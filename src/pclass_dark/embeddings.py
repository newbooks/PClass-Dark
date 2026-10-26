"""Frozen ESM-2 residue pooling with durable, independently verified shards."""

from __future__ import annotations

import csv
import importlib.metadata
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .artifacts import (
    IntegrityError,
    archive,
    file_record,
    identity,
    locked,
    publish,
    read_json,
    safe_file,
    schema_checked,
    sha256,
    staging,
    sync_dir,
    verify_bundle,
    write_json,
)
from .reference_data import read_fasta
from .setup_data import MODEL, REVISION, inventory_id, verify_prepared

LOG = logging.getLogger(__name__)
SCHEMA = "pclass-dark.embeddings.v1"
SHARD_ROWS = 64
ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWYXBZUO")
ASSET_HASHES = {
    "model.safetensors": (
        "c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566"
    ),
    "config.json": "e512f68ec444d99477703a9806639ca83da3dbc19f6c5fe428d6e5b7460972dc",
    "vocab.txt": "0b82cc0a7c7cf9e567b1e5892d793285b9fbae822c964ca48696f7db44598e03",
}
ROW_HEADER = ["row_index", "window_id", "sequence_id", "window_sha256"]


def pool_residues(hidden, attention_mask, special_tokens_mask, expected_lengths):
    """Pool residue rows only, validating both masks and float32 accumulation.

    Torch is imported only when this function is called. Invalid mask/layout or
    nonfinite residue states raise ValueError. Excluded rows never contribute.
    """
    import torch

    if hidden.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError(
            "Expected [batch,tokens,features] and two [batch,tokens] masks"
        )
    if tuple(hidden.shape[:2]) != tuple(attention_mask.shape) or tuple(
        special_tokens_mask.shape
    ) != tuple(attention_mask.shape):
        raise ValueError("Hidden states and masks do not align")
    for mask in (attention_mask, special_tokens_mask):
        if not torch.all((mask == 0) | (mask == 1)).item():
            raise ValueError("Masks must contain only 0/1")
    residue = attention_mask.bool() & ~special_tokens_mask.bool()
    lengths = torch.as_tensor(expected_lengths, device=residue.device)
    counts = residue.sum(dim=1)
    if (
        lengths.ndim != 1
        or lengths.shape != counts.shape
        or not torch.all(counts > 0).item()
        or not torch.equal(counts, lengths)
    ):
        raise ValueError("Residue count differs from window lengths or is empty")
    if not torch.isfinite(hidden[residue]).all().item():
        raise ValueError("Nonfinite selected residue hidden state")
    values = hidden.float().masked_fill(~residue.unsqueeze(-1), 0)
    pooled = values.sum(dim=1) / counts.unsqueeze(-1).float()
    if not torch.isfinite(pooled).all().item():
        raise ValueError("Nonfinite pooled embedding")
    return pooled


class EsmBackend:
    """Pinned offline-capable model adapter; instantiated only for new inference."""

    def __init__(self, *, model, revision, offline, device, seed):
        try:
            import torch
            from huggingface_hub import snapshot_download
            from transformers import EsmModel, EsmTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Embedding generation needs pip install 'pclass-dark[embeddings]'"
            ) from exc
        self.torch = torch
        self.model_id, self.revision = model, revision
        LOG.info("Resolving pinned ESM-2 checkpoint (offline=%s)", offline)
        snapshot = Path(
            snapshot_download(
                model,
                revision=revision,
                local_files_only=offline,
                allow_patterns=[
                    *ASSET_HASHES,
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ],
            )
        )
        assets = {}
        # Hub cache paths are symlinks by design: hash their resolved content.
        for name in [*ASSET_HASHES, "tokenizer_config.json", "special_tokens_map.json"]:
            assets[name] = sha256(snapshot / name)
            if name in ASSET_HASHES and assets[name] != ASSET_HASHES[name]:
                raise IntegrityError(f"Pinned model asset checksum mismatch: {name}")
        self.tokenizer = EsmTokenizer.from_pretrained(snapshot, local_files_only=True)
        self.tokenizer.padding_side = "right"
        vocab = self.tokenizer.get_vocab()
        for residue in sorted(ALPHABET):
            encoded = self.tokenizer(
                residue,
                return_special_tokens_mask=True,
                truncation=False,
                padding=False,
            )
            if (
                encoded["special_tokens_mask"] != [1, 0, 1]
                or len(encoded["input_ids"]) != 3
                or encoded["input_ids"][1] != vocab.get(residue)
                or encoded["input_ids"][1] == self.tokenizer.unk_token_id
            ):
                raise IntegrityError(f"Tokenizer cannot preserve residue: {residue}")
        torch.manual_seed(seed % (2**64))
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        requested = device
        resolved = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        self.model = (
            EsmModel.from_pretrained(
                snapshot, local_files_only=True, use_safetensors=True
            )
            .float()
            .eval()
        )
        if (
            self.model.config.hidden_size != 640
            or self.model.config.num_hidden_layers != 30
        ):
            raise IntegrityError("Unsupported ESM-2 model dimensions")
        try:
            self.model = self.model.to(resolved)
        except (RuntimeError, ValueError, AssertionError) as exc:
            if requested != "auto" or not resolved.startswith("cuda"):
                raise RuntimeError(
                    f"Cannot initialize requested device {resolved}"
                ) from exc
            LOG.warning("CUDA initialization failed; using CPU: %s", exc)
            resolved = "cpu"
            self.model = self.model.to(resolved)
        self.device = resolved
        self.contract = {
            "model": model,
            "revision": revision,
            "assets": assets,
            "torch_version": torch.__version__,
            "transformers_version": importlib.metadata.version("transformers"),
            "layer": 30,
            "width": 640,
            "inference_dtype": "float32",
            "output_dtype": "<f4",
            "pooling": "attention-and-not-special-mean-f32-v1",
            "residue_policy": "exact-vocab-ACDEFGHIKLMNPQRSTVWYXBZUO-v1",
            "algorithm": "esm2-shards64-v1",
            "window_length": 1022,
        }
        LOG.info(
            "Embedding device: %s; float32; final-layer residue-only mean", resolved
        )

    def embed(self, sequences):
        torch = self.torch
        if any(not s or len(s) > 1022 or set(s) - ALPHABET for s in sequences):
            raise ValueError("Unsupported or over-limit embedding window")
        encoded = self.tokenizer(
            sequences,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = encoded.pop("special_tokens_mask").to(self.device)
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        mask = encoded["attention_mask"].bool() & ~special.bool()
        vocab = self.tokenizer.get_vocab()
        for row, sequence in enumerate(sequences):
            actual = encoded["input_ids"][row][mask[row]].tolist()
            if actual != [vocab[c] for c in sequence]:
                raise IntegrityError("Residue-token IDs/order changed")
        with (
            torch.inference_mode(),
            torch.autocast(device_type=torch.device(self.device).type, enabled=False),
        ):
            output = self.model(**encoded).last_hidden_state
            result = pool_residues(
                output, encoded["attention_mask"], special, [len(s) for s in sequences]
            )
        return result.cpu().numpy().astype("<f4", copy=False)


def _finite_matrix(array, rows):
    if array.shape != (rows, 640) or array.dtype.str != "<f4":
        raise IntegrityError("Embedding matrix shape/dtype mismatch")
    for start in range(0, rows, 4096):
        if not np.isfinite(array[start : start + 4096]).all():
            raise IntegrityError("Nonfinite embedding matrix")


@schema_checked
def verify_embeddings(path, windows=None):
    path = Path(path)
    m = verify_bundle(path, SCHEMA)
    if m["id"] != identity(m["identity_inputs"]):
        raise IntegrityError("Embedding identity mismatch")
    if m["identity_inputs"]["inventory_id"] != m["inventory_id"]:
        raise IntegrityError("Embedding inventory identity mismatch")
    contract = m["identity_inputs"]["contract"]
    if (
        contract.get("model") != MODEL
        or contract.get("revision") != REVISION
        or any(contract.get("assets", {}).get(k) != v for k, v in ASSET_HASHES.items())
        or contract.get("width") != 640
        or contract.get("output_dtype") != "<f4"
    ):
        raise IntegrityError("Unsupported completed model contract")
    with (path / "rows.tsv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ROW_HEADER:
            raise IntegrityError("Invalid embedding row header")
        rows = list(reader)
    ids = [row["window_id"] for row in rows]
    if ids != sorted(set(ids)) or any(
        row["row_index"] != str(i) for i, row in enumerate(rows)
    ):
        raise IntegrityError("Embedding row indices/order mismatch")
    if inventory_id(rows) != m["inventory_id"]:
        raise IntegrityError("Embedding row inventory mismatch")
    if windows is not None:
        expected = [
            [w["window_id"], w["sequence_id"], w["window_sha256"]] for w in windows
        ]
        actual = [[w["window_id"], w["sequence_id"], w["window_sha256"]] for w in rows]
        if actual != expected:
            raise IntegrityError("Prepared/embedding row join mismatch")
    matrix = np.load(path / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    _finite_matrix(matrix, len(rows))
    if not matrix.flags.c_contiguous or m["rows"] != len(rows):
        raise IntegrityError("Embedding matrix layout/count mismatch")
    return m


def _compatible_existing(root, windows, inventory, model, revision):
    directory = root / "embeddings"
    if not directory.exists():
        return None
    matches = []
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or not path.is_dir():
            continue
        m = read_json(safe_file(path, "manifest.json"))
        if m.get("inventory_id") != inventory:
            continue
        contract = m.get("identity_inputs", {}).get("contract", {})
        if contract.get("model") != model or contract.get("revision") != revision:
            continue
        matches.append((path, m))
    if not matches:
        return None
    if len(matches) > 1:
        # Prefer exact installed execution versions without importing model code.
        try:
            versions = {
                "torch_version": importlib.metadata.version("torch"),
                "transformers_version": importlib.metadata.version("transformers"),
            }
        except importlib.metadata.PackageNotFoundError:
            versions = {}
        matching = [
            (p, m)
            for p, m in matches
            if versions
            and all(
                m["identity_inputs"]["contract"].get(k) == v
                for k, v in versions.items()
            )
        ]
        if len(matching) != 1:
            raise IntegrityError(
                "Multiple compatible embedding bundles; select a dedicated output root"
            )
        matches = matching
    path, _ = matches[0]
    with locked(root, "embedding-" + path.name):
        return path, verify_embeddings(path, windows), True


def _save_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".array-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _valid_shard(checkpoints, start, stop, embedding_id, inventory):
    array_path = checkpoints / f"{start:012d}.npy"
    receipt_path = checkpoints / f"{start:012d}.json"
    if not array_path.exists() and not receipt_path.exists():
        return None
    try:
        for path in (array_path, receipt_path):
            if path.is_symlink():
                raise IntegrityError("Symlink checkpoint")
        receipt = read_json(receipt_path)
        if not isinstance(receipt, dict):
            raise IntegrityError("Checkpoint receipt must be a JSON object")
        execution = receipt.get("execution")
        if not isinstance(execution, dict) or not isinstance(
            execution.get("contract"), dict
        ):
            raise IntegrityError(
                "Checkpoint execution and contract must be JSON objects"
            )
        expected = {
            "schema": "pclass-dark.embedding-shard.v1",
            "embedding_id": embedding_id,
            "inventory_id": inventory,
            "start": start,
            "stop": stop,
        }
        if any(receipt.get(k) != v for k, v in expected.items()):
            raise IntegrityError("Checkpoint row range/identity mismatch")
        if (
            receipt.get("shape") != [stop - start, 640]
            or receipt.get("dtype") != "<f4"
            or identity(
                {
                    "inventory_id": inventory,
                    "contract": execution["contract"],
                }
            )
            != embedding_id
        ):
            raise IntegrityError("Checkpoint execution/shape/dtype mismatch")
        if receipt["file"] != file_record(array_path):
            raise IntegrityError("Checkpoint checksum mismatch")
        array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        _finite_matrix(array, stop - start)
        return receipt
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        LOG.warning("Quarantining invalid shard %s: %s", start, exc)
        quarantine = checkpoints / ".quarantine"
        quarantine.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        for path in (array_path, receipt_path):
            if path.exists() or path.is_symlink():
                os.replace(path, quarantine / f"{path.name}-{stamp}")
        return None


def _assemble(stage, checkpoints, windows, embedding_id, inputs, prepared_id, receipts):
    array = np.lib.format.open_memmap(
        stage / "embeddings.npy", mode="w+", dtype="<f4", shape=(len(windows), 640)
    )
    for start in range(0, len(windows), SHARD_ROWS):
        stop = min(start + SHARD_ROWS, len(windows))
        source = np.load(
            checkpoints / f"{start:012d}.npy", mmap_mode="r", allow_pickle=False
        )
        array[start:stop] = source
    array.flush()
    del array
    with (stage / "embeddings.npy").open("rb") as handle:
        os.fsync(handle.fileno())
    with (stage / "rows.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(ROW_HEADER)
        for i, w in enumerate(windows):
            writer.writerow([i, w["window_id"], w["sequence_id"], w["window_sha256"]])
    m = {
        "schema": SCHEMA,
        "status": "complete",
        "id": embedding_id,
        "identity_inputs": inputs,
        "inventory_id": inputs["inventory_id"],
        "originating_preparation_id": prepared_id,
        "rows": len(windows),
        "columns": 640,
        "executions": [r["execution"] for r in receipts],
        "files": {p.name: file_record(p) for p in stage.iterdir()},
    }
    write_json(stage / "manifest.json", m)
    verify_embeddings(stage, windows)
    return m


def embed_prepared(
    root,
    prepared,
    manifest,
    *,
    model,
    revision,
    offline,
    device,
    batch_size,
    seed,
    force=False,
    transaction=None,
):
    # Revalidate under a shared identity lock before taking an immutable snapshot.
    with locked(root, "prepared-" + manifest["id"]):
        checked, windows, _ = verify_prepared(prepared, manifest["id"])
        if checked["inventory_id"] != manifest["inventory_id"]:
            raise IntegrityError("Prepared data changed before embedding")
        sequences = dict(read_fasta(prepared / "windows.fasta"))
    for wid, sequence in sequences.items():
        unsupported = set(sequence) - ALPHABET
        if unsupported or not sequence or len(sequence) > 1022:
            raise ValueError(
                f"Unsupported embedding input {wid}: {sorted(unsupported)}"
            )
    if not force:
        reused = _compatible_existing(
            root, windows, manifest["inventory_id"], model, revision
        )
        if reused:
            LOG.info("Reusing completed embeddings without loading a PLM")
            return reused
    # Factory indirection supports deterministic fault-injection tests.
    backend = EsmBackend(
        model=model, revision=revision, offline=offline, device=device, seed=seed
    )
    inputs = {"inventory_id": manifest["inventory_id"], "contract": backend.contract}
    embedding_id = identity(inputs)
    target = safe_file(root, "embeddings/" + embedding_id)
    checkpoints = safe_file(root, ".checkpoints/" + embedding_id)
    with locked(root, "embedding-" + embedding_id):
        if target.exists() and not force:
            return target, verify_embeddings(target, windows), True
        if force:
            archive(root, checkpoints)
        if checkpoints.is_symlink():
            raise IntegrityError("Symlink checkpoint directory")
        checkpoints.mkdir(parents=True, exist_ok=True)
        receipts = []
        for start in range(0, len(windows), SHARD_ROWS):
            stop = min(start + SHARD_ROWS, len(windows))
            receipt = _valid_shard(
                checkpoints, start, stop, embedding_id, manifest["inventory_id"]
            )
            if receipt is None:
                LOG.info("Embedding windows %s..%s of %s", start, stop, len(windows))
                array = np.empty((stop - start, 640), dtype="<f4")
                for batch_start in range(start, stop, batch_size):
                    batch_stop = min(batch_start + batch_size, stop)
                    batch = [
                        sequences[w["window_id"]]
                        for w in windows[batch_start:batch_stop]
                    ]
                    try:
                        values = backend.embed(batch)
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            raise RuntimeError(
                                "Embedding OOM: resume with smaller --batch-size; "
                                "committed shards remain valid"
                            ) from exc
                        raise
                    _finite_matrix(values, batch_stop - batch_start)
                    array[batch_start - start : batch_stop - start] = values
                path = checkpoints / f"{start:012d}.npy"
                _save_array(path, array)
                receipt = {
                    "schema": "pclass-dark.embedding-shard.v1",
                    "embedding_id": embedding_id,
                    "inventory_id": manifest["inventory_id"],
                    "start": start,
                    "stop": stop,
                    "shape": [stop - start, 640],
                    "dtype": "<f4",
                    "file": file_record(path),
                    "execution": {
                        "device": backend.device,
                        "batch_size": batch_size,
                        "seed": seed,
                        "contract": backend.contract,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                }
                write_json(checkpoints / f"{start:012d}.json", receipt)
            receipts.append(receipt)
        with staging(target.parent) as stage:
            m = _assemble(
                stage,
                checkpoints,
                windows,
                embedding_id,
                inputs,
                manifest["id"],
                receipts,
            )
            publish(root, stage, target, force, transaction)
        LOG.info("Durably committed %s/%s windows", len(windows), len(windows))
        # Shards remain as a verified local recovery cache. They are not packaged.
        return target, m, False
