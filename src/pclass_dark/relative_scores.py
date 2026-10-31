"""Final-signature empirical C/B distributions; no membership probabilities."""

from __future__ import annotations

import logging
import math
import shutil
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .artifacts import (
    IntegrityError,
    check_cleanup_pending,
    file_record,
    identity,
    read_json,
    safe_file,
    schema_checked,
    sha256,
    transaction_guard,
    verify_bundle,
    write_json,
)
from .cluster_paths import LAYOUT, cluster_directory_name, stored_cluster_name
from .fit_records import payload_identity
from .library_checkpoints import Checkpoints
from .library_runtime import freeze, score_signature

SCHEMA = "pclass-dark.relative-score-library.v1"
CLUSTER_SCHEMA = "pclass-dark.relative-score-cluster.v1"
SCORE_RULE = "ordered-float64-v1"
EVIDENCE_RULE = "empirical-inclusive-plus-one-B-v1"
SUMMARY = (
    "cluster_id status reason C_n B_n C_min C_max B_min B_max "
    "final_signature_sha256 metadata"
).split()
LOG = logging.getLogger(__name__)


class UnavailableDistributionError(ValueError):
    """A target does not have a usable empirical reference distribution."""


def _validate_array(array, count):
    if array.shape != (count,) or array.dtype.str != "<f8":
        raise IntegrityError("Relative array shape/dtype mismatch")
    previous = None
    for start in range(0, len(array), 1048576):
        chunk = array[start : start + 1048576]
        if not np.isfinite(chunk).all() or np.any(chunk[1:] < chunk[:-1]):
            raise IntegrityError("Relative array is nonfinite or unsorted")
        if len(chunk):
            if previous is not None and chunk[0] < previous:
                raise IntegrityError("Relative array is unsorted across blocks")
            previous = chunk[-1]


def compute_distributions(
    signature, matrix, populations, *, embedding_contract, block_size=4096
):
    """Score exact supplied C/B row indices, returning sorted float64 arrays.

    Raises ValueError/IntegrityError for unsupported signatures or invalid inputs.
    This does not fit parameters, discover features or generate embeddings.
    """
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if signature["status"] != "ok":
        raise UnavailableDistributionError(signature["status"])
    result = {}
    for key in ("C", "B"):
        rows = np.asarray(populations[key])
        if (
            rows.ndim != 1
            or rows.dtype.kind not in "iu"
            or not len(rows)
            or np.any(rows < 0)
            or np.any(rows >= len(matrix))
        ):
            raise IntegrityError("Relative scoring needs nonempty valid row indices")
        scores = np.empty(len(rows), dtype="<f8")
        last = time.monotonic()
        for start in range(0, len(rows), block_size):
            stop = min(start + block_size, len(rows))
            scores[start:stop], _ = score_signature(
                signature,
                matrix[rows[start:stop]],
                embedding_contract=embedding_contract,
            )
            if time.monotonic() - last >= 30:
                LOG.info(
                    "Cluster %s relative_scores %s: %s/%s windows",
                    signature["cluster_id"],
                    key,
                    stop,
                    len(rows),
                )
                last = time.monotonic()
        scores.sort(kind="stable")
        _validate_array(scores, len(rows))
        result[key] = scores
    return result


def distribution_checkpoint(
    root, fit_id, result, inputs, matrix, populations, contract, *, force=False
):
    from .artifacts import archive

    directory = safe_file(root, ".relative-checkpoints/" + fit_id)
    if force:
        archive(root, directory)
    store = Checkpoints(directory, fit_id)
    dep = identity([fit_id, payload_identity(result), SCORE_RULE, EVIDENCE_RULE])
    counts = {"C": inputs["C_n"], "B": inputs["B_n"]}
    status = result["signature"]["status"]

    def validate(payload):
        if (
            payload["fit_id"] != fit_id
            or payload["payload_id"] != payload_identity(result)
            or payload["counts"] != counts
            or payload["status"] != status
        ):
            raise IntegrityError("Relative checkpoint binding mismatch")

    saved = store.read("complete", dep, validate)
    if saved:
        try:
            if set(saved[1]) != ({"C", "B"} if status == "ok" else set()):
                raise IntegrityError("Relative checkpoint array inventory mismatch")
            for key, array in saved[1].items():
                _validate_array(array, counts[key])
            return directory, True
        except IntegrityError:
            # A valid checksum can still bind an invalid (e.g. unsorted) array.
            from .artifacts import archive

            archive(root, directory)
            store = Checkpoints(directory, fit_id)
    arrays = (
        compute_distributions(
            result["signature"], matrix, populations, embedding_contract=contract
        )
        if status == "ok"
        else {}
    )
    if any(len(arrays[k]) != counts[k] for k in arrays):
        raise IntegrityError("Relative population count mismatch")
    store.write(
        "complete",
        dep,
        {
            "fit_id": fit_id,
            "payload_id": payload_identity(result),
            "counts": counts,
            "status": status,
        },
        arrays=arrays,
    )
    return directory, False


def write_companion(stage, production, paths):
    """Publishable companion contents from verified per-target checkpoints."""
    from .library import _table

    pm = read_json(production / "manifest.json")
    root = production.parent
    bindings = []
    counts = {
        "targets": len(paths),
        "available": 0,
        "unsupported": 0,
        "total_C_values": 0,
        "total_B_values": 0,
    }
    with _table(stage / "summary.tsv", SUMMARY) as report:
        for path in paths:
            target = read_json(path)
            cluster = target["cluster_id"]
            inputs = target["fit_inputs"]
            signature_path = (
                production
                / "clusters"
                / stored_cluster_name(cluster, pm)
                / "final.json"
            )
            signature = read_json(signature_path)
            fit_id = signature["fit_identity"]
            source = safe_file(root, target["relative"])
            if sha256(source / "complete.json") != target["relative_receipt_sha256"]:
                raise IntegrityError("Relative target receipt changed")
            receipt = read_json(source / "complete.json")
            payload = receipt["payload"]
            if (
                receipt["task_id"] != fit_id
                or receipt["payload_id"] != identity(payload)
                or payload["payload_id"] != signature["scientific_payload_id"]
                or payload["counts"] != {"C": inputs["C_n"], "B": inputs["B_n"]}
                or payload["status"] != signature["status"]
            ):
                raise IntegrityError("Companion checkpoint/scientific binding mismatch")
            destination = stage / "clusters" / cluster_directory_name(cluster)
            destination.mkdir(parents=True)
            meta = {
                "schema": CLUSTER_SCHEMA,
                "cluster_id": cluster,
                "status": signature["status"],
                "reason": signature["reason"],
                "production_library_id": pm["id"],
                "final_fit_id": fit_id,
                "scientific_payload_id": signature["scientific_payload_id"],
                "final_signature_sha256": sha256(signature_path),
                "representation_contract_id": signature["representation_contract_id"],
                "prepared_id": target["prepared_id"],
                "embedding_id": target["embedding_id"],
                "matrix_sha256": inputs["matrix_sha256"],
                "C_ids_sha256": inputs["C_ids_sha256"],
                "B_ids_sha256": inputs["B_ids_sha256"],
                "C_n": inputs["C_n"],
                "B_n": inputs["B_n"],
                "fit_id": "final",
                "score_space": "log-arithmetic-gaussian-compatibility",
                "score_rule": SCORE_RULE,
                "evidence_rule": EVIDENCE_RULE,
                "C_role": "final-fit-in-sample",
                "B_role": "fixed-discovery-background-reused",
                "probability_calibrated": False,
                "arrays": {},
            }
            available = signature["status"] == "ok"
            if set(receipt["arrays"]) != ({"C", "B"} if available else set()):
                raise IntegrityError("Companion checkpoint inventory mismatch")
            for key in ("C", "B") if available else ():
                item = receipt["arrays"][key]
                src = safe_file(source, item["file"])
                if file_record(src) != receipt["files"][item["file"]]:
                    raise IntegrityError("Relative checkpoint checksum mismatch")
                array = np.load(src, mmap_mode="r", allow_pickle=False)
                _validate_array(array, meta[key + "_n"])
                dst = destination / f"{key}_log_scores.npy"
                shutil.copyfile(src, dst)
                with dst.open("rb") as handle:
                    import os

                    os.fsync(handle.fileno())
                meta["arrays"][key] = {
                    "file": dst.name,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    **file_record(dst),
                    "minimum": float(array[0]),
                    "maximum": float(array[-1]),
                }
                counts["total_" + key + "_values"] += len(array)
            write_json(destination / "distribution.json", meta)
            counts["available" if available else "unsupported"] += 1
            bindings.append(
                {
                    "cluster_id": cluster,
                    "fit_id": fit_id,
                    "signature": meta["final_signature_sha256"],
                    "payload": meta["scientific_payload_id"],
                    "prepared_id": meta["prepared_id"],
                    "embedding_id": meta["embedding_id"],
                    "inputs": inputs,
                }
            )
            report(
                {
                    "cluster_id": cluster,
                    "status": meta["status"],
                    "reason": meta["reason"],
                    "C_n": meta["C_n"],
                    "B_n": meta["B_n"],
                    **{
                        key + "_" + short: meta["arrays"].get(key, {}).get(long)
                        for key in ("C", "B")
                        for short, long in (("min", "minimum"), ("max", "maximum"))
                    },
                    "final_signature_sha256": meta["final_signature_sha256"],
                    "metadata": str(
                        (destination / "distribution.json").relative_to(stage)
                    ),
                }
            )
    inputs = {
        "contract": SCHEMA,
        "score_rule": SCORE_RULE,
        "evidence_rule": EVIDENCE_RULE,
        "production_id": pm["id"],
        "production_manifest_sha256": sha256(production / "manifest.json"),
        "targets": bindings,
    }
    manifest = {
        "schema": SCHEMA,
        "cluster_directory_layout": LAYOUT,
        "id": identity(inputs),
        "identity_inputs": inputs,
        "status": "complete",
        "counts": counts,
        "source_production": {
            "id": pm["id"],
            "path": str(production.relative_to(root)),
            "manifest_sha256": inputs["production_manifest_sha256"],
        },
        "files": {
            str(p.relative_to(stage)): file_record(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        },
        "software_version": __version__,
        "execution": {"device": "cpu", "dtype": "float64"},
    }
    from .artifacts import sync_tree

    sync_tree(stage)
    write_json(stage / "manifest.json", manifest)
    return manifest


@dataclass(frozen=True)
class RelativeEvidence:
    background_tail_fraction: np.ndarray
    background_percentile_le: np.ndarray
    background_evidence: np.ndarray
    cluster_percentile_le: np.ndarray
    background_n_ge: np.ndarray
    C_n: int
    B_n: int
    status: str
    C_role: str
    B_role: str
    probability_calibrated: bool
    final_fit_id: str
    scientific_payload_id: str


class RelativeEvaluator:
    """A lazy reference view valid only inside RelativeScores.for_cluster()."""

    def __init__(self, owner, cluster_id, metadata):
        self._owner, self._cluster, self._meta = owner, cluster_id, metadata
        self._arrays = None
        self._active = True

    def evaluate(self, log_scores):
        if not self._active:
            raise RuntimeError("Relative evaluator is closed")
        values = np.asarray(log_scores)
        if (
            values.ndim != 1
            or values.dtype.kind != "f"
            or not np.isfinite(values).all()
        ):
            raise ValueError("Expected finite floating one-dimensional log scores")
        values = values.astype(np.float64, copy=False)
        if self._arrays is None:
            self._arrays = self._owner._arrays_for(self._cluster, self._meta)
        return _empirical_evidence(values, self._arrays, self._meta)

    def close(self):
        self._active = False
        self._arrays = self._owner = None


class RelativeScores:
    def __init__(self, path, library, manifest, metadata, immutable):
        self._layout = manifest.get("cluster_directory_layout", "hex-v1")
        self.path, self.library = path, library
        self.relative_id = manifest["id"]
        self._digest = sha256(path / "manifest.json")
        self._metadata = metadata
        self._immutable = immutable
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self._lock = threading.RLock()

    @schema_checked
    def summarize(self, cluster_id):
        """Return verified C/B log-space linear quantiles for one cluster.

        Raises ValueError for an unknown ID, IntegrityError for a changed/corrupt
        generation, and OSError for inaccessible files. No query is evaluated.
        """
        check_cleanup_pending(self.library.source_root)
        if not isinstance(cluster_id, str) or not cluster_id:
            raise ValueError("cluster_id must be a nonempty string")
        with self._lock:
            meta = self._metadata.get(cluster_id)
            if meta is None:
                raise ValueError(f"Unknown cluster: {cluster_id}")
            with (
                nullcontext()
                if self._immutable
                else transaction_guard(self.library.source_root)
            ):
                self._check_generation()
                summary = {
                    "status": meta["status"],
                    "reason": meta["reason"],
                    "relative_id": self.relative_id,
                    "quantile_method": "linear-in-log-space-v1",
                    "C": None,
                    "B": None,
                }
                if meta["status"] == "ok":
                    arrays = _read_arrays(
                        self.path, meta, mmap=True, layout=self._layout
                    )
                    for key in ("C", "B"):
                        summary[key] = {
                            "n": len(arrays[key]),
                            "role": meta[key + "_role"],
                            "quantiles": _score_quantiles(arrays[key]),
                        }
                return freeze(summary)

    @property
    def manifest_sha256(self):
        """Captured companion manifest identity (read-only)."""
        return self._digest

    def _check_generation(self):
        check_cleanup_pending(self.library.source_root)
        try:
            valid = (
                sha256(self.path / "manifest.json") == self._digest
                and sha256(self.library.source_path / "manifest.json")
                == self.library.manifest_sha256
            )
        except OSError as exc:
            raise IntegrityError(
                "Relative generation removed; reload the selected pair"
            ) from exc
        if not valid:
            raise IntegrityError(
                "Relative generation changed; reload the selected pair"
            )

    def describe(self):
        """Return frozen target status/count metadata, without reading arrays."""
        with (
            self._lock,
            (
                nullcontext()
                if self._immutable
                else transaction_guard(self.library.source_root)
            ),
        ):
            self._check_generation()
            keys = (
                "status",
                "reason",
                "C_n",
                "B_n",
                "final_fit_id",
                "scientific_payload_id",
                "representation_contract_id",
            )
            return freeze(
                {c: {k: m[k] for k in keys} for c, m in self._metadata.items()}
            )

    @contextmanager
    def for_cluster(self, cluster_id):
        """Pin one lazy C/B pair under its guard; closed views cannot be reused.

        Raises UnavailableDistributionError for unsupported/unknown targets,
        IntegrityError for changed sources and SetupBusyError for contention.
        No arrays are materialized unless evaluate() is called.
        """
        with (
            self._lock,
            (
                nullcontext()
                if self._immutable
                else transaction_guard(self.library.source_root)
            ),
        ):
            self._check_generation()
            meta = self._metadata.get(cluster_id)
            if meta is None or meta["status"] != "ok":
                raise UnavailableDistributionError(
                    f"Unavailable relative scores: {cluster_id}"
                )
            view = RelativeEvaluator(self, cluster_id, meta)
            try:
                yield view
            finally:
                view.close()

    def evaluate(self, cluster_id, log_scores):
        with self.for_cluster(cluster_id) as view:
            return view.evaluate(log_scores)

    def _arrays_for(self, cluster_id, meta):
        if cluster_id in self._cache:
            arrays = self._cache.pop(cluster_id)
            self._cache[cluster_id] = arrays
        else:
            arrays = _read_arrays(self.path, meta, layout=self._layout)
            size = sum(a.nbytes for a in arrays.values())
            while self._cache and (
                len(self._cache) >= 16 or self._cache_bytes + size > 64 * 2**20
            ):
                _, old = self._cache.popitem(last=False)
                self._cache_bytes -= sum(a.nbytes for a in old.values())
            if size <= 64 * 2**20:
                self._cache[cluster_id] = arrays
                self._cache_bytes += size
        return arrays


def _empirical_evidence(values, arrays, meta):
    """Existing inclusive empirical formulas, shared by single and scoped calls."""
    c, b = arrays["C"], arrays["B"]
    ge = len(b) - np.searchsorted(b, values, side="left")
    tail = (1 + ge) / (len(b) + 1)
    outputs = [
        tail,
        np.searchsorted(b, values, side="right") / len(b),
        -np.log10(tail),
        np.searchsorted(c, values, side="right") / len(c),
        ge,
    ]
    for array in outputs:
        array.flags.writeable = False
    return RelativeEvidence(
        *outputs,
        len(c),
        len(b),
        "ok",
        meta["C_role"],
        meta["B_role"],
        False,
        meta["final_fit_id"],
        meta["scientific_payload_id"],
    )


def _score_quantiles(array):
    """Interpolate a verified nonempty sorted array without a population copy."""
    result = []
    for q in (0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0):
        h = (len(array) - 1) * q
        i, j = math.floor(h), math.ceil(h)
        value = (
            float(array[i])
            if i == j
            else float((1 - (h - i)) * array[i] + (h - i) * array[j])
        )
        score = math.exp(value)
        result.append(
            {
                "q": q,
                "log_score": value,
                "score": score,
                "score_underflow": score == 0.0,
            }
        )
    return result


def _read_arrays(path, meta, *, mmap=False, layout="hex-v1"):

    directory = safe_file(
        path,
        "clusters/"
        + stored_cluster_name(meta["cluster_id"], {"cluster_directory_layout": layout}),
    )
    arrays = {}
    for key in ("C", "B"):
        info = meta["arrays"][key]
        file = safe_file(directory, info["file"])
        if file_record(file) != {k: info[k] for k in ("sha256", "size_bytes")}:
            raise IntegrityError("Relative score file checksum mismatch")
        array = np.load(file, allow_pickle=False, mmap_mode="r" if mmap else None)
        _validate_array(array, meta[key + "_n"])
        if not len(array):
            raise IntegrityError("Available relative scores must be nonempty")
        if (
            tuple(info["shape"]) != array.shape
            or info["dtype"] != array.dtype.str
            or info["minimum"] != float(array[0])
            or info["maximum"] != float(array[-1])
        ):
            raise IntegrityError("Relative array metadata mismatch")
        array.flags.writeable = False
        arrays[key] = array
    return arrays


@schema_checked
def _load_relative(path, library, immutable, *, expected_targets=None):

    preview = read_json(path / "manifest.json")
    members = None
    if expected_targets is not None:
        if len(library.signatures) != 1:
            raise IntegrityError("Scoped companion requires exactly one signature")
        members = []
        for cluster, signature in library.signatures.items():
            prefix = "clusters/" + stored_cluster_name(cluster, preview) + "/"
            members.append(prefix + "distribution.json")
            if signature["status"] == "ok":
                members.extend(prefix + key + "_log_scores.npy" for key in ("C", "B"))
    manifest = verify_bundle(path, SCHEMA, members=members)
    if (
        identity(manifest["identity_inputs"]) != manifest["id"]
        or manifest["source_production"]["id"] != library.library_id
        or manifest["source_production"]["manifest_sha256"] != library.manifest_sha256
    ):
        raise IntegrityError("Companion/production identity mismatch")
    bindings = {b["cluster_id"]: b for b in manifest["identity_inputs"]["targets"]}
    if (
        set(bindings)
        != set(library.signatures if expected_targets is None else expected_targets)
        or not set(library.signatures) <= set(bindings)
        or len(bindings) != len(manifest["identity_inputs"]["targets"])
        or manifest["identity_inputs"]["score_rule"] != SCORE_RULE
        or manifest["identity_inputs"]["evidence_rule"] != EVIDENCE_RULE
        or manifest["identity_inputs"]["production_id"] != library.library_id
        or manifest["identity_inputs"]["production_manifest_sha256"]
        != library.manifest_sha256
        or manifest["source_production"]["path"]
        != str(library.source_path.relative_to(library.source_root))
    ):
        raise IntegrityError("Relative manifest source/target bindings mismatch")
    metadata = {}
    required = {"summary.tsv"}
    totals = {
        "targets": len(library.signatures),
        "available": 0,
        "unsupported": 0,
        "total_C_values": 0,
        "total_B_values": 0,
    }
    for cluster, signature in library.signatures.items():
        prefix = "clusters/" + stored_cluster_name(cluster, manifest) + "/"
        name = prefix + "distribution.json"
        required.add(name)
        meta = read_json(safe_file(path, name))
        if (
            meta["schema"] != CLUSTER_SCHEMA
            or meta["cluster_id"] != cluster
            or meta["status"] != signature["status"]
            or meta["fit_id"] != "final"
            or meta["production_library_id"] != library.library_id
            or meta["final_fit_id"] != signature["fit_identity"]
            or meta["scientific_payload_id"] != signature["scientific_payload_id"]
            or meta["final_signature_sha256"]
            != sha256(library.source_path / prefix / "final.json")
            or meta["representation_contract_id"]
            != signature["representation_contract_id"]
            or meta["score_rule"] != SCORE_RULE
            or meta["evidence_rule"] != EVIDENCE_RULE
            or meta["probability_calibrated"] is not False
            or meta["C_role"] != "final-fit-in-sample"
            or meta["B_role"] != "fixed-discovery-background-reused"
            or meta["C_n"] != signature["counts"]["training_positive_n"]
            or meta["B_n"] != signature["counts"]["background_n"]
        ):
            raise IntegrityError("Relative cluster binding mismatch")
        binding = bindings[cluster]
        if (
            identity(binding["inputs"]) != meta["final_fit_id"]
            or binding["inputs"]["cluster_id"] != cluster
            or binding["fit_id"] != meta["final_fit_id"]
            or binding["signature"] != meta["final_signature_sha256"]
            or binding["payload"] != meta["scientific_payload_id"]
            or binding["prepared_id"] != meta["prepared_id"]
            or binding["embedding_id"] != meta["embedding_id"]
            or any(
                meta[k] != binding["inputs"][k]
                for k in ("matrix_sha256", "C_ids_sha256", "B_ids_sha256", "C_n", "B_n")
            )
            or meta["reason"] != signature["reason"]
            or meta["score_space"] != "log-arithmetic-gaussian-compatibility"
        ):
            raise IntegrityError("Relative metadata/input identity mismatch")
        available = meta["status"] == "ok"
        if set(meta["arrays"]) != ({"C", "B"} if available else set()):
            raise IntegrityError("Relative array inventory mismatch")
        if available:
            for key in ("C", "B"):
                if meta["arrays"][key]["file"] != f"{key}_log_scores.npy":
                    raise IntegrityError("Unexpected score array name")
                required.add(prefix + meta["arrays"][key]["file"])
            _read_arrays(
                path,
                meta,
                mmap=True,
                layout=manifest.get("cluster_directory_layout", "hex-v1"),
            )  # Verify one target at a time, bounded RAM.
            totals["total_C_values"] += meta["C_n"]
            totals["total_B_values"] += meta["B_n"]
        totals["available" if available else "unsupported"] += 1
        metadata[cluster] = meta
    if expected_targets is not None:
        # The returned object intentionally contains only the requested metadata.
        # Other clusters' files and redundant shared reports are outside this scope.
        return RelativeScores(path, library, manifest, freeze(metadata), immutable)
    if set(manifest["files"]) != required or manifest["counts"] != totals:
        raise IntegrityError("Relative file inventory/count mismatch")
    import csv

    with (path / "summary.tsv").open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        if rows.fieldnames != SUMMARY or [r["cluster_id"] for r in rows] != sorted(
            metadata
        ):
            raise IntegrityError("Relative summary inventory mismatch")
    return RelativeScores(path, library, manifest, freeze(metadata), immutable)


@schema_checked
def load_relative_scores(relative_dir, *, library, immutable=False):
    """Validate a companion and expose bounded, read-only empirical evidence."""
    check_cleanup_pending(library.source_root)
    path = Path(relative_dir).absolute()
    immutable = immutable or library.immutable
    if path != library.source_root / "relative-scores" / path.name:
        raise IntegrityError(
            "Companion must be under the bound library's relative-scores root"
        )
    with nullcontext() if immutable else transaction_guard(library.source_root):
        if sha256(library.source_path / "manifest.json") != library.manifest_sha256:
            raise IntegrityError("Production generation changed")
        return _load_relative(path, library, immutable)


@dataclass(frozen=True)
class SelectedLibrary:
    library: object
    relative: RelativeScores | None
    selection_schema: str


@schema_checked
def load_selected_library(output_root=None, *, immutable=False, cluster_id=None):
    """Load a guarded pair; optionally verify only one v2 production target.

    With cluster_id, shared manifests/profiles and only the requested target's
    files are checked. Without it, retain the complete library audit.
    """
    from .library import _library_immutable, load_library
    from .paths import default_library_input_dir

    if cluster_id is not None and (not isinstance(cluster_id, str) or not cluster_id):
        raise ValueError("cluster_id must be a nonempty string")
    root = Path(
        default_library_input_dir() if output_root is None else output_root
    ).absolute()
    immutable = _library_immutable(root, immutable)
    with nullcontext() if immutable else transaction_guard(root):
        selected = read_json(safe_file(root, "selection.json"))
        if selected["schema"] not in {
            "pclass-dark.library-selection.v1",
            "pclass-dark.library-selection.v2",
        }:
            raise IntegrityError("Unknown library selection schema")
        path = safe_file(root, selected["library"])
        if sha256(path / "manifest.json") != selected["manifest_sha256"]:
            raise IntegrityError("Selected production manifest mismatch")
        library = load_library(path, immutable=immutable, cluster_id=cluster_id)
        if library.library_id != selected["library_id"]:
            raise IntegrityError("Selected production ID mismatch")
        relative = None
        if selected["schema"].endswith(".v2"):
            directory = safe_file(root, selected["relative"])
            if (
                sha256(directory / "manifest.json")
                != selected["relative_manifest_sha256"]
            ):
                raise IntegrityError("Selected relative manifest mismatch")
            if cluster_id is None:
                relative = load_relative_scores(
                    directory, library=library, immutable=immutable
                )
            else:
                if (
                    directory
                    != library.source_root / "relative-scores" / directory.name
                ):
                    raise IntegrityError(
                        "Companion is outside bound relative-scores root"
                    )
                with (
                    nullcontext()
                    if immutable
                    else transaction_guard(library.source_root)
                ):
                    if sha256(path / "manifest.json") != library.manifest_sha256:
                        raise IntegrityError("Production generation changed")
                    production = read_json(path / "manifest.json")
                    relative = _load_relative(
                        directory,
                        library,
                        immutable,
                        expected_targets=production["identity_inputs"]["cluster_ids"],
                    )
            if relative.relative_id != selected["relative_id"]:
                raise IntegrityError("Selected relative ID mismatch")
            if any(
                m["prepared_id"] != selected["preparation_id"]
                or m["embedding_id"] != selected["embedding_id"]
                for m in relative._metadata.values()
            ):
                raise IntegrityError("Selected setup/relative source mismatch")
        return SelectedLibrary(library, relative, selected["schema"])
