"""Mask correctness, checkpoint recovery, and artifact/vector identity."""

from pathlib import Path

import numpy as np
import pytest
import torch

from pclass_dark import artifacts, embeddings
from pclass_dark.artifacts import IntegrityError, read_json
from pclass_dark.embeddings import pool_residues, verify_embeddings
from pclass_dark.setup_data import MODEL, REVISION, prepare_data


def test_pooling_independent_numeric_oracle_and_masked_nan():
    hidden = torch.tensor(
        [
            [
                [100.0, 200.0],
                [2.0, 4.0],
                [6.0, 8.0],
                [900.0, 999.0],
                [float("nan"), float("inf")],
            ]
        ]
    )
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 0]])
    result = pool_residues(hidden, attention, special, [2])
    assert torch.equal(result, torch.tensor([[4.0, 6.0]]))
    assert result.dtype == torch.float32
    # Exercise all (attention,special) combinations; only (1,0) contributes.
    assert torch.equal(
        pool_residues(
            torch.tensor([[[90.0], [8.0], [99.0], [500.0]]]),
            torch.tensor([[0, 1, 1, 0]]),
            torch.tensor([[0, 0, 1, 1]]),
            [1],
        ),
        torch.tensor([[8.0]]),
    )


@pytest.mark.parametrize("case", ["shape", "nonbinary", "empty", "length", "nan"])
def test_pooling_rejects_invalid_data(case):
    hidden = torch.ones((1, 4, 3))
    attention = torch.tensor([[1, 1, 1, 1]])
    special = torch.tensor([[1, 0, 0, 1]])
    expected = [2]
    if case == "shape":
        special = special[:, :2]
    if case == "nonbinary":
        attention[0, 1] = 2
    if case == "empty":
        special[:] = 1
    if case == "length":
        expected = [3]
    if case == "nan":
        hidden[0, 1] = float("nan")
    with pytest.raises(ValueError):
        pool_residues(hidden, attention, special, expected)


def test_reduced_precision_hidden_accumulates_in_float32():
    hidden = torch.tensor(
        [[[999.0], [1.0], [2.0], [4.0], [999.0]]], dtype=torch.float16
    )
    result = pool_residues(
        hidden, torch.ones((1, 5)), torch.tensor([[1, 0, 0, 0, 1]]), [3]
    )
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.tensor([[7 / 3]], dtype=torch.float32))


@pytest.fixture
def fake_backend(monkeypatch):
    class FakeBackend:
        calls = []
        fail_after = None
        initialized = 0

        def __init__(self, **options):
            type(self).initialized += 1
            self.device = options["device"]
            self.contract = {
                "model": MODEL,
                "revision": REVISION,
                "assets": dict(embeddings.ASSET_HASHES),
                "torch_version": "fixture",
                "transformers_version": "fixture",
                "width": 640,
                "output_dtype": "<f4",
                "layer": 30,
                "pooling": "attention-and-not-special-mean-f32-v1",
            }

        def embed(self, sequences):
            if self.fail_after is not None and len(self.calls) >= self.fail_after:
                raise RuntimeError("injected out of memory")
            self.calls.extend(sequences)
            return np.asarray(
                [[sum(map(ord, s)) / len(s)] * 640 for s in sequences], dtype="<f4"
            )

    monkeypatch.setattr(embeddings, "EsmBackend", FakeBackend)
    return FakeBackend


def test_embedding_resume_batch_change_and_model_free_reuse(
    reference_factory, tmp_path, fake_backend
):
    reference = reference_factory()
    root = tmp_path / "output"
    fake_backend.fail_after = 64
    with pytest.raises(RuntimeError, match="smaller --batch-size"):
        prepare_data(root, reference_dir=reference, batch_size=8)
    assert len(fake_backend.calls) == 64
    assert len(list((root / ".checkpoints").glob("*/*.json"))) == 1
    fake_backend.fail_after = None
    fake_backend.calls.clear()
    result = prepare_data(root, reference_dir=reference, batch_size=3)
    assert len(fake_backend.calls) == 76
    assert not result.reused_embeddings
    assert verify_embeddings(result.embedding_dir)["rows"] == 140
    matrix = np.load(result.embedding_dir / "embeddings.npy")
    assert matrix.shape == (140, 640)
    before = fake_backend.initialized
    reused = prepare_data(root, reference_dir=reference)
    assert reused.reused_embeddings
    assert fake_backend.initialized == before
    changed = prepare_data(root, reference_dir=reference, seed=3)
    assert changed.embedding_id == result.embedding_id
    assert changed.preparation_id != result.preparation_id
    assert fake_backend.initialized == before


def test_corrupt_checkpoint_recomputed_and_final_assembly_resumes(
    reference_factory, tmp_path, fake_backend, monkeypatch
):
    reference = reference_factory()
    root = tmp_path / "output"
    original = embeddings._assemble

    def stop(*args, **kwargs):
        raise OSError("injected disk full")

    monkeypatch.setattr(embeddings, "_assemble", stop)
    with pytest.raises(OSError):
        prepare_data(root, reference_dir=reference)
    assert len(fake_backend.calls) == 140
    shards = sorted((root / ".checkpoints").glob("*/*.npy"))
    shards[0].write_bytes(b"corrupt")
    fake_backend.calls.clear()
    monkeypatch.setattr(embeddings, "_assemble", original)
    result = prepare_data(root, reference_dir=reference, batch_size=4)
    assert len(fake_backend.calls) == 64
    assert list((root / ".checkpoints").glob("*/.quarantine/*"))
    assert verify_embeddings(result.embedding_dir)["rows"] == 140


def test_interruption_after_array_before_receipt(
    reference_factory, tmp_path, fake_backend, monkeypatch
):
    reference = reference_factory()
    root = tmp_path / "output"
    original = embeddings.write_json
    failed = False

    def stop(path, value):
        nonlocal failed
        if Path(path).name == "000000000064.json" and not failed:
            failed = True
            raise KeyboardInterrupt()
        return original(path, value)

    monkeypatch.setattr(embeddings, "write_json", stop)
    with pytest.raises(KeyboardInterrupt):
        prepare_data(root, reference_dir=reference)
    fake_backend.calls.clear()
    result = prepare_data(root, reference_dir=reference)
    assert len(fake_backend.calls) == 76
    assert verify_embeddings(result.embedding_dir)["rows"] == 140


def test_vector_corruption_and_row_reordering_rejected(
    reference_factory, tmp_path, fake_backend
):
    ref = reference_factory()
    result = prepare_data(tmp_path / "out", reference_dir=ref)
    p = result.embedding_dir
    matrix = np.load(p / "embeddings.npy", mmap_mode="r+")
    matrix[0, 0] = float("nan")
    matrix.flush()
    with pytest.raises(IntegrityError):
        verify_embeddings(p)
    # Updating the outer hash still must not bypass finite-state checks.
    m = read_json(p / "manifest.json")
    m["files"]["embeddings.npy"] = artifacts.file_record(p / "embeddings.npy")
    artifacts.write_json(p / "manifest.json", m)
    with pytest.raises(IntegrityError, match="Nonfinite"):
        verify_embeddings(p)


def test_unsupported_residues_fail_before_backend(
    reference_factory, tmp_path, fake_backend
):
    ref = reference_factory([("p", "K", "AJ")])
    with pytest.raises(ValueError, match="Unsupported embedding input"):
        prepare_data(tmp_path / "out", reference_dir=ref, min_clusters=1)
    assert fake_backend.initialized == 0
    assert list((tmp_path / "out/prepared").glob("*/manifest.json"))


def test_local_real_encoder_pooling_batch_invariance(tmp_path):
    from transformers import EsmConfig, EsmModel, EsmTokenizer

    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(["<cls>", "<pad>", "<eos>", "<unk>", "<mask>", "A", "C", "D", "E"])
        + "\n"
    )
    tokenizer = EsmTokenizer(vocab_file=str(vocab))
    torch.manual_seed(7)
    model = EsmModel(
        EsmConfig(
            vocab_size=9,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            pad_token_id=1,
            mask_token_id=4,
            token_dropout=False,
        )
    ).eval()

    def embed(sequences):
        encoded = tokenizer(
            sequences,
            padding=True,
            truncation=False,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        special = encoded.pop("special_tokens_mask")
        with torch.inference_mode():
            states = model(**encoded).last_hidden_state
            pooled = pool_residues(
                states, encoded["attention_mask"], special, list(map(len, sequences))
            )
        for i, seq in enumerate(sequences):
            assert torch.allclose(pooled[i], states[i, 1 : len(seq) + 1].mean(0))
        return pooled

    alone = embed(["AC"])[0]
    padded = embed(["AC", "ACDEACDE"])[0]
    reordered = embed(["ACDEACDE", "AC"])[1]
    assert torch.allclose(alone, padded, atol=1e-6)
    assert torch.allclose(alone, reordered, atol=1e-6)


def test_pinned_tokenizer_residues_masks_and_boundaries():
    from transformers import EsmTokenizer

    tokenizer = EsmTokenizer(
        vocab_file=str(Path(__file__).parent / "fixtures/esm2/vocab.txt")
    )
    sequences = ["A", "XBZUO", "A" * 1021, "C" * 1022]
    encoded = tokenizer(
        sequences,
        padding=True,
        truncation=False,
        return_special_tokens_mask=True,
        return_tensors="pt",
    )
    residue = encoded["attention_mask"].bool() & ~encoded["special_tokens_mask"].bool()
    assert residue.sum(1).tolist() == [1, 5, 1021, 1022]
    assert encoded["input_ids"].shape == (4, 1024)
    for i, sequence in enumerate(sequences):
        assert encoded["input_ids"][i][residue[i]].tolist() == [
            tokenizer.get_vocab()[s] for s in sequence
        ]
    for text in ("<cls>", "A<eos>C", "a", "AJ"):
        assert set(text) - embeddings.ALPHABET


def test_shard_receipt_wrong_identity_and_indices(tmp_path):
    path = tmp_path / "checkpoints"
    path.mkdir()
    embeddings._save_array(path / "000000000000.npy", np.zeros((2, 640), dtype="<f4"))
    artifacts.write_json(
        path / "000000000000.json",
        {
            "schema": "pclass-dark.embedding-shard.v1",
            "embedding_id": "wrong",
            "inventory_id": "inventory",
            "start": 0,
            "stop": 2,
            "file": artifacts.file_record(path / "000000000000.npy"),
        },
    )
    assert embeddings._valid_shard(path, 0, 2, "expected", "inventory") is None
    assert list((path / ".quarantine").iterdir())


def test_uninterrupted_and_resumed_matrices_agree(
    reference_factory, tmp_path, fake_backend
):
    reference = reference_factory()
    first = prepare_data(tmp_path / "one", reference_dir=reference, batch_size=7)
    fake_backend.calls.clear()
    fake_backend.fail_after = 64
    with pytest.raises(RuntimeError):
        prepare_data(tmp_path / "two", reference_dir=reference, batch_size=8)
    fake_backend.calls.clear()
    fake_backend.fail_after = None
    resumed = prepare_data(tmp_path / "two", reference_dir=reference, batch_size=3)
    assert len(fake_backend.calls) == 76
    assert np.array_equal(
        np.load(first.embedding_dir / "embeddings.npy"),
        np.load(resumed.embedding_dir / "embeddings.npy"),
    )


def test_prepare_only_does_not_request_model_dependencies(
    reference_factory, tmp_path, monkeypatch
):
    import importlib.abc
    import sys

    reference = reference_factory()
    root = tmp_path / "out"

    class BlockModels(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in {
                "transformers",
                "huggingface_hub",
                "safetensors",
            }:
                raise AssertionError("Unexpected model dependency: " + fullname)

    monkeypatch.setattr(sys, "meta_path", [BlockModels(), *sys.meta_path])
    result = prepare_data(root, reference_dir=reference, prepare_only=True)
    assert result.embedding_dir is None


def test_failed_selection_rolls_back_force_replaced_bundles(
    reference_factory, tmp_path, fake_backend, monkeypatch
):
    import pclass_dark.setup_data as setup

    ref = reference_factory()
    root = tmp_path / "out"
    result = prepare_data(root, reference_dir=ref)
    selected = (root / "selection.json").read_bytes()
    previous_prepared = (result.prepared_dir / "manifest.json").read_bytes()
    previous_embedding = (result.embedding_dir / "manifest.json").read_bytes()
    original = setup.write_json

    def fail(path, value):
        if Path(path).name == "selection.json":
            raise OSError("injected selection publication failure")
        return original(path, value)

    monkeypatch.setattr(setup, "write_json", fail)
    with pytest.raises(OSError, match="selection publication"):
        prepare_data(root, reference_dir=ref, force=True)
    assert (root / "selection.json").read_bytes() == selected
    assert (result.prepared_dir / "manifest.json").read_bytes() == previous_prepared
    assert (result.embedding_dir / "manifest.json").read_bytes() == previous_embedding
    assert verify_embeddings(result.embedding_dir)["rows"] == 140


def test_durable_commit_message_only_after_publication(
    reference_factory, tmp_path, fake_backend, monkeypatch, caplog
):
    import logging

    ref = reference_factory()
    caplog.set_level(logging.INFO, logger="pclass_dark.embeddings")
    original = embeddings._assemble

    def fail(*args, **kwargs):
        raise OSError("injected finalization failure")

    monkeypatch.setattr(embeddings, "_assemble", fail)
    with pytest.raises(OSError):
        prepare_data(tmp_path / "out", reference_dir=ref)
    assert not any("Durably committed" in r.message for r in caplog.records)
    monkeypatch.setattr(embeddings, "_assemble", original)
    prepare_data(tmp_path / "out", reference_dir=ref)
    messages = [r.message for r in caplog.records if "Durably committed" in r.message]
    assert messages == ["Durably committed 140/140 windows"]


@pytest.mark.parametrize(
    "malformation",
    ["top_level_list", "execution_null", "execution_list", "contract_null"],
)
def test_malformed_receipt_is_quarantined_and_resume_recovers(
    reference_factory, tmp_path, fake_backend, monkeypatch, malformation
):
    reference = reference_factory()
    root = tmp_path / "output"
    original = embeddings._assemble

    def stop(*args, **kwargs):
        raise OSError("injected finalization failure")

    monkeypatch.setattr(embeddings, "_assemble", stop)
    with pytest.raises(OSError):
        prepare_data(root, reference_dir=reference)
    assert len(fake_backend.calls) == 140
    receipt_path = next((root / ".checkpoints").glob("*/000000000000.json"))
    receipt = read_json(receipt_path)
    if malformation == "top_level_list":
        receipt = []
    elif malformation == "execution_null":
        receipt["execution"] = None
    elif malformation == "execution_list":
        receipt["execution"] = []
    else:
        receipt["execution"]["contract"] = None
    artifacts.write_json(receipt_path, receipt)

    fake_backend.calls.clear()
    monkeypatch.setattr(embeddings, "_assemble", original)
    result = prepare_data(root, reference_dir=reference)
    assert len(fake_backend.calls) == 64  # Other committed shards remain reusable.
    quarantined = list((receipt_path.parent / ".quarantine").iterdir())
    assert len(quarantined) == 2
    assert any(p.name.startswith("000000000000.json-") for p in quarantined)
    assert any(p.name.startswith("000000000000.npy-") for p in quarantined)
    assert verify_embeddings(result.embedding_dir)["rows"] == 140

    fake_backend.calls.clear()
    assert prepare_data(root, reference_dir=reference).reused_embeddings
    assert not fake_backend.calls


def test_force_publication_blocks_concurrent_reuse_until_rollback_finishes(
    reference_factory, tmp_path, fake_backend, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import pclass_dark.setup_data as setup
    from pclass_dark.artifacts import SetupBusyError, sha256
    from pclass_dark.cluster_sets import load_sets

    reference = reference_factory()
    root = tmp_path / "output"
    result = prepare_data(root, reference_dir=reference)
    previous_selection = (root / "selection.json").read_bytes()
    published = Event()
    finish = Event()
    original = setup.write_json

    def fail_selection(path, value):
        if Path(path).name == "selection.json" and not published.is_set():
            published.set()  # Both force-rebuilt bundles are now published.
            assert finish.wait(timeout=10)
            raise OSError("injected selection failure after publication")
        return original(path, value)

    monkeypatch.setattr(setup, "write_json", fail_selection)
    with ThreadPoolExecutor(max_workers=1) as executor:
        writer = executor.submit(
            prepare_data, root, reference_dir=reference, force=True
        )
        try:
            assert published.wait(timeout=10)
            with pytest.raises(SetupBusyError):
                prepare_data(root, reference_dir=reference)
            with pytest.raises(SetupBusyError):
                load_sets(result.prepared_dir)
            with pytest.raises(SetupBusyError):
                verify_embeddings(result.embedding_dir)
        finally:
            finish.set()
        with pytest.raises(OSError, match="selection failure"):
            writer.result(timeout=10)

    assert (root / "selection.json").read_bytes() == previous_selection
    selected = read_json(root / "selection.json")
    assert (
        sha256(result.prepared_dir / "manifest.json")
        == selected["prepared_manifest_sha256"]
    )
    assert (
        sha256(result.embedding_dir / "manifest.json")
        == selected["embedding_manifest_sha256"]
    )
    monkeypatch.setattr(setup, "write_json", original)
    assert prepare_data(root, reference_dir=reference).reused_embeddings
