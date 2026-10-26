"""Opt-in real-checkpoint acceptance checks; never download model assets."""

import os

import numpy as np
import pytest

from pclass_dark.embeddings import EsmBackend
from pclass_dark.setup_data import MODEL, REVISION

pytestmark = pytest.mark.skipif(
    os.environ.get("PCLASS_DARK_TEST_CACHED_ESM") != "1",
    reason="Set PCLASS_DARK_TEST_CACHED_ESM=1 with the pinned checkpoint cached",
)


def test_cached_checkpoint_cpu_pooling_and_batching():
    backend = EsmBackend(
        model=MODEL, revision=REVISION, offline=True, device="cpu", seed=42
    )
    torch = backend.torch
    short = "ACDEFGHIKLMNPQRSTVWY"
    sequences = [short, "A" * 1022, "XBZUO"]
    batch = backend.embed(sequences)
    alone = backend.embed([short])[0]
    reordered = backend.embed(["XBZUO", short])[1]
    assert batch.shape == (3, 640)
    assert np.allclose(batch[0], alone, rtol=1e-4, atol=1e-5)
    assert np.allclose(batch[0], reordered, rtol=1e-4, atol=1e-5)
    encoded = backend.tokenizer(
        sequences, padding=True, truncation=False, return_tensors="pt"
    )
    with torch.inference_mode():
        hidden = backend.model(**encoded).last_hidden_state
        # Independent slice oracle, not the production mask/pooling helper.
        for i, sequence in enumerate(sequences):
            expected = hidden[i, 1 : len(sequence) + 1, :].float().mean(0).numpy()
            assert np.allclose(batch[i], expected, rtol=1e-4, atol=1e-5)


def test_cached_checkpoint_cuda_agrees_with_cpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("No compatible CUDA device available")
    cpu = EsmBackend(
        model=MODEL, revision=REVISION, offline=True, device="cpu", seed=42
    )
    gpu = EsmBackend(
        model=MODEL, revision=REVISION, offline=True, device="cuda", seed=42
    )
    sequences = ["ACDEFGHIKLMNPQRSTVWY", "XBZUO"]
    assert np.allclose(cpu.embed(sequences), gpu.embed(sequences), rtol=1e-4, atol=1e-5)
