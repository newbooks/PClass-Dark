"""Independent numerical and ownership checks for SPEC-0020."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from pclass_dark.artifacts import IntegrityError
from pclass_dark.structural_models import (
    BUNDLE_ID,
    StructuralModelBundle,
    bundled_structural_models,
    prepare_structural_models,
    verify_structural_models,
)
from pclass_dark.structure import (
    StructuralHeads,
    StructuralPredictor,
    _owned_ranges,
    structural_windows,
)


@pytest.fixture(scope="module")
def heads():
    return StructuralHeads()


def test_bundle_prepare_reuse_and_corruption(tmp_path):
    bundle = prepare_structural_models(tmp_path)
    assert bundle.model_id == BUNDLE_ID
    assert bundle.path == tmp_path / "structural-models" / BUNDLE_ID
    assert verify_structural_models(bundle.path) == bundle
    before = (bundle.path / "models.safetensors").stat().st_mtime_ns
    assert prepare_structural_models(tmp_path) == bundle
    assert (bundle.path / "models.safetensors").stat().st_mtime_ns == before
    with (bundle.path / "models.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(IntegrityError, match="checksum"):
        prepare_structural_models(tmp_path)


@pytest.mark.parametrize("kind", ["contract", "missing", "extra", "symlink"])
def test_bundle_rejects_invalid_artifacts(tmp_path, kind):
    target = tmp_path / "bundle"
    shutil.copytree(bundled_structural_models(), target)
    if kind == "contract":
        path = target / "manifest.json"
        data = json.loads(path.read_text())
        data["identity_inputs"]["contract"]["window_length"] = 1022
        path.write_text(json.dumps(data))
    if kind == "missing":
        (target / "models.safetensors").unlink()
    if kind == "extra":
        (target / "unexpected").write_text("x")
    if kind == "symlink":
        (target / "models.safetensors").unlink()
        (target / "models.safetensors").symlink_to(
            bundled_structural_models() / "models.safetensors"
        )
    with pytest.raises((IntegrityError, ValueError)):
        verify_structural_models(target)


@pytest.mark.parametrize("length", [1, 1021, 1022, 1500, 2044, 5000, 10001])
def test_windows_against_independent_residue_center_oracle(length):
    windows = structural_windows(length)
    assert windows[0][0] == 1 and windows[-1][1] == length
    assert all(b - a + 1 <= 1021 for a, b in windows)
    assert all(
        a[1] - b[0] + 1 >= 100 for a, b in zip(windows, windows[1:], strict=False)
    )
    ownership = _owned_ranges(windows)
    actual = []
    for ordinal, (left, right) in enumerate(ownership):
        assert windows[ordinal][0] <= left <= right <= windows[ordinal][1]
        actual.extend([ordinal] * (right - left + 1))
    expected = [
        min(
            (abs(2 * pos - a - b), a, i)
            for i, (a, b) in enumerate(windows)
            if a <= pos <= b
        )[2]
        for pos in range(1, length + 1)
    ]
    assert actual == expected


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_invalid_window_length(length):
    with pytest.raises(ValueError):
        structural_windows(length)


def test_real_heads_against_explicit_frozen_tensor_formula(heads):
    tensors = load_file(str(heads.bundle.path / "models.safetensors"))
    values = np.random.default_rng(20).normal(size=(7, 640)).astype(np.float32)
    raw, labels = heads.predict(values, batch_size=2, cpu_threads=1)
    for task in ("rsa", "q3"):
        normed = (
            (values.astype(np.float64) - tensors[f"{task}.mean"].numpy())
            / tensors[f"{task}.scale"].numpy()
        ).astype(np.float32)
        normed[:, tensors[f"{task}.constant"].numpy()] = 0
        y = torch.tensor(normed)
        # No use of the production module architecture or forward method.
        for layer in (0, 3, 5):
            y = (
                y @ tensors[f"{task}.{layer}.weight"].T
                + tensors[f"{task}.{layer}.bias"]
            )
            if layer != 5:
                y = 0.5 * y * (1 + torch.erf(y / np.sqrt(2)))
        expected = y.numpy()
        if task == "rsa":
            np.testing.assert_allclose(raw, expected[:, 0], atol=2e-6, rtol=2e-5)
        else:
            np.testing.assert_array_equal(
                labels, np.array(["C", "E", "H"])[expected.argmax(1)]
            )
    other_raw, other_labels = heads.predict(values, batch_size=7)
    np.testing.assert_allclose(raw, other_raw, atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(labels, other_labels)
    empty = heads.predict(np.empty((0, 640), np.float32))
    assert all(len(x) == 0 for x in empty)


def test_analytic_heads_normalization_constant_mask_and_raw_output(
    monkeypatch, tmp_path
):
    tensors = load_file(str(bundled_structural_models() / "models.safetensors"))
    for key in tensors:
        if key.endswith(("weight", "bias")):
            tensors[key].zero_()
    for task in ("rsa", "q3"):
        tensors[f"{task}.mean"].fill_(2)
        tensors[f"{task}.scale"].fill_(2)
        tensors[f"{task}.constant"].zero_()
        tensors[f"{task}.constant"][0] = True
        tensors[f"{task}.0.weight"][0, :2] = 1
        tensors[f"{task}.3.weight"][0, 0] = 1
    tensors["rsa.5.weight"][0, 0] = 1
    tensors["rsa.5.bias"].fill_(-2)  # Ensure the API does not clip negative ALL CEF.
    tensors["q3.5.bias"][:] = torch.tensor([0.0, 4.0, 4.0])  # tie E/H -> E
    save_file(tensors, str(tmp_path / "models.safetensors"))
    import pclass_dark.structure as module

    monkeypatch.setattr(
        module,
        "verify_structural_models",
        lambda _: StructuralModelBundle("test", tmp_path),
    )
    model = StructuralHeads(tmp_path)
    values = np.full((1, 640), 2, np.float32)
    values[0, 0] = 100  # Must be zeroed after normalization.
    values[0, 1] = 4
    raw, labels = model.predict(values)
    import math

    def gelu(x):
        return x / 2 * (1 + math.erf(x / math.sqrt(2)))

    assert raw[0] == pytest.approx(gelu(gelu(1)) - 2, abs=2e-7)
    assert raw[0] < 0 and labels.tolist() == ["E"]


@pytest.mark.parametrize("kind", ["nan", "shape", "scale", "dtype", "extra"])
def test_rejects_invalid_model_tensors(monkeypatch, tmp_path, kind):
    import pclass_dark.structure as module

    tensors = load_file(str(bundled_structural_models() / "models.safetensors"))
    if kind == "nan":
        tensors["rsa.0.weight"][0, 0] = float("nan")
    if kind == "shape":
        tensors["q3.5.bias"] = torch.zeros(4)
    if kind == "scale":
        tensors["rsa.scale"][0] = 0
    if kind == "dtype":
        tensors["q3.mean"] = tensors["q3.mean"].float()
    if kind == "extra":
        tensors["unexpected"] = torch.zeros(1)
    save_file(tensors, str(tmp_path / "models.safetensors"))
    monkeypatch.setattr(
        module,
        "verify_structural_models",
        lambda _: StructuralModelBundle("test", tmp_path),
    )
    with pytest.raises(IntegrityError):
        StructuralHeads(tmp_path)


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((2, 639), np.float32),
        np.zeros((1, 640)),
        np.full((1, 640), np.nan, np.float32),
    ],
)
def test_invalid_states(heads, values):
    with pytest.raises(ValueError):
        heads.predict(values)


@pytest.fixture
def fake_residues(monkeypatch):
    calls = []

    class Fake:
        def __init__(self, **options):
            self.device = "cpu"
            self.options = options

        def residue_states(self, seqs):
            calls.append(list(seqs))
            return [np.full((len(s), 640), 0.1, np.float32) for s in seqs]

    monkeypatch.setattr("pclass_dark.embeddings.EsmBackend", Fake)
    return calls


def test_sequence_prediction_masking_batches_and_lazy_loading(fake_residues):
    a = StructuralPredictor(device="cpu", offline=True, window_batch_size=1)
    assert not fake_residues
    sequence = "acD" * 750 + "XBZUO*?-"
    result = a.predict(sequence)
    assert result.sequence == sequence.upper() and result.device == "cpu"
    assert result.model_id == BUNDLE_ID
    assert all(
        len(s) <= 1021 and not set(s) - set("ACDX")
        for group in fake_residues
        for s in group
    )
    assert np.isfinite(result.rsa[:-8]).all()
    assert np.isnan(result.rsa[-8:]).all()
    assert result.q3[-8:].tolist() == ["NA"] * 8
    assert not result.rsa.flags.writeable and not result.q3.flags.writeable
    b = StructuralPredictor(device="cpu", window_batch_size=8)
    other = b.predict(sequence)
    np.testing.assert_allclose(result.rsa, other.rsa, equal_nan=True)
    np.testing.assert_array_equal(result.q3, other.q3)
    before = len(fake_residues)
    missing = a.predict("X?-")
    assert np.isnan(missing.rsa).all() and len(fake_residues) == before
    assert StructuralPredictor().predict("X").q3.tolist() == ["NA"]
    assert len(fake_residues) == before


@pytest.mark.parametrize("sequence", ["", "A C", "A\nC", "A1", "é", None])
def test_invalid_sequence_before_esm(sequence, fake_residues):
    model = StructuralPredictor()
    with pytest.raises(ValueError):
        model.predict(sequence)
    assert not fake_residues


def test_import_and_prepare_do_not_load_optional_stack(tmp_path):
    code = """
import sys
from pclass_dark.structural_models import prepare_structural_models
import pclass_dark.structure
assert "torch" not in sys.modules and "transformers" not in sys.modules
prepare_structural_models(sys.argv[1])
assert "torch" not in sys.modules and "transformers" not in sys.modules
assert "prostruct" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_esm_residue_path_excludes_bos_eos_padding_and_preserves_pooling():
    from types import SimpleNamespace

    from transformers import EsmTokenizer

    from pclass_dark.embeddings import EsmBackend

    backend = EsmBackend.__new__(EsmBackend)
    backend.torch = torch
    backend.device = "cpu"
    backend.tokenizer = EsmTokenizer(
        vocab_file=str(Path(__file__).parent / "fixtures/esm2/vocab.txt")
    )

    class TokenModel:
        def __call__(self, input_ids, **kwargs):
            hidden = input_ids[:, :, None].float().repeat(1, 1, 640)
            # Padding nonfinites must not contaminate any residue or pooled vector.
            hidden[input_ids == backend.tokenizer.pad_token_id] = float("nan")
            return SimpleNamespace(last_hidden_state=hidden)

    backend.model = TokenModel()
    sequences = ["A", "ACDX", "C" * 1021]
    blocks = backend.residue_states(sequences)
    for sequence, block in zip(sequences, blocks, strict=True):
        expected = np.array([backend.tokenizer.get_vocab()[r] for r in sequence])
        assert block.shape == (len(sequence), 640) and block.dtype == np.float32
        np.testing.assert_array_equal(block, np.repeat(expected[:, None], 640, axis=1))
    pooled = backend.embed(sequences)
    for i, block in enumerate(blocks):
        np.testing.assert_allclose(pooled[i], block.mean(axis=0))
    with pytest.raises(ValueError):
        backend.residue_states([])
    with pytest.raises(ValueError):
        backend.residue_states(["A" * 1023])


def test_build_clean_does_not_prepare_models(library_data, tmp_path, monkeypatch):
    from pclass_dark.library import build_library

    data, *_ = library_data()
    root = tmp_path / "library"
    result = build_library(data, root)
    shutil.rmtree(root / "structural-models")

    def unexpected(*args):
        pytest.fail("--clean attempted to prepare structural models")

    monkeypatch.setattr(
        "pclass_dark.structural_models.prepare_structural_models", unexpected
    )
    cleaned = build_library(data, root, clean=True)
    assert cleaned.library_id == result.library_id
    assert not (root / "structural-models").exists()


def test_cli_reports_prepared_models(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from typer.testing import CliRunner

    from pclass_dark.cli import app

    monkeypatch.setattr(
        "pclass_dark.library.build_library",
        lambda *a, **k: SimpleNamespace(
            library_dir=tmp_path,
            structural_model_dir=tmp_path / "models",
            relative_dir=None,
            research_dir=None,
        ),
    )
    result = CliRunner().invoke(app, ["build-library"])
    assert result.exit_code == 0, result.output
    assert f"Structural models: {tmp_path / 'models'}" in result.stdout


def test_source_and_wheel_distribute_self_contained_models(tmp_path):
    import tarfile
    import zipfile

    project = Path(__file__).resolve().parents[1]
    release = tmp_path / "release"
    release.mkdir()
    for name in (
        "pyproject.toml",
        "setup.py",
        "build_support.py",
        "MANIFEST.in",
        "README.md",
        "LICENSE",
    ):
        shutil.copyfile(project / name, release / name)
    shutil.copytree(
        project / "src/pclass_dark",
        release / "src/pclass_dark",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    completed = subprocess.run(
        [sys.executable, "-m", "build", "--no-isolation"],
        cwd=release,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    expected = {p.name for p in bundled_structural_models().iterdir()}
    with zipfile.ZipFile(next((release / "dist").glob("*.whl"))) as wheel:
        prefix = "pclass_dark/resources/structure/v1/"
        found = {
            name[len(prefix) :] for name in wheel.namelist() if name.startswith(prefix)
        }
        assert found == expected
        assert (
            wheel.read(prefix + "models.safetensors")
            == (bundled_structural_models() / "models.safetensors").read_bytes()
        )
        wheel.extractall(tmp_path / "installed")
    # Import the wheel's copy away from both the repository and ProStruct.
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from pclass_dark.structural_models import verify_structural_models
from pclass_dark.structure import StructuralHeads
assert verify_structural_models().path.is_relative_to(sys.argv[1])
assert set(StructuralHeads().models) == {"rsa", "q3"}
assert "prostruct" not in sys.modules
"""
    loaded = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "installed")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert loaded.returncode == 0, loaded.stderr
    with tarfile.open(next((release / "dist").glob("*.tar.gz"))) as source:
        assert {
            Path(name).name
            for name in source.getnames()
            if "/resources/structure/v1/" in name
        } == expected


@pytest.mark.parametrize("device", ["auto", "cuda"])
def test_head_initialization_device_fallback(monkeypatch, device):
    class FakeESM:
        device = "cuda:0"

        def __init__(self, **options):
            self.model = self

        def to(self, actual):
            assert actual == "cpu"
            self.device = actual

    monkeypatch.setattr("pclass_dark.embeddings.EsmBackend", FakeESM)
    predictor = StructuralPredictor(device=device)
    move = predictor.heads.to

    def fail_gpu(actual):
        if actual.startswith("cuda"):
            raise RuntimeError("injected GPU initialization failure")
        move(actual)

    monkeypatch.setattr(predictor.heads, "to", fail_gpu)
    if device == "auto":
        backend = predictor._runtime()
        assert backend.device == predictor.heads.device == "cpu"
    else:
        with pytest.raises(RuntimeError, match="Cannot initialize"):
            predictor._runtime()
        assert predictor._embedder is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "invalid"},
        {"window_batch_size": 0},
        {"window_batch_size": 9},
        {"cpu_threads": 0},
        {"cpu_threads": 9},
        {"offline": "yes"},
    ],
)
def test_predictor_option_validation(kwargs):
    with pytest.raises(ValueError):
        StructuralPredictor(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")
def test_gpu_head_agreement():
    model = StructuralHeads()
    values = np.random.default_rng(20).normal(size=(8, 640)).astype(np.float32)
    expected = model.predict(values)
    model.to("cuda:0")
    observed = model.predict(values)
    np.testing.assert_allclose(observed[0], expected[0], atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(observed[1], expected[1])


def test_failed_model_copy_does_not_publish_partial_output(tmp_path, monkeypatch):
    import pclass_dark.structural_models as module

    original = module.shutil.copyfile
    calls = 0

    def fail_mid_copy(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected copy failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(module.shutil, "copyfile", fail_mid_copy)
    with pytest.raises(OSError, match="injected"):
        prepare_structural_models(tmp_path)
    assert not (tmp_path / "structural-models").exists()
    assert not list(tmp_path.glob(".staging-*"))


def test_corrupt_companion_preserves_reused_production(library_data, tmp_path):
    from pclass_dark.library import build_library
    from pclass_dark.relative_scores import load_selected_library

    data, *_ = library_data()
    root = tmp_path / "library"
    result = build_library(data, root)
    selected = (root / "selection.json").read_bytes()
    with (result.structural_model_dir / "models.safetensors").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(IntegrityError, match="checksum") as error:
        build_library(data, root)
    assert error.value.production_result.library_id == result.library_id
    assert (root / "selection.json").read_bytes() == selected
    assert load_selected_library(root).library.library_id == result.library_id
