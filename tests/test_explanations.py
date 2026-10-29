"""Artifact-backed explanations and independent narrative/quantile checks."""

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pclass_dark.artifacts import IntegrityError, read_json, write_json
from pclass_dark.cli import app
from pclass_dark.explanations import (
    SignatureExplanation,
    explain_signature,
    render_signature_explanation,
)
from pclass_dark.library import build_library
from pclass_dark.relative_scores import _score_quantiles, load_selected_library


@pytest.fixture
def explanation_library(library_data, tmp_path):
    data, *_ = library_data(count=6, min_clusters=1)
    root = tmp_path / "library"
    result = build_library(data, root)
    return root, result


def test_report_and_cli_without_setup_data(explanation_library, monkeypatch):
    root, result = explanation_library
    selected_before = (root / "selection.json").read_bytes()
    # No preparation/embedding artifacts are required to explain a library.
    shutil.rmtree(root.parent / "data")
    report = explain_signature("K0", root, detailed=True)
    data = report.to_dict()
    assert data["schema"] == "pclass-dark.signature-explanation.v1"
    assert data["source"]["library_id"] == result.library_id
    assert data["populations"]["C_windows"] == 3
    assert len(data["dimensions"]["all"]) == 640
    assert len(data["selection"]["prefixes"]) == 640
    assert data["scoring"]["sigma_floor"] == 1e-8
    assert data["metrics"]["training"]["auroc"] == 1
    assert data["selection"]["best_prefix_length"] == 1  # Earliest tied maximum.
    assert data["selection"]["core"] == [0, 1]
    assert data["distributions"]["C"]["n"] == 3
    with pytest.raises(TypeError):
        report.data["source"]["cluster_id"] = "changed"
    data["source"]["cluster_id"] = "changed"
    assert report.to_dict()["source"]["cluster_id"] == "K0"
    monkeypatch.setattr(
        Path, "open", lambda *a, **k: pytest.fail("renderer read a file")
    )
    text = render_signature_explanation(report)
    for phrase in [
        "1023–2043",
        "above 2044",
        "attention mask",
        "special-token mask",
        "Gaussian similarity",
        "Individual AUROCs are not averaged",
        "highest cutoff",
        "in-sample",
        "deployment precision",
        "No single ABC value",
        "Every recorded removal trial",
    ]:
        assert phrase in text
    assert "\x1b" not in text
    assert "## Signature" in render_signature_explanation(report, format="markdown")
    assert (
        json.loads(render_signature_explanation(report, format="json"))
        == report.to_dict()
    )
    assert text == render_signature_explanation(report)
    monkeypatch.undo()
    for format in ["text", "markdown", "json"]:
        response = CliRunner().invoke(
            app,
            [
                "explain-signature",
                "--cluster",
                "K0",
                "--library-dir",
                str(root),
                "--detailed",
                "--format",
                format,
            ],
        )
        assert response.exit_code == 0, response.output
        assert response.stdout == render_signature_explanation(report, format=format)
    assert (root / "selection.json").read_bytes() == selected_before


def test_default_report_missing_companion_and_unknown_target(explanation_library):
    root, _ = explanation_library
    default = explain_signature("K0", root).to_dict()
    assert default["dimensions"]["all"] is None
    assert default["selection"]["iterations"] is None
    assert len(default["dimensions"]["top"]) == 3
    selector = read_json(root / "selection.json")
    selector["schema"] = "pclass-dark.library-selection.v1"
    for key in ["relative", "relative_id", "relative_manifest_sha256"]:
        selector.pop(key)
    write_json(root / "selection.json", selector)
    report = explain_signature("K0", root)
    assert report.to_dict()["distributions"]["reason"] == "no_selected_companion"
    assert "no_selected_companion" in render_signature_explanation(report)
    with pytest.raises(ValueError, match="Unknown cluster"):
        explain_signature(" K0", root)
    with pytest.raises(ValueError, match="boolean"):
        explain_signature("K0", root, detailed=1)
    with pytest.raises(ValueError, match="format"):
        render_signature_explanation(report, format="pdf")


def test_unsupported_fit_is_explained(library_data, tmp_path):
    data, *_ = library_data(count=1, min_clusters=1)
    root = tmp_path / "unsupported"
    build_library(data, root)
    report = explain_signature("K0", root, detailed=True)
    result = report.to_dict()
    assert result["status"] != "ok"
    assert result["selection"]["termination"] == "unsupported"
    assert result["selection"]["best_prefix_length"] is None
    assert result["metrics"]["training"] is None
    assert result["distributions"]["C"] is None
    text = render_signature_explanation(report)
    assert "no usable signature" in text and "No acceptance threshold" in text
    assert "are accepted (log score" not in text


def test_summary_quantiles_and_generation_checks(explanation_library):
    root, result = explanation_library
    relative = load_selected_library(root).relative
    summary = relative.summarize("K0")
    directory = result.relative_dir / "clusters/K0"
    for key in ("C", "B"):
        logs = np.load(directory / f"{key}_log_scores.npy")
        expected = np.quantile(
            logs, [0, 0.25, 0.5, 0.75, 0.95, 0.99, 1], method="linear"
        )
        np.testing.assert_allclose(
            [q["log_score"] for q in summary[key]["quantiles"]], expected
        )
    with pytest.raises(TypeError):
        summary["C"]["quantiles"][0]["score"] = 10
    with pytest.raises(ValueError, match="Unknown cluster"):
        relative.summarize("absent")
    manifest = read_json(result.library_dir / "manifest.json")
    manifest["test_changed"] = True
    write_json(result.library_dir / "manifest.json", manifest)
    with pytest.raises(IntegrityError, match="generation changed"):
        relative.summarize("K0")


def test_independent_quantile_and_rendering_edge_cases(explanation_library):
    # Linear interpolation of logs [-4, -2, 0] gives -3 at q=.25, not the
    # arithmetic interpolation of their exponentiated scores.
    quantiles = _score_quantiles(np.array([-4.0, -2.0, 0.0]))
    assert quantiles[1]["log_score"] == -3
    assert quantiles[1]["score"] == math.exp(-3)
    assert quantiles[1]["score"] != (math.exp(-4) + math.exp(-2)) / 2
    assert all(q["score_underflow"] for q in _score_quantiles(np.array([-1000.0])))
    data = explain_signature("K0", explanation_library[0]).to_dict()
    data["scoring"]["threshold"] = {
        "kind": "above_maximum",
        "score_value": None,
        "log_value": None,
    }
    data["metrics"]["training"]["precision"] = None
    data["metrics"]["training"]["unavailable_reasons"]["precision"] = "no_predictions"
    data["distributions"]["C"]["quantiles"] = _score_quantiles(np.array([-1000.0]))
    text = render_signature_explanation(SignatureExplanation(data))
    assert "predict-none: no windows" in text
    assert "unavailable: no_predictions" in text
    assert "underflow; log score -1000" in text
    assert "about 0.607" in text and "about 0.135" in text
    data["scoring"]["threshold"] = {
        "kind": "log_score_ge",
        "log_value": -1000.0,
        "score_value": 0.0,
    }
    assert "underflow; log score -1000" in render_signature_explanation(
        SignatureExplanation(data)
    )


def test_corrupt_companion_and_signature_fail_without_partial_cli(explanation_library):
    root, result = explanation_library
    file = result.relative_dir / "clusters/K0/C_log_scores.npy"
    file.write_bytes(b"corrupt")
    with pytest.raises(IntegrityError):
        explain_signature("K0", root)
    response = CliRunner().invoke(
        app, ["explain-signature", "--cluster", "K0", "--library-dir", str(root)]
    )
    assert response.exit_code == 1
    assert response.stdout == ""
    assert response.stderr


def test_legacy_rejected_before_six_fit_loader(tmp_path):
    root = tmp_path / "legacy"
    write_json(
        root / "selection.json",
        {"schema": "pclass-dark.library-selection.v1", "library": "old"},
    )
    write_json(root / "old/manifest.json", {"schema": "pclass-dark.library.v1"})
    with pytest.raises(IntegrityError, match="build-library to migrate"):
        explain_signature("K0", root)


def test_missing_paths_no_creation_and_cli_validation(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        explain_signature("K", missing)
    assert not missing.exists()
    monkeypatch.setattr(
        "pclass_dark.explanations.explain_signature",
        lambda *a, **k: pytest.fail("unexpected library read"),
    )
    for options, code in [
        ([], 2),
        (["-h"], 0),
        (["--help"], 0),
        (["--cluster", "K", "--cluster", "K"], 2),
        (["--cluster", "K", "--format", "invalid"], 2),
    ]:
        response = CliRunner().invoke(app, ["explain-signature", *options])
        assert response.exit_code == code, response.output


def test_immutable_bundled_root_without_heavy_imports(explanation_library, tmp_path):
    root, _ = explanation_library
    resources = tmp_path / "package/resources"
    bundled = resources / "library"
    shutil.copytree(root, bundled, ignore=shutil.ignore_patterns(".*"))
    script = """
import builtins, importlib.abc, json, os, pathlib, socket, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch','transformers','huggingface_hub'}:
            raise AssertionError('heavy import: '+fullname)
sys.meta_path.insert(0, Block())
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network'))
import pclass_dark.library as library
from pclass_dark.explanations import explain_signature, render_signature_explanation
library.__file__ = str(pathlib.Path(sys.argv[1]) / 'library.py')
root = pathlib.Path(sys.argv[1]) / 'resources/library'
original_open = builtins.open
original_os_open = os.open
def checked_open(file, mode='r', *args, **kwargs):
    if any(c in mode for c in 'wax+'):
        raise AssertionError('write: '+str(file))
    return original_open(file, mode, *args, **kwargs)
def checked_os_open(file, flags, *args, **kwargs):
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
        raise AssertionError('write: '+str(file))
    return original_os_open(file, flags, *args, **kwargs)
builtins.open = checked_open
os.open = checked_os_open
report = explain_signature('K0', root)
result = json.loads(render_signature_explanation(report, format='json'))
print(result['source']['cluster_id'])
"""
    response = subprocess.run(
        [sys.executable, "-B", "-c", script, str(resources.parent)],
        text=True,
        capture_output=True,
    )
    assert response.returncode == 0, response.stderr
    assert response.stdout.strip() == "K0"
    assert not list(bundled.glob(".*"))


def test_reader_cannot_observe_concurrent_publication(explanation_library):
    from concurrent.futures import ThreadPoolExecutor

    from pclass_dark.artifacts import transaction_guard

    root, _ = explanation_library
    with transaction_guard(root), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(explain_signature, "K0", root)
        with pytest.raises(RuntimeError, match="Setup busy"):
            future.result(timeout=10)


def test_trace_binding_and_missing_companion_are_errors(explanation_library):
    root, result = explanation_library
    selector = read_json(root / "selection.json")
    (result.relative_dir / "manifest.json").unlink()
    with pytest.raises(OSError):
        explain_signature("K0", root)
    # Intentionally remove the companion dependency, then corrupt a signature.
    selector["schema"] = "pclass-dark.library-selection.v1"
    write_json(root / "selection.json", selector)
    (result.library_dir / "clusters/K0/final.json").write_text("{}")
    with pytest.raises(IntegrityError):
        explain_signature("K0", root)


def test_zero_variance_and_ties_have_honest_evidence(library_data, tmp_path):
    from pclass_dark.artifacts import file_record

    data, _, embeddings, matrix, _ = library_data(count=4, min_clusters=1)
    np.save(embeddings / "embeddings.npy", np.zeros_like(matrix), allow_pickle=False)
    manifest = read_json(embeddings / "manifest.json")
    manifest["files"]["embeddings.npy"] = file_record(embeddings / "embeddings.npy")
    write_json(embeddings / "manifest.json", manifest)
    selected = read_json(data / "selection.json")
    from pclass_dark.artifacts import sha256

    selected["embedding_manifest_sha256"] = sha256(embeddings / "manifest.json")
    write_json(data / "selection.json", selected)
    root = tmp_path / "constant-library"
    build_library(data, root)
    report = explain_signature("K0", root)
    evidence = report.to_dict()
    assert evidence["selection"]["core"] == [0, 1]
    assert evidence["selection"]["best_prefix_length"] == 1
    for record in evidence["dimensions"]["selected"]:
        assert record["sigma_C"] == record["sigma_B"] == 0
        assert record["effective_sigma"] == 1e-8
        assert record["R"] == record["D"] == 0
        assert record["training_auroc"] == 0.5
    assert evidence["metrics"]["training"]["precision"] is None
    assert evidence["scoring"]["threshold"]["kind"] == "above_maximum"
    assert all(q["score"] == 1 for q in evidence["distributions"]["C"]["quantiles"])
    assert "0.5" in render_signature_explanation(report)


def test_tiny_selection_gain_is_visible(explanation_library):
    evidence = explain_signature("K0", explanation_library[0]).to_dict()
    evidence["selection"]["accepted_removals"] = [
        {
            "dimension": 10,
            "remaining_count": 2,
            "before_auroc": {"value": 0.98, "denominator": 100000000000},
            "after_auroc": {"value": 0.98000000001, "denominator": 100000000000},
            "gain_numerator": 1,
        }
    ]
    text = render_signature_explanation(SignatureExplanation(evidence))
    assert "1.000000e-11" in text


@pytest.mark.parametrize("cluster", [None, "", 1, ["K0"]])
def test_invalid_cluster_rejected_before_artifact_reads(cluster, tmp_path):
    with pytest.raises(ValueError, match="nonempty string"):
        explain_signature(cluster, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()


def test_symlink_root_and_nonfinite_report_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(IntegrityError, match="Symlink"):
        explain_signature("K0", alias)
    with pytest.raises(ValueError):
        SignatureExplanation({"bad": float("nan")})


def test_targeted_and_full_evidence_agree(explanation_library, monkeypatch):
    import pclass_dark.relative_scores as scores

    root, _ = explanation_library
    full = load_selected_library(root)
    scoped = load_selected_library(root, cluster_id="K1")
    assert set(scoped.library.signatures) == {"K1"}
    assert scoped.library.counts["targets"] == 1
    assert full.library.target_result is None
    assert len(scoped.library.target_result["profiles"]) == 640
    with pytest.raises(TypeError):
        scoped.library.target_result["signature"]["cluster_id"] = "changed"
    assert scoped.library.signatures["K1"] == full.library.signatures["K1"]
    assert scoped.relative.summarize("K1") == full.relative.summarize("K1")
    report = explain_signature("K1", root, detailed=True).to_dict()
    assert report["source"]["validation_scope"] == "cluster"
    # Drive the same report assembly from the full verified pair as an oracle.
    monkeypatch.setattr(scores, "load_selected_library", lambda *a, **k: full)
    assert explain_signature("K1", root, detailed=True).to_dict() == report


def test_targeted_explanation_does_not_read_other_cluster_files(
    explanation_library, monkeypatch
):
    import builtins
    import io
    import os

    root, _ = explanation_library
    real_open, real_io_open = builtins.open, io.open
    accessed = []

    def record(file):
        if isinstance(file, (str, bytes, os.PathLike)):
            path = Path(os.fsdecode(file))
            assert "K1" not in path.parts, f"Read unrelated cluster file: {path}"
            assert path.name not in {"cluster_metrics.tsv", "summary.tsv"}, path
            accessed.append(path)

    def checked_open(file, *args, **kwargs):
        record(file)
        return real_open(file, *args, **kwargs)

    def checked_io_open(file, *args, **kwargs):
        record(file)
        return real_io_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", checked_open)
    monkeypatch.setattr(io, "open", checked_io_open)
    report = explain_signature("K0", root)
    assert report.to_dict()["status"] == "ok"
    assert any(p.name == "dimension_final.tsv.gz" for p in accessed)
    assert any(p.name == "B_log_scores.npy" for p in accessed)
    # One binary checksum pass, one profile lookup and one short header check.
    # There must not be a second complete profile lookup for report assembly.
    assert sum(p.name == "dimension_final.tsv.gz" for p in accessed) == 3


def test_unrelated_corruption_does_not_block_targeted_report(explanation_library):
    root, result = explanation_library
    # Remove all of the other cluster's signature/trace/relative files and
    # redundant shared reports. They are outside the requested evidence scope.
    shutil.rmtree(result.library_dir / "clusters/K1")
    shutil.rmtree(result.relative_dir / "clusters/K1")
    (result.library_dir / "cluster_metrics.tsv").unlink()
    (result.relative_dir / "summary.tsv").unlink()
    assert explain_signature("K0", root).to_dict()["status"] == "ok"
    with pytest.raises((IntegrityError, OSError)):
        load_selected_library(root)
    with pytest.raises((IntegrityError, OSError)):
        explain_signature("K1", root)


@pytest.mark.parametrize(
    "member",
    [
        "trace",
        "distribution",
        "background",
        "profiles",
        "manifest_binding",
    ],
)
def test_targeted_dependencies_remain_verified(explanation_library, member):
    root, result = explanation_library
    if member == "manifest_binding":
        selected = read_json(root / "selection.json")
        selected["manifest_sha256"] = "0" * 64
        write_json(root / "selection.json", selected)
    else:
        file = {
            "trace": result.library_dir / "clusters/K0/selection_trace.json.gz",
            "distribution": result.relative_dir / "clusters/K0/distribution.json",
            "background": result.relative_dir / "clusters/K0/B_log_scores.npy",
            "profiles": result.library_dir / "dimension_final.tsv.gz",
        }[member]
        file.write_bytes(b"corrupt")
    with pytest.raises(IntegrityError):
        explain_signature("K0", root)
