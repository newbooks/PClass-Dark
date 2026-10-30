"""Final-only production, empirical score evidence and optional research."""

import numpy as np
import pytest

from pclass_dark import library_build
from pclass_dark.artifacts import sha256
from pclass_dark.library import build_library, load_library
from pclass_dark.relative_scores import load_selected_library


def test_default_final_only_and_relative_distributions(
    library_data, tmp_path, monkeypatch, caplog
):
    caplog.set_level(20, logger="pclass_dark.library_build")
    data, _, _, matrix, contract = library_data()
    original = library_build.fit_signature
    calls = []

    def final_only(*args, **kwargs):
        assert kwargs["context"]["fit_id"] == "final"
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(library_build, "fit_signature", final_only)
    root = tmp_path / "library"
    result = build_library(data, root)
    for cluster in ("K0", "K1"):
        assert (
            f"Cluster {cluster} ABC match evidence: "
            "computing or reusing reference distributions"
        ) in caplog.messages
    assert len(calls) == 2
    assert result.research_dir is None
    from pclass_dark.artifacts import read_json

    metadata = read_json(result.library_dir / "manifest.json")
    assert metadata["evaluation_mode"] == "all-C-in-sample"
    assert metadata["research_role"] == "separate-optional-artifact"
    assert metadata["probability_calibrated"] is False
    assert result.counts["dimension_final_rows"] == 1280
    assert not (result.library_dir / "dimension_folds.tsv.gz").exists()
    assert not list((root / ".fits").glob("*/final/kernels.*.npy"))
    pair = load_selected_library(root)
    assert pair.library.library_id == result.library_id
    scores, _ = pair.library.score("K0", matrix, embedding_contract=contract)
    evidence = pair.relative.evaluate("K0", scores)
    c, b = scores[::2], scores[1::2]
    expected_tail = [(1 + np.sum(b >= s)) / (len(b) + 1) for s in scores]
    expected_percentile = [np.sum(c <= s) / len(c) for s in scores]
    np.testing.assert_array_equal(evidence.background_tail_fraction, expected_tail)
    np.testing.assert_array_equal(evidence.cluster_percentile_le, expected_percentile)
    assert not evidence.background_evidence.flags.writeable
    assert pair.relative.evaluate("K0", np.array([], dtype=float)).C_n == 10
    monkeypatch.setattr(
        library_build, "fit_signature", lambda *a, **k: pytest.fail("recomputed fit")
    )
    again = build_library(data, root)
    assert again.reused
    assert again.library_id == result.library_id
    assert load_library(result.library_dir).manifest_sha256 == sha256(
        result.library_dir / "manifest.json"
    )


def test_research_is_optional_and_preserves_production(
    library_data, tmp_path, monkeypatch
):
    from pclass_dark.library_reports import load_research

    data, *_ = library_data()
    root = tmp_path / "library"
    initial = build_library(data, root)
    selected = (root / "selection.json").read_bytes()
    source = sha256(initial.library_dir / "manifest.json")
    original = library_build.fit_signature
    calls = []

    def fold_only(*args, **kwargs):
        fit = kwargs["context"]["fit_id"]
        assert fit != "final"
        calls.append(fit)
        # This must be readable while research input/run locks are held.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as reader:
            pair = reader.submit(load_selected_library, root).result(timeout=10)
            assert pair.library.library_id == initial.library_id
        return original(*args, **kwargs)

    monkeypatch.setattr(library_build, "fit_signature", fold_only)
    result = build_library(data, root, cross_validate=True, cluster_ids=["K0"])
    assert calls == [f"fold_{f}" for f in range(5)]
    assert result.library_id == initial.library_id
    assert result.research_dir is not None
    assert (root / "selection.json").read_bytes() == selected
    assert sha256(initial.library_dir / "manifest.json") == source
    research = load_research(result.research_dir)
    assert research["counts"]["dimension_fold_rows"] == 3200
    assert research["counts"]["oof_rows"] == 10
    assert not (result.research_dir / "dimension_final.tsv.gz").exists()
    monkeypatch.setattr(
        library_build,
        "fit_signature",
        lambda *a, **k: pytest.fail("recomputed research"),
    )
    again = build_library(data, root, cross_validate=True, cluster_ids=["K0"])
    assert again.reused and again.research_id == result.research_id


def test_research_failure_keeps_production_usable(library_data, tmp_path, monkeypatch):
    data, *_ = library_data()
    root = tmp_path / "library"
    initial = build_library(data, root)
    selected = (root / "selection.json").read_bytes()
    monkeypatch.setattr(
        library_build,
        "fit_signature",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("research failure")),
    )
    with pytest.raises(
        library_build.ResearchBuildError, match="Production remains usable"
    ) as error:
        build_library(data, root, cross_validate=True, cluster_ids=["K0"])
    assert error.value.production_result.library_id == initial.library_id
    assert (root / "selection.json").read_bytes() == selected
    assert load_selected_library(root).relative is not None


def test_legacy_private_and_published_import_without_discovery(
    library_data, tmp_path, monkeypatch
):
    import shutil

    from pclass_dark.library import _build_legacy_library

    data, *_ = library_data()
    root = tmp_path / "library"
    legacy = _build_legacy_library(data, root)
    legacy_manifest = sha256(legacy.library_dir / "manifest.json")
    legacy_final = (legacy.library_dir / "clusters/K0/final.json").read_bytes()
    # A private final is reusable without its surrounding six-fit target receipt.
    for path in (root / ".checkpoints").glob("*/c_*/target.json"):
        path.unlink()
    monkeypatch.setattr(
        library_build,
        "fit_signature",
        lambda *a, **k: pytest.fail("discovery during import"),
    )
    first = build_library(data, root, cluster_ids=["K0"])
    assert first.reuse["imported"] == 1
    assert sha256(legacy.library_dir / "manifest.json") == legacy_manifest
    assert (legacy.library_dir / "clusters/K0/final.json").read_bytes() == legacy_final
    # Now use published results only for K1; no private source survives.
    shutil.rmtree(root / ".checkpoints")
    second = build_library(data, root, cluster_ids=["K1"])
    assert second.reuse["imported"] == 1
    assert second.reuse["computed"] == 0


def test_production_parallel_bytes_match_serial(library_data, tmp_path):
    from pclass_dark.artifacts import read_json

    data, *_ = library_data()
    serial = build_library(data, tmp_path / "serial")
    parallel = build_library(data, tmp_path / "parallel", workers=2, memory_mib=128)
    assert serial.library_id == parallel.library_id
    a, b = (
        read_json(x.library_dir / "manifest.json")["files"] for x in [serial, parallel]
    )
    assert a == b
    a, b = (
        read_json(x.relative_dir / "manifest.json")["files"] for x in [serial, parallel]
    )
    assert a == b


@pytest.mark.parametrize("value", [0, 1, None, "yes"])
def test_cross_validate_requires_bool(tmp_path, value):
    with pytest.raises(ValueError, match="booleans"):
        build_library(tmp_path / "absent", tmp_path / "out", cross_validate=value)


def test_pair_force_rollback_and_published_backfill(
    library_data, tmp_path, monkeypatch
):
    import shutil

    from pclass_dark.artifacts import read_json, write_json

    data, *_ = library_data()
    root = tmp_path / "library"
    initial = build_library(data, root)
    selected = (root / "selection.json").read_bytes()
    original_select = library_build._select

    def fail(*args, **kwargs):
        raise OSError("paired selection failure")

    monkeypatch.setattr(library_build, "_select", fail)
    with pytest.raises(OSError, match="paired selection failure"):
        build_library(data, root, force=True)
    assert (root / "selection.json").read_bytes() == selected
    assert load_selected_library(root).relative is not None
    monkeypatch.setattr(library_build, "_select", original_select)
    # Backfill from published v2 science only, without its private fit/distributions.
    source = read_json(root / "selection.json")
    source["schema"] = "pclass-dark.library-selection.v1"
    for key in ["relative", "relative_id", "relative_manifest_sha256"]:
        source.pop(key)
    write_json(root / "selection.json", source)
    for directory in [
        root / ".fits",
        root / ".relative-checkpoints",
        initial.relative_dir,
    ]:
        shutil.rmtree(directory)
    monkeypatch.setattr(
        library_build,
        "fit_signature",
        lambda *a, **k: pytest.fail("discovery in backfill"),
    )
    result = build_library(data, root)
    assert result.reuse["computed"] == 0 and result.reuse["imported"] == 2
    assert result.library_id == initial.library_id
    assert load_selected_library(root).relative is not None


def test_new_cleanup_requires_distribution_gate_and_preserves_arrays(
    library_data, tmp_path
):
    from pclass_dark.artifacts import read_json
    from pclass_dark.library_cleanup import clean_library_cache

    data, *_ = library_data()
    root = tmp_path / "library"
    result = build_library(data, root, cluster_ids=["K0"])
    target = read_json(
        next((root / ".production-checkpoints").glob("*/c_*/target.json"))
    )
    scratch = (root / target["result"]).parent / "kernels.B.npy"
    scratch.write_bytes(b"disposable")
    distribution = root / target["relative"] / "complete.json"
    saved = distribution.read_bytes()
    distribution.unlink()
    assert clean_library_cache(root)["skipped"] == 1
    assert scratch.exists()
    distribution.write_bytes(saved)
    assert clean_library_cache(root)["files"] == 1
    assert not scratch.exists()
    assert load_selected_library(root).relative.relative_id == result.relative_id


def test_unsupported_distribution_is_explicit(library_data, tmp_path):
    from pclass_dark.relative_scores import UnavailableDistributionError

    data, *_ = library_data(count=1, min_clusters=1)
    result = build_library(data, tmp_path / "library")
    assert result.counts["unsupported_final"] == 1
    assert not list(result.relative_dir.rglob("*.npy"))
    with pytest.raises(UnavailableDistributionError):
        load_selected_library(result.library_dir.parent).relative.evaluate(
            "K0", np.array([0.0])
        )


def test_production_switch_archives_stale_research_and_remains_distributable(
    library_data, tmp_path
):
    from build_support import shared_files

    data, *_ = library_data()
    project = tmp_path / "release"
    root = project / "data/library"
    initial = build_library(data, root, cluster_ids=["K0"], cross_validate=True)
    selector = root / "research-selection.json"
    previous = selector.read_bytes()
    research_manifest = sha256(initial.research_dir / "manifest.json")
    # Reusing the same production must retain its compatible research selection.
    build_library(data, root, cluster_ids=["K0"])
    assert selector.read_bytes() == previous
    with shared_files(project) as files:
        assert any(str(dst) == "library/research-selection.json" for _, dst in files)
    result = build_library(data, root, cluster_ids=["K1"])
    assert not selector.exists()
    assert sha256(initial.research_dir / "manifest.json") == research_manifest
    archived = list((root / ".archives").glob("research-selection.json-*"))
    assert len(archived) == 1
    assert archived[0].read_bytes() == previous
    assert load_selected_library(root).library.library_id == result.library_id
    with shared_files(project) as files:
        assert any(str(dst) == "library/selection.json" for _, dst in files)
        assert not any(str(dst).startswith("library/research") for _, dst in files)


def test_research_cannot_reselect_after_concurrent_production_switch(
    library_data, tmp_path, monkeypatch
):
    data, *_ = library_data()
    root = tmp_path / "library"
    original = library_build._run
    switched = []

    def switch_after_folds(targets, make_task, workers, output_root, **kwargs):
        paths = original(targets, make_task, workers, output_root, **kwargs)
        if "fold_inputs" in make_task(targets[0]):
            switched.append(build_library(data, root, cluster_ids=["K1"]))
        return paths

    monkeypatch.setattr(library_build, "_run", switch_after_folds)
    with pytest.raises(
        library_build.ResearchBuildError,
        match="Selected production changed during research",
    ):
        build_library(data, root, cluster_ids=["K0"], cross_validate=True)
    assert load_selected_library(root).library.library_id == switched[0].library_id
    assert not (root / "research-selection.json").exists()


def test_research_rebinds_report_after_production_scope_change(
    library_data, tmp_path, monkeypatch
):
    from build_support import shared_files

    from pclass_dark.artifacts import IntegrityError, read_json, write_json
    from pclass_dark.library_reports import load_research

    data, *_ = library_data()
    project = tmp_path / "release"
    root = project / "data/library"
    initial = build_library(data, root, cluster_ids=["K0"], cross_validate=True)
    old_manifest = sha256(initial.research_dir / "manifest.json")
    full = build_library(data, root)
    original = library_build.fit_signature
    calls = []

    def fold_only(*args, **kwargs):
        fit = kwargs["context"]["fit_id"]
        assert fit != "final"
        calls.append(fit)
        return original(*args, **kwargs)

    monkeypatch.setattr(library_build, "fit_signature", fold_only)
    result = build_library(data, root, cluster_ids=["K0"], cross_validate=True)
    assert calls == [f"fold_{f}" for f in range(5)]
    assert result.library_id == full.library_id != initial.library_id
    assert result.research_id != initial.research_id
    assert sha256(initial.research_dir / "manifest.json") == old_manifest
    report = load_research(result.research_dir)
    selector = read_json(root / "research-selection.json")
    assert report["source_production"]["id"] == selector["production_library_id"]
    assert (
        report["source_production"]["manifest_sha256"]
        == selector["production_manifest_sha256"]
        == sha256(full.library_dir / "manifest.json")
    )
    with shared_files(project) as files:
        assert any(str(dst) == "library/research-selection.json" for _, dst in files)
    monkeypatch.setattr(
        library_build,
        "fit_signature",
        lambda *a, **k: pytest.fail("recomputed same-production research"),
    )
    again = build_library(data, root, cluster_ids=["K0"], cross_validate=True)
    assert again.reused and again.research_id == result.research_id
    # Top-level provenance must agree with its identity-bound counterpart.
    manifest_file = result.research_dir / "manifest.json"
    manifest = read_json(manifest_file)
    manifest["source_production"]["id"] = initial.library_id
    write_json(manifest_file, manifest)
    with pytest.raises(IntegrityError, match="Research production provenance mismatch"):
        load_research(result.research_dir)


def test_legacy_research_imports_empty_validation_folds(
    library_data, tmp_path, monkeypatch
):
    import shutil

    from pclass_dark.artifacts import read_json
    from pclass_dark.library import _build_legacy_library
    from pclass_dark.library_reports import load_research

    data, *_ = library_data(count=4, min_clusters=1)
    root = tmp_path / "library"
    legacy = _build_legacy_library(data, root)
    source_digest = sha256(legacy.library_dir / "manifest.json")
    # Require published migration; no private receipts may hide the empty folds.
    shutil.rmtree(root / ".checkpoints")
    monkeypatch.setattr(
        library_build,
        "fit_signature",
        lambda *a, **k: pytest.fail("recomputed published legacy signature"),
    )
    result = build_library(data, root, cross_validate=True)
    report = load_research(result.research_dir)
    assert report["counts"]["targets"] == 2
    assert report["counts"]["oof_rows"] == 4
    assert sha256(legacy.library_dir / "manifest.json") == source_digest
    assert load_selected_library(root).library.library_id == result.library_id
    empty = populated = 0
    for file in result.research_dir.glob("clusters/*/fold_*.json"):
        signature = read_json(file)
        count = signature["counts"]["validation_positive_n"]
        if count == 0:
            empty += 1
        else:
            populated += 1
        assert count in (0, 1)
    assert (empty, populated) == (6, 4)
    receipts = list((root / ".research-checkpoints").glob("*/c_*/fold_*/complete.json"))
    assert len(receipts) == 10
    for file in receipts:
        record = read_json(file)["payload"]
        assert (
            len(record["oof"]) == record["signature"]["counts"]["validation_positive_n"]
        )


def test_signature_carries_reference_for_live_description(library_data, tmp_path):
    from pclass_dark.artifacts import read_json
    from pclass_dark.cluster_descriptions import description_for

    data, prepared, *_ = library_data()
    result = build_library(data, tmp_path / "library")
    signature = read_json(result.library_dir / "clusters/K0/final.json")
    reference = read_json(prepared.prepared_dir / "manifest.json")["reference"]["files"]
    assert signature["input_identity"]["reference"] == reference
    catalog = {
        "reference": reference,
        "clusters": {"K0": {"description": "current function"}},
    }
    assert description_for(signature, catalog) == ("current function", None)
    catalog["reference"] = {
        **reference,
        "clusters.tsv": {"sha256": "a" * 64, "size_bytes": 100},
    }
    assert description_for(signature, catalog)[1] == "description_reference_mismatch"
