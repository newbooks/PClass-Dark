"""Portable library orchestration, scientific isolation, and recovery tests."""

from __future__ import annotations

import csv
import gzip

import numpy as np
import pytest

from pclass_dark import library
from pclass_dark.artifacts import (
    IntegrityError,
    SetupBusyError,
    file_record,
    identity,
    read_json,
    sha256,
    transaction_guard,
    write_json,
)
from pclass_dark.embeddings import ASSET_HASHES
from pclass_dark.library import _build_legacy_library as build_library
from pclass_dark.library import load_library
from pclass_dark.library_checkpoints import Checkpoints
from pclass_dark.library_runtime import (
    UnavailableSignatureError,
    representation_contract,
    score_signature,
)
from pclass_dark.setup_data import MODEL, REVISION
from pclass_dark.signatures import fit_signature


def test_build_library_fold_metrics_final_refit_and_loaded_scores(
    library_data, tmp_path
):
    data, prepared, ep, matrix, contract = library_data()
    result = build_library(data, tmp_path / "library")
    assert result.counts["targets"] == 2
    assert result.counts["available_final"] == 2
    assert result.counts["dimension_fold_rows"] == 2 * 5 * 640
    assert result.counts["dimension_final_rows"] == 2 * 640
    assert result.counts["oof_rows"] == 20
    private = tmp_path / "library" / ".checkpoints" / result.library_id
    assert len(list(private.glob("c_*/target.json"))) == 2
    assert len(list(private.glob("c_*/*/complete.json"))) == 12
    assert not list(private.glob("c_*/*/kernels.*.npy"))
    assert not list(private.glob("c_*/*/prefix-*.npy"))
    assert not list(private.glob("c_*/*/prefix-*.json"))
    loaded = load_library(result.library_dir)
    for cluster, signature in loaded.signatures.items():
        assert signature["fit_id"] == "final"
        assert signature["counts"]["training_positive_n"] == 10
        assert signature["validation_metrics"] is None
        assert signature["validation_status"] == "not_applicable_full_refit"
        assert list(signature["selected_dimension_ids"]) == [0, 1]
        assert signature["training_metrics"]["auroc"] == 1
        scores, decisions = score_signature(
            signature, matrix, embedding_contract=contract
        )
        assert scores.shape == (20,) and decisions.dtype == bool
        positive = np.arange(20) % 2 == int(cluster[-1])
        assert decisions[positive].all() and not decisions[~positive].any()
        mu = float(np.mean(matrix[positive, 0].astype(np.float64)))
        assert signature["selected_dimensions"][0]["mu"] == mu
        with pytest.raises(TypeError):
            signature["status"] = "changed"
    before = read_json(result.library_dir / "manifest.json")
    assert build_library(data, tmp_path / "library").reused
    assert read_json(result.library_dir / "manifest.json") == before
    with gzip.open(result.library_dir / "dimension_summary.tsv.gz", "rt") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 2 * 640 * 7
    assert {r["valid_folds"] for r in rows} == {"5"}


def test_scoped_build_uses_all_background_sources_and_parallel_is_identical(
    library_data, tmp_path
):
    data, *_ = library_data()
    serial = build_library(data, tmp_path / "serial")
    parallel = build_library(data, tmp_path / "parallel", workers=2, memory_mib=128)
    assert serial.library_id == parallel.library_id
    sm = read_json(serial.library_dir / "manifest.json")
    pm = read_json(parallel.library_dir / "manifest.json")
    assert sm["files"] == pm["files"]
    scoped = build_library(data, tmp_path / "scoped", cluster_ids=["K0"])
    assert scoped.library_id != serial.library_id
    assert (
        load_library(scoped.library_dir).signatures["K0"]["counts"]["background_n"]
        == 10
    )


def test_validation_perturbation_cannot_change_training_fit(tmp_path):
    matrix = np.zeros((12, 640), dtype="<f4")
    matrix[:6, :2] = np.arange(6)[:, None] * 0.01
    matrix[6:, :2] = 10
    matrix[:2, :2] = [[0.03], [0.04]]
    idx = {"C": np.arange(2, 6), "B": np.arange(6, 12), "V": np.arange(2)}
    contract = {
        "model": MODEL,
        "revision": REVISION,
        "assets": dict(ASSET_HASHES),
        "layer": 30,
        "width": 640,
        "window_length": 1022,
        "pooling": "attention-and-not-special-mean-f32-v1",
        "residue_policy": "exact-vocab-ACDEFGHIKLMNPQRSTVWYXBZUO-v1",
        "inference_dtype": "float32",
        "output_dtype": "<f4",
    }
    ctx = {
        "library_id": "fixture",
        "cluster_id": "K",
        "fit_id": "fold_0",
        "epsilon": 1e-8,
        "sigma_floor": 1e-8,
        "input_identity": {},
        "representation_contract_id": identity(representation_contract(contract)),
        "counts": {
            "training_positive_n": 4,
            "validation_positive_n": 2,
            "background_n": 6,
        },
        "validation_ids": ["a", "b"],
    }
    bm = [(10.0, 0.0) if d < 2 else (0.0, 0.0) for d in range(640)]
    a = fit_signature(
        matrix,
        idx,
        store=Checkpoints(tmp_path / "a", "fit"),
        context=ctx,
        background_moments=bm,
    )
    changed = matrix.copy()
    changed[:2, :2] = 100
    b = fit_signature(
        changed,
        idx,
        store=Checkpoints(tmp_path / "b", "fit"),
        context=ctx,
        background_moments=bm,
    )
    for key in ("threshold", "training_metrics", "selected_dimension_ids"):
        assert a["signature"][key] == b["signature"][key]
    for key in ("mu_C", "sigma_C", "R", "D", "training_auroc", "rank", "selected"):
        assert [p[key] for p in a["profiles"]] == [p[key] for p in b["profiles"]]
    assert (
        a["signature"]["validation_metrics"]["recall"]
        != b["signature"]["validation_metrics"]["recall"]
    )


def test_resume_reuses_completed_fits_and_recovers_malformed_receipt(
    library_data, tmp_path, monkeypatch
):
    data, *_ = library_data()
    root = tmp_path / "library"
    original = library._write_outputs

    def fail(*args, **kwargs):
        raise OSError("injected final report failure")

    monkeypatch.setattr(library, "_write_outputs", fail)
    with pytest.raises(OSError):
        build_library(data, root, cluster_ids=["K0"])
    complete = next((root / ".checkpoints").glob("*/c_*/fold_0/complete.json"))
    write_json(complete, [])
    import pclass_dark.signatures as signatures

    original_profile = signatures.dimension_profile

    def no_profile(*args, **kwargs):
        raise AssertionError("Committed profile was recomputed")

    monkeypatch.setattr(signatures, "dimension_profile", no_profile)
    original_kernels = signatures._kernel_cache
    regenerated = []

    def record_kernels(*args, **kwargs):
        regenerated.append(1)
        return original_kernels(*args, **kwargs)

    monkeypatch.setattr(signatures, "_kernel_cache", record_kernels)
    monkeypatch.setattr(library, "_write_outputs", original)
    result = build_library(data, root, cluster_ids=["K0"], memory_mib=64)
    assert result.counts["available_final"] == 1
    assert list(complete.parent.glob(".quarantine/*"))
    assert len(regenerated) == 1  # Only the damaged fit needs deleted arrays rebuilt.
    monkeypatch.setattr(signatures, "dimension_profile", original_profile)


def test_corruption_contract_and_actual_library_root_guard(library_data, tmp_path):
    data, *_ = library_data()
    result = build_library(data, tmp_path / "library", cluster_ids=["K0"])
    from concurrent.futures import ThreadPoolExecutor

    with (
        transaction_guard(result.library_dir.parent),
        ThreadPoolExecutor(max_workers=1) as executor,
    ):
        future = executor.submit(load_library, result.library_dir)
        with pytest.raises(SetupBusyError):
            future.result()
    final = next(result.library_dir.glob("clusters/*/final.json"))
    signature = read_json(final)
    signature["selected_dimensions"][0]["effective_sigma"] = -1
    write_json(final, signature)
    manifest = read_json(result.library_dir / "manifest.json")
    manifest["files"][str(final.relative_to(result.library_dir))] = file_record(final)
    write_json(result.library_dir / "manifest.json", manifest)
    with pytest.raises(IntegrityError):
        load_library(result.library_dir)


def test_unsupported_final_and_missing_embeddings(library_data, tmp_path):
    data, prepared, *_ = library_data(count=1, min_clusters=1)
    result = build_library(data, tmp_path / "library")
    loaded = load_library(result.library_dir)
    assert result.counts["available_final"] == 0
    with pytest.raises(UnavailableSignatureError):
        score_signature(
            loaded.signatures["K0"],
            np.zeros((1, 640)),
            embedding_contract=loaded.embedding_contract,
        )
    selected = read_json(data / "selection.json")
    selected["embedding"] = None
    write_json(data / "selection.json", selected)
    with pytest.raises(IntegrityError):
        build_library(data, tmp_path / "none")


def test_private_checkpoint_array_binding_and_bad_structure(tmp_path):
    store = Checkpoints(tmp_path / "state", "task")
    store.write("step", "dep", {"x": 1}, arrays={"a": np.array([1.0, 2.0])})
    assert store.read("step", "dep")[0] == {"x": 1}
    receipt = read_json(store.path / "step.json")
    receipt["arrays"]["a"] = None
    write_json(store.path / "step.json", receipt)
    assert store.read("step", "dep") is None
    assert list((store.path / ".quarantine").iterdir())


def test_prefix_checkpoint_resumes_without_repeating_committed_scores(
    tmp_path, monkeypatch
):
    import pclass_dark.signatures as signatures
    from pclass_dark.profiles import auroc_counts

    rng = np.random.default_rng(4)
    kernels = {"C": -rng.random((640, 3)) * 4, "B": -rng.random((640, 5)) * 4}
    order = list(range(640))
    first = auroc_counts(kernels["C"][0], kernels["B"][0])
    profiles = [
        {
            "training_auroc": first["value"],
            "training_auroc_numerator": first["numerator"],
            "training_auroc_denominator": first["denominator"],
        }
        for _ in order
    ]
    store = Checkpoints(tmp_path / "prefix", "fit")
    original_write = store.write

    def fail(name, *args, **kwargs):
        if name == "prefix-0032":
            raise OSError("injected prefix checkpoint failure")
        return original_write(name, *args, **kwargs)

    monkeypatch.setattr(store, "write", fail)
    with pytest.raises(OSError):
        signatures._prefixes(kernels, order, profiles, store, "params")
    monkeypatch.setattr(store, "write", original_write)
    calls = []

    def measured(p, b):
        calls.append(1)
        return auroc_counts(p, b)

    monkeypatch.setattr(signatures, "auroc_counts", measured)
    result = signatures._prefixes(kernels, order, profiles, store, "params")
    assert len(calls) == 624
    assert len(result) == 640
    for length in (1, 16, 32, 640):
        p = np.exp(kernels["C"][:length]).mean(0)
        b = np.exp(kernels["B"][:length]).mean(0)
        assert result[length - 1]["numerator"] == auroc_counts(p, b)["numerator"]


def test_backward_trial_receipts_resume_with_same_decisions(tmp_path, monkeypatch):
    from pclass_dark.signatures import select_dimensions

    order = list(range(5))
    individual = {d: {"numerator": 99, "denominator": 100} for d in order}
    prefix = [
        {"numerator": n, "denominator": 100, "value": n / 100}
        for n in [80, 81, 82, 83, 84]
    ]
    calls = []

    def evaluate_subset(dims):
        calls.append(tuple(dims))
        n = 95 - 10 * (2 in dims) - 5 * (3 in dims)
        return {"numerator": n, "denominator": 100, "value": n / 100}

    store = Checkpoints(tmp_path / "backward", "fit")
    original = store.write

    def fail(name, *args, **kwargs):
        if name == "decision-0000":
            raise OSError("injected decision failure")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(store, "write", fail)
    with pytest.raises(OSError):
        select_dimensions(order, individual, prefix, evaluate_subset, store, "params")
    assert len(calls) == 2
    calls.clear()
    monkeypatch.setattr(store, "write", original)
    result = select_dimensions(
        order, individual, prefix, evaluate_subset, store, "params"
    )
    assert len(calls) == 2
    assert [r["removed"] for r in result["iterations"]] == [2, 3, None]


def test_force_publication_cannot_be_reused_until_rollback_finishes(
    library_data, tmp_path, monkeypatch
):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    data, *_ = library_data()
    root = tmp_path / "library"
    result = build_library(data, root, cluster_ids=["K0"])
    # Permitted provenance makes the preexisting manifest observably distinct.
    manifest = read_json(result.library_dir / "manifest.json")
    manifest["previous_release_note"] = "retained"
    write_json(result.library_dir / "manifest.json", manifest)
    selected = read_json(root / "selection.json")
    selected["manifest_sha256"] = sha256(result.library_dir / "manifest.json")
    write_json(root / "selection.json", selected)
    original_selection = (root / "selection.json").read_bytes()
    original_manifest = (result.library_dir / "manifest.json").read_bytes()
    published, finish = Event(), Event()
    original = library._select_library

    def fail(root_dir, target, manifest):
        published.set()
        assert finish.wait(timeout=20)
        raise OSError("injected library selection failure")

    monkeypatch.setattr(library, "_select_library", fail)
    with ThreadPoolExecutor(max_workers=1) as pool:
        writer = pool.submit(build_library, data, root, cluster_ids=["K0"], force=True)
        try:
            assert published.wait(timeout=30)
            with pytest.raises(SetupBusyError):
                build_library(data, root, cluster_ids=["K0"])
            with pytest.raises(SetupBusyError):
                load_library(result.library_dir)
        finally:
            finish.set()
        with pytest.raises(OSError, match="selection failure"):
            writer.result(timeout=30)
    assert (root / "selection.json").read_bytes() == original_selection
    assert (result.library_dir / "manifest.json").read_bytes() == original_manifest
    monkeypatch.setattr(library, "_select_library", original)
    assert build_library(data, root, cluster_ids=["K0"]).reused


@pytest.mark.parametrize(
    "option,value",
    [
        ("epsilon", 0),
        ("epsilon", float("nan")),
        ("sigma_floor", float("inf")),
        ("workers", True),
        ("memory_mib", 63),
    ],
)
def test_invalid_options_fail_before_loading_data(tmp_path, option, value):
    with pytest.raises(ValueError):
        build_library(tmp_path / "missing", tmp_path / "out", **{option: value})


def test_rolling_scheduler_refills_without_waiting_for_slow_target(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    release = Event()
    third_started = Event()
    first_finished = Event()
    made = []

    def make_task(cluster):
        made.append(cluster)
        return cluster

    def job(cluster):
        if cluster == 0:
            assert release.wait(10)
            first_finished.set()
        elif cluster == 2:
            third_started.set()
            assert release.wait(10)
        return f"result-{cluster}"

    monkeypatch.setattr(library, "_target_job", job)
    with ThreadPoolExecutor(max_workers=2) as workers:
        with ThreadPoolExecutor(max_workers=1) as coordinator:
            result = coordinator.submit(
                lambda: list(
                    library._completed_targets(workers, range(5), make_task, 2)
                )
            )
            try:
                assert third_started.wait(5), "Freed worker did not take next target"
                assert not first_finished.is_set()
                # First and third are blocked: no fourth descriptor is expanded.
                assert made == [0, 1, 2]
            finally:
                release.set()
            completed = result.result(timeout=10)
    assert completed[0] == (1, "result-1")
    assert dict(completed) == {i: f"result-{i}" for i in range(5)}


def test_rolling_scheduler_propagates_failure_without_refilling(monkeypatch):
    from concurrent.futures import Future

    made = []

    class Executor:
        def submit(self, function, task):
            future = Future()
            if task == 0:
                future.set_exception(RuntimeError("target failed"))
            return future

    def make_task(cluster):
        made.append(cluster)
        return cluster

    with pytest.raises(RuntimeError, match="target failed"):
        list(library._completed_targets(Executor(), range(5), make_task, 2))
    assert made == [0, 1]
