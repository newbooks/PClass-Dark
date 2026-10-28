"""Durable-result gating, narrow deletion, and safe cache-maintenance retries."""

import pytest

from pclass_dark.artifacts import (
    IntegrityError,
    SetupBusyError,
    identity,
    locked,
    read_json,
    write_json,
)
from pclass_dark.library_checkpoints import Checkpoints
from pclass_dark.library_cleanup import FITS, clean_library_cache, cleanup_target_cache


@pytest.fixture
def complete_cache(tmp_path):
    root = tmp_path / "library"
    library_id, cluster_id = "a" * 64, "K0"
    checkpoint = root / ".checkpoints" / library_id / ("c_" + cluster_id.encode().hex())
    results = []
    for fit in FITS:
        trace = {"fit_id": fit}
        signature = {
            "schema": "pclass-dark.signature.v1",
            "library_id": library_id,
            "cluster_id": cluster_id,
            "fit_id": fit,
            "status": "empty_background",
            "selected_dimensions": [],
            "threshold": None,
            "input_identity": {},
            "selection_trace_identity": identity(trace),
        }
        result = {
            "signature": signature,
            "trace": trace,
            "profiles": [{"dimension": d} for d in range(640)],
            "oof": [],
        }
        store = Checkpoints(
            checkpoint / fit, identity([library_id, cluster_id, fit, {}])
        )
        store.write("complete", "fit-version", result)
        store.write("profiles-0016", "profiles", {"rows": []})
        for name in [
            "kernels.C.npy",
            "kernels.B.npy",
            "prefix-0016.B.npy",
            "prefix-0016.C.npy",
            "kernels.json",
            "prefix-0016.json",
        ]:
            (store.path / name).write_bytes(b"disposable")
        (store.path / "unknown.npy").write_bytes(b"preserve")
        results.append(result)
    write_json(checkpoint / "target.json", {"cluster_id": cluster_id, "fits": results})
    return root, checkpoint, library_id, cluster_id


def test_cleanup_preserves_complete_results_profiles_and_unknown_files(complete_cache):
    root, checkpoint, library_id, cluster_id = complete_cache
    before = {
        p: p.read_bytes()
        for p in checkpoint.rglob("*.json")
        if p.name not in {"kernels.json", "prefix-0016.json"}
    }
    result = clean_library_cache(root)
    assert result == {"targets": 1, "files": 36, "bytes": 36 * 10, "skipped": 0}
    assert all(p.read_bytes() == data for p, data in before.items())
    assert len(list(checkpoint.glob("*/unknown.npy"))) == 6
    assert cleanup_target_cache(checkpoint, library_id, cluster_id)["bytes"] == 0


@pytest.mark.parametrize(
    "damage", ["missing_target", "missing_complete", "payload", "binding"]
)
def test_incomplete_or_damaged_target_is_preserved(complete_cache, damage):
    root, checkpoint, library_id, cluster_id = complete_cache
    if damage == "missing_target":
        (checkpoint / "target.json").unlink()
    elif damage == "missing_complete":
        (checkpoint / "final/complete.json").unlink()
    elif damage == "payload":
        p = checkpoint / "final/complete.json"
        receipt = read_json(p)
        receipt["payload_id"] = "invalid"
        write_json(p, receipt)
    else:
        p = checkpoint / "target.json"
        target = read_json(p)
        target["fits"][-1]["signature"]["cluster_id"] = "wrong"
        write_json(p, target)
    before = {p: p.read_bytes() for p in checkpoint.rglob("*") if p.is_file()}
    with pytest.raises((IntegrityError, FileNotFoundError)):
        cleanup_target_cache(checkpoint, library_id, cluster_id)
    assert clean_library_cache(root)["skipped"] == 1
    assert all(p.read_bytes() == data for p, data in before.items())


def test_cleanup_rejects_symlink_before_any_deletion(complete_cache, tmp_path):
    root, checkpoint, library_id, cluster_id = complete_cache
    external = tmp_path / "external.npy"
    external.write_bytes(b"important")
    p = checkpoint / "final/kernels.C.npy"
    p.unlink()
    p.symlink_to(external)
    with pytest.raises(IntegrityError):
        cleanup_target_cache(checkpoint, library_id, cluster_id)
    assert (checkpoint / "fold_0/kernels.C.npy").exists()
    assert external.read_bytes() == b"important"


def test_cleanup_retry_after_partial_unlink_failure(complete_cache, monkeypatch):
    from pathlib import Path

    root, checkpoint, library_id, cluster_id = complete_cache
    original = Path.unlink
    count = 0

    def fail_once(path, *args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError("injected deletion failure")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    with pytest.raises(OSError, match="deletion failure"):
        cleanup_target_cache(checkpoint, library_id, cluster_id)
    monkeypatch.setattr(Path, "unlink", original)
    assert clean_library_cache(root)["files"] == 34
    assert (checkpoint / "target.json").exists()


def test_cleanup_respects_root_writer_lock(complete_cache):
    root, checkpoint, _, _ = complete_cache
    with locked(root, "setup-transaction"):
        with pytest.raises(SetupBusyError):
            clean_library_cache(root)
    assert (checkpoint / "fold_0/kernels.C.npy").exists()


def test_cleanup_respects_cluster_writer_lock(complete_cache):
    root, checkpoint, library_id, cluster_id = complete_cache
    with locked(root, "library-fit-" + identity([library_id, cluster_id])):
        with pytest.raises(SetupBusyError):
            clean_library_cache(root)
    assert (checkpoint / "fold_0/kernels.C.npy").exists()
