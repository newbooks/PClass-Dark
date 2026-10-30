"""Whole-generation retirement, shared inputs and crash recovery on tiny fixtures."""

import pytest

from pclass_dark.artifacts import IntegrityError, read_json, sha256, write_json
from pclass_dark.data_cleanup import (
    CleanupExecutionError,
    CleanupRecoveryRequiredError,
    CleanupStalePlanError,
    execute_cleanup,
    plan_cleanup,
    recover_cleanup,
)
from pclass_dark.library import build_library
from pclass_dark.relative_scores import load_selected_library

_template = None


@pytest.fixture
def generations(library_data, tmp_path, tmp_path_factory):
    global _template
    import shutil
    from dataclasses import replace

    if _template is None:
        built = _make_generations(library_data, tmp_path)
        template = tmp_path_factory.mktemp("cleanup-template")
        for name in ("data", "library", "reference"):
            shutil.copytree(tmp_path / name, template / name)
        _template = template, built
    template, built = _template
    work = tmp_path / "work"
    work.mkdir()
    for name in ("data", "library", "reference"):
        shutil.copytree(template / name, work / name)
    data, root = work / "data", work / "library"
    _, _, old, new, old_result, new_result, matrix, contract = built

    def rebound(result):
        return replace(
            result,
            library_dir=root / result.library_id,
            relative_dir=root / "relative-scores" / result.relative_id,
        )

    return (
        data,
        root,
        replace(old, prepared_dir=data / "prepared" / old.preparation_id),
        replace(new, prepared_dir=data / "prepared" / new.preparation_id),
        rebound(old_result),
        rebound(new_result),
        matrix,
        contract,
    )


def _make_generations(library_data, tmp_path, distinct_embedding=False):
    from pclass_dark.setup_data import prepare_data

    data, old, _, matrix, contract = library_data()
    root = tmp_path / "library"
    old_result = build_library(data, root)
    selection = read_json(data / "selection.json")
    reference = tmp_path / "reference"
    members = reference / "clusters.tsv"
    members.write_text(members.read_text().replace("K0", "K2"))
    m = read_json(reference / "manifest.json")
    from pclass_dark.artifacts import file_record

    m["files"]["clusters.tsv"] = file_record(members)
    write_json(reference / "manifest.json", m)
    new = prepare_data(data, reference_dir=reference, prepare_only=True)
    s = read_json(data / "selection.json")
    for key in ("embedding", "embedding_id", "embedding_manifest_sha256"):
        s[key] = selection[key]
    if distinct_embedding:
        import shutil

        from pclass_dark.artifacts import identity

        old_embedding = data / s["embedding"]
        em = read_json(old_embedding / "manifest.json")
        em["identity_inputs"]["contract"]["torch_version"] = "alternate-fixture"
        em["id"] = identity(em["identity_inputs"])
        em["originating_preparation_id"] = new.preparation_id
        target = data / "embeddings" / em["id"]
        shutil.copytree(old_embedding, target)
        write_json(target / "manifest.json", em)
        s.update(
            embedding="embeddings/" + em["id"],
            embedding_id=em["id"],
            embedding_manifest_sha256=sha256(target / "manifest.json"),
        )
    write_json(data / "selection.json", s)
    new_result = build_library(data, root)
    return data, root, old, new, old_result, new_result, matrix, contract


def test_plan_execute_shared_embedding_and_selected_scores(generations):
    data, root, old, new, old_result, new_result, matrix, contract = generations
    before = sha256(data / "selection.json"), sha256(root / "selection.json")
    plan = plan_cleanup(data, root)
    assert plan == plan_cleanup(data, root)
    removed = {e.key for e in plan.entries if e.disposition == "remove"}
    assert "setup:prepared/" + old.preparation_id in removed
    assert "library:" + old_result.library_id in removed
    assert "library:relative-scores/" + old_result.relative_id in removed
    assert not any(
        e.kind == "embedding" and e.disposition == "remove" for e in plan.entries
    )
    result = execute_cleanup(plan)
    assert result.status == "complete" and result.logical_bytes_removed > 0
    assert not old.prepared_dir.exists() and new.prepared_dir.exists()
    assert not old_result.library_dir.exists() and new_result.library_dir.exists()
    pair = load_selected_library(root)
    scores, _ = pair.library.score("K2", matrix, embedding_contract=contract)
    assert pair.relative.evaluate("K2", scores).C_n == 10
    assert before == (sha256(data / "selection.json"), sha256(root / "selection.json"))
    assert not execute_cleanup(plan_cleanup(data, root)).removed


def test_preview_no_mutation_and_extra_file_protection(generations):
    data, root, old, *_ = generations
    (old.prepared_dir / "notes.txt").write_text("retain me")

    def snapshot():
        return {
            str(p): sha256(p)
            for base in (data, root)
            for p in base.rglob("*")
            if p.is_file() and ".locks" not in p.parts
        }

    before = snapshot()
    plan = plan_cleanup(data, root)
    assert snapshot() == before
    entry = next(e for e in plan.entries if e.path == "prepared/" + old.preparation_id)
    assert entry.disposition == "protect" and "unexpected" in entry.reason


def test_stale_plan_and_forged_paths(generations):
    from dataclasses import replace

    data, root, old, *_ = generations
    plan = plan_cleanup(data, root)
    forged = replace(
        plan, entries=tuple(replace(e, path="../escape") for e in plan.entries)
    )
    with pytest.raises(CleanupStalePlanError):
        execute_cleanup(forged)
    (old.prepared_dir / "new.txt").write_text("changed")
    with pytest.raises(CleanupStalePlanError):
        execute_cleanup(plan)
    assert old.prepared_dir.exists()


@pytest.mark.parametrize("point", ["move", "remove_tree"])
def test_failure_rollback_or_recovery(generations, monkeypatch, point):
    import pclass_dark.data_cleanup as cleanup

    data, root, old, *_ = generations
    archive = root / ".archives"
    archive.mkdir()
    write_json(archive / "selection.json-backup", read_json(root / "selection.json"))
    plan = plan_cleanup(data, root)
    original = getattr(cleanup, point)
    count = []

    def fail(*args):
        original(*args)
        count.append(1)
        if len(count) == 1:
            raise OSError("injected failure after IO")

    with monkeypatch.context() as patch:
        patch.setattr(cleanup, point, fail)
        with pytest.raises(CleanupExecutionError) as error:
            execute_cleanup(plan)
    if point == "move":
        assert error.value.result.status == "rolled_back"
        assert old.prepared_dir.exists()
        assert archive.exists()
    else:
        assert error.value.result.recovery_required
        with pytest.raises(CleanupRecoveryRequiredError):
            load_selected_library(root)
        assert recover_cleanup(data, root).status == "complete"
        assert not old.prepared_dir.exists()
    load_selected_library(root)


def test_clean_option_reuses_build_and_invalid_scope(generations):
    data, root, old, *_ = generations
    with pytest.raises(ValueError, match="full build"):
        build_library(data, root, clean=True, cluster_ids=["K2"])
    with pytest.raises(ValueError, match="boolean"):
        build_library(data, root, clean=1)
    result = build_library(data, root, clean=True)
    assert result.reused and result.cleanup.status == "complete"
    assert not old.prepared_dir.exists()


def test_pending_markers_block_immutable_cache_and_packaging(generations):
    from build_support import shared_files

    data, root, *_ = generations
    pair = load_selected_library(root)
    import numpy as np

    pair.relative.evaluate("K2", np.array([0.0]))
    write_json(root / ".cleanup-journal.json", {"incomplete": True})
    for call in (
        lambda: load_selected_library(root, immutable=True),
        lambda: pair.relative.evaluate("K2", np.array([0.0])),
    ):
        with pytest.raises(CleanupRecoveryRequiredError):
            call()
    with pytest.raises(IntegrityError):
        recover_cleanup(data, root)
    # Packaging uses direct FileLock rather than the runtime guard.
    project = root.parent / "package"
    (project / "data/library").mkdir(parents=True)
    write_json(project / "data/library/.cleanup-journal.json", {})
    with pytest.raises(RuntimeError, match="Pending cleanup"):
        with shared_files(project):
            pass


@pytest.mark.parametrize("archived", ["old", "current"])
def test_valid_replacement_retires_archive_and_old_inputs(generations, archived):
    data, root, old, new, old_result, new_result, *_ = generations
    result = old_result if archived == "old" else new_result
    prep = old if archived == "old" else new
    selection = read_json(root / "selection.json")
    selection.update(
        preparation_id=prep.preparation_id,
        library_id=result.library_id,
        library=result.library_id,
        relative_id=result.relative_id,
        relative="relative-scores/" + result.relative_id,
    )
    archive = root / ".archives"
    archive.mkdir()
    write_json(archive / "selection.json-retained", selection)
    plan = plan_cleanup(data, root)
    entry = next(e for e in plan.entries if e.path == "prepared/" + old.preparation_id)
    assert entry.disposition == "remove"
    assert (
        next(e for e in plan.entries if e.path == ".archives").disposition == "remove"
    )
    execute_cleanup(plan)
    assert not archive.exists()
    assert not old.prepared_dir.exists()
    assert new.prepared_dir.exists()
    load_selected_library(root)


@pytest.mark.parametrize("kind", ["target", "fit"])
def test_partial_work_pins_old_preparation(generations, kind):
    data, root, old, _, old_result, *_ = generations
    if kind == "target":
        path = next(
            (root / ".production-checkpoints" / old_result.library_id).glob(
                "*/target.json"
            )
        )
    else:
        m = read_json(old_result.library_dir / "manifest.json")
        fit = m["identity_inputs"]["final_fits"][0][1]
        path = root / ".fits" / fit / "final/complete.json"
    path.unlink()
    plan = plan_cleanup(data, root)
    assert (
        next(
            e for e in plan.entries if e.path == "prepared/" + old.preparation_id
        ).disposition
        == "protect"
    )


def test_invalid_receipt_is_not_quarantined(generations):
    data, root, _, _, old_result, *_ = generations
    m = read_json(old_result.library_dir / "manifest.json")
    fit = m["identity_inputs"]["final_fits"][0][1]
    path = root / ".fits" / fit / "final/complete.json"
    path.write_text("{bad json")
    with pytest.raises(IntegrityError):
        plan_cleanup(data, root)
    assert path.read_text() == "{bad json"
    assert not list(root.rglob(".quarantine"))


def test_symlink_and_unknown_schema_preserved(generations, tmp_path):
    data, root, old, *_ = generations
    outside = tmp_path / "outside"
    outside.write_text("never touch")
    (old.prepared_dir / "extra").symlink_to(outside)
    plan = plan_cleanup(data, root)
    assert (
        next(
            e for e in plan.entries if e.path == "prepared/" + old.preparation_id
        ).disposition
        == "protect"
    )
    assert outside.read_text() == "never touch"
    m = read_json(old.prepared_dir / "manifest.json")
    m["schema"] = "future.schema"
    write_json(old.prepared_dir / "manifest.json", m)
    with pytest.raises(IntegrityError):
        plan_cleanup(data, root)


def test_busy_guard_and_same_thread_reentrance(generations):
    from filelock import FileLock

    from pclass_dark.artifacts import SetupBusyError, transaction_guard

    data, root, *_ = generations
    with FileLock(str(root / ".locks/setup-transaction.lock")):
        with pytest.raises(SetupBusyError):
            plan_cleanup(data, root)
    with transaction_guard(data), transaction_guard(root):
        assert plan_cleanup(data, root).entries


def test_invalid_clean_options_do_not_recover(generations, monkeypatch):
    data, root, *_ = generations
    import pclass_dark.data_cleanup as cleanup

    monkeypatch.setattr(
        cleanup, "recover_cleanup", lambda *a: pytest.fail("unexpected recovery")
    )
    with pytest.raises(ValueError, match="workers"):
        build_library(data, root, clean=True, workers=0)


def test_cli_clean_invalid_combinations(generations, monkeypatch):
    from typer.testing import CliRunner

    from pclass_dark.cli import app

    data, root, old, *_ = generations
    monkeypatch.setattr("pclass_dark.paths.default_setup_input_dir", lambda: data)
    monkeypatch.setattr("pclass_dark.paths.default_library_output_dir", lambda: root)
    for options in (["--force"], ["--cluster", "K2"], ["--cross-validate"]):
        result = CliRunner().invoke(app, ["build-library", "--clean", *options])
        assert result.exit_code == 1, result.output
    assert old.prepared_dir.exists()


@pytest.mark.parametrize(
    "operation,nth,after,committed",
    [
        ("prepared", 1, False, False),
        ("prepared", 1, True, False),
        ("peer", 1, False, False),
        ("peer", 1, True, False),
        ("detaching", 1, False, False),
        ("detaching", 1, True, False),
        ("move", 1, False, False),
        ("move", 1, True, False),
        ("move", 4, False, False),
        ("move", 4, True, False),
        ("committed", 1, False, False),
        ("committed", 1, True, True),
        ("remove_tree", 1, False, True),
        ("remove_tree", 1, True, True),
        ("remove_tree", 4, False, True),
        ("remove_tree", 4, True, True),
        ("done", 1, False, True),
        ("done", 1, True, True),
    ],
)
def test_process_death_and_recovery(generations, operation, nth, after, committed):
    import subprocess
    import sys

    data, root, old, *_ = generations
    script = r"""
import os, sys
from pathlib import Path
import pclass_dark.data_cleanup as c
op, nth, after = sys.argv[3], int(sys.argv[4]), sys.argv[5]=='True'
count = 0
def fault():
    global count
    count += 1
    if count == nth: os._exit(91)
if op in {'move','remove_tree'}:
    original = getattr(c, op)
    def wrapper(*args):
        if not after: fault()
        original(*args)
        if after: fault()
    setattr(c, op, wrapper)
elif op == 'peer':
    original = c.write_json
    def wrapper(path, value):
        matched = Path(path).name == c.PEER
        if matched and not after: fault()
        original(path,value)
        if matched and after: fault()
    c.write_json = wrapper
else:
    original = c._state
    def wrapper(roots,j,state):
        if state == op and not after: fault()
        original(roots,j,state)
        if state == op and after: fault()
    c._state = wrapper
c.execute_cleanup(c.plan_cleanup(sys.argv[1],sys.argv[2]))
"""
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(data),
            str(root),
            operation,
            str(nth),
            str(after),
        ],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 91, process.stderr
    result = recover_cleanup(data, root)
    if result:
        assert result.status == ("complete" if committed else "rolled_back")
    assert old.prepared_dir.exists() == (not committed)
    load_selected_library(root)


def test_force_clean_retires_validated_rollback_archives(generations):
    data, root, old, *_ = generations
    build_library(data, root, force=True)
    result = build_library(data, root, clean=True)
    assert result.cleanup.status == "complete"
    assert not (root / ".archives").exists()
    assert not old.prepared_dir.exists()
    load_selected_library(root)


def test_selected_explanation_survives_cleanup(generations):
    from pclass_dark.explanations import explain_signature

    data, root, *_ = generations
    before = explain_signature("K2", root).to_dict()
    execute_cleanup(plan_cleanup(data, root))
    assert explain_signature("K2", root).to_dict() == before


def test_hard_links_do_not_overclaim_freed_blocks(generations, tmp_path):
    import os

    data, root, old, *_ = generations
    file = old.prepared_dir / "windows.fasta"
    retained = tmp_path / "retained-link.fasta"
    os.link(file, retained)
    digest = sha256(file)
    plan = plan_cleanup(data, root)
    assert (
        next(
            e for e in plan.entries if e.path == "prepared/" + old.preparation_id
        ).allocated_bytes_estimate
        is not None
    )
    execute_cleanup(plan)
    assert retained.exists() and sha256(retained) == digest


def test_prebuild_recovery_and_successful_clean(generations, monkeypatch):
    import pclass_dark.data_cleanup as cleanup

    data, root, old, *_ = generations
    original = cleanup.remove_tree

    def fail(*args):
        original(*args)
        raise OSError("committed deletion stopped")

    with monkeypatch.context() as patch:
        patch.setattr(cleanup, "remove_tree", fail)
        with pytest.raises(CleanupExecutionError):
            execute_cleanup(plan_cleanup(data, root))
    monkeypatch.setattr(
        cleanup, "make_plan", lambda *a: pytest.fail("replanned completed recovery")
    )
    result = build_library(data, root, clean=True)
    assert result.cleanup.status == "complete"
    assert not old.prepared_dir.exists()


def test_cleanup_failure_reports_valid_production(generations, monkeypatch):
    import pclass_dark.data_cleanup as cleanup

    data, root, *_ = generations
    original = cleanup._after_build

    def changed(data, root, snapshot, result):
        write_json(
            root / "selection.json",
            {**read_json(root / "selection.json"), "extra": "changed"},
        )
        return original(data, root, snapshot, result)

    monkeypatch.setattr(cleanup, "_after_build", changed)
    with pytest.raises(CleanupExecutionError) as error:
        build_library(data, root, clean=True)
    assert error.value.production_result.library_dir.exists()
    assert isinstance(error.value.__cause__, CleanupStalePlanError)
    load_selected_library(root)


def test_resource_cap_fails_without_markers(generations, monkeypatch):
    import pclass_dark.data_cleanup as cleanup

    data, root, *_ = generations
    plan = plan_cleanup(data, root)
    monkeypatch.setattr(cleanup, "MAX_JSON", 10)
    with pytest.raises(IntegrityError, match="64 MiB"):
        execute_cleanup(plan)
    assert not (root / cleanup.JOURNAL).exists()
    assert not (data / cleanup.PEER).exists()


def test_exclusive_embedding_removed(library_data, tmp_path):
    data, root, *_ = _make_generations(library_data, tmp_path, distinct_embedding=True)
    selected = read_json(data / "selection.json")["embedding_id"]
    old = next(p for p in (data / "embeddings").iterdir() if p.name != selected)
    plan = plan_cleanup(data, root)
    assert (
        next(
            e for e in plan.entries if e.key == "setup:embeddings/" + old.name
        ).disposition
        == "remove"
    )
    execute_cleanup(plan)
    assert not old.exists() and (data / "embeddings" / selected).exists()
    load_selected_library(root)


def test_failure_before_first_journal_is_not_pending(generations, monkeypatch):
    import pclass_dark.data_cleanup as cleanup

    data, root, old, *_ = generations
    plan = plan_cleanup(data, root)

    def fail(*args):
        raise OSError("no journal could be written")

    monkeypatch.setattr(cleanup, "_state", fail)
    with pytest.raises(CleanupExecutionError) as error:
        execute_cleanup(plan)
    assert error.value.result.status == "rolled_back"
    assert not error.value.result.recovery_required
    assert old.prepared_dir.exists()


def test_unknown_archive_refuses_plan_without_mutation(generations):
    data, root, old, *_ = generations
    archive = root / ".archives"
    archive.mkdir()
    write_json(archive / "selection.json-unknown", {"schema": "future.selector"})
    with pytest.raises(IntegrityError, match="Unknown retained-history"):
        plan_cleanup(data, root)
    assert old.prepared_dir.exists()


def test_root_overlap_and_posix_limit(generations, monkeypatch):
    from types import SimpleNamespace

    import pclass_dark.data_cleanup as cleanup

    data, root, *_ = generations
    with pytest.raises(IntegrityError, match="disjoint"):
        plan_cleanup(data, data)
    plan = plan_cleanup(data, root)
    monkeypatch.setattr(cleanup, "os", SimpleNamespace(name="nt"))
    with pytest.raises(RuntimeError, match="POSIX"):
        execute_cleanup(plan)
    assert not (root / cleanup.JOURNAL).exists()


def test_interrupt_restores_before_commit(generations, monkeypatch):
    import pclass_dark.data_cleanup as cleanup

    data, root, old, *_ = generations
    plan = plan_cleanup(data, root)
    original = cleanup.move
    calls = []

    def interrupt(*args):
        original(*args)
        calls.append(1)
        if len(calls) == 1:
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(cleanup, "move", interrupt)
        with pytest.raises(KeyboardInterrupt):
            execute_cleanup(plan)
    assert old.prepared_dir.exists()
    assert not (root / cleanup.JOURNAL).exists()
    load_selected_library(root)


def test_research_clean_retains_complete_research(generations):
    from pclass_dark.library_reports import load_research

    data, root, old, *_ = generations
    built = build_library(data, root, cross_validate=True)
    result = build_library(data, root, clean=True)
    assert result.cleanup.status == "complete"
    assert not old.prepared_dir.exists()
    assert load_research(built.research_dir)["id"] == built.research_id


def test_lazy_cache_reloads_after_generation_retired(generations):
    import numpy as np

    from pclass_dark.library import load_library
    from pclass_dark.relative_scores import load_relative_scores

    data, root, _, _, old_result, *_ = generations
    lib = load_library(old_result.library_dir)
    rel = load_relative_scores(old_result.relative_dir, library=lib)
    rel.evaluate("K0", np.array([0.0]))
    execute_cleanup(plan_cleanup(data, root))
    with pytest.raises(IntegrityError, match="reload"):
        rel.evaluate("K0", np.array([0.0]))


@pytest.mark.parametrize("nth", [1, 2, 4, 8, 12, 20, 30, 35, 40, 50])
def test_process_death_at_fsync(generations, nth):
    import subprocess
    import sys

    import pclass_dark.data_cleanup as cleanup

    data, root, old, *_ = generations
    script = r"""
import os,sys
import pclass_dark.data_cleanup as c
plan=c.plan_cleanup(sys.argv[1],sys.argv[2])
nth=int(sys.argv[3]); count=0; original=os.fsync
def crash(fd):
    global count
    original(fd)
    count+=1
    if count==nth:os._exit(91)
os.fsync=crash
c.execute_cleanup(plan)
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(data), str(root), str(nth)],
        capture_output=True,
        text=True,
    )
    assert process.returncode == 91, process.stderr
    j = read_json(root / cleanup.JOURNAL) if (root / cleanup.JOURNAL).exists() else None
    committed = j is not None and j["state"] in {"committed", "done"}
    recover_cleanup(data, root)
    assert old.prepared_dir.exists() == (not committed)
    load_selected_library(root)


def test_unsupported_complete_targets_allow_clean(library_data, tmp_path):
    data, *_ = library_data(count=2, min_clusters=1)
    build_library(data, tmp_path / "library")
    result = build_library(data, tmp_path / "library", clean=True)
    assert result.cleanup.status == "complete"
    assert result.counts["unsupported_final"] == 2
    load_selected_library(tmp_path / "library")


def test_research_failure_never_enters_cleanup(generations, monkeypatch):
    import pclass_dark.data_cleanup as cleanup
    import pclass_dark.library_build as building

    data, root, old, *_ = generations
    original = building.fit_signature

    def fail(*args, **kwargs):
        if kwargs["context"]["fit_id"] != "final":
            raise RuntimeError("research failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(building, "fit_signature", fail)
    monkeypatch.setattr(
        cleanup, "_after_build", lambda *a: pytest.fail("failed research cleaned")
    )
    with pytest.raises(ValueError, match="cross-validate"):
        build_library(data, root, clean=True, cross_validate=True)
    assert old.prepared_dir.exists()
    load_selected_library(root)


def test_lfs_store_is_not_touched(generations):
    data, root, *_ = generations
    sentinel = root / ".git/lfs/objects/retained"
    sentinel.parent.mkdir(parents=True)
    sentinel.write_bytes(b"not a cleanup namespace")
    execute_cleanup(plan_cleanup(data, root))
    assert sentinel.read_bytes() == b"not a cleanup namespace"


def test_recovery_after_removing_one_candidate_hard_link(generations, monkeypatch):
    import os

    import pclass_dark.data_cleanup as cleanup

    data, root, _, _, old_result, *_ = generations
    m = read_json(old_result.library_dir / "manifest.json")
    cluster, fit_id = m["identity_inputs"]["final_fits"][0]
    published = old_result.relative_dir / "clusters" / cluster / "C_log_scores.npy"
    checkpoint = root / ".relative-checkpoints" / fit_id / "complete.C.npy"
    assert sha256(published) == sha256(checkpoint)
    checkpoint.unlink()
    os.link(published, checkpoint)
    digest = sha256(published)
    original = cleanup.remove_tree
    fired = []

    def stop(*args):
        target = args[0] / args[1]
        for file in target.rglob("*.npy"):
            if sha256(file) == digest and not fired:
                file.unlink()
                fired.append(True)
                raise OSError("one alias was unlinked")
        original(*args)

    plan = plan_cleanup(data, root)
    with monkeypatch.context() as patch:
        patch.setattr(cleanup, "remove_tree", stop)
        with pytest.raises(CleanupExecutionError) as error:
            execute_cleanup(plan)
    assert fired and error.value.result.logical_bytes_removed > 0
    assert recover_cleanup(data, root).status == "complete"
    load_selected_library(root)


def test_clean_never_builds_existing_or_missing(
    generations, tmp_path, monkeypatch, caplog
):
    import logging

    import pclass_dark.library_build as building

    data, root, *_ = generations
    monkeypatch.setattr(building, "build", lambda *a, **k: pytest.fail("cleanup built"))
    with caplog.at_level(logging.INFO):
        result = build_library(data, root, clean=True)
    assert result.reused and result.cleanup.status == "complete"
    assert "no library will be built" in caplog.text
    assert "Building " not in caplog.text
    missing = tmp_path / "absent-library"
    with pytest.raises(ValueError, match="No existing selected library"):
        build_library(data, missing, clean=True)
    assert not missing.exists()
    for options in ({"force": True}, {"cross_validate": True}):
        with pytest.raises(ValueError, match="cannot be combined"):
            build_library(data, root, clean=True, **options)


def test_cli_clean_missing_library_does_not_build(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from pclass_dark.cli import app

    data = tmp_path / "setup"
    data.mkdir()
    root = tmp_path / "library"
    monkeypatch.setattr("pclass_dark.paths.default_setup_input_dir", lambda: data)
    monkeypatch.setattr("pclass_dark.paths.default_library_output_dir", lambda: root)
    result = CliRunner().invoke(app, ["build-library", "--clean"])
    assert result.exit_code == 1
    assert "No existing selected library" in result.output
    assert not root.exists()


def test_clean_incomplete_library_never_repairs(generations, monkeypatch):
    import pclass_dark.library_build as building

    data, root, old, *_ = generations
    monkeypatch.setattr(building, "build", lambda *a, **k: pytest.fail("cleanup built"))
    selection = read_json(root / "selection.json")
    manifest = root / selection["relative"] / "manifest.json"
    manifest.unlink()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        build_library(data, root, clean=True)
    assert not manifest.exists()
    assert old.prepared_dir.exists()


def test_large_aggregate_inventory_uses_bounded_sidecars(generations, monkeypatch):
    import pclass_dark.cleanup_inventory as planning
    import pclass_dark.cleanup_io as io
    import pclass_dark.data_cleanup as cleanup
    from pclass_dark.artifacts import canonical

    data, root, old, *_ = generations
    expected, inventories = planning.make_plan(data, root)
    sizes = [len(canonical(tree)) for tree in inventories.values()]
    counts = [sum(len(v) for v in tree.values()) for tree in inventories.values()]
    inventories.close()
    # Allow each tree, but less than the aggregate: the former global limit failed.
    assert sum(counts) > max(counts) + 5
    monkeypatch.setattr(io, "MAX_ENTRIES", max(counts) + 5)
    monkeypatch.setattr(io, "MAX_INVENTORY", max(sizes) + 100)
    limit = max(max(sizes) + 100, 15_000)
    assert sum(sizes) > limit
    monkeypatch.setattr(cleanup, "MAX_JSON", limit)
    assert plan_cleanup(data, root) == expected
    result = execute_cleanup(expected)
    assert result.status == "complete"
    assert not old.prepared_dir.exists()
    assert not (root / ".cleanup").exists()


def test_planner_spool_closed_on_success_and_failure(generations, monkeypatch):
    import pclass_dark.cleanup_inventory as planning

    data, root, *_ = generations
    opened = []
    original = planning.tempfile.TemporaryFile

    def tracked(*a, **k):
        spool = original(*a, **k)
        opened.append(spool)
        return spool

    monkeypatch.setattr(planning.tempfile, "TemporaryFile", tracked)
    expected = plan_cleanup(data, root)
    assert opened and all(s.closed for s in opened)
    assert plan_cleanup(data, root) == expected

    def failed(*a):
        raise IntegrityError("injected planning failure")

    monkeypatch.setattr(planning.Planner, "finish", failed)
    with pytest.raises(IntegrityError, match="injected planning failure"):
        plan_cleanup(data, root)
    assert all(s.closed for s in opened)


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_invalid_replacement_preserves_archive(generations, failure):
    data, root, old, *_ = generations
    archive = root / ".archives"
    archive.mkdir()
    write_json(archive / "selection.json-backup", read_json(root / "selection.json"))
    selection = read_json(root / "selection.json")
    manifest = root / selection["relative"] / "manifest.json"
    if failure == "missing":
        manifest.unlink()
    else:
        manifest.write_text("{}")
    with pytest.raises((ValueError, RuntimeError, OSError, KeyError)):
        plan_cleanup(data, root)
    assert archive.exists()
    assert old.prepared_dir.exists()


def test_clean_plans_once_and_empty_clean_stops(generations, monkeypatch):
    import pclass_dark.data_cleanup as cleanup

    data, root, *_ = generations
    calls = []
    original = cleanup.make_plan

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(cleanup, "make_plan", counted)
    assert build_library(data, root, clean=True).cleanup.status == "complete"
    assert len(calls) == 1
    assert build_library(data, root, clean=True).cleanup.removed == ()
    assert len(calls) == 2


def test_inventory_publication_failure_rolls_back(generations, monkeypatch):
    import pclass_dark.data_cleanup as cleanup

    data, root, old, *_ = generations
    original = cleanup.write_json

    def fail(path, value):
        original(path, value)
        if path.name.startswith("inventory-"):
            raise OSError("inventory publication failed")

    with monkeypatch.context() as patch:
        patch.setattr(cleanup, "write_json", fail)
        with pytest.raises(CleanupExecutionError) as error:
            execute_cleanup(plan_cleanup(data, root))
    assert error.value.result.status == "rolled_back"
    assert old.prepared_dir.exists()
    assert not (root / cleanup.JOURNAL).exists()
    assert not (root / ".cleanup").exists()


@pytest.mark.parametrize("mode", ["v1", "corrupt", "done"])
def test_sidecar_recovery_and_legacy_compatibility(generations, monkeypatch, mode):
    import pclass_dark.data_cleanup as cleanup
    from pclass_dark.artifacts import identity

    data, root, old, *_ = generations
    original_finish = cleanup._finish_markers

    def stop(roots, journal):
        if mode == "done":
            sidecar = root / ".cleanup" / journal["token"] / "inventory-0.json"
            sidecar.unlink()
            raise OSError("stopped during marker finalization")
        raise OSError("stopped after commit")

    with monkeypatch.context() as patch:
        patch.setattr(
            cleanup, "_finish_markers" if mode == "done" else "_reclaim", stop
        )
        with pytest.raises(CleanupExecutionError):
            execute_cleanup(plan_cleanup(data, root))
    journal = read_json(root / cleanup.JOURNAL)
    assert journal["schema"] == "pclass-dark.cleanup-journal.v2"
    if mode == "corrupt":
        sidecar = root / ".cleanup" / journal["token"] / "inventory-0.json"
        write_json(sidecar, {})
        with pytest.raises(IntegrityError, match="sidecar changed"):
            recover_cleanup(data, root)
        assert (root / cleanup.JOURNAL).exists()
        return
    if mode == "v1":
        trees = list(cleanup._trees(journal))
        for i in range(len(trees)):
            (root / ".cleanup" / journal["token"] / f"inventory-{i}.json").unlink()
        journal["inventories"] = trees
        journal["schema"] = "pclass-dark.cleanup-journal.v1"
        journal.pop("identity")
        journal["identity"] = identity(journal)
        write_json(root / cleanup.JOURNAL, journal)
    assert cleanup._finish_markers is original_finish
    assert recover_cleanup(data, root).status == "complete"
    assert not old.prepared_dir.exists()
    assert not (root / ".cleanup").exists()
    load_selected_library(root)


def test_direct_remove_has_one_sync_and_preserves_external_links(tmp_path, monkeypatch):
    import os

    import pclass_dark.cleanup_io as io

    root = tmp_path / "root"
    trash = root / "trash"
    trash.mkdir(parents=True)
    outside = tmp_path / "retained"
    outside.write_text("keep")
    os.link(outside, trash / "alias")
    (trash / "outside-link").symlink_to(outside)
    for i in range(200):
        (trash / str(i)).write_text("obsolete")
    st = trash.stat()
    expected = {"inode": st.st_ino, "device": st.st_dev, "mode": st.st_mode}
    syncs = []
    original = os.fsync

    def synced(fd):
        syncs.append(fd)
        original(fd)

    monkeypatch.setattr(os, "fsync", synced)
    monkeypatch.setattr(io, "sha256", lambda *a: pytest.fail("rehashed trash"))
    monkeypatch.setattr(os, "link", lambda *a, **k: pytest.fail("relinked trash"))
    io.remove_tree(root, "trash", expected)
    assert not trash.exists()
    assert outside.read_text() == "keep"
    assert len(syncs) == 1


def test_direct_remove_refuses_replaced_root(tmp_path):
    from pclass_dark.cleanup_io import remove_tree

    trash = tmp_path / "trash"
    trash.mkdir()
    st = trash.stat()
    expected = {"inode": st.st_ino, "device": st.st_dev, "mode": st.st_mode}
    trash.rename(tmp_path / "original")
    trash.mkdir()
    (trash / "keep").write_text("replacement")
    with pytest.raises(IntegrityError, match="directory identity changed"):
        remove_tree(tmp_path, "trash", expected)
    assert (trash / "keep").read_text() == "replacement"


def test_direct_cleanup_does_not_reverify_committed_contents(generations, monkeypatch):
    import pclass_dark.data_cleanup as cleanup

    data, root, *_ = generations
    original = cleanup._verify_quarantines
    phases = []

    def checked(*a, **k):
        phases.append(k["partial"])
        return original(*a, **k)

    monkeypatch.setattr(cleanup, "_verify_quarantines", checked)
    assert execute_cleanup(plan_cleanup(data, root)).status == "complete"
    assert phases == [False]


def test_partial_embedding_without_assembly_survives_cleanup(generations):
    import numpy as np

    from pclass_dark.artifacts import file_record, identity
    from pclass_dark.embeddings import SHARD_ROWS, _valid_shard

    data, root, old, new, *_ = generations
    selected = read_json(data / "selection.json")
    manifest = read_json(data / selected["embedding"] / "manifest.json")
    contract = {
        **manifest["identity_inputs"]["contract"],
        "torch_version": "partial-unassembled-fixture",
    }
    inventory_id = manifest["inventory_id"]
    embedding_id = identity({"inventory_id": inventory_id, "contract": contract})
    assert not (data / "embeddings" / embedding_id).exists()
    checkpoints = data / ".checkpoints" / embedding_id
    checkpoints.mkdir(parents=True)
    array = checkpoints / "000000000000.npy"
    stop = min(SHARD_ROWS, manifest["rows"])
    np.save(array, np.zeros((stop, 640), dtype="<f4"), allow_pickle=False)
    receipt = checkpoints / "000000000000.json"
    write_json(
        receipt,
        {
            "schema": "pclass-dark.embedding-shard.v1",
            "embedding_id": embedding_id,
            "inventory_id": inventory_id,
            "start": 0,
            "stop": stop,
            "shape": [stop, 640],
            "dtype": "<f4",
            "file": file_record(array),
            "execution": {
                "device": "cpu",
                "batch_size": 1,
                "seed": 0,
                "contract": contract,
                "timestamp": "2026-10-30T00:00:00+00:00",
            },
        },
    )
    assert _valid_shard(checkpoints, 0, stop, embedding_id, inventory_id) is not None
    hashes = sha256(array), sha256(receipt)
    plan = plan_cleanup(data, root)
    entry = next(
        e for e in plan.entries if e.key == "setup:.checkpoints/" + embedding_id
    )
    assert entry.disposition == "protect"
    assert entry.reason == "partial resumable embedding job"
    assert "setup:prepared/" + old.preparation_id in entry.dependencies
    assert (
        next(
            e for e in plan.entries if e.path == "prepared/" + old.preparation_id
        ).disposition
        == "protect"
    )
    assert execute_cleanup(plan).status == "complete"
    assert build_library(data, root, clean=True).cleanup.status == "complete"
    assert (sha256(array), sha256(receipt)) == hashes
    assert old.prepared_dir.exists() and new.prepared_dir.exists()
    assert not (data / "embeddings" / embedding_id).exists()


@pytest.mark.parametrize(
    "stage,point",
    [
        ("prepared", "state"),
        ("prepared", "sidecar"),
        ("prepared_peer", "state"),
        ("prepared_peer", "sidecar"),
        ("prepared_peer", "peer"),
        ("detaching", "state"),
        ("detaching", "sidecar"),
        ("detaching", "peer"),
    ],
)
def test_rollback_finalization_is_resumable(generations, monkeypatch, stage, point):
    from pathlib import Path

    import pclass_dark.data_cleanup as cleanup
    from pclass_dark.artifacts import identity, transaction_guard
    from pclass_dark.cleanup_io import inventory

    class Crash(BaseException):
        pass

    data, root, old, new, *_ = generations
    plan = plan_cleanup(data, root)
    before = sha256(data / "selection.json"), sha256(root / "selection.json")
    original_state = cleanup._state
    original_write = cleanup.write_json
    original_unlink = Path.unlink

    def state(roots, journal, value):
        if value == ("detaching" if stage == "prepared_peer" else "committed"):
            raise OSError("start rollback before committing")
        original_state(roots, journal, value)
        if value == "rolled_back" and point == "state":
            raise Crash("after durable terminal state")

    def write(path, value):
        if stage == "prepared" and path.name == cleanup.PEER:
            raise OSError("cancel prepared transaction without peer")
        original_write(path, value)

    def unlink(path, *args, **kwargs):
        original_unlink(path, *args, **kwargs)
        if (point == "sidecar" and path.name == "inventory-0.json") or (
            point == "peer" and path.name == cleanup.PEER
        ):
            raise Crash("during rollback metadata removal")

    with monkeypatch.context() as patch:
        patch.setattr(cleanup, "_state", state)
        patch.setattr(cleanup, "write_json", write)
        patch.setattr(Path, "unlink", unlink)
        with pytest.raises(Crash):
            execute_cleanup(plan)
    assert read_json(root / cleanup.JOURNAL)["state"] == "rolled_back"
    for entry in plan.entries:
        if entry.disposition == "remove":
            base = data if entry.root == "setup" else root
            assert identity(inventory(base / entry.path)) == entry.tree_sha256
    result = recover_cleanup(data, root)
    assert result.status == "rolled_back"
    assert result.removed == () and result.logical_bytes_removed == 0
    assert not result.recovery_required
    assert old.prepared_dir.exists() and new.prepared_dir.exists()
    assert (sha256(data / "selection.json"), sha256(root / "selection.json")) == before
    assert not (root / cleanup.JOURNAL).exists()
    assert not (data / cleanup.PEER).exists()
    assert not (root / ".cleanup").exists()
    assert not (data / ".cleanup").exists()
    assert recover_cleanup(data, root) is None
    load_selected_library(root)
    with transaction_guard(data), transaction_guard(root):
        pass
