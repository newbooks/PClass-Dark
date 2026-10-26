"""Reference preparation, C/B provenance, and deterministic per-C folds."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from conftest import protein
from pclass_dark import artifacts
from pclass_dark.artifacts import (
    IntegrityError,
    SetupBusyError,
    identity,
    locked,
    read_json,
)
from pclass_dark.cli import app
from pclass_dark.cluster_sets import (
    ClusterSets,
    fold_assignments,
    load_sets,
    sample_source,
)
from pclass_dark.setup_data import prepare_data, read_tsv, window_records


def test_preparation_dedup_labels_threshold_and_reuse(reference_factory, tmp_path):
    records = [(f"A{i:02}", "A", protein(i)) for i in range(10)]
    records += [(f"B{i:02}", "B", protein(i + 20)) for i in range(9)]
    records += [
        ("Adup", "A", protein(0)),
        ("Bshared", "B", protein(0)),
        ("C0", "C", protein(0)),
        ("extra", None, "XYZ"),
    ]
    reference = reference_factory(records)
    result = prepare_data(tmp_path / "out", reference_dir=reference, prepare_only=True)
    assert (
        result.counts["retained_clusters"] == 2
    )  # B reaches exactly 10 with shared label.
    assert result.counts["excluded_clusters"] == 1
    assert result.counts["retained_sequences"] == 19
    assert result.counts["retained_memberships"] == 20
    sets = load_sets(result.prepared_dir)
    shared = set(sets.cluster_members("A")) & set(sets.cluster_members("B"))
    assert shared == {"A00"}
    assert not shared.intersection(sets.background_members("A"))
    folds = read_tsv(result.prepared_dir / "folds.tsv")
    for c in ("A", "B"):
        assert [
            sum(r["cluster_id"] == c and r["fold_id"] == str(f) for r in folds)
            for f in range(5)
        ] == [2] * 5
    before = (result.prepared_dir / "manifest.json").read_bytes()
    reused = prepare_data(tmp_path / "out", reference_dir=reference, prepare_only=True)
    assert reused.reused_prepared
    assert (reused.prepared_dir / "manifest.json").read_bytes() == before
    assert read_json(tmp_path / "out/selection.json")["embedding"] is None


@pytest.mark.parametrize("length", [1, 1021, 1022, 1023, 2043, 2044, 2045])
def test_terminal_windows(length):
    sequence = ("ACDEFGHIKL" * 205)[:length]
    rows = window_records("P", sequence)
    if length <= 1022:
        assert rows == [("P", sequence, "full", 1, length)]
    else:
        assert rows == [
            ("N-P", sequence[:1022], "N", 1, 1022),
            ("C-P", sequence[-1022:], "C", length - 1021, length),
        ]


def test_long_parent_and_small_custom_clusters(reference_factory, tmp_path):
    reference = reference_factory([("P", "K", "A" * 2045)])
    result = prepare_data(
        tmp_path / "out", reference_dir=reference, prepare_only=True, min_clusters=1
    )
    windows = read_tsv(result.prepared_dir / "windows.tsv")
    assert {w["window_id"] for w in windows} == {"N-P", "C-P"}
    assert result.counts["omitted_residues"] == 1
    assert len(read_tsv(result.prepared_dir / "cluster_folds.tsv")) == 5
    assert load_sets(result.prepared_dir).summary("K")["status"] == "empty_background"


def test_random_folds_are_per_cluster_and_per_window():
    ids = [f"W{i}" for i in range(20)]
    a = fold_assignments("A", ids, 42)
    assert a == fold_assignments("A", reversed(ids), 42)
    assert a != fold_assignments("B", ids, 42)
    assert [list(a.values()).count(f) for f in range(5)] == [4] * 5
    outcomes = {
        fold_assignments("A", ["N-P", "C-P", "Q"], seed)["N-P"]
        == fold_assignments("A", ["N-P", "C-P", "Q"], seed)["C-P"]
        for seed in range(100)
    }
    assert outcomes == {False}  # two records always differ if only three candidates
    longer = ["N-P", "C-P", *ids]
    assert {
        fold_assignments("A", longer, seed)["N-P"]
        == fold_assignments("A", longer, seed)["C-P"]
        for seed in range(100)
    } == {True, False}


def test_background_cap_and_exact_expansion():
    clusters = {
        "A": [f"A{i:03}" for i in range(250)],
        "B": ["B0", "B1"],
        "C": ["B0", "C1"],
    }
    parents = {w: w for ids in clusters.values() for w in ids}
    selections = {c: sample_source(c, ids, 42) for c, ids in clusters.items()}
    assert len(selections["A"]) == 200
    assert selections["B"] == "all"
    assert selections["A"] == sample_source("A", clusters["A"], 42)
    sets = ClusterSets(clusters, selections, parents)
    assert sets.background_members("A") == ("B0", "B1", "C1")
    for c in clusters:
        actual = sets.background_members(c)
        summary = sets.summary(c)
        assert summary["background_count"] == len(actual)
        assert summary["background_ids_sha256"] == identity(list(actual))
        assert not set(actual) & set(clusters[c])


def test_workers_and_seed_changes(reference_factory, tmp_path):
    ref = reference_factory()
    one = prepare_data(tmp_path / "one", reference_dir=ref, prepare_only=True)
    two = prepare_data(
        tmp_path / "two", reference_dir=ref, prepare_only=True, workers=2
    )
    assert one.preparation_id == two.preparation_id
    m = read_json(one.prepared_dir / "manifest.json")
    for name in m["files"]:
        assert (one.prepared_dir / name).read_bytes() == (
            two.prepared_dir / name
        ).read_bytes()
    changed = prepare_data(
        tmp_path / "one", reference_dir=ref, prepare_only=True, seed=7
    )
    assert changed.preparation_id != one.preparation_id
    assert changed.counts["windows"] == one.counts["windows"]


@pytest.mark.parametrize("mutation", ["reorder", "mixed", "index", "row"])
def test_corruption_rejected(reference_factory, tmp_path, mutation):
    ref = reference_factory()
    result = prepare_data(tmp_path / "out", reference_dir=ref, prepare_only=True)
    path = result.prepared_dir
    if mutation == "reorder":
        c = read_json(path / "cluster_members.json")
        c["clusters"]["K0"].reverse()
        artifacts.write_json(path / "cluster_members.json", c)
    elif mutation == "mixed":
        b = read_json(path / "background_members.json")
        b["cluster_members_sha256"] = "0" * 64
        artifacts.write_json(path / "background_members.json", b)
    elif mutation == "index":
        b = read_json(path / "background_members.json")
        b["source_samples"]["K0"] = [True]
        artifacts.write_json(path / "background_members.json", b)
    else:
        (path / "windows.tsv").write_text("wrong\n")
    # Even if the outer file digest is refreshed, semantic bindings must fail.
    m = read_json(path / "manifest.json")
    for name in m["files"]:
        m["files"][name] = artifacts.file_record(path / name)
    artifacts.write_json(path / "manifest.json", m)
    with pytest.raises(IntegrityError):
        load_sets(path)


def test_lock_and_rollback(tmp_path, monkeypatch):
    with locked(tmp_path, "one"):
        with pytest.raises(SetupBusyError):
            with locked(tmp_path, "one"):
                pass
    old, stage = tmp_path / "old", tmp_path / "stage"
    old.mkdir()
    stage.mkdir()
    (old / "data").write_text("previous")
    original = artifacts.os.replace

    def fail(source, target):
        if Path(source) == stage:
            raise OSError("injected publish failure")
        return original(source, target)

    monkeypatch.setattr(artifacts.os, "replace", fail)
    with pytest.raises(OSError):
        artifacts.publish(tmp_path, stage, old, force=True)
    assert (old / "data").read_text() == "previous"


def test_input_errors_and_cli(reference_factory, tmp_path, monkeypatch):
    ref = reference_factory()
    with pytest.raises(ValueError):
        prepare_data(tmp_path / "out", reference_dir=ref, min_clusters=0)
    with pytest.raises(ValueError):
        prepare_data(ref / "output", reference_dir=ref, prepare_only=True)
    (ref / "accession.fasta").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(IntegrityError):
        prepare_data(tmp_path / "out", reference_dir=ref, prepare_only=True)
    runner = CliRunner()
    assert runner.invoke(app, ["setup-data", "--help"]).exit_code == 0
    assert runner.invoke(app, ["setup-data", "--min-clusters", "0"]).exit_code == 2


def test_bins_accept_iterators():
    from pclass_dark.setup_data import _bins

    assert _bins(iter([9, 10, 19, 20, 40, 80, 160, 320, 640])) == [
        1,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
    ]


def test_symlink_outputs_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (tmp_path / ".locks").symlink_to(target, target_is_directory=True)
    with pytest.raises(IntegrityError):
        with locked(tmp_path, "one"):
            pass


def test_force_retains_identical_manifest_and_archives(reference_factory, tmp_path):
    ref = reference_factory()
    first = prepare_data(tmp_path / "out", reference_dir=ref, prepare_only=True)
    before = (first.prepared_dir / "manifest.json").read_bytes()
    rebuilt = prepare_data(
        tmp_path / "out", reference_dir=ref, prepare_only=True, force=True
    )
    assert not rebuilt.reused_prepared
    assert (rebuilt.prepared_dir / "manifest.json").read_bytes() == before
    assert list((tmp_path / "out/.archives").iterdir())


def test_counts_and_fold_counts_match_tables(reference_factory, tmp_path):
    ref = reference_factory()
    result = prepare_data(tmp_path / "out", reference_dir=ref, prepare_only=True)
    assert sum(result.counts["deduplicated_size_bins"]) == 3
    assert sum(result.counts["original_size_bins"]) == 3
    rows = read_tsv(result.prepared_dir / "cluster_folds.tsv")
    assert len(rows) == 15
    assert all(
        int(r["training_windows"]) + int(r["validation_windows"]) in (46, 47)
        for r in rows
    )


def test_reference_duplicate_empty_missing_and_noncanonical(
    reference_factory, tmp_path
):
    from pclass_dark.reference_data import read_fasta

    malformed = tmp_path / "bad.fasta"
    for text in (">P\nAC\n>P\nDE\n", ">P\n", "ACDE\n", ">P\nA C\n"):
        malformed.write_text(text)
        with pytest.raises(IntegrityError):
            list(read_fasta(malformed))
    malformed.write_text(">P description\n AC\nDE \n")
    assert list(read_fasta(malformed)) == [("P", "ACDE")]
    ref = reference_factory([("p", "K", "XBZUO")])
    result = prepare_data(
        tmp_path / "out", reference_dir=ref, prepare_only=True, min_clusters=1
    )
    assert result.counts["noncanonical_accessions"] == {"p": "BOUXZ"}


def test_small_empty_retained_and_collision(reference_factory, tmp_path, monkeypatch):
    ref = reference_factory([("p", "K", "AA"), ("q", "K", "AC")])
    with pytest.raises(ValueError, match="No clusters remain"):
        prepare_data(tmp_path / "out", reference_dir=ref, prepare_only=True)
    import pclass_dark.setup_data as setup

    monkeypatch.setattr(setup, "sequence_id", lambda sequence: "collision")
    with pytest.raises(IntegrityError, match="collision"):
        prepare_data(
            tmp_path / "out", reference_dir=ref, prepare_only=True, min_clusters=1
        )


def test_malformed_reference_manifest_is_integrity_error(reference_factory, tmp_path):
    ref = reference_factory()
    artifacts.write_json(ref / "manifest.json", [])
    with pytest.raises(IntegrityError, match="Malformed artifact"):
        prepare_data(tmp_path / "out", reference_dir=ref, prepare_only=True)


def test_transaction_guard_reentrancy_root_scope_and_process_exclusion(tmp_path):
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor

    from pclass_dark.artifacts import transaction_guard

    root = tmp_path / "root"
    other = tmp_path / "other"
    script = """
import sys
from pclass_dark.artifacts import transaction_guard, SetupBusyError
try:
    with transaction_guard(sys.argv[1]):
        print('acquired')
except SetupBusyError:
    print('busy')
"""
    with transaction_guard(root), transaction_guard(root):
        child = subprocess.run(
            [sys.executable, "-c", script, str(root)],
            capture_output=True,
            text=True,
            check=True,
        )
        assert child.stdout.strip() == "busy"
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Root ownership does not serialize an independent output directory.
            def independent():
                with transaction_guard(other):
                    return True

            assert executor.submit(independent).result(timeout=5)
    child = subprocess.run(
        [sys.executable, "-c", script, str(root)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.strip() == "acquired"
