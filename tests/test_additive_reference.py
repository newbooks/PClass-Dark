"""Snapshot binding and frozen-reference import on small analytical inputs."""

import csv

import pytest

from pclass_dark.additive_reference import (
    DEFAULT_RECIPE,
    prepare_additive_reference,
    validate_recipe,
    verify_reference,
)
from pclass_dark.artifacts import IntegrityError, file_record, read_json, write_json
from pclass_dark.ml_data import PAIR_HEADER


@pytest.fixture
def reference_source(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    write_json(
        prepared / "manifest.json",
        dict(id="P", reference={"files": {"clusters.tsv": {"sha256": "bound"}}}),
    )
    (prepared / "clusters.tsv").write_text("cluster_id\nA\nB\nC\n")
    matrix = tmp_path / "pairs.tsv"
    with matrix.open("w") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(PAIR_HEADER)
        w.writerows(
            [
                ("B", "mB", "A", "mA", 0.8, 0.2),
                ("A", "mA", "C", "mC", 0.3, 0.7),
                ("B", "mB", "C", "mC", 0.6, 0.4),
            ]
        )
    write_json(
        tmp_path / "members.json", {c: {"medoid_accession": "m" + c} for c in "ABC"}
    )
    write_json(tmp_path / "desc.json", {c: "function " + c for c in "ABC"})
    manifest = tmp_path / "source.json"
    write_json(
        manifest,
        dict(
            schema="pclass-dark.additive-source.v1",
            source_reference=read_json(prepared / "manifest.json")["reference"],
            raw_file=file_record(matrix),
            members_path="members.json",
            members_file=file_record(tmp_path / "members.json"),
            descriptions_path="desc.json",
            descriptions_file=file_record(tmp_path / "desc.json"),
        ),
    )
    return prepared, matrix, manifest


def test_recipe_validation():
    assert validate_recipe(DEFAULT_RECIPE)["configuration"] == "hard"
    for change in (
        {"epochs": True},
        {"alpha": 0.1},
        {"unexpected": 3},
        {"features": []},
        {"knots": 1},
    ):
        with pytest.raises(ValueError):
            validate_recipe(dict(DEFAULT_RECIPE, **change))


def test_import_and_reuse(reference_source, tmp_path):
    prepared, matrix, manifest = reference_source
    root = tmp_path / "setup"
    root.mkdir()
    path = prepare_additive_reference(
        root, prepared, source=matrix, source_manifest=manifest
    )
    assert verify_reference(path)["counts"]["retained_pairs"] == 3
    rows = list(csv.DictReader((path / "cluster_dmatrix.tsv").open(), delimiter="\t"))
    assert rows[0]["cluster_A"] == "A" and rows[0]["Medoid_A"] == "mA"
    assert prepare_additive_reference(root, prepared) == path
    (path / "recipe.json").write_text("{}")
    with pytest.raises(IntegrityError):
        prepare_additive_reference(root, prepared)


def test_snapshot_mismatch(reference_source, tmp_path):
    prepared, matrix, manifest = reference_source
    write_json(prepared / "manifest.json", dict(id="P", reference={"files": {}}))
    with pytest.raises(IntegrityError, match="reference"):
        prepare_additive_reference(
            tmp_path / "setup", prepared, source=matrix, source_manifest=manifest
        )


def test_default_frozen_binding_and_recipe():
    import gzip

    from pclass_dark.additive_reference import resource_root

    root = resource_root()
    source = read_json(root / "source.json")
    assert (
        source["raw_file"]["sha256"]
        == "d934cc904212c9e1d901c937ca818a7a6246386b5e51f97cb6962dc3faa9bd05"
    )
    assert source["compressed_file"] == file_record(root / "cluster_dmatrix.tsv.gz")
    assert validate_recipe(read_json(root / "recipe.json")) == DEFAULT_RECIPE
    with gzip.open(root / "cluster_dmatrix.tsv.gz", "rt") as f:
        assert f.readline().rstrip("\n").split("\t") == list(PAIR_HEADER)


def test_only_retained_prepared_clusters_imported(reference_source, tmp_path):
    prepared, matrix, manifest = reference_source
    (prepared / "clusters.tsv").write_text(
        "cluster_id\tretained\nA\ttrue\nB\ttrue\nC\tfalse\n"
    )
    root = tmp_path / "filtered"
    root.mkdir()
    path = prepare_additive_reference(
        root, prepared, source=matrix, source_manifest=manifest
    )
    assert verify_reference(path)["counts"]["retained_pairs"] == 1
    assert (
        len(list(csv.DictReader((path / "cluster_dmatrix.tsv").open(), delimiter="\t")))
        == 1
    )


def test_setup_failure_restores_both_selectors(
    reference_factory, tmp_path, monkeypatch
):
    from conftest import protein
    from pclass_dark.setup_data import prepare_data

    ref = reference_factory([(f"A{i}", "A", protein(i)) for i in range(10)])
    root = tmp_path / "setup"
    prepare_data(root, reference_dir=ref, prepare_only=True)
    write_json(root / "additive-selection.json", {"schema": "previous"})
    before = {
        p: p.read_bytes()
        for p in [root / "selection.json", root / "additive-selection.json"]
    }

    def fail(setup, *args, **kwargs):
        write_json(setup / "additive-selection.json", {"schema": "partial"})
        raise ValueError("interrupted additive import")

    monkeypatch.setattr(
        "pclass_dark.additive_reference.prepare_additive_reference", fail
    )
    with pytest.raises(ValueError, match="interrupted additive"):
        prepare_data(root, reference_dir=ref, prepare_only=True, additive=True)
    assert all(p.read_bytes() == content for p, content in before.items())


def test_integrated_build_normalizes_relative_guard_paths(tmp_path, monkeypatch):
    from pclass_dark import additive_build

    monkeypatch.chdir(tmp_path)
    (tmp_path / "setup").mkdir()
    monkeypatch.setattr(
        additive_build, "_build_additive_library", lambda data, root, **kw: (data, root)
    )
    data, root = additive_build.build_additive_library("setup", "library")
    assert data == tmp_path / "setup" and root == tmp_path / "library"
