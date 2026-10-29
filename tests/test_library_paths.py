"""Readable public paths, legacy compatibility, and guarded in-place migration."""

import csv
import io
import os
import subprocess
import sys

import numpy as np
import pytest

from pclass_dark import library_paths
from pclass_dark.artifacts import (
    file_record,
    identity,
    read_json,
    sha256,
    write_bytes,
    write_json,
)
from pclass_dark.cluster_paths import (
    FIELD,
    cluster_directory_name,
    legacy_cluster_name,
    validate_cluster_directories,
)
from pclass_dark.library import build_library
from pclass_dark.relative_scores import load_selected_library


def legacy_pair(root):
    """Independently reconstruct the prior on-disk layout from tiny new outputs."""
    selector = read_json(root / "selection.json")
    for key in ("library", "relative"):
        directory = root / selector[key]
        manifest = read_json(directory / "manifest.json")
        del manifest[FIELD]
        for cluster in ("K0", "K1"):
            os.rename(
                directory / "clusters" / cluster,
                directory / "clusters" / legacy_cluster_name(cluster),
            )
        manifest["files"] = {
            name.replace("clusters/K0/", "clusters/c_4b30/").replace(
                "clusters/K1/", "clusters/c_4b31/"
            ): record
            for name, record in manifest["files"].items()
        }
        if key == "relative":
            manifest["identity_inputs"]["production_manifest_sha256"] = selector[
                "manifest_sha256"
            ]
            manifest["source_production"]["manifest_sha256"] = selector[
                "manifest_sha256"
            ]
            manifest["id"] = identity(manifest["identity_inputs"])
            with (directory / "summary.tsv").open(newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                fields, rows = reader.fieldnames, list(reader)
            buf = io.StringIO()
            writer = csv.DictWriter(
                buf, fieldnames=fields, delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            for row in rows:
                row["metadata"] = (
                    f"clusters/{legacy_cluster_name(row['cluster_id'])}/distribution.json"
                )
                writer.writerow(row)
            write_bytes(directory / "summary.tsv", buf.getvalue().encode())
            manifest["files"]["summary.tsv"] = file_record(directory / "summary.tsv")
        write_json(directory / "manifest.json", manifest)
        if key == "library":
            selector["manifest_sha256"] = sha256(directory / "manifest.json")
        else:
            selector["relative_id"] = manifest["id"]
            selector["relative_manifest_sha256"] = sha256(directory / "manifest.json")
            selector["relative"] = "relative-scores/" + manifest["id"]
            os.rename(directory, root / selector["relative"])
    write_json(root / "selection.json", selector)
    load_selected_library(root)
    return selector


@pytest.mark.parametrize(
    "name", ["../foo", "/foo", "a/b", ".foo", "a.", "CON", "aux.txt", "", "é"]
)
def test_unsafe_names(name):
    with pytest.raises(ValueError):
        cluster_directory_name(name)


def test_case_collision():
    with pytest.raises(ValueError, match="collide"):
        validate_cluster_directories(["K0", "k0"])


@pytest.mark.parametrize("failure", [None, "exception", "crash"])
def test_migrate_legacy_pair(library_data, tmp_path, monkeypatch, failure):
    data, _, _, matrix, contract = library_data()
    root = tmp_path / "release/data/library"
    build_library(data, root)
    old = legacy_pair(root)
    old_bytes = (root / "selection.json").read_bytes()
    before = load_selected_library(root)
    before_score, _ = before.library.score("K0", matrix, embedding_contract=contract)
    before_evidence = before.relative.evaluate("K0", before_score)
    scoped_old = load_selected_library(root, cluster_id="K0")
    assert tuple(scoped_old.library.signatures) == ("K0",)
    old_pm = read_json(before.library.source_path / "manifest.json")
    old_rm = read_json(before.relative.path / "manifest.json")
    if failure == "exception":
        move = library_paths._move
        calls = []

        def fail(source, destination):
            move(source, destination)
            calls.append(1)
            if len(calls) == 3:
                raise RuntimeError("interrupted")

        with monkeypatch.context() as patch:
            patch.setattr(library_paths, "_move", fail)
            with pytest.raises(RuntimeError, match="interrupted"):
                library_paths.migrate_library_paths(root)
        assert (root / "selection.json").read_bytes() == old_bytes
        load_selected_library(root)
        assert not (root / library_paths.JOURNAL).exists()
    elif failure == "crash":
        program = """import os, sys
from pclass_dark import library_paths as p
original = p._move
calls = []
def move(a,b):
    original(a,b)
    calls.append(1)
    if len(calls)==3: os._exit(91)
p._move=move
p.migrate_library_paths(sys.argv[1])
"""
        result = subprocess.run([sys.executable, "-c", program, str(root)])
        assert result.returncode == 91
        assert (root / library_paths.JOURNAL).exists()
    result = library_paths.migrate_library_paths(root)
    assert result["clusters"] == 2
    backup = type(root)(result["backup"])
    assert (backup / "selection.json").read_bytes() == old_bytes
    load_selected_library(backup)
    after = load_selected_library(root)
    scoped = load_selected_library(root, cluster_id="K0")
    after_score, _ = scoped.library.score("K0", matrix, embedding_contract=contract)
    np.testing.assert_array_equal(before_score, after_score)
    np.testing.assert_array_equal(
        before_evidence.background_evidence,
        scoped.relative.evaluate("K0", after_score).background_evidence,
    )
    from pclass_dark.signature_summaries import list_signature_summaries

    assert len(list_signature_summaries(root).to_dict()["rows"]) == 2
    assert after.library.library_id == old["library_id"]
    assert after.library.manifest_sha256 != old["manifest_sha256"]
    assert read_json(root / "selection.json")["relative_id"] != old["relative_id"]
    for directory, manifest in (
        (after.library.source_path, old_pm),
        (after.relative.path, old_rm),
    ):
        for name, record in manifest["files"].items():
            if name.startswith("clusters/"):
                readable = name.replace("c_4b30", "K0").replace("c_4b31", "K1")
                assert sha256(directory / readable) == record["sha256"]
        assert (directory / "clusters/K0").is_dir()
        assert not (directory / "clusters/c_4b30").exists()
    from build_support import inventory

    exported = inventory(tmp_path / "release")
    assert any("clusters/K0/final.json" in str(path) for _, path in exported)
    assert not any(".archives" in str(path) for path, _ in exported)
    assert library_paths.migrate_library_paths(root)["reused"]
    from pclass_dark import library_build

    monkeypatch.setattr(
        library_build, "fit_signature", lambda *a, **k: pytest.fail("refitted")
    )
    assert build_library(data, root).reused
    assert not (root / library_paths.JOURNAL).exists()
