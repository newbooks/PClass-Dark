"""Release selection whitelist, corruption rejection and writer coordination."""

import shutil

import pytest
from build_support import shared_files
from filelock import FileLock


@pytest.fixture
def shared_project(library_data, tmp_path):
    source, *_ = library_data()
    project = tmp_path / "release"
    data = project / "data/setup-data"
    shutil.copytree(source, data, ignore=shutil.ignore_patterns(".*"))
    return project, data


def test_only_selected_completed_artifacts_export(shared_project):
    project, data = shared_project
    for name in [
        ".checkpoints/weights.npy",
        ".archives/old.npy",
        "embeddings/unselected/model.safetensors",
    ]:
        file = data / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"private")
    with shared_files(project) as files:
        destinations = [str(dst) for _, dst in files]
        assert "selection.json" in destinations
        assert any(
            p.startswith("embeddings/") and p.endswith("embeddings.npy")
            for p in destinations
        )
        assert not any(
            ".checkpoints" in p or ".archives" in p or "unselected" in p
            for p in destinations
        )


def test_corrupt_selected_artifact_rejects_release(shared_project):
    import json

    project, data = shared_project
    selected = json.loads((data / "selection.json").read_text())
    (data / selected["embedding"] / "embeddings.npy").write_bytes(b"bad")
    with pytest.raises(ValueError, match="checksum/size"):
        with shared_files(project):
            pass


def test_active_writer_rejects_release(shared_project):
    project, data = shared_project
    locks = data / ".locks"
    locks.mkdir(exist_ok=True)
    with FileLock(str(locks / "setup-transaction.lock")):
        with pytest.raises(RuntimeError, match="writer active"):
            with shared_files(project):
                pass


def test_no_completed_artifacts_is_a_valid_source_build(tmp_path):
    with shared_files(tmp_path) as files:
        assert files == []


def test_build_removes_retired_static_reference_resources(tmp_path, monkeypatch):
    from build_support import SharedBuildPy
    from setuptools import Distribution

    monkeypatch.chdir(tmp_path)
    resources = tmp_path / "build/pclass_dark/resources"
    stale = resources / "reference/manifest.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    (resources / "cluster_descriptions.json").write_text("{}")
    command = SharedBuildPy(Distribution({"packages": []}))
    command.ensure_finalized()
    command.build_lib = str(tmp_path / "build")
    command.run()
    assert not (resources / "reference").exists()
    assert not (resources / "cluster_descriptions.json").exists()
