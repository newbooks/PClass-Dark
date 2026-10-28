"""Defaults depend on package location; explicit paths remain caller-controlled."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from pclass_dark import library_build, paths, setup_data
from pclass_dark.library import build_library


@pytest.fixture
def checkout(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    package = root / "src/pclass_dark"
    package.mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "pclass-dark"\n')
    monkeypatch.setattr(paths, "PACKAGE_DIR", package)
    return root


def test_defaults_ignore_cwd_and_create_nothing(checkout, tmp_path, monkeypatch):
    for cwd in [checkout, checkout / "work", tmp_path / "elsewhere"]:
        cwd.mkdir(exist_ok=True)
        monkeypatch.chdir(cwd)
        assert paths.default_setup_input_dir() == checkout / "data/setup-data"
        assert paths.default_setup_output_dir() == checkout / "data/setup-data"
        assert paths.default_library_output_dir() == checkout / "data/library"
    assert not (checkout / "data").exists()


def test_prepare_default_and_explicit_override(
    checkout, reference_factory, tmp_path, monkeypatch
):
    reference = reference_factory()
    other = tmp_path / "unrelated"
    other.mkdir()
    monkeypatch.chdir(other)
    result = setup_data.prepare_data(prepare_only=True, reference_dir=reference)
    assert result.prepared_dir.is_relative_to(checkout / "data/setup-data")
    again = setup_data.prepare_data(prepare_only=True, reference_dir=reference)
    assert again.reused_prepared
    explicit = setup_data.prepare_data(
        "explicit", prepare_only=True, reference_dir=reference
    )
    assert explicit.prepared_dir.is_relative_to(other / "explicit")


def test_build_api_default_and_override(checkout, monkeypatch):
    calls = []

    def fake(data_dir, output_dir, **kwargs):
        calls.append((data_dir, output_dir))
        return SimpleNamespace()

    monkeypatch.setattr(library_build, "build", fake)
    build_library()
    assert calls[-1] == (checkout / "data/setup-data", checkout / "data/library")
    build_library("input", "output")
    assert calls[-1] == ("input", "output")


def test_installed_bundles_and_user_cache(tmp_path, monkeypatch):
    package = tmp_path / "site-packages/pclass_dark"
    package.mkdir(parents=True)
    monkeypatch.setattr(paths, "PACKAGE_DIR", package)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    cache = tmp_path / "cache/pclass-dark"
    assert paths.repository_root() is None
    assert paths.default_setup_output_dir() == cache / "setup-data"
    assert paths.default_setup_input_dir() == cache / "setup-data"
    resource = package / "resources"
    (resource / "library").mkdir(parents=True)
    (resource / "selection.json").write_text("{}")
    (resource / "library/selection.json").write_text("{}")
    assert paths.default_setup_input_dir() == resource
    assert paths.default_library_input_dir() == resource / "library"
    for directory in [cache / "setup-data", cache / "library"]:
        directory.mkdir(parents=True)
        (directory / "selection.json").write_text("{}")
    assert paths.default_setup_input_dir() == cache / "setup-data"
    assert paths.default_library_input_dir() == cache / "library"


def test_package_setup_root_is_immutable():
    import pclass_dark.library as library

    root = Path(library.__file__).resolve().parent / "resources"
    assert library._library_immutable(root, False)
