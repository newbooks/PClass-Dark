"""Working-directory-independent defaults and immutable bundled artifact paths."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def repository_root():
    """Locate this source checkout from the package, never from the caller cwd."""
    package = PACKAGE_DIR.resolve()
    if package.parent.name != "src":
        return None
    root = package.parent.parent
    metadata = root / "pyproject.toml"
    if package.name != "pclass_dark" or not metadata.is_file():
        return None
    text = metadata.read_text(encoding="utf-8")
    if 'name = "pclass-dark"' not in text:
        return None
    return root


def cache_root():
    base = os.environ.get("XDG_CACHE_HOME")
    path = Path(base).expanduser() if base else Path.home() / ".cache"
    if not path.is_absolute():
        path = Path.home() / ".cache"
    return path / "pclass-dark"


def shared_root():
    repo = repository_root()
    return repo / "data" if repo else cache_root()


def default_setup_output_dir():
    return shared_root() / "setup-data"


def default_library_output_dir():
    return shared_root() / "library"


def bundled_setup_data():
    return PACKAGE_DIR / "resources"


def bundled_library():
    return PACKAGE_DIR / "resources" / "library"


def default_setup_input_dir():
    local = default_setup_output_dir()
    if repository_root() or (local / "selection.json").is_file():
        return local
    bundled = bundled_setup_data()
    return bundled if (bundled / "selection.json").is_file() else local


def default_library_input_dir():
    local = default_library_output_dir()
    if repository_root() or (local / "selection.json").is_file():
        return local
    bundled = bundled_library()
    return bundled if (bundled / "selection.json").is_file() else local
