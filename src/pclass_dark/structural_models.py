"""Portable frozen ProStruct heads; artifact operations never import torch."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from .artifacts import (
    IntegrityError,
    file_record,
    identity,
    publish,
    read_json,
    safe_file,
    schema_checked,
    staging,
    transaction_guard,
)

BUNDLE_ID = "d0f2ef975fef2b86e7057643b250e467a77e81afc58aba42aba8c2f207e844b2"
SCHEMA = "pclass-dark.structural-models.v1"


@dataclass(frozen=True)
class StructuralModelBundle:
    """Verified portable model directory and its frozen content identity."""

    model_id: str
    path: Path


def bundled_structural_models() -> Path:
    return Path(__file__).resolve().parent / "resources" / "structure" / "v1"


@schema_checked
def verify_structural_models(directory=None) -> StructuralModelBundle:
    """Verify the supported frozen bundle, raising IntegrityError on corruption.

    File IO errors propagate. This validates bytes/contracts without importing
    the optional inference stack; tensor validation additionally occurs on load.
    """
    root = Path(directory) if directory is not None else bundled_structural_models()
    manifest = read_json(safe_file(root, "manifest.json"))
    if (
        manifest.get("schema") != SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("id") != BUNDLE_ID
        or identity(manifest["identity_inputs"]) != BUNDLE_ID
        or manifest["files"] != manifest["identity_inputs"]["files"]
    ):
        raise IntegrityError("Unsupported or corrupt structural model manifest")
    expected = set(manifest["files"]) | {"manifest.json"}
    if {p.name for p in root.iterdir()} != expected:
        raise IntegrityError("Unexpected or missing structural model files")
    for name, record in manifest["files"].items():
        path = safe_file(root, name)
        if file_record(path) != record:
            raise IntegrityError(f"Structural model checksum mismatch: {name}")
    return StructuralModelBundle(BUNDLE_ID, root.resolve())


def prepare_structural_models(output_dir) -> StructuralModelBundle:
    """Install/reuse frozen heads under a library root with atomic publication.

    Never download, train, import torch or replace corrupt existing outputs.
    Such outputs must be inspected/recovered explicitly; ordinary reuse fails.
    """
    source = verify_structural_models()
    root = Path(output_dir).absolute()
    # Reject symlinks before resolve, including ancestor substitutions.
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Symlink structural model output roots are unsupported")
    from .paths import PACKAGE_DIR

    resources = PACKAGE_DIR / "resources"
    if root == resources or resources in root.parents:
        raise ValueError("Cannot install models inside package resources")
    with transaction_guard(root):
        target = safe_file(root, f"structural-models/{BUNDLE_ID}")
        if target.exists():
            return verify_structural_models(target)
        with staging(root) as stage:
            for path in source.path.iterdir():
                shutil.copyfile(path, stage / path.name)
            verify_structural_models(stage)
            publish(root, stage, target)
        logging.getLogger(__name__).info("Structural models ready: %s", target)
        return verify_structural_models(target)
