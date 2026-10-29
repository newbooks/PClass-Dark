"""Guarded readable-path migration with original-generation backup and recovery."""

from __future__ import annotations

import base64
import copy
import csv
import io
import os
import re
import shutil
import uuid
from pathlib import Path

from .artifacts import (
    IntegrityError,
    canonical,
    file_record,
    identity,
    read_json,
    safe_file,
    schema_checked,
    sha256,
    sync_dir,
    transaction_guard,
    write_bytes,
    write_json,
)
from .cluster_paths import (
    FIELD,
    LAYOUT,
    cluster_directory_name,
    stored_cluster_name,
    validate_cluster_directories,
)

JOURNAL = ".cluster-path-migration.json"


def _move(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Migration destination already exists: {destination}")
    os.replace(source, destination)
    sync_dir(source.parent)
    sync_dir(destination.parent)


def _plain_copy(source, destination, manifest):
    """Hard-link immutable payloads, rename cluster folders in two phases."""
    shutil.copytree(source, destination, copy_function=os.link)
    targets = manifest["identity_inputs"].get("cluster_ids")
    if targets is None:
        targets = [r["cluster_id"] for r in manifest["identity_inputs"]["targets"]]
    validate_cluster_directories(targets)
    changes = {
        stored_cluster_name(c, manifest): cluster_directory_name(c) for c in targets
    }
    intermediate = destination / ".rename-clusters"
    for old, new in changes.items():
        if old != new:
            intermediate.mkdir(exist_ok=True)
            os.replace(destination / "clusters" / old, intermediate / new)
    for new in intermediate.iterdir() if intermediate.exists() else []:
        os.replace(new, destination / "clusters" / new.name)
    if intermediate.exists():
        intermediate.rmdir()
    result = copy.deepcopy(manifest)
    records = {}
    for name, record in manifest["files"].items():
        parts = Path(name).parts
        if len(parts) >= 3 and parts[0] == "clusters":
            name = str(Path("clusters", changes[parts[1]], *parts[2:]))
        records[name] = record
    result["files"] = records
    result[FIELD] = LAYOUT
    return result


def _relative_copy(source, destination, manifest, production_digest):
    result = _plain_copy(source, destination, manifest)
    result["identity_inputs"]["production_manifest_sha256"] = production_digest
    result["source_production"]["manifest_sha256"] = production_digest
    result["id"] = identity(result["identity_inputs"])
    # Break the hard link atomically when rewriting metadata, never truncate it.
    with (source / "summary.tsv").open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames
        rows = list(reader)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=columns, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        row["metadata"] = (
            f"clusters/{cluster_directory_name(row['cluster_id'])}/distribution.json"
        )
        writer.writerow(row)
    write_bytes(destination / "summary.tsv", buffer.getvalue().encode())
    result["files"]["summary.tsv"] = file_record(destination / "summary.tsv")
    write_json(destination / "manifest.json", result)
    return result


@schema_checked
def _recover(root, *, rollback=False):
    journal_file = safe_file(root, JOURNAL)
    if not journal_file.exists():
        return
    journal = read_json(journal_file)
    token = journal["token"]
    if journal["schema"] != "pclass-dark.cluster-path-migration.v1" or not re.fullmatch(
        r"[0-9a-f]{32}", token
    ):
        raise IntegrityError("Invalid cluster path migration journal")
    stage = safe_file(root, ".staging/cluster-paths-" + token)
    backup = safe_file(root, ".archives/cluster-paths-" + token)
    moves = [(safe_file(root, a), safe_file(root, b)) for a, b in journal["moves"]]
    # A journal may only move selected bundle trees between this root and its
    # own stage/backup. Never recover an arbitrary caller-provided filesystem plan.
    old = journal["old_selection"]
    new = journal["new_selection"]
    allowed = {
        safe_file(root, old["library"]),
        stage / "backup" / old["library"],
        stage / "candidate" / new["library"],
        stage / "backup",
        backup,
        safe_file(root, "research-selection.json"),
        stage / "backup/research-selection.original.json",
    }
    if "relative" in old:
        allowed.update(
            {
                safe_file(root, old["relative"]),
                stage / "backup" / old["relative"],
                safe_file(root, new["relative"]),
                stage / "candidate" / new["relative"],
            }
        )
    if any(a not in allowed or b not in allowed for a, b in moves):
        raise IntegrityError("Unsafe migration recovery paths")
    selection = root / "selection.json"
    committed = selection.read_bytes() == canonical(new) + b"\n"
    if committed and not rollback:
        # All bundle moves preceded the selector commit. Verify the marker's
        # manifest digests before treating this generation as committed.
        if sha256(root / new["library"] / "manifest.json") != new["manifest_sha256"]:
            raise IntegrityError("Committed migration manifest differs from journal")
        if (
            "relative" in new
            and sha256(root / new["relative"] / "manifest.json")
            != new["relative_manifest_sha256"]
        ):
            raise IntegrityError("Committed relative manifest differs from journal")
    else:
        for source, destination in reversed(moves):
            if destination.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, source)
                sync_dir(source.parent)
                sync_dir(destination.parent)
        write_bytes(selection, base64.b64decode(journal["old_bytes"], validate=True))
    shutil.rmtree(stage, ignore_errors=True)
    journal_file.unlink()
    sync_dir(root)


def migrate_library_paths(output_root):
    """Migrate the selected pair and preserve an original-generation backup.

    Runs no fitting or scoring. Uses root guards, verifies old/candidate bundles,
    and automatically recovers a pending journal. Returned paths identify the
    selected generation and the backup; already-readable inputs return reused=True.
    """
    from .relative_scores import load_selected_library

    root = Path(output_root).absolute()
    if not root.is_dir() or any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Migration requires an existing nonsymlink library root")
    with transaction_guard(root):
        _recover(root)
        original = (root / "selection.json").read_bytes()
        selected = read_json(root / "selection.json")
        pair = load_selected_library(root)
        production = pair.library.source_path
        pm = read_json(production / "manifest.json")
        relative = pair.relative.path if pair.relative else None
        rm = read_json(relative / "manifest.json") if relative else None
        if pm.get(FIELD) == LAYOUT and (rm is None or rm.get(FIELD) == LAYOUT):
            return {"reused": True, "library": str(production), "backup": None}
        validate_cluster_directories(pair.library.signatures)
        if production.parent != root:
            raise ValueError(
                "Migration requires production directly under its library root"
            )
        token = uuid.uuid4().hex
        stage = root / ".staging" / ("cluster-paths-" + token)
        candidate, backup_stage = stage / "candidate", stage / "backup"
        backup = root / ".archives" / ("cluster-paths-" + token)
        stage.mkdir(parents=True)
        try:
            new_pm = _plain_copy(production, candidate / selected["library"], pm)
            write_json(candidate / selected["library"] / "manifest.json", new_pm)
            new_selected = dict(selected)
            new_selected["manifest_sha256"] = sha256(
                candidate / selected["library"] / "manifest.json"
            )
            staged_relative = None
            if relative:
                temporary = candidate / "relative-tmp"
                new_rm = _relative_copy(
                    relative, temporary, rm, new_selected["manifest_sha256"]
                )
                new_selected.update(
                    relative_id=new_rm["id"],
                    relative="relative-scores/" + new_rm["id"],
                    relative_manifest_sha256=sha256(temporary / "manifest.json"),
                )
                staged_relative = candidate / new_selected["relative"]
                staged_relative.parent.mkdir(parents=True, exist_ok=True)
                os.replace(temporary, staged_relative)
            write_json(candidate / "selection.json", new_selected)
            load_selected_library(candidate, immutable=True)
            backup_stage.mkdir(parents=True)
            write_bytes(backup_stage / "selection.json", original)
            moves = [(production, backup_stage / selected["library"])]
            if relative:
                moves.append((relative, backup_stage / selected["relative"]))
            moves.append((candidate / selected["library"], production))
            if relative:
                new_relative = root / new_selected["relative"]
                if new_relative.exists() and new_relative != relative:
                    raise FileExistsError(
                        f"Migration companion destination exists: {new_relative}"
                    )
                moves.append((staged_relative, new_relative))
            research_selector = root / "research-selection.json"
            if research_selector.exists():
                moves.append(
                    (
                        research_selector,
                        backup_stage / "research-selection.original.json",
                    )
                )
            moves.append((backup_stage, backup))
            journal = {
                "schema": "pclass-dark.cluster-path-migration.v1",
                "token": token,
                "old_selection": selected,
                "new_selection": new_selected,
                "old_bytes": base64.b64encode(original).decode(),
                "moves": [
                    [str(a.relative_to(root)), str(b.relative_to(root))]
                    for a, b in moves
                ],
            }
            write_json(root / JOURNAL, journal)
            try:
                for source, destination in moves:
                    _move(source, destination)
                write_json(root / "selection.json", new_selected)
            except BaseException:
                _recover(root, rollback=True)
                raise
            _recover(root)
            return {
                "reused": False,
                "library": str(production),
                "relative": str(root / new_selected["relative"]) if relative else None,
                "backup": str(backup),
                "clusters": len(pair.library.signatures),
            }
        finally:
            # Preserve any stage with an unresolved journal for the next recovery.
            if not (root / JOURNAL).exists():
                shutil.rmtree(stage, ignore_errors=True)
