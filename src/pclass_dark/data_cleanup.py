"""Opt-in, dependency-aware retirement with recoverable two-root deletion.

Public entry points are plan_cleanup, execute_cleanup and recover_cleanup.
No model, subprocess or filesystem work occurs on import.
"""

import logging
import os
import re
import uuid
from dataclasses import replace
from pathlib import Path

from .artifacts import (
    IntegrityError,
    _cleanup_guard_pair,
    canonical,
    identity,
    safe_file,
    schema_checked,
    sha256,
    sync_dir,
    write_json,
)
from .cleanup_inventory import make_plan
from .cleanup_io import (
    MAX_JSON,
    inventory,
    make_directory,
    move,
    read_metadata,
    relative_path,
    remove_tree,
)
from .cleanup_store import InventoryStore
from .cleanup_types import (
    CleanupEntry,
    CleanupExecutionError,
    CleanupPlan,
    CleanupRecoveryRequiredError,
    CleanupResult,
    CleanupStalePlanError,
)

__all__ = [
    "CleanupEntry",
    "CleanupPlan",
    "CleanupResult",
    "CleanupExecutionError",
    "CleanupRecoveryRequiredError",
    "CleanupStalePlanError",
    "plan_cleanup",
    "execute_cleanup",
    "recover_cleanup",
]
LOG = logging.getLogger(__name__)

JOURNAL = ".cleanup-journal.json"
PEER = ".cleanup-peer.json"
UNITS = {
    "setup": re.compile(
        r"(?:\.archives|(?:prepared|embeddings|\.checkpoints)/[0-9a-f]{64})\Z"
    ),
    "library": re.compile(
        r"(?:\.archives|[0-9a-f]{64}|(?:relative-scores|research|\.fits|\.relative-checkpoints|\.production-checkpoints|\.research-checkpoints)/[0-9a-f]{64})\Z"
    ),
}


def _roots(data_dir=None, library_dir=None, *, _allow_missing_library=False):
    from .paths import default_library_output_dir, default_setup_input_dir

    data = Path(default_setup_input_dir() if data_dir is None else data_dir).absolute()
    library = Path(
        default_library_output_dir() if library_dir is None else library_dir
    ).absolute()
    resources = Path(__file__).resolve().parent / "resources"
    for path in (data, library):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise IntegrityError("Cleanup roots cannot contain symlinks")
        if (
            (
                not path.is_dir()
                and not (
                    _allow_missing_library and path == library and not path.exists()
                )
            )
            or path == resources
            or resources in path.parents
            or path in resources.parents
        ):
            raise IntegrityError(
                "Cleanup requires existing writable data roots "
                "outside installed resources"
            )
    data, library = data.resolve(), library.resolve()
    if data == library or data in library.parents or library in data.parents:
        raise IntegrityError("Cleanup roots must be disjoint")
    return {"setup": data, "library": library}


@schema_checked
def gate_snapshot(data, library):
    """Private build handoff token; call while the build owns its root guards."""
    roots = {"setup": Path(data), "library": Path(library)}
    result = []
    for role, root in roots.items():
        s = read_metadata(root / "selection.json")
        paths = ["selection.json"]
        if role == "setup":
            paths += [
                s["prepared"] + "/manifest.json",
                s["embedding"] + "/manifest.json",
            ]
        else:
            paths += [s["library"] + "/manifest.json", s["relative"] + "/manifest.json"]
            if (root / "research-selection.json").exists():
                r = read_metadata(root / "research-selection.json")
                paths += ["research-selection.json", r["research"] + "/manifest.json"]
        for name in paths:
            result.append((role, name, sha256(safe_file(root, name))))
    return tuple(sorted(result))


@schema_checked
def plan_cleanup(data_dir=None, library_dir=None):
    """Validate full selections and preview retirement without changing artifacts.

    Raises IntegrityError for unsafe, incomplete or ambiguous inputs,
    SetupBusyError for contention and CleanupRecoveryRequiredError for a journal.
    Only operational lock files may be created.
    """
    roots = _roots(data_dir, library_dir)
    with _cleanup_guard_pair(roots["setup"], roots["library"]):
        _no_pending(roots)
        plan, inventories = make_plan(roots["setup"], roots["library"])
        inventories.close()
        return plan


def _no_pending(roots):
    # Explicit check: the internal guard bypass is needed only for recovery.
    for root in roots.values():
        if any((root / n).exists() or (root / n).is_symlink() for n in (JOURNAL, PEER)):
            raise CleanupRecoveryRequiredError(
                "Pending cleanup; recover with the same setup/library roots"
            )
        directory = root / ".cleanup"
        if directory.is_symlink() or directory.exists() and any(directory.iterdir()):
            raise IntegrityError(
                "Unjournaled cleanup quarantine; manual inspection required"
            )


def _result(journal, status, *, failed=()):
    entries = journal.get("entries", [])
    complete = status == "complete"
    reclaimed = complete or journal.get("state") == "done"
    allocated = [e["allocated_bytes_estimate"] for e in entries]
    removed = [e["root"] + ":" + e["path"] for e in entries] if reclaimed else []
    logical = sum(e["logical_bytes"] for e in entries) if reclaimed else 0
    if status == "pending_recovery" and journal.get("state") == "committed":
        for i, (e, tree) in enumerate(zip(entries, _trees(journal), strict=True)):
            root = Path(journal["roots"][e["root"]]["path"])
            quarantine = safe_file(root, _quarantine(journal, i))
            if not quarantine.exists():
                removed.append(e["root"] + ":" + e["path"])
            logical += sum(
                f["size_bytes"]
                for f in tree["files"]
                if not (quarantine / f["path"]).exists()
            )
    return CleanupResult(
        "pclass-dark.cleanup-result.v1",
        journal["plan_id"],
        status,
        tuple(removed),
        tuple(tuple(v) for v in journal.get("protected", [])),
        tuple(failed),
        logical,
        sum(allocated) if complete and all(v is not None for v in allocated) else None,
        status == "pending_recovery",
    )


def _state(roots, journal, state):
    journal["state"] = state
    journal.pop("identity", None)
    journal["identity"] = identity(journal)
    write_json(roots["library"] / JOURNAL, journal)


def _quarantine(journal, index):
    return f".cleanup/{journal['token']}/{index}"


def _tree(journal, index):
    if journal["schema"] == "pclass-dark.cleanup-journal.v1":
        return journal["inventories"][index]
    ref = journal["inventories"][index]
    if ref != {
        "sidecar": f"inventory-{index}.json",
        "sha256": journal["entries"][index]["tree_sha256"],
    }:
        raise IntegrityError("Invalid cleanup inventory reference")
    root = Path(journal["roots"]["library"]["path"])
    path = safe_file(root, f".cleanup/{journal['token']}/{ref['sidecar']}")
    tree = read_metadata(path)
    if identity(tree) != ref["sha256"]:
        raise IntegrityError("Cleanup inventory sidecar changed")
    return tree


def _trees(journal):
    for i in range(len(journal["entries"])):
        yield _tree(journal, i)


def _root_identity(root):
    st = root.lstat()
    return {"device": st.st_dev, "inode": st.st_ino}


@schema_checked
def _validate_journal(roots):
    file = roots["library"] / JOURNAL
    peer = roots["setup"] / PEER
    if file.is_symlink() or peer.is_symlink():
        raise IntegrityError("Symlink cleanup marker; manual inspection required")
    if not file.exists():
        if peer.exists() or peer.is_symlink():
            raise IntegrityError("Lone cleanup peer marker; coordinator missing")
        _no_pending(roots)
        return None
    j = read_metadata(file)
    if (
        j.get("schema")
        not in {"pclass-dark.cleanup-journal.v1", "pclass-dark.cleanup-journal.v2"}
        or j.get("identity")
        != identity({k: v for k, v in j.items() if k != "identity"})
        or not re.fullmatch(r"[0-9a-f]{32}", j.get("token", ""))
        or j.get("state") not in {"prepared", "detaching", "committed", "done"}
        or j.get("roots")
        != {k: {"path": str(p), **_root_identity(p)} for k, p in roots.items()}
    ):
        raise IntegrityError("Invalid cleanup journal schema, identity or root binding")
    entries = j["entries"]
    if len(entries) != len(j["inventories"]) or not re.fullmatch(
        r"[0-9a-f]{64}", j["plan_id"]
    ):
        raise IntegrityError("Invalid cleanup journal inventory")
    previous = set()
    skip_trees = j["schema"].endswith(".v2") and (
        j["state"] == "done" or (j["state"] == "prepared" and not peer.exists())
    )
    for index, entry in enumerate(entries):
        tree = None if skip_trees else _tree(j, index)
        role, name = entry["root"], entry["path"]
        if (
            role not in roots
            or not UNITS[role].fullmatch(name)
            or (tree is not None and identity(tree) != entry["tree_sha256"])
        ):
            raise IntegrityError("Unsafe cleanup journal candidate")
        key = role + ":" + name
        if key in previous:
            raise IntegrityError("Duplicate cleanup candidate")
        previous.add(key)
        if tree is None:
            continue
        paths = set()
        for f in tree["files"]:
            relative_path(f["path"])
            if f["path"] in paths:
                raise IntegrityError("Duplicate cleanup inventory path")
            paths.add(f["path"])
            if f["size_bytes"] < 0 or not re.fullmatch(r"[0-9a-f]{64}", f["sha256"]):
                raise IntegrityError("Invalid cleanup file record")
        dirs = {d["path"] for d in tree["directories"]}
        if "." not in dirs or len(dirs) != len(tree["directories"]) or dirs & paths:
            raise IntegrityError("Invalid cleanup directory records")
        for directory in dirs - {"."}:
            relative_path(directory)
        if entry["files"] != len(paths) or entry["logical_bytes"] != sum(
            f["size_bytes"] for f in tree["files"]
        ):
            raise IntegrityError("Cleanup journal count mismatch")
    if peer.exists():
        p = read_metadata(peer)
        if p != {
            "schema": "pclass-dark.cleanup-peer.v1",
            "token": j["token"],
            "library_root": str(roots["library"]),
            "plan_id": j["plan_id"],
        }:
            raise IntegrityError("Cleanup peer does not match coordinator")
    elif j["state"] not in {"prepared", "done"}:
        raise IntegrityError("Cleanup peer missing after detachment began")
    return j


def _verify_quarantines(roots, journal, *, partial):
    """Verify all remaining trees before any further unlink, including hard links."""
    store = InventoryStore()
    db = store.db
    db.execute(
        "CREATE TABLE checks (device INTEGER, inode INTEGER, "
        "missing INTEGER DEFAULT 0, actual_min INTEGER, actual_max INTEGER, "
        "expected_min INTEGER, expected_max INTEGER, "
        "PRIMARY KEY(device,inode))"
    )
    try:
        for i, (entry, tree) in enumerate(
            zip(journal["entries"], _trees(journal), strict=True)
        ):
            root = roots[entry["root"]]
            original = safe_file(root, entry["path"])
            target = safe_file(root, _quarantine(journal, i))
            if original.exists() or original.is_symlink():
                raise IntegrityError("Committed cleanup original has been recreated")
            actual = (
                inventory(target)
                if target.exists()
                else {"files": [], "directories": []}
            )
            expected_files = {f["path"]: f for f in tree["files"]}
            expected_dirs = {d["path"]: d for d in tree["directories"]}
            if {f["path"] for f in actual["files"]} - expected_files.keys() or {
                d["path"] for d in actual["directories"]
            } - expected_dirs.keys():
                raise IntegrityError("Unexpected quarantine contents")
            present = {f["path"] for f in actual["files"]}
            for name, f in expected_files.items():
                if name not in present:
                    group = (f["device"], f["inode"])
                    if f["nlink"] > 1:
                        db.execute(
                            "INSERT INTO checks(device,inode,missing) VALUES (?,?,1) "
                            "ON CONFLICT(device,inode) DO UPDATE SET missing=missing+1",
                            group,
                        )
            for d in actual["directories"]:
                if d != expected_dirs[d["path"]]:
                    raise IntegrityError("Quarantine directory identity changed")
            for f in actual["files"]:
                expected = expected_files[f["path"]]
                if any(
                    f[k] != expected[k]
                    for k in expected
                    if k not in {"nlink", "blocks"}
                ):
                    raise IntegrityError("Quarantine file changed")
                if expected["nlink"] == 1:
                    if f["nlink"] != 1:
                        raise IntegrityError("Unexpected quarantine hard-link count")
                else:
                    db.execute(
                        "INSERT INTO checks(device,inode,actual_min,actual_max,"
                        "expected_min,expected_max) "
                        "VALUES (?,?,?,?,?,?) ON CONFLICT(device,inode) DO UPDATE SET "
                        "actual_min=MIN(COALESCE(actual_min,excluded.actual_min),"
                        "excluded.actual_min), "
                        "actual_max=MAX(COALESCE(actual_max,excluded.actual_max),"
                        "excluded.actual_max), "
                        "expected_min=MIN(COALESCE(expected_min,"
                        "excluded.expected_min),excluded.expected_min), "
                        "expected_max=MAX(COALESCE(expected_max,"
                        "excluded.expected_max),excluded.expected_max)",
                        (
                            f["device"],
                            f["inode"],
                            f["nlink"],
                            f["nlink"],
                            expected["nlink"],
                            expected["nlink"],
                        ),
                    )
            if not partial and actual != tree:
                raise IntegrityError("Detached candidate inventory changed")
        if db.execute(
            "SELECT 1 FROM checks WHERE actual_min IS NOT NULL AND "
            "(actual_min != expected_min-missing OR actual_max != expected_max-missing "
            "OR expected_min != expected_max) LIMIT 1"
        ).fetchone():
            raise IntegrityError("Unexpected quarantine hard-link count")
    finally:
        store.close()


def _finish_markers(roots, j):
    for root in roots.values():
        token_dir = safe_file(root, ".cleanup/" + j["token"])
        if token_dir.exists():
            if root == roots["library"] and j["schema"].endswith(".v2"):
                for i in range(len(j["entries"])):
                    sidecar = token_dir / f"inventory-{i}.json"
                    if sidecar.exists() or sidecar.is_symlink():
                        sidecar.unlink()
                sync_dir(token_dir)
            token_dir.rmdir()
            sync_dir(token_dir.parent)
        container = root / ".cleanup"
        if container.exists() and not any(container.iterdir()):
            container.rmdir()
            sync_dir(root)
    peer = roots["setup"] / PEER
    if peer.exists():
        peer.unlink()
        sync_dir(peer.parent)
    (roots["library"] / JOURNAL).unlink()
    sync_dir(roots["library"])


def _reclaim(roots, j):
    if j["state"] == "done":
        _finish_markers(roots, j)
        return _result(j, "complete")
    LOG.info("Cleanup: removing %s obsolete artifacts directly", len(j["entries"]))
    removed_files = 0
    for i, (entry, tree) in enumerate(zip(j["entries"], _trees(j), strict=True)):
        root = roots[entry["root"]]
        original = safe_file(root, entry["path"])
        if original.exists() or original.is_symlink():
            raise IntegrityError("Committed cleanup original has been recreated")
        directory = next(d for d in tree["directories"] if d["path"] == ".")
        remove_tree(root, _quarantine(j, i), directory)
        removed_files += entry["files"]
        if (i + 1) % 100 == 0 or i + 1 == len(j["entries"]):
            LOG.info(
                "Cleanup: removed %s/%s artifacts (%s planned files)",
                i + 1,
                len(j["entries"]),
                removed_files,
            )
    _state(roots, j, "done")
    _finish_markers(roots, j)
    return _result(j, "complete")


def _recover(roots):
    j = _validate_journal(roots)
    if j is None:
        return None
    if j["state"] == "prepared" and not (roots["setup"] / PEER).exists():
        # Nothing may be detached until the peer is durable. Setup writers could
        # have changed inputs after this crash; cancel without touching them.
        sidecar_names = {f"inventory-{i}.json" for i in range(len(j["entries"]))}
        for root in roots.values():
            token = safe_file(root, ".cleanup/" + j["token"])
            if token.exists() and any(
                p.name not in sidecar_names for p in token.iterdir()
            ):
                raise IntegrityError("Detached data exists without a durable peer")
        _finish_markers(roots, j)
        return _result(j, "rolled_back")
    if j["state"] in {"committed", "done"}:
        # Never reclaim a newly selected generation, even after manual selector edits.
        snapshot = gate_snapshot(roots["setup"], roots["library"])
        if [list(v) for v in snapshot] != j["gate"]:
            raise CleanupStalePlanError(
                "Selected pipeline changed during committed cleanup; "
                "manual inspection required"
            )
        return _reclaim(roots, j)
    for i in reversed(range(len(j["entries"]))):
        e, expected = j["entries"][i], _tree(j, i)
        root = roots[e["root"]]
        original, quarantine = e["path"], _quarantine(j, i)
        a, b = safe_file(root, original), safe_file(root, quarantine)
        if a.exists() == b.exists():
            raise IntegrityError(
                "Ambiguous precommit cleanup state; manual repair required"
            )
        if inventory(b if b.exists() else a) != expected:
            raise IntegrityError("Precommit candidate changed; refusing rollback")
        if b.exists():
            move(root, quarantine, original)
    _finish_markers(roots, j)
    return _result(j, "rolled_back")


@schema_checked
def recover_cleanup(data_dir=None, library_dir=None):
    """Rollback an uncommitted cleanup or finish committed deletion.

    Root bindings and exact inventories are checked. Raises IntegrityError for
    corrupt/ambiguous state and OSError for IO failure; retains recovery markers.
    """
    if os.name != "posix":
        raise RuntimeError("Cleanup recovery requires POSIX IO")
    roots = _roots(data_dir, library_dir)
    with _cleanup_guard_pair(roots["setup"], roots["library"]):
        return _recover(roots)


@schema_checked
def execute_cleanup(plan):
    """Revalidate an immutable preview and retire only the same candidate trees.

    A stale plan raises CleanupStalePlanError before mutation. After journaling,
    failures raise CleanupExecutionError carrying recovery status; KeyboardInterrupt
    propagates after rollback is attempted. POSIX nonfollowing IO is required.
    """
    if not isinstance(plan, CleanupPlan):
        raise ValueError("execute_cleanup requires a CleanupPlan")
    if os.name != "posix":
        raise RuntimeError("Destructive cleanup requires POSIX descriptor-relative IO")
    roots = _roots(plan.data_dir, plan.library_dir)
    with _cleanup_guard_pair(roots["setup"], roots["library"]):
        _no_pending(roots)
        current, inventories = make_plan(roots["setup"], roots["library"])
        try:
            if current != plan:
                raise CleanupStalePlanError("Cleanup plan changed; preview again")
            return _execute_validated(roots, plan, inventories)
        finally:
            inventories.close()


def _execute_validated(roots, plan, inventories):
    entries = [e.to_dict() for e in plan.entries if e.disposition == "remove"]
    j = {
        "schema": "pclass-dark.cleanup-journal.v2",
        "token": uuid.uuid4().hex,
        "plan_id": plan.plan_id,
        "roots": {k: {"path": str(p), **_root_identity(p)} for k, p in roots.items()},
        "gate": [list(v) for v in gate_snapshot(roots["setup"], roots["library"])],
        "entries": entries,
        "protected": [
            [e.key, e.reason] for e in plan.entries if e.disposition == "protect"
        ],
        "inventories": [
            {"sidecar": f"inventory-{i}.json", "sha256": e["tree_sha256"]}
            for i, e in enumerate(entries)
        ],
        "state": "prepared",
    }
    if not entries:
        return _result(j, "complete")
    if len(canonical(j)) + 100 > MAX_JSON:
        raise IntegrityError("Cleanup journal exceeds 64 MiB; no data changed")
    for entry in entries:
        tree = inventories[entry["root"] + ":" + entry["path"]]
        if len(canonical(tree)) + 1 > MAX_JSON:
            raise IntegrityError(
                "Cleanup inventory sidecar exceeds document budget; no data changed"
            )
    LOG.info(
        "Cleanup: publishing recovery inventories for %s obsolete artifacts",
        len(entries),
    )
    try:
        _state(roots, j, "prepared")
        for root in roots.values():
            make_directory(root / ".cleanup")
            make_directory(root / ".cleanup" / j["token"])
        for i, entry in enumerate(entries):
            write_json(
                roots["library"] / ".cleanup" / j["token"] / f"inventory-{i}.json",
                inventories[entry["root"] + ":" + entry["path"]],
            )
        write_json(
            roots["setup"] / PEER,
            {
                "schema": "pclass-dark.cleanup-peer.v1",
                "token": j["token"],
                "library_root": str(roots["library"]),
                "plan_id": plan.plan_id,
            },
        )
        LOG.info("Cleanup: detaching %s obsolete artifacts", len(entries))
        _state(roots, j, "detaching")
        for i, e in enumerate(entries):
            root = roots[e["root"]]
            original = safe_file(root, e["path"])
            if inventory(original) != _tree(j, i):
                raise CleanupStalePlanError("Candidate changed before detachment")
            move(root, e["path"], _quarantine(j, i))
        _verify_quarantines(roots, j, partial=False)
        if gate_snapshot(roots["setup"], roots["library"]) != tuple(
            tuple(v) for v in j["gate"]
        ):
            raise CleanupStalePlanError(
                "Selected pipeline changed before cleanup commit"
            )
        _state(roots, j, "committed")
        return _reclaim(roots, j)
    except BaseException as exc:
        status = "pending_recovery"
        try:
            persisted = read_metadata(roots["library"] / JOURNAL)
            if persisted["state"] in {"prepared", "detaching"}:
                _recover(roots)
                status = "rolled_back"
        except (OSError, ValueError, RuntimeError):
            if (
                not (roots["library"] / JOURNAL).exists()
                and not (roots["setup"] / PEER).exists()
                and j["state"] == "prepared"
            ):
                status = "rolled_back"
        if isinstance(exc, KeyboardInterrupt):
            raise
        if not isinstance(exc, Exception):
            raise
        result = _result(j, status, failed=(("cleanup", str(exc)),))
        raise CleanupExecutionError(
            f"Cleanup failed ({status}): {exc}", result=result
        ) from exc


def _after_build(data, library, expected, production_result):
    try:
        roots = _roots(data, library)
        with _cleanup_guard_pair(roots["setup"], roots["library"]):
            _no_pending(roots)
            if gate_snapshot(roots["setup"], roots["library"]) != expected:
                raise CleanupStalePlanError("Build selection changed before cleanup")
            plan, inventories = make_plan(roots["setup"], roots["library"])
            try:
                result = _execute_validated(roots, plan, inventories)
            finally:
                inventories.close()
        return replace(production_result, cleanup=result)
    except (OSError, ValueError, RuntimeError) as exc:
        if isinstance(exc, CleanupExecutionError):
            exc.production_result = production_result
            raise
        raise CleanupExecutionError(
            f"Cleanup failed; production remains at "
            f"{production_result.library_dir}: {exc}",
            production_result=production_result,
        ) from exc
