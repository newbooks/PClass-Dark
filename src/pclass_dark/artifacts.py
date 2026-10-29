"""Checksummed artifact IO and publication shared by setup and embedding."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from filelock import FileLock, Timeout


class IntegrityError(ValueError):
    """An artifact does not match its recorded content or schema."""


class SetupBusyError(RuntimeError):
    """Another process owns an artifact's writer lock."""


def schema_checked(function):
    """Translate malformed structured-data fields into an actionable integrity error."""

    @wraps(function)
    def checked(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise IntegrityError(
                f"Malformed artifact in {function.__name__}: {exc}"
            ) from exc

    return checked


def canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def identity(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path):
    try:
        return json.loads(
            Path(path).read_bytes(),
            object_pairs_hook=_pairs,
            parse_constant=lambda x: (_ for _ in ()).throw(
                IntegrityError(f"Invalid JSON number: {x}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"Invalid JSON artifact: {path}") from exc


def write_bytes(path, data):
    """Durably replace a file, without exposing partially written bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    write_bytes(path, canonical(value) + b"\n")


def sync_dir(path):
    if os.name == "posix":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def sync_tree(root):
    """Flush a completed staged payload tree before committing its manifest."""
    root = Path(root)
    paths = [safe_file(root, str(p.relative_to(root))) for p in root.rglob("*")]
    for path in paths:
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in sorted(
        (p for p in paths if p.is_dir()), key=lambda p: len(p.parts), reverse=True
    ):
        sync_dir(path)
    sync_dir(root)


def file_record(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise IntegrityError(f"Expected regular artifact file: {path}")
    return {"sha256": sha256(path), "size_bytes": path.stat().st_size}


def safe_file(root, name):
    root = Path(root)
    relative = Path(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise IntegrityError(f"Unsafe artifact member: {name}")
    path = root / relative
    if any(p.is_symlink() for p in [path, *path.parents] if p != root.parent):
        raise IntegrityError(f"Symlink artifact member: {name}")
    return path


def verify_files(root, records):
    if not isinstance(records, dict) or not records:
        raise IntegrityError("Missing artifact file inventory")
    for name, expected in records.items():
        path = safe_file(root, name)
        if not isinstance(expected, dict) or file_record(path) != {
            k: expected.get(k) for k in ("sha256", "size_bytes")
        }:
            raise IntegrityError(f"Checksum/size mismatch: {path}")


def verify_bundle(path, schema, expected_id=None, *, members=None):
    """Verify a bundle, optionally restricting file checks to explicit members."""
    path = Path(path)
    manifest = read_json(safe_file(path, "manifest.json"))
    if (
        manifest.get("schema") != schema
        or manifest.get("status") != "complete"
        or (expected_id is not None and manifest.get("id") != expected_id)
    ):
        raise IntegrityError(f"Incomplete or incompatible bundle: {path}")
    records = manifest.get("files")
    if members is not None:
        members = tuple(members)
        if not isinstance(records, dict) or any(m not in records for m in members):
            raise IntegrityError("Missing requested artifact file inventory")
        records = {m: records[m] for m in members}
    verify_files(path, records)
    return manifest


@contextmanager
def locked(root, key):
    directory = safe_file(root, ".locks")
    directory.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(directory / f"{key}.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise SetupBusyError(
            f"Setup busy: {key}; retry after the writer exits"
        ) from exc
    try:
        yield
    finally:
        lock.release()


_transaction_context = threading.local()


@contextmanager
def transaction_guard(root):
    """Own a working root until selection commits or rollback finishes.

    Reentrant only in the same thread/process. Other invocations and mutable
    readers fail promptly rather than observe a rollback-eligible generation.
    """
    root = Path(root).resolve()
    token = (os.getpid(), str(root))
    owned = getattr(_transaction_context, "owned", set())
    if token in owned:
        yield
        return
    with locked(root, "setup-transaction"):
        owned.add(token)
        _transaction_context.owned = owned
        try:
            yield
        finally:
            owned.remove(token)


@contextmanager
def bundle_read_guard(path, *, immutable=False):
    """Coordinate mutable bundle readers with selection publication/rollback."""
    path = Path(path).resolve()
    resources = Path(__file__).resolve().parent / "resources"
    if immutable or resources in path.parents:
        yield
    else:
        with transaction_guard(path.parent.parent):
            yield


def guarded_bundle_reader(function):
    @wraps(function)
    def checked(path, *args, **kwargs):
        with bundle_read_guard(path, immutable=kwargs.get("immutable", False)):
            return function(path, *args, **kwargs)

    return checked


def archive(root, path):
    path = Path(path)
    if not path.exists():
        return None
    directory = safe_file(root, ".archives")
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = directory / f"{path.name}-{stamp}"
    os.replace(path, target)
    return target


def publish(root, staged, target, force=False, transaction=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not force:
        raise FileExistsError(f"Output exists: {target}; use --force")
    previous = archive(root, target) if force else None
    try:
        os.replace(staged, target)
        sync_dir(target.parent)
    except BaseException:
        if previous is not None:
            if target.exists():
                archive(root, target)
            os.replace(previous, target)
        raise
    if transaction is not None and previous is not None:
        transaction.append((target, previous, sha256(target / "manifest.json")))


@contextmanager
def publication_transaction(root, *, domain=None):
    """Restore force-replaced bundles if a later phase or selection write fails."""
    if domain not in {
        None,
        "prepared",
        "embedding",
        "library",
        "relative_scores",
        "research",
    }:
        raise ValueError("Unknown publication domain")
    with transaction_guard(root):
        actions = []
        try:
            yield actions
        except BaseException as exc:
            for target, previous, new_digest in reversed(actions):
                owner = domain or (
                    "prepared" if target.parent.name == "prepared" else "embedding"
                )
                key = owner + "-" + target.name
                with locked(root, key):
                    if sha256(target / "manifest.json") != new_digest:
                        raise IntegrityError(
                            "Concurrent replacement prevents safe rollback"
                        ) from exc
                    archive(root, target)
                    os.replace(previous, target)
                    sync_dir(target.parent)
            raise


@contextmanager
def staging(parent):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    path = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path)
