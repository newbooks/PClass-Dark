"""Bounded read-only inventories and descriptor-relative cleanup IO."""

import os
import shutil
import stat
from pathlib import Path

from .artifacts import IntegrityError, canonical, read_json, sha256, sync_dir

MAX_JSON = 64 * 2**20
MAX_ENTRIES = 1_000_000
MAX_INVENTORY = 512 * 2**20


class UnsafeTree(IntegrityError):
    """A candidate contains files that must be preserved."""


def read_metadata(path):
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > MAX_JSON:
        raise IntegrityError(f"Unsafe or oversized cleanup metadata: {path}")
    return read_json(path)


def inventory(path):
    path = Path(path)
    device = path.lstat().st_dev
    files, directories = [], []
    pending = [(path, ".")]
    while pending:
        current, relative = pending.pop()
        st = current.lstat()
        if st.st_dev != device:
            raise UnsafeTree(f"Cross-device subtree: {current}")
        base = dict(path=relative, device=st.st_dev, inode=st.st_ino, mode=st.st_mode)
        if stat.S_ISDIR(st.st_mode):
            directories.append(base)
            with os.scandir(current) as children:
                for child in children:
                    name = (
                        child.name if relative == "." else relative + "/" + child.name
                    )
                    pending.append((Path(child.path), name))
        elif stat.S_ISREG(st.st_mode):
            files.append(
                {
                    **base,
                    "sha256": sha256(current),
                    "size_bytes": st.st_size,
                    "nlink": st.st_nlink,
                    "mtime_ns": st.st_mtime_ns,
                    "blocks": getattr(st, "st_blocks", None),
                }
            )
        else:
            raise UnsafeTree(f"Symlink or special file: {current}")
        if len(files) + len(directories) + len(pending) > MAX_ENTRIES:
            raise IntegrityError("Cleanup inventory exceeds 1,000,000 entries")
    result = {
        "files": sorted(files, key=lambda f: f["path"]),
        "directories": sorted(directories, key=lambda f: f["path"]),
    }
    if len(canonical(result)) > MAX_INVENTORY:
        raise IntegrityError("Cleanup inventory exceeds 512 MiB")
    return result


def relative_path(value):
    p = Path(value)
    if (
        not isinstance(value, str)
        or p.is_absolute()
        or not p.parts
        or any(part in {".", ".."} for part in p.parts)
        or p.as_posix() != value
    ):
        raise IntegrityError(f"Unsafe cleanup relative path: {value}")
    return p


def _parent(root, relative):
    """Return an open, nonfollowing parent descriptor and basename."""
    p = relative_path(relative)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in p.parts[:-1]:
            next_fd = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd
            )
            os.close(fd)
            fd = next_fd
        return fd, p.name
    except BaseException:
        os.close(fd)
        raise


def move(root, source, destination):
    a, name = _parent(root, source)
    try:
        b, dest = _parent(root, destination)
        try:
            # Destination existence is checked without following links.
            try:
                os.stat(dest, dir_fd=b, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise IntegrityError("Cleanup rename destination already exists")
            os.rename(name, dest, src_dir_fd=a, dst_dir_fd=b)
            os.fsync(a)
            os.fsync(b)
        finally:
            os.close(b)
    finally:
        os.close(a)


def make_directory(path):
    path = Path(path)
    if path.is_symlink():
        raise IntegrityError(f"Symlink cleanup directory: {path}")
    path.mkdir(exist_ok=True)
    sync_dir(path.parent)


def remove_tree(root, relative, expected):
    """Unlink committed trash without hashing, relinking, or per-file fsync.

    Keep a nonfollowing parent descriptor, validate the directory identity and
    use shutil's descriptor-based recursive removal. Partial trees are resumable.
    """
    if not shutil.rmtree.avoids_symlink_attacks:
        raise RuntimeError("Cleanup requires symlink-resistant recursive removal")
    fd, name = _parent(root, relative)
    try:
        try:
            st = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            os.fsync(fd)
            return
        if (
            not stat.S_ISDIR(st.st_mode)
            or st.st_ino != expected["inode"]
            or st.st_dev != expected["device"]
            or st.st_mode != expected["mode"]
        ):
            raise IntegrityError("Quarantined directory identity changed")
        shutil.rmtree(name, dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)
