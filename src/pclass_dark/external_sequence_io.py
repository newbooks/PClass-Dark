"""Bounded HTTP, durable metadata and recoverable paired publication."""

import gzip
import hashlib
import json
import logging
import os
import re
import shutil
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

from .external_sequence_types import (
    ExternalDownloadError,
    ExternalRecoveryError,
    ExternalSnapshotError,
)

GAF_URL = "https://ftp.ebi.ac.uk/pub/databases/GO/goa/UNIPROT/goa_uniprot_all.gaf.gz"
SEQUENCE_URL = "https://rest.uniprot.org/uniprotkb/stream"
OUTPUTS = ("external_sequences.fasta", "external_filtered.txt")


def now():
    return datetime.now(timezone.utc).isoformat()


def record(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"size_bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_file(path):
    with Path(path).open("rb") as handle:
        os.fsync(handle.fileno())


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    sync_dir(path.parent)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("policy_version", 1) != 1
    ):
        raise ValueError(f"Unsupported external sequence state: {path}")
    return value


def _log_gaf_progress(size, total):
    downloaded = f"{size // 1000:,}"
    if total is not None and size <= total:
        message = (
            f"Downloaded GOA: {downloaded} / {total // 1000:,} KB "
            f"({100 * size / total:.1f}%)"
        )
    else:
        message = f"Downloaded GOA: {downloaded} KB (percentage unavailable)"
    logging.getLogger(__name__).info(message)


def _response_total(headers):
    headers = {key.lower(): value for key, value in headers.items()}
    if headers.get("content-encoding", "identity").lower() != "identity":
        return None
    try:
        total = int(headers.get("content-length", ""))
    except (TypeError, ValueError):
        return None
    return total if total > 0 else None


def download(url, path, *, params=None, retries=3):
    """Stream one response to disk; transient failures retry from byte zero."""
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    for attempt in range(retries + 1):
        if url == GAF_URL:
            logging.getLogger(__name__).info(
                "Downloading GOA (attempt %d/%d)", attempt + 1, retries + 1
            )
        delay = min(60, 2**attempt)
        try:
            with requests.Session() as session:
                with session.get(
                    url, params=params, stream=True, timeout=(10, 120)
                ) as response:
                    if response.status_code != 200:
                        transient = response.status_code in {
                            408,
                            429,
                            500,
                            502,
                            503,
                            504,
                        }
                        if not transient:
                            raise ExternalDownloadError(
                                f"HTTP {response.status_code} from {url}"
                            )
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = float(retry_after)
                            except ValueError:
                                try:
                                    date = parsedate_to_datetime(retry_after)
                                    if date.tzinfo is None:
                                        date = date.replace(tzinfo=timezone.utc)
                                    wait = (
                                        date - datetime.now(timezone.utc)
                                    ).total_seconds()
                                except (ValueError, TypeError, OverflowError):
                                    wait = 0  # Invalid advice: use exponential backoff.
                            delay = max(delay, wait)
                        if delay > 300:
                            raise ExternalDownloadError(
                                "Retry-After exceeds 300 seconds; resume later"
                            )
                        raise requests.ConnectionError(
                            f"HTTP {response.status_code} from {url}"
                        )
                    total = _response_total(response.headers)
                    digest = hashlib.sha256()
                    size = 0
                    with tmp.open("wb") as handle:
                        for chunk in response.iter_content(1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                                digest.update(chunk)
                                before = size
                                size += len(chunk)
                                if url == GAF_URL and size // (
                                    128 * 1024**2
                                ) != before // (128 * 1024**2):
                                    _log_gaf_progress(size, total)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if url == GAF_URL:
                        _log_gaf_progress(size, total)
                    headers = dict(response.headers)
            os.replace(tmp, path)
            sync_dir(path.parent)
            return {
                "schema_version": 1,
                "url": url,
                "retrieved_at": now(),
                "headers": headers,
                "record": {"size_bytes": size, "sha256": digest.hexdigest()},
            }
        except requests.RequestException as exc:
            if attempt == retries:
                raise ExternalDownloadError(f"Download failed: {url}: {exc}") from exc
            time.sleep(delay)
        finally:
            tmp.unlink(missing_ok=True)
    raise AssertionError("unreachable")


def cached_gaf(root, state, refresh, retries):
    path = root / "goa_uniprot_all.gaf.gz"
    meta = state / "gaf.json"
    previous = state / "gaf.previous"
    previous_meta = state / "gaf.previous.json"
    if previous.exists():
        registered = False
        try:
            registered = meta.exists() and read_json(meta).get("record") == record(path)
        except (OSError, ValueError):
            pass
        if registered:
            previous.unlink()
        else:
            os.replace(previous, path)
            if previous_meta.exists():
                os.replace(previous_meta, meta)
            else:
                meta.unlink(missing_ok=True)
        previous_meta.unlink(missing_ok=True)
        sync_dir(root)
        sync_dir(state)
    if path.exists() and not refresh:
        if not meta.exists() or read_json(meta).get("record") != record(path):
            raise ExternalSnapshotError(
                "Unregistered/changed GAF cache; use --gaf or --refresh"
            )
        return path, read_json(meta)
    staged = state / "gaf.download"
    info = download(GAF_URL, staged, retries=retries)
    try:
        try:
            with gzip.open(staged, "rb") as handle:
                for _ in iter(lambda: handle.read(1024 * 1024), b""):
                    pass
        except EOFError as exc:
            raise ValueError("Truncated downloaded GAF gzip") from exc
        previous = state / "gaf.previous"
        previous_meta = state / "gaf.previous.json"
        had_file, had_meta = path.exists(), meta.exists()
        if had_meta:
            shutil.copyfile(meta, previous_meta)
            sync_file(previous_meta)
        if had_file:
            # A hard link retains the old inode without copying a 12 GB source.
            previous.unlink(missing_ok=True)
            os.link(path, previous)
            sync_dir(state)
        try:
            os.replace(staged, path)
            sync_dir(root)
            write_json(meta, info)
        except BaseException:
            if had_file:
                os.replace(previous, path)
            else:
                path.unlink(missing_ok=True)
            if had_meta:
                os.replace(previous_meta, meta)
            else:
                meta.unlink(missing_ok=True)
            sync_dir(root)
            sync_dir(state)
            raise
        previous.unlink(missing_ok=True)
        previous_meta.unlink(missing_ok=True)
        sync_dir(state)
    finally:
        staged.unlink(missing_ok=True)
    return path, info


def verify_outputs(root, manifest):
    for name in OUTPUTS:
        path = root / name
        expected = manifest["outputs"][name]
        if not path.is_file() or record(path) != {
            k: expected[k] for k in ("size_bytes", "sha256")
        }:
            raise ExternalSnapshotError(
                f"Changed output {path}; restore its recorded bytes or move both "
                "outputs and .external-seq aside before starting again"
            )


def check_owned_outputs(root, state, *, refresh=False):
    manifest_path = state / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        # Even refresh must not overwrite user-modified outputs.
        verify_outputs(root, manifest)
        return manifest
    if any((root / name).exists() for name in OUTPUTS):
        raise ExternalSnapshotError("Unregistered outputs; move them aside manually")
    return None


def recover(root, state):
    """Roll back an interrupted replacement, or finish a verified completed pair."""
    journal_path = state / "publication.json"
    if not journal_path.exists():
        return
    journal = read_json(journal_path)
    run = state / "runs" / journal["run_id"]
    if not re.fullmatch(r"[0-9a-f]{32}", journal["run_id"]):
        raise ExternalRecoveryError("Invalid publication run ID")
    committed = False
    try:
        manifest = read_json(state / "manifest.json")
        verify_outputs(root, manifest)
        committed = manifest == journal["new"]
    except (OSError, ValueError, ExternalSnapshotError):
        pass
    if not committed:
        old = journal["old"]
        required = {*OUTPUTS, "manifest.json"} if old is not None else set()
        if set(journal["backups"]) != required:
            raise ExternalRecoveryError("Incomplete rollback backup inventory")
        # Verify every backup before touching any public file.
        for name, expected in journal["backups"].items():
            backup = run / "backups" / name
            if name not in (*OUTPUTS, "manifest.json"):
                raise ExternalRecoveryError("Invalid backup name")
            if not backup.is_file() or record(backup) != expected:
                raise ExternalRecoveryError(
                    f"Missing/corrupt rollback backup: {backup}"
                )
        for name in OUTPUTS:
            target = root / name
            if old is None:
                target.unlink(missing_ok=True)
            else:
                restore = run / (name + ".restore")
                shutil.copyfile(run / "backups" / name, restore)
                sync_file(restore)
                os.replace(restore, target)
        if old is None:
            (state / "manifest.json").unlink(missing_ok=True)
        else:
            write_json(state / "manifest.json", old)
    run_info = read_json(run / "run.json")
    run_info["phase"] = "complete" if committed else "ready"
    write_json(run / "run.json", run_info)
    sync_dir(root)
    sync_dir(state)
    journal_path.unlink()
    sync_dir(state)


def publish(root, state, run, manifest):
    old = check_owned_outputs(root, state)
    backups = run / "backups"
    backups.mkdir(exist_ok=True)
    hashes = {}
    if old is not None:
        for name, source in [
            *((name, root / name) for name in OUTPUTS),
            ("manifest.json", state / "manifest.json"),
        ]:
            target = backups / name
            shutil.copyfile(source, target)
            sync_file(target)
            hashes[name] = record(target)
        sync_dir(backups)
    for name in OUTPUTS:
        sync_file(run / name)
    sync_dir(run)
    write_json(
        state / "publication.json",
        {
            "schema_version": 1,
            "run_id": run.name,
            "old": old,
            "new": manifest,
            "backups": hashes,
        },
    )
    for name in OUTPUTS:
        os.replace(run / name, root / name)
    sync_dir(root)
    write_json(state / "manifest.json", manifest)
    recover(root, state)
