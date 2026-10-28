"""Reclaim disposable library arrays only after durable complete target results."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from .artifacts import (
    IntegrityError,
    file_record,
    identity,
    locked,
    read_json,
    safe_file,
    sha256,
    sync_dir,
    transaction_guard,
)
from .library_runtime import validate_signature

LOG = logging.getLogger(__name__)
FITS = ("fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "final")
DISPOSABLE = re.compile(
    r"(?:kernels\.[CBV]\.npy|kernels\.json|prefix-\d{4}(?:\.[CB]\.npy|\.json))\Z"
)


def cleanup_target_cache(checkpoint, library_id, cluster_id):
    """Validate durable results, then unlink disposable files under caller's lock.

    Incomplete or inconsistent targets raise IntegrityError without deleting any
    files. Partial deletion after an IO error is safe and retryable.
    """
    checkpoint = Path(checkpoint)
    try:
        if (
            checkpoint.name != "c_" + cluster_id.encode("utf-8").hex()
            or checkpoint.parent.name != library_id
        ):
            raise IntegrityError("Cleanup target directory identity mismatch")
        target = read_json(safe_file(checkpoint, "target.json"))
        if target["cluster_id"] != cluster_id or len(target["fits"]) != len(FITS):
            raise IntegrityError("Cleanup requires all six durable target fits")
        for fit_id, result in zip(FITS, target["fits"], strict=True):
            signature = result["signature"]
            validate_signature(
                signature, library_id=library_id, cluster_id=cluster_id, fit_id=fit_id
            )
            fit_dir = safe_file(checkpoint, fit_id)
            receipt = read_json(safe_file(fit_dir, "complete.json"))
            if (
                receipt["schema"] != "pclass-dark.library-checkpoint.v1"
                or receipt["task_id"]
                != identity(
                    [library_id, cluster_id, fit_id, signature["input_identity"]]
                )
                or receipt["files"] != {}
                or receipt["arrays"] != {}
                or receipt["payload_id"] != identity(result)
                or receipt["payload"] != result
                or identity(result["trace"]) != signature["selection_trace_identity"]
                or len(result["profiles"]) != 640
                or [p["dimension"] for p in result["profiles"]] != list(range(640))
            ):
                raise IntegrityError(f"Inconsistent durable cleanup result: {fit_id}")
        # Resolve every candidate before the first unlink: symlink rejection must
        # not leave a partially cleaned target. No recursive/wildcard deletion.
        files = [
            safe_file(safe_file(checkpoint, fit_id), p.name)
            for fit_id in FITS
            for p in safe_file(checkpoint, fit_id).iterdir()
            if DISPOSABLE.fullmatch(p.name)
        ]
        if any(not p.is_file() for p in files):
            raise IntegrityError("Disposable cache member is not a regular file")
        released = sum(p.stat().st_size for p in files)
    except (KeyError, TypeError, AttributeError, IndexError, ValueError) as exc:
        raise IntegrityError(
            f"Cannot clean incomplete/inconsistent target: {exc}"
        ) from exc
    for path in files:
        path.unlink()
    for fit_id in FITS:
        sync_dir(checkpoint / fit_id)
    if released:
        LOG.info(
            "Cluster %s: released %.3f GiB disposable cache",
            cluster_id,
            released / 2**30,
        )
    return {"targets": 1, "files": len(files), "bytes": released}


def clean_library_cache(output_dir):
    """Clean verified completed clusters across checkpoint versions; skip partials.

    Raises SetupBusyError if a build owns the root. Never touch prepared data,
    embeddings, published libraries, or partial/inconsistent target caches.
    """
    root = Path(output_dir).absolute()
    totals = {"targets": 0, "files": 0, "bytes": 0, "skipped": 0}
    with transaction_guard(root):
        private = safe_file(root, ".checkpoints")
        if not private.exists():
            return _clean_new_caches(root)
        for library in sorted(private.iterdir()):
            if library.is_symlink() or not re.fullmatch(r"[0-9a-f]{64}", library.name):
                raise IntegrityError("Invalid checkpoint library directory")
            for checkpoint in sorted(library.iterdir()):
                if not checkpoint.name.startswith("c_"):
                    continue
                checkpoint = safe_file(library, checkpoint.name)
                try:
                    cluster_id = bytes.fromhex(checkpoint.name[2:]).decode("utf-8")
                except (ValueError, UnicodeError) as exc:
                    raise IntegrityError("Invalid encoded cluster directory") from exc
                with locked(
                    root, "library-fit-" + identity([library.name, cluster_id])
                ):
                    try:
                        result = cleanup_target_cache(
                            checkpoint, library.name, cluster_id
                        )
                    except (IntegrityError, FileNotFoundError) as exc:
                        totals["skipped"] += 1
                        LOG.debug("Preserving cache for %s: %s", cluster_id, exc)
                        continue
                    for key, value in result.items():
                        totals[key] += value
        for key, value in _clean_new_caches(root).items():
            totals[key] += value
    return totals


def _clean_new_caches(root):
    """Caller owns the root; validate mode-specific gates and lock each fit."""
    from .fit_records import payload_identity, validate_result
    from .library_build import _clean_fit
    from .library_checkpoints import Checkpoints

    totals = {"targets": 0, "files": 0, "bytes": 0, "skipped": 0}
    seen = set()
    candidates = [
        *root.glob(".production-checkpoints/*/c_*/target.json"),
        *root.glob(".research-checkpoints/*/c_*/fold_*-target.json"),
    ]
    for path in sorted(candidates):
        try:
            record = read_json(safe_file(root, str(path.relative_to(root))))
            fit_id = record["fit_id"]
            if fit_id in seen:
                continue
            with locked(root, "fit-" + fit_id):
                inputs = record["fit_inputs"]
                if (
                    record["schema"] != "pclass-dark.target-result.v2"
                    or record["status"] != "complete"
                    or identity(inputs) != fit_id
                ):
                    raise IntegrityError("Invalid mode-specific cleanup target")
                file = safe_file(root, record["result"])
                if sha256(file) != record["result_sha256"]:
                    raise IntegrityError("Cleanup complete receipt checksum mismatch")
                receipt = read_json(file)
                result = receipt["payload"]
                signature = result["signature"]
                fit = signature["fit_id"]
                if record["required_fits"] != [fit] or record[
                    "payload_id"
                ] != payload_identity(result):
                    raise IntegrityError("Cleanup scientific payload mismatch")
                validate_result(
                    result,
                    cluster=record["cluster_id"],
                    fit=fit,
                    counts=signature["counts"],
                    sigma_floor=inputs["sigma_floor"],
                    validation_ids=[r["window_id"] for r in result["oof"]],
                )
                if (
                    signature["input_identity"] != inputs
                    or signature["library_id"] != fit_id
                    or receipt["task_id"]
                    != identity([fit_id, record["cluster_id"], fit, inputs])
                ):
                    raise IntegrityError("Cleanup normalized identity mismatch")
                if fit == "final":
                    if record["result"] != f".fits/{fit_id}/final/complete.json":
                        raise IntegrityError("Unexpected production cleanup fit path")
                    relative = safe_file(root, record["relative"])
                    if relative != root / ".relative-checkpoints" / fit_id:
                        raise IntegrityError("Unexpected distribution gate path")
                    if (
                        sha256(relative / "complete.json")
                        != record["relative_receipt_sha256"]
                    ):
                        raise IntegrityError("Cleanup distribution receipt mismatch")
                    gate = read_json(relative / "complete.json")
                    if (
                        gate["payload_id"] != identity(gate["payload"])
                        or gate["payload"]["payload_id"] != payload_identity(result)
                        or gate["payload"]["fit_id"] != fit_id
                        or gate["payload"]["status"] != signature["status"]
                        or gate["payload"]["counts"]
                        != {"C": inputs["C_n"], "B": inputs["B_n"]}
                    ):
                        raise IntegrityError("Unverified distribution cleanup gate")
                    for name, expected in gate["files"].items():
                        if file_record(safe_file(relative, name)) != expected:
                            raise IntegrityError("Missing/corrupt distribution output")
                elif (
                    not file.is_relative_to(root / ".research-checkpoints")
                    or file.parent.name != fit
                    or file.parent.parent != path.parent
                ):
                    raise IntegrityError("Unexpected research cleanup fit path")
                files = [
                    p for p in file.parent.iterdir() if DISPOSABLE.fullmatch(p.name)
                ]
                released = sum(p.stat().st_size for p in files)
                store = Checkpoints(file.parent, receipt["task_id"])
                _clean_fit(store, result)
                totals["targets"] += 1
                totals["files"] += len(files)
                totals["bytes"] += released
                seen.add(fit_id)
        except (
            ValueError,
            OSError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
        ) as exc:
            totals["skipped"] += 1
            LOG.debug("Preserving mode-specific cache at %s: %s", path, exc)
    return totals
