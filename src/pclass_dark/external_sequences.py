"""Collect external experimentally annotated sequences (SPEC-0014).

No annotation, reference or network work occurs at import. Public failures are
ValueError (inputs), OSError (I/O), and the exported External*Error types.
"""

import hashlib
import json
import logging
import re
import shutil
import sqlite3
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock, Timeout

from . import external_sequence_data as data
from . import external_sequence_io as io
from .external_sequence_types import (  # noqa: F401 (public API)
    ExternalDownloadError,
    ExternalRecoveryError,
    ExternalSequenceResult,
    ExternalSnapshotError,
    StageCount,
)

LOGGER = logging.getLogger(__name__)


def _safe_paths(root, inputs):
    state = root / ".external-seq"
    owned = [
        state,
        root / "goa_uniprot_all.gaf.gz",
        *(root / name for name in io.OUTPUTS),
    ]
    if state.exists() and not state.is_dir():
        raise ValueError(".external-seq must be a directory")
    for path in [*owned, *(state.rglob("*") if state.is_dir() else [])]:
        if path.is_symlink():
            raise ValueError(f"Symlink at tool-owned path: {path}")
    for path in inputs:
        if path is None:
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Input must be a regular non-symlink file: {path}")
        if path in [root / name for name in io.OUTPUTS] or state in path.parents:
            raise ValueError(f"Input overlaps tool-owned output/state: {path}")
    return state


def _result(root, state, manifest):
    return ExternalSequenceResult(
        root / io.OUTPUTS[0],
        root / io.OUTPUTS[1],
        state / "manifest.json",
        manifest["run_id"],
        tuple(StageCount(**row) for row in manifest["counts"]),
    )


def _batch(ids, run, retries):
    batch_id = hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()
    path = run / "batches" / (batch_id + ".fasta")
    info = io.download(
        io.SEQUENCE_URL,
        path.with_suffix(".download"),
        retries=retries,
        params={
            "query": "(" + " OR ".join(f"accession:{a}" for a in ids) + ")",
            "format": "fasta",
            "includeIsoform": "false",
        },
    )
    temporary = path.with_suffix(".download")
    headers = {k.lower(): v for k, v in info["headers"].items()}
    release = (headers.get("x-uniprot-release"), headers.get("x-uniprot-release-date"))
    if not all(release):
        raise ExternalSnapshotError("Missing UniProt release headers; use --refresh")
    records = data.fasta_records(temporary, set(ids))
    temporary.replace(path)
    io.sync_dir(path.parent)
    return batch_id, path, info, release, records


def _fetch(db, run, workers, batch_size, retries):
    cursor = db.execute(
        "SELECT accession FROM candidates WHERE excluded_swissprot=0 ORDER BY accession"
    )
    release = None
    # Release metadata survives interruption independently of completion order.
    meta_path = run / "retrieval.json"
    if meta_path.exists():
        info = io.read_json(meta_path)
        release = tuple(info["release"]) if info["release"] else None
    iterator_done = False
    pending = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            while not iterator_done or pending:
                while not iterator_done and len(pending) < 2 * workers:
                    ids = [row[0] for row in cursor.fetchmany(batch_size)]
                    if not ids:
                        iterator_done = True
                        break
                    batch_id = hashlib.sha256(
                        ("\n".join(ids) + "\n").encode()
                    ).hexdigest()
                    cached = db.execute(
                        "SELECT response_sha256,release,release_date FROM batches "
                        "WHERE batch_id=? AND status='complete'",
                        (batch_id,),
                    ).fetchone()
                    path = run / "batches" / (batch_id + ".fasta")
                    if (
                        cached
                        and path.is_file()
                        and io.record(path)["sha256"] == cached[0]
                    ):
                        observed = tuple(cached[1:])
                        if release is not None and release != observed:
                            raise ExternalSnapshotError(
                                "Mixed cached releases; use --refresh"
                            )
                        release = observed
                        # Recreate derived sequence state from validated response bytes.
                        data.store_sequences(
                            db, ids, data.fasta_records(path, set(ids))
                        )
                        db.commit()
                        continue
                    future = pool.submit(_batch, ids, run, retries)
                    pending[future] = ids
                if not pending:
                    continue
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    ids = pending.pop(future)
                    batch_id, path, info, observed, records = future.result()
                    if release is not None and release != observed:
                        raise ExternalSnapshotError(
                            "UniProt release changed; use --refresh"
                        )
                    release = observed
                    io.write_json(meta_path, {"schema_version": 1, "release": release})
                    io.write_json(path.with_suffix(".json"), info)
                    data.store_sequences(db, ids, records)
                    db.execute(
                        "INSERT OR REPLACE INTO batches VALUES (?,?,?,?,?,?,?)",
                        (
                            batch_id,
                            json.dumps(ids, separators=(",", ":")),
                            path.name,
                            info["record"]["sha256"],
                            *release,
                            "complete",
                        ),
                    )
                    db.commit()
                    LOGGER.info(
                        "Retrieved batch %s (%d accessions)", batch_id[:12], len(ids)
                    )
        finally:
            for future in pending:
                future.cancel()
    return release


def _report(path, stages, rejections, provenance):
    rows = [["Stage", "Annotation rows", "Accessions", "Sequences"]]
    rows += [
        [
            row.stage,
            *[
                "-" if n is None else str(n)
                for n in (row.annotation_rows, row.accessions, row.sequences)
            ],
        ]
        for row in stages
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = []
    for i, row in enumerate(rows):
        lines.append(
            "  ".join(
                value.ljust(widths[j]) if j == 0 else value.rjust(widths[j])
                for j, value in enumerate(row)
            ).rstrip()
        )
        if i == 0:
            lines.append("  ".join("-" * width for width in widths))
    lines.extend(["", "Rejections:"])
    lines.extend(f"{key}: {value}" for key, value in sorted(rejections.items()))
    lines.extend(["", "Provenance:", json.dumps(provenance, sort_keys=True, indent=2)])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_external_sequences(
    *,
    work_dir: str | Path,
    go_graph: str | Path | None = None,
    swissprot_accessions: str | Path | None = None,
    reference_fasta: str | Path | None = None,
    gaf: str | Path | None = None,
    refresh: bool = False,
    workers: int = 4,
    batch_size: int = 100,
    retries: int = 3,
) -> ExternalSequenceResult:
    """Download/filter all experimental MF accessions and publish a recoverable pair.

    All paths resolve against the caller's cwd. Existing completed outputs are
    reused only under the same source/policy identity; refresh starts a new run.
    This checks local Swiss-Prot accession membership, not current reviewed status
    or sequence homology. Explicit graph pickles must be trusted local inputs.
    """
    if not isinstance(refresh, bool):
        raise ValueError("refresh must be a boolean")
    if swissprot_accessions is not None and reference_fasta is not None:
        raise ValueError(
            "reference_fasta and swissprot_accessions are mutually exclusive"
        )
    for name, value, low, high in (
        ("workers", workers, 1, 32),
        ("batch_size", batch_size, 1, 100),
        ("retries", retries, 0, 10),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not low <= value <= high
        ):
            raise ValueError(f"{name} must be between {low} and {high}")
    root = Path(work_dir).absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("work_dir must be an existing non-symlink directory")
    paths = [
        Path(p).absolute() if p is not None else None
        for p in (go_graph, swissprot_accessions, reference_fasta, gaf)
    ]
    # Canonicalize parent directories without concealing a final-component symlink.
    for p in paths:
        if p is not None and p.is_symlink():
            raise ValueError(f"Input symlink: {p}")
    root = root.resolve()
    paths = [p.resolve() if p is not None else None for p in paths]
    state = _safe_paths(root, paths)
    state.mkdir(exist_ok=True)
    try:
        with FileLock(str(state / "lock"), timeout=0):
            io.recover(root, state)
            old = io.check_owned_outputs(root, state)
            try:
                return _collect_locked(
                    root, state, paths, old, refresh, workers, batch_size, retries
                )
            except sqlite3.Error as exc:
                raise ExternalRecoveryError(
                    f"External sequence database error: {exc}"
                ) from exc
    except Timeout as exc:
        raise RuntimeError("External sequence collection busy in this cwd") from exc


def _collect_locked(root, state, paths, old, refresh, workers, batch_size, retries):
    graph, swiss, fasta, source = paths
    nodes, excluded, reference = data.references(graph, swiss, fasta)
    source_info = {"explicit": True}
    if source is None:
        source, source_info = io.cached_gaf(root, state, refresh, retries)
    source_record = io.record(source)
    identity = {
        "schema_version": 1,
        "policy_version": 1,
        "reference": reference,
        "gaf": {"path": str(source), **source_record},
        "batch_size": batch_size,
        "retries": retries,
    }
    active = state / "active.json"
    if active.exists() and not refresh:
        active_info = io.read_json(active)
        run_id = active_info["run_id"]
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("Invalid active run ID")
        run = state / "runs" / run_id
        info = io.read_json(run / "run.json")
        if info.get("snapshot_failed"):
            raise ExternalSnapshotError("Previous snapshot failure; use --refresh")
        if info["identity"] != identity:
            raise ExternalSnapshotError(
                "Changed run inputs/configuration; use --refresh"
            )
        if info["phase"] == "complete":
            if old is None or old["run_id"] != run_id:
                raise ExternalSnapshotError("Completed run/output mismatch")
            return _result(root, state, old)
    else:
        run = state / "runs" / uuid.uuid4().hex
        run.mkdir(parents=True)
        (run / "batches").mkdir()
        info = {
            "schema_version": 1,
            "policy_version": 1,
            "identity": identity,
            "phase": "parsing",
            "created_at": io.now(),
            "parsed": False,
        }
        io.write_json(run / "run.json", info)
        io.write_json(active, {"schema_version": 1, "run_id": run.name})
    info["workers"] = workers
    db = data.connect(run / "state.sqlite")
    try:
        if not info["parsed"]:
            LOGGER.info(
                "Streaming GAF %s (%d bytes); free disk %d bytes",
                source,
                source_record["size_bytes"],
                shutil.disk_usage(root).free,
            )
            stages, headers = data.parse_gaf(db, source, nodes, excluded)
            info.update(
                parsed=True, counts=[asdict(s) for s in stages], headers=headers
            )
        else:
            stages = [StageCount(**row) for row in info["counts"]]
        info["phase"] = "fetching"
        io.write_json(run / "run.json", info)
        release = _fetch(db, run, workers, batch_size, retries)
        info["phase"] = "deduplicating"
        io.write_json(run / "run.json", info)
        stages, rejections = data.deduplicate(db, stages)
        rejections["invalid_accession_rows"] = info["headers"].get(
            "invalid_accession_rows", 0
        )
        data.write_fasta(db, run / io.OUTPUTS[0])
        if io.record(source) != source_record:
            raise ExternalSnapshotError("GAF changed during collection")
        for name, expected in reference["files"].items():
            if io.record(Path(name)) != expected:
                raise ExternalSnapshotError(f"Reference changed: {name}")
        if reference["runtime"] is not None:
            from .reference_data import resolve_profunc

            if resolve_profunc().manifest != reference["runtime"]:
                raise ExternalSnapshotError("Profunc runtime changed")
        provenance = {
            "created_at": info["created_at"],
            "input_contract": identity,
            "gaf_source": source_info,
            "gaf_headers": info["headers"],
            "retrieval_release": release,
            "experimental_evidence": sorted(data.EVIDENCE),
            "isoforms": "parent approximation",
            "annotation_date": "max qualifying GAF date",
            "tie_break": "lexicographically smallest accession",
            "swissprot_scope": (
                "local primary/specified accessions, not current reviewed status"
            ),
            "fasta_sha256": io.record(run / io.OUTPUTS[0])["sha256"],
        }
        _report(run / io.OUTPUTS[1], stages, rejections, provenance)
        manifest = {
            "schema_version": 1,
            "policy_version": 1,
            "run_id": run.name,
            "created_at": info["created_at"],
            "input_contract": identity,
            "counts": [asdict(s) for s in stages],
            "rejections": rejections,
            "retrieval_release": release,
            "outputs": {
                name: {"path": name, **io.record(run / name)} for name in io.OUTPUTS
            },
        }
        info["phase"] = "ready"
        io.write_json(run / "run.json", info)
        io.publish(root, state, run, manifest)
        if rejections["unresolved_accessions"]:
            LOGGER.warning(
                "Unresolved accessions: %d", rejections["unresolved_accessions"]
            )
        return _result(root, state, manifest)
    except BaseException as exc:
        if isinstance(exc, ExternalSnapshotError):
            info["snapshot_failed"] = True
        info["phase"] = "failed"
        io.write_json(run / "run.json", info)
        if isinstance(exc, EOFError):
            raise ValueError("Truncated source GAF gzip") from exc
        raise
    finally:
        db.close()


def script_main():
    """Run the standalone script's Typer adapter."""
    from .external_sequence_cli import app

    app()
