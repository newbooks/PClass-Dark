"""Experimental end-to-end feature extraction; see SPEC-0021."""

from __future__ import annotations

import hashlib
import importlib.metadata
import logging
import multiprocessing
import os
import platform
import re
import shutil
import sqlite3
import time
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from filelock import FileLock, Timeout
from threadpoolctl import threadpool_limits

from . import feature_reports as reports
from . import feature_store as store
from .artifacts import (
    IntegrityError,
    SetupBusyError,
    canonical,
    identity,
    read_json,
    safe_file,
    sha256,
    sync_dir,
    write_json,
)
from .structural_features import (
    FEATURES,
    HYDROPATHY,
    THRESHOLD,
    summarize_structural_features,
)
from .structural_models import verify_structural_models
from .structure import StructuralPrediction, StructuralPredictor

LOG = logging.getLogger(__name__)
WORK_SCHEMA = "pclass-dark.feature-work.v1"
BUDGET = 256 * 1024 * 1024


@dataclass(frozen=True)
class FeatureExtractionResult:
    tsv_path: Path
    qc_path: Path
    sequences_path: Path
    manifest_path: Path
    extraction_id: str
    sequence_count: int
    membership_count: int
    cluster_count: int
    computed_sequences: int
    reused_sequences: int
    devices: tuple[str, ...]
    reused_output: bool


def _versions():
    result = {"python": platform.python_version()}
    for package in ("numpy", "scipy", "torch", "transformers", "safetensors"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not_installed"
    return result


def _code_hashes():
    modules = (
        "feature_extraction",
        "feature_store",
        "feature_reports",
        "structural_features",
        "structure",
        "structural_models",
        "embeddings",
        "setup_data",
        "reference_data",
        "artifacts",
        "paths",
    )
    code = {name: sha256(Path(__file__).with_name(name + ".py")) for name in modules}
    return code


def _contract(manifest, manifest_hash, bundle, batch):
    from .embeddings import ASSET_HASHES
    from .setup_data import MODEL, REVISION

    code = _code_hashes()
    return {
        "schema": "pclass-dark.feature-contract.v1",
        "preparation_id": manifest["id"],
        "prepared_manifest_sha256": manifest_hash,
        "input_files": manifest["files"],
        "model_id": bundle.model_id,
        "model_manifest_sha256": sha256(bundle.path / "manifest.json"),
        "model_tensor_sha256": sha256(bundle.path / "models.safetensors"),
        "esm": {
            "model": MODEL,
            "revision": REVISION,
            "assets": ASSET_HASHES,
            "layer": 30,
            "width": 640,
            "dtype": "float32",
            "window_length": 1021,
            "overlap": 100,
            "ownership": "center-selected-earlier-start",
        },
        "features": list(FEATURES),
        "hydropathy": HYDROPATHY,
        "threshold": THRESHOLD,
        "policies": {
            "aggregation": "prostruct-v2-f64-threshold-v1",
            "precision": ".3f",
            "undefined": "NA",
            "scd": "fft-integer-1e-6-direct-fallback-v1",
            "sequence_limit": store.LIMIT,
            "header_limit": 4096,
            "row_unit": "retained-sequence-cluster-membership",
        },
        "code": code,
        "versions": _versions(),
        "window_batch_size": batch,
        "seed": 42,
    }


def _options(device, offline, batch, workers, resume, force):
    if not isinstance(device, str) or not re.fullmatch(
        r"auto|cpu|cuda(?::\d+)?", device
    ):
        raise ValueError("Device must be auto, cpu, cuda or cuda:<index>")
    if any(type(v) is not bool for v in (offline, resume, force)):
        raise ValueError("offline, resume and force must be boolean")
    if resume and force:
        raise ValueError("--resume and --force cannot be combined")
    if (
        type(batch) is not int
        or not 1 <= batch <= 8
        or type(workers) is not int
        or not 1 <= workers <= 8
    ):
        raise ValueError(
            "window_batch_size and workers must be integers between 1 and 8"
        )


def _family(output, prepared, root, bundle):
    from .paths import PACKAGE_DIR, default_library_output_dir, default_setup_input_dir

    path = Path(output).absolute()
    if (
        path.suffix != ".tsv"
        or not path.stem
        or not path.parent.is_dir()
        or any(ord(c) < 32 for c in path.name)
    ):
        raise ValueError("Output must be a .tsv path with an existing parent")
    family = reports.Family(path)
    protected = [
        prepared,
        bundle.path,
        PACKAGE_DIR / "resources",
        Path(default_library_output_dir()),
        Path(default_setup_input_dir()),
    ]
    if prepared.parent.name == "prepared":
        protected.append(root)
    for parent in protected:
        parent = parent.resolve()
        if path.parent.resolve() == parent or parent in path.parent.resolve().parents:
            raise ValueError(
                "Experimental output cannot be inside protected input/data directories"
            )
    for target in [
        *(family.path(n) for n in family.names),
        family.work,
        family.lock,
        family.archives,
    ]:
        safe_file(family.parent, target.name)
        if target.exists() and target.is_file():
            stamp = (target.stat().st_dev, target.stat().st_ino)
            for source_root in (prepared, bundle.path):
                for source in source_root.iterdir():
                    if source.is_file() and stamp == (
                        source.stat().st_dev,
                        source.stat().st_ino,
                    ):
                        raise ValueError("Output hardlinks an input file")
    return reports.Family(path.resolve())


def _owned_run(family):
    for path in family.work.rglob("*"):
        safe_file(family.work, str(path.relative_to(family.work)))
    run = read_json(safe_file(family.work, "run.json"))
    if (
        run.get("schema") != WORK_SCHEMA
        or run.get("output") != str(family.output)
        or run.get("status") not in ("indexing", "ready", "running", "complete")
        or run.get("contract_id") != identity(run["contract_inputs"])
    ):
        raise IntegrityError("Unrecognized feature work directory")
    allowed = {
        "run.json",
        "state.sqlite",
        "state.sqlite-wal",
        "state.sqlite-shm",
        "models",
        "stage",
        "publication.json",
    }
    if any(p.name not in allowed for p in family.work.iterdir()):
        raise IntegrityError("Unrecognized file in feature work directory")
    from .structural_models import bundled_structural_models

    model_names = {p.name for p in bundled_structural_models().iterdir()}
    for subdir, names in (("models", model_names), ("stage", set(family.names))):
        folder = family.work / subdir
        if folder.exists() and any(
            p.name not in names or not p.is_file() for p in folder.iterdir()
        ):
            raise IntegrityError("Unrecognized nested feature work file")
    if run.get("extraction_id") is not None:
        eid, fp = reports.extraction_identity(
            run["contract_id"], run["execution_metadata"]
        )
        if run["extraction_id"] != eid or run["execution_fingerprint"] != fp:
            raise IntegrityError("Feature work identity mismatch")
    return run


def _plan(workers):
    available = max(1, getattr(os, "process_cpu_count", os.cpu_count)() or 1)
    admitted = min(workers, 8, max(0, available - 1))
    processes = admitted if workers > 1 and admitted >= 2 else 0
    return {
        "workers_requested": workers,
        "workers_used": processes or 1,
        "processes": processes,
        "cpu_threads": 1 if processes else min(8, available),
        "task_limit": 2 * processes if processes else 1,
        "payload_budget": BUDGET,
        "max_reserved_bytes": 0,
        "oversized_sequences": [],
    }


def _device_matches(requested, actual):
    return (
        actual == "not_run"
        or requested == "auto"
        or requested == actual
        or requested == "cuda"
        and actual.startswith("cuda:")
    )


def _metadata(predictor):
    from .embeddings import ASSET_HASHES
    from .setup_data import MODEL, REVISION

    metadata = predictor.prepare()
    contract = metadata["embedding_contract"]
    if (
        contract.get("model") != MODEL
        or contract.get("revision") != REVISION
        or contract.get("assets") is None
        or any(contract["assets"].get(k) != v for k, v in ASSET_HASHES.items())
        or contract.get("layer") != 30
        or contract.get("width") != 640
        or contract.get("inference_dtype") != "float32"
        or not re.fullmatch(r"cpu|cuda:\d+", metadata["device"])
    ):
        raise IntegrityError("Structural runtime differs from pinned representation")
    return metadata


def _aggregate(sequence, prediction):
    with threadpool_limits(limits=1):
        return summarize_structural_features(sequence, prediction)


def _compute(db, run, predictor, plan, save_run):
    eid = run["extraction_id"]
    # Validate every completed receipt even when it no longer corresponds to pending
    # rows.
    if db.execute(
        "SELECT 1 FROM completed c LEFT JOIN sequences s USING(sequence_id) WHERE "
        "s.sequence_id IS NULL LIMIT 1"
    ).fetchone():
        raise IntegrityError("Orphan feature receipt")
    reused = 0
    for row in db.execute("SELECT * FROM sequences ORDER BY accession_id"):
        if store.read_record(db, row, eid) is not None:
            reused += 1
    computed = 0
    uncommitted = 0
    last_commit = time.monotonic()
    inflight = {}
    reserved = 0
    executor = (
        ProcessPoolExecutor(
            max_workers=plan["processes"],
            mp_context=multiprocessing.get_context("spawn"),
        )
        if plan["processes"]
        else None
    )

    def consume(row, summary, device):
        nonlocal computed, uncommitted, last_commit
        record = store.make_record(row[0], row[1], row[2], summary, eid, device)
        text = canonical(record).decode()
        store.validate_record(store.loads(text), row, eid)
        db.execute(
            "INSERT INTO completed VALUES(?,?,?)",
            (row[0], text, hashlib.sha256(text.encode()).hexdigest()),
        )
        computed += 1
        uncommitted += 1
        if uncommitted >= 32 or time.monotonic() - last_commit >= 10:
            db.commit()
            uncommitted = 0
            last_commit = time.monotonic()
        if (computed + reused) % 1000 == 0:
            LOG.info("Features complete: %d", computed + reused)

    def drain(all_tasks=False):
        nonlocal reserved
        if not inflight:
            return
        done, _ = wait(
            inflight, return_when=ALL_COMPLETED if all_tasks else FIRST_COMPLETED
        )
        for future in done:
            row, size, device = inflight.pop(future)
            try:
                summary = future.result()
            except Exception as exc:
                raise RuntimeError(
                    f"Feature worker failed for {row[1]} (length={row[3]}): {exc}"
                ) from exc
            consume(row, summary, device)
            reserved -= size

    try:
        for row in db.execute(
            "SELECT s.* FROM sequences s LEFT JOIN completed c USING(sequence_id) "
            "WHERE c.sequence_id IS NULL ORDER BY s.accession_id"
        ):
            sid, accession, sequence, length, count = row
            size = 64 * length + 4096
            oversized = size > BUDGET
            while inflight and (
                oversized
                or len(inflight) >= plan["task_limit"]
                or reserved + size > BUDGET
            ):
                drain()
            if oversized:
                LOG.warning(
                    "Oversized sequence %s (%d residues): serial aggregation",
                    accession,
                    length,
                )
                plan["oversized_sequences"].append(
                    {"accession_id": accession, "length": length}
                )
            plan["max_reserved_bytes"] = max(
                plan["max_reserved_bytes"], reserved + size
            )
            try:
                if count:
                    prediction = predictor.predict(sequence)
                    if not _device_matches(
                        prediction.device, run["execution_metadata"]["device"]
                    ):
                        raise RuntimeError("Structural device changed mid-run")
                    device = run["execution_metadata"]["device"]
                else:
                    from .structural_models import BUNDLE_ID

                    prediction = StructuralPrediction(
                        sequence,
                        np.full(length, np.nan, np.float32),
                        np.full(length, "NA", dtype="<U2"),
                        BUNDLE_ID,
                        "cpu",
                    )
                    device = "not_run"
                if executor is not None and not oversized:
                    future = executor.submit(_aggregate, sequence, prediction)
                    inflight[future] = (row, size, device)
                    reserved += size
                else:
                    consume(row, _aggregate(sequence, prediction), device)
            except (ValueError, RuntimeError, MemoryError) as exc:
                actual_device = (
                    run["execution_metadata"]["device"] if count else "not_run"
                )
                window_batch = run["contract_inputs"]["window_batch_size"]
                raise RuntimeError(
                    f"Feature extraction failed for {accession}, length={length}, "
                    f"device={actual_device}, "
                    f"window_batch_size={window_batch}: {exc}"
                ) from exc
        drain(all_tasks=True)
        db.commit()
    except BaseException:
        # Only fully validated receipts have entered this transaction.
        db.commit()
        for future in inflight:
            future.cancel()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    LOG.info("Features complete: %d computed, %d reused", computed, reused)
    save_run()
    return computed, reused


def _stage(db, run, family, execution):
    stage = safe_file(family.work, "stage")
    if stage.exists():
        if any(
            p.name not in family.names or p.is_symlink() or not p.is_file()
            for p in stage.iterdir()
        ):
            raise IntegrityError("Unknown staged feature file")
        shutil.rmtree(stage)
    stage.mkdir()
    counts = {
        "unique_sequences": 0,
        "memberships": db.execute("SELECT COUNT(*) FROM memberships").fetchone()[0],
        "retained_clusters": db.execute(
            "SELECT COUNT(DISTINCT cluster_id) FROM memberships"
        ).fetchone()[0],
        "with_noncanonical": 0,
        "no_canonical": 0,
        "with_undefined_features": 0,
        "rsa_out_of_range_residues": 0,
        "scd_fallbacks": 0,
    }

    def main_rows():
        for row in db.execute("SELECT * FROM sequences ORDER BY accession_id"):
            record = store.read_record(db, row, run["extraction_id"])
            if record is None:
                raise IntegrityError("Cannot publish missing sequence features")
            for (cluster,) in db.execute(
                "SELECT cluster_id FROM memberships WHERE sequence_id=? ORDER BY "
                "cluster_id",
                (row[0],),
            ):
                yield (row[1], *record["values"], cluster)

    files = {
        family.names[0]: reports.write_table(
            stage / family.names[0], reports.HEADER, main_rows()
        )
    }

    def qc_rows():
        for row in db.execute("SELECT * FROM sequences ORDER BY accession_id"):
            record = store.read_record(db, row, run["extraction_id"])
            qc = record["qc"]
            counts["unique_sequences"] += 1
            counts["with_noncanonical"] += int(qc["noncanonical_count"] > 0)
            counts["no_canonical"] += int(qc["canonical_count"] == 0)
            counts["with_undefined_features"] += int(bool(qc["undefined_features"]))
            counts["rsa_out_of_range_residues"] += qc["rsa_out_of_range_count"]
            counts["scd_fallbacks"] += int(qc["scd_method"] == "direct_fallback")
            values = {
                "accession_id": row[1],
                "sequence_id": row[0],
                **qc,
                "device": record["device"],
                "undefined_feature_count": len(qc["undefined_features"]),
                "undefined_features": ",".join(qc["undefined_features"]) or "none",
            }
            yield [values[k] for k in reports.QC_HEADER]

    files[family.names[1]] = reports.write_table(
        stage / family.names[1], reports.QC_HEADER, qc_rows()
    )

    def index_rows():
        yield from db.execute(
            "SELECT s.accession_id,s.sequence_id,s.length,COUNT(m.cluster_id) "
            "FROM sequences s JOIN memberships m USING(sequence_id) GROUP BY "
            "s.sequence_id ORDER BY s.accession_id"
        )

    files[family.names[2]] = reports.write_table(
        stage / family.names[2], reports.INDEX_HEADER, index_rows()
    )
    manifest = {
        "schema": reports.SCHEMA,
        "status": "complete",
        "id": run["extraction_id"],
        "contract_id": run["contract_id"],
        "execution_fingerprint": run["execution_fingerprint"],
        "identity_inputs": {
            "contract_inputs": run["contract_inputs"],
            "execution_metadata": run["execution_metadata"],
        },
        "inputs": run["contract_inputs"],
        "counts": counts,
        "execution": execution,
        "files": files,
    }
    write_json(stage / family.names[-1], manifest)
    reports.verify_family(family, directory=stage, db=db)


def _result(family, manifest, computed, reused, reused_output):
    c = manifest["counts"]
    metadata = manifest["identity_inputs"]["execution_metadata"]
    devices = () if metadata == "not_run" else (metadata["device"],)
    return FeatureExtractionResult(
        *(family.path(n) for n in family.names),
        manifest["id"],
        c["unique_sequences"],
        c["memberships"],
        c["retained_clusters"],
        computed,
        reused,
        devices,
        reused_output,
    )


def extract_features(
    prepared_dir=None,
    output="features.tsv",
    model_dir=None,
    *,
    device="auto",
    offline=False,
    window_batch_size=8,
    workers=1,
    resume=False,
    force=False,
):
    """Extract the selected unique sequences to a cwd-local labeled feature family.

    See SPEC-0021: atomic receipts, bounded aggregation and journaled publication.
    Raises ValueError/IntegrityError on invalid input or corrupt/incompatible work,
    SetupBusyError on contention, and RuntimeError on prediction/worker failures.
    Does not train or evaluate a classifier and never changes source selectors.
    """
    _options(device, offline, window_batch_size, workers, resume, force)
    for name, value in (
        ("output", output),
        ("prepared_dir", prepared_dir),
        ("model_dir", model_dir),
    ):
        if value is None and name != "output":
            continue
        if not isinstance(value, (str, os.PathLike)) or not isinstance(
            os.fspath(value), str
        ):
            raise ValueError(f"{name} must be a text filesystem path")
    path, selection, root = store.resolve_preparation(prepared_dir)
    bundle = verify_structural_models(model_dir)
    family = _family(output, path, root, bundle)
    try:
        with FileLock(str(family.lock), timeout=0):
            return _extract(
                path,
                selection,
                root,
                bundle,
                family,
                device,
                offline,
                window_batch_size,
                workers,
                resume,
                force,
            )
    except Timeout as exc:
        raise SetupBusyError("Feature output family is busy") from exc
    except (sqlite3.Error, KeyError, TypeError, AttributeError, IndexError) as exc:
        raise IntegrityError(f"Invalid feature input/state: {exc}") from exc


def _extract(
    path,
    selection,
    root,
    bundle,
    family,
    device,
    offline,
    batch,
    workers,
    resume,
    force,
):
    started = time.monotonic()
    run = _owned_run(family) if family.work.exists() else None
    if run is not None and (family.work / "publication.json").exists():
        if not (resume or force):
            raise ValueError("Interrupted publication; use --resume")
        reports.recover(family)
    old_report = reports.verify_family(family)
    if (run is not None or old_report is not None) and not (resume or force):
        raise ValueError("Existing feature outputs/work; use --resume or --force")
    with store.prepared_guard(root):
        if selection is not None and read_json(root / "selection.json") != selection:
            raise IntegrityError("Selected preparation changed")
        prepared = store.inspect_preparation(path, selection)
        contract = _contract(prepared, sha256(path / "manifest.json"), bundle, batch)
    cid = identity(contract)
    if force and run is not None:
        archive = reports.archive_directory(family)
        os.replace(family.work, archive / "work")
        sync_dir(family.parent)
        run = None
    if run is not None and run["contract_id"] != cid:
        raise IntegrityError(
            "Incompatible feature work contract; restore configuration or use --force"
        )
    reuse_complete = bool(
        resume
        and old_report is not None
        and (run is None or run["extraction_id"] == old_report["id"])
    )
    if reuse_complete:
        if old_report["contract_id"] != cid:
            raise IntegrityError("Incompatible complete feature contract; use --force")
        metadata = old_report["identity_inputs"]["execution_metadata"]
        if not _device_matches(
            device, "not_run" if metadata == "not_run" else metadata["device"]
        ):
            raise IntegrityError(
                "Requested device differs from completed feature report"
            )
    if reuse_complete and run is None:
        # A verified complete family does not need a disposable receipt cache.
        # Return before creating work state that cannot reproduce those receipts.
        return _result(
            family, old_report, 0, old_report["counts"]["unique_sequences"], True
        )
    if run is not None and run["status"] == "indexing":
        shutil.rmtree(family.work)
        run = None
    if run is None:
        family.work.mkdir()
        run = {
            "schema": WORK_SCHEMA,
            "output": str(family.output),
            "status": "indexing",
            "contract_inputs": contract,
            "contract_id": cid,
            "extraction_id": None,
            "execution_metadata": None,
            "execution_fingerprint": None,
            "snapshot_hashes": None,
            "counts": None,
            "sqlite_schema": 1,
            "created_at": time.time(),
        }
        write_json(family.work / "run.json", run)
        shutil.copytree(bundle.path, family.work / "models")
        verify_structural_models(family.work / "models")
        verify_structural_models(bundle.path)
        db = store.connect(family.work / "state.sqlite")
        try:
            store.create(db)
            with store.prepared_guard(root):
                if (
                    store.inspect_preparation(path, selection) != prepared
                    or sha256(path / "manifest.json")
                    != contract["prepared_manifest_sha256"]
                ):
                    raise IntegrityError("Prepared source changed before indexing")
                store.snapshot(db, path, prepared)
                if (
                    store.inspect_preparation(path, selection) != prepared
                    or sha256(path / "manifest.json")
                    != contract["prepared_manifest_sha256"]
                ) or (
                    selection is not None
                    and read_json(root / "selection.json") != selection
                ):
                    raise IntegrityError("Prepared source changed during indexing")
            run["snapshot_hashes"] = store.snapshot_hashes(db)
            run["counts"] = store.validate_snapshot(db, prepared)
            run["status"] = "ready"
            write_json(family.work / "run.json", run)
        finally:
            db.close()
    db = store.connect(safe_file(family.work, "state.sqlite"))
    try:
        store.validate_snapshot(db, prepared)
        if store.snapshot_hashes(db) != run["snapshot_hashes"]:
            raise IntegrityError("Feature snapshot checksum mismatch")
        verify_structural_models(family.work / "models")
        if reuse_complete:
            reports.verify_family(family, db=db)
            run.update(
                status="complete",
                extraction_id=old_report["id"],
                execution_metadata=old_report["identity_inputs"]["execution_metadata"],
                execution_fingerprint=old_report["execution_fingerprint"],
            )
            write_json(family.work / "run.json", run)
            return _result(
                family, old_report, 0, old_report["counts"]["unique_sequences"], True
            )
        plan = _plan(workers)
        canonical = (
            db.execute(
                "SELECT 1 FROM sequences WHERE canonical_count>0 LIMIT 1"
            ).fetchone()
            is not None
        )
        predictor = None
        metadata = "not_run"
        if canonical:
            predictor = StructuralPredictor(
                family.work / "models",
                device=device,
                offline=offline,
                window_batch_size=batch,
                cpu_threads=plan["cpu_threads"],
            )
            metadata = _metadata(predictor)
        eid, fp = reports.extraction_identity(cid, metadata)
        if run["extraction_id"] is not None and run["extraction_id"] != eid:
            raise IntegrityError(
                "Incompatible feature execution fingerprint; restore device/thread "
                "settings or use --force"
            )
        run.update(
            extraction_id=eid,
            execution_metadata=metadata,
            execution_fingerprint=fp,
            status="running",
        )

        def save_run():
            run["updated_at"] = time.time()
            write_json(family.work / "run.json", run)

        save_run()
        LOG.info(
            "Prepared %s; structural models %s; %d sequences, %d memberships; device "
            "%s; plan %s",
            prepared["id"],
            bundle.model_id,
            run["counts"]["retained_sequences"],
            run["counts"]["retained_memberships"],
            "not_run" if metadata == "not_run" else metadata["device"],
            plan,
        )
        computed, reused = _compute(db, run, predictor, plan, save_run)
        execution = {
            "configuration": {
                "device": device,
                "offline": offline,
                "window_batch_size": batch,
                "workers": workers,
                "resume": resume,
                "force": force,
            },
            "created_at": run["created_at"],
            "finished_at": time.time(),
            "plan": plan,
            "versions": contract["versions"],
            "code": contract["code"],
            "elapsed_seconds": time.monotonic() - started,
            "computed_sequences": computed,
            "reused_sequences": reused,
            "warnings": ["oversized_serial"] if plan["oversized_sequences"] else [],
        }
        if _code_hashes() != contract["code"] or _versions() != contract["versions"]:
            raise IntegrityError("Extraction code/dependencies changed during the run")
        if store.snapshot_hashes(db) != run["snapshot_hashes"]:
            raise IntegrityError("Feature snapshot changed during the run")
        _stage(db, run, family, execution)
        reports.publish_family(family)
        manifest = reports.verify_family(family, db=db)
        run["status"] = "complete"
        save_run()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return _result(family, manifest, computed, reused, False)
    finally:
        db.close()
