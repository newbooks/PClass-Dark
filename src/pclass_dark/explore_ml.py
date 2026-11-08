"""Explicit, resumable five-fold SIAC experiments (SPEC-0022).

Imports do not fit models, initialize devices, parse CLI options or read files.
``run_exploration`` raises ValueError for configuration, IntegrityError for corrupt
or incompatible data/state, OSError for IO and RuntimeError for training failures.
"""

from __future__ import annotations

import importlib.metadata
import logging
import multiprocessing
import os
import shutil
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from filelock import FileLock

from .artifacts import (
    IntegrityError,
    file_record,
    identity,
    read_json,
    sha256,
    sync_dir,
    write_json,
)
from .ml_data import (
    ALIAS_HEADER,
    EXCLUSION_HEADER,
    LabelConflictError,
    load_cohort,
    load_groups,
    load_similarity,
)
from .ml_progress import worker_progress
from .ml_reports import HEADERS, aggregate_reports, write_tsv
from .ml_splits import make_split_plan

LOG = logging.getLogger(__name__)

PREFIX = "pclass-dark.explore-ml"
ROOT_FILES = {
    "run.json",
    "manifest.json",
    "preflight.json",
    "exclusions.tsv",
    "conflicts.tsv",
    "aliases.tsv",
    "labels.tsv",
    "folds.tsv",
    "inner_folds.tsv",
    "fold_metrics.tsv",
    "summary.json",
    "per_cluster.tsv",
    "oof_predictions.tsv",
    "oof_topk.tsv",
    "stability.tsv",
    "explanations.tsv",
}
ROOT_DIRS = {"inputs", "arrays", "folds", ".stage", "audits"}


@dataclass(frozen=True)
class ExploreMLConfig:
    features: Path = Path("features.tsv")
    similarities: Path = Path("cluster_dmatrix.tsv")
    output_dir: Path = Path("explore-ml")
    sequence_index: Path | None = None
    groups: Path | None = None
    group_provenance: Path | None = None
    similarity_provenance: Path | None = None
    split_mode: str = "accession"
    conflict_policy: str = "exclude"
    repeats: int = 1
    seed: int = 42
    alpha: float = 0.1
    neighbors: int = 10
    similarity_power: float = 2.0
    knots: int = 8
    max_epochs: int = 100
    patience: int = 10
    min_delta: float = 1e-5
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 1024
    device: str = "auto"
    workers: int = 8
    threads: int = 1
    memory_budget_gib: float = 8.0
    validate_only: bool = False
    resume: bool = False

    def __post_init__(self):
        for name in (
            "features",
            "similarities",
            "output_dir",
            "sequence_index",
            "groups",
            "group_provenance",
            "similarity_provenance",
        ):
            value = getattr(self, name)
            if value is not None:
                value = Path(value).absolute()
                if value.is_symlink() or (
                    name == "output_dir"
                    and any(parent.is_symlink() for parent in value.parents)
                ):
                    raise ValueError(f"Symlink paths are not accepted: {name}")
                object.__setattr__(self, name, value.resolve())
        ints = {
            "repeats": 1,
            "seed": 0,
            "neighbors": 1,
            "knots": 2,
            "max_epochs": 1,
            "patience": 1,
            "batch_size": 1,
            "workers": 1,
            "threads": 1,
        }
        for name, lower in ints.items():
            if type(getattr(self, name)) is not int or getattr(self, name) < lower:
                raise ValueError(f"Invalid {name}")
        if self.seed > 2**31 - 1 or self.workers > 8:
            raise ValueError("Seed/worker count outside allowed range")
        for name, strict in (
            ("similarity_power", True),
            ("learning_rate", True),
            ("weight_decay", False),
            ("min_delta", False),
            ("memory_budget_gib", True),
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or (strict and value == 0):
                raise ValueError(f"Invalid {name}")
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("SIAC alpha must be strictly between zero and one")
        if self.split_mode not in (
            "homology",
            "accession",
        ) or self.conflict_policy not in ("error", "exclude"):
            raise ValueError("Invalid split/conflict policy")
        if self.split_mode == "homology" and (
            self.groups is None or self.group_provenance is None
        ):
            raise ValueError("Homology mode requires groups and group_provenance")
        if self.split_mode == "accession" and (
            self.groups is not None or self.group_provenance is not None
        ):
            raise ValueError("Accession mode rejects homology map/provenance")
        if type(self.resume) is not bool or type(self.validate_only) is not bool:
            raise ValueError("resume/validate_only must be boolean")
        if self.resume and self.validate_only:
            raise ValueError("validate_only and resume cannot be combined")
        import re

        if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?", self.device):
            raise ValueError("Invalid device")
        if self.threads > _available_cpus():
            raise ValueError("threads exceeds available CPU count")
        for name in (
            "features",
            "similarities",
            "sequence_index",
            "groups",
            "group_provenance",
            "similarity_provenance",
        ):
            path = getattr(self, name)
            if path is not None and (
                path == self.output_dir or self.output_dir in path.parents
            ):
                raise ValueError("Input/output overlap")
        # Prevent publication inside an authoritative artifact bundle.
        for parent in (self.output_dir, *self.output_dir.parents):
            if parent.name.endswith(".features-work"):
                raise ValueError("Output inside an extraction work directory")
            marker = parent / "manifest.json"
            if marker.is_file():
                record = read_json(marker)
                schema = record.get("schema", "") if isinstance(record, dict) else ""
                if schema.startswith("pclass-dark.") and not schema.startswith(PREFIX):
                    raise ValueError("Output inside a prepared/model/extraction bundle")
        for parent in self.output_dir.parents:
            if parent.is_symlink():
                raise ValueError("Symlink output ancestors are not accepted")


@dataclass(frozen=True)
class PreflightResult:
    preflight_path: Path
    conflicts_path: Path
    status: str
    error_count: int
    example_count: int
    label_count: int
    estimated_resources: dict


@dataclass(frozen=True)
class ExploreMLResult:
    output_dir: Path
    manifest_path: Path
    summary_path: Path
    oof_path: Path
    experiment_id: str
    example_count: int
    label_count: int
    repeat_count: int
    status: str
    reused_folds: int
    devices: tuple[str, ...]


def _versions():
    result = {}
    for package in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "Install ML dependencies: pip install -e '.[ml]'"
            ) from exc
    return result


def _git_revision():
    import subprocess

    project = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "-C", str(project), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(project), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return {"revision": revision, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "dirty": None, "status": None}


def _code():
    return {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            "explore_ml.py",
            "ml_data.py",
            "ml_splits.py",
            "siac.py",
            "ml_reports.py",
            "ml_progress.py",
            "feature_reports.py",
            "feature_store.py",
            "structural_features.py",
            "artifacts.py",
        )
    }


def _settings(config):
    omit = {
        "features",
        "similarities",
        "output_dir",
        "sequence_index",
        "groups",
        "group_provenance",
        "similarity_provenance",
        "resume",
        "validate_only",
        "workers",
        "threads",
        "memory_budget_gib",
        "device",
    }
    return {k: v for k, v in asdict(config).items() if k not in omit}


def _sources(config):
    sources = {
        "features/" + config.features.name: config.features,
        "similarities/" + config.similarities.name: config.similarities,
    }
    index = config.sequence_index
    auto_index = config.features.with_name(config.features.stem + ".sequences.tsv")
    if index is None and auto_index.exists():
        index = auto_index
    if index is not None:
        sources["index/" + index.name] = index
    for name in ("groups", "group_provenance", "similarity_provenance"):
        value = getattr(config, name)
        if value is not None:
            sources[name + "/" + value.name] = value
    return sources, index


def _copy_sources(config, root):
    from .feature_reports import Family, verify_family

    sources, index = _sources(config)
    family = Family(config.features)
    marker = family.path(family.names[-1])
    qc = family.path(family.names[1])
    extraction_verified = marker.exists() or qc.exists()
    # A full family is immutable under its lock; an arbitrary standalone table is not.
    lock = FileLock(str(family.lock), timeout=0) if extraction_verified else None
    if lock is not None:
        lock.acquire()
    try:
        if extraction_verified:
            LOG.info("Verifying extraction report family: %s", config.features)
            verify_family(family)
            LOG.info("Extraction report family verified")
            for name in family.names:
                sources["features/" + name] = family.path(name)
        LOG.info("Hashing %d source input files", len(sources))
        before = {name: file_record(path) for name, path in sources.items()}
        for name, path in sources.items():
            destination = root / "inputs" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            LOG.info(
                "Snapshotting input %s (%s bytes)",
                path.name,
                f"{before[name]['size_bytes']:,}",
            )
            shutil.copyfile(path, destination)
            if (
                file_record(destination) != before[name]
                or file_record(path) != before[name]
            ):
                raise IntegrityError("Source input changed during snapshot")
        feature_path = root / "inputs" / "features" / config.features.name
        index_path = (
            root / "inputs" / "index" / index.name if index is not None else None
        )
        return sources, before, feature_path, index_path, extraction_verified
    finally:
        if lock is not None:
            lock.release()


def _inventory(root, *, exclude=()):
    inventory = {}
    for path in sorted(Path(root).rglob("*")):
        if path.is_symlink():
            raise IntegrityError("Symlink artifact rejected")
        relative = str(path.relative_to(root))
        if path.is_file() and relative not in exclude:
            record = file_record(path)
            if path.suffix == ".tsv":
                import csv

                with path.open(encoding="utf-8", newline="") as handle:
                    reader = csv.reader(handle, delimiter="\t")
                    record["columns"] = next(reader, [])
                    record["rows"] = sum(1 for _ in reader)
            inventory[relative] = record
    return inventory


def _verify_files(root, inventory):
    for name, record in inventory.items():
        path = Path(root) / name
        if Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise IntegrityError("Unsafe artifact reference")
        if not path.is_file() or file_record(path) != {
            k: record[k] for k in ("sha256", "size_bytes")
        }:
            raise IntegrityError(f"Artifact checksum mismatch: {name}")


def _owned(root):
    for path in root.rglob("*"):
        if path.is_symlink():
            raise IntegrityError("Symlink experiment artifacts rejected")
    if any(p.name not in ROOT_FILES | ROOT_DIRS for p in root.iterdir()):
        raise IntegrityError("Unrecognized experiment output")
    run = read_json(root / "run.json")
    if (
        run.get("schema") != PREFIX + ".run.v1"
        or run.get("status")
        not in ("validating", "invalid", "ready", "running", "failed", "complete")
        or (
            run.get("contract") is not None
            and identity(run["contract"]) != run.get("experiment_id")
        )
    ):
        raise IntegrityError("Unrecognized experiment owner/identity")
    return run


def _available_cpus():
    return max(1, getattr(os, "process_cpu_count", os.cpu_count)() or 1)


def _resources(config, cohort, similarity, device):
    import resource

    from .siac import sparse_target_memory

    targets = sparse_target_memory(
        len(cohort.labels), config.neighbors, config.batch_size
    )
    # B <= 18 * (knots + degree - 2), with include_bias=False.
    n, c, b = len(cohort.y), len(cohort.labels), 18 * (config.knots + 1)
    immutable = cohort.X.nbytes + cohort.y.nbytes + similarity.S.nbytes
    # Conservative spawned-worker baseline, including imported dependencies.
    baseline = max(
        256 * 1024**2, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    )
    variable = (
        targets["host_bytes"]
        + n * 18 * 16
        + n * b * 4
        + n * 32
        + c * b * 4 * 5
        + config.batch_size * (4 * c + b) * 4
    )
    per_worker = baseline + int(1.5 * variable)
    budget = int(config.memory_budget_gib * 1024**3)
    workers = (
        min(config.workers, 5 * config.repeats, _available_cpus() // config.threads)
        if device == "cpu"
        else 1
    )
    while workers > 1 and immutable + workers * per_worker > budget:
        workers -= 1
    if immutable + workers * per_worker > budget:
        raise ValueError(
            f"Estimated arrays/workers need {(immutable + per_worker) / 1024**3:.2f} "
            "GiB; increase memory budget"
        )
    fallback = None
    if device.startswith("cuda"):
        import torch

        free, total = torch.cuda.mem_get_info(device)
        gpu_need = int(
            1.5
            * (
                c * b * 4 * 5
                + config.batch_size * (4 * c + b) * 4
                + targets["device_bytes"]
            )
        )
        if gpu_need > free:
            if config.device != "auto":
                raise ValueError(
                    "Requested GPU cannot fit estimated optimizer/work buffers"
                )
            fallback, device = "GPU memory admission failed", "cpu"
        gpu = {"free_bytes": free, "total_bytes": total, "estimated_bytes": gpu_need}
    else:
        gpu = None
    return device, {
        "sparse_targets": targets,
        "immutable_bytes": immutable,
        "worker_baseline_bytes": baseline,
        "per_worker_bytes": per_worker,
        "estimated_peak_bytes": immutable + workers * per_worker,
        "budget_bytes": budget,
        "requested_workers": config.workers,
        "available_cpus": _available_cpus(),
        "effective_workers": workers,
        "threads": config.threads,
        "basis_width_upper_bound": b,
        "gpu": gpu,
        "fallback": fallback,
        "formula": (
            "immutable+J*(baseline+1.5*(N*(18*16+B*4+32)+C*B*20+M*(4*C+B)*4+targets_host))"
        ),
    }


def _preflight(config, root, run):
    from .siac import resolve_device

    LOG.info("Preflight: validating and snapshotting source tables")
    versions = _versions()
    sources, inventory, features, index, verified = _copy_sources(config, root)
    for name in ("group_provenance", "similarity_provenance"):
        original = getattr(config, name)
        if original is not None:
            sidecar = read_json(root / "inputs" / name / original.name)
            if not isinstance(sidecar, dict):
                raise IntegrityError("Provenance sidecar must be a JSON object")
            if name == "group_provenance":
                required = (
                    "method",
                    "tool_version",
                    "command",
                    "source_sequences_sha256",
                    "parameters",
                )
                if any(not sidecar.get(k) for k in required) or not isinstance(
                    sidecar["parameters"], dict
                ):
                    raise IntegrityError("Incomplete homology provenance")
    try:
        cohort = load_cohort(
            features, sequence_index=index, conflict_policy=config.conflict_policy
        )
    except LabelConflictError as exc:
        write_tsv(root / "conflicts.tsv", EXCLUSION_HEADER, exc.conflicts)
        write_tsv(root / "exclusions.tsv", EXCLUSION_HEADER, ())
        LOG.error("Conflict audit prepared; no model training has started")
        raise
    write_tsv(root / "conflicts.tsv", EXCLUSION_HEADER, cohort.conflicts)
    write_tsv(root / "exclusions.tsv", EXCLUSION_HEADER, cohort.exclusions)
    write_tsv(root / "aliases.tsv", ALIAS_HEADER, cohort.aliases)
    similarity = load_similarity(
        root / "inputs" / "similarities" / config.similarities.name,
        labels=cohort.labels,
        memory_budget_bytes=int(config.memory_budget_gib * 1024**3),
    )
    group_path = (
        root / "inputs" / "groups" / config.groups.name if config.groups else None
    )
    groups = load_groups(cohort, group_path, split_mode=config.split_mode)
    LOG.info(
        "Planning %d repeat(s) of five-fold CV; split mode=%s",
        config.repeats,
        config.split_mode,
    )
    splits = make_split_plan(
        cohort,
        groups=groups,
        split_mode=config.split_mode,
        repeats=config.repeats,
        seed=config.seed,
    )
    write_tsv(
        root / "labels.tsv",
        HEADERS["labels.tsv"],
        (
            (j, label, int(np.sum(cohort.y == j)), similarity.medoids[j])
            for j, label in enumerate(cohort.labels)
        ),
    )
    write_tsv(
        root / "folds.tsv",
        HEADERS["folds.tsv"],
        (
            (p.repeat, p.fold, cohort.example_ids[i], groups[i])
            for p in splits.folds
            for i in p.test
        ),
    )
    write_tsv(
        root / "inner_folds.tsv",
        HEADERS["inner_folds.tsv"],
        (
            (p.repeat, p.fold, cohort.example_ids[i], role)
            for p in splits.folds
            for role, indices in (
                ("train", p.inner_train),
                ("validation", p.inner_validation),
            )
            for i in indices
        ),
    )
    LOG.info("Probing requested training device: %s", config.device)
    device, fallback = resolve_device(config.device, threads=config.threads)
    device, resources = _resources(config, cohort, similarity, device)
    LOG.info(
        "Preflight ready: %s examples, %s labels; device=%s; "
        "workers=%d requested/%d admitted",
        f"{len(cohort.y):,}",
        f"{len(cohort.labels):,}",
        device,
        config.workers,
        resources["effective_workers"],
    )
    arrays = root / "arrays"
    arrays.mkdir(exist_ok=True)
    np.save(arrays / "X.npy", cohort.X, allow_pickle=False)
    np.save(arrays / "y.npy", cohort.y, allow_pickle=False)
    np.save(arrays / "S.npy", similarity.S, allow_pickle=False)
    split_inventory = {
        name: file_record(root / name) for name in ("folds.tsv", "inner_folds.tsv")
    }
    contract = {
        "schema": PREFIX + ".contract.v1",
        "settings": _settings(config),
        "inputs": inventory,
        "code": _code(),
        "versions": versions,
        "device": device,
        "deterministic_algorithms": True,
        "splits": split_inventory,
    }
    eid = identity(contract)
    population = {
        "source_rows": cohort.source_rows,
        "prefilter_counts": cohort.prefilter_counts,
        "example_count": len(cohort.y),
        "label_count": len(cohort.labels),
        "excluded_identity_count": len({r[0] for r in cohort.exclusions}),
        "excluded_accession_count": len({r[1] for r in cohort.exclusions}),
        "conflict_policy": config.conflict_policy,
        "split_mode": config.split_mode,
        "sequence_hashes_available": cohort.sequence_verified,
        "provenance_status": {
            "extraction": "verified" if verified else "unverified",
            "similarity": "declared" if config.similarity_provenance else "unverified",
            "homology": "declared" if config.group_provenance else "uncontrolled",
        },
    }
    preflight = {
        "schema": PREFIX + ".preflight.v1",
        "status": "ready",
        "experiment_id": eid,
        "population": population,
        "source_inventory": inventory,
        "resources": resources,
        "device": device,
        "device_fallback": fallback,
        "matrix_only_labels": list(similarity.extra_labels),
        "seeds": {
            "outer": list(splits.outer_seeds),
            "folds": [
                {
                    "repeat": p.repeat,
                    "fold": p.fold,
                    "inner_seed": p.inner_seed,
                    "inner_fold": p.inner_fold,
                    "select": p.selection_seed,
                    "refit": p.refit_seed,
                    "rejected_inner": list(p.rejected_inner),
                    "train_examples": len(p.train),
                    "test_examples": len(p.test),
                    "unsupported_test_examples": int(
                        np.sum(~np.isin(cohort.y[p.test], np.unique(cohort.y[p.train])))
                    ),
                    "unsupported_inner_examples": int(
                        np.sum(
                            ~np.isin(
                                cohort.y[p.inner_validation],
                                np.unique(cohort.y[p.inner_train]),
                            )
                        )
                    ),
                }
                for p in splits.folds
            ],
        },
        "warnings": ["Homology uncontrolled"]
        if config.split_mode == "accession"
        else [],
    }
    write_json(root / "preflight.json", preflight)
    return cohort, similarity, splits, contract, preflight, sources


def _task_job(root, cohort, similarity, plan, parameters, device, contract_id):
    """Commit one spawn-safe fold task, including models and controls."""
    import resource
    import sys

    from threadpoolctl import threadpool_limits

    from .ml_reports import produce_task_reports
    from .siac import (
        fit_preprocessing,
        fit_siac,
        load_model,
        make_soft_targets,
        save_model,
    )

    started = time.monotonic()
    LOG.info(
        "Starting repeat %d fold %d/5 on %s", plan.repeat + 1, plan.fold + 1, device
    )
    root = Path(root)
    stage = root / ".stage" / f"r{plan.repeat:03d}-f{plan.fold:03d}"
    final = root / "folds" / f"r{plan.repeat:03d}" / f"f{plan.fold:03d}"
    if final.exists():
        raise IntegrityError("Unexpected existing committed task")
    if stage.exists():
        owner = read_json(stage / "owner.json")
        if owner != {
            "experiment_id": contract_id,
            "repeat": plan.repeat,
            "fold": plan.fold,
            "purpose": "task",
        }:
            raise IntegrityError("Unowned task stage")
        allowed = {
            "owner.json",
            "oof_predictions.tsv",
            "oof_topk.tsv",
            "explanations.tsv",
            "fold_metrics.tsv",
            "per_cluster.tsv",
            "task.json",
            *[
                f"{c}/{name}"
                for c in ("siac", "hard")
                for name in ("model.npz", "model.json", "training.tsv")
            ],
        }
        if set(_inventory(stage)) - allowed:
            raise IntegrityError("Foreign file in interrupted task stage")
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    write_json(
        stage / "owner.json",
        {
            "experiment_id": contract_id,
            "repeat": plan.repeat,
            "fold": plan.fold,
            "purpose": "task",
        },
    )
    # Read-only shared inputs replace pickled large arrays in spawn workers.
    from dataclasses import replace

    cohort = replace(
        cohort,
        X=np.load(root / "arrays" / "X.npy", mmap_mode="r"),
        y=np.load(root / "arrays" / "y.npy", mmap_mode="r"),
    )
    similarity = replace(
        similarity, S=np.load(root / "arrays" / "S.npy", mmap_mode="r")
    )
    models = {}
    with threadpool_limits(limits=parameters["threads"]):
        for model_index, (name, alpha) in enumerate(
            (("siac", parameters["alpha"]), ("hard", 0.0))
        ):
            context = (
                f"Repeat {plan.repeat + 1}/{parameters['repeats']} | "
                f"Fold {plan.fold + 1}/5"
            )
            selection_label = (
                f"[{context} | Stage {2 * model_index + 1}/4 "
                f"{name.upper()}:epoch-selection]"
            )
            refit_label = (
                f"[{context} | Stage {2 * model_index + 2}/4 "
                f"{name.upper()}:outer-refit]"
            )
            LOG.info("%s START: fit splines and select epoch count", selection_label)
            fit_parameters = {**parameters, "alpha": alpha}
            selection_p = fit_preprocessing(
                cohort.X[plan.inner_train], knots=parameters["knots"]
            )
            targets = make_soft_targets(
                similarity,
                active_labels=np.unique(cohort.y[plan.inner_train]),
                alpha=alpha,
                neighbors=parameters["neighbors"],
                gamma=parameters["similarity_power"],
            )
            selected, logs = fit_siac(
                cohort.X[plan.inner_train],
                cohort.y[plan.inner_train],
                preprocess=selection_p,
                targets=targets,
                parameters=fit_parameters,
                seed=plan.selection_seed,
                device=device,
                progress_label=selection_label,
                validation=(
                    cohort.X[plan.inner_validation],
                    cohort.y[plan.inner_validation],
                ),
            )
            LOG.info(
                "%s DONE: selected %d epochs",
                selection_label,
                selected.selected_epochs,
            )
            LOG.info("%s START: refit on all outer training examples", refit_label)
            preprocess = fit_preprocessing(
                cohort.X[plan.train], knots=parameters["knots"]
            )
            targets = make_soft_targets(
                similarity,
                active_labels=np.unique(cohort.y[plan.train]),
                alpha=alpha,
                neighbors=parameters["neighbors"],
                gamma=parameters["similarity_power"],
            )
            model, refit_logs = fit_siac(
                cohort.X[plan.train],
                cohort.y[plan.train],
                preprocess=preprocess,
                targets=targets,
                parameters=fit_parameters,
                seed=plan.refit_seed,
                device=device,
                epochs=selected.selected_epochs,
                progress_label=refit_label,
            )
            LOG.info("%s DONE: fitted %d epochs", refit_label, model.selected_epochs)
            save_model(stage / name, model, cohort.labels)
            reloaded = load_model(stage / name)
            probe = cohort.X[plan.test[: min(16, len(plan.test))]]
            if not np.allclose(
                model.logits(probe), reloaded.logits(probe), atol=1e-6, rtol=0
            ):
                raise IntegrityError("Saved model prediction mismatch")
            models[name] = reloaded
            write_tsv(
                stage / name / "training.tsv",
                ("phase", "epoch", "n", "mean_loss", "validation_loss", "best_epoch"),
                (
                    (phase, *row)
                    for phase, values in (("selection", logs), ("refit", refit_logs))
                    for row in values
                ),
            )
        LOG.info(
            "repeat %d fold %d/5: predicting held-out examples and writing reports",
            plan.repeat + 1,
            plan.fold + 1,
        )
        produce_task_reports(
            stage, cohort, similarity, plan, models, batch_size=parameters["batch_size"]
        )
    files = _inventory(stage)
    for relative in files:
        with (stage / relative).open("rb") as handle:
            os.fsync(handle.fileno())
    write_json(
        stage / "task.json",
        {
            "schema": PREFIX + ".task.v1",
            "experiment_id": contract_id,
            "repeat": plan.repeat,
            "fold": plan.fold,
            "files": files,
            "device": device,
            "threads": parameters["threads"],
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        },
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stage, final)
    sync_dir(final.parent)
    LOG.info(
        "Committed repeat %d fold %d/5 in %.1fs",
        plan.repeat + 1,
        plan.fold + 1,
        time.monotonic() - started,
    )
    return str(final)


def _task_verify(path, eid):
    task = read_json(path / "task.json")
    if task.get("schema") != PREFIX + ".task.v1" or task.get("experiment_id") != eid:
        raise IntegrityError("Stale task contract")
    _verify_files(path, task["files"])
    if set(_inventory(path, exclude=("task.json",))) != set(task["files"]):
        raise IntegrityError("Unknown/missing committed task artifact")
    return task


def _result(root, manifest, reused):
    p = manifest["population"]
    return ExploreMLResult(
        root,
        root / "manifest.json",
        root / "summary.json",
        root / "oof_predictions.tsv",
        manifest["experiment_id"],
        p["example_count"],
        p["label_count"],
        manifest["contract"]["settings"]["repeats"],
        "complete",
        reused,
        (manifest["contract"]["device"],),
    )


def _execute(config):
    root = config.output_dir
    root.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("SIAC exploration: output=%s; requested workers=%d", root, config.workers)
    with FileLock(str(root) + ".lock", timeout=0):
        exists = root.exists() and any(root.iterdir())
        if exists and not config.resume:
            raise ValueError(
                "Existing experiment; use --resume or a new output directory"
            )
        root.mkdir(exist_ok=True)
        if exists:
            run = _owned(root)
        else:
            run = {
                "schema": PREFIX + ".run.v1",
                "status": "validating",
                "experiment_id": None,
                "created_at": time.time(),
            }
            run["git"] = _git_revision()
            write_json(root / "run.json", run)
        # Never overwrite a corrupt public completion marker with a regenerated one.
        if (root / "manifest.json").exists():
            manifest = read_json(root / "manifest.json")
            if (
                manifest.get("schema") != PREFIX + ".complete.v1"
                or manifest.get("status") != "complete"
                or identity(manifest["contract"]) != manifest["experiment_id"]
            ):
                raise IntegrityError("Corrupt complete experiment manifest")
            _verify_files(root, manifest["files"])
            if set(_inventory(root, exclude=("run.json", "manifest.json"))) != set(
                manifest["files"]
            ):
                raise IntegrityError("Unknown/missing completed experiment artifact")
            if (
                manifest["contract"]["settings"] != _settings(config)
                or manifest["contract"]["code"] != _code()
                or manifest["contract"]["versions"] != _versions()
            ):
                raise IntegrityError("Incompatible complete experiment contract")
            sources, _ = _sources(config)
            expected = manifest["source_inventory"]
            # Companion inventories are checked through their recorded original paths.
            for name, original in manifest["source_paths"].items():
                if file_record(original) != expected[name]:
                    raise IntegrityError("Changed source for complete experiment")
            for name, path in sources.items():
                if name not in expected or file_record(path) != expected[name]:
                    raise IntegrityError("Changed source configuration")
            from .siac import resolve_device

            device, _ = resolve_device(config.device, threads=config.threads)
            # Reapply preflight admission: a probed GPU may still require the
            # same automatic CPU fallback as when this experiment was fitted.
            cohort = SimpleNamespace(
                X=np.load(root / "arrays/X.npy", mmap_mode="r", allow_pickle=False),
                y=np.load(root / "arrays/y.npy", mmap_mode="r", allow_pickle=False),
                labels=range(manifest["population"]["label_count"]),
            )
            similarity = SimpleNamespace(
                S=np.load(root / "arrays/S.npy", mmap_mode="r", allow_pickle=False)
            )
            device, _ = _resources(config, cohort, similarity, device)
            if device != manifest["contract"]["device"]:
                raise IntegrityError("Changed actual device")
            LOG.info("Reusing verified complete experiment: %s", root / "summary.json")
            return _result(root, manifest, 5 * config.repeats)
        # Previously invalid, untrained audits may be replaced explicitly.
        if exists and run["status"] == "invalid":
            archive = root / "audits" / str(time.time_ns())
            archive.mkdir(parents=True)
            for name in (
                "preflight.json",
                "conflicts.tsv",
                "exclusions.tsv",
                "folds.tsv",
                "inner_folds.tsv",
            ):
                if (root / name).exists():
                    shutil.copyfile(root / name, archive / name)
        old_contract = run.get("contract")
        committed = (
            list((root / "folds").glob("r*/f*")) if (root / "folds").exists() else []
        )
        for path in committed:
            _task_verify(path, run["experiment_id"])
        if old_contract is not None and run.get("preflight_files"):
            _verify_files(root, run["preflight_files"])
        # Validate new inputs into a separate owned stage. A failed or incompatible
        # resume must never replace the input arrays of previously committed tasks.
        candidate = root / ".stage" / "preflight"
        owner = {"purpose": "preflight", "created_at": run["created_at"]}
        if candidate.exists():
            if read_json(candidate / "owner.json") != owner:
                raise IntegrityError("Unowned preflight stage")
            shutil.rmtree(candidate)
        candidate.mkdir(parents=True)
        write_json(candidate / "owner.json", owner)
        try:
            cohort, similarity, splits, contract, audit, sources = _preflight(
                config, candidate, run
            )
            if (
                old_contract is not None
                and run["status"] != "invalid"
                and contract != old_contract
            ):
                raise IntegrityError(
                    "Incompatible experiment contract; choose a new output directory"
                )
        except (ValueError, OSError, RuntimeError) as exc:
            if isinstance(exc, LabelConflictError):
                exc.args = (
                    f"{exc}. Audit: {root / 'conflicts.tsv'}. "
                    "To exclude these identities and retry this output: "
                    "--resume --conflict-policy exclude",
                )
            LOG.error("Preflight stopped: %s", exc)
            # Preserve previous ready inputs/audits on any rejected resume.
            if old_contract is None:
                for name in ("conflicts.tsv", "exclusions.tsv"):
                    if (candidate / name).exists():
                        os.replace(candidate / name, root / name)
                run.update(status="invalid", last_error=str(exc))
                write_json(root / "run.json", run)
                write_json(
                    root / "preflight.json",
                    {
                        "schema": PREFIX + ".preflight.v1",
                        "status": "invalid",
                        "errors": [str(exc)],
                    },
                )
            shutil.rmtree(candidate)
            if config.validate_only:
                return PreflightResult(
                    root / "preflight.json",
                    root / "conflicts.tsv",
                    "invalid",
                    1,
                    0,
                    0,
                    {},
                )
            raise
        eid = identity(contract)
        if old_contract is not None and run["status"] != "invalid":
            # Existing immutable arrays/splits already passed their checksums.
            # Keep them in place: replacing a directory creates a crash window
            # in which a valid committed task loses its required inputs.
            os.replace(candidate / "preflight.json", root / "preflight.json")
        else:
            for child in candidate.iterdir():
                if child.name == "owner.json":
                    continue
                target = root / child.name
                if target.is_dir():
                    shutil.rmtree(target)
                os.replace(child, target)
        shutil.rmtree(candidate)
        run.update(
            status="ready",
            experiment_id=eid,
            contract=contract,
            population=audit["population"],
            preflight_files={
                str(p.relative_to(root)): file_record(p)
                for directory in (root / "arrays", root / "inputs")
                for p in directory.rglob("*")
                if p.is_file()
            }
            | {
                name: file_record(root / name)
                for name in (
                    "folds.tsv",
                    "inner_folds.tsv",
                    "aliases.tsv",
                    "labels.tsv",
                    "conflicts.tsv",
                    "exclusions.tsv",
                )
            },
        )
        write_json(root / "run.json", run)
        if config.validate_only:
            return PreflightResult(
                root / "preflight.json",
                root / "conflicts.tsv",
                "ready",
                0,
                len(cohort.y),
                len(cohort.labels),
                audit["resources"],
            )
        run["status"] = "running"
        write_json(root / "run.json", run)
        parameters = {**_settings(config), "threads": config.threads}
        pending, tasks, reused = [], [], 0
        expected_paths = set()
        for plan in splits.folds:
            path = root / "folds" / f"r{plan.repeat:03d}" / f"f{plan.fold:03d}"
            expected_paths.add(path)
            if path.exists():
                _task_verify(path, eid)
                LOG.info(
                    "Reusing committed repeat %d fold %d/5",
                    plan.repeat + 1,
                    plan.fold + 1,
                )
                tasks.append(path)
                reused += 1
            else:
                pending.append(plan)
        if set(committed) - expected_paths:
            raise IntegrityError("Unexpected committed fold")
        # Pickle thin metadata; mmap immutable large arrays per task.
        from dataclasses import replace

        shared_cohort = replace(
            cohort, X=np.empty((0, 18)), y=np.empty(0, dtype=np.int64)
        )
        shared_similarity = replace(similarity, S=np.empty((0, 0)))
        try:
            workers = audit["resources"]["effective_workers"]
            LOG.info(
                "Training: %d pending fold tasks, %d reused, %d worker(s)",
                len(pending),
                reused,
                workers,
            )
            if workers == 1:
                for plan in pending:
                    tasks.append(
                        Path(
                            _task_job(
                                root,
                                shared_cohort,
                                shared_similarity,
                                plan,
                                parameters,
                                contract["device"],
                                eid,
                            )
                        )
                    )
            elif pending:
                with ProcessPoolExecutor(
                    max_workers=workers,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=worker_progress,
                    initargs=(
                        logging.getLogger("pclass_dark").isEnabledFor(logging.INFO),
                    ),
                ) as pool:
                    iterator = iter(pending)
                    futures = {}

                    def submit():
                        plan = next(iterator, None)
                        if plan is not None:
                            f = pool.submit(
                                _task_job,
                                root,
                                shared_cohort,
                                shared_similarity,
                                plan,
                                parameters,
                                contract["device"],
                                eid,
                            )
                            futures[f] = plan

                    for _ in range(workers):
                        submit()
                    try:
                        while futures:
                            done, _ = wait(futures, return_when=FIRST_COMPLETED)
                            for future in done:
                                del futures[future]
                                tasks.append(Path(future.result()))
                                LOG.info(
                                    "Fold tasks completed: %d/%d",
                                    len(tasks),
                                    5 * config.repeats,
                                )
                            for _ in done:
                                submit()
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        raise
            for path in tasks:
                _task_verify(path, eid)
            stage = root / ".stage" / "aggregate"
            if stage.exists():
                owner = read_json(stage / "owner.json")
                if owner != {"experiment_id": eid, "purpose": "aggregate"}:
                    raise IntegrityError("Unowned aggregate stage")
                allowed = {
                    "owner.json",
                    "summary.json",
                    "oof_predictions.tsv",
                    "oof_topk.tsv",
                    "explanations.tsv",
                    "fold_metrics.tsv",
                    "per_cluster.tsv",
                    "stability.tsv",
                }
                if set(_inventory(stage)) - allowed:
                    raise IntegrityError("Foreign file in aggregate stage")
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
            write_json(
                stage / "owner.json", {"experiment_id": eid, "purpose": "aggregate"}
            )
            LOG.info(
                "Aggregating held-out predictions, hit rates, and stability reports"
            )
            aggregate_reports(
                stage,
                tasks,
                cohort,
                similarity,
                repeats=config.repeats,
                experiment_id=eid,
                population=audit["population"],
            )
            for name, path in sources.items():
                if file_record(path) != contract["inputs"][name]:
                    raise IntegrityError("Source changed during experiment")
            for path in sorted(stage.iterdir()):
                if path.name != "owner.json":
                    os.replace(path, root / path.name)
            sync_dir(root)
            shutil.rmtree(stage)
            # Own snapshots/tasks are part of the final complete inventory.
            LOG.info("Verifying report inventory and publishing complete manifest")
            files = _inventory(root, exclude=("run.json", "manifest.json"))
            manifest = {
                "schema": PREFIX + ".complete.v1",
                "status": "complete",
                "experiment_id": eid,
                "contract": contract,
                "source_inventory": contract["inputs"],
                "source_paths": {k: str(v) for k, v in sources.items()},
                "task_inventory": {
                    str(p.relative_to(root)): file_record(p / "task.json")
                    for p in sorted(tasks)
                },
                "files": files,
                "execution": {
                    "resources": audit["resources"],
                    "device": contract["device"],
                    "completed_at": time.time(),
                    "elapsed_seconds": time.time() - run["created_at"],
                    "git": run.get("git"),
                    "cpu_count": os.cpu_count(),
                    "platform": os.uname()._asdict()
                    if hasattr(os.uname(), "_asdict")
                    else list(os.uname()),
                },
                "population": audit["population"],
                "created_at": run["created_at"],
                "completed_at": time.time(),
            }
            write_json(root / "manifest.json", manifest)
            run.update(status="complete", last_error=None)
            write_json(root / "run.json", run)
            LOG.info("Exploration complete: %s", root / "summary.json")
            return _result(root, manifest, reused)
        except BaseException as exc:
            run.update(status="failed", last_error=str(exc))
            write_json(root / "run.json", run)
            raise


def run_exploration(config):
    """Run five-fold SIAC/controls or reuse a matching completed experiment."""
    if not isinstance(config, ExploreMLConfig):
        raise ValueError("Expected ExploreMLConfig")
    return _execute(config)


def validate_exploration(config):
    """Write preflight audit without fitting; return a result for invalid data."""
    from dataclasses import replace

    return _execute(replace(config, validate_only=True, resume=False))
