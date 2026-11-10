"""Full-cohort additive fitting and portable query prediction (SPEC-0023).

Only explicit calls read artifacts, initialize optional runtimes or train models.
Full fitting uses the deduplicated CV snapshot once, never concatenates repeats.
"""

from __future__ import annotations

import importlib.metadata
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from filelock import FileLock

from .artifacts import (
    IntegrityError,
    file_record,
    identity,
    read_json,
    safe_file,
    schema_checked,
    sha256,
    sync_dir,
    write_json,
)
from .ml_data import FEATURES, identifier, readonly, rows
from .ml_reports import read_dicts, write_tsv

LOG = logging.getLogger(__name__)
SCHEMA = "pclass-dark.ml-model.v1"
BUNDLE_FILES = ("model.json", "model.npz", "feature_contract.json", "training.tsv")


@dataclass(frozen=True)
class FinalFitResult:
    model_dir: Path
    model_id: str
    configuration: str
    example_count: int
    label_count: int
    epochs: int
    device: str
    reused: bool


@dataclass(frozen=True)
class FinalMLModel:
    classifier: object
    labels: tuple[str, ...]
    feature_contract: dict
    manifest: dict


@dataclass(frozen=True)
class RankedCluster:
    rank: int
    cluster_id: str
    logit: float
    softmax_score: float


@dataclass(frozen=True)
class QueryPrediction:
    query_id: str
    model_id: str
    configuration: str
    ranked: tuple[RankedCluster, ...]
    features: dict
    margin: float
    bias_difference: float
    feature_contributions: dict
    imputed_features: tuple[str, ...]
    outside_support_features: tuple[str, ...]
    feature_device: str
    chosen_feature_contributions: dict = field(default_factory=dict)
    chosen_bias: float | None = None
    chosen_logit: float | None = None
    training_example_count: int = 0
    sequence_length: int | None = None


@schema_checked
def _verify_inventory(root, files):
    for name, expected in files.items():
        path = safe_file(root, name)
        if file_record(path) != {k: expected[k] for k in ("sha256", "size_bytes")}:
            raise IntegrityError(f"Artifact checksum mismatch: {name}")


@schema_checked
def _training_source(experiment, configuration, epochs):
    root = Path(experiment).resolve()
    if configuration not in ("hard", "siac"):
        raise ValueError("config must be hard or siac")
    if epochs is not None and (type(epochs) is not int or epochs < 1):
        raise ValueError("epochs must be a positive integer")
    manifest = read_json(root / "manifest.json")
    if (
        manifest["schema"] != "pclass-dark.explore-ml.complete.v1"
        or manifest["status"] != "complete"
        or identity(manifest["contract"]) != manifest["experiment_id"]
    ):
        raise IntegrityError("Expected a verified completed exploration")
    inventory = manifest["files"]
    required = [
        "arrays/X.npy",
        "arrays/y.npy",
        "arrays/S.npy",
        "labels.tsv",
        "folds.tsv",
        "summary.json",
    ]
    settings = dict(manifest["contract"]["settings"])
    repeat_count = settings["repeats"]
    selected = []
    for r in range(repeat_count):
        for f in range(5):
            name = f"folds/r{r:03d}/f{f:03d}/{configuration}/model.json"
            required.append(name)
    feature_manifests = [
        name
        for name in inventory
        if name.startswith("inputs/features/") and name.endswith(".manifest.json")
    ]
    if len(feature_manifests) > 1:
        raise IntegrityError("Ambiguous extraction provenance")
    required.extend(feature_manifests)
    consumed = {name: inventory[name] for name in required}
    LOG.info(
        "Verifying %d CV snapshot artifacts for full-cohort fitting", len(consumed)
    )
    _verify_inventory(root, consumed)
    for r in range(repeat_count):
        for f in range(5):
            meta = read_json(
                root / f"folds/r{r:03d}/f{f:03d}/{configuration}/model.json"
            )
            epoch = meta["selected_epochs"]
            if type(epoch) is not int or epoch < 1:
                raise IntegrityError("Invalid selected fold epoch count")
            selected.append(epoch)
    budget = (
        epochs if epochs is not None else math.floor(float(np.median(selected)) + 0.5)
    )
    X = np.load(root / "arrays/X.npy", mmap_mode="r", allow_pickle=False)
    y = np.load(root / "arrays/y.npy", mmap_mode="r", allow_pickle=False)
    S = np.load(root / "arrays/S.npy", mmap_mode="r", allow_pickle=False)
    label_rows = list(read_dicts(root / "labels.tsv"))
    labels = tuple(row["cluster_id"] for row in label_rows)
    n, c = len(y), len(labels)
    if (
        X.shape != (n, 18)
        or X.dtype != np.float64
        or y.shape != (n,)
        or y.dtype != np.int64
        or S.shape != (c, c)
        or S.dtype != np.float64
        or n < 1
        or c < 2
        or labels != tuple(sorted(set(labels)))
        or np.isinf(X).any()
        or np.isnan(X).all(axis=0).any()
        or not np.isfinite(S).all()
        or (S < 0).any()
        or (S > 1).any()
        or not np.array_equal(S, S.T)
        or not np.all(np.diag(S) == 1)
        or y.min() < 0
        or y.max() >= c
    ):
        raise IntegrityError("Invalid full-cohort array/label contract")
    if [int(row["label_index"]) for row in label_rows] != list(range(c)):
        raise IntegrityError("Noncanonical label indices")
    if not np.array_equal(
        np.bincount(y, minlength=c), [int(row["example_count"]) for row in label_rows]
    ):
        raise IntegrityError("Label supports disagree with training snapshot")
    if (
        manifest["population"]["example_count"] != n
        or manifest["population"]["label_count"] != c
    ):
        raise IntegrityError("Population and snapshot dimensions disagree")
    assignments = {}
    for row in read_dicts(root / "folds.tsv"):
        r, f = int(row["repeat_id"]), int(row["outer_fold_id"])
        if not 0 <= r < repeat_count or not 0 <= f < 5:
            raise IntegrityError("Invalid fold index")
        bucket = assignments.setdefault(r, {})
        if row["example_id"] in bucket:
            raise IntegrityError("Repeated example within CV repeat")
        bucket[row["example_id"]] = f
    if set(assignments) != set(range(repeat_count)):
        raise IntegrityError("Missing CV repeat")
    first = set(assignments[0])
    if len(first) != n or any(
        set(a) != first or set(a.values()) != set(range(5))
        for a in assignments.values()
    ):
        raise IntegrityError("CV examples are not a single common full cohort")
    summary = read_json(root / "summary.json")
    if (
        summary["status"] != "complete"
        or summary["experiment_id"] != manifest["experiment_id"]
    ):
        raise IntegrityError("Invalid CV summary identity")
    evidence = [r for r in summary["repeat_metrics"] if r["config_id"] == configuration]
    extraction = None
    precision = "identity"
    if feature_manifests:
        fmeta = read_json(root / feature_manifests[0])
        from .feature_reports import manifest_identity

        manifest_identity(fmeta)
        extraction = fmeta["identity_inputs"]["contract_inputs"]
        precision = fmeta.get("serialization", {}).get(
            "precision", extraction["policies"]["precision"]
        )
        if extraction["features"][:-1] != list(FEATURES) or precision not in (
            ".3f",
            ".17g",
        ):
            raise IntegrityError("Unsupported extraction schema/export precision")
    feature_contract = {
        "schema": "pclass-dark.ml-feature-contract.v1",
        "feature_names": list(FEATURES),
        "precision": precision,
        "extraction": extraction,
    }
    return (
        root,
        manifest,
        consumed,
        X,
        y,
        S,
        labels,
        settings,
        budget,
        selected,
        evidence,
        feature_contract,
    )


def _fit_code():
    return {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            "ml_model.py",
            "siac.py",
            "ml_data.py",
            "ml_splits.py",
            "ml_reports.py",
            "ml_progress.py",
            "artifacts.py",
        )
    }


def _versions():
    try:
        return {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "torch", "threadpoolctl")
        }
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Install ML dependencies: pip install -e '.[ml]'") from exc


def _admit(X, y, S, settings, device, requested, budget_gib):
    import resource

    from .siac import sparse_target_memory

    n, c, b = len(y), len(S), 18 * (settings["knots"] + 1)
    targets = sparse_target_memory(c, settings["neighbors"], settings["batch_size"])
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    variable = (
        targets["host_bytes"]
        + n * (18 * 16 + b * 4 + 32)
        + c * b * 20
        + settings["batch_size"] * (4 * c + b) * 4
    )
    peak = base + X.nbytes + y.nbytes + S.nbytes + int(1.5 * variable)
    if peak > budget_gib * 1024**3:
        raise ValueError(
            f"Estimated full fit needs {peak / 1024**3:.2f} GiB; increase memory budget"
        )
    gpu = None
    if device.startswith("cuda"):
        import torch

        free, total = torch.cuda.mem_get_info(device)
        need = int(
            1.5
            * (
                c * b * 20
                + settings["batch_size"] * (4 * c + b) * 4
                + targets["device_bytes"]
            )
        )
        gpu = {"free_bytes": free, "total_bytes": total, "estimated_bytes": need}
        if need > free:
            if requested != "auto":
                raise ValueError("Requested GPU lacks estimated fitting memory")
            LOG.warning("GPU memory admission failed; using CPU before fitting")
            device = "cpu"
    return device, {
        "sparse_targets": targets,
        "estimated_peak_bytes": peak,
        "budget_bytes": int(budget_gib * 1024**3),
        "gpu": gpu,
    }


@schema_checked
def load_final_model(path):
    """Verify a standalone complete bundle; source CV files are not required."""
    from .siac import load_model

    root = Path(path).resolve()
    m = read_json(root / "manifest.json")
    if (
        m["schema"] != SCHEMA
        or m["status"] != "complete"
        or identity(m["contract"]) != m["model_id"]
    ):
        raise IntegrityError("Incomplete/invalid final model manifest")
    if set(m["files"]) != set(BUNDLE_FILES):
        raise IntegrityError("Unexpected final model inventory")
    _verify_inventory(root, m["files"])
    allowed = set(BUNDLE_FILES) | {"manifest.json", "run.json"}
    if any(p.is_symlink() or p.name not in allowed for p in root.iterdir()):
        raise IntegrityError("Foreign/symlink final model artifact")
    model = load_model(root)
    meta = read_json(root / "model.json")
    feature_contract = read_json(root / "feature_contract.json")
    if (
        feature_contract != m["contract"]["feature_contract"]
        or feature_contract["schema"] != "pclass-dark.ml-feature-contract.v1"
        or feature_contract["feature_names"] != list(FEATURES)
        or feature_contract["precision"] not in ("identity", ".3f", ".17g")
        or model.training_example_count != m["contract"]["example_count"]
        or model.selected_epochs != m["contract"]["epochs"]
        or len(model.bias) != m["contract"]["label_count"]
        or meta["active_global_indices"] != list(range(len(meta["global_labels"])))
        or model.parameters["alpha"] != m["contract"]["parameters"]["alpha"]
    ):
        raise IntegrityError("Final model metadata/contract mismatch")
    return FinalMLModel(model, tuple(meta["global_labels"]), feature_contract, m)


def _fit_result(root, manifest, reused):
    c = manifest["contract"]
    return FinalFitResult(
        root,
        manifest["model_id"],
        c["configuration"],
        c["example_count"],
        c["label_count"],
        c["epochs"],
        c["device"],
        reused,
    )


def _fit_model(
    experiment="explore-ml",
    output_dir="ml-model",
    *,
    configuration="hard",
    epochs=None,
    device="auto",
    threads=8,
    memory_budget_gib=8.0,
    resume=False,
    _reference=None,
):
    """Refit all verified reference examples once; publish a portable complete model.

    ValueError: options. IntegrityError: incompatible/corrupt source or output.
    OSError: IO. RuntimeError: fitting/device failure. GPU/CPU runtime is lazy.
    """
    from threadpoolctl import threadpool_limits

    from .ml_splits import derived_seed
    from .siac import (
        fit_preprocessing,
        fit_siac,
        load_model,
        make_soft_targets,
        resolve_device,
        save_model,
    )

    available = max(1, getattr(os, "process_cpu_count", os.cpu_count)() or 1)
    if (
        type(threads) is not int
        or threads < 1
        or not np.isfinite(memory_budget_gib)
        or memory_budget_gib <= 0
    ):
        raise ValueError("Invalid thread count/memory budget")
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    import re

    if not isinstance(device, str) or not re.fullmatch(
        r"auto|cpu|cuda(?::\d+)?", device
    ):
        raise ValueError("Invalid training device")
    threads = min(threads, available)
    raw = Path(output_dir).absolute()
    source = Path(experiment).resolve()
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ValueError("Symlink model output rejected")
    root = raw.resolve()
    if root == source or root in source.parents or source in root.parents:
        raise ValueError("Final model output must be outside source experiment")
    root.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root) + ".lock", timeout=0):
        existing = root.exists() and any(root.iterdir())
        if existing and not resume:
            raise ValueError("Existing model output; use --resume or a new directory")
        root.mkdir(exist_ok=True)
        run = None
        if existing:
            run = read_json(root / "run.json")
            if run.get("schema") != "pclass-dark.ml-model.run.v1":
                raise IntegrityError("Unowned final model directory")
            allowed = set(BUNDLE_FILES) | {"manifest.json", "run.json", ".fit-stage"}
            if any(p.name not in allowed or p.is_symlink() for p in root.iterdir()):
                raise IntegrityError("Foreign final model output")
        LOG.info(
            "Final fit: configuration=%s; source=%s",
            configuration,
            source if _reference is None else "prepared reference arrays",
        )
        (
            source,
            m,
            consumed,
            X,
            y,
            S,
            labels,
            settings,
            epoch_budget,
            selected,
            evidence,
            fc,
        ) = (
            _training_source(source, configuration, epochs)
            if _reference is None
            else _reference
        )
        actual, fallback = resolve_device(device, threads=threads)
        actual, resources = _admit(X, y, S, settings, actual, device, memory_budget_gib)
        parameters = {
            **settings,
            "threads": threads,
            "alpha": settings["alpha"] if configuration == "siac" else 0.0,
        }
        seed = derived_seed(settings["seed"], 0, 0, "final-" + configuration)
        contract = {
            "schema": "pclass-dark.ml-model.contract.v1",
            "source_experiment_id": m["experiment_id"],
            "source_files": consumed,
            "configuration": configuration,
            "epochs": epoch_budget,
            "epoch_rule": "explicit"
            if epochs is not None
            else "rounded_median_inner_selected_epochs",
            "source_selected_epochs": selected,
            "parameters": parameters,
            "feature_contract": fc,
            "example_count": len(y),
            "label_count": len(labels),
            "seed": seed,
            "device": actual,
            "versions": _versions(),
            "code": _fit_code(),
        }
        if _reference is not None:
            contract["source_kind"] = "prepared_reference"
            contract["source_experiment_id"] = None
            contract["source_reference_id"] = m["experiment_id"]
        model_id = identity(contract)
        if (root / "manifest.json").exists():
            loaded = load_final_model(root)
            if loaded.manifest["contract"] != contract:
                raise IntegrityError("Incompatible completed final model")
            LOG.info("Reusing complete model %s", model_id)
            return _fit_result(root, loaded.manifest, True)
        if run is not None and run.get("contract") != contract:
            raise IntegrityError(
                "Incompatible incomplete final model; use a new directory"
            )
        started = time.time()
        run = {
            "schema": "pclass-dark.ml-model.run.v1",
            "status": "fitting",
            "contract": contract,
            "model_id": model_id,
            "started_at": started,
        }
        write_json(root / "run.json", run)
        stage = root / ".fit-stage"
        if stage.exists():
            if stage.is_symlink() or read_json(stage / "owner.json") != {
                "model_id": model_id
            }:
                raise IntegrityError("Unowned final-fit stage")
            if any(
                p.is_symlink() or p.name not in set(BUNDLE_FILES) | {"owner.json"}
                for p in stage.iterdir()
            ):
                raise IntegrityError("Foreign file in final-fit stage")
            shutil.rmtree(stage)
        stage.mkdir()
        write_json(stage / "owner.json", {"model_id": model_id})
        try:
            LOG.info(
                "Fitting ALL %s unique sequences / %s labels for %d epochs on %s",
                f"{len(y):,}",
                f"{len(labels):,}",
                epoch_budget,
                actual,
            )
            with threadpool_limits(limits=threads):
                preprocess = fit_preprocessing(X, knots=settings["knots"])
                targets = make_soft_targets(
                    S,
                    active_labels=np.arange(len(labels)),
                    alpha=parameters["alpha"],
                    neighbors=settings["neighbors"],
                    gamma=settings["similarity_power"],
                )
                model, logs = fit_siac(
                    X,
                    y,
                    preprocess=preprocess,
                    targets=targets,
                    parameters=parameters,
                    seed=seed,
                    device=actual,
                    epochs=epoch_budget,
                    progress_label=(
                        f"[Final {configuration.upper()} | all {len(y):,} sequences]"
                    ),
                )
            save_model(stage, model, labels)
            write_json(stage / "feature_contract.json", fc)
            write_tsv(
                stage / "training.tsv",
                ("epoch", "n", "mean_loss", "validation_loss", "best_epoch"),
                logs,
            )
            reloaded = load_model(stage)
            probe = np.asarray(X[: min(32, len(X))])
            if not np.allclose(
                model.logits(probe), reloaded.logits(probe), atol=1e-6, rtol=0
            ):
                raise IntegrityError("Full-model reload prediction mismatch")
            _verify_inventory(source, consumed)
            files = {name: file_record(stage / name) for name in BUNDLE_FILES}
            manifest = {
                "schema": SCHEMA,
                "status": "complete",
                "model_id": model_id,
                "contract": contract,
                "files": files,
                "source_experiment_id": m["experiment_id"],
                "source_cv_evidence": evidence,
                "source_path": str(source),
                "evaluation_note": (
                    "No new held-out estimate. "
                    + (
                        "Source CV evidence only."
                        if _reference is None
                        else "Full prepared-reference fit; no exploration required."
                    )
                ),
                "execution": {
                    "device": actual,
                    "fallback": fallback,
                    "threads": threads,
                    "resources": resources,
                    "started_at": started,
                    "completed_at": time.time(),
                },
            }
            if _reference is not None:
                manifest.update(
                    source_kind="prepared_reference",
                    source_experiment_id=None,
                    source_cv_evidence=None,
                    source_path=None,
                    source_reference=dict(
                        id=m["experiment_id"], provenance=m["provenance"]
                    ),
                )
            for name in BUNDLE_FILES:
                with (stage / name).open("rb") as stream:
                    os.fsync(stream.fileno())
                os.replace(stage / name, root / name)
            sync_dir(root)
            shutil.rmtree(stage)
            sync_dir(root)
            write_json(root / "manifest.json", manifest)
            run["status"] = "complete"
            write_json(root / "run.json", run)
            loaded = load_final_model(root)
            LOG.info("Full-data model ready: %s (%s)", root, model_id)
            return _fit_result(root, loaded.manifest, False)
        except BaseException as exc:
            run.update(status="failed", last_error=str(exc))
            write_json(root / "run.json", run)
            raise


def fit_final_model(experiment="explore-ml", output_dir="ml-model", **options):
    """Refit a verified completed exploration, preserving its historical API."""
    return _fit_model(experiment, output_dir, **options)


def fit_reference_model(
    X,
    y,
    labels,
    similarity,
    feature_contract,
    recipe,
    output_dir,
    *,
    provenance,
    device="auto",
    threads=8,
    memory_budget_gib=8.0,
    resume=False,
):
    """Fit explicit reference arrays without a CV artifact. Return FinalFitResult.

    Raises ValueError/IntegrityError for invalid inputs/state, OSError for IO,
    RuntimeError for fitting failures. Does not mutate input arrays.
    """
    from .additive_reference import validate_recipe

    recipe = validate_recipe(recipe)
    X, y = np.asarray(X), np.asarray(y)
    S = np.asarray(similarity.S if hasattr(similarity, "S") else similarity)
    labels = tuple(labels)
    if (
        X.dtype != np.float64
        or y.dtype != np.int64
        or S.dtype != np.float64
        or X.shape != (len(y), 18)
        or y.ndim != 1
        or not len(y)
        or len(labels) < 2
        or labels != tuple(sorted(set(labels)))
        or S.shape != (len(labels), len(labels))
        or np.isinf(X).any()
        or np.isnan(X).all(axis=0).any()
        or y.min() < 0
        or y.max() >= len(labels)
        or set(y) != set(range(len(labels)))
        or not np.isfinite(S).all()
        or (S < 0).any()
        or (S > 1).any()
        or not np.array_equal(S, S.T)
        or not np.all(np.diag(S) == 1)
    ):
        raise IntegrityError("Invalid explicit reference arrays/labels")
    if (
        feature_contract.get("schema") != "pclass-dark.ml-feature-contract.v1"
        or feature_contract.get("feature_names") != list(FEATURES)
        or feature_contract.get("precision") != ".3f"
    ):
        raise IntegrityError("Invalid reference feature contract")
    import hashlib

    def digest(array):
        return hashlib.sha256(
            memoryview(np.ascontiguousarray(array)).cast("B")
        ).hexdigest()

    source_id = identity(
        {
            "arrays": [digest(a) for a in (X, y, S)],
            "labels": labels,
            "provenance": provenance,
        }
    )
    source = Path(output_dir).resolve().parent / ".reference-input"
    settings = dict(recipe, max_epochs=recipe["epochs"])
    reference = (
        source,
        {"experiment_id": source_id, "provenance": provenance},
        {},
        X,
        y,
        S,
        labels,
        settings,
        recipe["epochs"],
        [],
        {},
        feature_contract,
    )
    return _fit_model(
        source,
        output_dir,
        configuration=recipe["configuration"],
        epochs=recipe["epochs"],
        device=device,
        threads=threads,
        memory_budget_gib=memory_budget_gib,
        resume=resume,
        _reference=reference,
    )


def _rounded_features(model, X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 18 or not len(X) or np.isinf(X).any():
        raise ValueError("Query features must be [N,18], finite or NaN")
    precision = model.feature_contract["precision"]
    if precision == "identity":
        return X.copy()
    return np.array(
        [
            [float(format(v, precision)) if np.isfinite(v) else np.nan for v in row]
            for row in X
        ]
    )


def predict_features(
    model, X, query_ids, *, batch_size=1024, feature_device="precomputed"
):
    """Yield CPU rankings and additive explanations without initializing Torch."""
    from .ml_interpretation import chosen_support
    from .siac import predict_siac

    X = _rounded_features(model, X)
    ids = tuple(query_ids)
    if len(ids) != len(X) or len(set(ids)) != len(ids):
        raise ValueError("Query IDs must be unique and aligned")
    for q in ids:
        identifier(q)
    for batch in predict_siac(model.classifier, X, batch_size=batch_size):
        own_terms, own_bias = chosen_support(
            model.classifier,
            X[batch.start : batch.start + len(batch.indices)],
            batch.indices[:, 0],
        )
        if not np.allclose(
            own_terms.sum(axis=1) + own_bias, batch.logits[:, 0], atol=1e-4, rtol=1e-4
        ):
            raise IntegrityError(
                "Chosen-cluster explanation does not reconstruct logit"
            )
        for j, local in enumerate(batch.indices):
            ix = batch.start + j
            values = X[ix]
            active = model.classifier.active_labels[local]
            ranked = tuple(
                RankedCluster(
                    k + 1,
                    model.labels[c],
                    float(batch.logits[j, k]),
                    float(batch.probabilities[j, k]),
                )
                for k, c in enumerate(active)
            )
            margin = float(batch.logits[j, 0] - batch.logits[j, 1])
            if not np.isclose(
                margin,
                batch.contributions[j].sum() + batch.bias_difference[j],
                atol=1e-4,
                rtol=1e-4,
            ):
                raise IntegrityError(
                    "Query explanation does not reconstruct score margin"
                )
            p = model.classifier.preprocess
            yield QueryPrediction(
                ids[ix],
                model.manifest["model_id"],
                model.manifest["contract"]["configuration"],
                ranked,
                {
                    k: float(v) if np.isfinite(v) else None
                    for k, v in zip(FEATURES, values, strict=True)
                },
                margin,
                float(batch.bias_difference[j]),
                dict(zip(FEATURES, batch.contributions[j].tolist(), strict=True)),
                tuple(FEATURES[k] for k in np.flatnonzero(np.isnan(values))),
                tuple(
                    FEATURES[k]
                    for k in np.flatnonzero(
                        (values < p.training_min) | (values > p.training_max)
                    )
                ),
                feature_device,
                chosen_feature_contributions=dict(
                    zip(FEATURES, own_terms[j].tolist(), strict=True)
                ),
                chosen_bias=float(own_bias[j]),
                chosen_logit=float(batch.logits[j, 0]),
                training_example_count=model.classifier.training_example_count,
            )


def read_query_features(path):
    """Read exact query_id + 18-feature TSV; reject targets and reordered columns."""
    before = file_record(path)
    ids, values = [], []
    for row in rows(path, ("query_id", *FEATURES)):
        identifier(row[0])
        ids.append(row[0])
        vector = []
        for text in row[1:]:
            if text == "NA":
                vector.append(np.nan)
            else:
                try:
                    value = float(text)
                except ValueError as exc:
                    raise IntegrityError("Invalid query feature number") from exc
                if not np.isfinite(value):
                    raise IntegrityError("Use literal NA for undefined query features")
                vector.append(value)
        values.append(vector)
    if not ids or len(set(ids)) != len(ids):
        raise IntegrityError("Empty/duplicate query identifiers")
    X = np.asarray(values, dtype=np.float64)
    if file_record(path) != before:
        raise IntegrityError("Query feature input changed during reading")
    return tuple(ids), readonly(X)


def _raw_contract(model):
    contract = model.feature_contract.get("extraction")
    if contract is None:
        raise IntegrityError(
            "Raw-sequence prediction requires saved extraction provenance; "
            "supply precomputed features"
        )
    for name in ("structural_features", "structure", "structural_models", "embeddings"):
        recorded = contract["code"].get(name)
        # This revision only adds a borrowed-backend constructor. Standalone
        # extraction arithmetic and the pinned feature recipe are unchanged.
        compatible = name == "structure" and recorded == (
            "5414b040d300fb09598b5b7566fcdab36ecbe4fd8f43ef62501cf8a364fe336d"
        )
        if (
            recorded != sha256(Path(__file__).with_name(name + ".py"))
            and not compatible
        ):
            raise IntegrityError(
                f"Query extraction implementation differs from training: {name}"
            )
    from .setup_data import MODEL, REVISION
    from .structural_models import BUNDLE_ID

    if (
        contract["model_id"] != BUNDLE_ID
        or contract["esm"]["model"] != MODEL
        or contract["esm"]["revision"] != REVISION
    ):
        raise IntegrityError("Query feature models differ from reference models")
    return contract


def predict_sequences(model, queries, *, device="auto", offline=True, threads=8):
    """Stream full-sequence features using the frozen reference extraction recipe."""
    _raw_contract(model)
    from .structural_features import summarize_structural_features
    from .structure import StructuralPredictor

    predictor = None
    seen = set()
    for query_id, sequence in queries:
        identifier(query_id)
        if query_id in seen:
            raise ValueError("Duplicate sequence query ID")
        seen.add(query_id)
        if not isinstance(sequence, str) or not 1 <= len(sequence) <= 100000:
            raise ValueError("Query sequence length must be 1..100000")
        if predictor is None:
            predictor = StructuralPredictor(
                device=device, offline=offline, cpu_threads=min(8, threads)
            )
        LOG.info(
            "Extracting reference-compatible features for %s (%s residues)",
            query_id,
            f"{len(sequence):,}",
        )
        prediction = predictor.predict(sequence)
        summary = summarize_structural_features(prediction.sequence, prediction)
        X = np.array([[np.nan if x is None else x for x in summary.values[:-1]]])
        for result in predict_features(
            model, X, [query_id], feature_device=prediction.device
        ):
            yield replace(result, sequence_length=len(prediction.sequence))


def score_candidates(model, X, candidate_labels, *, batch_size=32):
    """Yield global and requested scores without retaining a full-input matrix.

    Unknown candidates are explicit unscored rows. Ties use lexical cluster IDs.
    """
    from scipy.special import logsumexp

    X = _rounded_features(model, X)
    if len(candidate_labels) != len(X) or type(batch_size) is not int or batch_size < 1:
        raise ValueError("Invalid candidate/query alignment or batch size")
    index = {label: i for i, label in enumerate(model.labels)}
    for start in range(0, len(X), batch_size):
        logits = model.classifier.logits(X[start : start + batch_size])
        for j, values in enumerate(logits):
            candidates = tuple(candidate_labels[start + j])
            if len(candidates) != len(set(candidates)):
                raise ValueError("Duplicate requested cluster ID")
            order = np.lexsort((np.asarray(model.labels), -values))
            ranks = np.empty(len(order), dtype=np.int64)
            ranks[order] = np.arange(1, len(order) + 1)
            probs = np.exp(
                values.astype(np.float64) - logsumexp(values.astype(np.float64))
            )

            def row(label, values=values, probs=probs, ranks=ranks):
                k = index.get(label)
                return dict(
                    cluster_id=label,
                    logit=float(values[k]) if k is not None else None,
                    softmax_score=float(probs[k]) if k is not None else None,
                    global_rank=int(ranks[k]) if k is not None else None,
                    score_available=k is not None,
                    reason=None if k is not None else "not_in_classifier_vocabulary",
                )

            yield dict(
                global_top5=[
                    dict(row(model.labels[k]), rank=r + 1)
                    for r, k in enumerate(order[:5])
                ],
                candidates=[row(label) for label in candidates],
            )
