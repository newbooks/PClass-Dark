"""Final-only production, empirical reference scores, and independent research."""

from __future__ import annotations

import copy
import importlib.metadata
import logging
import math
import multiprocessing
import numbers
import os
import platform
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from .artifacts import (
    IntegrityError,
    archive,
    identity,
    locked,
    publication_transaction,
    publish,
    read_json,
    safe_file,
    sha256,
    staging,
    sync_dir,
    transaction_guard,
    write_json,
)
from .cluster_paths import validate_cluster_directories
from .fit_records import payload_identity, validate_result
from .library_checkpoints import Checkpoints
from .library_cleanup import DISPOSABLE
from .library_migration import import_source, prepare_sources
from .library_reports import (
    PRODUCTION,
    RESEARCH,
    load_research,
    write_production,
    write_research,
)
from .relative_scores import (
    _load_relative,
    distribution_checkpoint,
    load_selected_library,
    write_companion,
)
from .signatures import SELECTION_VERSION, background_profiles, fit_signature

LOG = logging.getLogger(__name__)


class ResearchBuildError(RuntimeError):
    """Research failed after a production pair was made available."""

    def __init__(self, production_result, research_id, cause):
        self.production_result, self.research_id = production_result, research_id
        super().__init__(
            f"Research incomplete: {cause}. Production remains usable: "
            f"{production_result.library_dir}"
        )


def _numeric():
    return {
        "engine": "cpu-logdiff-reference-fallback-exact-count-v2",
        "numpy": np.__version__,
        "scipy": importlib.metadata.version("scipy"),
        "architecture": platform.machine(),
        "inner_threads": 1,
    }


@contextmanager
def _guards(data, root, immutable):
    with ExitStack() as stack:
        for path in sorted({root} | (set() if immutable else {data})):
            stack.enter_context(transaction_guard(path))
        yield


def _legacy_base(pm, em, contract, prepared, embedding, epsilon, floor):
    return {
        "schema": "pclass-dark.library.v1",
        "algorithm": "library-" + SELECTION_VERSION,
        "cluster_ids": [],
        "epsilon": epsilon,
        "sigma_floor": floor,
        "numerical": _numeric(),
        "representation_contract_id": identity(contract),
        "representation_contract": contract,
        "prepared_manifest": sha256(prepared / "manifest.json"),
        "embedding_manifest": sha256(embedding / "manifest.json"),
        "prepared_id": pm["id"],
        "embedding_id": em["id"],
        "inventory_id": pm["inventory_id"],
        **{
            k: em["files"][name]
            for k, name in (("matrix", "embeddings.npy"), ("row_map", "rows.tsv"))
        },
        **{
            k: pm["files"][name]
            for k, name in (
                ("folds", "folds.tsv"),
                ("C", "cluster_members.json"),
                ("B", "background_members.json"),
            )
        },
        "selection_rule": "top-two-prefix-union-gt90-backward-largest-strict-v1",
        "threshold_rule": "max-training-balanced-accuracy-highest-cutoff-v1",
    }


def _fit_inputs(
    cluster,
    cids,
    summary,
    base,
    *,
    reference,
    fold=None,
    validation=None,
    partition=None,
):
    result = {
        "schema": "pclass-dark.fit-identity.v1",
        "cluster_id": cluster,
        "selection": SELECTION_VERSION,
        "threshold": base["threshold_rule"],
        "score_rule": "ordered-float64-v1",
        "epsilon": base["epsilon"],
        "sigma_floor": base["sigma_floor"],
        "numerical": _numeric(),
        "representation_contract_id": base["representation_contract_id"],
        "matrix_sha256": base["matrix"]["sha256"],
        "row_map_sha256": base["row_map"]["sha256"],
        "reference": reference,
        "order": "ascending-window-id-v1",
        "C_ids_sha256": identity(list(cids)),
        "C_n": len(cids),
        "B_ids_sha256": summary["background_ids_sha256"],
        "B_n": summary["background_count"],
    }
    if fold is not None:
        result.update(
            fold_id=fold,
            V_ids_sha256=identity(validation),
            V_n=len(validation),
            partition=partition,
        )
    return result


def _check_complete(store, inputs, context):
    def validate(result):
        validate_result(
            result,
            cluster=context["cluster_id"],
            fit=context["fit_id"],
            counts=context["counts"],
            sigma_floor=context["sigma_floor"],
            validation_ids=context["validation_ids"],
        )
        if result["signature"]["input_identity"] != inputs or result["signature"][
            "library_id"
        ] != identity(inputs):
            raise IntegrityError("Stable fit identity mismatch")

    return store.read(
        "complete", identity([store.task_id, SELECTION_VERSION]), validate
    )


def _fit(
    task, inputs, fit, cids, validation, rows, vrows, background, matrix, target_store
):
    fit_id = identity(inputs)
    root = Path(task["output_root"])
    directory = (
        safe_file(root, f".fits/{fit_id}/final")
        if fit == "final"
        else safe_file(
            root, f".research-checkpoints/{task['research_id']}/{task['encoded']}/{fit}"
        )
    )
    if task["force"]:
        archive(root, directory)
    source_meta = directory.parent / (
        "identity.json" if fit == "final" else fit + "-identity.json"
    )
    write_json(
        source_meta,
        {
            "schema": "pclass-dark.fit-inputs.v1",
            "id": fit_id,
            "identity_inputs": inputs,
        },
    )
    parents = task["parents"]
    counts = {
        "training_positive_n": len(cids),
        "validation_positive_n": len(validation),
        "background_n": len(background),
        "training_parent_n": len({parents[w] for w in cids}),
        "validation_parent_n": len({parents[w] for w in validation}),
        "background_parent_n": task["background_parents"],
    }
    context = {
        "library_id": fit_id,
        "cluster_id": task["cluster_id"],
        "fit_id": fit,
        "epsilon": task["epsilon"],
        "sigma_floor": task["sigma_floor"],
        "representation_contract_id": task["representation_contract_id"],
        "input_identity": inputs,
        "validation_ids": validation,
        "counts": counts,
    }
    store = Checkpoints(directory, identity([fit_id, task["cluster_id"], fit, inputs]))
    found = _check_complete(store, inputs, context)
    mode, provenance = "reused", None
    if found:
        result = found[0]
        provenance_path = directory / "source.json"
        if provenance_path.exists():
            provenance = read_json(provenance_path)
    else:
        matches = []
        legacy_identity = {
            **task["legacy_input"],
            "training_ids_sha256": identity(cids),
            "validation_ids_sha256": identity(validation),
            "background_ids_sha256": task["background_id"],
        }
        if not task["force"]:
            for source in task["sources"].get(fit, []):
                try:
                    old = import_source(
                        source,
                        cluster=task["cluster_id"],
                        fit=fit,
                        counts=counts,
                        validation_ids=validation,
                        input_identity=legacy_identity,
                        sigma_floor=task["sigma_floor"],
                        normalized_inputs=inputs,
                    )
                    matches.append((payload_identity(old), old, source))
                except (
                    ValueError,
                    OSError,
                    KeyError,
                    TypeError,
                    AttributeError,
                ) as exc:
                    if source["kind"] in {"published", "production"}:
                        raise IntegrityError(
                            f"Invalid published migration source: {exc}"
                        ) from exc
                    LOG.warning(
                        "Legacy %s %s not reusable: %s", task["cluster_id"], fit, exc
                    )
                    archive(root, Path(source["path"]))
        if len({m[0] for m in matches}) > 1:
            raise IntegrityError("Conflicting scientific results for a normalized fit")
        if matches:
            _, old, source = matches[0]
            result = copy.deepcopy(old)
            result["signature"]["library_id"] = fit_id
            result["signature"]["input_identity"] = inputs
            if payload_identity(result) != payload_identity(old):
                raise IntegrityError("Migration changed scientific payload")
            provenance = {
                "source_library_id": source["source_id"],
                "source_path": source["path"],
                "source_sha256": sha256(source["path"]),
                "source_kind": source["kind"],
            }
            write_json(directory / "source.json", provenance)
            store.write(
                "complete", identity([store.task_id, SELECTION_VERSION]), result
            )
            mode = "imported"
        else:
            LOG.info("Cluster %s %s: computing signature", task["cluster_id"], fit)
            bm = background_profiles(
                matrix, background, target_store, task["background_id"]
            )
            result = fit_signature(
                matrix,
                {"C": rows, "B": background, "V": vrows},
                store=store,
                context=context,
                background_moments=bm,
            )
            validate_result(
                result,
                cluster=task["cluster_id"],
                fit=fit,
                counts=counts,
                sigma_floor=task["sigma_floor"],
                validation_ids=validation,
            )
            mode = "computed"
    if mode != "computed":
        LOG.info("Cluster %s %s: %s verified signature", task["cluster_id"], fit, mode)
    return result, store, mode, provenance


def _clean_fit(store, result):
    # No quarantine/mutation while establishing the deletion gate.
    receipt = read_json(store.path / "complete.json")
    if (
        receipt["task_id"] != store.task_id
        or receipt["dependency"] != identity([store.task_id, SELECTION_VERSION])
        or receipt["payload_id"] != identity(result)
        or receipt["payload"] != result
        or receipt["arrays"]
        or receipt["files"]
    ):
        raise IntegrityError("Cannot clean an unverified complete fit")
    files = [
        safe_file(store.path, p.name)
        for p in store.path.iterdir()
        if DISPOSABLE.fullmatch(p.name)
    ]
    if any(not p.is_file() for p in files):
        raise IntegrityError("Unsafe disposable cache member")
    released = sum(p.stat().st_size for p in files)
    for file in files:
        file.unlink()
    sync_dir(store.path)
    if released:
        LOG.info(
            "Cluster %s %s: released %.3f GiB disposable cache",
            result["signature"]["cluster_id"],
            result["signature"]["fit_id"],
            released / 2**30,
        )


def target_job(task):
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    root = Path(task["output_root"])
    cluster = task["cluster_id"]
    phase_id = task.get("research_id") or task["production_id"]
    phase = "research" if task.get("research_id") else "production"
    with (
        locked(root, phase + "-target-" + identity([phase_id, cluster])),
        threadpool_limits(limits=1),
    ):
        matrix = np.load(task["matrix_path"], mmap_mode="r", allow_pickle=False)
        background = np.asarray(task["background_rows"], dtype=np.int64)
        all_rows = np.asarray(task["c_rows"], dtype=np.int64)
        checkpoint = safe_file(
            root, f".{phase}-checkpoints/{phase_id}/{task['encoded']}"
        )
        target_store = Checkpoints(checkpoint, identity([phase_id, cluster]))
        fit_paths = []
        for fold in range(5) if phase == "research" else [None]:
            fit = "final" if fold is None else f"fold_{fold}"
            mask = (
                np.asarray([f == fold for f in task["folds"]])
                if fold is not None
                else np.zeros(len(all_rows), dtype=bool)
            )
            cids = [
                w
                for w, excluded in zip(task["c_ids"], mask, strict=True)
                if not excluded
            ]
            validation = [
                w for w, excluded in zip(task["c_ids"], mask, strict=True) if excluded
            ]
            inputs = task["fit_inputs"] if fold is None else task["fold_inputs"][fold]
            fit_id = identity(inputs)
            with locked(root, "fit-" + fit_id):
                result, store, reuse, source = _fit(
                    task,
                    inputs,
                    fit,
                    cids,
                    validation,
                    all_rows[~mask],
                    all_rows[mask],
                    background,
                    matrix,
                    target_store,
                )
                record = {
                    "schema": "pclass-dark.target-result.v2",
                    "status": "complete",
                    "cluster_id": cluster,
                    "required_fits": [fit],
                    "fit_id": fit_id,
                    "fit_inputs": inputs,
                    "result": str((store.path / "complete.json").relative_to(root)),
                    "result_sha256": sha256(store.path / "complete.json"),
                    "payload_id": payload_identity(result),
                    "source": source,
                    "reuse": reuse,
                    "prepared_id": task["prepared_id"],
                    "embedding_id": task["embedding_id"],
                }
                if phase == "production":
                    LOG.info(
                        "Cluster %s ABC match evidence: "
                        "computing or reusing reference distributions",
                        cluster,
                    )
                    relative, rel_reused = distribution_checkpoint(
                        root,
                        fit_id,
                        result,
                        inputs,
                        matrix,
                        {"C": all_rows, "B": background},
                        task["contract"],
                        force=task["force"],
                    )
                    record.update(
                        relative=str(relative.relative_to(root)),
                        relative_reused=rel_reused,
                        relative_receipt_sha256=sha256(relative / "complete.json"),
                    )
                file = checkpoint / (
                    "target.json" if fold is None else fit + "-target.json"
                )
                write_json(file, record)
                _clean_fit(store, result)
                fit_paths.append(str(file))
        if phase == "research":
            file = checkpoint / "target.json"
            write_json(
                file,
                {
                    "schema": "pclass-dark.research-target.v1",
                    "status": "complete",
                    "cluster_id": cluster,
                    "required_fits": [f"fold_{f}" for f in range(5)],
                    "fits": fit_paths,
                    "final_status": task["final_status"],
                },
            )
        return str(file)


def _run(targets, make_task, workers, root, job=target_job):
    from .library import _completed_targets

    with ExitStack() as stack:
        executor = (
            stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                )
            )
            if workers > 1
            else None
        )
        completions = (
            _completed_targets(executor, targets, make_task, workers, job=job)
            if executor
            else ((i, job(make_task(c))) for i, c in enumerate(targets))
        )
        paths = {}
        for index, path in completions:
            paths[index] = path
            LOG.info("Library targets durable: %s/%s", len(paths), len(targets))
    return [paths[i] for i in range(len(targets))]


def _select(root, production, relative, prepared_id, embedding_id):
    pm, rm = (
        read_json(production / "manifest.json"),
        read_json(relative / "manifest.json"),
    )
    production_digest = sha256(production / "manifest.json")
    research_selector = root / "research-selection.json"
    if research_selector.exists():
        research = read_json(research_selector)
        if (
            research.get("production_library_id") != pm["id"]
            or research.get("production_manifest_sha256") != production_digest
        ):
            archive(root, research_selector)
    write_json(
        root / "selection.json",
        {
            "schema": "pclass-dark.library-selection.v2",
            "library_id": pm["id"],
            "library": str(production.relative_to(root)),
            "manifest_sha256": production_digest,
            "preparation_id": prepared_id,
            "embedding_id": embedding_id,
            "relative_id": rm["id"],
            "relative": str(relative.relative_to(root)),
            "relative_manifest_sha256": sha256(relative / "manifest.json"),
        },
    )


def build(
    data_dir,
    output_dir,
    *,
    workers,
    memory_mib,
    epsilon,
    sigma_floor,
    cluster_ids,
    force,
    cross_validate,
):
    from .library import (
        LibraryResult,
        _library_immutable,
        _read_inputs,
        encoded_cluster,
        load_library,
    )

    for name, value, minimum in (
        ("workers", workers, 1),
        ("memory_mib", memory_mib, 64),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if type(cross_validate) is not bool or type(force) is not bool:
        raise ValueError("cross_validate and force must be booleans")
    for name, value in (("epsilon", epsilon), ("sigma_floor", sigma_floor)):
        if (
            isinstance(value, bool)
            or not isinstance(value, numbers.Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be positive and finite")
    epsilon, sigma_floor = float(epsilon), float(sigma_floor)
    data, root = Path(data_dir).absolute(), Path(output_dir).absolute()
    if any(p.is_symlink() for path in (data, root) for p in (path, *path.parents)):
        raise ValueError("Symlink data/output roots are unsupported")
    data, root = data.resolve(), root.resolve()
    resources = Path(__file__).resolve().parent / "resources"
    if (
        data == root
        or data in root.parents
        or root in data.parents
        or resources == root
        or resources in root.parents
    ):
        raise ValueError(
            "Input/output roots must be disjoint and outside installed resources"
        )
    immutable = _library_immutable(data, False)
    root.mkdir(parents=True, exist_ok=True)
    with _guards(data, root, immutable), threadpool_limits(limits=1):
        selected, prepared, embedding, pm, em, windows, sets, contract = _read_inputs(
            data, immutable
        )
        if isinstance(cluster_ids, (str, bytes)):
            raise ValueError("cluster_ids must be a sequence, not a string")
        targets = (
            sorted(sets.cluster_ids) if cluster_ids is None else sorted(cluster_ids)
        )
        if (
            not targets
            or len(targets) != len(set(targets))
            or any(c not in sets.clusters for c in targets)
        ):
            raise ValueError("Unknown, duplicate, or empty target cluster selection")
        validate_cluster_directories(targets)
        base = _legacy_base(pm, em, contract, prepared, embedding, epsilon, sigma_floor)
        bg = read_json(prepared / "background_members.json")["backgrounds"]
        reference = {
            n: {k: r[k] for k in ("sha256", "size_bytes")}
            for n, r in pm["reference"]["files"].items()
        }
        fit_inputs = {
            c: _fit_inputs(c, sets.cluster_members(c), bg[c], base, reference=reference)
            for c in targets
        }
        fit_ids = {c: identity(fit_inputs[c]) for c in targets}
        inputs = {
            "schema": PRODUCTION,
            "contract": "final-only-v1",
            "cluster_ids": targets,
            "final_fits": sorted(fit_ids.items()),
            "representation_contract_id": identity(contract),
            "epsilon": epsilon,
            "sigma_floor": sigma_floor,
        }
        pid = identity(inputs)
        production = safe_file(root, pid)
        needed = max(
            64,
            max(
                math.ceil(
                    384
                    * (len(sets.cluster_members(c)) + bg[c]["background_count"])
                    / 2**20
                )
                + 8
                for c in targets
            ),
        )
        if memory_mib < needed:
            raise ValueError(f"At least {needed} MiB numerical workspace is required")
        active = min(
            workers, len(targets), os.cpu_count() or 1, 32, memory_mib // needed
        )
        LOG.info(
            "Building %s targets on CPU float64 with %s worker(s); final-only",
            len(targets),
            active,
        )
        reuse_superset = None
        if cross_validate and not force and (root / "selection.json").exists():
            current = load_selected_library(root)
            if current.relative is not None and all(
                c in current.library.signatures
                and current.library.signatures[c].get("fit_identity") == fit_ids[c]
                for c in targets
            ):
                reuse_superset = current
                production = current.library.source_path
        row_index = {w["window_id"]: i for i, w in enumerate(windows)}
        parents = {w["window_id"]: w["sequence_id"] for w in windows}
        folds = defaultdict(dict)
        from .setup_data import read_tsv

        for row in read_tsv(prepared / "folds.tsv"):
            folds[row["cluster_id"]][row["window_id"]] = int(row["fold_id"])
        sources = {}

        def make_task(cluster):
            cids, bids = (
                list(sets.cluster_members(cluster)),
                list(sets.background_members(cluster)),
            )
            return {
                "production_id": pid,
                "cluster_id": cluster,
                "encoded": encoded_cluster(cluster),
                "output_root": str(root),
                "matrix_path": str(embedding / "embeddings.npy"),
                "c_ids": cids,
                "c_rows": [row_index[w] for w in cids],
                "parents": {w: parents[w] for w in cids},
                "folds": [folds[cluster][w] for w in cids],
                "background_rows": [row_index[w] for w in bids],
                "background_parents": len({parents[w] for w in bids}),
                "background_id": bg[cluster]["background_ids_sha256"],
                "epsilon": epsilon,
                "sigma_floor": sigma_floor,
                "representation_contract_id": identity(contract),
                "contract": contract,
                "fit_inputs": fit_inputs[cluster],
                "prepared_id": pm["id"],
                "embedding_id": em["id"],
                "legacy_input": {
                    k: base[k]
                    for k in (
                        "prepared_id",
                        "embedding_id",
                        "inventory_id",
                        "folds",
                        "C",
                        "B",
                        "matrix",
                    )
                },
                "sources": {
                    fit: items for (c, fit), items in sources.items() if c == cluster
                },
                "force": force,
            }

        existing_pair = None
        if not force and (root / "selection.json").exists():
            selection = read_json(root / "selection.json")
            if (
                selection.get("schema") == "pclass-dark.library-selection.v2"
                and selection.get("library_id") == pid
            ):
                existing_pair = load_selected_library(root)
        production_reused = reuse_superset is not None or (
            production.exists() and not force
        )
        execution = {
            "device": "cpu",
            "dtype": "float64",
            "workers_requested": workers,
            "workers_used": active,
            "memory_mib": memory_mib,
            "per_worker_workspace_mib": memory_mib // active,
        }
        if reuse_superset or existing_pair:
            pair = reuse_superset or existing_pair
            result = LibraryResult(
                pair.library.library_id,
                pair.library.source_path,
                dict(pair.library.counts),
                True,
                pair.relative.relative_id,
                pair.relative.path,
                reuse={"production": True, "relative": True},
            )
        else:
            if production_reused:
                load_library(production)
            sources = prepare_sources(
                root,
                base,
                [targets, sorted(sets.cluster_ids)],
                targets,
                final_ids=fit_ids,
            )
            paths = _run(targets, make_task, active, root)
            with publication_transaction(root, domain="library") as core_transaction:
                if not production_reused:
                    with staging(root) as stage:
                        manifest = write_production(
                            stage, paths, inputs, contract, execution
                        )
                        load_library(stage)
                        publish(root, stage, production, force, core_transaction)
                library = load_library(production)
                with publication_transaction(
                    root, domain="relative_scores"
                ) as relative_transaction:
                    with staging(root) as stage:
                        manifest = write_companion(stage, production, paths)
                        relative = safe_file(root, "relative-scores/" + manifest["id"])
                        _load_relative(stage, library, False)
                        if relative.exists() and not force:
                            from .relative_scores import load_relative_scores

                            load_relative_scores(relative, library=library)
                        else:
                            publish(root, stage, relative, force, relative_transaction)
                    _select(root, production, relative, pm["id"], em["id"])
            reuse = {
                "production": production_reused,
                "imported": 0,
                "computed": 0,
                "reused": 0,
                "relative": True,
            }
            for path in paths:
                record = read_json(path)
                reuse[record["reuse"]] += 1
                reuse["relative"] &= record["relative_reused"]
            result = LibraryResult(
                library.library_id,
                production,
                dict(library.counts),
                production_reused and reuse["relative"],
                manifest["id"],
                relative,
                reuse=reuse,
            )
        input_snapshot = (
            sha256(prepared / "manifest.json"),
            sha256(embedding / "manifest.json"),
        )
        production_digest = sha256(result.library_dir / "manifest.json")
        LOG.info(
            "Production complete: %s targets; C/B distributions available",
            result.counts["targets"],
        )
    if not cross_validate:
        return result
    # Root guard is deliberately released: selected production remains readable.
    research_id = None
    try:
        with (
            transaction_guard(data) if not immutable else ExitStack(),
            threadpool_limits(limits=1),
        ):
            if input_snapshot != (
                sha256(prepared / "manifest.json"),
                sha256(embedding / "manifest.json"),
            ):
                raise IntegrityError("Setup generation changed before research")
            fold_inputs = {}
            for c in targets:
                cids = list(sets.cluster_members(c))
                partition = identity([[w, folds[c][w]] for w in cids])
                fold_inputs[c] = [
                    _fit_inputs(
                        c,
                        [w for w in cids if folds[c][w] != f],
                        bg[c],
                        base,
                        reference=reference,
                        fold=f,
                        validation=[w for w in cids if folds[c][w] == f],
                        partition=partition,
                    )
                    for f in range(5)
                ]
            source_production = {
                "id": result.library_id,
                "manifest_sha256": production_digest,
                "library": str(result.library_dir.relative_to(root)),
            }
            ri = {
                "schema": RESEARCH,
                "source_production": source_production,
                "cluster_ids": targets,
                "final_fits": sorted(fit_ids.items()),
                "fold_fits": [
                    [c, [identity(i) for i in fold_inputs[c]]] for c in targets
                ],
                "representation_contract_id": identity(contract),
            }
            research_id = identity(ri)
            research = safe_file(root, "research/" + research_id)
            with locked(root, "research-run-" + research_id):
                research_reused = research.exists() and not force
                if research_reused:
                    load_research(research)
                    rpaths = None
                else:
                    sources = prepare_sources(
                        root,
                        base,
                        [targets, sorted(sets.cluster_ids)],
                        targets,
                        include_folds=True,
                        final_ids=fit_ids,
                    )
                    production_library = load_library(result.library_dir)

                    def research_task(cluster):
                        task = make_task(cluster)
                        task.update(
                            research_id=research_id,
                            fold_inputs=fold_inputs[cluster],
                            final_status=production_library.signatures[cluster][
                                "status"
                            ],
                        )
                        return task

                    rpaths = _run(targets, research_task, active, root)
        with _guards(data, root, immutable):
            if (
                input_snapshot
                != (
                    sha256(prepared / "manifest.json"),
                    sha256(embedding / "manifest.json"),
                )
                or sha256(result.library_dir / "manifest.json") != production_digest
            ):
                raise IntegrityError(
                    "Input/production generation changed during research"
                )
            selected = read_json(root / "selection.json")
            if (
                selected.get("library_id") != result.library_id
                or selected.get("manifest_sha256") != production_digest
            ):
                raise IntegrityError("Selected production changed during research")
            with publication_transaction(root, domain="research") as transaction:
                if not research_reused:
                    with staging(root) as stage:
                        write_research(
                            stage,
                            rpaths,
                            ri,
                            contract,
                            execution,
                            source_production=source_production,
                        )
                        load_research(stage, immutable=True)
                        publish(root, stage, research, force, transaction)
                write_json(
                    root / "research-selection.json",
                    {
                        "schema": "pclass-dark.library-research-selection.v1",
                        "research_id": research_id,
                        "research": str(research.relative_to(root)),
                        "manifest_sha256": sha256(research / "manifest.json"),
                        "production_library_id": result.library_id,
                        "production_manifest_sha256": production_digest,
                    },
                )
        LOG.info("Research complete: %s targets, five rotations each", len(targets))
        return replace(
            result,
            research_id=research_id,
            research_dir=research,
            reused=result.reused and research_reused,
        )
    except Exception as exc:
        raise ResearchBuildError(result, research_id, exc) from exc
