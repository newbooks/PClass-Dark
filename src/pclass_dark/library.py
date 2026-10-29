"""Build, validate, publish, and load portable fold-wise PLM signature libraries."""

from __future__ import annotations

import csv
import gzip
import importlib.metadata
import io
import logging
import math
import multiprocessing
import os
import platform
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np
from threadpoolctl import threadpool_limits

from . import __version__
from .artifacts import (
    IntegrityError,
    archive,
    canonical,
    file_record,
    identity,
    locked,
    publication_transaction,
    publish,
    read_json,
    safe_file,
    schema_checked,
    sha256,
    staging,
    transaction_guard,
    verify_bundle,
    write_json,
)
from .cluster_paths import (
    LAYOUT,
    cluster_directory_name,
    stored_cluster_name,
    validate_cluster_directories,
)
from .cluster_sets import validate_sets
from .embeddings import verify_embeddings
from .library_checkpoints import Checkpoints
from .library_cleanup import cleanup_target_cache
from .library_runtime import (
    freeze,
    representation_contract,
    score_signature,
    validate_signature,
)
from .setup_data import read_tsv, verify_prepared
from .signature_metrics import METRICS
from .signatures import SELECTION_VERSION, background_profiles, fit_signature

LOG = logging.getLogger(__name__)
SCHEMA = "pclass-dark.library.v1"
STATS = ("mu_C", "sigma_C", "R", "D", "training_auroc", "validation_auroc", "rank")
PROFILE = (
    "dimension positive_n background_n mu_C sigma_C mu_B sigma_B "
    "effective_sigma R D training_auroc training_auroc_numerator "
    "training_auroc_denominator validation_auroc "
    "validation_auroc_numerator validation_auroc_denominator rank "
    "core candidate selected status"
).split()
FINAL_PROFILE = [x for x in PROFILE if not x.startswith("validation_")]
HEADERS = {
    "dimension_folds.tsv.gz": ["cluster_id", "fold_id", *PROFILE],
    "dimension_final.tsv.gz": ["cluster_id", *FINAL_PROFILE],
    "dimension_summary.tsv.gz": (
        "cluster_id dimension statistic mean sample_sd valid_folds "
        "selection_count selection_frequency core_count"
    ).split(),
    "cluster_metrics.tsv": (
        "cluster_id fit_id population metric value positive_n negative_n status"
    ).split(),
    "cluster_summary.tsv": (
        "cluster_id population metric mean sample_sd valid_folds final_status"
    ).split(),
    "out_of_fold_positive_scores.tsv.gz": (
        "cluster_id window_id fold_id signature_digest log_score score accepted status"
    ).split(),
}


@dataclass(frozen=True)
class LibraryResult:
    library_id: str
    library_dir: Path
    counts: dict
    reused: bool
    relative_id: str | None = None
    relative_dir: Path | None = None
    research_id: str | None = None
    research_dir: Path | None = None
    reuse: dict | None = None


@dataclass(frozen=True)
class Library:
    library_id: str
    signatures: MappingProxyType
    embedding_contract: MappingProxyType
    counts: MappingProxyType
    source_root: Path | None = None
    manifest_sha256: str | None = None
    source_path: Path | None = None
    immutable: bool = False
    target_result: MappingProxyType | None = field(
        default=None, repr=False, compare=False
    )

    def score(self, cluster_id, vectors, *, embedding_contract):
        return score_signature(
            self.signatures[cluster_id], vectors, embedding_contract=embedding_contract
        )


def encoded_cluster(cluster):
    result = "c_" + cluster.encode("utf-8").hex()
    if len(result) > 200:
        raise ValueError(f"Encoded cluster ID exceeds 200 bytes: {cluster}")
    return result


def _stat(values):
    values = [x for x in values if x is not None and math.isfinite(x)]
    return {
        "mean": float(np.mean(values)) if values else None,
        "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "valid_folds": len(values),
    }


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Nonfinite report value")
        return format(value, ".17g")
    return str(value)


@contextmanager
def _table(path, header):
    with ExitStack() as stack:
        raw = stack.enter_context(Path(path).open("wb"))
        stream = (
            stack.enter_context(
                gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0)
            )
            if str(path).endswith(".gz")
            else raw
        )
        text = stack.enter_context(
            io.TextIOWrapper(stream, encoding="utf-8", newline="")
        )
        writer = csv.writer(text, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        yield lambda row: writer.writerow([_cell(row.get(k)) for k in header])


def _gzip_json(path, value):
    with Path(path).open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle:
            handle.write(canonical(value) + b"\n")


def _target_job(task):
    # Only the coordinator owns input-root locks. Workers receive verified
    # immutable row descriptors and cannot publish a library selection.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    with threadpool_limits(limits=1):
        matrix = np.load(task["matrix_path"], mmap_mode="r", allow_pickle=False)
        root = Path(task["output_root"])
        checkpoint = safe_file(
            root,
            f".checkpoints/{task['library_id']}/{encoded_cluster(task['cluster_id'])}",
        )
        with locked(
            root, "library-fit-" + identity([task["library_id"], task["cluster_id"]])
        ):
            target_store = Checkpoints(
                checkpoint, identity([task["library_id"], task["cluster_id"]])
            )
            background = np.asarray(task["background_rows"], dtype=np.int64)
            bm = background_profiles(
                matrix, background, target_store, task["background_id"]
            )
            all_rows = np.asarray(task["c_rows"], dtype=np.int64)
            folds = np.asarray(task["folds"], dtype=np.int8)
            results = []
            for fold in [0, 1, 2, 3, 4, None]:
                fit_id = f"fold_{fold}" if fold is not None else "final"
                validation_mask = (
                    folds == fold
                    if fold is not None
                    else np.zeros(len(folds), dtype=bool)
                )
                c, v = all_rows[~validation_mask], all_rows[validation_mask]
                validation_ids = [
                    wid
                    for wid, selected in zip(
                        task["c_ids"], validation_mask, strict=True
                    )
                    if selected
                ]
                train_ids = [
                    wid
                    for wid, selected in zip(
                        task["c_ids"], validation_mask, strict=True
                    )
                    if not selected
                ]
                input_id = {
                    **task["input_identity"],
                    "training_ids_sha256": identity(train_ids),
                    "validation_ids_sha256": identity(validation_ids),
                    "background_ids_sha256": task["background_id"],
                }
                context = {
                    "library_id": task["library_id"],
                    "cluster_id": task["cluster_id"],
                    "fit_id": fit_id,
                    "epsilon": task["epsilon"],
                    "sigma_floor": task["sigma_floor"],
                    "representation_contract_id": task["representation_contract_id"],
                    "input_identity": input_id,
                    "validation_ids": validation_ids,
                    "counts": {
                        "training_positive_n": len(c),
                        "validation_positive_n": len(v),
                        "background_n": len(background),
                        "training_parent_n": len(
                            {
                                p
                                for p, selected in zip(
                                    task["c_parents"], validation_mask, strict=True
                                )
                                if not selected
                            }
                        ),
                        "validation_parent_n": len(
                            {
                                p
                                for p, selected in zip(
                                    task["c_parents"], validation_mask, strict=True
                                )
                                if selected
                            }
                        ),
                        "background_parent_n": task["background_parents"],
                    },
                }
                fit_key = identity(
                    [task["library_id"], task["cluster_id"], fit_id, input_id]
                )
                store = Checkpoints(checkpoint / fit_id, fit_key)
                LOG.info(
                    "Cluster %s %s: computing signature",
                    task["cluster_id"],
                    fit_id,
                )
                try:
                    result = fit_signature(
                        matrix,
                        {"C": c, "B": background, "V": v},
                        store=store,
                        context=context,
                        background_moments=bm,
                    )
                except (ValueError, FloatingPointError) as exc:
                    raise ValueError(
                        f"Cluster {task['cluster_id']} {fit_id}: {exc}"
                    ) from exc
                validate_signature(
                    result["signature"],
                    library_id=task["library_id"],
                    cluster_id=task["cluster_id"],
                    fit_id=fit_id,
                )
                results.append(result)
            path = checkpoint / "target.json"
            write_json(path, {"cluster_id": task["cluster_id"], "fits": results})
            cleanup_target_cache(checkpoint, task["library_id"], task["cluster_id"])
            return str(path)


def _write_outputs(stage, paths, identity_inputs, contract, *, execution=None):
    counts = {
        "targets": len(paths),
        "available_final": 0,
        "unsupported_final": 0,
        "valid_discovery_fits": 0,
        "dimension_fold_rows": 0,
        "dimension_final_rows": 0,
        "oof_rows": 0,
    }
    with ExitStack() as stack:
        tables = {
            name: stack.enter_context(_table(stage / name, header))
            for name, header in HEADERS.items()
        }
        for path in paths:
            target = read_json(path)
            cluster, fits = target["cluster_id"], target["fits"]
            if len(fits) != 6:
                raise IntegrityError("Incomplete target result")
            directory = stage / "clusters" / cluster_directory_name(cluster)
            directory.mkdir(parents=True)
            final_status = fits[5]["signature"]["status"]
            counts[
                "available_final" if final_status == "ok" else "unsupported_final"
            ] += 1
            traces = []
            for i, result in enumerate(fits):
                signature, profiles = result["signature"], result["profiles"]
                fit_id = f"fold_{i}" if i < 5 else "final"
                validate_signature(
                    signature,
                    library_id=identity(identity_inputs),
                    cluster_id=cluster,
                    fit_id=fit_id,
                )
                if len(profiles) != 640 or [p["dimension"] for p in profiles] != list(
                    range(640)
                ):
                    raise IntegrityError("Invalid per-fit profile coverage")
                if identity(result["trace"]) != signature["selection_trace_identity"]:
                    raise IntegrityError("Signature trace identity mismatch")
                write_json(directory / f"{fit_id}.json", signature)
                traces.append(result["trace"])
                table = "dimension_folds.tsv.gz" if i < 5 else "dimension_final.tsv.gz"
                for profile in profiles:
                    tables[table]({"cluster_id": cluster, "fold_id": i, **profile})
                counts["dimension_fold_rows" if i < 5 else "dimension_final_rows"] += (
                    len(profiles)
                )
                if i < 5:
                    counts["valid_discovery_fits"] += signature["status"] == "ok"
                    digest = sha256(directory / f"{fit_id}.json")
                    for row in result["oof"]:
                        tables["out_of_fold_positive_scores.tsv.gz"](
                            {
                                "cluster_id": cluster,
                                "fold_id": i,
                                "signature_digest": digest,
                                **row,
                            }
                        )
                        counts["oof_rows"] += 1
                for population in (
                    ("training", "validation") if i < 5 else ("training",)
                ):
                    metrics = signature[f"{population}_metrics"]
                    status = (
                        metrics["status"]
                        if metrics is not None
                        else signature["status"]
                    )
                    positive = signature["counts"][f"{population}_positive_n"]
                    negative = signature["counts"]["background_n"]
                    for metric in sorted(METRICS):
                        tables["cluster_metrics.tsv"](
                            {
                                "cluster_id": cluster,
                                "fit_id": fit_id,
                                "population": population,
                                "metric": metric,
                                "value": metrics.get(metric) if metrics else None,
                                "positive_n": positive,
                                "negative_n": negative,
                                "status": status,
                            }
                        )
            _gzip_json(directory / "selection_trace.json.gz", traces)
            valid_fits = sum(f["signature"]["status"] == "ok" for f in fits[:5])
            for dimension in range(640):
                rows = [f["profiles"][dimension] for f in fits[:5]]
                selected = sum(r["selected"] for r in rows)
                for statistic in sorted(STATS):
                    tables["dimension_summary.tsv.gz"](
                        {
                            "cluster_id": cluster,
                            "dimension": dimension,
                            "statistic": statistic,
                            **_stat([r[statistic] for r in rows]),
                            "selection_count": selected,
                            "selection_frequency": selected / valid_fits
                            if valid_fits
                            else None,
                            "core_count": sum(r["core"] for r in rows),
                        }
                    )
            for population in ("training", "validation"):
                for metric in sorted(METRICS):
                    values = []
                    for f in fits[:5]:
                        m = f["signature"][f"{population}_metrics"]
                        if m is not None and m["status"] == "ok":
                            values.append(m.get(metric))
                    tables["cluster_summary.tsv"](
                        {
                            "cluster_id": cluster,
                            "population": population,
                            "metric": metric,
                            **_stat(values),
                            "final_status": final_status,
                        }
                    )
            tables["cluster_summary.tsv"](
                {
                    "cluster_id": cluster,
                    "population": "discovery",
                    "metric": "signature_size",
                    **_stat(
                        [
                            len(f["signature"]["selected_dimensions"])
                            for f in fits[:5]
                            if f["signature"]["status"] == "ok"
                        ]
                    ),
                    "final_status": final_status,
                }
            )
    manifest = {
        "schema": SCHEMA,
        "cluster_directory_layout": LAYOUT,
        "status": "complete",
        "id": identity(identity_inputs),
        "identity_inputs": identity_inputs,
        "representation_contract": contract,
        "software_version": __version__,
        "execution": execution or {},
        "counts": counts,
        "files": {
            str(p.relative_to(stage)): file_record(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        },
    }
    write_json(stage / "manifest.json", manifest)
    return manifest


def _library_immutable(path, immutable):
    resources = Path(__file__).resolve().parent / "resources"
    return immutable or path == resources or resources in path.parents


@schema_checked
def _load_library(path, *, cluster_id=None):
    if (
        read_json(safe_file(path, "manifest.json")).get("schema")
        == "pclass-dark.library.v2"
    ):
        from .library_reports import load_production

        return load_production(path, cluster_id=cluster_id)
    if cluster_id is not None:
        raise IntegrityError(
            "Legacy library: run pclass-dark build-library to migrate first"
        )
    manifest = verify_bundle(path, SCHEMA)
    if identity(manifest["identity_inputs"]) != manifest["id"]:
        raise IntegrityError("Library identity mismatch")
    contract = representation_contract(manifest["representation_contract"])
    if identity(contract) != manifest["identity_inputs"]["representation_contract_id"]:
        raise IntegrityError("Library representation binding mismatch")
    targets = manifest["identity_inputs"]["cluster_ids"]
    if targets != sorted(set(targets)) or manifest["counts"]["targets"] != len(targets):
        raise IntegrityError("Invalid library target inventory")
    required = set(HEADERS)
    signatures = {}
    for cluster in targets:
        prefix = "clusters/" + stored_cluster_name(cluster, manifest) + "/"
        required.add(prefix + "selection_trace.json.gz")
        with gzip.open(
            safe_file(path, prefix + "selection_trace.json.gz"), "rt", encoding="utf-8"
        ) as handle:
            import json

            traces = json.load(handle)
        if len(traces) != 6:
            raise IntegrityError("Incomplete trace inventory")
        for index, fit in enumerate(
            ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "final"]
        ):
            name = prefix + fit + ".json"
            required.add(name)
            signature = read_json(safe_file(path, name))
            validate_signature(
                signature, library_id=manifest["id"], cluster_id=cluster, fit_id=fit
            )
            if (
                signature["representation_contract_id"] != identity(contract)
                or identity(traces[index]) != signature["selection_trace_identity"]
            ):
                raise IntegrityError("Signature representation/trace mismatch")
            if fit == "final":
                signatures[cluster] = freeze(signature)
    if set(manifest["files"]) != required:
        raise IntegrityError("Library file inventory mismatch")
    for name, header in HEADERS.items():
        opener = gzip.open if name.endswith(".gz") else open
        with opener(path / name, "rt", encoding="utf-8", newline="") as handle:
            if next(csv.reader(handle, delimiter="\t"), None) != header:
                raise IntegrityError(f"Library table header mismatch: {name}")
    return Library(
        manifest["id"],
        MappingProxyType(signatures),
        freeze(contract),
        freeze(manifest["counts"]),
        source_root=path.parent,
        manifest_sha256=sha256(path / "manifest.json"),
        source_path=path,
    )


def load_library(library_dir, *, immutable=False, cluster_id=None):
    """Verify a library under its actual parent-root guard, then detach parameters.

    cluster_id limits v2 verification and returned signatures/counts to one
    target; the shared profile table is still checksummed. None audits all targets.
    Scoped loads retain the immutable parsed fit in target_result to avoid rereads.
    """
    if cluster_id is not None and (not isinstance(cluster_id, str) or not cluster_id):
        raise ValueError("cluster_id must be a nonempty string")
    path = Path(library_dir).absolute()
    if _library_immutable(path.resolve(), immutable):
        from dataclasses import replace

        return replace(_load_library(path, cluster_id=cluster_id), immutable=True)
    with transaction_guard(path.parent):
        return _load_library(path, cluster_id=cluster_id)


def _read_inputs(root, immutable):
    selection = read_json(safe_file(root, "selection.json"))
    if selection.get("schema") != "pclass-dark.selection.v1" or not selection.get(
        "embedding"
    ):
        raise IntegrityError(
            "build-library requires a completed setup-data embedding selection"
        )
    prepared, embedding = (
        safe_file(root, selection[k]) for k in ("prepared", "embedding")
    )
    if (
        sha256(prepared / "manifest.json") != selection["prepared_manifest_sha256"]
        or sha256(embedding / "manifest.json") != selection["embedding_manifest_sha256"]
    ):
        raise IntegrityError("Selected input manifest digest mismatch")
    pm, windows, memberships = verify_prepared(
        prepared, selection["preparation_id"], immutable=immutable
    )
    em = verify_embeddings(embedding, windows, immutable=immutable)
    if (
        em["id"] != selection["embedding_id"]
        or pm["inventory_id"] != em["inventory_id"]
        or selection["inventory_id"] != pm["inventory_id"]
    ):
        raise IntegrityError("Selected prepared/embedding identity mismatch")
    sets = validate_sets(prepared, pm, windows, memberships)
    contract = representation_contract(em["identity_inputs"]["contract"])
    return selection, prepared, embedding, pm, em, windows, sets, contract


def _completed_targets(executor, targets, make_task, limit, job=None):
    """Yield indexed completions with at most limit expanded target jobs queued."""
    job = _target_job if job is None else job
    remaining = iter(enumerate(targets))
    pending = {}

    def submit_next():
        item = next(remaining, None)
        if item is None:
            return False
        index, cluster = item
        pending[executor.submit(job, make_task(cluster))] = index
        return True

    try:
        for _ in range(limit):
            if not submit_next():
                break
        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            # Resolve failures before admitting more work. Stable ordering only
            # breaks simultaneous completion ties; it never waits for an index.
            completed = [
                (pending[f], f.result()) for f in sorted(done, key=pending.__getitem__)
            ]
            for future in done:
                del pending[future]
            for _ in completed:
                submit_next()
            yield from completed
    finally:
        for future in pending:
            future.cancel()


def _build_legacy_library(
    data_dir="work/setup-data",
    output_dir="work/library",
    *,
    workers=1,
    memory_mib=1024,
    epsilon=1e-8,
    sigma_floor=1e-8,
    cluster_ids=None,
    force=False,
):
    """Discover five fold signatures and an all-C refit; resume validated fit stages."""
    for name, value, minimum in (
        ("workers", workers, 1),
        ("memory_mib", memory_mib, 64),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    import numbers

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
    with ExitStack() as guards, threadpool_limits(limits=1):
        for directory in sorted({root} | (set() if immutable else {data})):
            guards.enter_context(transaction_guard(directory))
        selected, prepared, embedding, pm, em, windows, sets, contract = _read_inputs(
            data, immutable
        )
        if cluster_ids is not None and isinstance(cluster_ids, (str, bytes)):
            raise ValueError(
                "cluster_ids must be a sequence of exact IDs, not one string"
            )
        targets = sorted(sets.cluster_ids) if cluster_ids is None else list(cluster_ids)
        if (
            not targets
            or len(set(targets)) != len(targets)
            or any(t not in sets.clusters for t in targets)
        ):
            raise ValueError("Unknown, duplicate, or empty target cluster selection")
        targets.sort()
        validate_cluster_directories(targets)
        numerical = {
            "engine": "cpu-logdiff-reference-fallback-exact-count-v2",
            "numpy": np.__version__,
            "scipy": importlib.metadata.version("scipy"),
            "architecture": platform.machine(),
            "inner_threads": 1,
        }
        inputs = {
            "schema": SCHEMA,
            "algorithm": "library-" + SELECTION_VERSION,
            "cluster_ids": targets,
            "epsilon": epsilon,
            "sigma_floor": sigma_floor,
            "numerical": numerical,
            "representation_contract_id": identity(contract),
            "representation_contract": contract,
            "prepared_manifest": sha256(prepared / "manifest.json"),
            "embedding_manifest": sha256(embedding / "manifest.json"),
            "prepared_id": pm["id"],
            "embedding_id": em["id"],
            "inventory_id": pm["inventory_id"],
            "matrix": em["files"]["embeddings.npy"],
            "row_map": em["files"]["rows.tsv"],
            "folds": pm["files"]["folds.tsv"],
            "C": pm["files"]["cluster_members.json"],
            "B": pm["files"]["background_members.json"],
            "selection_rule": "top-two-prefix-union-gt90-backward-largest-strict-v1",
            "threshold_rule": "max-training-balanced-accuracy-highest-cutoff-v1",
        }
        library_id = identity(inputs)
        target = safe_file(root, library_id)
        private = safe_file(root, ".checkpoints/" + library_id)
        if target.exists() and not force:
            loaded = _load_library(target)
            manifest = read_json(target / "manifest.json")
            _select_library(root, target, manifest)
            return LibraryResult(library_id, target, dict(loaded.counts), True)
        if force:
            archive(root, private)
        row_index = {w["window_id"]: i for i, w in enumerate(windows)}
        parents = {w["window_id"]: w["sequence_id"] for w in windows}
        folds = defaultdict(dict)
        for row in read_tsv(prepared / "folds.tsv"):
            folds[row["cluster_id"]][row["window_id"]] = int(row["fold_id"])
        bg_summary = read_json(prepared / "background_members.json")["backgrounds"]
        # Conservative peak for sort/group/count temporaries, not mapped caches.
        required_workspace = max(
            64,
            max(
                math.ceil(
                    384
                    * (len(sets.cluster_members(c)) + bg_summary[c]["background_count"])
                    / 2**20
                )
                + 8
                for c in targets
            ),
        )
        if memory_mib < required_workspace:
            raise ValueError(
                f"At least {required_workspace} MiB numerical workspace is required"
            )
        active_workers = min(
            workers,
            len(targets),
            os.cpu_count() or 1,
            32,
            memory_mib // required_workspace,
        )
        LOG.info(
            "Building %s targets on CPU float64 with %s worker(s)",
            len(targets),
            active_workers,
        )

        def make_task(cluster):
            cids, bids = (
                list(sets.cluster_members(cluster)),
                sets.background_members(cluster),
            )
            return {
                "library_id": library_id,
                "cluster_id": cluster,
                "output_root": str(root),
                "matrix_path": str(embedding / "embeddings.npy"),
                "c_ids": cids,
                "c_rows": [row_index[w] for w in cids],
                "c_parents": [parents[w] for w in cids],
                "folds": [folds[cluster][w] for w in cids],
                "background_rows": [row_index[w] for w in bids],
                "background_parents": len({parents[w] for w in bids}),
                "background_id": sets.summary(cluster)["background_ids_sha256"],
                "epsilon": epsilon,
                "sigma_floor": sigma_floor,
                "representation_contract_id": identity(contract),
                "input_identity": {
                    k: inputs[k]
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
            }

        # Refill free slots immediately; expand B only for admitted jobs.
        completed_paths = {}
        executor = (
            guards.enter_context(
                ProcessPoolExecutor(
                    max_workers=active_workers,
                    mp_context=multiprocessing.get_context("spawn"),
                )
            )
            if active_workers > 1
            else None
        )
        completions = (
            _completed_targets(executor, targets, make_task, active_workers)
            if executor
            else (
                (index, _target_job(make_task(cluster)))
                for index, cluster in enumerate(targets)
            )
        )
        for index, path in completions:
            completed_paths[index] = path
            LOG.info(
                "Library targets durable: %s/%s", len(completed_paths), len(targets)
            )
        paths = [completed_paths[index] for index in range(len(targets))]
        with publication_transaction(root, domain="library") as transaction:
            with staging(root) as stage:
                manifest = _write_outputs(
                    stage,
                    paths,
                    inputs,
                    contract,
                    execution={
                        "device": "cpu",
                        "dtype": "float64",
                        "workers_requested": workers,
                        "workers_used": active_workers,
                        "memory_mib": memory_mib,
                        "per_worker_workspace_mib": memory_mib // active_workers,
                    },
                )
                _load_library(stage)
                publish(root, stage, target, force, transaction)
            _select_library(root, target, manifest)
        LOG.info(
            "Library complete: %s targets, %s usable final signatures",
            len(targets),
            manifest["counts"]["available_final"],
        )
        return LibraryResult(library_id, target, manifest["counts"], False)


def _select_library(root, target, manifest):
    write_json(
        root / "selection.json",
        {
            "schema": "pclass-dark.library-selection.v1",
            "library_id": manifest["id"],
            "library": str(target.relative_to(root)),
            "manifest_sha256": sha256(target / "manifest.json"),
            "preparation_id": manifest["identity_inputs"]["prepared_id"],
            "embedding_id": manifest["identity_inputs"]["embedding_id"],
        },
    )


def build_library(
    data_dir=None,
    output_dir=None,
    *,
    workers=1,
    memory_mib=1024,
    epsilon=1e-8,
    sigma_floor=1e-8,
    cluster_ids=None,
    force=False,
    cross_validate=False,
):
    """Build all-C signatures and reference distributions; optionally run research."""
    from .library_build import build
    from .paths import default_library_output_dir, default_setup_input_dir

    data_dir = default_setup_input_dir() if data_dir is None else data_dir
    output_dir = default_library_output_dir() if output_dir is None else output_dir

    return build(
        data_dir,
        output_dir,
        workers=workers,
        memory_mib=memory_mib,
        epsilon=epsilon,
        sigma_floor=sigma_floor,
        cluster_ids=cluster_ids,
        force=force,
        cross_validate=cross_validate,
    )
