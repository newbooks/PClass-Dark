"""Native Swiss-Prot comparison with frozen profunc truth and resumable evidence.

No model/service/resource work on import. See SPEC-0017 for scientific scope and
resource limits. Errors use the external evaluator's public exception classes.
"""

import os
import shutil
import sqlite3
import sys
import uuid
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock, Timeout

from . import external_evaluation as shared
from . import external_evaluation_collect as common
from . import external_evaluation_resources as resources
from . import external_evaluation_store as store
from . import swiss_evaluation_collect as collect
from . import swiss_evaluation_truth as truth
from .external_evaluation_types import (
    EvaluationIntegrityError,
    EvaluationPreflight,
    ScoringConfig,
)
from .swiss_evaluation_types import (
    SwissDarkConfig,
    SwissExecutionConfig,
    SwissPClassConfig,
)

__all__ = [
    "evaluate_swiss",
    "SwissDarkConfig",
    "SwissExecutionConfig",
    "SwissPClassConfig",
    "ScoringConfig",
]


def evaluate_swiss(
    fasta,
    *,
    work_dir,
    output=None,
    pclass=None,
    dark=None,
    scoring=None,
    execution=None,
    score_only=False,
    preflight_only=False,
):
    """Evaluate supplied Swiss-Prot records, preserving normal self matches.

    Raises ValueError for unsupported input/config/truth, OSError for filesystem
    failures, and Evaluation* errors for inference, resources or corrupt state.
    Score-only needs the input FASTA and saved state, not live annotations/models.
    """
    if os.name != "posix" or sys.version_info < (3, 11):
        raise ValueError("Swiss evaluation requires POSIX/Python >=3.11")
    if (
        type(score_only) is not bool
        or type(preflight_only) is not bool
        or score_only
        and preflight_only
    ):
        raise ValueError("Invalid/mutually exclusive evaluation modes")
    if score_only and any(c is not None for c in (pclass, dark, execution)):
        raise ValueError("Score-only rejects inference configuration")
    for value, cls in (
        (pclass, SwissPClassConfig),
        (dark, SwissDarkConfig),
        (execution, SwissExecutionConfig),
        (scoring, ScoringConfig),
    ):
        if value is not None and not isinstance(value, cls):
            raise ValueError(f"Expected {cls.__name__}")
    sc = scoring or ScoringConfig()
    _, state, output, paths = shared._paths(
        work_dir,
        output,
        [fasta, None, sc.go_graph, sc.cluster_go_map],
        state_name=".swiss-evaluation",
        table_name="swiss_evaluation.tsv",
    )
    fasta = paths[0]
    state.mkdir(exist_ok=True)
    output_lock = output.with_name("." + output.name + ".evaluation.lock")
    try:
        with (
            FileLock(str(state / "lock"), timeout=0),
            FileLock(str(output_lock), timeout=0),
        ):
            if not (state / "workspace.json").exists():
                store.write_json(
                    state / "workspace.json",
                    {"schema_version": 1, "workspace_uuid": uuid.uuid4().hex},
                )
            store.recover_publication(output, state)
            store.owner_check(output, state)
            if score_only:
                selection_path = state / "selection.json"
                if not selection_path.exists():
                    raise ValueError(
                        "No saved selection; run collection before score-only"
                    )
                selection = store.read(selection_path)
                resolver, source, db = truth.saved_resolver(state, selection)
                try:
                    cohort = store.snapshot_inputs(
                        fasta,
                        None,
                        state,
                        truth_resolver=resolver,
                        truth_source=source,
                        accession_resolver=truth.primary_accession,
                    )
                finally:
                    db.close()
                selection = shared._selection(state, cohort["input_id"])
                if selection is None or selection["truth_id"] != cohort["truth_id"]:
                    raise ValueError("Input does not match saved Swiss selection")
                generations = {
                    e: collect.verify_generation(
                        state, e, selection[e + "_generation"], cohort["input_id"]
                    )
                    for e in ("pclass", "dark")
                }
                graph_id, index, graph_path, payload, mapping = (
                    shared._scoring_resources(state, sc, selection, True)
                )
            else:
                source, annotations, runtime = truth.freeze_annotations(state)
                truth.validate_headers(fasta, annotations)
                cohort = store.snapshot_inputs(
                    fasta,
                    None,
                    state,
                    truth_resolver=truth.TruthResolver(annotations),
                    truth_source=source,
                    accession_resolver=truth.primary_accession,
                )
                del annotations
                graph_id, index, graph_path = resources.snapshot_graph(
                    state, sc.go_graph or runtime.dataset / "go_graph.pkl"
                )
                payload = resources.snapshot_map(
                    state, sc.cluster_go_map, runtime=runtime
                )
                mapping = resources.map_from_payload(payload)
            shared._validate_terms(state, cohort, index, mapping)
            if not score_only:
                pc, dc, ex = (
                    pclass or SwissPClassConfig(),
                    dark or SwissDarkConfig(),
                    execution or SwissExecutionConfig(),
                )
                # Resolve all paths before passing a config across a spawn boundary.
                dc = SwissDarkConfig(
                    **{
                        k: str(Path(v).resolve())
                        if k in {"library_dir", "embedding_cache", "prepared_dir"}
                        and v is not None
                        else v
                        for k, v in asdict(dc).items()
                    }
                )
                from .swiss_evaluation_worker import pclass_contract

                pclass_contract()  # dependency/API check only; no service activity
                generations, details = {}, {}
                # Validate Dark resources before any remote predictions. Dark runs
                # first so its pinned runtime need only be loaded once.
                for engine, config, count in (
                    ("dark", dc, dc.chunk_size),
                    ("pclass", pc, pc.batch_size),
                ):
                    with collect.NativeWorker(
                        engine, config, state, ex, mapping.clusters
                    ) as worker:
                        if preflight_only:
                            info = (
                                {
                                    "contract": worker.contract,
                                    "service_device": "unverified",
                                }
                                if engine == "pclass"
                                else {"windows": 0, "cache_hits": 0, "cache_misses": 0}
                            )
                            if engine == "dark":
                                for rows in common.plan_chunks(
                                    common.input_rows(state, cohort["input_id"]),
                                    engine,
                                    count,
                                ):
                                    attempt = state / "attempts" / uuid.uuid4().hex
                                    attempt.mkdir(parents=True)
                                    try:
                                        found = worker.task(
                                            rows, attempt, inspect_only=True
                                        )["details"]
                                        for key in info:
                                            info[key] += found[key]
                                    finally:
                                        shutil.rmtree(attempt)
                            info["contract"] = worker.contract
                            info.update(getattr(worker, "details", {}))
                            info.update(
                                collect.progress_summary(state, cohort, worker, count)
                            )
                            info["max_evidence_bytes"] = ex.max_raw_chunk_mib * 2**20
                            details[engine] = info
                        else:
                            from .swiss_evaluation_resume import completed_generation

                            generations[engine] = completed_generation(
                                state, cohort, worker, count
                            ) or collect.collect_generation(
                                state, cohort, worker, count
                            )
                if preflight_only:
                    return EvaluationPreflight(
                        cohort["input_id"],
                        cohort["truth_id"],
                        cohort["query_count"],
                        details,
                        graph_id,
                        payload["mapping_id"],
                    )
            return shared._score_and_publish(
                state,
                output,
                cohort,
                generations["pclass"],
                generations["dark"],
                graph_id,
                index,
                graph_path,
                payload,
                mapping,
                sc,
                saved_reader=collect.saved_queries,
                table_name="swiss_evaluation.tsv",
                evaluation_kind="swiss",
            )
    except Timeout as exc:
        raise RuntimeError("Swiss evaluation workspace/output busy") from exc
    except (KeyError, TypeError, AttributeError) as exc:
        raise EvaluationIntegrityError(
            f"Malformed Swiss evaluation metadata: {exc}"
        ) from exc
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Duplicate/invalid Swiss input: {exc}") from exc
