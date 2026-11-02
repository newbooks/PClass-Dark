"""Durable comparison of external PClass / Dark inference (SPEC-0015).

Inference dependencies are resolved only for pending collection. Cached scoring
never contacts a service, loads a model or invokes an inference executable.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sqlite3
import sys
import tempfile
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from filelock import FileLock, Timeout

from . import external_evaluation_collect as collect
from . import external_evaluation_resources as resources
from . import external_evaluation_scoring as scoring_ops
from . import external_evaluation_store as store
from .artifacts import identity
from .external_evaluation_scoring import score_external_predictions  # noqa: F401
from .external_evaluation_types import (  # noqa: F401 (public API)
    HEADERS,
    ClusterGoMap,
    ComparisonRow,
    DarkConfig,
    EvaluationInferenceError,
    EvaluationIntegrityError,
    EvaluationPreflight,
    EvaluationQuery,
    EvaluationRecoveryError,
    EvaluationResourceError,
    EvaluationResult,
    EvaluationSnapshotError,
    ExecutionConfig,
    PClassConfig,
    SavedDarkQuery,
    SavedPClassQuery,
    ScoringConfig,
    digest_id,
)


def _selection(state, input_id):
    path = state / "selection.json"
    if not path.exists():
        return None
    s = store.read(path)
    score_dir = state / "scores" / digest_id(s["score_id"])
    store.checked(score_dir / "manifest.json", s["score_manifest"])
    m = store.read(score_dir / "manifest.json")
    store.verify_files(score_dir, m)
    if any(
        s[k] != m[k]
        for k in [
            "input_id",
            "pclass_generation",
            "dark_generation",
            "graph_id",
            "mapping_id",
        ]
    ):
        raise EvaluationIntegrityError("Selection/score manifest mismatch")
    return s if s["input_id"] == input_id else None


def _paths(work_dir, output, inputs):
    if not isinstance(work_dir, (str, Path)):
        raise ValueError("work_dir must be a path")
    root = Path(work_dir).absolute()
    if not root.is_dir() or root.is_symlink():
        raise ValueError("work_dir must be an existing non-symlink directory")
    root = root.resolve()
    state = root / ".external-evaluation"
    if state.is_symlink():
        raise ValueError("State directory must not be a symlink")
    output = (
        Path(output).absolute()
        if output is not None
        else root / "external_evaluation.tsv"
    )
    if output.is_symlink():
        raise ValueError("Output must not be a symlink")
    output = output.resolve()
    if not output.parent.is_dir() or output == state or state in output.parents:
        raise ValueError("Invalid output location/parent")
    resolved = []
    for p in inputs:
        if p is None:
            resolved.append(None)
            continue
        p = Path(p).resolve()
        if not p.is_file() or p == output:
            raise ValueError(f"Missing/overlapping input: {p}")
        resolved.append(p)
    if any(p is not None and state in p.parents for p in resolved[:2]):
        raise ValueError("FASTA/truth input overlaps private evaluation state")
    marker, journal = store.publication_paths(output)
    for p in [
        state,
        output,
        marker,
        journal,
        output.with_name("." + output.name + ".evaluation.lock"),
        *(state.rglob("*") if state.is_dir() else []),
    ]:
        if p.is_symlink():
            raise ValueError(f"Symlink at owned evaluation path: {p}")
    if any(p in [marker, journal] for p in resolved if p is not None):
        raise ValueError("Input overlaps publication metadata")
    return root, state, output, resolved


def _query_records(state, cohort):
    truth = state / "truth" / cohort["truth_id"]
    store.verify_files(truth, store.read(truth / "manifest.json"))
    with sqlite3.connect(truth / "truth.sqlite") as db:
        for r in collect.input_rows(state, cohort["input_id"]):
            row = db.execute(
                "SELECT go_ids_json FROM truth WHERE query_id=?", (r["id"],)
            ).fetchone()
            if row is None:
                raise EvaluationIntegrityError("Missing saved truth row")
            yield EvaluationQuery(
                r["ordinal"],
                r["id"],
                r["length"],
                r["sequence_sha256"],
                tuple(json.loads(row[0])),
            )


def _validate_terms(state, cohort, index, mapping):
    unknown = set()
    for q in _query_records(state, cohort):
        unknown.update(set(q.truth_go) - index.all_terms)
        if len(unknown) > 100:
            break
    for c in mapping.clusters:
        unknown.update(set(mapping.terms(c)) - index.all_terms)
    if unknown:
        raise ValueError(f"Unknown/nonactive evaluation MF terms: {sorted(unknown)}")


def _scoring_resources(state, config, selection, score_only):
    graph, map_path = config.go_graph, config.cluster_go_map
    if score_only:
        if selection is None:
            raise ValueError("No saved selection; run collection before score-only")
        if graph is None:
            folder = state / "graphs" / digest_id(selection["graph_id"])
            store.verify_files(folder, store.read(folder / "manifest.json"))
            graph = folder / "go_graph.pkl"
        if map_path is None:
            map_path = (
                state / "mappings" / (digest_id(selection["mapping_id"]) + ".json")
            )
    graph_id, index, graph_path = resources.snapshot_graph(state, graph)
    payload = resources.snapshot_map(state, map_path)
    return graph_id, index, graph_path, payload, resources.map_from_payload(payload)


def _blocks(rows, count=64):
    block = []
    for row in rows:
        block.append(row)
        if len(block) == count:
            yield block
            block = []
    if block:
        yield block


def _score_rows(db, index, mapping, graph_path, map_payload, workers):
    if workers == 1:
        for q, p, d in scoring_ops.joined_records(db):
            yield scoring_ops.score_query(q, p, d, index, mapping)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=scoring_ops.init_score_worker,
            initargs=(str(graph_path), map_payload),
        ) as pool:
            pending = []
            try:
                for block in _blocks(scoring_ops.joined_records(db)):
                    pending.append(pool.submit(scoring_ops.score_block, block))
                    if len(pending) >= 2 * workers:
                        for row, status in pending.pop(0).result():
                            row["pclass_go"] = tuple(row["pclass_go"])
                            row["dark_go"] = tuple(row["dark_go"])
                            yield ComparisonRow(**row), status
                for future in pending:
                    for row, status in future.result():
                        yield ComparisonRow(**row), status
            finally:
                for future in pending:
                    future.cancel()


def _score_and_publish(
    state,
    output,
    cohort,
    pclass,
    dark,
    graph_id,
    index,
    graph_path,
    map_payload,
    mapping,
    config,
):
    from profunc.go import similarity

    key = {
        "input_id": cohort["input_id"],
        "truth_id": cohort["truth_id"],
        "pclass_generation": pclass["id"],
        "dark_generation": dark["id"],
        "graph_id": graph_id,
        "mapping_id": map_payload["mapping_id"],
        "metric_version": "lin-amb-v1",
        "reduction_version": "top-neighbor-v2",
        "serialization_version": 1,
        "tolerance": 1e-12,
        "scoring_code": identity(
            {
                name: store.record(Path(__file__).parent / name)["sha256"]
                for name in [
                    "external_evaluation_scoring.py",
                    "external_evaluation_types.py",
                ]
            }
        ),
        "metric_code": store.record(Path(similarity.__file__))["sha256"],
    }
    score_id = identity(key)
    score_dir = state / "scores" / score_id
    if score_dir.exists():
        manifest = store.read(score_dir / "manifest.json")
        if manifest.get("id") != score_id or any(
            manifest.get(k) != v for k, v in key.items()
        ):
            raise EvaluationIntegrityError("Saved score identity mismatch")
        store.verify_files(score_dir, manifest)
    else:
        staged = state / (".score-" + uuid.uuid4().hex)
        staged.mkdir()
        db = scoring_ops.scoring_db(staged / "join.sqlite")
        try:
            count = scoring_ops.join_predictions(
                db,
                _query_records(state, cohort),
                collect.saved_queries(state, "pclass", pclass),
                collect.saved_queries(state, "dark", dark),
            )
            with (
                (staged / "external_evaluation.tsv").open(
                    "w", encoding="utf-8", newline="\n"
                ) as table,
                (staged / "query_status.jsonl").open(
                    "w", encoding="utf-8", newline="\n"
                ) as statuses,
            ):
                table.write("\t".join(HEADERS) + "\n")
                written = 0
                for row, status in _score_rows(
                    db, index, mapping, graph_path, map_payload, config.workers
                ):
                    table.write("\t".join(row.cells()) + "\n")
                    statuses.write(
                        json.dumps(status, sort_keys=True, allow_nan=False) + "\n"
                    )
                    written += 1
                if written != count:
                    raise EvaluationIntegrityError("Scoring row count mismatch")
            store.source_unchanged(cohort)
            manifest = {
                "schema_version": 1,
                "id": score_id,
                **key,
                "query_count": count,
                "created_at": store.now(),
                "files": {
                    n: store.record(staged / n)
                    for n in ["external_evaluation.tsv", "query_status.jsonl"]
                },
            }
            db.close()
            (staged / "join.sqlite").unlink()
            store.publish_directory(staged, score_dir, manifest)
        finally:
            db.close()
            if staged.exists():
                shutil.rmtree(staged)
    store.source_unchanged(cohort)
    store.publish_table(output, state, score_dir, manifest)
    return EvaluationResult(
        output,
        score_dir / "manifest.json",
        score_id,
        cohort["input_id"],
        cohort["truth_id"],
        pclass["id"],
        dark["id"],
        cohort["query_count"],
    )


def evaluate_external(
    fasta,
    *,
    work_dir,
    output=None,
    truth_tsv=None,
    pclass=None,
    dark=None,
    scoring=None,
    execution=None,
    score_only=False,
    preflight_only=False,
):
    """Run existing classifier commands and score their top matches.

    No classifier models, optional embedding packages or PClass Python modules
    are imported. score_only never resolves classifier commands. See SPEC-0015.
    """
    if sys.version_info < (3, 11) or os.name != "posix":
        raise ValueError("Evaluation requires POSIX and Python >=3.11")
    if (
        type(score_only) is not bool
        or type(preflight_only) is not bool
        or (score_only and preflight_only)
    ):
        raise ValueError("Invalid/mutually exclusive evaluation modes")
    if score_only and any(v is not None for v in (pclass, dark, execution)):
        raise ValueError("Score-only does not accept inference configuration")
    for value, kind in (
        (pclass, PClassConfig),
        (dark, DarkConfig),
        (scoring, ScoringConfig),
        (execution, ExecutionConfig),
    ):
        if value is not None and not isinstance(value, kind):
            raise ValueError(f"Expected {kind.__name__}")
    sc = scoring or ScoringConfig()
    _, state, output, paths = _paths(
        work_dir, output, [fasta, truth_tsv, sc.go_graph, sc.cluster_go_map]
    )
    fasta, truth_tsv, _, _ = paths
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
            with tempfile.NamedTemporaryFile(
                prefix=".evaluation-writable-", dir=output.parent
            ):
                pass
            cohort = store.snapshot_inputs(fasta, truth_tsv, state)
            selection = _selection(state, cohort["input_id"])
            runtime = {}
            if score_only:
                if selection is None:
                    raise ValueError(
                        "No saved selection; run collection before score-only"
                    )
                generations = {
                    e: collect.verify_generation(
                        state,
                        e,
                        digest_id(selection[e + "_generation"]),
                        cohort["input_id"],
                    )
                    for e in ("pclass", "dark")
                }
            else:
                pc, dc, ex = (
                    pclass or PClassConfig(),
                    dark or DarkConfig(),
                    execution or ExecutionConfig(),
                )
                runtime = {
                    e: collect.command_resource(e, c, ex)
                    for e, c in (("pclass", pc), ("dark", dc))
                }
            graph_id, index, graph_path, payload, mapping = _scoring_resources(
                state, sc, selection, score_only
            )
            _validate_terms(state, cohort, index, mapping)
            if preflight_only:
                return EvaluationPreflight(
                    cohort["input_id"],
                    cohort["truth_id"],
                    cohort["query_count"],
                    {e: r["contract"]["argv"] for e, r in runtime.items()},
                    graph_id,
                    payload["mapping_id"],
                )
            if not score_only:
                generations = {
                    e: collect.collect_generation(state, cohort, runtime[e], count, ex)
                    for e, count in (("pclass", pc.batch_size), ("dark", dc.chunk_size))
                }
            return _score_and_publish(
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
            )
    except Timeout as exc:
        raise RuntimeError("Evaluation workspace/output busy") from exc
    except (KeyError, TypeError, AttributeError) as exc:
        raise EvaluationIntegrityError(
            f"Malformed/incompatible evaluation metadata: {exc}"
        ) from exc
    except sqlite3.IntegrityError as exc:
        raise ValueError(f"Duplicate/invalid evaluation input: {exc}") from exc
