"""Network/model-free Lin/AMB scoring and disk-backed query joins."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from .external_evaluation_types import (
    ComparisonRow,
    EvaluationQuery,
    SavedDarkQuery,
    SavedPClassQuery,
    digest_id,
    go_set,
    integer,
    number,
    query_id,
)


def metric(index, truth, prediction):
    from profunc.go.similarity import go_set_similarity_with_details

    truth, prediction = go_set(truth, nonempty=True), go_set(prediction)
    unknown = (set(truth) | set(prediction)) - index.all_terms
    if unknown:
        warnings.warn(
            f"Unknown evaluation MF terms: {sorted(unknown)}; excluding these terms "
            "from semantic scoring (possible GO graph/annotation version mismatch)",
            UserWarning,
            stacklevel=2,
        )
    if not prediction:
        return None, {
            "status": "unavailable",
            "reason": "empty_prediction",
            "source_counts": [len(truth), 0],
            "defined_counts": [0, 0],
            "directional_means": [None, None],
            "clamp": None,
            **({"missing_terms": [sorted(unknown), []]} if unknown else {}),
        }
    detail = go_set_similarity_with_details(index, truth, prediction)
    value = detail.similarity
    clamp = None
    if value is not None:
        number("semantic similarity", value, -1e-12, 1 + 1e-12)
        bounded = min(1.0, max(0.0, value))
        if bounded != value:
            clamp = {"original": value, "clamped": bounded}
        value = bounded
    return value, {
        "status": "defined" if value is not None else "unavailable",
        "reason": None if value is not None else "undefined_similarity",
        "source_counts": [len(truth), len(prediction)],
        "defined_counts": [
            len(detail.best_matches_1_to_2),
            len(detail.best_matches_2_to_1),
        ],
        "directional_means": [detail.mean_1_to_2, detail.mean_2_to_1],
        "clamp": clamp,
        **(
            {
                "missing_terms": [
                    sorted(set(truth) & unknown),
                    sorted(set(prediction) & unknown),
                ]
            }
            if unknown
            else {}
        ),
    }


def project_pclass(result):
    if not isinstance(result, Mapping) or not isinstance(result.get("id"), str):
        raise ValueError("Invalid PClass query result")
    neighbors = result.get("neighbors")
    if not isinstance(neighbors, list):
        raise ValueError("Invalid PClass neighbor result")
    projected = []
    for neighbor in neighbors:
        if (
            not isinstance(neighbor, Mapping)
            or not isinstance(neighbor.get("accession"), str)
            or not neighbor["accession"]
        ):
            raise ValueError("Invalid PClass neighbor")
        distance = number("embedding distance", neighbor.get("distance"), 0, 2)
        projected.append(
            {
                "accession": neighbor["accession"],
                "distance": distance,
                "go_mf_terms": go_set(neighbor.get("go_mf_terms")),
            }
        )
    nearest = min(projected, key=lambda n: n["distance"], default=None)
    return {
        "id": result["id"],
        "go_terms": nearest["go_mf_terms"] if nearest else (),
        "neighbors": projected,
    }


def project_dark(result):
    if (
        not isinstance(result, Mapping)
        or result.get("schema") != "pclass-dark.infer-query.v1"
        or not isinstance(result.get("query_id"), str)
        or result.get("status") not in {"matches", "no_accepted_matches"}
        or not isinstance(result.get("matches"), list)
    ):
        raise ValueError("Invalid Dark query result/schema")
    matches, seen = [], set()
    for match in result["matches"]:
        if not isinstance(match, Mapping):
            raise ValueError("Invalid Dark match")
        cluster = match.get("cluster_id")
        if (
            not isinstance(cluster, str)
            or not cluster
            or cluster in seen
            or match.get("accepted") is not True
        ):
            raise ValueError("Invalid/duplicate/nonaccepted Dark cluster")
        seen.add(cluster)
        b = number("Dark B", match.get("background_tail_fraction"), 0, 1, positive=True)
        matches.append({"cluster_id": cluster, "B": b})
    if bool(matches) != (result["status"] == "matches"):
        raise ValueError("Inconsistent Dark status")
    return {
        "query_id": result["query_id"],
        "status": result["status"],
        "matches": matches,
    }


def cached_metric(index):
    """Bound repeated full-set comparisons to one frozen graph/worker lifetime."""
    return lru_cache(maxsize=4096)(
        lambda truth, prediction: metric(index, truth, prediction)
    )


def score_query(query, pclass, dark, index, mapping, *, metric_func=None):
    truth = go_set(query.truth_go, nonempty=True)
    terms = go_set(pclass["go_terms"])
    nearest = min(pclass["neighbors"], key=lambda n: n["distance"], default=None)
    winner = min(dark["matches"], key=lambda m: (m["B"], m["cluster_id"]), default=None)
    dark_terms = mapping.terms(winner["cluster_id"]) if winner else ()
    compare = metric_func or (lambda a, b: metric(index, a, b))
    p_match, p_status = compare(truth, terms)
    d_match, d_status = compare(truth, dark_terms)
    row = ComparisonRow(
        query.query_id,
        terms,
        nearest["distance"] if nearest else None,
        p_match,
        dark_terms,
        winner["B"] if winner else None,
        d_match,
    )
    status = {
        "schema_version": 1,
        "ordinal": query.ordinal,
        "query_id": query.query_id,
        "truth_go": truth,
        "selected_dark_cluster": winner["cluster_id"] if winner else None,
        "pclass_nearest_accession": nearest["accession"] if nearest else None,
        "pclass_status": "predicted"
        if terms
        else ("empty_go_prediction" if nearest else "no_neighbors"),
        "dark_status": "predicted" if winner else "no_accepted_matches",
        "pclass_metric": p_status,
        "dark_metric": d_status,
    }
    return row, status


def scoring_db(path):
    db = sqlite3.connect(path)
    db.execute("PRAGMA cache_size=-16384")
    db.execute("PRAGMA temp_store=FILE")
    db.executescript("""
    CREATE TABLE queries(ordinal INTEGER PRIMARY KEY,query_id TEXT UNIQUE,
        length INTEGER,sequence_sha256 TEXT,truth_go TEXT);
    CREATE TABLE pclass(query_id TEXT PRIMARY KEY,length INTEGER,
        sequence_sha256 TEXT,payload TEXT);
    CREATE TABLE dark(query_id TEXT PRIMARY KEY,length INTEGER,
        sequence_sha256 TEXT,payload TEXT);
    """)
    return db


def join_predictions(db, inputs, pclass_results, dark_results):
    count = 0
    try:
        for q in inputs:
            if (
                not isinstance(q, EvaluationQuery)
                or type(q.ordinal) is not int
                or q.ordinal != count
            ):
                raise ValueError("Queries must have consecutive input ordinals")
            query_id(q.query_id)
            integer("query length", q.length, 1, 1000000)
            digest_id(q.sequence_sha256)
            db.execute(
                "INSERT INTO queries VALUES (?,?,?,?,?)",
                (
                    q.ordinal,
                    q.query_id,
                    q.length,
                    q.sequence_sha256,
                    json.dumps(go_set(q.truth_go, nonempty=True)),
                ),
            )
            count += 1
            if count % 1000 == 0:
                db.commit()
        if not count:
            raise ValueError("Empty evaluation cohort")
        for engine, records, projector in (
            ("pclass", pclass_results, project_pclass),
            ("dark", dark_results, project_dark),
        ):
            total = 0
            for r in records:
                expected_type = (
                    SavedPClassQuery if engine == "pclass" else SavedDarkQuery
                )
                if not isinstance(r, expected_type):
                    raise ValueError(f"Expected {expected_type.__name__}")
                integer("saved query length", r.length, 1, 1000000)
                digest_id(r.sequence_sha256)
                if engine == "dark" and (
                    r.result.get("length") != r.length
                    or r.result.get("sequence_sha256") != r.sequence_sha256
                ):
                    raise ValueError("Dark echoed sequence identity differs")
                expected = db.execute(
                    "SELECT length,sequence_sha256 FROM queries WHERE query_id=?",
                    (r.query_id,),
                ).fetchone()
                if expected != (r.length, r.sequence_sha256):
                    raise ValueError(
                        f"Unexpected/mismatched {engine} query: {r.query_id}"
                    )
                projected = projector(r.result)
                if projected.get("id", projected.get("query_id")) != r.query_id:
                    raise ValueError("Result ID differs from request")
                db.execute(
                    f"INSERT INTO {engine} VALUES (?,?,?,?)",
                    (
                        r.query_id,
                        r.length,
                        r.sequence_sha256,
                        json.dumps(projected, allow_nan=False),
                    ),
                )
                total += 1
                if total % 1000 == 0:
                    db.commit()
            if total != count:
                raise ValueError(f"Missing {engine} query results")
        db.commit()
    except sqlite3.IntegrityError as exc:
        raise ValueError("Duplicate query/result ID") from exc
    return count


def joined_records(db):
    for ordinal, qid, length, digest, truth, p, d in db.execute(
        "SELECT q.ordinal,q.query_id,q.length,q.sequence_sha256,q.truth_go,"
        "p.payload,d.payload "
        "FROM queries q JOIN pclass p USING(query_id) "
        "JOIN dark d USING(query_id) ORDER BY q.ordinal"
    ):
        yield (
            EvaluationQuery(ordinal, qid, length, digest, tuple(json.loads(truth))),
            json.loads(p),
            json.loads(d),
        )


def score_external_predictions(
    inputs, pclass_results, dark_results, *, graph, cluster_go_map
):
    """Yield ordered comparisons with bounded RAM; close the generator for cleanup.

    Raises ValueError for malformed, duplicate, extra or missing records and
    unknown clusters.
    Unknown GO terms warn and are excluded from semantic scoring.
    No inference dependencies, network or permanent files are used.
    """
    with tempfile.TemporaryDirectory(prefix="pclass-evaluation-score-") as directory:
        db = scoring_db(Path(directory) / "join.sqlite")
        try:
            join_predictions(db, inputs, pclass_results, dark_results)
            for q, p, d in joined_records(db):
                yield score_query(q, p, d, graph, cluster_go_map)[0]
        finally:
            db.close()


# Spawned scoring workers hold only the fixed graph/map, never an embedding model.
_worker_index = _worker_map = _worker_metric = _worker_threads = None


def init_score_worker(graph_path, mapping_payload):
    from threadpoolctl import threadpool_limits

    from .external_evaluation_resources import load_graph, map_from_payload

    global _worker_index, _worker_map, _worker_metric, _worker_threads
    _worker_threads = threadpool_limits(limits=1)
    _worker_index = load_graph(graph_path)
    _worker_map = map_from_payload(mapping_payload)
    _worker_metric = cached_metric(_worker_index)


def score_block(block):
    return [
        (asdict(row), status)
        for row, status in (
            score_query(q, p, d, _worker_index, _worker_map, metric_func=_worker_metric)
            for q, p, d in block
        )
    ]
