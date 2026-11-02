"""Freeze existing profunc scoring data; never inspect classifier models."""

import shutil
import uuid
from pathlib import Path

from .artifacts import identity
from .external_evaluation_store import (
    checked,
    publish_directory,
    read,
    record,
    verify_files,
    write_json,
)
from .external_evaluation_types import ClusterGoMap, go_set


def load_graph(path):
    from profunc.go.inspect import _load_go_graph
    from profunc.go.similarity import build_fast_go_similarity_index

    graph = _load_go_graph(Path(path))
    terms = {
        k: v
        for k, v in graph["terms"].items()
        if not v.get("obsolete")
        and not v.get("is_obsolete")
        and v.get("namespace", "molecular_function") == "molecular_function"
    }
    return build_fast_go_similarity_index({"meta": graph["meta"], "terms": terms})


def snapshot_graph(state, path=None):
    if path is None:
        from profunc.runtime import resolve_runtime

        path = resolve_runtime().dataset / "go_graph.pkl"
    path = Path(path).resolve()
    expected = record(path)
    index = load_graph(path)
    checked(path, expected)
    graph_id = expected["sha256"]
    target = state / "graphs" / graph_id
    if target.exists():
        verify_files(target, read(target / "manifest.json"))
    else:
        staged = state / (".graph-" + uuid.uuid4().hex)
        staged.mkdir()
        shutil.copyfile(path, staged / "go_graph.pkl")
        checked(staged / "go_graph.pkl", expected)
        publish_directory(
            staged,
            target,
            {
                "schema_version": 1,
                "id": graph_id,
                "source": str(path),
                "files": {"go_graph.pkl": expected},
            },
        )
    return graph_id, index, target / "go_graph.pkl"


def map_from_payload(payload):
    key = {k: v for k, v in payload.items() if k != "mapping_id"}
    if payload.get("mapping_id") != identity(key):
        raise ValueError("Cluster GO map identity mismatch")
    clusters = payload.get("clusters")
    if not isinstance(clusters, dict):
        raise ValueError("Invalid cluster GO map")
    for cluster, row in clusters.items():
        if not isinstance(cluster, str) or not cluster or not isinstance(row, dict):
            raise ValueError("Invalid cluster GO map entry")
        go_set(row.get("go_mf_terms"))
    return ClusterGoMap(clusters, payload.get("reference", {}))


def snapshot_map(state, path=None):
    if path is None:
        from profunc.cluster_listing import load_cluster_records
        from profunc.runtime import resolve_runtime

        runtime = resolve_runtime()
        payload = {
            "schema_version": 1,
            "reference": runtime.manifest,
            "clusters": {
                r.cluster_id: {
                    "go_mf_terms": list(
                        go_set(r.composed_go.split(";") if r.composed_go else [])
                    )
                }
                for r in load_cluster_records(runtime.clusters)
            },
        }
        payload["mapping_id"] = identity(payload)
        if resolve_runtime().manifest != runtime.manifest:
            raise ValueError("Profunc runtime changed while reading GO labels; retry")
    else:
        payload = read(Path(path))
    map_from_payload(payload)
    target = state / "mappings" / (payload["mapping_id"] + ".json")
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        if read(target) != payload:
            raise ValueError("Conflicting saved GO map")
    else:
        write_json(target, payload)
    return payload
