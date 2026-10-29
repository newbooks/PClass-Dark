"""Export profunc function descriptions bound to this repository's setup data.

Run from an installed/editable PClass-Dark environment. Profunc is imported only
by this explicit maintenance operation; the application never depends on it.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from pclass_dark.artifacts import (
    identity,
    read_json,
    sha256,
    transaction_guard,
    write_json,
)
from pclass_dark.cluster_descriptions import SCHEMA, clean_description, validate_catalog
from pclass_dark.embeddings import verify_embeddings
from pclass_dark.setup_data import verify_prepared


def export(profunc_repo, data_dir, output):
    source = Path(profunc_repo).resolve()
    reference = (
        Path(__file__).resolve().parents[1] / "src/pclass_dark/resources/reference"
    )
    cluster_manifest = read_json(reference / "source_manifests/clusters.json")
    dataset_manifest = read_json(reference / "source_manifests/dataset.json")
    bundle = source / "src/profunc/library/default/bundle"
    paths = {
        "membership": bundle / "clusters/clusters.tsv",
        "metadata": bundle / "clusters/clusters_meta.tsv",
        "graph": bundle / "dataset/go_graph.pkl",
    }
    records = {r["path"]: r for r in cluster_manifest["members"]}
    dataset_records = {r["path"]: r for r in dataset_manifest["members"]}
    expected = {
        "membership": records["clusters.tsv"],
        "metadata": records["clusters_meta.tsv"],
        "graph": dataset_records["go_graph.pkl"],
    }
    for key, path in paths.items():
        if (
            sha256(path) != expected[key]["sha256"]
            or path.stat().st_size != expected[key]["size"]
        ):
            raise ValueError(
                f"Source differs from bundled reference provenance: {path}"
            )
    ref = read_json(reference / "manifest.json")
    data = Path(data_dir).resolve()
    with transaction_guard(data):
        selection = read_json(data / "selection.json")
        prepared, embedding = (
            data / selection["prepared"],
            data / selection["embedding"],
        )
        if (
            sha256(prepared / "manifest.json") != selection["prepared_manifest_sha256"]
            or sha256(embedding / "manifest.json")
            != selection["embedding_manifest_sha256"]
        ):
            raise ValueError("Setup selection manifest mismatch")
        pm, windows, memberships = verify_prepared(
            prepared, selection["preparation_id"]
        )
        for name in ("clusters.tsv", "accession.fasta"):
            if pm["reference"]["files"][name]["sha256"] != ref["files"][name]["sha256"]:
                raise ValueError("Prepared reference differs from bundled snapshot")
        em = verify_embeddings(embedding, windows)
        if (
            em["id"] != selection["embedding_id"]
            or em["inventory_id"] != pm["inventory_id"]
            or selection["inventory_id"] != pm["inventory_id"]
        ):
            raise ValueError("Embedding/setup inventory mismatch")
        by_parent = defaultdict(list)
        for window in windows:
            by_parent[window["sequence_id"]].append(window["window_id"])
        by_cluster = defaultdict(set)
        for row in memberships:
            by_cluster[row["cluster_id"]].update(by_parent[row["sequence_id"]])
        c_digests = {c: identity(sorted(ids)) for c, ids in by_cluster.items()}
        row_map_digest = em["files"]["rows.tsv"]["sha256"]
    sys.path.insert(0, str(source / "src"))
    import profunc.go.description as description_module
    from profunc.cli.inspect import CLUSTER_DESCRIPTION_MAX_MODIFIERS
    from profunc.cluster_listing import load_cluster_records
    from profunc.go.description import describe_go_function
    from profunc.go.inspect import _load_go_graph

    if source not in Path(description_module.__file__).resolve().parents:
        raise ValueError("Imported profunc module is outside the requested checkout")
    graph = _load_go_graph(paths["graph"])
    clusters = {}
    for row in load_cluster_records(bundle / "clusters"):
        terms = row.composed_go.split(";") if row.composed_go else []
        result = describe_go_function(
            row.cluster_id,
            terms,
            graph["terms"],
            max_modifiers=CLUSTER_DESCRIPTION_MAX_MODIFIERS,
        )
        clusters[row.cluster_id] = {
            "description": clean_description(row.cluster_id, result.summary)
            if result.head
            else None,
            "composed_go": terms,
            "C_ids_sha256": c_digests.get(row.cluster_id),
        }
    catalog = {
        "schema": SCHEMA,
        "status": "complete",
        "row_map_sha256": row_map_digest,
        "source": {
            "function": "profunc.go.description.describe_go_function",
            "max_modifiers": CLUSTER_DESCRIPTION_MAX_MODIFIERS,
            "files": {
                k: {"sha256": expected[k]["sha256"], "size_bytes": expected[k]["size"]}
                for k in paths
            },
            "code": {
                name: sha256(source / "src/profunc/go" / name)
                for name in ("description.py", "semantic_roles.py", "build.py")
            },
            "reference_membership_sha256": ref["files"]["clusters.tsv"]["sha256"],
            "reference_fasta_sha256": ref["files"]["accession.fasta"]["sha256"],
        },
        "clusters": dict(sorted(clusters.items())),
    }
    catalog["source"]["cli_inspect_sha256"] = sha256(
        source / "src/profunc/cli/inspect.py"
    )
    catalog["id"] = identity(catalog)
    validate_catalog(catalog)
    output = Path(output)
    write_json(output, catalog)
    output.chmod(0o644)
    print(
        f"Exported {len(clusters)} descriptions; {len(c_digests)} membership bindings"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profunc-repo", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.profunc_repo, args.data_dir, args.output)
