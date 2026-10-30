"""Reference-bound live cluster descriptions from profunc."""

from .artifacts import IntegrityError, identity


def clean_description(cluster_id, summary):
    """Remove the exact profunc sentence prefix/article; retain function wording."""
    prefix = cluster_id + " is annotated as "
    if not summary.startswith(prefix):
        raise ValueError("Unexpected profunc description prefix")
    text = summary[len(prefix) :]
    for article in ("an ", "a "):
        if text.startswith(article):
            text = text[len(article) :]
            break
    return text.removesuffix(".")


def load_catalog(cluster_ids=None):
    """Describe current profunc clusters through its validated runtime modules.

    Raises the runtime's OSError/ValueError/RuntimeError on missing or invalid
    source data. The result is operation-local; no static catalog is consulted.
    """
    from profunc.cli.inspect import CLUSTER_DESCRIPTION_MAX_MODIFIERS
    from profunc.cluster_listing import load_cluster_records
    from profunc.go.description import describe_go_function
    from profunc.go.inspect import _load_go_graph

    from .reference_data import profunc_reference_contract, resolve_profunc

    runtime = resolve_profunc()
    reference = profunc_reference_contract(runtime)
    records = load_cluster_records(runtime.clusters)
    if cluster_ids is not None:
        selected = set(cluster_ids)
        records = [r for r in records if r.cluster_id in selected]
    graph = _load_go_graph(runtime.dataset / "go_graph.pkl") if records else None
    clusters = {}
    for row in records:
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
        }
    # Detect concurrent runtime replacement before returning labels.
    after = resolve_profunc()
    if (
        after.manifest != runtime.manifest
        or profunc_reference_contract(after) != reference
    ):
        raise IntegrityError("Profunc library changed during description lookup; retry")
    catalog = {"reference": reference, "clusters": clusters}
    return {**catalog, "id": identity(catalog)}


def description_for(signature, catalog):
    """Attach a live label only to a signature with the same source reference."""
    if catalog is None:
        return None, "no_description_catalog"
    reference = signature["input_identity"].get("reference")
    if reference is None:
        return None, "missing_profunc_reference_binding"
    if reference != catalog["reference"]:
        return None, "description_reference_mismatch"
    row = catalog["clusters"].get(signature["cluster_id"])
    if row is None:
        return None, "cluster_not_in_catalog"
    if row["description"] is None:
        return None, "unresolved_function_description"
    return row["description"], None
