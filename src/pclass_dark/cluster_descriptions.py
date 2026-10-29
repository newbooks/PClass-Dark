"""Portable, membership-bound cluster labels generated offline by profunc."""

from importlib.resources import files

from .artifacts import IntegrityError, identity, read_json, schema_checked

SCHEMA = "pclass-dark.cluster-descriptions.v1"


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


@schema_checked
def validate_catalog(catalog):
    """Validate a catalog's content identity and per-cluster data structure."""
    if (
        catalog["schema"] != SCHEMA
        or catalog["status"] != "complete"
        or catalog["id"] != identity({k: v for k, v in catalog.items() if k != "id"})
    ):
        raise IntegrityError("Cluster description catalog identity/schema mismatch")
    rows = catalog["clusters"]
    if not isinstance(rows, dict):
        raise IntegrityError("Invalid cluster description mapping")
    for cluster, row in rows.items():
        if (
            not isinstance(cluster, str)
            or not cluster
            or row["description"] is not None
            and (not isinstance(row["description"], str) or not row["description"])
            or not isinstance(row["composed_go"], list)
            or not all(isinstance(go, str) for go in row["composed_go"])
        ):
            raise IntegrityError("Invalid cluster description entry")
        for digest in (catalog["row_map_sha256"], row["C_ids_sha256"]):
            if digest is not None and (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise IntegrityError("Invalid description membership binding")
    return catalog


def load_catalog():
    """Load optional packaged labels; no reference files or GO runtimes are read."""
    resource = files("pclass_dark").joinpath("resources", "cluster_descriptions.json")
    if not resource.is_file():
        return None
    # Package resources are ordinary files in supported wheel/editable installs.
    return validate_catalog(read_json(resource))


def description_for(signature, catalog):
    """Return (full description, unavailable reason) for a matching reference fit."""
    if catalog is None:
        return None, "no_description_catalog"
    row = catalog["clusters"].get(signature["cluster_id"])
    if row is None:
        return None, "cluster_not_in_catalog"
    inputs = signature["input_identity"]
    if (
        row["C_ids_sha256"] is None
        or catalog["row_map_sha256"] is None
        or row["C_ids_sha256"] != inputs.get("C_ids_sha256")
        or catalog["row_map_sha256"] != inputs.get("row_map_sha256")
    ):
        return None, "description_reference_mismatch"
    if row["description"] is None:
        return None, "unresolved_function_description"
    return row["description"], None
