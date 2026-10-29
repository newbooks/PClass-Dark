"""Portable published cluster paths and explicit legacy-layout reads."""

import re

from .artifacts import IntegrityError

LAYOUT = "cluster-id-v1"
FIELD = "cluster_directory_layout"


def cluster_directory_name(cluster):
    """Return a safe readable directory component, or raise ValueError."""
    reserved = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }
    if (
        not isinstance(cluster, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,98}", cluster)
        or cluster.endswith(".")
        or cluster.split(".")[0].upper() in reserved
    ):
        raise ValueError(f"Unsafe cluster directory ID: {cluster!r}")
    return cluster


def legacy_cluster_name(cluster):
    result = "c_" + cluster.encode("utf-8").hex()
    if len(result) > 200:
        raise ValueError(f"Encoded cluster ID exceeds 200 bytes: {cluster}")
    return result


def stored_cluster_name(cluster, manifest):
    layout = manifest.get(FIELD, "hex-v1")
    if layout == LAYOUT:
        return cluster_directory_name(cluster)
    if layout == "hex-v1":
        return legacy_cluster_name(cluster)
    raise IntegrityError(f"Unknown cluster directory layout: {layout}")


def validate_cluster_directories(clusters):
    names = [cluster_directory_name(c) for c in clusters]
    if len({c.casefold() for c in names}) != len(names):
        raise ValueError("Cluster directory IDs collide after case folding")
