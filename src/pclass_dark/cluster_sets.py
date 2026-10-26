"""Compact, checksum-bound C sets, fixed capped B sets, and per-target folds."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .artifacts import IntegrityError, canonical, identity, read_json, sha256

C_SCHEMA = "pclass-dark.cluster-members.v1"
B_SCHEMA = "pclass-dark.background-members.v1"
RULE = "union-other-sources-exclude-target-parents-v1"
CAP = 200


def fold_assignments(cluster_id, window_ids, seed):
    def key(window):
        encoded = canonical(["pclass-dark.c-folds.v1", seed, cluster_id, window])
        return hashlib.sha256(encoded).digest(), window

    return {window: rank % 5 for rank, window in enumerate(sorted(window_ids, key=key))}


def sample_source(cluster_id, windows, seed):
    if len(windows) <= CAP:
        return "all"
    encoded = canonical(["pclass-dark.background-windows.v1", seed, cluster_id])
    derived = int.from_bytes(hashlib.sha256(encoded).digest(), "big")
    rng = np.random.Generator(np.random.PCG64(derived))
    return sorted(int(i) for i in rng.choice(len(windows), size=CAP, replace=False))


def _one_cluster(task):
    cluster, windows, seed = task
    return (
        cluster,
        fold_assignments(cluster, windows, seed),
        sample_source(cluster, windows, seed),
    )


def build_selections(clusters, seed, workers=1):
    tasks = [(key, value, seed) for key, value in sorted(clusters.items())]
    if workers == 1:
        records = map(_one_cluster, tasks)
        return {c: (folds, sample) for c, folds, sample in records}
    # Submit bounded chunks to avoid copying the sequence corpus to workers.
    result = {}
    with ProcessPoolExecutor(
        max_workers=min(workers, len(tasks), os.cpu_count() or 1, 32),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        for start in range(0, len(tasks), workers * 4):
            for c, folds, sample in executor.map(
                _one_cluster, tasks[start : start + workers * 4]
            ):
                result[c] = folds, sample
    return result


class ClusterSets:
    """Validated compact definitions; background expansion is target-specific."""

    def __init__(self, clusters, samples, parents):
        self.clusters = {k: tuple(v) for k, v in sorted(clusters.items())}
        self.samples = samples
        self.parents = dict(parents)
        self.cluster_ids = tuple(self.clusters)
        self.selected = {}
        contribution_counts = Counter()
        for cluster, windows in self.clusters.items():
            sample = samples[cluster]
            selected = windows if sample == "all" else tuple(windows[i] for i in sample)
            self.selected[cluster] = selected
            contribution_counts.update(selected)
        self.pool = tuple(sorted(contribution_counts))
        self.contribution_counts = contribution_counts
        self.total_contributions = sum(map(len, self.selected.values()))
        self.positions = {window: i for i, window in enumerate(self.pool)}
        self.parent_positions = defaultdict(set)
        for window, index in self.positions.items():
            self.parent_positions[parents[window]].add(index)
        # Canonical array body supports hashing filtered sets without repeatedly
        # serializing or sorting hundreds of thousands of background IDs.
        tokens = [canonical(window) for window in self.pool]
        self.body = b",".join(tokens)
        self.starts, self.ends = [], []
        position = 0
        for token in tokens:
            self.starts.append(position)
            position += len(token)
            self.ends.append(position)
            position += 1

    def cluster_members(self, cluster):
        return self.clusters[cluster]

    def _excluded(self, cluster):
        parents = {self.parents[w] for w in self.clusters[cluster]}
        excluded = set()
        for parent in parents:
            excluded.update(self.parent_positions.get(parent, ()))
        return excluded

    def background_members(self, cluster):
        excluded = self._excluded(cluster)
        return tuple(w for i, w in enumerate(self.pool) if i not in excluded)

    def summary(self, cluster):
        excluded = sorted(self._excluded(cluster))
        digest = hashlib.sha256(b"[")
        first = True
        start = 0
        for stop in [*excluded, len(self.pool)]:
            if start < stop:
                if not first:
                    digest.update(b",")
                digest.update(
                    memoryview(self.body)[self.starts[start] : self.ends[stop - 1]]
                )
                first = False
            start = stop + 1
        digest.update(b"]")
        # Only samples unique to the excluded source disappear before removing
        # target parents; overlapping labels require set-union accounting.
        sole = sum(self.contribution_counts[w] == 1 for w in self.selected[cluster])
        union_count = len(self.pool) - sole
        count = len(self.pool) - len(excluded)
        return {
            "cluster_id": cluster,
            "source_contributions": self.total_contributions
            - len(self.selected[cluster]),
            "unique_union_count": union_count,
            "target_excluded_count": union_count - count,
            "background_count": count,
            "background_ids_sha256": digest.hexdigest(),
            "status": "ok" if count else "empty_background",
        }


def pair_payload(c, c_hash, samples, seed):
    return {
        "preparation_id": c["preparation_id"],
        "inventory_id": c["inventory_id"],
        "cluster_members_sha256": c_hash,
        "source_samples": samples,
        "cap": CAP,
        "seed": seed,
        "sampler": "sha256-PCG64-choice-v1",
        "numpy_version": np.__version__,
        "rule": RULE,
    }


def load_sets(prepared_dir, *, immutable=False):
    """Verify C/B against a snapshot; immutable releases require no write lock.

    Set immutable=True only when the bundle cannot be concurrently replaced.
    Installed package resources are immutable automatically.
    """
    from .artifacts import bundle_read_guard, locked
    from .setup_data import verify_prepared

    path = Path(prepared_dir).resolve()
    root = path.parent.parent
    package_resources = Path(__file__).resolve().parent / "resources"
    immutable = immutable or package_resources in path.parents
    context = nullcontext() if immutable else locked(root, "prepared-" + path.name)
    with bundle_read_guard(path, immutable=immutable), context:
        manifest, windows, memberships = verify_prepared(path, immutable=immutable)
        return validate_sets(path, manifest, windows, memberships)


def validate_sets(path, manifest, windows, memberships):
    c = read_json(Path(path) / "cluster_members.json")
    b = read_json(Path(path) / "background_members.json")
    try:
        if c["schema"] != C_SCHEMA or b["schema"] != B_SCHEMA:
            raise IntegrityError("Unsupported compact set schema")
        if (
            c["preparation_id"] != manifest["id"]
            or c["inventory_id"] != manifest["inventory_id"]
        ):
            raise IntegrityError("C inventory/preparation mismatch")
        for name in ("windows.tsv", "memberships.tsv"):
            if c["input_files"][name] != manifest["files"][name]:
                raise IntegrityError("C input binding mismatch")
        parents = {w["window_id"]: w["sequence_id"] for w in windows}
        parent_windows = defaultdict(list)
        for row in windows:
            parent_windows[row["sequence_id"]].append(row["window_id"])
        expected = defaultdict(set)
        for row in memberships:
            expected[row["cluster_id"]].update(parent_windows[row["sequence_id"]])
        expected = {k: sorted(v) for k, v in sorted(expected.items())}
        if list(c["clusters"]) != sorted(expected) or c["clusters"] != expected:
            raise IntegrityError("C membership/order mismatch")
        if c.get("counts") != {
            "clusters": len(expected),
            "memberships": sum(map(len, expected.values())),
        }:
            raise IntegrityError("C membership count mismatch")
        samples = b["source_samples"]
        if set(samples) != set(expected) or set(b["backgrounds"]) != set(expected):
            raise IntegrityError("B source/target coverage mismatch")
        for cluster, members in expected.items():
            selection = samples[cluster]
            if len(members) <= CAP:
                if selection != "all":
                    raise IntegrityError("Small B source must use all")
            elif (
                not isinstance(selection, list)
                or len(selection) != CAP
                or any(
                    type(i) is not int or i < 0 or i >= len(members) for i in selection
                )
                or selection != sorted(set(selection))
            ):
                raise IntegrityError("Invalid background source indices")
        payload = pair_payload(
            c,
            sha256(Path(path) / "cluster_members.json"),
            samples,
            manifest["configuration"]["seed"],
        )
        # NumPy version is frozen provenance, not the loader's installed version.
        payload["numpy_version"] = manifest["configuration"]["numpy_version"]
        if any(b.get(k) != v for k, v in payload.items()) or b["pair_id"] != identity(
            payload
        ):
            raise IntegrityError("C/B pair identity mismatch")
        if manifest["pair_id"] != b["pair_id"]:
            raise IntegrityError("Prepared C/B identity mismatch")
        sets = ClusterSets(expected, samples, parents)
        for cluster in expected:
            if b["backgrounds"][cluster] != sets.summary(cluster):
                raise IntegrityError("Background expansion count/digest mismatch")
        return sets
    except (KeyError, TypeError, IndexError) as exc:
        raise IntegrityError("Malformed compact cluster definitions") from exc
