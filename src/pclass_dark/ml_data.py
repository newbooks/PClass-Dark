"""Validated sequence-level SIAC cohorts and medoid similarity matrices (SPEC-0022)."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .artifacts import IntegrityError
from .feature_store import identifier
from .ml_progress import Progress
from .structural_features import FEATURES as SOURCE_FEATURES
from .structural_features import HEADER

LOG = logging.getLogger(__name__)

FEATURES = SOURCE_FEATURES[:-1]
PAIR_HEADER = (
    "cluster_A",
    "Medoid_A",
    "Cluster_B",
    "Medoid_B",
    "Distance",
    "Similarity",
)
INDEX_HEADER = ("accession_id", "sequence_id", "length", "membership_count")
ALIAS_HEADER = (
    "example_id",
    "accession_id",
    "representative_accession",
    "sequence_id",
    "cluster_id",
    "source_row_count",
)
EXCLUSION_HEADER = (
    "example_id",
    "accession_id",
    "cluster_id",
    "reason",
    "source_row_count",
)


def readonly(array):
    """Return an array with mutation disabled."""
    array = np.asarray(array)
    array.flags.writeable = False
    return array


def rows(path, header):
    """Yield exact-width TSV rows, rejecting malformed headers and empty cells."""
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if next(reader, None) != list(header):
            raise IntegrityError(f"Invalid header: {path}")
        for line, row in enumerate(reader, 2):
            if len(row) != len(header) or any(not x for x in row):
                raise IntegrityError(f"Invalid TSV row {line}: {path}")
            yield row


@dataclass(frozen=True)
class MLCohort:
    example_ids: tuple[str, ...]
    representatives: tuple[str, ...]
    aliases: tuple[tuple, ...]
    exclusions: tuple[tuple, ...]
    X: np.ndarray
    y: np.ndarray
    labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    source_rows: int
    conflicts: tuple[tuple, ...]
    sequence_verified: bool
    prefilter_counts: dict = field(default_factory=dict)


class LabelConflictError(IntegrityError):
    """Single-label input conflicts; full audit is available as ``conflicts``."""

    def __init__(self, conflicts):
        self.conflicts = tuple(conflicts)
        super().__init__(
            f"Conflicting labels for {len({r[0] for r in conflicts})} identities; "
            "multiple cluster_id labels for the same sequence. "
            "Correct the labels or explicitly use --conflict-policy exclude "
            "(Python: conflict_policy='exclude')"
        )


@dataclass(frozen=True)
class SimilarityData:
    labels: tuple[str, ...]
    S: np.ndarray
    medoids: tuple[str, ...]
    extra_labels: tuple[str, ...]
    pair_count: int


def _numbers(values):
    parsed = tuple(None if v == "NA" else float(v) for v in values)
    if any(v is not None and not np.isfinite(v) for v in parsed):
        raise IntegrityError("Nonfinite feature value")
    # Rounded exported ratios need not reconcile exactly.
    fractions = (*range(1, 10), 11, 12, *range(13, 17))
    for i in fractions:
        if parsed[i] is not None and not -0.001 <= parsed[i] <= 1.001:
            raise IntegrityError("Feature fraction outside domain")
    if parsed[0] is not None and parsed[0] < -0.001:
        raise IntegrityError("Negative log length")
    if parsed[10] is not None and not -1.001 <= parsed[10] <= 1.001:
        raise IntegrityError("Charge balance outside domain")
    if parsed[17] is not None and not -4.501 <= parsed[17] <= 4.501:
        raise IntegrityError("Hydropathy outside domain")
    return parsed


def load_cohort(features, *, sequence_index=None, conflict_policy="error"):
    """Read/deduplicate a single-label cohort; raise IntegrityError on invalid data."""
    if conflict_policy not in ("error", "exclude"):
        raise ValueError("Invalid conflict policy")
    accessions, count = {}, 0
    progress = Progress(LOG, "Reading feature table")
    for row in rows(features, HEADER):
        accession, label = row[0], row[-1]
        identifier(accession)
        identifier(label)
        try:
            values = _numbers(row[1:-1])
        except ValueError as exc:
            raise IntegrityError(f"Invalid features for {accession}: {exc}") from exc
        if accession not in accessions:
            accessions[accession] = (values, {})
        existing, labels = accessions[accession]
        if existing != values:
            raise IntegrityError(f"Conflicting feature vectors: {accession}")
        labels[label] = labels.get(label, 0) + 1
        count += 1
        if count % 10000 == 0:
            progress.update(count)
    progress.finish(count)
    index = {}
    if sequence_index is not None:
        import re

        progress = Progress(LOG, "Reading sequence identity index")
        for accession, sid, length, members in rows(sequence_index, INDEX_HEADER):
            if (
                accession in index
                or not re.fullmatch(r"[a-f0-9]{64}", sid)
                or not length.isdecimal()
                or int(length) < 1
                or not members.isdecimal()
                or int(members) < 1
            ):
                raise IntegrityError("Invalid/duplicate sequence index record")
            if accession not in accessions or len(accessions[accession][1]) != int(
                members
            ):
                raise IntegrityError("Sequence index membership mismatch")
            index[accession] = (sid, int(length))
            if len(index) % 10000 == 0:
                progress.update(len(index))
        progress.finish(len(index))
        if set(index) != set(accessions):
            raise IntegrityError("Sequence index coverage mismatch")
    identities = {}
    for accession in sorted(accessions):
        eid = f"sha256:{index[accession][0]}" if index else f"accession:{accession}"
        identities.setdefault(eid, []).append(accession)
    conflicts, excluded, aliases, examples, vectors, targets = [], [], [], [], [], []
    representatives = []
    for eid, members in sorted(identities.items()):
        first = accessions[members[0]][0]
        if any(accessions[a][0] != first for a in members):
            raise IntegrityError(f"Hash aliases have different features: {eid}")
        if index and len({index[a][1] for a in members}) != 1:
            raise IntegrityError(f"Hash aliases have different lengths: {eid}")
        labels = sorted({label for a in members for label in accessions[a][1]})
        if len(labels) > 1:
            group_conflicts = []
            for a in members:
                for label, n in sorted(accessions[a][1].items()):
                    group_conflicts.append((eid, a, label, "conflicting_labels", n))
            conflicts.extend(group_conflicts)
            excluded.extend(group_conflicts)
            continue
        examples.append(eid)
        representatives.append(members[0])
        targets.append(labels[0])
        vectors.append([np.nan if v is None else v for v in first[:-1]])
        for a in members:
            aliases.append(
                (
                    eid,
                    a,
                    members[0],
                    index[a][0] if index else None,
                    labels[0],
                    sum(accessions[a][1].values()),
                )
            )
    if conflicts and conflict_policy == "error":
        LOG.error(
            "Conflicting cluster labels for %s sequence identities; "
            "fitting has not started",
            f"{len({r[0] for r in conflicts}):,}",
        )
        raise LabelConflictError(conflicts)
    excluded_rows = sum(row[-1] for row in excluded)
    retained_source_rows = sum(row[-1] for row in aliases)
    prefilter_counts = {
        "source_rows": count,
        "source_accessions": len(accessions),
        "unique_identities": len(identities),
        "repeated_identity_rows": count - len(identities),
        "same_target_rows_collapsed": retained_source_rows - len(examples),
        "ambiguous_identities_removed": len({r[0] for r in excluded}),
        "ambiguous_source_rows_removed": excluded_rows,
        "total_rows_removed": count - len(examples),
        "retained_sequences": len(examples),
        "identity_basis": "sequence_hash" if index else "accession_only",
    }
    if (
        count - len(examples)
        != prefilter_counts["same_target_rows_collapsed"] + excluded_rows
    ):
        raise IntegrityError("Prefilter removal counts do not reconcile")
    LOG.info(
        "Sequence prefilter: %s feature rows, %s unique identities (%s)",
        f"{count:,}",
        f"{len(identities):,}",
        prefilter_counts["identity_basis"],
    )
    LOG.info(
        "Sequence prefilter removed %s ambiguous sequences (%s source rows); "
        "collapsed %s same-target duplicate rows; retained %s sequences",
        f"{prefilter_counts['ambiguous_identities_removed']:,}",
        f"{excluded_rows:,}",
        f"{prefilter_counts['same_target_rows_collapsed']:,}",
        f"{len(examples):,}",
    )
    if not index:
        LOG.warning(
            "No sequence hash index: duplicate accessions are filtered, "
            "but cross-accession identical sequences cannot be verified"
        )
    labels = tuple(sorted(set(targets)))
    if len(labels) < 2:
        raise IntegrityError("At least two retained labels required")
    mapping = {label: i for i, label in enumerate(labels)}
    return MLCohort(
        tuple(examples),
        tuple(representatives),
        tuple(aliases),
        tuple(excluded),
        readonly(np.asarray(vectors, dtype=np.float64)),
        readonly(np.array([mapping[t] for t in targets], dtype=np.int64)),
        labels,
        FEATURES,
        count,
        tuple(conflicts),
        bool(index),
        prefilter_counts,
    )


def load_similarity(path, *, labels, memory_budget_bytes=8 * 1024**3):
    """Read medoid pair similarities with bounded duplicate detection memory."""
    labels = tuple(labels)
    if labels != tuple(sorted(set(labels))) or len(labels) < 2:
        raise ValueError("Labels must be sorted, unique and contain at least two IDs")
    medoids, count = {}, 0
    progress = Progress(LOG, "Similarity pass 1/2: validating pairs")
    for a, ma, b, mb, d, s in rows(path, PAIR_HEADER):
        for c, m in ((a, ma), (b, mb)):
            identifier(c)
            identifier(m)
            if c in medoids and medoids[c] != m:
                raise IntegrityError(f"Inconsistent medoid: {c}")
            medoids[c] = m
        try:
            d, s = float(d), float(s)
        except ValueError as exc:
            raise IntegrityError("Invalid similarity number") from exc
        if (
            a == b
            or not np.isfinite([d, s]).all()
            or not (0 <= d <= 1 and 0 <= s <= 1)
            or abs(d + s - 1) > 1.00000001e-5
        ):
            raise IntegrityError("Invalid similarity pair/score")
        count += 1
        if count % 10000 == 0:
            progress.update(count)
    progress.finish(count)
    if set(labels) - medoids.keys():
        raise IntegrityError("Retained label absent from similarity table")
    all_ids = {c: i for i, c in enumerate(sorted(medoids))}
    n = len(all_ids)
    bit_bytes = (n * (n - 1) // 2 + 7) // 8
    if bit_bytes + len(labels) ** 2 * 8 > memory_budget_bytes:
        raise ValueError("Similarity arrays exceed memory budget")
    seen = np.zeros(bit_bytes, dtype=np.uint8)
    S = np.full((len(labels), len(labels)), np.nan, dtype=np.float64)
    np.fill_diagonal(S, 1)
    local = {label: i for i, label in enumerate(labels)}
    progress = Progress(LOG, "Similarity pass 2/2: aligning target labels", total=count)
    for processed, (a, _, b, _, _, value) in enumerate(rows(path, PAIR_HEADER), 1):
        i, j = sorted((all_ids[a], all_ids[b]))
        bit = i * (2 * n - i - 1) // 2 + j - i - 1
        byte, shift = divmod(bit, 8)
        if seen[byte] & (1 << shift):
            raise IntegrityError(f"Duplicate unordered similarity pair: {a}, {b}")
        seen[byte] |= 1 << shift
        if a in local and b in local:
            S[local[a], local[b]] = S[local[b], local[a]] = float(value)
        if processed % 10000 == 0:
            progress.update(processed)
    progress.finish(count)
    if np.isnan(S).any():
        raise IntegrityError("Incomplete retained-label similarity coverage")
    return SimilarityData(
        labels,
        readonly(S),
        tuple(medoids[c] for c in labels),
        tuple(sorted(set(medoids) - set(labels))),
        count,
    )


def load_groups(cohort, path=None, *, split_mode="homology"):
    """Return aligned group IDs, with alias consistency and full retained coverage."""
    if split_mode == "accession":
        if path is not None:
            raise ValueError("Accession mode rejects a homology group map")
        return tuple(cohort.example_ids)
    if split_mode != "homology" or path is None:
        raise ValueError("Homology mode requires a group map")
    mapping = {}
    for accession, group in rows(path, ("accession_id", "group_id")):
        identifier(accession)
        identifier(group)
        if accession in mapping:
            raise IntegrityError("Duplicate group-map accession")
        mapping[accession] = group
    grouped = {}
    for eid, accession, *_ in cohort.aliases:
        if accession not in mapping:
            raise IntegrityError(f"Missing group: {accession}")
        if eid in grouped and grouped[eid] != mapping[accession]:
            raise IntegrityError("Exact-sequence aliases have different groups")
        grouped[eid] = mapping[accession]
    return tuple(grouped[e] for e in cohort.example_ids)
