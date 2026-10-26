"""Offline preparation, per-cluster folds/backgrounds, and embedding orchestration."""

from __future__ import annotations

import csv
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .artifacts import (
    IntegrityError,
    file_record,
    guarded_bundle_reader,
    identity,
    locked,
    publication_transaction,
    publish,
    schema_checked,
    sha256,
    staging,
    verify_bundle,
    write_json,
)
from .cluster_sets import (
    B_SCHEMA,
    C_SCHEMA,
    CAP,
    ClusterSets,
    build_selections,
    pair_payload,
    validate_sets,
)
from .reference_data import load_reference, read_fasta, reference_root, sequence_id

LOG = logging.getLogger(__name__)


def reference_identity(reference):
    return {
        n: {k: r[k] for k in ("sha256", "size_bytes")}
        for n, r in reference["files"].items()
    }


PREP_SCHEMA = "pclass-dark.prepared.v1"
MODEL = "facebook/esm2_t30_150M_UR50D"
REVISION = "a695f6045e2e32885fa60af20c13cb35398ce30c"
WINDOW_LENGTH = 1022
HEADERS = {
    "aliases.tsv": (
        "accession original_cluster_id sequence_id representative "
        "retained_cluster retained_sequence exclusion_reason"
    ),
    "memberships.tsv": "cluster_id sequence_id representative",
    "clusters.tsv": (
        "cluster_id original_accessions unique_sequences retained exclusion_reason"
    ),
    "cluster_folds.tsv": (
        "cluster_id fold_id validation_windows training_windows "
        "validation_parents training_parents status"
    ),
    "folds.tsv": "cluster_id window_id sequence_id representative fold_id",
    "background_summary.tsv": (
        "cluster_id source_contributions unique_union_count "
        "target_excluded_count background_count background_ids_sha256 "
        "status"
    ),
    "windows.tsv": (
        "window_id sequence_id representative parent_length kind start "
        "end length window_sha256"
    ),
    "window_identity.tsv": (
        "window_sha256 window_id sequence_id cluster_id fold_id shared_across_folds"
    ),
}
HEADERS = {k: v.split() for k, v in HEADERS.items()}


@dataclass(frozen=True)
class SetupResult:
    preparation_id: str
    prepared_dir: Path
    embedding_id: str | None
    embedding_dir: Path | None
    counts: dict
    reused_prepared: bool
    reused_embeddings: bool


def write_tsv(path, rows):
    name = Path(path).name
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, HEADERS[name], delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: str(v).lower() if isinstance(v, bool) else v
                    for k, v in row.items()
                }
            )


def read_tsv(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != HEADERS[Path(path).name]:
            raise IntegrityError(f"Invalid table header: {path}")
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise IntegrityError(f"Malformed table: {path}")
    return rows


def write_fasta(path, records):
    with Path(path).open("w", encoding="ascii", newline="") as handle:
        for accession, sequence in records:
            handle.write(f">{accession}\n")
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def window_records(accession, sequence):
    length = len(sequence)
    if length <= WINDOW_LENGTH:
        return [(accession, sequence, "full", 1, length)]
    return [
        (f"N-{accession}", sequence[:WINDOW_LENGTH], "N", 1, WINDOW_LENGTH),
        (
            f"C-{accession}",
            sequence[-WINDOW_LENGTH:],
            "C",
            length - WINDOW_LENGTH + 1,
            length,
        ),
    ]


def inventory_id(windows):
    return identity(
        [[w["window_id"], w["sequence_id"], w["window_sha256"]] for w in windows]
    )


def _bins(counts):
    counts = list(counts)
    boundaries = [0, 10, 20, 40, 80, 160, 320, 640, float("inf")]
    return [
        sum(lo <= n < hi for n in counts)
        for lo, hi in zip(boundaries, boundaries[1:], strict=False)
    ]


def _create_prepared(
    stage, prep_id, configuration, ref, members, sequences, noncanonical
):
    LOG.info("Grouping %s clustered accessions by exact sequence", len(members))
    grouped = {}
    by_accession = {}
    original = Counter(members.values())
    for accession in sorted(members):
        sequence = sequences[accession]
        sid = sequence_id(sequence)
        if sid in grouped and grouped[sid]["sequence"] != sequence:
            raise IntegrityError("Full sequence SHA-256 collision")
        if sid not in grouped:
            grouped[sid] = {
                "representative": accession,
                "sequence": sequence,
                "clusters": set(),
            }
        grouped[sid]["clusters"].add(members[accession])
        by_accession[accession] = sid
    cluster_sequences = defaultdict(set)
    for sid, entry in grouped.items():
        for cluster in entry["clusters"]:
            cluster_sequences[cluster].add(sid)
    retained = {
        c
        for c, ids in cluster_sequences.items()
        if len(ids) >= configuration["min_clusters"]
    }
    if not retained:
        raise ValueError("No clusters remain after deduplication and minimum filtering")
    retained_ids = set().union(*(cluster_sequences[c] for c in retained))
    LOG.info(
        "Retained %s clusters and %s distinct full sequences",
        len(retained),
        len(retained_ids),
    )
    aliases = []
    for accession in sorted(members):
        sid, cluster = by_accession[accession], members[accession]
        aliases.append(
            dict(
                accession=accession,
                original_cluster_id=cluster,
                sequence_id=sid,
                representative=grouped[sid]["representative"],
                retained_cluster=cluster in retained,
                retained_sequence=sid in retained_ids,
                exclusion_reason="" if cluster in retained else "cluster_below_minimum",
            )
        )
    membership_rows = [
        dict(
            cluster_id=c, sequence_id=sid, representative=grouped[sid]["representative"]
        )
        for c in sorted(retained)
        for sid in sorted(cluster_sequences[c])
    ]
    cluster_rows = [
        dict(
            cluster_id=c,
            original_accessions=original[c],
            unique_sequences=len(ids),
            retained=c in retained,
            exclusion_reason="" if c in retained else "cluster_below_minimum",
        )
        for c, ids in sorted(cluster_sequences.items())
    ]
    windows, window_sequences = [], {}
    representatives = sorted(
        (grouped[sid]["representative"], sid) for sid in retained_ids
    )
    omitted = 0
    covered_residues = omitted_residues = 0
    for accession, sid in representatives:
        seq = grouped[sid]["sequence"]
        missing = max(0, len(seq) - 2 * WINDOW_LENGTH)
        omitted += bool(missing)
        omitted_residues += missing
        covered_residues += len(seq) - missing
        for wid, wseq, kind, start, end in window_records(accession, seq):
            if wid in window_sequences:
                raise IntegrityError(f"Generated window ID collision: {wid}")
            window_sequences[wid] = wseq
            windows.append(
                dict(
                    window_id=wid,
                    sequence_id=sid,
                    representative=accession,
                    parent_length=len(seq),
                    kind=kind,
                    start=start,
                    end=end,
                    length=len(wseq),
                    window_sha256=sequence_id(wseq),
                )
            )
    windows.sort(key=lambda w: w["window_id"])
    inv_id = inventory_id(windows)
    parents = {w["window_id"]: w["sequence_id"] for w in windows}
    parent_windows = defaultdict(list)
    for w in windows:
        parent_windows[w["sequence_id"]].append(w["window_id"])
    clusters = {
        c: sorted(w for sid in cluster_sequences[c] for w in parent_windows[sid])
        for c in sorted(retained)
    }
    LOG.info("Selecting per-cluster folds and capped background sources")
    selections = build_selections(
        clusters, configuration["seed"], configuration["workers"]
    )
    samples = {c: selections[c][1] for c in clusters}
    sets = ClusterSets(clusters, samples, parents)
    fold_rows, fold_counts, diagnostics = [], [], []
    digest_windows = defaultdict(list)
    for w in windows:
        digest_windows[w["window_sha256"]].append(w["window_id"])
    # Verify digest groups against actual window strings.
    for group in digest_windows.values():
        if any(window_sequences[w] != window_sequences[group[0]] for w in group):
            raise IntegrityError("Window SHA-256 collision")
    window_map = {w["window_id"]: w for w in windows}
    sibling_crossings = identical_crossings = 0
    for c in clusters:
        assignments = selections[c][0]
        per_parent = defaultdict(set)
        digest_folds = defaultdict(set)
        for wid in clusters[c]:
            w = window_map[wid]
            fold = assignments[wid]
            fold_rows.append(
                dict(
                    cluster_id=c,
                    window_id=wid,
                    sequence_id=w["sequence_id"],
                    representative=w["representative"],
                    fold_id=fold,
                )
            )
            per_parent[w["sequence_id"]].add(fold)
            digest_folds[w["window_sha256"]].add(fold)
        sibling_crossings += sum(len(v) > 1 for v in per_parent.values())
        identical_crossings += sum(len(v) > 1 for v in digest_folds.values())
        for fold in range(5):
            validation = [w for w in clusters[c] if assignments[w] == fold]
            training = [w for w in clusters[c] if assignments[w] != fold]
            status = (
                "insufficient_training"
                if len(training) < 2
                else "empty_validation"
                if not validation
                else "ok"
            )
            fold_counts.append(
                dict(
                    cluster_id=c,
                    fold_id=fold,
                    validation_windows=len(validation),
                    training_windows=len(training),
                    validation_parents=len({parents[w] for w in validation}),
                    training_parents=len({parents[w] for w in training}),
                    status=status,
                )
            )
        for wid in clusters[c]:
            digest = window_map[wid]["window_sha256"]
            if len(digest_windows[digest]) > 1:
                diagnostics.append(
                    dict(
                        window_sha256=digest,
                        window_id=wid,
                        sequence_id=parents[wid],
                        cluster_id=c,
                        fold_id=assignments[wid],
                        shared_across_folds=len(digest_folds[digest]) > 1,
                    )
                )
    diagnostics.sort(
        key=lambda r: (r["window_sha256"], r["cluster_id"], r["window_id"])
    )
    for name, rows in [
        ("aliases.tsv", aliases),
        ("memberships.tsv", membership_rows),
        ("clusters.tsv", cluster_rows),
        ("windows.tsv", windows),
        ("folds.tsv", fold_rows),
        ("cluster_folds.tsv", fold_counts),
        ("window_identity.tsv", diagnostics),
    ]:
        write_tsv(stage / name, rows)
    write_fasta(
        stage / "sequences.fasta",
        [(a, grouped[sid]["sequence"]) for a, sid in representatives],
    )
    write_fasta(stage / "windows.fasta", sorted(window_sequences.items()))
    c = {
        "schema": C_SCHEMA,
        "preparation_id": prep_id,
        "inventory_id": inv_id,
        "reference_id": identity(reference_identity(ref)),
        "clusters": clusters,
        "input_files": {
            n: file_record(stage / n) for n in ("windows.tsv", "memberships.tsv")
        },
        "counts": {
            "clusters": len(clusters),
            "memberships": sum(map(len, clusters.values())),
        },
    }
    write_json(stage / "cluster_members.json", c)
    payload = pair_payload(
        c, sha256(stage / "cluster_members.json"), samples, configuration["seed"]
    )
    summaries = []
    for index, cluster in enumerate(clusters):
        summaries.append(sets.summary(cluster))
        if index % 100 == 0:
            LOG.info("Background identities: %s/%s targets", index + 1, len(clusters))
    write_tsv(stage / "background_summary.tsv", summaries)
    b = {
        "schema": B_SCHEMA,
        **payload,
        "pair_id": identity(payload),
        "backgrounds": {row["cluster_id"]: row for row in summaries},
    }
    write_json(stage / "background_members.json", b)
    counts = {
        "source_accessions": len(members),
        "source_clusters": len(original),
        "retained_clusters": len(retained),
        "excluded_clusters": len(original) - len(retained),
        "retained_sequences": len(retained_ids),
        "retained_memberships": len(membership_rows),
        "windows": len(windows),
        "proteins_with_omitted_centers": omitted,
        "covered_residues": covered_residues,
        "omitted_residues": omitted_residues,
        "cluster_parent_cross_fold_cases": sibling_crossings,
        "cluster_identical_window_cross_fold_cases": identical_crossings,
        "original_size_bins": _bins(original.values()),
        "deduplicated_size_bins": _bins(map(len, cluster_sequences.values())),
        "noncanonical_accessions": noncanonical,
    }
    manifest = {
        "schema": PREP_SCHEMA,
        "status": "complete",
        "id": prep_id,
        "configuration": {k: v for k, v in configuration.items() if k != "workers"},
        "reference": ref,
        "inventory_id": inv_id,
        "pair_id": b["pair_id"],
        "counts": counts,
        "software_version": __version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {p.name: file_record(p) for p in sorted(stage.iterdir())},
    }
    write_json(stage / "manifest.json", manifest)
    return manifest


@guarded_bundle_reader
@schema_checked
def verify_prepared(path, expected_id=None, *, immutable=False):
    """Validate a complete portable prepared bundle and inventory joins."""
    path = Path(path)
    m = verify_bundle(path, PREP_SCHEMA, expected_id)
    configuration = m["configuration"]
    expected = identity(
        {
            "reference": reference_identity(m["reference"]),
            "configuration": configuration,
        }
    )
    if m["id"] != expected:
        raise IntegrityError("Prepared identity/configuration mismatch")
    windows = read_tsv(path / "windows.tsv")
    memberships = read_tsv(path / "memberships.tsv")
    ids = [w["window_id"] for w in windows]
    if ids != sorted(set(ids)) or inventory_id(windows) != m["inventory_id"]:
        raise IntegrityError("Window inventory/order mismatch")
    seqs = dict(read_fasta(path / "sequences.fasta"))
    wseqs = dict(read_fasta(path / "windows.fasta"))
    if set(wseqs) != set(ids):
        raise IntegrityError("Window FASTA coverage mismatch")
    by_parent = defaultdict(list)
    for w in windows:
        seq = seqs.get(w["representative"])
        if seq is None or sequence_id(seq) != w["sequence_id"]:
            raise IntegrityError("Parent sequence mismatch")
        valid = window_records(w["representative"], seq)
        match = next((x for x in valid if x[0] == w["window_id"]), None)
        if (
            match is None
            or match[1] != wseqs[w["window_id"]]
            or sequence_id(match[1]) != w["window_sha256"]
            or int(w["length"]) != len(match[1])
            or int(w["parent_length"]) != len(seq)
            or (w["kind"], int(w["start"]), int(w["end"])) != match[2:]
        ):
            raise IntegrityError("Window sequence/coordinate mismatch")
        by_parent[w["sequence_id"]].append(w["window_id"])
    for accession, seq in seqs.items():
        if set(by_parent[sequence_id(seq)]) != {
            w[0] for w in window_records(accession, seq)
        }:
            raise IntegrityError("Incomplete parent window coverage")
    seen = set()
    for row in memberships:
        key = row["cluster_id"], row["sequence_id"]
        if key in seen or row["sequence_id"] not in by_parent:
            raise IntegrityError("Invalid retained membership join")
        seen.add(key)
    folds = read_tsv(path / "folds.tsv")
    expected_folds = {
        (row["cluster_id"], w)
        for row in memberships
        for w in by_parent[row["sequence_id"]]
    }
    keys = [(r["cluster_id"], r["window_id"]) for r in folds]
    if len(set(keys)) != len(keys) or set(keys) != expected_folds:
        raise IntegrityError("Fold assignment coverage mismatch")
    from .cluster_sets import fold_assignments

    grouped = defaultdict(list)
    for row in folds:
        grouped[row["cluster_id"]].append(row["window_id"])
    expected_assignments = {
        c: fold_assignments(c, w, configuration["seed"]) for c, w in grouped.items()
    }
    window_map = {w["window_id"]: w for w in windows}
    for row in folds:
        w = window_map[row["window_id"]]
        if (
            type(row["fold_id"]) is not str
            or row["fold_id"] not in {"0", "1", "2", "3", "4"}
            or int(row["fold_id"])
            != expected_assignments[row["cluster_id"]][row["window_id"]]
            or row["sequence_id"] != w["sequence_id"]
            or row["representative"] != w["representative"]
        ):
            raise IntegrityError("Invalid seeded fold assignment")
    cluster_rows = read_tsv(path / "clusters.tsv")
    aliases = read_tsv(path / "aliases.tsv")
    if len({r["accession"] for r in aliases}) != len(aliases):
        raise IntegrityError("Duplicate original accession aliases")
    expected_counts = {
        "source_accessions": len(aliases),
        "source_clusters": len(cluster_rows),
        "retained_clusters": sum(r["retained"] == "true" for r in cluster_rows),
        "excluded_clusters": sum(r["retained"] == "false" for r in cluster_rows),
        "retained_sequences": len(seqs),
        "retained_memberships": len(memberships),
        "windows": len(windows),
        "original_size_bins": _bins(
            int(r["original_accessions"]) for r in cluster_rows
        ),
        "deduplicated_size_bins": _bins(
            int(r["unique_sequences"]) for r in cluster_rows
        ),
    }
    if any(m["counts"].get(k) != v for k, v in expected_counts.items()):
        raise IntegrityError("Prepared observed counts mismatch")
    return m, windows, memberships


def _prepare(root, reference, min_clusters, seed, workers, force, transaction):
    LOG.info("Verifying authoritative bundled references")
    ref, members, sequences, noncanonical = load_reference(reference)
    config = {
        "min_clusters": min_clusters,
        "seed": seed,
        "folds": 5,
        "window_length": WINDOW_LENGTH,
        "background_cap": CAP,
        "numpy_version": np.__version__,
        "algorithm": "setup-v1",
        "schema": PREP_SCHEMA,
        "fold_algorithm": "sha256-round-robin-v1",
        "background_algorithm": "sha256-PCG64-choice-v1",
    }
    prep_id = identity({"reference": reference_identity(ref), "configuration": config})
    from .artifacts import safe_file

    target = safe_file(root, "prepared/" + prep_id)
    with locked(root, "prepared-" + prep_id):
        if target.exists() and not force:
            m, windows, memberships = verify_prepared(target, prep_id)
            validate_sets(target, m, windows, memberships)
            LOG.info("Reusing verified prepared bundle %s", prep_id)
            return target, m, True
        with staging(target.parent) as stage:
            m = _create_prepared(
                stage,
                prep_id,
                {**config, "workers": workers},
                ref,
                members,
                sequences,
                noncanonical,
            )
            if force and target.exists():
                try:
                    prior, _, _ = verify_prepared(target, prep_id)
                except (ValueError, OSError):
                    prior = None
                if prior is not None and all(
                    prior[k] == m[k]
                    for k in (
                        "files",
                        "counts",
                        "configuration",
                        "inventory_id",
                        "pair_id",
                    )
                ):
                    m = prior
                    write_json(stage / "manifest.json", m)
            checked, windows, memberships = verify_prepared(stage, prep_id)
            validate_sets(stage, checked, windows, memberships)
            publish(root, stage, target, force, transaction)
    return target, m, False


def _validate_options(min_clusters, seed, workers, batch_size, model, revision, device):
    for name, value in (
        ("min_clusters", min_clusters),
        ("workers", workers),
        ("batch_size", batch_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if model != MODEL or revision != REVISION:
        raise ValueError(
            "Unsupported model/revision; use the pinned ESM-2 150M checkpoint"
        )
    if not re.fullmatch(r"auto|cpu|cuda(?::[0-9]+)?", device):
        raise ValueError("device must be auto, cpu, cuda, or cuda:<index>")


def prepare_data(
    output_dir="work/setup-data",
    *,
    min_clusters=10,
    seed=42,
    workers=1,
    prepare_only=False,
    offline=False,
    model=MODEL,
    revision=REVISION,
    device="auto",
    batch_size=8,
    force=False,
    reference_dir=None,
):
    """Prepare references and optionally embed; resume matching inference shards.

    Raises ValueError/IntegrityError for bad input, SetupBusyError on contention,
    and OSError for filesystem errors. reference_dir is a local manifest-bound
    override for integrations/tests; the default always uses packaged resources.
    """
    _validate_options(min_clusters, seed, workers, batch_size, model, revision, device)
    root_input = Path(output_dir).absolute()
    if any(p.is_symlink() for p in (root_input, *root_input.parents)):
        raise ValueError("Symlink output paths are not supported")
    root = root_input.resolve()
    # Python 3.10 importlib.resources.as_file supports resource files; the
    # installed package is a filesystem tree after wheel installation.
    ref_path = (
        Path(reference_dir)
        if reference_dir is not None
        else Path(str(reference_root()))
    )
    reference = ref_path.resolve()
    package_resources = Path(__file__).resolve().parent / "resources"
    if (
        root == reference
        or root in reference.parents
        or reference in root.parents
        or root == package_resources
        or package_resources in root.parents
    ):
        raise ValueError("Output must not overlap reference or installed resources")
    root.mkdir(parents=True, exist_ok=True)
    with publication_transaction(root) as transaction:
        prepared, m, reused = _prepare(
            root, reference, min_clusters, seed, workers, force, transaction
        )
        embedding_id = embedding_dir = None
        reused_embeddings = False
        if not prepare_only:
            from .embeddings import embed_prepared

            embedding_dir, em, reused_embeddings = embed_prepared(
                root,
                prepared,
                m,
                model=model,
                revision=revision,
                offline=offline,
                device=device,
                batch_size=batch_size,
                seed=seed,
                force=force,
                transaction=transaction,
            )
            embedding_id = em["id"]
        selection = {
            "schema": "pclass-dark.selection.v1",
            "preparation_id": m["id"],
            "prepared": str(prepared.relative_to(root)),
            "prepared_manifest_sha256": sha256(prepared / "manifest.json"),
            "inventory_id": m["inventory_id"],
            "embedding_id": embedding_id,
            "embedding": str(embedding_dir.relative_to(root))
            if embedding_dir
            else None,
            "embedding_manifest_sha256": sha256(embedding_dir / "manifest.json")
            if embedding_dir
            else None,
        }
        write_json(root / "selection.json", selection)
        return SetupResult(
            m["id"],
            prepared,
            embedding_id,
            embedding_dir,
            m["counts"],
            reused,
            reused_embeddings,
        )
