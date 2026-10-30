"""Validate references obtained through the installed profunc library."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from .artifacts import IntegrityError, read_json, schema_checked, verify_files


def resolve_profunc():
    """Resolve and validate profunc's active library without invoking its CLI."""
    from profunc.runtime import resolve_runtime

    return resolve_runtime()


def profunc_reference_contract(runtime):
    """Return content records binding membership and dataset sequences."""
    from .artifacts import file_record

    dataset = read_json(runtime.dataset / "manifest.json")
    record = next(
        (r for r in dataset["members"] if r["path"] == "accession.fasta"), None
    )
    if record is None:
        raise IntegrityError("Profunc dataset manifest has no accession.fasta")
    clusters = read_json(runtime.clusters / "manifest.json")
    membership = next(
        (r for r in clusters["members"] if r["path"] == "clusters.tsv"), None
    )
    observed = file_record(runtime.clusters / "clusters.tsv")
    if membership is None or observed != {
        "sha256": membership["sha256"],
        "size_bytes": membership["size"],
    }:
        raise IntegrityError("Profunc membership differs from its cluster manifest")
    return {
        "clusters.tsv": observed,
        "accession.fasta": {"sha256": record["sha256"], "size_bytes": record["size"]},
    }


def profunc_fasta_path(runtime, reference_fasta=None):
    """Locate the source dataset FASTA without downloading or copying it."""
    from profunc.workspace import default_workspace

    dataset = read_json(runtime.dataset / "manifest.json")
    fasta = (
        Path(reference_fasta)
        if reference_fasta is not None
        else (runtime.dataset / "accession.fasta")
    )
    if reference_fasta is None and not fasta.is_file():
        fasta = (
            default_workspace()
            / "datasets"
            / dataset["generation_id"]
            / "accession.fasta"
        )
    if not fasta.is_file():
        raise FileNotFoundError(
            "Profunc runtime omits sequence FASTA. Supply the source dataset's "
            "accession.fasta with --reference-fasta (Python: reference_fasta)."
        )
    return fasta.resolve()


def load_profunc_reference(runtime, reference_fasta=None):
    """Read the active membership and its checksum-matched source FASTA.

    Raises IntegrityError for mismatched input and FileNotFoundError when the
    original dataset FASTA is unavailable. Never fetches sequences from a network.
    """
    from .artifacts import file_record

    records = profunc_reference_contract(runtime)
    fasta = profunc_fasta_path(runtime, reference_fasta)
    if file_record(fasta) != records["accession.fasta"]:
        raise IntegrityError("Reference FASTA differs from profunc dataset manifest")
    manifest = {
        "schema_version": 1,
        "authority": "profunc",
        "source": {"runtime": runtime.source, "identity": runtime.manifest},
        "files": records,
    }
    result = _read_reference(manifest, runtime.clusters / "clusters.tsv", fasta)
    # Catch replacement of source files during the read before publishing inputs.
    if (
        file_record(fasta) != records["accession.fasta"]
        or file_record(runtime.clusters / "clusters.tsv") != records["clusters.tsv"]
    ):
        raise IntegrityError("Profunc reference changed during preparation; retry")
    return result


def sequence_id(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def read_fasta(path):
    """Yield unique primary IDs and nonempty, unmodified alphabetic sequences."""
    seen = set()
    current = None
    pieces = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if current is not None:
                    sequence = "".join(pieces)
                    if not sequence:
                        raise IntegrityError(f"Empty sequence: {current}")
                    yield current, sequence
                header = line[1:].strip()
                if not header:
                    raise IntegrityError("Empty FASTA ID")
                current = header.split()[0]
                if current in seen:
                    raise IntegrityError(f"Duplicate FASTA ID: {current}")
                seen.add(current)
                pieces = []
            else:
                sequence = line.strip(" \t\r\n\v\f")
                if not sequence:
                    continue
                if current is None or not sequence.isascii() or not sequence.isalpha():
                    raise IntegrityError(f"Invalid FASTA sequence near {current}")
                pieces.append(sequence)
    if current is None or not pieces:
        raise IntegrityError("Missing or empty final FASTA sequence")
    yield current, "".join(pieces)


@schema_checked
def load_reference(root):
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("counts"), dict
    ):
        raise IntegrityError("Unsupported reference manifest")
    verify_files(root, manifest.get("files"))
    for name in ("clusters.tsv", "accession.fasta"):
        if name not in manifest["files"]:
            raise IntegrityError(f"Missing reference member: {name}")
    source = root / "source_manifests"
    if source.exists():
        bundle = read_json(source / "bundle.json")
        dataset = read_json(source / "dataset.json")
        if (
            bundle["clusters"]["source_generation_id"] != dataset["generation_id"]
            or bundle["dataset"]["generation_id"] != dataset["generation_id"]
        ):
            raise IntegrityError("Reference cluster/dataset generation mismatch")
        for name, group in (
            ("clusters.tsv", "clusters"),
            ("accession.fasta", "dataset"),
        ):
            src = read_json(source / f"{group}.json")
            record = next((v for v in src["members"] if v["path"] == name), None)
            expected = manifest["files"][name]
            if (
                record is None
                or record["sha256"] != expected["sha256"]
                or record["size"] != expected["size_bytes"]
            ):
                raise IntegrityError(f"Source manifest disagrees: {name}")
    return _read_reference(manifest, root / "clusters.tsv", root / "accession.fasta")


def _read_reference(manifest, membership_path, fasta_path):
    memberships = {}
    with Path(membership_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["member_id", "cluster_id"]:
            raise IntegrityError("Expected member_id/cluster_id header")
        for row in reader:
            accession, cluster = row.get("member_id"), row.get("cluster_id")
            if (
                None in row
                or not accession
                or not cluster
                or any(
                    x != x.strip() or any(c.isspace() for c in x)
                    for x in (accession, cluster)
                )
                or accession in memberships
            ):
                raise IntegrityError("Malformed or duplicate accession membership")
            memberships[accession] = cluster
    if not memberships:
        raise IntegrityError("Empty reference membership")
    sequences = {}
    all_count = residues = 0
    noncanonical = {}
    for accession, sequence in read_fasta(fasta_path):
        all_count += 1
        residues += len(sequence)
        if accession in memberships:
            sequences[accession] = sequence
            extra = "".join(sorted(set(sequence) - set("ACDEFGHIKLMNPQRSTVWY")))
            if extra:
                noncanonical[accession] = extra
    if sequences.keys() != memberships.keys():
        raise IntegrityError("Clustered accessions missing from FASTA")
    counts = {
        "fasta_records": all_count,
        "fasta_residues": residues,
        "clustered_accessions": len(memberships),
        "clusters": len(set(memberships.values())),
        "unclustered_fasta_accessions": all_count - len(memberships),
        "clustered_noncanonical_sequences": len(noncanonical),
    }
    if "counts" in manifest and manifest["counts"] != counts:
        raise IntegrityError("Reference observed counts disagree with manifest")
    manifest = {**manifest, "counts": counts}
    return manifest, memberships, sequences, noncanonical
