"""Read immutable bundled references without requiring profunc or a PLM."""

from __future__ import annotations

import csv
import hashlib
from importlib.resources import files
from pathlib import Path

from .artifacts import IntegrityError, read_json, schema_checked, verify_files


def reference_root():
    return files("pclass_dark").joinpath("resources", "reference")


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
    if manifest.get("schema_version") != 1:
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
    memberships = {}
    with (root / "clusters.tsv").open(encoding="utf-8", newline="") as handle:
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
    for accession, sequence in read_fasta(root / "accession.fasta"):
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
    if manifest.get("counts") != counts:
        raise IntegrityError("Reference observed counts disagree with manifest")
    return manifest, memberships, sequences, noncanonical
