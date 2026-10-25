"""Validate the bundled snapshot and its recorded provenance."""

import csv
import hashlib
import json
from importlib.resources import files


def test_reference_integrity_and_coverage():
    reference = files("pclass_dark").joinpath("resources", "reference")
    manifest = json.loads(reference.joinpath("manifest.json").read_text())
    for name, expected in manifest["files"].items():
        digest = hashlib.sha256()
        size = 0
        with reference.joinpath(*name.split("/")).open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        assert digest.hexdigest() == expected["sha256"], name
        assert size == expected["size_bytes"], name

    members = set()
    clusters = set()
    with reference.joinpath("clusters.tsv").open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        assert reader.fieldnames == ["member_id", "cluster_id"]
        for row in reader:
            assert row["member_id"] and row["cluster_id"]
            assert row["member_id"] not in members
            members.add(row["member_id"])
            clusters.add(row["cluster_id"])

    sequences = set()
    residues = 0
    noncanonical = set()
    current = None
    length = 0
    with reference.joinpath("accession.fasta").open() as handle:
        for line in handle:
            if line.startswith(">"):
                if current is not None:
                    assert length > 0
                current = line[1:].split()[0]
                assert current not in sequences
                sequences.add(current)
                length = 0
            else:
                assert current is not None
                sequence = line.strip()
                length += len(sequence)
                residues += len(sequence)
                if set(sequence) - set("ACDEFGHIKLMNPQRSTVWY"):
                    noncanonical.add(current)
    assert current is not None and length > 0
    assert members <= sequences
    assert manifest["counts"] == {
        "clustered_accessions": len(members),
        "clusters": len(clusters),
        "fasta_records": len(sequences),
        "fasta_residues": residues,
        "unclustered_fasta_accessions": len(sequences - members),
        "clustered_noncanonical_sequences": len(noncanonical & members),
    }
