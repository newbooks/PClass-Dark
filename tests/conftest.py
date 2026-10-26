"""Small independent reference snapshots for setup tests."""

import pytest

from pclass_dark.artifacts import file_record, write_json


def protein(index):
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    result = []
    while index:
        result.append(alphabet[index % len(alphabet)])
        index //= len(alphabet)
    return "M" + "".join(result).ljust(7, "A")


@pytest.fixture
def reference_factory(tmp_path):
    def make(records=None):
        root = tmp_path / "reference"
        root.mkdir(exist_ok=True)
        if records is None:
            records = [(f"P{i:04d}", f"K{i % 3}", protein(i)) for i in range(140)]
        clustered = [(a, c, s) for a, c, s in records if c is not None]
        with (root / "clusters.tsv").open("w") as handle:
            handle.write("member_id\tcluster_id\n")
            for a, c, _ in clustered:
                handle.write(f"{a}\t{c}\n")
        with (root / "accession.fasta").open("w") as handle:
            for a, _, s in records:
                handle.write(f">{a}\n{s}\n")
        counts = {
            "fasta_records": len(records),
            "fasta_residues": sum(len(s) for _, _, s in records),
            "clustered_accessions": len(clustered),
            "clusters": len({c for _, c, _ in clustered}),
            "unclustered_fasta_accessions": len(records) - len(clustered),
            "clustered_noncanonical_sequences": sum(
                bool(set(s) - set("ACDEFGHIKLMNPQRSTVWY")) for _, _, s in clustered
            ),
        }
        write_json(
            root / "manifest.json",
            {
                "schema_version": 1,
                "counts": counts,
                "files": {
                    n: file_record(root / n)
                    for n in ("clusters.tsv", "accession.fasta")
                },
            },
        )
        return root

    return make
