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


@pytest.fixture
def library_data(reference_factory, tmp_path):
    import csv

    import numpy as np

    from pclass_dark.artifacts import identity, read_json, sha256
    from pclass_dark.embeddings import ASSET_HASHES, ROW_HEADER
    from pclass_dark.embeddings import SCHEMA as EMB_SCHEMA
    from pclass_dark.setup_data import MODEL, REVISION, prepare_data, read_tsv

    def make(count=20, min_clusters=10):
        records = [(f"P{i:04}", f"K{i % 2}", protein(i)) for i in range(count)]
        reference = reference_factory(records)
        root = tmp_path / "data"
        prepared = prepare_data(
            root, reference_dir=reference, prepare_only=True, min_clusters=min_clusters
        )
        pm = read_json(prepared.prepared_dir / "manifest.json")
        windows = read_tsv(prepared.prepared_dir / "windows.tsv")
        matrix = np.zeros((len(windows), 640), dtype="<f4")
        for i in range(len(windows)):
            matrix[i, :2] = (i % 2) * 10 + i * 0.001
        contract = {
            "model": MODEL,
            "revision": REVISION,
            "assets": dict(ASSET_HASHES),
            "layer": 30,
            "width": 640,
            "window_length": 1022,
            "inference_dtype": "float32",
            "output_dtype": "<f4",
            "pooling": "attention-and-not-special-mean-f32-v1",
            "residue_policy": "exact-vocab-ACDEFGHIKLMNPQRSTVWYXBZUO-v1",
            "torch_version": "synthetic-fixture",
            "transformers_version": "synthetic-fixture",
        }
        inputs = {"inventory_id": pm["inventory_id"], "contract": contract}
        eid = identity(inputs)
        ep = root / "embeddings" / eid
        ep.mkdir(parents=True, exist_ok=True)
        np.save(ep / "embeddings.npy", matrix, allow_pickle=False)
        with (ep / "rows.tsv").open("w", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(ROW_HEADER)
            for i, w in enumerate(windows):
                writer.writerow(
                    [i, w["window_id"], w["sequence_id"], w["window_sha256"]]
                )
        write_json(
            ep / "manifest.json",
            {
                "schema": EMB_SCHEMA,
                "status": "complete",
                "id": eid,
                "identity_inputs": inputs,
                "inventory_id": pm["inventory_id"],
                "rows": len(windows),
                "columns": 640,
                "files": {
                    n: file_record(ep / n) for n in ("embeddings.npy", "rows.tsv")
                },
            },
        )
        selected = read_json(root / "selection.json")
        selected.update(
            embedding=str(ep.relative_to(root)),
            embedding_id=eid,
            embedding_manifest_sha256=sha256(ep / "manifest.json"),
        )
        write_json(root / "selection.json", selected)
        return root, prepared, ep, matrix, contract

    return make
