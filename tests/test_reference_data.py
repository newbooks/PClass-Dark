"""Validate live profunc inputs and explicit custom-reference contracts."""


def test_custom_reference_integrity_and_coverage(reference_factory):
    from pclass_dark.reference_data import load_reference

    root = reference_factory(
        [
            ("P1", "K1", "ACD"),
            ("P2", "K1", "MX"),
            ("P3", None, "AAAA"),
        ]
    )
    manifest, members, sequences, noncanonical = load_reference(root)
    assert members == {"P1": "K1", "P2": "K1"}
    assert sequences == {"P1": "ACD", "P2": "MX"}
    assert noncanonical == {"P2": "X"}
    assert manifest["counts"] == {
        "fasta_records": 3,
        "fasta_residues": 9,
        "clustered_accessions": 2,
        "clusters": 1,
        "unclustered_fasta_accessions": 1,
        "clustered_noncanonical_sequences": 1,
    }


def test_profunc_preparation_refreshes_memberships(
    reference_factory, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import pytest

    from pclass_dark.artifacts import IntegrityError, file_record, read_json, write_json
    from pclass_dark.setup_data import prepare_data

    root = reference_factory()
    write_json(
        root / "manifest.json",
        {
            "generation_id": "source-test",
            "members": [
                {
                    "path": "clusters.tsv",
                    "sha256": file_record(root / "clusters.tsv")["sha256"],
                    "size": file_record(root / "clusters.tsv")["size_bytes"],
                },
                {
                    "path": "accession.fasta",
                    "size": file_record(root / "accession.fasta")["size_bytes"],
                    "sha256": file_record(root / "accession.fasta")["sha256"],
                },
            ],
        },
    )
    runtime = SimpleNamespace(dataset=root, clusters=root, manifest={}, source="test")
    monkeypatch.setattr("pclass_dark.setup_data.resolve_profunc", lambda: runtime)
    output = tmp_path / "prepared"
    first = prepare_data(output, prepare_only=True)
    assert prepare_data(output, prepare_only=True).reused_prepared
    path = root / "clusters.tsv"
    path.write_text(path.read_text().replace("K0", "K3"))
    with pytest.raises(IntegrityError, match="membership differs"):
        prepare_data(output, prepare_only=True)
    manifest = read_json(root / "manifest.json")
    manifest["members"][0].update(
        sha256=file_record(path)["sha256"], size=file_record(path)["size_bytes"]
    )
    write_json(root / "manifest.json", manifest)
    changed = prepare_data(output, prepare_only=True)
    assert changed.preparation_id != first.preparation_id
    assert not changed.reused_prepared
    assert read_json(changed.prepared_dir / "manifest.json")["reference"]["files"][
        "clusters.tsv"
    ] == file_record(path)


def test_profunc_fasta_validation_and_override(
    reference_factory, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    import pytest

    from pclass_dark.artifacts import IntegrityError, file_record, write_json
    from pclass_dark.reference_data import load_profunc_reference

    root = reference_factory()
    fasta = root / "accession.fasta"
    record = file_record(fasta)
    write_json(
        root / "manifest.json",
        {
            "generation_id": "unavailable",
            "members": [
                {
                    "path": "clusters.tsv",
                    "sha256": file_record(root / "clusters.tsv")["sha256"],
                    "size": file_record(root / "clusters.tsv")["size_bytes"],
                },
                {
                    "path": "accession.fasta",
                    "size": record["size_bytes"],
                    "sha256": record["sha256"],
                },
            ],
        },
    )
    runtime = SimpleNamespace(dataset=root, clusters=root, manifest={}, source="test")
    monkeypatch.setattr(
        "profunc.workspace.default_workspace", lambda: tmp_path / "empty"
    )
    source_dir = tmp_path / "sequence-source"
    source_dir.mkdir()
    external = source_dir / "source.fasta"
    fasta.rename(external)
    with pytest.raises(FileNotFoundError, match="--reference-fasta"):
        load_profunc_reference(runtime)
    assert load_profunc_reference(runtime, external)[1]
    from pclass_dark.setup_data import prepare_data

    monkeypatch.setattr("pclass_dark.setup_data.resolve_profunc", lambda: runtime)
    with pytest.raises(ValueError, match="overlap"):
        prepare_data(source_dir, reference_fasta=external, prepare_only=True)
    external.write_text(">P0000\nAAAA\n")
    with pytest.raises(IntegrityError, match="differs"):
        load_profunc_reference(runtime, external)
