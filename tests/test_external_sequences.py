"""Independent SPEC-0014 fixtures; no live downloads or large sources."""

import gzip
import json
import pickle
import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pclass_dark import external_sequence_data as data
from pclass_dark import external_sequence_io as io
from pclass_dark import external_sequences as external
from pclass_dark.cli import app as root_app
from pclass_dark.external_sequence_cli import app
from pclass_dark.external_sequence_types import ExternalSnapshotError


def gaf_row(
    acc="P00001",
    *,
    db="UniProtKB",
    qualifier="enables",
    aspect="F",
    evidence="EXP",
    go="GO:0003824",
    date="20200101",
):
    fields = [
        db,
        acc,
        "symbol",
        qualifier,
        go,
        "PMID:1",
        evidence,
        "",
        aspect,
        "name",
        "",
        "protein",
        "taxon:1",
        date,
        "TEST",
        "",
        "",
    ]
    return "\t".join(fields) + "\n"


def write_gaf(path, rows):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("!gaf-version: 2.2\n")
        handle.writelines(rows)


@pytest.fixture
def inputs(tmp_path):
    graph = tmp_path / "graph.pkl"
    with graph.open("wb") as handle:
        pickle.dump(
            {
                "meta": {"schema_version": "2.0", "namespace": "molecular_function"},
                "terms": {"GO:0003824": {}, "GO:0005488": {}},
            },
            handle,
        )
    swiss = tmp_path / "swiss.txt"
    swiss.write_text("P00003\n")
    gaf = tmp_path / "annotations.gz"
    rows = [
        gaf_row("other-id", db="OtherDB"),
        gaf_row(qualifier="NOT"),
        gaf_row(aspect="P"),
        gaf_row(evidence="IEA"),
        gaf_row(go="GO:9999999", date="20260101"),
        gaf_row(),
        gaf_row("P00001-2", evidence="IDA", go="GO:0005488", date="20210101"),
        gaf_row("P00002", date="20220101"),
        gaf_row("P00003", date="20230101"),
        gaf_row("P00004", go="GO:0005488", date="20240101"),
        gaf_row("P00005", date="20250101"),
    ]
    write_gaf(gaf, rows)
    return dict(go_graph=graph, swissprot_accessions=swiss, gaf=gaf)


@pytest.fixture
def transport(monkeypatch):
    calls = []

    def fake_download(url, path, *, params=None, retries=3):
        assert url == io.SEQUENCE_URL
        ids = [
            c.removeprefix("accession:") for c in params["query"][1:-1].split(" OR ")
        ]
        calls.append(ids)
        seqs = {"P00001": "AAAA", "P00002": "AAAA", "P00004": "AA*"}
        path.write_text("".join(f">tr|{a}|NAME\n{seqs[a]}\n" for a in ids if a in seqs))
        return {
            "schema_version": 1,
            "url": url,
            "retrieved_at": "2026-10-31",
            "headers": {
                "X-UniProt-Release": "2026_01",
                "X-UniProt-Release-Date": "2026-01-01",
            },
            "record": io.record(path),
        }

    monkeypatch.setattr(io, "download", fake_download)
    return calls


def test_acceptance_counts_and_winner(tmp_path, inputs, transport):
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert [s.annotation_rows for s in result.counts[:6]] == [11, 10, 9, 8, 7, 6]
    assert [s.accessions for s in result.counts[1:6]] == [5] * 5
    assert [s.accessions for s in result.counts[6:]] == [5, 4, 3, 2, 1]
    assert result.fasta_path.read_text() == (
        ">P00002 go_mf_terms=GO:0003824 latest_annotation_date=20220101\nAAAA\n"
    )
    report = result.report_path.read_text()
    assert "unresolved_accessions: 1" in report
    assert "duplicate_sequences: 1" in report
    assert all(line == line.rstrip() for line in report.splitlines())
    manifest = io.read_json(result.manifest_path)
    io.verify_outputs(tmp_path, manifest)
    before = result.report_path.read_bytes()
    calls = len(transport)
    again = external.collect_external_sequences(work_dir=tmp_path, workers=1, **inputs)
    assert again == result and again.report_path.read_bytes() == before
    assert len(transport) == calls


def test_serial_parallel_and_tie(tmp_path, inputs, transport):
    write_gaf(inputs["gaf"], [gaf_row("P00002"), gaf_row("P00001")])
    results = []
    for workers in [1, 3]:
        rows = [gaf_row("P00002"), gaf_row("P00001")]
        write_gaf(inputs["gaf"], rows if workers == 1 else list(reversed(rows)))
        work = tmp_path / str(workers)
        work.mkdir()
        result = external.collect_external_sequences(
            work_dir=work, workers=workers, batch_size=1, **inputs
        )
        results.append(result.fasta_path.read_bytes())
    assert results[0] == results[1]
    assert results[0].startswith(b">P00001 ")


@pytest.mark.parametrize("mutation", ["date", "columns", "version", "gzip"])
def test_invalid_gaf_fails_without_publication(tmp_path, inputs, transport, mutation):
    if mutation == "date":
        write_gaf(inputs["gaf"], [gaf_row(date="20260230")])
    elif mutation == "columns":
        write_gaf(inputs["gaf"], ["bad\trow\n"])
    elif mutation == "version":
        with gzip.open(inputs["gaf"], "wt") as f:
            f.write("!gaf-version: 1.0\n")
    else:
        content = inputs["gaf"].read_bytes()
        inputs["gaf"].write_bytes(content[:-8])
    with pytest.raises((ValueError, EOFError, OSError)):
        external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert not (tmp_path / io.OUTPUTS[0]).exists()
    assert not transport


def test_empty_gaf_no_sequence_http(tmp_path, inputs, transport):
    write_gaf(inputs["gaf"], [])
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert result.fasta_path.read_bytes() == b""
    assert result.counts[-1].sequences == 0
    assert not transport


def test_all_experimental_codes_and_not_token(tmp_path):
    rows = [gaf_row(evidence=code) for code in sorted(data.EVIDENCE)]
    rows += [gaf_row(qualifier="enables|NOT"), gaf_row(qualifier="NOTHING")]
    path = tmp_path / "gaf.gz"
    write_gaf(path, rows)
    db = data.connect(tmp_path / "db")
    try:
        stages, _ = data.parse_gaf(db, path, {"GO:0003824"}, set())
        assert stages[1].annotation_rows == 13
        assert stages[2].annotation_rows == 12
        assert db.execute("SELECT COUNT(*) FROM annotations").fetchone()[0] == 11
    finally:
        db.close()


@pytest.mark.parametrize(
    "body",
    [
        ">P00099\nAAAA\n",
        ">P00001-2\nAAAA\n",
        "AAAA\n",
        ">bad|P00001|NAME\nAAAA\n",
        ">P00001\nAAAA\n>P00001\nCCCC\n",
    ],
)
def test_invalid_response(tmp_path, body):
    path = tmp_path / "response"
    path.write_text(body)
    with pytest.raises(ValueError):
        data.fasta_records(path, {"P00001"})


def test_resume_reuses_completed_batches(tmp_path, inputs, transport, monkeypatch):
    original = external._batch

    def fail(ids, *args):
        if ids == ["P00004"]:
            raise io.ExternalDownloadError("interrupted")
        return original(ids, *args)

    monkeypatch.setattr(external, "_batch", fail)
    with pytest.raises(io.ExternalDownloadError):
        external.collect_external_sequences(
            work_dir=tmp_path, workers=1, batch_size=1, **inputs
        )
    completed = {a for batch in transport for a in batch}
    monkeypatch.setattr(external, "_batch", original)
    transport.clear()
    result = external.collect_external_sequences(
        work_dir=tmp_path, workers=1, batch_size=1, **inputs
    )
    assert result.counts[-1].sequences == 1
    assert "P00001" in completed
    assert not any("P00001" in batch for batch in transport)


def test_refresh_failure_preserves_old_pair(tmp_path, inputs, transport, monkeypatch):
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    before = [p.read_bytes() for p in (result.fasta_path, result.report_path)]

    def fail(*args, **kwargs):
        raise io.ExternalDownloadError("offline")

    monkeypatch.setattr(io, "download", fail)
    with pytest.raises(io.ExternalDownloadError):
        external.collect_external_sequences(work_dir=tmp_path, refresh=True, **inputs)
    assert before == [p.read_bytes() for p in (result.fasta_path, result.report_path)]


@pytest.mark.parametrize("existing", [False, True])
def test_publication_fault_rollback(tmp_path, inputs, transport, monkeypatch, existing):
    if existing:
        result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
        before = [p.read_bytes() for p in (result.fasta_path, result.report_path)]
        # Refresh really changes the first public file before the injected fault.
        write_gaf(inputs["gaf"], [gaf_row(date="20270101")])
    original = io.os.replace

    def interrupt(source, target):
        if Path(target) == tmp_path / "external_filtered.txt":
            raise OSError("publication fault")
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(io.os, "replace", interrupt)
        with pytest.raises(OSError, match="publication fault"):
            external.collect_external_sequences(
                work_dir=tmp_path, refresh=True, **inputs
            )
    state = tmp_path / ".external-seq"
    assert (state / "publication.json").exists()
    io.recover(tmp_path, state)
    if existing:
        assert before == [(tmp_path / name).read_bytes() for name in io.OUTPUTS]
    else:
        assert not any((tmp_path / name).exists() for name in io.OUTPUTS)
    io.recover(tmp_path, state)
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert result.counts[-1].sequences == 1


def test_snapshot_and_symlink_rejection(tmp_path, inputs, transport):
    external.collect_external_sequences(work_dir=tmp_path, **inputs)
    inputs["swissprot_accessions"].write_text("P00001\n")
    with pytest.raises(ExternalSnapshotError, match="Changed run"):
        external.collect_external_sequences(work_dir=tmp_path, **inputs)
    (tmp_path / ".external-seq" / "link").symlink_to(inputs["gaf"])
    with pytest.raises(ValueError, match="Symlink"):
        external.collect_external_sequences(work_dir=tmp_path, **inputs)


def test_release_change_and_missing_headers(tmp_path, inputs, transport, monkeypatch):
    original = io.download
    counter = 0

    def changed(*args, **kwargs):
        nonlocal counter
        info = original(*args, **kwargs)
        counter += 1
        info["headers"]["X-UniProt-Release"] = str(counter)
        return info

    monkeypatch.setattr(io, "download", changed)
    with pytest.raises(ExternalSnapshotError, match="release changed"):
        external.collect_external_sequences(work_dir=tmp_path, batch_size=1, **inputs)
    assert not (tmp_path / io.OUTPUTS[0]).exists()


def test_graph_filtering_and_local_list(tmp_path, inputs):
    graph = pickle.loads(inputs["go_graph"].read_bytes())
    graph["terms"]["GO:0005488"]["obsolete"] = True
    inputs["go_graph"].write_bytes(pickle.dumps(graph))
    inputs["swissprot_accessions"].write_text("# comment\nP00003-2\nP00003\n\n")
    nodes, excluded, _ = data.references(
        inputs["go_graph"], inputs["swissprot_accessions"], None
    )
    assert nodes == {"GO:0003824"} and excluded == {"P00003"}


def test_cli_script_and_cwd(tmp_path, inputs, transport, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = [
        "--go-graph",
        str(inputs["go_graph"]),
        "--swissprot-accessions",
        str(inputs["swissprot_accessions"]),
        "--gaf",
        str(inputs["gaf"]),
    ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert (tmp_path / "external_sequences.fasta").exists()
    assert CliRunner().invoke(app, ["--workers", "0"]).exit_code == 2
    script = Path(__file__).resolve().parents[1] / "scripts/get-external-seq.py"
    result = subprocess.run(
        [str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--go-graph" in result.stdout


class Response:
    def __init__(self, code, body=b"", headers=None, failure=False):
        self.status_code = code
        self.body = body
        self.headers = headers or {}
        self.failure = failure

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, chunk_size):
        yield self.body
        if self.failure:
            raise io.requests.ConnectionError("truncated stream")


def fake_session(monkeypatch, responses):
    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, *args, **kwargs):
            return responses.pop(0)

    monkeypatch.setattr(io.requests, "Session", Session)
    monkeypatch.setattr(io.time, "sleep", lambda delay: None)


def test_http_retry_restarts_and_closes(tmp_path, monkeypatch):
    fake_session(
        monkeypatch,
        [
            Response(429, headers={"Retry-After": "0"}),
            Response(200, b"partial", failure=True),
            Response(200, b"complete"),
        ],
    )
    path = tmp_path / "response"
    meta = io.download("https://example.test", path, retries=2)
    assert path.read_bytes() == b"complete" and meta["record"] == io.record(path)
    assert not path.with_name("response.part").exists()


@pytest.mark.parametrize(
    "response", [Response(400), Response(429, headers={"Retry-After": "301"})]
)
def test_http_terminal_failures(tmp_path, monkeypatch, response):
    fake_session(monkeypatch, [response])
    with pytest.raises(io.ExternalDownloadError):
        io.download("https://example.test", tmp_path / "response")


def test_unregistered_output_refused_before_http(tmp_path, inputs, transport):
    (tmp_path / io.OUTPUTS[0]).write_text("user data")
    with pytest.raises(ExternalSnapshotError, match="Unregistered outputs"):
        external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert not transport


def test_default_reference_includes_unclustered_accessions(
    tmp_path, inputs, monkeypatch
):
    from types import SimpleNamespace

    from pclass_dark import reference_data

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    graph = dataset / "go_graph.pkl"
    graph.write_bytes(inputs["go_graph"].read_bytes())
    fasta = dataset / "accession.fasta"
    fasta.write_text(">sp|P00003|CLUSTERED\nAAAA\n>P00004\nCCCC\n")
    members = [
        {
            "path": p.name,
            "size": io.record(p)["size_bytes"],
            "sha256": io.record(p)["sha256"],
        }
        for p in (graph, fasta)
    ]
    (dataset / "manifest.json").write_text(json.dumps({"members": members}))
    runtime = SimpleNamespace(dataset=dataset, manifest={"id": "fixture"})
    monkeypatch.setattr(reference_data, "resolve_profunc", lambda: runtime)
    monkeypatch.setattr(reference_data, "profunc_fasta_path", lambda r, f: fasta)
    monkeypatch.setattr(
        reference_data,
        "profunc_reference_contract",
        lambda r: {"accession.fasta": io.record(fasta)},
    )
    nodes, excluded, contract = data.references(None, None, None)
    assert excluded == {"P00003", "P00004"}
    assert nodes == {"GO:0003824", "GO:0005488"}
    assert contract["runtime"] == runtime.manifest
    # A duplicate accession in the complete source is never silently hidden.
    fasta.write_text(">P00003\nAAAA\n>P00003\nCCCC\n")
    with pytest.raises(ValueError, match="Duplicate Swiss-Prot"):
        data.references(None, None, None)


def test_cached_download_validation_and_failed_refresh(tmp_path, monkeypatch):
    state = tmp_path / ".external-seq"
    state.mkdir()
    gz = gzip.compress(b"!gaf-version: 2.2\n")
    bodies = [gz, b"not gzip"]

    def download(url, path, **kwargs):
        path.write_bytes(bodies.pop(0))
        return {"schema_version": 1, "record": io.record(path)}

    monkeypatch.setattr(io, "download", download)
    path, _ = io.cached_gaf(tmp_path, state, False, 0)
    assert path.read_bytes() == gz
    io.cached_gaf(tmp_path, state, False, 0)
    with pytest.raises(OSError):
        io.cached_gaf(tmp_path, state, True, 0)
    assert path.read_bytes() == gz
    assert len(bodies) == 0
    path.write_bytes(b"changed")
    with pytest.raises(ExternalSnapshotError):
        io.cached_gaf(tmp_path, state, False, 0)


def test_missing_release_requires_refresh(tmp_path, inputs, transport, monkeypatch):
    original = io.download

    def missing(*args, **kwargs):
        info = original(*args, **kwargs)
        info["headers"] = {}
        return info

    with monkeypatch.context() as patch:
        patch.setattr(io, "download", missing)
        with pytest.raises(ExternalSnapshotError, match="Missing UniProt"):
            external.collect_external_sequences(work_dir=tmp_path, **inputs)
    with pytest.raises(ExternalSnapshotError, match="Previous snapshot failure"):
        external.collect_external_sequences(work_dir=tmp_path, **inputs)
    result = external.collect_external_sequences(
        work_dir=tmp_path, refresh=True, **inputs
    )
    assert result.counts[-1].sequences == 1


def test_corrupt_batch_is_refetched(tmp_path, inputs, transport, monkeypatch):
    original = io.publish

    def fail(*args):
        raise OSError("before publication")

    with monkeypatch.context() as patch:
        patch.setattr(io, "publish", fail)
        with pytest.raises(OSError):
            external.collect_external_sequences(work_dir=tmp_path, **inputs)
    state = tmp_path / ".external-seq"
    run = state / "runs" / io.read_json(state / "active.json")["run_id"]
    batch = next((run / "batches").glob("*.fasta"))
    batch.write_text("corruption")
    transport.clear()
    monkeypatch.setattr(io, "publish", original)
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert transport and result.counts[-1].sequences == 1


def test_recovery_refuses_missing_backup(tmp_path, inputs, transport, monkeypatch):
    external.collect_external_sequences(work_dir=tmp_path, **inputs)
    original = io.os.replace

    def interrupt(source, target):
        if Path(target) == tmp_path / "external_filtered.txt":
            raise OSError("stop")
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(io.os, "replace", interrupt)
        with pytest.raises(OSError):
            external.collect_external_sequences(
                work_dir=tmp_path, refresh=True, **inputs
            )
    state = tmp_path / ".external-seq"
    journal = io.read_json(state / "publication.json")
    (
        state / "runs" / journal["run_id"] / "backups" / "external_sequences.fasta"
    ).unlink()
    with pytest.raises(io.ExternalRecoveryError, match="Missing/corrupt"):
        io.recover(tmp_path, state)
    assert (state / "publication.json").exists()


def test_recovery_finishes_committed_pair(tmp_path, inputs, transport, monkeypatch):
    original = io.recover

    def fail_after_commit(root, state):
        if (state / "publication.json").exists():
            raise OSError("after commit")
        original(root, state)

    with monkeypatch.context() as patch:
        patch.setattr(io, "recover", fail_after_commit)
        with pytest.raises(OSError):
            external.collect_external_sequences(work_dir=tmp_path, **inputs)
    calls = len(transport)
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert result.counts[-1].sequences == 1 and calls == len(transport)
    assert not (tmp_path / ".external-seq/publication.json").exists()


def test_hash_collision_keeps_distinct_sequences(tmp_path):
    db = data.connect(tmp_path / "state.sqlite")
    try:
        db.executemany(
            "INSERT INTO candidates VALUES (?,?,0)",
            [("P00001", "20200101"), ("P00002", "20210101")],
        )
        db.executemany(
            "INSERT INTO sequences VALUES (?,'valid',?,'samehash')",
            [("P00001", "AAAA"), ("P00002", "CCCC")],
        )
        stages = [external.StageCount(name, None, 2, None) for name in data.STAGES[:8]]
        final, rejected = data.deduplicate(db, stages)
        assert final[-1].sequences == 2
        assert rejected["duplicate_sequences"] == 0
    finally:
        db.close()


@pytest.mark.parametrize(
    "params",
    [
        {"workers": 0},
        {"workers": True},
        {"batch_size": 101},
        {"retries": -1},
        {"swissprot_accessions": "a", "reference_fasta": "b"},
    ],
)
def test_invalid_api_configuration(tmp_path, params):
    with pytest.raises(ValueError):
        external.collect_external_sequences(work_dir=tmp_path, **params)


def test_lock_contention(tmp_path, inputs, transport):
    from filelock import FileLock

    state = tmp_path / ".external-seq"
    state.mkdir()
    with FileLock(state / "lock"):
        with pytest.raises(RuntimeError, match="busy"):
            external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert not transport


def test_report_exact_table(tmp_path, inputs, transport):
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    expected = """\
Stage                             Annotation rows  Accessions  Sequences
--------------------------------  ---------------  ----------  ---------
Input GAF records                              11           -          -
UniProtKB                                      10           5          -
Without NOT                                     9           5          -
MF aspect F                                     8           5          -
Experimental evidence                           7           5          -
Present in local MF graph                       6           5          -
Grouped parent accessions                       -           5          -
After local Swiss-Prot exclusion                -           4          -
Retrieved full sequences                        -           3          3
Valid full sequences                            -           2          2
Deduplicated (latest annotation)                -           1          1"""
    assert result.report_path.read_text().split("\n\n")[0] == expected


def test_invalid_utf8_and_residues(tmp_path):
    path = tmp_path / "response"
    path.write_bytes(b">P00001\n\xff\n")
    with pytest.raises(UnicodeDecodeError):
        data.fasta_records(path, {"P00001"})
    path.write_text(">P00001\nacdubzxoj\n>P00002\n\n>P00004\nAA-\n")
    records = data.fasta_records(path, {"P00001", "P00002", "P00004"})
    db = data.connect(tmp_path / "db")
    try:
        data.store_sequences(db, list(records), records)
        assert list(
            db.execute("SELECT accession,status FROM sequences ORDER BY accession")
        ) == [("P00001", "valid"), ("P00002", "invalid"), ("P00004", "invalid")]
    finally:
        db.close()


def test_gaf_registration_failure_restores_cache(tmp_path, monkeypatch):
    state = tmp_path / ".external-seq"
    state.mkdir()
    old = gzip.compress(b"!gaf-version: 2.1\n")
    new = gzip.compress(b"!gaf-version: 2.2\n")
    path = tmp_path / "goa_uniprot_all.gaf.gz"
    path.write_bytes(old)
    io.write_json(state / "gaf.json", {"schema_version": 1, "record": io.record(path)})
    old_meta = (state / "gaf.json").read_bytes()

    def download(url, target, **kwargs):
        target.write_bytes(new)
        return {"schema_version": 1, "record": io.record(target)}

    def fail(*args):
        raise OSError("registration failed")

    monkeypatch.setattr(io, "download", download)
    monkeypatch.setattr(io, "write_json", fail)
    with pytest.raises(OSError, match="registration failed"):
        io.cached_gaf(tmp_path, state, True, 0)
    assert path.read_bytes() == old
    assert (state / "gaf.json").read_bytes() == old_meta


def test_nonascii_uppercase_does_not_create_valid_residues(tmp_path):
    path = tmp_path / "response"
    path.write_text(">P00001\nß\n")
    rows = data.fasta_records(path, {"P00001"})
    db = data.connect(tmp_path / "db")
    try:
        data.store_sequences(db, ["P00001"], rows)
        assert db.execute("SELECT status FROM sequences").fetchone()[0] == "invalid"
    finally:
        db.close()


def test_interrupted_gaf_registration_recovers_old_inode(tmp_path):
    state = tmp_path / ".external-seq"
    state.mkdir()
    path = tmp_path / "goa_uniprot_all.gaf.gz"
    old = gzip.compress(b"!gaf-version: 2.1\n")
    path.write_bytes(old)
    io.write_json(state / "gaf.json", {"schema_version": 1, "record": io.record(path)})
    (state / "gaf.previous.json").write_bytes((state / "gaf.json").read_bytes())
    io.os.link(path, state / "gaf.previous")
    staged = state / "new.download"
    staged.write_bytes(gzip.compress(b"!gaf-version: 2.2\n"))
    io.os.replace(staged, path)
    # Simulate process death between source replacement and metadata replacement.
    recovered, _ = io.cached_gaf(tmp_path, state, False, 0)
    assert recovered.read_bytes() == old
    assert not (state / "gaf.previous").exists()


def test_unknown_state_versions(tmp_path):
    state = tmp_path / "unknown.json"
    state.write_text('{"schema_version": 2}')
    with pytest.raises(ValueError, match="Unsupported"):
        io.read_json(state)
    import sqlite3

    db_path = tmp_path / "unknown.sqlite"
    with sqlite3.connect(db_path) as db:
        db.execute("PRAGMA user_version=2")
    with pytest.raises(ValueError, match="Unsupported"):
        data.connect(db_path)


def test_all_unresolved_is_reported_zero_result(tmp_path, inputs, transport):
    write_gaf(inputs["gaf"], [gaf_row("P00005")])
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert result.fasta_path.read_bytes() == b""
    assert result.counts[-1].sequences == 0
    assert "unresolved_accessions: 1" in result.report_path.read_text()


def test_retry_after_http_date(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    retry_at = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=10))
    fake_session(
        monkeypatch,
        [Response(429, headers={"Retry-After": retry_at}), Response(200, b"complete")],
    )
    delays = []
    monkeypatch.setattr(io.time, "sleep", delays.append)
    io.download("https://example.test", tmp_path / "response", retries=1)
    assert len(delays) == 1 and 8 <= delays[0] <= 10


def test_pending_batches_are_bounded(tmp_path, inputs, transport, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock

    outstanding = peak = 0
    mutex = Lock()

    class ObservedExecutor(ThreadPoolExecutor):
        def submit(self, *args, **kwargs):
            nonlocal outstanding, peak
            with mutex:
                outstanding += 1
                peak = max(peak, outstanding)
            future = super().submit(*args, **kwargs)
            original = future.result

            def result(*args, **kwargs):
                nonlocal outstanding
                try:
                    return original(*args, **kwargs)
                finally:
                    with mutex:
                        outstanding -= 1

            future.result = result
            return future

    write_gaf(inputs["gaf"], [gaf_row(f"P{i:05d}") for i in range(1, 21)])
    monkeypatch.setattr(external, "ThreadPoolExecutor", ObservedExecutor)
    external.collect_external_sequences(
        work_dir=tmp_path, workers=2, batch_size=1, **inputs
    )
    assert peak == 4 and outstanding == 0


def test_help_is_lazy_and_interrupt_exit_code(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(external, "collect_external_sequences", fail)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert not (tmp_path / ".external-seq").exists()
    result = runner.invoke(app, [])
    assert result.exit_code == 130 and not result.stdout


def test_external_sequence_command_removed():
    runner = CliRunner()
    assert "get-external-seq" not in runner.invoke(root_app, ["--help"]).stdout
    result = runner.invoke(root_app, ["get-external-seq"])
    assert result.exit_code == 2


def test_progress_kb_commas_and_percentage(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger=io.__name__):
        io._log_gaf_progress(18790481920, 28598433883)
    assert caplog.messages == ["Downloaded GOA: 18,790,481 / 28,598,433 KB (65.7%)"]


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Content-Length": "invalid"},
        {"Content-Length": "0"},
        {"Content-Length": "-1"},
        {"Content-Length": "2000", "Content-Encoding": "gzip"},
    ],
)
def test_progress_unknown_total(headers, caplog):
    import logging

    total = io._response_total(headers)
    assert total is None
    with caplog.at_level(logging.INFO, logger=io.__name__):
        io._log_gaf_progress(1234000, total)
    assert caplog.messages == ["Downloaded GOA: 1,234 KB (percentage unavailable)"]


def test_download_final_progress(tmp_path, monkeypatch, caplog):
    import logging

    fake_session(
        monkeypatch, [Response(200, b"A" * 2000, headers={"Content-Length": "2000"})]
    )
    with caplog.at_level(logging.INFO, logger=io.__name__):
        io.download(io.GAF_URL, tmp_path / "download")
    assert caplog.messages[-1] == "Downloaded GOA: 2 / 2 KB (100.0%)"


@pytest.mark.parametrize(
    "source_id", ["CPX", "CPX-1234", "", "P12345-0", "not_an_accession"]
)
def test_bad_gaf_accessions_are_counted_not_fatal(
    tmp_path, inputs, transport, caplog, source_id
):
    write_gaf(inputs["gaf"], [gaf_row(source_id), gaf_row("P00001")])
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert [s.annotation_rows for s in result.counts[:6]] == [2, 1, 1, 1, 1, 1]
    assert result.counts[-1].sequences == 1
    assert transport == [["P00001"]]
    manifest = io.read_json(result.manifest_path)
    assert manifest["rejections"]["invalid_accession_rows"] == 1
    assert "invalid_accession_rows: 1" in result.report_path.read_text()
    run = tmp_path / ".external-seq/runs" / result.run_id
    headers = io.read_json(run / "run.json")["headers"]
    assert headers["invalid_accession_examples"] == [
        {"line": 2, "source_id": source_id}
    ]
    assert "Skipping GAF line 2" in caplog.text


def test_bad_accession_examples_and_warnings_are_bounded(
    tmp_path, inputs, transport, caplog
):
    write_gaf(inputs["gaf"], [gaf_row("CPX-1234")] * 25)
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert result.counts[0].annotation_rows == 25
    assert result.counts[1].annotation_rows == 0
    assert result.fasta_path.read_bytes() == b""
    assert not transport
    run = tmp_path / ".external-seq/runs" / result.run_id
    headers = io.read_json(run / "run.json")["headers"]
    assert headers["invalid_accession_rows"] == 25
    assert len(headers["invalid_accession_examples"]) == 10
    assert sum("Skipping GAF line" in m for m in caplog.messages) == 10
    assert (
        caplog.messages[-1] == "Skipped 25 GAF rows with invalid UniProtKB accessions"
    )


def test_parse_progress_uses_commas(caplog):
    import logging

    with caplog.at_level(logging.INFO, logger=data.__name__):
        data._log_parse_progress(648000000, 582876)
    assert caplog.messages == [
        "Parsed 648,000,000 GAF rows; 582,876 graph-qualified rows"
    ]


def test_failed_legacy_parse_restarts_with_cached_download(
    tmp_path, inputs, transport, monkeypatch
):
    write_gaf(inputs["gaf"], [gaf_row("CPX-1234"), gaf_row("P00001")])
    cached = tmp_path / "goa_uniprot_all.gaf.gz"
    cached.write_bytes(inputs.pop("gaf").read_bytes())
    state = tmp_path / ".external-seq"
    state.mkdir()
    io.write_json(
        state / "gaf.json", {"schema_version": 1, "record": io.record(cached)}
    )
    original = data.parse_gaf

    def old_failure(db, *args):
        # Model a previous strict parser leaving committed partial stage state.
        db.execute("INSERT INTO stage_accessions VALUES ('P99999',31)")
        db.commit()
        raise ValueError("GAF line 648144824: Invalid UniProt accession: 'CPX'")

    with monkeypatch.context() as patch:
        patch.setattr(data, "parse_gaf", old_failure)
        with pytest.raises(ValueError, match="CPX"):
            external.collect_external_sequences(work_dir=tmp_path, **inputs)
    failed_run_id = io.read_json(state / "active.json")["run_id"]
    monkeypatch.setattr(data, "parse_gaf", original)
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert result.run_id == failed_run_id
    assert result.counts[0].annotation_rows == 2
    assert result.counts[1].accessions == 1
    assert result.counts[-1].sequences == 1
    # Transport fixture accepts only sequence URLs, so a GAF redownload would fail.
    assert transport == [["P00001"]]


def test_nonprotein_id_still_invalid_in_local_reference(tmp_path, inputs):
    inputs["swissprot_accessions"].write_text("CPX-1234\n")
    with pytest.raises(ValueError, match="Invalid UniProt accession"):
        data.references(inputs["go_graph"], inputs["swissprot_accessions"], None)


def test_real_goa_complex_rows_are_rejected(tmp_path, inputs, transport):
    fixture = Path(__file__).parent / "fixtures/goa_nonprotein_ids.gaf"
    with gzip.open(inputs["gaf"], "wt", encoding="utf-8") as handle:
        handle.write(fixture.read_text())
        handle.write(gaf_row("P00001"))
    result = external.collect_external_sequences(work_dir=tmp_path, **inputs)
    assert result.counts[0].annotation_rows == 4
    assert [s.annotation_rows for s in result.counts[1:6]] == [1] * 5
    assert result.counts[-1].sequences == 1
    manifest = io.read_json(result.manifest_path)
    assert manifest["rejections"]["invalid_accession_rows"] == 3
    run = tmp_path / ".external-seq/runs" / result.run_id
    assert [
        r["source_id"]
        for r in io.read_json(run / "run.json")["headers"]["invalid_accession_examples"]
    ] == ["CPX-10041", "CPX-1014", "CPX-10161"]
    assert transport == [["P00001"]]
