"""SPEC-0015 analytical and failure/recovery tests; no production inference."""

import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pclass_dark import external_evaluation as evaluation
from pclass_dark import external_evaluation_collect as collect
from pclass_dark import external_evaluation_resources as resources
from pclass_dark import external_evaluation_scoring as scoring
from pclass_dark import external_evaluation_store as store
from pclass_dark.artifacts import identity
from pclass_dark.external_evaluation_cli import app
from pclass_dark.external_evaluation_types import (
    DarkConfig,
    EvaluationInferenceError,
    EvaluationIntegrityError,
    EvaluationQuery,
    EvaluationResourceError,
    ExecutionConfig,
    PClassConfig,
    SavedDarkQuery,
    SavedPClassQuery,
    ScoringConfig,
)

R, G, X, Y, Z = "GO:0003674", "GO:0003824", "GO:0000001", "GO:0000002", "GO:0000003"


@pytest.fixture
def graph_path(tmp_path):
    descriptions = {
        R: (0, [], []),
        G: (2, [R], [R]),
        X: (4, [G], [R, G]),
        Y: (4, [G], [R, G]),
        Z: (4, [R], [R]),
    }
    graph = {
        "meta": {"schema_version": "2.0", "namespace": "molecular_function"},
        "terms": {
            k: {"ic": ic, "parents": parents, "ancestors": ancestors}
            for k, (ic, parents, ancestors) in descriptions.items()
        },
    }
    path = tmp_path / "graph.pkl"
    path.write_bytes(pickle.dumps(graph))
    return path


@pytest.fixture
def map_payload():
    p = {
        "schema_version": 1,
        "clusters": {
            c: {"go_mf_terms": terms}
            for c, terms in [("C1", [Y]), ("C2", [X, Y]), ("C3", [Z])]
        },
    }
    p["mapping_id"] = identity(p)
    return p


@pytest.fixture
def mapping(map_payload):
    return resources.map_from_payload(map_payload)


def p_result(qid):
    terms = [{"go_id": X, "name": "X"}] if qid == "q1" else []
    neighbors = (
        []
        if qid == "q2"
        else [
            {
                "accession": "P00001",
                "distance": 0.2 if qid == "q1" else 0.3,
                "confidence": 0.9,
                "go_mf_terms": [X] if qid == "q1" else [],
                "used_for_go_voting": bool(terms),
            }
        ]
    )
    return {
        "id": qid,
        "description": "prediction",
        "go_terms": terms,
        "neighbors": neighbors,
        "confidence": 0.9 if neighbors else None,
        "sequence_confirmation": None,
        "prosig_evidence": None,
    }


def d_result(qid, sequence="AAAA", ordinal=0):
    from pclass_dark.inference_records import QuerySequence, make_query_windows

    windows = [
        w.info.to_dict()
        for w in make_query_windows(QuerySequence(qid, sequence)).windows
    ]
    choices = (
        [("C1", 0.01), ("C2", 0.02)]
        if qid == "q1"
        else ([("C3", 1.0)] if qid == "q3" else [])
    )
    matches = []
    for c, b in choices:
        support = {
            "window": windows[0],
            "background_tail_fraction": b,
            "cluster_percentile_le": 0.5,
            "accepted": True,
        }
        matches.append(
            {
                "cluster_id": c,
                "background_tail_fraction": b,
                "cluster_percentile_le": 0.5,
                "C_n": 10,
                "B_n": 20,
                "representative": windows[0],
                "supporting_windows": [support],
                "accepted": True,
            }
        )
    return {
        "schema": "pclass-dark.infer-query.v1",
        "query_index": ordinal,
        "query_id": qid,
        "length": len(sequence),
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "windows": windows,
        "status": "matches" if matches else "no_accepted_matches",
        "matches": matches,
    }


def records():
    queries = [
        EvaluationQuery(i, q, 4, hashlib.sha256(b"AAAA").hexdigest(), truth)
        for i, (q, truth) in enumerate([("q1", (X, Y)), ("q2", (X,)), ("q3", (X,))])
    ]
    p = [
        SavedPClassQuery(q.query_id, q.length, q.sequence_sha256, p_result(q.query_id))
        for q in queries
    ]
    d = [
        SavedDarkQuery(q.query_id, q.length, q.sequence_sha256, d_result(q.query_id))
        for q in queries
    ]
    return queries, p, d


def test_analytical_seven_column_golden(graph_path, mapping):
    graph = resources.load_graph(graph_path)
    q, p, d = records()
    rows = list(
        evaluation.score_external_predictions(
            q, reversed(p), reversed(d), graph=graph, cluster_go_map=mapping
        )
    )
    assert [r.cells() for r in rows] == [
        ("q1", X, "0.2", "0.875", Y, "0.01", "0.875"),
        ("q2", "NA", "NA", "NA", "NA", "NA", "NA"),
        ("q3", "NA", "0.3", "NA", Z, "1.0", "0.0"),
    ]
    assert rows[0].pclass_match != pytest.approx(5 / 6)
    assert scoring.metric(graph, (R,), (R,))[0] is None
    assert scoring.metric(graph, (X,), ())[1]["reason"] == "empty_prediction"


def test_top_b_tie_is_not_truth_oracle(graph_path, mapping):
    q, p, d = records()
    result = d[0].result
    result["matches"][0]["background_tail_fraction"] = 0.02
    row = list(
        evaluation.score_external_predictions(
            q, p, d, graph=resources.load_graph(graph_path), cluster_go_map=mapping
        )
    )[0]
    assert row.dark_go == (Y,) and row.dark_match == 0.875


@pytest.mark.parametrize("kind", ["missing", "duplicate", "extra", "hash"])
def test_scoring_rejects_incomplete_or_incompatible_data(graph_path, mapping, kind):
    q, p, d = records()
    if kind == "missing":
        p.pop()
    elif kind == "duplicate":
        p.append(p[0])
    elif kind == "extra":
        p.append(SavedPClassQuery("extra", 4, q[0].sequence_sha256, p_result("extra")))
    elif kind == "hash":
        p[0] = SavedPClassQuery("q1", 4, "0" * 64, p[0].result)
    with pytest.raises(ValueError):
        list(
            evaluation.score_external_predictions(
                q, p, d, graph=resources.load_graph(graph_path), cluster_go_map=mapping
            )
        )


def test_partial_ic_coverage(graph_path):
    data = pickle.loads(graph_path.read_bytes())
    data["terms"][Y]["ic"] = None
    graph_path.write_bytes(pickle.dumps(data))
    value, status = scoring.metric(resources.load_graph(graph_path), (X, Y), (X,))
    assert value == 1.0
    assert status["defined_counts"] == [1, 1]
    assert status["source_counts"] == [2, 1]


@pytest.fixture
def cohort(tmp_path, graph_path, map_payload):
    fasta = tmp_path / "queries.fasta"
    fasta.write_text(
        f">q1 go_mf_terms={X}|{Y}\nAAAA\n>q2 go_mf_terms={X}\nCCCC\n"
        f">q3 go_mf_terms={X}\nDDDD\n"
    )
    path = tmp_path / "map.json"
    store.write_json(path, map_payload)
    return fasta, ScoringConfig(graph_path, path)


@pytest.fixture
def backends(tmp_path, cohort, monkeypatch):
    calls = []
    original_graph, original_map = resources.snapshot_graph, resources.snapshot_map
    monkeypatch.setattr(
        resources,
        "snapshot_graph",
        lambda state, path=None: original_graph(state, path or cohort[1].go_graph),
    )
    monkeypatch.setattr(
        resources,
        "snapshot_map",
        lambda state, path=None: original_map(state, path or cohort[1].cluster_go_map),
    )
    monkeypatch.setattr(collect.shutil, "which", lambda name: sys.executable)

    def child(resource, fasta, execution, attempt):
        lines = fasta.read_text().splitlines()
        assert "go_mf_terms" not in fasta.read_text()
        rows = [(lines[i][1:], lines[i + 1]) for i in range(0, len(lines), 2)]
        engine = resource["engine"]
        calls.append((engine, [q for q, _ in rows], resource["contract"]["argv"]))
        if engine == "pclass":
            payload = {"results": [p_result(q) for q, _ in reversed(rows)]}
        else:
            payload = {
                "schema": "pclass-dark.infer-report.v1",
                "queries": [d_result(q, seq, i) for i, (q, seq) in enumerate(rows)],
            }
        path = attempt / "response.json"
        path.write_text(json.dumps(payload))
        return path

    monkeypatch.setattr(collect, "run_child", child)
    return calls, child


def run(tmp_path, cohort, **kwargs):
    return evaluation.evaluate_external(cohort[0], work_dir=tmp_path, **kwargs)


@pytest.mark.parametrize("chunk_size", [1, 2])
def test_collection_logs_percentage(tmp_path, cohort, backends, caplog, chunk_size):
    with caplog.at_level("INFO", logger=collect.LOG.name):
        run(
            tmp_path,
            cohort,
            pclass=PClassConfig(batch_size=chunk_size),
        )
    messages = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("pclass: completed")
    ]
    expected = ["33.3", "66.7", "100.0"] if chunk_size == 1 else ["66.7", "100.0"]
    assert len(messages) == len(expected)
    assert all(
        m.endswith(f"{pct}% finished")
        for m, pct in zip(messages, expected, strict=True)
    )


def test_default_command_pipeline_resume_and_offline(
    tmp_path, cohort, backends, monkeypatch
):
    result = run(tmp_path, cohort)
    table = result.tsv_path.read_bytes()
    assert result.query_count == 3
    assert [c[0] for c in backends[0]] == ["pclass", "dark"]
    assert all(c[2][1:] == ["infer"] for c in backends[0])
    assert table.decode().splitlines()[1] == f"q1\t{X}\t0.2\t0.875\t{Y}\t0.01\t0.875"
    assert run(tmp_path, cohort).score_id == result.score_id
    assert len(backends[0]) == 2

    def fail(*a, **k):
        raise AssertionError("Inference dependency accessed in score-only")

    monkeypatch.setattr(collect, "command_resource", fail)
    monkeypatch.setattr(collect, "run_child", fail)
    again = run(tmp_path, cohort, score_only=True)
    assert again.tsv_path.read_bytes() == table


def test_optional_inference_packages_never_imported(
    tmp_path, cohort, backends, monkeypatch
):
    import builtins

    original = builtins.__import__

    def blocked(name, *a, **k):
        if name.split(".")[0] in {"ijson", "pclass", "torch", "prosig", "transformers"}:
            raise AssertionError(f"Unnecessary import: {name}")
        return original(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    run(tmp_path, cohort)
    run(tmp_path, cohort, score_only=True)


def test_top_neighbor_ignores_alignment_and_voted_terms(graph_path, mapping):
    q, p, d = records()
    p[0].result["go_terms"] = [{"go_id": Z, "name": "voted"}]
    p[0].result["sequence_confirmation"] = {"top_identity_neighbor": "farther"}
    p[0].result["neighbors"].insert(
        0, {"accession": "farther", "distance": 0.4, "go_mf_terms": [Z]}
    )
    row = next(
        evaluation.score_external_predictions(
            q, p, d, graph=resources.load_graph(graph_path), cluster_go_map=mapping
        )
    )
    assert (
        row.pclass_go == (X,) and row.embedding_d == 0.2 and row.pclass_match == 0.875
    )


def test_truth_and_graph_changes_do_not_repeat_inference(tmp_path, cohort, backends):
    first = run(tmp_path, cohort)
    cohort[0].write_text(cohort[0].read_text().replace(f"{X}|{Y}", Y))
    second = run(tmp_path, cohort)
    assert first.score_id != second.score_id and len(backends[0]) == 2
    graph = pickle.loads(cohort[1].go_graph.read_bytes())
    graph["terms"][G]["ic"] = 1
    cohort[1].go_graph.write_bytes(pickle.dumps(graph))
    third = run(tmp_path, cohort)
    assert third.score_id != second.score_id and len(backends[0]) == 2


def test_run_tag_recollects_external_resources(tmp_path, cohort, backends):
    run(tmp_path, cohort)
    run(tmp_path, cohort, execution=ExecutionConfig(run_tag="new-model"))
    assert len(backends[0]) == 4


def test_parallel_score_matches_serial(tmp_path, cohort, backends):
    first = run(tmp_path, cohort, scoring=ScoringConfig(workers=1))
    original = first.tsv_path.read_bytes()
    shutil.rmtree(tmp_path / ".external-evaluation" / "scores")
    (tmp_path / ".external-evaluation" / "selection.json").unlink()
    # Collection receipts are reused; a fresh score directory exercises the pool.
    second = run(tmp_path, cohort, scoring=ScoringConfig(workers=2))
    assert second.tsv_path.read_bytes() == original and len(backends[0]) == 2


@pytest.mark.parametrize("kind", ["raw", "output", "receipt"])
def test_corrupt_completed_results_are_preserved(tmp_path, cohort, backends, kind):
    result = run(tmp_path, cohort)
    state = tmp_path / ".external-evaluation"
    if kind == "output":
        result.tsv_path.write_text("manual change")
    elif kind == "raw":
        next((state / "raw" / "pclass").glob("*/response.json")).write_text("{}")
    else:
        path = next((state / "raw" / "pclass").glob("*/receipt.json"))
        receipt = store.read(path)
        receipt["contract"]["run_tag"] = "tampered"
        store.write_json(path, receipt)
    with pytest.raises(EvaluationIntegrityError):
        run(tmp_path, cohort)
    assert len(backends[0]) == 2


def test_one_engine_failure_resumes(tmp_path, cohort, backends, monkeypatch):
    original = backends[1]

    def failed(resource, *args):
        if resource["engine"] == "dark":
            raise EvaluationInferenceError("simulated failure")
        return original(resource, *args)

    monkeypatch.setattr(collect, "run_child", failed)
    with pytest.raises(EvaluationInferenceError):
        run(tmp_path, cohort)
    assert not (tmp_path / "external_evaluation.tsv").exists()
    monkeypatch.setattr(collect, "run_child", original)
    run(tmp_path, cohort)
    assert [c[0] for c in backends[0]] == ["pclass", "dark"]


def test_size_rejection_splits_and_reuses_leaves(
    tmp_path, cohort, backends, monkeypatch, caplog
):
    caplog.set_level("INFO", logger=collect.LOG.name)
    original = backends[1]
    rejected = []

    def split(resource, fasta, *args):
        if resource["engine"] == "pclass" and fasta.read_text().count(">") > 1:
            rejected.append(1)
            return None
        return original(resource, fasta, *args)

    monkeypatch.setattr(collect, "run_child", split)
    run(tmp_path, cohort)
    messages = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("pclass: completed")
    ]
    assert [m.rsplit("; ", 1)[1] for m in messages] == [
        "33.3% finished",
        "66.7% finished",
        "100.0% finished",
    ]
    assert len(rejected) == 2
    assert len([c for c in backends[0] if c[0] == "pclass"]) == 3
    run(tmp_path, cohort)
    assert len(rejected) == 2


@pytest.mark.parametrize("count,expected", [(1024, [1024]), (1025, [1024, 1])])
def test_pclass_count_boundary(count, expected):
    rows = ({"id": f"q{i}", "sequence": "AAAA"} for i in range(count))
    assert [len(c) for c in collect.plan_chunks(rows, "pclass", 1024)] == expected


def test_pclass_body_bound():
    rows = ({"id": f"q{i}", "sequence": "A" * 1000000} for i in range(10))
    assert [len(c) for c in collect.plan_chunks(rows, "pclass", 128)] == [8, 2]


@pytest.mark.parametrize("bad", ["missing", "extra", "duplicate", "hash", "schema"])
def test_report_validation(tmp_path, bad):
    path = tmp_path / "response.json"
    query = d_result("q1")
    members = [{"id": "q1", "length": 4, "sequence_sha256": query["sequence_sha256"]}]
    payload = {"schema": "pclass-dark.infer-report.v1", "queries": [query]}
    if bad == "missing":
        payload["queries"] = []
    elif bad == "extra":
        payload["queries"].append(d_result("q2"))
    elif bad == "duplicate":
        payload["queries"].append(query)
    elif bad == "hash":
        query["sequence_sha256"] = "0" * 64
    else:
        payload["schema"] = "unknown"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        collect.report_results(path, "dark", members)


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":NaN}', "{}{}"])
def test_strict_json(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        collect.load_report(path)


@pytest.mark.parametrize("kind", ["timeout", "cap", "exit", "size", "not-size"])
def test_real_child_limits_and_errors(tmp_path, kind):
    import time

    scripts = {
        "timeout": "import time; time.sleep(10)",
        "cap": 'print("X" * (2 * 1024**2))',
        "exit": "raise SystemExit(9)",
        "size": (
            'import sys; print("REQUEST_TOO_LARGE: limit",file=sys.stderr); sys.exit(4)'
        ),
        "not-size": (
            'import sys; print("not a size error",file=sys.stderr); sys.exit(4)'
        ),
    }
    resource = {
        "engine": "pclass",
        "contract": {"argv": [sys.executable, "-c", scripts[kind]]},
    }
    execution = ExecutionConfig(
        process_timeout=0.2 if kind == "timeout" else 10, max_raw_chunk_mib=1
    )
    before = time.monotonic()
    if kind == "size":
        assert (
            collect.run_child(resource, tmp_path / "unused", execution, tmp_path)
            is None
        )
    else:
        with pytest.raises((EvaluationInferenceError, EvaluationResourceError)):
            collect.run_child(resource, tmp_path / "unused", execution, tmp_path)
    assert time.monotonic() - before < 5


def test_cli_only_fasta_and_no_attestation(tmp_path, cohort, backends, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, [str(cohort[0])])
    assert result.exit_code == 0, result.output
    assert "Queries: 3" in result.output
    help_result = CliRunner().invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "bundle" not in help_result.output and "ijson" not in help_result.output


def test_preflight_does_no_inference(tmp_path, cohort, backends):
    result = run(tmp_path, cohort, preflight_only=True)
    assert result.query_count == 3 and not backends[0]
    assert not (tmp_path / "external_evaluation.tsv").exists()


def test_publication_recovers_after_interrupt(tmp_path, cohort, backends, monkeypatch):
    original = store.os.replace

    def interrupted(source, target):
        if Path(target) == tmp_path / "external_evaluation.tsv":
            raise OSError("simulated publication failure")
        return original(source, target)

    monkeypatch.setattr(store.os, "replace", interrupted)
    with pytest.raises(OSError):
        run(tmp_path, cohort)
    monkeypatch.setattr(store.os, "replace", original)
    run(tmp_path, cohort)
    assert (tmp_path / "external_evaluation.tsv").is_file() and len(backends[0]) == 2


def test_unowned_output_not_overwritten(tmp_path, cohort, backends):
    (tmp_path / "external_evaluation.tsv").write_text("mine")
    with pytest.raises(EvaluationIntegrityError):
        run(tmp_path, cohort)
    assert not backends[0]


def test_missing_classifier_command(tmp_path, cohort, backends, monkeypatch):
    monkeypatch.setattr(collect.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="PATH"):
        run(tmp_path, cohort)
    assert not backends[0]


def test_recovery_after_marker_commit_uses_no_inference(
    tmp_path, cohort, backends, monkeypatch
):
    original = store.write_json

    def fail(path, data):
        if Path(path).name == "selection.json":
            raise OSError("selection fault")
        original(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(store, "write_json", fail)
        with pytest.raises(OSError, match="selection fault"):
            run(tmp_path, cohort)
    result = evaluation.evaluate_external(cohort[0], work_dir=tmp_path, score_only=True)
    assert result.query_count == 3 and len(backends[0]) == 2
    assert not store.publication_paths(result.tsv_path)[1].exists()


def test_recovery_refuses_corrupt_backup(tmp_path, cohort, backends, monkeypatch):
    run(tmp_path, cohort)
    cohort[0].write_text(cohort[0].read_text().replace(f"{X}|{Y}", Y))
    original = store.write_json

    def fail(path, data):
        if str(path).endswith(".tsv.evaluation.json"):
            raise OSError("marker fault")
        original(path, data)

    with monkeypatch.context() as patch:
        patch.setattr(store, "write_json", fail)
        with pytest.raises(OSError):
            evaluation.evaluate_external(cohort[0], work_dir=tmp_path, score_only=True)
    state = tmp_path / ".external-evaluation"
    output = tmp_path / "external_evaluation.tsv"
    journal = store.read(store.publication_paths(output)[1])
    (state / journal["backup"] / "table").write_text("corrupt")
    with pytest.raises(evaluation.EvaluationRecoveryError):
        store.recover_publication(output, state)
    assert store.publication_paths(output)[1].exists()


def test_external_output_is_staged_on_destination_filesystem(
    tmp_path, cohort, backends, monkeypatch
):
    import errno

    output_dir = tmp_path / "other-filesystem"
    output_dir.mkdir()
    output = output_dir / "result.tsv"
    replace = store.os.replace

    def guarded(source, target):
        if Path(target) == output and Path(source).parent != output_dir:
            raise OSError(errno.EXDEV, "simulated cross-device link")
        return replace(source, target)

    monkeypatch.setattr(store.os, "replace", guarded)
    result = run(tmp_path, cohort, output=output)
    assert result.tsv_path == output and output.exists()


@pytest.mark.parametrize("lock_kind", ["state", "output"])
def test_locks_fail_before_inference(tmp_path, cohort, backends, lock_kind):
    from filelock import FileLock

    run(tmp_path, cohort)
    state = tmp_path / ".external-evaluation"
    lock = (
        state / "lock"
        if lock_kind == "state"
        else tmp_path / ".external_evaluation.tsv.evaluation.lock"
    )
    with FileLock(lock):
        with pytest.raises(RuntimeError, match="busy"):
            evaluation.evaluate_external(cohort[0], work_dir=tmp_path, score_only=True)
    assert len(backends[0]) == 2


def test_path_commands_execute_end_to_end(tmp_path, cohort, monkeypatch):
    """Exercise real PATH subprocesses; no classifier packages/models required."""
    import os

    binary = tmp_path / "bin"
    binary.mkdir()
    code = """import json,sys,hashlib
from pathlib import Path
lines=Path(sys.argv[sys.argv.index('--fasta')+1]).read_text().splitlines()
rows=[(lines[i][1:],lines[i+1]) for i in range(0,len(lines),2)]
if Path(sys.argv[0]).name == 'pclass':
    payload={'results':[{'id':q,'neighbors':[{'accession':'ref','distance':0.2,
        'go_mf_terms':['GO:0000001']}]} for q,s in rows]}
else:
    payload={'schema':'pclass-dark.infer-report.v1','queries':[
        {'schema':'pclass-dark.infer-query.v1','query_id':q,'length':len(s),
         'sequence_sha256':hashlib.sha256(s.encode()).hexdigest(),
         'status':'matches','matches':[{'accepted':True,'cluster_id':'C1',
         'background_tail_fraction':0.01}]} for q,s in rows]}
print(json.dumps(payload))
"""
    for name in ["pclass", "pclass-dark"]:
        path = binary / name
        path.write_text(f"#!{sys.executable}\n" + code)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", str(binary) + os.pathsep + os.environ["PATH"])
    result = evaluation.evaluate_external(
        cohort[0],
        work_dir=tmp_path,
        scoring=cohort[1],
        pclass=PClassConfig(batch_size=2),
        dark=DarkConfig(chunk_size=2),
    )
    assert result.query_count == 3
    assert result.tsv_path.read_text().splitlines()[1] == (
        f"q1\t{X}\t0.2\t0.875\t{Y}\t0.01\t0.875"
    )
    manifests = list((tmp_path / ".external-evaluation/raw").glob("*/*/receipt.json"))
    assert len(manifests) == 4


@pytest.mark.parametrize("kind", ["truth", "unknown", "duplicate", "alphabet"])
def test_invalid_cohort_prevents_collection(tmp_path, cohort, backends, kind):
    text = cohort[0].read_text()
    if kind == "truth":
        text = text.replace(f" go_mf_terms={X}|{Y}", "")
    elif kind == "unknown":
        text = text.replace(Y, "GO:9999999")
    elif kind == "duplicate":
        text = text.replace(">q2 ", ">q1 ")
    else:
        text = text.replace("AAAA", "AAJA")
    cohort[0].write_text(text)
    with pytest.raises(ValueError):
        run(tmp_path, cohort)
    assert not backends[0]


def test_score_only_rejects_explicit_inference_flags(
    tmp_path, cohort, backends, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(
        app, [str(cohort[0]), "--score-only", "--run-tag", "new"]
    )
    assert result.exit_code == 2 and not backends[0]


def test_unknown_dark_cluster_blocks_scoring(graph_path, mapping):
    q, p, d = records()
    d[0].result["matches"][0]["cluster_id"] = "unknown"
    with pytest.raises(ValueError, match="GO mapping"):
        list(
            evaluation.score_external_predictions(
                q, p, d, graph=resources.load_graph(graph_path), cluster_go_map=mapping
            )
        )


def test_grandchild_cleanup(tmp_path):
    import os
    import time

    pidfile = tmp_path / "pid"
    code = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
        f"Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)"
    )
    resource = {"engine": "pclass", "contract": {"argv": [sys.executable, "-c", code]}}
    with pytest.raises(EvaluationInferenceError, match="timeout"):
        collect.run_child(
            resource, tmp_path / "unused", ExecutionConfig(1, 1), tmp_path
        )
    pid = int(pidfile.read_text())
    for _ in range(20):
        try:
            status = Path(f"/proc/{pid}/stat").read_text().split()[2]
        except FileNotFoundError:
            break
        if status == "Z":
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        pytest.fail("Grandchild survived timeout")


def test_dark_override_reuses_pclass_output(tmp_path, cohort, backends):
    run(tmp_path, cohort)
    run(tmp_path, cohort, dark=DarkConfig(device="cpu"))
    assert [row[0] for row in backends[0]] == ["pclass", "dark", "dark"]
    assert backends[0][-1][2][-2:] == ["--device", "cpu"]


@pytest.mark.parametrize(
    "overrides,expected",
    [
        ([], (512, 512, 8)),
        (
            [
                "--pclass-batch-size",
                "32",
                "--dark-chunk-size",
                "64",
                "--score-workers",
                "1",
            ],
            (32, 64, 1),
        ),
    ],
)
def test_cli_and_api_processing_defaults(monkeypatch, overrides, expected):
    assert (
        PClassConfig().batch_size,
        DarkConfig().chunk_size,
        ScoringConfig().workers,
    ) == (512, 512, 8)
    captured = {}

    def stub(fasta, **kwargs):
        captured.update(kwargs)
        return evaluation.EvaluationPreflight("input", "truth", 1, {}, "graph", "map")

    monkeypatch.setattr(evaluation, "evaluate_external", stub)
    result = CliRunner().invoke(app, ["input.fasta", *overrides])
    assert result.exit_code == 0, result.output
    assert (
        captured["pclass"].batch_size,
        captured["dark"].chunk_size,
        captured["scoring"].workers,
    ) == expected


@pytest.mark.parametrize(
    "truth,prediction,expected",
    [
        ((X,), (X, "GO:0018491"), 1.0),
        ((X, "GO:0018491"), (X,), 1.0),
        (("GO:0018491",), (X,), None),
        ((X,), ("GO:0018491",), None),
    ],
)
def test_unknown_metric_terms_warn(graph_path, truth, prediction, expected):
    with pytest.warns(UserWarning, match="Unknown evaluation MF terms.*GO:0018491"):
        value, status = scoring.metric(
            resources.load_graph(graph_path), truth, prediction
        )
    assert value == expected
    assert status["source_counts"] == [len(truth), len(prediction)]
    assert status["missing_terms"] == [
        [t for t in truth if t == "GO:0018491"],
        [t for t in prediction if t == "GO:0018491"],
    ]
    assert status["status"] == ("defined" if expected is not None else "unavailable")
