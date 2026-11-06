"""SPEC-0017 scientific parity, native persistence and bounded lifecycle tests."""

import json
from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from pclass_dark import external_evaluation_store as store
from pclass_dark import inference as infer
from pclass_dark import inference_runtime as resident
from pclass_dark import swiss_evaluation as swiss
from pclass_dark import swiss_evaluation_collect as collect
from pclass_dark import swiss_evaluation_truth as truth
from pclass_dark.artifacts import IntegrityError, identity, write_json
from pclass_dark.external_audit import read_audit_table, read_query_annotations
from pclass_dark.external_evaluation_types import (
    EvaluationInferenceError,
    EvaluationIntegrityError,
    EvaluationResourceError,
    ScoringConfig,
)
from pclass_dark.inference_records import QuerySequence
from pclass_dark.library import build_library
from pclass_dark.reference_data import read_fasta
from pclass_dark.swiss_evaluation_cli import app
from pclass_dark.swiss_evaluation_types import (
    SCHEMAS,
    SwissDarkConfig,
    SwissExecutionConfig,
    SwissPClassConfig,
)
from test_external_evaluation import graph_path as graph_path  # noqa: F401

X, Y = "GO:0000001", "GO:0000002"


@pytest.fixture
def runtime_data(library_data, tmp_path):
    data, prepared, ep, matrix, contract = library_data()
    # Older fixture manifests omit this optional provenance field.
    manifest = json.loads((ep / "manifest.json").read_text())
    manifest["originating_preparation_id"] = prepared.preparation_id
    write_json(ep / "manifest.json", manifest)
    selection = json.loads((data / "selection.json").read_text())
    selection["embedding_manifest_sha256"] = store.record(ep / "manifest.json")[
        "sha256"
    ]
    write_json(data / "selection.json", selection)
    root = tmp_path / "library"
    build_library(data, root)
    queries = [
        QuerySequence(f"P{i:05d}", sequence)
        for i, (_, sequence) in enumerate(
            read_fasta(prepared.prepared_dir / "sequences.fasta")
        )
    ]
    return root, prepared.prepared_dir, ep, matrix, contract, queries


def test_resident_native_parity_and_one_initialization(runtime_data, monkeypatch):
    root, _, _, _, contract, _ = runtime_data
    calls = []

    @contextmanager
    def backend(*args, **kwargs):
        calls.append("initialize")

        class Model:
            def embed(self, seqs):
                result = np.zeros((len(seqs), 640), dtype="<f4")
                result[:, :2] = 0.009
                return result

        model = Model()
        model.contract = contract
        yield model, {"device": "cpu"}

    monkeypatch.setattr(infer, "_backend_context", backend)
    monkeypatch.setattr(resident, "backend_context", backend)
    queries = [QuerySequence("P00001", "A"), QuerySequence("P00002", "A" * 1023)]
    with infer.infer_queries(queries, library_dir=root) as baseline:
        expected = [r.to_dict() for r in baseline]
    calls.clear()
    with resident.DarkInferenceRuntime(root, device="cpu") as runtime:
        actual = []
        for q in queries:
            with runtime.infer_chunk([q]) as session:
                result = next(session).to_dict()
                result["query_index"] = len(actual)
                actual.append(result)
        assert calls == ["initialize"]
        assert runtime.execution["device"] == "cpu"
        with runtime.infer_chunk([queries[0]]):
            with pytest.raises(RuntimeError, match="previous|Previous"):
                runtime.infer_chunk([queries[1]])
    assert actual == expected


def test_cache_all_hit_avoids_model_and_matches_low_level(
    runtime_data, tmp_path, monkeypatch
):
    root, prepared, ep, matrix, contract, queries = runtime_data

    def forbidden(*args, **kwargs):
        raise AssertionError("all-hit run tried to create a model")

    monkeypatch.setattr(resident, "backend_context", forbidden)
    with resident.DarkInferenceRuntime(
        root, embedding_cache=ep, prepared_dir=prepared, _index_dir=tmp_path / "index"
    ) as runtime:
        assert runtime.inspect_queries(queries) == {
            "windows": len(queries),
            "cache_hits": len(queries),
            "cache_misses": 0,
        }
        windows = tuple(
            infer.make_query_windows(q, query_index=i) for i, q in enumerate(queries)
        )
        expected = infer.score_query_windows(
            runtime.pair, windows, matrix, embedding_contract=contract
        )
        with runtime.infer_chunk(queries) as session:
            actual = tuple(session)
        assert actual == expected
        assert runtime.execution["embedding_device"] == "not_used"
        assert runtime.execution["cache_hits"] == len(queries)
        assert list((tmp_path / "index").glob("*/index.sqlite"))


@pytest.mark.parametrize("corrupt", ["matrix", "contract", "coordinates"])
def test_cache_corruption_fails_closed(runtime_data, tmp_path, corrupt):
    root, prepared, ep, _, _, _ = runtime_data
    if corrupt == "matrix":
        with (ep / "embeddings.npy").open("ab") as f:
            f.write(b"changed")
    elif corrupt == "contract":
        m = json.loads((ep / "manifest.json").read_text())
        m["identity_inputs"]["contract"]["pooling"] = "wrong"
        m["id"] = identity(m["identity_inputs"])
        write_json(ep / "manifest.json", m)
    else:
        p = prepared / "windows.tsv"
        p.write_text(p.read_text().replace("\tfull\t1\t", "\tfull\t2\t", 1))
    with pytest.raises((IntegrityError, ValueError)):
        with resident.DarkInferenceRuntime(
            root,
            embedding_cache=ep,
            prepared_dir=prepared,
            _index_dir=tmp_path / "index",
        ):
            pass


def test_runtime_budget_and_generation_change(runtime_data):
    root, _, _, _, _, queries = runtime_data
    with pytest.raises(infer.InferenceResourceError, match="workspace"):
        with resident.DarkInferenceRuntime(
            root, memory_mib=128, score_batch_size=65536
        ):
            pass
    with resident.DarkInferenceRuntime(root) as runtime:
        p = root / "selection.json"
        p.write_text(p.read_text() + " ")
        with pytest.raises(IntegrityError, match="changed"):
            with runtime.infer_chunk(queries):
                pass


@pytest.mark.parametrize("qid", ["P00001", "sp|P00001|ENTRY"])
def test_primary_accession_and_full_truth(qid):
    terms = ("GO:0003674", "GO:0005515", X)
    resolve = truth.TruthResolver({"P00001": terms})
    assert resolve(qid, None) == tuple(sorted(terms))
    with pytest.raises(ValueError, match="Duplicate normalized"):
        resolve("P00001", None)


@pytest.mark.parametrize(
    "qid", ["tr|P00001|ENTRY", "sp|P00001|", "P00001-2", "query", "sp|bad|X"]
)
def test_invalid_accession(qid):
    with pytest.raises(ValueError):
        truth.primary_accession(qid)


def test_missing_and_conflicting_truth():
    with pytest.raises(ValueError):
        truth.TruthResolver({})("P00001", None)
    with pytest.raises(ValueError, match="Conflicting"):
        truth.TruthResolver({"P00001": (X,)})("P00001", (Y,))


@pytest.mark.parametrize(
    "cls,kw",
    [
        (SwissPClassConfig, {"batch_size": True}),
        (SwissPClassConfig, {"timeout_seconds": float("nan")}),
        (SwissPClassConfig, {"server_url": "https://user:secret@localhost"}),
        (SwissDarkConfig, {"embedding_cache": "missing"}),
        (SwissDarkConfig, {"batch_size": 65}),
        (SwissDarkConfig, {"offline": 1}),
        (SwissExecutionConfig, {"process_timeout": 0}),
    ],
)
def test_invalid_configs(cls, kw):
    with pytest.raises(ValueError):
        cls(**kw)


@pytest.fixture
def pipeline(tmp_path, graph_path, monkeypatch):
    fasta = tmp_path / "queries.fasta"
    fasta.write_text(">P00001\nAAAA\n>sp|P00002|ENTRY\nCCCC\n>P00003\nDDDD\n")
    annotation = {"id": "fixture-annotations-v1"}
    monkeypatch.setattr(
        truth,
        "freeze_annotations",
        lambda state: (
            dict(annotation),
            {"P00001": (X,), "P00002": (X,), "P00003": (X,)},
            SimpleNamespace(dataset=graph_path.parent),
        ),
    )
    payload = {"schema_version": 1, "clusters": {"C1": {"go_mf_terms": [Y]}}}
    payload["mapping_id"] = identity(payload)
    map_path = tmp_path / "mapping.json"
    write_json(map_path, payload)
    import pclass_dark.swiss_evaluation_worker as worker

    monkeypatch.setattr(worker, "pclass_contract", lambda: {"version": "fixture"})
    calls = []

    class FakeWorker:
        def __init__(self, engine, config, state, execution, mapping_clusters):
            self.engine = engine
            self.contract = {
                "format": SCHEMAS[engine],
                "engine": engine,
                "config": asdict(config),
                "tag": execution.run_tag,
            }
            self.contract_id = identity(self.contract)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def task(self, rows, attempt, inspect_only=False):
            if inspect_only:
                return {
                    "details": {
                        "windows": len(rows),
                        "cache_hits": 0,
                        "cache_misses": len(rows),
                    }
                }
            calls.append((self.engine, len(rows)))
            if self.engine == "pclass" and len(rows) > 1:
                return {"split": True}
            if self.engine == "pclass":
                payload = {
                    "results": [
                        {
                            "id": r["id"],
                            "neighbors": [
                                {
                                    "accession": "P12345",
                                    "distance": 0.2,
                                    "confidence": 0.8,
                                    "go_mf_terms": [X],
                                }
                            ],
                        }
                        for r in reversed(rows)
                    ]
                }
            else:
                from pclass_dark.inference_records import (
                    combine_matches,
                    make_query_windows,
                )

                payload = []
                for i, r in enumerate(rows):
                    q = make_query_windows(
                        QuerySequence(r["id"], r["sequence"]), query_index=i
                    )
                    result = combine_matches(
                        i,
                        q.query_id,
                        q.length,
                        q.sequence_sha256,
                        tuple(w.info for w in q.windows),
                        [("C1", 0, 0.1, 0.5)],
                        {"C1": {"C_n": 10, "B_n": 20}},
                    )
                    payload.append(result.to_dict())
            path = attempt / "response.json"
            path.write_text(
                json.dumps(
                    {
                        "schema": SCHEMAS[self.engine],
                        "source": self.contract,
                        "execution": {},
                        "members": collect.members(rows),
                        "payload": payload,
                    }
                )
            )
            return {"path": str(path), "record": store.record(path)}

    monkeypatch.setattr(collect, "NativeWorker", FakeWorker)
    return fasta, ScoringConfig(graph_path, map_path, 1), annotation, calls


def run_pipeline(tmp_path, pipeline, **kw):
    fasta, sc, _, _ = pipeline
    return swiss.evaluate_swiss(fasta, work_dir=tmp_path, scoring=sc, **kw)


def test_pipeline_analytical_scores_resume_offline_and_audit(
    tmp_path, pipeline, monkeypatch
):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    output = output_dir / "swiss.tsv"
    result = run_pipeline(tmp_path, pipeline, output=output)
    rows = result.tsv_path.read_text().splitlines()
    # X/Y share G (IC 2), each has IC 4: Lin = 2*2/(4+4) = 0.5.
    assert rows[1].split("\t") == ["P00001", X, "0.2", "1.0", Y, "0.1", "0.5"]
    assert len(rows) == 4
    assert run_pipeline(tmp_path, pipeline, output=output).score_id == result.score_id
    assert pipeline[3] == [
        ("dark", 3),
        ("pclass", 3),
        ("pclass", 1),
        ("pclass", 2),
        ("pclass", 1),
        ("pclass", 1),
    ]
    data = read_audit_table(output)
    assert read_query_annotations(output, data) == {
        qid: (X,) for qid in ("P00001", "sp|P00002|ENTRY", "P00003")
    }
    old_import = __import__("builtins").__import__

    def blocked(name, *args, **kwargs):
        if name.startswith(("pclass.", "torch", "profunc.runtime")):
            raise AssertionError(f"Offline operation imported {name}")
        return old_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    monkeypatch.setattr(
        truth, "freeze_annotations", lambda state: pytest.fail("live truth")
    )
    assert (
        run_pipeline(tmp_path, pipeline, output=output, score_only=True).score_id
        == result.score_id
    )


def test_truth_provenance_change_reuses_predictions(tmp_path, pipeline):
    first = run_pipeline(tmp_path, pipeline)
    count = len(pipeline[3])
    pipeline[2]["id"] = "fixture-annotations-v2"
    second = run_pipeline(tmp_path, pipeline)
    assert first.truth_id != second.truth_id and first.score_id != second.score_id
    assert len(pipeline[3]) == count
    assert first.pclass_generation == second.pclass_generation


def test_preflight_does_not_collect_or_publish(tmp_path, pipeline):
    result = run_pipeline(tmp_path, pipeline, preflight_only=True)
    assert result.query_count == 3
    assert result.commands["dark"]["windows"] == 3
    assert not pipeline[3]
    assert not (tmp_path / "swiss_evaluation.tsv").exists()
    assert not (tmp_path / ".swiss-evaluation" / "selection.json").exists()


@pytest.mark.parametrize("kind", ["raw", "receipt", "output"])
def test_corrupt_published_state_is_refused(tmp_path, pipeline, kind):
    result = run_pipeline(tmp_path, pipeline)
    if kind == "output":
        result.tsv_path.write_text("manual edit")
    else:
        path = next(
            (tmp_path / ".swiss-evaluation" / "raw").glob(
                f"*/*/{'response.json' if kind == 'raw' else 'receipt.json'}"
            )
        )
        path.write_text("corrupt")
    with pytest.raises(EvaluationIntegrityError):
        run_pipeline(tmp_path, pipeline)


def test_cli_help_and_inference_option_rejection():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"], terminal_width=200, env={"COLUMNS": "200"})
    assert result.exit_code == 0 and "--dark-embedding-cache" in result.stdout
    result = runner.invoke(app, ["missing.fa", "--score-only", "--device", "cpu"])
    assert result.exit_code != 0 and "Score-only" in result.output


def test_cache_source_change_and_input_change_refused_offline(tmp_path, pipeline):
    run_pipeline(tmp_path, pipeline)
    pipeline[0].write_text(pipeline[0].read_text().replace("AAAA", "AAAW"))
    with pytest.raises(ValueError, match="Input does not match"):
        run_pipeline(tmp_path, pipeline, score_only=True)


@pytest.fixture
def neighbor_server():
    """Local protocol fixture; no real model or remote service is exercised."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    calls = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, value):
            encoded = json.dumps(value).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            self.respond(200, {"max_request_bytes": 10 * 2**20})

        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append(data)
            if len(data["sequences"]) > 1:
                self.respond(413, {"error": "fixture size limit"})
                return
            results = []
            for q in data["sequences"]:
                results.append(
                    {
                        "id": q["id"],
                        "normalization": {
                            "input_length": len(q["sequence"]),
                            "normalized_length": len(q["sequence"]),
                            "ignored_characters": {
                                "whitespace": {"count": 0},
                                "other": {"count": 0, "symbols": []},
                            },
                        },
                        "neighbors": [
                            {
                                "accession": "P12345",
                                "distance": 0.2,
                                "confidence": 0.8,
                                "go_mf_terms": [X],
                            }
                        ],
                    }
                )
            self.respond(200, {"results": results})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_spawned_pclass_worker_splits_and_reuses(tmp_path, neighbor_server):
    pytest.importorskip("pclass.query.client")
    url, calls = neighbor_server
    state = tmp_path / "state"
    state.mkdir()
    fasta = tmp_path / "input.fa"
    fasta.write_text(f">P00001 go_mf_terms={X}\nAAAA\n>P00002 go_mf_terms={X}\nCCCC\n")
    cohort = store.snapshot_inputs(fasta, None, state)
    config = SwissPClassConfig(server_url=url)
    with collect.NativeWorker(
        "pclass", config, state, SwissExecutionConfig(), []
    ) as worker:
        generation = collect.collect_generation(state, cohort, worker, 512)
        assert len(generation["chunks"]) == 2
        results = list(collect.saved_queries(state, "pclass", generation))
        assert [r.query_id for r in results] == ["P00001", "P00002"]
        assert collect.collect_generation(state, cohort, worker, 512) == generation
    assert len(calls) == 3
    assert "go_mf_terms" not in json.dumps([q["sequences"] for q in calls])


def test_spawned_dark_all_cache_and_offline_evidence(runtime_data, tmp_path):
    root, prepared, ep, _, _, queries = runtime_data
    state = tmp_path / "state"
    state.mkdir()
    config = SwissDarkConfig(
        library_dir=root, embedding_cache=ep, prepared_dir=prepared
    )
    rows = [
        {
            "ordinal": i,
            "id": q.query_id,
            "sequence": q.sequence,
            "length": len(q.sequence),
            "sequence_sha256": __import__("hashlib")
            .sha256(q.sequence.encode())
            .hexdigest(),
        }
        for i, q in enumerate(queries[:2])
    ]
    with collect.NativeWorker(
        "dark", config, state, SwissExecutionConfig(), ["K0", "K1"]
    ) as worker:
        for row in rows:
            attempt = state / row["id"]
            attempt.mkdir()
            reply = worker.task([row], attempt)
            assert reply["path"] == str(attempt / "response.json")
            envelope = json.loads((attempt / "response.json").read_text())
            assert envelope["execution"]["embedding_device"] == "not_used"
            assert envelope["execution"]["cache_hits"] == 1
            assert (
                len(
                    collect.evidence(
                        attempt / "response.json", "dark", [row], worker.contract
                    )
                )
                == 1
            )


def test_worker_deadline_and_process_cleanup(tmp_path):
    worker = collect.NativeWorker(
        "pclass",
        SwissPClassConfig(),
        tmp_path,
        SwissExecutionConfig(process_timeout=0.001),
        [],
    )
    with pytest.raises(EvaluationResourceError, match="timed out"):
        with worker:
            pass
    assert worker.process is None and worker.connection is None


def test_worker_initialization_error(tmp_path):
    worker = collect.NativeWorker(
        "dark",
        SwissDarkConfig(library_dir=tmp_path / "missing"),
        tmp_path,
        SwissExecutionConfig(),
        [],
    )
    with pytest.raises(EvaluationInferenceError):
        with worker:
            pass
    assert worker.process is None


def test_cache_detects_mutation_and_model_misses(runtime_data, monkeypatch):
    root, prepared, ep, _, contract, queries = runtime_data
    creations, forwards = [], []

    @contextmanager
    def backend(*args, **kwargs):
        creations.append(True)

        class Backend:
            def embed(self, sequences):
                forwards.extend(sequences)
                return np.zeros((len(sequences), 640), dtype="<f4")

        model = Backend()
        model.contract = contract
        yield model, {"device": "cpu"}

    monkeypatch.setattr(resident, "backend_context", backend)
    with resident.DarkInferenceRuntime(
        root, embedding_cache=ep, prepared_dir=prepared
    ) as runtime:
        with runtime.infer_chunk([queries[0], QuerySequence("P99991", "A")]) as session:
            assert len(list(session)) == 2
            assert (
                session.execution["cache_hits"]
                == session.execution["cache_misses"]
                == 1
            )
        with runtime.infer_chunk([QuerySequence("P99992", "W")]) as session:
            assert len(list(session)) == 1
        assert len(creations) == 1 and forwards == ["A", "W"]
        with (ep / "embeddings.npy").open("ab") as stream:
            stream.write(b"mutated")
        with pytest.raises(IntegrityError, match="sources changed"):
            with runtime.infer_chunk([queries[0]]):
                pass


@pytest.mark.parametrize("kind", ["duplicate", "junk", "empty"])
def test_malformed_annotation_mapping(tmp_path, kind):
    path = tmp_path / "accession_mf_go.tsv"
    path.write_text(
        {
            "duplicate": f"P00001\t{X}\nP00001\t{Y}\n",
            "junk": f"P00001\tgarbage {X}\n",
            "empty": "P00001\t\n",
        }[kind]
    )
    with pytest.raises(ValueError):
        truth.validate_annotation_rows(path)


def test_bounded_missing_annotation_diagnostics(tmp_path):
    path = tmp_path / "input.fa"
    path.write_text("".join(f">P{i:05d}\nA\n" for i in range(100)))
    with pytest.raises(ValueError, match=r"\(100\)") as error:
        truth.validate_headers(path, {})
    assert str(error.value).count("missing MF truth") == 20


def test_metric_cache_is_bounded_and_graph_scoped(graph_path, monkeypatch):
    from pclass_dark import external_evaluation_resources as resources
    from pclass_dark import external_evaluation_scoring as scoring

    calls = []
    original = scoring.metric

    def metric(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(scoring, "metric", metric)
    first = scoring.cached_metric(resources.load_graph(graph_path))
    second = scoring.cached_metric(resources.load_graph(graph_path))
    assert first((X,), (Y,))[0] == 0.5
    assert first((X,), (Y,))[0] == 0.5
    assert second((X,), (Y,))[0] == 0.5
    assert len(calls) == 2
    assert first.cache_info().maxsize == 4096


@pytest.mark.parametrize("corrupt", ["absolute", "uuid", "symlink", "kind"])
def test_audit_marker_v2_path_and_identity_checks(tmp_path, pipeline, corrupt):
    from pclass_dark.external_audit import read_query_annotations

    result = run_pipeline(tmp_path, pipeline)
    marker_path, _ = store.publication_paths(result.tsv_path)
    marker = store.read(marker_path)
    if corrupt == "absolute":
        marker["state_relative"] = str(tmp_path / ".swiss-evaluation")
    elif corrupt == "uuid":
        marker["workspace_uuid"] = "another workspace"
    elif corrupt == "kind":
        marker["evaluation_kind"] = "unknown"
    else:
        (tmp_path / "linked-state").symlink_to(
            tmp_path / ".swiss-evaluation", target_is_directory=True
        )
        marker["state_relative"] = "linked-state"
    store.write_json(marker_path, marker)
    with pytest.raises(ValueError):
        read_query_annotations(result.tsv_path, read_audit_table(result.tsv_path))


def test_parallel_swiss_scoring_matches_serial(tmp_path, pipeline):
    first = run_pipeline(tmp_path, pipeline)
    expected = first.tsv_path.read_bytes()
    sc = pipeline[1]
    second = swiss.evaluate_swiss(
        pipeline[0],
        work_dir=tmp_path,
        scoring=ScoringConfig(sc.go_graph, sc.cluster_go_map, 2),
    )
    assert second.tsv_path.read_bytes() == expected


def test_failure_after_committed_dark_resumes(tmp_path, pipeline, monkeypatch):
    original = collect.NativeWorker.task

    def fail(self, *args, **kwargs):
        if self.engine == "pclass":
            raise EvaluationInferenceError("fixture failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(collect.NativeWorker, "task", fail)
    with pytest.raises(EvaluationInferenceError, match="fixture failure"):
        run_pipeline(tmp_path, pipeline)
    assert pipeline[3] == [("dark", 3)]
    assert not (tmp_path / "swiss_evaluation.tsv").exists()
    monkeypatch.setattr(collect.NativeWorker, "task", original)
    assert run_pipeline(tmp_path, pipeline).query_count == 3
    assert [engine for engine, _ in pipeline[3]].count("dark") == 1


def test_publication_interruption_recovers_swiss_marker(
    tmp_path, pipeline, monkeypatch
):
    original = store.write_json
    crashed = []

    def crash(path, value):
        if str(path).endswith(".tsv.evaluation.json") and not crashed:
            crashed.append(True)
            raise OSError("fixture interruption")
        return original(path, value)

    monkeypatch.setattr(store, "write_json", crash)
    with pytest.raises(OSError, match="fixture interruption"):
        run_pipeline(tmp_path, pipeline)
    assert list(tmp_path.glob(".*evaluation-journal.json"))
    calls = len(pipeline[3])
    result = run_pipeline(tmp_path, pipeline)
    assert result.query_count == 3 and len(pipeline[3]) == calls
    assert not list(tmp_path.glob(".*evaluation-journal.json"))
    assert (
        store.read(store.publication_paths(result.tsv_path)[0])["marker_version"] == 2
    )


def test_native_neighbor_only_has_no_infer_imports(monkeypatch):
    pytest.importorskip("pclass.query.client")
    import builtins

    import pclass.query.client

    from pclass_dark.swiss_evaluation_worker import neighbors

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith(("pclass.infer", "prosig", "pclass.sequence_identity")):
            raise AssertionError(f"Unneeded inference import: {name}")
        return original(name, *args, **kwargs)

    def request(records, **kwargs):
        assert len(records) == 1 and records[0].sequence == "AAAA"
        assert "max_neighbors" not in kwargs and "max_distance" not in kwargs
        return SimpleNamespace(model_dump=lambda **kw: {"results": []})

    monkeypatch.setattr(pclass.query.client, "request_neighbors", request)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert neighbors(
        [{"id": "P00001", "sequence": "AAAA"}], asdict(SwissPClassConfig())
    ) == {"results": []}


def test_native_worker_evidence_cap(tmp_path):
    worker = collect.NativeWorker(
        "dark",
        SwissDarkConfig(),
        tmp_path,
        SwissExecutionConfig(max_raw_chunk_mib=1),
        [],
    )
    (tmp_path / "response.json").write_bytes(b" " * (2**20 + 1))
    with pytest.raises(EvaluationResourceError, match="cap exceeded"):
        worker._receive("done", tmp_path)


@pytest.mark.parametrize("kind", ["payload", "confidence", "missing", "duplicate"])
def test_native_response_validation(tmp_path, pipeline, kind):
    from pclass_dark.external_evaluation_collect import input_rows

    run_pipeline(tmp_path, pipeline)
    state = tmp_path / ".swiss-evaluation"
    root = next((state / "raw" / "pclass").iterdir())
    receipt = store.read(root / "receipt.json")
    data = json.loads((root / "response.json").read_text())
    ids = {m["id"] for m in receipt["members"]}
    selection = store.read(state / "selection.json")
    rows = [r for r in input_rows(state, selection["input_id"]) if r["id"] in ids]
    if kind == "payload":
        data["payload"] = data["payload"]["results"]
    elif kind == "confidence":
        data["payload"]["results"][0]["neighbors"][0]["confidence"] = 1.1
    elif kind == "missing":
        data["payload"]["results"] = []
    else:
        data["payload"]["results"] *= 2
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(data))
    with pytest.raises((EvaluationIntegrityError, ValueError)):
        collect.evidence(path, "pclass", rows, receipt["contract"])


def test_cached_runtime_limits_and_restores_numerical_threads(runtime_data):
    from threadpoolctl import threadpool_info

    root, prepared, ep, _, _, queries = runtime_data
    before = {p["filepath"]: p["num_threads"] for p in threadpool_info()}
    with resident.DarkInferenceRuntime(
        root, workers=1, embedding_cache=ep, prepared_dir=prepared
    ) as runtime:
        assert runtime.cpu_threads == 1
        assert all(p["num_threads"] == 1 for p in threadpool_info())
        with runtime.infer_chunk(queries[:1]) as session:
            assert len(list(session)) == 1
            assert session.execution["cpu_threads"] == 1
            assert session.execution["embedding_device"] == "not_used"
    assert {p["filepath"]: p["num_threads"] for p in threadpool_info()} == before


@pytest.mark.parametrize("changed", ["scoring", "config", "unreviewed", "corrupt"])
def test_resume_completed_before_publication(tmp_path, pipeline, monkeypatch, changed):
    from pclass_dark import external_evaluation as shared
    from pclass_dark.swiss_evaluation_resume import _SCORING_ONLY_EQUIVALENTS

    old, new = sorted(_SCORING_ONLY_EQUIVALENTS)
    original_init = collect.NativeWorker.__init__
    version = [old]

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.contract["code"] = {"pclass_dark.external_evaluation_scoring": version[0]}
        self.contract_id = identity(self.contract)

    monkeypatch.setattr(collect.NativeWorker, "__init__", init)
    original_score = shared._score_and_publish

    def fail(*args, **kwargs):
        raise RuntimeError("scoring interrupted")

    monkeypatch.setattr(shared, "_score_and_publish", fail)
    with pytest.raises(RuntimeError, match="scoring interrupted"):
        run_pipeline(tmp_path, pipeline)
    state = tmp_path / ".swiss-evaluation"
    assert not (state / "selection.json").exists()
    calls = list(pipeline[3])
    version[0] = new if changed != "unreviewed" else "unreviewed-code-change"
    monkeypatch.setattr(shared, "_score_and_publish", original_score)
    if changed == "corrupt":
        path = next((state / "raw" / "dark").glob("*/response.json"))
        path.write_text("corrupt")
        with pytest.raises(
            EvaluationIntegrityError, match="corrupt committed artifact"
        ):
            run_pipeline(tmp_path, pipeline)
        assert pipeline[3] == calls
        return
    kwargs = {}
    if changed == "config":
        kwargs["execution"] = SwissExecutionConfig(run_tag="changed")
    result = run_pipeline(tmp_path, pipeline, **kwargs)
    assert result.tsv_path.exists()
    if changed == "scoring":
        assert pipeline[3] == calls
    else:
        assert len(pipeline[3]) > len(calls)
