"""Window/evidence oracles and resource-isolated query inference acceptance."""

import io
import json
import shutil
from contextlib import contextmanager

import numpy as np
import pytest
from typer.testing import CliRunner

from pclass_dark.artifacts import IntegrityError, write_json
from pclass_dark.cli import app
from pclass_dark.inference import (
    InferenceResourceError,
    QuerySequence,
    infer_queries,
    make_query_windows,
    read_query_fasta,
    score_query_windows,
    write_inference_report,
)
from pclass_dark.library import build_library
from pclass_dark.relative_scores import load_selected_library

_template = None


@pytest.fixture
def query_library(library_data, tmp_path, tmp_path_factory):
    global _template
    if _template is None:
        data, _, _, _, contract = library_data()
        original = tmp_path / "original-library"
        build_library(data, original)
        template = tmp_path_factory.mktemp("inference-library")
        shutil.copytree(original, template / "library")
        _template = template / "library", contract
    source, contract = _template
    root = tmp_path / "library"
    shutil.copytree(source, root)
    return root, contract


@pytest.fixture
def fake_model(monkeypatch):
    import pclass_dark.inference as infer

    calls = []

    @contextmanager
    def context(contract, **options):
        class Backend:
            def embed(self, sequences):
                calls.append(tuple(sequences))
                matrix = np.zeros((len(sequences), 640), dtype="<f4")
                for i, sequence in enumerate(sequences):
                    matrix[i, :2] = {"A": 0.009, "W": 10.010, "Z": 5.0}.get(
                        sequence[0], 0.009
                    )
                return matrix

        backend = Backend()
        backend.contract = contract
        yield (
            backend,
            {
                "requested_device": options["device"],
                "device": "cpu",
                "fallback_reason": None,
                "workers_requested": options["workers"],
                "workers_used": options["workers"],
                "model": contract["model"],
                "revision": contract["revision"],
                "seed": 42,
                "inference_dtype": "float32",
                "torch_version": "fake",
                "transformers_version": "fake",
            },
        )

    monkeypatch.setattr(infer, "_backend_context", context)
    return calls


@pytest.mark.parametrize(
    "length,coordinates",
    [
        (1, [(1, 1)]),
        (1022, [(1, 1022)]),
        (1023, [(1, 1022), (2, 1023)]),
        (2044, [(1, 1022), (1023, 2044)]),
        (2045, [(1, 1022), (1024, 2045)]),
    ],
)
def test_exact_reference_windowing(length, coordinates):
    q = make_query_windows(QuerySequence("X", "A" * length))
    assert [(w.info.start, w.info.end) for w in q.windows] == coordinates
    assert all(
        w.info.window_sha256
        == __import__("hashlib").sha256(w.sequence.encode()).hexdigest()
        for w in q.windows
    )
    assert (
        q.sequence_sha256
        == __import__("hashlib").sha256(("A" * length).encode()).hexdigest()
    )
    assert len(q.windows) == len(coordinates)


@pytest.mark.parametrize(
    "sequence", ["", "   ", "abc", "AA*", "A-A", "A A", "A\nA", "J", "é"]
)
def test_invalid_residues(sequence):
    with pytest.raises(ValueError):
        make_query_windows(QuerySequence("q", sequence))


def test_ambiguities_and_outer_whitespace():
    q = make_query_windows(QuerySequence("q", "  XBZUO\n"))
    assert q.length == 5 and q.windows[0].sequence == "XBZUO"


def test_fasta_order_wrapping_duplicate_and_limits(tmp_path, monkeypatch):
    import pclass_dark.inference as infer

    path = tmp_path / "queries.fasta"
    path.write_text(">one description\nACD\nEFG\n>two\nXBZUO\n")
    assert list(read_query_fasta(path)) == [
        QuerySequence("one", "ACDEFG"),
        QuerySequence("two", "XBZUO"),
    ]
    path.write_text(">one\nA\n>one duplicate\nW\n")
    with pytest.raises(IntegrityError, match="Duplicate"):
        list(read_query_fasta(path))
    monkeypatch.setattr(infer, "MAX_SEQUENCE", 3)
    path.write_text(">one\nAA\nAA\n")
    with pytest.raises(InferenceResourceError):
        list(read_query_fasta(path))


def test_batch_reports_oracle_no_setup_dependency(query_library, fake_model, tmp_path):
    root, _ = query_library
    queries = [
        QuerySequence("X", "A" + "W" * 1022),
        QuerySequence("N-X", "Z"),
        QuerySequence("repeat", "A"),
    ]
    with infer_queries(
        queries, library_dir=root, batch_size=1, score_batch_size=1
    ) as run:
        assert run.execution["workers_requested"] == 8
        results = list(run)
        private = run._directory
        assert not (private / "vectors.npy").exists()
        assert not (private / "queries.jsonl").exists()
    assert not private.exists()
    assert [r.query_id for r in results] == ["X", "N-X", "repeat"]
    assert [m.cluster_id for m in results[0].matches] == ["K0", "K1"]
    assert results[1].status == "no_accepted_matches"
    assert [m.cluster_id for m in results[2].matches] == ["K0"]
    for match in results[0].matches:
        # Ten background samples all score below the query near its class mean.
        assert match.background_tail_fraction == pytest.approx(1 / 11)
        assert match.cluster_percentile_le == 1.0
    assert all(len(batch) == 1 for batch in fake_model)


def test_window_union_and_pairing_and_lowlevel_parity(query_library, fake_model):
    root, contract = query_library
    pair = load_selected_library(root)
    q = make_query_windows(QuerySequence("q", "A" * 2045))
    vectors = np.zeros((2, 640), dtype="<f4")
    vectors[:, :2] = 0.009
    result = score_query_windows(pair, [q], vectors, embedding_contract=contract)[0]
    assert len(result.matches) == 1
    match = result.matches[0]
    assert len(match.supporting_windows) == 2
    assert match.representative.start == 1  # Equal B: earlier coordinate wins.
    assert (
        match.cluster_percentile_le == match.supporting_windows[0].cluster_percentile_le
    )
    with infer_queries([QuerySequence("q", "A" * 2045)], library_dir=root) as run:
        assert next(run) == result
        with pytest.raises(StopIteration):
            next(run)
        with pytest.raises(StopIteration):
            next(run)
    with pytest.raises(RuntimeError):
        next(run)


def test_one_reference_materialization_per_scoring_run(
    query_library, fake_model, monkeypatch
):
    import pclass_dark.relative_scores as relative

    root, _ = query_library
    calls = []
    original = relative._read_arrays

    def read(*args, **kwargs):
        if not kwargs.get("mmap", False):
            calls.append(args[1]["cluster_id"])
        return original(*args, **kwargs)

    monkeypatch.setattr(relative, "_read_arrays", read)
    queries = [QuerySequence(str(i), "A") for i in range(6)]
    with infer_queries(
        queries, library_dir=root, batch_size=2, score_batch_size=1
    ) as run:
        assert len(list(run)) == 6
    assert calls == ["K0"]  # K1 rejects every window; audit mmap is separate.


def test_json_text_and_fresh_session_contract(query_library, fake_model):
    root, _ = query_library
    for format in ("json", "text"):
        with infer_queries(
            [QuerySequence("[red]é", "A"), QuerySequence("empty", "Z")],
            library_dir=root,
        ) as run:
            output = io.StringIO()
            write_inference_report(run, output, format=format)
            text = output.getvalue()
            assert "\x1b" not in text
            if format == "json":
                report = json.loads(text)
                assert report["schema"] == "pclass-dark.infer-report.v1"
                assert report["queries"][0]["matches"][0]["accepted"] is True
                assert report["queries"][1]["matches"] == []
                assert report["counts"]["queries_total"] == 2
            else:
                assert "[red]é" in text and "No accepted matches" in text
                assert "Yes" in text and "9.09091%" in text
    with infer_queries([QuerySequence("a", "A")], library_dir=root) as run:
        next(run)
        with pytest.raises(RuntimeError, match="fresh"):
            write_inference_report(run, io.StringIO())


def test_cli_stdout_and_atomic_invalid_fasta(query_library, fake_model, tmp_path):
    root, _ = query_library
    runner = CliRunner()
    result = runner.invoke(
        app, ["infer", "--seq", "A", "--library-dir", str(root), "--format", "json"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["queries"][0]["query_id"] == "query"
    path = tmp_path / "bad.fasta"
    path.write_text(">first\nA\n>second\nbad\n")
    calls = len(fake_model)
    result = runner.invoke(
        app, ["infer", "--fasta", str(path), "--library-dir", str(root)]
    )
    assert result.exit_code == 1 and result.stdout == ""
    assert len(fake_model) == calls
    for args in (
        [],
        ["--seq", "A", "--fasta", str(path)],
        ["--seq", "A", "--format", "bad"],
        ["--seq", "A", "--workers", "0"],
    ):
        result = runner.invoke(app, ["infer", *args])
        assert result.exit_code == 2


def test_failed_entry_and_budget_cleanup(query_library, fake_model, monkeypatch):
    import pclass_dark.inference as infer

    root, _ = query_library
    session = infer_queries([QuerySequence("q", "A")], library_dir=root, memory_mib=128)
    with pytest.raises(InferenceResourceError):
        session.__enter__()
    assert session._temporary is None and session._db is None
    assert not fake_model
    monkeypatch.setattr(infer, "MAX_TEMP", 100)
    with pytest.raises(InferenceResourceError):
        with infer_queries([QuerySequence("q", "A")], library_dir=root):
            pass


def test_model_failure_no_report_and_snapshot_race(query_library, monkeypatch):
    import pclass_dark.inference as infer

    root, contract = query_library

    @contextmanager
    def context(*args, **kwargs):
        class Backend:
            def embed(self, seqs):
                raise RuntimeError("CUDA out of memory")

        b = Backend()
        b.contract = contract
        yield b, {"device": "cuda:0"}

    monkeypatch.setattr(infer, "_backend_context", context)
    result = CliRunner().invoke(
        app, ["infer", "--seq", "A", "--library-dir", str(root)]
    )
    assert result.exit_code == 1 and not result.stdout
    assert "--device cpu" in result.stderr
    original = (root / "selection.json").read_bytes()

    @contextmanager
    def race(*args, **kwargs):
        (root / "selection.json").write_bytes(original + b"\n")

        class Backend:
            def embed(self, s):
                pytest.fail("stale snapshot embedded")

        b = Backend()
        b.contract = contract
        yield b, {}

    monkeypatch.setattr(infer, "_backend_context", race)
    with pytest.raises(IntegrityError, match="changed"):
        with infer_queries([QuerySequence("q", "A")], library_dir=root):
            pass


def test_relative_scope_closed_and_pending_markers(query_library):
    root, _ = query_library
    pair = load_selected_library(root)
    assert pair.relative.describe()["K0"]["C_n"] == 10
    with pair.relative.for_cluster("K0") as view:
        view.evaluate(np.array([0.0]))
    with pytest.raises(RuntimeError, match="closed"):
        view.evaluate(np.array([0.0]))
    write_json(root / ".cleanup-peer.json", {})
    with pytest.raises(RuntimeError, match="maintenance"):
        pair.relative.evaluate("K0", np.array([0.0]))


def test_help_does_not_initialize_model():
    runner = CliRunner()
    result = runner.invoke(app, ["infer", "-h"])
    assert result.exit_code == 0 and "--score-batch-size" in result.stdout
    assert result.stdout == runner.invoke(app, ["infer", "--help"]).stdout


@pytest.mark.parametrize(
    "device,available,resolved,fallback",
    [
        ("auto", False, "cpu", "cuda_unavailable"),
        ("auto", True, "cuda", None),
        ("auto", True, "cpu", "cuda_initialization_failed"),
        ("cpu", True, "cpu", None),
        ("cuda", True, "cuda", None),
    ],
)
def test_backend_device_threads_and_restoration(
    monkeypatch, device, available, resolved, fallback
):
    import sys
    from types import SimpleNamespace

    import pclass_dark.inference_backend as module

    matmul = SimpleNamespace(allow_tf32=True)
    cudnn = SimpleNamespace(allow_tf32=True)
    threads = [16]
    torch = SimpleNamespace(
        get_num_threads=lambda: threads[0],
        set_num_threads=lambda n: threads.__setitem__(0, n),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul), cudnn=cudnn),
        cuda=SimpleNamespace(is_available=lambda: available, current_device=lambda: 0),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(module, "cpu_budget", lambda requested: min(requested, 2))

    class Backend:
        def __init__(self, **kwargs):
            assert kwargs["device"] == (
                "cpu" if device == "auto" and not available else device
            )
            assert kwargs["seed"] == 42
            assert threads[0] == 2
            matmul.allow_tf32 = cudnn.allow_tf32 = False
            if device == "cuda":
                raise RuntimeError("deliberate device initialization failure")
            self.device = resolved
            self.contract = {"torch_version": "test", "transformers_version": "test"}

    monkeypatch.setattr(module, "EsmBackend", Backend)
    with pytest.raises(RuntimeError, match="deliberate"):
        with module.backend_context(
            {"model": "m", "revision": "r"}, device=device, workers=8, offline=True
        ) as (_, execution):
            assert execution["fallback_reason"] == fallback
            assert execution["device"] == ("cuda:0" if resolved == "cuda" else resolved)
            assert execution["workers_used"] == 2
            raise RuntimeError("deliberate")
    assert threads == [16] and matmul.allow_tf32 and cudnn.allow_tf32


def test_cache_hit_rejects_changed_generation(query_library):
    root, _ = query_library
    pair = load_selected_library(root)
    pair.relative.evaluate("K0", np.array([0.0]))
    manifest = pair.relative.path / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(IntegrityError, match="reload"):
        pair.relative.evaluate("K0", np.array([0.0]))


def test_invalid_lowlevel_window_contract(query_library):
    from dataclasses import replace

    root, contract = query_library
    pair = load_selected_library(root)
    q = make_query_windows(QuerySequence("q", "A"))
    for invalid in (
        replace(q, query_id="bad id"),
        replace(q, length=True),
        replace(q, windows=None),
        replace(q, windows=(None,)),
        replace(q, windows=(replace(q.windows[0], sequence=None),)),
    ):
        with pytest.raises(ValueError):
            score_query_windows(
                pair, [invalid], np.zeros((1, 640)), embedding_contract=contract
            )


def test_reducer_native_sort_and_same_window_C():
    from pclass_dark.inference_records import WindowInfo, combine_matches

    windows = (
        WindowInfo("N-q", "N", 1, 1022, "a" * 64),
        WindowInfo("C-q", "C", 1024, 2045, "b" * 64),
    )
    metadata = {c: {"C_n": 10, "B_n": 10000000} for c in ("a", "z")}
    # Both B values round to the same six-digit percentage. Sort native values.
    result = combine_matches(
        0,
        "q",
        2045,
        "c" * 64,
        windows,
        [("a", 0, 0.100000004, 0.9), ("z", 0, 0.4, 1.0), ("z", 1, 0.100000001, 0.2)],
        metadata,
    )
    assert [m.cluster_id for m in result.matches] == ["z", "a"]
    assert result.matches[0].cluster_percentile_le == 0.2
    assert result.matches[0].representative == windows[1]
    assert len(result.matches[0].supporting_windows) == 2


def test_fasta_replacement_detected(tmp_path):
    path = tmp_path / "query.fasta"
    path.write_text(">q\nA\n")
    iterator = read_query_fasta(path)
    assert next(iterator) == QuerySequence("q", "A")
    other = tmp_path / "replacement"
    other.write_text(">q\nW\n")
    other.replace(path)
    with pytest.raises(IntegrityError, match="changed"):
        next(iterator)


def test_lowlevel_invalid_vectors(query_library):
    root, contract = query_library
    pair = load_selected_library(root)
    q = make_query_windows(QuerySequence("q", "A"))
    for vectors in (
        np.zeros((2, 640)),
        np.zeros((1, 640), dtype=int),
        np.full((1, 640), np.nan),
    ):
        with pytest.raises(ValueError):
            score_query_windows(pair, [q], vectors, embedding_contract=contract)


@pytest.mark.parametrize("model_batch,score_batch,workers", [(1, 1, 1), (3, 2, 8)])
def test_chunk_and_worker_consistency(
    query_library, fake_model, model_batch, score_batch, workers
):
    root, _ = query_library
    queries = [QuerySequence("q", "A" + "W" * 1022), QuerySequence("none", "Z")]
    with infer_queries(queries, library_dir=root) as baseline:
        expected = list(baseline)
    with infer_queries(
        queries,
        library_dir=root,
        batch_size=model_batch,
        score_batch_size=score_batch,
        workers=workers,
    ) as run:
        assert list(run) == expected


@pytest.mark.parametrize(
    "bad", [np.zeros((1, 639), dtype="<f4"), np.full((1, 640), np.nan, dtype="<f4")]
)
def test_bad_backend_output_fails_closed(query_library, monkeypatch, bad):
    import pclass_dark.inference as infer

    root, contract = query_library

    @contextmanager
    def backend(*args, **kwargs):
        from types import SimpleNamespace

        yield SimpleNamespace(contract=contract, embed=lambda batch: bad), {}

    monkeypatch.setattr(infer, "_backend_context", backend)
    session = infer_queries([QuerySequence("q", "A")], library_dir=root)
    with pytest.raises(IntegrityError, match="matrix"):
        session.__enter__()
    assert session._temporary is None and session._db is None


def test_report_quota_and_index_scans(query_library, fake_model, monkeypatch):
    root, _ = query_library
    with infer_queries([QuerySequence("q", "A")], library_dir=root) as run:
        plan = run._db.execute(
            "EXPLAIN QUERY PLAN SELECT cluster_id FROM accepted "
            "WHERE query_index=? ORDER BY cluster_id,window_index",
            (0,),
        ).fetchall()
        assert not any("TEMP B-TREE" in row[3] for row in plan)
        assert any("PRIMARY KEY" in row[3] for row in plan)
        monkeypatch.setattr("pclass_dark.inference.MAX_TEMP", run._usage() + 10)
        with pytest.raises(InferenceResourceError):
            write_inference_report(run, io.StringIO(), format="json")
