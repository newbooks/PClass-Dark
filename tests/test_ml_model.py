"""Full-cohort fitting, portable prediction and extraction-parity acceptance."""

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pclass_dark import ml_model as final
from pclass_dark.artifacts import IntegrityError, read_json, sha256, write_json
from pclass_dark.cli import app
from pclass_dark.explore_ml import ExploreMLConfig, run_exploration
from pclass_dark.ml_data import FEATURES, PAIR_HEADER
from pclass_dark.ml_reports import read_dicts, write_tsv
from pclass_dark.structural_features import HEADER

pytest.importorskip("sklearn")
pytest.importorskip("torch")


@pytest.fixture(scope="module")
def experiment(tmp_path_factory):
    root = tmp_path_factory.mktemp("final-model-source")
    examples = []
    for c in range(3):
        for j in range(10):
            values = [0.1] * 19
            values[0] = 3 + c + j * 0.001
            values[-1] = None
            examples.append((f"P{c}_{j:02d}", *values, f"C{c}"))
    write_tsv(root / "features.tsv", HEADER, examples)
    write_tsv(
        root / "cluster_dmatrix.tsv",
        PAIR_HEADER,
        [
            ("C0", "M0", "C1", "M1", 0.2, 0.8),
            ("C0", "M0", "C2", "M2", 0.6, 0.4),
            ("C1", "M1", "C2", "M2", 0.75, 0.25),
        ],
    )
    # Two repeats must still produce a final training sample of 30, not 60.
    run_exploration(
        ExploreMLConfig(
            features=root / "features.tsv",
            similarities=root / "cluster_dmatrix.tsv",
            output_dir=root / "cv",
            device="cpu",
            workers=1,
            repeats=2,
            max_epochs=2,
            batch_size=16,
            learning_rate=0.05,
        )
    )
    return root / "cv"


@pytest.fixture(scope="module")
def model_dir(experiment, tmp_path_factory):
    path = tmp_path_factory.mktemp("final-bundle") / "model"
    result = final.fit_final_model(experiment, path, device="cpu", threads=1)
    assert result.example_count == 30 and result.label_count == 3 and result.epochs == 2
    return path


def test_full_cohort_settings_and_queries(experiment, model_dir):
    model = final.load_final_model(model_dir)
    assert model.classifier.training_example_count == 30
    assert model.classifier.parameters["alpha"] == 0
    assert (
        model.manifest["contract"]["epoch_rule"]
        == "rounded_median_inner_selected_epochs"
    )
    assert len(model.manifest["contract"]["source_selected_epochs"]) == 10
    X = np.load(experiment / "arrays/X.npy")
    assert np.array_equal(model.classifier.preprocess.medians, np.median(X, axis=0))
    predictions = list(final.predict_features(model, X[:2], ["q1", "q2"]))
    expected = model.classifier.logits(X[:2])
    for i, p in enumerate(predictions):
        assert p.query_id == f"q{i + 1}" and p.configuration == "hard"
        order = np.argsort(-expected[i], kind="stable")
        assert [r.cluster_id for r in p.ranked] == [model.labels[j] for j in order]
        assert sum(r.softmax_score for r in p.ranked) == pytest.approx(1)
        assert sum(
            p.feature_contributions.values()
        ) + p.bias_difference == pytest.approx(p.margin, abs=1e-4)
    logs = list(read_dicts(model_dir / "training.tsv"))
    assert len(logs) == 2 and all(
        r["n"] == "30" and r["validation_loss"] == "NA" for r in logs
    )
    reused = final.fit_final_model(
        experiment, model_dir, device="cpu", threads=1, resume=True
    )
    assert reused.reused
    with pytest.raises(IntegrityError, match="Incompatible"):
        final.fit_final_model(
            experiment, model_dir, device="cpu", threads=1, resume=True, epochs=3
        )


def test_similarity_final_fit_and_epoch_override(experiment, tmp_path):
    result = final.fit_final_model(
        experiment,
        tmp_path / "siac",
        configuration="siac",
        epochs=3,
        device="cpu",
        threads=1,
    )
    model = final.load_final_model(result.model_dir)
    assert result.configuration == "siac" and result.epochs == 3
    assert model.classifier.parameters["alpha"] == 0.1
    assert model.manifest["contract"]["epoch_rule"] == "explicit"


def test_source_and_bundle_tampering(experiment, model_dir, tmp_path):
    source = tmp_path / "cv"
    shutil.copytree(experiment, source)
    with (source / "arrays/X.npy").open("ab") as h:
        h.write(b"changed")
    with pytest.raises(IntegrityError, match="checksum"):
        final.fit_final_model(source, tmp_path / "bad", device="cpu", threads=1)
    target = tmp_path / "copied"
    shutil.copytree(model_dir, target)
    with (target / "model.npz").open("ab") as h:
        h.write(b"changed")
    with pytest.raises(IntegrityError, match="checksum"):
        final.load_final_model(target)


def test_portable_model_without_source(model_dir, tmp_path):
    target = tmp_path / "portable"
    shutil.copytree(model_dir, target)
    m = read_json(target / "manifest.json")
    m["source_path"] = "/source/no/longer/exists"
    write_json(target / "manifest.json", m)
    model = final.load_final_model(target)
    assert len(list(final.predict_features(model, np.zeros((1, 18)), ["Q"]))) == 1


def test_fit_failure_and_restart(experiment, tmp_path, monkeypatch):
    from pclass_dark import siac

    original = siac.fit_siac

    def failure(*args, **kwargs):
        raise RuntimeError("injected fit failure")

    monkeypatch.setattr(siac, "fit_siac", failure)
    target = tmp_path / "retry"
    with pytest.raises(RuntimeError, match="injected"):
        final.fit_final_model(experiment, target, device="cpu", threads=1)
    assert not (target / "manifest.json").exists()
    monkeypatch.setattr(siac, "fit_siac", original)
    assert (
        final.fit_final_model(
            experiment, target, device="cpu", threads=1, resume=True
        ).example_count
        == 30
    )


def test_query_feature_validation_and_rounding(model_dir, tmp_path):
    model = final.load_final_model(model_dir)
    fc = {**model.feature_contract, "precision": ".3f"}
    model = replace(model, feature_contract=fc)
    X = np.zeros((1, 18))
    X[0, 0] = 1.23456
    X[0, 1] = np.nan
    pred = list(final.predict_features(model, X, ["q"]))[0]
    assert (
        pred.features["ln_length"] == 1.235 and pred.features["acid_fraction"] is None
    )
    assert "acid_fraction" in pred.imputed_features
    for bad, ids in (
        (np.zeros(18), ["q"]),
        (np.full((1, 18), np.inf), ["q"]),
        (np.zeros((2, 18)), ["q", "q"]),
        (np.zeros((0, 18)), []),
    ):
        with pytest.raises(ValueError):
            list(final.predict_features(model, bad, ids))
    path = tmp_path / "queries.tsv"
    write_tsv(path, ("query_id", *FEATURES), [("q", 1.23456, None, *([0.1] * 16))])
    ids, values = final.read_query_features(path)
    assert ids == ("q",) and np.isnan(values[0, 1])
    path.write_text(path.read_text().replace("\tNA\t", "\tnan\t"))
    with pytest.raises(IntegrityError):
        final.read_query_features(path)
    write_tsv(path, ("query_id", *FEATURES, "cluster_id"), [("q", *([0.1] * 18), "C0")])
    with pytest.raises(IntegrityError, match="header"):
        final.read_query_features(path)


def test_raw_sequence_contract_and_analytical_feature_parity(model_dir, monkeypatch):
    from pclass_dark import structure
    from pclass_dark.setup_data import MODEL, REVISION
    from pclass_dark.structural_models import BUNDLE_ID

    model = final.load_final_model(model_dir)
    with pytest.raises(IntegrityError, match="provenance"):
        list(final.predict_sequences(model, [("q", "DEKRH")]))
    code = {
        name: sha256(Path(final.__file__).with_name(name + ".py"))
        for name in (
            "structural_features",
            "structure",
            "structural_models",
            "embeddings",
        )
    }
    contract = {
        "model_id": BUNDLE_ID,
        "esm": {"model": MODEL, "revision": REVISION},
        "code": code,
    }
    model = replace(
        model,
        feature_contract={
            **model.feature_contract,
            "precision": ".3f",
            "extraction": contract,
        },
    )
    calls = []

    class Predictor:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def predict(self, sequence):
            return structure.StructuralPrediction(
                sequence.upper(),
                np.array([0.1, 0.05, 0.2, -0.1, 1.2]),
                np.array(list("HECCH")),
                BUNDLE_ID,
                "cpu",
            )

    monkeypatch.setattr(structure, "StructuralPredictor", Predictor)
    result = list(
        final.predict_sequences(model, [("q", "dekrh"), ("q2", "DEKRH")], device="cpu")
    )
    assert len(calls) == 1 and calls[0]["offline"] is True
    f = result[0].features
    assert f["ln_length"] == 1.609 and f["acid_fraction"] == 0.4
    assert f["surface_acid_over_all"] == 0.2 and f["base_fraction"] == 0.6
    assert f["surface_fraction"] == 0.6 and f["mean_kyte_doolittle_hydropathy"] == -3.72
    assert f["surface_acid_over_surface"] == 0.333
    expected = list(
        final.predict_features(model, np.array([[f[k] for k in FEATURES]]), ["q"])
    )[0]
    assert result[0].ranked == expected.ranked
    broken = {**contract, "model_id": "wrong"}
    with pytest.raises(IntegrityError, match="differ"):
        list(
            final.predict_sequences(
                replace(
                    model,
                    feature_contract={**model.feature_contract, "extraction": broken},
                ),
                [("q", "DEKRH")],
            )
        )


def test_cli_feature_prediction_and_atomic_errors(model_dir, tmp_path):
    runner = CliRunner()
    for name in ("train-ml", "predict-ml"):
        assert runner.invoke(app, [name, "--help"]).exit_code == 0
    queries = tmp_path / "queries.tsv"
    write_tsv(queries, ("query_id", *FEATURES), [("q", *([0.1] * 18))])
    result = runner.invoke(
        app,
        [
            "predict-ml",
            "--model-dir",
            str(model_dir),
            "--features",
            str(queries),
            "--format",
            "tsv",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (
        result.stdout.splitlines()[0]
        == "query_id\trank\tcluster_id\tlogit\tsoftmax_score"
    )
    result = runner.invoke(
        app,
        [
            "predict-ml",
            "--model-dir",
            str(model_dir),
            "--features",
            str(queries),
            "--format",
            "jsonl",
        ],
    )
    assert result.exit_code == 0, result.output
    row = json.loads(result.stdout)
    assert row["query_id"] == "q" and "feature_contributions" in row
    output = tmp_path / "predictions.md"
    args = [
        "predict-ml",
        "--model-dir",
        str(model_dir),
        "--features",
        str(queries),
        "--output",
        str(output),
    ]
    assert runner.invoke(app, args).exit_code == 0
    assert output.read_text().startswith("# Prediction interpretation: q")
    before = output.read_bytes()
    assert runner.invoke(app, args).exit_code == 1
    assert output.read_bytes() == before
    assert (
        runner.invoke(
            app,
            [
                "predict-ml",
                "--model-dir",
                str(model_dir),
                "--seq",
                "A",
                "--features",
                str(queries),
            ],
        ).exit_code
        == 2
    )
    assert runner.invoke(app, ["train-ml", "--config", "unknown"]).exit_code == 2


def test_lazy_imports_and_script_help():
    code = (
        "import sys; import pclass_dark.ml_model; "
        "assert 'torch' not in sys.modules; assert 'sklearn' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )
    for name in ("train-ml.py", "predict-ml.py"):
        result = subprocess.run(
            [sys.executable, "scripts/" + name, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("format", ["tsv", "jsonl", "markdown"])
def test_late_query_error_publishes_no_partial_output(
    model_dir, tmp_path, monkeypatch, format
):
    model = final.load_final_model(model_dir)
    prediction = next(final.predict_features(model, np.zeros((1, 18)), ["first"]))

    def fail_late(*args, **kwargs):
        yield prediction
        raise ValueError("late invalid query")

    monkeypatch.setattr(final, "predict_sequences", fail_late)
    runner = CliRunner()
    args = [
        "predict-ml",
        "--model-dir",
        str(model_dir),
        "--seq",
        "AAA",
        "--format",
        format,
        "--stdout",
    ]
    result = runner.invoke(app, args)
    assert result.exit_code == 1 and result.stdout == ""
    path = tmp_path / "output.tsv"
    result = runner.invoke(
        app, [*[a for a in args if a != "--stdout"], "--output", str(path)]
    )
    assert result.exit_code == 1 and not path.exists()


def test_partial_fit_publication_restart(experiment, tmp_path, monkeypatch):
    root = tmp_path / "partial"
    original = final.os.replace
    fired = []

    def fail_once(source, target):
        if str(source).endswith(".fit-stage/training.tsv") and not fired:
            fired.append(True)
            raise OSError("publication failure")
        return original(source, target)

    monkeypatch.setattr(final.os, "replace", fail_once)
    with pytest.raises(OSError, match="publication"):
        final.fit_final_model(experiment, root, device="cpu", threads=1)
    assert not (root / "manifest.json").exists()
    monkeypatch.setattr(final.os, "replace", original)
    result = final.fit_final_model(
        experiment, root, device="cpu", threads=1, resume=True
    )
    assert result.example_count == 30
    assert not (root / ".fit-stage").exists()


def test_median_epoch_rounding_and_invalid_options(experiment, tmp_path):
    copied = tmp_path / "cv"
    shutil.copytree(experiment, copied)
    manifest = read_json(copied / "manifest.json")
    from pclass_dark.artifacts import file_record

    for i, (r, f) in enumerate((r, f) for r in range(2) for f in range(5)):
        name = f"folds/r{r:03d}/f{f:03d}/hard/model.json"
        meta = read_json(copied / name)
        meta["selected_epochs"] = 1 if i < 5 else 2
        write_json(copied / name, meta)
        manifest["files"][name] = file_record(copied / name)
    write_json(copied / "manifest.json", manifest)
    source = final._training_source(copied, "hard", None)
    assert source[8] == 2  # Median 1.5 rounds upward, not bankers rounding.
    with pytest.raises(ValueError):
        final.fit_final_model(experiment, tmp_path / "bad", configuration="ensemble")
    with pytest.raises(ValueError):
        final.fit_final_model(experiment, tmp_path / "bad", epochs=0)
    with pytest.raises(ValueError):
        final.fit_final_model(experiment, tmp_path / "bad", device="unknown")


def test_markdown_cli_and_added_json_fields(model_dir, tmp_path):
    from pclass_dark.ml_interpretation import render_interpretation

    model = final.load_final_model(model_dir)
    X = np.full((2, 18), 0.1)
    predictions = list(final.predict_features(model, X, ["q1", "q2"]))
    for p in predictions:
        assert sum(
            p.chosen_feature_contributions.values()
        ) + p.chosen_bias == pytest.approx(p.chosen_logit, abs=1e-4)
        assert sum(
            p.feature_contributions.values()
        ) + p.bias_difference == pytest.approx(p.margin, abs=1e-4)
    queries = tmp_path / "queries.tsv"
    write_tsv(
        queries,
        ("query_id", *FEATURES),
        [(p.query_id, *X[i]) for i, p in enumerate(predictions)],
    )
    args = ["predict-ml", "--model-dir", str(model_dir), "--features", str(queries)]
    result = CliRunner().invoke(app, [*args, "--stdout"])
    assert result.exit_code == 0, result.output
    assert result.stdout == "\n---\n\n".join(
        render_interpretation(p) for p in predictions
    )
    from pclass_dark.ml_interpretation import render_top_predictions

    saved = CliRunner().invoke(app, [*args, "--output", str(tmp_path / "report.md")])
    assert saved.exit_code == 0, saved.output
    for p in predictions:
        assert f'Query: "{p.query_id}"' in saved.stdout
        assert render_top_predictions(p) in saved.stdout
    assert "## Plain-language" not in saved.stdout
    result = CliRunner().invoke(app, [*args, "--format", "jsonl"])
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    assert (
        rows[0]["chosen_feature_contributions"]
        == predictions[0].chosen_feature_contributions
    )


def test_automatic_interpretation_file(model_dir, tmp_path, monkeypatch):
    queries = tmp_path / "seq2.tsv"
    write_tsv(queries, ("query_id", *FEATURES), [("q", *([0.1] * 18))])
    runner = CliRunner()
    args = ["predict-ml", "--model-dir", str(model_dir), "--features", str(queries)]
    result = runner.invoke(app, args)
    report = tmp_path / "seq2-interpretation.md"
    assert result.exit_code == 0, result.output
    assert 'Query: "q"' in result.stdout
    assert "## Top 5 predictions" in result.stdout
    assert "Short description | Softmax score" in result.stdout
    assert "## Plain-language" not in result.stdout
    assert str(report) in result.stderr
    before = report.read_bytes()
    assert before.startswith(b"# Prediction interpretation: q")
    collision = runner.invoke(app, args)
    assert collision.exit_code == 1 and collision.stdout == ""
    assert report.read_bytes() == before
    assert runner.invoke(app, [*args, "--stdout"]).stdout == before.decode()
    assert (
        runner.invoke(
            app, [*args, "--stdout", "--output", str(tmp_path / "x.md")]
        ).exit_code
        != 0
    )

    model = final.load_final_model(model_dir)
    p = next(final.predict_features(model, np.zeros((1, 18)), ["q"]))
    monkeypatch.setattr(final, "predict_sequences", lambda *a, **kw: iter([p]))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app,
        [
            "predict-ml",
            "--model-dir",
            str(model_dir),
            "--seq",
            "DEKRH",
            "--query-id",
            "../../unsafe/name",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "unsafe_name-interpretation.md").exists()
    fasta = tmp_path / "seq2.fasta"
    result = runner.invoke(
        app, ["predict-ml", "--model-dir", str(model_dir), "--fasta", str(fasta)]
    )
    # Same default name as the existing TSV report: refuse before inference.
    assert result.exit_code == 1
    fasta = tmp_path / "other.fasta"
    result = runner.invoke(
        app, ["predict-ml", "--model-dir", str(model_dir), "--fasta", str(fasta)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "other-interpretation.md").exists()
