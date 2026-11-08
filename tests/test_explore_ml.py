"""Independent SIAC analytical examples and five-fold artifact integration."""

import csv
import os
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest
from typer.testing import CliRunner

from pclass_dark import explore_ml as ml
from pclass_dark.artifacts import IntegrityError, read_json, sha256, write_json
from pclass_dark.cli import app
from pclass_dark.ml_data import (
    INDEX_HEADER,
    PAIR_HEADER,
    LabelConflictError,
    load_cohort,
    load_groups,
    load_similarity,
)
from pclass_dark.ml_reports import read_dicts, score_predictions, write_tsv
from pclass_dark.ml_splits import make_split_plan
from pclass_dark.siac import (
    fit_preprocessing,
    fit_siac,
    load_model,
    make_soft_targets,
    predict_siac,
    save_model,
    soft_loss,
)
from pclass_dark.structural_features import HEADER

pytest.importorskip("sklearn")
torch = pytest.importorskip("torch")


@pytest.fixture
def inputs(tmp_path):
    features, similarity = tmp_path / "features.tsv", tmp_path / "cluster_dmatrix.tsv"
    samples = []
    for label in range(3):
        for j in range(12):
            x = [0.1] * 19
            x[0] = 3 + label + j * 0.001
            x[1] = 0.1 + label * 0.1
            x[17] = -0.3 + label * 0.2
            x[-1] = None  # excluded SCD must never be an input
            samples.append((f"P{label}_{j:02d}", *x, f"C{label}"))
    write_tsv(features, HEADER, samples)
    write_tsv(
        similarity,
        PAIR_HEADER,
        [
            ("C0", "M0", "C1", "M1", 0.2, 0.8),
            ("C0", "M0", "C2", "M2", 0.6, 0.4),
            ("C1", "M1", "C2", "M2", 0.75, 0.25),
        ],
    )
    return features, similarity


def config(inputs, tmp_path, **kwargs):
    f, s = inputs
    return ml.ExploreMLConfig(
        features=f,
        similarities=s,
        output_dir=tmp_path / "experiment",
        split_mode="accession",
        device="cpu",
        max_epochs=2,
        learning_rate=0.05,
        batch_size=16,
        workers=1,
        **kwargs,
    )


def test_target_and_loss_independent_oracle():
    S = np.array([[1, 0.8, 0.4], [0.8, 1, 0.25], [0.4, 0.25, 1]])
    targets = make_soft_targets(
        S, active_labels=[0, 1, 2], alpha=0.1, neighbors=2, gamma=2
    )
    assert targets.weights[0] == pytest.approx([0.8, 0.2])
    logits = torch.tensor([[1.0, -2.0, 0.3]], requires_grad=True)
    loss = soft_loss(logits, torch.tensor([0]), targets)
    q = np.array([0.9, 0.08, 0.02])
    z = np.array([1.0, -2.0, 0.3])
    expected = -np.sum(q * (z - np.log(np.exp(z).sum())))
    assert float(loss.detach()) == pytest.approx(expected, abs=1e-6)
    loss.backward()
    assert logits.grad.numpy()[0] == pytest.approx(
        np.exp(z) / np.exp(z).sum() - q, abs=1e-6
    )
    hard = make_soft_targets(S, active_labels=[0, 1, 2], alpha=0, neighbors=2, gamma=2)
    assert soft_loss(logits, torch.tensor([0]), hard).item() == pytest.approx(
        torch.nn.functional.cross_entropy(logits, torch.tensor([0])).item()
    )
    isolated = make_soft_targets(
        np.eye(3), active_labels=[0, 1, 2], alpha=0.1, neighbors=2, gamma=2
    )
    assert np.all(isolated.alpha == 0)
    restricted = make_soft_targets(
        S, active_labels=[0, 2], alpha=0.1, neighbors=10, gamma=1e308
    )
    assert restricted.neighbor_indices[0, 0] == 1
    assert restricted.weights[0, 0] == 1
    for bad in (-0.1, 1, float("nan")):
        with pytest.raises(ValueError):
            make_soft_targets(
                S, active_labels=[0, 1, 2], alpha=bad, neighbors=2, gamma=2
            )


def test_metric_oracle_macro_and_empty():
    S = np.array([[1, 0.2, 0.1], [0.2, 1, 0.25], [0.1, 0.25, 1]])
    m = score_predictions(np.array([0, 1, 2]), np.array([0, 2, 1]), S)
    assert m["exact_hit_rate"] == pytest.approx(1 / 3)
    assert m["weighted_hit_rate"] == 0.5
    m = score_predictions(np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]), S)
    assert m["exact_hit_rate"] == 0.75 and m["macro_recall"] == 0.5
    empty = score_predictions(np.array([], dtype=int), np.array([], dtype=int), S)
    assert empty["n"] == 0 and empty["weighted_hit_rate"] is None


def test_training_only_splines_and_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    X = np.ones((80, 18))
    X[:, 0] = rng.uniform(-1, 1, 80)
    X[0, 0] = np.nan
    y = (np.nan_to_num(X[:, 0]) > 0).astype(int)
    preprocess = fit_preprocessing(X, knots=5)
    assert preprocess.offsets[1] == preprocess.offsets[-1]
    assert preprocess.medians[0] == pytest.approx(np.nanmedian(X[:, 0]))
    extreme = X[:2].copy()
    extreme[:, 0] = [-100, 100]
    boundary = extreme.copy()
    boundary[:, 0] = [preprocess.training_min[0], preprocess.training_max[0]]
    assert np.array_equal(preprocess.transform(extreme), preprocess.transform(boundary))
    targets = make_soft_targets(
        np.eye(2), active_labels=[0, 1], alpha=0, neighbors=1, gamma=2
    )
    params = dict(
        learning_rate=0.05,
        weight_decay=0.0001,
        batch_size=16,
        max_epochs=35,
        patience=10,
        min_delta=1e-5,
        threads=1,
    )
    model, _ = fit_siac(
        X,
        y,
        preprocess=preprocess,
        targets=targets,
        parameters=params,
        seed=42,
        device="cpu",
        epochs=35,
    )
    # Independently known sign rule, evaluated on new feature values.
    test = np.ones((40, 18))
    test[:, 0] = np.linspace(-0.95, 0.95, 40)
    expected = (test[:, 0] > 0).astype(int)
    pred = np.argmax(model.logits(test), axis=1)
    assert np.mean(pred == expected) >= 0.95
    save_model(tmp_path, model, ["A", "B"])
    restored = load_model(tmp_path)
    assert np.allclose(model.logits(test), restored.logits(test), atol=1e-6, rtol=0)
    for batch in predict_siac(restored, test, batch_size=7):
        assert np.allclose(
            batch.logits[:, 0] - batch.logits[:, 1],
            batch.contributions.sum(1) + batch.bias_difference,
            atol=1e-4,
        )
        assert np.all(batch.contributions[:, 1:] == 0)
    metadata = read_json(tmp_path / "model.json")
    metadata["basis_offsets"][-1] += 1
    write_json(tmp_path / "model.json", metadata)
    with pytest.raises(IntegrityError):
        load_model(tmp_path)


def test_conflicts_aliases_and_excluded_scd(inputs, tmp_path):
    f, _ = inputs
    with f.open() as h:
        rows = list(csv.reader(h, delimiter="\t"))
    original = rows[1]
    rows.append([*original[:-1], "C2"])
    with f.open("w") as h:
        csv.writer(h, delimiter="\t", lineterminator="\n").writerows(rows)
    with pytest.raises(LabelConflictError) as error:
        load_cohort(f)
    assert len(error.value.conflicts) == 2
    cohort = load_cohort(f, conflict_policy="exclude")
    assert (
        len(cohort.X) == 35 and cohort.X.shape[1] == 18 and not np.isnan(cohort.X).any()
    )
    assert len(cohort.exclusions) == 2
    # Alias the conflicting identity to a seemingly unambiguous accession: exclude all.
    idx = tmp_path / "idx.tsv"
    accessions = {r[0] for r in rows[1:]}
    table = []
    for i, a in enumerate(sorted(accessions)):
        sid = f"{i + 1:064x}"
        table.append((a, sid, 100, len({r[-1] for r in rows[1:] if r[0] == a})))
    write_tsv(idx, INDEX_HEADER, table)
    indexed = load_cohort(f, sequence_index=idx, conflict_policy="exclude")
    assert indexed.example_ids[0].startswith("sha256:")
    assert len(indexed.X) == 35


def test_similarity_extra_duplicates_and_missing(inputs, tmp_path):
    _, s = inputs
    sim = load_similarity(s, labels=("C0", "C1", "C2"))
    assert sim.S[0, 1] == 0.8 and np.array_equal(sim.S, sim.S.T)
    with s.open("a") as h:
        h.write("X\tMX\tY\tMY\t1\t0\nY\tMY\tX\tMX\t1\t0\n")
    with pytest.raises(IntegrityError, match="Duplicate"):
        load_similarity(s, labels=sim.labels)
    s.write_text("\t".join(PAIR_HEADER) + "\nC0\tM0\tC1\tM1\t0.2\t0.8\n")
    with pytest.raises(IntegrityError):
        load_similarity(s, labels=("C0", "C1", "C2"))


def test_split_coverage_and_seed(inputs):
    cohort = load_cohort(inputs[0])
    groups = load_groups(cohort, split_mode="accession")
    plan = make_split_plan(
        cohort, groups=groups, split_mode="accession", repeats=2, seed=42
    )
    again = make_split_plan(
        cohort, groups=groups, split_mode="accession", repeats=2, seed=42
    )
    for r in range(2):
        test = np.concatenate([p.test for p in plan.folds if p.repeat == r])
        assert sorted(test) == list(range(len(cohort.y)))
    for a, b in zip(plan.folds, again.folds, strict=True):
        assert np.array_equal(a.test, b.test)
        assert not set(a.train) & set(a.test)
        assert not set(a.inner_train) & set(a.inner_validation)
        assert set(a.inner_train) | set(a.inner_validation) == set(a.train)


def test_five_fold_reports_resume_and_corruption(inputs, tmp_path):
    c = config(inputs, tmp_path)
    result = ml.run_exploration(c)
    summary = read_json(result.summary_path)
    assert summary["status"] == "complete" and len(summary["repeat_metrics"]) == 4
    rows = list(read_dicts(result.oof_path))
    assert len(rows) == 36 * 4
    for metric in summary["repeat_metrics"]:
        subset = [r for r in rows if r["config_id"] == metric["config_id"]]
        assert metric["exact_hit_rate"] == pytest.approx(
            sum(int(r["exact_hit"]) for r in subset) / 36
        )
        assert metric["weighted_hit_rate"] == pytest.approx(
            sum(float(r["similarity_credit"]) for r in subset) / 36
        )
    assert all(
        r["modal_agreement"] == "NA" for r in read_dicts(c.output_dir / "stability.tsv")
    )
    before = sha256(result.summary_path)
    reused = ml.run_exploration(replace(c, resume=True))
    assert reused.reused_folds == 5 and sha256(result.summary_path) == before
    with pytest.raises(IntegrityError, match="Incompatible"):
        ml.run_exploration(replace(c, resume=True, knots=4))
    assert sha256(result.summary_path) == before
    model = c.output_dir / "folds" / "r000" / "f000" / "siac" / "model.npz"
    with model.open("ab") as h:
        h.write(b"corrupt")
    with pytest.raises(IntegrityError, match="checksum"):
        ml.run_exploration(replace(c, resume=True))


def test_validation_and_cli(inputs, tmp_path):
    c = config(inputs, tmp_path, validate_only=True)
    ready = ml.run_exploration(c)
    assert ready.status == "ready" and not (c.output_dir / "manifest.json").exists()
    result = ml.run_exploration(replace(c, validate_only=False, resume=True))
    assert result.status == "complete"
    runner = CliRunner()
    assert runner.invoke(app, ["explore-ml", "--help"]).exit_code == 0
    bad = runner.invoke(
        app, ["explore-ml", "--split-mode", "accession", "--alpha", "0"]
    )
    assert bad.exit_code == 2
    cli = runner.invoke(
        app,
        [
            "explore-ml",
            "--features",
            str(inputs[0]),
            "--similarities",
            str(inputs[1]),
            "--output-dir",
            str(tmp_path / "cli"),
            "--split-mode",
            "accession",
            "--device",
            "cpu",
            "--validate-only",
        ],
    )
    assert cli.exit_code == 0, cli.output


def test_failed_fold_resume(inputs, tmp_path, monkeypatch):
    c = config(inputs, tmp_path)
    original = ml._task_job

    def fail(root, cohort, similarity, plan, *args):
        if plan.fold == 1:
            raise RuntimeError("injected fold failure")
        return original(root, cohort, similarity, plan, *args)

    monkeypatch.setattr(ml, "_task_job", fail)
    with pytest.raises(RuntimeError, match="injected"):
        ml.run_exploration(c)
    assert not (c.output_dir / "manifest.json").exists()
    assert (c.output_dir / "folds" / "r000" / "f000" / "task.json").exists()
    monkeypatch.setattr(ml, "_task_job", original)
    result = ml.run_exploration(replace(c, resume=True))
    assert result.reused_folds == 1


def test_parallel_serial_consistency(inputs, tmp_path):
    if (os.cpu_count() or 1) < 2:
        pytest.skip("At least two CPUs required")
    a = config(inputs, tmp_path)
    b = replace(a, output_dir=tmp_path / "parallel", workers=2)
    x = ml.run_exploration(a)
    y = ml.run_exploration(b)
    assert x.oof_path.read_bytes() == y.oof_path.read_bytes()
    assert (
        read_json(x.summary_path)["repeat_metrics"]
        == read_json(y.summary_path)["repeat_metrics"]
    )


def test_lazy_imports_and_script_help():
    script = (
        "import sys; import pclass_dark.explore_ml; "
        "assert 'torch' not in sys.modules; "
        "assert 'sklearn' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    result = subprocess.run(
        [sys.executable, "scripts/explore-ml.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "similarities" in result.stdout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU unavailable")
@pytest.mark.parametrize("cutoff,probability_tolerance", [(0.0, 5e-4), (0.1, 1e-4)])
def test_gpu_cpu_prediction_agreement(cutoff, probability_tolerance):
    X = np.zeros((20, 18))
    X[:, 0] = np.linspace(-1, 1, 20)
    y = (X[:, 0] > cutoff).astype(int)
    p = fit_preprocessing(X, knots=4)
    t = make_soft_targets(
        np.eye(2), active_labels=[0, 1], alpha=0.1, neighbors=1, gamma=2
    )
    params = dict(
        learning_rate=0.01,
        weight_decay=0.0001,
        batch_size=20,
        max_epochs=2,
        patience=2,
        min_delta=1e-5,
    )
    cpu, _ = fit_siac(
        X,
        y,
        preprocess=p,
        targets=t,
        parameters=params,
        seed=42,
        device="cpu",
        epochs=2,
    )
    gpu, _ = fit_siac(
        X,
        y,
        preprocess=p,
        targets=t,
        parameters=params,
        seed=42,
        device="cuda:0",
        epochs=2,
    )

    # At exact class balance, cancellation-scale bias gradients can differ at
    # float32 precision and AdamW's eps=1e-8 amplifies that near-zero component.
    # Measure probabilities, the specified cross-device prediction quantity,
    # and separately require identical labels on this non-tied analytic example.
    def probabilities(model):
        logits = model.logits(X).astype(np.float64)
        values = np.exp(logits - logits.max(axis=1, keepdims=True))
        return values / values.sum(axis=1, keepdims=True)

    cp, gp = probabilities(cpu), probabilities(gpu)
    assert np.allclose(cp, gp, atol=probability_tolerance, rtol=1e-4)
    assert np.array_equal(cp.argmax(axis=1), gp.argmax(axis=1))
    if cutoff == 0.0:
        # Independent analytical initial-bias derivative is exactly zero.
        for device in ("cpu", "cuda:0"):
            bias = torch.zeros(2, device=device, requires_grad=True)
            loss = torch.nn.functional.cross_entropy(
                bias.expand(len(y), 2), torch.tensor(y, device=device)
            )
            loss.backward()
            assert (
                np.max(np.abs(bias.grad.detach().cpu().numpy()))
                <= np.finfo(np.float32).eps
            )


def test_repeated_cv_stability_matches_predictions(inputs, tmp_path):
    c = config(inputs, tmp_path, repeats=2)
    result = ml.run_exploration(c)
    cohort = load_cohort(inputs[0])
    similarity = load_similarity(inputs[1], labels=cohort.labels)
    labels = {label: i for i, label in enumerate(cohort.labels)}
    predictions = {}
    for row in read_dicts(result.oof_path):
        predictions.setdefault((row["config_id"], row["example_id"]), []).append(
            row["pred_cluster_id"]
        )
    for row in read_dicts(c.output_dir / "stability.tsv"):
        a, b = predictions[(row["config_id"], row["example_id"])]
        assert float(row["modal_agreement"]) == (1 if a == b else 0.5)
        assert float(row["semantic_agreement"]) == similarity.S[labels[a], labels[b]]
    assert len(read_json(result.summary_path)["repeat_metrics"]) == 8


def test_homology_and_unsupported_labels(inputs, tmp_path):
    groups = tmp_path / "groups.tsv"
    cohort = load_cohort(inputs[0])
    write_tsv(
        groups,
        ("accession_id", "group_id"),
        [
            (a, "whole_C0" if label == "C0" else a)
            for _, a, _, _, label, _ in cohort.aliases
        ],
    )
    provenance = tmp_path / "groups.json"
    write_json(
        provenance,
        {
            "method": "synthetic grouping",
            "tool_version": "fixture-v1",
            "command": "fixture",
            "source_sequences_sha256": "0" * 64,
            "parameters": {"definition": "C0 shares one group"},
        },
    )
    c = replace(
        config(inputs, tmp_path),
        split_mode="homology",
        groups=groups,
        group_provenance=provenance,
    )
    result = ml.run_exploration(c)
    predictions = list(read_dicts(result.oof_path))
    unsupported = [r for r in predictions if int(r["train_class_count"]) == 0]
    assert unsupported and all(r["exact_hit"] == "0" for r in unsupported)
    assert len(predictions) == len(cohort.y) * 4
    plan = list(read_dicts(c.output_dir / "folds.tsv"))
    assert len({r["outer_fold_id"] for r in plan if r["group_id"] == "whole_C0"}) == 1


def test_invalid_conflicts_audit_then_explicit_exclusion(inputs, tmp_path):
    f, _ = inputs
    with f.open() as h:
        rows = list(csv.reader(h, delimiter="\t"))
    with f.open("a") as h:
        h.write("\t".join([*rows[1][:-1], "C2"]) + "\n")
    c = config(inputs, tmp_path, validate_only=True, conflict_policy="error")
    audit = ml.run_exploration(c)
    assert (
        audit.status == "invalid" and len(list(read_dicts(audit.conflicts_path))) == 2
    )
    assert not (c.output_dir / "arrays").exists()
    result = ml.run_exploration(
        replace(c, validate_only=False, resume=True, conflict_policy="exclude")
    )
    assert result.example_count == 35
    assert len(list((c.output_dir / "audits").iterdir())) == 1


def test_incompatible_partial_resume_does_not_mutate_inputs(
    inputs, tmp_path, monkeypatch
):
    c = config(inputs, tmp_path)
    original = ml._task_job

    def fail(root, cohort, similarity, plan, *args):
        if plan.fold == 1:
            raise RuntimeError("pause")
        return original(root, cohort, similarity, plan, *args)

    monkeypatch.setattr(ml, "_task_job", fail)
    with pytest.raises(RuntimeError):
        ml.run_exploration(c)
    before = {
        str(p.relative_to(c.output_dir)): sha256(p)
        for directory in ("inputs", "arrays")
        for p in (c.output_dir / directory).rglob("*")
        if p.is_file()
    }
    audit_before = (c.output_dir / "preflight.json").read_bytes()
    with pytest.raises(IntegrityError, match="Incompatible"):
        ml.run_exploration(replace(c, resume=True, knots=3))
    assert audit_before == (c.output_dir / "preflight.json").read_bytes()
    assert all(sha256(c.output_dir / p) == value for p, value in before.items())
    monkeypatch.setattr(ml, "_task_job", original)
    assert ml.run_exploration(replace(c, resume=True)).reused_folds == 1


def test_partial_aggregate_publication_is_recoverable(inputs, tmp_path, monkeypatch):
    c = config(inputs, tmp_path)
    original = ml.os.replace
    fired = []

    def fail(source, target):
        if (
            ".stage/aggregate" in str(source)
            and str(source).endswith("oof_predictions.tsv")
            and not fired
        ):
            fired.append(True)
            raise OSError("injected aggregate publication failure")
        return original(source, target)

    monkeypatch.setattr(ml.os, "replace", fail)
    with pytest.raises(OSError, match="publication"):
        ml.run_exploration(c)
    assert not (c.output_dir / "manifest.json").exists()
    monkeypatch.setattr(ml.os, "replace", original)
    result = ml.run_exploration(replace(c, resume=True))
    assert result.reused_folds == 5
    assert len(list(read_dicts(result.oof_path))) == 144


def test_changed_sources_and_foreign_outputs_rejected(inputs, tmp_path):
    c = config(inputs, tmp_path)
    ml.run_exploration(c)
    original = c.features.read_bytes()
    c.features.write_bytes(original + b"\n")
    with pytest.raises(IntegrityError, match="source"):
        ml.run_exploration(replace(c, resume=True))
    c.features.write_bytes(original)
    (c.output_dir / "foreign.txt").write_text("keep")
    with pytest.raises(IntegrityError, match="Unrecognized"):
        ml.run_exploration(replace(c, resume=True))
    assert (c.output_dir / "foreign.txt").read_text() == "keep"


def test_resource_and_config_errors(inputs, tmp_path):
    c = config(inputs, tmp_path, validate_only=True, memory_budget_gib=0.001)
    assert ml.run_exploration(c).status == "invalid"
    with pytest.raises(ValueError):
        replace(c, alpha=0)
    with pytest.raises(ValueError):
        replace(c, device="mps")
    with pytest.raises(ValueError):
        replace(c, split_mode="homology")
    with pytest.raises(ValueError):
        replace(c, resume=True)


def test_sequence_alias_conflict_excludes_whole_identity(inputs, tmp_path):
    f, _ = inputs
    with f.open() as h:
        table = list(csv.reader(h, delimiter="\t"))
    first = table[1]
    table.append(["alias", *first[1:-1], "C2"])
    with f.open("w") as h:
        csv.writer(h, delimiter="\t", lineterminator="\n").writerows(table)
    idx = tmp_path / "index.tsv"
    index = []
    for i, row in enumerate(table[1:]):
        sid = "0" * 64 if row[0] in (first[0], "alias") else f"{i + 1:064x}"
        index.append((row[0], sid, 100, 1))
    write_tsv(idx, INDEX_HEADER, index)
    with pytest.raises(LabelConflictError):
        load_cohort(f, sequence_index=idx)
    cohort = load_cohort(f, sequence_index=idx, conflict_policy="exclude")
    assert len(cohort.y) == 35 and {r[1] for r in cohort.exclusions} == {
        first[0],
        "alias",
    }


def test_missing_group_duplicate_index_and_nonfinite_rejected(inputs, tmp_path):
    f, _ = inputs
    cohort = load_cohort(f)
    groups = tmp_path / "groups.tsv"
    write_tsv(groups, ("accession_id", "group_id"), [(cohort.representatives[0], "g")])
    with pytest.raises(IntegrityError, match="Missing group"):
        load_groups(cohort, groups)
    with groups.open("a") as h:
        h.write(f"{cohort.representatives[0]}\tg\n")
    with pytest.raises(IntegrityError, match="Duplicate"):
        load_groups(cohort, groups)
    with f.open() as h:
        table = list(csv.reader(h, delimiter="\t"))
    table[1][1] = "nan"
    with f.open("w") as h:
        csv.writer(h, delimiter="\t", lineterminator="\n").writerows(table)
    with pytest.raises(IntegrityError, match="Nonfinite"):
        load_cohort(f)


def test_foreign_interrupted_stage_is_preserved(inputs, tmp_path):
    c = config(inputs, tmp_path, validate_only=True)
    ml.run_exploration(c)
    run = read_json(c.output_dir / "run.json")
    stage = c.output_dir / ".stage" / "r000-f000"
    stage.mkdir(parents=True)
    write_json(
        stage / "owner.json",
        {
            "experiment_id": run["experiment_id"],
            "repeat": 0,
            "fold": 0,
            "purpose": "task",
        },
    )
    foreign = stage / "foreign.txt"
    foreign.write_text("preserve")
    with pytest.raises(IntegrityError, match="Foreign"):
        ml.run_exploration(replace(c, validate_only=False, resume=True))
    assert foreign.read_text() == "preserve"
    assert not (c.output_dir / "manifest.json").exists()


def test_resume_preserves_ready_input_inodes(inputs, tmp_path, monkeypatch):
    c = config(inputs, tmp_path, validate_only=True)
    ml.run_exploration(c)
    arrays = c.output_dir / "arrays"
    before = {p.name: p.stat().st_ino for p in arrays.iterdir()}
    original = ml._task_job

    def fail(*args, **kwargs):
        raise RuntimeError("stop before fitting")

    monkeypatch.setattr(ml, "_task_job", fail)
    with pytest.raises(RuntimeError, match="before fitting"):
        ml.run_exploration(replace(c, validate_only=False, resume=True))
    assert before == {p.name: p.stat().st_ino for p in arrays.iterdir()}
    monkeypatch.setattr(ml, "_task_job", original)
    assert (
        ml.run_exploration(replace(c, validate_only=False, resume=True)).status
        == "complete"
    )


def test_invalid_metric_inputs_and_collapsed_duplicate(inputs):
    with pytest.raises(ValueError, match="similarity"):
        score_predictions(np.array([0]), np.array([0]), np.array([[0.0]]))
    f, _ = inputs
    with f.open() as handle:
        table = list(csv.reader(handle, delimiter="\t"))
    with f.open("a") as handle:
        handle.write("\t".join(table[1]) + "\n")
    cohort = load_cohort(f)
    assert len(cohort.y) == 36 and cohort.source_rows == 37
    assert cohort.aliases[0][-1] == 2


def test_float32_fit_independent_of_torch_default_dtype():
    X = np.zeros((10, 18))
    X[:, 0] = np.linspace(-1, 1, 10)
    y = (X[:, 0] > 0).astype(int)
    p = fit_preprocessing(X, knots=3)
    targets = make_soft_targets(
        np.eye(2), active_labels=[0, 1], alpha=0, neighbors=1, gamma=2
    )
    params = dict(
        learning_rate=0.01,
        weight_decay=0,
        batch_size=10,
        max_epochs=1,
        patience=1,
        min_delta=0,
    )
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        model, _ = fit_siac(
            X,
            y,
            preprocess=p,
            targets=targets,
            parameters=params,
            seed=42,
            device="cpu",
            epochs=1,
        )
        assert model.weight.dtype == np.float32
    finally:
        torch.set_default_dtype(previous)


def test_default_random_splits_need_no_homology_files(inputs, tmp_path):
    from sklearn.model_selection import StratifiedKFold

    from pclass_dark.ml_splits import derived_seed

    # Exercise API defaults and the actual sklearn random splitter independently.
    c = ml.ExploreMLConfig(
        features=inputs[0],
        similarities=inputs[1],
        output_dir=tmp_path / "random-api",
        device="cpu",
        validate_only=True,
    )
    assert c.split_mode == "accession" and c.groups is None
    result = ml.run_exploration(c)
    assert result.status == "ready"
    audit = read_json(result.preflight_path)
    assert audit["population"]["split_mode"] == "accession"
    assert audit["population"]["provenance_status"]["homology"] == "uncontrolled"
    cohort = load_cohort(inputs[0])
    groups = load_groups(cohort, split_mode="accession")
    plan = make_split_plan(
        cohort, groups=groups, split_mode=c.split_mode, repeats=1, seed=c.seed
    )
    oracle = StratifiedKFold(
        5, shuffle=True, random_state=derived_seed(42, 0, 0, "outer")
    )
    for actual, (train, test) in zip(
        plan.folds, oracle.split(cohort.X, cohort.y), strict=True
    ):
        assert np.array_equal(actual.train, train)
        assert np.array_equal(actual.test, test)
        counts = np.bincount(cohort.y[test], minlength=3)
        assert np.all((counts >= 2) & (counts <= 3))
        inner = StratifiedKFold(
            5,
            shuffle=True,
            random_state=derived_seed(42, actual.repeat, actual.fold, "inner"),
        )
        inner_train, inner_val = next(inner.split(cohort.X[train], cohort.y[train]))
        assert np.array_equal(actual.inner_train, train[inner_train])
        assert np.array_equal(actual.inner_validation, train[inner_val])
    changed = make_split_plan(
        cohort, groups=groups, split_mode="accession", repeats=1, seed=43
    )
    assert any(
        not np.array_equal(a.test, b.test)
        for a, b in zip(plan.folds, changed.folds, strict=True)
    )

    # Omit --split-mode from the command as well as all grouping file options.
    cli = CliRunner().invoke(
        app,
        [
            "explore-ml",
            "--features",
            str(inputs[0]),
            "--similarities",
            str(inputs[1]),
            "--output-dir",
            str(tmp_path / "random-cli"),
            "--device",
            "cpu",
            "--validate-only",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert read_json(tmp_path / "random-cli" / "preflight.json")["status"] == "ready"


def test_default_eight_workers_and_cpu_task_limits(inputs, tmp_path, monkeypatch):
    monkeypatch.setattr(ml, "_available_cpus", lambda: 2)
    c = ml.ExploreMLConfig(
        features=inputs[0],
        similarities=inputs[1],
        output_dir=tmp_path / "workers",
        device="cpu",
        memory_budget_gib=100,
    )
    assert c.workers == 8  # A two-CPU machine still accepts the default.
    cohort = load_cohort(inputs[0])
    similarity = load_similarity(inputs[1], labels=cohort.labels)
    _, resources = ml._resources(c, cohort, similarity, "cpu")
    assert resources["requested_workers"] == 8
    assert resources["effective_workers"] == 2
    monkeypatch.setattr(ml, "_available_cpus", lambda: 16)
    assert ml._resources(c, cohort, similarity, "cpu")[1]["effective_workers"] == 5
    assert (
        ml._resources(replace(c, repeats=2), cohort, similarity, "cpu")[1][
            "effective_workers"
        ]
        == 8
    )
    assert (
        ml._resources(replace(c, workers=1), cohort, similarity, "cpu")[1][
            "effective_workers"
        ]
        == 1
    )
    with pytest.raises(ValueError):
        replace(c, workers=9)
    cli = CliRunner().invoke(
        app,
        [
            "explore-ml",
            "--features",
            str(inputs[0]),
            "--similarities",
            str(inputs[1]),
            "--output-dir",
            str(tmp_path / "workers-cli"),
            "--device",
            "cpu",
            "--validate-only",
        ],
    )
    assert cli.exit_code == 0, cli.output
    audit = read_json(tmp_path / "workers-cli" / "preflight.json")
    assert audit["resources"]["requested_workers"] == 8
    assert audit["resources"]["effective_workers"] <= 5


def test_cli_progress_and_actionable_conflict_audit(inputs, tmp_path):
    import logging

    runner = CliRunner()
    logger = logging.getLogger("pclass_dark")
    before_handlers, before_level = list(logger.handlers), logger.level
    command = [
        "explore-ml",
        "--features",
        str(inputs[0]),
        "--similarities",
        str(inputs[1]),
        "--output-dir",
        str(tmp_path / "logged"),
        "--device",
        "cpu",
        "--workers",
        "1",
        "--max-epochs",
        "1",
    ]
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    for message in (
        "Preflight:",
        "Reading feature table",
        "Similarity pass 1/2",
        "Similarity pass 2/2",
        "Preflight ready:",
        "Starting repeat 1 fold 1/5",
        "Stage 1/4 SIAC:epoch-selection] epoch 1/1",
        "Stage 4/4 HARD:outer-refit] epoch 1/1",
        "Committed repeat 1 fold 5/5",
        "Aggregating held-out predictions",
        "Exploration complete:",
    ):
        assert message in result.stderr
    assert logger.handlers == before_handlers and logger.level == before_level
    assert "summary.json" in result.stdout
    assert "Repeat 1/1 | Fold 1/5" in result.stderr
    assert "Stage 2/4 SIAC:outer-refit] epoch 1/1" in result.stderr
    assert "Stage 3/4 HARD:epoch-selection] epoch 1/1" in result.stderr
    assert "validation_loss=" not in result.stderr
    assert "best_epoch=" not in result.stderr
    history = list(
        read_dicts(
            tmp_path / "logged" / "folds" / "r000" / "f000" / "hard" / "training.tsv"
        )
    )
    assert any(
        row["validation_loss"] != "NA" for row in history if row["phase"] == "selection"
    )

    # Repeated CLI invocations must not retain handlers pointing at closed streams.
    second = runner.invoke(app, [*command, "--resume"])
    assert second.exit_code == 0, second.output
    assert "Reusing verified complete experiment" in second.stderr
    assert logger.handlers == before_handlers

    with inputs[0].open() as handle:
        table = list(csv.reader(handle, delimiter="\t"))
    with inputs[0].open("a") as handle:
        handle.write("\t".join([*table[1][:-1], "C2"]) + "\n")
    conflict_command = [*command, "--conflict-policy", "error"]
    conflict_command[conflict_command.index("--output-dir") + 1] = str(
        tmp_path / "logged-conflicts"
    )
    failed = runner.invoke(app, conflict_command)
    assert failed.exit_code == 1
    assert "multiple cluster_id labels for the same sequence" in failed.stderr
    assert "conflicts.tsv" in failed.stderr
    assert "--resume --conflict-policy exclude" in failed.stderr
    assert "Starting repeat" not in failed.stderr
    assert (tmp_path / "logged-conflicts" / "conflicts.tsv").is_file()


def test_progress_counter_throttling(caplog, monkeypatch):
    import logging

    from pclass_dark import ml_progress

    now = [0.0]
    monkeypatch.setattr(ml_progress.time, "monotonic", lambda: now[0])
    with caplog.at_level(logging.INFO):
        counter = ml_progress.Progress(
            logging.getLogger("siac-counter-test"), "Scan", total=100
        )
        counter.update(1)
        assert len(caplog.records) == 1  # Start only; throttle small updates.
        now[0] = 5.0
        counter.update(50)
        assert "50.0%" in caplog.records[-1].message
        assert len(caplog.records) == 2
        counter.update(60)
        assert len(caplog.records) == 2
        counter.finish(100)
    assert "completed 100 rows" in caplog.records[-1].message


def test_spawned_workers_emit_progress(inputs, tmp_path):
    if (os.cpu_count() or 1) < 2:
        pytest.skip("At least two CPUs required")
    # A real script subprocess verifies spawn initialization, not parent-only logs.
    command = [
        sys.executable,
        "scripts/explore-ml.py",
        "--features",
        str(inputs[0]),
        "--similarities",
        str(inputs[1]),
        "--output-dir",
        str(tmp_path / "spawn-logs"),
        "--workers",
        "2",
        "--device",
        "cpu",
        "--max-epochs",
        "1",
        "--memory-budget-gib",
        "100",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "SpawnProcess" in result.stderr
    assert "Stage 1/4 SIAC:epoch-selection] epoch 1/1" in result.stderr
    assert "Fold tasks completed: 5/5" in result.stderr


def test_default_prefilter_deduplicates_and_excludes_ambiguous_sequences(
    inputs, tmp_path
):
    f, _ = inputs
    with f.open() as handle:
        table = list(csv.reader(handle, delimiter="\t"))
    # One same-label sequence alias and one conflicting-label alias.
    table.extend(
        [
            ["alias_same", *table[1][1:]],
            ["alias_conflict", *table[2][1:-1], "C2"],
        ]
    )
    with f.open("w") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(table)
    index = tmp_path / "features.sequences.tsv"
    index_rows = []
    for j, row in enumerate(table[1:]):
        if row[0] == "alias_same":
            sid = f"{1:064x}"
        elif row[0] == "alias_conflict":
            sid = f"{2:064x}"
        else:
            sid = f"{j + 1:064x}"
        index_rows.append((row[0], sid, 100, 1))
    write_tsv(index, INDEX_HEADER, index_rows)
    c = ml.ExploreMLConfig(
        features=f,
        similarities=inputs[1],
        output_dir=tmp_path / "prefilter",
        device="cpu",
        validate_only=True,
    )
    assert c.conflict_policy == "exclude"
    result = ml.run_exploration(c)
    assert result.status == "ready"
    counts = read_json(result.preflight_path)["population"]["prefilter_counts"]
    assert counts == {
        "source_rows": 38,
        "source_accessions": 38,
        "unique_identities": 36,
        "repeated_identity_rows": 2,
        "same_target_rows_collapsed": 1,
        "ambiguous_identities_removed": 1,
        "ambiguous_source_rows_removed": 2,
        "total_rows_removed": 3,
        "retained_sequences": 35,
        "identity_basis": "sequence_hash",
    }
    assert len(list(read_dicts(c.output_dir / "exclusions.tsv"))) == 2
    # Different accessions with equal rounded descriptors aren't sequence proof.
    without_hashes = load_cohort(f, conflict_policy="exclude")
    assert len(without_hashes.y) == 38
    assert without_hashes.prefilter_counts["identity_basis"] == "accession_only"


def test_default_cli_prefilter_logs_and_proceeds(inputs, tmp_path):
    f, _ = inputs
    with f.open() as handle:
        table = list(csv.reader(handle, delimiter="\t"))
    with f.open("a") as handle:
        handle.write("\t".join([*table[1][:-1], "C2"]) + "\n")
        handle.write("\t".join(table[2]) + "\n")
    result = CliRunner().invoke(
        app,
        [
            "explore-ml",
            "--features",
            str(f),
            "--similarities",
            str(inputs[1]),
            "--output-dir",
            str(tmp_path / "auto-filter"),
            "--device",
            "cpu",
            "--workers",
            "1",
            "--max-epochs",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "removed 1 ambiguous sequences (2 source rows)" in result.stderr
    assert (
        "collapsed 1 same-target duplicate rows; retained 35 sequences" in result.stderr
    )
    summary = read_json(tmp_path / "auto-filter" / "summary.json")
    counts = summary["population"]["prefilter_counts"]
    assert counts["total_rows_removed"] == 3
    assert counts["retained_sequences"] == 35
    assert all(row["n"] == 35 for row in summary["repeat_metrics"])
