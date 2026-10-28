"""Independent small population oracles for profiles, selection, and metrics."""

import itertools
import math

import numpy as np
import pytest

from pclass_dark.profiles import auroc_counts, dimension_profile, gaussian_log
from pclass_dark.signature_metrics import accepted, choose_threshold, evaluate
from pclass_dark.signatures import composite_log, select_dimensions


def test_moments_conservation_discrimination():
    result = dimension_profile([1.0, 3.0], [0.0, 4.0], [2.0])
    assert result["mu_C"] == 2
    assert result["sigma_C"] == pytest.approx(math.sqrt(2))
    assert result["sigma_B"] == pytest.approx(math.sqrt(8))
    assert result["R"] == pytest.approx(math.sqrt(2) / (math.sqrt(8) + 1e-8))
    assert result["D"] == 0
    assert result["training_auroc"] == 1
    assert result["validation_auroc"] == 1


def test_auroc_matches_pairwise_oracle():
    rng = np.random.default_rng(8)
    for _ in range(30):
        p, b = rng.integers(-3, 4, size=3), rng.integers(-3, 4, size=7)
        q = sum(2 if x > y else 1 if x == y else 0 for x in p for y in b)
        count = auroc_counts(p, b)
        assert count == {"value": q / 42, "numerator": q, "denominator": 42}
        assert auroc_counts(b, p)["numerator"] == 42 - q
    assert auroc_counts([1], [1])["value"] == 0.5
    assert auroc_counts([], [1])["value"] is None


def test_gaussian_and_arithmetic_mean_log_score():
    kernels = np.array([gaussian_log([0.0, 1.0], 0, 1), gaussian_log([2.0, 0.0], 0, 1)])
    actual = composite_log(kernels, [0, 1])
    expected = np.log((np.exp(kernels[0]) + np.exp(kernels[1])) / 2)
    assert np.allclose(actual, expected)
    assert not np.allclose(actual, (kernels[0] + kernels[1]) / 2)
    assert np.isfinite(composite_log(np.array([[-1000.0], [-1001.0]]), [0, 1])).all()


def test_threshold_independent_enumeration_and_validation():
    p, b = np.array([-0.1, -0.2, -0.8]), np.array([-0.3, -0.6, -0.8, -1.0])
    threshold = choose_threshold(p, b)
    candidates = [None, *sorted(set([*p, *b]), reverse=True)]
    objectives = []
    for cutoff in candidates:
        tp = sum(x >= cutoff for x in p) if cutoff is not None else 0
        tn = sum(x < cutoff for x in b) if cutoff is not None else len(b)
        objectives.append(tp * len(b) + tn * len(p))
    best = candidates[objectives.index(max(objectives))]
    assert threshold["log_value"] == best
    metrics = evaluate(p, b, threshold)
    validation = evaluate(np.array([-0.15, -2.0]), b, threshold)
    assert metrics["specificity"] == validation["specificity"]
    assert metrics["false_positive_rate"] == validation["false_positive_rate"]
    assert metrics["tp"] + metrics["fn"] == 3
    assert validation["tp"] == 1
    tied = choose_threshold(np.zeros(3), np.zeros(4))
    assert tied["kind"] == "above_maximum"
    assert not accepted(np.zeros(3), tied).any()
    assert evaluate(np.zeros(3), np.zeros(4), tied)["precision"] is None


@pytest.mark.parametrize("fit_id", ["fold_1", "final"])
def test_selection_protects_core_and_accepts_first_gain(monkeypatch, caplog, fit_id):
    ticks = itertools.count(step=31)
    monkeypatch.setattr("pclass_dark.signatures.time.monotonic", lambda: next(ticks))
    caplog.set_level("INFO", logger="pclass_dark.signatures")
    order = [0, 1, 2, 3]
    individual = {
        0: {"numerator": 99, "denominator": 100},
        1: {"numerator": 98, "denominator": 100},
        2: {"numerator": 95, "denominator": 100},
        3: {"numerator": 90, "denominator": 100},
    }
    prefixes = [
        {"numerator": n, "denominator": 100, "value": n / 100} for n in [80, 85, 89, 90]
    ]
    scores = {(0, 1, 2, 3): 80, (0, 1, 3): 91, (0, 1, 2): 85, (0, 1): 95}
    result = select_dimensions(
        order,
        individual,
        prefixes,
        lambda dims: {
            "numerator": scores[tuple(dims)],
            "denominator": 100,
            "value": scores[tuple(dims)] / 100,
        },
        cluster_id="cluster_0004",
        fit_id=fit_id,
    )
    assert (
        f"Cluster cluster_0004 {fit_id}: Removal iteration 0: 1/2 trials"
        in caplog.messages
    )
    assert result["core"] == [0, 1]
    assert result["selected"] == [0, 1]
    assert [r["removed"] for r in result["iterations"]] == [2, 3, None]


def test_single_background_and_constant_features():
    profile = dimension_profile([0.0, 0.0], [0.0], [0.0])
    assert profile["R"] is None and profile["D"] is None
    assert profile["training_auroc"] == 0.5
    assert profile["effective_sigma"] == 1e-8
    assert np.isfinite(gaussian_log([100.0], 0, 1e-8)).all()


def test_tied_average_precision_and_threshold_objective():
    tied = evaluate(
        np.array([-1.0, -1.0]),
        np.array([-1.0, -1.0]),
        choose_threshold(np.array([-1.0, -1.0]), np.array([-1.0, -1.0])),
    )
    assert tied["average_precision"] == 0.5
    p, b = np.array([0.0, -2.0]), np.array([-1.0, -3.0])
    cutoff = choose_threshold(p, b)
    assert cutoff["log_value"] == 0
    metrics = evaluate(p, b, cutoff)
    assert metrics["average_precision"] == pytest.approx(5 / 6)
    assert metrics["auroc"] == 0.75


def test_exact_candidate_cutoff_and_protected_core():
    order = list(range(5))
    individual = {
        d: {"numerator": n, "denominator": 100}
        for d, n in enumerate([99, 98, 91, 90, 91])
    }
    prefixes = [
        {"numerator": n, "denominator": 100, "value": n / 100}
        for n in [80, 95, 95, 94, 94]
    ]
    calls = []

    def evaluate_subset(dims):
        calls.append(dims)
        assert 0 in dims and 1 in dims
        return {"numerator": 90, "denominator": 100, "value": 0.9}

    result = select_dimensions(order, individual, prefixes, evaluate_subset)
    assert result["prefilter_peak_length"] == 2
    assert result["candidates"] == [0, 1, 2, 4]
    assert calls


def test_overflow_is_a_numeric_error_not_a_dropped_dimension():
    with pytest.raises((ValueError, FloatingPointError, OverflowError)):
        gaussian_log([np.finfo(float).max], 0.0, 1e-300)


def test_first_improvement_skips_later_better_trial():
    calls = []

    def evaluate(dimensions):
        calls.append(tuple(dimensions))
        # Removing dimension 3 would improve more, but dimension 2 is visited first.
        numerator = {(0, 1, 2, 3): 50, (0, 1, 3): 60, (0, 1, 2): 90, (0, 1): 55}[
            tuple(dimensions)
        ]
        return {"numerator": numerator, "denominator": 100, "value": numerator / 100}

    individual = {d: {"numerator": 95, "denominator": 100} for d in range(4)}
    prefixes = [{"numerator": 50, "denominator": 100}] * 4
    result = select_dimensions(list(range(4)), individual, prefixes, evaluate)
    assert result["selected"] == [0, 1, 3]
    assert (0, 1, 2) not in calls
    assert [t["dimension"] for t in result["iterations"][0]["trials"]] == [2]


@pytest.mark.parametrize("scale", [1, 1000, 1000000])
def test_fast_removal_matches_reference_and_reuses_buffers(scale):
    from pclass_dark.signatures import RemovalScores

    rng = np.random.default_rng(91)
    kernels = {k: -rng.uniform(0, scale, (25, n)) for k, n in [("C", 11), ("B", 37)]}
    evaluator = RemovalScores(kernels)
    selected = list(range(25))
    for omitted in [2, 11, 24]:
        p, b = evaluator.scores(selected, omitted)
        buffers = (id(p), id(b))
        for key, actual in [("C", p), ("B", b)]:
            expected = composite_log(
                kernels[key], [d for d in selected if d != omitted]
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-10)
            np.testing.assert_array_equal(evaluator.reference(key, omitted), expected)
            assert len(evaluator.prefixes[key]) <= 8
        assert buffers == tuple(id(x) for x in evaluator.scores(selected, omitted))
    selected.remove(11)
    p, b = evaluator.scores(selected, 24)
    assert evaluator.selected == tuple(selected)
    np.testing.assert_allclose(p, composite_log(kernels["C"], selected[:-1]))


def test_fast_removal_dominant_term_underflow_and_near_ties():
    from pclass_dark.signatures import RemovalScores

    kernels = {
        k: np.array([[-10000.0, -1e9], [-10001.0, -1e9], [0.0, 0.0], [-10002.0, -1e9]])
        for k in ["C", "B"]
    }
    evaluator = RemovalScores(kernels)
    p, b = evaluator.scores([0, 1, 2, 3], 2)
    np.testing.assert_array_equal(p, composite_log(kernels["C"], [0, 1, 3]))
    assert np.isfinite(p).all()
    result = evaluator([0, 1, 2, 3], 2, {"numerator": 0})
    assert result["value"] == 0.5
    np.testing.assert_array_equal(p, b)


def test_selection_version_invalidates_old_receipts(tmp_path, monkeypatch):
    import pclass_dark.signatures as module
    from pclass_dark.library_checkpoints import Checkpoints

    store = Checkpoints(tmp_path, "fit")
    individual = {d: {"numerator": 90, "denominator": 100} for d in range(3)}
    prefixes = [{"numerator": 50, "denominator": 100}] * 3
    calls = []

    def evaluate(dimensions):
        calls.append(tuple(dimensions))
        return {"numerator": 50, "denominator": 100, "value": 0.5}

    monkeypatch.setattr(module, "SELECTION_VERSION", "old-version")
    select_dimensions([0, 1, 2], individual, prefixes, evaluate, store)
    calls.clear()
    monkeypatch.setattr(module, "SELECTION_VERSION", "new-version")
    select_dimensions([0, 1, 2], individual, prefixes, evaluate, store)
    assert calls
    assert list((tmp_path / ".quarantine").iterdir())


@pytest.mark.parametrize("seed", range(5))
def test_fast_first_improvement_matches_reference_selection(seed):
    from pclass_dark.signatures import RemovalScores

    rng = np.random.default_rng(seed)
    kernels = {k: -rng.uniform(0, 8, (15, n)) for k, n in [("C", 9), ("B", 23)]}
    denominator = 2 * 9 * 23
    individual = {
        d: {"numerator": denominator, "denominator": denominator} for d in range(15)
    }
    prefixes = [{"numerator": 1, "denominator": denominator}] * 15

    def evaluate(dimensions):
        return auroc_counts(
            *(composite_log(kernels[k], dimensions) for k in ["C", "B"])
        )

    expected = select_dimensions(list(range(15)), individual, prefixes, evaluate)
    actual = select_dimensions(
        list(range(15)),
        individual,
        prefixes,
        evaluate,
        evaluate_removal=RemovalScores(kernels),
    )
    assert actual == expected
