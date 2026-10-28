"""Training-only exact balanced-accuracy thresholds and fixed-cutoff metrics."""

from __future__ import annotations

import math

import numpy as np

from .profiles import auroc_counts

METRICS = (
    "tp",
    "fp",
    "fn",
    "tn",
    "auroc",
    "precision",
    "recall",
    "specificity",
    "accuracy",
    "balanced_accuracy",
    "f1",
    "false_positive_rate",
    "average_precision",
    "positive_prevalence",
)


def _groups(p, b):
    scores = np.concatenate((p, b))
    labels = np.concatenate(
        (np.ones(len(p), dtype=np.int64), np.zeros(len(b), dtype=np.int64))
    )
    order = np.argsort(-scores, kind="stable")
    scores, labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    return (
        scores,
        ends,
        np.cumsum(labels)[ends] if len(scores) else np.array([], dtype=np.int64),
    )


def choose_threshold(positive, negative):
    p, b = (np.asarray(x, dtype=np.float64) for x in (positive, negative))
    if not len(p) or not len(b) or not np.isfinite(p).all() or not np.isfinite(b).all():
        raise ValueError("Threshold fitting requires finite nonempty C and B scores")
    scores, ends, tp = _groups(p, b)
    best_objective = len(p) * len(b)  # Predict none: TP=0, TN=len(B).
    threshold = {"kind": "above_maximum", "log_value": None, "score_value": None}
    for end, true_positive in zip(ends, tp, strict=True):
        fp = int(end + 1 - true_positive)
        objective = int(true_positive) * len(b) + (len(b) - fp) * len(p)
        # Descending score groups implement highest-cutoff tie resolution.
        if objective > best_objective:
            best_objective = objective
            value = float(scores[end])
            threshold = {
                "kind": "log_score_ge",
                "log_value": value,
                "score_value": math.exp(value),
            }
    return {
        **threshold,
        "objective": best_objective,
        "rule": "max-training-balanced-accuracy-highest-cutoff-v1",
    }


def accepted(scores, threshold):
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("Acceptance needs finite log scores")
    if threshold["kind"] == "above_maximum":
        if threshold["log_value"] is not None or threshold["score_value"] is not None:
            raise ValueError("Invalid predict-none threshold")
        return np.zeros(scores.shape, dtype=bool)
    if threshold["kind"] != "log_score_ge" or not math.isfinite(threshold["log_value"]):
        raise ValueError("Invalid threshold")
    return scores >= threshold["log_value"]


def evaluate(positive, negative, threshold):
    p, b = (np.asarray(x, dtype=np.float64) for x in (positive, negative))
    pa, ba = accepted(p, threshold), accepted(b, threshold)
    tp, fp = int(pa.sum()), int(ba.sum())
    fn, tn = len(p) - tp, len(b) - fp

    def ratio(a, denominator):
        return a / denominator if denominator else None

    precision, recall = ratio(tp, tp + fp), ratio(tp, len(p))
    specificity = ratio(tn, len(b))
    auc = auroc_counts(p, b)
    ap = None
    if len(p) and len(b):
        _, ends, cumulative = _groups(p, b)
        ap = float(
            np.sum(np.diff(np.r_[0, cumulative]) / len(p) * (cumulative / (ends + 1)))
        )
    values = {
        "positive_n": len(p),
        "negative_n": len(b),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "auroc": auc["value"],
        "auroc_numerator": auc["numerator"],
        "auroc_denominator": auc["denominator"],
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": ratio(tp + tn, len(p) + len(b)),
        "balanced_accuracy": (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None,
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
        "false_positive_rate": ratio(fp, len(b)),
        "average_precision": ap,
        "positive_prevalence": ratio(len(p), len(p) + len(b)),
    }
    values["unavailable_reasons"] = {
        k: "undefined_denominator_or_population" for k, v in values.items() if v is None
    }
    values["status"] = (
        "ok"
        if len(p) and len(b)
        else "empty_validation"
        if not len(p)
        else "empty_background"
    )
    return values
