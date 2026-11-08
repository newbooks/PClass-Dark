"""SIAC metric oracles, stable TSV schemas and out-of-fold report production."""

from __future__ import annotations

import csv
import math
import os
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .artifacts import IntegrityError, file_record, write_json
from .ml_data import FEATURES

CONFIGS = ("siac", "hard", "majority", "semantic_constant")
HEADERS = {
    "folds.tsv": ("repeat_id", "outer_fold_id", "example_id", "group_id"),
    "inner_folds.tsv": ("repeat_id", "outer_fold_id", "example_id", "role"),
    "labels.tsv": ("label_index", "cluster_id", "example_count", "medoid_accession"),
    "fold_metrics.tsv": tuple(
        (
            "repeat_id config_id outer_fold_id stratum stratum_value n "
            "n_exact similarity_sum "
            "exact_hit_rate weighted_hit_rate macro_recall macro_similarity "
            "top5_hit_rate"
        ).split()
    ),
    "per_cluster.tsv": tuple(
        (
            "repeat_id config_id scope outer_fold_id cluster_id n n_exact "
            "similarity_sum "
            "recall mean_similarity"
        ).split()
    ),
    "oof_predictions.tsv": tuple(
        (
            "repeat_id config_id outer_fold_id example_id true_cluster_id "
            "pred_cluster_id "
            "train_class_count exact_hit similarity_credit top5_hit "
            "runner_up_cluster_id logit_margin"
        ).split()
    ),
    "oof_topk.tsv": tuple(
        (
            "repeat_id config_id outer_fold_id example_id rank cluster_id "
            "logit softmax_score"
        ).split()
    ),
    "stability.tsv": tuple(
        (
            "config_id example_id n_repeats modal_cluster_id "
            "modal_agreement n_distinct semantic_agreement"
        ).split()
    ),
    "explanations.tsv": (
        "repeat_id",
        "config_id",
        "outer_fold_id",
        "example_id",
        "pred_cluster_id",
        "runner_up_cluster_id",
        "logit_margin",
        "bias_difference",
        *FEATURES,
        "imputed_features",
        "outside_support_features",
    ),
}
METRIC_NAMES = (
    "n",
    "n_exact",
    "similarity_sum",
    "exact_hit_rate",
    "weighted_hit_rate",
    "macro_recall",
    "macro_similarity",
    "top5_hit_rate",
)


def cell(value):
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise IntegrityError("Nonfinite report number")
        return format(value, ".17g")
    if isinstance(value, (bool, np.bool_)):
        return str(int(value))
    return str(value)


def write_tsv(path, header, data):
    """Durably write deterministic plain TSV and return its inventory record."""
    path = Path(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        n = 0
        for row in data:
            if len(row) != len(header):
                raise IntegrityError("Output row width mismatch")
            writer.writerow([cell(v) for v in row])
            n += 1
        handle.flush()
        os.fsync(handle.fileno())
    return {**file_record(path), "rows": n, "columns": list(header)}


def read_dicts(path):
    with Path(path).open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def score_predictions(truth, predicted, similarity, *, labels=None, top5=None):
    """Compute exact/micro recall, semantic credit and true-label macro metrics."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    S = similarity.S if hasattr(similarity, "S") else np.asarray(similarity)
    if not hasattr(similarity, "S") and (
        S.ndim != 2
        or S.shape != (len(S), len(S))
        or not np.isfinite(S).all()
        or (S < 0).any()
        or (S > 1).any()
        or not np.all(np.diag(S) == 1)
        or not np.array_equal(S, S.T)
    ):
        raise ValueError("Invalid similarity metric matrix")
    if labels is not None and truth.dtype.kind not in "iu":
        index = {label: i for i, label in enumerate(labels)}
        truth = np.array([index[x] for x in truth])
        predicted = np.array([index[x] for x in predicted])
    if truth.shape != predicted.shape or truth.ndim != 1:
        raise ValueError("Truth and predictions must be aligned vectors")
    if len(truth) and (
        truth.dtype.kind not in "iu"
        or predicted.dtype.kind not in "iu"
        or min(truth.min(), predicted.min()) < 0
        or max(truth.max(), predicted.max()) >= len(S)
    ):
        raise ValueError("Invalid global prediction label index")
    n = len(truth)
    exact = truth == predicted
    credit = S[truth, predicted] if n else np.array([], dtype=float)
    counts = np.bincount(truth, minlength=len(S)) if n else np.zeros(len(S))
    hits = np.bincount(truth, weights=exact, minlength=len(S)) if n else counts
    credits = np.bincount(truth, weights=credit, minlength=len(S)) if n else counts
    present = counts > 0
    result = {
        "n": n,
        "n_exact": int(exact.sum()),
        "similarity_sum": float(credit.sum(dtype=np.float64)),
        "exact_hit_rate": float(exact.mean()) if n else None,
        "weighted_hit_rate": float(credit.mean()) if n else None,
        "macro_recall": float(np.mean(hits[present] / counts[present])) if n else None,
        "macro_similarity": float(np.mean(credits[present] / counts[present]))
        if n
        else None,
        "top5_hit_rate": float(np.mean(top5)) if top5 is not None and n else None,
    }
    if n and not -1e-12 <= result["weighted_hit_rate"] - result["exact_hit_rate"]:
        raise IntegrityError("Similarity credit below exact credit")
    return result


def size_bin(count):
    if count == 0:
        return "0"
    if count < 20:
        return "1-19"
    if count < 100:
        return "20-99"
    if count < 500:
        return "100-499"
    return "500+"


def metric_rows(cohort, similarity, plan, config, predicted, top5):
    truth = cohort.y[plan.test]
    train_counts = np.bincount(cohort.y[plan.train], minlength=len(cohort.labels))
    support = train_counts[truth]
    strata = [("all", "all", np.ones(len(truth), dtype=bool))]
    strata.extend(
        ("train_support", value, (support > 0) == flag)
        for value, flag in (("supported", True), ("unsupported", False))
    )
    bins = np.array([size_bin(c) for c in support])
    strata.extend(
        ("train_size", b, bins == b) for b in ("0", "1-19", "20-99", "100-499", "500+")
    )
    for stratum, value, mask in strata:
        m = score_predictions(
            truth[mask],
            predicted[mask],
            similarity,
            top5=top5[mask] if top5 is not None else None,
        )
        yield (
            plan.repeat,
            config,
            plan.fold,
            stratum,
            value,
            *[m[k] for k in METRIC_NAMES],
        )


def cluster_rows(cohort, similarity, plan, config, predicted):
    y = cohort.y[plan.test]
    counts = np.bincount(y, minlength=len(cohort.labels))
    hits = np.bincount(y, weights=(y == predicted), minlength=len(cohort.labels))
    credits = np.bincount(
        y, weights=similarity.S[y, predicted], minlength=len(cohort.labels)
    )
    for j, label in enumerate(cohort.labels):
        n = int(counts[j])
        yield (
            plan.repeat,
            config,
            "fold",
            plan.fold,
            label,
            n,
            int(hits[j]),
            float(credits[j]),
            float(hits[j] / n) if n else None,
            float(credits[j] / n) if n else None,
        )


def produce_task_reports(path, cohort, similarity, plan, models, *, batch_size):
    """Stream all four configurations for one outer fold; no full probability matrix."""
    from .siac import predict_siac

    path = Path(path)
    train_counts = np.bincount(cohort.y[plan.train], minlength=len(cohort.labels))
    active = np.flatnonzero(train_counts)
    majority = active[np.argmax(train_counts[active])]
    # Multiply the shared matrix first; selecting columns would copy C-by-active
    # values in each worker. Only the resulting score vector needs indexing.
    central = active[np.argmax((train_counts @ similarity.S)[active])]
    truth = cohort.y[plan.test]
    metrics, clusters = [], []
    names = ("oof_predictions.tsv", "oof_topk.tsv", "explanations.tsv")
    with ExitStack() as stack:
        handles, writers = {}, {}
        for name in names:
            handle = stack.enter_context(
                (path / name).open("w", encoding="utf-8", newline="")
            )
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(HEADERS[name])
            handles[name], writers[name] = handle, writer

        def emit(name, values):
            writers[name].writerow([cell(v) for v in values])

        for config in CONFIGS:
            predicted = np.empty(len(truth), dtype=np.int64)
            hits5 = np.empty(len(truth), dtype=bool) if config in models else None
            if config in models:
                model = models[config]
                for result in predict_siac(
                    model, cohort.X[plan.test], batch_size=batch_size
                ):
                    start = result.start
                    global_top = model.active_labels[result.indices]
                    stop = start + len(global_top)
                    predicted[start:stop] = global_top[:, 0]
                    hits5[start:stop] = (global_top == truth[start:stop, None]).any(
                        axis=1
                    )
                    for j, ix in enumerate(plan.test[start:stop]):
                        prefix = (
                            plan.repeat,
                            config,
                            plan.fold,
                            cohort.example_ids[ix],
                        )
                        first, runner = global_top[j, :2]
                        margin = float(result.logits[j, 0] - result.logits[j, 1])
                        reconstructed = (
                            result.contributions[j].sum() + result.bias_difference[j]
                        )
                        if not np.isclose(margin, reconstructed, atol=1e-4, rtol=1e-4):
                            raise IntegrityError("Additive margin does not reconcile")
                        emit(
                            "oof_predictions.tsv",
                            (
                                *prefix,
                                cohort.labels[truth[start + j]],
                                cohort.labels[first],
                                int(train_counts[truth[start + j]]),
                                first == truth[start + j],
                                similarity.S[truth[start + j], first],
                                hits5[start + j],
                                cohort.labels[runner],
                                margin,
                            ),
                        )
                        for rank, c in enumerate(global_top[j], 1):
                            emit(
                                "oof_topk.tsv",
                                (
                                    *prefix,
                                    rank,
                                    cohort.labels[c],
                                    result.logits[j, rank - 1],
                                    result.probabilities[j, rank - 1],
                                ),
                            )
                        x = cohort.X[ix]
                        imputed = [FEATURES[k] for k in np.flatnonzero(np.isnan(x))]
                        outside = [
                            FEATURES[k]
                            for k in np.flatnonzero(
                                (x < model.preprocess.training_min)
                                | (x > model.preprocess.training_max)
                            )
                        ]
                        emit(
                            "explanations.tsv",
                            (
                                *prefix,
                                cohort.labels[first],
                                cohort.labels[runner],
                                margin,
                                result.bias_difference[j],
                                *result.contributions[j],
                                ";".join(imputed) or "none",
                                ";".join(outside) or "none",
                            ),
                        )
            else:
                predicted[:] = majority if config == "majority" else central
                for j, ix in enumerate(plan.test):
                    pred = predicted[j]
                    emit(
                        "oof_predictions.tsv",
                        (
                            plan.repeat,
                            config,
                            plan.fold,
                            cohort.example_ids[ix],
                            cohort.labels[truth[j]],
                            cohort.labels[pred],
                            int(train_counts[truth[j]]),
                            pred == truth[j],
                            similarity.S[truth[j], pred],
                            None,
                            None,
                            None,
                        ),
                    )
            metrics.extend(
                metric_rows(cohort, similarity, plan, config, predicted, hits5)
            )
            clusters.extend(cluster_rows(cohort, similarity, plan, config, predicted))
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
    write_tsv(path / "fold_metrics.tsv", HEADERS["fold_metrics.tsv"], metrics)
    write_tsv(path / "per_cluster.tsv", HEADERS["per_cluster.tsv"], clusters)


def _descriptive(values):
    values = np.array(values, dtype=float)
    if not len(values):
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def aggregate_reports(
    path, task_paths, cohort, similarity, *, repeats, experiment_id, population
):
    """Rebuild aggregates from immutable task reports, verifying exact OOF coverage."""
    path = Path(path)
    tasks = sorted(task_paths)
    index = {e: i for i, e in enumerate(cohort.example_ids)}
    label_index = {c: i for i, c in enumerate(cohort.labels)}
    predictions = {
        c: np.full((repeats, len(index)), -1, dtype=np.int32) for c in CONFIGS
    }
    top5 = {c: np.zeros((repeats, len(index)), dtype=bool) for c in CONFIGS[:2]}
    cluster_sums = {}
    fold_summaries = []
    for name in (
        "oof_predictions.tsv",
        "oof_topk.tsv",
        "explanations.tsv",
        "fold_metrics.tsv",
    ):

        def combined(name=name):
            for r in range(repeats):
                for config in CONFIGS:
                    for task in tasks:
                        if int(task.parent.name[1:]) != r:
                            continue
                        for row in read_dicts(task / name):
                            if row["config_id"] != config:
                                continue
                            if name == "oof_predictions.tsv":
                                ix = index[row["example_id"]]
                                pred = label_index[row["pred_cluster_id"]]
                                if predictions[config][r, ix] != -1:
                                    raise IntegrityError("Duplicate OOF prediction")
                                true = cohort.y[ix]
                                if (
                                    row["true_cluster_id"] != cohort.labels[true]
                                    or int(row["exact_hit"]) != (true == pred)
                                    or not np.isclose(
                                        float(row["similarity_credit"]),
                                        similarity.S[true, pred],
                                        atol=1e-12,
                                        rtol=0,
                                    )
                                ):
                                    raise IntegrityError("OOF truth/credit mismatch")
                                predictions[config][r, ix] = pred
                                if config in top5:
                                    top5[config][r, ix] = bool(int(row["top5_hit"]))
                            if name == "fold_metrics.tsv" and row["stratum"] == "all":
                                fold_summaries.append(row)
                            yield tuple(row[h] for h in HEADERS[name])

        write_tsv(path / name, HEADERS[name], combined())
    if any((a < 0).any() for a in predictions.values()):
        raise IntegrityError("Incomplete OOF coverage")
    per_rows = []
    for task in tasks:
        for row in read_dicts(task / "per_cluster.tsv"):
            key = (int(row["repeat_id"]), row["config_id"], row["cluster_id"])
            total = cluster_sums.setdefault(key, [0, 0, 0.0])
            total[0] += int(row["n"])
            total[1] += int(row["n_exact"])
            total[2] += float(row["similarity_sum"])
            per_rows.append(tuple(row[h] for h in HEADERS["per_cluster.tsv"]))
    for (r, c, label), (n, hit, credit) in sorted(cluster_sums.items()):
        per_rows.append(
            (
                r,
                c,
                "pooled",
                None,
                label,
                n,
                hit,
                credit,
                hit / n if n else None,
                credit / n if n else None,
            )
        )
    per_rows.sort(key=lambda r: (int(r[0]), CONFIGS.index(r[1]), r[2], str(r[3]), r[4]))
    write_tsv(path / "per_cluster.tsv", HEADERS["per_cluster.tsv"], per_rows)
    repeat_metrics, differences, fold_desc, repeat_desc = [], [], [], []
    for r in range(repeats):
        for c in CONFIGS:
            metric = score_predictions(
                cohort.y,
                predictions[c][r],
                similarity,
                top5=top5[c][r] if c in top5 else None,
            )
            repeat_metrics.append(
                {
                    "repeat_id": r,
                    "config_id": c,
                    "scope": "pooled",
                    "outer_fold_id": None,
                    "stratum": "all",
                    "stratum_value": "all",
                    **metric,
                }
            )
            folds = [
                row
                for row in fold_summaries
                if int(row["repeat_id"]) == r and row["config_id"] == c
            ]
            if len(folds) != 5 or sum(int(row["n"]) for row in folds) != len(cohort.y):
                raise IntegrityError("Fold metric support mismatch")
            for metric_name in (
                "exact_hit_rate",
                "weighted_hit_rate",
                "macro_recall",
                "macro_similarity",
                "top5_hit_rate",
            ):
                values = [
                    float(row[metric_name]) for row in folds if row[metric_name] != "NA"
                ]
                fold_desc.append(
                    {
                        "repeat_id": r,
                        "config_id": c,
                        "metric": metric_name,
                        **_descriptive(values),
                    }
                )
            if sum(int(row["n_exact"]) for row in folds) != metric[
                "n_exact"
            ] or not np.isclose(
                sum(float(row["similarity_sum"]) for row in folds),
                metric["similarity_sum"],
                atol=1e-8,
                rtol=0,
            ):
                raise IntegrityError("Fold metrics do not reconcile with OOF")
    for c in CONFIGS:
        for name in (
            "exact_hit_rate",
            "weighted_hit_rate",
            "macro_recall",
            "macro_similarity",
            "top5_hit_rate",
        ):
            values = [
                row[name]
                for row in repeat_metrics
                if row["config_id"] == c and row[name] is not None
            ]
            repeat_desc.append({"config_id": c, "metric": name, **_descriptive(values)})
    for scope, source in (("pooled", repeat_metrics), ("fold", fold_summaries)):
        for row in source:
            for other in source:
                if (
                    row["repeat_id"] != other["repeat_id"]
                    or row.get("outer_fold_id") != other.get("outer_fold_id")
                    or row["config_id"] == other["config_id"]
                ):
                    continue
                differences.append(
                    {
                        "scope": scope,
                        "repeat_id": int(row["repeat_id"]),
                        "outer_fold_id": row.get("outer_fold_id"),
                        "config_id": row["config_id"],
                        "reference": other["config_id"],
                        **{
                            name: float(row[name]) - float(other[name])
                            for name in ("exact_hit_rate", "weighted_hit_rate")
                        },
                    }
                )
    stability_summary = []
    counts = np.bincount(cohort.y, minlength=len(cohort.labels))

    def stability_rows():
        for c in CONFIGS:
            agreement, semantic = [], []
            for ix, eid in enumerate(cohort.example_ids):
                preds = predictions[c][:, ix]
                labels, freqs = np.unique(preds, return_counts=True)
                modal = labels[np.argmax(freqs)]
                exact_agree = float(freqs.max() / repeats) if repeats > 1 else None
                sem = (
                    float(
                        np.mean(
                            [
                                similarity.S[preds[a], preds[b]]
                                for a in range(repeats)
                                for b in range(a + 1, repeats)
                            ]
                        )
                    )
                    if repeats > 1
                    else None
                )
                agreement.append(exact_agree)
                semantic.append(sem)
                yield (
                    c,
                    eid,
                    repeats,
                    cohort.labels[modal],
                    exact_agree,
                    len(labels),
                    sem,
                )
            agreement = np.asarray(agreement, dtype=float)
            semantic = np.asarray(semantic, dtype=float)
            bins = np.array([size_bin(counts[y]) for y in cohort.y])
            for stratum, values in (
                ("all", ["all"]),
                ("true_cluster", list(cohort.labels)),
                ("cohort_size", ["0", "1-19", "20-99", "100-499", "500+"]),
            ):
                for value in values:
                    mask = (
                        np.ones(len(cohort.y), dtype=bool)
                        if stratum == "all"
                        else (
                            cohort.y == label_index[value]
                            if stratum == "true_cluster"
                            else bins == value
                        )
                    )
                    item = {
                        "config_id": c,
                        "stratum": stratum,
                        "stratum_value": value,
                        "n": int(mask.sum()),
                    }
                    for name, data in (
                        ("modal_agreement", agreement),
                        ("semantic_agreement", semantic),
                    ):
                        selected = data[mask]
                        item[name] = (
                            None
                            if repeats == 1 or not len(selected)
                            else {
                                "mean": float(selected.mean()),
                                "median": float(np.median(selected)),
                                "p10": float(np.quantile(selected, 0.1)),
                                "p90": float(np.quantile(selected, 0.9)),
                                "fraction_one": float(np.mean(selected == 1)),
                            }
                        )
                    stability_summary.append(item)

    write_tsv(path / "stability.tsv", HEADERS["stability.tsv"], stability_rows())
    summary = {
        "schema": "pclass-dark.explore-ml.summary.v1",
        "status": "complete",
        "experiment_id": experiment_id,
        "population": population,
        "split_mode": population["split_mode"],
        "homology_controlled": population["split_mode"] == "homology",
        "provenance_status": population["provenance_status"],
        "configurations": list(CONFIGS),
        "repeat_metrics": repeat_metrics,
        "fold_descriptive": fold_desc,
        "repeat_descriptive": repeat_desc,
        "paired_differences": differences,
        "stability_summary": stability_summary,
        "limitations": [
            "Fold variability is not a confidence interval.",
            "Softmax scores are not calibrated membership probabilities.",
            "Repeated training sets overlap.",
        ],
        "artifacts": list(HEADERS),
    }
    write_json(path / "summary.json", summary)
    return summary
