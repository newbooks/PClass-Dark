"""Fixed-bin, unweighted distributions of saved per-cluster reference metrics."""

from bisect import bisect_right

METRICS = ("auroc", "recall", "specificity", "precision")
BIN_COUNT = 20


def metric_distributions(rows):
    """Count non-null, unrounded fractions; raises ValueError for invalid values."""
    edges = [i / BIN_COUNT for i in range(BIN_COUNT + 1)]
    metrics = {
        key: {"counts": [0] * BIN_COUNT, "valid": 0, "unavailable": 0}
        for key in METRICS
    }
    for row in rows:
        for key in METRICS:
            try:
                value = row[key]
            except (KeyError, TypeError) as exc:
                raise ValueError(f"Missing {key} metric for distribution") from exc
            if value is None:
                metrics[key]["unavailable"] += 1
                continue
            if type(value) not in (int, float) or not 0 <= value <= 1:
                raise ValueError(f"Invalid {key} fraction for metric distribution")
            index = min(BIN_COUNT - 1, bisect_right(edges, value) - 1)
            metrics[key]["counts"][index] += 1
            metrics[key]["valid"] += 1
    return {
        "bin_count": BIN_COUNT,
        "edges": edges,
        "closure": "left-inclusive-right-exclusive-final-inclusive",
        "metrics": metrics,
    }
