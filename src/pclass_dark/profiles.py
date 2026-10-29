"""Float64 fold moments and exact empirical AUROC for PLM coordinates."""

from __future__ import annotations

import math

import numpy as np


def _integer_sum(values, upper_bound):
    if len(values) * upper_bound <= np.iinfo(np.int64).max:
        return int(np.sum(values, dtype=np.int64))
    return sum(map(int, values))


def auroc_counts(positive, negative):
    """Higher score wins; retain twice-wins-plus-ties as an exact integer."""
    p, b = (
        np.asarray(positive, dtype=np.float64),
        np.asarray(negative, dtype=np.float64),
    )
    if (
        p.ndim != 1
        or b.ndim != 1
        or not np.isfinite(p).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("AUROC needs finite one-dimensional scores")
    denominator = 2 * len(p) * len(b)
    if not denominator:
        return {"value": None, "numerator": None, "denominator": denominator}
    if len(p) <= len(b):
        ordered = np.sort(p)
        lower = np.searchsorted(ordered, b, side="left")
        upper = np.searchsorted(ordered, b, side="right")
        wins = _integer_sum(len(p) - upper, len(p))
    else:
        ordered = np.sort(b)
        lower = np.searchsorted(ordered, p, side="left")
        upper = np.searchsorted(ordered, p, side="right")
        wins = _integer_sum(lower, len(b))
    ties = _integer_sum(upper - lower, max(len(p), len(b)))
    numerator = 2 * wins + ties
    return {
        "value": numerator / denominator,
        "numerator": numerator,
        "denominator": denominator,
    }


def moments(values):
    values = np.ascontiguousarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Moments need finite one-dimensional values")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        mean = float(np.mean(values)) if len(values) else None
        sigma = float(np.std(values, ddof=1)) if len(values) >= 2 else None
    return mean, sigma


def dimension_profile(
    c, b, v, *, epsilon=1e-8, sigma_floor=1e-8, background_moments=None
):
    for name, value in (("epsilon", epsilon), ("sigma_floor", sigma_floor)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    c, b, v = (np.ascontiguousarray(x, dtype=np.float64) for x in (c, b, v))
    if any(x.ndim != 1 or not np.isfinite(x).all() for x in (c, b, v)):
        raise ValueError("Profile populations must be finite vectors")
    mu, sigma = moments(c)
    bm, bs = background_moments if background_moments is not None else moments(b)
    valid = sigma is not None and bs is not None
    r = sigma / (bs + epsilon) if valid else None
    d = (
        abs(mu - bm) / (math.hypot(sigma, bs) / math.sqrt(2) + epsilon)
        if valid
        else None
    )
    if any(
        value is not None and not math.isfinite(value)
        for value in (mu, sigma, bm, bs, r, d)
    ):
        raise ValueError("Nonfinite dimension-profile evidence")
    train = (
        auroc_counts(-np.abs(c - mu), -np.abs(b - mu))
        if sigma is not None and len(b)
        else {"value": None, "numerator": None, "denominator": 2 * len(c) * len(b)}
    )
    validation = (
        auroc_counts(-np.abs(v - mu), -np.abs(b - mu))
        if sigma is not None and len(b)
        else {"value": None, "numerator": None, "denominator": 2 * len(v) * len(b)}
    )
    return {
        "positive_n": len(c),
        "background_n": len(b),
        "mu_C": mu,
        "sigma_C": sigma,
        "mu_B": bm,
        "sigma_B": bs,
        "effective_sigma": max(sigma, sigma_floor) if sigma is not None else None,
        "R": r,
        "D": d,
        "training_auroc": train["value"],
        "training_auroc_numerator": train["numerator"],
        "training_auroc_denominator": train["denominator"],
        "validation_auroc": validation["value"],
        "validation_auroc_numerator": validation["numerator"],
        "validation_auroc_denominator": validation["denominator"],
    }


def gaussian_log(values, mu, width):
    if not math.isfinite(mu) or not math.isfinite(width) or width <= 0:
        raise ValueError("Gaussian parameters must be finite with positive width")
    with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
        result = -0.5 * np.square((np.asarray(values, dtype=np.float64) - mu) / width)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite Gaussian log kernel")
    return result
