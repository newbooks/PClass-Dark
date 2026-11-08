"""Seeded random or homology-grouped outer/inner evaluation plans for SIAC."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .artifacts import IntegrityError
from .ml_data import readonly


def derived_seed(seed, repeat, fold, purpose):
    text = f"siac-v1:{seed}:{repeat}:{fold}:{purpose}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


@dataclass(frozen=True)
class FoldPlan:
    repeat: int
    fold: int
    train: np.ndarray
    test: np.ndarray
    inner_train: np.ndarray
    inner_validation: np.ndarray
    selection_seed: int
    refit_seed: int
    inner_seed: int
    inner_fold: int
    rejected_inner: tuple[str, ...]


@dataclass(frozen=True)
class SplitPlan:
    folds: tuple[FoldPlan, ...]
    groups: tuple[str, ...]
    outer_seeds: tuple[int, ...]


def make_split_plan(cohort, *, groups, split_mode, repeats, seed):
    """Create all five-fold plans before fitting; fail on infeasible partitions."""
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    groups = tuple(groups)
    if split_mode not in ("homology", "accession") or len(groups) != len(cohort.y):
        raise ValueError("Invalid split mode/group coverage")
    if repeats < 1 or not 0 <= seed <= 2**31 - 1:
        raise ValueError("Invalid repeats/seed")
    if len(set(groups)) < 5:
        raise IntegrityError("Five-fold CV requires at least five groups")
    if split_mode == "accession" and len(set(groups)) != len(groups):
        raise ValueError(
            "Random splitting requires one deduplicated identity per example"
        )
    g, y, X = np.asarray(groups), cohort.y, cohort.X
    splitter = StratifiedGroupKFold if split_mode == "homology" else StratifiedKFold

    def split(model, features, labels, group_ids):
        if split_mode == "homology":
            return model.split(features, labels, group_ids)
        return model.split(features, labels)

    plans, outer_seeds = [], []

    def viable(train):
        return (
            len(np.unique(y[train])) >= 2 and not np.isnan(X[train]).all(axis=0).any()
        )

    for r in range(repeats):
        outer_seed = derived_seed(seed, r, 0, "outer")
        outer_seeds.append(outer_seed)
        outer = splitter(5, shuffle=True, random_state=outer_seed)
        for f, (train, test) in enumerate(split(outer, X, y, g)):
            if not len(test) or not len(train) or not viable(train):
                raise IntegrityError(f"Infeasible outer training partition {r}/{f}")
            if set(g[train]) & set(g[test]):
                raise IntegrityError("Group leakage in outer plan")
            if len(set(g[train])) < 5:
                raise IntegrityError("Outer training has fewer than five inner groups")
            inner_seed = derived_seed(seed, r, f, "inner")
            inner = splitter(5, shuffle=True, random_state=inner_seed)
            rejected, selected = [], None
            for j, (itr, iva) in enumerate(split(inner, X[train], y[train], g[train])):
                a, b = train[itr], train[iva]
                if (
                    len(a)
                    and len(b)
                    and viable(a)
                    and np.isin(y[b], np.unique(y[a])).any()
                ):
                    selected = (a, b, j)
                    break
                rejected.append(
                    f"inner fold {j}: insufficient labels/support/finite data"
                )
            if selected is None:
                raise IntegrityError(f"No viable inner validation split {r}/{f}")
            a, b, j = selected
            plans.append(
                FoldPlan(
                    r,
                    f,
                    readonly(train),
                    readonly(test),
                    readonly(a),
                    readonly(b),
                    derived_seed(seed, r, f, "select"),
                    derived_seed(seed, r, f, "refit"),
                    inner_seed,
                    j,
                    tuple(rejected),
                )
            )
    return SplitPlan(tuple(plans), groups, tuple(outer_seeds))
