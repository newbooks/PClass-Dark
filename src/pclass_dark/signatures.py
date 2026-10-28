"""ProStruct-style 640-coordinate signatures, with fold-local fitted kernels."""

from __future__ import annotations

import logging
import math
import shutil
import time

import numpy as np

from .artifacts import IntegrityError, identity
from .profiles import auroc_counts, dimension_profile, gaussian_log, moments
from .signature_metrics import accepted, choose_threshold, evaluate

LOG = logging.getLogger(__name__)
WIDTH = 640
SELECTION_VERSION = "first-improvement-logdiff-prefix-v2"


def composite_log(kernels, dimensions):
    """Reference left-to-right arithmetic Gaussian mean, in stored rank order."""
    if not dimensions:
        raise ValueError("An empty signature has no compatibility score")
    total = np.array(kernels[dimensions[0]], dtype=np.float64, copy=True)
    for dimension in dimensions[1:]:
        np.logaddexp(total, kernels[dimension], out=total)
    total -= math.log(len(dimensions))
    if not np.isfinite(total).all():
        raise ValueError("Nonfinite composite log score")
    return total


class RemovalScores:
    """Bounded subset caches and stable leave-one-out scores for training C/B.

    Returned scratch arrays remain valid only until the next call. Final metrics
    and accepted improvements are evaluated with rank-ordered reference scores.
    """

    def __init__(self, kernels):
        self.kernels = {key: kernels[key] for key in ("C", "B")}
        self.selected = None
        self.prefixes = {}
        self.totals = {}
        self.buffers = {}

    def prepare(self, dimensions):
        selected = tuple(dimensions)
        if selected == self.selected:
            return
        self.selected = selected
        self.prefixes.clear()
        self.totals.clear()
        stride = max(1, math.ceil(len(selected) / 8))
        for key, kernels in self.kernels.items():
            running = np.array(kernels[selected[0]], dtype=np.float64, copy=True)
            prefixes = {}
            for index, dimension in enumerate(selected[1:], 1):
                if index % stride == 0:
                    prefixes[index] = running.copy()  # First index dimensions.
                np.logaddexp(running, kernels[dimension], out=running)
            self.prefixes[key] = prefixes
            self.totals[key] = running
            if key not in self.buffers:
                self.buffers[key] = np.empty_like(running)

    def reference(self, key, omitted, rows=None):
        """Reuse a preceding exact prefix; never add the omitted dimension."""
        kernels = self.kernels[key]
        selected = self.selected
        index = selected.index(omitted)
        start = max((i for i in self.prefixes[key] if i <= index), default=0)
        take = slice(None) if rows is None else rows
        if start:
            total = self.prefixes[key][start][take].copy()
        else:
            total = None
        for dimension in selected[start:]:
            if dimension == omitted:
                continue
            if total is None:
                total = np.array(kernels[dimension][take], copy=True)
            else:
                np.logaddexp(total, kernels[dimension][take], out=total)
        total -= math.log(len(selected) - 1)
        return total

    def scores(self, dimensions, omitted):
        self.prepare(dimensions)
        for key, kernels in self.kernels.items():
            out = self.buffers[key]
            np.subtract(kernels[omitted], self.totals[key], out=out)
            # A removed term close to the total amplifies subtraction error.
            unsafe = out > -1e-4
            np.minimum(out, 0, out=out)
            np.expm1(out, out=out)
            np.negative(out, out=out)
            with np.errstate(divide="ignore"):
                np.log(out, out=out)
            out += self.totals[key]
            out -= math.log(len(dimensions) - 1)
            if unsafe.any():
                out[unsafe] = self.reference(key, omitted, unsafe)
            if not np.isfinite(out).all():
                raise ValueError("Nonfinite leave-one-out score")
        return self.buffers["C"], self.buffers["B"]

    def __call__(self, dimensions, omitted, current):
        positive, background = self.scores(dimensions, omitted)
        count = auroc_counts(positive, background)
        # Detect score ties/near-ties where subtraction rounding can change ranks.
        ordered = np.sort(positive)
        positions = np.searchsorted(ordered, background)
        near = False
        for offset in (-1, 0):
            neighbor = ordered[np.clip(positions + offset, 0, len(ordered) - 1)]
            tolerance = 1e-8 * (1 + np.maximum(np.abs(neighbor), np.abs(background)))
            if np.any(np.abs(neighbor - background) <= tolerance):
                near = True
                break
        if near or count["numerator"] > current["numerator"]:
            count = auroc_counts(
                self.reference("C", omitted), self.reference("B", omitted)
            )
        return count


def _require(condition, message):
    if not condition:
        raise IntegrityError(message)


def _valid_count(value, denominator):
    return (
        isinstance(value, dict)
        and type(value.get("numerator")) is int
        and 0 <= value["numerator"] <= denominator
        and value.get("denominator") == denominator
        and value.get("value") == value["numerator"] / denominator
    )


def _validate_profiles(rows, dimensions, n_c, n_b, n_v, sigma_floor):
    _require(
        isinstance(rows, list) and [r["dimension"] for r in rows] == list(dimensions),
        "Invalid profile dimension coverage",
    )
    for row in rows:
        _require(
            row["positive_n"] == n_c and row["background_n"] == n_b,
            "Invalid profile sample counts",
        )
        required = {
            "mu_C": n_c > 0,
            "sigma_C": n_c >= 2,
            "mu_B": n_b > 0,
            "sigma_B": n_b >= 2,
            "effective_sigma": n_c >= 2,
            "R": n_c >= 2 and n_b >= 2,
            "D": n_c >= 2 and n_b >= 2,
        }
        for key, defined in required.items():
            value = row[key]
            _require(
                type(value) in (int, float) and math.isfinite(value)
                if defined
                else value is None,
                f"Invalid profile statistic: {key}",
            )
            if defined and key not in ("mu_C", "mu_B"):
                _require(value >= 0, f"Negative profile spread/evidence: {key}")
        if n_c >= 2:
            _require(
                row["effective_sigma"] == max(row["sigma_C"], sigma_floor),
                "Gaussian width/floor mismatch",
            )
        for prefix, positives in (("training", n_c), ("validation", n_v)):
            count = {
                "value": row[prefix + "_auroc"],
                "numerator": row[prefix + "_auroc_numerator"],
                "denominator": row[prefix + "_auroc_denominator"],
            }
            defined = n_c >= 2 and positives > 0 and n_b > 0
            _require(
                _valid_count(count, 2 * positives * n_b)
                if defined
                else count
                == {
                    "value": None,
                    "numerator": None,
                    "denominator": 2 * positives * n_b,
                },
                "Invalid individual AUROC counts",
            )


def select_dimensions(
    order,
    individual_counts,
    prefix_counts,
    evaluate_subset,
    checkpoints=None,
    dependency="",
    *,
    cluster_id=None,
    fit_id=None,
    evaluate_removal=None,
):
    """Protect top two; accept the first strict gain in individual-rank order."""
    dependency = identity([SELECTION_VERSION, dependency])
    if len(order) < 2:
        return {
            "status": "insufficient_core",
            "core": [],
            "candidates": [],
            "selected": [],
            "iterations": [],
        }
    peak = (
        max(range(len(prefix_counts)), key=lambda i: prefix_counts[i]["numerator"]) + 1
    )
    core = order[:2]
    candidates = [
        d
        for rank, d in enumerate(order, 1)
        if rank <= max(2, peak)
        or 10 * individual_counts[d]["numerator"]
        > 9 * individual_counts[d]["denominator"]
    ]
    selected = candidates.copy()
    denominator = individual_counts[order[0]]["denominator"]
    initial_dependency = identity([dependency, candidates])
    saved = (
        checkpoints.read(
            "initial",
            initial_dependency,
            lambda p: _require(
                _valid_count(p["count"], denominator), "Invalid initial AUROC"
            ),
        )
        if checkpoints
        else None
    )
    initial = saved[0]["count"] if saved else evaluate_subset(selected)
    if checkpoints and not saved:
        checkpoints.write("initial", initial_dependency, {"count": initial})
    current = initial
    iterations = []
    while True:
        iteration = len(iterations)
        dep = identity([dependency, selected, core, current])
        name = f"decision-{iteration:04d}"

        def validate_decision(
            p, current_numerator=current["numerator"], current_count=current
        ):
            _require(
                p["before"] == selected and _valid_count(p["after_count"], denominator),
                "Invalid removal decision",
            )
            _require(
                p["removed"] is None
                or p["removed"] in selected
                and p["removed"] not in core,
                "Invalid protected-core removal",
            )
            _require(
                p["removed"] is None
                or p["after_count"]["numerator"] > current_numerator,
                "Non-improving removal",
            )
            eligible = [d for d in selected if d not in core]
            visited = [trial["dimension"] for trial in p["trials"]]
            _require(
                visited == eligible[: len(visited)]
                and p["before_count"] == current_count
                and all(_valid_count(t["count"], denominator) for t in p["trials"]),
                "Invalid first-improvement trial order/counts",
            )
            _require(
                all(
                    t["count"]["numerator"] <= current_numerator
                    for t in (p["trials"] if p["removed"] is None else p["trials"][:-1])
                )
                and (
                    visited == eligible and p["after_count"] == current_count
                    if p["removed"] is None
                    else bool(visited)
                    and visited[-1] == p["removed"]
                    and p["trials"][-1]["count"] == p["after_count"]
                ),
                "Invalid first-improvement decision",
            )

        saved = checkpoints.read(name, dep, validate_decision) if checkpoints else None
        if saved:
            decision = saved[0]
        else:
            trials = []
            best, best_count = None, current
            last_log = time.monotonic()
            for dimension in selected:
                if dimension in core:
                    continue
                trial_dep = identity([dep, dimension])
                trial_name = f"trial-{iteration:04d}-{dimension:03d}"
                saved_trial = (
                    checkpoints.read(
                        trial_name,
                        trial_dep,
                        lambda p: _require(
                            _valid_count(p["count"], denominator),
                            "Invalid removal trial",
                        ),
                    )
                    if checkpoints
                    else None
                )
                count = (
                    saved_trial[0]["count"]
                    if saved_trial
                    else evaluate_removal(selected, dimension, current)
                    if evaluate_removal is not None
                    else evaluate_subset([d for d in selected if d != dimension])
                )
                if checkpoints and not saved_trial:
                    checkpoints.write(trial_name, trial_dep, {"count": count})
                trials.append(
                    {
                        "dimension": dimension,
                        "count": count,
                        "gain_numerator": count["numerator"] - current["numerator"],
                    }
                )
                if count["numerator"] > best_count["numerator"]:
                    best, best_count = dimension, count
                if time.monotonic() - last_log >= 30:
                    LOG.info(
                        "Cluster %s %s: Removal iteration %s: %s/%s trials",
                        cluster_id,
                        fit_id,
                        iteration,
                        len(trials),
                        len(selected) - 2,
                    )
                    last_log = time.monotonic()
                if best is not None:
                    break
            decision = {
                "before": selected.copy(),
                "before_count": current,
                "trials": trials,
                "removed": best,
                "after_count": best_count,
            }
            if checkpoints:
                checkpoints.write(name, dep, decision)
        iterations.append(decision)
        if decision["removed"] is None:
            break
        selected.remove(decision["removed"])
        current = decision["after_count"]
    return {
        "status": "ok",
        "core": core,
        "candidates": candidates,
        "selected": selected,
        "prefilter_peak_length": peak,
        "initial_count": initial,
        "final_count": current,
        "iterations": iterations,
    }


def _kernel_cache(matrix, indices, profiles, store, dependency):
    saved = store.read("kernels", dependency)
    if saved and set(saved[1]) == {key for key, idx in indices.items() if len(idx)}:
        if all(saved[1][k].shape == (WIDTH, len(indices[k])) for k in saved[1]):
            return {
                k: saved[1].get(k, np.empty((WIDTH, 0), dtype=np.float64))
                for k in indices
            }
    required = 8 * WIDTH * sum(map(len, indices.values()))
    if shutil.disk_usage(store.path).free < required * 1.1:
        raise OSError(
            f"Insufficient cache storage: need {required} bytes for Gaussian kernels"
        )
    caches = {}
    for key, idx in indices.items():
        if not len(idx):
            caches[key] = np.empty((WIDTH, 0), dtype=np.float64)
            continue
        path = store.path / f"kernels.{key}.npy"
        cache = np.lib.format.open_memmap(
            path, mode="w+", dtype="<f8", shape=(WIDTH, len(idx))
        )
        last_log = time.monotonic()
        for dimension, profile in enumerate(profiles):
            for start in range(0, len(idx), 4096):
                stop = min(start + 4096, len(idx))
                cache[dimension, start:stop] = gaussian_log(
                    matrix[idx[start:stop], dimension],
                    profile["mu_C"],
                    profile["effective_sigma"],
                )
            if time.monotonic() - last_log >= 30:
                LOG.info(
                    "%s %s Gaussian cache: %s/%s dimensions",
                    store.path.parent.name,
                    store.path.name,
                    dimension + 1,
                    WIDTH,
                )
                last_log = time.monotonic()
        cache.flush()
        caches[key] = cache
    store.write(
        "kernels",
        dependency,
        {"width": WIDTH},
        external_arrays={
            k: store.path / f"kernels.{k}.npy" for k, idx in indices.items() if len(idx)
        },
    )
    return caches


def _prefixes(kernels, order, profiles, store, dependency):
    counts = []
    running = {}
    denominator = 2 * kernels["C"].shape[1] * kernels["B"].shape[1]
    for end in range(16, len(order) + 16, 16):
        stop = min(end, len(order))
        name = f"prefix-{stop:04d}"
        dep = identity([dependency, order, stop])

        def valid(p, stop=stop):
            _require(
                p["stop"] == stop
                and len(p["counts"]) == stop
                and all(_valid_count(c, denominator) for c in p["counts"]),
                "Invalid prefix counts",
            )

        saved = store.read(name, dep, valid)
        if (
            saved
            and set(saved[1]) == {"C", "B"}
            and all(saved[1][k].shape == (kernels[k].shape[1],) for k in ("C", "B"))
        ):
            counts = saved[0]["counts"]
            running = {k: np.array(v, copy=True) for k, v in saved[1].items()}
            continue
        begin = len(counts)
        for rank in range(begin, stop):
            dimension = order[rank]
            for key in ("C", "B"):
                if rank == 0:
                    running[key] = np.array(kernels[key][dimension], copy=True)
                else:
                    np.logaddexp(
                        running[key], kernels[key][dimension], out=running[key]
                    )
            count = (
                {
                    "value": profiles[dimension]["training_auroc"],
                    "numerator": profiles[dimension]["training_auroc_numerator"],
                    "denominator": profiles[dimension]["training_auroc_denominator"],
                }
                if rank == 0
                else auroc_counts(
                    running["C"] - math.log(rank + 1), running["B"] - math.log(rank + 1)
                )
            )
            counts.append(count)
        store.write(name, dep, {"counts": counts, "stop": stop}, arrays=running)
        LOG.debug("Prefix checkpoint: %s/%s dimensions", stop, len(order))
    return counts


def fit_signature(matrix, indices, *, store, context, background_moments):
    """One fold or all-C refit. No validation value participates in selection."""
    task = identity([store.task_id, SELECTION_VERSION])
    c, b, v = (indices[k] for k in ("C", "B", "V"))

    def complete_valid(p):
        from .library_runtime import validate_signature

        signature = p["signature"]
        validate_signature(
            signature,
            library_id=context["library_id"],
            cluster_id=context["cluster_id"],
            fit_id=context["fit_id"],
        )
        _require(
            isinstance(p["profiles"], list) and len(p["profiles"]) == WIDTH,
            "Invalid completed profile inventory",
        )
        _require(
            signature["selection_trace_identity"] == identity(p["trace"]),
            "Completed trace digest mismatch",
        )
        _validate_profiles(
            p["profiles"], range(WIDTH), len(c), len(b), len(v), context["sigma_floor"]
        )
        for d, row in enumerate(p["profiles"]):
            _require(
                row["dimension"] == d and row["status"] == signature["status"],
                "Invalid completed profile dimension/status",
            )
            for key in (
                "mu_C",
                "sigma_C",
                "mu_B",
                "sigma_B",
                "effective_sigma",
                "R",
                "D",
                "training_auroc",
                "validation_auroc",
            ):
                value = row[key]
                _require(
                    value is None
                    or isinstance(value, (int, float))
                    and math.isfinite(value),
                    "Invalid completed numeric profile",
                )
        _require(
            [r["window_id"] for r in p["oof"]] == context["validation_ids"],
            "Completed OOF coverage mismatch",
        )

    complete = store.read("complete", task, complete_valid)
    if complete:
        return complete[0]
    status = (
        "empty_background"
        if not len(b)
        else "insufficient_training"
        if len(c) < 2
        else "ok"
    )
    profiles = []
    for begin in range(0, WIDTH, 16):
        end = min(begin + 16, WIDTH)
        name = f"profiles-{end:04d}"
        dep = identity([task, begin, end])

        def valid(p, begin=begin, end=end):
            _validate_profiles(
                p["rows"],
                range(begin, end),
                len(c),
                len(b),
                len(v),
                context["sigma_floor"],
            )

        saved = store.read(name, dep, valid)
        if saved:
            rows = saved[0]["rows"]
        else:
            rows = []
            for d in range(begin, end):
                row = dimension_profile(
                    matrix[c, d],
                    matrix[b, d],
                    matrix[v, d],
                    epsilon=context["epsilon"],
                    sigma_floor=context["sigma_floor"],
                    background_moments=background_moments[d],
                )
                rows.append({"dimension": d, **row})
            store.write(name, dep, {"rows": rows})
        profiles.extend(rows)
    order = sorted(
        [d for d in range(WIDTH) if profiles[d]["training_auroc"] is not None],
        key=lambda d: (-profiles[d]["training_auroc_numerator"], d),
    )
    if status == "ok" and len(order) < 2:
        status = "insufficient_core"
    trace = {
        "fit_id": context["fit_id"],
        "order": order,
        "prefixes": [],
        "selection": {
            "status": status,
            "core": [],
            "candidates": [],
            "selected": [],
            "iterations": [],
        },
    }
    signature = {
        "schema": "pclass-dark.signature.v1",
        "library_id": context["library_id"],
        "cluster_id": context["cluster_id"],
        "fit_id": context["fit_id"],
        "status": status,
        "reason": None if status == "ok" else status,
        "input_identity": context["input_identity"],
        "representation_contract_id": context["representation_contract_id"],
        "counts": context["counts"],
        "selected_dimensions": [],
        "threshold": None,
        "training_metrics": None,
        "validation_metrics": None,
        "validation_status": "not_applicable_full_refit"
        if context["fit_id"] == "final"
        else "empty_validation"
        if not len(v)
        else status,
    }
    oof = []
    if status == "ok":
        params = identity(
            [[p["mu_C"], p["sigma_C"], p["effective_sigma"]] for p in profiles]
        )
        kernels = _kernel_cache(
            matrix, indices, profiles, store, identity([task, params])
        )
        prefix = _prefixes(kernels, order, profiles, store, identity([task, params]))
        individual = {
            d: {
                "numerator": profiles[d]["training_auroc_numerator"],
                "denominator": profiles[d]["training_auroc_denominator"],
            }
            for d in order
        }

        def evaluate_subset(dimensions):
            return auroc_counts(
                composite_log(kernels["C"], dimensions),
                composite_log(kernels["B"], dimensions),
            )

        selection = select_dimensions(
            order,
            individual,
            prefix,
            evaluate_subset,
            store,
            identity([task, params, order]),
            cluster_id=context["cluster_id"],
            fit_id=context["fit_id"],
            evaluate_removal=RemovalScores(kernels),
        )
        selected = selection["selected"]
        trace.update(
            prefixes=[
                {
                    "length": i + 1,
                    "count": x,
                    "delta_numerator": None
                    if not i
                    else x["numerator"] - prefix[i - 1]["numerator"],
                }
                for i, x in enumerate(prefix)
            ],
            selection=selection,
        )
        scores = {k: composite_log(kernels[k], selected) for k in kernels}
        threshold = choose_threshold(scores["C"], scores["B"])
        signature["threshold"] = threshold
        signature["training_metrics"] = evaluate(scores["C"], scores["B"], threshold)
        if context["fit_id"] != "final":
            signature["validation_metrics"] = evaluate(
                scores["V"], scores["B"], threshold
            )
            for wid, score, decision in zip(
                context["validation_ids"],
                scores["V"],
                accepted(scores["V"], threshold),
                strict=True,
            ):
                oof.append(
                    {
                        "window_id": wid,
                        "log_score": float(score),
                        "score": math.exp(float(score)),
                        "accepted": bool(decision),
                        "status": "ok",
                    }
                )
        for d in selected:
            p = profiles[d]
            signature["selected_dimensions"].append(
                {
                    "dimension": d,
                    "individual_rank": order.index(d) + 1,
                    "mu": p["mu_C"],
                    "sigma": p["sigma_C"],
                    "effective_sigma": p["effective_sigma"],
                    "R": p["R"],
                    "D": p["D"],
                    "individual_training_auroc": {
                        "value": p["training_auroc"],
                        "numerator": p["training_auroc_numerator"],
                        "denominator": p["training_auroc_denominator"],
                    },
                    "individual_validation_auroc": {
                        "value": p["validation_auroc"],
                        "numerator": p["validation_auroc_numerator"],
                        "denominator": p["validation_auroc_denominator"],
                    },
                    "core": d in selection["core"],
                    "weight": 1 / len(selected),
                }
            )
    else:
        oof = [
            {
                "window_id": wid,
                "log_score": None,
                "score": None,
                "accepted": None,
                "status": status,
            }
            for wid in context["validation_ids"]
        ]
    selection = trace["selection"]
    for p in profiles:
        d = p["dimension"]
        p.update(
            rank=order.index(d) + 1 if d in order else None,
            core=d in selection["core"],
            candidate=d in selection["candidates"],
            selected=d in selection["selected"],
            status=status,
        )
    signature.update(
        core_dimensions=selection["core"],
        candidate_dimensions=selection["candidates"],
        selected_dimension_ids=selection["selected"],
        selection_trace_identity=identity(trace),
    )
    result = {"signature": signature, "profiles": profiles, "trace": trace, "oof": oof}
    store.write("complete", task, result)
    return result


def background_profiles(matrix, rows, store, dependency):
    def valid(p):
        _require(
            isinstance(p["moments"], list) and len(p["moments"]) == WIDTH,
            "Invalid background profile cache",
        )
        for pair in p["moments"]:
            _require(
                isinstance(pair, list) and len(pair) == 2,
                "Invalid background moment pair",
            )
            for index, value in enumerate(pair):
                defined = len(rows) > index
                _require(
                    type(value) in (int, float) and math.isfinite(value)
                    if defined
                    else value is None,
                    "Invalid background moment value",
                )
                if defined and index == 1:
                    _require(value >= 0, "Negative background spread")

    cached = store.read("background", dependency, valid)
    if cached:
        return cached[0]["moments"]
    result = [moments(matrix[rows, d]) for d in range(WIDTH)]
    store.write("background", dependency, {"moments": result})
    return result
