"""Validate recorded selection decisions without rerunning scientific fitting."""

from .artifacts import IntegrityError, schema_checked
from .signatures import SELECTION_VERSION, _valid_count


def require(condition, message):
    if not condition:
        raise IntegrityError("Explanation trace: " + message)


@schema_checked
def selection_evidence(result, *, detailed=False):
    """Return checked prefix/removal evidence from a parsed final fit result."""
    signature, trace = result["signature"], result["trace"]
    require(
        signature["input_identity"]["selection"] == SELECTION_VERSION,
        "unsupported selection algorithm; rebuild with current build-library",
    )
    order, selection = trace["order"], trace["selection"]
    output = {
        "status": selection["status"],
        "order": order,
        "core": selection["core"],
        "candidates": selection["candidates"],
        "selected": selection["selected"],
        "best_prefix_length": None,
        "comparisons": [],
        "accepted_removals": [],
        "termination": "unsupported",
        "prefixes": trace["prefixes"] if detailed else None,
        "iterations": selection["iterations"] if detailed else None,
    }
    if signature["status"] != "ok":
        require(not selection["selected"], "unsupported fit has selected dimensions")
        return output
    denominator = (
        2
        * signature["counts"]["training_positive_n"]
        * signature["counts"]["background_n"]
    )

    def checked(count):
        require(_valid_count(count, denominator), "invalid AUROC count")
        return count

    profiles = result["profiles"]
    ranked = sorted(
        [p["dimension"] for p in profiles if p["training_auroc"] is not None],
        key=lambda d: (-profiles[d]["training_auroc_numerator"], d),
    )
    require(order == ranked and len(order) >= 2, "profile ranking mismatch")
    prefixes = trace["prefixes"]
    require(len(prefixes) == len(order), "incomplete prefixes")
    for index, prefix in enumerate(prefixes):
        require(prefix["length"] == index + 1, "prefix length mismatch")
        checked(prefix["count"])
        previous = prefixes[index - 1]["count"]["numerator"] if index else None
        require(
            prefix["delta_numerator"]
            == (prefix["count"]["numerator"] - previous if index else None),
            "prefix delta mismatch",
        )
    first = profiles[order[0]]
    require(
        prefixes[0]["count"]["numerator"] == first["training_auroc_numerator"],
        "single dimension prefix mismatch",
    )
    peak = (
        max(range(len(prefixes)), key=lambda i: prefixes[i]["count"]["numerator"]) + 1
    )
    core = order[:2]
    candidates = [
        d
        for rank, d in enumerate(order, 1)
        if rank <= max(2, peak)
        or 10 * profiles[d]["training_auroc_numerator"] > 9 * denominator
    ]
    require(
        selection["prefilter_peak_length"] == peak
        and selection["core"] == core
        and selection["candidates"] == candidates,
        "candidate/core/prefix mismatch",
    )
    current = checked(selection["initial_count"])
    selected = list(candidates)
    removals = []
    terminal = False
    for index, decision in enumerate(selection["iterations"]):
        require(not terminal, "decision after terminal scan")
        require(
            decision["before"] == selected and decision["before_count"] == current,
            "removal transition mismatch",
        )
        eligible = [d for d in selected if d not in core]
        trials = decision["trials"]
        visited = [t["dimension"] for t in trials]
        require(visited == eligible[: len(visited)], "trial order mismatch")
        for trial in trials:
            checked(trial["count"])
            require(
                trial["gain_numerator"]
                == trial["count"]["numerator"] - current["numerator"],
                "trial gain mismatch",
            )
        after = checked(decision["after_count"])
        removed = decision["removed"]
        if removed is None:
            require(
                visited == eligible
                and after == current
                and all(t["gain_numerator"] <= 0 for t in trials)
                and index == len(selection["iterations"]) - 1,
                "invalid terminal scan",
            )
            terminal = True
        else:
            require(
                bool(trials)
                and removed == visited[-1]
                and removed not in core
                and trials[-1]["count"] == after
                and trials[-1]["gain_numerator"] > 0
                and all(t["gain_numerator"] <= 0 for t in trials[:-1]),
                "not a first strict improvement",
            )
            selected.remove(removed)
            removals.append(
                {
                    "dimension": removed,
                    "remaining_count": len(selected),
                    "before_auroc": current,
                    "after_auroc": after,
                    "gain_numerator": after["numerator"] - current["numerator"],
                }
            )
            current = after
    require(terminal or selected == core, "missing terminal scan")
    checked(selection["final_count"])
    require(
        selected == selection["selected"] == signature["selected_dimension_ids"]
        and current == selection["final_count"],
        "final selection mismatch",
    )
    metrics = signature["training_metrics"]
    require(
        metrics["auroc_numerator"] == current["numerator"]
        and metrics["auroc_denominator"] == denominator
        and metrics["auroc"] == current["value"],
        "final metrics mismatch",
    )
    comparisons = []

    def add(label, dimensions, count):
        checked(count)
        for row in comparisons:
            if row["dimension_ids"] == dimensions:
                require(row["auroc"] == count, "same subset has conflicting AUROC")
                row["labels"].append(label)
                return
        comparisons.append(
            {"labels": [label], "dimension_ids": dimensions, "auroc": count}
        )

    add("Best dimension", order[:1], prefixes[0]["count"])
    add("Top two", core, prefixes[1]["count"])
    add("Best ranked prefix", order[:peak], prefixes[peak - 1]["count"])
    add("All ranked dimensions", order, prefixes[-1]["count"])
    add("Initial candidates", candidates, selection["initial_count"])
    add("Final signature", selected, current)
    output.update(
        best_prefix_length=peak,
        comparisons=comparisons,
        accepted_removals=removals,
        termination="protected_core_only"
        if selected == core
        else "no_improving_removal",
    )
    return output
