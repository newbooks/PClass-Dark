"""Small independent recorded-decision examples, without matrix-based discovery."""

import copy

import pytest

from pclass_dark.artifacts import IntegrityError
from pclass_dark.explanation_trace import selection_evidence
from pclass_dark.signatures import SELECTION_VERSION


def recorded_fit():
    def count(n):
        return {"value": n / 100, "numerator": n, "denominator": 100}

    profiles = [
        {
            "dimension": d,
            "training_auroc": (95 - d) / 100,
            "training_auroc_numerator": 95 - d,
        }
        for d in range(4)
    ]
    selection = {
        "status": "ok",
        "core": [0, 1],
        "candidates": [0, 1, 2, 3],
        "selected": [0, 1, 3],
        "prefilter_peak_length": 4,
        "initial_count": count(98),
        "final_count": count(99),
        "iterations": [
            {
                "before": [0, 1, 2, 3],
                "before_count": count(98),
                "trials": [{"dimension": 2, "count": count(99), "gain_numerator": 1}],
                "removed": 2,
                "after_count": count(99),
            },
            {
                "before": [0, 1, 3],
                "before_count": count(99),
                "trials": [{"dimension": 3, "count": count(96), "gain_numerator": -3}],
                "removed": None,
                "after_count": count(99),
            },
        ],
    }
    return {
        "signature": {
            "status": "ok",
            "input_identity": {"selection": SELECTION_VERSION},
            "counts": {"training_positive_n": 5, "background_n": 10},
            "selected_dimension_ids": [0, 1, 3],
            "training_metrics": {
                "auroc": 0.99,
                "auroc_numerator": 99,
                "auroc_denominator": 100,
            },
        },
        "profiles": profiles,
        "trace": {
            "order": [0, 1, 2, 3],
            "prefixes": [
                {
                    "length": i + 1,
                    "count": count(95 + i),
                    "delta_numerator": 1 if i else None,
                }
                for i in range(4)
            ],
            "selection": selection,
        },
    }


def test_recorded_first_improvement_and_candidate_union():
    result = recorded_fit()
    explanation = selection_evidence(result, detailed=True)
    assert explanation["accepted_removals"][0]["dimension"] == 2
    assert explanation["accepted_removals"][0]["gain_numerator"] == 1
    assert explanation["termination"] == "no_improving_removal"
    # Best prefix is now two, but >.90 individual dimensions extend candidates.
    result["trace"]["prefixes"][1]["count"]["value"] = 0.99
    result["trace"]["prefixes"][1]["count"]["numerator"] = 99
    result["trace"]["prefixes"][1]["delta_numerator"] = 4
    result["trace"]["prefixes"][2]["delta_numerator"] = -2
    result["trace"]["selection"]["prefilter_peak_length"] = 2
    explanation = selection_evidence(result)
    assert explanation["best_prefix_length"] == 2
    assert explanation["candidates"] == [0, 1, 2, 3]
    assert explanation["comparisons"][-1]["auroc"]["value"] == 0.99


@pytest.mark.parametrize(
    "change",
    [
        lambda r: r["trace"]["selection"]["iterations"][0]["trials"][0].update(
            dimension=0
        ),
        lambda r: r["trace"]["selection"]["iterations"][0]["trials"][0].update(
            gain_numerator=2
        ),
        lambda r: r["trace"]["selection"]["iterations"].pop(),
        lambda r: r["trace"]["prefixes"][0].update(length=2),
        lambda r: r["trace"]["selection"].update(prefilter_peak_length=1),
        lambda r: r["trace"]["selection"]["iterations"].append(
            copy.deepcopy(r["trace"]["selection"]["iterations"][-1])
        ),
        lambda r: r["trace"]["selection"]["final_count"].update(value=0.5),
        lambda r: r["trace"].update(order=[1, 0, 2, 3]),
        lambda r: r["signature"]["input_identity"].update(selection="unknown"),
    ],
)
def test_invalid_recorded_reasoning_rejected(change):
    result = recorded_fit()
    change(result)
    with pytest.raises(IntegrityError):
        selection_evidence(result)


def test_malformed_trace_is_integrity_error():
    result = recorded_fit()
    result["trace"]["selection"] = None
    with pytest.raises(IntegrityError):
        selection_evidence(result)


def test_core_only_can_terminate_without_extra_scan():
    result = recorded_fit()
    result["profiles"] = result["profiles"][:2]
    result["trace"]["order"] = [0, 1]
    result["trace"]["prefixes"] = result["trace"]["prefixes"][:2]
    count = result["trace"]["prefixes"][1]["count"]
    result["trace"]["selection"].update(
        candidates=[0, 1],
        selected=[0, 1],
        prefilter_peak_length=2,
        initial_count=count,
        final_count=count,
        iterations=[],
    )
    result["signature"]["selected_dimension_ids"] = [0, 1]
    result["signature"]["training_metrics"].update(auroc=0.96, auroc_numerator=96)
    evidence = selection_evidence(result)
    assert evidence["termination"] == "protected_core_only"
    assert evidence["accepted_removals"] == []
