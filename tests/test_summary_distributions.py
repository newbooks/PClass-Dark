"""Independent edge/count oracles for the 20-bin per-cluster metric report."""

import json
import math

import pytest

from pclass_dark.explanations import SignatureSummaryListing, render_signature_summaries
from pclass_dark.summary_distributions import METRICS, metric_distributions


def row(value):
    return {key: value for key in METRICS}


def test_all_edges_and_final_inclusive_bin():
    data = metric_distributions([row(i / 20) for i in range(21)])
    assert data["edges"] == [i / 20 for i in range(21)]
    assert data["bin_count"] == 20
    for value in data["metrics"].values():
        assert value["counts"] == [1] * 19 + [2]
        assert value["valid"] == 21 and value["unavailable"] == 0


def test_just_below_edges_and_unrounded_values():
    rows = [row(math.nextafter(i / 20, -math.inf)) for i in range(1, 21)]
    assert metric_distributions(rows)["metrics"]["auroc"]["counts"] == [1] * 20
    # Both values display as 0.050, but belong to different bins.
    data = metric_distributions([row(0.0499), row(0.0501), row(0.0)])
    assert data["metrics"]["recall"]["counts"][:2] == [2, 1]


def test_missing_metrics_are_not_zero_and_do_not_weight_windows():
    rows = [
        {**row(0.0), "precision": None, "C_windows": 1000},
        {**row(1.0), "recall": None, "C_windows": 2},
        row(None),
    ]
    data = metric_distributions(rows)
    assert data["metrics"]["auroc"]["counts"][0] == 1
    assert data["metrics"]["auroc"]["counts"][-1] == 1
    assert data["metrics"]["precision"]["counts"][0] == 0
    for key in METRICS:
        metric = data["metrics"][key]
        assert sum(metric["counts"]) == metric["valid"]
        assert metric["valid"] + metric["unavailable"] == len(rows)


@pytest.mark.parametrize("rows", [[], [row(None)]])
def test_empty_and_all_unavailable(rows):
    for metric in metric_distributions(rows)["metrics"].values():
        assert metric["counts"] == [0] * 20
        assert metric["valid"] == 0 and metric["unavailable"] == len(rows)


@pytest.mark.parametrize("bad", [-0.01, 1.01, float("nan"), float("inf"), True, "0.5"])
def test_invalid_fractions_rejected(bad):
    with pytest.raises(ValueError):
        metric_distributions([row(bad)])


def test_reports_append_twenty_bins_and_preserve_api_json():
    rows = [
        {
            "cluster_id": "K",
            "description": None,
            "status": "ok",
            "reason": None,
            "C_windows": 20,
            "B_windows": 200,
            "dimensions": 3,
            "auroc": 0.95,
            "recall": 1.0,
            "specificity": 0.8,
            "precision": None,
        }
    ]
    listing = SignatureSummaryListing({"rows": rows})
    for format in ("text", "markdown"):
        output = render_signature_summaries(listing, format=format)
        section = output.split("In-sample metric distributions", 1)[1]
        bins = [
            line for line in section.splitlines() if line.lstrip("| ").startswith("[")
        ]
        assert len(bins) == 20
        assert "[0.000,0.050)" in bins[0] and "[0.000,5.000)" in bins[0]
        assert "[0.950,1.000]" in bins[-1] and "[95.000,100.000]" in bins[-1]
        assert "Unavailable_n" in section and "Precision" in section
    data = listing.to_dict()
    assert data["distributions"]["metrics"]["precision"]["unavailable"] == 1
    assert json.loads(render_signature_summaries(listing, format="json")) == data
    with pytest.raises(TypeError):
        listing.data["distributions"]["metrics"]["precision"]["counts"][0] = 99
    data["distributions"]["metrics"]["precision"]["counts"][0] = 99
    assert listing.to_dict()["distributions"]["metrics"]["precision"]["counts"][0] == 0
