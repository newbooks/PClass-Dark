"""Portable profunc description metadata, membership binding and display limits."""

import json

import pytest
from rich.cells import cell_len

from pclass_dark.cluster_descriptions import (
    clean_description,
    description_for,
    load_catalog,
)
from pclass_dark.explanations import SignatureSummaryListing, render_signature_summaries
from pclass_dark.signature_summary_render import short_description


def catalog():
    return {
        "reference": {"clusters.tsv": {"sha256": "a" * 64, "size_bytes": 10}},
        "clusters": {"K": {"description": "ATP-dependent test function"}},
    }


@pytest.mark.parametrize("article", ["a ", "an ", ""])
def test_exact_profunc_prefix_removal(article):
    assert (
        clean_description(
            "cluster_0001",
            f"cluster_0001 is annotated as {article}enzyme with activity.",
        )
        == "enzyme with activity"
    )
    with pytest.raises(ValueError, match="prefix"):
        clean_description("cluster_0001", "cluster_0002 is annotated as an enzyme.")


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, "—"),
        ("short", "short"),
        ("x" * 24, "x" * 24),
        ("x" * 25, "x" * 21 + "..."),
        ("x" * 100, "x" * 21 + "..."),
        ("界" * 13, "界" * 10 + "..."),
    ],
)
def test_description_limit_includes_ellipsis(value, expected):
    result = short_description(value)
    assert result == expected
    assert len(result) <= 24 and cell_len(result) <= 24
    assert "\n" not in short_description("abc\n" + "x" * 25)


def test_live_reference_binding():
    data = catalog()
    signature = {"cluster_id": "K", "input_identity": {}}
    assert description_for(signature, data)[1] == "missing_profunc_reference_binding"
    signature["input_identity"]["reference"] = data["reference"]
    assert description_for(signature, data) == ("ATP-dependent test function", None)
    signature["input_identity"]["reference"] = {}
    assert description_for(signature, data)[1] == "description_reference_mismatch"
    assert description_for(signature, None) == (None, "no_description_catalog")
    signature["input_identity"]["reference"] = data["reference"]
    signature["cluster_id"] = "unknown"
    assert description_for(signature, data)[1] == "cluster_not_in_catalog"
    signature["cluster_id"] = "K"
    data["clusters"]["K"]["description"] = None
    assert description_for(signature, data)[1] == "unresolved_function_description"


def test_description_column_retains_full_json_text():
    description = "ATP-dependent enzyme with additional supporting activity"
    data = {
        "rows": [
            {
                "cluster_id": "K",
                "description": description,
                "status": "ok",
                "reason": None,
                "C_windows": 20,
                "B_windows": 200,
                "dimensions": 3,
                "auroc": 0.9,
                "recall": 1.0,
                "specificity": 0.8,
                "precision": 0.3,
            }
        ],
    }
    listing = SignatureSummaryListing(data)
    text = render_signature_summaries(listing)
    assert text.splitlines()[2].split()[:2] == ["Cluster", "Description"]
    assert short_description(description) in text
    assert description not in text
    raw = json.loads(render_signature_summaries(listing, format="json"))
    assert raw["rows"][0]["description"] == description
    assert "Description" in render_signature_summaries(listing, format="markdown")


def test_live_catalog_calls_profunc_modules(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import profunc.cli.inspect
    import profunc.cluster_listing
    import profunc.go.description
    import profunc.go.inspect

    import pclass_dark.reference_data as reference_data

    runtime = SimpleNamespace(dataset=tmp_path, clusters=tmp_path, manifest={})
    monkeypatch.setattr(reference_data, "resolve_profunc", lambda: runtime)
    monkeypatch.setattr(
        reference_data,
        "profunc_reference_contract",
        lambda runtime: catalog()["reference"],
    )
    monkeypatch.setattr(
        profunc.go.inspect, "_load_go_graph", lambda path: {"terms": {"GO:1": {}}}
    )
    monkeypatch.setattr(
        profunc.cluster_listing,
        "load_cluster_records",
        lambda path: [SimpleNamespace(cluster_id="K", composed_go="GO:1")],
    )
    calls = []

    def describe(cluster, terms, graph, **kwargs):
        calls.append((cluster, terms, graph, kwargs))
        return SimpleNamespace(
            head=True, summary="K is annotated as an updated enzyme."
        )

    monkeypatch.setattr(profunc.go.description, "describe_go_function", describe)
    result = load_catalog()
    assert result["clusters"]["K"]["description"] == "updated enzyme"
    assert calls == [
        (
            "K",
            ["GO:1"],
            {"GO:1": {}},
            {"max_modifiers": profunc.cli.inspect.CLUSTER_DESCRIPTION_MAX_MODIFIERS},
        )
    ]


def test_live_runtime_catalog_smoke():
    data = load_catalog()
    assert data["clusters"]
    assert data["clusters"]["cluster_0001"]["description"]
