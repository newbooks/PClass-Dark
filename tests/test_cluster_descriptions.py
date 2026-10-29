"""Portable profunc description metadata, membership binding and display limits."""

import copy
import json
from importlib.resources import files

import pytest
from rich.cells import cell_len

from pclass_dark.artifacts import IntegrityError, identity
from pclass_dark.cluster_descriptions import (
    SCHEMA,
    clean_description,
    description_for,
    load_catalog,
    validate_catalog,
)
from pclass_dark.explanations import SignatureSummaryListing, render_signature_summaries
from pclass_dark.signature_summary_render import short_description


def catalog():
    result = {
        "schema": SCHEMA,
        "status": "complete",
        "source": {},
        "row_map_sha256": "a" * 64,
        "clusters": {
            "K": {
                "description": "ATP-dependent test function",
                "composed_go": ["GO:0000001"],
                "C_ids_sha256": "b" * 64,
            }
        },
    }
    result["id"] = identity(result)
    return result


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


def test_catalog_binding_and_corruption():
    data = validate_catalog(catalog())
    signature = {
        "cluster_id": "K",
        "input_identity": {"row_map_sha256": "a" * 64, "C_ids_sha256": "b" * 64},
    }
    assert description_for(signature, data) == ("ATP-dependent test function", None)
    signature["input_identity"]["C_ids_sha256"] = "c" * 64
    assert description_for(signature, data) == (None, "description_reference_mismatch")
    assert description_for(signature, None) == (None, "no_description_catalog")
    signature["cluster_id"] = "unknown"
    assert description_for(signature, data) == (None, "cluster_not_in_catalog")
    bad = copy.deepcopy(data)
    bad["clusters"]["K"]["description"] = "tampered"
    with pytest.raises(IntegrityError):
        validate_catalog(bad)


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


def test_packaged_catalog_matches_reference_provenance():
    data = load_catalog()
    assert len(data["clusters"]) == 3524
    reference = json.loads(
        files("pclass_dark").joinpath("resources/reference/manifest.json").read_text()
    )
    assert (
        data["source"]["reference_membership_sha256"]
        == reference["files"]["clusters.tsv"]["sha256"]
    )
    assert (
        data["source"]["reference_fasta_sha256"]
        == reference["files"]["accession.fasta"]["sha256"]
    )
    first = data["clusters"]["cluster_0001"]
    assert first["description"] and not first["description"].startswith(
        "cluster_0001 is annotated as "
    )
    assert first["C_ids_sha256"] is not None


def test_optional_missing_catalog_and_row_map_mismatch(tmp_path, monkeypatch):
    monkeypatch.setattr("pclass_dark.cluster_descriptions.files", lambda name: tmp_path)
    assert load_catalog() is None
    data = catalog()
    signature = {
        "cluster_id": "K",
        "input_identity": {"row_map_sha256": "c" * 64, "C_ids_sha256": "b" * 64},
    }
    assert description_for(signature, data)[1] == "description_reference_mismatch"
    signature["input_identity"]["row_map_sha256"] = "a" * 64
    data["clusters"]["K"]["description"] = None
    assert description_for(signature, data) == (None, "unresolved_function_description")
