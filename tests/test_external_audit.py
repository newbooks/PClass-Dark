"""Independent small-table checks for descriptive external audit plots."""

import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pclass_dark.cli import app
from pclass_dark.external_audit import (
    audit_external,
    confidence_bins,
    coverage_curve,
    read_audit_table,
    summarize_audit,
)
from pclass_dark.external_evaluation_types import HEADERS


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "custom-name.tsv"
    rows = [
        ["q1", "GO:0000001", "0", "0.7", "GO:0000001", "0.1", "0.8"],
        ["q2", "GO:0000001", "0.5", "0.2", "NA", "NA", "NA"],
        ["q3", "GO:0000001", "1", "NA", "GO:0000001", "0.1", "0.1"],
        ["q4", "GO:0000001", "1.2", "1", "GO:0000001", "NA", "NA"],
        ["q5", "NA", "NA", "NA", "GO:0000001", "0.9", "0.7"],
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(HEADERS)
        writer.writerows(rows)
    return path


def test_analytical_counts_bins_and_tied_coverage(table):
    data = read_audit_table(table)
    rows, diagnostic = summarize_audit(data)
    assert rows[0]["predictions"] == 4
    assert rows[0]["scorable"] == 3
    assert rows[0]["hits"] == 2
    assert rows[0]["coverage_pct"] == 80
    assert rows[0]["hit_rate_pct"] == pytest.approx(200 / 3)
    assert rows[0]["hits_per_query_pct"] == 40
    assert rows[0]["both_hit"] == rows[0]["method_only_hit"] == 1
    assert rows[1]["method_only_hit"] == 1
    assert diagnostic["distance_over_one"] == 1
    assert diagnostic["distance_over_one_pct"] == 25
    labels, counts = confidence_bins(
        data.predicted[0], data.matches[0], data.evidence[0], bins=2
    )
    assert labels == ["0–0.5", "0.5–1", ">1"]
    np.testing.assert_array_equal(counts, [[1, 0, 1], [0, 1, 0], [0, 1, 0]])
    x, y = coverage_curve(data.predicted[1], data.matches[1], data.evidence[1])
    # q1/q3 are inseparable at evidence=0.1. Missing confidence q4 is unrankable.
    np.testing.assert_allclose(x, [40, 60])
    np.testing.assert_allclose(y, [50, 200 / 3])
    labels, counts = confidence_bins(
        data.predicted[1], data.matches[1], data.evidence[1], bins=2
    )
    assert labels[-1] == "NA" and counts[2, -1] == 1


@pytest.mark.parametrize("bad", ["nan", "inf", "-0.1", "2.1", "text"])
def test_invalid_distance_rejected(table, bad):
    table.write_text(table.read_text().replace("\t1.2\t", f"\t{bad}\t"))
    with pytest.raises(ValueError, match="Embedding_d"):
        read_audit_table(table)


def test_duplicate_and_inconsistent_rows_rejected(table):
    original = table.read_text()
    table.write_text(original.replace("q2\t", "q1\t"))
    with pytest.raises(ValueError, match="duplicate"):
        read_audit_table(table)
    table.write_text(original.replace("q2\tGO:0000001", "q2\tNA"))
    with pytest.raises(ValueError, match="without GO"):
        read_audit_table(table)
    table.write_text("Query_id\tQuery_id\n")
    with pytest.raises(ValueError, match="headers"):
        read_audit_table(table)


def test_empty_and_unscorable_data(tmp_path):
    path = tmp_path / "empty.tsv"
    path.write_text("\t".join(HEADERS) + "\n")
    data = read_audit_table(path)
    rows, diagnostic = summarize_audit(data)
    assert rows[0]["hit_rate_pct"] is None
    assert diagnostic["distance_max"] is None
    assert (
        coverage_curve(data.predicted[0], data.matches[0], data.evidence[0])[0].size
        == 0
    )
    _, counts = confidence_bins(np.array([True]), np.array([np.nan]), np.array([0.0]))
    assert counts[2, 0] == 1
    _, y = coverage_curve(np.array([True]), np.array([np.nan]), np.array([0.0]))
    assert np.isnan(y[0])
    paths, _, _ = audit_external(path)
    assert all(p.exists() for p in paths)


def test_pngs_summary_and_both_cli_interfaces(table, tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    coverage_labels = []
    original_save = Figure.savefig

    def save(fig, path, *args, **kwargs):
        if Path(path).name == "external_audit_coverage.png":
            coverage_labels.extend(ax.get_xlabel() for ax in fig.axes)
        return original_save(fig, path, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    original = table.read_bytes()
    paths, _, _ = audit_external(table)
    assert len(paths) == 5
    for path in paths[:4]:
        assert path.name.startswith("external_audit_")
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with paths[-1].open() as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    assert rows[0]["hits"] == "2" and rows[0]["distance_over_one"] == "1"
    truth = tmp_path / "truth.tsv"
    truth.write_text(
        "Query_id\tgo_mf_terms\n" + "".join(f"q{i}\tGO:0000001\n" for i in range(1, 6))
    )
    out = tmp_path / "cli"
    result = CliRunner().invoke(
        app,
        [
            "external-audit",
            str(table),
            "--output-dir",
            str(out),
            "--truth-tsv",
            str(truth),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Embedding distances >1: 1 / 4" in result.output
    script = Path(__file__).resolve().parents[1] / "scripts" / "external-audit.py"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            str(table),
            "--output-dir",
            str(out),
            "--truth-tsv",
            str(truth),
        ],
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    assert table.read_bytes() == original
    assert coverage_labels
    assert all(label.endswith("\nStrong to weak") for label in coverage_labels)


def test_output_cannot_replace_input_and_bad_settings(table):
    for kwargs in ({"bins": 0}, {"cutoff": np.nan}):
        with pytest.raises(ValueError):
            audit_external(table, **kwargs)
    source = table.with_name("external_audit_summary.tsv")
    source.write_bytes(table.read_bytes())
    before = source.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        audit_external(source)
    assert source.read_bytes() == before


def test_v2_filters_query_truth_and_preserves_v1(table, tmp_path):
    from pclass_dark.external_audit import filter_query_go

    original = table.read_text().replace("q3\tGO:0000001", "q3\tGO:0005515")
    table.write_text(original)
    annotations = {
        "q1": ["GO:0005515"],
        "q2": ["GO:0005515", "GO:0000001"],
        "q3": ["GO:0000001"],
        "q4": ["GO:0000002"],
        "q5": ["GO:0000003"],
    }
    data = filter_query_go(read_audit_table(table), annotations)
    assert data.query_ids == ("q3", "q4", "q5")
    x, y = coverage_curve(data.predicted[1], data.matches[1], data.evidence[1])
    np.testing.assert_allclose(x, [100 / 3, 200 / 3])
    np.testing.assert_allclose(y, [0, 50])
    truth = tmp_path / "truth.tsv"
    truth.write_text(
        "Query_id\tgo_mf_terms\n"
        + "".join(f"{q}\t{'|'.join(go)}\n" for q, go in annotations.items())
    )
    old_paths, _, _ = audit_external(table)
    previous = {p: p.read_bytes() for p in old_paths}
    paths, rows, counts = audit_external(
        table, exclude_go="GO:0005515", truth_tsv=truth
    )
    assert counts["original_queries"] == 5
    assert counts["excluded_queries"] == 2 and counts["retained_queries"] == 3
    assert rows[0]["queries"] == 3 and rows[0]["hits"] == 1
    assert rows[0]["hit_rate_pct"] == 100
    assert rows[1]["hit_rate_pct"] == 50
    assert all(p.stem.endswith("_v2") for p in paths)
    assert all(p.read_bytes() == content for p, content in previous.items())
    with paths[-1].open() as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        assert len(reader.fieldnames) == len(set(reader.fieldnames))
        assert next(reader)["excluded_queries"] == "2"


def test_annotation_metadata_verified_and_required(table, tmp_path):
    import json

    from pclass_dark.artifacts import file_record
    from pclass_dark.external_audit import read_query_annotations

    data = read_audit_table(table)
    with pytest.raises(ValueError, match="--truth-tsv"):
        read_query_annotations(table, data)
    score_id = "a" * 64
    score = table.parent / ".external-evaluation" / "scores" / score_id
    score.mkdir(parents=True)
    status = score / "query_status.jsonl"
    status.write_text(
        "".join(
            json.dumps(
                dict(
                    ordinal=i,
                    query_id=q,
                    truth_go=["GO:0005515" if i == 0 else "GO:0000001"],
                )
            )
            + "\n"
            for i, q in enumerate(data.query_ids)
        )
    )
    manifest = score / "manifest.json"
    manifest.write_text(
        json.dumps(
            dict(
                id=score_id,
                query_count=5,
                files={"query_status.jsonl": file_record(status)},
            )
        )
    )
    marker = table.with_name("." + table.name + ".evaluation.json")
    marker.write_text(
        json.dumps(
            dict(
                score_id=score_id,
                tsv=file_record(table),
                score_manifest=file_record(manifest),
            )
        )
    )
    annotations = read_query_annotations(table, data)
    assert annotations["q1"] == ("GO:0005515",)
    result = CliRunner().invoke(app, ["external-audit", str(table)])
    assert result.exit_code == 0, result.output
    assert "Excluded 1 / 5" in result.output
    assert (table.parent / "external_audit_coverage_v2.png").exists()
    status.write_text(status.read_text() + "\n")
    with pytest.raises(ValueError, match="identity mismatch"):
        read_query_annotations(table, data)


@pytest.mark.parametrize(
    "rows", ["q1\tGO:0005515\n", "q1\tGO:0005515\nq1\tGO:0000001\n"]
)
def test_incomplete_or_duplicate_truth_rejected(table, tmp_path, rows):
    from pclass_dark.external_audit import read_query_annotations

    truth = tmp_path / "truth.tsv"
    truth.write_text("Query_id\tgo_mf_terms\n" + rows)
    with pytest.raises(ValueError, match="exactly|duplicate"):
        read_query_annotations(table, read_audit_table(table), truth)


def test_exclusion_can_leave_empty_cohort(table, tmp_path):
    truth = tmp_path / "all-binding.tsv"
    truth.write_text(
        "Query_id\tgo_mf_terms\n" + "".join(f"q{i}\tGO:0005515\n" for i in range(1, 6))
    )
    paths, rows, counts = audit_external(
        table, exclude_go="GO:0005515", truth_tsv=truth
    )
    assert counts["excluded_queries"] == 5 and counts["retained_queries"] == 0
    assert rows[0]["queries"] == 0 and rows[0]["hit_rate_pct"] is None
    assert all(p.exists() for p in paths)
