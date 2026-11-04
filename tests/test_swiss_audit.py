"""Swiss audit interfaces, analytical parity, verified truth and safe publication."""

import builtins
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pclass_dark import external_audit as shared
from pclass_dark.artifacts import file_record
from pclass_dark.cli import app
from pclass_dark.swiss_audit import audit_swiss
from test_external_audit import table as table  # noqa: F401


@pytest.fixture
def truth(table):
    path = table.parent / "truth.tsv"
    path.write_text(
        "Query_id\tgo_mf_terms\nq1\tGO:0005515\n"
        "q2\tGO:0005515|GO:0000001\nq3\tGO:0000001\n"
        "q4\tGO:0000001\nq5\tGO:0000001\n"
    )
    return path


def test_analytical_parity_outputs_and_title(table, truth, monkeypatch):
    from matplotlib.figure import Figure

    titles = []
    original = Figure.savefig

    def save(fig, *args, **kwargs):
        titles.extend(ax.get_title() for ax in fig.axes)
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    external, expected, diagnostics = shared.audit_external(
        table, exclude_go="GO:0005515", truth_tsv=truth
    )
    before = {p: p.read_bytes() for p in external}
    paths, rows, counts = audit_swiss(table, truth_tsv=truth)
    assert rows == expected and counts == diagnostics
    assert counts["original_queries"] == 5
    assert counts["excluded_queries"] == 2
    assert counts["retained_queries"] == 3
    assert rows[0]["predictions"] == 2
    assert rows[1]["predictions"] == 3
    assert rows[0]["scorable"] == 1 and rows[0]["hit_rate_pct"] == 100
    assert rows[1]["scorable"] == 2 and rows[1]["hit_rate_pct"] == 50
    assert rows[0]["both_hit"] == 0
    assert rows[0]["method_only_hit"] == rows[1]["method_only_hit"] == 1
    assert paths[-1].name == "swiss_audit_summary_v2.tsv"
    assert all(p.read_bytes() == content for p, content in before.items())
    assert any(title.startswith("Swiss-Prot audit") for title in titles)
    for p in paths[:-1]:
        assert p.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert paths[-1].read_bytes() == external[-1].read_bytes()
    with paths[-1].open() as stream:
        saved = list(csv.DictReader(stream, delimiter="\t"))
    assert saved[0]["excluded_queries"] == "2"
    assert saved[0]["distance_over_one"] == "1"


def test_script_cli_and_help(table, truth, tmp_path):
    args = [str(table), "--truth-tsv", str(truth), "--bins", "2"]
    result = CliRunner().invoke(app, ["swiss-audit", *args])
    assert result.exit_code == 0, result.output
    assert "Excluded 2 / 5" in result.output
    expected = (tmp_path / "swiss_audit_summary_v2.tsv").read_bytes()
    script = Path(__file__).parents[1] / "scripts/swiss-audit.py"
    completed = subprocess.run(
        [sys.executable, str(script), *args], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert "swiss_audit_summary_v2.tsv" in completed.stdout
    assert (tmp_path / "swiss_audit_summary_v2.tsv").read_bytes() == expected
    for command in (
        ["swiss-audit", "--help"],
        ["swiss-audit", str(table), "--bins", "0"],
    ):
        result = CliRunner().invoke(app, command)
        assert result.exit_code == (0 if "--help" in command else 2)
    assert CliRunner().invoke(app, ["swiss-audit", str(table)]).exit_code == 1


def publication(table, tmp_path):
    """Minimal checksum-bound Swiss publication, stored away from table parent."""
    state = tmp_path / "state" / ".swiss-evaluation"
    score_id = "b" * 64
    score = state / "scores" / score_id
    score.mkdir(parents=True)
    (state / "workspace.json").write_text(json.dumps({"workspace_uuid": "fixture"}))
    status = score / "query_status.jsonl"
    status.write_text(
        "".join(
            json.dumps(dict(ordinal=i, query_id=q, truth_go=["GO:0000001"])) + "\n"
            for i, q in enumerate(shared.read_audit_table(table).query_ids)
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
                marker_version=2,
                evaluation_kind="swiss",
                workspace_uuid="fixture",
                state_relative=os.path.relpath(state, table.parent),
                score_id=score_id,
                tsv=file_record(table),
                score_manifest=file_record(manifest),
            )
        )
    )
    return status, marker


def test_swiss_publication_and_explicit_truth_precedence(table, truth, tmp_path):
    status, marker = publication(table, tmp_path)
    assert audit_swiss(table)[2]["excluded_queries"] == 0
    marker.write_text("not JSON")
    assert audit_swiss(table, truth_tsv=truth)[2]["excluded_queries"] == 2
    # Restore a valid marker, then corrupt its referenced status.
    marker.unlink()
    # Reuse existing publication files by keeping the original status checksum.
    marker_data = dict(
        marker_version=2,
        evaluation_kind="swiss",
        workspace_uuid="fixture",
        state_relative=os.path.relpath(status.parents[2], table.parent),
        score_id=status.parent.name,
        tsv=file_record(table),
        score_manifest=file_record(status.parent / "manifest.json"),
    )
    marker.write_text(json.dumps(marker_data))
    status.write_text("corrupt")
    with pytest.raises(ValueError, match="identity mismatch"):
        audit_swiss(table)


@pytest.mark.parametrize("mode", ["headers", "binding", "undefined"])
def test_empty_or_unscorable_cohort(table, truth, mode):
    if mode == "headers":
        table.write_text(table.read_text().splitlines()[0] + "\n")
        truth.write_text("Query_id\tgo_mf_terms\n")
    elif mode == "binding":
        truth.write_text(
            "Query_id\tgo_mf_terms\n"
            + "".join(f"q{i}\tGO:0005515\n" for i in range(1, 6))
        )
    else:
        lines = table.read_text().splitlines()
        for i in range(1, len(lines)):
            row = lines[i].split("\t")
            row[3] = row[6] = row[2] = row[5] = "NA"
            lines[i] = "\t".join(row)
        table.write_text("\n".join(lines) + "\n")
    paths, rows, _ = audit_swiss(table, truth_tsv=truth)
    assert rows[0]["hit_rate_pct"] is None
    assert all(p.exists() for p in paths)


@pytest.mark.parametrize("mode", ["path", "truth", "hardlink", "symlink"])
def test_source_and_destination_protection(table, truth, tmp_path, mode):
    destination = tmp_path / "swiss_audit_summary_v2.tsv"
    original = table.read_bytes()
    if mode == "path":
        table.rename(destination)
        table = destination
    elif mode == "truth":
        truth.rename(destination)
        truth = destination
    elif mode == "hardlink":
        os.link(table, destination)
    else:
        destination.symlink_to(table)
    with pytest.raises(ValueError, match="overwrite|symlink"):
        audit_swiss(table, truth_tsv=truth)
    assert table.read_bytes() == original


def test_render_failure_preserves_outputs(table, truth, monkeypatch):
    from matplotlib.figure import Figure

    paths, _, _ = audit_swiss(table, truth_tsv=truth)
    previous = {p: p.read_bytes() for p in paths}
    original = Figure.savefig
    calls = []

    def fail(fig, *args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise OSError("injected rendering failure")
        return original(fig, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="injected"):
        audit_swiss(table, truth_tsv=truth)
    assert all(p.read_bytes() == content for p, content in previous.items())
    assert not list(table.parent.glob(".external-audit-*"))


def test_presence_parser_unfiltered_and_options(table, tmp_path):
    lines = table.read_text().replace("GO:0000001", "prediction-text").splitlines()
    # Reorder columns and retain an extra uniquely named column.
    table.write_text(
        "\n".join("extra\t" + "\t".join(reversed(line.split("\t"))) for line in lines)
        + "\n"
    )
    paths, rows, _ = audit_swiss(
        table, exclude_go=None, truth_tsv=tmp_path / "missing-truth.tsv"
    )
    assert rows[0]["hits"] == 2
    assert paths[-1].name == "swiss_audit_summary.tsv"
    with pytest.raises(ValueError, match="exclude_go"):
        audit_swiss(table, exclude_go="GO:0000001")
    for kwargs in ({"bins": True}, {"bins": 101}, {"cutoff": float("nan")}):
        with pytest.raises(ValueError):
            audit_swiss(table, exclude_go=None, **kwargs)


def test_malformed_truth_and_missing_plot_dependency(table, truth, monkeypatch):
    truth.write_text(truth.read_text().replace("GO:0000001", "invalid"))
    with pytest.raises(ValueError):
        audit_swiss(table, truth_tsv=truth)
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="plotting support"):
        audit_swiss(table, exclude_go=None)


def test_import_has_no_plotting_or_inference_side_effects():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.startswith(("matplotlib", "torch", "pclass.", "profunc")):
        raise RuntimeError("unexpected import: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import pclass_dark.swiss_audit
import pclass_dark.swiss_audit_cli
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
