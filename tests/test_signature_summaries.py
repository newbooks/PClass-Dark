"""Summary-only loading, analytical metric checks and aligned table rendering."""

import copy
import json
import math
import shutil
from pathlib import Path

import pytest
from rich.cells import cell_len
from typer.testing import CliRunner

from pclass_dark.artifacts import (
    IntegrityError,
    identity,
    read_json,
    sha256,
    write_json,
)
from pclass_dark.cli import app
from pclass_dark.explanations import (
    SignatureSummaryListing,
    list_signature_summaries,
    render_signature_summaries,
)
from pclass_dark.library import build_library
from pclass_dark.signature_summaries import validate_signature_summary
from pclass_dark.signature_summary_render import HEADERS, metric, visible


@pytest.fixture
def summary_library(library_data, tmp_path):
    data, *_ = library_data(count=4, min_clusters=1)
    root = tmp_path / "library"
    result = build_library(data, root)
    return root, result


def analytical_signature():
    # Four positives and six negatives, with TP=3, FP=1, TN=5, FN=1.
    # A possible ordering gives 21 wins among 24 C/B pairs: AUROC=42/48.
    inputs = {
        "cluster_id": "K",
        "C_n": 4,
        "B_n": 6,
        "representation_contract_id": "c" * 64,
    }
    fit = identity(inputs)
    signature = {
        "schema": "pclass-dark.signature.v2",
        "library_id": "a" * 64,
        "cluster_id": "K",
        "fit_id": "final",
        "fit_identity": fit,
        "scientific_payload_id": "b" * 64,
        "input_identity": inputs,
        "representation_contract_id": "c" * 64,
        "evaluation_mode": "all-C-in-sample",
        "status": "ok",
        "reason": None,
        "counts": {"training_positive_n": 4, "background_n": 6},
        "selected_dimension_ids": [0, 1],
        "core_dimensions": [0, 1],
        "candidate_dimensions": [0, 1],
        "selected_dimensions": [
            {
                "dimension": d,
                "mu": 0.0,
                "sigma": 1.0,
                "effective_sigma": 1.0,
                "weight": 0.5,
                "individual_rank": d + 1,
                "core": True,
            }
            for d in (0, 1)
        ],
        "threshold": {
            "kind": "log_score_ge",
            "log_value": -0.13,
            "score_value": math.exp(-0.13),
        },
        "training_metrics": {
            "status": "ok",
            "positive_n": 4,
            "negative_n": 6,
            "tp": 3,
            "fp": 1,
            "tn": 5,
            "fn": 1,
            "auroc_numerator": 42,
            "auroc_denominator": 48,
            "auroc": 0.875,
            "recall": 0.75,
            "specificity": 5 / 6,
            "precision": 0.75,
            "unavailable_reasons": {},
        },
    }
    manifest = {
        "id": "a" * 64,
        "identity_inputs": {
            "final_fits": [["K", fit]],
            "representation_contract_id": "c" * 64,
        },
    }
    return signature, manifest


def test_summary_metrics_against_analytical_counts():
    signature, manifest = analytical_signature()
    row = validate_signature_summary(signature, manifest=manifest, cluster_id="K")
    assert row["recall"] == row["precision"] == 3 / 4
    assert row["specificity"] == 5 / 6
    assert row["auroc"] == 21 / 24
    assert row["dimensions"] == 2 and "fp" not in row
    signature["threshold"] = {
        "kind": "above_maximum",
        "log_value": None,
        "score_value": None,
    }
    signature["training_metrics"].update(
        tp=0,
        fp=0,
        tn=6,
        fn=4,
        recall=0.0,
        specificity=1.0,
        precision=None,
        auroc=0.5,
        auroc_numerator=24,
        unavailable_reasons={"precision": "undefined_denominator_or_population"},
    )
    row = validate_signature_summary(signature, manifest=manifest, cluster_id="K")
    assert row["status"] == "ok" and row["precision"] is None
    assert (
        row["unavailable_reasons"]["precision"] == "undefined_denominator_or_population"
    )


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s["counts"].update(training_positive_n=True),
        lambda s: s["training_metrics"].update(fp=-1),
        lambda s: s["training_metrics"].update(precision=0.75001),
        lambda s: s["training_metrics"].update(recall=None),
        lambda s: s["training_metrics"].update(specificity=float("nan")),
        lambda s: s["training_metrics"].update(auroc=1.1),
        lambda s: s["training_metrics"].update(auroc_numerator=True),
        lambda s: s["training_metrics"].update(auroc_denominator=0),
        lambda s: s["training_metrics"].update(positive_n=5),
        lambda s: s["training_metrics"]["unavailable_reasons"].update(
            precision="unexpected"
        ),
        lambda s: s.update(reason="unsupported"),
        lambda s: s.update(training_metrics=None),
        lambda s: s.update(scientific_payload_id="bad"),
    ],
)
def test_inconsistent_summary_rejected(change):
    signature, manifest = analytical_signature()
    change(signature)
    with pytest.raises(IntegrityError):
        validate_signature_summary(signature, manifest=manifest, cluster_id="K")


def make_listing(rows):
    return SignatureSummaryListing(
        {
            "schema": "pclass-dark.signature-summary-list.v1",
            "source": {},
            "counts": {"clusters": len(rows)},
            "rows": rows,
            "limitations": [],
        }
    )


def test_alignment_precision_unicode_controls_and_raw_json():
    signature, manifest = analytical_signature()
    row = validate_signature_summary(signature, manifest=manifest, cluster_id="K")
    ids = [
        "A",
        "cluster_10",
        "cluster_2",
        "e\u0301",
        "界-long-identifier",
        "z\x1b[31m\n\t|",
    ]
    rows = [{**copy.deepcopy(row), "cluster_id": key} for key in sorted(ids)]
    rows[0].update(
        precision=None, status="insufficient_training", reason="long status 界"
    )
    listing = make_listing(rows)
    text = render_signature_summaries(listing)
    lines = text.splitlines()
    header, table_rows = lines[2], lines[3 : 3 + len(rows)]
    assert "FP" not in header and "\x1b" not in text
    assert "\\x1b" in text and "\\n" in text and "\\t" in text
    assert "0.875" in text and "75.000" in text and "83.333" in text
    assert "—" in table_rows[0] and table_rows[0].endswith("—")
    assert all(
        visible(key) in line for key, line in zip(sorted(ids), table_rows, strict=True)
    )
    # Numeric columns have the same display-cell right boundaries on every row.
    starts = [cell_len(header[: header.index(label)]) for label in HEADERS[3:]]
    expected_right = [
        start + cell_len(label)
        for start, label in zip(starts, HEADERS[3:], strict=True)
    ]
    for line, row in zip(table_rows, rows, strict=True):
        cells = [str(row[k]) for k in ("C_windows", "B_windows", "dimensions")]
        cells += [
            metric(row["auroc"]),
            *(
                metric(row[k], percentage=True)
                for k in ("recall", "specificity", "precision")
            ),
        ]
        cursor = 0
        # Walk from right to left: numeric strings may repeat in other columns.
        for cell, right in zip(reversed(cells), reversed(expected_right), strict=True):
            index = line.rfind(cell, 0, len(line) - cursor if cursor else len(line))
            assert cell_len(line[: index + len(cell)]) == right
            cursor = len(line) - index
    raw = json.loads(render_signature_summaries(listing, format="json"))
    assert raw["rows"][1]["specificity"] == 5 / 6
    assert raw["rows"][-1]["cluster_id"] == sorted(ids)[-1]
    markdown = render_signature_summaries(listing, format="markdown")
    assert "---:" in markdown and "\\|" in markdown
    assert metric(1.0) == "1.000" and metric(1.0, percentage=True) == "100.000"
    assert metric(0.000001) == metric(-0.0) == "0.000"
    assert render_signature_summaries(listing) == text
    with pytest.raises(TypeError):
        listing.data["rows"][0]["cluster_id"] = "changed"
    detached = listing.to_dict()
    detached["rows"].clear()
    assert len(listing.to_dict()["rows"]) == len(rows)


def test_loader_ignores_unused_artifacts_and_cli_matches_api(
    summary_library, monkeypatch
):
    import builtins
    import io
    import os

    root, result = summary_library
    shutil.rmtree(root.parent / "data")
    shutil.rmtree(result.relative_dir)
    for file in result.library_dir.glob("*.tsv*"):
        file.unlink()
    for file in result.library_dir.glob("clusters/*/selection_trace.json.gz"):
        file.unlink()
    original, original_io = builtins.open, io.open
    accessed = []

    def record(file):
        if isinstance(file, (str, bytes, os.PathLike)):
            p = Path(os.fsdecode(file))
            if root in p.parents:
                assert p.name in {"selection.json", "manifest.json", "final.json"}, p
                accessed.append(p)

    def opened(file, *a, **k):
        record(file)
        return original(file, *a, **k)

    def io_opened(file, *a, **k):
        record(file)
        return original_io(file, *a, **k)

    monkeypatch.setattr(builtins, "open", opened)
    monkeypatch.setattr(io, "open", io_opened)
    listing = list_signature_summaries(root)
    assert listing.to_dict()["counts"] == {
        "clusters": 2,
        "available": 2,
        "unsupported": 0,
    }
    assert [r["cluster_id"] for r in listing.to_dict()["rows"]] == ["K0", "K1"]
    assert (
        sum(p.name == "final.json" for p in accessed) == 4
    )  # Hash + parse, two signatures.
    for format in ("text", "markdown", "json"):
        response = CliRunner().invoke(
            app,
            [
                "explain-signature",
                "--list",
                "--library-dir",
                str(root),
                "--format",
                format,
            ],
        )
        assert response.exit_code == 0, response.output
        assert response.stdout == render_signature_summaries(listing, format=format)


def test_cli_invalid_modes_do_not_read_data(monkeypatch):
    monkeypatch.setattr(
        "pclass_dark.explanations.list_signature_summaries",
        lambda *a, **k: pytest.fail("read listing"),
    )
    monkeypatch.setattr(
        "pclass_dark.explanations.explain_signature",
        lambda *a, **k: pytest.fail("read explanation"),
    )
    cases = [
        ([], 2),
        (["--cluster", "K", "--list"], 2),
        (["--list", "--detailed"], 2),
        (["--list", "--format", "invalid"], 2),
        (["--cluster", "K", "--cluster", "K"], 2),
        (["-h"], 0),
        (["--help"], 0),
    ]
    for options, status in cases:
        response = CliRunner().invoke(app, ["explain-signature", *options])
        assert response.exit_code == status, response.output


def test_unsupported_and_empty_libraries(library_data, tmp_path):
    data, *_ = library_data(count=1, min_clusters=1)
    root = tmp_path / "library"
    result = build_library(data, root)
    listing = list_signature_summaries(root)
    row = listing.to_dict()["rows"][0]
    assert row["dimensions"] == 0 and row["precision"] is None
    assert listing.to_dict()["counts"]["unsupported"] == 1
    assert "—" in render_signature_summaries(listing)
    manifest = read_json(result.library_dir / "manifest.json")
    manifest["identity_inputs"]["cluster_ids"] = []
    manifest["identity_inputs"]["final_fits"] = []
    manifest["id"] = identity(manifest["identity_inputs"])
    manifest["counts"] = {k: 0 for k in manifest["counts"]}
    write_json(result.library_dir / "manifest.json", manifest)
    selected = read_json(root / "selection.json")
    selected.update(
        library_id=manifest["id"],
        manifest_sha256=sha256(result.library_dir / "manifest.json"),
    )
    write_json(root / "selection.json", selected)
    listing = list_signature_summaries(root)
    assert listing.to_dict()["rows"] == []
    assert "No clusters in selected library." in render_signature_summaries(listing)


def test_corrupt_summary_fails_without_partial_output(summary_library):
    root, result = summary_library
    (result.library_dir / "clusters/K1/final.json").write_text("{}")
    with pytest.raises(IntegrityError):
        list_signature_summaries(root)
    response = CliRunner().invoke(
        app, ["explain-signature", "--list", "--library-dir", str(root)]
    )
    assert response.exit_code == 1 and response.stdout == "" and response.stderr


def test_missing_paths_and_bad_formats(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        list_signature_summaries(missing)
    assert not missing.exists()
    with pytest.raises(ValueError):
        list_signature_summaries(True)
    with pytest.raises(ValueError):
        render_signature_summaries(make_listing([]), format="csv")


def test_summary_read_guard(summary_library):
    from concurrent.futures import ThreadPoolExecutor

    from pclass_dark.artifacts import transaction_guard

    root, _ = summary_library
    with transaction_guard(root), ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="Setup busy"):
            executor.submit(list_signature_summaries, root).result(timeout=10)


@pytest.mark.parametrize(
    "status,c,b",
    [
        ("empty_background", 2, 0),
        ("insufficient_training", 1, 6),
        ("insufficient_core", 4, 6),
    ],
)
def test_unsupported_statuses_preserve_reasons(status, c, b):
    signature, manifest = analytical_signature()
    signature["counts"].update(training_positive_n=c, background_n=b)
    signature["input_identity"].update(C_n=c, B_n=b)
    signature["fit_identity"] = identity(signature["input_identity"])
    manifest["identity_inputs"]["final_fits"] = [["K", signature["fit_identity"]]]
    signature.update(
        status=status,
        reason=status,
        training_metrics=None,
        threshold=None,
        selected_dimensions=[],
        selected_dimension_ids=[],
    )
    row = validate_signature_summary(signature, manifest=manifest, cluster_id="K")
    assert row["status"] == row["reason"] == status
    assert row["dimensions"] == 0
    assert set(row["unavailable_reasons"].values()) == {status}


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["counts"].update(available_final=True),
        lambda m: m["counts"].update(dimension_final_rows=0),
        lambda m: m["identity_inputs"]["final_fits"].append(
            m["identity_inputs"]["final_fits"][0]
        ),
    ],
)
def test_bad_manifest_summary_counts_and_inventory(summary_library, change):
    root, result = summary_library
    manifest = read_json(result.library_dir / "manifest.json")
    change(manifest)
    manifest["id"] = identity(manifest["identity_inputs"])
    write_json(result.library_dir / "manifest.json", manifest)
    selected = read_json(root / "selection.json")
    selected.update(
        library_id=manifest["id"],
        manifest_sha256=sha256(result.library_dir / "manifest.json"),
    )
    write_json(root / "selection.json", selected)
    with pytest.raises(IntegrityError):
        list_signature_summaries(root)


def test_old_selection_and_missing_signature(summary_library):
    root, result = summary_library
    selected = read_json(root / "selection.json")
    selected["schema"] = "pclass-dark.library-selection.v1"
    write_json(root / "selection.json", selected)
    assert list_signature_summaries(root).to_dict()["counts"]["clusters"] == 2
    (result.library_dir / "clusters/K0/final.json").unlink()
    with pytest.raises((IntegrityError, OSError)):
        list_signature_summaries(root)


def test_unknown_and_legacy_production_schema(summary_library):
    root, result = summary_library
    manifest = read_json(result.library_dir / "manifest.json")
    manifest["schema"] = "pclass-dark.library.v1"
    write_json(result.library_dir / "manifest.json", manifest)
    selected = read_json(root / "selection.json")
    selected["manifest_sha256"] = sha256(result.library_dir / "manifest.json")
    write_json(root / "selection.json", selected)
    with pytest.raises(IntegrityError, match="build-library to migrate"):
        list_signature_summaries(root)
    selected["schema"] = "unknown"
    write_json(root / "selection.json", selected)
    with pytest.raises(IntegrityError, match="unknown selection"):
        list_signature_summaries(root)


def test_report_types_cannot_be_rendered_by_wrong_renderer():
    from pclass_dark.explanations import (
        SignatureExplanation,
        render_signature_explanation,
    )

    with pytest.raises(ValueError, match="SignatureExplanation"):
        render_signature_explanation(make_listing([]))
    with pytest.raises(ValueError, match="SignatureSummaryListing"):
        render_signature_summaries(SignatureExplanation({}))


def test_installed_bundled_summary_without_unused_files(summary_library, tmp_path):
    import subprocess
    import sys

    root, result = summary_library
    package = tmp_path / "package"
    bundled = package / "resources/library"
    production = bundled / result.library_id
    production.mkdir(parents=True)
    shutil.copy2(root / "selection.json", bundled / "selection.json")
    shutil.copy2(result.library_dir / "manifest.json", production / "manifest.json")
    for source in result.library_dir.glob("clusters/*/final.json"):
        destination = production / source.relative_to(result.library_dir)
        destination.parent.mkdir(parents=True)
        shutil.copy2(source, destination)
    script = """
import builtins, importlib.abc, json, os, pathlib, socket, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        blocked = {'torch','transformers','huggingface_hub','profunc'}
        if fullname.split('.')[0] in blocked:
            raise AssertionError('heavy import: '+fullname)
sys.meta_path.insert(0, Block())
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network'))
import pclass_dark.library as library
import pclass_dark.paths as paths
from pclass_dark.explanations import (
    list_signature_summaries, render_signature_summaries,
)
package = pathlib.Path(sys.argv[1])
library.__file__ = str(package / 'library.py')
paths.PACKAGE_DIR = package
os.environ['XDG_CACHE_HOME'] = str(package.parent / 'empty-cache')
original_open, original_os_open = builtins.open, os.open
def checked_open(file, mode='r', *args, **kwargs):
    if any(c in mode for c in 'wax+'):
        raise AssertionError('write: '+str(file))
    return original_open(file, mode, *args, **kwargs)
def checked_os_open(file, flags, *args, **kwargs):
    if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
        raise AssertionError('write: '+str(file))
    return original_os_open(file, flags, *args, **kwargs)
builtins.open, os.open = checked_open, checked_os_open
listing = list_signature_summaries()
assert listing.to_dict()['counts']['clusters'] == 2
assert not (package / 'resources/library/.locks').exists()
print(json.loads(render_signature_summaries(listing, format='json'))['schema'])
"""
    response = subprocess.run(
        [sys.executable, "-B", "-c", script, str(package)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert response.returncode == 0, response.stderr
    assert response.stdout.strip() == "pclass-dark.signature-summary-list.v1"


def test_validator_accepts_immutable_signature_inputs():
    from pclass_dark.library_runtime import freeze

    signature, manifest = analytical_signature()
    expected = validate_signature_summary(signature, manifest=manifest, cluster_id="K")
    assert (
        validate_signature_summary(
            freeze(signature), manifest=freeze(manifest), cluster_id="K"
        )
        == expected
    )


def test_headers_and_ordinary_rows_are_whitespace_separated():
    signature, manifest = analytical_signature()
    row = validate_signature_summary(signature, manifest=manifest, cluster_id="K")
    text = render_signature_summaries(make_listing([row]))
    lines = text.splitlines()
    assert lines[2].split() == [
        "Cluster",
        "Description",
        "C_windows",
        "B_windows",
        "Dims",
        "AUROC",
        "Recall%",
        "Specificity%",
        "Precision%",
    ]
    assert lines[3].index("2", lines[3].index("6") + 1) == lines[2].index("Dims") + 3
    assert lines[3].split() == [
        "K",
        "—",
        "4",
        "6",
        "2",
        "0.875",
        "75.000",
        "83.333",
        "75.000",
    ]


def test_status_column_is_conditional_without_changing_json():
    signature, manifest = analytical_signature()
    ok = validate_signature_summary(signature, manifest=manifest, cluster_id="K")
    for format in ("text", "markdown"):
        listing = make_listing([ok])
        output = render_signature_summaries(listing, format=format)
        assert "Status" not in output.splitlines()[2]
        # C_windows is numeric even though it is now the second displayed column.
        if format == "markdown":
            assert output.splitlines()[3].split("|")[3].strip() == "---:"
        other = {
            **copy.deepcopy(ok),
            "cluster_id": "L",
            "status": "insufficient_core",
            "reason": "insufficient_core",
        }
        output = render_signature_summaries(make_listing([ok, other]), format=format)
        assert "Status" in output.splitlines()[2]
        assert "insufficient_core" in output
        assert (
            "Status"
            in render_signature_summaries(make_listing([]), format=format).splitlines()[
                2
            ]
        )
    assert (
        json.loads(render_signature_summaries(make_listing([ok]), format="json"))[
            "rows"
        ][0]["status"]
        == "ok"
    )
