"""Build → portable additive envelope → inference, without a PLM download or CV."""

import csv
import io
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pclass_dark.additive_reference import DEFAULT_RECIPE, prepare_additive_reference
from pclass_dark.artifacts import (
    IntegrityError,
    file_record,
    read_json,
    sha256,
    write_json,
)
from pclass_dark.ml_data import INDEX_HEADER, PAIR_HEADER
from pclass_dark.setup_data import MODEL, REVISION
from pclass_dark.structural_features import FEATURES, HEADER
from pclass_dark.structural_models import BUNDLE_ID
from test_feature_extraction import fake_predictor as _fake_predictor
from test_inference import fake_model as _fake_model

fake_model = _fake_model
fake_predictor = _fake_predictor


@pytest.fixture
def additive_data(library_data, tmp_path):
    data, _, _, _, _ = library_data()
    data = Path(data)
    sel = read_json(data / "selection.json")
    prepared = data / sel["prepared"]
    pm = read_json(prepared / "manifest.json")
    src = tmp_path / "pairs.tsv"
    with src.open("w") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(PAIR_HEADER)
        w.writerow(("K0", "m0", "K1", "m1", 0.8, 0.2))
    write_json(
        tmp_path / "members.json",
        {"K0": {"medoid_accession": "m0"}, "K1": {"medoid_accession": "m1"}},
    )
    write_json(
        tmp_path / "desc.json", {"K0": "test function zero", "K1": "test function one"}
    )
    manifest = tmp_path / "source.json"
    write_json(
        manifest,
        dict(
            schema="pclass-dark.additive-source.v1",
            source_reference=pm["reference"],
            raw_file=file_record(src),
            members_path="members.json",
            members_file=file_record(tmp_path / "members.json"),
            descriptions_path="desc.json",
            descriptions_file=file_record(tmp_path / "desc.json"),
        ),
    )
    recipe = tmp_path / "recipe.json"
    write_json(recipe, dict(DEFAULT_RECIPE, epochs=2, batch_size=4))
    prepare_additive_reference(
        data, prepared, source=src, source_manifest=manifest, recipe=recipe
    )
    return data


def test_build_and_integrated_infer(additive_data, tmp_path, monkeypatch, fake_model):
    from typer.testing import CliRunner

    from pclass_dark.additive_build import verify_integrated
    from pclass_dark.cli import app
    from pclass_dark.inference import QuerySequence
    from pclass_dark.integrated_inference import infer_integrated_queries
    from pclass_dark.integrated_render import write_integrated_reports
    from pclass_dark.library import build_library
    from pclass_dark.structure import StructuralPrediction

    data = additive_data
    sel = read_json(data / "selection.json")
    feature = tmp_path / "features.tsv"
    index = tmp_path / "sequences.tsv"
    fm = tmp_path / "features.manifest.json"
    with feature.open("w") as f, index.open("w") as g:
        fw = csv.writer(f, delimiter="\t", lineterminator="\n")
        fw.writerow(HEADER)
        iw = csv.writer(g, delimiter="\t", lineterminator="\n")
        iw.writerow(INDEX_HEADER)
        for i in range(20):
            values = [0.1] * 19
            values[0] = 3 + i // 10
            values[-1] = 0.0
            fw.writerow((f"Q{i}", *values, f"K{i // 10}"))
            iw.writerow((f"Q{i}", f"{i:064x}", 20, 1))
    from pclass_dark import ml_model

    code = {
        n: sha256(Path(ml_model.__file__).with_name(n + ".py"))
        for n in ("structure", "structural_features", "structural_models", "embeddings")
    }
    extraction = dict(
        model_id=BUNDLE_ID,
        preparation_id=sel["preparation_id"],
        esm={"model": MODEL, "revision": REVISION},
        code=code,
        features=list(FEATURES),
    )
    write_json(fm, {"identity_inputs": {"contract_inputs": extraction}})
    monkeypatch.setattr(
        "pclass_dark.feature_extraction.extract_features",
        lambda *a, **kw: SimpleNamespace(
            tsv_path=feature, sequences_path=index, manifest_path=fm
        ),
    )
    root = tmp_path / "library"
    result = build_library(data, root, additive=True, additive_device="cpu", workers=1)
    selected = read_json(root / "selection.json")
    path, envelope = verify_integrated(root, selected)
    assert result.additive_model_dir == path / "additive/model"
    assert selected["schema"] == "pclass-dark.library-selection.v3"
    assert not list(tmp_path.glob("**/explore-ml"))

    # Both signature explanation interfaces must read the selector published by
    # the default integrated builder, while retaining their signature evidence.
    from pclass_dark.explanations import explain_signature, list_signature_summaries

    report = explain_signature("K0", root).to_dict()
    assert report["source"]["library_id"] == result.library_id
    assert report["distributions"]["C"]["n"] > 0
    listing = list_signature_summaries(root).to_dict()
    assert {row["cluster_id"] for row in listing["rows"]} == {"K0", "K1"}
    for options, expected in [(["--cluster", "K0"], report), (["--list"], listing)]:
        response = CliRunner().invoke(
            app,
            [
                "explain-signature",
                *options,
                "--library-dir",
                str(root),
                "--format",
                "json",
            ],
        )
        assert response.exit_code == 0, response.output
        assert json.loads(response.stdout) == expected
    assert read_json(root / "selection.json") == selected

    class Predictor:
        def __init__(self, *a, _backend=None, **kw):
            assert _backend is not None

        def predict(self, seq):
            return StructuralPrediction(
                seq, np.full(len(seq), 0.5), np.full(len(seq), "H"), BUNDLE_ID, "cpu"
            )

    monkeypatch.setattr("pclass_dark.structure.StructuralPredictor", Predictor)
    with infer_integrated_queries(
        [QuerySequence("q", "A")],
        library_dir=root,
        additive="required",
        score_batch_size=1,
    ) as run:
        js, md, text = io.StringIO(), io.StringIO(), io.StringIO()
        write_integrated_reports(run, js, md, terminal_stream=text)
    q = json.loads(js.getvalue())["queries"][0]
    assert q["additive"]["status"] == "available"
    assert q["classification"]["selected_cluster_id"] == "K0"
    assert q["additive"]["explanations"][0]["cluster_id"] == "K0"
    assert "test function zero" in md.getvalue()
    prefix = tmp_path / "output"
    cli = CliRunner().invoke(
        app,
        [
            "infer",
            "--seq",
            "A",
            "--library-dir",
            str(root),
            "--report-prefix",
            str(prefix),
            "--score-batch-size",
            "1",
        ],
    )
    assert cli.exit_code == 0, cli.output
    assert "Top 5" in cli.stdout
    assert (tmp_path / "output-classification.complete.json").exists()
    import shutil

    import build_support

    project = tmp_path / "release-project"
    shutil.copytree(data, project / "data/setup-data")
    shutil.copytree(root, project / "data/library")
    files = build_support.inventory(project)
    assert any(
        str(destination).endswith("additive/model/model.npz")
        for _, destination in files
    )
    assert any(
        str(destination).endswith("structural/models.safetensors")
        for _, destination in files
    )
    before = (root / "selection.json").read_bytes()
    # Reuses all-C artifacts and the final numerical fit.
    assert (
        build_library(
            data, root, additive=True, additive_device="cpu", workers=1
        ).additive_model_id
        == result.additive_model_id
    )
    assert (root / "selection.json").read_bytes() == before
    original_code = ml_model._fit_code
    monkeypatch.setattr(ml_model, "_fit_code", lambda: {"simulated_revision": "next"})
    updated = build_library(data, root, additive=True, additive_device="cpu", workers=1)
    assert updated.additive_model_id != result.additive_model_id
    monkeypatch.setattr(ml_model, "_fit_code", original_code)
    restored = build_library(
        data, root, additive=True, additive_device="cpu", workers=1
    )
    assert restored.additive_model_id == result.additive_model_id
    assert list((root / "additive-features").glob("*/cohort-exclusions.tsv"))
    calls_before = len(fake_model)
    with pytest.raises(ValueError, match="length limit"):
        with infer_integrated_queries(
            [QuerySequence("long", "A" * 100001)],
            library_dir=root,
            additive="required",
            score_batch_size=1,
        ):
            pass
    assert len(fake_model) == calls_before
    with infer_integrated_queries(
        [QuerySequence("long", "A" * 100001)],
        library_dir=root,
        additive="auto",
        score_batch_size=1,
    ) as run:
        long = next(run).to_dict()
        assert long["additive"]["reason"] == "length_limit"
        assert long["classification"]["selected_cluster_id"] == "K0"
    (path / "additive/descriptions.json").write_text("{}")
    with pytest.raises(IntegrityError, match="Changed integrated"):
        with infer_integrated_queries([QuerySequence("q", "A")], library_dir=root):
            pass


@pytest.mark.parametrize("fail_first", [False, True])
def test_default_library_real_extraction_build_and_required_infer(
    additive_data, tmp_path, monkeypatch, fake_model, fake_predictor, fail_first
):
    """Run the production pipeline; replace PLM prediction, not extraction."""
    from typer.testing import CliRunner

    from pclass_dark.additive_build import verify_integrated
    from pclass_dark.cli import app
    from pclass_dark.library import build_library

    root = tmp_path / "library"
    monkeypatch.setattr("pclass_dark.paths.default_library_output_dir", lambda: root)
    monkeypatch.setattr(
        "pclass_dark.paths.default_setup_input_dir", lambda: additive_data
    )
    if fail_first:
        fake_predictor.fail_after = 1
        with pytest.raises(RuntimeError, match="injected prediction failure"):
            build_library(additive=True, additive_device="cpu", workers=1, offline=True)
        assert not (root / "selection.json").exists()
        assert list(root.glob("*/manifest.json"))
        assert list((root / "relative-scores").glob("*/manifest.json"))
        fake_predictor.fail_after = None

    result = build_library(
        additive=True, additive_device="cpu", workers=1, offline=True
    )
    selected = read_json(root / "selection.json")
    envelope, _ = verify_integrated(root, selected)
    assert result.additive_model_dir == envelope / "additive/model"
    assert selected["additive_status"] == "complete"
    assert list((root / "additive-features").glob("*/features.tsv.manifest.json"))
    calls = len(fake_predictor.calls)
    original = (root / "selection.json").read_bytes()
    reused = build_library(
        additive=True, additive_device="cpu", workers=1, offline=True
    )
    assert reused.additive_model_id == result.additive_model_id
    assert len(fake_predictor.calls) == calls
    assert (root / "selection.json").read_bytes() == original

    # Extend the embedding-only fixture with synthetic residue states. Actual
    # structural heads, aggregation, scoring and report publication run here.
    from pclass_dark import inference

    embedding_context = inference._backend_context

    @contextmanager
    def residue_context(*args, **kwargs):
        with embedding_context(*args, **kwargs) as (backend, metadata):
            backend.device = "cpu"
            backend.residue_states = lambda sequences: [
                np.zeros((len(sequence), 640), dtype=np.float32)
                for sequence in sequences
            ]
            yield backend, metadata

    monkeypatch.setattr(inference, "_backend_context", residue_context)
    prefix = tmp_path / "required"
    cli = CliRunner().invoke(
        app,
        [
            "infer",
            "--seq",
            "ACDEKR",
            "--additive",
            "required",
            "--device",
            "cpu",
            "--offline",
            "--report-prefix",
            str(prefix),
            "--score-batch-size",
            "1",
        ],
    )
    assert cli.exit_code == 0, cli.output
    report = read_json(tmp_path / "required-classification.json")
    assert report["queries"][0]["additive"]["status"] == "available"
    assert (tmp_path / "required-classification.complete.json").exists()
