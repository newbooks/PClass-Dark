"""Direct fitting, all-candidate ranking and integrated result preservation."""

import numpy as np
import pytest

from pclass_dark.additive_reference import DEFAULT_RECIPE
from pclass_dark.inference_records import ClusterMatch, QueryResult, WindowInfo
from pclass_dark.integrated_inference import describe_features, unavailable
from pclass_dark.integrated_records import combine_evidence
from pclass_dark.integrated_render import render_query
from pclass_dark.ml_data import FEATURES
from pclass_dark.ml_model import (
    fit_reference_model,
    load_final_model,
    predict_features,
    score_candidates,
)


@pytest.fixture
def direct_model(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("sklearn")
    X = np.array(
        [[float(i // 4) + j * 0.01 for j in range(18)] for i in range(12)],
        dtype=np.float64,
    )
    y = np.repeat(np.arange(3, dtype=np.int64), 4)
    fc = dict(
        schema="pclass-dark.ml-feature-contract.v1",
        feature_names=list(FEATURES),
        precision=".3f",
        extraction=None,
    )
    recipe = dict(DEFAULT_RECIPE, epochs=2, batch_size=4)
    args = (X, y, ("A", "B", "Z"), np.eye(3), fc, recipe, tmp_path / "model")
    fit = fit_reference_model(
        *args, provenance={"prepared": "p"}, device="cpu", threads=1
    )
    assert fit.example_count == 12
    assert fit_reference_model(
        *args, provenance={"prepared": "p"}, device="cpu", threads=1, resume=True
    ).reused
    return load_final_model(fit.model_dir)


def original(labels=("A", "B")):
    window = WindowInfo("w", "full", 1, 10, "a" * 64)
    matches = tuple(ClusterMatch(c, 0.1, 0.8, 10, 10, window, ()) for c in labels)
    return QueryResult(
        0,
        "query",
        10,
        "b" * 64,
        (window,),
        "matches" if labels else "no_accepted_matches",
        matches,
    )


def test_direct_fit_and_global_normalization(direct_model):
    X = np.full((1, 18), 0.1)
    scores = next(score_candidates(direct_model, X, [["A", "B", "missing"]]))
    assert sum(r["softmax_score"] for r in scores["global_top5"]) == pytest.approx(1)
    assert scores["candidates"][-1]["logit"] is None
    p = next(predict_features(direct_model, X, ["q"]))
    a = describe_features(direct_model, p, scores, 10)
    q = combine_evidence(original(("A", "B", "missing")), a).to_dict()
    assert q["classification"]["selected_cluster_id"] in ("A", "B")
    assert q["additive"]["accepted_reranking"][-1]["cluster_id"] == "missing"
    assert len(a["explanations"]) <= 3
    assert q["pclass_dark"]["matches"][0]["background_tail_fraction"] == 0.1
    for e in a["explanations"]:
        assert sum(e["feature_contributions"].values()) + e["bias"] == pytest.approx(
            e["logit"], abs=1e-4
        )
    md, text = render_query(q)
    assert (
        md.index("PClass-Dark accepts")
        < md.index("Detailed PClass-Dark")
        < md.index("Additive reranking analysis")
    )
    assert "Top 5" in text


def test_selection_does_not_promote_global_winner():
    result = original()
    a = unavailable(result, "missing_component")
    a.update(status="available", reason=None)
    a["global_top5"] = [dict(cluster_id="Z", softmax_score=0.8, logit=4, rank=1)]
    a["accepted_reranking"] = [
        dict(cluster_id=c, logit=v, score_available=True, softmax_score=0.1)
        for c, v in [("A", 1.0), ("B", 2.0)]
    ]
    q = combine_evidence(result, a).to_dict()
    assert q["classification"]["selected_cluster_id"] == "B"
    assert not q["classification"]["in_additive_top5"]
    assert q["pclass_dark"]["matches"][0]["cluster_id"] == "A"
    a["accepted_reranking"][0]["logit"] = 2.0
    assert (
        combine_evidence(result, a).to_dict()["classification"]["selected_cluster_id"]
        == "A"
    )


def test_no_accepted_and_unavailable():
    result = original(())
    q = combine_evidence(result, unavailable(result, "missing_component")).to_dict()
    assert q["classification"]["selected_cluster_id"] is None
    assert "no accepted cluster" in render_query(q)[0]
    result = original()
    q = combine_evidence(result, unavailable(result, "disabled_by_user")).to_dict()
    assert q["classification"]["selected_cluster_id"] == "A"
    assert "original ordering is retained" in render_query(q)[0]


# Existing fake embedding fixture verifies the actual signature path unchanged.
from test_inference import fake_model as _fake_model  # noqa: E402
from test_inference import query_library as _query_library  # noqa: E402

fake_model = _fake_model
query_library = _query_library


def test_integrated_legacy_library_preserves_acceptance(
    query_library, fake_model, tmp_path
):
    import io
    import json

    from pclass_dark.inference import QuerySequence, infer_queries
    from pclass_dark.integrated_inference import infer_integrated_queries
    from pclass_dark.integrated_render import write_integrated_reports

    root, _ = query_library
    with infer_queries(
        [QuerySequence("q", "A")], library_dir=root, score_batch_size=1
    ) as run:
        original_result = next(run).to_dict()
    with infer_integrated_queries(
        [QuerySequence("q", "A")], library_dir=root, score_batch_size=1
    ) as run:
        js, md, text = io.StringIO(), io.StringIO(), io.StringIO()
        write_integrated_reports(run, js, md, terminal_stream=text)
    report = json.loads(js.getvalue())
    q = report["queries"][0]
    assert q["additive"]["reason"] == "disabled_by_build"
    assert [
        {k: v for k, v in m.items() if k != "original_rank"}
        for m in q["pclass_dark"]["matches"]
    ] == json.loads(json.dumps(original_result["matches"]))
    assert "original ordering is retained" in text.getvalue()
    with pytest.raises(ValueError, match="no additive model"):
        with infer_integrated_queries(
            [QuerySequence("q", "A")], library_dir=root, additive="required"
        ):
            pass


def test_query_record_fixture_and_missing_candidate_validation():
    import json
    from pathlib import Path

    from pclass_dark.artifacts import IntegrityError

    fixture = json.loads(
        Path("docs/schemas/fixtures/integrated-infer-query.valid.json").read_text()
    )
    assert fixture["schema"] == "pclass-dark.integrated-infer-query.v1"
    result = original()
    a = unavailable(result, "missing_component")
    a["accepted_reranking"].pop()
    with pytest.raises(IntegrityError, match="accepted set"):
        combine_evidence(result, a)
