"""Analytical and presentation checks for own-class explanations."""

from types import SimpleNamespace

import numpy as np
import pytest

from pclass_dark.ml_data import FEATURES
from pclass_dark.ml_interpretation import chosen_support, render_interpretation


def prediction():
    terms = dict.fromkeys(FEATURES, 0.0)
    terms[FEATURES[0]] = 2.0
    terms["surface_base_over_surface"] = 0.144
    terms["surface_charged_over_surface"] = -0.19
    return SimpleNamespace(
        query_id="seq2",
        features=dict.fromkeys(FEATURES, 0.1),
        chosen_feature_contributions=terms,
        chosen_bias=1.0,
        chosen_logit=2.954,
        training_example_count=30,
        sequence_length=129,
        ranked=[SimpleNamespace(cluster_id="1096", logit=2.954, softmax_score=0.85)],
        imputed_features=(),
        outside_support_features=(),
    )


def test_analytical_own_terms_not_pairwise_margin():
    basis = np.arange(1, 19, dtype=np.float32)[None, :]
    classifier = SimpleNamespace(
        preprocess=SimpleNamespace(offsets=np.arange(19), transform=lambda X: basis),
        weight=np.array([[2.0] * 18, [5.0] * 18]),
        bias=np.array([3.0, 7.0]),
    )
    terms, bias = chosen_support(classifier, np.zeros((1, 18)), [0])
    np.testing.assert_array_equal(terms[0], 2 * np.arange(1, 19))
    assert terms.sum() + bias[0] == 345
    assert terms[0, 0] != (2 - 5) * basis[0, 0]
    with pytest.raises(ValueError):
        chosen_support(classifier, np.zeros((1, 18)), [2])


def test_template_uses_own_signs_and_exact_length():
    p = prediction()
    text = render_interpretation(p, descriptions={"1096": "hydrolase activity"})
    assert "129-amino-acid" in text and "hydrolase activity" in text
    assert (
        "| Basic residues among predicted exposed residues | 10.0% | +0.144 |" in text
    )
    assert (
        "| Charged residues among predicted exposed residues | 10.0% | -0.190 |" in text
    )
    assert "runner-up" not in text
    assert len([line for line in text.splitlines() if line.startswith("| ")]) == 23
    p.chosen_logit += 1
    with pytest.raises(ValueError, match="reconciled"):
        render_interpretation(p)


def test_missing_approximate_extreme_length_and_escaping():
    p = prediction()
    p.sequence_length = None
    p.features["ln_length"] = 4.86
    p.query_id = "<script>|[query]"
    text = render_interpretation(p)
    assert "about 129 amino acids" in text
    assert "<script>" not in text and r"\|" in text
    assert "annotation is unavailable" in text
    p.features["ln_length"] = 1000
    assert "ln(length) = 1000.000" in render_interpretation(p)
    p.features["surface_base_over_surface"] = None
    p.imputed_features = ("surface_base_over_surface",)
    p.outside_support_features = ("ln_length",)
    text = render_interpretation(p)
    assert "NA (imputed)" in text and "training medians" in text
    assert "outside the saved training ranges" in text


@pytest.mark.parametrize("sign", [0, -1])
def test_no_positive_support(sign):
    p = prediction()
    p.chosen_feature_contributions = dict.fromkeys(FEATURES, float(sign))
    p.chosen_logit = 18 * sign + p.chosen_bias
    p.ranked[0].logit = p.chosen_logit
    assert "None of the observed feature values" in render_interpretation(p)


def test_description_provenance_binding(tmp_path, monkeypatch):
    from pclass_dark import cluster_descriptions, paths, reference_data
    from pclass_dark.artifacts import sha256, write_json
    from pclass_dark.ml_interpretation import resolve_descriptions

    clusters = tmp_path / "clusters"
    dataset = tmp_path / "dataset"
    clusters.mkdir()
    dataset.mkdir()
    members = []
    for root, name, relative in [
        (clusters, "clusters_meta.tsv", "clusters/clusters_meta.tsv"),
        (dataset, "go_graph.pkl", "dataset/go_graph.pkl"),
    ]:
        path = root / name
        path.write_bytes(b"reference")
        members.append({"path": relative, "sha256": sha256(path), "size": 9})
    manifest = tmp_path / "prepared/p1/manifest.json"
    manifest.parent.mkdir(parents=True)
    write_json(
        manifest,
        {
            "id": "p1",
            "reference": {
                "files": {"source": "bound"},
                "source": {"identity": {"members": members}},
            },
        },
    )
    model = SimpleNamespace(
        labels=("C",),
        feature_contract={
            "extraction": {
                "preparation_id": "p1",
                "prepared_manifest_sha256": sha256(manifest),
            }
        },
    )
    monkeypatch.setattr(paths, "default_setup_input_dir", lambda: tmp_path)
    monkeypatch.setattr(
        reference_data,
        "resolve_profunc",
        lambda: SimpleNamespace(clusters=clusters, dataset=dataset),
    )
    monkeypatch.setattr(
        cluster_descriptions,
        "load_catalog",
        lambda labels: {
            "reference": {"source": "bound"},
            "clusters": {"C": {"description": "activity"}},
        },
    )
    assert resolve_descriptions(model) == {"C": "activity"}
    (clusters / "clusters_meta.tsv").write_bytes(b"changed")
    assert resolve_descriptions(model) == {}
    manifest.unlink()
    assert resolve_descriptions(model) == {}


def test_top_five_order_short_descriptions_and_scores():
    p = prediction()
    p.ranked += [
        SimpleNamespace(cluster_id=f"C{i}", logit=0.0, softmax_score=0.01 * i)
        for i in range(1, 7)
    ]
    descriptions = {
        "1096": "a" * 25,
        "C1": "b" * 24,
        "C2": "short\n description",
        "C3": "<tag>|[x]",
    }
    text = render_interpretation(p, descriptions=descriptions)
    table = text.split("## Top 5 predictions\n", 1)[1].split("## Plain-language", 1)[0]
    rows = [line for line in table.splitlines() if line.startswith("| ")][2:]
    assert len(rows) == 5
    assert rows[0] == "| 1096 | " + "a" * 23 + "… | 0.850 |"
    assert rows[1] == "| C1 | " + "b" * 24 + " | 0.010 |"
    assert rows[2] == "| C2 | short description | 0.020 |"
    assert "&lt;tag&gt;" in rows[3] and r"\|" in rows[3]
    assert rows[4] == "| C4 | Unavailable | 0.040 |"
    assert "C5" not in table
