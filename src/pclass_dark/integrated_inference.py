"""Disk-backed integrated inference, sharing the signature inference backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import IntegrityError, read_json, schema_checked, sha256
from .inference import InferenceSession
from .integrated_records import combine_evidence
from .paths import default_library_input_dir


def unavailable(result, reason, *, model=None):
    return dict(
        status="disabled" if reason == "disabled_by_user" else "unavailable",
        reason=reason,
        model_id=model.manifest["model_id"] if model else None,
        configuration=model.manifest["contract"]["configuration"] if model else None,
        reference_id=None,
        recipe_id=None,
        global_top5=[],
        accepted_reranking=[
            dict(
                cluster_id=m.cluster_id,
                description=None,
                score_available=False,
                reason=reason,
                logit=None,
                softmax_score=None,
                global_rank=None,
            )
            for m in result.matches
        ],
        features=None,
        feature_device=None,
        sequence_length=result.length,
        imputed_features=[],
        outside_support_features=[],
        explanations=[],
    )


def describe_features(model, prediction, scores, sequence_length):
    """Explain the selected accepted class and accepted global top five."""
    from .ml_interpretation import EPSILON, chosen_support
    from .ml_model import _rounded_features

    ranked = sorted(
        (r for r in scores["candidates"] if r["score_available"]),
        key=lambda r: (-r["logit"], r["cluster_id"]),
    )
    accepted = {r["cluster_id"] for r in ranked}
    selected = ranked[0]["cluster_id"] if ranked else None
    ids = ([selected] if selected else []) + [
        r["cluster_id"] for r in scores["global_top5"] if r["cluster_id"] in accepted
    ]
    features = prediction.features
    X = _rounded_features(
        model,
        np.array(
            [
                [
                    np.nan if features[f] is None else features[f]
                    for f in model.feature_contract["feature_names"]
                ]
            ]
        ),
    )
    explanations = []
    for label in dict.fromkeys(ids):
        k = model.labels.index(label)
        terms, bias = chosen_support(model.classifier, X, np.array([k]))
        contrib = dict(
            zip(
                model.feature_contract["feature_names"],
                map(float, terms[0]),
                strict=True,
            )
        )
        logit = float(model.classifier.logits(X)[0, k])
        if not np.isclose(sum(contrib.values()) + bias[0], logit, atol=1e-4, rtol=1e-4):
            raise IntegrityError(
                "Selected-class additive explanation does not reconcile"
            )
        positive = sorted(
            (f for f, v in contrib.items() if v > EPSILON and features[f] is not None),
            key=lambda f: -contrib[f],
        )
        negative = sorted(
            (f for f, v in contrib.items() if v < -EPSILON and features[f] is not None),
            key=lambda f: contrib[f],
        )
        explanations.append(
            dict(
                cluster_id=label,
                feature_contributions=contrib,
                bias=float(bias[0]),
                logit=logit,
                reference="training_mean",
                training_example_count=model.classifier.training_example_count,
                positive_observed_features=positive,
                negative_observed_features=negative,
            )
        )
    return dict(
        status="available",
        reason=None,
        model_id=prediction.model_id,
        configuration=prediction.configuration,
        reference_id=None,
        recipe_id=None,
        global_top5=scores["global_top5"],
        accepted_reranking=scores["candidates"],
        features=features,
        sequence_length=sequence_length,
        feature_device=prediction.feature_device,
        imputed_features=list(prediction.imputed_features),
        outside_support_features=list(prediction.outside_support_features),
        explanations=explanations,
    )


class IntegratedSession(InferenceSession):
    def __init__(self, queries, *, additive="auto", **kwargs):
        if additive not in ("auto", "off", "required"):
            raise ValueError("Invalid additive mode")
        self.additive_mode = additive
        self.model = self.envelope = self.descriptions = None
        self.supplemental_reason = (
            "disabled_by_user" if additive == "off" else "missing_component"
        )
        super().__init__(queries, **kwargs)

    @schema_checked
    def _compute(self):
        root = Path(self._library_dir or default_library_input_dir())
        self._integrated_root = root
        if self.additive_mode != "off":
            from .additive_build import verify_integrated
            from .ml_model import _raw_contract, load_final_model

            selected = read_json(root / "selection.json")
            if "integrated" in selected:
                path, self.envelope = verify_integrated(root, selected)
                self.model = load_final_model(path / "additive/model")
                if (
                    self.model.manifest["model_id"]
                    != self.envelope["identity_inputs"]["model_id"]
                ):
                    raise IntegrityError("Integrated/model identity mismatch")
                extraction = _raw_contract(self.model)
                if extraction.get("preparation_id") != selected["preparation_id"]:
                    raise IntegrityError("Integrated/model preparation mismatch")
                self.heads_path = path / "structural"
                from .structural_models import verify_structural_models

                heads = verify_structural_models(self.heads_path)
                if (
                    heads.model_id
                    != self.model.feature_contract["extraction"]["model_id"]
                ):
                    raise IntegrityError("Classifier/head extraction identity mismatch")
                self.descriptions = read_json(path / "additive/descriptions.json")
                self._selected_digest = sha256(root / "selection.json")
            elif self.additive_mode == "required":
                raise ValueError(
                    "Library has no additive model; rebuild with additive enabled"
                )
        if self.additive_mode != "off" and self.model is None:
            selected = read_json(root / "selection.json")
            if selected.get("additive_status") == "disabled_by_build":
                self.supplemental_reason = "disabled_by_build"
        self._db.execute(
            "CREATE TABLE supplemental(query_index INTEGER PRIMARY "
            "KEY,data TEXT NOT NULL)"
        )
        if self.model and self.additive_mode == "required":
            if self._db.execute(
                "SELECT 1 FROM queries WHERE length>100000 LIMIT 1"
            ).fetchone():
                raise ValueError("Additive sequence length limit exceeded")
        super()._compute()

    def _supplemental_compute(self, backend):
        if not self.model:
            return
        from .ml_model import predict_features, score_candidates
        from .structural_features import summarize_structural_features
        from .structure import StructuralPredictor

        predictor = None
        c = len(self.model.labels)
        b = self.model.classifier.preprocess.offsets[-1]
        need = b * 16 + c * 32 + self.model.classifier.weight.nbytes + 64 * 2**20
        if need > self._options["memory_mib"] * 2**20:
            raise ValueError("Insufficient additive scoring workspace")
        previous = self._emitted
        state = self._state
        self._state = "active"
        with (self._directory / "queries.jsonl").open() as source:
            for i, line in enumerate(source):
                self._emitted = i
                result = InferenceSession.__next__(self)
                name, sequence = json.loads(line)
                if len(sequence) > 100000:
                    if self.additive_mode == "required":
                        raise ValueError("Additive sequence length limit exceeded")
                    value = unavailable(result, "length_limit", model=self.model)
                else:
                    if predictor is None:
                        predictor = StructuralPredictor(
                            self.heads_path,
                            device=self._options["device"],
                            offline=self._options["offline"],
                            window_batch_size=min(8, self._options["batch_size"]),
                            cpu_threads=min(8, self._options["workers"]),
                            _backend=backend,
                        )
                    structural = predictor.predict(sequence)
                    summary = summarize_structural_features(
                        structural.sequence, structural
                    )
                    X = np.array(
                        [[np.nan if v is None else v for v in summary.values[:-1]]]
                    )
                    prediction = next(
                        predict_features(
                            self.model, X, [name], feature_device=structural.device
                        )
                    )
                    scores = next(
                        score_candidates(
                            self.model,
                            X,
                            [[m.cluster_id for m in result.matches]],
                            batch_size=1,
                        )
                    )
                    for row in scores["global_top5"] + scores["candidates"]:
                        row["description"] = self.descriptions.get(row["cluster_id"])
                    value = describe_features(
                        self.model, prediction, scores, len(sequence)
                    )
                for row in value["accepted_reranking"]:
                    row["description"] = self.descriptions.get(row["cluster_id"])
                value["reference_id"] = self.envelope["identity_inputs"]["reference_id"]
                value["recipe_id"] = self.envelope["identity_inputs"]["recipe_id"]
                encoded = json.dumps(value, allow_nan=False)
                self._quota(len(encoded.encode()))
                self._db.execute("INSERT INTO supplemental VALUES(?,?)", (i, encoded))
        self._emitted = previous
        self._state = state
        if sha256(self._integrated_root / "selection.json") != self._selected_digest:
            raise IntegrityError("Integrated selection changed during inference")

    def __next__(self):
        result = super().__next__()
        row = self._db.execute(
            "SELECT data FROM supplemental WHERE query_index=?", (result.query_index,)
        ).fetchone()
        a = json.loads(row[0]) if row else unavailable(result, self.supplemental_reason)
        return combine_evidence(result, a)


def infer_integrated_queries(queries, *, additive="auto", **options):
    """Return a context-managed one-shot integrated session; imports are inert."""
    return IntegratedSession(queries, additive=additive, **options)
