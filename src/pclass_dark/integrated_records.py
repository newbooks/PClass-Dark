"""Pure accepted-candidate reranking and serializable integrated query records."""

from copy import deepcopy
from dataclasses import dataclass

from .artifacts import IntegrityError


@dataclass(frozen=True)
class IntegratedQuery:
    data: dict

    def to_dict(self):
        return self.data


def combine_evidence(result, supplemental):
    """Preserve accepted records; select only within accepted candidates."""
    original = result.to_dict()
    matches = [
        dict(row, original_rank=i + 1) for i, row in enumerate(original["matches"])
    ]
    a = deepcopy(supplemental)
    original_rank = {r["cluster_id"]: r["original_rank"] for r in matches}
    if a.get("status") not in {"available", "unavailable", "disabled"}:
        raise IntegrityError("Invalid additive availability status")
    candidate_ids = [r["cluster_id"] for r in a["accepted_reranking"]]
    if len(candidate_ids) != len(set(candidate_ids)) or set(candidate_ids) != set(
        original_rank
    ):
        raise IntegrityError("Additive candidates must preserve the accepted set")
    if a["status"] != "available" and a["global_top5"]:
        raise IntegrityError("Unavailable additive evidence cannot contain predictions")
    top = a["global_top5"]
    for r in top:
        r["accepted_by_pclass_dark"] = r["cluster_id"] in original_rank
    rows = a["accepted_reranking"]
    for row in rows:
        row["original_rank"] = original_rank[row["cluster_id"]]
    rows.sort(
        key=lambda r: (
            not r["score_available"],
            -r["logit"] if r["score_available"] else r["original_rank"],
            r["cluster_id"] if r["score_available"] else "",
        )
    )
    for i, row in enumerate(rows):
        row["position"] = i + 1
    chosen = next((r for r in rows if r["score_available"]), None)
    chosen = chosen or (rows[0] if rows else None)
    topids = [r["cluster_id"] for r in top if r["accepted_by_pclass_dark"]]
    a["top5_accepted_cluster_ids"] = topids
    explanations = {e["cluster_id"]: e for e in a["explanations"]}
    selected = chosen["cluster_id"] if chosen else None
    classification = dict(
        selected_cluster_id=selected,
        selection_rule="none"
        if not chosen
        else "additive_reranked"
        if chosen["score_available"]
        else "pclass_original",
        original_rank=chosen["original_rank"] if chosen else None,
        reranked_position=chosen["position"] if chosen else None,
        in_additive_top5=selected in [r["cluster_id"] for r in top],
        selected_explanation_cluster_id=selected if selected in explanations else None,
    )
    agreement = (
        "no_accepted_matches"
        if not chosen
        else "additive_unavailable"
        if a["status"] != "available"
        else "selected_in_global_top5"
        if classification["in_additive_top5"]
        else "selected_outside_global_top5"
    )
    return IntegratedQuery(
        dict(
            schema="pclass-dark.integrated-infer-query.v1",
            **{k: v for k, v in original.items() if k not in ("schema", "matches")},
            pclass_dark=dict(matches=matches),
            additive=a,
            classification=classification,
            analysis=dict(agreement=agreement),
        )
    )
