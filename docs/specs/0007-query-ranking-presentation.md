# SPEC-0007: Query ranking and A/B/C presentation

- Date: 2026-10-27.
- Status: Planned; documentation only. Query ranking is not implemented by this change.
- Provenance: User-approved naming and existing acceptance/evidence contracts in
  [SPEC-0004](0004-build-library.md) and [SPEC-0005](0005-relative-score-distributions.md).

## Purpose and labels

Future top-matching-cluster results will use **A — Acceptance**,
**B — Background score ↓**, and **C — Cluster score ↑**. Arrows show score
direction, not change over time. A is a Yes/No flag and needs no direction arrow.

| Label | Existing source | Meaning / tooltip |
|---|---|---|
| A — Acceptance | Signature acceptance flag | Does the query pass this cluster's fitted threshold? Yes/No. |
| B — Background score ↓ | `background_tail_fraction` | Reference nonmembers scoring at least this high, with plus-one adjustment. Lower is better. |
| C — Cluster score ↑ | `cluster_percentile_le` | Reference members scoring at or below the query. Higher means a higher score rank among members. |

## Data and calculation contracts

Use a verified production signature and its paired C/B distributions. For query
log score `l`, preserve existing formulas and inclusive, exact score ties:

```text
A = (l >= fitted log cutoff), or No for a predict-none threshold
B = (1 + count(B_log_scores >= l)) / (n_B + 1)
C = count(C_log_scores <= l) / n_C
```

The cutoff maximizes training balanced accuracy, with highest-cutoff ties.
A is not derived from B or C. Display B/C as percentages, retaining fractions
in [0, 1] and existing descriptive field names in the underlying Python API.
Provide population sizes and availability status with the evidence.

Illustrative presentation:

| Rank | Cluster | A — Acceptance | B — Background score ↓ | C — Cluster score ↑ |
|---|---|---|---|---|
| 1 | cluster_example | Yes | 1% | 70% |

Lay explanation: "Only about 1% of reference nonmembers score at least this
high; the query scores at least as highly as 70% of reference members."
B includes the plus-one adjustment; it is not exactly the raw observed fraction.
Neither percentage describes how many windows have similar scores to the query.

## Planned ranking and edge cases

The initial planned heuristic keeps accepted clusters and sorts by ascending
unrounded B, displaying C as supporting evidence. Do not combine A/B/C into an
unvalidated numerical score. A low C means weak compatibility compared with
known members; a high C means higher score rank, not greater member typicality.
C=100% can also occur above every observed member score.

Equal B values retain equal evidence rank; cluster ID may give deterministic
display order within ties without implying stronger evidence. Include tied
matches at a top-k boundary. Round only for display; avoid displaying small
positive B values as zero. Final display precision remains an implementation choice.

When none are accepted, report no accepted matches. An optional rejected-match
view must retain A=No. Unavailable signatures or distributions carry an explicit
unavailable status/reason, not fabricated zero scores or No flags, and are
excluded from ranking.

These are descriptive comparisons, not membership probabilities. Each cluster
has its own signature/background, and final C/B were used during discovery.
B ordering compares background-relative evidence; it does not establish relative
membership likelihood. Multiple clusters may be accepted. Do not rank raw
signature scores across clusters or multiply/normalize B and C into probabilities.

## Future implementation and validation

Expose reusable Python ranking logic and a thin `pclass-dark` query adapter.
Reuse existing scoring/evidence operations without refitting parameters or
thresholds. Sequence embedding, window aggregation, candidate retrieval,
resource budgets and final query interfaces require specification before code
implementation. This document does not define protein-level aggregation of
multiple window results or introduce new runtime dependencies.

Tests must cover field/label mapping, percentages, cutoff equality, predict-none,
inclusive ties, plus-one B, sorting before rounding, tied top-k boundaries,
unavailable data and no accepted candidates. Assess ranking utility on held-out
queries with homology controls; fit any later ranking parameters on training data
only. Ranking performance is not established by the current in-sample reports.

Documentation-only validation: local links, labels/formulas and whitespace;
no runtime behavior changed and no code tests required.
