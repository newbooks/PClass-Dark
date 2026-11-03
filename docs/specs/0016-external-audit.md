# SPEC-0016: External evaluation audit

- Date: 2026-11-02.
- Status: Implemented and validated (2026-11-02).

## Purpose and provenance

Analyze SPEC-0015 comparison TSVs without rerunning inference. Use the existing
AMB Lin match scores directly; a hit is match >= 0.7 by default. This is semantic
hit rate, not exact-label accuracy. No fitting or reference-method dependency.

## Interfaces and contracts

Provide scripts/external-audit.py INPUT and pclass-dark external-audit INPUT;
reusable pclass_dark.external_audit parsing/analysis/rendering APIs. Optional
[audit] dependency supplies Matplotlib, imported only for rendering. Input uses
the seven SPEC-0015 headers, arbitrary filename, unique nonempty query IDs,
NA/empty missing values, match/B scores in [0,1], distances in [0,2]. Predictions
require a nonempty GO column. Reject duplicate headers/IDs, malformed rows,
nonfinite/out-of-range numbers and scored rows without predictions. Numeric zero
is defined. Do not treat missing match scores as failures in hit-rate denominators.

Default output is beside INPUT; optional output directory, bins (10), hit cutoff
(0.7). All output basenames start external_audit_. Scatter, confidence, coverage,
and summary-table PNGs plus a machine-readable summary TSV. Never overwrite input.
Stage complete outputs before publication; replace existing audit outputs.

## Analysis and visualization

- Scatter: x Pclass_match, y Dark_match, equal [0,1] axes, identity line, cutoff
  lines. Only jointly scorable queries; report excluded count and unique hits.
- Confidence: two panels, distance and B fraction, same number of [0,1] equal-width
  bins. Stacked hit/non-hit/undefined-score counts; total bar height is predictions,
  hit percentage above each bar uses scorable predictions. Zero denominator is NA.
  Retain distance >1 in a separate overflow category; retain predictions with
  missing confidence in an NA category. Small distance/B is stronger. Bins are
  descriptive, not a calibration/equivalence between methods.
- Coverage: two panels, smaller evidence first, tied values retained together.
  x retained predictions / all queries (%), labeled "Strong to weak"; y hits / scorable retained predictions
  (%). Only predictions with finite evidence are rankable. Undefined matches count
  in coverage but not hit-rate denominator; undefined rates are gaps. Never extend
  curves past available rankable coverage or manufacture a (0,100) origin.
- Summary: total, predicted/scorable/hit counts, prediction coverage, hit rate,
  hits/all queries, both-hit and method-only-hit counts, undefined matches and
  missing confidence. Report distance >1 count and fraction among finite distances
  across all rows, separately from prediction coverage.

## Resources, edge cases and validation

Read once, hold O(N) numeric arrays and GO-presence masks; CPU NumPy operations,
no GPU benefit, no workers or nested thread pools needed. Stable O(N log N)
ranking with deterministic ties. Headless Matplotlib; no computation on import.
Support headers-only tables and empty panels. Analytical tests cover bin boundaries,
overflow, zero, missingness, ties, denominators, invalid input and CLI integration;
verify PNG artifacts on a small fixture and run the available real TSV, recording
its SHA-256, counts, commands and artifact paths. No claims about homology leakage
or held-out model quality from these descriptive plots.

## Validation and actual-data run

Audit/evaluation/CLI suites: 87 passed (30.31 s), log
`/tmp/external-audit-final-tests.log`. Analytical examples verify hit/scorable
counts, exact bin boundaries, overflow/NA, tied thresholds and denominators.
Tests cover invalid values, duplicate IDs/headers, empty/unscorable inputs,
PNG headers, summary TSV and both script/installed CLI adapters.

Ran `python scripts/external-audit.py work-unknown/external_evaluation.tsv` with
10 bins and hit cutoff 0.7; final headless run used
`MPLCONFIGDIR=/tmp/pclass-external-audit-mpl` (restricted home cache in this
execution environment). Input SHA256:
`09943c5a25388cf4f38e1b712c7470f0e5c33824f4841bbe50df92c8e5b97821`.
27,910 queries; no distances >1 (0/27,910), maximum 0.3711543083190918.
Both methods have GO predictions for all queries. Scorable counts are 27,845
(PClass) and 27,847 (Dark); hits 7,017 and 3,229; hit rates 25.20% and 11.60%.
Both hit 2,190 queries; method-only hits 4,827 and 1,039. These describe this cohort
and operating point, not independently controlled held-out generalization.

Outputs: `work-unknown/external_audit_{match_scores,confidence,coverage,summary}.png`
and `work-unknown/external_audit_summary.tsv`. Visually checked all plot designs;
confidence panels share the count scale. CPU rendering with Python 3.14.6,
NumPy 2.4.4, Matplotlib 3.11.0; no random sampling/seeds or model computation.
Baseline revision 609e2a6e5099603e3f3afa2795f7a68d6f37ee55 plus uncommitted audit
implementation. Per-file publication is staged; replacing the entire output set
is not an atomic multi-file transaction. All bins on [0,1] are preserved even
when a measure occupies a narrow range. No score calibration is inferred.

## V2 protein-binding exclusion (2026-11-02)

CLI/script defaults now exclude every query whose own truth MF annotations contain
GO:0005515, including mixed sets. Never use PClass_go or Dark_go for this filter.
Resolve truth from the input's verified evaluation publication/score manifest and
query_status.jsonl, or accept explicit --truth-tsv with Query_id/go_mf_terms for
standalone/renamed inputs. Require exact ID coverage, valid nonempty GO sets and
unique IDs. Missing/unverifiable truth fails rather than silently retaining rows.
Python audit_external adds optional exclude_go and truth_tsv; its default remains
unfiltered for compatibility. Filter before all counts, binning and coverage so
all-query denominators use retained queries. Report original/excluded/retained
counts. Filtered outputs use external_audit_*_v2.png and summary_v2.tsv, preserving
original outputs. Stage before publication and protect input/truth paths.
Test annotation-based exclusion, mixed sets, denominators, missing truth, verified
metadata identity and CLI output preservation. Generate the actual v2 PNGs.

Implemented/validated: audit/evaluation/CLI suites **92 passed** (32.79 s;
`/tmp/external-audit-v2-final-tests.log`). Cases include truth versus predicted
GO filtering, mixed sets, exact ID coverage, duplicate/missing annotations,
metadata tampering, empty retained cohorts, v1 preservation and recalculated
coverage denominators. Ruff and whitespace checks passed.

Real run: `MPLCONFIGDIR=/tmp/pclass-external-audit-mpl python
scripts/external-audit.py work-unknown/external_evaluation.tsv`. Same source hash
and numerical settings as original audit; annotation metadata verified from the
published score ID 5f3b999fddc68ad6e1e08e046a235b07c0899b1d14d1fdbcc06ca8f64a74c2c2.
Excluded 13,530 / 27,910; retained 14,380. No retained distances >1; maximum
0.3559103012084961. Both methods predict all retained queries; each has 14,317
scorable matches. PClass hits 5,439 (37.99%); Dark hits 2,548 (17.80%). Both hit
1,780; method-only hits 3,659 and 768. Wrote all four _v2 PNGs and summary_v2.tsv
beside the input; checked the summary PNG visually. This annotation-selected
posthoc cohort is descriptive, not a new independently held-out evaluation.
