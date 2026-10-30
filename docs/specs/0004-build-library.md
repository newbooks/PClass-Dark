# SPEC-0004: Production signatures and optional cross-validation

- Revised: 2026-10-27.
- Status: Implemented on 2026-10-27; final-only production, empirical C/B
  distributions and optional research are available. Verification recorded below.
- Supersedes the mandatory-six-fit execution contract preserved in the
  [historical specification](0004-build-library-legacy-six-fit.md).
- Inputs: [SPEC-0003 setup-data](0003-setup-data.md).
- Production score distributions: [draft SPEC-0005](0005-relative-score-distributions.md).
- Method provenance: ProStruct's cluster profiling/composition, adapted to PLM
  coordinates with the approved first-improvement removal policy.

## Purpose and approved change

Make a usable signature library available without mandatory cross-validation.
By default, fit one signature per target using all C windows and its existing
fixed B. Calculate final-signature C/B score distributions within build-library
as specified in SPEC-0005. Five-fold discovery becomes an optional research phase,
usually scoped to selected clusters.

The all-C fitting algorithm remains the same one currently used for `final`.
This change removes five mandatory discovery fits; it does not weaken feature
screening, sample B differently, change parameters, or introduce approximate
AUROC. Optional research does not contribute parameters to production signatures.

The expected discovery-work reduction is approximately six fits to one per target,
not a guaranteed sixfold command speedup. Fit difficulty, scoring, IO, validation,
and publication overhead vary. No new timing results are claimed by this revision.

The CLI `--workers` default is 8 (updated 2026-10-29). Explicit
`--workers 1` retains serial execution. Python API defaults remain 1; existing
worker admission limits still apply. CLI default/override regression validation
is recorded in the progress log.

## Commands and Python API

```bash
# Default: one all-C discovery fit per target, plus final C/B distributions.
pclass-dark build-library

# Optional research for a particular cluster; reuse its final signature if valid.
pclass-dark build-library --cross-validate --cluster cluster_0002
```

```python
from pclass_dark.library import build_library

result = build_library(
    data_dir=None, output_dir=None,  # Shared paths; see SPEC-0006.
    workers=4, memory_mib=4096, cross_validate=False,
)
```

Use the shared path defaults in SPEC-0006. The CLI uses internal workspace
(1024 MiB), epsilon (1e-8) and sigma-floor (1e-8) defaults; Python accepts explicit
settings. Expose workers, repeated exact `--cluster`, and `--force`.
Add `--cross-validate/--no-cross-validate`, default false; Python keyword
`cross_validate=False`. Workers remain bounded by CPUs, target count, requested
workspace, and 32. Workspace minimum is 64 MiB. CLI/API inputs must be validated
before computational work; require a real boolean for `cross_validate`.
Keep existing error/exit conventions. A failed optional research phase raises
`ResearchBuildError(RuntimeError)` with `production_result`, `research_id` and
chained cause, so Python callers can use the already committed production pair.
CLI reports its paths and exits 1; Ctrl-C remains exit 130 rather than being
wrapped as a successful or ordinary failed research result.

`--cross-validate` requests five research fits in addition to the production
signature; it never removes or changes that signature. Without `--cluster`, the
flag applies to all requested targets, so help must explain the additional cost.
A scoped build retains B selected from all original retained clusters.

`--force` remains an explicit intentional rebuild of requested work. Normal
resume, migration, distribution backfill, or adding research must not require it.
Do not prune other libraries, research scopes, or checkpoint generations.

Extend the frozen result compatibly with defaulted `relative_id`, `relative_dir`,
`research_id`, `research_dir`, and reuse metadata. Absent research returns null
research fields, not an empty research artifact presented as completed CV.
Raw scoring API return values remain unchanged. Imports must not compute,
launch workers, download weights, or write package resources.

## Populations and representation

Validate the selected prepared/embedding manifests, matrix checksums, inventory,
row joins, compact C/B indices, counts, model revision and tokenizer hashes,
residue policy, dtype, width 640, and representation contract. Setup-data remains
responsible for residue-only mean pooling using attention AND NOT special-token
masks. Build-library consumes the fixed vectors and never pools tokens again.

For each target k:

- Production C is every window in its exact retained C pool.
- B is the existing fixed union of capped samples from other source clusters,
  up to 200 windows per source cluster, with target-parent exclusion and existing
  deduplication rules. No resampling or refill is introduced.
- With research requested, rotation f uses C minus fold f for discovery and fold
  f for held-out positives. B stays unchanged. Folds are within each cluster,
  window-level, and may separate N/C windows or identical windows, as approved.

A default fit has no V population. Keep prepared-data structural integrity
checks, but do not schedule research merely because folds exist in setup-data.
Only research fit dependencies include fold assignments. B remains a reused
selection background, not independent calibration data.

## Unchanged scientific fitting rules

For each dimension j, fit C/B means and sample SDs (`ddof=1`) in canonical window
order using CPU float64. Cache B moments once per target. Store:

```text
R_j = sigma_C,j / (sigma_B,j + epsilon)
D_j = abs(mu_C,j - mu_B,j)
      / (hypot(sigma_C,j, sigma_B,j) / sqrt(2) + epsilon)
width_j = max(sigma_C,j, sigma_floor)
log_g_j(x) = -0.5 * ((x_j - mu_C,j) / width_j)**2
log_S(x) = logsumexp(log_g_j(x) for j in selected) - log(n_selected)
```

Use ordered left-to-right `logaddexp` for reference composite scoring. Do not
replace log of an arithmetic mean with mean log compatibility. Individual
Gaussian AUROC uses the equivalent negative-distance ordering to avoid underflow.
For empirical AUROC, retain exact `2*wins + ties` integer numerators and
`2*n_C*n_B` denominators; compare integer counts for strict improvements.

1. Rank all 640 dimensions by individual training AUROC, then numeric dimension
   ID for exact ties. Profile all dimensions, not just those eventually selected.
2. Evaluate every ranked prefix, selecting the earliest maximum-AUROC prefix.
3. Protect the first two ranked dimensions. Candidates are their union with
   dimensions through the best prefix and every individually ranked dimension
   with AUROC strictly greater than 0.90 (`10*q > 9*denominator`).
4. Visit remaining removable dimensions in rank order. Accept the first removal
   that strictly improves training composite AUROC, then restart the scan.
   Stop on a complete non-improving scan; never remove the protected core.
5. Select the threshold maximizing training balanced accuracy. Use exact integer
   objectives and highest-cutoff ties; explicitly serialize predict-none rather
   than a nonfinite cutoff. This is not precision or probability optimization.

Retain the implemented bounded running-prefix caches, reusable buffers, stable
leave-one-out log subtraction, cancellation/near-tie reference fallback, and
reference confirmation of proposed gains. Final metrics, thresholds, and query
scoring use ordered reference scores. Preserve the selection-version identity;
mode separation must not implicitly change its arithmetic or tolerances.

Report unsupported populations rather than invent parameters: empty B,
insufficient training C for sample SD, or insufficient eligible core. A usable
fit with one B observation can have empirical AUROC but unavailable B sample SD
and R/D. Nonfinite inputs/overflow are errors, not dimensions to drop. Preserve
existing detailed statuses, metric zero-denominator rules, and regression cases
from the historical implementation.

## Default production outputs

A successful default target requires exactly one final fit (or explicit
unsupported-final record) and its SPEC-0005 C/B distributions/status. No fold fit,
fold summary, or stability statistic is a prerequisite for completion or cleanup.

The production scientific bundle contains:

```text
manifest.json
dimension_final.tsv.gz
cluster_metrics.tsv
clusters/<cluster_id>/
    final.json
    selection_trace.json.gz
```

`dimension_final.tsv.gz` retains the existing final-profile column contract:
cluster/dimension IDs, C/B counts, means/SDs, effective width, R/D, individual
training AUROC/counts, ranks, core/candidate/selected flags and status. It has
640 rows per target, including unavailable values for unsupported targets.

`cluster_metrics.tsv` retains its long-table columns. Default rows use
`fit_id=final`, `population=training`: AUROC, precision, recall, specificity,
accuracy, balanced accuracy, F1, FPR, average precision, support counts and
availability reasons. Preserve the full confusion counts in final metadata.
The final signature has no held-out validation metrics; declare them not
applicable. Do not fabricate fold means, SD=0, selection frequencies of 1, OOF
rows, or validation scores from final training data.

Production metadata declares `evaluation_mode=all-C-in-sample`,
`research_role=separate-optional-artifact`, and `probability_calibrated=false`.
Whether this invocation requested research belongs only in invocation logs and
returned result metadata, not even in the production manifest execution block;
it must not change production artifact bytes or identity. Historical optional
research references may be shown separately; their existence never relabels final-fit metrics as held-out.

Keep a versioned production schema and unambiguous required-file inventory.
Support old complete six-fit library loads through their legacy schema. New
loaders must not require fold files in a final-only production bundle. Selected
signature dimensions/parameters, trace binding, thresholds, model provenance,
checksums, unsupported cases, and immutable scoring protections remain mandatory.

## Optional five-fold research

Publish research independently of production, with its own ID, manifest,
checkpoint namespace, selected scope and selector/reference. It contains only
five fold fits; the unchanged production final signature is linked as context,
not treated as a sixth observation. Required root:

```text
research/<research-id>/
    manifest.json
    dimension_folds.tsv.gz
    dimension_summary.tsv.gz
    cluster_metrics.tsv
    cluster_summary.tsv
    out_of_fold_positive_scores.tsv.gz
    clusters/<cluster_id>/
        fold_0.json ... fold_4.json
        selection_trace.json.gz
```

Use the existing fold-profile, summary, long-metric and OOF column contracts.
Store all five raw per-dimension R/D/AUROC/parameter values; aggregate mean,
sample SD and valid counts; record selection frequency and signature size
variation. Keep training and validation populations separate. Each positive
window has one OOF row with fold, exact signature digest, score/acceptance or
explicit unsupported status. Different folds define different score functions;
do not pool their scores into final C/B reference distributions.

Each research fit selects dimensions, Gaussian parameters and threshold using
only its four training folds and B, then evaluates the held-out C fold with
that unchanged model/threshold. Empty validation or unsupported fit cases remain
explicit. Five overlapping rotations are not independent experiments; reused B
and related windows limit the meaning of internal validation.

Default execution must not create these files. Requesting research later may
reuse a compatible production library covering the selected targets, including
a larger full library, without replacing its selector with a scoped build.
Research completion updates its own selector; it never overwrites the selected
production signature or its C/B distributions. A research failure exits nonzero
with a clear message, preserving an already published production library as
usable while leaving research resumable. CV success is not implied by production
success. Reuse existing final signatures; do not refit them after CV completes.

## Identities, migration, and existing work

Separate three dependencies:

- A per-target final-fit identity binds fitting algorithm/numerical versions,
  representation, exact C/B members and row mappings, embedding values, epsilon,
  floor and ordering. It excludes CV request, research scope, and fold partition.
- A production bundle identity binds selected target inventory, final fit
  identities and its versioned output contract. Output scope remains explicit.
- A research identity additionally binds the source production library ID,
  root-relative path and manifest SHA-256, research target scope, five-fold
  assignments, fold-fit algorithm and result contract. It cannot change the
  already fixed production parameters or invalidate relative-score data.

Changing only `cross_validate` must never invalidate a compatible final fit.
Scope changes may require a new aggregate bundle, but may reuse matching
per-target final results. Distribution dependencies are described in SPEC-0005.

Existing legacy library IDs and complete receipts include global scope/fold/six-
fit information. Do not merely rename them or assume a new hash is compatible.
Implement explicit validated import/reuse:

1. Locate complete published legacy libraries and private legacy final receipts
   for requested targets under the selected output root, using guarded reads.
2. Verify content hashes, task/target/library/trace bindings, profile coverage,
   representation, original matrix/C/B identities, fitting parameters and exact
   supported selection/numerical versions. A final can be reusable even if its
   surrounding cluster has not completed every legacy fold; validate the final
   receipt on its own rather than requiring a six-fit `target.json`.
3. Match normalized final-fit dependencies to the requested all-C fit. Different
   data, B pools, numerical/selection rules or unverified receipts are not reuse.
4. Copy the exact scientific model/results into a new verified checkpoint or
   publication with explicit source-path/source-digest provenance. New schema or
   library-ID bookkeeping may change serialized wrappers, but selected dimensions,
   numerical Gaussian values/weights, threshold and metric/trace values must not
   change. Retain an identity for this preserved scientific payload and prove
   reference query scores are bitwise unchanged.
5. Keep source bundles immutable and leave unrelated checkpoints alone. If more
   than one verified candidate claims the same normalized identity but different
   scientific payloads, raise a conflict rather than choose arbitrarily.
6. Reuse compatible legacy fold fits only when research is requested, validating
   each fold's exact training/validation dependencies. Skip unneeded fold work.

No discovery, threshold refitting, or PLM inference may run on a successfully
imported final fit. Corrupt published sources are explicit errors, not repaired
in place; malformed private receipts follow quarantine/recompute behavior.
Do not reuse old best-improvement fits as first-improvement fits or force changes
solely to bypass validation. Retain source evidence for every migration decision.

## Execution, resume, cleanup and publication

Use rolling bounded target jobs, lazy C/B descriptor expansion, memory mapping,
CPU float64 scoring, one numerical inner thread, and the admitted shared worker
budget. Preserve deterministic artifact ordering independently of completion.

Default target states are `final` then `relative_scores` then complete. Research
states are five independently resumable fits followed by research aggregation.
Run/publish the production phase first; optional research must not delay initial
production availability or roll it back on a later research failure. Parallel
or scoped research remains bounded by the same resource controls.

Preserve profile blocks, prefix blocks and per-trial selection checkpoints for
unfinished fits. Target completion receipts explicitly list required fit IDs:
`[final]` for production and five fold IDs for research. Do not hardcode six fits
in orchestration, loaders, merger counts, completion logging, or cleanup.

After a production cluster's final result and distributions/status are durable,
reclaim its disposable Gaussian/prefix caches. Retain compact results/profiles,
source provenance and distribution arrays. For research, fit-level cleanup after
a durable complete fold receipt bounds cache use; it need not wait for all five
folds. That receipt must independently contain everything needed to resume and
assemble reports. Update the cleanup validator to distinguish legacy six-fit,
new single-final, and research fit contracts before deleting anything. Keep the
legacy cleanup API safe and usable; never delete unfinished or inconsistent work.

Default active Gaussian storage is roughly one fit per worker, approximately
8–9 GiB for eight workers on the current dataset, plus prefix scratch, retained
relative-score arrays, compact results, migration copies and publication staging.
These are estimates, not an RSS or total-disk guarantee. Do not resurrect the
previous all-cluster Gaussian-cache accumulation.

Maintain input/output transaction guards, per-target ownership, symlink checks,
atomic fsync/rename receipts, manifest inventories and rollback-aware selectors.
Production and companion distributions publish as a verified pair under
SPEC-0005; research publishes separately with an explicit domain. Immutable
installed reads create no files. Readers must not guess a lock root from a
legacy directory depth. Completed scientific/relative artifacts stay usable if
an unrelated research phase fails.

Logs distinguish `final`, `relative_scores`, and research `fold_0`–`fold_4`, with
cluster IDs, throttled progress, and separate production/research completion
counts. Retain the current concise `computing signature` wording. Report
verified reuse/import and unsupported states accurately; do not call unrequested
research "completed" or "failed".

## Concrete schema and identity contracts

These details close the implementation choices left open by the initial revision.
Use the repository's canonical JSON/SHA-256 helpers throughout, serialize finite
float64 values round-trip, preserve deterministic gzip headers, and reject
symlink/escaping paths. Optional metadata must not override required fields.

### Production, fit and research manifests

- Production manifest schema: `pclass-dark.library.v2`. Keep `id`,
  `identity_inputs`, `status=complete`, `representation_contract`, `counts`,
  `files`, software/execution provenance. `id=identity(identity_inputs)`.
  Identity inputs contain contract version and sorted `[cluster_id, final_fit_id]`
  pairs; final fit IDs transitively bind verified scientific inputs/engine.
- Signature schema: `pclass-dark.signature.v2`. Retain every v1 signature field
  and add `fit_identity`, `scientific_payload_id`, `evaluation_mode`, and optional
  source provenance. `library_id` binds the containing production/research bundle,
  not the legacy source. Loaders dispatch explicitly on v1/v2.
- Production `counts`: `targets`, `available_final`, `unsupported_final`, and
  `dimension_final_rows=640*targets`. No implied discovery/OOF counts. Final
  training metric rows enumerate the current `METRICS` keys, including tp/fp/fn/tn
  and positive prevalence. Existing per-row support/status and signature-level
  undefined-denominator reasons remain unchanged.
- `selection_trace.json.gz` is a list of exactly one `final` trace in production,
  or exactly five ordered fold traces in research; verify each signature's trace
  digest and reject extra/missing traces.
- Research manifest schema: `pclass-dark.library-research.v1`, with the same
  canonical identity/status/files/provenance conventions. Counts include targets,
  valid/unsupported fold fits, dimension rows (`5*640*targets`) and OOF rows
  (one per target C window). Bind sorted requested targets, each target's final
  fit ID, exact fold membership digests, fold fit IDs and research contract.
  Bind exact production manifest digest separately as provenance: a larger
  compatible production scope need not alter the actual research fits.
- Research signatures use v2 and bind the research ID as `library_id`; their
  fit identities include exact training/validation memberships and fold ID.
  `cluster_summary.tsv` includes `population=discovery, metric=signature_size`
  and its five-fit mean/sample SD/valid count, as in the existing reports.

For means/sample SD, use finite available values only: no values gives null/null
and count zero, one gives mean/null and count one. Selection frequency divides
by valid discovery fits; zero valid fits gives null, not zero. These quantities
exist only in research artifacts. Use the full legacy header definitions for
unchanged tables; define separate production/research inventories instead of one
unconditional global `HEADERS` map.

### Final fit identity and preserved payload

A final-fit identity hashes a versioned canonical dictionary containing cluster
ID, selection/threshold/scoring algorithm versions, epsilon/floor, numerical
engine/NumPy/SciPy/architecture, representation ID, matrix and row-map hashes,
canonical window order, and exact target C/B window-ID digests/counts. Source
prepared IDs/manifests are verified provenance, not a hidden fold-partition input.
Membership digests are over the existing canonical ascending window-ID lists,
not sorted score arrays. New fit receipts store their full normalized identity
inputs, not only a digest whose preimage would be unavailable on resume.

`scientific_payload_id` hashes the canonical dictionary `{signature, profiles,
trace, oof}` of the existing fit result, with only these top-level signature
wrapper fields removed before hashing: `schema`, `library_id`, `input_identity`,
`fit_identity`, `scientific_payload_id`, `evaluation_mode`, and `source_provenance`.
Keep all other scientific fields, status, counts, parameters, ranks, evidence,
metrics, threshold, trace and OOF values. New writers use `source_provenance` for
source references; do not introduce extra unnormalized wrapper fields. This
allows an explicit v1-to-v2 rebind without concealing scientific changes.

Store final fits under `.fits/<fit-id>/final/`, including normalized identity
metadata and existing stage receipts; production target receipts under
`.production-checkpoints/<production-id>/c_<hex>/target.json` reference that fit
and its payload digest. Research uses `.research-checkpoints/<research-id>/c_<hex>/`
with five fold directories and its own target receipt. Legacy `.checkpoints/`
remains a read/import source. Adding research never archives or prunes `.fits/`.

### Proof available for private legacy sources

Current private legacy receipts have `task_id`, `dependency` and payload hashes,
but no full library identity preimage, epsilon/floor or engine metadata. The
signature's `library_id` alone cannot establish these settings. For unpublished
sources, verify all available bindings and the complete-receipt dependency
`identity([task_id, SELECTION_VERSION])`, and reconstruct the supported legacy
library identity input dictionary against verified setup inputs, numerical
versions/settings and a provable target scope. A source root ID must match that
reconstructed hash. The currently selected full setup scope and the exact
requested scope are supported candidate reconstructions; do not assume the set
of partially created checkpoint directories was the original requested scope.

Recognize the actual legacy identity spelling: the first-improvement implementation
retained the string `top-two-prefix-union-gt90-backward-largest-strict-v1` in its
`selection_rule` field, while its `algorithm`, engine, and receipt dependency
version identify first-improvement. Validate the complete known recipe, not that
misleading field alone. Published legacy manifests supply their actual recipe.
A private source whose recipe/scope/engine cannot be proved is logged as not
reusable and preserved; compute that requested final anew rather than guessing
or rejecting all other reusable sources. Emit reasoned reused/imported/computed
counts. Unknown or old best-improvement algorithm recipes are never accepted.

### Selectors, force and failure boundaries

SPEC-0005 defines the paired production selector. Research selects independently
using `research-selection.json` with schema
`pclass-dark.library-research-selection.v1`, fields `research_id`, `research`
(root-relative path), `manifest_sha256`, `production_library_id`, and
`production_manifest_sha256`. A legacy selector is never silently interpreted as
a research selector. Missing research is not a production load failure.

`--force` bypasses reuse only for requested target work: production final fits
and distributions, plus the five folds if `cross_validate=True`. Archive matching
state under explicit domain locks without deleting unrelated identities. A
research-only normal rerun reuses compatible production; force explicitly requests
its target rebuild. Shared final fit state is archived only for explicitly forced
fit identities; already published bundles stay readable until their protected
replacement. Do not force merely to add missing distributions or research.

A default `--cluster` build still selects its scoped production pair, preserving
existing scoped-build semantics. For `--cross-validate --cluster`, first reuse a
verified selected production superset if every requested target/dependency matches;
leave that production selection unchanged and publish only scoped research.
If none exists, build/select the scoped production pair before research. A paired
selector lacking distributions is not a complete reusable production pair; finish
its scoring backfill before entering research.

Cleanup gates are explicit, read-only validated inventories, not directory counts:
legacy requires all six complete fits plus its target; new production requires
one final fit plus the referenced committed distribution receipt/status; research
requires the complete receipt of the individual fold being reclaimed. Never
change these gates merely because removable files are present. Malformed/missing
distribution receipts preserve production arrays for recovery, even if the final
fit is complete. Keep compact receipt/source evidence after cleanup.

Use publication domains `library`, `relative_scores` and `research` consistently
for writer, reader and rollback locks, with explicit output roots passed to nested
bundle readers. `load_library` retains non-scientific source-root/manifest-digest
metadata alongside detached parameters, so paired readers receive an explicit
owner root; this is excluded from model identity.
Production/relative pair publication has one root-owned rollback
boundary. Research begins a new transaction after that pair is committed. A
research failure never rolls back a selected production pair.

Release the production output-root guard before lengthy research computation;
otherwise guarded query readers cannot actually use the published pair until CV
finishes. Own research computation with its own run/target locks and a verified
input snapshot/read guard, not the production output-root transaction guard.
For research publication, release phase-specific guards, reacquire the standard
input/output guards in canonical order, and revalidate input/production bindings
before committing. If they changed, report a resumable integrity conflict instead
of mixing generations. Hold the production root only for brief verification/
publication work in this phase. Test production loading/scoring while a research
worker is deliberately blocked; no reader may have to wait for that fold. Tests
must also cover source replacement between compute and publication.

## Acceptance tests and implementation order

1. Extend orchestration and schemas with final-only production and optional
   research. Default tests replace fold discovery with a fail-fast sentinel;
   prove one fit per target, no fold reports and correct unsupported handling.
2. Implement stable final-fit identity and legacy imports. Exercise published
   and private final-only sources, partially completed old clusters, scoped/full
   compatibility, corruption/conflicts, provenance and version rejection. Replace
   discovery/threshold fit functions with sentinels to prove migration is reuse.
3. Implement SPEC-0005 distributions/backfill using those preserved parameters.
   Assert existing model payloads/query scores remain unchanged after migration.
4. Add optional research: exactly five folds only when requested; no repeat
   final fit; unchanged production signature/distributions/selection; proper
   held-out isolation, production-reader availability while CV runs, and explicit
   research failure/recovery behavior.
5. Update report/loader/cleanup contracts: no six-fit assumptions, no fake
   stability metrics, idempotent correct-mode cleanup, retained distribution
   arrays, interrupted writes, safe publication and selector concurrency.
6. Run the numerical/first-improvement, checkpoint and publication regression
   suites plus serial/parallel byte consistency and rolling scheduling tests.
   Validate final-only and research package loads/scoring outside the checkout
   without PLM imports or writes to installed resources.
7. Record a small actual-data final-only build/backfill: reused versus computed
   fits, target counts, output sizes, elapsed time and cache cleanup. Run optional
   CV on a small scoped target and show production is unchanged. Do not claim an
   unmeasured whole-library speedup or launch a full research sweep implicitly.

Update code, CLI/API help, packaging/LFS rules, README, index and progress records
in the implementation change. Historical 102-test and benchmark results in earlier records apply to the legacy
implementation; new verification is recorded below.

## Boundaries

Default output is in-sample discovery evidence plus empirical C/B comparisons,
not independent validation, a calibrated probability, a forced cluster decision,
or proof of recovery beyond nearest neighbors. Optional research supplies
stability and internal held-out-positive metrics, not production parameters.
Query-window aggregation, a combined C/B ranking rule, deployment priors and
probability calibration remain separate scientific decisions. No further decision
is needed to implement the approved mode change and the explicit SPEC-0005
empirical distributions.

## Readiness review — 2026-10-27

Reviewed against the current orchestrator, private receipts, runtime loader,
publication helper and packaging configuration. Closed technical gaps in concrete
schemas, research-independent production metadata, signature-payload migration,
private-source proof requirements, typed cleanup gates, force/scope behavior,
read guards and failure semantics. SPEC-0005 fixes paired publication, array
bindings and query-evidence contracts. The review established readiness before implementation; no scientific decision
was needed. The subsequent implementation/verification record below supersedes
the earlier documentation-only status.

## Implementation and verification — 2026-10-27

Implemented the final-only default, paired C/B distributions and optional research.
`library_build.py` orchestrates phases; `fit_records.py` normalizes scientific
payloads and validates profiles/parameters/ranks; `library_migration.py` proves
legacy identities/imports; `library_reports.py` writes/loads separate v2 production
and research reports. The legacy builder is retained privately for migration and
regression fixtures, not as the default CLI path.

Verified complete legacy finals import from published bundles or private receipts
without a six-fit target marker. New production sources also support scoring-only
backfill when private fits are missing. Published source files are unchanged;
malformed private sources are archived for recovery instead of silently trusted.
Unknown legacy identity recipes remain untouched and are logged as non-reusable.
A selected complete v2 superset is reused for scoped research; otherwise normal
scoped production/import semantics apply rather than implicitly building an
entire full-library companion just for one research target.

Per-fit private results reuse the existing checkpoint payload schema internally;
versioned signature v2 wrappers and preserved payload IDs are applied at report
publication. Research source-production binding is provenance. Production
metadata explicitly declares in-sample evaluation and uncalibrated status.
Durable staged file trees are fsynced before manifests and selectors commit.

Validation: full suite **127 passed, 2 opt-in real-checkpoint tests skipped** in
239.88 s, with one existing Torch environment warning. Subsequent strengthened
fit/metadata checks: **34 focused tests passed**, then **3 publication/rollback
checks passed** after staged-tree fsync changes and **2 default/research tests
passed** after additive production-role metadata. Ruff lint/format, local doc
links/fences and whitespace checks passed. No skipped checks are counted as
executed real-checkpoint inference.

A real-data isolated run imported six cluster_0001 fits without discovery,
generated distributions for 20 C/213,241 B windows, and finished in 20.97 s.
Migrated query scores/acceptance matched the original bitwise for 32 vectors;
no Gaussian caches remained. [Record and timing limits](../research/notes/2026-10-27-final-only-backfill.md).
Wheel/source builds and installed production/relative/research loading and query
evidence were verified outside the checkout without PLM imports, setup-data
matrices or package writes. No all-cluster production build was launched.

## Shared path and distribution update — 2026-10-27

[SPEC-0006](0006-shared-artifacts.md) supersedes earlier cwd-relative defaults.
CLI and Python omitted paths resolve to `<repo>/data/setup-data` and
`<repo>/data/library` in a checkout, regardless of cwd. Explicit overrides remain
caller-controlled. Installed packages read selected bundled data or existing
user-cache data and write new outputs only to the user cache. Package builds
export selected complete shared bundles through validated build hooks; scientific
identities, cache safety and historical experiment paths are unchanged.

### Legacy migration with empty validation folds

A valid legacy fold with zero validation members has no rows in the OOF table.
When migrating published research, represent this as an empty OOF list, verified
against the fold signature's validation count and the current validation IDs.
Missing OOF rows for a nonempty validation fold remain an integrity error.
Preserve the source bundle and import both populated and empty folds without
recomputing signatures. Regression coverage uses four proteins in two clusters
(`min_clusters=1`), removes private checkpoints, and imports published folds.

Empty-fold migration validation: the regression passed independently, followed by
**33 passing production/legacy library tests**. Lint, formatting and whitespace
checks passed.

CLI help accepts both `pclass-dark build-library -h` and
`pclass-dark build-library --help`, with identical output and exit status zero.

Published cluster directories now use validated cluster IDs; readers retain legacy
hex-layout compatibility. See [SPEC-0011](0011-readable-cluster-paths.md) for
layout metadata, migration, checksum rebinding and backup contracts. Internal
checkpoint directory names remain unchanged.

## CLI simplification (2026-10-30)

Remove CLI options data-dir, output-dir, memory-mib, epsilon, sigma-floor and
dry-run. Resolve paths through the existing default resolvers (None), use internal
1024 MiB / 1e-8 / 1e-8 numerical defaults, and execute clean directly with no CLI
preview mode. Keep workers, cluster, cross-validate, force and clean exposed.
Python build_library configuration and plan_cleanup preview remain available.
Validate removed flags are rejected, help omits them and defaults are forwarded;
update cleanup CLI integration, documentation and progress records.

Reduced CLI validation: CLI and cleanup suites passed **85 tests** (108.85 s),
including removed-option rejection, help, default forwarding and default-root
cleanup integration. Ruff and Git whitespace checks passed.
