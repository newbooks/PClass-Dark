# SPEC-0005: Final-signature background and cluster score distributions

- Date: 2026-10-27.
- Status: Implemented on 2026-10-27 with final-only production and optional
  research; verification recorded below.
- Extends: [SPEC-0004 build-library](0004-build-library.md).
- Inputs: Frozen final signatures and the verified C/B memberships and embeddings
  defined in [SPEC-0003](0003-setup-data.md).
- Motivation: [Score interpretation and decision record](../research/notes/2026-10-27-relative-score-reasoning.md).

## Purpose

Extend `pclass-dark build-library` to retain the empirical distributions of each
fixed final signature's composite scores on its cluster members (C) and background
(B). These distributions let downstream code interpret a query against both
populations, using the same signature that scored the query.

The immediate objective is useful cluster comparison, without repeating signature
selection or claiming calibrated membership probabilities. The revised SPEC-0004
makes one all-C fit the default and five rotations optional research. This stage
uses the production final signature in either mode. It does not change fitting
parameters, background selection, thresholds or the preserved scientific model
payload; schema/bundle identities may change under validated legacy migration.

## Statistical interpretation

For cluster k, let `L_k(x) = log(S_k(x))`, where S is the existing equal-weight
arithmetic mean of selected dimensions' Gaussian compatibility values. Use the
existing ordered float64 scoring implementation and residue-only embedding
contract. This is log of an arithmetic mean, not mean of log kernels.

After the final signature is frozen, compute two arrays:

- `C_log_scores`: L for every member of the original final-fit C pool.
- `B_log_scores`: L for every member of the original fixed B pool.

Use the existing window observations and exact target-specific memberships.
Do not substitute training-fold C, fold-specific scores, other clusters' scores,
all reference proteins, or a new sample of B. Preserve multiplicity of distinct
window IDs having identical vectors/scores. Repeated membership across target
clusters is allowed under the existing data contract.

B directly answers how unusual a match is among sampled nonmembers. C directly
answers where the query falls among known members' compatibility scores. C also
influences B scores indirectly through the fitted signature. C is not an extra
factor already incorporated into the B percentile.

Both are descriptive distributions: final C was used to fit/select the signature,
and B was used to select it. They are not independent calibration populations.
Their empirical tail fractions are not posterior membership probabilities, formal
held-out p-values, or false-discovery guarantees. The capped B mixture also does
not establish deployment class frequencies. Report these limitations in metadata
and API documentation without making users choose additional scientific options.

## Query evidence and tie conventions

Store sorted arrays in ascending log-score order. For a finite query log score l,
use integer binary searches with exact float64 comparisons:

```text
B_left  = searchsorted(B, l, side="left")
B_right = searchsorted(B, l, side="right")
C_right = searchsorted(C, l, side="right")

background_n_ge          = n_B - B_left
background_tail_fraction = (1 + background_n_ge) / (n_B + 1)
background_percentile_le = B_right / n_B
background_evidence      = -log10(background_tail_fraction)
cluster_percentile_le    = C_right / n_C
```

The plus-one convention prevents zero background tail fractions and infinite
evidence above the observed maximum. It does not confer statistical validity on
a distribution fitted with reused B. Count all score ties; do not use midranks,
interpolation, density estimation, or floating-point tie tolerances. The inclusive
B percentile and upper-tail fraction are not complements when ties occur.

Higher background evidence means a stronger match relative to that cluster's B.
A low C percentile means a lower score than most known members; a high percentile
means the score exceeds many known members, not a density-based typicality
measure or guaranteed assignment. A score above every observed C value has
percentile one even outside the observed range. Do not multiply the two
percentiles, apply softmax across raw scores, or invent a combined probability.

Return population sizes alongside fractions because finite B limits tail
resolution. At approximately 213,000 B windows, the smallest reported fraction
is about 4.7e-6. Queries above several clusters' maxima can consequently tie;
retain the tie rather than claim an ordering unsupported by the stored data.

Existing five-fold OOF-positive scores remain separate stability/internal
validation diagnostics. Do not pool them with the final C distribution or compare
a final-signature query to them as if the scoring functions were identical.

## Placement in build-library

For each target:

1. Compute, resume, or import the all-C production final signature under the
   revised SPEC-0004. No five-fold work is required for this stage.
2. Durably save and verify the final scientific fit, source provenance, and the
   production target state whose required-fit inventory is `[final]`; it is not
   complete until a distribution receipt or unsupported status is committed.
3. For a usable final signature, score its exact final C and fixed B windows in
   bounded row blocks; sort and commit the two distributions with a receipt.
4. Commit the complete production target receipt referencing its verified final
   fit and distribution receipt/status. The receipt must contain the exact
   required-fit inventory, normalized dependencies and referenced content hashes.
5. Reclaim disposable Gaussian/prefix caches using the correct-mode cleanup,
   then report production target completion. Distribution arrays/receipts must
   survive cleanup. If arrays were removed by legacy cleanup, recover scores
   directly from embeddings, without regenerating Gaussian caches. Rolling
   workers take the next cluster without introducing a batch barrier.
6. Publish the production/relative pair before optional research. A later
   `--cross-validate` phase reuses these distributions; five-fold completion is
   neither a prerequisite nor grounds to overwrite or recompute them.

If the final signature is unsupported, emit its status and reason with zero
stored score arrays. Do not substitute a fold signature or silently omit the
cluster. An otherwise usable final signature with missing/empty required pools
is an integrity error, not a successful distribution stage.

## Compatibility and backfill without rediscovery

Distribution output is an independently checksummed companion artifact, produced
within `build-library`. Bind it to stable final-model payload identities and exact
serialized signatures in the selected production bundle. Adding research must
not change these dependencies. SPEC-0004 permits versioned production schemas
and new bundle IDs while importing old final results without discovery: preserve
legacy source files byte-for-byte and prove the migrated scientific payload and
query scores unchanged, even if new wrappers rebind library metadata.

On a normal rerun:

- If scientific discovery is still incomplete, reuse intact matching final fits
  through validated legacy import or current final-fit resume. Backfill missing
  distributions when each target is admitted; do not restart a completed search.
- If the scientific library is already published, verify it and run only missing
  distribution work from its final signatures and matching setup-data inputs.
  Keep legacy research reports as source evidence; never call ranking, prefix
  screening, backward removal, threshold fitting, or PLM inference for this path.
- A complete legacy final-fit receipt can be imported without a published library
  or six-fit `target.json`. Verify its own content/dependencies and provenance,
  then score distributions. Do not wait for unrequested legacy folds.
- `--force` retains its existing intentional-discovery-rebuild meaning. Backfill
  must not require `--force` or modification of a complete scientific bundle.

The current completed-library early-return path must check companion completion
before declaring the extended operation complete. Existing raw-score-only
libraries remain loadable; missing relative scores must raise an actionable
availability error if requested, rather than trigger implicit computation.

## Artifacts, identities, and publication

Required layout, relative to the library output root:

```text
<production-library-id>/                # Versioned scientific bundle per SPEC-0004
.relative-checkpoints/<final-fit-id>/
    complete.C.npy
    complete.B.npy
    complete.json
relative-scores/<distribution-id>/
    manifest.json
    summary.tsv
    clusters/c_<hex>/
        distribution.json
        C_log_scores.npy
        B_log_scores.npy
selection.json                         # Selects verified scientific/relative pair
```

Arrays are one-dimensional, ascending, finite little-endian float64 (`<f8`),
NumPy format with `allow_pickle=False`. Store no object arrays. Sorting loses
window order intentionally: the declared source membership identity and counts
bind each multiset. These files support percentiles, not per-accession lookup.

Each cluster's `distribution.json` declares:

- Schema `pclass-dark.relative-score-cluster.v1`, cluster ID and final status.
- Production-library ID, preserved final scientific-payload identity, SHA-256
  of exact serialized `final.json`, and the
  representation-contract identity.
- Prepared/embedding IDs, verified embedding matrix digest, and canonical C/B
  membership digests/counts using setup-data's existing conventions.
- `fit_id: final`; `score_space: log-arithmetic-gaussian-compatibility`;
  `score_rule: ordered-float64-v1`; query-evidence formula version.
- `C_role: final-fit-in-sample`; `B_role: fixed-discovery-background-reused`;
  `probability_calibrated: false`.
- Relative file names, array shapes/dtypes, byte sizes and SHA-256 digests; summary
  statistics count/min/max for available arrays. Unsupported cases have a reason,
  no array paths, and null numerical summaries.

The companion manifest uses `pclass-dark.relative-score-library.v1` and includes
canonical identity inputs, completion status, target/usable/unsupported counts,
verified scientific manifest-byte digest, and a complete file inventory. Its ID
hashes the distribution contract, scientific manifest digest, exact final
signature digests, C/B/matrix identities, output scope, and numerical engine.
Execution timing, worker count, and workspace budget are provenance, not new
scientific selection settings. Fixed CPU scoring and deterministic target ordering
must give identical array, cluster-metadata and summary bytes across worker
counts for a fixed production source. Force rebuilding a production manifest
with different execution provenance may change its byte digest and thus the
companion ID; that does not justify changing scores or repeating discovery.

`summary.tsv` has one row per target: cluster ID, status, reason, C/B counts,
C/B score minima/maxima, final-signature SHA-256, and relative metadata path.
Do not persist rounded quantiles in place of full arrays. Add explicit schema
constants and validators rather than treating arbitrary dictionaries as trusted.

Publish the companion through a staging directory with array/metadata writes
flushed and fsynced, checksums verified, and manifest committed last. Extend the
publication helper with an explicit companion domain where necessary; retain
output-root ownership through selector publication and rollback. Never overwrite
a complete scientific bundle to insert these files.

Add a versioned selector schema preserving existing scientific path/ID/digest
fields and adding companion path/ID/manifest digest. Readers support legacy
selectors without the companion. A selected v2 pair is valid only if both
manifests and their bindings verify. On failure retain the previous selection;
newly created complete artifacts may remain unselected or be rolled back under
the existing guarded transaction rules. Do not leave a selector pointing at an
incomplete companion. Acquire mutable input/output guards in the existing sorted
order; companion readers use the actual output root explicitly, not a guessed
parent-depth inherited from scientific bundle readers.

## Checkpoint and cleanup behavior

Distribution receipts are distinct from scientific `complete.json` receipts.
Do not amend model parameters, traces, or source scientific receipts to add
this stage. Versioned migration wrappers/provenance follow SPEC-0004; distribution
backfill itself does not refit or mutate scientific results. Bind private
distribution receipts to the stable final-fit/scientific-payload identity, exact pools,
embedding matrix, representation, and distribution algorithm version. Arrays are
committed before their receipt; validate identities, counts, dtype, finite values,
sorting, and file hashes before reuse. Quarantine malformed private receipts or
bad arrays and regenerate that target's distributions only.

Initial implementation may repeat the current target's distribution scoring
if interrupted before the two arrays/receipt are committed. At the observed
population sizes this is bounded scoring work, not repeated discovery. Reuse
all completed targets. Interrupted cleanup or companion publication must not
invalidate scientific complete fits or force Gaussian caches to be recreated.
Never delete distributions through the Gaussian/prefix cleanup API. Copy or
otherwise safely publish verified arrays without mutable aliasing between private
checkpoints and a force-replaceable published generation.

## Python interfaces and installed use

Required interfaces:

```python
from pclass_dark.library import build_library, load_library
from pclass_dark.relative_scores import load_relative_scores

result = build_library()  # Shared defaults from SPEC-0006.
library = load_library(result.library_dir)
relative = load_relative_scores(result.relative_dir, library=library)
log_scores, accepted = library.score(cluster_id, query_vectors,
                                     embedding_contract=query_contract)
evidence = relative.evaluate(cluster_id, log_scores)
```

Extend `LibraryResult` compatibly with defaulted `relative_id`/`relative_dir`
fields alongside the optional research fields in SPEC-0004. `reused` is true
only if all requested artifacts, including research when requested, were already
verified complete. Expose discovery reuse separately in execution
metadata so scoring-only backfill is not described as rediscovery.
Keep existing `Library.score()` return values unchanged. Provide a
reusable computation API accepting a frozen signature, verified matrix and row
indices; CLI adapters must not contain scoring mathematics. A selector-loading
helper should resolve the scientific/companion pair for downstream applications.

`evaluate` returns arrays of background tail fractions, background inclusive
percentiles, background evidence, and C inclusive percentiles, with source counts
and evidence-role metadata. Accept finite one-dimensional log scores only;
validate the representation and exact final-signature binding at load time.
The caller supplies scores from that bound signature; arbitrary numerical score
inputs do not carry independently verifiable provenance.
Raise `UnavailableDistributionError` (a `ValueError` subclass) for unsupported/
missing targets; never manufacture neutral evidence. Empty queries return empty
float64 evidence arrays with valid population sizes/status metadata. Preserve
query order. Reject integer, boolean, complex, multidimensional and nonfinite
query inputs; accept floating 1-D arrays and convert to float64 explicitly.

Do not load all clusters' B arrays into RAM. Load distributions per requested
cluster with a bounded cache, holding the correct read guard while verifying and
copying arrays into read-only snapshots. Use an LRU cache limited to 16 clusters
and 64 MiB of array storage, evicting to satisfy both bounds; an oversized
cluster is evaluated without retaining it in the cache. Immutable installed resource reads
must create no lock/cache files. Plain NumPy scoring requires no PLM stack,
model downloads, or regeneration of embeddings.

## Concrete bindings, selector and API result contract

The cluster metadata JSON has these required keys: `schema`, `cluster_id`,
`status`, `reason`, `production_library_id`, `final_fit_id`,
`scientific_payload_id`, `final_signature_sha256`, `representation_contract_id`,
`prepared_id`, `embedding_id`, `matrix_sha256`, `C_ids_sha256`, `B_ids_sha256`,
`C_n`, `B_n`, `fit_id`, `score_space`, `score_rule`, `evidence_rule`, `C_role`,
`B_role`, `probability_calibrated`, `arrays`. Use the previously declared literal
roles/rules; `evidence_rule=empirical-inclusive-plus-one-B-v1`.
For each available population `arrays.C`/`arrays.B` contains `file`, `shape`,
`dtype`, `size_bytes`, `sha256`, `minimum`, `maximum`; unsupported targets use
`arrays={}` with actual pool counts and explicit reason. Do not manufacture
zero-length reference arrays or zero support for a populated unsupported pool.

`summary.tsv` header is exactly:

```text
cluster_id status reason C_n B_n C_min C_max B_min B_max final_signature_sha256 metadata
```

The displayed header names are tab-separated in the actual file. Use canonical
cluster order, finite numbers at 17 significant digits, empty TSV cells for null,
and the existing lowercase boolean and deterministic IO conventions.

The companion manifest keys are `schema`, `id`, `identity_inputs`, `status`,
`counts`, `files`, `source_production`, `software_version`, `execution`.
`source_production` carries production ID/root-relative path/manifest SHA-256;
`files` covers exactly summary, all cluster metadata and available arrays.
`id=identity(identity_inputs)`, with the distribution/evidence/score versions,
production ID/manifest SHA, sorted target inventory, final-signature/payload IDs,
representation, exact pools/matrix identity and engine. Counts are targets,
available, unsupported, total stored C values and total stored B values, with
no fold counts. Totals count available arrays only; unsupported metadata still
records its true input-pool sizes.

Freeze default row blocks at 4096 and CPU ordered-float64 scoring. Chunking is
an execution detail only while per-row arithmetic is unchanged. Private receipts
bind final-fit and scientific payload identities, pools/matrix and scoring rule,
not destination library IDs or signature wrapper digests. This allows verified
array reuse when legacy wrappers are migrated or output scope changes. When
publishing, regenerate destination cluster metadata binding the exact final JSON
and production manifest, without rescoring identical data. Do not trust private
arrays merely because their shape matches.

Paired selector schema: `pclass-dark.library-selection.v2`. Required fields:
`library_id`, `library`, `manifest_sha256`, `preparation_id`, `embedding_id`,
`relative_id`, `relative`, `relative_manifest_sha256`, plus `schema`. All artifact
paths are root-relative and checked. The relative manifest must name exactly
the selected production manifest digest, and cluster signature hashes must
match the corresponding scientific files. Legacy v1 selectors remain valid for
raw-score loading only. A v2 selector with missing relative fields is malformed.

Production and relative artifacts form a directed dependency, not a hash cycle:
final scientific results determine the production ID/manifest first; its exact
manifest determines the companion ID/manifest second; selector selects the pair
last. The production manifest does not hash the companion or research result.
Private distribution scoring may precede production publication, because its
receipt binds the stable scientific payload and pools, not the final manifest.
Use the explicit `relative_scores` transaction domain and actual library output
root for companion locking; legacy helper parent-depth inference is invalid.
Direct companion loading obtains the owner root/expected scientific manifest
digest from the bound library snapshot and verifies the required relative-scores
layout under that root. The selector helper passes its guarded root explicitly
to internal loaders. Immutable resource pairs need no write locks. A misplaced
mutable companion raises an actionable layout error, rather than guessing a
different transaction root.

`load_selected_library(output_root=None, *, immutable=False)` returns a frozen
`SelectedLibrary(library, relative, selection_schema)` after guarded pair
validation; `relative=None` is allowed only for a legacy v1 selection. Use
`load_relative_scores(relative_dir, *, library, immutable=False)` to load a
companion directly; verify all expected file hashes, counts, bindings, dtypes,
finite values and sortedness before first use, in bounded chunks/one target at a
time, without retaining every array. Reject extra manifest members, unexpected
cluster IDs, count mismatches and mutable aliases. Its runtime cache stores
read-only detached copies. Before every uncached mutable array load, reacquire
the explicit root read guard and verify expected manifest/file digests; if a
force replacement changed the generation, raise IntegrityError instead of mixing
new arrays with old parameters. Cached copies remain valid detached snapshots.
Legacy/default/research files must not be guessed
from arbitrary directory contents.

`relative.evaluate(cluster_id, log_scores)` returns a frozen `RelativeEvidence`
with read-only float64 arrays named `background_tail_fraction`,
`background_percentile_le`, `background_evidence`, `cluster_percentile_le`, and
read-only integer `background_n_ge`; scalar `C_n`, `B_n`; `status`, `C_role`,
`B_role`, `probability_calibrated=false`, and final-fit/payload IDs. Empty queries
keep these same fields. No output is named probability or confidence.

Serial/parallel determinism tests compare arrays, metadata, reports and preserved
scientific payload IDs, excluding manifest execution/provenance fields whose
worker settings intentionally differ. Research requested afterward must leave
the selected production/relative pair and all these bytes unchanged.

## Resource use and distribution

At about 213,000 B windows, one float64 B array is about 1.6 MiB. Across 3,443
clusters, B arrays occupy approximately 5.5 GiB before compression/format overhead;
C arrays are much smaller. These are retained output data, unlike the roughly
22 TiB of previously accumulating feature-level caches. Budget private arrays
plus staged/published copies during publication; do not claim 5.5 GiB is the peak
whole-run requirement. Old companion generations accumulate unless explicitly
removed; do not prune them silently.

Score only final selected dimensions in bounded blocks, reusing output buffers
and the ordered reference scoring kernel. No `(640, n_B)` Gaussian cache is
needed. Use the existing bounded rolling worker budget, one numerical inner
thread, and memory estimates that include sorting temporaries. CPU is the first
backend to preserve current scoring behavior; a future GPU backend requires
versioning and parity tests. Log target, population and completed/total windows
approximately every 30 seconds during long work and report stage completion.

For releases, ship both companion and scientific manifests, all required
arrays/metadata, and the paired selector under package resources. Add package
patterns for `resources/library/relative-scores/<id>/...` and private-checkpoint
exclusions. Existing `resources/library/**` Git LFS rules cover these large data.
Do not bundle model weights or require setup-data matrices to evaluate query
percentiles once the distributions have been installed.

## Tests and acceptance criteria

- Compare fixed-signature C/B arrays with independent hand-calculated Gaussian
  arithmetic means and sorted logs, including extreme values that underflow in
  ordinary score space. Never average log kernels.
- Hand-check binary-search counts, inclusive ties, all-equal arrays, queries
  below/above the observed ranges, finite-sample plus-one fractions, and differing
  C/B sizes. Prove B percentiles/tails are not accidentally treated as complements.
- Verify same data/window multiplicities as compact setup pools, no accidental
  fold restriction, no new B sample, correct row joins and metadata identities.
- Show that the full final-signature scoring path and the installed query scorer
  produce identical reference log scores for the same vectors/contract.
- Test serial/parallel byte consistency, bounded task/cache memory behavior,
  interrupted array/receipt writes, malformed receipts, stale signature/matrix
  digests, wrong shapes/dtypes/counts, unsorted/nonfinite arrays, and symlinks.
- Backfill a legacy published library with discovery/PLM/threshold-fit functions
  replaced by fail-fast sentinels. Prove source scientific files and the migrated
  model payload remain unchanged. Test partial
  scientific runs after Gaussian cleanup and imports without six-fit targets.
- Verify companion publication/selection failure leaves the old selected pair
  valid; readers coordinate with force rebuilds; completed corruption is an
  explicit error rather than silently repaired in place.
- Verify cleanup removes disposable Gaussian/prefix arrays while preserving
  distribution outputs and both populations' complete statistics.
- Build/install a release containing a small scientific/companion pair; load and
  evaluate outside the checkout with PLM imports blocked, package writes denied,
  and no setup-data matrices present.

Completion requires synchronized code/docs, tests, and a small end-to-end backfill
record showing elapsed scoring time, output sizes, identity checks, and absence
of signature rediscovery. No independent calibration claim is required.

## Boundaries and remaining scientific work

No new scientific choice is needed to store the exact C/B distributions and
expose the explicit empirical quantities above. This draft deliberately leaves
automatic joint ranking rules, acceptance cutoffs on C percentiles, independent
calibration, membership priors/probabilities, multiple-testing control, N/C query
aggregation, and forced single-cluster assignment to later decisions. Users may
inspect background evidence and C compatibility side by side now; do not label
those outputs as probabilities. Five-fold work is optional research under
SPEC-0004 and must not delay default production or be discarded when already
completed and reusable.

## Readiness review — 2026-10-27

The readiness review preceded implementation with revised SPEC-0004. It resolved stage/cleanup
ordering, source versus destination identity binding, deterministic-data versus
execution-provenance comparisons, concrete schema/selector/API fields, cache
bounds, empty/invalid queries, unsupported counts and lazy-read guards. None of
these changes adds probability calibration or a scientific decision. The implementation verification record below supersedes the earlier
documentation-only status.

## Implementation and verification — 2026-10-27

Implemented `relative_scores.py`: fixed-signature block scoring, sorted float64
arrays, bound private checkpoints, companion publication, bounded thread-safe
LRU snapshots, paired loading and frozen empirical-evidence results. Private
array names follow the shared checkpoint writer (`complete.C.npy`/`complete.B.npy`);
published names remain `C_log_scores.npy`/`B_log_scores.npy`. Final and research
cleanup preserve these durable outputs and verify mode-specific completion gates.

The full 127-pass suite and subsequent focused checks are recorded in
[SPEC-0004](0004-build-library.md#implementation-and-verification--2026-10-27).
Independent tests cover arithmetic Gaussian scoring, tied/inclusive counts,
finite-sample tails, all-equal arrays, extreme/invalid/empty query inputs and
window multiplicity. Integration tests exercise default production, unsupported
populations, immutable evidence, legacy/v2 backfill without discovery, parallel
byte consistency, paired force rollback, research isolation and post-cleanup
recovery. Installed package verification includes both artifact families and
research without embedding matrices or PLM dependencies.

The [real-data record](../research/notes/2026-10-27-final-only-backfill.md) reports
1,706,344 bytes for one target's C/B array files and exact input/model bindings.
This is a scoped import/scoring check, not an independent calibration experiment
or full-library discovery-speed measurement. Probability calibration and a
combined cluster-ranking decision rule remain deliberately outside this command.

## Shared path and distribution update — 2026-10-27

[SPEC-0006](0006-shared-artifacts.md) supersedes earlier cwd-relative defaults.
CLI and Python omitted paths resolve to `<repo>/data/setup-data` and
`<repo>/data/library` in a checkout, regardless of cwd. Explicit overrides remain
caller-controlled. Installed packages read selected bundled data or existing
user-cache data and write new outputs only to the user cache. Package builds
export selected complete shared bundles through validated build hooks; scientific
identities, cache safety and historical experiment paths are unchanged.

The user-facing build progress label is `ABC match evidence`, for example
`Cluster cluster_0001 ABC match evidence: computing or reusing reference distributions`.
This denotes preparation of the fixed-signature C/B reference distributions;
it does not imply probability calibration. Artifact names and APIs retain their
`relative_scores` terminology.
