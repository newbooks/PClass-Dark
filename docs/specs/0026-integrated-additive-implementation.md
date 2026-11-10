# SPEC-0026: Production additive reranking implementation

- Date: 2026-11-08.
- Status: implemented and validated; operational limits are recorded in section 20.
- Revision: 2026-11-09, managed feature-cache path repair; see section 22.
- Parent requirements: [SPEC-0025](0025-integrated-additive-evidence.md).
- References: setup 0003, library 0004, inference 0007, structure 0020,
  features 0021, exploration 0022, final fitting 0023, interpretation 0024.
- Scope: existing 18 full-sequence descriptors and existing cluster-ID targets.
  Direct GO-set targets and direct embedding classifiers are outside this change.

## 1. Decisions and invariants

Production is setup-data → build-library → infer. No CV artifact, exploration
manifest, inner-selected epoch list or exploration command is required. Preserve
all three standalone ML scripts and their APIs. The root ML registrations
were retired by explicit user request on 2026-11-08 (section 21).

PClass-Dark acceptance, B/C values, terminal windows and representative selection
remain unchanged. Additive ranking is supplemental. Rank all accepted candidates
that have classifier outputs, not only the global-top-five intersection. Never
promote an additive-only suggestion to an accepted cluster. Preserve original rank.

Reports call the component an additive classifier and record actual HARD/SIAC
configuration. Initial default is HARD because the recorded five-fold experiment
favored it; this is not a claim of validated integrated reranking accuracy. SIAC
remains an explicit recipe choice. No combined probability, B/C multiplication,
new acceptance threshold or automatic configuration search is introduced.

## 2. Ownership and artifact layout

Use existing setup/library path resolvers, locks and generation publication.
Paths below are relative to their enclosing immutable bundle/generation:

| Owner | Artifact | Contents |
|---|---|---|
| Setup data root | additive-reference/<id>/manifest.json | Versioned frozen reference contract and inventory |
| Same | clusters.tsv | Canonical retained classifier cluster vocabulary |
| Same | cluster_dmatrix.tsv | Existing six-column medoid pair schema, retained-target pairs |
| Same | descriptions.json | Snapshot-bound full descriptions for all retained targets |
| Same | recipe.json | Validated frozen training recipe |
| Library build cache | additive-features/<id>/ | Resumable full-sequence feature extraction and exclusion audit |
| Library generation | additive/model/ | Existing final numerical bundle, feature contract and model manifest |
| Same | additive/reference.json | Setup snapshot identity, matrix checksum, vocabulary and description provenance |
| Same | additive/descriptions.json | Portable full descriptions; no live catalog required by inference |
| Same | additive/training.json | Recipe, counts, exclusions, extraction identity, execution and timing |

Signature library.v2 manifests and IDs are preserved. An integrated-library.v1
envelope under library-root/integrated/<id>/ owns the nested additive/model,
additive/reference.json, additive/descriptions.json, additive/training.json and
structural/ assets. Its identity_inputs bind signature_selection, reference_id,
recipe_id, model_id and heads_id; its files inventory covers every payload.
Library-selection.v3 retains the original signature/distribution locations and
adds integrated path/ID/manifest checksum plus additive status/model ID. Selection
changes only after complete verification. A disabled signature-only build uses a
v2 selector with additive_status=disabled_by_build and no integration pointer.
Do not mutate historical signature generations to install the component.

Do not duplicate the full feature table or similarity matrix in the inference
bundle: their identities and inventories suffice. Keep resumable training inputs
in the managed build cache, and record references for cleanup reachability.

## 3. setup-data changes

Add --cluster-similarity PATH for an explicit import and --additive-recipe PATH
for an optional recipe override. Default resolution uses the frozen asset specified
in section 12; do not infer
an arbitrary work/ file or neighboring checkout. Explicit imports also require
--cluster-similarity-manifest PATH containing the source-binding fields in section
12. A manifest without its paired table, or vice versa, is invalid. The supplied
source must match prepared provenance; never infer historical export provenance
from matching IDs.

Preserve prepare-only: preparation never initializes a PLM or extracts features.
Use the existing prepared cluster definitions and selected cohort as authority.
The additive reference ID hashes the prepared manifest identity, source cluster
membership identity, similarity canonicalization version, canonical pair checksum,
descriptions checksum and recipe. Source paths/timestamps are provenance, not
scientific identity. Do not rewrite historical prepared bundles to add this data.

Import using ml_data.PAIR_HEADER exactly:
cluster_A, Medoid_A, Cluster_B, Medoid_B, Distance, Similarity (TSV).
Keep these six columns in the managed table so load_similarity remains reusable.
Orient each retained pair lexicographically and sort rows by (cluster_A, Cluster_B),
reordering medoid cells with their clusters. Use float values exactly as imported
and serialize with .17g; do not introduce rounding during canonicalization. Reject
all duplicate unordered pairs, including equal reverse duplicates, consistently
with the existing loader. Require finite [0,1] scores, no self-pairs, and
abs(Distance+Similarity-1)<=1.00000001e-5. Validate source medoids against the bound
metadata and complete retained-label coverage. Extra source clusters are audited
and filtered; no missing pair is zero-filled. Diagonal similarity is implicitly 1.
Canonical clusters.tsv has header cluster_id,medoid_accession (TSV), sorted by ID.
Use external SQLite ORDER BY on a disk-backed unique pair key to bound sorting
memory; set its cache to 64 MiB and temporary workspace budget to 2 GiB. Run
load_similarity validation on canonical output within its declared memory budget.

Descriptions come from the same reference snapshot; reuse existing catalog
verification during import and freeze its output. Resolve no live annotations at
infer time. Training exclusions may shrink the classifier vocabulary later.

An omitted explicit import can reuse a matching verified frozen reference.
If additive building is requested without a valid reference, fail before training
with an actionable setup-data instruction. --no-additive library builds need no
additive reference. For this initial release, absence of a compatible explicit, selected or bundled
reference is an error for default setup; report how to import a bound snapshot.
An explicit --no-additive setup flag preserves signature-only preparation and
records no additive selection. With --no-additive, reject additive import/recipe
options. An override never rewrites the frozen default.

## 4. Training recipe and build order

Recipe schema pclass-dark.additive-recipe.v1 uses these initial defaults:

| Field | Value |
|---|---|
| configuration | hard |
| epochs | 100, fixed full-data fit |
| knots / spline degree | 8 / 3 |
| learning_rate / weight_decay | 0.001 / 0.0001 |
| optimizer | AdamW, eps=1e-8, existing deterministic settings |
| batch_size / seed | 1024 / 42 |
| alpha | 0 for hard; 0.1 for siac |
| neighbors / similarity_power | 10 / 2 |
| feature_precision | .3f |
| preprocessing | Existing saved training medians, knots and centered spline basis |
| features | FEATURES in ml_data, excluding sequence_charge_decoration |
| identity_conflicts | Existing deterministic identical-sequence prefilter; exclude conflicting identities |

Validate recipe keys and types; unknown fields fail. Require alpha/configuration
consistency. Derive final fit seed from seed and purpose final-hard/final-siac as
existing final fitting does. Do not perform validation-based epoch selection in
production. Include prior evaluation provenance as optional descriptive metadata,
not a prerequisite or an accuracy claim on a different reference population.

build-library gains --additive/--no-additive (default enabled),
--additive-device auto|cpu|cuda[:index] (default auto), and
--additive-memory-gib (default 8). Reuse --workers as the numerical thread budget for this single fitting process,
bounded by available CPUs. Structural extraction uses min(8, budget) CPU threads
and min(8, requested embedding batch) windows, matching StructuralPredictor limits.
New Python arguments mirror these controls, with additive=False retained on the
legacy Python preparation/build APIs; the root CLI defaults to additive enabled. Add --offline (default false) to forbid
checkpoint downloads for full-sequence extraction; fitting itself never downloads.
Missing cached assets under --offline fail without publishing a generation.
--clean performs validation/cleanup only and never starts extraction or fitting.
A cluster-subset signature build still trains/reuses the global prepared-cohort
classifier: do not accidentally fit on only the selected signature clusters.

Build dependency graph:

1. Verify prepared references and frozen additive recipe/reference.
2. Prepare existing structural prediction heads and required signature artifacts
   in non-selected staging/checkpoint generations; defer selected-pointer publication.
3. Extract full-sequence features from unique selected reference sequences using
   the staged heads and pinned PLM. Do not pool terminal windows into full features.
4. Apply existing duplicate/conflict exclusion contracts, retaining alias and
   exclusion audits. Each retained identity supplies one training example.
5. Fit preprocessing and the classifier on this entire eligible cohort once.
6. Validate model, vocabularies, descriptions and feature extraction identities.
7. Publish the whole generation and update its selection pointer.

Feature cache identity binds sequence population, structural heads, PLM revision,
extraction code, precision and feature order. Model identity also binds recipe,
actual device, fitting versions/code, target labels and frozen similarity.
A compatible resume reuses completed stages. A failed additive stage leaves the
previous selected generation usable; do not publish a newly required component as
silently disabled. --no-additive explicitly publishes a signature-only generation.

## 5. Library and scientific APIs

Refactor ml_model fitting into fit_reference_model(X, y, labels, similarity,
feature_contract, recipe, output_dir, *, provenance, device, threads,
memory_budget_gib, resume), using explicit validated/read-only arrays. Preserve
fit_final_model(experiment, ...) as the existing CV-derived adapter. Shared fitting,
serialization, admission and validation must not diverge between entry points.
The new API has documented ValueError/IntegrityError, OSError and RuntimeError
behavior and performs no computation at import time.

Add score_candidates(model, X, candidate_labels, *, batch_size), returning global
top five, requested-candidate logits/global ranks/softmax and availability. Compute
global logsumexp over all outputs; do not normalize only over candidates. Bound
score blocks and preserve cluster-ID tie breaking. No full queries-by-labels
matrix is retained. Extend chosen_support to arbitrary requested local classes;
retain the existing prediction API's comparative-field meanings.

Add a library additive loader verifying nested/enclosing inventories and matching
preparation, model vocabulary, description identity and extraction contracts.
No reference lookup, training or download occurs while loading numerical weights.

## 6. Inference execution and fallback

infer gains --additive auto|off|required, default auto:

- auto: use a valid library component; old/disabled libraries produce explicit
  unavailable supplemental evidence and retain original PClass-Dark output.
- off: bypass supplemental extraction/scoring and record disabled_by_user.
- required: fail on absent component or unsupported supplemental query.
- A present corrupt/incompatible component fails in auto and required modes;
  it is never downgraded to missing. off does not load the component.

Existing sequence validation and PClass-Dark limits stay in force. Additive
extraction initially supports at most 100,000 residues, matching predict_sequences
(the standalone feature extractor has a different, larger input cap). Longer valid PClass-Dark queries receive unavailable/length_limit in
auto mode; required mode fails before report publication. Do not silently truncate.
Unexpected extraction, device or scoring failures fail the run; only the listed
availability states permit fallback. Use existing GPU probing/admission and CPU fallback policy for extraction.
Additive inference uses saved NumPy/SciPy bases and CPU matrix multiplication, as
predict_siac already does; record additive_score_device=cpu. Record actual extraction
and training devices separately. A requested GPU never implies GPU additive scoring.

Reuse model/backend instances within a run where their pinned contracts match.
Refactor full-sequence extraction to accept existing heads/backend explicitly,
without making integrations invoke CLI subprocesses. Do not promise zero extra
PLM passes: whole-sequence and terminal-window representations may differ.
Introduce a borrowed-backend context (section 13), keep one resident PLM, and
schedule forwards serially initially. Numerical thread limits apply across both
components. Mismatched PLM/head contracts are incompatible-library errors, not a
reason to load a second model.

Preserve matches in original PClass-Dark order in JSON. Produce a separate
reranking array of accepted candidates: scored rows by descending logit then
cluster ID; unscored rows after them, preserving original order. Rows have unique
1-based positions, score_available and reason. If no accepted candidate is scored,
select the original first candidate and mark selection_rule=pclass_original.
Otherwise select the first scored candidate, selection_rule=additive_reranked.
No accepted candidates means selected_cluster_id=null regardless of global top five.

Compute full 18-feature own-class explanations for the selected accepted candidate
and each accepted candidate in the global top five, at most six classes/query.
If none is selected/scored, selected_explanation_cluster_id is null. A global
winner outside acceptance receives a score/description but no implied acceptance.

## 7. Comprehensive JSON schema

Introduce pclass-dark.integrated-infer-report.v1 for integrated runs. Preserve the
legacy infer-report.v1 renderer/API for callers explicitly using legacy rendering;
CLI --format json uses the integrated schema after this migration and documents it.
Root keys:

- schema; source (signature/additive preparation and library identities);
- execution (options, actual devices, versions, warnings); coverage (existing
  signature coverage plus additive vocabulary counts); queries; counts.

Each query contains:

- query_index, query_id, length, sequence_sha256, windows, status;
- pclass_dark: matches with every existing ClusterMatch field plus original_rank;
- additive: status, reason, model_id, configuration, reference_id, recipe_id,
  global_top5, accepted_reranking, top5_accepted_cluster_ids, features,
  imputed_features, outside_support_features, explanations;
- classification: selected_cluster_id, selection_rule, original_rank,
  reranked_position, in_additive_top5, selected_explanation_cluster_id;
- analysis: structured strongest positive/negative observed feature IDs, agreement
  and availability codes used to render prose.

Global top-five rows: rank, cluster_id, description (full), logit, softmax_score,
accepted_by_pclass_dark. Reranking rows additionally carry original_rank, position,
global_rank, score_available and reason. Unavailable numerical fields are null.
Feature arrays/maps retain canonical FEATURES order. Explanation records include
cluster_id, feature_contributions (own-class), bias, logit, reference=training_mean,
training_example_count; reconstruction tolerance follows SPEC-0024. Do not confuse
these terms with existing standalone comparative feature_contributions.

Use finite JSON numbers with full available precision, never NaN. Include all
accepted candidate/window records, even if classifier coverage is incomplete.
Descriptions are full text in JSON, not truncated table labels. Both renderers
consume these records; no separate inference or selection is allowed in rendering.

## 8. Human report and screen contract

For each query: optional heading, then the opening paragraph required by SPEC-0025.
It states accepted-candidate count, leading accepted classification after reranking,
original position if changed, global-top-five agreement and up to three strongest
positive observed feature supports for that selected cluster. If none are positive,
state this explicitly. Missing/imputed values cannot be described as observations.

Then emit Detailed PClass-Dark analysis (all original evidence), followed by
Additive reranking analysis (accepted reranking, global top five, agreement and
selected-class contribution table). For accepted/top-five intersection members,
provide brief actual feature support explanations from their own-class records.
Retain signs, training-mean reference and uncalibrated-score caveats. No language
model calls, causal claims or independent-confirmation claims. Escape external
IDs/descriptions in Markdown and terminal output; cap compact descriptions at
24 displayed characters including ellipsis using the existing formatter.

infer retains --format text|json and adds markdown. Default text stdout contains
the opening summary and compact accepted-reranking/global-top-five tables; detailed
human analysis is in the Markdown file. json prints only JSON; markdown prints the
full human report. Progress and saved-path messages go to stderr.

By default publish both <input-stem>-classification.json and
<input-stem>-interpretation.md beside FASTA input. For --seq use query as the stem
in cwd. --report-prefix PATH overrides directory/stem; --no-reports disables file
publication for pipelines. Reject --report-prefix with --no-reports. Do not alter
standalone predict-ml file naming. Existing filenames are never overwritten.

Precheck both destinations before computation; stage both reports before publishing.
Use a locked publication transaction with a recoverable journal and an inventory
receipt <prefix>-classification.complete.json, installed last. Receipt records
schema, query count and both file checksums; outputs are reusable as a pair only
with a verified receipt. Check receipt collision too. Roll back newly published
members after a publication failure; recovery must never delete unowned files.
This is recoverable multi-file publication, not a claim that two filesystem names
can change atomically. Do not emit stdout success summaries until publication
succeeds. With --no-reports, stage stdout until the run is fully validated.

## 9. Resource limits and packaging

Retain existing input/query/window caps and 2 GiB temporary-storage quota; account
for JSON, Markdown, receipts and stdout staging together. Stream records/reports
using the existing disk-backed session, not an in-memory list of all queries.
Accepted-candidate scoring is blocked; explanation work scales with at most six
classes per query. Limit each block by actual basis/output/workspace estimates.

Reuse sparse-target memory admission in both fitting APIs, including simultaneous
copies and batch workspaces. Read-only similarity arrays remain shared/mapped;
semantic controls do not copy matrix columns. Training is one GPU fit or one CPU
process; do not nest it inside signature worker pools. Close worker pools before
feature/model stages as needed to bound combined resource peaks.

Synchronize pyproject dependencies, package-data patterns and source-only build
rules: production training needs the ml extra, and extraction/infer need embeddings.
Before starting a default additive build, missing extras produce an actionable
install instruction rather than a partial published generation. Numerical model loading/scoring uses NumPy and SciPy BSpline and requires neither
scikit-learn nor Torch. Fitting requires the ml extra; raw feature extraction
requires embeddings. Test these import boundaries explicitly. Add no new runtime
dependency on profunc beyond the existing declared package/reference contracts.
Include frozen recipe/reference data in managed distribution assets, without
accidentally packaging work/ or training caches. Update cleanup, migration and
portable-library validation to include nested additive ownership.

## 10. Implementation sequence and acceptance gates

1. Frozen reference import, recipe schemas and setup selection tests.
2. Direct reference fitting API; retain CV adapter/script regression coverage.
3. Feature cache/build dependency graph; library manifest and atomic publication.
4. Candidate-scoring API, arbitrary-class explanations and integrated records.
5. infer fallback modes, session integration and bounded per-query execution.
6. JSON/Markdown/text renderers and recoverable report-family publication.
7. Packaging, README/API/CLI help, cleanup/migration and source-only distribution.

Required automated tests:

- Exact reference binding, missing/inconsistent pairs, stale recipes/descriptions,
  deterministic canonical import, no PLM initialization during prepare-only.
- Tiny full-data fit without any exploration files; direct/CV-adapter numerical
  parity given identical inputs/settings; deterministic CPU reload/prediction.
- Feature/head identity parity, duplicate/conflict audit, resume/corrupt checkpoints,
  failed fit preserves selected library, subset build/global classifier behavior.
- Analytical candidate ranks, tie order, global normalization, absent-label order,
  no accepted results, selected accepted class differing from global winner,
  six-class explanation bound and contribution reconciliation.
- Byte/numeric parity of original PClass-Dark acceptance, B/C, representative and
  support records; unchanged results under additive off.
- Old-library auto fallback, required errors, explicit off, corrupt-present failures,
  length-limit behavior and unexpected extraction failures.
- JSON/text/Markdown agreement; first paragraph comes before details and explains
  the selected accepted class; imputed features never become observed evidence.
- Multiple queries, escaping, target collisions/races, quota exhaustion, late input
  errors, transaction failure/recovery, and no partial terminal success.
- Serial/thread/resource admission, CPU path, optional real GPU check and mocked
  GPU admission/fallback; tests must report GPU skips honestly.
- Installed CLI/library smoke, source-only wheel/sdist contents and self-contained
  inference without work/, setup cache or neighboring repository paths.

Performance experiments are separate from implementation correctness: measure
added extraction/scoring latency, peak host/GPU memory, candidate recall, reranked
top-one exact/semantic quality and agreement coverage on held-out candidate sets.
Both components must exclude held-out proteins; use homology-separated evaluation
before claiming generalization benefits. No performance improvement is assumed.

## 11. Readiness and implementation completion

All algorithm and interface decisions for this version are specified below. The production integration and reference freeze are implemented. Section 20 records
actual interfaces and limitations; validation outcomes are recorded below. The
separate held-out experiments remain research work. Do not promise measured
accuracy or latency improvements from implementation tests.

## 12. Frozen source and selection contracts

The initial import source is the existing user-provided work/cluster_dmatrix.tsv:

| Property | Verified value, 2026-11-08 |
|---|---|
| Raw SHA-256 | d934cc904212c9e1d901c937ca818a7a6246386b5e51f97cb6962dc3faa9bd05 |
| Bytes | 354315214 |
| Cluster IDs / pair rows | 3524 / 6207526 = 3524*3523/2 |
| Source medoids | All match the bound clusters_meta.tsv |
| clusters_meta.tsv SHA-256 | 9dd8837c195ec275001f0b1b9f7477a4a33d6bd4df849adbe15e97f76608cbd0 |
| Source clusters.tsv SHA-256 | b616109726acb0e0403060725c097c19bb245062aba62beabf4fcc6d3db8f1de |
| Source GO graph SHA-256 | 819a5675ebd00edd6622d1bd45c8feb05928a8fc2f12d7dc67e555fc843b4f06 |
| Source accession_mf_go.tsv SHA-256 | 267099636e2dddf81ddce895a065b2af99a7812867a40462d6d64dfdf5de97ef |
| ProFunc bundle ID | 28f87149af01ed624c2814ca93ce1111ed6c915ad8aac4545ed1af48c00f0ee0 |
| Bound prepared ID | 8e5994726cb958fce5d2396835f9a3767da897007110a9fda4f2e3b06cccbe4e |

The prepared reference is the authority for all member digest values. Current
ProFunc cluster_listing.iter_medoid_distances uses medoid-profile AMB Lin
similarity and emits Distance=1-Similarity to five decimal places. This matches
the provided format, but the original export command/code revision is unrecorded.
Record origin=user-provided-export, historical_generator=unknown, and
semantic_method=declared-medoid-profile-AMB-Lin; do not claim a verified original
execution or recomputation of all pair scores. Matching medoids and bound source
snapshots establish the adopted reference, not historical score generation.
Full pair/schema validation remains an implementation gate. If these bytes have
changed, stop default import with an integrity error; do not silently approve a
new matrix. New data requires an explicit manifest and a new reference identity.

Implementation copies this source once into
src/pclass_dark/resources/additive-reference/v1/cluster_dmatrix.tsv.gz, using
reproducible gzip with empty filename and mtime=0, together with source.json,
recipe.json, descriptions.json and reference-members.json. Track large payloads
with an explicit Git LFS rule; do not commit/push as part of implementation unless
requested. Add source.json's schema pclass-dark.additive-source.v1 with fields
raw_file={sha256,size_bytes}, compressed_file={sha256,size_bytes}, header,
source_reference (prepared reference identity/files), medoids_sha256,
semantic_method, origin, historical_generator, generator_reference (informational
source file/hash), scores_precision=.5f, and validation counts. reference-members
stores the exact medoid accession/profile and composed_go for each source cluster,
bound to clusters_meta.tsv; descriptions are derived from that frozen graph.

Package this explicit resource directory in wheels/sdists; reject Git LFS pointer
text with actionable materialization guidance. Never regenerate scores or consult
neighboring checkout paths during runtime. Decompress through a bounded streaming
reader to owned setup staging, checking compressed and raw digests before import.
Budgets cover uncompressed size, pair-sort SQLite and canonical output; reject
expansion beyond declared raw size. setup-data can use installed ProFunc only to
verify selected reference identities, as existing preparation does.

Managed selection is additive-selection.json under the setup root, schema
pclass-dark.additive-selection.v1, fields preparation_id,
prepared_manifest_sha256, reference_id, reference (safe relative bundle path),
reference_manifest_sha256 and recipe_id. Publish it last under the existing setup
transaction guard; extend its allowed publication domains for additive_reference.
Verify the ordinary setup selection is unchanged before updating this companion.
No additive pointer is added to the historical setup selection.v1 object.
Resolution order: explicit pair+manifest, compatible managed selection, bundled
source. A broken selected pointer is an integrity error, never silently bypassed.
A missing pointer permits bundled resolution; an incompatible valid selection
permits deriving a new reference only from a matching explicit/bundled source.

For a valid new preparation over the same source reference but a different retained
subset, deterministically filter the frozen full table. Bind new prepared ID and
new vocabulary into the derived reference. Unsupported source-reference revisions
require explicit import; never reuse cluster numbers across snapshots.

Recipe JSON is a complete object, not a partial override. Its required keys are
schema, configuration, epochs, knots, degree, learning_rate, weight_decay,
optimizer, optimizer_eps, batch_size, seed, alpha, neighbors, similarity_power,
feature_precision, features, identity_conflicts. Values follow section 4;
optimizer=adamw and identity_conflicts=exclude. Require >=1 epochs/batch/neighbors,
knots>=2, degree=3, seed>=0, finite positive rate/epsilon/power, nonnegative decay,
alpha=0 for hard or 0<alpha<1 for siac, exact feature order and precision=.3f.
Booleans are invalid integers. The implementation supports no other optimizers,
degrees or identity policy in this schema. Canonical JSON identity excludes paths.

## 13. Integration seams and lifecycle

Introduce these modules with no import-time model/data work:

- additive_reference.py: freeze/verify/import reference, canonical tables, selection.
- additive_build.py: cache/extraction/cohort/direct fitting orchestration.
- integrated_inference.py: compose original session records with additive evidence.
- integrated_records.py: immutable records, validated status/selection enums.
- integrated_render.py: JSON, Markdown, terminal output from shared records.
- integrated_reports.py: report-family staging, ownership, recovery and publication.

Reuse ml_data, ml_model, siac and ml_interpretation for scientific operations;
do not copy target, feature or spline formulas into these modules. Extend
setup_data.prepare_data with additive=False, cluster_similarity=None,
cluster_similarity_manifest=None, additive_recipe=None. Extend SetupResult with
optional additive_reference_id/dir fields. Extend build_library with additive=False,
additive_device='auto', additive_memory_gib=8.0, offline=False; extend LibraryResult
with optional additive_model_id/dir. Preserve existing positional fields and add
new dataclass fields with defaults. Use normal validation for booleans/options.

Preserve production library.v2 manifests, including their cluster-path layout and
scientific IDs. The new integrated-library.v1 envelope and library-selection.v3
provide the aggregate lifecycle (section 2). Existing signature checkpoints remain
keyed by their original scientific inputs. Additive recipe changes do not force
signature refits. Structural heads are nested in the portable envelope; no new
selector structural path is needed. The existing distribution selector fields
continue to bind relative-score IDs and checksums. Package-selected envelopes
with all their nested files, including NPZ weights, safetensors and model cards.

Extend validation, library_paths migration and library_cleanup/data_cleanup ownership
for all v2/v3 forms and additive reference/cache roots. Missing additive keys in
v1 are supported absence; missing declared v2 complete files are corruption.
Keep --clean signature-only semantics: it validates declared additive inventories
without fitting, preserves live feature/reference dependencies, and prunes only
unselected owned generations using the existing cleanup safety model.

Full-sequence extraction can emit the existing 19-feature schema for compatibility,
but training drops its last SCD column through ml_data.FEATURES. Always provide
sequence_index from checksum-verified prepared aliases to load_cohort; never use
accession fallback identity grouping. Require N>=1, C>=2 and at least one observed
value in every modeled feature; use the existing invalid-cohort checks. Preserve
all target clusters with >=1 eligible example, without a new minimum-count filter.
Clusters losing every example appear in training.json vocabulary_exclusions and
remain valid, explicitly unscored signature candidates.

For fit_reference_model, require float64 X[N,18] (NaN allowed, infinity rejected),
int64 y[N], sorted unique labels[C], float64 read-only S[C,C] with existing
symmetry/range/diagonal checks, and the exact extraction feature contract. Return
FinalFitResult, write the existing BUNDLE_FILES and standalone manifest format,
and extend its provenance with source_kind='prepared_reference'. Keep CV-adapter
source_kind='exploration'. Loader checks depend on saved numerical/feature
contracts, never the continued presence of an exploration directory. Compare
actual-device admission on completed resume using the same policy as fitting.

StructuralPredictor gains an internal borrowed backend/heads constructor or context
with explicit ownership: it never closes, moves or independently initializes a
borrowed backend. Validate PLM assets/layer/dtype/window recipe and head IDs first.
Use inference_backend.backend_context once; pass the resulting backend to both
window embedding and structural extraction. Do not re-enter the execution lock
or restore global Torch settings until both consumers finish. If structural auto
fallback moves the shared backend to CPU, synchronize recorded execution and
subsequent signature forwards with its actual device; explicit GPU failure errors.
Feature-cache identities must bind the actual frozen structural contract, not a
staging path. Additive inference batches at most eight windows and caps native
threads at eight for the structural head, even when --workers is larger.

## 14. Scoring, analysis and schema precision

score_candidates accepts X[N,18] and candidate_labels as N sequences of unique
cluster IDs aligned with input rows. Return an iterator of CandidateScores records
in query order; each has global_top5 and candidates in supplied order. Reject
misaligned rows/duplicate candidate IDs. Unknown IDs have score_available=false,
reason=not_in_classifier_vocabulary and null logit/softmax/global_rank. Score
available IDs even when they are outside global top five. Rank globally by
(-logit,cluster_id), using saved float32 basis/weights/logit arithmetic and float64
stable softmax normalization, consistent with predict_siac. Scores have no
posthoc probability calibration. Tests must cover float ties explicitly.

Default score block is min(32,N); reduce until estimated basis, logits,
probabilities, stable-sort indices and requested explanation workspace fit the
infer --memory-mib allocation, reserving 64 MiB for SQLite/stream buffers. For a
block Q and C labels, reserve at least Q*(B*16+C*32) bytes plus weight bytes;
limit all operations to this reservation and fail if one row cannot fit. Holding
one row's sorted C indices gives global ranks without O(accepted*C) comparisons.
No full-input X or score matrix is held. Explanation blocks handle one query and
one requested class at a time. Reserve up to six explanation records/query; their
class order is selected accepted class first, then accepted global-top-five order,
with duplicates removed. Do not fit or recenter curves at inference.

Additive status enum: available, unavailable, disabled. Reason enum: null for
available; missing_component, disabled_by_build, length_limit for unavailable;
disabled_by_user for disabled. With no accepted candidates, available is still
valid and global_top5 is computed. When unavailable/disabled, global_top5 and
explanations are empty, features=null, imputed/outside lists empty, every accepted
row unscored with the query reason. Feature NA serializes as null; model inputs
remain imputed as recorded. There is no selected-explanation object to conflict
with selected_explanation_cluster_id: explanations is the single authoritative
array and that ID references it, or is null.

classification.selection_rule enum: additive_reranked, pclass_original, none.
Empty acceptance has all selected/rank/explanation fields null, in_additive_top5
false and selection_rule=none. With any scored accepted candidate choose the
first scored row even if unscored candidates existed earlier; report partial
coverage explicitly. Reranking does not change top-level status, which remains
the original matches/no_accepted_matches. All positions/ranks are 1-based except
existing query_index, which stays 0-based. Softmax global-rank tie rules are
identical to candidate sorting. No tolerance-based rank ties are introduced.

analysis.agreement enum: selected_in_global_top5, selected_outside_global_top5,
no_accepted_matches, additive_unavailable. Store positive/negative observed
feature IDs per explained cluster, sorted by signed magnitude then FEATURES order;
use abs(term)<=1e-6 as near-zero. The opening paragraph uses up to three positive
IDs, natural values and signed terms. If the selected class has only negative or
near-zero observed terms, describe this honestly and explain baseline/imputation
when applicable. Preserve full 18-term tables and exact raw sequence length.
No-acceptance text must not name a supplemental suggestion as a classification.

Descriptions in every accepted row come from the frozen source; accepted clusters
outside classifier vocabulary still have descriptions. Unavailable descriptions
are null in JSON and Unavailable in compact tables. Model ID/configuration are
present for length-limited queries with a valid loaded model, but null when absent.
Every root query uses schema=pclass-dark.integrated-infer-query.v1. Root source,
execution and coverage retain their original objects under pclass_dark and add a
separate additive object rather than overwriting existing field meanings. Counts
retain original query/window/match totals and add additive_available_queries,
reranked_queries and selected_top5_agreement_queries; off/fallback do not inflate
these. Unknown keys in public JSON readers may be ignored; required schema fields,
finite values and enums must be validated in the producer and fixtures.

Text, Markdown and JSON use the same completed integrated record stream. Introduce
infer_integrated_queries(..., additive='auto', ...) as the context-managed Python
API, leaving infer_queries and its legacy report writer intact. Integrated result
storage extends the existing bounded spool; it never launches legacy inference
again for another format. Render the complete report family in one pass and
stage requested stdout alongside it. Expose write_integrated_reports(session,
json_stream, markdown_stream, *, terminal_stream=None) for Python callers; CLI
publication lives outside this scientific/session API. No log scraping.

## 15. Publication and recovery protocol

Default file prefix is fasta.with_suffix('') for FASTA (remove only the last
suffix), or cwd/query for --seq. --report-prefix is a literal path prefix, not an
output directory; append suffixes without replacing any existing prefix suffix.
Resolve destinations before reading models; reject symlinks, collisions, paths
inside owned library/setup roots, and report paths identical to an input. Create
an explicitly requested parent directory, but never overwrite its files. Receipt
schema is pclass-dark.integrated-report-completion.v1 with transaction_id,
query_count and relative-basename file records. Hash bytes actually published.
There is no implicit completed-report reuse or overwrite option in this version.

Use <prefix>-classification.lock and <prefix>-classification.journal.json under
one FileLock. Lock/journal paths must also be safe regular paths, never symlinks.
Create transaction staging in the destination filesystem; journal schema
pclass-dark.integrated-report-journal.v1 records transaction UUID, target basenames,
staged file dev/inode, checksums and sizes before publication. Use no-overwrite
hard links for destination installation, with fsync of payloads and directory.
A crash between link and journal update is recoverable because the staged inode
was recorded before linking. Install receipt last; a valid matching receipt marks
the transaction complete. Journal writes use existing atomic JSON utilities.

On recovery, verify ownership via recorded inode and digest before deleting a
newly installed member. A different inode/digest or unexpected member is a recovery
conflict, never grounds for deletion. With valid completed receipt, clean only
owned staging/journal and then report the normal output-exists error. With an
incomplete owned family, remove only owned members/staging and allow a fresh run.
Keep journal/staging on a recovery conflict and surface actionable paths. Tests
inject failures before/after each install and cover concurrent target creation.
Temporary quota includes transaction payloads, spool files, journals and staged
stdout; hard-linked published bytes are counted once by inode. No files are
published and no success summary is printed after a late-query validation failure.
A broken stdout pipe after successful file publication does not delete valid files.
With --no-reports do not create report lock, journal, receipt or target directories.

## 16. Readiness audit and implementation work boundaries

Verified on 2026-11-08: source matrix raw digest/size, streamed row count and medoid
coverage; prepared reference member identities; existing six-column parser,
float64 similarity loader and duplicate rejection; NumPy/SciPy inference bases;
structural constructor thread/window caps; setup/library selection versions;
separate structural/distribution artifacts and legacy inference schemas.

This is a documentation readiness audit, not a new fit, a full score recomputation,
or a passing integration test suite. Actual matrix import must validate every
score and pair against section 3; current observation alone does not replace it.
Implementation-ready means no unspecified decision is delegated to an implementer:
follow concrete defaults and migrations above, with packaging/import and tests as
required tasks. Changes to scientific acceptance or default recipe beyond this
contract require a specification update. No commits or pushes are authorized.

## 17. Concrete report cases and acceptance assertions

The following cases are normative fixtures, independent of learned model accuracy:

| Original accepted IDs | Global classifier order | Candidate vocabulary | Selected / supplemental action |
|---|---|---|---|
| A,B | Z,B,A | A,B,Z | B; retain original A rank 1, B rank 2; highlight B and A in global top five |
| A,B | Z,Y,X,W,V,B,A | All | B; selected_outside_global_top5, no highlighted intersection |
| A,B | B,Z | B,Z | B; A remains accepted, unscored and placed after B |
| A,B | Z,Y | Z,Y | A; pclass_original, both accepted scores null |
| Empty | Z,Y | Z,Y | No accepted classification; show Z/Y only as suggestions |
| A,B | Equal logits for A/B | A,B | A by lexical ID tie rule, independent of original ordering |
| A | Component missing, auto | None | A with additive_unavailable; preserve evidence |
| A | Component missing, required | None | Whole run fails before publication |

C may be less than five; global_top5 then contains exactly C entries. Global rank
is 1-based among the full C outputs. Top-five intersection retains global-rank
order. Every accepted candidate appears exactly once in both original matches
and accepted_reranking, with equal cluster IDs and unchanged PClass-Dark records.
IDs not accepted never appear in accepted_reranking.

A representative opening paragraph template (values are substituted from records):

> PClass-Dark accepts 2 clusters for this query. Additive reranking places B first
> (original rank 2), and B is also among the classifier's global top five. Protein
> length (129 residues, +2.459), predicted beta-strand fraction (10.9%, +2.341)
> and mean hydropathy (-0.472, +2.153) support B relative to their average training
> contributions.

This is a template for formatting fixtures, not measured evidence for an invented
cluster B. Use full bound descriptions in actual prose. In unavailable mode:

> PClass-Dark accepts A as its leading cluster. Additive reranking is unavailable
> because this library has no additive classifier; the original ordering is retained.

For no matches, begin "PClass-Dark finds no accepted cluster for this query" and
state whether supplemental suggestions exist. Never describe these suggestions
as accepted. Required prose order is enforced per query in multi-query reports.

## 18. Execution/error/configuration details

Default setup/build dependency failure uses the existing CLI error exit 1 and
prints an actionable reason on stderr. Invalid option combinations/types exit 2
through Typer. KeyboardInterrupt exits 130, preserving previous complete library
and report files. Success exits 0. Do not catch integrity errors and continue a
partially scored run. Cache locks use existing bounded locking conventions; report
FileLock timeout is zero and reports a busy destination without waiting forever.
Public APIs propagate documented exceptions without process exits or printing.

All source file inventories use {sha256,size_bytes}; source_reference preserves the
existing prepared reference object's field conventions. Manifest identity is the
canonical JSON identity of its explicit identity_inputs, never a self-referential
hash of a manifest containing its own ID. Content inventories and status belong
outside identity_inputs. Publication validates both identity and actual checksums.
Model numerical equality tests use existing float32 tolerances; extraction records
actual devices and numerical versions. Stable ranking is deterministic for equal
computed logits, but the specification does not promise identical near-tied ranks
across different numerical backends. Record such differences in device tests.
The additive recipe's id is identity(recipe), not an extra required recipe key.
The direct final-model ID binds the recipe's scientific values and actual device;
execution thread count is recorded and handled consistently with current fitting
resume contracts. Do not compare mutable free-memory/RSS estimates for identity.

setup-data locks/snapshots the ordinary prepared selection, imports its matching
additive companion, then updates additive-selection under the same guard. Inference
and building take one validated selection snapshot; recheck selected dependency
checksums before publishing a newly built library. Concurrent source changes
abort the build rather than relabeling old data. Externally supplied matrices and
manifests are read-only; source stat/digest checks surround streaming import.

For external imports, the source manifest contains the complete selected prepared
reference.source.identity and reference.files objects, raw-file digest/size,
source medoid/profile/composed-go snapshot inventory and the semantic-method
label. Require all bound member hashes in section 12 (or the corresponding new
reference revision) to match. Historical generator may remain explicitly unknown;
source provenance must not. No serialized Python object from an untrusted source
manifest is executed. Use only verified installed reference graph assets to derive
labels, then freeze plain JSON descriptions.

Build cache policy: a matching complete feature report can be imported only after
verifying all input sequence/preparation/head/precision identities. The existing
work/features.tsv is not an implicit default. If adopted during implementation,
copy it into an owned cache with a verified provenance record; otherwise re-extract.
Model import is similarly explicit and never inferred from work/ml-model. Existing
HARD weights can serve parity tests but cannot bypass new library binding checks.

The default hard recipe still requires a bound matrix for lifecycle consistency
and because the current target builder allocates targets even for alpha=0. Do not
optimize that allocation as part of integration unless scientific parity and
resource expectations are separately tested and documented.

When --no-additive is used for a build, additive-specific device/memory flags may
remain at defaults; reject nondefault values because they would be ignored.
--clean rejects additive recipe/device/memory overrides and performs only verified
cleanup. setup-data --additive-recipe without a source override reuses a compatible
frozen source and creates a new recipe-bound reference identity. Missing declared
structural heads in a new complete library are corruption; auto cannot load a
substitute packaged head silently. Old libraries may still use their existing
head-resolution behavior when additive is absent/off.

The lower-level legacy infer_queries API preserves its schema and accepted results.
The root CLI routes through the new integrated API, including --additive off;
that mode changes reporting schema/availability only, never original scientific
results. CLI help documents the new JSON schema and automatic report files, and
shows --no-reports for prior stdout-only workflows. Do not silently call old and
new APIs twice or run scoring separately for JSON versus Markdown.

## 19. Required implementation artifacts and test ownership

Create plain-JSON schema definitions for additive-source, recipe, reference,
selection, library additive block, integrated query/report and report completion/
journal under docs/schemas/, with matching explicit runtime validators in the
owning modules. JSON Schema files are documentation/tooling artifacts; no additional
runtime validator dependency is required. Each schema has a checked-in minimal
valid fixture and invalid-enum/missing-field/nonfinite producer tests. Freeze
actual artifact schemas only after these agree with sections 2–18.

Add tests/test_additive_reference.py, tests/test_additive_build.py,
tests/test_integrated_inference.py, tests/test_integrated_render.py and
tests/test_integrated_reports.py. Extend existing setup/library/CLI/cleanup tests;
retain all existing ML, memory and interpretation tests. Use small generated
references in ordinary tests; no test downloads, hundreds-of-MB fixture copies
or full-cohort training. One optional local-reference acceptance test verifies the
frozen default matrix/header/coverage digest and packaged binding. Do not label
that expensive test run if it was skipped. Build source-only artifacts in a clean
temporary copy to avoid packaging work/ and multi-GB caches.

Completion requires: synchronized index/parent/README/API/help/schema docs;
selected bundled source and recipe materialized with inventories; relevant tests
executed and results logged; clean output/transaction checks; documented CPU/GPU
coverage and old-library migration. There are no remaining design questions in
this specification; implementation failures requiring a changed contract must
revise this document with the concrete evidence rather than silently diverge.


## 20. Implementation record and reconciled interfaces

The code inspection during implementation identified that production signatures
already use library.v2. Preserve that schema and its scientific ID; do not reuse
v2 to mean a different format. New immutable integrated-library.v1 envelopes under
library-root/integrated/<id>/ own the additive model, frozen descriptions/training
metadata and structural heads, and bind the original signature and distribution
selection. library-selection.v3 selects that envelope alongside unchanged signature
paths. Integrated inventory validation is mandatory before using the model.
Existing v1/v2 readers and the legacy infer_queries API remain available.

Root setup/build commands enable additive by default. Existing prepare_data and
build_library Python defaults remain additive=False for backward compatibility;
explicit additive=True provides the integrated behavior. The initial frozen source
was imported and every score/pair validated through the six-column loader; the
managed retained reference has 3443 clusters and complete pair coverage. Source
historical generator remains explicitly unknown. No neighboring checkout is a
runtime dependency after freezing. Source-only packaging includes the frozen gzip
asset and portable envelope support.

Integrated --force is rejected with a fresh-root instruction to avoid mutating a
selected signature generation before supplemental fitting succeeds. Signature
research remains available via --no-additive --cross-validate. Selected integrated
inventories are checked during cleanup; additive feature/model caches and old
envelopes are conservatively retained in this release. They are not candidates for
automatic deletion until an extended reachability collector is implemented. These
are explicit operational limits, not changes to acceptance or scoring.

The borrowed-backend constructor changes no standalone extraction arithmetic;
raw prediction accepts the prior known structure.py hash
5414b040d300fb09598b5b7566fcdab36ecbe4fd8f43ef62501cf8a364fe336d
as compatible, while still rejecting unknown scientific implementations. Numerical
inference is CPU NumPy/SciPy. Full-cohort deployment is not automatically trained
as part of code implementation; synthetic build/infer acceptance covers the new
workflow. Actual test and packaging outcomes are recorded in the validation record below.


### Final validation record — 2026-11-08

Implementation revision: 1ece3d5c plus this uncommitted change on issue#42.
Full-suite snapshot: 811 passed, 5 skipped, with one existing spawned-worker test
blocked by sandbox socket creation. That exact test passed separately outside the
sandbox (1 passed). Existing GPU/CPU classifier and structural-head agreement
checks passed on the local GPU (3 passed). Final core setup/inference/production/
cleanup/path-migration regression run: 108 passed. Final focused additive,
standalone ML, CLI and setup run: 94 passed. The focused run includes direct fitting
without CV, source filtering, selector rollback, cache regeneration after a
simulated fitter revision, candidate preservation, selected-class explanation,
long-query fallback, file collision/recovery and release-envelope inventory.

Source-only wheel and sdist builds passed, as did installed help/numerical model
prediction checks with optional training libraries absent from sys.modules.
Ruff, whitespace, local-document links and schema/fixture JSON syntax passed.
No jsonschema runtime dependency was added; producers use explicit validators.
Source-only packaging avoids selected multi-GB development data and training caches.

Full frozen import validated 6207526 source pairs, retaining 5925403 pairs over
3443 prepared retained clusters. All medoids bind to the adopted reference.
Feature cache keys now bind the full extraction contract and requested device;
model cache keys bind recipe/reference/features, fitter code/versions, admitted
device and thread budget. Final publication rechecks source inventories and both
selected input snapshots under ordered guards. Alias/exclusion audits are stored
beside the managed feature table. JSON and detailed Markdown stream under the
shared temporary quota rather than accumulating full report strings.

A real offline GPU parity smoke used the existing saved HARD weights and read-only
production signatures. It accepted 37 clusters and reranked cluster_1096 from
original rank 4 to rank 1; only that global top-five suggestion was accepted.
Feature extraction used cuda:0 and numerical scoring CPU. The production selector
was unchanged. [Retained evidence](../research/notes/2026-11-08-integrated-reranking-smoke.md)
explains the explicit saved-weight harness; this is not a new production fit or
held-out performance result. No full-cohort retraining or deployment was performed.


## 21. Root ML command retirement — 2026-11-08

Explicit user follow-up supersedes the earlier root-command compatibility promise:
remove explore-ml, train-ml and predict-ml registrations and their adapter imports
from cli.py. Preserve all script files, standalone Typer apps, scientific APIs,
model/data schemas and numerical behavior. setup-data/build-library/infer remain
the production hierarchy; exploration remains available through its script/API.
No migration or retraining is needed. Existing root invocations now fail with the
normal unknown-command exit; documentation points to the script alternatives.

Move former root-command adapter tests to the actual standalone apps, verify root
help and rejection of all three removed names, and retain subprocess script-help
checks. Run CLI, exploration, fitting, interpretation and integration regressions.
This is an interface-only change and requires no GPU or full-data experiment.
Validation: 94 passed, 2 GPU tests skipped across CLI, exploration, final-model,
interpretation and integrated inference/build suites. Tests exercise real standalone
apps, reject all three root names and run all script entry-point help paths. Ruff,
local documentation links and git diff --check passed. No numerical or device
implementation changed; no new full-data/GPU experiment was required.


## 22. Managed feature-cache path repair — 2026-11-09

Bug: the integrated builder writes additive-features/<contract-id>/features.tsv
below the library root, but the experimental extractor rejects every output below
the default library root. The first production build therefore finishes signatures,
relative distributions and structural-head installation, then fails before feature
extraction and never publishes selection.json. The previous end-to-end fixture
mocked extraction and did not exercise this path admission rule.

Implemented repair: retain the managed cache layout and add private `_library_root`
context to the extraction API/path validator. Admit only the exact
<library-root>/additive-features/<64-lowercase-hex-id>/features.tsv family. This
context exempts only that library root from the experimental library-output guard;
prepared inputs, setup roots, model bundles, package resources, symlinks and input
hardlinks remain protected. Standalone extraction keeps its existing restrictions.
No CLI option, scientific formula, worker/device policy or artifact schema changes.
Existing code-bound extraction cache identities change with the implementation.

Validation gates: exercise real feature extraction, fitting, selected-envelope
verification, additive-required inference and report publication against a small
CPU cohort with only the PLM predictor replaced. Set the fixture library as the
default library root, cover resume and extraction failure without selector
publication, and retain standalone rejection and invalid managed-path tests.
Run feature extraction, additive build, inference and CLI regression suites; record
results here and in the progress log. Full-cohort production fitting is separate
from synthetic correctness validation.

Initial validation: 61 extractor/additive-build tests passed, including fresh and
interrupted/resumed real extraction, direct CPU fitting, envelope verification,
additive-required CLI inference and complete report receipts. The query test uses
real frozen structural heads with synthetic residue states; no PLM download is
needed. Standalone rejection, malformed managed paths, protected setup roots and
input symlink/hardlink rejection passed. Existing production cache path admission
also passed without starting extraction or changing artifacts.

Broader regression command: pytest -q tests/test_integrated_inference.py
tests/test_inference.py tests/test_cli.py tests/test_structural_features.py
tests/test_structure.py tests/test_production_library.py — 164 passed. Combined
with the extractor/build run: 225 passed. Ruff check/format, git diff --check,
local documentation links and Markdown fences passed. CUDA is unavailable in
this environment, so no GPU or full-cohort refit was run. The production selector
remains absent until the corrected ordinary build is resumed; this change does
not substitute previously fitted research weights for a production model.
