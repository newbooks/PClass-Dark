# SPEC-0025: Integrated PClass-Dark and additive classification evidence

- Date: 2026-11-08.
- Status: implemented; actual interfaces, validation and operational limits in SPEC-0026.
- Provenance: user-approved integration and report ordering in this conversation;
  existing specifications 0003, 0004, 0007 and 0020–0024.
- Terminology: SIAC means Similarity-Informed Additive Classifier. Reports identify
  the actual classifier configuration (HARD or SIAC); do not label HARD weights
  as similarity-trained. User references to SAIC refer to this additive component.

## Purpose and scope

Use PClass-Dark acceptance and additive reranking together as supplemental
classification evidence. Production lifecycle is setup-data → build-library →
infer. Exploration and cross-validation are not production stages or prerequisites.
Preserve scripts/explore-ml.py, scripts/train-ml.py and scripts/predict-ml.py,
including their reusable Python interfaces and independent operational behavior.
Explicit user follow-up on 2026-11-08 retires the three root ML commands;
standalone scripts and Python APIs remain supported (SPEC-0026 section 21).

This specification supersedes SPEC-0007's prohibitions on interpretation and
supplemental ranking only. Acceptance, window coverage, representative selection,
residue mappings and B/C formulas retain their existing definitions.

## Shared preparation and model publication

setup-data owns the cluster membership snapshot and optional frozen cluster
similarity artifact. Store source provenance, cluster vocabulary, matrix/pair
schema, missing-pair and distance/similarity conventions, and content checksums.
Validate against the prepared membership identity, not just cluster names. No
runtime dependency on neighboring repositories. Prepare full-sequence features
using SPEC-0021 contracts, excluding sequence_charge_decoration for classifier
training. Keep deduplication/conflict decisions and exclusions auditable.

build-library trains the selected additive configuration on eligible full-sequence
examples and publishes its checked bundle inside the immutable library generation.
No exploration run is required: separate reusable final fitting from the existing
CV-derived adapter, accepting explicit validated data and a frozen training recipe.
Record settings, epoch budget, seed, actual device, target vocabulary, feature and
preparation identities, and any prior evaluation provenance. Existing validated
HARD settings are a starting recipe; do not silently switch to SIAC. Configuration
selection must be explicit and its rationale documented before production fitting.
Stage all new artifacts and publish atomically, preserving prior valid generations.

## Inference and ranking contract

1. Compute existing PClass-Dark accepted candidates and window evidence.
2. Obtain reference-compatible full-sequence descriptors and additive scores.
3. Rank all accepted candidates represented in the classifier vocabulary by their
   additive logits, with deterministic cluster-ID tie breaking. Preserve original
   PClass-Dark rank and every B/C/supporting-window record.
4. Mark membership in the additive classifier's global top five. The intersection
   is highlighted, not the complete reranking candidate set.
5. Show global top-five classifier suggestions outside the accepted set separately;
   these do not become accepted predictions.

Softmax remains normalized over the classifier's full target vocabulary. Do not
renormalize over accepted candidates or present a product of B/C and softmax as a
probability. Classifier contribution signs remain relative to each chosen-cluster
curve's average training contribution (SPEC-0024).

Missing classifier, unsupported sequence lengths and candidates absent from its
vocabulary need explicit availability states and reasons. Do not interpret missing
scores as low support or rejection. Retain PClass-Dark results if supplemental
inference is unavailable under a documented compatibility policy. Corrupt present
artifacts must not be silently treated as absent. Preserve bounded batching,
resource admission, CPU fallback and shared model loading where extraction
contracts permit. Window pooling cannot substitute for full-sequence features.

## Reports: one structured result, two representations

Provide comprehensive versioned JSON and a human-readable Markdown report generated
from the same structured result. Terminal summaries use the same ranking and
explanation fields. Keep scientific calculations outside rendering code.

### Required human-readable order

The first prose paragraph, immediately after an optional query heading, gives the
combined classification: which clusters PClass-Dark accepts, which is first after
additive reranking, whether it is also in the classifier's global top five, and the
strongest observed additive feature evidence supporting that selected cluster.
For many candidates, summarize the count and leading result here; list all later.
Do not lead with score definitions, provenance, a table or model configuration.
Use measured feature values and actual signed own-class contributions. If no
positive observed feature support exists, state that rather than fabricate support.

When there are no accepted candidates, the paragraph states that PClass-Dark made
no accepted classification. Additive-only suggestions remain explicitly
supplemental. When additive evidence is unavailable, summarize the PClass-Dark
classification and the reason reranking could not be provided.

Follow the paragraph with:

1. Detailed PClass-Dark analysis: accepted clusters, original ordering, A/B/C
   evidence, representative and supporting windows, and library coverage.
2. Additive reranking analysis: all accepted candidates with original/new ranks,
   logits, global softmax scores and top-five agreement; global top-five table
   with cluster IDs and descriptions shortened to 24 displayed characters.
   Explain disagreements and distinguish window-based evidence from full-sequence
   feature evidence. Include selected-cluster feature values and centered supports
   for all 18 features, with strongest positive and negative observed terms in prose.

Feature evidence must explain the reranked selection, not blindly the globally
highest additive prediction when that prediction was not accepted by PClass-Dark.
Retain full cluster descriptions elsewhere; shortening applies only to compact
summary tables. State that model evidence is statistical, scores are uncalibrated,
and the two methods are not independent experimental confirmations.

### Comprehensive JSON contents

Record query identity/length/sequence digest; library, preparation and classifier
identities/configuration; execution and availability; original accepted matches
with all window evidence and original ranks; supplemental scores and reranked
positions for every accepted candidate; classifier-global top five; intersection
membership; chosen classification and its selection rule; raw feature values,
imputation and out-of-support flags; selected-cluster own-class feature terms,
bias/logit and centering reference/count. Preserve full precision and full
annotations. Record ties and unscored candidates consistently.

Do not drop acceptance records or replace B/C values when adding classifier fields.
Use a versioned report schema with explicit null/reason fields for unavailable
supplemental evidence. Human prose must be traceable to these numerical fields.

## Implementation specification

[SPEC-0026](0026-integrated-additive-implementation.md) proposes concrete schemas,
artifacts, recipe, API/CLI changes, compatibility policies, publication rules,
resource limits and acceptance gates. It is now an implementation-ready contract after source/data readiness review;
implementation and its documented operational limits are recorded there.

## Validation and completion expectations

Test shared-reference mismatch rejection, frozen matrix validation, full-data fit
without exploration artifacts, atomic library publication and reuse, old/partial
library handling, and unchanged PClass-Dark acceptance/window evidence. Verify
reranking across all accepted candidates, top-five intersection/disagreement,
absent labels, ties, no accepted results, unavailable supplementary inference,
and selected-cluster explanation reconstruction against independent examples.

Test Python API/CLI parity, JSON/human/terminal consistency, opening paragraph
ordering, escaped annotations/IDs, multi-query inputs, output collisions and late
failures. Validate CPU and resource admission; exercise GPU where available.
Report ranking benefits separately on held-out candidate sets; do not infer
combined accuracy from standalone cluster CV. Implementation and actual validation outcomes are recorded in SPEC-0026.
No new held-out accuracy experiment is claimed by this requirements document.
