# PClass-Dark

**Protein Function Classification Beyond Nearest Neighbors**

PClass-Dark is a protein function classification research project for proteins that lack reliable nearest-neighbor matches in protein language model (PLM) embedding space.

The central idea is to identify **conserved functional signatures across protein function clusters**. Each cluster may carry functional information in a different subset of embedding dimensions. By selecting dimensions that are both consistent within a cluster and informative against a background, PClass-Dark aims to reduce protein-specific variation and recover signals obscured in the full embedding space.

PClass-Dark complements [PClass](https://github.com/newbooks/pclass), which uses learned PLM embeddings and approximate nearest-neighbor (ANN) search to infer function from annotated reference proteins.

## Installation and Interfaces

The initial package version is **0.0.1** and requires Python 3.10 or newer.
Install from this checkout:

```bash
git lfs install
git lfs pull
pip install -e ../profunc
pip install -e .
pclass-dark --help
pclass-dark --version
```

The CLI uses a command/subcommand hierarchy: `pclass-dark <command> [options]`.
Use `-h` or `--help` after any subcommand to show its options.

`setup-data` prepares references and frozen PLM embeddings. `build-library`
discovers and evaluates cluster signatures. The same entry
point is available through `python -m pclass_dark`.

Reusable functionality lives in the importable `pclass_dark` package. Data
preparation, signature discovery, library loading, and signature scoring expose
Python APIs so other code can integrate them directly.

```python
import pclass_dark

print(pclass_dark.__version__)
```

Core dependencies include profunc, NumPy, SciPy, Typer, and filelock. Install the optional
embedding stack with `pip install -e '.[embeddings]'`. For development, use
`pip install -e '.[dev,embeddings]'` and run `pytest`.

## Shared Data and Distribution

In an editable/source checkout, default paths come from the imported package's
repo location, not the directory you run the command from:

- `<repo>/data/setup-data/`: prepared references and embeddings.
- `<repo>/data/library/`: signatures, C/B distributions and optional research.

These commands work from the repo root, `work/`, or another directory:

```bash
pclass-dark setup-data --workers 8
pclass-dark build-library --workers 8
```

Both commands default to `--workers 8`; use `--workers 1` for serial execution.

`setup-data` supports explicit path options; `build-library` uses the default
paths. Python APIs accept custom paths; relative paths use your current directory. Clearing `work/` no longer
removes the shared artifacts. Existing custom `work/` outputs are not moved
implicitly; use the Python API to keep building libraries at custom paths.

Completed shared artifacts are tracked with Git LFS. `python -m build` includes
only the selected complete bundles in the wheel and source archive, validating
manifests/checksums and coordinating with active writers. Checkpoints, archives,
staging files, locks and model weights are excluded. A stage without a selection
contributes no generated data yet; building does not run the scientific commands.
See [data instructions](data/README.md) and [SPEC-0006](docs/specs/0006-shared-artifacts.md).

Published signatures use `data/library/<library-id>/clusters/<cluster-id>/final.json`,
for example `clusters/cluster_0001/final.json`. Relative-score and research
bundles use the same cluster directory names. Older hexadecimal layouts remain
readable. To migrate an existing selection without refitting, run
`python scripts/migrate_library_paths.py data/library` from the repository root.
The operation validates the selected pair and preserves its original generation
under `data/library/.archives/`; those immutable backups share unchanged payloads
with the migrated generation and must not be edited in place. See
[SPEC-0011](docs/specs/0011-readable-cluster-paths.md).

A normal installed wheel reads bundled artifacts through the same APIs, including
`load_selected_library()` without a path. New writable output goes under
`$XDG_CACHE_HOME/pclass-dark/` or `~/.cache/pclass-dark/`; an existing cache selection
takes precedence over bundled data. Package files are never used as writable
output. Git LFS is needed for obtaining LFS files from Git, not for reading
artifacts already included in an installed wheel.

## Prepare Data and Embeddings

```bash
# Offline: validate references, deduplicate, filter, window, and define C/B sets.
pclass-dark setup-data --prepare-only --workers 4

# Prepare and embed; use cached model assets or download the pinned checkpoint.
pip install -e '.[embeddings]'
pclass-dark setup-data --device auto --batch-size 8

# Require an already cached checkpoint; use CPU explicitly if desired.
pclass-dark setup-data --offline --device cpu
```

The default output is `<repo>/data/setup-data/`, independent of your current
directory; change it with `--output-dir`.
`--min-clusters 10` keeps clusters with at least 10 distinct full sequences after
deduplication. Long proteins produce 1022-residue terminal windows named
`N-<accession>` and `C-<accession>`. Five seeded folds are assigned separately
within each target cluster C. Background B pools fixed samples of up to 200
windows per other retained cluster. N/C windows and identical windows are allowed
in different folds; these folds measure window-level internal validation.

Embedding uses pinned ESM-2 150M final-layer hidden states and averages only
residue positions, using both attention and special-token masks. Each window
produces 640 float32 values. Automatic device selection prefers CUDA, with CPU
fallback if CUDA initialization fails.

Re-run the same command to resume. Completed 64-window shards are checksummed
and reused; interruption can repeat only the active shard. After an out-of-memory
error, lower `--batch-size` and re-run. Complete bundles are verified before
reuse. `--force` archives matching prior outputs and starts intentional
recomputation. `selection.json` identifies the completed prepared/embedding pair.

The Python API provides the same workflow:

```python
from pclass_dark.setup_data import prepare_data
from pclass_dark.cluster_sets import load_sets

result = prepare_data(prepare_only=True, workers=4)
sets = load_sets(result.prepared_dir)
```

Completed selected shared bundles are copied into package resources during
builds. Temporary checkpoints and model weights are excluded. Prepared data and vector consumers
do not need the embedding-generation dependencies. See
[SPEC-0003](docs/specs/0003-setup-data.md) for schemas, integrity checks, and
the release layout.

## Profunc Reference Data

`setup-data` imports the installed **profunc** library and resolves its active
publication (current-directory clustering when present, otherwise profunc's
packaged default). It reads memberships directly and validates the source
sequence FASTA against that publication's dataset manifest. PClass-Dark's
legacy reference copies have been removed; there is no bundled-reference fallback.

Profunc runtime bundles omit FASTA. An editable profunc installation normally
retains it in the original dataset generation. Otherwise provide it explicitly:

```bash
pclass-dark setup-data --reference-fasta /path/to/source/accession.fasta
pclass-dark build-library --workers 4
pclass-dark explain-signature --list
```

After updating/building profunc, rerun `setup-data` and `build-library` in that
order. Changed memberships or sequences produce new identities; matching inputs
are reused. Neither command upgrades the installed profunc package itself.
Descriptions are computed through profunc at listing time and checked against
the signatures' source hashes. Legacy signatures without this binding show `—`
until rebuilt. See [reference instructions](docs/reference-data.md).

## Build a Signature Library

```bash
# Default: one signature fitted using all cluster members, plus C/B distributions.
pclass-dark build-library --workers 4

# Build a smaller scope; backgrounds still use all original source clusters.
pclass-dark build-library --cluster cluster_0945

# Optional research: five rotations for a selected cluster, reusing production.
pclass-dark build-library --cross-validate --cluster cluster_0945
```

The default output is `<repo>/data/library/`, independent of your current directory. Discovery uses a CPU float64 backend,
bounded multiprocessing, and existing 640-dimensional embeddings. It requires
no model download or embedding dependencies. Each free worker immediately takes
the next cluster. Re-running resumes verified checkpoints; compatible final fits
from older six-fit runs are imported without repeating discovery when their
provenance and algorithm dependencies can be proved. `--force` explicitly
archives and recomputes the requested work.

Each production signature is fitted using all C windows and the fixed capped B.
Dimensions are ranked by individual training AUROC; the top two are protected.
Backward selection accepts the first strict training AUROC improvement in rank
order and restarts. Cached totals, bounded partial sums, reusable buffers, and
numerical fallbacks accelerate scoring. The acceptance threshold maximizes
training balanced accuracy. Default metrics are **in-sample**, with no implied
held-out validation or stability estimate.

After the final signature is fixed, `build-library` scores its exact C and B
pools and preserves sorted composite log-score distributions. For a query:

- **B upper-tail fraction:** how often sampled nonmembers score at least as high.
- **C percentile:** the fraction of known members scoring at or below the query.

These are empirical comparisons, not membership probabilities or density-based
typicality. Final C is in-sample, and B was also used during selection. A high C
percentile can occur above every observed member score. Background evidence and
member compatibility should be interpreted together; automatic assignment,
combined ranking/calibration, and query N/C aggregation remain future work.

Future top-cluster query results will use **A — Acceptance** (Yes/No),
**B — Background score ↓** (background upper-tail percentage; lower is better),
and **C — Cluster score ↑** (member percentile; higher score rank among members).
The planned initial ranking orders accepted clusters by ascending B and displays
C as supporting evidence. These are not membership probabilities. See
[SPEC-0007](docs/specs/0007-query-ranking-presentation.md) for the planned
presentation and tie handling; this ranking is not yet implemented.

```python
from pclass_dark.library import build_library
from pclass_dark.relative_scores import load_selected_library

result = build_library(workers=4, memory_mib=4096)
pair = load_selected_library()
# query_vectors: finite (n, 640) embeddings; query_contract describes their pipeline.
log_scores, accepted = pair.library.score(
    "cluster_0945", query_vectors, embedding_contract=query_contract
)
evidence = pair.relative.evaluate("cluster_0945", log_scores)
print(evidence.background_tail_fraction, evidence.cluster_percentile_le)
```

Production scientific outputs and the `relative-scores/` companion are selected
as a verified pair by `selection.json`. Distributions can be backfilled from
completed signatures without PLM inference, threshold fitting, or discovery.
`--cross-validate` adds separate `research/` reports and `research-selection.json`:
five fold signatures, stability summaries, training/held-out-positive metrics,
and OOF scores. It does not modify production parameters or its selector, and
production remains readable while research runs. Research is window-level and
reuses B; it is not independent homology-controlled validation.

After each production target's final result and distributions are durable,
disposable Gaussian/prefix caches are removed. Research cleans each completed
fold. Unfinished fits retain their caches. Complete results, profiles, compact
checkpoints and C/B distributions survive cleanup. The maintenance API
`pclass_dark.library_cleanup.clean_library_cache(output_dir)` also recognizes
legacy and new completion gates. Final B arrays total approximately 5.5 GiB for
the current reference collection; private/staged copies and reports add storage.

Completed selected production, relative and optional research artifacts are
automatically bundled under `pclass_dark/resources/library/` with their selectors. Loading and scoring
need neither setup-data matrices nor PLM imports or writes to installed files.
Private checkpoints and model weights are excluded. See
[SPEC-0004](docs/specs/0004-build-library.md),
[SPEC-0005](docs/specs/0005-relative-score-distributions.md), and the
[score-interpretation notes](docs/research/notes/2026-10-27-relative-score-reasoning.md).

## Clean Obsolete Dataset Generations

Cleanup validates an existing full library and never builds missing results.
The CLI uses default data/library paths and internal numerical settings
(1024 MiB workspace, epsilon and sigma floor both `1e-8`). Custom paths and
settings remain available through the Python API.

```bash
# Validate the existing library, then permanently remove eligible old generations.
pclass-dark build-library --clean
```

`--clean` requires an existing complete library; missing or invalid selections fail
without building. It rejects `--force`, `--cross-validate`, and `--cluster`.
`--clean` removes obsolete preparations and their exclusive completed signatures,
C/B distributions, embedding generations and checkpoint caches. Shared embeddings,
the selected pipeline, current-preparation experiments and resumable work are
retained. Rollback/migration backups under `.archives/` become obsolete once the
selected replacement is fully validated and are eligible for cleanup; their
references no longer pin old datasets. Legacy/migration caches and ambiguous data are protected;
unbounded or corrupt ownership blocks cleanup with an error. Python
`plan_cleanup()` provides a preview with
removal/protection reasons and logical byte estimates. It uses the current
validated selections, regardless of proposed fitting settings.

Do not combine cleanup with `--cluster`, `--force`, or
`--cross-validate`. Normal builds continue to retain data. A cleanup failure does
not invalidate an already published library. If interrupted cleanup leaves a
journal, rerun `build-library --clean` to
recover before validation. Other commands and packaging refuse pending cleanup.
POSIX descriptor-relative IO is required for deletion/recovery. Planning caps are
64 MiB per JSON document (coordinator journal or inventory sidecar) and
1,000,000 entries / 512 MiB serialized inventory per tree. There is no aggregate
entry or inventory-size cap across deletion candidates. Inventories and hard-link
accounting use temporary disk storage; execution publishes bounded recovery
sidecars before detaching any data. After commit, cleanup directly removes
obsolete directory trees without relinking, re-hashing their contents or syncing
each file deletion. Interrupted committed removal resumes from remaining trees.
Temporary/recovery disk space scales with the
scanned inventory; a per-tree/document cap or disk-space failure stops safely.

Only files in the supplied writable setup/library roots are considered. Cleanup
does not discover external consumers. Keep old experiments elsewhere if needed.
Tracked deletions remain for Git review; `.git/lfs` objects, Git history and remote
copies are not removed. Retained hard links can also retain disk blocks, so
estimated reclaimed space can differ from logical size.

```python
from pclass_dark.data_cleanup import plan_cleanup, execute_cleanup, recover_cleanup
from pclass_dark.library import build_library

plan = plan_cleanup("data/setup-data", "data/library")
print(plan.to_dict())
# Execute only this unchanged, freshly revalidated plan:
result = execute_cleanup(plan)
# Or integrate cleanup with a successful full build:
cleaned = build_library("data/setup-data", "data/library", clean=True)
# Recovery of pending cleanup is also importable:
recover_cleanup("data/setup-data", "data/library")
```

See [SPEC-0013](docs/specs/0013-build-library-clean.md) for dependency, journal and
retention contracts. No model or profunc source data is needed for cleanup.

## Explain a Signature

```bash
pclass-dark explain-signature --cluster cluster_0001
pclass-dark explain-signature --cluster cluster_0001 --detailed
pclass-dark explain-signature --cluster cluster_0001 --format markdown > signature.md
pclass-dark explain-signature --cluster cluster_0001 --format json > signature.json
```

The report explains window membership, individual dimension evidence, candidate
selection and removal history, Gaussian similarity and the composite threshold,
in-sample metrics, and reference distributions for ABC match evidence. It uses
the selected library from the shared default location; use `--library-dir` to
select another library root. It does not need setup-data matrices, model weights
or network access, and does not evaluate a query or assign probabilities.
The command validates only the requested cluster's signature, trace and C/B
arrays, together with their manifest/provenance bindings. The shared compressed
profile table is still checksummed; locating a late cluster requires sequential
lookup. Unrelated clusters are outside this validation scope. `--detailed`
includes all profiles, prefixes and recorded removal trials.

```python
from pclass_dark.explanations import explain_signature, render_signature_explanation

report = explain_signature("cluster_0001")
print(render_signature_explanation(report), end="")
evidence = report.to_dict()  # Detached structured data for integrations.
```

The report is immutable; rendering performs no artifact reads. Relative-score
objects also provide `summarize(cluster_id)` for verified, bounded-memory C/B
quantiles. `load_selected_library(cluster_id="cluster_0001")` and
`load_library(path, cluster_id="cluster_0001")` expose targeted loading to Python
integrations; omitting `cluster_id` retains full-library validation. See [SPEC-0008](docs/specs/0008-explain-signature.md) for the report
schema, availability rules and interpretation.

### List Signature Summaries

```bash
pclass-dark explain-signature --list
pclass-dark explain-signature --list --format markdown > signatures.md
pclass-dark explain-signature --list --format json > signatures.json
```

The listing includes every selected cluster in alphabetic order, including
unsupported signatures. Columns are Cluster, Description, Status, C_windows, B_windows,
Dimensions, AUROC, Recall%, Specificity% and Precision%. Status is omitted from
nonempty text/Markdown tables when every signature is `ok`; JSON always retains
it. Description uses profunc's composed-GO function description, with the
annotation prefix/article removed. Its display is limited to 24 characters
including `...`; JSON retains the full text. Clusters without a resolved
function description show `—`. Text/Markdown use
three decimal places for AUROC and percentages, integer counts and `—` for
unavailable metrics. Text columns align by display width; a narrow terminal may
still wrap long lines. JSON preserves unrounded fractions and availability
reasons. These are final **in-sample reference metrics**, not query-match scores. The report ends with 20 equal-width distributions for AUROC,
recall, specificity and precision, counting one observation per cluster. AUROC
bins span 0–1; rate bins display 0–100%. Internal bins include their lower edge
and exclude their upper edge; the final bin includes the maximum. Unavailable
values are excluded and counted separately. Binning uses unrounded values; JSON
and the Python listing include the edges, counts and availability totals.

Choose either `--cluster` or `--list`. `--detailed` applies only to individual
explanations. The list verifies production manifests and saved signature summaries;
it does not read profiles, traces, relative-score arrays or research artifacts.

Descriptions come from the installed profunc modules and validated active library.
Listing requires profunc but no Internet access. Different reference hashes or
legacy signatures without a reference binding show `—` with a JSON reason,
rather than matching labels by cluster ID alone. Description text contains
spaces; JSON provides full structured fields for scripts.

```python
from pclass_dark.explanations import list_signature_summaries, render_signature_summaries

listing = list_signature_summaries()
print(render_signature_summaries(listing), end="")
rows = listing.to_dict()["rows"]
```

See [SPEC-0009](docs/specs/0009-list-signature-summaries.md) for the schema and
summary validation scope.

## Motivation

PLM embeddings encode multiple properties of protein sequences. A distance computed across the full representation may dilute a functional signal carried by only a subset of dimensions. The dimensions most informative for one function may also differ from those informative for another.

PClass-Dark explores the hypothesis that:

> Proteins sharing a biological function may preserve characteristic patterns in a subset of PLM embedding dimensions, even when their overall embedding vectors are substantially different.

Instead of requiring proximity to an individual reference protein, the project evaluates compatibility with a signature learned from a functional cluster. Whether this improves classification for proteins lacking reliable neighbors is a research question to be tested.

## Core Approach

The planned workflow adapts the feature-profiling and signature-composition concepts studied in the neighboring ProStruct repository, replacing structural features with PLM embedding dimensions.

### 1. Function Space Clustering

Functional clusters are curated from Swiss-Prot accessions annotated with Gene Ontology molecular function (GO MF) terms. The separate **profunc** project implements this clustering using Lin semantic similarity between GO terms.

Calibration against four-number Enzyme Commission (EC) classifications indicates comparable or finer cluster resolution. Each cluster represents a group of proteins with related molecular functions; cluster granularity determines the specificity of the resulting predictions.

### 2. PLM Embedding Representation

Protein sequences are represented by fixed-length PLM embedding vectors. Each dimension becomes a candidate feature for cluster profiling and signature discovery.

The embedding model, sequence pooling, and preprocessing must be consistent between reference and query proteins. A signature is specific to the representation in which it was learned.

### 3. Cluster and Background Profiling

For each functional cluster, individual embedding dimensions are profiled against a defined background of proteins outside that cluster. Background composition matters: separation from unrelated proteins may be easier than separation from proteins with closely related functions.

Following ProStruct's `scripts/profile_clusters.py`, candidate dimensions can be characterized by:

- **Relative conservation:** Within-cluster variation compared with background variation.
- **Distributional separation:** The difference between cluster and background means relative to their spreads.
- **Compatibility discrimination:** How well a score based on the cluster's characteristic values separates cluster members from background proteins, measured by the area under the receiver operating characteristic curve (AUROC).

The starting compatibility model is a Gaussian-shaped score centered on the cluster mean, with width set by the within-cluster standard deviation and a numerical floor. Values near the cluster mean receive higher scores. Low variation alone does not establish functional relevance; a dimension must also help distinguish the cluster from its background.

### 4. Signature Composition and Selection

Following ProStruct's `scripts/compose_signature.py`, dimensions are ranked by their individual compatibility AUROC and combined into candidate signatures. The starting composite score is the equal-weight arithmetic mean of the selected dimensions' Gaussian compatibility scores.

Incrementally adding ranked dimensions reveals whether a larger signature improves separation. Backward removal then tests whether discarding individual dimensions improves separation as measured by composite AUROC. A dimension that performs well alone may contribute little when combined with others.

The resulting signature records the selected dimensions, their cluster-specific means and spreads, background statistics, and selection diagnostics. ProStruct's precise selection rules provide an initial reference; their suitability for high-dimensional PLM embeddings remains to be evaluated.

### 5. Cluster-Level Classification and Validation

For a query protein, its embedding is scored against candidate cluster signatures. The intended prediction is a **functional cluster**, supported by compatibility scores and validated acceptance criteria.

Compatibility scores are not calibrated probabilities of cluster membership, and raw scores may not be directly comparable across clusters. Cluster-specific thresholds, ambiguous assignments, and rejection of unsupported predictions require validation.

Feature ranking, parameter fitting, subset selection, and threshold tuning on the same proteins produce descriptive, in-sample results. Independent evaluation must keep these steps within the training data and assess performance on held-out proteins, with controls for homology leakage. Evaluation should measure precision, recall, and functional coverage, particularly among proteins without reliable nearest-neighbor matches.

## PClass vs. PClass-Dark

| Feature | PClass | PClass-Dark |
|---|---|---|
| Representation | Learned PLM embeddings | PLM embeddings |
| Reference information | Individual annotated proteins | Predefined functional clusters and backgrounds |
| Classification principle | Nearest-neighbor similarity | Cluster-specific functional signatures |
| Features used | Full learned embedding representation | Selected informative embedding dimensions |
| Primary prediction | Function inferred from reference neighbors | Functional cluster |
| Intended strength | Proteins with reliable functional neighbors | Proteins lacking reliable nearest-neighbor matches |

The methods are complementary. PClass-Dark aims to extend functional coverage beyond the region where nearest-neighbor predictions are reliable, while maintaining acceptable precision.

## Project Status and Scope

**Research and early development.** ProStruct provides preliminary work on structural feature signatures; PClass-Dark will investigate whether the same profiling and composition strategy transfers to PLM embeddings. Data preparation and signature discovery/scoring are implemented. Production calibration and independent predictive benchmarks remain to be established.

PClass-Dark is intended to predict membership in functional clusters represented by the reference dataset. Its goal is to recover known functional information beyond reliable nearest-neighbor similarity. It does not inherently discover biological functions absent from the reference annotations.

## Related Projects

- **[PClass](https://github.com/newbooks/pclass):** PLM-based protein function classification using nearest-neighbor search.
- **ProStruct:** Preliminary structural feature profiling and cluster-signature composition, especially `scripts/profile_clusters.py` and `scripts/compose_signature.py` in the neighboring repository.
- **[profunc](https://github.com/newbooks/profunc):** Functional clustering of reference proteins using GO semantic similarity.

## License

PClass-Dark is released under the [MIT License](LICENSE).

---

*PClass-Dark: Finding conserved functional signals beyond nearest neighbors.*
