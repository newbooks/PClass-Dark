# Integrated classification and additive reranking

`setup-data`, `build-library` and `infer` now provide a production path for
PClass-Dark acceptance with supplemental additive evidence. The independent
exploration/training/prediction scripts and Python APIs remain available. Their
three root registrations have been removed; use the scripts for standalone work.
Production fitting requires no exploration directory or cross-validation reports.

```bash
pip install 'pclass-dark[ml,embeddings]'
pclass-dark setup-data --prepare-only
pclass-dark setup-data --offline
pclass-dark build-library --offline
pclass-dark infer --fasta queries.fasta --offline
```

The first setup command prepares reference membership and imports the frozen
cluster medoid similarity table; it loads no PLM. The second prepares embeddings.
The build prepares structural heads, extracts full-sequence features, trains the
additive model on eligible unique sequences and selects the integrated generation
only after all components verify. Numerical fitting prefers an admitted GPU;
`--additive-device cpu` explicitly selects CPU. Numerical prediction uses CPU;
PLM feature extraction uses infer's `--device`.

Builds retain resumable feature and model caches under the library root. If an
additive stage fails, rerun `pclass-dark build-library --offline` with the same
build options; completed signatures and distributions are reused. Do not use
`--force` to resume. A first build publishes `selection.json` only after the full
integrated generation verifies, so inference cannot use an unfinished build.
The managed feature cache is admitted only for the builder's exact cache family;
standalone feature extraction still rejects outputs inside protected data roots.

The shipped initial recipe is **HARD**, selected from the earlier experiment,
not similarity-trained SIAC. SIAC is available through an explicit complete recipe
using `configuration=siac` and `0<alpha<1`. Both share the additive architecture.
The component is labeled with its actual configuration in reports.

## Shared data and recipes

Setup uses the repository/package frozen table and verifies its original cluster
membership, medoids and reference snapshot. It never reads `work/` implicitly or
reaches into a neighboring checkout. A different reference requires both
`--cluster-similarity table.tsv` and `--cluster-similarity-manifest source.json`.
`--additive-recipe recipe.json` supplies a complete validated recipe; see
[the implementation specification](specs/0026-integrated-additive-implementation.md)
and [recipe schema](schemas/additive-recipe.v1.json).

The 18 modeled descriptors exclude `sequence_charge_decoration`. Features are
quantized to three decimals; identical sequences sharing one target are collapsed,
and identities carrying conflicting targets are excluded with an audit.
Training does not change signature acceptance or B/C formulas.

## Inference and report files

For `queries.fasta`, inference writes:

- `queries-classification.json`: comprehensive structured evidence.
- `queries-interpretation.md`: human-readable analysis.
- `queries-classification.complete.json`: completion receipt with file checksums.

It also prints the classification summary, accepted-candidate ranking and global
top-five additive table. For `--seq`, the default prefix is `query` in the current
directory. Existing files are never overwritten.

```bash
pclass-dark infer --fasta queries.fasta --report-prefix results/run1
pclass-dark infer --fasta queries.fasta --format json --no-reports > results.json
pclass-dark infer --seq ACDEFGHIK --format markdown --no-reports
```

Each human report opens with the accepted classification, additive reranking and
supporting observed features, then gives detailed PClass-Dark evidence and additive
analysis. The feature explanation concerns the selected **accepted** cluster,
which may differ from the classifier's global first choice. Accepted clusters in
the global top five receive additional feature explanations.

All accepted candidates are reranked by additive logit when scored; unscored
candidates follow in their original order. Original PClass-Dark ranks, B/C values
and supporting windows remain in the JSON. The classifier's global top five are
also shown, with descriptions capped at 24 characters in compact tables. A global
suggestion outside acceptance does not become an accepted classification.

Softmax scores cover the full classifier vocabulary and are not calibrated
probabilities. Feature support is relative to each curve's mean training
contribution. Window-based signature evidence and full-sequence additive evidence
are complementary model evidence, not independent experimental confirmations.

## Compatibility and operational limits

`infer --additive auto` is the default: older libraries without this component
retain original acceptance with an explicit unavailable reason. `required` fails
if supplemental inference is unavailable; `off` bypasses it. Corrupt present
components fail rather than being treated as absent. Additive extraction currently
supports up to 100,000 residues; longer valid signature queries retain their
PClass-Dark result under auto mode. Unexpected extraction failures abort reporting.

Use `setup-data --no-additive` and `build-library --no-additive` for the independent
signature workflow. Existing Python `prepare_data` and `build_library` APIs retain
`additive=False` defaults for compatibility; pass `additive=True` to opt in.
`infer_queries` and its legacy renderer remain unchanged; the CLI uses the new
`infer_integrated_queries` API and integrated JSON schema.

Integrated builds currently reject `--force` on an existing workflow: use a fresh
library root through the Python API for a forced full rebuild. Signature research
remains explicit with `build-library --no-additive --cross-validate`. Cleanup audits
the selected integrated inventory; additive caches and unselected envelopes are
conservatively retained rather than automatically reclaimed.

## Python interfaces

```python
from pclass_dark.inference import QuerySequence
from pclass_dark.integrated_inference import infer_integrated_queries
from pclass_dark.integrated_render import write_integrated_reports

with infer_integrated_queries(
    [QuerySequence('query', 'ACDEFGHIK')],
    library_dir='data/library', additive='auto', offline=True,
) as run:
    with open('result.json', 'w') as js, open('result.md', 'w') as md:
        write_integrated_reports(run, js, md)
```

Python callers own publication of their supplied streams. The CLI adds staging,
no-overwrite checks and recoverable multi-file publication. Source APIs also expose
`ml_model.fit_reference_model` and `score_candidates`; imports never start fitting,
workers, downloads or model loading.
