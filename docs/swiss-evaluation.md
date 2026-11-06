# Swiss-Prot classifier comparison

Evaluate a Swiss-Prot FASTA against direct MF annotations from the installed
profunc dataset. The comparison uses PClass's closest embedding neighbor and
Dark's accepted cluster with the smallest background-tail fraction, with profunc
AMB Lin semantic similarity to truth. Self matches are retained: this is a
potentially in-sample reference-cohort diagnostic, not held-out performance.

Run from the results directory:

```bash
python /path/to/PClass-Dark/scripts/evaluate_swiss.py sequences.fasta
```

The default output is `swiss_evaluation.tsv` in the working directory, with
`Query_id`, `PClass_go`, `Embedding_d`, `Pclass_match`, `Dark_go`, `Dark_Bscore`,
and `Dark_match` columns. Undefined/absent values are `NA`; zero remains zero.
The TSV schema and prediction selection match [external evaluation](external-evaluation.md).

## Inputs and dependencies

Use uncompressed UTF-8 FASTA with unique primary accessions, either `P12345` or
`sp|P12345|ENTRY` as first header token. The original token is retained in output.
Isoforms, secondary accessions absent from the annotation mapping, `tr|` records,
and duplicate normalized accessions fail. Sequences follow the common classifier
alphabet `ACDEFGHIKLMNPQRSTVWYBOUXZ`. Every query needs nonempty profunc MF truth.
Optional `go_mf_terms` header annotations must agree with profunc. There is no
truth TSV override or silent skipping of unannotated queries.

Truth is the complete direct MF set underlying
`profunc analyze function ACCESSION -v`, including description-pruned terms.
No per-query profunc subprocesses or annotation downloads occur. `GO:0005515`
is retained during evaluation; the audit's normal exclusion is separate.

Install this project and profunc, plus the actual PClass project in the same
Python environment (initial native API target: PClass 0.3.x; tested with 0.3.0).
PClass collection uses its neighbor client and an already-running PClass service,
not its inference CLI. No Needle, ProSig, alignment sequence table or local PClass
inference artifact directory is needed. Server URL defaults to `PCLASS_SERVER_URL`
or `http://127.0.0.1:8000`. This evaluator does not start the server or verify its
model/index identity. Set a new `--run-tag` after changing those resources/defaults.

Dark requires a completed production/C/B library. Embedding cache misses require
`pip install -e '.[embeddings]'` and model assets; `--offline` forbids downloading
assets. A fully cached Dark run does not initialize a model or import Torch.
Both classifiers require scientific references for semantic scoring.
Unknown predicted GO terms warn and are excluded from semantic scoring; missing
terms are recorded in metric status, original annotations are retained, and scores
with no usable terms are `NA`. Preflight truth/map incompatibilities and unknown
Dark cluster IDs still fail clearly. POSIX/Python >=3.11 is required.

## Cache and processing controls

For the repository's selected prepared data, use the paths named by
`data/setup-data/selection.json`:

```bash
python /path/to/PClass-Dark/scripts/evaluate_swiss.py sequences.fasta \
  --dark-library-dir /path/to/PClass-Dark/data/library \
  --dark-embedding-cache /path/to/completed/embedding-directory \
  --dark-prepared-dir /path/to/completed/prepared-directory \
  --preflight-only
```

Supply both cache directories together. They are fully verified before use;
matching accessions or vector dimensions alone are insufficient. The join checks
sequence hashes, window coordinates/content, row inventory and representation.
An earlier originating preparation ID is allowed when the supplied preparation's
verified window inventory and representation match. Both IDs are recorded.
Cache files are read-only; missing windows are embedded in bounded microbatches.
Cache mismatch/corruption is an error, not a reason to silently recompute.

Preflight validates inputs, truth, native APIs, Dark library/cache and scoring
resources, and reports window hit/miss and cached/pending query counts. It does
not contact the PClass service, load a model, infer proteins or publish a TSV.
Whole-source verification can take time and temporarily materializes prepared
reference sequences. Device availability remains unverified until model use.

Checkpoint chunks default to 512 queries per classifier. Dark embedding
microbatches default to 8 windows (`--dark-batch-size`, maximum 64), and scoring
blocks to 4096 windows (`--dark-score-batch-size`). Dark worker threads and GO
scoring processes each default to 8, run in separate phases, and are configurable
with `--dark-workers` and `--score-workers`; use 1 for serial debugging.
`--device cpu` selects CPU for Dark; PClass service execution is server-managed.
Memory admission defaults to 1024 MiB (`--dark-memory-mib`) for scoring workspace,
not total process/model RSS. CPU fallback follows normal Dark inference policy.

One native inference worker runs at a time, Dark first. It keeps validated
library/model resources across chunks. PClass explicitly rejected oversized
requests are bisected; completed leaves survive failures. `--process-timeout`
(default 3600 seconds) bounds initialization and each task;
`--pclass-timeout` defaults to 60 seconds per HTTP operation. Raw evidence is
capped at 128 MiB per chunk by default (`--max-raw-chunk-mib`); this is not a total
workspace storage cap. See `--help` for all controls.

## Resume, rescore and audit

Keep `.swiss-evaluation/` and the output's hidden publication marker. This workspace
is separate from `.external-evaluation/`. It retains frozen annotations/truth,
input snapshots, native numeric evidence, receipts, graph/maps and scored results.
Rerun to resume. Normal runs refresh truth from current profunc while reusing
compatible predictions. Inference identities include adapter source fingerprints,
configs, source/cache identities and run tag. A remote model change still requires
an explicit new tag.

```bash
python /path/to/PClass-Dark/scripts/evaluate_swiss.py sequences.fasta --score-only
python /path/to/PClass-Dark/scripts/swiss-audit.py swiss_evaluation.tsv
```

Score-only requires the original FASTA and saved workspace, but no live annotation
source, service, model or embedding cache. It rejects inference flags, uses saved
truth/resources, and permits explicit graph/map overrides for rescoring.

Output publication has locks, checksums and journaled recovery. Altered/unowned
outputs and corrupt committed observations fail. `--output` can select another
existing directory; marker v2 tells the audit where the source workspace lives.
Moving the output alone loses that relationship. Keep the workspace/marker layout
or supply the audit with explicit truth. Use `swiss-audit` for `swiss_audit_*_v2` plots. The external audit command
remains compatible and retains its `external_audit_*` names.

## Python API

```python
from pclass_dark.swiss_evaluation import (
    evaluate_swiss, SwissDarkConfig, SwissPClassConfig, ScoringConfig,
)

result = evaluate_swiss(
    'sequences.fasta', work_dir='.',
    pclass=SwissPClassConfig(server_url='http://127.0.0.1:8000'),
    dark=SwissDarkConfig(device='cpu'),
    scoring=ScoringConfig(workers=1),
)
print(result.tsv_path)
```

`SwissExecutionConfig` controls timeout/output cap/run tag. Library failures use
`ValueError`, `OSError` and the existing `Evaluation*` exceptions. Imports perform
no computation. The reusable `pclass_dark.inference_runtime.DarkInferenceRuntime`
is context managed; consume each `with runtime.infer_chunk(queries) as session`
before starting another chunk. Results are the normal `QueryResult` records.

See [SPEC-0017](specs/0017-evaluate-swiss.md) for contracts and validation status.
Full-cohort inference, real-model throughput comparisons and GPU parity remain
separate validation work; no measured speedup factor is claimed.

Completed inference is also reusable when scoring failed before publication:
rerun the same `scripts/evaluate_swiss.py sequences.fasta` command with the same
options. The script verifies compatible generation receipts and query coverage
and logs `reused completed generation` for each engine. The original unknown-GO
error version and its warning-only replacement are explicitly compatible;
unreviewed code changes and changes to inference settings remain incompatible.
No selection file or artifact rewriting is required. Resource preflight still
runs. `--score-only` continues to require a successful prior publication.

## Plot a Swiss evaluation audit

```bash
pip install -e '.[audit]'
python scripts/swiss-audit.py work-swiss/swiss_evaluation.tsv
```

The command writes `swiss_audit_{match_scores,confidence,coverage,summary}_v2.png`
and `swiss_audit_summary_v2.tsv` beside the input. It excludes whole queries whose
own truth annotations contain `GO:0005515`, including mixed annotation sets,
before computing any counts. Saved publication metadata supplies verified truth;
standalone tables require `--truth-tsv` with `Query_id` and pipe-separated
`go_mf_terms` columns covering exactly the input IDs. Explicit truth takes
precedence over metadata. No inference, model or live annotation lookup runs.

Options: `--output-dir`, `--bins` (1–100, default 10), `--hit-cutoff` (0–1,
default 0.7), and `--truth-tsv`. A hit is saved semantic similarity >= cutoff,
not exact-label accuracy. Undefined scores are excluded from hit-rate denominators
but retained in prediction coverage. Smaller distance/B evidence is stronger;
confidence bins are descriptive and do not calibrate the methods.

```python
from pclass_dark.swiss_audit import audit_swiss

paths, rows, diagnostics = audit_swiss("work-swiss/swiss_evaluation.tsv")
# Explicitly unfiltered; filenames omit _v2 and no truth is loaded:
paths, rows, diagnostics = audit_swiss("comparison.tsv", exclude_go=None)
```

The API supports only `exclude_go="GO:0005515"` or `None`, returns five output
paths plus per-method rows and diagnostics, and raises `ValueError` for invalid
data/options or unsafe output paths, `OSError` for I/O failures and `RuntimeError`
for missing plotting support. Prediction GO text is used only for presence; truth
GO sets are validated. Extra uniquely named input columns are accepted.
Existing outputs are replaced per file after all five files are staged; the
whole output set is not an atomic transaction. Inputs are protected from replacement.

These plots describe an annotation-selected cohort; Swiss self/training and
homologous matches may remain. They do not establish held-out model quality.
See [SPEC-0018](specs/0018-swiss-audit.md) for the audit contract and validation.
