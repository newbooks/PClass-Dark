# External classifier comparison

Run from the directory where you want the results:

```bash
../scripts/evaluate_external.py external_sequences.fasta
```

The evaluator uses the working `pclass infer --fasta` and `pclass-dark infer
--fasta` commands on PATH, requesting JSON output. It needs no evaluation extra,
`ijson`, bundle fingerprint or separate model setup. Each classifier manages its
own dependencies, server, artifacts and internal computations. The evaluator
does not check Needle/ProSig separately or use alignment/sequence identity scores.
The evaluator environment needs the existing PClass-Dark base dependencies,
including profunc, with its local GO graph and cluster data. It requires POSIX and
Python >=3.11; classifier executables may use separate Python environments.

## Input and results

Use regular uncompressed UTF-8 FASTA with unique first-token IDs and MF truth:

```text
>Q12345 go_mf_terms=GO:0003824|GO:0005524
ACDEFGHIKLMNPQRSTVWY
```

Alternatively supply `--truth-tsv annotations.tsv` with columns `Query_id` and
`go_mf_terms`, pipe-separated GO IDs, and exactly one row per query. Truth sources
must agree if both are present. Sequences are uppercased; the common classifier
alphabet is `ACDEFGHIKLMNPQRSTVWYBOUXZ`. J, gaps and stops are rejected. Truth is
removed from FASTA sent to classifiers.

The primary result is `external_evaluation.tsv`, with one row per input query:

| Column | Value |
| --- | --- |
| Query_id | Original first-token identifier, in input order. |
| PClass_go | GO MF set of the closest embedding neighbor. |
| Embedding_d | That neighbor's embedding distance. |
| Pclass_match | Truth versus PClass_go, using profunc AMB Lin similarity. |
| Dark_go | Composed GO set of the accepted cluster with the smallest B score. |
| Dark_Bscore | That cluster's background-tail fraction, not percentage. |
| Dark_match | Truth versus Dark_go, using the same similarity. |

PClass ties retain report order; Dark ties use cluster ID. Selection never uses
truth similarity, alignments or sequence identity. GO sets are sorted and joined
by `|`. Empty sets and unavailable/undefined scores are `NA`; defined zero is
`0.0`. The complete PClass predicted GO set and other evidence remain in raw JSON
although the table compares the top neighbor's GO set.

Similarity uses the same Python helper as the installed
`profunc analyze go-set-sim` command. It avoids per-query subprocesses and rounded
text parsing. No additional similarity package is introduced. The graph's IC is
frozen, not estimated from the evaluated cohort.

Dark JSON contains cluster IDs, so the evaluator automatically loads their
composed GO sets from profunc. Use the profunc data corresponding to the Dark
classifier's cluster IDs; the current report does not prove that reference
identity. Unknown clusters or GO terms fail clearly. `--cluster-go-map` and
`--go-graph` are optional explicit scoring-data overrides. A graph pickle must
come from a trusted source.

## Optional controls

The defaults use classifier defaults. Override only what you need:

```bash
../scripts/evaluate_external.py external_sequences.fasta \
  --pclass-server-url http://127.0.0.1:8000 --device cpu \
  --pclass-batch-size 64 --dark-chunk-size 64 --score-workers 2
```

`--pclass-artifact-dir` and `--dark-library-dir` are optional pass-through paths.
`--preflight-only` validates inputs, PATH commands and scoring data without
starting inference. `--help` lists all options. PClass batches default to 512
queries, never exceed 1,024, and use a conservative request byte bound. Explicit
size rejections are split and retried. Dark defaults to 512 queries per chunk and
manages its own model quotas. One classifier child runs at a time.

`--process-timeout` defaults to 3,600 seconds per child; `--max-raw-chunk-mib`
defaults to 128 MiB of stdout per chunk. Timeouts/output-limit failures terminate
the process group, including grandchildren. Failed attempts retain stderr.
Standard-library JSON parsing holds one bounded report chunk in memory; its
Python representation can use several times its on-disk size. Each scoring
worker holds a graph/map copy; worker count defaults to 8 and is bounded to 1–8.

## Resume and rescore

Full raw JSON, hashes, receipts, input/truth snapshots, graph/map copies and
scoring records are retained in `.external-evaluation/` in the current directory.
Rerun the command to resume and reuse completed chunks. Annotation or scoring-data
changes reuse the costly predictions.

```bash
../scripts/evaluate_external.py external_sequences.fasta --score-only
```

Score-only uses the last saved prediction generations and frozen scoring data.
It does not resolve classifier commands or contact servers. Inference flags
cannot be supplied in this mode. Use explicit graph/map paths to rescore with
new data. Keep the whole workspace for reproducibility; retained reports and
versions accumulate and the per-chunk cap is not a total storage cap.

The evaluator records commands, executable hashes, relevant environment settings
and raw source metadata, but cannot detect a changed remote server model/index
or a changed library behind the same command. After such a change, use a new tag:

```bash
../scripts/evaluate_external.py external_sequences.fasta --run-tag updated-models
```

Existing artifact-attested evaluator caches from the earlier design are preserved
but are not automatically adopted by this command-output revision. Collect a new
run before using score-only with that format.

Output publication uses locks, hashes, an ownership marker and a recovery journal.
Interrupted publication is completed or rolled back on the next invocation.
Unowned outputs, manually changed results and corrupt committed raw files cause
an error. Use `--output` with an existing parent to select another destination.

## Python interface

```python
from pclass_dark.external_evaluation import evaluate_external

result = evaluate_external('external_sequences.fasta', work_dir='.')
print(result.tsv_path)
```

Optional `PClassConfig`, `DarkConfig`, `ScoringConfig` and `ExecutionConfig`
objects configure the same operations. `score_only=True` reuses saved inference;
`preflight_only=True` returns an `EvaluationPreflight`. Pure
`score_external_predictions` accepts typed query/result records, a profunc graph
index and `ClusterGoMap`, and yields ordered `ComparisonRow` records without
invoking classifiers. See [SPEC-0015](specs/0015-evaluate-external.md) for contracts.

Tests use analytical scores and simulated classifier commands; full external
inference and GPU execution remain untested.

## Audit the comparison table

Install the optional plotting dependency with `pip install -e '.[audit]'`, then:

```bash
python scripts/external-audit.py path/to/comparison.tsv
# Equivalent installed command:
pclass-dark external-audit path/to/comparison.tsv
# Optional display settings:
pclass-dark external-audit path/to/comparison.tsv --bins 10 --hit-cutoff 0.7 --output-dir audit
```

The script/CLI now excludes all queries whose annotated MF GO set contains
`GO:0005515`, including mixed sets. It reads truth from the input's hash-verified
saved evaluation metadata, not from predicted GO columns. For standalone or
renamed inputs without publication metadata, supply `--truth-tsv annotations.tsv`
with `Query_id` and `go_mf_terms` columns and exactly one row per input query.
Missing or unverifiable annotations fail with an error.

The filename is arbitrary; the seven evaluation column headers are required.
By default outputs are written beside the input (existing audit outputs replaced):

- `external_audit_match_scores_v2.png`: paired GO match scores with equality/cutoff lines.
- `external_audit_confidence_v2.png`: side-by-side stacked hit/non-hit/undefined counts,
  identical [0,1] bins and shared count scale. Bar labels give hit percentages among
  defined matches. A separate >1 distance bin and missing-confidence category are
  retained when needed. Smaller distance and smaller B are stronger evidence;
  bin numbers do not equate/calibrate these measures.
- `external_audit_coverage_v2.png`: side-by-side hit rate versus retained predictions
  as a percentage of all queries. Ties remain together. Missing match scores count
  toward coverage but not hit-rate denominators; missing confidence is unrankable.
- `external_audit_summary_v2.png` and `external_audit_summary_v2.tsv`: counts, coverage,
  hit rates, method-only hits, missingness and distance >1 diagnostics.

Original unfiltered files are preserved. Counts, hit rates and coverage denominators
are recomputed on retained queries. The summary records original, excluded and
retained query counts.

A hit means GO semantic match >=0.7 by default, not exact-label accuracy. Missing
scores are never converted to zero. Prediction coverage uses nonempty predicted
GO sets; scatter uses jointly defined scores. Header-only tables produce empty
plots/NA rates. Dark contains accepted candidates only, so curves cannot extend
beyond observed predictions. These descriptive plots do not establish independent
held-out performance or control reference homology leakage.

Python callers can use `read_audit_table`, `summarize_audit`, `confidence_bins`,
`coverage_curve`, and `audit_external` from `pclass_dark.external_audit`. The first
four functions do not import Matplotlib or render files. See
[SPEC-0016](specs/0016-external-audit.md) for contracts and validation.

For the same v2 filtering through Python, call
`audit_external(input_path, exclude_go="GO:0005515", truth_tsv=None)`.
`read_query_annotations` and `filter_query_go` are reusable independently. The
Python API's default remains unfiltered for compatibility; no `exclude_go` writes
the original filenames. The CLI/script always requests the v2 exclusion.
