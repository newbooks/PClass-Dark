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
