# SPEC-0007: Query ranking and A/B/C presentation

- Date: 2026-10-27.
- Status: Planned; documentation only. Query ranking is not implemented by this change.
- Provenance: User-approved naming and existing acceptance/evidence contracts in
  [SPEC-0004](0004-build-library.md) and [SPEC-0005](0005-relative-score-distributions.md).

## Planned command and inputs

The command name is **`pclass-dark infer`**. This document is its implementation
specification; the command remains planned and is not implemented by this edit.

Provide exactly one of the mutually exclusive input options:

- `--seq TEXT`: one protein sequence supplied directly on the command line.
- `--fasta PATH`: a FASTA file containing one or more protein sequences, processed
  as a batch. Preserve FASTA record order and identifiers in the report. Each
  record is a separate query; never merge matches across different proteins.

Reject missing input, both input options together, an empty sequence, an empty
FASTA file, duplicate FASTA identifiers, or malformed/unsupported sequence input
with a clear error and nonzero exit status. Apply the reference pipeline's
residue policy and windowing consistently for both input forms.

```bash
pclass-dark infer --seq ACDEFGHIKLMNPQRSTVWY
pclass-dark infer --fasta queries.fasta

# The user redirects the on-screen report when a file is wanted.
pclass-dark infer --fasta queries.fasta > matches.txt
```

## Planned embedding execution and CPU budget

Embedding is GPU first with CPU fallback. Planned `--device auto` is the default:
prefer a compatible CUDA GPU; if unavailable or GPU initialization fails, use
CPU. Provide `--device cpu` for explicit CPU execution. Report the device actually
used and any automatic fallback on stderr. Keep the pinned model, residue
pooling and embedding representation compatible with the selected library.

The default requested CPU count is **8**, exposed through `--workers 8`.
When embedding on CPU, use this as the CPU execution budget; provide
`--workers 1` for serial execution. Bound effective CPU use by available resources
and avoid multiplying process workers by an unrestricted numerical-library thread
count. GPU embedding must not launch one model replica per CPU worker. Batch
FASTA records/windows with bounded memory and deterministic output order; exact
batch-size controls and admission limits must be finalized before implementation.

## Planned on-screen reporting

Both `--seq` and `--fasta` write the result report to **stdout**, which displays
on screen by default. Users are responsible for redirecting stdout to a file;
no report file is created automatically and no report-output path option is
planned. Logs, progress, device/fallback messages and errors go to stderr so
redirection captures the report cleanly.

For `--seq`, show one query result. For `--fasta`, show a separately identified
result for each FASTA record in input order, including records with no accepted
matches. Combine and rank windows only within each query as specified below.
Do not generate an interpretation or batch-wide ranking across proteins.

## Purpose and labels

Future top-matching-cluster results will use **A — Acceptance**,
**B — Background score ↓**, and **C — Cluster score ↑**. Arrows show score
direction, not change over time. A is a Yes/No flag and needs no direction arrow.

| Label | Existing source | Meaning / tooltip |
|---|---|---|
| A — Acceptance | Signature acceptance flag | Does the query pass this cluster's fitted threshold? Yes/No. |
| B — Background score ↓ | `background_tail_fraction` | Reference nonmembers scoring at least this high, with plus-one adjustment. Lower is better. |
| C — Cluster score ↑ | `cluster_percentile_le` | Reference members scoring at or below the query. Higher means a higher score rank among members. |

## Data and calculation contracts

Use a verified production signature and its paired C/B distributions. For query
log score `l`, preserve existing formulas and inclusive, exact score ties:

```text
A = (l >= fitted log cutoff), or No for a predict-none threshold
B = (1 + count(B_log_scores >= l)) / (n_B + 1)
C = count(C_log_scores <= l) / n_C
```

The cutoff maximizes training balanced accuracy, with highest-cutoff ties.
A is not derived from B or C. Display B/C as percentages, retaining fractions
in [0, 1] and existing descriptive field names in the underlying Python API.
Provide population sizes and availability status with the evidence.

Illustrative presentation:

| Rank | Cluster | A — Acceptance | B — Background score ↓ | C — Cluster score ↑ |
|---|---|---|---|---|
| 1 | cluster_example | Yes | 1% | 70% |

Lay explanation: "Only about 1% of reference nonmembers score at least this
high; the query scores at least as highly as 70% of reference members."
B includes the plus-one adjustment; it is not exactly the raw observed fraction.
Neither percentage describes how many windows have similar scores to the query.

## Planned infer output and window combination

Updated 2026-10-30: keep `infer` focused on reporting accepted matches and their
B/C evidence. Do not generate likely/unlikely labels, B cutoffs, gap-based
partitions, confidence categories or an interpretive summary. Interpretation is
left to users. This section specifies planned behavior; no infer command exists
yet.

Use the same residue handling, window length and terminal-window construction as
reference preparation. Score each query window against all available production
signatures and evaluate its paired B/C evidence. Preserve window IDs and 1-based
inclusive sequence coordinates. Batch FASTA proteins have separate result sets;
the union below applies only to windows of one query.

Combine accepted matches into one result table for the query:

1. A cluster is included if at least one query window has A=Yes for that cluster.
2. Include exactly one row per matched cluster. From its accepted windows, choose
   the window with the lowest unrounded B as the representative; break equal-B
   ties by sequence start, sequence end and window ID, in that order.
3. Report B and C from that same representative window, never independent minima
   or maxima across windows. Preserve all accepted supporting windows with their
   individual scores and coordinates in the Python/structured result.
4. Sort cluster rows by ascending unrounded representative B. Equal B values
   retain equal evidence rank; cluster ID provides deterministic display order
   within ties without implying stronger evidence.

The compact table contains Cluster, A (always Yes), B ↓ and C ↑. Representative
window coordinates and supporting-window evidence remain available in structured
output; their final text layout and precision are implementation choices. Display
B/C as percentages, retaining full fractions in structured output and population
sizes with the evidence. Avoid displaying positive B as zero. Include all matched
clusters; no top-k filter or additional evidence cutoff is planned.

The union and representative-window selection are reporting rules, not a new
protein-level threshold or calibrated combined score. Do not average embeddings,
combine raw cluster scores across windows, refit acceptance thresholds or
multiply/normalize B/C values.

When no usable window/cluster pair is accepted, return an empty match list and
an informational `No accepted matches` message. This is not a conclusion that the
query has no known function. No A=No rows are listed. Unavailable signatures or
distributions carry explicit status/reason metadata and are excluded from match
ranking, rather than being assigned zero scores or fabricated acceptance flags.
Integrity errors must fail visibly rather than appear as absent matches.

These are descriptive comparisons, not membership probabilities. Each cluster
has its own signature/background, and final C/B were used during discovery.
B ordering compares background-relative evidence; it does not establish relative
membership likelihood. Multiple clusters may be accepted. Do not rank raw
signature scores across clusters or multiply/normalize B and C into probabilities.

## Future implementation and validation

Expose reusable Python inference/ranking logic and a thin `pclass-dark infer`
adapter. Python integrations must not invoke CLI subprocesses or print reports
unless explicitly rendering a result.
Reuse existing scoring/evidence operations without refitting parameters or
thresholds. The accepted-window union and representative selection above define
result combination. The command name, input options, device policy, CPU default
and stdout behavior are defined above. Exact Python API signatures, embedding
batch-size controls, resource admission limits and structured output schemas
still require specification before implementation.
No runtime dependency or protein-level calibration is introduced here.

CLI tests must cover `--seq`, multi-record `--fasta`, exactly-one-input validation,
invalid/empty input, stable per-query input order, clean stdout redirection and
stderr diagnostics, GPU preference/CPU fallback, explicit CPU mode, and the
default CPU budget of eight plus overrides. Use mocked embedding backends for
deterministic integration tests and exercise real CPU/GPU paths when available.

Tests must cover field/label mapping, percentages, cutoff equality, predict-none,
inclusive ties, plus-one B, sorting before rounding, deterministic equal-B ordering,
unavailable data and no accepted candidates. Also test a cluster accepted by only
one terminal window, duplicate cluster matches combined into one row, B/C paired
to the selected accepted window, representative-window ties, and separate query
results. Assess ranking utility on held-out
queries with homology controls; fit any later ranking parameters on training data
only. Ranking performance is not established by the current in-sample reports.

Documentation-only validation: local links, labels/formulas and whitespace;
no runtime behavior changed and no code tests required.
