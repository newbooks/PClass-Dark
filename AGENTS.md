# Repository Agent Guidelines

## Branch-aware editing

- Check the current Git branch and working tree before modifying files.
- On `main`, do not edit code unless the user explicitly authorizes code edits
  on that branch. A general implementation request alone is not an exception.
  Read-only inspection, analysis, recommendations, and requested documentation
  edits are allowed.
- On any branch other than `main`, proceed with implementation, fixes,
  refactoring, tests, and documentation within the user's requested scope
  without asking for additional confirmation.
- Preserve unrelated user changes. These rules do not authorize destructive
  or out-of-scope changes. Do not commit or push unless requested.

## Project and neighboring repositories

- PClass-Dark studies functional-cluster signatures in PLM embedding dimensions
  to classify proteins lacking reliable nearest-neighbor matches.
- Consult `../ProStruct/`, especially `scripts/profile_clusters.py`,
  `scripts/compose_signature.py`, their underlying modules, tests, and
  specifications, when adapting feature profiling and signature composition.
  ProStruct's preliminary work uses structural features; transfer to PLM
  embeddings requires evaluation rather than an assumption of equivalent
  behavior or performance.
- Consult `../profunc/` for functional clustering and GO semantic similarity,
  and the neighboring PClass repository for embedding and nearest-neighbor
  behavior when relevant. Inspect the actual paths and implementations rather
  than guessing their behavior.
- Treat neighboring repositories as references, not undocumented runtime
  dependencies. Keep required implementations, schemas, constants, tests, and
  operational documentation here. Record provenance and explain meaningful
  deviations from reference methods.

## Command-line and Python interfaces

- Provide both the `pclass-dark` command/subcommand hierarchy and importable
  `pclass_dark` modules. Use the `src/pclass_dark/` package layout and Typer,
  following the neighboring projects' CLI conventions.
- Keep reusable scientific logic in library modules with explicit inputs,
  return values, and documented exceptions. CLI subcommands should be thin
  adapters for argument parsing, output formatting, and exit codes.
- Expose new scientific operations through both interfaces where applicable.
  Python integrations must not need to invoke CLI subprocesses. Imports must
  not parse arguments, launch workers, download models, or start computation.
- Keep packaging metadata, dependency declarations, CLI help, API documentation,
  and README examples synchronized. Test library behavior and CLI integration.

## Implementation specifications

- Before substantive implementation or algorithm changes, create or update a
  numbered implementation specification under `docs/specs/`.
- Record purpose, provenance, inputs and outputs, data contracts, algorithm
  steps, assumptions, invariants, edge cases, configuration, resource use,
  and validation expectations as applicable.
- Keep specifications synchronized with code in the same change. Update
  implementation status, actual behavior, deviations, and test results before
  considering the work complete. Mark planned behavior clearly.
- Maintain `docs/specs/index.md` as a concise entry point as specifications
  are added. Routine small fixes may update the relevant existing specification
  rather than require a new one.

## Progress log

- Maintain a dated, chronological log in `docs/research/progress.md` for
  substantive implementation and experimental work. Record milestones,
  decisions, results, failures, validation, and next steps, with links to
  relevant specifications and artifacts.
- Preserve earlier findings and negative results. Mark superseded conclusions
  and link to their replacements rather than silently rewriting history.
- For experiments, record enough detail to reproduce the work: data identity,
  code revision when available, preprocessing, splits, parameters, seeds,
  hardware, commands, and artifact locations.
- Distinguish in-sample diagnostics from held-out performance. Control homology
  leakage and fit preprocessing, feature selection, scoring parameters, and
  thresholds on training data only for held-out evaluations.

## Parallel processing and GPU use

- Support multiprocessing for independent CPU-intensive work whenever practical,
  such as per-cluster profiling and signature selection. Expose a configurable
  worker count and retain a serial mode for debugging and constrained systems.
- Prefer GPU execution for suitable embedding and numerical workloads when a
  compatible GPU is available. Provide an explicit CPU option and a usable CPU
  fallback, and report the device actually used.
- Bound worker counts, memory use, and task queues; avoid nested process or
  numerical-library thread oversubscription and unnecessary copies of large
  embedding matrices. Use threads or batching where they better suit the work.
- Preserve deterministic ordering and seeded reproducibility across worker
  counts where practical. Surface worker failures and protect outputs from
  partial publication. Document material numerical differences between devices.

## Tests and completion checks

- Every code implementation or change must include appropriate automated tests
  and execution of the relevant test suite before completion. Add or update
  pytest tests for new behavior, regressions, and meaningful edge cases.
- Test scientific calculations against independent analytical or small reference
  examples, not just copies of the implementation. Cover invalid inputs and
  insufficient data where relevant.
- For parallel or device-dependent changes, verify serial/parallel consistency
  and CPU behavior; exercise GPU behavior when hardware is available and report
  when it could not be tested.
- If tests cannot run, state the blocker and what remains unverified. Do not
  report unexecuted or failing tests as passing.
- At handoff, summarize changes, tests and outcomes, synchronized specifications
  and progress records, and any remaining limitations. Documentation-only edits
  require appropriate content and formatting checks rather than code tests.
