# SPEC-0012: Live profunc dependency and reference binding

Status: implemented and validated. Supersedes SPEC-0002's bundled-reference
runtime and SPEC-0010's static description catalog. Legacy reference snapshots are removed from the working tree; Git history and
the research log retain their provenance.

## Purpose and provenance

Use installed profunc modules rather than a copied reference publication or
exported descriptions. Resolve through `profunc.runtime.resolve_runtime`, with
its local-current then packaged-default semantics and manifest validation.
Use `profunc.cluster_listing.load_cluster_records` and
`profunc.go.description.describe_go_function` for current labels, retaining the
modifier setting used by profunc's inspection command.

## Contracts and algorithm

`prepare_data` resolves profunc at invocation, reads its cluster memberships,
and validates the matching sequence FASTA against the dataset manifest. Profunc
runtime bundles exclude FASTA: use the resolved dataset's FASTA when present,
otherwise the original generation under `profunc.workspace.default_workspace`,
or an explicit Python `reference_fasta` / CLI `--reference-fasta` path. Missing
or mismatched FASTA fails with actionable guidance; never download or fall back
to PClass-Dark's historical snapshot. `reference_dir` remains an explicit
manifest-bound Python override for integrations and tests.

Record cluster and FASTA content hashes/sizes in the prepared reference contract.
Existing preparation, embedding inventory, and library identities continue to
invalidate downstream artifacts when their inputs change. The library propagates
the reference contract into final-fit inputs. Description lookup resolves and
validates current profunc, loads its metadata/GO graph through its modules, and
compares reference hashes before attaching descriptions. Missing reference binding
in legacy signatures or a changed membership/sequence snapshot yields an explicit
unavailable reason rather than a label joined solely on cluster ID. No process
or network activity occurs on module import. No automatic package upgrade occurs.
Description-only metadata/algorithm changes
are reflected at lookup time when source memberships/sequences still match; they
do not require refitting.

## Resources, failures and validation

Existing preparation workers and embedding CPU/GPU options are unchanged. Runtime
validation and label generation are serial, linear in source size. Resolve at
operation boundaries and repeat validation before returning labels to detect
concurrent publication. Verify graph provenance before profunc loads its pickle. Surface
missing runtime, invalid manifests and incompatible module interfaces as errors.
No network required. A user must install/update profunc and build its library.

Test changed membership identity/reuse, FASTA mismatch/missing input, explicit
FASTA, dependency-backed descriptions and stale/legacy binding rejection. Run
full pytest, lint and CLI smoke checks. Full embedding/signature recomputation
is outside the regression checks; synthetic embeddings exercise pipeline binding.

## Implementation outcome

Implemented lazy dependency imports, validated runtime resolution, source FASTA
lookup/override and output-overlap protection, new fit reference bindings, and
live descriptions with content identities. Historical references are removed from the working tree and excluded from
wheels/sdists; static descriptions and the
export script were removed. No neighboring repository was changed.

Validation on 2026-10-29: full pytest passed with **261 passed, 2 skipped**
(524.15 seconds). The two pinned real-checkpoint tests require the explicit
`PCLASS_DARK_TEST_CACHED_ESM=1` setting and were not enabled. One existing warning
was reported by the suite. Focused reference/CLI/listing checks, lint and diff
checks passed. Isolated wheel/sdist builds declared the dependency and excluded
retired resources, including a simulated stale build directory. Real source
validation covered 3524 clusters and 455508 accessions with complete sequence
coverage; listing the existing library returned all 3443 signatures and explicit
missing-binding reasons. Synthetic embedding tests verified new signature
bindings and CPU/parallel behavior. No full PLM inference or signature-dataset
rebuild was performed. See the dated [progress record](../research/progress.md).

## Legacy-artifact cleanup (2026-10-29)

Remove the retired `resources/reference/` membership TSV, sequence FASTA,
manifest and four source-manifest/statistics files, and their obsolete LFS rule.
Remove the obsolete copying/release procedure from current reference docs.
Retain clearly retired specifications and chronological research results for
historical provenance. Replace the bundled-snapshot test with an independent
small fixture exercising the supported explicit-reference API. Keep package
exclusions and build cleanup guards so stale build directories cannot reintroduce
retired resources. Do not remove selected/archived prepared data, embeddings, or
signature libraries: those are still supported reproducible algorithm outputs.
Validation: 58 focused reference, setup, description, CLI and distribution tests
passed in 4.51 seconds. Ruff and scoped Git diff checks passed. Direct live
profunc lookup succeeded with the retired reference tree absent.
