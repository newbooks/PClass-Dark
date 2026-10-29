# SPEC-0010: Portable profunc cluster descriptions

- Date: 2026-10-29.
- Status: Implemented (2026-10-29); verification recorded below.
- Purpose: Add Description immediately after Cluster in signature listings,
  using the exact function-description logic behind `profunc browse cluster`.
- Related: [SPEC-0009](0009-list-signature-summaries.md).

## Source and generation

Import the neighboring profunc module `profunc.go.description.describe_go_function`
in an explicit offline maintenance script, not as an installed runtime dependency.
Pass each cluster's `composed_go` annotations and the matching dataset graph,
with `CLUSTER_DESCRIPTION_MAX_MODIFIERS` imported from `profunc.cli.inspect`
(currently 3), matching the actual command. Verify the membership,
cluster metadata and GO graph checksums against the manifests preserved with our
bundled reference snapshot before importing/reading them. Never unpickle a graph
at application runtime; generation reads only the verified local source artifact.

Remove the exact `<cluster> is annotated as ` prefix and following `a ` or `an `
article, and the final sentence period. Preserve other wording, including
supporting activities. A description with no resolved head is unavailable.
Do not approximate the result from a GO term name or invent a functional label.

Bundle full descriptions in `resources/cluster_descriptions.json`, outside the
reference directory: adding display metadata must not change reference identity
or require refitting embeddings/signatures. Include source file/code hashes,
module/function name and parameters, a content identity, and all source cluster
IDs/composed GO IDs. Bind usable descriptions to the verified setup snapshot's
row-map SHA-256 and each retained cluster's C-window ID digest. At runtime, require
both to match the signature input identity before attaching text; this avoids
mislabeling a custom library that reuses the same cluster ID. A different window
inventory requires regenerating this optional metadata, not signature refitting.

## Runtime and display

Load the small packaged catalog once per listing; validate its schema/content ID
and structure. No neighboring paths, profunc import, GO graph, setup-data, network
or model download is needed at runtime. Missing entries or incompatible bindings
produce `description=null`, a machine-readable unavailable reason and `—` on
screen. A corrupt installed catalog is an integrity error. Preserve full text in
JSON/API rows and identify the catalog in source metadata.

Add Description immediately after Cluster in text/Markdown, left-aligned. Truncate
displayed descriptions longer than 24 characters to a prefix plus literal `...`,
with at most 24 characters including the ellipsis. Also respect 24 display cells
for wide Unicode; escape controls before measuring/truncating. Keep other column
alignment, conditional Status and three-decimal metrics unchanged. Descriptions
naturally contain spaces; headers stay single tokens, and JSON remains available
for structured parsing. Store full descriptions, never pre-truncated catalog text.

## Maintenance and validation

Provide `scripts/update_cluster_descriptions.py` with explicit profunc repository,
setup-data and output paths. It verifies matching reference provenance and setup
artifacts, imports the real describer, generates deterministically and writes an
atomic JSON output. Record code/source hashes rather than depend on profunc at
runtime. Document regeneration in the reference update instructions.

Test exact prefix/article removal, unresolved descriptions, short/exact/overlong
and wide-Unicode truncation, alignment after the new column, full JSON text,
missing/mismatched metadata, catalog corruption, packaging and operation without
profunc installed. Verify cluster_0001 against the actual profunc description and
inspect the full local listing. Update README, SPEC 0009, index and progress log.

## Implementation and verification — 2026-10-29

Implemented `cluster_descriptions.py`, the offline exporter, the packaged catalog,
and the summary loader/rendering changes. The catalog contains all 3524 source
clusters, with membership bindings for 3443 retained clusters. It is a 709 KiB
JSON resource distributed outside the identity-bearing reference tree; existing
reference/prepared/embedding/signature identities are unchanged.

All 3524 results were compared with the imported profunc function, using the CLI's
actual modifier constant; cluster_0001 was also compared with the command formatter.
Its full function text is `NADP+ nucleosidase with NAD+ nucleosidase activity`,
displayed as `NADP+ nucleosidase wi...`. Six retained clusters have empty composed
GO sets and therefore null descriptions, displayed as `—`. Unknown or incompatible
references also receive no guessed description.

Validation: initial focused **52 passed**; full regression **227 passed, 2 opt-in
tests skipped** in 380.51 s (existing Torch environment warning). Ruff lint/format
checks passed for 46 Python files, including the maintenance script. Wheel and
sdist inspection confirmed the catalog, with the exporter in the source archive.
Installed CLI/API listing worked outside the checkout with profunc/heavy PLM
imports, network and package writes blocked. Documentation link/fence/whitespace
checks passed. No GPU work or scientific refitting was required.
