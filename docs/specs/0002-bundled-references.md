# SPEC-0002: Bundled functional-cluster references

## Purpose and contract

Distribute a frozen, matched pair of profunc cluster memberships and Swiss-Prot
sequences with the Python package. The authoritative sources are
`../profunc/src/profunc/library/default/bundle/clusters/clusters.tsv` and
`../profunc/src/profunc/library/default/bundle/dataset/accession.fasta`.
ProStruct copies are corroborating references, not acquisition sources.

Copy both files byte-for-byte into `src/pclass_dark/resources/reference/`.
Preserve the `member_id`/`cluster_id` header, accession identifiers, sequences,
noncanonical residues, and additional unclustered sequences. Require unique
membership IDs, unique FASTA IDs, nonempty sequences, and full sequence coverage
of clustered accessions. Functional clusters do not define homology-separated
splits. Do not interpret unclustered accessions as validated negatives.

Record source paths, sizes, SHA-256 digests, generation IDs, and observed counts
in `manifest.json`. Preserve bundle, cluster, dataset manifests and cluster
statistics under `source_manifests/`. Bind those provenance files by checksum.
Verify source files against their source manifests before copying and rehash
copies. Reject mismatched cluster/dataset generation identities.

## Distribution and updates

All files under `src/pclass_dark/resources/reference/` are tracked recursively
by Git LFS through `.gitattributes`. Release builders must fetch LFS objects
and pass reference-integrity checks before packaging. Distribution archives
contain full data, not LFS pointers; installing a wheel does not require LFS.

Setuptools package-data includes reference TSV, FASTA, JSON, and source
manifest JSON files. Access resources with `importlib.resources`; callers needing
filesystem paths should use `as_file` within its context. Bundled resources are
read-only inputs. No dependency on profunc at runtime is introduced.

Updates replace the pair and provenance together after validation, following
`docs/reference-data.md`. Rebuild and inspect both source and wheel distributions;
validate installed resources outside the checkout. Embeddings and signatures
must retain the reference identity used to create them and be regenerated or
explicitly checked for compatibility after a reference update.

## Status and validation

Implemented. Automated resource tests validate manifest checksums, membership
schema, unique IDs, complete nonempty sequence coverage, and observed counts.
Build and installed-resource verification results are in the progress log.
