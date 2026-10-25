# Updating bundled cluster references

The authoritative inputs are these files in the neighboring profunc checkout:

- `src/profunc/library/default/bundle/clusters/clusters.tsv`
- `src/profunc/library/default/bundle/dataset/accession.fasta`

PClass-Dark stores unchanged copies in
`src/pclass_dark/resources/reference/`. `manifest.json` records their identities,
checksums, sizes, and coverage. `source_manifests/` preserves the profunc bundle,
cluster, and dataset manifests and cluster statistics. ProStruct's enrichment
copies are useful for comparison but are not the authoritative update source.

## Update procedure

1. Select a completed profunc publication. Read its bundle, cluster, and dataset
   manifests; confirm that the clusters' source dataset generation matches the
   selected sequence dataset. Check file sizes and SHA-256 digests against those
   manifests. Filesystem timestamps alone do not establish a matched snapshot.
2. Stage both source files together outside the live reference directory. Copy
   them byte-for-byte, retaining the original header, accession IDs, noncanonical
   residues, and additional unclustered FASTA records. Do not filter or normalize
   the bundled snapshot. Preserve the source manifests and cluster statistics.
3. Validate the staged pair: membership columns must be `member_id` and
   `cluster_id`; IDs must be nonempty and memberships unique. FASTA IDs must be
   unique, sequences nonempty, and every clustered accession must have a
   sequence. Count clusters, accessions, residues, extra FASTA records, and
   clustered sequences with noncanonical residues. Compare expected source
   counts and rehash the copies.
4. Replace the bundled pair and provenance files together. Regenerate
   `manifest.json`, including source paths, copy timestamp, generation IDs,
   bundle ID, recipe/schema versions, file sizes, SHA-256 digests for every
   copied file, and observed counts. Never retain old checksums for new data.
5. Update README snapshot counts and record the old/new identities, reason for
   updating, and validation results in `docs/research/progress.md`. Retain the
   previous snapshot's provenance in version history. Review whether existing
   embeddings and signatures must be rebuilt; never silently combine artifacts
   from incompatible references. Bump the package version for a new release.
6. Run the tests and build both distribution formats:

   ```bash
   pip install -e '.[dev]'
   pytest
   python -m build
   ```

   Inspect the source archive and wheel for both full data files, the manifest,
   and source manifests. Install the wheel into a separate environment and verify
   resource checksums through `importlib.resources` from outside the checkout.
   Ensure the files are real data rather than Git LFS pointer files if LFS is
   used to store large reference files.

## Access and interpretation

Use `importlib.resources.files("pclass_dark").joinpath("resources", "reference")`.
For an API requiring a filesystem path, use `importlib.resources.as_file` and
keep the returned path inside its context. Write generated outputs elsewhere.

The profunc CLI currently does not export this exact pair:
`browse clusters --output` writes cluster metadata, while `browse accession`
fetches sequences live from UniProt. Neither replaces copying the matched,
frozen local snapshot. No profunc installation is required to use the bundled
PClass-Dark references.

These are functional clusters, not homology-separated evaluation splits.
Unclustered FASTA records are not automatically negative examples. The snapshot
includes noncanonical residues; embedding generation must define how it handles
those sequences without modifying the reference files.
