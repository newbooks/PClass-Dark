# Exact sequence identity check — 2026-10-25

Question: how often do distinct reference accessions have identical sequences?
This is an exact full-sequence comparison, not a homology search or alignment.

## Inputs and method

Used the bundled authoritative profunc references. Both file SHA-256 digests
were verified against the bundled manifest before analysis; exact identities
and full counts are in [the result JSON](2026-10-25-sequence-identity.json).
Cluster generation: `20260911T225655Z-153032271c41`; dataset generation:
`20260911T184540Z-d7e84090016d`.

Parsed FASTA primary IDs, concatenated sequence lines, and grouped by actual
complete sequence strings. No case conversion, residue substitution, sequence
filtering, alignment, or windowing was applied. Grouping uses string equality,
not just digest equality. Membership and FASTA IDs were unique, every sequence
was nonempty, and all clustered IDs were found in FASTA. Serial Python analysis
completed locally in seconds; no model, GPU, multiprocessing, or Internet access
was needed. Sanity checks covered duplicate and all-unique group summaries.

For clustered-only counts, restrict each sequence group to clustered accessions
before counting; an unclustered sequence match alone does not count as a
clustered duplicate. A redundant accession is any accession beyond one per
unique sequence, without choosing which accession to retain.

## Results

| Measure | All FASTA accessions | Clustered accessions only |
|---|---:|---:|
| Accessions | 575,748 | 455,508 |
| Distinct sequences | 487,476 | 381,183 |
| Sequence groups with multiple accessions | 38,466 | 32,384 |
| Accessions in those groups | 126,738 (22.01%) | 106,709 (23.43%) |
| Redundant accessions beyond one per sequence | 88,272 (15.33%) | 74,325 (16.32%) |
| Largest group | 114 | 66 |

Among clustered accessions, 1,071 identical-sequence groups span multiple
functional clusters, involving 5,386 accessions across 739 clusters. These
memberships are not automatically annotation errors: no biological explanation
or reconciliation was attempted. Within-cluster duplication occurs in 2,010
clusters (32,009 cluster/sequence groups, involving 105,249 accessions).
Within-cluster and cross-cluster counts overlap and must not be added.

## Setup-data direction and interpretation

Use bundled reference resources directly, without duplicating them or downloading
new reference data. The planned setup step is offline and will handle exact
sequence identity, embedding-window planning, and training/validation splits.
Homology searches are intentionally excluded per user direction. PLM weight
acquisition and embedding computation remain separate from this offline setup.
This records scope; setup-data has not been implemented yet.

Identical sequences should stay in the same split, including matches across
clusters, and windows should inherit their parent sequence's split. Otherwise,
identical inputs can appear in both training and validation. Even with exact
identity grouping, related but nonidentical sequences can span splits, so this
protocol does not establish generalization to distant homologs.

Local detailed artifacts (ignored by Git): `work/sequence-identity/summary.json`,
`duplicate_groups.tsv`, and `check_identity.py`. Run the analysis from this
checkout using `python work/sequence-identity/check_identity.py` while those
local artifacts are present. No reference data or memberships were modified.

## Cluster sizes after within-cluster exact deduplication

Count each full sequence once per cluster. A sequence shared across distinct
clusters is retained once in each; no arbitrary cluster reassignment occurs.
This gives 382,268 cluster/sequence memberships versus 455,508 accession
memberships, across all 3,524 clusters. The membership total exceeds the 381,183
 globally distinct clustered sequences because of cross-cluster matches.

| Unique sequences per cluster | Clusters | Percent |
|---|---:|---:|
| <10 | 81 | 2.30% |
| [10, 20) | 1,295 | 36.75% |
| [20, 40) | 799 | 22.67% |
| [40, 80) | 481 | 13.65% |
| [80, 160) | 354 | 10.05% |
| [160, 320) | 218 | 6.19% |
| [320, 640) | 197 | 5.59% |
| >=640 | 99 | 2.81% |

Source checksums were reverified and bin counts sum to 3,524. Detailed local
outputs: `work/sequence-identity/cluster_sizes.tsv` and
`work/sequence-identity/cluster_size_distribution.json`. Reference files remain
unchanged; this is a hypothetical deduplication summary.
