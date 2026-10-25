# PClass-Dark

**Protein Function Classification Beyond Nearest Neighbors**

PClass-Dark is a protein function classification research project for proteins that lack reliable nearest-neighbor matches in protein language model (PLM) embedding space.

The central idea is to identify **conserved functional signatures across protein function clusters**. Each cluster may carry functional information in a different subset of embedding dimensions. By selecting dimensions that are both consistent within a cluster and informative against a background, PClass-Dark aims to reduce protein-specific variation and recover signals obscured in the full embedding space.

PClass-Dark complements [PClass](https://github.com/newbooks/pclass), which uses learned PLM embeddings and approximate nearest-neighbor (ANN) search to infer function from annotated reference proteins.

## Motivation

PLM embeddings encode multiple properties of protein sequences. A distance computed across the full representation may dilute a functional signal carried by only a subset of dimensions. The dimensions most informative for one function may also differ from those informative for another.

PClass-Dark explores the hypothesis that:

> Proteins sharing a biological function may preserve characteristic patterns in a subset of PLM embedding dimensions, even when their overall embedding vectors are substantially different.

Instead of requiring proximity to an individual reference protein, the project evaluates compatibility with a signature learned from a functional cluster. Whether this improves classification for proteins lacking reliable neighbors is a research question to be tested.

## Core Approach

The planned workflow adapts the feature-profiling and signature-composition concepts studied in the neighboring ProStruct repository, replacing structural features with PLM embedding dimensions.

### 1. Function Space Clustering

Functional clusters are curated from Swiss-Prot accessions annotated with Gene Ontology molecular function (GO MF) terms. The separate **profunc** project implements this clustering using Lin semantic similarity between GO terms.

Calibration against four-number Enzyme Commission (EC) classifications indicates comparable or finer cluster resolution. Each cluster represents a group of proteins with related molecular functions; cluster granularity determines the specificity of the resulting predictions.

### 2. PLM Embedding Representation

Protein sequences are represented by fixed-length PLM embedding vectors. Each dimension becomes a candidate feature for cluster profiling and signature discovery.

The embedding model, sequence pooling, and preprocessing must be consistent between reference and query proteins. A signature is specific to the representation in which it was learned.

### 3. Cluster and Background Profiling

For each functional cluster, individual embedding dimensions are profiled against a defined background of proteins outside that cluster. Background composition matters: separation from unrelated proteins may be easier than separation from proteins with closely related functions.

Following ProStruct's `scripts/profile_clusters.py`, candidate dimensions can be characterized by:

- **Relative conservation:** Within-cluster variation compared with background variation.
- **Distributional separation:** The difference between cluster and background means relative to their spreads.
- **Compatibility discrimination:** How well a score based on the cluster's characteristic values separates cluster members from background proteins, measured by the area under the receiver operating characteristic curve (AUROC).

The starting compatibility model is a Gaussian-shaped score centered on the cluster mean, with width set by the within-cluster standard deviation and a numerical floor. Values near the cluster mean receive higher scores. Low variation alone does not establish functional relevance; a dimension must also help distinguish the cluster from its background.

### 4. Signature Composition and Selection

Following ProStruct's `scripts/compose_signature.py`, dimensions are ranked by their individual compatibility AUROC and combined into candidate signatures. The starting composite score is the equal-weight arithmetic mean of the selected dimensions' Gaussian compatibility scores.

Incrementally adding ranked dimensions reveals whether a larger signature improves separation. Backward removal then tests whether discarding individual dimensions improves separation as measured by composite AUROC. A dimension that performs well alone may contribute little when combined with others.

The resulting signature records the selected dimensions, their cluster-specific means and spreads, background statistics, and selection diagnostics. ProStruct's precise selection rules provide an initial reference; their suitability for high-dimensional PLM embeddings remains to be evaluated.

### 5. Cluster-Level Classification and Validation

For a query protein, its embedding is scored against candidate cluster signatures. The intended prediction is a **functional cluster**, supported by compatibility scores and validated acceptance criteria.

Compatibility scores are not calibrated probabilities of cluster membership, and raw scores may not be directly comparable across clusters. Cluster-specific thresholds, ambiguous assignments, and rejection of unsupported predictions require validation.

Feature ranking, parameter fitting, subset selection, and threshold tuning on the same proteins produce descriptive, in-sample results. Independent evaluation must keep these steps within the training data and assess performance on held-out proteins, with controls for homology leakage. Evaluation should measure precision, recall, and functional coverage, particularly among proteins without reliable nearest-neighbor matches.

## PClass vs. PClass-Dark

| Feature | PClass | PClass-Dark |
|---|---|---|
| Representation | Learned PLM embeddings | PLM embeddings |
| Reference information | Individual annotated proteins | Predefined functional clusters and backgrounds |
| Classification principle | Nearest-neighbor similarity | Cluster-specific functional signatures |
| Features used | Full learned embedding representation | Selected informative embedding dimensions |
| Primary prediction | Function inferred from reference neighbors | Functional cluster |
| Intended strength | Proteins with reliable functional neighbors | Proteins lacking reliable nearest-neighbor matches |

The methods are complementary. PClass-Dark aims to extend functional coverage beyond the region where nearest-neighbor predictions are reliable, while maintaining acceptable precision.

## Project Status and Scope

**Research and early development.** ProStruct provides preliminary work on structural feature signatures; PClass-Dark will investigate whether the same profiling and composition strategy transfers to PLM embeddings. Signature selection, scoring, calibration, and benchmark results for this representation remain to be established.

PClass-Dark is intended to predict membership in functional clusters represented by the reference dataset. Its goal is to recover known functional information beyond reliable nearest-neighbor similarity. It does not inherently discover biological functions absent from the reference annotations.

## Related Projects

- **[PClass](https://github.com/newbooks/pclass):** PLM-based protein function classification using nearest-neighbor search.
- **ProStruct:** Preliminary structural feature profiling and cluster-signature composition, especially `scripts/profile_clusters.py` and `scripts/compose_signature.py` in the neighboring repository.
- **[profunc](https://github.com/newbooks/profunc):** Functional clustering of reference proteins using GO semantic similarity.

## License

PClass-Dark is released under the [MIT License](LICENSE).

---

*PClass-Dark: Finding conserved functional signals beyond nearest neighbors.*
