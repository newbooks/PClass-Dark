# PClass-Dark frozen structural prediction heads

Unmodified ProStruct full-data RSA/ALL-CEF and Q3 heads, imported from library
`9ed4d06c9282266bc1b400f440e15ef61b9c5454195857750c1bd8a8dfa90315` at source
revision `fa8fa0da7a9f2ecb1ab096f979b3936dd2409d54`.

Both heads use final-layer 640-dimensional residue states from pinned ESM-2
`facebook/esm2_t30_150M_UR50D`, revision
`a695f6045e2e32885fa60af20c13cb35398ce30c`. Each uses 640->512->128->output
with exact GELU and task-specific training-population normalization stored in
the safetensors file. Training provenance, configuration, history and in-sample
metrics are preserved beside this card. No retraining was performed here.

RSA output is raw, unclipped ALL contextual exposure fraction, not conventional
maximum-ASA-normalized RSA. Q3 classes are C/E/H, using raw argmax without
smoothing. Sequence inference uses full-coverage overlapping windows of 1021
residues, minimum overlap 100, center-selected ownership with earlier-start ties.
Noncanonical positions use X context and missing output values.

These are predicted structural properties, not physical electrostatic fields or
experimentally observed structure. Full-data training metrics are resubstitution
diagnostics, not evidence of held-out accuracy. Historical chain-stratified
validation did not separate homologous families; unseen-family performance and
reranking improvement require independent evaluation. GPU/CPU batch rounding may
change low numerical bits and argmax at near ties.

Provenance and distribution: original structural training inputs derive from PDB
records; original ESM-2 weights are distributed separately under their upstream
terms. ProStruct's bundled RSA model card notes that its repository declares no
separate project-level license for generated weights. This import records origin
and does not assert a new upstream license or change applicable terms.
