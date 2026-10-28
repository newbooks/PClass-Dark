# Final-only production and optional research check — 2026-10-27

Implemented SPEC-0004/0005 on the uncommitted `issue#10` branch. This scoped
check used real existing setup-data vectors and exact cluster/background pools.
No PLM inference or signature discovery was run. Exact identities and counters
are in the [JSON record](2026-10-27-final-only-backfill.json).

Under the source root's transaction guard, copied only the six complete fit
receipts for `cluster_0001` from the verified first-improvement legacy run
`844465e3a89102ff1e44d3769995a54ed6ba8758f1a94129fdc4cc9378d275e4` into an isolated
`work/library-v2-validated/.checkpoints/` source tree. No source target marker,
Gaussian caches, prefix arrays or published source library were needed. Original
user outputs were not replaced. The new builder reconstructed the exact legacy
identity recipe from the verified setup inputs/full scope before import.

```bash
PYTHONPATH=src python -m pclass_dark build-library \
  --data-dir work/work/setup-data \
  --output-dir work/library-v2-validated \
  --cluster cluster_0001 --cross-validate
```

Results:

- Imported the all-C final signature and all five research signatures; **zero
  discovery fits computed**. Thirteen dimensions remain in the final signature.
- Generated final distributions for **20 C windows and 213,241 B windows**.
  Published array files total **1,706,344 bytes**, including NumPy headers.
- End-to-end elapsed time was **20.97 seconds**, including input verification,
  import, distribution scoring and optional research publication. It is not a
  measurement of new signature-discovery speed or full-library performance.
- Loaded the production/relative pair and research reports successfully. Compared
  original versus migrated final query scores/acceptance for 32 existing embedding
  rows: both matched bit for bit. No Gaussian cache files remained in the isolated
  output root.
- Production, relative and research IDs/paths, source data IDs and model payload
  bindings are recorded in JSON. Production stayed separate from five-fold
  research; their diagnostics were not merged into the final C distribution.

The measured process preceded the final additive production-manifest role flags;
those flags do not alter scientific parameters or score arrays. Final-code
integrity/loading/parity checks passed against the recorded artifacts.
The current user production run was not restarted or expanded to all clusters.
