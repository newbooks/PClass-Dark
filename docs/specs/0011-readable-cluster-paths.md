# SPEC-0011: Readable cluster directory names

- Date: 2026-10-29.
- Status: Implemented; current selected library migrated (2026-10-29).
- Purpose: Write `clusters/<cluster_id>/final.json` and corresponding relative
  and research directories; migrate the currently selected library in place.

Use safe, portable ASCII cluster IDs as path components, rejecting traversal,
separators, leading/trailing dots, reserved device names and case-fold collisions.
Keep existing hexadecimal names for internal checkpoints/migration caches to
preserve resumability. New public bundles declare
`cluster_directory_layout="cluster-id-v1"`; absent metadata means legacy hex.
Readers support both declared layouts without guessing from directory contents.

Directory layout is storage metadata, outside scientific identity inputs. Keep
production ID, signatures, selected dimensions, thresholds, metrics, traces and
score-array bytes unchanged. Changing the production manifest file inventory
changes its digest, so rebuild the relative companion's production binding and
content ID, update its metadata paths/manifest checksums and select the new pair.
Do not refit or recompute scores. Existing unsupported/legacy unsafe IDs stay
readable in hex layout, but cannot be written or migrated to unsafe plain paths.

Provide a reusable `migrate_library_paths(root)` operation and maintenance script.
Under the root transaction guard, validate the old pair, stage hard-linked payloads
with readable names, rewrite metadata atomically, and verify the new pair before
publication. Preserve a complete original selected generation under `.archives/`.
Move original payload trees into that backup during guarded publication and
update selection last. A durable journal supports rollback/recovery after an
exception or interruption; no other invocation can reuse partial publication.
Reject destination collisions rather than overwrite unrelated data. Repeated
migration of an already-readable selection is a no-op. Archive incompatible
research selection while retaining its research artifacts/backup; current local
selection has no research output.

Tests cover new output/read paths, backward-compatible legacy loading/imports,
unsafe IDs/collisions, migration identity/scientific/array parity, idempotence,
rollback/recovery and release packaging. Execute migration on the current local
library only after synthetic tests pass. Record backup location and verification;
update relevant specs/README and progress log. No model/scoring changes are made.

## Validation and local migration

The selected production scientific ID remains
`5f9117a77f64b30426661dff87533a462a8c0771c7150ae76eb903574a5e8100`.
Its relative companion is now
`cab57dc6d25c492f233f59197d2cc62c4bf2c05cdc89d19fbe3580863ba3e5b0`.
All 3443 production and 3443 companion cluster directories use cluster IDs.
Original generation: `data/library/.archives/cluster-paths-693395768393481da3f98b38a140a6fa/`.
Checksums confirm 17215 unchanged cluster payloads. Real explanation/list commands
and release inventory (17238 files) passed. Migration tests cover query-score and
ABC parity, legacy scoped/full loading, listing/export, idempotence, build reuse,
rollback and process-crash recovery: 13 passed. Initial relevant suites: 63 passed.
Full suite: 252 passed, 2 opt-in real-checkpoint tests skipped. Latest migration
regressions separately passed (13 tests). Ruff/format checks passed for 51 Python
files and scoped whitespace checks passed. See the
[progress log](../research/progress.md) for commands, artifacts and limitations.
