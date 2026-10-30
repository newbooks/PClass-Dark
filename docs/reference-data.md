# Profunc reference dependency

Current behavior is specified by [SPEC-0012](specs/0012-profunc-dependency.md).
Install profunc in the same environment, for example `pip install -e ../profunc`.
After updating its publication, run:

```bash
pclass-dark setup-data
pclass-dark build-library
pclass-dark explain-signature --list
```

`setup-data` calls `profunc.runtime.resolve_runtime()`, honoring profunc's
current-directory then packaged-default resolution and integrity validation.
It reads current memberships directly. Profunc's portable runtime excludes the
sequence FASTA, so preparation finds the original dataset generation through
profunc's workspace module. If unavailable, supply
`--reference-fasta /path/to/source/accession.fasta` (Python: `reference_fasta=`).
Its size and checksum must match the active dataset manifest. There is no
automatic sequence download or fallback to PClass-Dark's old copied references.
Changed inputs invalidate preparation and downstream fit identities; unchanged
inputs can be reused. `build-library` consumes the selected prepared data and
embeddings, so run preparation first. Commands do not upgrade profunc itself.

The Python `prepare_data(reference_dir=...)` override still accepts a custom
manifest-bound reference pair for integrations; it cannot be combined with
`reference_fasta`. Custom references receive descriptions only if their content
contract exactly matches the active profunc reference.

Legacy copies under `src/pclass_dark/resources/reference/` and the static
catalog/exporter have been removed. Do not copy profunc data into package
resources. Git history and the research log retain earlier snapshot provenance.

## Live reference and description behavior

The static catalog and its maintenance exporter have been retired. Listing
imports profunc's cluster records and composed-GO describer at runtime. It uses
the validated active runtime and only attaches labels when membership/FASTA hashes
match those recorded in signature fit inputs. Missing bindings in older
signatures and incompatible source references show `—` with a reason in JSON.
Rebuild signatures to establish the new binding.
