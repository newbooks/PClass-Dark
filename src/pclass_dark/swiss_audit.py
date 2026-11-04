"""Descriptive Swiss-Prot audit using the shared evaluation analysis/renderer."""

from .external_audit import _audit


def audit_swiss(
    input_path,
    output_dir=None,
    *,
    bins=10,
    cutoff=0.7,
    exclude_go="GO:0005515",
    truth_tsv=None,
):
    """Write four PNG plots and a summary TSV; return paths, rows, diagnostics.

    By default exclude whole queries annotated with GO:0005515 using verified
    saved truth or explicit truth_tsv. Set exclude_go=None for an unfiltered
    audit (truth_tsv is then unopened but protected against output collisions).
    Original annotations and scores are never recomputed or modified.

    Raises ValueError for invalid options/data or unsafe destinations, OSError
    for I/O failures, and RuntimeError for unavailable optional plotting support.
    Existing audit outputs are replaced per file after all outputs are staged.
    """
    if exclude_go not in (None, "GO:0005515"):
        raise ValueError("Swiss audit exclude_go must be GO:0005515 or None")
    return _audit(
        input_path,
        output_dir,
        bins=bins,
        cutoff=cutoff,
        exclude_go=exclude_go,
        truth_tsv=truth_tsv,
        prefix="swiss_audit",
        summary_title="Swiss-Prot audit",
    )
