"""Audited whole-file extractor revisions sharing the same numerical recipe.

Only the enumerated versions are equivalent. Unknown hashes remain strict.
See SPEC-0021, 2026-11-10: progress wording and publication diagnostics/compatibility
changed; feature mathematics, receipts, model settings and schemas did not.
"""

CANONICAL_EXTRACTOR = "622223dd6288afcd583ece890ff82866e6f5fd790abb00393252dd2344fc1ea7"
EQUIVALENT_EXTRACTORS = frozenset(
    (
        CANONICAL_EXTRACTOR,
        "18cb0f5cddb55f0af6059abae634369d629de2e150161d8f4b8c5130c4a4b1d6",
        "4b34e839c8dabab1a6a1e326f953a271c2bd7059e806975133caf5e30b2d02d5",
    )
)


def canonical_extractor_hash(raw_hash):
    """Normalize only the independently reviewed source revisions above."""
    return CANONICAL_EXTRACTOR if raw_hash in EQUIVALENT_EXTRACTORS else raw_hash
