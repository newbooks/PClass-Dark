"""Conservative recovery of completed native inference before score publication."""

from . import external_evaluation_store as store
from . import swiss_evaluation_collect as collect
from .swiss_evaluation_types import FORMAT

# Reviewed scoring-only warning change: project_dark/project_pclass are identical.
# Do not exempt arbitrary future edits: this file also holds evidence projections.
_SCORING_ONLY_EQUIVALENTS = frozenset(
    {
        "4a107aec6c5cbedded1eafdaa30d8a3183a8342ce21088b8745e5b93330df799",
        "155e99d75fcda430e6f105df81f1cce62e08c445e99b539a720adbe17c893b94",
    }
)


def compatible_contract(contract):
    """Normalize only reviewed scoring versions, retaining every other field."""
    result = dict(contract)
    if isinstance(result.get("code"), dict):
        code = dict(result["code"])
        name = "pclass_dark.external_evaluation_scoring"
        if code.get(name) in _SCORING_ONLY_EQUIVALENTS:
            code[name] = "reviewed-unknown-go-warning-equivalence-v1"
        result["code"] = code
    return result


def completed_generation(state, cohort, worker, count):
    """Return a verified compatible completed generation, or None.

    No inference, artifact mutation, or selection file is required. Invalid
    compatible artifacts raise integrity errors through the native verifier.
    """
    expected = compatible_contract(worker.contract)
    for path in sorted((state / "generations" / worker.engine).glob("*/manifest.json")):
        manifest = store.read(path)
        if any(
            manifest.get(k) != v
            for k, v in {
                "format": FORMAT,
                "engine": worker.engine,
                "input_id": cohort["input_id"],
                "chunk_count": count,
            }.items()
        ):
            continue
        if not manifest["chunks"]:
            continue
        _, receipt = collect.raw_receipt(
            state, worker.engine, manifest["chunks"][0]["id"]
        )
        if compatible_contract(receipt["contract"]) != expected:
            continue
        verified = collect.verify_generation(
            state, worker.engine, path.parent.name, cohort["input_id"]
        )
        collect.LOG.info(
            "%s: reused completed generation %s (%s chunks); 100.0%% finished",
            worker.engine,
            verified["id"][:12],
            len(verified["chunks"]),
        )
        return verified
    return None
