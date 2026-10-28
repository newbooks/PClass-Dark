"""Immutable signature snapshots and NumPy-only compatibility scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

from .artifacts import IntegrityError, identity
from .embeddings import ASSET_HASHES
from .setup_data import MODEL, REVISION
from .signature_metrics import accepted

CONTRACT_FIELDS = (
    "model",
    "revision",
    "assets",
    "layer",
    "width",
    "window_length",
    "pooling",
    "residue_policy",
    "inference_dtype",
    "output_dtype",
)


class UnavailableSignatureError(ValueError):
    """The target has no statistically supported fitted signature."""


def representation_contract(contract):
    result = {key: contract[key] for key in CONTRACT_FIELDS}
    result["assets"] = dict(result["assets"])
    expected = {
        "model": MODEL,
        "revision": REVISION,
        "layer": 30,
        "width": 640,
        "window_length": 1022,
        "pooling": "attention-and-not-special-mean-f32-v1",
        "residue_policy": "exact-vocab-ACDEFGHIKLMNPQRSTVWYXBZUO-v1",
        "inference_dtype": "float32",
        "output_dtype": "<f4",
    }
    if (
        any(result[k] != v for k, v in expected.items())
        or not isinstance(result["assets"], Mapping)
        or any(result["assets"].get(k) != v for k, v in ASSET_HASHES.items())
    ):
        raise IntegrityError("Unsupported residue-only representation contract")
    return result


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value


def validate_signature(signature, *, library_id=None, cluster_id=None, fit_id=None):
    try:
        if (
            signature["schema"]
            not in {"pclass-dark.signature.v1", "pclass-dark.signature.v2"}
            or library_id is not None
            and signature["library_id"] != library_id
            or cluster_id is not None
            and signature["cluster_id"] != cluster_id
            or fit_id is not None
            and signature["fit_id"] != fit_id
        ):
            raise IntegrityError("Signature binding/schema mismatch")
        if signature["status"] not in {
            "ok",
            "empty_background",
            "insufficient_training",
            "insufficient_core",
        }:
            raise IntegrityError("Unknown signature status")
        if signature["schema"] == "pclass-dark.signature.v2":
            for key in ("fit_identity", "scientific_payload_id"):
                value = signature[key]
                if (
                    not isinstance(value, str)
                    or len(value) != 64
                    or any(c not in "0123456789abcdef" for c in value)
                ):
                    raise IntegrityError("Invalid v2 signature identity")
            if signature["evaluation_mode"] not in {
                "all-C-in-sample",
                "four-fold-training-held-out-positive",
            }:
                raise IntegrityError("Invalid signature evaluation mode")
        dimensions = signature["selected_dimensions"]
        if not isinstance(dimensions, (list, tuple)):
            raise IntegrityError("Invalid selected dimensions")
        if signature["status"] != "ok":
            if dimensions or signature["threshold"] is not None:
                raise IntegrityError(
                    "Unsupported signature contains scoring parameters"
                )
            return
        ids = [d["dimension"] for d in dimensions]
        if (
            len(ids) < 2
            or len(set(ids)) != len(ids)
            or any(type(d) is not int or not 0 <= d < 640 for d in ids)
            or list(signature["selected_dimension_ids"]) != ids
        ):
            raise IntegrityError("Invalid dimension IDs/order")
        ranks = [d["individual_rank"] for d in dimensions]
        if ranks != sorted(set(ranks)) or any(
            type(r) is not int or not 1 <= r <= 640 for r in ranks
        ):
            raise IntegrityError("Invalid parameter rank order")
        core = list(signature["core_dimensions"])
        candidate = list(signature["candidate_dimensions"])
        if (
            len(core) != 2
            or len(set(core)) != 2
            or not set(core) <= set(ids)
            or not set(ids) <= set(candidate)
            or len(candidate) != len(set(candidate))
        ):
            raise IntegrityError("Invalid protected core/candidate binding")
        for record in dimensions:
            if (
                any(
                    not isinstance(record[k], (int, float))
                    or not math.isfinite(record[k])
                    for k in ("mu", "sigma", "effective_sigma", "weight")
                )
                or record["sigma"] < 0
                or record["effective_sigma"] <= 0
                or record["effective_sigma"] < record["sigma"]
                or record["weight"] != 1 / len(ids)
                or record["core"] != (record["dimension"] in core)
            ):
                raise IntegrityError("Invalid Gaussian parameters or weights")
        accepted(np.empty(0), signature["threshold"])
        threshold = signature["threshold"]
        if threshold["kind"] == "log_score_ge" and threshold["score_value"] != math.exp(
            threshold["log_value"]
        ):
            raise IntegrityError("Threshold log/display mismatch")
    except (
        KeyError,
        TypeError,
        AttributeError,
        IndexError,
        ValueError,
        OverflowError,
    ) as exc:
        if isinstance(exc, IntegrityError):
            raise
        raise IntegrityError("Malformed signature record") from exc


def score_signature(signature, vectors, *, embedding_contract):
    """Return (float64 log compatibility, accepted bool), without assigning clusters."""
    validate_signature(signature)
    if signature["status"] != "ok":
        raise UnavailableSignatureError(
            f"Unavailable {signature['cluster_id']}: {signature['status']}"
        )
    if (
        identity(representation_contract(embedding_contract))
        != signature["representation_contract_id"]
    ):
        raise IntegrityError("Query embedding representation differs from the library")
    vectors = np.asarray(vectors)
    if (
        vectors.ndim != 2
        or vectors.shape[1] != 640
        or vectors.dtype.kind != "f"
        or not np.isfinite(vectors).all()
    ):
        raise ValueError(
            "Expected a finite real floating-point (n,640) embedding matrix"
        )
    from .profiles import gaussian_log

    total = None
    for record in signature["selected_dimensions"]:
        kernel = gaussian_log(
            vectors[:, record["dimension"]], record["mu"], record["effective_sigma"]
        )
        if total is None:
            total = kernel
        else:
            np.logaddexp(total, kernel, out=total)
    scores = total - math.log(len(signature["selected_dimensions"]))
    return scores, accepted(scores, signature["threshold"])
