"""Pure 19-descriptor aggregation (SPEC-0021; ProStruct SPEC-0026 provenance)."""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

import numpy as np

from .structural_models import BUNDLE_ID
from .structure import CANONICAL, StructuralPrediction

FEATURES = (
    "ln_length",
    "acid_fraction",
    "base_fraction",
    "surface_acid_over_all",
    "buried_acid_over_all",
    "surface_base_over_all",
    "buried_base_over_all",
    "q3_h_fraction",
    "q3_e_fraction",
    "q3_c_fraction",
    "net_charge_balance",
    "charged_fraction",
    "histidine_fraction",
    "surface_fraction",
    "surface_acid_over_surface",
    "surface_base_over_surface",
    "surface_charged_over_surface",
    "mean_kyte_doolittle_hydropathy",
    "sequence_charge_decoration",
)
HEADER = ("accession_id", *FEATURES, "cluster_id")
HYDROPATHY = dict(
    zip(
        "ACDEFGHIKLMNPQRSTVWY",
        (
            1.8,
            2.5,
            -3.5,
            -3.5,
            2.8,
            -0.4,
            -3.2,
            4.5,
            -3.9,
            3.8,
            1.9,
            -3.5,
            -1.6,
            -3.5,
            -4.5,
            -0.8,
            -0.7,
            4.2,
            -0.9,
            -1.3,
        ),
        strict=True,
    )
)
THRESHOLD = 0.05
LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeatureQC:
    length: int
    canonical_count: int
    noncanonical_count: int
    predicted_residue_count: int
    surface_count: int
    buried_count: int
    rsa_out_of_range_count: int
    scd_status: str
    scd_method: str
    undefined_features: tuple[str, ...]
    prediction_status: str


@dataclass(frozen=True)
class FeatureSummary:
    values: tuple[float | None, ...]
    qc: FeatureQC


def sequence_charge_decoration(sequence):
    """Signed Sawle–Ghosh Eq.14; FFT integer-lag check with direct fallback."""
    if not isinstance(sequence, str) or not re.fullmatch(r"[A-Z*?\-]+", sequence):
        raise ValueError("Expected nonempty uppercase ASCII protein sequence")
    if any(r not in CANONICAL for r in sequence):
        return None, "not_applicable"
    q = np.array(
        [1 if r in "KR" else -1 if r in "DE" else 0 for r in sequence], dtype=np.float64
    )
    if np.count_nonzero(q) < 2:
        return 0.0, "zero_shortcut"
    length = len(q)
    size = 1 << (2 * length - 2).bit_length()
    spectrum = np.fft.rfft(q, n=size)
    correlation = np.fft.irfft(spectrum * spectrum.conjugate(), n=size)[1:length]
    integers = np.rint(correlation)
    if np.isfinite(correlation).all() and np.all(
        np.abs(correlation - integers) <= 1e-6
    ):
        value = float(np.dot(integers, np.sqrt(np.arange(1, length)))) / length
        method = "fft"
    else:
        LOG.warning("SCD FFT integrity failed for length %d; direct fallback", length)
        value = (
            math.fsum(
                float(np.dot(q[:-d], q[d:])) * math.sqrt(d) for d in range(1, length)
            )
            / length
        )
        method = "direct_fallback"
    if not math.isfinite(value):
        raise ValueError("Nonfinite sequence charge decoration")
    return value, method


def summarize_structural_features(sequence, prediction) -> FeatureSummary:
    """Validate residue predictions and return full-sequence descriptors and QC.

    Undefined values are None. Bad identity/length, missing canonical predictions,
    or nonmissing unknown predictions raise ValueError. No model is loaded here.
    """
    if not isinstance(sequence, str) or not re.fullmatch(r"[A-Z*?\-]+", sequence):
        raise ValueError("Expected nonempty uppercase ASCII protein sequence")
    if not isinstance(prediction, StructuralPrediction) or (
        prediction.sequence != sequence or prediction.model_id != BUNDLE_ID
    ):
        raise ValueError("Structural prediction sequence/model identity mismatch")
    raw, labels = np.asarray(prediction.rsa), np.asarray(prediction.q3)
    length = len(sequence)
    if (
        raw.shape != (length,)
        or raw.dtype.kind != "f"
        or (labels.shape != (length,) or labels.dtype.kind not in "US")
    ):
        raise ValueError("Invalid residue prediction shape/dtype")
    # Promote BEFORE the comparison: float32(0.05) is greater than float64 0.05.
    raw = raw.astype(np.float64)
    known = np.fromiter((r in CANONICAL for r in sequence), dtype=bool)
    labels = labels.astype(str)
    if (
        not np.isfinite(raw[known]).all()
        or not np.isin(labels[known], ["C", "E", "H"]).all()
    ):
        raise ValueError("Missing/nonfinite canonical structural predictions")
    if not np.isnan(raw[~known]).all() or not (labels[~known] == "NA").all():
        raise ValueError("Unknown residues require nan/NA predictions")
    canonical = int(known.sum())
    acid = np.fromiter((r in "DE" for r in sequence), dtype=bool)
    base = np.fromiter((r in "KRH" for r in sequence), dtype=bool)
    charged = np.fromiter((r in "DEKR" for r in sequence), dtype=bool)
    surface = known & (raw > THRESHOLD)
    buried = known & ~surface
    ns = int(surface.sum())
    scd, method = sequence_charge_decoration(sequence)
    values = dict.fromkeys(FEATURES)
    values.update(
        ln_length=math.log(length),
        acid_fraction=float(acid.sum()) / length,
        base_fraction=float(base.sum()) / length,
        net_charge_balance=(sum(r in "KR" for r in sequence) - int(acid.sum()))
        / length,
        charged_fraction=float(charged.sum()) / length,
        histidine_fraction=sequence.count("H") / length,
        mean_kyte_doolittle_hydropathy=(
            math.fsum(HYDROPATHY[r] for r in sequence if r in CANONICAL) / canonical
            if canonical
            else None
        ),
        sequence_charge_decoration=scd,
    )
    if canonical:
        for name, mask in (("acid", acid), ("base", base)):
            values[f"surface_{name}_over_all"] = float((surface & mask).sum()) / length
            values[f"buried_{name}_over_all"] = float((buried & mask).sum()) / length
        for label in "HEC":
            values[f"q3_{label.lower()}_fraction"] = (
                float((labels == label).sum()) / length
            )
        values["surface_fraction"] = ns / length
        if ns:
            for name, mask in (("acid", acid), ("base", base), ("charged", charged)):
                values[f"surface_{name}_over_surface"] = (
                    float((surface & mask).sum()) / ns
                )
    output = tuple(values[f] for f in FEATURES)
    if any(v is not None and not math.isfinite(v) for v in output):
        raise ValueError("Nonfinite aggregated feature")
    return FeatureSummary(
        output,
        FeatureQC(
            length,
            canonical,
            length - canonical,
            canonical,
            ns,
            canonical - ns,
            int((known & ((raw < 0) | (raw > 1))).sum()),
            "defined" if canonical == length else "noncanonical",
            method,
            tuple(f for f, v in zip(FEATURES, output, strict=True) if v is None),
            "complete" if canonical else "no_canonical_residues",
        ),
    )
