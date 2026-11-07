"""Independent analytical oracles for all 19 descriptors."""

import math

import numpy as np
import pytest

from pclass_dark.structural_features import (
    FEATURES,
    sequence_charge_decoration,
    summarize_structural_features,
)
from pclass_dark.structural_models import BUNDLE_ID
from pclass_dark.structure import StructuralPrediction


def prediction(sequence, raw=None, labels=None, dtype=np.float64):
    return StructuralPrediction(
        sequence,
        np.array(
            [0.1 if r in "ACDEFGHIKLMNPQRSTVWY" else np.nan for r in sequence]
            if raw is None
            else raw,
            dtype=dtype,
        ),
        np.array(
            ["C" if r in "ACDEFGHIKLMNPQRSTVWY" else "NA" for r in sequence]
            if labels is None
            else labels
        ),
        BUNDLE_ID,
        "cpu",
    )


def test_analytical_19_feature_oracle_and_unknown():
    seq = "DEKRH"
    raw = [0.1, 0.05, 0.2, -0.1, 1.2]
    labels = list("HECCH")
    result = summarize_structural_features(seq, prediction(seq, raw, labels))
    expected = [
        math.log(5),
        0.4,
        0.6,
        0.2,
        0.2,
        0.4,
        0.2,
        0.4,
        0.2,
        0.4,
        0,
        0.8,
        0.2,
        0.6,
        1 / 3,
        2 / 3,
        2 / 3,
        -3.72,
        (1 - 2 * math.sqrt(2) - math.sqrt(3)) / 5,
    ]
    np.testing.assert_allclose(result.values, expected, atol=1e-14, rtol=1e-14)
    assert len(FEATURES) == 19 and result.qc.rsa_out_of_range_count == 2
    assert result.qc.surface_count == 3 and result.qc.buried_count == 2
    other = summarize_structural_features(
        seq + "X", prediction(seq + "X", raw + [np.nan], labels + ["NA"])
    )
    assert other.values[-1] is None and other.values[-2] == pytest.approx(-3.72)
    assert sum(other.values[7:10]) == pytest.approx(5 / 6)


def test_threshold_float32_boundary_and_no_surface():
    exact = summarize_structural_features("D", prediction("D", [0.05]))
    rounded = summarize_structural_features(
        "D", prediction("D", [0.05], dtype=np.float32)
    )
    assert exact.qc.surface_count == 0 and rounded.qc.surface_count == 1
    assert exact.values[14:17] == (None, None, None)
    assert exact.values[-1] == 0 and exact.values[0] == 0


def test_unknown_all_unknown_and_whole_sequence_denominators():
    result = summarize_structural_features("XX", prediction("XX"))
    assert len(result.qc.undefined_features) == 13
    assert result.qc.prediction_status == "no_canonical_residues"
    assert result.values[1:3] == (0, 0)
    assert result.values[17:] == (None, None)
    seq = "DDDDAA"
    result = summarize_structural_features(seq, prediction(seq))
    assert result.values[1] == pytest.approx(4 / 6)
    assert (
        result.values[1] != (1 + 0) / 2
    )  # Unweighted window-ratio averaging is wrong.


def test_scd_independent_pair_sums_and_fft_fallback(monkeypatch):
    rng = np.random.default_rng(21)
    for length in range(1, 30):
        seq = "".join(rng.choice(list("DEKRAG"), length))
        q = [1 if r in "KR" else -1 if r in "DE" else 0 for r in seq]
        expected = (
            sum(
                q[i] * q[j] * math.sqrt(j - i)
                for i in range(length)
                for j in range(i + 1, length)
            )
            / length
        )
        assert sequence_charge_decoration(seq)[0] == pytest.approx(expected, abs=1e-12)
    original = np.fft.irfft
    monkeypatch.setattr(np.fft, "irfft", lambda *a, **k: original(*a, **k) + 0.2)
    value, method = sequence_charge_decoration("DEKRH")
    assert method == "direct_fallback"
    assert value == pytest.approx((1 - 2 * math.sqrt(2) - math.sqrt(3)) / 5)


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "lowercase",
        "identity",
        "model",
        "shape",
        "nan",
        "infinity",
        "label",
        "unknown_raw",
        "unknown_label",
    ],
)
def test_invalid_predictions(kind):
    seq = "AX"
    p = prediction(seq)
    if kind == "empty":
        seq = ""
    if kind == "lowercase":
        seq = "ax"
    if kind == "identity":
        seq = "CX"
    if kind == "model":
        p = StructuralPrediction(seq, p.rsa, p.q3, "bad", "cpu")
    if kind == "shape":
        p = StructuralPrediction(seq, p.rsa[:1], p.q3, BUNDLE_ID, "cpu")
    if kind == "nan":
        p.rsa[0] = np.nan
    if kind == "infinity":
        p.rsa[0] = np.inf
    if kind == "label":
        p.q3[0] = "NA"
    if kind == "unknown_raw":
        p.rsa[1] = 0.2
    if kind == "unknown_label":
        p.q3[1] = "C"
    with pytest.raises(ValueError):
        summarize_structural_features(seq, p)


@pytest.mark.parametrize("sequence", ["", "ac", "A C", None])
def test_scd_rejects_invalid_direct_inputs(sequence):
    with pytest.raises(ValueError):
        sequence_charge_decoration(sequence)
