"""Independent empirical-count and fixed-signature scoring oracles."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest

from pclass_dark.artifacts import IntegrityError, identity, sha256, write_json
from pclass_dark.library_runtime import representation_contract
from pclass_dark.relative_scores import (
    RelativeScores,
    UnavailableDistributionError,
    _validate_array,
    compute_distributions,
)


def toy(tmp_path, c=(-2.0, -1.0), b=(-5.0, -3.0, -3.0, -1.0)):
    write_json(tmp_path / "manifest.json", {"test": True})
    reference = RelativeScores(
        tmp_path,
        SimpleNamespace(
            source_root=tmp_path,
            source_path=tmp_path,
            manifest_sha256=sha256(tmp_path / "manifest.json"),
        ),
        {"id": "fixture"},
        {
            "K": {
                "status": "ok",
                "C_role": "final-fit-in-sample",
                "B_role": "fixed-discovery-background-reused",
                "final_fit_id": "f",
                "scientific_payload_id": "s",
            }
        },
        True,
    )
    reference._cache = OrderedDict(K={"C": np.array(c), "B": np.array(b)})
    return reference


def test_empirical_ties_bounds_and_empty_queries(tmp_path):
    reference = toy(tmp_path)
    scores = np.array([-6.0, -3.0, -1.0, 0.0])
    result = reference.evaluate("K", scores)
    np.testing.assert_array_equal(result.background_n_ge, [4, 3, 1, 0])
    np.testing.assert_array_equal(result.background_tail_fraction, [1.0, 0.8, 0.4, 0.2])
    np.testing.assert_array_equal(
        result.background_percentile_le, [0.0, 0.75, 1.0, 1.0]
    )
    np.testing.assert_array_equal(result.cluster_percentile_le, [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(
        result.background_evidence, -np.log10([1.0, 0.8, 0.4, 0.2])
    )
    assert result.background_tail_fraction[1] != 1 - result.background_percentile_le[1]
    assert not result.cluster_percentile_le.flags.writeable
    empty = reference.evaluate("K", np.array([], dtype=float))
    assert empty.background_n_ge.shape == (0,) and empty.B_n == 4
    tied = toy(tmp_path, c=(0.0, 0.0), b=(0.0, 0.0))
    assert tied.evaluate("K", np.array([0.0])).background_tail_fraction[0] == 1


@pytest.mark.parametrize("bad", [[1], [True], [1 + 2j], [[1.0]], [np.nan], [np.inf]])
def test_invalid_queries_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        toy(tmp_path).evaluate("K", np.asarray(bad))


def test_missing_target_is_not_neutral_evidence(tmp_path):
    with pytest.raises(UnavailableDistributionError):
        toy(tmp_path).evaluate("missing", np.array([0.0]))


@pytest.mark.parametrize(
    "array,count",
    [
        (np.array([1.0, 0.0]), 2),
        (np.array([np.nan]), 1),
        (np.array([1.0], dtype="f4"), 1),
        (np.zeros((1, 2)), 2),
    ],
)
def test_invalid_reference_arrays_rejected(array, count):
    with pytest.raises(IntegrityError):
        _validate_array(array, count)


def test_fixed_signature_matches_independent_arithmetic_gaussian(library_data):
    _, _, _, matrix, contract = library_data()
    signature = {
        "schema": "pclass-dark.signature.v1",
        "cluster_id": "K",
        "status": "ok",
        "selected_dimension_ids": [0, 1],
        "core_dimensions": [0, 1],
        "candidate_dimensions": [0, 1],
        "representation_contract_id": identity(representation_contract(contract)),
        "threshold": {"kind": "above_maximum", "log_value": None, "score_value": None},
        "selected_dimensions": [
            {
                "dimension": d,
                "individual_rank": d + 1,
                "core": True,
                "mu": float(d),
                "sigma": 2.0,
                "effective_sigma": 2.0,
                "weight": 0.5,
            }
            for d in [0, 1]
        ],
    }
    populations = {"C": np.array([0, 0, 2, 4]), "B": np.array([1, 3, 5])}
    result = compute_distributions(
        signature, matrix, populations, embedding_contract=contract, block_size=2
    )
    for key, rows in populations.items():
        x = matrix[rows, :2].astype(float)
        expected = np.sort(
            np.log(
                (
                    np.exp(-0.5 * (x[:, 0] / 2) ** 2)
                    + np.exp(-0.5 * ((x[:, 1] - 1) / 2) ** 2)
                )
                / 2
            )
        )
        np.testing.assert_allclose(result[key], expected, rtol=1e-14, atol=1e-14)
        assert len(result[key]) == len(rows)  # Preserve identical-window multiplicity.
