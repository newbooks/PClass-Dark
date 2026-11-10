"""Memory estimates use analytical sizes without allocating large target tables."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pclass_dark import explore_ml as ml
from pclass_dark.ml_model import _admit
from pclass_dark.siac import make_soft_targets, sparse_target_memory


def test_target_memory_matches_arrays_and_caps_width():
    targets = make_soft_targets(
        np.ones((3, 3)), active_labels=[0, 1, 2], alpha=0, neighbors=99, gamma=1
    )
    actual = sum(
        a.nbytes
        for a in (
            targets.active_labels,
            targets.alpha,
            targets.neighbor_indices,
            targets.weights,
        )
    )
    estimate = sparse_target_memory(3, 99, 4)
    assert actual == 108 == estimate["table_bytes"]
    assert estimate["host_bytes"] == 3 * 108 + 4 * 2 * 24
    assert estimate["device_bytes"] == 108 + 4 * 2 * 24
    assert estimate == sparse_target_memory(3, 2, 4)
    assert sparse_target_memory(10000, 9999, 1)["table_bytes"] == 1200000000


@pytest.mark.parametrize("operation", ["explore", "final"])
def test_large_sparse_targets_admission(operation, monkeypatch):
    torch = pytest.importorskip("torch")
    C = 10000

    # Shape/byte metadata only; no large allocation or training.
    class Array:
        nbytes = 0

        def __len__(self):
            return C

    X = y = S = Array()
    cohort = SimpleNamespace(X=X, y=y, labels=range(C))
    similarity = SimpleNamespace(S=S)
    config = ml.ExploreMLConfig(
        device="auto",
        workers=4,
        threads=1,
        neighbors=9999,
        batch_size=1,
        memory_budget_gib=100,
    )
    settings = {"knots": config.knots, "neighbors": 9999, "batch_size": 1}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *a: (1024**3, 2 * 1024**3))
    monkeypatch.setattr(ml, "_available_cpus", lambda: 4)

    def admit(device, requested="auto", budget=100, neighbors=9999):
        if operation == "explore":
            return ml._resources(
                replace(
                    config,
                    device=requested,
                    memory_budget_gib=budget,
                    neighbors=neighbors,
                ),
                cohort,
                similarity,
                device,
            )
        return _admit(
            X, y, S, dict(settings, neighbors=neighbors), device, requested, budget
        )

    # Small target tables fit this GPU; the large configuration must fall back.
    assert admit("cuda:0", neighbors=1)[0] == "cuda:0"
    device, audit = admit("cuda:0")
    assert device == "cpu"
    assert audit["sparse_targets"]["table_bytes"] == 1200000000
    with pytest.raises(ValueError, match="GPU"):
        admit("cuda:0", requested="cuda:0")
    _, small = admit("cpu", neighbors=1)
    _, large = admit("cpu")
    small_peak = small["estimated_peak_bytes"]
    large_peak = large["estimated_peak_bytes"]
    assert large_peak > small_peak + 4_000_000_000
    budget = (small_peak + large_peak) / 2 / 1024**3
    if operation == "explore":
        _, reduced = admit("cpu", budget=budget)
        assert reduced["effective_workers"] < small["effective_workers"]
    else:
        with pytest.raises(ValueError, match="increase memory budget"):
            admit("cpu", budget=budget)
    with pytest.raises(ValueError, match="increase memory budget"):
        admit("cpu", budget=1)
