"""Similarity-Informed Additive Classifier: pure targets, splines and batched fits.

Optional sklearn/Torch imports occur only inside scientific operations. Models are
plain arrays, never pickle, with one spline curve per source feature and class.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import IntegrityError, read_json, write_json
from .ml_data import FEATURES, readonly
from .ml_progress import Progress

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class SparseTargets:
    active_labels: np.ndarray
    neighbor_indices: np.ndarray
    weights: np.ndarray
    alpha: np.ndarray

    @property
    def fingerprint(self):
        from .artifacts import identity

        return identity(
            {
                "active_labels": self.active_labels.tolist(),
                "neighbor_indices": self.neighbor_indices.tolist(),
                "weights": self.weights.tolist(),
                "alpha": self.alpha.tolist(),
            }
        )


def sparse_target_memory(label_count, neighbors, batch_size):
    """Conservative host/device bytes for sparse targets and batched loss buffers.

    Host reserves original arrays, a temporary copy and CPU tensors. Device
    reserves transferred tensors. Both include gather/product/backward workspace.
    Hard targets currently allocate the same tables and must use this estimate.
    """
    width = min(neighbors, max(0, label_count - 1))
    table = label_count * (width * 12 + 12)
    batch = batch_size * width * 24
    return {
        "table_bytes": table,
        "host_bytes": 3 * table + batch,
        "device_bytes": table + batch,
    }


def make_soft_targets(similarity, *, active_labels, alpha, neighbors, gamma):
    """Construct normalized similarity targets over active global label indices."""
    active = np.asarray(active_labels, dtype=np.int64)
    S = similarity.S if hasattr(similarity, "S") else np.asarray(similarity)
    if (
        not np.isfinite(alpha)
        or not 0 <= alpha < 1
        or not np.isfinite(gamma)
        or gamma <= 0
        or type(neighbors) is not int
        or neighbors < 1
        or active.ndim != 1
        or len(active) < 2
        or not np.array_equal(active, np.unique(active))
        or active.min() < 0
        or active.max() >= len(S)
        or S.shape != (len(S), len(S))
        or not np.isfinite(S).all()
        or (S < 0).any()
        or (S > 1).any()
    ):
        raise ValueError("Invalid soft-target parameters/similarity/active labels")
    width = min(neighbors, len(active) - 1)
    indices = np.zeros((len(active), width), dtype=np.int64)
    weights = np.zeros((len(active), width), dtype=np.float32)
    effective = np.zeros(len(active), dtype=np.float32)
    for i, label in enumerate(active):
        scores = S[label, active]
        candidates = [j for j in range(len(active)) if j != i and scores[j] > 0]
        candidates.sort(key=lambda j: (-scores[j], active[j]))
        selected = candidates[:width]
        if selected and alpha:
            logs = np.log(scores[selected])
            with np.errstate(over="ignore", under="ignore"):
                scaled = np.exp(gamma * (logs - logs.max()))
            w = scaled / scaled.sum()
            indices[i, : len(selected)] = selected
            weights[i, : len(selected)] = w
            effective[i] = alpha
            if not np.isclose(weights[i].sum(), 1, atol=1e-6):
                raise IntegrityError("Soft target weights fail normalization")
    return SparseTargets(
        readonly(active.copy()),
        readonly(indices),
        readonly(weights),
        readonly(effective),
    )


def soft_loss(logits, local_y, targets):
    """Exactly soft cross-entropy using sparse neighbor gathers, no N-by-C targets."""
    import torch

    logp = torch.log_softmax(logits, dim=1)
    device = logits.device
    a = torch.as_tensor(targets.alpha.copy(), device=device)[local_y]
    own = logp.gather(1, local_y[:, None]).squeeze(1)
    indices = torch.as_tensor(targets.neighbor_indices.copy(), device=device)[local_y]
    weights = torch.as_tensor(targets.weights.copy(), device=device)[local_y]
    neighbor = (logp.gather(1, indices) * weights).sum(1)
    return -((1 - a) * own + a * neighbor).mean()


@dataclass(frozen=True)
class Preprocessing:
    medians: np.ndarray
    training_min: np.ndarray
    training_max: np.ndarray
    means: np.ndarray
    offsets: tuple[int, ...]
    splines: tuple[tuple[np.ndarray, np.ndarray] | None, ...]
    degree: int = 3

    def transform(self, X):
        """Evaluate centered bases in float64 and return float32 design rows."""
        from scipy.interpolate import BSpline

        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(FEATURES) or np.isinf(X).any():
            raise ValueError("Expected finite-or-NA [N,18] features")
        result = np.empty((len(X), self.offsets[-1]), dtype=np.float64)
        for j, spec in enumerate(self.splines):
            if spec is None:
                continue
            value = np.where(np.isnan(X[:, j]), self.medians[j], X[:, j])
            value = np.clip(value, self.training_min[j], self.training_max[j])
            t, c = spec
            start, end = self.offsets[j : j + 2]
            result[:, start:end] = BSpline(t, c, self.degree)(value)[:, :-1]
        result -= self.means
        return result.astype(np.float32)


def fit_preprocessing(X, *, knots=8):
    """Fit training-only imputation, quantile splines and basis centering."""
    from sklearn.preprocessing import SplineTransformer

    X = np.asarray(X, dtype=np.float64)
    if (
        X.ndim != 2
        or X.shape[1] != 18
        or not len(X)
        or np.isinf(X).any()
        or np.isnan(X).all(axis=0).any()
        or type(knots) is not int
        or knots < 2
    ):
        raise ValueError("Invalid training features/knots (including all-NA column)")
    medians = np.nanmedian(X, axis=0)
    clean = np.where(np.isnan(X), medians, X)
    minima, maxima = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
    splines, offsets, means = [], [0], []
    for j in range(18):
        q = np.unique(
            np.quantile(clean[:, j], np.linspace(0, 1, knots), method="linear")
        )
        if len(q) < 2:
            splines.append(None)
            offsets.append(offsets[-1])
            continue
        transformer = SplineTransformer(
            knots=q[:, None],
            degree=3,
            extrapolation="constant",
            include_bias=False,
            order="C",
        ).fit(clean[:, j : j + 1])
        spline = transformer.bsplines_[0]
        splines.append((readonly(spline.t.copy()), readonly(spline.c.copy())))
        # Chunk the mean, rather than retaining the whole expanded N-by-B matrix.
        total = np.zeros(transformer.n_features_out_, dtype=np.float64)
        for begin in range(0, len(clean), 4096):
            total += transformer.transform(clean[begin : begin + 4096, j : j + 1]).sum(
                axis=0
            )
        means.extend(total / len(clean))
        offsets.append(offsets[-1] + transformer.n_features_out_)
    return Preprocessing(
        readonly(medians),
        readonly(minima),
        readonly(maxima),
        readonly(np.array(means)),
        tuple(offsets),
        tuple(splines),
    )


@dataclass(frozen=True)
class SIACModel:
    preprocess: Preprocessing
    weight: np.ndarray
    bias: np.ndarray
    active_labels: np.ndarray
    selected_epochs: int
    fit_seed: int
    parameters: dict
    training_example_count: int

    def logits(self, X):
        return self.preprocess.transform(X) @ self.weight.T + self.bias

    def explain(self, X, predicted_local, runner_local):
        """Return per-feature margin contributions and bias differences."""
        basis = self.preprocess.transform(X)
        delta = self.weight[predicted_local] - self.weight[runner_local]
        result = np.zeros((len(X), 18), dtype=np.float64)
        for j, (a, b) in enumerate(
            zip(self.preprocess.offsets[:-1], self.preprocess.offsets[1:], strict=True)
        ):
            result[:, j] = (basis[:, a:b] * delta[:, a:b]).sum(1)
        return result, self.bias[predicted_local] - self.bias[runner_local]


@dataclass(frozen=True)
class PredictionBatch:
    start: int
    indices: np.ndarray
    logits: np.ndarray
    probabilities: np.ndarray
    contributions: np.ndarray
    bias_difference: np.ndarray


def predict_siac(model, X, *, batch_size=1024):
    """Yield top-five local labels and additive explanations in bounded batches."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(X), batch_size):
        block = X[start : start + batch_size]
        logits = model.logits(block)
        if not np.isfinite(logits).all():
            raise RuntimeError("Nonfinite prediction logits")
        # Stable sorting preserves lexicographic active-label order for exact ties.
        order = np.argsort(-logits, axis=1, kind="stable")[:, : min(5, len(model.bias))]
        shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        contrib, bias = model.explain(block, order[:, 0], order[:, 1])
        yield PredictionBatch(
            start,
            order,
            np.take_along_axis(logits, order, axis=1),
            np.take_along_axis(probabilities, order, axis=1),
            contrib,
            bias,
        )


def configure_torch(seed, *, threads=1):
    """Configure deterministic float32 optimization; no implicit model loading."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def resolve_device(requested, *, threads=1):
    """Probe deterministic fitting on requested device; auto may fall back to CPU."""
    torch = configure_torch(42, threads=threads)
    choices = ["cuda:0", "cpu"] if requested == "auto" else [requested]
    errors = []
    for device in choices:
        try:
            device = str(torch.device("cuda:0" if device == "cuda" else device))
            a = torch.zeros(
                (2, 2), device=device, dtype=torch.float32, requires_grad=True
            )
            optimizer = torch.optim.AdamW([a], foreach=False, fused=False)
            torch.log_softmax(a, 1).sum().backward()
            optimizer.step()
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            return device, errors
        except (RuntimeError, AssertionError, ValueError) as exc:
            errors.append(f"{device}: {exc}")
    raise RuntimeError("No usable requested device: " + "; ".join(errors))


def fit_siac(
    X,
    y,
    *,
    preprocess,
    targets,
    parameters,
    seed,
    device,
    validation=None,
    epochs=None,
    progress_label="SIAC",
):
    """Fit one active-label model; select epochs only using supplied inner validation.

    Returns (canonical model, immutable epoch-record tuple). No outer test data is
    accepted by this operation. Explicit epochs performs a fresh fixed-length refit.
    """
    torch = configure_torch(seed, threads=parameters.get("threads", 1))
    active = targets.active_labels
    local_y = np.searchsorted(active, y)
    if (local_y >= len(active)).any() or not np.array_equal(active[local_y], y):
        raise ValueError("Training labels not in active set")
    weight = torch.zeros(
        (len(active), preprocess.offsets[-1]),
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    bias = torch.zeros(
        len(active), device=device, dtype=torch.float32, requires_grad=True
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": [weight], "weight_decay": parameters["weight_decay"]},
            {"params": [bias], "weight_decay": 0.0},
        ],
        lr=parameters["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        foreach=False,
        fused=False,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batch = parameters["batch_size"]

    def design(values):
        result = np.empty((len(values), preprocess.offsets[-1]), dtype=np.float32)
        for begin in range(0, len(values), 4096):
            result[begin : begin + 4096] = preprocess.transform(
                values[begin : begin + 4096]
            )
        return result

    LOG.info(
        "%s: preparing spline bases for %s training examples",
        progress_label,
        f"{len(X):,}",
    )
    train_basis = design(X)
    val_basis, val_y = None, None
    if validation is not None:
        vx, vy = validation
        mask = np.isin(vy, active)
        if not mask.any():
            raise ValueError("No supported inner-validation labels")
        val_basis = design(vx[mask])
        val_y = np.searchsorted(active, vy[mask])
    logs, best, best_epoch, stale = [], float("inf"), 1, 0
    max_epochs = epochs if epochs is not None else parameters["max_epochs"]
    # Precompute sparse targets once on device, not once per mini-batch.
    target_alpha = torch.tensor(targets.alpha.copy(), device=device)
    target_indices = torch.tensor(targets.neighbor_indices.copy(), device=device)
    target_weights = torch.tensor(targets.weights.copy(), device=device)

    def loss_fn(logits, labels):
        if not np.any(targets.alpha):
            return torch.nn.functional.cross_entropy(logits, labels)
        logp = torch.log_softmax(logits, 1)
        own = logp.gather(1, labels[:, None]).squeeze(1)
        neighbor = (
            logp.gather(1, target_indices[labels]) * target_weights[labels]
        ).sum(1)
        a = target_alpha[labels]
        return -((1 - a) * own + a * neighbor).mean()

    LOG.info("%s: optimizing up to %d epochs on %s", progress_label, max_epochs, device)
    for epoch in range(1, max_epochs + 1):
        progress = Progress(
            LOG,
            f"{progress_label} epoch {epoch}/{max_epochs}",
            total=len(X),
            unit="examples",
            announce=False,
        )
        order = torch.randperm(len(X), generator=generator).numpy()
        total = 0.0
        for start in range(0, len(X), batch):
            ix = order[start : start + batch]
            basis = torch.from_numpy(train_basis[ix]).to(device)
            labels = torch.as_tensor(local_y[ix].copy(), device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(basis @ weight.T + bias, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite SIAC training loss")
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in (weight, bias)):
                raise RuntimeError("Nonfinite SIAC gradient")
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in (weight, bias)):
                raise RuntimeError("Nonfinite SIAC parameters")
            total += float(loss.detach().cpu()) * len(ix)
            progress.update(start + len(ix))
        val_loss = None
        if validation is not None:
            val_total = 0.0
            with torch.no_grad():
                for start in range(0, len(val_y), batch):
                    b = torch.from_numpy(val_basis[start : start + batch]).to(device)
                    labels = torch.as_tensor(
                        val_y[start : start + batch].copy(), device=device
                    )
                    loss = loss_fn(b @ weight.T + bias, labels)
                    val_total += float(loss.cpu()) * len(labels)
            val_loss = val_total / len(val_y)
            if not np.isfinite(val_loss):
                raise RuntimeError("Nonfinite validation loss")
            if val_loss < best - parameters["min_delta"]:
                best, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1
        else:
            best_epoch = epoch
        logs.append((epoch, len(X), total / len(X), val_loss, best_epoch))
        LOG.info(
            "%s epoch %d/%d loss=%.6f",
            progress_label,
            epoch,
            max_epochs,
            total / len(X),
        )
        if validation is not None and stale >= parameters["patience"]:
            LOG.info(
                "%s: early stopping at epoch %d; selected epoch %d",
                progress_label,
                epoch,
                best_epoch,
            )
            break
    w, b = weight.detach().cpu().numpy(), bias.detach().cpu().numpy()
    # Remove the unidentifiable common class-score curve/intercept.
    w = w.astype(np.float64) - w.astype(np.float64).mean(axis=0)
    b = b.astype(np.float64) - b.astype(np.float64).mean()
    model = SIACModel(
        preprocess,
        readonly(w.astype(np.float32)),
        readonly(b.astype(np.float32)),
        readonly(active.copy()),
        best_epoch,
        seed,
        dict(parameters),
        len(X),
    )
    return model, tuple(logs)


def save_model(path, model, global_labels):
    """Serialize research model as schema-validated JSON and plain numerical NPZ."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    p = model.preprocess
    arrays = {
        "weight": model.weight,
        "bias": model.bias,
        "medians": p.medians,
        "training_min": p.training_min,
        "training_max": p.training_max,
        "basis_mean": p.means,
    }
    for j, spline in enumerate(p.splines):
        if spline is not None:
            arrays[f"f{j:02d}_t"], arrays[f"f{j:02d}_c"] = spline
    np.savez(path / "model.npz", **arrays)
    write_json(
        path / "model.json",
        {
            "schema": "pclass-dark.explore-ml.siac-model.v1",
            "feature_names": list(FEATURES),
            "active_global_indices": model.active_labels.tolist(),
            "global_labels": list(global_labels),
            "basis_offsets": list(p.offsets),
            "basis_width": p.offsets[-1],
            "class_count": len(model.bias),
            "degree": 3,
            "include_bias": False,
            "extrapolation": "constant",
            "constant_features": [j for j, x in enumerate(p.splines) if x is None],
            "selected_epochs": model.selected_epochs,
            "fit_seed": model.fit_seed,
            "parameters": model.parameters,
            "gauge": "class_mean_zero",
            "dtype": "float32",
            "training_example_count": model.training_example_count,
            "arrays": {
                k: {"dtype": str(v.dtype), "shape": list(v.shape)}
                for k, v in arrays.items()
            },
        },
    )


def load_model(path):
    """Load only finite plain arrays; reject malformed dimensions/class/basis maps."""
    path = Path(path)
    m = read_json(path / "model.json")
    try:
        metadata_keys = {
            "schema",
            "feature_names",
            "active_global_indices",
            "global_labels",
            "basis_offsets",
            "basis_width",
            "class_count",
            "degree",
            "include_bias",
            "extrapolation",
            "constant_features",
            "selected_epochs",
            "fit_seed",
            "parameters",
            "gauge",
            "dtype",
            "training_example_count",
            "arrays",
        }
        if set(m) != metadata_keys:
            raise IntegrityError("Unexpected model metadata keys")
        if (
            m["schema"] != "pclass-dark.explore-ml.siac-model.v1"
            or m["feature_names"] != list(FEATURES)
            or m["degree"] != 3
            or m["include_bias"] is not False
            or m["extrapolation"] != "constant"
            or m["dtype"] != "float32"
            or m["gauge"] != "class_mean_zero"
        ):
            raise IntegrityError("Invalid SIAC model metadata")
        with np.load(path / "model.npz", allow_pickle=False) as archive:
            arrays = {k: archive[k] for k in archive.files}
        if set(arrays) != set(m["arrays"]):
            raise IntegrityError("Model array inventory mismatch")
        for key, value in arrays.items():
            record = m["arrays"][key]
            if (
                str(value.dtype) != record["dtype"]
                or list(value.shape) != record["shape"]
                or value.dtype.kind != "f"
                or not np.isfinite(value).all()
            ):
                raise IntegrityError("Invalid model array schema/values")
        offsets = tuple(m["basis_offsets"])
        if (
            any(type(i) is not int for i in m["active_global_indices"])
            or type(m["selected_epochs"]) is not int
            or m["selected_epochs"] < 1
            or type(m["training_example_count"]) is not int
            or m["training_example_count"] < 1
        ):
            raise IntegrityError("Invalid model label/epoch metadata")
        active = np.array(m["active_global_indices"], dtype=np.int64)
        labels = m["global_labels"]
        if (
            len(offsets) != 19
            or offsets[0] != 0
            or any(type(x) is not int for x in offsets)
            or any(a > b for a, b in zip(offsets[:-1], offsets[1:], strict=True))
            or offsets[-1] != m["basis_width"]
            or len(active) != m["class_count"]
            or len(active) < 2
            or labels != sorted(set(labels))
            or not np.array_equal(active, np.unique(active))
            or active.min() < 0
            or active.max() >= len(labels)
            or arrays["weight"].shape != (len(active), offsets[-1])
            or arrays["bias"].shape != (len(active),)
            or arrays["weight"].dtype != np.float32
            or arrays["bias"].dtype != np.float32
        ):
            raise IntegrityError("Invalid model dimensions/class map")
        for key in ("medians", "training_min", "training_max"):
            if arrays[key].shape != (18,) or arrays[key].dtype != np.float64:
                raise IntegrityError("Invalid preprocessing vector")
        if (
            arrays["basis_mean"].shape != (offsets[-1],)
            or arrays["basis_mean"].dtype != np.float64
            or m["constant_features"]
            != [j for j in range(18) if offsets[j] == offsets[j + 1]]
        ):
            raise IntegrityError("Invalid basis centering")
        splines, expected = (
            [],
            {"weight", "bias", "medians", "training_min", "training_max", "basis_mean"},
        )
        for j in range(18):
            if j in m["constant_features"]:
                if offsets[j] != offsets[j + 1]:
                    raise IntegrityError("Constant feature has nonempty basis")
                splines.append(None)
            else:
                t, c = arrays[f"f{j:02d}_t"], arrays[f"f{j:02d}_c"]
                expected.update((f"f{j:02d}_t", f"f{j:02d}_c"))
                if (
                    t.dtype != np.float64
                    or c.dtype != np.float64
                    or t.ndim != 1
                    or (np.diff(t) < 0).any()
                    or c.ndim != 2
                    or c.shape[0] != len(t) - 4
                    or c.shape[0] != c.shape[1]
                    or c.shape[1] - 1 != offsets[j + 1] - offsets[j]
                ):
                    raise IntegrityError("Invalid persisted spline")
                splines.append((readonly(t), readonly(c)))
        if set(arrays) != expected:
            raise IntegrityError("Unexpected model arrays")
        if (
            np.any(abs(arrays["weight"].astype(float).mean(axis=0)) > 1e-6)
            or abs(arrays["bias"].astype(float).mean()) > 1e-6
            or (arrays["training_min"] > arrays["training_max"]).any()
        ):
            raise IntegrityError("Invalid model gauge/support")
        p = Preprocessing(
            readonly(arrays["medians"]),
            readonly(arrays["training_min"]),
            readonly(arrays["training_max"]),
            readonly(arrays["basis_mean"]),
            offsets,
            tuple(splines),
        )
        return SIACModel(
            p,
            readonly(arrays["weight"]),
            readonly(arrays["bias"]),
            readonly(active),
            m["selected_epochs"],
            m["fit_seed"],
            m["parameters"],
            m["training_example_count"],
        )
    except (KeyError, TypeError, IndexError, ValueError) as exc:
        raise IntegrityError(f"Malformed SIAC model: {exc}") from exc
