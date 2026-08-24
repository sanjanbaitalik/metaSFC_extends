"""Information Bottleneck tracking for prior-gated FC-SC models.

Implements the two Tishby-style quantities tracked during the ICLR 2027
dual-task matrix experiments:

- ``I(X; Z)`` - compression: how much input (connectome) information the
  latent representation carries.
- ``I(Z; Y)`` - predictive capacity: how much of the target the latent
  preserves.

Estimator (Gaussian / VIB proxy)
--------------------------------
Both quantities use simple Gaussian channel assumptions on the embeddings,
which is deterministic, hyperparameter-light, and stable for our sample
sizes (n ~ 300 per split):

- The stochastic encoder is approximated as q(z|x) = N(mu_d(x), sigma_nu^2)
  with a *fixed noise floor* sigma_nu^2 = ``noise_floor`` * mean feature
  variance (identical across all runs of an experiment, so comparisons are
  meaningful).  Under this model the VIB rate bound gives

      I(X; Z) ~= 0.5 * sum_d log(1 + Var(mu_d) / sigma_nu^2)   [nats]

- For I(Z; Y) we fit a linear probe y_hat = Z w (+ intercept), compute its
  R^2, and invert the Gaussian channel

      I(Z; Y) ~= -0.5 * log(1 - R^2)                            [nats]

  clipped to R^2 in [0, 1 - 1e-4] for numerical safety.

Caveats (documented, not hidden): these are proxies, not tight MI values;
they assume (approximately) Gaussian marginals and a linear decoder for the
Z -> Y channel.  All scientific claims therefore rest on *relative*
comparisons between matched architecture/prior variants evaluated with the
same estimator settings - exactly the dual-task matrix design.  Under the
Inductive Bottleneck hypothesis, a mismatched prior yields high I(X;Z)
(the gate over-filters toward irrelevant structure yet the representation
still varies strongly with the input) but low I(Z;Y) (the surviving signal
is uninformative about the target), while a matched prior optimizes the
trade-off.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def _as_2d(z: np.ndarray) -> np.ndarray:
    arr = np.asarray(z, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D latent, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Latent contains NaN/Inf values")
    return arr


def compression_mi(z: np.ndarray, noise_floor: float = 0.05) -> float:
    """I(X; Z) proxy [nats]: Gaussian-channel rate of the embedding.

    Parameters
    ----------
    z : np.ndarray, shape (n,) or (n, d)
        Latent representations (one row per subject).  A 1-D vector (e.g.
        the scalar projection x^T beta of a linear model) is treated as one
        channel dimension.
    noise_floor : float
        Encoder noise level as a fraction of the mean feature variance
        (fixed across all compared runs).
    """
    arr = _as_2d(z)
    if len(arr) < 3:
        raise ValueError("Need >= 3 samples to estimate variance")
    variances = arr.var(axis=0, ddof=1)
    reference = float(variances.mean())
    if reference <= 1e-12:
        return 0.0
    sigma_nu_sq = max(float(noise_floor), 1e-8) * reference
    return float(0.5 * np.log1p(variances / sigma_nu_sq).sum())


def predictive_mi(z: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> Dict[str, float]:
    """I(Z; Y) proxy [nats] via a linear probe's R^2 (Gaussian channel).

    Returns {"I_ZY": ..., "probe_r2": ...}.
    """
    arr = _as_2d(z)
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(arr) != len(y_vec):
        raise ValueError(f"z/y length mismatch: {len(arr)} vs {len(y_vec)}")
    if len(y_vec) < 3:
        raise ValueError("Need >= 3 samples to fit the probe")
    zc = arr - arr.mean(axis=0, keepdims=True)
    yc = y_vec - y_vec.mean()
    # Tiny ridge for stability when d approaches n.
    gram = zc.T @ zc + ridge * len(y_vec) * np.eye(zc.shape[1])
    w = np.linalg.solve(gram, zc.T @ yc)
    resid = yc - zc @ w
    ss_res = float(resid @ resid)
    ss_tot = float(yc @ yc)
    r2 = 0.0 if ss_tot <= 1e-12 else max(0.0, min(1.0 - 1e-4, 1.0 - ss_res / ss_tot))
    return {"I_ZY": float(-0.5 * np.log1p(-r2)), "probe_r2": float(r2)}


def information_bottleneck_metrics(
    z: np.ndarray,
    y: np.ndarray,
    noise_floor: float = 0.05,
) -> Dict[str, float]:
    """Both IB quantities for one latent/target pair.

    Returns {"I_XZ": ..., "I_ZY": ..., "probe_r2": ...} (nats for the MI
    entries).
    """
    out = {"I_XZ": compression_mi(z, noise_floor=noise_floor)}
    out.update(predictive_mi(z, y))
    return out


class IBEpochTracker:
    """Accumulates per-epoch IB metrics inside a training loop."""

    def __init__(self, noise_floor: float = 0.05) -> None:
        self.noise_floor = float(noise_floor)
        self.epochs: list[int] = []
        self.I_XZ: list[float] = []
        self.I_ZY: list[float] = []
        self.final: Optional[Dict[str, float]] = None

    def log_epoch(self, epoch: int, z: np.ndarray, y: np.ndarray) -> None:
        metrics = information_bottleneck_metrics(z, y, noise_floor=self.noise_floor)
        self.epochs.append(int(epoch))
        self.I_XZ.append(metrics["I_XZ"])
        self.I_ZY.append(metrics["I_ZY"])

    def log_final(self, z: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        self.final = information_bottleneck_metrics(z, y, noise_floor=self.noise_floor)
        return dict(self.final)

    def to_dict(self) -> Dict:
        return {
            "noise_floor": self.noise_floor,
            "epochs": self.epochs,
            "I_XZ": self.I_XZ,
            "I_ZY": self.I_ZY,
            "final": self.final,
        }


__all__ = [
    "IBEpochTracker",
    "compression_mi",
    "information_bottleneck_metrics",
    "predictive_mi",
]
