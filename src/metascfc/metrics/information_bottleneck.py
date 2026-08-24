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


# ---------------------------------------------------------------------------
# MINE: Mutual Information Neural Estimation (lightweight critic)
# ---------------------------------------------------------------------------
class MINEEstimator:
    """Neural MI estimator with a small MLP critic (Donsker-Varadhan bound).

        I(X; Y) >= E_joint[T(x, y)] - log E_marginal[exp T(x, y')]

    where the marginal pairs are formed by permuting y against x.  The critic
    is intentionally tiny (two ``hidden``-unit layers by default) so that a
    handful of full-batch gradient steps add well under ~10% overhead to a
    training epoch at our sample sizes (n ~ 300).

    The log-mean-exp is max-shifted for numerical stability.  Estimates are
    lower bounds with finite-sample bias; as with the Gaussian proxy, only
    *relative* comparisons across matched runs are interpreted.
    """

    def __init__(
        self,
        x_dim: int,
        y_dim: int,
        hidden: int = 32,
        n_steps: int = 50,
        learning_rate: float = 1e-2,
        batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        import torch

        self.torch = torch
        if min(x_dim, y_dim) < 1 or hidden < 1 or n_steps < 1:
            raise ValueError("dims/hidden/n_steps must be positive")
        self.x_dim, self.y_dim = int(x_dim), int(y_dim)
        self.hidden, self.n_steps = int(hidden), int(n_steps)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        generator = torch.Generator().manual_seed(int(seed))
        self._generator = generator
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(x_dim + y_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1),
        )
        for p in self.critic.parameters():  # deterministic init
            torch.nn.init.normal_(p, std=0.1, generator=generator)

    def estimate(self, x: np.ndarray, y: np.ndarray) -> float:
        torch = self.torch
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        if len(x_arr) != len(y_arr):
            raise ValueError(f"x/y length mismatch: {len(x_arr)} vs {len(y_arr)}")
        x_arr = _as_2d(x_arr)
        if x_arr.std() > 0:
            x_arr = (x_arr - x_arr.mean(0)) / np.where(
                x_arr.std(0) > 0, x_arr.std(0), 1.0
            )
        y2 = _as_2d(y_arr)
        y_mu, y_sd = y2.mean(0), y2.std(0)
        y2 = (y2 - y_mu) / np.where(y_sd > 0, y_sd, 1.0)
        xt = torch.from_numpy(x_arr.astype(np.float32))
        yt = torch.from_numpy(y2.astype(np.float32))
        n = len(xt)

        opt = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        joint = torch.cat([xt, yt], dim=1)
        for _ in range(self.n_steps):
            if self.batch_size < n:
                idx = torch.randint(
                    n, (self.batch_size,), generator=self._generator
                )
                j_batch, xb, yb = joint[idx], xt[idx], yt[idx]
            else:
                j_batch, xb, yb = joint, xt, yt
            # Marginal pairing: permute y WITHIN the sampled batch.
            perm = torch.randperm(len(xb), generator=self._generator)
            marg = torch.cat([xb, yb[perm]], dim=1)
            t_joint = self.critic(j_batch).mean()
            t_marg = self.critic(marg)
            shift = t_marg.max().detach()
            log_e_t = shift + torch.log(torch.exp(t_marg - shift).mean())
            loss = -(t_joint - log_e_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            perm = torch.randperm(n, generator=self._generator)
            t_j = float(self.critic(joint).mean())
            t_m = self.critic(torch.cat([xt, yt[perm]], dim=1))
            shift = t_m.max()
            mi = t_j - (shift + torch.log(torch.exp(t_m - shift).mean()))
        return float(max(mi.item(), 0.0))


def random_project(x: np.ndarray, dims: int = 16, seed: int = 0) -> np.ndarray:
    """Fixed seeded Gaussian sketch of high-dim features (JL projection).

    Used to make I(X; Z) tractable for edge-space inputs (26k+ dims): MI is
    estimated against this low-dim summary, identically across all compared
    runs.
    """
    arr = _as_2d(x)
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((arr.shape[1], int(dims))) / np.sqrt(arr.shape[1])
    return arr @ proj


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
    """Accumulates per-epoch IB metrics inside a training loop.

    ``method`` selects the estimator:

    - ``"gaussian"`` (default): deterministic Gaussian/VIB proxy - the
      compression rate bound needs only Z, and I(Z;Y) uses a linear probe.
    - ``"mine"``: neural estimation with a small MLP critic.  I(Z;Y) is
      estimated between Z and Y; I(X;Z) is estimated between Z and a fixed
      random projection of the input features (``x_summary_dims``), which
      keeps the critic tiny and the overhead negligible.

    Both paths share the same call sites; switching method changes no other
    protocol detail, so cross-method comparisons of *trends* remain valid
    but exact values are not comparable (different estimands in the strict
    sense) - pick one method per experiment and report it.
    """

    def __init__(
        self,
        noise_floor: float = 0.05,
        method: str = "gaussian",
        mine_hidden: int = 32,
        mine_steps: int = 50,
        x_summary_dims: int = 16,
        seed: int = 0,
    ) -> None:
        if method not in ("gaussian", "mine"):
            raise ValueError(f"Unknown IB method '{method}'")
        self.noise_floor = float(noise_floor)
        self.method = method
        self.mine_hidden = int(mine_hidden)
        self.mine_steps = int(mine_steps)
        self.x_summary_dims = int(x_summary_dims)
        self.seed = int(seed)
        self.epochs: list[int] = []
        self.I_XZ: list[float] = []
        self.I_ZY: list[float] = []
        self.final: Optional[Dict[str, float]] = None
        self.alpha_final: Optional[list[float]] = None

    def _estimate(self, z: np.ndarray, y: np.ndarray,
                  x: Optional[np.ndarray]) -> Dict[str, float]:
        if self.method == "gaussian":
            z_arr = _as_2d(z)
            # Reference scale for the encoder noise floor: the input feature
            # variance when X is available (correct even for scalar latents
            # like z = X beta), otherwise the latent's own variance.
            if x is not None:
                reference = float(
                    np.asarray(x, dtype=np.float64).reshape(len(x), -1)
                    .var(axis=0, ddof=1).mean()
                )
            else:
                reference = float(z_arr.var(axis=0, ddof=1).mean())
            sigma_nu_sq = max(self.noise_floor, 1e-8) * max(reference, 1e-12)
            i_xz = float(0.5 * np.log1p(z_arr.var(axis=0, ddof=1) / sigma_nu_sq).sum())
            probe = predictive_mi(z, y)
            return {"I_XZ": i_xz, "I_ZY": probe["I_ZY"], "probe_r2": probe["probe_r2"]}
        from .information_bottleneck import MINEEstimator, random_project
        zy = MINEEstimator(
            x_dim=_as_2d(z).shape[1], y_dim=1, hidden=self.mine_hidden,
            n_steps=self.mine_steps, seed=self.seed,
        ).estimate(z, y)
        summary = random_project(x, dims=self.x_summary_dims, seed=self.seed) \
            if x is not None else z
        xz = MINEEstimator(
            x_dim=_as_2d(summary).shape[1], y_dim=_as_2d(z).shape[1],
            hidden=self.mine_hidden, n_steps=self.mine_steps, seed=self.seed + 1,
        ).estimate(summary, z)
        return {"I_XZ": xz, "I_ZY": zy, "probe_r2": float("nan")}

    def log_epoch(self, epoch: int, z: np.ndarray, y: np.ndarray,
                  x: Optional[np.ndarray] = None) -> None:
        metrics = self._estimate(z, y, x)
        self.epochs.append(int(epoch))
        self.I_XZ.append(metrics["I_XZ"])
        self.I_ZY.append(metrics["I_ZY"])

    def log_final(self, z: np.ndarray, y: np.ndarray,
                  x: Optional[np.ndarray] = None) -> Dict[str, float]:
        self.final = self._estimate(z, y, x)
        return dict(self.final)

    def to_dict(self) -> Dict:
        return {
            "noise_floor": self.noise_floor,
            "method": self.method,
            "epochs": self.epochs,
            "I_XZ": self.I_XZ,
            "I_ZY": self.I_ZY,
            "final": self.final,
        }


__all__ = [
    "IBEpochTracker",
    "MINEEstimator",
    "compression_mi",
    "information_bottleneck_metrics",
    "predictive_mi",
    "random_project",
]
