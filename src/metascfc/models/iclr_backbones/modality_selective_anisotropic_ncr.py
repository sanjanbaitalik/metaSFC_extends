"""Modality-Selective Anisotropic Network-Constrained Ridge (MS-A-NCR).

Solves, using the repository's feature-count-scaled Ridge convention:
    min_β  ||y - Xβ||^2
         + λ_FC β_FC^T D(q;γ) β_FC
         + λ_SC ||β_SC||^2
         + λ_L  β_FC^T L_q β_FC

where X = [FC, SC], β = [β_FC; β_SC], D is a prior-dependent diagonal
shrinkage matrix, and L_q is an FC-only edge Laplacian from the prior.  The
written penalty is multiplied by ``s = 2 * n_edges`` internally, matching the
existing NCR/generalized dual-Ridge baseline convention.

SC receives ordinary Ridge (no prior).  FC receives anisotropic shrinkage
plus network smoothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from metascfc.models.iclr_backbones.network_constrained_ridge import (
    EdgeLaplacian,
    build_edge_laplacian,
    node_saliency_from_beta,
)


# ---------------------------------------------------------------------------
# Diagonal penalty
# ---------------------------------------------------------------------------

def compute_diagonal_penalty(
    edge_prior: np.ndarray,
    gamma: float,
    epsilon: float = 1e-3,
    normalize: bool = True,
) -> np.ndarray:
    """Compute D(q; gamma) = (epsilon + |q|)^{-gamma}.

    Parameters
    ----------
    edge_prior : (p,) continuous edge-level prior scores.
    gamma : exponent. gamma=0 gives isotropic penalty.
    epsilon : small constant to avoid division by zero.
    normalize : if True, normalize D so mean(D) = 1.

    Returns
    -------
    D : (p,) diagonal penalty weights.  High D_e means MORE shrinkage.
    """
    if gamma < 0:
        raise ValueError(f"gamma must be non-negative, got {gamma}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    q = np.asarray(edge_prior, dtype=np.float64)
    if not np.isfinite(q).all():
        raise ValueError("edge_prior contains NaN/Inf values")
    d = (epsilon + np.abs(q)) ** (-gamma)
    if not np.isfinite(d).all() or np.any(d <= 0):
        raise ValueError("diagonal penalty must be finite and strictly positive")
    if normalize:
        d = d / d.mean()
    return d


# ---------------------------------------------------------------------------
# Edge-level prior lifting
# ---------------------------------------------------------------------------

def lift_roi_to_edge(
    roi_prior: np.ndarray,
    n_rois: int,
    rule: str = "prod",
) -> np.ndarray:
    """Lift ROI-level prior to edge-level (upper triangle).

    Parameters
    ----------
    roi_prior : (n_rois,) ROI prior scores.
    n_rois : atlas size.
    rule : 'prod' or 'mean'.

    Returns
    -------
    edge_prior : (n_edges,) edge-level prior scores.
    """
    p = np.asarray(roi_prior, dtype=np.float64).ravel()
    if len(p) != n_rois:
        raise ValueError(f"roi_prior has {len(p)} entries; expected {n_rois}")
    iu = np.triu_indices(n_rois, k=1)
    pi, pj = p[iu[0]], p[iu[1]]
    if rule == "prod":
        return pi * pj
    elif rule == "mean":
        return 0.5 * (pi + pj)
    else:
        raise ValueError(f"Unknown lifting rule: {rule!r}")


# ---------------------------------------------------------------------------
# MS-A-NCR solver (dual Woodbury form)
# ---------------------------------------------------------------------------

@dataclass
class _MSANCRCache:
    """Precomputed quantities for one (gamma, lifting_rule) pair.

    These depend on the prior and gamma but NOT on the data or
    hyperparameters lambda_fc/lambda_sc/lambda_l.
    """
    D: np.ndarray                   # (n_edges,) diagonal penalty
    D_inv_sqrt: np.ndarray          # (n_edges,) 1/sqrt(D)
    active_indices: np.ndarray      # indices of active edges
    D_active: np.ndarray            # D at active indices
    active_laplacian: np.ndarray    # (n_active, n_active) Laplacian on active edges
    generalized_u: np.ndarray       # eigvecs of D_A^-1/2 L_A D_A^-1/2
    generalized_mu: np.ndarray      # corresponding non-negative eigenvalues
    n_edges: int
    n_rois: int
    gamma: float
    lifting: str

    @property
    def n_active(self) -> int:
        return int(len(self.active_indices))


def build_msancr_cache(
    roi_prior: np.ndarray,
    n_rois: int,
    gamma: float,
    lifting: str = "prod",
    top_k: int = 30,
    epsilon: float = 1e-3,
    weighting: str = "binary",
    couple_modalities: bool = False,
    normalize_laplacian: str = "sym",
    edge_laplacian: Optional[EdgeLaplacian] = None,
    prior_space: str = "node",
) -> _MSANCRCache:
    """Precompute quantities that depend on (prior, gamma) only.

    Parameters
    ----------
    prior_space : str, default "node"
        If "node", roi_prior is a (n_rois,) vector and lifting is applied.
        If "edge", roi_prior is already an (n_edges,) edge-level vector;
        lifting is skipped and the FIP edge vector is used directly as q_e.
    """
    if couple_modalities:
        raise ValueError("MS-A-NCR is FC-selective; couple_modalities must be False")
    if prior_space == "edge":
        edge_prior = np.asarray(roi_prior, dtype=np.float64).ravel()
        if len(edge_prior) != n_rois * (n_rois - 1) // 2:
            raise ValueError(
                f"edge prior has {len(edge_prior)} entries; "
                f"expected {n_rois * (n_rois - 1) // 2}"
            )
    else:
        edge_prior = lift_roi_to_edge(roi_prior, n_rois, lifting)
    D = compute_diagonal_penalty(edge_prior, gamma, epsilon, normalize=True)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D, 1e-30))

    edge_lap = edge_laplacian
    if edge_lap is None:
        if prior_space == "edge":
            # For edge-level priors, build a simple Laplacian from edge weights
            # Select top_k edges by prior weight as active
            top_k_actual = min(top_k, len(edge_prior))
            active_by_weight = np.argsort(edge_prior)[-top_k_actual:] if top_k_actual > 0 else np.array([], dtype=int)
            n_edges_total = n_rois * (n_rois - 1) // 2
            active_laplacian = np.zeros((len(active_by_weight), len(active_by_weight)), dtype=np.float64)
            from metascfc.models.iclr_backbones.network_constrained_ridge import EdgeLaplacian
            edge_lap = EdgeLaplacian(
                n_rois=n_rois,
                n_edges=n_edges_total,
                active_indices=active_by_weight,
                active_laplacian=active_laplacian,
                top_k=top_k,
                weighting="fip_edge",
                couple_modalities=False,
            )
        else:
            edge_lap = build_edge_laplacian(
                n_rois,
                prior_scores=roi_prior,
                top_k=top_k,
                weighting=weighting,
                couple_modalities=couple_modalities,
                normalize=normalize_laplacian,
            )
    if edge_lap.n_rois != n_rois or edge_lap.n_edges != len(D):
        raise ValueError("edge_laplacian dimensions do not match the prior/atlas")
    if edge_lap.couple_modalities:
        raise ValueError("MS-A-NCR requires an FC-only edge Laplacian")
    active = edge_lap.active_indices
    if len(active):
        d_active_inv_sqrt = D_inv_sqrt[active]
        whitened_laplacian = (
            d_active_inv_sqrt[:, None]
            * edge_lap.active_laplacian
            * d_active_inv_sqrt[None, :]
        )
        whitened_laplacian = 0.5 * (whitened_laplacian + whitened_laplacian.T)
        generalized_mu, generalized_u = np.linalg.eigh(whitened_laplacian)
        generalized_mu = np.clip(generalized_mu, 0.0, None)
    else:
        generalized_u = np.empty((0, 0), dtype=np.float64)
        generalized_mu = np.empty(0, dtype=np.float64)
    n_edges = int(n_rois * (n_rois - 1) / 2)

    return _MSANCRCache(
        D=D,
        D_inv_sqrt=D_inv_sqrt,
        active_indices=edge_lap.active_indices,
        D_active=D[edge_lap.active_indices],
        active_laplacian=edge_lap.active_laplacian,
        generalized_u=generalized_u,
        generalized_mu=generalized_mu,
        n_edges=n_edges,
        n_rois=n_rois,
        gamma=float(gamma),
        lifting=lifting,
    )


def _solve_msancr_kernel(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    cache: _MSANCRCache,
    lambda_fc: float,
    lambda_sc: float,
    lambda_l: float,
    fc_only: bool = False,
    eig_cache: Optional[Dict[float, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dual coefficients alpha and the kernel for predictions.

    Uses the Woodbury identity in the n_subjects x n_subjects dual space.

    Parameters
    ----------
    fc_only : bool, default False
        If True, SC block is omitted entirely (FC-only prior-aware Ridge).
        In this mode, lambda_sc is ignored and X_sc is not used.

    Returns (alpha, kernel_info) where kernel_info is a tuple of
    precomputed matrices for predictions on new data.
    """
    del eig_cache
    X_fc = np.asarray(X_fc, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if lambda_fc <= 0 or lambda_l < 0:
        raise ValueError("lambda_fc must be positive and lambda_l non-negative")
    if not fc_only and lambda_sc <= 0:
        raise ValueError("lambda_sc must be positive in FC+SC mode")
    if X_fc.ndim != 2:
        raise ValueError(f"Expected 2D FC design; got {X_fc.shape}")
    if not fc_only:
        X_sc = np.asarray(X_sc, dtype=np.float64)
        if X_sc.ndim != 2 or X_fc.shape != X_sc.shape:
            raise ValueError(f"Expected matched 2D FC/SC designs; got {X_fc.shape} and {X_sc.shape}")
    if X_fc.shape[1] != cache.n_edges or len(y) != len(X_fc):
        raise ValueError("MS-A-NCR design/target dimensions do not match the cache")

    n = X_fc.shape[0]
    n_active = cache.n_active
    n_edges = cache.n_edges

    active = cache.active_indices
    inactive_fc_mask = np.ones(n_edges, dtype=bool)
    inactive_fc_mask[active] = False
    inactive_fc_idx = np.where(inactive_fc_mask)[0]

    # --- FC active block: P_A = lambda_fc D_A + lambda_l L_A ---
    K_active = np.zeros((n, n), dtype=np.float64)
    if n_active > 0:
        whitened_active = X_fc[:, active] * cache.D_inv_sqrt[active][None, :]
        W = whitened_active @ cache.generalized_u
        denom = lambda_fc + lambda_l * cache.generalized_mu
        K_active = (W / denom) @ W.T

    # --- FC inactive block: P_I = lambda_fc * diag(D_I) ---
    K_inactive_fc = np.zeros((n, n), dtype=np.float64)
    if len(inactive_fc_idx) > 0:
        whitened_inactive = (
            X_fc[:, inactive_fc_idx]
            * cache.D_inv_sqrt[inactive_fc_idx][None, :]
        )
        K_inactive_fc = (whitened_inactive @ whitened_inactive.T) / lambda_fc

    # --- SC block ---
    if fc_only:
        K = K_active + K_inactive_fc
    else:
        K_sc = (1.0 / lambda_sc) * (X_sc @ X_sc.T)
        K = K_active + K_inactive_fc + K_sc

    K = K / float(max(1, n_edges * 2))

    # --- Solve ---
    alpha = np.linalg.solve(K + np.eye(n), y)

    if fc_only:
        return alpha, (K_active, K_inactive_fc, None, cache.generalized_mu)
    return alpha, (K_active, K_inactive_fc, K_sc, cache.generalized_mu)


def _predict_msancr(
    X_fc_new: np.ndarray,
    X_sc_new: np.ndarray,
    X_fc_train: np.ndarray,
    X_sc_train: np.ndarray,
    alpha: np.ndarray,
    cache: _MSANCRCache,
    lambda_fc: float,
    lambda_sc: float,
    lambda_l: float,
    fc_only: bool = False,
) -> np.ndarray:
    """Predict using the dual form."""
    n_active = cache.n_active
    n_edges = cache.n_edges
    active = cache.active_indices
    inactive_fc_mask = np.ones(n_edges, dtype=bool)
    inactive_fc_mask[active] = False
    inactive_fc_idx = np.where(inactive_fc_mask)[0]

    n_test = X_fc_new.shape[0]
    pred = np.zeros(n_test, dtype=np.float64)

    if lambda_fc <= 0 or lambda_l < 0:
        raise ValueError("lambda_fc must be positive and lambda_l non-negative")

    # FC active
    if n_active > 0:
        W_train = (
            X_fc_train[:, active] * cache.D_inv_sqrt[active][None, :]
        ) @ cache.generalized_u
        W_test = (
            X_fc_new[:, active] * cache.D_inv_sqrt[active][None, :]
        ) @ cache.generalized_u
        denom = lambda_fc + lambda_l * cache.generalized_mu
        K_test_active = (W_test / denom) @ W_train.T
        pred += K_test_active @ alpha

    # FC inactive (diagonal prior weighting, no Laplacian contribution)
    if len(inactive_fc_idx) > 0:
        X_new_w = X_fc_new[:, inactive_fc_idx] * cache.D_inv_sqrt[inactive_fc_idx][None, :]
        X_train_w = X_fc_train[:, inactive_fc_idx] * cache.D_inv_sqrt[inactive_fc_idx][None, :]
        pred += (X_new_w @ X_train_w.T) @ alpha / lambda_fc

    # SC (only if not fc_only)
    if not fc_only:
        pred += (1.0 / lambda_sc) * (X_sc_new @ X_sc_train.T) @ alpha

    scale = float(max(1, n_edges * 2))
    return pred / scale


def recover_msancr_beta(
    X_fc_train: np.ndarray,
    X_sc_train: np.ndarray,
    alpha: np.ndarray,
    cache: _MSANCRCache,
    lambda_fc: float,
    lambda_sc: float,
    lambda_l: float,
    fc_only: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover exact primal coefficients on standardized FC/SC features.

    The returned vectors satisfy ``prediction_z = X_fc @ beta_fc +
    X_sc @ beta_sc`` under the same feature-count-scaled objective used by
    :func:`_solve_msancr_kernel`.

    If fc_only=True, beta_sc is returned as zeros (SC is unused).
    """
    if lambda_fc <= 0 or lambda_l < 0:
        raise ValueError("lambda_fc must be positive and lambda_l non-negative")
    X_fc_train = np.asarray(X_fc_train, dtype=np.float64)
    if not fc_only:
        X_sc_train = np.asarray(X_sc_train, dtype=np.float64)
    alpha = np.asarray(alpha, dtype=np.float64).reshape(-1)
    if len(alpha) != len(X_fc_train):
        raise ValueError("Training designs and alpha have incompatible dimensions")

    n_edges = cache.n_edges
    scale = float(max(1, n_edges * 2))
    active = cache.active_indices
    inactive_mask = np.ones(n_edges, dtype=bool)
    inactive_mask[active] = False
    inactive = np.where(inactive_mask)[0]

    beta_fc = np.zeros(n_edges, dtype=np.float64)
    if len(active):
        rhs_w = cache.D_inv_sqrt[active] * (X_fc_train[:, active].T @ alpha)
        denom = lambda_fc + lambda_l * cache.generalized_mu
        beta_fc[active] = (
            cache.D_inv_sqrt[active]
            * (cache.generalized_u @ ((cache.generalized_u.T @ rhs_w) / denom))
            / scale
        )
    if len(inactive):
        beta_fc[inactive] = (
            (cache.D_inv_sqrt[inactive] ** 2)
            * (X_fc_train[:, inactive].T @ alpha)
            / (lambda_fc * scale)
        )

    if fc_only:
        beta_sc = np.zeros(X_fc_train.shape[1], dtype=np.float64)
    else:
        beta_sc = (X_sc_train.T @ alpha) / (lambda_sc * scale)
    return beta_fc, beta_sc


class ModalitySelectiveAnisotropicNCR:
    """MS-A-NCR: Modality-Selective Anisotropic Network-Constrained Ridge.

    FC gets anisotropic diagonal penalty + Laplacian smoothing.
    SC gets ordinary Ridge.

    Parameters
    ----------
    lambda_fc : float
        Ridge penalty for FC (modulates D).
    lambda_sc : float
        Ridge penalty for SC (isotropic). Ignored when fc_only=True.
    lambda_l : float
        Laplacian smoothing strength for FC.
    gamma : float
        Anisotropy exponent for D(q; gamma).
    cache : _MSANCRCache
        Precomputed prior-dependent quantities.
    n_rois : int
        Atlas size.
    fc_only : bool, default False
        If True, FC-only prior-aware Ridge (no SC branch).
    """

    def __init__(
        self,
        lambda_fc: float = 1.0,
        lambda_sc: float = 1.0,
        lambda_l: float = 0.0,
        gamma: float = 0.0,
        cache: Optional[_MSANCRCache] = None,
        n_rois: int = 116,
        fc_only: bool = False,
    ) -> None:
        self.lambda_fc = float(lambda_fc)
        self.lambda_sc = float(lambda_sc)
        self.lambda_l = float(lambda_l)
        self.gamma = float(gamma)
        self.cache = cache
        self.n_rois = n_rois
        self.fc_only = bool(fc_only)

        self.scaler_fc_: Optional[StandardScaler] = None
        self.scaler_sc_: Optional[StandardScaler] = None
        self.target_mean_ = 0.0
        self.target_std_ = 1.0
        self.alpha_: Optional[np.ndarray] = None
        self.X_fc_train_: Optional[np.ndarray] = None
        self.X_sc_train_: Optional[np.ndarray] = None

    def fit(
        self,
        X_fc: np.ndarray,
        X_sc: np.ndarray,
        y: np.ndarray,
    ) -> "ModalitySelectiveAnisotropicNCR":
        """Fit the model on training data.

        Parameters
        ----------
        X_fc : (n, n_edges) FC upper-triangle features.
        X_sc : (n, n_edges) SC upper-triangle features. Ignored when fc_only.
        y : (n,) target values.
        """
        X_fc = np.asarray(X_fc, dtype=np.float64)
        X_sc = np.asarray(X_sc, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if self.cache is None:
            raise ValueError("cache is required")
        if self.n_rois != self.cache.n_rois:
            raise ValueError("n_rois does not match the supplied cache")
        if not np.isclose(self.gamma, self.cache.gamma):
            raise ValueError(
                f"model gamma={self.gamma} does not match cache gamma={self.cache.gamma}"
            )

        # Standardize on training data
        self.scaler_fc_ = StandardScaler()
        X_fc_z = self.scaler_fc_.fit_transform(X_fc)

        if not self.fc_only:
            self.scaler_sc_ = StandardScaler()
            X_sc_z = self.scaler_sc_.fit_transform(X_sc)
        else:
            self.scaler_sc_ = None
            X_sc_z = np.zeros_like(X_fc_z)

        self.target_mean_ = float(y.mean())
        y_std = float(y.std())
        self.target_std_ = y_std if y_std >= 1e-8 else 1.0
        y_z = (y - self.target_mean_) / self.target_std_

        # Solve
        self.alpha_, _ = _solve_msancr_kernel(
            X_fc_z, X_sc_z, y_z, self.cache,
            self.lambda_fc, self.lambda_sc, self.lambda_l,
            fc_only=self.fc_only,
        )
        self.X_fc_train_ = X_fc_z
        self.X_sc_train_ = X_sc_z
        return self

    def predict(
        self,
        X_fc: np.ndarray,
        X_sc: np.ndarray,
    ) -> np.ndarray:
        """Predict on new data."""
        if self.alpha_ is None:
            raise RuntimeError("predict() called before fit()")
        X_fc = np.asarray(X_fc, dtype=np.float64)
        X_fc_z = self.scaler_fc_.transform(X_fc)

        if not self.fc_only:
            X_sc = np.asarray(X_sc, dtype=np.float64)
            X_sc_z = self.scaler_sc_.transform(X_sc)
        else:
            X_sc_z = np.zeros_like(X_fc_z)

        pred_z = _predict_msancr(
            X_fc_z, X_sc_z,
            self.X_fc_train_, self.X_sc_train_,
            self.alpha_, self.cache,
            self.lambda_fc, self.lambda_sc, self.lambda_l,
            fc_only=self.fc_only,
        )
        return pred_z * self.target_std_ + self.target_mean_

    def beta(self) -> np.ndarray:
        """Return exact FC+SC coefficients on standardized features."""
        if self.alpha_ is None:
            raise RuntimeError("beta() called before fit()")
        beta_fc, beta_sc = recover_msancr_beta(
            self.X_fc_train_, self.X_sc_train_, self.alpha_, self.cache,
            self.lambda_fc, self.lambda_sc, self.lambda_l,
            fc_only=self.fc_only,
        )
        return np.concatenate([beta_fc, beta_sc])

    def node_saliency(self) -> np.ndarray:
        """ROI-level saliency (biomarker) of the learned weights."""
        return node_saliency_from_beta(self.beta(), self.n_rois)


# ---------------------------------------------------------------------------
# Convenience function for pilot evaluation
# ---------------------------------------------------------------------------

def fit_predict_msancr(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    cache: _MSANCRCache,
    lambda_fc_grid: Sequence[float],
    lambda_sc_grid: Sequence[float],
    lambda_l_grid: Sequence[float],
    n_rois: int = 116,
    fc_only: bool = False,
) -> Tuple[np.ndarray, float, float, float, float, dict]:
    """Nested hyperparameter selection for MS-A-NCR.

    Returns (test_pred, best_lambda_fc, best_lambda_sc, best_lambda_l,
             best_val_rmse, selected_info).
    """
    X_fc = np.asarray(X_fc, dtype=np.float64)
    X_sc = np.asarray(X_sc, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    n_edges = cache.n_edges
    n_active = cache.n_active
    active = cache.active_indices
    inactive_fc_mask = np.ones(n_edges, dtype=bool)
    inactive_fc_mask[active] = False
    inactive_fc_idx = np.where(inactive_fc_mask)[0]

    # --- Inner training partition (selection) ---
    scaler_fc = StandardScaler()
    X_fc_train = scaler_fc.fit_transform(X_fc[train_idx])
    if fc_only:
        X_sc_train = np.zeros_like(X_fc_train)
    else:
        scaler_sc = StandardScaler()
        X_sc_train = scaler_sc.fit_transform(X_sc[train_idx])
    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_z = (y[train_idx] - y_mean) / y_std

    X_fc_val = scaler_fc.transform(X_fc[val_idx])
    if fc_only:
        X_sc_val = np.zeros_like(X_fc_val)
    else:
        X_sc_val = scaler_sc.transform(X_sc[val_idx])

    best = (float("inf"), None, None, None)

    for lambda_fc in lambda_fc_grid:
        for lambda_sc in lambda_sc_grid if not fc_only else [1.0]:
            for lambda_l in lambda_l_grid:
                try:
                    alpha, _ = _solve_msancr_kernel(
                        X_fc_train, X_sc_train, y_z, cache,
                        lambda_fc, lambda_sc, lambda_l,
                        fc_only=fc_only,
                    )
                    pred_z = _predict_msancr(
                        X_fc_val, X_sc_val,
                        X_fc_train, X_sc_train,
                        alpha, cache, lambda_fc, lambda_sc, lambda_l,
                        fc_only=fc_only,
                    )
                    pred = pred_z * y_std + y_mean
                    rmse = float(np.sqrt(np.mean((y[val_idx] - pred) ** 2)))
                    if rmse < best[0]:
                        best = (rmse, lambda_fc, lambda_sc, lambda_l)
                except Exception:
                    continue

    best_rmse, best_lfc, best_lsc, best_ll = best
    if best_lfc is None:
        best_lfc, best_lsc, best_ll = lambda_fc_grid[0], lambda_sc_grid[0], 0.0

    # --- Refit on train + val, predict test ---
    fit_idx = np.concatenate([train_idx, val_idx])
    final_fc = StandardScaler()
    X_fc_fit = final_fc.fit_transform(X_fc[fit_idx])
    if fc_only:
        X_sc_fit = np.zeros_like(X_fc_fit)
        final_sc = None
    else:
        final_sc = StandardScaler()
        X_sc_fit = final_sc.fit_transform(X_sc[fit_idx])
    X_fc_test = final_fc.transform(X_fc[test_idx])
    if fc_only:
        X_sc_test = np.zeros_like(X_fc_test)
    else:
        X_sc_test = final_sc.transform(X_sc[test_idx])

    fit_mean = float(y[fit_idx].mean())
    fit_std = max(float(y[fit_idx].std()), 1e-8)
    y_fit_z = (y[fit_idx] - fit_mean) / fit_std

    alpha, _ = _solve_msancr_kernel(
        X_fc_fit, X_sc_fit, y_fit_z, cache,
        best_lfc, best_lsc, best_ll,
        fc_only=fc_only,
    )
    pred_z = _predict_msancr(
        X_fc_test, X_sc_test,
        X_fc_fit, X_sc_fit,
        alpha, cache, best_lfc, best_lsc, best_ll,
        fc_only=fc_only,
    )
    pred = pred_z * fit_std + fit_mean

    info = {
        "best_lambda_fc": best_lfc,
        "best_lambda_sc": best_lsc,
        "best_lambda_l": best_ll,
        "n_active": n_active,
        "fc_only": fc_only,
    }
    return pred, best_lfc, best_lsc, best_ll, best_rmse, info
