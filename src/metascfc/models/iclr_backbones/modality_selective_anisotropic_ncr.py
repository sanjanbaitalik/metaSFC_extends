"""Modality-Selective Anisotropic Network-Constrained Ridge (MS-A-NCR).

Solves:
    min_β  ||y - Xβ||^2
         + λ_FC β_FC^T D(q;γ) β_FC
         + λ_SC ||β_SC||^2
         + λ_L  β_FC^T L_q β_FC

where X = [FC, SC], β = [β_FC; β_SC], D is a prior-dependent diagonal
shrinkage matrix, and L_q is an FC-only edge Laplacian from the prior.

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
    _LaplacianEig,
    build_edge_laplacian,
    factor_laplacian_eig,
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
    d = (epsilon + np.abs(np.asarray(edge_prior, dtype=np.float64))) ** (-gamma)
    if normalize and d.mean() > 1e-12:
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
    eig: _LaplacianEig              # eigendecomposition of L_q
    n_edges: int
    n_rois: int


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
) -> _MSANCRCache:
    """Precompute quantities that depend on (prior, gamma) only."""
    edge_prior = lift_roi_to_edge(roi_prior, n_rois, lifting)
    D = compute_diagonal_penalty(edge_prior, gamma, epsilon, normalize=True)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D, 1e-30))

    edge_lap = build_edge_laplacian(
        n_rois,
        prior_scores=roi_prior,
        top_k=top_k,
        weighting=weighting,
        couple_modalities=couple_modalities,
        normalize=normalize_laplacian,
    )
    eig = factor_laplacian_eig(edge_lap)
    n_edges = int(n_rois * (n_rois - 1) / 2)

    return _MSANCRCache(
        D=D,
        D_inv_sqrt=D_inv_sqrt,
        active_indices=edge_lap.active_indices,
        D_active=D[edge_lap.active_indices],
        active_laplacian=edge_lap.active_laplacian,
        eig=eig,
        n_edges=n_edges,
        n_rois=n_rois,
    )


def _solve_msancr_kernel(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    cache: _MSANCRCache,
    lambda_fc: float,
    lambda_sc: float,
    lambda_l: float,
    eig_cache: Optional[Dict[float, Tuple[np.ndarray, np.ndarray]]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute dual coefficients alpha and the kernel for predictions.

    Uses the Woodbury identity in the n_subjects x n_subjects dual space.

    Parameters
    ----------
    eig_cache : optional dict mapping ratio -> (eigvals, eigvecs) for the
        active penalty matrix.  When provided, repeated solves with the
        same (gamma, lifting) but different (lambda_fc, lambda_l) reuse
        eigendecompositions via the ratio r = lambda_l / lambda_fc.

    Returns (alpha, kernel_info) where kernel_info is a tuple of
    precomputed matrices for predictions on new data.
    """
    n = X_fc.shape[0]
    n_active = cache.eig.n_active
    n_edges = cache.n_edges

    active = cache.active_indices
    inactive_fc_mask = np.ones(n_edges, dtype=bool)
    inactive_fc_mask[active] = False
    inactive_fc_idx = np.where(inactive_fc_mask)[0]

    # --- FC active block: P_fc_active = lambda_fc * diag(D_active) + lambda_l * L_q ---
    # P_active = lambda_fc * (diag(D_active) + (lambda_l/lambda_fc) * L_active)
    # Eigendecomposition depends only on ratio r = lambda_l / lambda_fc
    K_active = np.zeros((n, n), dtype=np.float64)
    if n_active > 0:
        if lambda_fc > 0:
            ratio = lambda_l / lambda_fc
        else:
            ratio = float(lambda_l) if lambda_l > 0 else 0.0

        if eig_cache is not None and ratio in eig_cache:
            eigvals, eigvecs = eig_cache[ratio]
        else:
            P_active = lambda_fc * np.diag(cache.D_active) + lambda_l * cache.active_laplacian
            P_active = 0.5 * (P_active + P_active.T)
            eigvals, eigvecs = np.linalg.eigh(P_active)
            eigvals = np.maximum(eigvals, 1e-12)
            if eig_cache is not None:
                eig_cache[ratio] = (eigvals.copy(), eigvecs.copy())

        W = X_fc[:, active] @ eigvecs
        K_active = (W / eigvals) @ W.T

    # --- FC inactive block: P_fc_inactive = lambda_fc * I (no prior weighting) ---
    K_inactive_fc = np.zeros((n, n), dtype=np.float64)
    if len(inactive_fc_idx) > 0:
        K_inactive_fc = (1.0 / lambda_fc) * (X_fc[:, inactive_fc_idx] @ X_fc[:, inactive_fc_idx].T)

    # --- SC block: P_sc = lambda_sc * I ---
    K_sc = (1.0 / lambda_sc) * (X_sc @ X_sc.T)

    # --- Total kernel ---
    K = K_active + K_inactive_fc + K_sc
    K = K / float(max(1, n_edges * 2))

    # --- Solve ---
    alpha = np.linalg.solve(K + np.eye(n), y)

    return alpha, (K_active, K_inactive_fc, K_sc, eigvals if n_active > 0 else None)


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
) -> np.ndarray:
    """Predict using the dual form."""
    n_active = cache.eig.n_active
    n_edges = cache.n_edges
    active = cache.active_indices
    inactive_fc_mask = np.ones(n_edges, dtype=bool)
    inactive_fc_mask[active] = False
    inactive_fc_idx = np.where(inactive_fc_mask)[0]

    n_test = X_fc_new.shape[0]
    pred = np.zeros(n_test, dtype=np.float64)

    # FC active
    if n_active > 0:
        P_active = lambda_fc * np.diag(cache.D_active) + lambda_l * cache.active_laplacian
        P_active = 0.5 * (P_active + P_active.T)
        eigvals, eigvecs = np.linalg.eigh(P_active)
        eigvals = np.maximum(eigvals, 1e-12)

        W_train = X_fc_train[:, active] @ eigvecs
        W_test = X_fc_new[:, active] @ eigvecs
        K_test_active = (W_test / eigvals) @ W_train.T
        pred += K_test_active @ alpha

    # FC inactive (ordinary ridge, no prior weighting)
    if len(inactive_fc_idx) > 0:
        pred += (1.0 / lambda_fc) * (X_fc_new[:, inactive_fc_idx] @ X_fc_train[:, inactive_fc_idx].T) @ alpha

    # SC
    pred += (1.0 / lambda_sc) * (X_sc_new @ X_sc_train.T) @ alpha

    scale = float(max(1, n_edges * 2))
    return pred / scale


class ModalitySelectiveAnisotropicNCR:
    """MS-A-NCR: Modality-Selective Anisotropic Network-Constrained Ridge.

    FC gets anisotropic diagonal penalty + Laplacian smoothing.
    SC gets ordinary Ridge.

    Parameters
    ----------
    lambda_fc : float
        Ridge penalty for FC (modulates D).
    lambda_sc : float
        Ridge penalty for SC (isotropic).
    lambda_l : float
        Laplacian smoothing strength for FC.
    gamma : float
        Anisotropy exponent for D(q; gamma).
    cache : _MSANCRCache
        Precomputed prior-dependent quantities.
    n_rois : int
        Atlas size.
    """

    def __init__(
        self,
        lambda_fc: float = 1.0,
        lambda_sc: float = 1.0,
        lambda_l: float = 0.0,
        gamma: float = 0.0,
        cache: Optional[_MSANCRCache] = None,
        n_rois: int = 116,
    ) -> None:
        self.lambda_fc = float(lambda_fc)
        self.lambda_sc = float(lambda_sc)
        self.lambda_l = float(lambda_l)
        self.gamma = float(gamma)
        self.cache = cache
        self.n_rois = n_rois

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
        X_sc : (n, n_edges) SC upper-triangle features.
        y : (n,) target values.
        """
        X_fc = np.asarray(X_fc, dtype=np.float64)
        X_sc = np.asarray(X_sc, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)

        # Standardize on training data
        self.scaler_fc_ = StandardScaler()
        self.scaler_sc_ = StandardScaler()
        X_fc_z = self.scaler_fc_.fit_transform(X_fc)
        X_sc_z = self.scaler_sc_.fit_transform(X_sc)

        self.target_mean_ = float(y.mean())
        y_std = float(y.std())
        self.target_std_ = y_std if y_std >= 1e-8 else 1.0
        y_z = (y - self.target_mean_) / self.target_std_

        # Solve
        self.alpha_, _ = _solve_msancr_kernel(
            X_fc_z, X_sc_z, y_z, self.cache,
            self.lambda_fc, self.lambda_sc, self.lambda_l,
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
        X_sc = np.asarray(X_sc, dtype=np.float64)
        X_fc_z = self.scaler_fc_.transform(X_fc)
        X_sc_z = self.scaler_sc_.transform(X_sc)
        pred_z = _predict_msancr(
            X_fc_z, X_sc_z,
            self.X_fc_train_, self.X_sc_train_,
            self.alpha_, self.cache,
            self.lambda_fc, self.lambda_sc, self.lambda_l,
        )
        return pred_z * self.target_std_ + self.target_mean_

    def node_saliency(self) -> np.ndarray:
        """ROI-level saliency (biomarker) of the learned weights."""
        if self.alpha_ is None:
            raise RuntimeError("node_saliency() called before fit()")
        # Approximate beta from dual: beta = P^{-1} X^T alpha
        # For simplicity, use |X^T alpha| as a proxy
        n_edges = self.cache.n_edges
        beta_fc = self.X_fc_train_.T @ self.alpha_
        beta_sc = self.X_sc_train_.T @ self.alpha_
        beta = np.concatenate([beta_fc, beta_sc])
        return node_saliency_from_beta(beta, self.n_rois)


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
) -> Tuple[np.ndarray, float, float, float, float, dict]:
    """Nested hyperparameter selection for MS-A-NCR.

    Returns (test_pred, best_lambda_fc, best_lambda_sc, best_lambda_l,
             best_val_rmse, selected_info).
    """
    X_fc = np.asarray(X_fc, dtype=np.float64)
    X_sc = np.asarray(X_sc, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    n_edges = cache.n_edges
    n_active = cache.eig.n_active
    active = cache.active_indices
    inactive_fc_mask = np.ones(n_edges, dtype=bool)
    inactive_fc_mask[active] = False
    inactive_fc_idx = np.where(inactive_fc_mask)[0]

    # --- Inner training partition (selection) ---
    scaler_fc = StandardScaler()
    scaler_sc = StandardScaler()
    X_fc_train = scaler_fc.fit_transform(X_fc[train_idx])
    X_sc_train = scaler_sc.fit_transform(X_sc[train_idx])
    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_z = (y[train_idx] - y_mean) / y_std

    X_fc_val = scaler_fc.transform(X_fc[val_idx])
    X_sc_val = scaler_sc.transform(X_sc[val_idx])

    best = (float("inf"), None, None, None)

    for lambda_fc in lambda_fc_grid:
        for lambda_sc in lambda_sc_grid:
            for lambda_l in lambda_l_grid:
                try:
                    alpha, _ = _solve_msancr_kernel(
                        X_fc_train, X_sc_train, y_z, cache,
                        lambda_fc, lambda_sc, lambda_l,
                    )
                    pred_z = _predict_msancr(
                        X_fc_val, X_sc_val,
                        X_fc_train, X_sc_train,
                        alpha, cache, lambda_fc, lambda_sc, lambda_l,
                    )
                    pred = pred_z * y_std + y_mean
                    rmse = float(np.sqrt(np.mean((y[val_idx] - pred) ** 2)))
                    if rmse < best[0]:
                        best = (rmse, lambda_fc, lambda_sc, lambda_l)
                except Exception:
                    continue

    best_rmse, best_lfc, best_lsc, best_ll = best
    if best_lfc is None:
        # Fallback to isotropic ridge
        best_lfc, best_lsc, best_ll = lambda_fc_grid[0], lambda_sc_grid[0], 0.0

    # --- Refit on train + val, predict test ---
    fit_idx = np.concatenate([train_idx, val_idx])
    final_fc = StandardScaler()
    final_sc = StandardScaler()
    X_fc_fit = final_fc.fit_transform(X_fc[fit_idx])
    X_sc_fit = final_sc.fit_transform(X_sc[fit_idx])
    X_fc_test = final_fc.transform(X_fc[test_idx])
    X_sc_test = final_sc.transform(X_sc[test_idx])

    fit_mean = float(y[fit_idx].mean())
    fit_std = max(float(y[fit_idx].std()), 1e-8)
    y_fit_z = (y[fit_idx] - fit_mean) / fit_std

    alpha, _ = _solve_msancr_kernel(
        X_fc_fit, X_sc_fit, y_fit_z, cache,
        best_lfc, best_lsc, best_ll,
    )
    pred_z = _predict_msancr(
        X_fc_test, X_sc_test,
        X_fc_fit, X_sc_fit,
        alpha, cache, best_lfc, best_lsc, best_ll,
    )
    pred = pred_z * fit_std + fit_mean

    info = {
        "best_lambda_fc": best_lfc,
        "best_lambda_sc": best_lsc,
        "best_lambda_l": best_ll,
        "n_active": n_active,
    }
    return pred, best_lfc, best_lsc, best_ll, best_rmse, info
