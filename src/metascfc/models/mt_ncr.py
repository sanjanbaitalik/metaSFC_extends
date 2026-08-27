#!/usr/bin/env python3
"""Multi-Task Network-Constrained Ridge (MT-NCR) for dual-task prediction.

When lambda3 = 0 the two tasks are solved independently using the standard
NCR dual-form solver (no joint penalty).  When lambda3 > 0 the l2,1 norm
encourages shared feature selection across tasks via Iteratively Reweighted
Least Squares (IRLS): each IRLS iteration solves independent per-task
weighted ridge regressions via CG with a diagonal preconditioner.  The
diagonal preconditioner ensures fast convergence (~10 CG iterations).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg
from sklearn.preprocessing import StandardScaler

from metascfc.models.iclr_backbones.network_constrained_ridge import (
    EdgeLaplacian,
    _LaplacianEig,
    factor_laplacian_eig,
    fit_predict_network_constrained,
    node_saliency_from_beta,
)


@dataclass
class MTNCRResult:
    pred_fluid: np.ndarray
    pred_wm: np.ndarray
    beta_fluid: np.ndarray
    beta_wm: np.ndarray
    best_lambda1: float
    best_lambda2: float
    best_lambda3: float
    best_val_rmse_fluid: float
    best_val_rmse_wm: float
    jaccard: float


def _cg_solve_weighted_ridge(
    x: np.ndarray,
    y: np.ndarray,
    lambda1: float,
    lambda2: float,
    lambda3: float,
    irls_weights: np.ndarray,
    edge_laplacian: EdgeLaplacian,
    eig: _LaplacianEig,
) -> np.ndarray:
    """Solve (X^T X + P) beta = X^T y via CG with diagonal preconditioner.

    P = scale*(lambda1*I + lambda2*L + lambda3*diag(w))
    """
    n, p = x.shape
    scale = float(max(1, p))
    n_active = eig.n_active
    n_edges = edge_laplacian.n_edges

    active_fc = edge_laplacian.active_indices
    active_sc = active_fc + n_edges
    active_all = np.concatenate([active_fc, active_sc]) if n_active > 0 else np.empty(0, dtype=np.int64)

    L0 = edge_laplacian.active_laplacian

    def matvec(v):
        Xv = x @ v
        XtXv = x.T @ Xv
        Pv = scale * (lambda1 * v + lambda3 * irls_weights * v)
        if n_active > 0:
            v_active = v[active_all]
            Lv_fc = L0 @ v_active[:n_active]
            Lv_sc = L0 @ v_active[n_active:]
            Pv[active_all] += scale * lambda2 * np.concatenate([Lv_fc, Lv_sc])
        return XtXv + Pv

    A = LinearOperator((p, p), matvec=matvec, dtype=np.float64)
    Xty = x.T @ y

    diag_P = np.full(p, scale * (lambda1 + lambda3 * irls_weights))
    if n_active > 0:
        diag_P[active_fc] += scale * lambda2 * np.diag(L0)
        diag_P[active_sc] += scale * lambda2 * np.diag(L0)
    diag_P = np.maximum(diag_P, 1e-12)
    M_inv = LinearOperator((p, p), matvec=lambda v: v / diag_P, dtype=np.float64)

    beta, _ = cg(A, Xty, maxiter=200, rtol=1e-5, M=M_inv)
    return beta


def compute_biomarker_jaccard(
    beta_f: np.ndarray,
    beta_m: np.ndarray,
    n_rois: int,
    top_fraction: float = 0.10,
) -> float:
    n_edges = int(n_rois * (n_rois - 1) / 2)
    k = max(1, int(n_edges * top_fraction))
    top_f = set(np.argsort(np.abs(beta_f[:n_edges]))[-k:])
    top_m = set(np.argsort(np.abs(beta_m[:n_edges]))[-k:])
    inter = len(top_f & top_m)
    union = len(top_f | top_m)
    return float(inter / union) if union > 0 else 0.0


def fit_predict_multitask_ncr(
    x: np.ndarray,
    y_fluid: np.ndarray,
    y_wm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    edge_laplacian_fluid: EdgeLaplacian,
    edge_laplacian_wm: EdgeLaplacian,
    alpha1_grid: Sequence[float],
    alpha2_grid: Sequence[float],
    alpha3_grid: Sequence[float] = (0.0,),
    eig_fluid: Optional[_LaplacianEig] = None,
    eig_wm: Optional[_LaplacianEig] = None,
    irls_max_iter: int = 10,
    irls_tol: float = 1e-4,
    n_rois: int = 116,
) -> MTNCRResult:
    x = np.asarray(x, dtype=np.float64)
    y_fluid = np.asarray(y_fluid, dtype=np.float64).reshape(-1)
    y_wm = np.asarray(y_wm, dtype=np.float64).reshape(-1)

    if eig_fluid is None:
        eig_fluid = factor_laplacian_eig(edge_laplacian_fluid)
    if eig_wm is None:
        eig_wm = factor_laplacian_eig(edge_laplacian_wm)

    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    all_zero_l3 = all(l3 == 0.0 for l3 in alpha3_grid)
    if all_zero_l3:
        pred_f, _, _, rmse_f, beta_f = fit_predict_network_constrained(
            x, y_fluid, train_idx, val_idx, test_idx,
            edge_laplacian_fluid, alpha1_grid, alpha2_grid,
            laplacian_eig=eig_fluid)
        pred_m, _, _, rmse_m, beta_m = fit_predict_network_constrained(
            x, y_wm, train_idx, val_idx, test_idx,
            edge_laplacian_wm, alpha1_grid, alpha2_grid,
            laplacian_eig=eig_wm)
        jaccard = compute_biomarker_jaccard(beta_f, beta_m, n_rois)
        return MTNCRResult(
            pred_fluid=pred_f, pred_wm=pred_m,
            beta_fluid=beta_f, beta_wm=beta_m,
            best_lambda1=0.0, best_lambda2=0.0, best_lambda3=0.0,
            best_val_rmse_fluid=rmse_f, best_val_rmse_wm=rmse_m,
            jaccard=jaccard,
        )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx])
    x_val = scaler.transform(x[val_idx])
    y_f_mean = float(y_fluid[train_idx].mean())
    y_f_std = max(float(y_fluid[train_idx].std()), 1e-8)
    y_m_mean = float(y_wm[train_idx].mean())
    y_m_std = max(float(y_wm[train_idx].std()), 1e-8)
    yf_train = (y_fluid[train_idx] - y_f_mean) / y_f_std
    ym_train = (y_wm[train_idx] - y_m_mean) / y_m_std

    best = (float("inf"), None, None, None, None, None, None, None)

    for l1 in alpha1_grid:
        for l2 in alpha2_grid:
            for l3 in alpha3_grid:
                betas_f = np.zeros(x_train.shape[1])
                betas_m = np.zeros(x_train.shape[1])
                for _irls in range(irls_max_iter):
                    r_norms = np.sqrt(betas_f**2 + betas_m**2 + 1e-10)
                    irls_w = 1.0 / r_norms
                    new_f = _cg_solve_weighted_ridge(
                        x_train, yf_train, l1, l2, l3, irls_w,
                        edge_laplacian_fluid, eig_fluid)
                    new_m = _cg_solve_weighted_ridge(
                        x_train, ym_train, l1, l2, l3, irls_w,
                        edge_laplacian_wm, eig_wm)
                    if max(np.max(np.abs(new_f - betas_f)),
                           np.max(np.abs(new_m - betas_m))) < irls_tol:
                        betas_f, betas_m = new_f, new_m
                        break
                    betas_f, betas_m = new_f, new_m

                pred_vf = (x_val @ betas_f) * y_f_std + y_f_mean
                pred_vm = (x_val @ betas_m) * y_m_std + y_m_mean
                rmse_f = float(np.sqrt(np.mean((y_fluid[val_idx] - pred_vf)**2)))
                rmse_m = float(np.sqrt(np.mean((y_wm[val_idx] - pred_vm)**2)))
                mean_rmse = 0.5 * (rmse_f + rmse_m)
                if mean_rmse < best[0]:
                    best = (mean_rmse, l1, l2, l3,
                            betas_f.copy(), betas_m.copy(), rmse_f, rmse_m)

    best_rmse, best_l1, best_l2, best_l3, _, _, rmse_f, rmse_m = best
    if best_l1 is None:
        raise RuntimeError("No hyperparameter candidate selected")

    fit_idx = np.concatenate([train_idx, val_idx])
    final_scaler = StandardScaler()
    x_fit = final_scaler.fit_transform(x[fit_idx])
    x_test = final_scaler.transform(x[test_idx])
    yf_fit_mean = float(y_fluid[fit_idx].mean())
    yf_fit_std = max(float(y_fluid[fit_idx].std()), 1e-8)
    ym_fit_mean = float(y_wm[fit_idx].mean())
    ym_fit_std = max(float(y_wm[fit_idx].std()), 1e-8)
    yf_fit = (y_fluid[fit_idx] - yf_fit_mean) / yf_fit_std
    ym_fit = (y_wm[fit_idx] - ym_fit_mean) / ym_fit_std

    betas_f = np.zeros(x_fit.shape[1])
    betas_m = np.zeros(x_fit.shape[1])
    for _irls in range(irls_max_iter):
        r_norms = np.sqrt(betas_f**2 + betas_m**2 + 1e-10)
        irls_w = 1.0 / r_norms
        new_f = _cg_solve_weighted_ridge(
            x_fit, yf_fit, best_l1, best_l2, best_l3, irls_w,
            edge_laplacian_fluid, eig_fluid)
        new_m = _cg_solve_weighted_ridge(
            x_fit, ym_fit, best_l1, best_l2, best_l3, irls_w,
            edge_laplacian_wm, eig_wm)
        if max(np.max(np.abs(new_f - betas_f)),
               np.max(np.abs(new_m - betas_m))) < irls_tol:
            betas_f, betas_m = new_f, new_m
            break
        betas_f, betas_m = new_f, new_m

    pred_f = (x_test @ betas_f) * yf_fit_std + yf_fit_mean
    pred_m = (x_test @ betas_m) * ym_fit_std + ym_fit_mean

    beta_dev_f = betas_f * (final_scaler.scale_ / yf_fit_std)
    beta_dev_m = betas_m * (final_scaler.scale_ / ym_fit_std)

    jaccard = compute_biomarker_jaccard(beta_dev_f, beta_dev_m, n_rois)

    return MTNCRResult(
        pred_fluid=np.asarray(pred_f),
        pred_wm=np.asarray(pred_m),
        beta_fluid=beta_dev_f,
        beta_wm=beta_dev_m,
        best_lambda1=best_l1,
        best_lambda2=best_l2,
        best_lambda3=best_l3,
        best_val_rmse_fluid=rmse_f,
        best_val_rmse_wm=rmse_m,
        jaccard=jaccard,
    )
