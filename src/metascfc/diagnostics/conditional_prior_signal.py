"""Conditional / Residual Prior-Signal Audit — core computation.

Determines whether the matched prior contains predictive information
that is *not already captured* by the no-prior FC+SC Ridge baseline.
All diagnostics use training-only data (no outer-test labels for selection).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from metascfc.diagnostics.prior_predictive_enrichment import (
    bootstrap_ci,
    fit_ridge_dual,
    holm_correction,
    paired_effect_size,
    paired_wilcoxon,
    roi_to_edge_prior,
)


# ---------------------------------------------------------------------------
# Cross-fitted Ridge residuals
# ---------------------------------------------------------------------------

def crossfit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    n_folds: int = 5,
    alpha_grid: Sequence[float] = (0.01, 1.0, 100.0, 10000.0, 1000000.0),
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Cross-fitted Ridge predictions for subjects at *indices*.

    For each inner fold the held-out fold is never used for alpha
    selection or fitting.

    Returns
    -------
    pred : (len(indices),) out-of-fold predictions
    residuals : y[indices] - pred
    """
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    if rng is None:
        rng = np.random.RandomState(42)

    indices = np.asarray(indices, dtype=int)
    n = len(indices)
    pred = np.zeros(n)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(0, 2**31))

    for train_local, val_local in kf.split(np.zeros(n)):
        train_global = indices[train_local]
        val_global = indices[val_local]

        n_tr = len(train_global)
        perm = rng.permutation(n_tr)
        n_sel = max(int(n_tr * 0.8), 1)
        sel_tr = train_global[perm[:n_sel]]
        sel_val = train_global[perm[n_sel:]]
        if len(sel_val) < 3:
            sel_val = val_global

        scaler = StandardScaler()
        X_sel_tr = scaler.fit_transform(X[sel_tr])
        X_sel_val = scaler.transform(X[sel_val])
        y_mean = float(y[sel_tr].mean())
        y_std = max(float(y[sel_tr].std()), 1e-8)
        y_sel_tr = (y[sel_tr] - y_mean) / y_std
        y_sel_val = (y[sel_val] - y_mean) / y_std

        best_a, best_rmse = alpha_grid[0], float("inf")
        for a in alpha_grid:
            beta = fit_ridge_dual(X_sel_tr, y_sel_tr, a)
            rmse = float(np.sqrt(np.mean((y_sel_val - X_sel_val @ beta) ** 2)))
            if rmse < best_rmse:
                best_a, best_rmse = a, rmse

        X_tr = scaler.fit_transform(X[train_global])
        y_tr_mean = float(y[train_global].mean())
        y_tr_std = max(float(y[train_global].std()), 1e-8)
        y_tr = (y[train_global] - y_tr_mean) / y_tr_std
        beta = fit_ridge_dual(X_tr, y_tr, best_a)

        X_val = scaler.transform(X[val_global])
        pred[val_local] = X_val @ beta * y_tr_std + y_tr_mean

    return pred, y[indices] - pred


# ---------------------------------------------------------------------------
# Baseline Ridge on test
# ---------------------------------------------------------------------------

def fit_ridge_baseline(
    X_all: np.ndarray,
    y_full: np.ndarray,
    trainval_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha_grid: Sequence[float],
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Fit Ridge on trainval, predict on test.

    Returns (pred_test, pred_trainval_insample, best_alpha).
    """
    from sklearn.preprocessing import StandardScaler

    if rng is None:
        rng = np.random.RandomState(42)

    scaler = StandardScaler()
    X_tv = scaler.fit_transform(X_all[trainval_idx])
    y_tv_mean = float(y_full[trainval_idx].mean())
    y_tv_std = max(float(y_full[trainval_idx].std()), 1e-8)
    y_tv = (y_full[trainval_idx] - y_tv_mean) / y_tv_std

    n_tv = len(trainval_idx)
    perm = rng.permutation(n_tv)
    n_sel = max(int(n_tv * 0.8), 1)
    sel_tr, sel_val = perm[:n_sel], perm[n_sel:]

    best_a, best_rmse = alpha_grid[0], float("inf")
    for a in alpha_grid:
        beta = fit_ridge_dual(X_tv[sel_tr], y_tv[sel_tr], a)
        rmse = float(np.sqrt(np.mean((y_tv[sel_val] - X_tv[sel_val] @ beta) ** 2)))
        if rmse < best_rmse:
            best_a, best_rmse = a, rmse

    beta = fit_ridge_dual(X_tv, y_tv, best_a)
    X_test = scaler.transform(X_all[test_idx])
    pred_test = X_test @ beta * y_tv_std + y_tv_mean
    pred_trainval = X_tv @ beta * y_tv_std + y_tv_mean
    return pred_test, pred_trainval, best_a


# ---------------------------------------------------------------------------
# Diagnostic A — residual marginal enrichment
# ---------------------------------------------------------------------------

def residual_enrichment(
    X: np.ndarray,
    residuals: np.ndarray,
    edge_prior: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Pearson/Spearman between edge prior and |corr(X_e, residuals)|."""
    n, p = X.shape
    if n < 3:
        return 0.0, 0.0, np.zeros(p)

    r = residuals - residuals.mean()
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-12:
        return 0.0, 0.0, np.zeros(p)

    x_stds = X.std(axis=0, ddof=1)
    valid = x_stds > 1e-12
    m_e = np.zeros(p)
    if valid.any():
        Xv = X[:, valid]
        corrs = (Xv.T @ r) / (n - 1) / (x_stds[valid] * (r_norm / np.sqrt(n - 1) + 1e-30))
        corrs = np.clip(corrs, -1.0, 1.0)
        m_e[valid] = np.abs(corrs)

    pr, _ = pearsonr(edge_prior, m_e)
    sr, _ = spearmanr(edge_prior, m_e)
    return (
        float(pr) if np.isfinite(pr) else 0.0,
        float(sr) if np.isfinite(sr) else 0.0,
        m_e,
    )


# ---------------------------------------------------------------------------
# Diagnostic C — top-prior residual enrichment
# ---------------------------------------------------------------------------

def fast_top_fraction(
    edge_prior: np.ndarray,
    m_e: np.ndarray,
    fractions: Sequence[float],
    n_random: int = 1000,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    """Mean |corr| in top-prior edges vs random subsets."""
    if rng is None:
        rng = np.random.RandomState(42)

    p = len(edge_prior)
    rank = np.argsort(edge_prior)[::-1]
    results = []
    for frac in fractions:
        k = max(1, int(p * frac))
        top_idx = rank[:k]
        observed = float(m_e[top_idx].mean())
        rand_idx = rng.randint(0, p, size=(n_random, k))
        null_means = np.array([m_e[ri].mean() for ri in rand_idx])
        z = (observed - null_means.mean()) / max(null_means.std(), 1e-12)
        p_val = float(np.mean(null_means >= observed))
        results.append({
            "fraction": frac, "k": k,
            "observed_mean": observed,
            "null_mean": float(null_means.mean()),
            "null_std": float(null_means.std()),
            "enrichment_ratio": observed / max(float(null_means.mean()), 1e-12),
            "z_score": float(z), "p_value": p_val,
        })
    return results


# ---------------------------------------------------------------------------
# Diagnostic D — prior-only residual prediction variants
# ---------------------------------------------------------------------------

def _select_alpha_and_predict(
    X_tv: np.ndarray,
    y_norm: np.ndarray,
    X_test: np.ndarray,
    alpha_grid: Sequence[float],
    rng: np.random.RandomState,
) -> Tuple[np.ndarray, np.ndarray, float]:
    n_tv = X_tv.shape[0]
    perm = rng.permutation(n_tv)
    n_sel = max(int(n_tv * 0.8), 1)
    sel_tr, sel_val = perm[:n_sel], perm[n_sel:]

    best_a, best_rmse = alpha_grid[0], float("inf")
    for a in alpha_grid:
        beta = fit_ridge_dual(X_tv[sel_tr], y_norm[sel_tr], a)
        rmse = float(np.sqrt(np.mean((y_norm[sel_val] - X_tv[sel_val] @ beta) ** 2)))
        if rmse < best_rmse:
            best_a, best_rmse = a, rmse

    beta = fit_ridge_dual(X_tv, y_norm, best_a)
    return X_tv @ beta, X_test @ beta, best_a


def fit_prior_topk_ridge(
    X_all: np.ndarray,
    y_res_tv: np.ndarray,
    trainval_idx: np.ndarray,
    test_idx: np.ndarray,
    edge_prior: np.ndarray,
    top_frac: float,
    alpha_grid: Sequence[float],
    scaler_tv=None,
    rng: Optional[np.random.RandomState] = None,
) -> Dict:
    """D1: top-k Ridge on prior-selected features."""
    from sklearn.preprocessing import StandardScaler

    if rng is None:
        rng = np.random.RandomState(42)

    k = max(1, int(len(edge_prior) * top_frac))
    top_features = np.argsort(np.abs(edge_prior))[-k:]

    if scaler_tv is None:
        scaler_tv = StandardScaler()
        X_tv = scaler_tv.fit_transform(X_all[trainval_idx][:, top_features])
    else:
        X_tv = scaler_tv.transform(X_all[trainval_idx][:, top_features])
    X_test = scaler_tv.transform(X_all[test_idx][:, top_features])

    y_mean = float(y_res_tv.mean())
    y_std = max(float(y_res_tv.std()), 1e-8)
    y_norm = (y_res_tv - y_mean) / y_std

    pv_train, pv_test, best_a = _select_alpha_and_predict(X_tv, y_norm, X_test, alpha_grid, rng)

    return {
        "variant": "topk", "top_fraction": top_frac, "k": k,
        "best_alpha": best_a,
        "pred_test": pv_test * y_std + y_mean,
        "pred_trainval": pv_train * y_std + y_mean,
    }


def fit_prior_weighted_ridge(
    X_all: np.ndarray,
    y_res_tv: np.ndarray,
    trainval_idx: np.ndarray,
    test_idx: np.ndarray,
    edge_prior: np.ndarray,
    gamma: float,
    epsilon: float,
    alpha_grid: Sequence[float],
    rng: Optional[np.random.RandomState] = None,
) -> Dict:
    """D2: prior-weighted Ridge."""
    from sklearn.preprocessing import StandardScaler

    if rng is None:
        rng = np.random.RandomState(42)

    weights = (epsilon + np.abs(edge_prior)) ** gamma
    X_w = X_all * weights[np.newaxis, :]

    scaler = StandardScaler()
    X_tv = scaler.fit_transform(X_w[trainval_idx])
    X_test = scaler.transform(X_w[test_idx])

    y_mean = float(y_res_tv.mean())
    y_std = max(float(y_res_tv.std()), 1e-8)
    y_norm = (y_res_tv - y_mean) / y_std

    pv_train, pv_test, best_a = _select_alpha_and_predict(X_tv, y_norm, X_test, alpha_grid, rng)

    return {
        "variant": "weighted", "gamma": gamma, "epsilon": epsilon,
        "best_alpha": best_a,
        "pred_test": pv_test * y_std + y_mean,
        "pred_trainval": pv_train * y_std + y_mean,
    }


def fit_prior_pca_ridge(
    X_all: np.ndarray,
    y_res_tv: np.ndarray,
    trainval_idx: np.ndarray,
    test_idx: np.ndarray,
    edge_prior: np.ndarray,
    n_components: int,
    alpha_grid: Sequence[float],
    rng: Optional[np.random.RandomState] = None,
) -> Dict:
    """D3: PCA on top-prior features + Ridge."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    if rng is None:
        rng = np.random.RandomState(42)

    k = min(max(n_components + 10, 2 * n_components), len(edge_prior))
    top_features = np.argsort(np.abs(edge_prior))[-k:]

    scaler = StandardScaler()
    X_tv_raw = scaler.fit_transform(X_all[trainval_idx][:, top_features])
    X_test_raw = scaler.transform(X_all[test_idx][:, top_features])

    pca = PCA(n_components=min(n_components, X_tv_raw.shape[1], X_tv_raw.shape[0]))
    X_tv = pca.fit_transform(X_tv_raw)
    X_test = pca.transform(X_test_raw)

    y_mean = float(y_res_tv.mean())
    y_std = max(float(y_res_tv.std()), 1e-8)
    y_norm = (y_res_tv - y_mean) / y_std

    pv_train, pv_test, best_a = _select_alpha_and_predict(X_tv, y_norm, X_test, alpha_grid, rng)

    return {
        "variant": "pca", "n_components": n_components,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
        "best_alpha": best_a,
        "pred_test": pv_test * y_std + y_mean,
        "pred_trainval": pv_train * y_std + y_mean,
    }


# ---------------------------------------------------------------------------
# Diagnostic E — additive prediction test
# ---------------------------------------------------------------------------

def compute_additive_metrics(
    baseline_pred_test: np.ndarray,
    prior_pred_test: np.ndarray,
    y_test: np.ndarray,
    baseline_pred_trainval: np.ndarray,
    prior_pred_trainval: np.ndarray,
    y_trainval: np.ndarray,
    eta_grid: Sequence[float],
) -> Dict:
    """Select eta on trainval, evaluate combined prediction on test."""
    from metascfc.benchmark_utils import prediction_metrics

    best_eta, best_rmse = 0.0, float("inf")
    for eta in eta_grid:
        combined_tv = baseline_pred_trainval + eta * prior_pred_trainval
        rmse = float(np.sqrt(np.mean((y_trainval - combined_tv) ** 2)))
        if rmse < best_rmse:
            best_eta, best_rmse = eta, rmse

    baseline_m = prediction_metrics(y_test, baseline_pred_test)
    combined_pred = baseline_pred_test + best_eta * prior_pred_test
    combined_m = prediction_metrics(y_test, combined_pred)

    return {
        "selected_eta": best_eta,
        "baseline_pearson": baseline_m["pearson"],
        "baseline_rmse": baseline_m["rmse"],
        "baseline_mae": baseline_m["mae"],
        "combined_pearson": combined_m["pearson"],
        "combined_rmse": combined_m["rmse"],
        "combined_mae": combined_m["mae"],
        "delta_pearson": combined_m["pearson"] - baseline_m["pearson"],
        "delta_rmse": combined_m["rmse"] - baseline_m["rmse"],
        "delta_mae": combined_m["mae"] - baseline_m["mae"],
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def seed_level_stats(
    matched_vals: np.ndarray,
    control_vals: np.ndarray,
    label: str = "matched",
) -> Dict:
    """Paired stats: matched vs a control at seed level."""
    n = min(len(matched_vals), len(control_vals))
    if n < 3:
        return {"label": label, "wilcoxon_p": 1.0, "mean_diff": 0.0,
                "ci_lo": 0.0, "ci_hi": 0.0, "effect_size": 0.0,
                "n_seeds": n, "n_positive": 0, "median_delta": 0.0}

    a, b = matched_vals[:n], control_vals[:n]
    test = paired_wilcoxon(a, b)
    ci = bootstrap_ci(a, b)
    es = paired_effect_size(a, b)
    diff = a - b
    return {
        "label": label,
        "wilcoxon_p": test["p_value"],
        "mean_diff": ci["mean_diff"],
        "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
        "effect_size": es,
        "n_seeds": n,
        "n_positive": int(np.sum(diff > 0)),
        "median_delta": float(np.median(diff)),
    }
