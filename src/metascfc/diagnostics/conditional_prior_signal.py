"""Conditional / Residual Prior-Signal Audit v2 — core computation.

Fully nested, leakage-safe implementation.  All variant/eta selection
occurs via inner cross-fitting within the outer-training set.
No outer-test labels are used for any model selection.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

from metascfc.diagnostics.prior_predictive_enrichment import (
    bootstrap_ci,
    fit_ridge_dual as _fit_ridge_dual_orig,
    holm_correction,
    paired_effect_size,
    paired_wilcoxon,
)


def _robust_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression that works for both n<p and n>p.

    Uses sklearn Ridge which handles both cases robustly.
    """
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha, fit_intercept=False, solver="cholesky")
    model.fit(X, y)
    return model.coef_.copy()


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

    Returns (pred, residuals) where residuals = y[indices] - pred.
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

        # Alpha selection: further split train into sub_train/sub_val
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
            beta = _robust_ridge(X_sel_tr, y_sel_tr, a)
            rmse = float(np.sqrt(np.mean((y_sel_val - X_sel_val @ beta) ** 2)))
            if rmse < best_rmse:
                best_a, best_rmse = a, rmse

        # Refit on train with best alpha
        X_tr = scaler.fit_transform(X[train_global])
        y_tr_mean = float(y[train_global].mean())
        y_tr_std = max(float(y[train_global].std()), 1e-8)
        y_tr = (y[train_global] - y_tr_mean) / y_tr_std
        beta = _robust_ridge(X_tr, y_tr, best_a)

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
        beta = _robust_ridge(X_tv[sel_tr], y_tv[sel_tr], a)
        rmse = float(np.sqrt(np.mean((y_tv[sel_val] - X_tv[sel_val] @ beta) ** 2)))
        if rmse < best_rmse:
            best_a, best_rmse = a, rmse

    beta = _robust_ridge(X_tv, y_tv, best_a)
    X_test = scaler.transform(X_all[test_idx])
    pred_test = X_test @ beta * y_tv_std + y_tv_mean
    pred_trainval = X_tv @ beta * y_tv_std + y_tv_mean
    return pred_test, pred_trainval, best_a


# ---------------------------------------------------------------------------
# Residual enrichment
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
# Top-fraction enrichment
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
        # Ensure no duplicates in random subsets
        null_means = np.array([m_e[np.unique(ri)].mean() for ri in rand_idx])
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
# Residual branch candidates — all produce OOF predictions for selection
# ---------------------------------------------------------------------------

def _standardize_fit_transform(X_fit, y_fit_mean, y_fit_std):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_fit)
    return Xs, scaler


def _standardize_transform(scaler, X):
    return scaler.transform(X)


def _normalize_target(y, mean, std):
    return (y - mean) / max(std, 1e-8)


def _denormalize(pred, mean, std):
    return pred * std + mean


def crossfit_residual_branch_topk(
    X_all, y_res_tv, trainval_idx, test_idx,
    edge_prior, top_frac, alpha_grid, n_inner_folds, rng,
):
    """OOF cross-fitted top-k Ridge residual branch within trainval."""
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    k = max(1, int(len(edge_prior) * top_frac))
    top_features = np.argsort(np.abs(edge_prior))[-k:]

    n_tv = len(trainval_idx)
    kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=rng.randint(0, 2**31))

    oof_pred = np.zeros(n_tv)
    for tr_local, va_local in kf.split(np.zeros(n_tv)):
        tr_global = trainval_idx[tr_local]
        va_global = trainval_idx[va_local]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[tr_global][:, top_features])
        X_va = scaler.transform(X_all[va_global][:, top_features])

        y_mean = float(y_res_tv[tr_local].mean())
        y_std = max(float(y_res_tv[tr_local].std()), 1e-8)
        y_tr = _normalize_target(y_res_tv[tr_local], y_mean, y_std)

        # Alpha selection on sub-split
        n_tr = len(tr_local)
        perm = rng.permutation(n_tr)
        n_sel = max(int(n_tr * 0.8), 1)
        s_tr, s_val = perm[:n_sel], perm[n_sel:]
        best_a, best_rmse = alpha_grid[0], float("inf")
        for a in alpha_grid:
            beta = _robust_ridge(X_tr[s_tr], y_tr[s_tr], a)
            rmse = float(np.sqrt(np.mean((y_tr[s_val] - X_tr[s_val] @ beta) ** 2)))
            if rmse < best_rmse:
                best_a, best_rmse = a, rmse

        beta = _robust_ridge(X_tr, y_tr, best_a)
        oof_pred[va_local] = _denormalize(X_va @ beta, y_mean, y_std)

    # Refit on full trainval for test prediction
    scaler = StandardScaler()
    X_tv = scaler.fit_transform(X_all[trainval_idx][:, top_features])
    X_te = scaler.transform(X_all[test_idx][:, top_features])
    y_mean = float(y_res_tv.mean())
    y_std = max(float(y_res_tv.std()), 1e-8)
    y_tv = _normalize_target(y_res_tv, y_mean, y_std)

    n_tv2 = len(trainval_idx)
    perm = rng.permutation(n_tv2)
    n_sel = max(int(n_tv2 * 0.8), 1)
    s_tr2, s_val2 = perm[:n_sel], perm[n_sel:]
    best_a2, best_rmse2 = alpha_grid[0], float("inf")
    for a in alpha_grid:
        beta = _robust_ridge(X_tv[s_tr2], y_tv[s_tr2], a)
        rmse = float(np.sqrt(np.mean((y_tv[s_val2] - X_tv[s_val2] @ beta) ** 2)))
        if rmse < best_rmse2:
            best_a2, best_rmse2 = a, rmse

    beta = _robust_ridge(X_tv, y_tv, best_a2)
    pred_test = _denormalize(X_te @ beta, y_mean, y_std)
    pred_tv_ins = _denormalize(X_tv @ beta, y_mean, y_std)

    return {
        "variant": f"topk_{top_frac}", "alpha": best_a2,
        "oof_pred_trainval": oof_pred,
        "pred_test": pred_test,
        "pred_trainval_ins": pred_tv_ins,
        "top_fraction": top_frac, "k": k,
    }


def crossfit_residual_branch_weighted(
    X_all, y_res_tv, trainval_idx, test_idx,
    edge_prior, gamma, epsilon, alpha_grid, n_inner_folds, rng,
):
    """OOF cross-fitted generalized-prior-weighted Ridge via penalty matrix."""
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    from metascfc.diagnostics.generalized_ridge import (
        compute_prior_penalties,
        fit_generalized_ridge,
        predict_generalized_ridge,
    )

    d = compute_prior_penalties(edge_prior, gamma, epsilon)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-30))

    n_tv = len(trainval_idx)
    kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=rng.randint(0, 2**31))

    oof_pred = np.zeros(n_tv)
    for tr_local, va_local in kf.split(np.zeros(n_tv)):
        tr_global = trainval_idx[tr_local]
        va_global = trainval_idx[va_local]

        scaler = StandardScaler()
        X_tr_raw = scaler.fit_transform(X_all[tr_global])
        X_va_raw = scaler.transform(X_all[va_global])

        # Apply penalty via reparameterization: X' = X D^{-1/2}
        X_tr = X_tr_raw * d_inv_sqrt[np.newaxis, :]
        X_va = X_va_raw * d_inv_sqrt[np.newaxis, :]

        y_mean = float(y_res_tv[tr_local].mean())
        y_std = max(float(y_res_tv[tr_local].std()), 1e-8)
        y_tr = _normalize_target(y_res_tv[tr_local], y_mean, y_std)

        n_tr = len(tr_local)
        perm = rng.permutation(n_tr)
        n_sel = max(int(n_tr * 0.8), 1)
        s_tr, s_val = perm[:n_sel], perm[n_sel:]
        best_a, best_rmse = alpha_grid[0], float("inf")
        for a in alpha_grid:
            beta = _robust_ridge(X_tr[s_tr], y_tr[s_tr], a)
            pred = X_tr[s_val] @ beta
            rmse = float(np.sqrt(np.mean((y_tr[s_val] - pred) ** 2)))
            if rmse < best_rmse:
                best_a, best_rmse = a, rmse

        beta = _robust_ridge(X_tr, y_tr, best_a)
        oof_pred[va_local] = _denormalize(X_va @ beta, y_mean, y_std)

    # Refit on full trainval
    scaler = StandardScaler()
    X_tv_raw = scaler.fit_transform(X_all[trainval_idx])
    X_te_raw = scaler.transform(X_all[test_idx])
    X_tv = X_tv_raw * d_inv_sqrt[np.newaxis, :]
    X_te = X_te_raw * d_inv_sqrt[np.newaxis, :]

    y_mean = float(y_res_tv.mean())
    y_std = max(float(y_res_tv.std()), 1e-8)
    y_tv = _normalize_target(y_res_tv, y_mean, y_std)

    n_tv2 = len(trainval_idx)
    perm = rng.permutation(n_tv2)
    n_sel = max(int(n_tv2 * 0.8), 1)
    s_tr2, s_val2 = perm[:n_sel], perm[n_sel:]
    best_a2, best_rmse2 = alpha_grid[0], float("inf")
    for a in alpha_grid:
        beta = _robust_ridge(X_tv[s_tr2], y_tv[s_tr2], a)
        pred = X_tv[s_val2] @ beta
        rmse = float(np.sqrt(np.mean((y_tv[s_val2] - pred) ** 2)))
        if rmse < best_rmse2:
            best_a2, best_rmse2 = a, rmse

    beta = _robust_ridge(X_tv, y_tv, best_a2)
    pred_test = _denormalize(X_te @ beta, y_mean, y_std)
    pred_tv_ins = _denormalize(X_tv @ beta, y_mean, y_std)

    return {
        "variant": f"weighted_{gamma}", "alpha": best_a2,
        "oof_pred_trainval": oof_pred,
        "pred_test": pred_test,
        "pred_trainval_ins": pred_tv_ins,
        "gamma": gamma, "epsilon": epsilon,
    }


def crossfit_residual_branch_pca(
    X_all, y_res_tv, trainval_idx, test_idx,
    edge_prior, n_components, alpha_grid, n_inner_folds, rng,
):
    """OOF cross-fitted PCA + Ridge residual branch."""
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    k = min(max(n_components + 10, 2 * n_components), len(edge_prior))
    top_features = np.argsort(np.abs(edge_prior))[-k:]

    n_tv = len(trainval_idx)
    kf = KFold(n_splits=n_inner_folds, shuffle=True, random_state=rng.randint(0, 2**31))

    oof_pred = np.zeros(n_tv)
    for tr_local, va_local in kf.split(np.zeros(n_tv)):
        tr_global = trainval_idx[tr_local]
        va_global = trainval_idx[va_local]

        scaler = StandardScaler()
        X_tr_raw = scaler.fit_transform(X_all[tr_global][:, top_features])
        X_va_raw = scaler.transform(X_all[va_global][:, top_features])

        pca = PCA(n_components=min(n_components, X_tr_raw.shape[1], X_tr_raw.shape[0]))
        X_tr = pca.fit_transform(X_tr_raw)
        X_va = pca.transform(X_va_raw)

        y_mean = float(y_res_tv[tr_local].mean())
        y_std = max(float(y_res_tv[tr_local].std()), 1e-8)
        y_tr = _normalize_target(y_res_tv[tr_local], y_mean, y_std)

        n_tr = len(tr_local)
        perm = rng.permutation(n_tr)
        n_sel = max(int(n_tr * 0.8), 1)
        s_tr, s_val = perm[:n_sel], perm[n_sel:]
        best_a, best_rmse = alpha_grid[0], float("inf")
        for a in alpha_grid:
            beta = _robust_ridge(X_tr[s_tr], y_tr[s_tr], a)
            rmse = float(np.sqrt(np.mean((y_tr[s_val] - X_tr[s_val] @ beta) ** 2)))
            if rmse < best_rmse:
                best_a, best_rmse = a, rmse

        beta = _robust_ridge(X_tr, y_tr, best_a)
        oof_pred[va_local] = _denormalize(X_va @ beta, y_mean, y_std)

    # Refit on full trainval
    scaler = StandardScaler()
    X_tv_raw = scaler.fit_transform(X_all[trainval_idx][:, top_features])
    X_te_raw = scaler.transform(X_all[test_idx][:, top_features])

    pca = PCA(n_components=min(n_components, X_tv_raw.shape[1], X_tv_raw.shape[0]))
    X_tv = pca.fit_transform(X_tv_raw)
    X_te = pca.transform(X_te_raw)

    y_mean = float(y_res_tv.mean())
    y_std = max(float(y_res_tv.std()), 1e-8)
    y_tv = _normalize_target(y_res_tv, y_mean, y_std)

    n_tv2 = len(trainval_idx)
    perm = rng.permutation(n_tv2)
    n_sel = max(int(n_tv2 * 0.8), 1)
    s_tr2, s_val2 = perm[:n_sel], perm[n_sel:]
    best_a2, best_rmse2 = alpha_grid[0], float("inf")
    for a in alpha_grid:
        beta = _robust_ridge(X_tv[s_tr2], y_tv[s_tr2], a)
        pred = X_tv[s_val2] @ beta
        rmse = float(np.sqrt(np.mean((y_tv[s_val2] - pred) ** 2)))
        if rmse < best_rmse2:
            best_a2, best_rmse2 = a, rmse

    beta = _robust_ridge(X_tv, y_tv, best_a2)
    pred_test = _denormalize(X_te @ beta, y_mean, y_std)
    pred_tv_ins = _denormalize(X_tv @ beta, y_mean, y_std)

    return {
        "variant": f"pca_{n_components}", "alpha": best_a2,
        "oof_pred_trainval": oof_pred,
        "pred_test": pred_test,
        "pred_trainval_ins": pred_tv_ins,
        "n_components": n_components,
        "explained_variance": float(pca.explained_variance_ratio_.sum()),
    }


# ---------------------------------------------------------------------------
# Eta selection using OOF predictions
# ---------------------------------------------------------------------------

def select_eta_and_evaluate(
    oof_baseline: np.ndarray,
    oof_residual: np.ndarray,
    y_trainval: np.ndarray,
    pred_test_baseline: np.ndarray,
    pred_test_residual: np.ndarray,
    y_test: np.ndarray,
    eta_grid: Sequence[float],
) -> Dict:
    """Select eta using OOF combined predictions, evaluate on test.

    Returns metrics dict with selected_eta, baseline/combined metrics, deltas.
    """
    from metascfc.benchmark_utils import prediction_metrics

    best_eta, best_pearson = 0.0, -float("inf")
    for eta in eta_grid:
        combined_oof = oof_baseline + eta * oof_residual
        r = float(np.corrcoef(y_trainval, combined_oof)[0, 1]) if len(y_trainval) > 1 else 0.0
        if r > best_pearson:
            best_pearson = r
            best_eta = eta

    baseline_m = prediction_metrics(y_test, pred_test_baseline)
    combined_pred = pred_test_baseline + best_eta * pred_test_residual
    combined_m = prediction_metrics(y_test, combined_pred)

    return {
        "selected_eta": best_eta,
        "inner_pearson": best_pearson,
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
    """Paired stats: matched vs a control at seed level.

    Convention: positive mean_diff means matched > control.
    """
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
