"""Prior Predictive Enrichment Audit for ICLR 2027.

Determines whether the continuous cognitive meta-analysis prior concentrates
edges that are predictive of the corresponding cognitive target. All
diagnostics use training-only data (no outer-test labels).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr, wilcoxon


# ---------------------------------------------------------------------------
# Edge lifting: ROI prior scores -> edge prior scores
# ---------------------------------------------------------------------------

def roi_to_edge_prior(
    roi_prior: np.ndarray,
    n_rois: int,
    rule: str = "prod",
) -> np.ndarray:
    """Lift an ROI-level prior to edge-level prior scores.

    The edge ordering matches the FC/SC upper-triangle feature ordering:
    upper-triangle edges in row-major order (i<j), then repeated for the
    second modality.  Returns an array of length n_edges = n_rois*(n_rois-1)/2.

    Parameters
    ----------
    roi_prior : (n_rois,) continuous ROI prior scores.
    n_rois : atlas size.
    rule : one of 'prod', 'mean', 'max', 'bridge'.

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
    if rule == "mean":
        return 0.5 * (pi + pj)
    if rule == "max":
        return np.maximum(pi, pj)
    if rule == "bridge":
        return pi * (1.0 - pj) + pj * (1.0 - pi)
    raise ValueError(f"Unknown lifting rule: {rule!r}")


# ---------------------------------------------------------------------------
# Ridge solver (training-only, dual form for n < p)
# ---------------------------------------------------------------------------

def fit_ridge_dual(
    x: np.ndarray,
    y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Ridge regression in dual form: beta = X^T (X X^T + alpha I)^{-1} y."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    n = x.shape[0]
    K = x @ x.T
    alpha_reg = K + alpha * np.eye(n)
    dual_coef = np.linalg.solve(alpha_reg, y)
    return x.T @ dual_coef


def select_ridge_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    alpha_grid: Sequence[float],
) -> Tuple[float, np.ndarray]:
    """Select alpha by inner validation, refit on train+val."""
    best_alpha, best_rmse, best_beta = alpha_grid[0], float("inf"), None
    for a in alpha_grid:
        beta = fit_ridge_dual(x_train, y_train, a)
        pred = x_val @ beta
        rmse = float(np.sqrt(np.mean((y_val - pred) ** 2)))
        if rmse < best_rmse:
            best_alpha, best_rmse, best_beta = a, rmse, beta
    return best_alpha, best_beta


# ---------------------------------------------------------------------------
# Enrichment metrics
# ---------------------------------------------------------------------------

def marginal_predictive_enrichment(
    X: np.ndarray,
    y: np.ndarray,
    edge_prior: np.ndarray,
) -> Dict[str, float]:
    """Diagnostic A: correlation between edge prior and |corr(X_e, y)|.

    All data is from the outer-training partition only.
    """
    n, p = X.shape
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if n < 3:
        return {"pearson": 0.0, "spearman": 0.0}
    m = np.zeros(p, dtype=np.float64)
    for e in range(p):
        r = pearsonr(X[:, e], y).statistic
        m[e] = abs(r) if np.isfinite(r) else 0.0
    pr, _ = pearsonr(edge_prior, m)
    sr, _ = spearmanr(edge_prior, m)
    return {
        "pearson": float(pr) if np.isfinite(pr) else 0.0,
        "spearman": float(sr) if np.isfinite(sr) else 0.0,
        "m_e": m,
    }


def top_fraction_enrichment(
    edge_prior: np.ndarray,
    m_e: np.ndarray,
    fractions: Sequence[float],
    n_random: int = 1000,
    rng: Optional[np.random.RandomState] = None,
) -> List[Dict]:
    """Diagnostic B: mean |corr| in top prior-selected edges vs random."""
    if rng is None:
        rng = np.random.RandomState(42)
    p = len(edge_prior)
    rank = np.argsort(edge_prior)[::-1]
    results = []
    for frac in fractions:
        k = max(1, int(p * frac))
        top_idx = rank[:k]
        observed = float(m_e[top_idx].mean())
        null_means = np.empty(n_random)
        for b in range(n_random):
            rand_idx = rng.choice(p, size=k, replace=False)
            null_means[b] = m_e[rand_idx].mean()
        z = (observed - null_means.mean()) / max(null_means.std(), 1e-12)
        p_val = float(np.mean(null_means >= observed))
        results.append({
            "fraction": frac,
            "k": k,
            "observed_mean": observed,
            "null_mean": float(null_means.mean()),
            "null_std": float(null_means.std()),
            "enrichment_ratio": observed / max(float(null_means.mean()), 1e-12),
            "z_score": float(z),
            "p_value": p_val,
        })
    return results


def ridge_coefficient_enrichment(
    beta: np.ndarray,
    edge_prior: np.ndarray,
    fractions: Sequence[float],
    n_random: int = 1000,
    rng: Optional[np.random.RandomState] = None,
) -> Dict:
    """Diagnostic C: correlation between edge prior and |Ridge beta|."""
    b_e = np.abs(np.asarray(beta, dtype=np.float64))
    pr, _ = pearsonr(edge_prior, b_e)
    sr, _ = spearmanr(edge_prior, b_e)
    top = top_fraction_enrichment(edge_prior, b_e, fractions, n_random, rng)
    return {
        "pearson": float(pr) if np.isfinite(pr) else 0.0,
        "spearman": float(sr) if np.isfinite(sr) else 0.0,
        "top_fraction": top,
    }


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> Dict:
    """Two-sided Wilcoxon signed-rank test on paired observations."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    diff = diff[diff != 0]
    if len(diff) < 3:
        return {"statistic": 0.0, "p_value": 1.0}
    stat, p = wilcoxon(diff, alternative="two-sided")
    return {"statistic": float(stat), "p_value": float(p)}


def holm_correction(pvalues: Sequence[float]) -> np.ndarray:
    """Holm step-down correction for multiple comparisons."""
    p = np.asarray(list(pvalues), dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict:
    """Bootstrap 95% CI of paired difference mean."""
    rng = np.random.RandomState(seed)
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(diff)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means[i] = diff[idx].mean()
    lo = np.percentile(boot_means, (1 - ci) / 2 * 100)
    hi = np.percentile(boot_means, (1 + ci) / 2 * 100)
    return {
        "mean_diff": float(diff.mean()),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
    }


def paired_effect_size(a: np.ndarray, b: np.ndarray) -> float:
    """Paired standardized effect size (Cohen's d)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if diff.std() < 1e-12:
        return 0.0
    return float(diff.mean() / diff.std())


def roi_to_edge_prior_with_threshold(
    roi_prior: np.ndarray,
    n_rois: int,
    top_k: int,
    rule: str = "prod",
) -> Tuple[np.ndarray, int]:
    """Lift ROI prior after thresholding to top-k ROIs."""
    p = np.copy(np.asarray(roi_prior, dtype=np.float64).ravel())
    if top_k >= n_rois:
        return roi_to_edge_prior(p, n_rois, rule), n_rois
    threshold = np.sort(p)[-top_k]
    p[p < threshold] = 0.0
    n_active = int(np.sum(p > 0))
    return roi_to_edge_prior(p, n_rois, rule), n_active
