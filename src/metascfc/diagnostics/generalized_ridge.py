"""Generalized Ridge regression with diagonal prior penalty.

Solves:
    min_beta ||y - X beta||^2 + alpha * sum_e d_e * beta_e^2

where d_e = (epsilon + |q_e|)^{-gamma} and q_e are prior scores.

High-prior features (large |q_e|) receive lower penalty (less shrinkage).
The penalty survives feature standardization because it enters through
the penalty matrix, not through feature scaling.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np


def compute_prior_penalties(
    edge_prior: np.ndarray,
    gamma: float,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Compute diagonal penalty weights d_e = (epsilon + |q_e|)^{-gamma}.

    Parameters
    ----------
    edge_prior : (p,) prior scores (can be negative).
    gamma : exponent (> 0 gives less shrinkage to high-prior features).
    epsilon : small constant to avoid division by zero.

    Returns
    -------
    d : (p,) penalty weights.  High d_e means MORE shrinkage.
    """
    return (epsilon + np.abs(np.asarray(edge_prior, dtype=np.float64))) ** (-gamma)


def fit_generalized_ridge(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    d: np.ndarray,
) -> np.ndarray:
    """Generalized Ridge with diagonal penalty via reparameterization.

    Solves: min ||y - X beta||^2 + alpha * d_e beta_e^2

    Uses the transformation: X' = X D^{-1/2}, solve standard Ridge on X',
    recover beta = D^{-1/2} theta.

    Parameters
    ----------
    X : (n, p) feature matrix.
    y : (n,) target.
    alpha : Ridge regularization strength.
    d : (p,) diagonal penalty weights.

    Returns
    -------
    beta : (p,) coefficient vector in original feature space.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    d = np.asarray(d, dtype=np.float64).ravel()

    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-30))
    X_transformed = X * d_inv_sqrt[np.newaxis, :]

    K = X_transformed @ X_transformed.T
    n = K.shape[0]
    alpha_reg = K + alpha * np.eye(n)
    dual_coef = np.linalg.solve(alpha_reg, y)
    theta = X_transformed.T @ dual_coef

    beta = d_inv_sqrt * theta
    return beta


def predict_generalized_ridge(
    X: np.ndarray,
    beta: np.ndarray,
) -> np.ndarray:
    """Predict using generalized Ridge coefficients."""
    return np.asarray(X, dtype=np.float64) @ np.asarray(beta, dtype=np.float64)


def fit_predict_generalized_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float,
    d: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit generalized Ridge on training data, predict on both train and test.

    Returns (pred_train, pred_test).
    """
    beta = fit_generalized_ridge(X_train, y_train, alpha, d)
    pred_train = predict_generalized_ridge(X_train, beta)
    pred_test = predict_generalized_ridge(X_test, beta)
    return pred_train, pred_test


def generalized_ridge_cv(
    X: np.ndarray,
    y: np.ndarray,
    alpha_grid: Sequence[float],
    d: np.ndarray,
    n_folds: int = 5,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Select alpha by inner CV, refit on full data.

    Returns (best_alpha, pred_oof, beta_full).

    pred_oof are out-of-fold predictions for model-selection purposes.
    """
    from sklearn.model_selection import KFold

    if rng is None:
        rng = np.random.RandomState(42)

    n = X.shape[0]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(0, 2**31))

    best_alpha, best_score = alpha_grid[0], -float("inf")
    oof_preds = {}

    for a in alpha_grid:
        oof = np.zeros(n)
        for tr_local, va_local in kf.split(np.zeros(n)):
            beta = fit_generalized_ridge(X[tr_local], y[tr_local], a, d)
            oof[va_local] = predict_generalized_ridge(X[va_local], beta)
        score = float(np.corrcoef(y, oof)[0, 1])
        if score > best_score:
            best_score = score
            best_alpha = a

    # Refit on full data with best alpha
    beta_full = fit_generalized_ridge(X, y, best_alpha, d)

    # Compute full OOF predictions with best alpha
    oof_final = np.zeros(n)
    for tr_local, va_local in kf.split(np.zeros(n)):
        beta = fit_generalized_ridge(X[tr_local], y[tr_local], best_alpha, d)
        oof_final[va_local] = predict_generalized_ridge(X[va_local], beta)

    return best_alpha, oof_final, beta_full
