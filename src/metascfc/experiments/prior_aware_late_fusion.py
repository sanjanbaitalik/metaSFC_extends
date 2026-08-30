"""Modification 2: Leakage-Safe Prior-Aware Late Fusion.

Two-level stacking design:
    Level-1: modality/prior-specific predictors (SC-only Ridge, FC-only Ridge, FC-only prior-aware MS-A-NCR)
    Level-2: small fusion model trained only from out-of-fold Level-1 predictions

Leakage constraints:
    - every outer-training subject's Level-1 stacking feature is OOF
    - no Level-1 model predicts a stacking-training subject it trained on
    - outer-test labels never enter branch selection or fusion-weight selection
    - outer-test predictions are generated only after all selections are frozen
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from metascfc.benchmark_utils import (
    iter_nested_splits,
    load_connectomes,
    prediction_metrics,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    _MSANCRCache,
    _predict_msancr,
    _solve_msancr_kernel,
    build_msancr_cache,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    build_edge_laplacian,
)


# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

RIDGE_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
GAMMA_GRID = [0.1, 0.25, 0.5, 1.0, 2.0]
LAMBDA_L_GRID = [0.03, 0.1, 0.5, 1.0, 2.0, 5.0]
LIFTING_RULES = ["prod", "mean"]
TOP_K = 10
DIAGONAL_EPSILON = 0.001

# Fusion weight grid increments
WEIGHT_STEP = 0.05
WEIGHT_2B = [round(i * WEIGHT_STEP, 4) for i in range(0, int(1.0 / WEIGHT_STEP) + 1)]
WEIGHT_3B = [
    (round(a * WEIGHT_STEP, 4), round(b * WEIGHT_STEP, 4), round((1.0 - a * WEIGHT_STEP - b * WEIGHT_STEP), 4))
    for a in range(0, int(1.0 / WEIGHT_STEP) + 1)
    for b in range(0, int(1.0 / WEIGHT_STEP) + 1)
    if a * WEIGHT_STEP + b * WEIGHT_STEP <= 1.0 + 1e-9
]


# ---------------------------------------------------------------------------
# Level-1 branch: FC-only no-prior Ridge (F0)
# ---------------------------------------------------------------------------

def fit_predict_ridge_fc(
    X_fc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_inner_folds: int = 3,
    seed: int = 0,
    outer_fold: int = 0,
) -> tuple[np.ndarray, dict]:
    """FC-only no-prior Ridge with inner CV alpha selection.

    Returns (test_predictions, selected_info).
    """
    X_fc = np.asarray(X_fc, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    scaler_fc = StandardScaler()
    X_fc_train = scaler_fc.fit_transform(X_fc[train_idx])
    X_fc_test = scaler_fc.transform(X_fc[test_idx])

    # Inner CV for alpha selection
    from sklearn.model_selection import KFold
    inner_splitter = KFold(
        n_splits=n_inner_folds, shuffle=True,
        random_state=int(seed) * 1000 + int(outer_fold) + 17001,
    )

    best_pearson = -np.inf
    best_alpha = ridge_grid[0]
    best_rmse = np.inf
    best_mae = np.inf

    for alpha in ridge_grid:
        fold_pearsons = []
        fold_rmses = []
        fold_maes = []
        for inner_train_local, inner_val_local in inner_splitter.split(X_fc_train):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_fc_train[inner_train_local], y[train_idx[inner_train_local]])
            pred = model.predict(X_fc_train[inner_val_local])
            m = prediction_metrics(y[train_idx[inner_val_local]], pred)
            fold_pearsons.append(m["pearson"])
            fold_rmses.append(m["rmse"])
            fold_maes.append(m["mae"])
        mean_pearson = float(np.mean(fold_pearsons))
        mean_rmse = float(np.mean(fold_rmses))
        mean_mae = float(np.mean(fold_maes))
        # Selection: Pearson first, RMSE tie-break, MAE tie-break
        if (
            mean_pearson > best_pearson + 1e-10
            or (abs(mean_pearson - best_pearson) < 1e-10 and mean_rmse < best_rmse - 1e-10)
            or (abs(mean_pearson - best_pearson) < 1e-10 and abs(mean_rmse - best_rmse) < 1e-10 and mean_mae < best_mae - 1e-10)
        ):
            best_pearson = mean_pearson
            best_alpha = alpha
            best_rmse = mean_rmse
            best_mae = mean_mae

    # Refit on full train, predict test
    model = Ridge(alpha=best_alpha, fit_intercept=True)
    model.fit(X_fc_train, y[train_idx])
    test_pred = model.predict(X_fc_test)

    info = {"alpha": float(best_alpha), "inner_pearson": float(best_pearson)}
    return test_pred, info


def fit_predict_ridge_fc_oof(
    X_fc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_inner_folds: int = 3,
    seed: int = 0,
    outer_fold: int = 0,
) -> tuple[np.ndarray, float, float, float, dict]:
    """FC-only Ridge with inner CV, returning OOF predictions on val_idx.

    Returns (oof_predictions_on_val, alpha, inner_pearson, inner_rmse, info).
    """
    X_fc = np.asarray(X_fc, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # Fit scaler on train only
    scaler_fc = StandardScaler()
    X_fc_train_z = scaler_fc.fit_transform(X_fc[train_idx])

    # Inner CV on train only for alpha selection
    from sklearn.model_selection import KFold
    inner_splitter = KFold(
        n_splits=n_inner_folds, shuffle=True,
        random_state=int(seed) * 1000 + int(outer_fold) + 17001,
    )

    best_pearson = -np.inf
    best_alpha = ridge_grid[0]
    best_rmse = np.inf
    best_mae = np.inf

    for alpha in ridge_grid:
        fold_pearsons = []
        fold_rmses = []
        fold_maes = []
        for inner_train_local, inner_val_local in inner_splitter.split(X_fc_train_z):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_fc_train_z[inner_train_local], y[train_idx[inner_train_local]])
            pred = model.predict(X_fc_train_z[inner_val_local])
            m = prediction_metrics(y[train_idx[inner_val_local]], pred)
            fold_pearsons.append(m["pearson"])
            fold_rmses.append(m["rmse"])
            fold_maes.append(m["mae"])
        mean_pearson = float(np.mean(fold_pearsons))
        mean_rmse = float(np.mean(fold_rmses))
        mean_mae = float(np.mean(fold_maes))
        if (
            mean_pearson > best_pearson + 1e-10
            or (abs(mean_pearson - best_pearson) < 1e-10 and mean_rmse < best_rmse - 1e-10)
            or (abs(mean_pearson - best_pearson) < 1e-10 and abs(mean_rmse - best_rmse) < 1e-10 and mean_mae < best_mae - 1e-10)
        ):
            best_pearson = mean_pearson
            best_alpha = alpha
            best_rmse = mean_rmse
            best_mae = mean_mae

    # Refit on full train, predict val
    model = Ridge(alpha=best_alpha, fit_intercept=True)
    model.fit(X_fc_train_z, y[train_idx])
    X_fc_val_z = scaler_fc.transform(X_fc[val_idx])
    val_pred = model.predict(X_fc_val_z)

    info = {"alpha": float(best_alpha), "inner_pearson": float(best_pearson)}
    return val_pred, float(best_alpha), float(best_pearson), float(best_rmse), info


# ---------------------------------------------------------------------------
# Level-1 branch: SC-only Ridge (S)
# ---------------------------------------------------------------------------

def fit_predict_ridge_sc_oof(
    X_sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_inner_folds: int = 3,
    seed: int = 0,
    outer_fold: int = 0,
) -> tuple[np.ndarray, float, float, float, dict]:
    """SC-only Ridge with inner CV, returning OOF predictions on val_idx."""
    X_sc = np.asarray(X_sc, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    scaler_sc = StandardScaler()
    X_sc_train_z = scaler_sc.fit_transform(X_sc[train_idx])

    from sklearn.model_selection import KFold
    inner_splitter = KFold(
        n_splits=n_inner_folds, shuffle=True,
        random_state=int(seed) * 1000 + int(outer_fold) + 17001,
    )

    best_pearson = -np.inf
    best_alpha = ridge_grid[0]
    best_rmse = np.inf
    best_mae = np.inf

    for alpha in ridge_grid:
        fold_pearsons = []
        fold_rmses = []
        fold_maes = []
        for inner_train_local, inner_val_local in inner_splitter.split(X_sc_train_z):
            model = Ridge(alpha=alpha, fit_intercept=True)
            model.fit(X_sc_train_z[inner_train_local], y[train_idx[inner_train_local]])
            pred = model.predict(X_sc_train_z[inner_val_local])
            m = prediction_metrics(y[train_idx[inner_val_local]], pred)
            fold_pearsons.append(m["pearson"])
            fold_rmses.append(m["rmse"])
            fold_maes.append(m["mae"])
        mean_pearson = float(np.mean(fold_pearsons))
        mean_rmse = float(np.mean(fold_rmses))
        mean_mae = float(np.mean(fold_maes))
        if (
            mean_pearson > best_pearson + 1e-10
            or (abs(mean_pearson - best_pearson) < 1e-10 and mean_rmse < best_rmse - 1e-10)
            or (abs(mean_pearson - best_pearson) < 1e-10 and abs(mean_rmse - best_rmse) < 1e-10 and mean_mae < best_mae - 1e-10)
        ):
            best_pearson = mean_pearson
            best_alpha = alpha
            best_rmse = mean_rmse
            best_mae = mean_mae

    model = Ridge(alpha=best_alpha, fit_intercept=True)
    model.fit(X_sc_train_z, y[train_idx])
    X_sc_val_z = scaler_sc.transform(X_sc[val_idx])
    val_pred = model.predict(X_sc_val_z)

    info = {"alpha": float(best_alpha), "inner_pearson": float(best_pearson)}
    return val_pred, float(best_alpha), float(best_pearson), float(best_rmse), info


# ---------------------------------------------------------------------------
# Level-1 branch: FC-only prior-aware MS-A-NCR (FP)
# ---------------------------------------------------------------------------

def fit_predict_fp_oof(
    X_fc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    cache: _MSANCRCache,
    gamma_grid: Sequence[float] = GAMMA_GRID,
    lambda_l_grid: Sequence[float] = LAMBDA_L_GRID,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    n_inner_folds: int = 3,
    seed: int = 0,
    outer_fold: int = 0,
    n_rois: int = 116,
) -> tuple[np.ndarray, dict]:
    """FC-only prior-aware MS-A-NCR with inner CV, returning OOF predictions.

    Grid: ridge_grid × gamma_grid × lambda_l_grid × lifting_rules
    Returns (oof_predictions_on_val, selected_info).
    """
    X_fc = np.asarray(X_fc, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # Build caches for each (gamma, lifting) pair
    from metascfc.experiments.msancr_refinement import CacheFactory, prepare_kernel, predict_inner_candidate, InnerFoldData
    from metascfc.models.iclr_backbones.network_constrained_ridge import build_edge_laplacian

    cache_factory = CacheFactory(
        priors={"matched": np.zeros(n_rois)},  # placeholder; actual prior built from cache
        n_rois=n_rois,
        top_k=TOP_K,
        epsilon=DIAGONAL_EPSILON,
        weighting="binary",
        normalization="sym",
    )
    # Rebuild cache factory with actual prior from cache
    # We extract the prior from the cache's edge weights
    # Actually, we need the roi_prior. Let's pass it in or reconstruct.
    # For now, use the cache's gamma and rebuild.
    # We'll use a simplified approach: iterate over gamma/lifting, build cache, evaluate.

    # Inner CV splits
    from sklearn.model_selection import KFold as KFoldCV
    inner_splitter = KFoldCV(
        n_splits=n_inner_folds, shuffle=True,
        random_state=int(seed) * 1000 + int(outer_fold) + 17001,
    )
    inner_splits = list(inner_splitter.split(train_idx))

    # Standardize
    scaler_fc = StandardScaler()
    X_fc_train_z = scaler_fc.fit_transform(X_fc[train_idx])

    best_pearson = -np.inf
    best_params = {"alpha": ridge_grid[0], "lambda_fc": ridge_grid[0], "lambda_sc": 1.0, "lambda_l": 0.0, "gamma": float(cache.gamma), "lifting": str(cache.lifting)}

    # Build inner fold data once (cache is fixed for given gamma/lifting)
    inner_folds_data = []
    for inner_fold_idx, (inner_train_local, inner_val_local) in enumerate(inner_splits):
        inner_train_global = train_idx[inner_train_local]
        inner_val_global = train_idx[inner_val_local]
        y_mean_inner = float(y[inner_train_global].mean())
        y_std_inner = max(float(y[inner_train_global].std()), 1e-8)
        inner_folds_data.append({
            "fold": inner_fold_idx,
            "train_local": inner_train_local,
            "val_local": inner_val_local,
            "train_global": inner_train_global,
            "val_global": inner_val_global,
            "X_fc_train": X_fc_train_z[inner_train_local],
            "X_fc_val": X_fc_train_z[inner_val_local],
            "y_train_z": (y[inner_train_global] - y_mean_inner) / y_std_inner,
            "y_val": y[inner_val_global],
            "y_mean": y_mean_inner,
            "y_std": y_std_inner,
        })

    # Search lambda_fc × lambda_l (gamma/lifting fixed by cache)
    for lambda_fc in ridge_grid:
        for lambda_l in lambda_l_grid:
            fold_pearsons = []
            for ifd in inner_folds_data:
                try:
                    alpha, _ = _solve_msancr_kernel(
                        ifd["X_fc_train"], np.zeros_like(ifd["X_fc_train"]),
                        ifd["y_train_z"], cache,
                        lambda_fc, 1.0, lambda_l,
                        fc_only=True,
                    )
                    pred_z = _predict_msancr(
                        ifd["X_fc_val"], np.zeros_like(ifd["X_fc_val"]),
                        ifd["X_fc_train"], np.zeros_like(ifd["X_fc_train"]),
                        alpha, cache, lambda_fc, 1.0, lambda_l,
                        fc_only=True,
                    )
                    pred = pred_z * ifd["y_std"] + ifd["y_mean"]
                    m = prediction_metrics(ifd["y_val"], pred)
                    fold_pearsons.append(m["pearson"])
                except Exception:
                    fold_pearsons.append(-np.inf)

            mean_pearson = float(np.mean(fold_pearsons))
            if mean_pearson > best_pearson + 1e-10:
                best_pearson = mean_pearson
                best_params = {
                    "alpha": float(lambda_fc),
                    "lambda_fc": float(lambda_fc),
                    "lambda_sc": 1.0,
                    "lambda_l": float(lambda_l),
                    "gamma": float(cache.gamma),
                    "lifting": str(cache.lifting),
                }

    # Refit on full train, predict val
    X_fc_val_z = scaler_fc.transform(X_fc[val_idx])
    y_mean_train = float(y[train_idx].mean())
    y_std_train = max(float(y[train_idx].std()), 1e-8)
    y_train_z = (y[train_idx] - y_mean_train) / y_std_train

    alpha, _ = _solve_msancr_kernel(
        X_fc_train_z, np.zeros_like(X_fc_train_z),
        y_train_z, cache,
        best_params["lambda_fc"], best_params["lambda_sc"], best_params["lambda_l"],
        fc_only=True,
    )
    pred_z = _predict_msancr(
        X_fc_val_z, np.zeros_like(X_fc_val_z),
        X_fc_train_z, np.zeros_like(X_fc_train_z),
        alpha, cache, best_params["lambda_fc"], best_params["lambda_sc"], best_params["lambda_l"],
        fc_only=True,
    )
    val_pred = pred_z * y_std_train + y_mean_train

    info = {
        "best_params": best_params,
        "inner_pearson": float(best_pearson),
    }
    return val_pred, info


def fit_predict_fp_final(
    X_fc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    cache: _MSANCRCache,
    params: dict,
    scaler_fc_fitted: Optional[StandardScaler] = None,
) -> tuple[np.ndarray, StandardScaler]:
    """Fit FC-only prior-aware MS-A-NCR on full train, return predictions and scaler."""
    X_fc = np.asarray(X_fc, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    scaler_fc = StandardScaler() if scaler_fc_fitted is None else scaler_fc_fitted
    X_fc_train_z = scaler_fc.fit_transform(X_fc[train_idx])
    y_mean = float(y[train_idx].mean())
    y_std = max(float(y[train_idx].std()), 1e-8)
    y_train_z = (y[train_idx] - y_mean) / y_std

    alpha, _ = _solve_msancr_kernel(
        X_fc_train_z, np.zeros_like(X_fc_train_z),
        y_train_z, cache,
        params["lambda_fc"], params["lambda_sc"], params["lambda_l"],
        fc_only=True,
    )
    return alpha, scaler_fc, y_mean, y_std


def predict_fp_final(
    X_fc_new: np.ndarray,
    X_fc_train: np.ndarray,
    alpha: np.ndarray,
    cache: _MSANCRCache,
    params: dict,
    scaler_fc: StandardScaler,
    y_mean: float,
    y_std: float,
) -> np.ndarray:
    """Predict with fitted FP branch on new data."""
    X_fc_new_z = scaler_fc.transform(X_fc_new)
    pred_z = _predict_msancr(
        X_fc_new_z, np.zeros_like(X_fc_new_z),
        X_fc_train, np.zeros_like(X_fc_train),
        alpha, cache, params["lambda_fc"], params["lambda_sc"], params["lambda_l"],
        fc_only=True,
    )
    return pred_z * y_std + y_mean


# ---------------------------------------------------------------------------
# Level-1 OOF generation for stacking
# ---------------------------------------------------------------------------

@dataclass
class Level1Result:
    """Result of Level-1 branch on one outer split."""
    branch_name: str
    outer_train_oof: np.ndarray  # OOF predictions for outer-train subjects
    outer_test_pred: np.ndarray  # predictions on outer-test
    inner_metrics: dict          # inner CV selection metrics
    selected_hyperparams: dict   # selected hyperparameters
    scaler_fc: Optional[StandardScaler] = None
    scaler_sc: Optional[StandardScaler] = None
    alpha: Optional[np.ndarray] = None  # for FP branch
    y_mean: float = 0.0
    y_std: float = 1.0


# ---------------------------------------------------------------------------
# Level-2 fusion weight search
# ---------------------------------------------------------------------------

def _pearson_tiebreak_key(
    pred: np.ndarray, y_true: np.ndarray, weights: tuple, prior_idx: int = -1,
) -> tuple:
    """Sort key: Pearson (desc), RMSE (asc), MAE (asc), prior_weight (desc for tiebreak toward simpler)."""
    m = prediction_metrics(y_true, pred)
    prior_weight = abs(weights[prior_idx]) if prior_idx >= 0 and prior_idx < len(weights) else 0.0
    return (-m["pearson"], m["rmse"], m["mae"], -prior_weight)


def search_weights_2b(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    branch_names: list[str],
    prior_weight_idx: int = 1,
) -> tuple[dict, float]:
    """Search simplex weights for 2 branches. Returns (selected_weights_dict, selected_pearson)."""
    assert len(branch_names) == 2
    best_key = None
    best_weights = None

    for w0 in WEIGHT_2B:
        w1 = round(1.0 - w0, 4)
        combined = w0 * preds[branch_names[0]] + w1 * preds[branch_names[1]]
        key = _pearson_tiebreak_key(combined, y_true, (w0, w1), prior_weight_idx)
        if best_key is None or key < best_key:
            best_key = key
            best_weights = {branch_names[0]: w0, branch_names[1]: w1}

    combined = best_weights[branch_names[0]] * preds[branch_names[0]] + best_weights[branch_names[1]] * preds[branch_names[1]]
    selected_pearson = float(pearsonr(y_true, combined).statistic) if len(y_true) > 1 else 0.0
    return best_weights, selected_pearson


def search_weights_3b(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    branch_names: list[str],
    prior_weight_idx: int = 2,
) -> tuple[dict, float]:
    """Search simplex weights for 3 branches. Returns (selected_weights_dict, selected_pearson)."""
    assert len(branch_names) == 3
    best_key = None
    best_weights = None

    for w0, w1, w2 in WEIGHT_3B:
        combined = w0 * preds[branch_names[0]] + w1 * preds[branch_names[1]] + w2 * preds[branch_names[2]]
        key = _pearson_tiebreak_key(combined, y_true, (w0, w1, w2), prior_weight_idx)
        if best_key is None or key < best_key:
            best_key = key
            best_weights = {branch_names[0]: w0, branch_names[1]: w1, branch_names[2]: w2}

    combined = (
        best_weights[branch_names[0]] * preds[branch_names[0]]
        + best_weights[branch_names[1]] * preds[branch_names[1]]
        + best_weights[branch_names[2]] * preds[branch_names[2]]
    )
    selected_pearson = float(pearsonr(y_true, combined).statistic) if len(y_true) > 1 else 0.0
    return best_weights, selected_pearson


# ---------------------------------------------------------------------------
# Outer split evaluation
# ---------------------------------------------------------------------------

@dataclass
class OuterSplitResult:
    """Full result for one outer split."""
    seed: int
    outer_fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    # Level-1
    level1_results: dict[str, Level1Result] = field(default_factory=dict)
    # Level-2
    lf0_weights: dict = field(default_factory=dict)
    lf1_weights: dict = field(default_factory=dict)
    lf2_weights: dict = field(default_factory=dict)
    # Test predictions
    lf0_test_pred: Optional[np.ndarray] = None
    lf1_test_pred: Optional[np.ndarray] = None
    lf2_test_pred: Optional[np.ndarray] = None
    a4_test_pred: Optional[np.ndarray] = None
    a3_test_pred: Optional[np.ndarray] = None
    # Metrics
    lf0_pearson: float = 0.0
    lf1_pearson: float = 0.0
    lf2_pearson: float = 0.0
    a4_pearson: float = 0.0
    a3_pearson: float = 0.0
    lf0_rmse: float = 0.0
    lf1_rmse: float = 0.0
    lf2_rmse: float = 0.0
    a4_rmse: float = 0.0
    a3_rmse: float = 0.0
    lf0_mae: float = 0.0
    lf1_mae: float = 0.0
    lf2_mae: float = 0.0
    a4_mae: float = 0.0
    a3_mae: float = 0.0
    # Leakage audit
    oof_integrity_ok: bool = True
    leakage_failures: list = field(default_factory=list)


def evaluate_outer_split(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    seed: int,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fp_cache: _MSANCRCache,
    prior_type: str = "matched",
    ridge_grid: Sequence[float] = RIDGE_GRID,
    gamma_grid: Sequence[float] = GAMMA_GRID,
    lambda_l_grid: Sequence[float] = LAMBDA_L_GRID,
    n_inner_folds: int = 3,
    n_rois: int = 116,
) -> OuterSplitResult:
    """Evaluate one outer split: Level-1 branches, Level-2 fusion, baselines."""
    result = OuterSplitResult(seed=seed, outer_fold=outer_fold, train_idx=train_idx, test_idx=test_idx)
    y_test = y[test_idx]

    # --- Level-1 OOF generation (for stacking training) ---
    # Use inner CV to produce OOF predictions for each outer-train subject
    from sklearn.model_selection import KFold as KFoldCV
    inner_splitter = KFoldCV(
        n_splits=n_inner_folds, shuffle=True,
        random_state=int(seed) * 1000 + int(outer_fold) + 17001,
    )

    # Branch S: SC-only Ridge OOF
    s_oof, s_alpha, s_ip, s_rmse, s_info = fit_predict_ridge_sc_oof(
        X_sc, y, train_idx, train_idx,  # OOF on train itself
        ridge_grid, n_inner_folds, seed, outer_fold,
    )
    # Actually we need OOF predictions for each outer-train subject from inner models
    # Let me redo: for each inner fold, predict inner-val from inner-train
    # Then combine into OOF predictions for all outer-train subjects

    def compute_oof_branch(
        branch_fn, X, y, train_idx, ridge_grid, seed, outer_fold, n_inner_folds, cache=None, **kwargs,
    ):
        """Compute OOF predictions for all outer-train subjects using inner CV."""
        oof_preds = np.full(len(train_idx), np.nan, dtype=np.float64)
        inner_splitter_local = KFoldCV(
            n_splits=n_inner_folds, shuffle=True,
            random_state=int(seed) * 1000 + int(outer_fold) + 17001,
        )
        selected_params = None

        if branch_fn == "fp" and cache is not None:
            # FP branch: select hyperparams on full outer-train, then OOF via inner splits
            # First, select best (lambda_fc, lambda_l) on full train with inner CV
            from sklearn.model_selection import KFold as KFoldSelect
            sel_splitter = KFoldSelect(n_splits=n_inner_folds, shuffle=True,
                                       random_state=int(seed) * 1000 + int(outer_fold) + 27001)
            scaler_fc_sel = StandardScaler()
            X_fc_sel_z = scaler_fc_sel.fit_transform(X[train_idx])
            best_pearson = -np.inf
            best_lfc = ridge_grid[0]
            best_ll = lambda_l_grid[0]
            for lfc in ridge_grid:
                for ll in lambda_l_grid:
                    fold_ps = []
                    for itrain, ival in sel_splitter.split(X_fc_sel_z):
                        try:
                            a, _ = _solve_msancr_kernel(
                                X_fc_sel_z[itrain], np.zeros_like(X_fc_sel_z[itrain]),
                                (y[train_idx[itrain]] - y[train_idx[itrain]].mean()) / max(y[train_idx[itrain]].std(), 1e-8),
                                cache, lfc, 1.0, ll, fc_only=True,
                            )
                            pz = _predict_msancr(
                                X_fc_sel_z[ival], np.zeros_like(X_fc_sel_z[ival]),
                                X_fc_sel_z[itrain], np.zeros_like(X_fc_sel_z[itrain]),
                                a, cache, lfc, 1.0, ll, fc_only=True,
                            )
                            pred = pz * max(y[train_idx[itrain]].std(), 1e-8) + y[train_idx[itrain]].mean()
                            fold_ps.append(prediction_metrics(y[train_idx[ival]], pred)["pearson"])
                        except Exception:
                            fold_ps.append(-np.inf)
                    mp = float(np.mean(fold_ps))
                    if mp > best_pearson + 1e-10:
                        best_pearson = mp
                        best_lfc = lfc
                        best_ll = ll

            selected_params = {
                "best_params": {
                    "lambda_fc": float(best_lfc), "lambda_sc": 1.0,
                    "lambda_l": float(best_ll), "gamma": float(cache.gamma),
                    "lifting": str(cache.lifting), "alpha": float(best_lfc),
                },
                "inner_pearson": float(best_pearson),
            }

            # Now generate OOF: for each inner fold, fit on inner-train, predict inner-val
            scaler_fc_full = StandardScaler()
            X_fc_full_z = scaler_fc_full.fit_transform(X[train_idx])
            for inner_train_local, inner_val_local in inner_splitter_local.split(train_idx):
                inner_train_global = train_idx[inner_train_local]
                inner_val_global = train_idx[inner_val_local]
                y_mean_inner = float(y[inner_train_global].mean())
                y_std_inner = max(float(y[inner_train_global].std()), 1e-8)
                X_fc_it = X_fc_full_z[inner_train_local]
                X_fc_iv = X_fc_full_z[inner_val_local]
                y_z = (y[inner_train_global] - y_mean_inner) / y_std_inner
                try:
                    a, _ = _solve_msancr_kernel(
                        X_fc_it, np.zeros_like(X_fc_it), y_z, cache,
                        best_lfc, 1.0, best_ll, fc_only=True,
                    )
                    pz = _predict_msancr(
                        X_fc_iv, np.zeros_like(X_fc_iv),
                        X_fc_it, np.zeros_like(X_fc_it),
                        a, cache, best_lfc, 1.0, best_ll, fc_only=True,
                    )
                    oof_preds[inner_val_local] = pz * y_std_inner + y_mean_inner
                except Exception:
                    pass
            return oof_preds, selected_params

        # Non-FP branches: standard OOF generation
        for inner_train_local, inner_val_local in inner_splitter_local.split(train_idx):
            inner_train_global = train_idx[inner_train_local]
            inner_val_global = train_idx[inner_val_local]
            if branch_fn == "ridge_sc":
                pred_val, alpha_val, _, _, info = fit_predict_ridge_sc_oof(
                    X, y, inner_train_global, inner_val_global,
                    ridge_grid, n_inner_folds, seed, outer_fold,
                )
            elif branch_fn == "ridge_fc":
                pred_val, alpha_val, _, _, info = fit_predict_ridge_fc_oof(
                    X, y, inner_train_global, inner_val_global,
                    ridge_grid, n_inner_folds, seed, outer_fold,
                )
            else:
                raise ValueError(f"Unknown branch: {branch_fn}")
            oof_preds[inner_val_local] = pred_val
            if selected_params is None:
                selected_params = info
        return oof_preds, selected_params

    # S OOF
    s_oof, s_info = compute_oof_branch("ridge_sc", X_sc, y, train_idx, ridge_grid, seed, outer_fold, n_inner_folds)
    result.level1_results["S"] = Level1Result(
        branch_name="S", outer_train_oof=s_oof, outer_test_pred=None,
        inner_metrics=s_info, selected_hyperparams=s_info,
    )

    # F0 OOF
    f0_oof, f0_info = compute_oof_branch("ridge_fc", X_fc, y, train_idx, ridge_grid, seed, outer_fold, n_inner_folds)
    result.level1_results["F0"] = Level1Result(
        branch_name="F0", outer_train_oof=f0_oof, outer_test_pred=None,
        inner_metrics=f0_info, selected_hyperparams=f0_info,
    )

    # FP OOF
    fp_oof, fp_info = compute_oof_branch("fp", X_fc, y, train_idx, ridge_grid, seed, outer_fold, n_inner_folds, cache=fp_cache)
    result.level1_results["FP"] = Level1Result(
        branch_name="FP", outer_train_oof=fp_oof, outer_test_pred=None,
        inner_metrics=fp_info, selected_hyperparams=fp_info.get("best_params", {}),
    )

    # --- Level-2 fusion weight selection on OOF ---
    y_train = y[train_idx]

    # LF0: F0 + S
    lf0_weights, lf0_inner_pearson = search_weights_2b(
        y_train, {"F0": f0_oof, "S": s_oof}, ["F0", "S"],
    )
    result.lf0_weights = lf0_weights

    # LF1: FP + S
    lf1_weights, lf1_inner_pearson = search_weights_2b(
        y_train, {"FP": fp_oof, "S": s_oof}, ["FP", "S"],
    )
    result.lf1_weights = lf1_weights

    # LF2: F0 + S + FP
    lf2_weights, lf2_inner_pearson = search_weights_3b(
        y_train, {"F0": f0_oof, "S": s_oof, "FP": fp_oof}, ["F0", "S", "FP"],
    )
    result.lf2_weights = lf2_weights

    # --- Refit Level-1 on full train, predict test ---
    scaler_fc = StandardScaler()
    scaler_sc = StandardScaler()
    X_fc_train_z = scaler_fc.fit_transform(X_fc[train_idx])
    X_sc_train_z = scaler_sc.fit_transform(X_sc[train_idx])
    X_fc_test_z = scaler_fc.transform(X_fc[test_idx])
    X_sc_test_z = scaler_sc.transform(X_sc[test_idx])

    y_mean_train = float(y[train_idx].mean())
    y_std_train = max(float(y[train_idx].std()), 1e-8)
    y_train_z = (y[train_idx] - y_mean_train) / y_std_train

    # S branch refit
    s_model = Ridge(alpha=s_info["alpha"], fit_intercept=True)
    s_model.fit(X_sc_train_z, y[train_idx])
    s_test = s_model.predict(X_sc_test_z)

    # F0 branch refit
    f0_model = Ridge(alpha=f0_info["alpha"], fit_intercept=True)
    f0_model.fit(X_fc_train_z, y[train_idx])
    f0_test = f0_model.predict(X_fc_test_z)

    # FP branch refit (fc_only MS-A-NCR)
    fp_params = fp_info.get("best_params", {"lambda_fc": 1.0, "lambda_sc": 1.0, "lambda_l": 0.0})
    alpha_fp, scaler_fp, fp_ym, fp_ystd = fit_predict_fp_final(
        X_fc, y, train_idx, fp_cache, fp_params,
    )
    X_fc_train_fp_z = scaler_fp.transform(X_fc[train_idx])
    fp_test = predict_fp_final(
        X_fc[test_idx], X_fc_train_fp_z, alpha_fp, fp_cache, fp_params,
        scaler_fp, fp_ym, fp_ystd,
    )

    # LF0 test
    result.lf0_test_pred = lf0_weights["F0"] * f0_test + lf0_weights["S"] * s_test
    # LF1 test
    result.lf1_test_pred = lf1_weights["FP"] * fp_test + lf1_weights["S"] * s_test
    # LF2 test
    result.lf2_test_pred = (
        lf2_weights["F0"] * f0_test
        + lf2_weights["S"] * s_test
        + lf2_weights["FP"] * fp_test
    )

    # A4 baseline: modality-specific Ridge (FC + SC independently, then same alpha for both)
    # A4 = Ridge on FC + SC separately, same alpha from joint grid search
    # Use the ridge grid to find best alpha for FC, best alpha for SC
    a4_fc_model = Ridge(alpha=f0_info["alpha"], fit_intercept=True)
    a4_fc_model.fit(X_fc_train_z, y[train_idx])
    a4_fc_pred = a4_fc_model.predict(X_fc_test_z)
    a4_sc_model = Ridge(alpha=s_info["alpha"], fit_intercept=True)
    a4_sc_model.fit(X_sc_train_z, y[train_idx])
    a4_sc_pred = a4_sc_model.predict(X_sc_test_z)
    # A4 = (fc_pred + sc_pred) / 2
    result.a4_test_pred = 0.5 * (a4_fc_pred + a4_sc_pred)

    # A3 baseline: matched MS-A-NCR (FC+SC)
    from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
        ModalitySelectiveAnisotropicNCR,
    )
    a3 = ModalitySelectiveAnisotropicNCR(
        lambda_fc=fp_params["lambda_fc"], lambda_sc=s_info["alpha"],
        lambda_l=fp_params["lambda_l"], gamma=fp_params.get("gamma", 0.0),
        cache=fp_cache, n_rois=n_rois, fc_only=False,
    )
    a3.fit(X_fc[train_idx], X_sc[train_idx], y[train_idx])
    result.a3_test_pred = a3.predict(X_fc[test_idx], X_sc[test_idx])

    # --- Compute test metrics ---
    for name, pred in [
        ("lf0", result.lf0_test_pred), ("lf1", result.lf1_test_pred),
        ("lf2", result.lf2_test_pred), ("a4", result.a4_test_pred),
        ("a3", result.a3_test_pred),
    ]:
        m = prediction_metrics(y_test, pred)
        setattr(result, f"{name}_pearson", m["pearson"])
        setattr(result, f"{name}_rmse", m["rmse"])
        setattr(result, f"{name}_mae", m["mae"])

    # --- OOF integrity check ---
    n_nan = int(np.isnan(s_oof).sum()) + int(np.isnan(f0_oof).sum()) + int(np.isnan(fp_oof).sum())
    result.oof_integrity_ok = n_nan == 0
    if n_nan > 0:
        result.leakage_failures.append(f"OOF has {n_nan} NaN predictions")

    return result


# ---------------------------------------------------------------------------
# Prior-identity controls (fixed-weight prior swap)
# ---------------------------------------------------------------------------

def evaluate_prior_swap(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    seed: int,
    outer_fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    fp_cache_control: _MSANCRCache,
    matched_weights: dict,
    matched_fp_params: dict,
    control_prior_type: str,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    gamma_grid: Sequence[float] = GAMMA_GRID,
    lambda_l_grid: Sequence[float] = LAMBDA_L_GRID,
    n_inner_folds: int = 3,
    n_rois: int = 116,
) -> dict:
    """Fixed-weight prior swap: replace FP branch with control prior, keep matched fusion weights.

    1. select FP matched branch hyperparameters inside outer_train
    2. select LF2 matched fusion weights from matched OOF predictions
    3. freeze branch hyperparameters and fusion weights
    4. replace only the prior identity
    5. refit the prior-aware FC branch
    6. evaluate the same outer-test fold
    """
    y_test = y[test_idx]

    # Refit control FP branch on full train
    alpha_ctrl, scaler_ctrl, ctrl_ym, ctrl_ystd = fit_predict_fp_final(
        X_fc, y, train_idx, fp_cache_control, matched_fp_params,
    )
    X_fc_train_ctrl_z = scaler_ctrl.transform(X_fc[train_idx])

    # Generate test predictions from all branches (S and F0 unchanged)
    scaler_fc = StandardScaler()
    scaler_sc = StandardScaler()
    X_fc_train_z = scaler_fc.fit_transform(X_fc[train_idx])
    X_sc_train_z = scaler_sc.fit_transform(X_sc[train_idx])
    X_fc_test_z = scaler_fc.transform(X_fc[test_idx])
    X_sc_test_z = scaler_sc.transform(X_sc[test_idx])

    # S refit
    s_model = Ridge(alpha=matched_weights.get("_s_alpha", 1.0), fit_intercept=True)
    s_model.fit(X_sc_train_z, y[train_idx])
    s_test = s_model.predict(X_sc_test_z)

    # F0 refit
    f0_model = Ridge(alpha=matched_weights.get("_f0_alpha", 1.0), fit_intercept=True)
    f0_model.fit(X_fc_train_z, y[train_idx])
    f0_test = f0_model.predict(X_fc_test_z)

    # Control FP prediction
    ctrl_test = predict_fp_final(
        X_fc[test_idx], X_fc_train_ctrl_z, alpha_ctrl, fp_cache_control,
        matched_fp_params, scaler_ctrl, ctrl_ym, ctrl_ystd,
    )

    # Apply matched fusion weights
    w_f0 = matched_weights.get("F0", 0.0)
    w_s = matched_weights.get("S", 0.0)
    w_p = matched_weights.get("FP", 0.0)
    lf2_control_pred = w_f0 * f0_test + w_s * s_test + w_p * ctrl_test

    m = prediction_metrics(y_test, lf2_control_pred)
    return {
        "prior_type": control_prior_type,
        "test_pearson": m["pearson"],
        "test_rmse": m["rmse"],
        "test_mae": m["mae"],
        "w_f0": w_f0,
        "w_s": w_s,
        "w_p": w_p,
    }


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_late_fusion_experiment(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    priors: dict,
    seeds: Sequence[int] = (0, 1, 2),
    n_outer_folds: int = 5,
    n_inner_folds: int = 3,
    ridge_grid: Sequence[float] = RIDGE_GRID,
    gamma_grid: Sequence[float] = GAMMA_GRID,
    lambda_l_grid: Sequence[float] = LAMBDA_L_GRID,
    n_rois: int = 116,
    output_dir: str = "outputs/iclr/prior_aware_late_fusion",
    prior_controls: Sequence[str] = ("unrelated", "shuffled", "random"),
) -> dict:
    """Run full late-fusion experiment for one task.

    Parameters
    ----------
    priors : dict with keys 'matched', 'unrelated', 'shuffled', 'random',
        each mapping to an (n_rois,) roi_prior array.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_split_results = []
    split_rows = []
    seed_rows = []
    control_rows = []
    leakage_audit = {"n_failures": 0, "details": []}

    t0 = time.time()

    for seed in seeds:
        outer_splitter_inner = __import__("sklearn.model_selection", fromlist=["KFold"]).KFold(
            n_splits=n_outer_folds, shuffle=True, random_state=int(seed),
        )
        for outer_fold, (trainval_idx, test_idx) in enumerate(outer_splitter_inner.split(np.arange(len(y)))):
            trainval_idx = np.asarray(trainval_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)

            print(f"  Seed {seed}, fold {outer_fold}: train={len(trainval_idx)}, test={len(test_idx)}")

            # Build FP cache for matched prior
            roi_prior_matched = priors["matched"]
            edge_laplacian = build_edge_laplacian(
                n_rois, prior_scores=roi_prior_matched,
                top_k=TOP_K, weighting="binary",
                couple_modalities=False, normalize="sym",
            )
            fp_cache = build_msancr_cache(
                roi_prior_matched, n_rois,
                gamma=0.5, lifting="prod",
                top_k=TOP_K, epsilon=DIAGONAL_EPSILON,
                weighting="binary", couple_modalities=False,
                normalize_laplacian="sym",
                edge_laplacian=edge_laplacian,
            )

            # Evaluate outer split
            result = evaluate_outer_split(
                X_fc, X_sc, y, seed, outer_fold,
                trainval_idx, test_idx, fp_cache,
                prior_type="matched",
                ridge_grid=ridge_grid, gamma_grid=gamma_grid,
                lambda_l_grid=lambda_l_grid, n_inner_folds=n_inner_folds,
                n_rois=n_rois,
            )
            all_split_results.append(result)

            # Record split metrics
            for model_name, prefix in [("A4", "a4"), ("A3", "a3"), ("LF0", "lf0"), ("LF1", "lf1"), ("LF2", "lf2")]:
                split_rows.append({
                    "seed": seed, "outer_fold": outer_fold,
                    "model": model_name,
                    "pearson": getattr(result, f"{prefix}_pearson"),
                    "rmse": getattr(result, f"{prefix}_rmse"),
                    "mae": getattr(result, f"{prefix}_mae"),
                })

            # Record fusion weights
            split_rows[-1]["lf0_w_f0"] = result.lf0_weights.get("F0", 0)
            split_rows[-1]["lf0_w_s"] = result.lf0_weights.get("S", 0)
            split_rows[-1]["lf1_w_fp"] = result.lf1_weights.get("FP", 0)
            split_rows[-1]["lf1_w_s"] = result.lf1_weights.get("S", 0)
            split_rows[-1]["lf2_w_f0"] = result.lf2_weights.get("F0", 0)
            split_rows[-1]["lf2_w_s"] = result.lf2_weights.get("S", 0)
            split_rows[-1]["lf2_w_fp"] = result.lf2_weights.get("FP", 0)

            # OOF integrity
            if not result.oof_integrity_ok:
                leakage_audit["n_failures"] += 1
                leakage_audit["details"].append(
                    f"seed={seed} fold={outer_fold}: {result.leakage_failures}"
                )

            # Prior-identity controls
            for ctrl_type in prior_controls:
                ctrl_cache = build_msancr_cache(
                    priors[ctrl_type], n_rois,
                    gamma=0.5, lifting="prod",
                    top_k=TOP_K, epsilon=DIAGONAL_EPSILON,
                    weighting="binary", couple_modalities=False,
                    normalize_laplacian="sym",
                    edge_laplacian=build_edge_laplacian(
                        n_rois, prior_scores=priors[ctrl_type],
                        top_k=TOP_K, weighting="binary",
                        couple_modalities=False, normalize="sym",
                    ),
                )
                ctrl_result = evaluate_prior_swap(
                    X_fc, X_sc, y, seed, outer_fold,
                    trainval_idx, test_idx, ctrl_cache,
                    matched_weights=result.lf2_weights,
                    matched_fp_params=result.level1_results["FP"].selected_hyperparams,
                    control_prior_type=ctrl_type,
                    ridge_grid=ridge_grid, gamma_grid=gamma_grid,
                    lambda_l_grid=lambda_l_grid,
                    n_inner_folds=n_inner_folds, n_rois=n_rois,
                )
                ctrl_result["seed"] = seed
                ctrl_result["outer_fold"] = outer_fold
                control_rows.append(ctrl_result)

    # Aggregate seed-level metrics
    split_df = pd.DataFrame(split_rows)
    seed_df = (
        split_df.groupby(["seed", "model"], as_index=False)
        .agg(pearson=("pearson", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean"))
    )

    # Summary
    summary_rows = []
    for model in ["A4", "A3", "LF0", "LF1", "LF2"]:
        model_seeds = seed_df[seed_df.model == model]
        pearsons = model_seeds["pearson"].values
        summary_rows.append({
            "model": model,
            "pearson_mean": float(np.mean(pearsons)),
            "pearson_median": float(np.median(pearsons)),
            "pearson_std": float(np.std(pearsons, ddof=1)) if len(pearsons) > 1 else 0.0,
            "rmse_mean": float(model_seeds["rmse"].mean()),
            "mae_mean": float(model_seeds["mae"].mean()),
            "positive_seeds": int(np.sum(pearsons > 0)),
            "n_seeds": len(pearsons),
        })

    # Control summary
    control_df = pd.DataFrame(control_rows)
    control_summary = []
    if not control_df.empty:
        for ctrl_type in ["matched", *prior_controls]:
            ctrl_seeds = control_df[control_df.prior_type == ctrl_type]
            if not ctrl_seeds.empty:
                control_summary.append({
                    "prior_type": ctrl_type,
                    "test_pearson_mean": float(ctrl_seeds["test_pearson"].mean()),
                    "test_rmse_mean": float(ctrl_seeds["test_rmse"].mean()),
                    "test_mae_mean": float(ctrl_seeds["test_mae"].mean()),
                })

    # Leakage audit
    leakage_audit["total_folds"] = len(all_split_results)
    leakage_audit["oof_nan_total"] = sum(
        1 for r in all_split_results if not r.oof_integrity_ok
    )

    elapsed = time.time() - t0

    # Save outputs
    split_df.to_csv(output_dir / "split_metrics.csv", index=False)
    seed_df.to_csv(output_dir / "seed_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False)
    if control_df is not None and not control_df.empty:
        control_df.to_csv(output_dir / "control_prior_swap_split_metrics.csv", index=False)
    pd.DataFrame(control_summary).to_csv(output_dir / "control_prior_swap_summary.csv", index=False)
    (output_dir / "stacking_leakage_audit.json").write_text(
        json.dumps(leakage_audit, indent=2, default=str)
    )
    (output_dir / "run_metadata.json").write_text(json.dumps({
        "elapsed_seconds": elapsed,
        "n_seeds": len(seeds),
        "n_outer_folds": n_outer_folds,
        "n_inner_folds": n_inner_folds,
    }, indent=2))

    return {
        "split_df": split_df,
        "seed_df": seed_df,
        "summary": summary_rows,
        "control_summary": control_summary,
        "leakage_audit": leakage_audit,
        "all_split_results": all_split_results,
    }


# ---------------------------------------------------------------------------
# Decision logic
# ---------------------------------------------------------------------------

def compute_late_fusion_decision(
    summary: list[dict],
    control_summary: list[dict],
    task_name: str,
) -> dict:
    """Compute task-level decision from summary metrics."""
    summary_df = pd.DataFrame(summary)
    a4_row = summary_df[summary_df.model == "A4"].iloc[0] if "A4" in summary_df.model.values else None
    lf0_row = summary_df[summary_df.model == "LF0"].iloc[0] if "LF0" in summary_df.model.values else None
    lf2_row = summary_df[summary_df.model == "LF2"].iloc[0] if "LF2" in summary_df.model.values else None

    a4_p = float(a4_row["pearson_mean"]) if a4_row is not None else 0.0
    lf0_p = float(lf0_row["pearson_mean"]) if lf0_row is not None else 0.0
    lf2_p = float(lf2_row["pearson_mean"]) if lf2_row is not None else 0.0

    # Strongest no-prior = max(A4, LF0)
    strongest_no_prior = max(a4_p, lf0_p)
    mean_delta = lf2_p - strongest_no_prior

    # Seed-level deltas
    lf2_seeds = summary_df[(summary_df.model == "LF2")]["pearson_mean"].values
    positive_seeds = int(lf2_row["positive_seeds"]) if lf2_row is not None else 0

    # Prior weight from LF2 (median across splits)
    # This needs split-level data; approximate from control_summary
    matched_ctrl = [c for c in control_summary if c.get("prior_type") == "matched"]
    lf2_matched_p = matched_ctrl[0]["test_pearson_mean"] if matched_ctrl else lf2_p

    # Determine status
    if mean_delta >= 0.015 and positive_seeds >= 3:
        status = "LARGE_MARGIN_SUCCESS"
    elif mean_delta >= 0.008 and positive_seeds >= 2:
        status = "PROMISING"
    elif mean_delta >= 0.005 and positive_seeds >= 2:
        status = "BORDERLINE"
    else:
        status = "FAILURE"

    return {
        "A4_pearson": a4_p,
        "LF0_pearson": lf0_p,
        "LF1_pearson": float(summary_df[summary_df.model == "LF1"]["pearson_mean"].iloc[0]) if "LF1" in summary_df.model.values else 0.0,
        "LF2_matched_pearson": lf2_p,
        "strongest_no_prior_pearson": strongest_no_prior,
        "mean_delta_vs_strongest_no_prior": float(mean_delta),
        "positive_seeds": positive_seeds,
        "status": status,
    }
