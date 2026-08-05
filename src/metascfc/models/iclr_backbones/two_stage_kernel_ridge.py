#!/usr/bin/env python3
"""Two-Stage Biomarker-Guided Kernel Ridge (ICLR 2027, Method 3).

Instead of treating the MetaSFC saliency maps purely as post-hoc
explanations, the saliency becomes an *explicit feature-space projector* for
a secondary strong predictor:

    Stage 1 (biomarker): per-split, min-max normalized node saliency
        c in [0, 1]^n_rois is obtained from a *previously trained* model
        (the AAAI MS-Inter-GCN runs E0/E7/E8/E9, exported per split), so no
        retraining is needed and the splits are identical to every other
        method in the family.

    Stage 2 (predictor): the saliency is lifted to the connectome edge space
        (upper triangle) and broadcast over the FC and SC blocks,

            X_gated = X_std  ⊙  [ repeat(g)_FC | repeat(g)_SC ],

        where g_ij = (c_i * c_j)   (gate_mode="product", default)
                 or g_ij = (c_i + c_j)/2   (gate_mode="sum").

        A Kernel Ridge Regression (RBF kernel) is then trained on the gated
        (biomarker-constrained) feature space.  Candidate (alpha, gamma)
        pairs are selected only on the inner validation split (RMSE in raw
        target units); the winner is refit on train+val and evaluated on the
        outer test split.  This is exactly the "subject-specific
        prior-weighted kernel" idea: K(s, t) = k(x_s ⊙ c, x_t ⊙ c) with a
        common stabilized projection c per split.

Faithfulness to the shared protocol
------------------------------------
- Identical cohort, seeds (0-9), 5 outer folds, 15% inner validation, and
  resumable runner as scripts/40 and scripts/41.
- Feature standardization and target z-statistics are fitted on the inner
  training (or refit) partition only; the gate c is fixed per (seed, fold)
  and comes from the same split's stage-1 model (leakage-free by
  construction - the AAAI saliency was computed on fit subjects only).
- True / shuffled / random stage-1 biomarkers (E7/E8/E9) plus the no-prior
  stage-1 biomarker (E0) share identical grids, seeds, and folds.

Edge cases
----------
- Degenerate (all-equal) saliency: the gate falls back to an all-ones
  vector (no gating) and a note is logged, so KRR still runs.
- RBF kernel: the Gram matrix is n x n (<= 412), solved exactly; no
  GPU-specific kernels are required (sklearn/LAPACK), the runner still
  records the chosen device in its metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from sklearn.kernel_ridge import KernelRidge


# ---------------------------------------------------------------------------
# Hyperparameter container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KRRConfig:
    """One stage-2 Kernel Ridge candidate (selected by inner CV).

    Attributes
    ----------
    alpha : float
        Ridge regularization of the kernel solve.
    gamma : float
        RBF kernel width (gamma = 1 / (2 sigma^2), sklearn convention).
    kernel : str
        Kernel name (only 'rbf' is used here).
    """

    alpha: float
    gamma: float
    kernel: str = "rbf"


def upper_triangle_indices(n_rois: int) -> np.ndarray:
    """ij pairs of the upper triangle (row-major), shape (n_edges, 2)."""
    i, j = np.triu_indices(n_rois, k=1)
    return np.stack([i, j], axis=1)


def lift_node_saliency_to_edges(
    saliency: np.ndarray,
    n_rois: int,
    mode: str = "product",
) -> np.ndarray:
    """Lift a node-level saliency c in [0, 1]^n_rois to edge features.

    Each upper-triangle edge (i, j) receives the gate

        g_ij = c_i * c_j        (mode "product", emphasizes both endpoints)
        g_ij = (c_i + c_j) / 2  (mode "sum")

    Parameters
    ----------
    saliency : np.ndarray, shape (n_rois,)
        Min-max normalized node saliency (Stage-1 biomarker).
    n_rois : int
    mode : str
        "product" or "sum".

    Returns
    -------
    np.ndarray, shape (n_edges,)
        Non-negative edge gate in [0, 1].
    """
    if mode not in ("product", "sum"):
        raise ValueError(f"gate_mode must be 'product' or 'sum', got {mode!r}")
    c = np.asarray(saliency, dtype=np.float64).reshape(-1)
    if c.shape[0] != n_rois:
        raise ValueError(
            f"saliency has {c.shape[0]} entries; expected {n_rois}"
        )
    ii, jj = upper_triangle_indices(n_rois).T
    if mode == "product":
        return c[ii] * c[jj]
    return (c[ii] + c[jj]) / 2.0


def build_gated_features(
    x: np.ndarray,
    n_rois: int,
    gate: np.ndarray,
) -> np.ndarray:
    """Broadcast an edge gate over both the FC and SC feature blocks.

    Parameters
    ----------
    x : np.ndarray, shape (n_subjects, 2 * n_edges)
        Standardized [FC_upper | SC_upper] edge features.
    n_rois : int
    gate : np.ndarray, shape (n_edges,)
        Edge gate in [0, 1] (same lift applied to FC and SC blocks).

    Returns
    -------
    np.ndarray, shape (n_subjects, 2 * n_edges)
    """
    n_edges = upper_triangle_indices(n_rois).shape[0]
    gate_blk = np.asarray(gate, dtype=np.float64).reshape(1, -1)
    blk = np.repeat(gate_blk, 2, axis=1)  # = [g | g]
    return np.asarray(x, dtype=np.float64) * blk


def extract_upper(n_rois: int) -> np.ndarray:
    """Boolean mask selecting the upper-triangle entries of an (n, n) matrix."""
    mask = np.zeros((n_rois, n_rois), dtype=bool)
    mask[np.triu_indices(n_rois, k=1)] = True
    return mask


# ---------------------------------------------------------------------------
# Nested CV entry point (the script-42 workhorse)
# ---------------------------------------------------------------------------
def fit_predict_two_stage_krr(
    fc: np.ndarray,
    sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    saliency: np.ndarray,
    alpha_grid: Sequence[float],
    gamma_grid: Sequence[float],
    gate_mode: str = "product",
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict[str, float], float, np.ndarray, int]:
    """Nested selection + refit for one outer split (two-stage KRR).

    Protocol (identical to the other ICLR methods):
      1. X = [FC_upper | SC_upper] is standardized on the inner training
         partition only.
      2. The Stage-1 node saliency is lifted to the edge space and broadcast
         over both blocks (gating = feature-space projection).
      3. Every (alpha, gamma) candidate is fit on the *gated* train
         features; the best is selected by validation RMSE (raw target
         units).
      4. The winner is refit on train+val (scaler recomputed on the refit
         partition; the gate c is fixed per split) and predicts the outer
         test subjects in raw target units.
      5. The gate saliency (the Stage-1 biomarker actually used) is returned
         for the alignment/rank-stability pipeline.

    Parameters
    ----------
    fc / sc : np.ndarray, shape (n, n_rois, n_rois)
        Functional / structural connectomes of all subjects.
    y : np.ndarray, shape (n,)
        Raw target scores.
    train_idx / val_idx / test_idx : np.ndarray
        Partition indices from iter_nested_splits.
    saliency : np.ndarray, shape (n_rois,)
        Stage-1 min-max normalized node saliency of this split.
    alpha_grid / gamma_grid : Sequence[float]
        Candidate grids (cartesian product).
    gate_mode : str
        "product" or "sum" edge lift.
    verbose : bool
        Print the winning config for the split.

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float], float, np.ndarray, int]
        (test predictions in raw units, best config dict, best validation
        RMSE, gate saliency used (n_rois,), effective parameter count).
    """
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)

    fc = np.asarray(fc, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    n_rois = fc.shape[1]
    mask = extract_upper(n_rois)
    x = np.concatenate(
        [fc[:, mask], sc[:, mask]], axis=1
    )  # (n, 2 * n_edges) already upper-triangle float64
    n_edges = mask.sum()
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # Degenerate-saliency guard: fall back to no gating (all-ones gate).
    s = np.asarray(saliency, dtype=np.float64).reshape(-1)
    if s.max() - s.min() < 1e-12:
        s = np.ones_like(s)
    gate = lift_node_saliency_to_edges(s, n_rois, mode=gate_mode)

    # ---- inner training partition (model selection) ----
    x_train = x[train_idx]
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    x_train = build_gated_features((x_train - x_mean) / x_std, n_rois, gate)
    x_val = build_gated_features((x[val_idx] - x_mean) / x_std, n_rois, gate)
    y_train_mean, y_train_std = float(y[train_idx].mean()), float(y[train_idx].std())
    y_train_std = y_train_std if y_train_std >= 1e-8 else 1.0
    y_train_z = (y[train_idx] - y_train_mean) / y_train_std
    y_val_raw = y[val_idx]

    # ---- inner model selection on validation RMSE ----
    best_rmse = float("inf")
    best_cfg: Optional[KRRConfig] = None
    for alpha in alpha_grid:
        for gamma in gamma_grid:
            krr = KernelRidge(kernel="rbf", alpha=float(alpha), gamma=float(gamma))
            krr.fit(x_train, y_train_z)
            pred_val = krr.predict(x_val) * y_train_std + y_train_mean
            rmse = float(np.sqrt(np.mean((pred_val - y_val_raw) ** 2)))
            if rmse < best_rmse - 1e-12:
                best_rmse = rmse
                best_cfg = KRRConfig(alpha=float(alpha), gamma=float(gamma))
    if best_cfg is None:
        raise RuntimeError("No candidate configuration was selected")

    # ---- refit on train + validation, predict the outer test split ----
    fit_idx = np.concatenate([train_idx, val_idx])
    x_fit = x[fit_idx]
    x_mean = x_fit.mean(axis=0, keepdims=True)
    x_std = x_fit.std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    x_fit = build_gated_features((x_fit - x_mean) / x_std, n_rois, gate)
    x_test = build_gated_features((x[test_idx] - x_mean) / x_std, n_rois, gate)
    fit_mean, fit_std = float(y[fit_idx].mean()), float(y[fit_idx].std())
    fit_std = fit_std if fit_std >= 1e-8 else 1.0
    y_fit_z = (y[fit_idx] - fit_mean) / fit_std

    final_krr = KernelRidge(
        kernel="rbf", alpha=best_cfg.alpha, gamma=best_cfg.gamma
    )
    final_krr.fit(x_fit, y_fit_z)
    pred = (final_krr.predict(x_test) * fit_std + fit_mean).astype(np.float64)

    # Effective model size: dual weights + kernel width (sklearn KRR stores
    # n_train dual coefficients).
    n_params = int(len(fit_idx)) + 1

    best_cfg_dict = {
        "alpha": best_cfg.alpha,
        "gamma": best_cfg.gamma,
        "kernel": best_cfg.kernel,
        "gate_mode": gate_mode,
    }
    if verbose:
        print(
            f"    two-stage KRR: alpha={best_cfg.alpha} gamma={best_cfg.gamma} "
            f"val_rmse={best_rmse:.3f}",
            flush=True,
        )
    return pred, best_cfg_dict, best_rmse, s, n_params


# ---------------------------------------------------------------------------
# Stage-1 saliency loading (per-split, from the AAAI E* exports)
# ---------------------------------------------------------------------------
def load_split_node_saliency(saliency_dir: str, seed: int, fold: int) -> np.ndarray:
    """Load the per-split node saliency npz from an AAAI E* output directory.

    The E* runners wrote `saliency/seed{seed:02d}_fold{fold:02d}.npz`
    containing `node_saliency` (n_rois,), min-max normalized.

    Parameters
    ----------
    saliency_dir : str | Path
        Directory with the per-split npz files (e.g.
        ``outputs/aaai/final/E7_edge_true/saliency``).
    seed / fold : int
        Split identifiers (identical across all ICLR/AAAI methods).

    Returns
    -------
    np.ndarray, shape (n_rois,)
    """
    path = Path(saliency_dir) / f"seed{int(seed):02d}_fold{int(fold):02d}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Stage-1 saliency missing for (seed={seed}, fold={fold}): {path}"
        )
    sal = np.load(path)["node_saliency"]
    sal = np.asarray(sal, dtype=np.float64).reshape(-1)
    span = sal.max() - sal.min()
    if span > 1e-12:
        sal = (sal - sal.min()) / span
    return sal


__all__ = [
    "KRRConfig",
    "extract_upper",
    "fit_predict_two_stage_krr",
    "lift_node_saliency_to_edges",
    "load_split_node_saliency",
    "upper_triangle_indices",
]