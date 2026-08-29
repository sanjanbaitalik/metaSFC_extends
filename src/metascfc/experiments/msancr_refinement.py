"""Corrected, targeted Working-Memory MS-A-NCR refinement.

This module is intentionally importable: the split, selection, prior-swap,
and decision behavior used by the CLI is covered directly by unit tests.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from metascfc.benchmark_utils import (
    atomic_write_csv,
    iter_nested_splits,
    load_connectomes,
    prediction_metrics,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    _MSANCRCache,
    _predict_msancr,
    _solve_msancr_kernel,
    build_msancr_cache,
    recover_msancr_beta,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    EdgeLaplacian,
    build_edge_laplacian,
    node_saliency_from_beta,
)

MODEL_A4 = "A4_modality_ridge"
MODEL_A4_ISO = "A4_iso_same_solver"
MODEL_A2 = "A2_fc_laplacian"
MODEL_A3 = "A3_msancr"
BASE_MODELS = (MODEL_A4, MODEL_A4_ISO, MODEL_A2, MODEL_A3)
CONTROL_PRIORS = ("unrelated", "shuffled", "random")
REQUIRED_PRIORS = ("matched", *CONTROL_PRIORS)
PEARSON_TIE_TOLERANCE = 0.002


def stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def atomic_write_json(value: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def validate_refinement_config(cfg: Mapping[str, Any], enforce_seed_gate: bool = True) -> None:
    if list(cfg.get("targets", {})) != ["working_memory"]:
        raise ValueError("The refinement config must contain Working Memory only")
    seeds = [int(s) for s in cfg.get("seeds", [])]
    if enforce_seed_gate:
        if seeds != [0, 1, 2]:
            raise ValueError("The refinement is restricted to seeds [0, 1, 2]")
    elif seeds != list(range(10)):
        raise ValueError("The final 10x5 runner requires seeds [0..9]")
    if int(cfg.get("n_outer_folds", 0)) != 5 or int(cfg.get("n_inner_folds", 0)) != 3:
        raise ValueError("The refinement requires 5 outer folds and 3 inner folds")
    prior_cfg = cfg.get("priors", {}).get("working_memory", {})
    missing = [name for name in REQUIRED_PRIORS if not prior_cfg.get(name)]
    if missing:
        raise ValueError(f"Missing required Working-Memory priors: {missing}")
    requested = tuple(cfg.get("prior_controls", []))
    if requested != CONTROL_PRIORS:
        raise ValueError(f"prior_controls must be exactly {list(CONTROL_PRIORS)}")
    if set(cfg.get("lifting_rules", [])) != {"prod", "mean"}:
        raise ValueError("lifting_rules must execute both prod and mean")
    if any(float(g) <= 0 for g in cfg.get("gamma_grid", [])):
        raise ValueError("A3 gamma_grid values must all be positive")
    if any(float(v) <= 0 for v in cfg.get("lambda_laplacian_grid", [])):
        raise ValueError("A2/A3 lambda_L candidates must all be positive")
    for key in ("ridge_grid", "ridge_expanded_grid"):
        vals = [float(v) for v in cfg.get(key, [])]
        if not vals or any(v <= 0 for v in vals) or vals != sorted(set(vals)):
            raise ValueError(f"{key} must be a sorted, unique, positive list")


def load_roi_prior(path: str | Path, n_rois: int) -> np.ndarray:
    frame = pd.read_csv(path)
    if "roi_index" in frame:
        frame = frame.sort_values("roi_index")
    if "prior_score" not in frame:
        raise ValueError(f"Prior {path} lacks prior_score")
    values = frame["prior_score"].to_numpy(np.float64)
    if values.shape != (n_rois,) or not np.isfinite(values).all():
        raise ValueError(f"Prior {path} is not a finite ({n_rois},) vector")
    return values


def upper_triangle_features(mats: np.ndarray) -> np.ndarray:
    iu = np.triu_indices(mats.shape[1], k=1)
    return mats[:, iu[0], iu[1]].astype(np.float64)


def iter_refinement_outer_splits(
    y: np.ndarray,
    seeds: Sequence[int],
    n_folds: int,
    groups: Optional[np.ndarray],
    historical_val_fraction: float = 0.15,
):
    """Yield the historical outer folds, recombining old train/val as trainval."""
    for seed, fold, train_idx, val_idx, test_idx in iter_nested_splits(
        y, seeds, n_folds, historical_val_fraction, groups,
    ):
        trainval_idx = np.sort(np.concatenate([train_idx, val_idx])).astype(int)
        if np.intersect1d(trainval_idx, test_idx).size:
            raise RuntimeError("Outer train/test leakage detected")
        yield seed, fold, trainval_idx, np.asarray(test_idx, dtype=int)


def make_inner_cv_splits(
    outer_train_idx: np.ndarray,
    y: np.ndarray,
    seed: int,
    outer_fold: int,
    n_splits: int = 3,
    groups: Optional[np.ndarray] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic global-index inner folds inside outer train only."""
    outer_train_idx = np.asarray(outer_train_idx, dtype=int)
    if groups is None:
        splitter = KFold(
            n_splits=n_splits, shuffle=True,
            random_state=int(seed) * 1000 + int(outer_fold) + 17001,
        )
        iterator = splitter.split(outer_train_idx)
    else:
        splitter = GroupKFold(n_splits=n_splits)
        iterator = splitter.split(
            outer_train_idx, y[outer_train_idx], groups[outer_train_idx]
        )
    splits = []
    validation_seen: list[int] = []
    for train_local, val_local in iterator:
        train_global = outer_train_idx[np.asarray(train_local, dtype=int)]
        val_global = outer_train_idx[np.asarray(val_local, dtype=int)]
        if np.intersect1d(train_global, val_global).size:
            raise RuntimeError("Inner train/validation overlap detected")
        if groups is not None and np.intersect1d(groups[train_global], groups[val_global]).size:
            raise RuntimeError("Inner family/group leakage detected")
        splits.append((train_global, val_global))
        validation_seen.extend(val_global.tolist())
    if sorted(validation_seen) != sorted(outer_train_idx.tolist()):
        raise RuntimeError("Inner CV did not produce exactly one OOF prediction per subject")
    return splits


@dataclass
class InnerFoldData:
    inner_fold: int
    train_idx: np.ndarray
    val_idx: np.ndarray
    X_fc_train: np.ndarray
    X_sc_train: np.ndarray
    X_fc_val: np.ndarray
    X_sc_val: np.ndarray
    y_train_z: np.ndarray
    y_val: np.ndarray
    y_mean: float
    y_std: float
    scaler_fc_mean: np.ndarray
    scaler_sc_mean: np.ndarray


def prepare_inner_folds(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> list[InnerFoldData]:
    prepared = []
    for inner_fold, (train_idx, val_idx) in enumerate(splits):
        scaler_fc, scaler_sc = StandardScaler(), StandardScaler()
        X_fc_train = scaler_fc.fit_transform(X_fc[train_idx])
        X_sc_train = scaler_sc.fit_transform(X_sc[train_idx])
        y_mean = float(np.mean(y[train_idx]))
        y_std = max(float(np.std(y[train_idx])), 1e-8)
        prepared.append(InnerFoldData(
            inner_fold=inner_fold,
            train_idx=np.asarray(train_idx, dtype=int),
            val_idx=np.asarray(val_idx, dtype=int),
            X_fc_train=X_fc_train,
            X_sc_train=X_sc_train,
            X_fc_val=scaler_fc.transform(X_fc[val_idx]),
            X_sc_val=scaler_sc.transform(X_sc[val_idx]),
            y_train_z=(y[train_idx] - y_mean) / y_std,
            y_val=np.asarray(y[val_idx], dtype=np.float64),
            y_mean=y_mean,
            y_std=y_std,
            scaler_fc_mean=scaler_fc.mean_.copy(),
            scaler_sc_mean=scaler_sc.mean_.copy(),
        ))
    return prepared


def candidate_id(candidate: Mapping[str, Any]) -> str:
    keys = ("lambda_fc", "lambda_sc", "lambda_l", "gamma", "lifting", "cache_key")
    return stable_hash({key: candidate.get(key) for key in keys})


def _simplicity_key(row: Mapping[str, Any]) -> tuple:
    return (
        float(row.get("gamma", 0.0)),
        float(row.get("lambda_l", 0.0)),
        0 if row.get("lifting") in (None, "none", "prod") else 1,
        abs(np.log10(float(row.get("lambda_fc", 1.0))) - np.log10(float(row.get("lambda_sc", 1.0)))),
    )


def select_best_candidate(
    summary: pd.DataFrame,
    pearson_tolerance: float = PEARSON_TIE_TOLERANCE,
) -> dict[str, Any]:
    """Select by Pearson first; RMSE/MAE/simplicity break near-ties only."""
    if summary.empty:
        raise ValueError("Cannot select from an empty candidate table")
    if not np.isfinite(summary[["mean_pearson", "mean_rmse", "mean_mae"]]).all().all():
        raise ValueError("Candidate summary contains non-finite metrics")
    best_pearson = float(summary["mean_pearson"].max())
    eligible = summary[summary.mean_pearson >= best_pearson - pearson_tolerance].copy()
    eligible["_simplicity"] = [
        _simplicity_key(row) for row in eligible.to_dict("records")
    ]
    eligible = eligible.sort_values(
        ["mean_rmse", "mean_mae", "_simplicity", "candidate_id"],
        kind="mergesort",
    )
    result = eligible.iloc[0].drop(labels="_simplicity").to_dict()
    result["pearson_tolerance"] = float(pearson_tolerance)
    return result


class CacheFactory:
    """Lazy prior/lifting/gamma cache with one structural Laplacian per prior."""

    def __init__(
        self,
        priors: Mapping[str, np.ndarray],
        n_rois: int,
        top_k: int,
        epsilon: float,
        weighting: str,
        normalization: str,
    ) -> None:
        self.priors = {key: np.asarray(value, dtype=np.float64) for key, value in priors.items()}
        self.n_rois = int(n_rois)
        self.top_k = int(top_k)
        self.epsilon = float(epsilon)
        self.weighting = str(weighting)
        self.normalization = str(normalization)
        self._laplacians: dict[str, EdgeLaplacian] = {}
        self._caches: dict[str, _MSANCRCache] = {}

    @staticmethod
    def key(prior_type: str, gamma: float, lifting: str) -> str:
        return f"{prior_type}|g={float(gamma):g}|lift={lifting}"

    def get(self, prior_type: str, gamma: float, lifting: str) -> tuple[str, _MSANCRCache]:
        if prior_type not in self.priors:
            raise KeyError(f"Unknown prior type: {prior_type}")
        key = self.key(prior_type, gamma, lifting)
        if key in self._caches:
            return key, self._caches[key]
        if prior_type not in self._laplacians:
            self._laplacians[prior_type] = build_edge_laplacian(
                self.n_rois,
                prior_scores=self.priors[prior_type],
                top_k=self.top_k,
                weighting=self.weighting,
                couple_modalities=False,
                normalize=self.normalization,
            )
        self._caches[key] = build_msancr_cache(
            self.priors[prior_type],
            self.n_rois,
            gamma=float(gamma),
            lifting=lifting,
            top_k=self.top_k,
            epsilon=self.epsilon,
            weighting=self.weighting,
            couple_modalities=False,
            normalize_laplacian=self.normalization,
            edge_laplacian=self._laplacians[prior_type],
        )
        return key, self._caches[key]


@dataclass
class PreparedKernel:
    active_train: np.ndarray
    active_val: np.ndarray
    inactive_train_gram: np.ndarray
    inactive_val_cross: np.ndarray
    sc_train_gram: np.ndarray
    sc_val_cross: np.ndarray
    fold: InnerFoldData
    scale: float


def prepare_kernel(fold: InnerFoldData, cache: _MSANCRCache) -> PreparedKernel:
    active = cache.active_indices
    inactive_mask = np.ones(cache.n_edges, dtype=bool)
    inactive_mask[active] = False
    inactive = np.where(inactive_mask)[0]
    if len(active):
        active_train = (
            fold.X_fc_train[:, active] * cache.D_inv_sqrt[active][None, :]
        ) @ cache.generalized_u
        active_val = (
            fold.X_fc_val[:, active] * cache.D_inv_sqrt[active][None, :]
        ) @ cache.generalized_u
    else:
        active_train = np.empty((len(fold.train_idx), 0), dtype=np.float64)
        active_val = np.empty((len(fold.val_idx), 0), dtype=np.float64)
    if len(inactive):
        inactive_train = (
            fold.X_fc_train[:, inactive] * cache.D_inv_sqrt[inactive][None, :]
        )
        inactive_val = (
            fold.X_fc_val[:, inactive] * cache.D_inv_sqrt[inactive][None, :]
        )
        inactive_train_gram = inactive_train @ inactive_train.T
        inactive_val_cross = inactive_val @ inactive_train.T
    else:
        inactive_train_gram = np.zeros((len(fold.train_idx), len(fold.train_idx)))
        inactive_val_cross = np.zeros((len(fold.val_idx), len(fold.train_idx)))
    return PreparedKernel(
        active_train=active_train,
        active_val=active_val,
        inactive_train_gram=inactive_train_gram,
        inactive_val_cross=inactive_val_cross,
        sc_train_gram=fold.X_sc_train @ fold.X_sc_train.T,
        sc_val_cross=fold.X_sc_val @ fold.X_sc_train.T,
        fold=fold,
        scale=float(max(1, cache.n_edges * 2)),
    )


def predict_inner_candidate(
    prepared: PreparedKernel,
    cache: _MSANCRCache,
    lambda_fc: float,
    lambda_sc: float,
    lambda_l: float,
) -> np.ndarray:
    if lambda_fc <= 0 or lambda_sc <= 0 or lambda_l < 0:
        raise ValueError("Invalid non-positive Ridge or negative Laplacian penalty")
    denom = lambda_fc + lambda_l * cache.generalized_mu
    active_train = (
        (prepared.active_train / denom) @ prepared.active_train.T
        if prepared.active_train.shape[1]
        else np.zeros_like(prepared.sc_train_gram)
    )
    kernel = (
        active_train
        + prepared.inactive_train_gram / lambda_fc
        + prepared.sc_train_gram / lambda_sc
    ) / prepared.scale
    alpha = np.linalg.solve(
        kernel + np.eye(kernel.shape[0], dtype=np.float64),
        prepared.fold.y_train_z,
    )
    active_cross = (
        (prepared.active_val / denom) @ prepared.active_train.T
        if prepared.active_train.shape[1]
        else np.zeros_like(prepared.sc_val_cross)
    )
    cross = (
        active_cross
        + prepared.inactive_val_cross / lambda_fc
        + prepared.sc_val_cross / lambda_sc
    ) / prepared.scale
    pred_z = cross @ alpha
    return pred_z * prepared.fold.y_std + prepared.fold.y_mean


def evaluate_candidates(
    candidates: Sequence[Mapping[str, Any]],
    inner_folds: Sequence[InnerFoldData],
    cache_factory: CacheFactory,
    stage: str,
    seed: int,
    outer_fold: int,
    model_id: str,
    prior_type: str,
    prepared_cache: Optional[dict[tuple[int, str], PreparedKernel]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate candidates on identical inner folds and return fold/mean rows."""
    if not candidates:
        raise ValueError(f"No candidates supplied for {model_id} {stage}")
    prepared = prepared_cache if prepared_cache is not None else {}
    fold_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        cid = candidate_id(candidate)
        key, cache = cache_factory.get(
            prior_type,
            float(candidate["gamma"]),
            str(candidate["lifting"]),
        )
        if candidate.get("cache_key") not in (None, key):
            raise ValueError("Candidate cache identity disagrees with its geometry")
        for fold in inner_folds:
            prepared_key = (fold.inner_fold, key)
            if prepared_key not in prepared:
                prepared[prepared_key] = prepare_kernel(fold, cache)
            try:
                pred = predict_inner_candidate(
                    prepared[prepared_key], cache,
                    float(candidate["lambda_fc"]),
                    float(candidate["lambda_sc"]),
                    float(candidate["lambda_l"]),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Candidate failed: model={model_id} stage={stage} seed={seed} "
                    f"outer_fold={outer_fold} inner_fold={fold.inner_fold} "
                    f"candidate={dict(candidate)}"
                ) from exc
            metrics = prediction_metrics(fold.y_val, pred)
            fold_rows.append({
                "seed": int(seed),
                "outer_fold": int(outer_fold),
                "inner_fold": int(fold.inner_fold),
                "stage": stage,
                "model_id": model_id,
                "prior_type": prior_type,
                "candidate_id": cid,
                "cache_key": key,
                "lambda_fc": float(candidate["lambda_fc"]),
                "lambda_sc": float(candidate["lambda_sc"]),
                "lambda_l": float(candidate["lambda_l"]),
                "gamma": float(candidate["gamma"]),
                "lifting": str(candidate["lifting"]),
                "n_inner_train": int(len(fold.train_idx)),
                "n_inner_val": int(len(fold.val_idx)),
                "inner_train_indices_hash": stable_hash(fold.train_idx.tolist()),
                "inner_val_indices_hash": stable_hash(fold.val_idx.tolist()),
                "scaler_fc_mean_hash": stable_hash(fold.scaler_fc_mean.tolist()),
                "scaler_sc_mean_hash": stable_hash(fold.scaler_sc_mean.tolist()),
                **metrics,
            })
    fold_df = pd.DataFrame(fold_rows)
    summary = (
        fold_df.groupby(
            ["candidate_id", "cache_key", "lambda_fc", "lambda_sc", "lambda_l", "gamma", "lifting"],
            as_index=False,
        )
        .agg(
            mean_pearson=("pearson", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_mae=("mae", "mean"),
            n_inner_folds=("inner_fold", "nunique"),
        )
    )
    if not (summary.n_inner_folds == len(inner_folds)).all():
        raise RuntimeError("A candidate is missing an inner fold")
    return fold_df, summary


def _candidates_for_ridge(grid: Sequence[float], isotropic: bool) -> list[dict[str, Any]]:
    if isotropic:
        pairs = [(value, value) for value in grid]
    else:
        pairs = [(left, right) for left in grid for right in grid]
    return [
        {
            "lambda_fc": float(left), "lambda_sc": float(right),
            "lambda_l": 0.0, "gamma": 0.0, "lifting": "prod",
            "cache_key": CacheFactory.key("matched", 0.0, "prod"),
        }
        for left, right in pairs
    ]


def local_neighbors(value: float, grid: Sequence[float]) -> list[float]:
    grid = [float(v) for v in grid]
    index = min(range(len(grid)), key=lambda idx: abs(grid[idx] - float(value)))
    return grid[max(0, index - 1):min(len(grid), index + 2)]


def tune_outer_split(
    inner_folds: Sequence[InnerFoldData],
    cache_factory: CacheFactory,
    cfg: Mapping[str, Any],
    seed: int,
    outer_fold: int,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    """Run Stage A/B/C selection for A4, A4-iso, A2, and matched A3."""
    all_inner_rows: list[pd.DataFrame] = []
    prepared_cache: dict[tuple[int, str], PreparedKernel] = {}
    ridge_grid = [float(v) for v in cfg["ridge_grid"]]
    expanded_grid = [float(v) for v in cfg["ridge_expanded_grid"]]

    a4_fold, a4_summary = evaluate_candidates(
        _candidates_for_ridge(ridge_grid, isotropic=False), inner_folds,
        cache_factory, "A", seed, outer_fold, MODEL_A4, "matched", prepared_cache,
    )
    all_inner_rows.append(a4_fold)
    a4_initial = select_best_candidate(a4_summary)
    initial_boundary = (
        float(a4_initial["lambda_fc"]) in (ridge_grid[0], ridge_grid[-1])
        or float(a4_initial["lambda_sc"]) in (ridge_grid[0], ridge_grid[-1])
    )
    expand = initial_boundary and expanded_grid != ridge_grid
    final_grid = expanded_grid if expand else ridge_grid
    if expand:
        expanded_fold, expanded_summary = evaluate_candidates(
            _candidates_for_ridge(expanded_grid, isotropic=False), inner_folds,
            cache_factory, "A_expanded", seed, outer_fold, MODEL_A4, "matched", prepared_cache,
        )
        all_inner_rows.append(expanded_fold)
        a4 = select_best_candidate(expanded_summary)
    else:
        a4 = a4_initial

    iso_fold, iso_summary = evaluate_candidates(
        _candidates_for_ridge(final_grid, isotropic=True), inner_folds,
        cache_factory, "A_iso", seed, outer_fold, MODEL_A4_ISO, "matched", prepared_cache,
    )
    all_inner_rows.append(iso_fold)
    a4_iso = select_best_candidate(iso_summary)

    a2_stage_b = [
        {
            "lambda_fc": float(a4["lambda_fc"]),
            "lambda_sc": float(a4["lambda_sc"]),
            "lambda_l": float(lambda_l), "gamma": 0.0, "lifting": "prod",
            "cache_key": CacheFactory.key("matched", 0.0, "prod"),
        }
        for lambda_l in cfg["lambda_laplacian_grid"]
    ]
    a2_b_fold, a2_b_summary = evaluate_candidates(
        a2_stage_b, inner_folds, cache_factory, "B", seed, outer_fold,
        MODEL_A2, "matched", prepared_cache,
    )
    all_inner_rows.append(a2_b_fold)
    a2_geometry = select_best_candidate(a2_b_summary)

    a3_stage_b = []
    for lifting in cfg["lifting_rules"]:
        for gamma in cfg["gamma_grid"]:
            for lambda_l in cfg["lambda_laplacian_grid"]:
                a3_stage_b.append({
                    "lambda_fc": float(a4["lambda_fc"]),
                    "lambda_sc": float(a4["lambda_sc"]),
                    "lambda_l": float(lambda_l), "gamma": float(gamma),
                    "lifting": str(lifting),
                    "cache_key": CacheFactory.key("matched", float(gamma), str(lifting)),
                })
    a3_b_fold, a3_b_summary = evaluate_candidates(
        a3_stage_b, inner_folds, cache_factory, "B", seed, outer_fold,
        MODEL_A3, "matched", prepared_cache,
    )
    all_inner_rows.append(a3_b_fold)
    a3_geometry = select_best_candidate(a3_b_summary)

    fc_local = local_neighbors(float(a4["lambda_fc"]), final_grid)
    sc_local = local_neighbors(float(a4["lambda_sc"]), final_grid)
    a2_stage_c = [
        {
            "lambda_fc": lfc, "lambda_sc": lsc,
            "lambda_l": float(a2_geometry["lambda_l"]),
            "gamma": 0.0, "lifting": "prod",
            "cache_key": CacheFactory.key("matched", 0.0, "prod"),
        }
        for lfc in fc_local for lsc in sc_local
    ]
    a2_c_fold, a2_c_summary = evaluate_candidates(
        a2_stage_c, inner_folds, cache_factory, "C", seed, outer_fold,
        MODEL_A2, "matched", prepared_cache,
    )
    all_inner_rows.append(a2_c_fold)
    a2 = select_best_candidate(a2_c_summary)

    a3_stage_c = [
        {
            "lambda_fc": lfc, "lambda_sc": lsc,
            "lambda_l": float(a3_geometry["lambda_l"]),
            "gamma": float(a3_geometry["gamma"]),
            "lifting": str(a3_geometry["lifting"]),
            "cache_key": CacheFactory.key(
                "matched", float(a3_geometry["gamma"]), str(a3_geometry["lifting"])
            ),
        }
        for lfc in fc_local for lsc in sc_local
    ]
    if len(a3_stage_c) > 9:
        raise RuntimeError("Stage C exceeded the requested 3x3 coordinate refinement")
    a3_c_fold, a3_c_summary = evaluate_candidates(
        a3_stage_c, inner_folds, cache_factory, "C", seed, outer_fold,
        MODEL_A3, "matched", prepared_cache,
    )
    all_inner_rows.append(a3_c_fold)
    a3 = select_best_candidate(a3_c_summary)

    selected = {
        MODEL_A4: a4,
        MODEL_A4_ISO: a4_iso,
        MODEL_A2: a2,
        MODEL_A3: a3,
    }
    boundary = {
        "seed": int(seed), "fold": int(outer_fold),
        "initial_grid_min": ridge_grid[0], "initial_grid_max": ridge_grid[-1],
        "initial_lambda_fc": float(a4_initial["lambda_fc"]),
        "initial_lambda_sc": float(a4_initial["lambda_sc"]),
        "initial_boundary_selected": bool(initial_boundary),
        "expanded": bool(initial_boundary),
        "final_grid_min": final_grid[0], "final_grid_max": final_grid[-1],
        "final_lambda_fc": float(a4["lambda_fc"]),
        "final_lambda_sc": float(a4["lambda_sc"]),
        "final_boundary_selected": bool(
            float(a4["lambda_fc"]) in (final_grid[0], final_grid[-1])
            or float(a4["lambda_sc"]) in (final_grid[0], final_grid[-1])
        ),
    }
    return selected, pd.concat(all_inner_rows, ignore_index=True), boundary


def hyperparameter_payload(selected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "lambda_fc": float(selected["lambda_fc"]),
        "lambda_sc": float(selected["lambda_sc"]),
        "lambda_l": float(selected["lambda_l"]),
        "gamma": float(selected["gamma"]),
        "lifting": str(selected["lifting"]),
    }


def make_fixed_prior_swap_configs(
    matched_selected: Mapping[str, Any],
    controls: Sequence[str] = CONTROL_PRIORS,
) -> dict[str, dict[str, Any]]:
    """Return control identities mapped to exact copies of matched-selected HPs."""
    if tuple(controls) != CONTROL_PRIORS:
        raise ValueError(f"Fixed swaps require exactly {list(CONTROL_PRIORS)}")
    payload = hyperparameter_payload(matched_selected)
    return {control: dict(payload) for control in controls}


def fit_final_model(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    test_idx: np.ndarray,
    cache: _MSANCRCache,
    selected: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Refit a selected exact model on all outer-training subjects."""
    fit_idx = np.asarray(fit_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    scaler_fc, scaler_sc = StandardScaler(), StandardScaler()
    X_fc_fit = scaler_fc.fit_transform(X_fc[fit_idx])
    X_sc_fit = scaler_sc.fit_transform(X_sc[fit_idx])
    X_fc_test = scaler_fc.transform(X_fc[test_idx])
    X_sc_test = scaler_sc.transform(X_sc[test_idx])
    y_mean = float(np.mean(y[fit_idx]))
    y_std = max(float(np.std(y[fit_idx])), 1e-8)
    y_fit_z = (y[fit_idx] - y_mean) / y_std
    hp = hyperparameter_payload(selected)
    alpha, _ = _solve_msancr_kernel(
        X_fc_fit, X_sc_fit, y_fit_z, cache,
        hp["lambda_fc"], hp["lambda_sc"], hp["lambda_l"],
    )
    pred_z = _predict_msancr(
        X_fc_test, X_sc_test, X_fc_fit, X_sc_fit, alpha, cache,
        hp["lambda_fc"], hp["lambda_sc"], hp["lambda_l"],
    )
    beta_fc, beta_sc = recover_msancr_beta(
        X_fc_fit, X_sc_fit, alpha, cache,
        hp["lambda_fc"], hp["lambda_sc"], hp["lambda_l"],
    )
    prediction = pred_z * y_std + y_mean
    prediction_from_beta = (X_fc_test @ beta_fc + X_sc_test @ beta_sc) * y_std + y_mean
    if not np.allclose(prediction, prediction_from_beta, atol=1e-7, rtol=1e-6):
        raise RuntimeError("Dual prediction and recovered primal coefficients disagree")
    beta_fc_original_target = beta_fc * y_std
    beta_sc_original_target = beta_sc * y_std
    fc_only_saliency = node_saliency_from_beta(
        np.concatenate([beta_fc_original_target, np.zeros_like(beta_sc_original_target)]),
        cache.n_rois,
    )
    return prediction, beta_fc_original_target, beta_sc_original_target, fc_only_saliency


def safe_correlation(left: np.ndarray, right: np.ndarray, kind: str) -> float:
    result = pearsonr(left, right) if kind == "pearson" else spearmanr(left, right)
    value = float(result.statistic)
    return value if np.isfinite(value) else 0.0


def topk_jaccard(left: np.ndarray, right: np.ndarray, k: int = 10) -> float:
    if k <= 0 or k > len(left) or len(left) != len(right):
        raise ValueError("Invalid top-k Jaccard inputs")
    a = set(np.argpartition(left, -k)[-k:].tolist())
    b = set(np.argpartition(right, -k)[-k:].tolist())
    return float(len(a & b) / len(a | b)) if a | b else 1.0


def biomarker_alignment(saliency: np.ndarray, reference_prior: np.ndarray) -> dict[str, float]:
    return {
        "prior_alignment_pearson": safe_correlation(saliency, reference_prior, "pearson"),
        "prior_alignment_spearman": safe_correlation(saliency, reference_prior, "spearman"),
        "prior_alignment_top10_jaccard": topk_jaccard(saliency, reference_prior, 10),
    }


def save_coefficient_artifact(
    output_dir: Path,
    method_id: str,
    prior_type: str,
    seed: int,
    fold: int,
    beta_fc: np.ndarray,
    beta_sc: np.ndarray,
    saliency: np.ndarray,
    selected_hash: str,
) -> Path:
    directory = output_dir / "coefficients" / f"{method_id}__{prior_type}"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed{seed:02d}_fold{fold:02d}.npz"
    tmp = path.with_suffix(".npz.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(
            handle,
            beta_fc=np.asarray(beta_fc, dtype=np.float64),
            beta_sc=np.asarray(beta_sc, dtype=np.float64),
            node_saliency=np.asarray(saliency, dtype=np.float64),
            selected_hyperparameter_hash=np.asarray(selected_hash),
            feature_coordinate_system=np.asarray("standardized_edges_original_target_units"),
            edge_order=np.asarray("numpy_triu_indices_k1"),
        )
    tmp.replace(path)
    return path


def aggregate_seed_metrics(split_df: pd.DataFrame) -> pd.DataFrame:
    return (
        split_df.groupby(["model_id", "prior_type", "evaluation_type", "seed"], as_index=False)
        .agg(
            pearson=("pearson", "mean"),
            rmse=("rmse", "mean"),
            mae=("mae", "mean"),
            n_folds=("fold", "nunique"),
        )
        .sort_values(["model_id", "prior_type", "seed"])
        .reset_index(drop=True)
    )


def aggregate_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    return (
        seed_df.groupby(["model_id", "prior_type", "evaluation_type"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            pearson_mean=("pearson", "mean"), pearson_std=("pearson", "std"),
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        )
        .sort_values(["evaluation_type", "model_id", "prior_type"])
        .reset_index(drop=True)
    )


def build_paired_comparisons(seed_df: pd.DataFrame) -> pd.DataFrame:
    required = [
        ("A3 matched - A4", MODEL_A3, "matched", MODEL_A4, "none"),
        ("A3 matched - A2", MODEL_A3, "matched", MODEL_A2, "matched"),
        ("A3 matched - A3 unrelated-fixed", MODEL_A3, "matched", MODEL_A3, "unrelated"),
        ("A3 matched - A3 shuffled-fixed", MODEL_A3, "matched", MODEL_A3, "shuffled"),
        ("A3 matched - A3 random-fixed", MODEL_A3, "matched", MODEL_A3, "random"),
    ]
    rows = []
    for label, left_model, left_prior, right_model, right_prior in required:
        left = seed_df[
            (seed_df.model_id == left_model) & (seed_df.prior_type == left_prior)
        ].set_index("seed")
        right = seed_df[
            (seed_df.model_id == right_model) & (seed_df.prior_type == right_prior)
        ].set_index("seed")
        common = sorted(set(left.index) & set(right.index))
        if not common:
            continue
        row: dict[str, Any] = {"comparison": label, "n_seeds": len(common)}
        for metric in ("pearson", "rmse", "mae"):
            delta = left.loc[common, metric].to_numpy(float) - right.loc[common, metric].to_numpy(float)
            row[f"mean_delta_{metric}"] = float(np.mean(delta))
            row[f"median_delta_{metric}"] = float(np.median(delta))
            row[f"positive_{metric}"] = int(np.sum(delta > 0))
        rows.append(row)
    return pd.DataFrame(rows)


def build_biomarker_metrics(
    output_dir: Path,
    expected: Sequence[tuple[str, str]],
    seeds: Sequence[int],
    n_folds: int,
    reference_prior: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for model_id, prior_type in expected:
        directory = output_dir / "coefficients" / f"{model_id}__{prior_type}"
        for seed in seeds:
            vectors = []
            alignments = []
            for fold in range(n_folds):
                path = directory / f"seed{seed:02d}_fold{fold:02d}.npz"
                if not path.exists():
                    raise ValueError(f"Missing biomarker artifact: {path}")
                with np.load(path, allow_pickle=False) as artifact:
                    vector = artifact["node_saliency"].astype(np.float64)
                vectors.append(vector)
                alignments.append(safe_correlation(vector, reference_prior, "spearman"))
            rank_values = [safe_correlation(a, b, "spearman") for a, b in combinations(vectors, 2)]
            jaccard_values = [topk_jaccard(a, b, 10) for a, b in combinations(vectors, 2)]
            rows.append({
                "model_id": model_id, "prior_type": prior_type, "seed": int(seed),
                "n_folds": len(vectors), "n_fold_pairs": len(rank_values),
                "wm_alignment": float(np.mean(alignments)),
                "rank_stability": float(np.mean(rank_values)) if rank_values else 0.0,
                "top10_jaccard": float(np.mean(jaccard_values)) if jaccard_values else 0.0,
            })
    return pd.DataFrame(rows)


def component_diagnostic(summary: pd.DataFrame, tolerance: float = 0.005) -> dict[str, Any]:
    def mean_for(model: str, prior: str) -> float:
        row = summary[(summary.model_id == model) & (summary.prior_type == prior)]
        if len(row) != 1:
            raise ValueError(f"Missing unique summary row for {model}/{prior}")
        return float(row.pearson_mean.iloc[0])
    a4, a2, a3 = mean_for(MODEL_A4, "none"), mean_for(MODEL_A2, "matched"), mean_for(MODEL_A3, "matched")
    if a3 - a2 > tolerance and a2 - a4 > tolerance:
        interpretation = "anisotropy + Laplacian are complementary"
        ordering = "A3 > A2 > A4"
    elif a2 - a4 > tolerance and abs(a3 - a2) <= tolerance:
        interpretation = "Laplacian drives the gain; simplify method"
        ordering = "A3 ~= A2 > A4"
    elif abs(a2 - a4) <= tolerance and a3 - a4 > tolerance:
        interpretation = "interaction between anisotropy and Laplacian drives gain"
        ordering = "A2 ~= A4 and A3 > A4"
    elif a4 >= a3:
        interpretation = "prior does not improve prediction under corrected implementation"
        ordering = "A4 >= A3"
    else:
        interpretation = "mixed component ordering; no canonical simplification"
        ordering = "mixed"
    return {
        "A4_pearson": a4, "A2_pearson": a2, "A3_pearson": a3,
        "equivalence_tolerance": tolerance,
        "ordering": ordering, "interpretation": interpretation,
    }


def make_refinement_decision(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    max_rmse_degradation: float,
) -> dict[str, Any]:
    def comparison(label: str) -> pd.Series:
        row = comparisons[comparisons.comparison == label]
        if len(row) != 1:
            raise ValueError(f"Missing paired comparison: {label}")
        return row.iloc[0]
    main = comparison("A3 matched - A4")
    median_delta = float(main.median_delta_pearson)
    mean_delta = float(main.mean_delta_pearson)
    positive = int(main.positive_pearson)
    # Positive delta RMSE means A3 has larger (worse) RMSE under the table's left-right convention.
    rmse_degradation = float(main.mean_delta_rmse)
    no_material_rmse_degradation = rmse_degradation <= float(max_rmse_degradation)
    specificity = {}
    for control in CONTROL_PRIORS:
        row = comparison(f"A3 matched - A3 {control}-fixed")
        specificity[control] = {
            "mean_delta_pearson": float(row.mean_delta_pearson),
            "median_delta_pearson": float(row.median_delta_pearson),
            "matched_better_mean_or_median": bool(
                row.mean_delta_pearson > 0 or row.median_delta_pearson > 0
            ),
        }
    specificity_pass_count = sum(v["matched_better_mean_or_median"] for v in specificity.values())
    go_primary = median_delta >= 0.010 and positive >= 2
    go = go_primary and no_material_rmse_degradation and specificity_pass_count >= 2
    if go:
        recommendation = "full_10x5_msancr"
    elif 0.005 <= median_delta < 0.010 and positive >= 2:
        recommendation = "one_final_small_refinement"
    else:
        recommendation = "ct_mac_prior_rebuild"
    return {
        "recommended_next_step": recommendation,
        "inference_status": "descriptive_only_n_equals_3_no_significance_claim",
        "gate": {
            "median_delta_pearson_vs_A4": median_delta,
            "mean_delta_pearson_vs_A4": mean_delta,
            "positive_seeds_vs_A4": positive,
            "median_threshold_go": 0.010,
            "positive_seed_threshold_go": 2,
            "preferred_mean_threshold": 0.010,
            "preferred_mean_threshold_met": bool(mean_delta >= 0.010),
            "mean_delta_rmse_A3_minus_A4": rmse_degradation,
            "max_material_rmse_degradation": float(max_rmse_degradation),
            "no_material_rmse_degradation": no_material_rmse_degradation,
            "specificity_pass_count": int(specificity_pass_count),
            "specificity_required": 2,
            "go_primary_prediction_gate": bool(go_primary),
            "go_all_requirements": bool(go),
        },
        "fixed_prior_swap_specificity": specificity,
        "component_contribution": component_diagnostic(summary),
    }


def _read_records(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict("records") if path.exists() and path.stat().st_size else []


def _artifact_exists(output_dir: Path, model_id: str, prior_type: str, seed: int, fold: int) -> bool:
    return (
        output_dir / "coefficients" / f"{model_id}__{prior_type}"
        / f"seed{seed:02d}_fold{fold:02d}.npz"
    ).exists()


def _outer_split_complete(
    output_dir: Path,
    base_rows: Sequence[Mapping[str, Any]],
    swap_rows: Sequence[Mapping[str, Any]],
    selected_rows: Sequence[Mapping[str, Any]],
    seed: int,
    fold: int,
) -> bool:
    base_expected = {
        (MODEL_A4, "none"), (MODEL_A4_ISO, "uniform"),
        (MODEL_A2, "matched"), (MODEL_A3, "matched"),
    }
    swap_expected = {(MODEL_A3, name) for name in CONTROL_PRIORS}
    base_found = {
        (str(row["model_id"]), str(row["prior_type"]))
        for row in base_rows if int(row["seed"]) == seed and int(row["fold"]) == fold
    }
    swap_found = {
        (str(row["model_id"]), str(row["prior_type"]))
        for row in swap_rows if int(row["seed"]) == seed and int(row["fold"]) == fold
    }
    selected_found = {
        str(row["model_id"])
        for row in selected_rows if int(row["seed"]) == seed and int(row["fold"]) == fold
    }
    artifacts = all(
        _artifact_exists(output_dir, model, prior, seed, fold)
        for model, prior in (base_expected | swap_expected)
    )
    return (
        base_expected.issubset(base_found)
        and swap_expected.issubset(swap_found)
        and set(BASE_MODELS).issubset(selected_found)
        and artifacts
    )


def _model_prior_for_final(model_id: str) -> str:
    if model_id in (MODEL_A4, MODEL_A4_ISO, MODEL_A2, MODEL_A3):
        return "matched"
    raise ValueError(f"Unknown base model: {model_id}")


def _display_prior(model_id: str) -> str:
    if model_id == MODEL_A4:
        return "none"
    if model_id == MODEL_A4_ISO:
        return "uniform"
    return "matched"


def _make_figures(
    seed_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    biomarker_df: pd.DataFrame,
    figure_dir: Path,
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        MODEL_A4: "A4", MODEL_A4_ISO: "A4-iso", MODEL_A2: "A2", MODEL_A3: "A3",
    }
    base = seed_df[seed_df.evaluation_type == "retuned_base"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    order = [MODEL_A4, MODEL_A4_ISO, MODEL_A2, MODEL_A3]
    means = [base[base.model_id == model].pearson.mean() for model in order]
    stds = [float(np.nan_to_num(base[base.model_id == model].pearson.std())) for model in order]
    ax.bar(range(len(order)), means, yerr=stds, capsize=4)
    ax.set_xticks(range(len(order)), [labels[m] for m in order])
    ax.set_ylabel("Seed-mean Pearson r")
    ax.set_title("Working Memory: corrected MS-A-NCR refinement")
    fig.tight_layout()
    fig.savefig(figure_dir / "wm_model_comparison.pdf")
    plt.close(fig)

    pivot = base.pivot(index="seed", columns="model_id", values="pearson")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    delta = pivot[MODEL_A3] - pivot[MODEL_A4]
    ax.bar(delta.index.astype(str), delta.values)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axhline(0.010, color="tab:green", linestyle="--", linewidth=1)
    ax.set_xlabel("Seed")
    ax.set_ylabel("A3 matched − A4 Pearson")
    fig.tight_layout()
    fig.savefig(figure_dir / "wm_seed_delta_vs_A4.pdf")
    plt.close(fig)

    a3 = seed_df[seed_df.model_id == MODEL_A3].copy()
    prior_order = ["matched", *CONTROL_PRIORS]
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [a3[a3.prior_type == prior].pearson.mean() for prior in prior_order]
    errors = [float(np.nan_to_num(a3[a3.prior_type == prior].pearson.std())) for prior in prior_order]
    ax.bar(range(4), values, yerr=errors, capsize=4)
    ax.set_xticks(range(4), prior_order, rotation=15)
    ax.set_ylabel("Seed-mean Pearson r")
    ax.set_title("Matched-selected A3 with fixed prior swaps")
    fig.tight_layout()
    fig.savefig(figure_dir / "wm_matched_vs_prior_swaps.pdf")
    plt.close(fig)

    a3_selected = selected_df[selected_df.model_id == MODEL_A3]
    fig, ax = plt.subplots(figsize=(7, 5))
    for lifting, group in a3_selected.groupby("lifting"):
        ax.scatter(group.gamma, group.lambda_l, label=lifting, s=45)
    ax.set_xlabel("Selected gamma")
    ax.set_ylabel("Selected lambda_L")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "wm_selected_gamma_lambdaL.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    bio_labels, rank, jacc = [], [], []
    for (model, prior), group in biomarker_df.groupby(["model_id", "prior_type"]):
        bio_labels.append(f"{labels.get(model, model)}\n{prior}")
        rank.append(group.rank_stability.mean())
        jacc.append(group.top10_jaccard.mean())
    axes[0].bar(range(len(rank)), rank)
    axes[1].bar(range(len(jacc)), jacc)
    for ax, values, title in zip(axes, (rank, jacc), ("Rank stability", "Top-10 Jaccard")):
        ax.set_xticks(range(len(values)), bio_labels, rotation=35, ha="right")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(figure_dir / "wm_biomarker_stability.pdf")
    plt.close(fig)


def finalize_refinement_outputs(
    output_dir: Path,
    figure_dir: Path,
    cfg: Mapping[str, Any],
    config_hash: str,
    elapsed_seconds: float,
    groups_available: bool,
    completed_full_grid: bool,
    inference_status: str = "descriptive_only_n_equals_3_no_significance_claim",
) -> dict[str, Any]:
    split_df = pd.DataFrame(_read_records(output_dir / "split_metrics.csv"))
    swap_df = pd.DataFrame(_read_records(output_dir / "prior_swap_split_metrics.csv"))
    selected_df = pd.DataFrame(_read_records(output_dir / "selected_hyperparameters.csv"))
    if split_df.empty or swap_df.empty:
        raise ValueError("Cannot finalize without base and prior-swap split metrics")
    combined = pd.concat([split_df, swap_df], ignore_index=True)
    seed_df = aggregate_seed_metrics(combined)
    summary = aggregate_summary(seed_df)
    swap_seed = seed_df[seed_df.evaluation_type == "fixed_prior_swap"].reset_index(drop=True)
    swap_summary = summary[summary.evaluation_type == "fixed_prior_swap"].reset_index(drop=True)
    comparisons = build_paired_comparisons(seed_df)
    expected_biomarkers = [
        (MODEL_A4, "none"), (MODEL_A4_ISO, "uniform"),
        (MODEL_A2, "matched"), (MODEL_A3, "matched"),
        *[(MODEL_A3, control) for control in CONTROL_PRIORS],
    ]
    biomarker = build_biomarker_metrics(
        output_dir, expected_biomarkers,
        [int(v) for v in cfg["seeds"]] if completed_full_grid else sorted(combined.seed.unique()),
        int(cfg["n_outer_folds"]) if completed_full_grid else int(combined.fold.nunique()),
        load_roi_prior(cfg["priors"]["working_memory"]["matched"], int(cfg["n_rois"])),
    )

    atomic_write_csv(seed_df, output_dir / "seed_metrics.csv")
    atomic_write_csv(summary, output_dir / "summary_metrics.csv")
    atomic_write_csv(swap_seed, output_dir / "prior_swap_seed_metrics.csv")
    atomic_write_csv(swap_summary, output_dir / "prior_swap_summary.csv")
    atomic_write_csv(biomarker, output_dir / "biomarker_metrics.csv")
    atomic_write_csv(comparisons, output_dir / "paired_comparisons.csv")

    decision: dict[str, Any] = {
        "status": "complete" if completed_full_grid else "partial_smoke",
        "target": "working_memory",
        "inference_status": inference_status,
    }
    if completed_full_grid:
        decision.update(make_refinement_decision(
            summary, comparisons, float(cfg["max_material_rmse_degradation"])
        ))
        decision["inference_status"] = inference_status
    else:
        decision["recommended_next_step"] = "complete_full_3_seed_refinement_before_decision"
        decision["inference_status"] = "smoke_only"
    atomic_write_json(decision, output_dir / "refinement_decision.json")
    metadata = {
        "experiment_name": cfg["experiment_name"],
        "config_hash": config_hash,
        "target": "working_memory",
        "seeds_configured": cfg["seeds"],
        "n_outer_folds": int(cfg["n_outer_folds"]),
        "n_inner_folds": int(cfg["n_inner_folds"]),
        "selection_metric": "mean_inner_validation_pearson",
        "tie_breakers": ["mean_rmse", "mean_mae", "simpler_configuration"],
        "pearson_tie_tolerance": PEARSON_TIE_TOLERANCE,
        "group_aware": bool(groups_available),
        "group_limitation": None if groups_available else "No family IDs available; subject-wise CV retained.",
        "feature_count_penalty_scale": "2*n_edges",
        "full_fc_diagonal_penalty": True,
        "fixed_prior_swap_controls": list(CONTROL_PRIORS),
        "retuned_controls_run": False,
        "elapsed_seconds": float(elapsed_seconds),
        "completed_full_grid": bool(completed_full_grid),
    }
    atomic_write_json(metadata, output_dir / "run_metadata.json")
    _make_figures(seed_df, selected_df, biomarker, figure_dir)
    if completed_full_grid:
        (output_dir / "COMPLETE").write_text("done\n", encoding="utf-8")
    return decision


def run_refinement(
    cfg: Mapping[str, Any],
    seeds_override: Optional[Sequence[int]] = None,
    folds_override: Optional[Sequence[int]] = None,
    output_dir_override: Optional[str | Path] = None,
    figure_dir_override: Optional[str | Path] = None,
    overwrite: bool = False,
    enforce_seed_gate: bool = True,
    inference_status: str = "descriptive_only_n_equals_3_no_significance_claim",
) -> dict[str, Any]:
    validate_refinement_config(cfg, enforce_seed_gate=enforce_seed_gate)
    started = time.time()
    output_dir = Path(output_dir_override or cfg["output_dir"])
    figure_dir = Path(figure_dir_override or cfg["figures_dir"])
    audit_source = output_dir / "audit_before_changes.json"
    audit_text = audit_source.read_text(encoding="utf-8") if audit_source.exists() else None
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if audit_text is not None and not audit_source.exists():
        audit_source.write_text(audit_text, encoding="utf-8")
    (output_dir / "coefficients").mkdir(parents=True, exist_ok=True)

    cfg_hash = stable_hash(cfg)
    metadata_path = output_dir / "run_metadata.json"
    if metadata_path.exists():
        prior_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if prior_metadata.get("config_hash") != cfg_hash:
            raise ValueError("Refusing to resume outputs produced by a different config hash")
    atomic_write_json({
        "experiment_name": cfg["experiment_name"],
        "config_hash": cfg_hash,
        "status": "running",
        "target": "working_memory",
        "seeds_configured": cfg["seeds"],
        "n_outer_folds": int(cfg["n_outer_folds"]),
        "n_inner_folds": int(cfg["n_inner_folds"]),
    }, metadata_path)

    fc, sc, _, subject_ids, groups = load_connectomes(cfg["data"])
    n_rois = int(fc.shape[1])
    if n_rois != int(cfg["n_rois"]):
        raise ValueError(f"Expected {cfg['n_rois']} ROIs, found {n_rois}")
    target_info = cfg["targets"]["working_memory"]
    y = np.load(target_info["label_path"], allow_pickle=False).astype(np.float64).reshape(-1)
    if len(y) != len(fc) or len(subject_ids) != len(y):
        raise ValueError("Working-Memory labels/subject manifest do not align with connectomes")
    X_fc, X_sc = upper_triangle_features(fc), upper_triangle_features(sc)

    priors = {
        name: load_roi_prior(cfg["priors"]["working_memory"][name], n_rois)
        for name in REQUIRED_PRIORS
    }
    cache_factory = CacheFactory(
        priors, n_rois, int(cfg["top_k"]), float(cfg["diagonal_epsilon"]),
        str(cfg["laplacian_weighting"]), str(cfg["laplacian_normalization"]),
    )
    selected_seeds = [int(v) for v in (seeds_override or cfg["seeds"])]
    selected_folds = set(int(v) for v in folds_override) if folds_override is not None else None

    paths = {
        "base": output_dir / "split_metrics.csv",
        "swap": output_dir / "prior_swap_split_metrics.csv",
        "inner": output_dir / "inner_cv_metrics.csv",
        "selected": output_dir / "selected_hyperparameters.csv",
        "boundary": output_dir / "boundary_selection_report.csv",
    }
    records = {key: _read_records(path) for key, path in paths.items()}
    for seed, fold, trainval_idx, test_idx in iter_refinement_outer_splits(
        y, selected_seeds, int(cfg["n_outer_folds"]), groups,
        float(cfg.get("historical_val_fraction", 0.15)),
    ):
        if selected_folds is not None and fold not in selected_folds:
            continue
        if _outer_split_complete(
            output_dir, records["base"], records["swap"], records["selected"], seed, fold
        ):
            print(f"SKIP complete seed={seed} fold={fold}", flush=True)
            continue
        split_started = time.time()
        inner_splits = make_inner_cv_splits(
            trainval_idx, y, seed, fold, int(cfg["n_inner_folds"]), groups
        )
        inner_folds = prepare_inner_folds(X_fc, X_sc, y, inner_splits)
        selected, inner_df, boundary = tune_outer_split(
            inner_folds, cache_factory, cfg, seed, fold
        )

        base_new, swap_new, selected_new = [], [], []
        for model_id in BASE_MODELS:
            chosen = selected[model_id]
            prior_for_cache = _model_prior_for_final(model_id)
            _, cache = cache_factory.get(
                prior_for_cache, float(chosen["gamma"]), str(chosen["lifting"])
            )
            pred, beta_fc, beta_sc, saliency = fit_final_model(
                X_fc, X_sc, y, trainval_idx, test_idx, cache, chosen
            )
            display_prior = _display_prior(model_id)
            hp = hyperparameter_payload(chosen)
            hp_hash = stable_hash(hp)
            artifact = save_coefficient_artifact(
                output_dir, model_id, display_prior, seed, fold,
                beta_fc, beta_sc, saliency, hp_hash,
            )
            base_new.append({
                "target": "working_memory", "seed": seed, "fold": fold,
                "model_id": model_id, "prior_type": display_prior,
                "evaluation_type": "retuned_base", "n_train": len(trainval_idx),
                "n_test": len(test_idx), "test_indices_hash": stable_hash(test_idx.tolist()),
                "selected_hyperparameter_hash": hp_hash,
                "coefficient_artifact": str(artifact),
                **biomarker_alignment(saliency, priors["matched"]),
                **prediction_metrics(y[test_idx], pred),
            })
            selected_new.append({
                "target": "working_memory", "seed": seed, "fold": fold,
                "model_id": model_id, "prior_type": display_prior,
                **hp,
                "selected_hyperparameter_hash": hp_hash,
                "mean_inner_pearson": float(chosen["mean_pearson"]),
                "mean_inner_rmse": float(chosen["mean_rmse"]),
                "mean_inner_mae": float(chosen["mean_mae"]),
                "selection_metric": "pearson",
                "pearson_tie_tolerance": float(chosen["pearson_tolerance"]),
            })

        matched_hp = hyperparameter_payload(selected[MODEL_A3])
        matched_hash = stable_hash(matched_hp)
        swap_configs = make_fixed_prior_swap_configs(selected[MODEL_A3])
        for control, swap_hp in swap_configs.items():
            _, cache = cache_factory.get(
                control, swap_hp["gamma"], swap_hp["lifting"]
            )
            pred, beta_fc, beta_sc, saliency = fit_final_model(
                X_fc, X_sc, y, trainval_idx, test_idx, cache, swap_hp
            )
            artifact = save_coefficient_artifact(
                output_dir, MODEL_A3, control, seed, fold,
                beta_fc, beta_sc, saliency, matched_hash,
            )
            swap_new.append({
                "target": "working_memory", "seed": seed, "fold": fold,
                "model_id": MODEL_A3, "prior_type": control,
                "evaluation_type": "fixed_prior_swap", "n_train": len(trainval_idx),
                "n_test": len(test_idx), "test_indices_hash": stable_hash(test_idx.tolist()),
                "selected_from_prior": "matched",
                "selected_hyperparameter_hash": matched_hash,
                "coefficient_artifact": str(artifact),
                **swap_hp,
                **biomarker_alignment(saliency, priors["matched"]),
                **prediction_metrics(y[test_idx], pred),
            })

        records["base"] = [
            row for row in records["base"]
            if not (int(row["seed"]) == seed and int(row["fold"]) == fold)
        ] + base_new
        records["swap"] = [
            row for row in records["swap"]
            if not (int(row["seed"]) == seed and int(row["fold"]) == fold)
        ] + swap_new
        records["selected"] = [
            row for row in records["selected"]
            if not (int(row["seed"]) == seed and int(row["fold"]) == fold)
        ] + selected_new
        records["inner"] = [
            row for row in records["inner"]
            if not (int(row["seed"]) == seed and int(row["outer_fold"]) == fold)
        ] + inner_df.to_dict("records")
        records["boundary"] = [
            row for row in records["boundary"]
            if not (int(row["seed"]) == seed and int(row["fold"]) == fold)
        ] + [boundary]
        for key, path in paths.items():
            frame = pd.DataFrame(records[key])
            sort_cols = [column for column in ("seed", "fold", "outer_fold", "model_id", "prior_type", "stage", "candidate_id", "inner_fold") if column in frame]
            if sort_cols:
                frame = frame.sort_values(sort_cols)
            atomic_write_csv(frame, path)
        print(
            f"DONE seed={seed} fold={fold} runtime={time.time()-split_started:.1f}s "
            f"A4={base_new[0]['pearson']:.4f} A2={base_new[2]['pearson']:.4f} "
            f"A3={base_new[3]['pearson']:.4f}",
            flush=True,
        )

    base_df = pd.DataFrame(records["base"])
    swap_df = pd.DataFrame(records["swap"])
    configured = {(seed, fold) for seed in cfg["seeds"] for fold in range(int(cfg["n_outer_folds"]))}
    completed_pairs = {
        (int(row["seed"]), int(row["fold"]))
        for row in records["base"]
        if _outer_split_complete(
            output_dir, records["base"], records["swap"], records["selected"],
            int(row["seed"]), int(row["fold"]),
        )
    }
    full = configured.issubset(completed_pairs)
    if full:
        n_cfg_seeds = len(cfg["seeds"])
        n_outer = int(cfg["n_outer_folds"])
        expected_base = n_cfg_seeds * n_outer * len(BASE_MODELS)
        expected_swap = n_cfg_seeds * n_outer * len(CONTROL_PRIORS)
        if (
            len(base_df[base_df.seed.isin(cfg["seeds"])]) != expected_base
            or len(swap_df[swap_df.seed.isin(cfg["seeds"])]) != expected_swap
        ):
            raise RuntimeError("Full-grid cardinality check failed")
    return finalize_refinement_outputs(
        output_dir, figure_dir, cfg, cfg_hash, time.time() - started,
        groups is not None, full,
        inference_status=inference_status,
    )