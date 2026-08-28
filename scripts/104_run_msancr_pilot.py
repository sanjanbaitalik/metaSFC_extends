#!/usr/bin/env python3
"""MS-A-NCR Pilot: Modality-Selective Anisotropic Network-Constrained Ridge.

3-seed pilot (seeds 0, 1, 2; 5 folds) evaluating modality-selective
anisotropic NCR for Fluid Intelligence and Working Memory.

Usage:
    python scripts/104_run_msancr_pilot.py --config configs/iclr/msancr_pilot.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from metascfc.benchmark_utils import iter_nested_splits, load_connectomes, prediction_metrics, save_json
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    ModalitySelectiveAnisotropicNCR,
    _MSANCRCache,
    build_msancr_cache,
    fit_predict_msancr,
    lift_roi_to_edge,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def upper_triangle_features(mats: np.ndarray) -> Tuple[np.ndarray, int]:
    iu = np.triu_indices(mats.shape[1], k=1)
    return mats[:, iu[0], iu[1]].astype(np.float64), len(iu[0])


def load_roi_prior(path: str, n_rois: int) -> np.ndarray:
    df = pd.read_csv(path)
    if "roi_index" in df.columns:
        df = df.sort_values("roi_index")
    p = df["prior_score"].to_numpy(np.float64)
    if p.shape != (n_rois,):
        raise ValueError(f"Prior {path}: shape {p.shape} != ({n_rois},)")
    return p


def config_hash(cfg: Dict) -> str:
    s = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ── model definitions ────────────────────────────────────────────────────────

MODEL_A0 = "A0_ridge"
MODEL_A1 = "A1_aniso_ridge"
MODEL_A2 = "A2_fc_laplacian"
MODEL_A3 = "A3_msancr"
MODEL_A4 = "A4_modality_ridge"

ALL_MODELS = [MODEL_A0, MODEL_A1, MODEL_A2, MODEL_A3, MODEL_A4]


# ── pre-build caches ─────────────────────────────────────────────────────────

def build_all_caches(
    roi_priors: Dict[str, Dict[str, np.ndarray]],
    n_rois: int,
    lifting_rules: List[str],
    gamma_grid: List[float],
    top_k: int,
    epsilon: float,
    n_edges: int,
) -> Dict[str, Dict[str, _MSANCRCache]]:
    """Pre-build all (prior_type x gamma x lifting) caches.

    Returns nested dict: caches[target_key][prior_type][gamma][lifting] = cache.
    """
    all_caches: Dict[str, Dict[str, Dict[float, Dict[str, _MSANCRCache]]]] = {}
    total_builds = 0
    t0 = time.time()

    for tgt, priors in roi_priors.items():
        all_caches[tgt] = {}
        for pt_name, roi_prior in priors.items():
            all_caches[tgt][pt_name] = {}
            for gamma in gamma_grid:
                all_caches[tgt][pt_name][gamma] = {}
                for lifting in lifting_rules:
                    cache = build_msancr_cache(
                        roi_prior, n_rois, gamma, lifting, top_k, epsilon,
                    )
                    all_caches[tgt][pt_name][gamma][lifting] = cache
                    total_builds += 1

    elapsed = time.time() - t0
    print(f"Built {total_builds} caches in {elapsed:.0f}s", flush=True)
    return all_caches


# ── model fitting with pre-built caches ──────────────────────────────────────

def _solve_kernel_cached(
    X_fc: np.ndarray, X_sc: np.ndarray, y: np.ndarray,
    cache: _MSANCRCache,
    lambda_fc: float, lambda_sc: float, lambda_l: float,
) -> np.ndarray:
    """Solve MS-A-NCR using pre-built cache. Returns dual coefficients."""
    from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
        _solve_msancr_kernel,
    )
    alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, lambda_fc, lambda_sc, lambda_l)
    return alpha


def _predict_cached(
    X_fc_new, X_sc_new, X_fc_train, X_sc_train, alpha, cache,
    lambda_fc, lambda_sc, lambda_l,
) -> np.ndarray:
    from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
        _predict_msancr,
    )
    return _predict_msancr(X_fc_new, X_sc_new, X_fc_train, X_sc_train, alpha, cache,
                           lambda_fc, lambda_sc, lambda_l)


def fit_ridge_a0(
    x_fc, x_sc, y, tr, va, te, lfc_grid, lsc_grid,
) -> Tuple[np.ndarray, Dict]:
    """A0: Standard FC+SC Ridge (lambda_fc = lambda_sc)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.concatenate([x_fc[tr], x_sc[tr]], axis=1))
    X_val = scaler.transform(np.concatenate([x_fc[va], x_sc[va]], axis=1))
    X_test = scaler.transform(np.concatenate([x_fc[te], x_sc[te]], axis=1))

    y_mean = float(y[tr].mean())
    y_std = max(float(y[tr].std()), 1e-8)
    y_z = (y[tr] - y_mean) / y_std

    # Use only lambda_fc grid (same for both modalities)
    best_rmse, best_alpha = float("inf"), lfc_grid[0]
    for alpha_val in lfc_grid:
        model = Ridge(alpha=alpha_val, fit_intercept=False)
        model.fit(X_train, y_z)
        pred_val = model.predict(X_val) * y_std + y_mean
        rmse = float(np.sqrt(np.mean((y[va] - pred_val) ** 2)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha_val

    model = Ridge(alpha=best_alpha, fit_intercept=False)
    model.fit(X_train, y_z)
    pred_test = model.predict(X_test) * y_std + y_mean

    return pred_test, {
        "selected_lambda_fc": best_alpha, "selected_lambda_sc": best_alpha,
        "selected_lambda_l": 0.0, "selected_gamma": 0.0, "selected_lifting": "none",
    }


def fit_ridge_a4(
    x_fc, x_sc, y, tr, va, te, lfc_grid, lsc_grid,
) -> Tuple[np.ndarray, Dict]:
    """A4: Modality-specific Ridge (no prior, lambda_fc != lambda_sc)."""
    from sklearn.preprocessing import StandardScaler

    best_rmse, best_lfc, best_lsc = float("inf"), lfc_grid[0], lsc_grid[0]
    n_edges = x_fc.shape[1]
    scale = float(max(1, n_edges * 2))

    scaler_fc = StandardScaler()
    scaler_sc = StandardScaler()
    X_fc_tr = scaler_fc.fit_transform(x_fc[tr])
    X_sc_tr = scaler_sc.fit_transform(x_sc[tr])
    X_fc_va = scaler_fc.transform(x_fc[va])
    X_sc_va = scaler_sc.transform(x_sc[va])

    y_mean = float(y[tr].mean())
    y_std = max(float(y[tr].std()), 1e-8)
    y_z = (y[tr] - y_mean) / y_std
    n = len(tr)

    for lfc in lfc_grid:
        for lsc in lsc_grid:
            K = (1.0 / lfc) * (X_fc_tr @ X_fc_tr.T) + (1.0 / lsc) * (X_sc_tr @ X_sc_tr.T)
            K = K / scale
            alpha = np.linalg.solve(K + np.eye(n), y_z)

            pred_val = (
                (1.0 / lfc) * (X_fc_va @ X_fc_tr.T) @ alpha +
                (1.0 / lsc) * (X_sc_va @ X_sc_tr.T) @ alpha
            ) / scale * y_std + y_mean
            rmse = float(np.sqrt(np.mean((y[va] - pred_val) ** 2)))
            if rmse < best_rmse:
                best_rmse, best_lfc, best_lsc = rmse, lfc, lsc

    # Refit on train+val
    fit_idx = np.concatenate([tr, va])
    X_fc_fit = scaler_fc.fit_transform(x_fc[fit_idx])
    X_sc_fit = scaler_sc.fit_transform(x_sc[fit_idx])
    X_fc_te = scaler_fc.transform(x_fc[te])
    X_sc_te = scaler_sc.transform(x_sc[te])

    fit_mean = float(y[fit_idx].mean())
    fit_std = max(float(y[fit_idx].std()), 1e-8)
    y_fit_z = (y[fit_idx] - fit_mean) / fit_std
    n_fit = len(fit_idx)

    K = (1.0 / best_lfc) * (X_fc_fit @ X_fc_fit.T) + (1.0 / best_lsc) * (X_sc_fit @ X_sc_fit.T)
    K = K / scale
    alpha = np.linalg.solve(K + np.eye(n_fit), y_fit_z)
    pred_test = (
        (1.0 / best_lfc) * (X_fc_te @ X_fc_fit.T) @ alpha +
        (1.0 / best_lsc) * (X_sc_te @ X_sc_fit.T) @ alpha
    ) / scale * fit_std + fit_mean

    return pred_test, {
        "selected_lambda_fc": best_lfc, "selected_lambda_sc": best_lsc,
        "selected_lambda_l": 0.0, "selected_gamma": 0.0, "selected_lifting": "none",
    }


def fit_prior_model_cached(
    x_fc, x_sc, y, tr, va, te,
    cache_dict: Dict[float, Dict[str, _MSANCRCache]],
    model_id: str,
    lambda_fc_grid: Sequence[float],
    lambda_sc_grid: Sequence[float],
    lambda_l_grid: Sequence[float],
    n_edges: int,
) -> Tuple[np.ndarray, Dict]:
    """Fit A1/A2/A3 with staged selection using pre-built caches."""
    from sklearn.preprocessing import StandardScaler
    from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
        _solve_msancr_kernel,
    )

    # Determine which (gamma, lambda_l) combos to search
    if model_id == MODEL_A1:
        gammas = [g for g in cache_dict if g > 0]
        ll_grid = [0.0]
    elif model_id == MODEL_A2:
        gammas = [0.0]
        ll_grid = [l for l in lambda_l_grid if l > 0]
    else:  # A3
        gammas = [g for g in cache_dict if g > 0]
        ll_grid = [l for l in lambda_l_grid if l > 0]

    best_rmse = float("inf")
    best_info: Dict[str, Any] = {}

    for gamma in gammas:
        liftings = list(cache_dict[gamma].keys())
        for lifting in liftings:
            cache = cache_dict[gamma][lifting]
            for lambda_l in ll_grid:
                eig_cache: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}

                scaler_fc = StandardScaler()
                scaler_sc = StandardScaler()
                X_fc_tr = scaler_fc.fit_transform(x_fc[tr])
                X_sc_tr = scaler_sc.fit_transform(x_sc[tr])
                X_fc_va = scaler_fc.transform(x_fc[va])
                X_sc_va = scaler_sc.transform(x_sc[va])

                y_mean = float(y[tr].mean())
                y_std = max(float(y[tr].std()), 1e-8)
                y_z = (y[tr] - y_mean) / y_std

                for lfc in lambda_fc_grid:
                    for lsc in lambda_sc_grid:
                        try:
                            alpha, _ = _solve_msancr_kernel(
                                X_fc_tr, X_sc_tr, y_z, cache, lfc, lsc, lambda_l,
                                eig_cache=eig_cache,
                            )
                            pred_z = _predict_cached(
                                X_fc_va, X_sc_va, X_fc_tr, X_sc_tr,
                                alpha, cache, lfc, lsc, lambda_l,
                            )
                            pred = pred_z * y_std + y_mean
                            rmse = float(np.sqrt(np.mean((y[va] - pred) ** 2)))
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_info = {
                                    "selected_lambda_fc": lfc,
                                    "selected_lambda_sc": lsc,
                                    "selected_lambda_l": lambda_l,
                                    "selected_gamma": gamma,
                                    "selected_lifting": lifting,
                                }
                        except Exception:
                            continue

    # Refit on train+val with best config
    if not best_info:
        return np.zeros(len(te)), {
            "selected_lambda_fc": 1.0, "selected_lambda_sc": 1.0,
            "selected_lambda_l": 0.0, "selected_gamma": 0.0, "selected_lifting": "prod",
        }

    cache = cache_dict[best_info["selected_gamma"]][best_info["selected_lifting"]]
    fit_idx = np.concatenate([tr, va])

    scaler_fc = StandardScaler()
    scaler_sc = StandardScaler()
    X_fc_fit = scaler_fc.fit_transform(x_fc[fit_idx])
    X_sc_fit = scaler_sc.fit_transform(x_sc[fit_idx])
    X_fc_te = scaler_fc.transform(x_fc[te])
    X_sc_te = scaler_sc.transform(x_sc[te])

    fit_mean = float(y[fit_idx].mean())
    fit_std = max(float(y[fit_idx].std()), 1e-8)
    y_fit_z = (y[fit_idx] - fit_mean) / fit_std

    alpha, _ = _solve_msancr_kernel(
        X_fc_fit, X_sc_fit, y_fit_z, cache,
        best_info["selected_lambda_fc"], best_info["selected_lambda_sc"],
        best_info["selected_lambda_l"],
    )
    pred_z = _predict_cached(
        X_fc_te, X_sc_te, X_fc_fit, X_sc_fit, alpha, cache,
        best_info["selected_lambda_fc"], best_info["selected_lambda_sc"],
        best_info["selected_lambda_l"],
    )
    pred = pred_z * fit_std + fit_mean

    return pred, best_info


# ── main pilot ───────────────────────────────────────────────────────────────

def run_pilot(config: Dict, output_dir_override: Optional[str] = None) -> None:
    t0 = time.time()
    out_dir = Path(output_dir_override or config["output_dir"])
    fig_dir = Path(config.get("figures_dir", out_dir.parent / "msancr_pilot_figs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg_hash = config_hash(config)
    seeds = config["seeds"]
    n_folds = config["n_folds"]
    val_frac = config["val_fraction"]
    lifting_rules = config["lifting_rules"]
    top_k = int(config["top_k"])
    gamma_grid = [float(g) for g in config["gamma_grid"]]
    lambda_l_grid = [float(l) for l in config["lambda_laplacian_grid"]]
    lambda_fc_grid = [float(l) for l in config["lambda_fc_grid"]]
    lambda_sc_grid = [float(l) for l in config["lambda_sc_grid"]]
    epsilon = float(config.get("diagonal_epsilon", 1e-3))
    pilot_prior_types = config.get("pilot_prior_types", None)

    fc, sc, _, _, groups = load_connectomes(config["data"])
    n_rois = int(fc.shape[1])
    x_fc, n_edges = upper_triangle_features(fc)
    x_sc, _ = upper_triangle_features(sc)

    # Load all priors
    roi_priors: Dict[str, Dict[str, np.ndarray]] = {}
    for target_key, target_info in config["targets"].items():
        tp = config["priors"][target_key]
        roi_priors[target_key] = {}
        for pt in ["matched", "unrelated", "shuffled", "random"]:
            if tp.get(pt):
                if pilot_prior_types is not None and pt not in pilot_prior_types:
                    continue
                roi_priors[target_key][pt] = load_roi_prior(tp[pt], n_rois)

    # Pre-build ALL caches (one-time cost)
    print("Pre-building caches...", flush=True)
    all_caches = build_all_caches(
        roi_priors, n_rois, lifting_rules, gamma_grid, top_k, epsilon, n_edges,
    )

    all_split_rows: List[Dict] = []
    all_hyper_rows: List[Dict] = []

    total_evals = len(seeds) * n_folds * len(ALL_MODELS) * sum(len(p) for p in roi_priors.values())
    eval_count = 0

    for target_key, target_info in config["targets"].items():
        y_full = np.load(target_info["label_path"]).astype(np.float64).reshape(-1)
        if len(y_full) != len(fc):
            print(f"SKIP {target_key}: labels mismatch")
            continue

        for seed in seeds:
            for fold, (_, _seed, tr, va, te) in enumerate(
                iter_nested_splits(y_full, [seed], n_folds, val_frac, groups)
            ):
                for model_id in ALL_MODELS:
                    for pt_name in roi_priors[target_key]:
                        eval_count += 1
                        t_eval = time.time()

                        if model_id == MODEL_A0:
                            pred, info = fit_ridge_a0(
                                x_fc, x_sc, y_full, tr, va, te,
                                lambda_fc_grid, lambda_sc_grid,
                            )
                        elif model_id == MODEL_A4:
                            pred, info = fit_ridge_a4(
                                x_fc, x_sc, y_full, tr, va, te,
                                lambda_fc_grid, lambda_sc_grid,
                            )
                        else:
                            cache_dict = all_caches[target_key][pt_name]
                            pred, info = fit_prior_model_cached(
                                x_fc, x_sc, y_full, tr, va, te,
                                cache_dict, model_id,
                                lambda_fc_grid, lambda_sc_grid, lambda_l_grid,
                                n_edges,
                            )

                        metrics = prediction_metrics(y_full[te], pred)
                        all_split_rows.append({
                            "target": target_key, "seed": seed, "fold": fold,
                            "model_id": model_id, "prior_type": pt_name,
                            "pearson": metrics["pearson"],
                            "rmse": metrics["rmse"],
                            "mae": metrics["mae"],
                            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                            **info,
                        })
                        all_hyper_rows.append({
                            "target": target_key, "seed": seed, "fold": fold,
                            "model_id": model_id, "prior_type": pt_name,
                            **info,
                        })

                        elapsed = time.time() - t_eval
                        if eval_count % 20 == 0 or elapsed > 60:
                            print(f"  [{time.time()-t0:.0f}s] eval {eval_count}/{total_evals} "
                                  f"{target_key} seed={seed} fold={fold} {model_id}/{pt_name} "
                                  f"pearson={metrics['pearson']:.4f} ({elapsed:.1f}s)", flush=True)

    elapsed_total = time.time() - t0
    print(f"\nMain loop: {elapsed_total:.0f}s ({len(all_split_rows)} evaluations)", flush=True)

    # ── Save raw CSVs ──────────────────────────────────────────────────────
    split_df = pd.DataFrame(all_split_rows)
    split_df.to_csv(out_dir / "split_metrics.csv", index=False)
    pd.DataFrame(all_hyper_rows).to_csv(out_dir / "selected_hyperparameters.csv", index=False)

    # ── Seed-level aggregation ─────────────────────────────────────────────
    metric_cols = ["pearson", "rmse", "mae"]
    seed_agg = (
        split_df.groupby(["target", "model_id", "prior_type", "seed"])
        .agg({c: "mean" for c in metric_cols})
        .reset_index()
    )
    seed_agg.to_csv(out_dir / "seed_metrics.csv", index=False)

    # ── Summary across seeds ──────────────────────────────────────────────
    summary_rows = []
    for (tgt, mid, pt), grp in seed_agg.groupby(["target", "model_id", "prior_type"]):
        row = {
            "target": tgt, "model_id": mid, "prior_type": pt,
            "n_seeds": len(grp),
        }
        for c in metric_cols:
            row[f"{c}_mean"] = float(grp[c].mean())
            row[f"{c}_std"] = float(grp[c].std()) if len(grp) > 1 else 0.0
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary_metrics.csv", index=False)

    # ── Prior control metrics ─────────────────────────────────────────────
    ctrl_rows = []
    for tgt in seed_agg["target"].unique():
        for model_id in [MODEL_A1, MODEL_A2, MODEL_A3]:
            sub = seed_agg[(seed_agg.target == tgt) & (seed_agg.model_id == model_id)]
            matched = sub[sub.prior_type == "matched"]
            if matched.empty:
                continue
            for ctrl in ["unrelated", "shuffled", "random"]:
                ctrl_sub = sub[sub.prior_type == ctrl]
                if ctrl_sub.empty:
                    continue
                common_seeds = sorted(set(matched.seed) & set(ctrl_sub.seed))
                if len(common_seeds) < 2:
                    continue
                m_vals = matched.set_index("seed").loc[common_seeds, "pearson"].values
                c_vals = ctrl_sub.set_index("seed").loc[common_seeds, "pearson"].values
                diff = m_vals - c_vals
                ctrl_rows.append({
                    "target": tgt, "model_id": model_id,
                    "control": ctrl, "n_seeds": len(common_seeds),
                    "mean_paired_delta": float(diff.mean()),
                    "median_paired_delta": float(np.median(diff)),
                    "n_positive": int(np.sum(diff > 0)),
                })
    pd.DataFrame(ctrl_rows).to_csv(out_dir / "prior_control_metrics.csv", index=False)

    # ── Pairwise model comparisons ────────────────────────────────────────
    comp_rows = []
    for tgt in seed_agg["target"].unique():
        for model_id in [MODEL_A1, MODEL_A2, MODEL_A3]:
            for ctrl_name, ctrl_model, ctrl_prior in [
                ("A4", MODEL_A4, "matched"),
                (f"{model_id}_unrelated", model_id, "unrelated"),
                (f"{model_id}_shuffled", model_id, "shuffled"),
                (f"{model_id}_random", model_id, "random"),
            ]:
                m_sub = seed_agg[(seed_agg.target == tgt) & (seed_agg.model_id == model_id)
                                  & (seed_agg.prior_type == "matched")]
                c_sub = seed_agg[(seed_agg.target == tgt) & (seed_agg.model_id == ctrl_model)
                                  & (seed_agg.prior_type == ctrl_prior)]
                common_seeds = sorted(set(m_sub.seed) & set(c_sub.seed))
                if len(common_seeds) < 2:
                    continue
                m_vals = m_sub.set_index("seed").loc[common_seeds, "pearson"].values
                c_vals = c_sub.set_index("seed").loc[common_seeds, "pearson"].values
                diff = m_vals - c_vals
                comp_rows.append({
                    "target": tgt, "model_id": model_id,
                    "vs": ctrl_name, "n_seeds": len(common_seeds),
                    "mean_delta": float(diff.mean()),
                    "median_delta": float(np.median(diff)),
                    "n_positive": int(np.sum(diff > 0)),
                })
    pd.DataFrame(comp_rows).to_csv(out_dir / "paired_comparisons.csv", index=False)

    # ── Pilot decision ────────────────────────────────────────────────────
    decisions = _make_pilot_decision(seed_agg)
    decisions["run_metadata"] = {
        "seeds": seeds, "n_folds": n_folds, "val_fraction": val_frac,
        "lifting_rules": lifting_rules, "top_k": top_k,
        "gamma_grid": gamma_grid, "lambda_l_grid": lambda_l_grid,
        "lambda_fc_grid": lambda_fc_grid, "lambda_sc_grid": lambda_sc_grid,
        "config_hash": cfg_hash, "elapsed_seconds": time.time() - t0,
    }
    save_json(decisions, out_dir / "pilot_decision.json")
    save_json(decisions, out_dir / "run_metadata.json")

    # ── Figures ───────────────────────────────────────────────────────────
    _make_pilot_figures(seed_agg, fig_dir)

    (out_dir / "COMPLETE").write_text("done", encoding="utf-8")
    _print_validation_report(out_dir, split_df, cfg_hash)

    print(f"\nPilot complete: {out_dir} ({time.time()-t0:.0f}s)")
    print(f"Overall decision: {json.dumps(decisions.get('overall_recommendation', ''), indent=2)}")


# ── decision logic ───────────────────────────────────────────────────────────

def _make_pilot_decision(seed_agg: pd.DataFrame) -> Dict:
    decisions = {}
    for tgt in seed_agg["target"].unique():
        tgt_sub = seed_agg[seed_agg.target == tgt]

        a0 = tgt_sub[(tgt_sub.model_id == MODEL_A0) & (tgt_sub.prior_type == "matched")]
        a4 = tgt_sub[(tgt_sub.model_id == MODEL_A4) & (tgt_sub.prior_type == "matched")]
        a3 = tgt_sub[(tgt_sub.model_id == MODEL_A3) & (tgt_sub.prior_type == "matched")]

        a0_mean = float(a0["pearson"].mean()) if not a0.empty else 0.0
        a4_mean = float(a4["pearson"].mean()) if not a4.empty else 0.0
        a3_mean = float(a3["pearson"].mean()) if not a3.empty else 0.0
        best_no_prior = max(a0_mean, a4_mean)

        delta_vs_a0 = a3_mean - a0_mean
        delta_vs_a4 = a3_mean - a4_mean
        delta_vs_best_no_prior = a3_mean - best_no_prior

        common_seeds_34 = sorted(set(a3.seed) & set(a4.seed)) if not a3.empty and not a4.empty else []
        n_pos_vs_a4 = 0
        if len(common_seeds_34) >= 2:
            m_vals = a3.set_index("seed").loc[common_seeds_34, "pearson"].values
            c_vals = a4.set_index("seed").loc[common_seeds_34, "pearson"].values
            n_pos_vs_a4 = int(np.sum(m_vals > c_vals))

        a1 = tgt_sub[(tgt_sub.model_id == MODEL_A1) & (tgt_sub.prior_type == "matched")]
        a2 = tgt_sub[(tgt_sub.model_id == MODEL_A2) & (tgt_sub.prior_type == "matched")]
        a1_mean = float(a1["pearson"].mean()) if not a1.empty else 0.0
        a2_mean = float(a2["pearson"].mean()) if not a2.empty else 0.0

        if delta_vs_best_no_prior >= 0.015 and n_pos_vs_a4 >= 2:
            rec = "full_10x5_msancr"
        elif delta_vs_best_no_prior >= 0.008 and n_pos_vs_a4 >= 1:
            rec = "one_targeted_msancr_refinement"
        elif delta_vs_best_no_prior < 0.005:
            rec = "ct_mac_prior_rebuild"
        elif a1_mean >= a2_mean and abs(a1_mean - a3_mean) < 0.005:
            rec = "simplify_to_anisotropic_ridge"
        elif a2_mean > a1_mean:
            rec = "fc_laplacian_matters_more"
        elif a4_mean >= a3_mean - 0.002:
            rec = "modality_specific_only"
        else:
            rec = "anisotropic_ncr_promising"

        decisions[tgt] = {
            "best_model": "A3" if a3_mean >= best_no_prior else ("A4" if a4_mean > a0_mean else "A0"),
            "best_prior": "matched",
            "delta_r_vs_A0": round(delta_vs_a0, 6),
            "delta_r_vs_A4": round(delta_vs_a4, 6),
            "delta_r_vs_strongest_no_prior": round(delta_vs_best_no_prior, 6),
            "positive_seeds_vs_A4": n_pos_vs_a4,
            "total_seeds": len(common_seeds_34),
            "A0_pearson": round(a0_mean, 6),
            "A1_pearson": round(a1_mean, 6),
            "A2_pearson": round(a2_mean, 6),
            "A3_pearson": round(a3_mean, 6),
            "A4_pearson": round(a4_mean, 6),
            "diagnostic_A1_vs_A2": "aniso_dominates" if a1_mean > a2_mean else "laplacian_dominates",
            "recommended_next_step": rec,
        }

    recs = {k: v["recommended_next_step"] for k, v in decisions.items()}
    if all(r == "full_10x5_msancr" for r in recs.values()):
        overall = "full_10x5_msancr"
    elif any(r in ("full_10x5_msancr", "one_targeted_msancr_refinement") for r in recs.values()):
        overall = "refine_before_full_run"
    else:
        overall = "ct_mac_prior_rebuild"

    return {"per_task": decisions, "overall_recommendation": overall}


# ── figures ──────────────────────────────────────────────────────────────────

def _make_pilot_figures(seed_agg: pd.DataFrame, fig_dir: Path) -> None:
    try:
        for tgt in seed_agg["target"].unique():
            sub = seed_agg[seed_agg.target == tgt]
            matched = sub[sub.prior_type == "matched"]
            if matched.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            models = sorted(matched.model_id.unique())
            x = np.arange(len(matched.seed.unique()))
            w = 0.15
            for i, mid in enumerate(models):
                ms = matched[matched.model_id == mid].sort_values("seed")
                ax.bar(x + i * w, ms["pearson"].values, w, label=mid)
            ax.set_xlabel("Seed")
            ax.set_ylabel("Pearson r")
            ax.set_title(f"MS-A-NCR Pilot: {tgt} (matched prior)")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(fig_dir / f"model_comparison_{tgt}.pdf", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"Plot error: {e}")


# ── validation ───────────────────────────────────────────────────────────────

def _print_validation_report(out_dir: Path, split_df: pd.DataFrame, cfg_hash: str) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)
    n_seeds = split_df["seed"].nunique()
    n_folds = split_df["fold"].nunique()
    n_targets = split_df["target"].nunique()
    n_models = split_df["model_id"].nunique()
    n_priors = split_df["prior_type"].nunique()
    print(f"  Seeds: {n_seeds}, Folds: {n_folds}, Targets: {n_targets}")
    print(f"  Models: {n_models}, Priors: {n_priors}")
    print(f"  Total rows: {len(split_df)}")
    nan_count = split_df.isna().sum().sum()
    print(f"  NaN values: {nan_count}")
    dup = split_df.duplicated(subset=["target", "seed", "fold", "model_id", "prior_type"]).sum()
    print(f"  Duplicate rows: {dup}")
    print(f"  Config hash: {cfg_hash}")
    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/iclr/msancr_pilot.yaml")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_pilot(cfg, output_dir_override=args.output_dir)


if __name__ == "__main__":
    main()
