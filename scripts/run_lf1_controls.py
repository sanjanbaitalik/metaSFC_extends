"""Run LF1 fixed prior controls for both tasks."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from metascfc.benchmark_utils import load_connectomes, prediction_metrics
from metascfc.experiments.prior_aware_late_fusion import (
    RIDGE_GRID, GAMMA_GRID, LAMBDA_L_GRID, TOP_K, DIAGONAL_EPSILON,
    fit_predict_ridge_sc_oof, fit_predict_ridge_fc_oof,
    fit_predict_fp_oof, fit_predict_fp_final, predict_fp_final,
    search_weights_2b,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    build_msancr_cache,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import build_edge_laplacian
from metascfc.experiments.msancr_refinement import load_roi_prior, upper_triangle_features

BASE = Path("/home/iemiedc2026/Documents/Sanjan/iclr")
OUT = BASE / "outputs/iclr/prior_aware_late_fusion_integrity_audit/lf1_fixed_prior_swaps"
OUT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load((BASE / "configs/iclr/prior_aware_late_fusion.yaml").read_text())

# Load data
print("Loading HCP data...")
fc_mats, sc_mats, y_all, subject_ids, groups = load_connectomes(cfg["data"])
n_rois = int(cfg.get("n_rois", fc_mats.shape[1]))
X_fc = upper_triangle_features(fc_mats)
X_sc = upper_triangle_features(sc_mats)
print(f"  FC: {fc_mats.shape}, SC: {sc_mats.shape}, X_fc: {X_fc.shape}")

seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
n_outer = int(cfg.get("n_outer_folds", 5))
n_inner = int(cfg.get("n_inner_folds", 3))

from metascfc.experiments.prior_aware_late_fusion import evaluate_outer_split

all_control_rows = []

for target_key in ["working_memory", "fluid_intelligence"]:
    target_cfg = cfg["targets"][target_key]
    print(f"\n{'='*60}")
    print(f"LF1 Controls: {target_cfg['name']}")
    print(f"{'='*60}")

    y_task = np.asarray(np.load(target_cfg["label_path"], allow_pickle=False), dtype=np.float64).reshape(-1)

    prior_cfg = cfg.get("priors", {}).get(target_key, {})
    priors = {}
    for prior_name in ["matched", "unrelated", "shuffled", "random"]:
        path = prior_cfg.get(prior_name)
        if path:
            priors[prior_name] = load_roi_prior(path, n_rois)

    for seed in seeds:
        kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
        for outer_fold, (trainval_idx, test_idx) in enumerate(kf.split(np.arange(len(y_task)))):
            trainval_idx = np.asarray(trainval_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)

            t0 = time.time()
            print(f"  Seed {seed}, fold {outer_fold}: train={len(trainval_idx)}, test={len(test_idx)}")

            # Build matched cache
            roi_prior_matched = priors["matched"]
            edge_laplacian = build_edge_laplacian(n_rois, prior_scores=roi_prior_matched, top_k=TOP_K, weighting="binary", couple_modalities=False, normalize="sym")
            fp_cache_matched = build_msancr_cache(roi_prior_matched, n_rois, gamma=0.5, lifting="prod", top_k=TOP_K, epsilon=DIAGONAL_EPSILON, weighting="binary", couple_modalities=False, normalize_laplacian="sym", edge_laplacian=edge_laplacian)

            # Run matched outer split to get LF1 weights and FP params
            result = evaluate_outer_split(
                X_fc, X_sc, y_task, seed, outer_fold,
                trainval_idx, test_idx, fp_cache_matched,
                prior_type="matched",
                ridge_grid=RIDGE_GRID, gamma_grid=GAMMA_GRID,
                lambda_l_grid=LAMBDA_L_GRID, n_inner_folds=n_inner, n_rois=n_rois,
            )

            lf1_weights = result.lf1_weights
            fp_params = result.level1_results["FP"].selected_hyperparams

            # Run LF1 controls: FP branch with control prior, same weights
            for ctrl_type in ["unrelated", "shuffled", "random"]:
                ctrl_cache = build_msancr_cache(
                    priors[ctrl_type], n_rois, gamma=0.5, lifting="prod",
                    top_k=TOP_K, epsilon=DIAGONAL_EPSILON, weighting="binary",
                    couple_modalities=False, normalize_laplacian="sym",
                    edge_laplacian=build_edge_laplacian(n_rois, prior_scores=priors[ctrl_type], top_k=TOP_K, weighting="binary", couple_modalities=False, normalize="sym"),
                )

                # Refit control FP branch
                alpha_ctrl, scaler_ctrl, ctrl_ym, ctrl_ystd = fit_predict_fp_final(
                    X_fc, y_task, trainval_idx, ctrl_cache, fp_params,
                )
                X_fc_train_ctrl_z = scaler_ctrl.transform(X_fc[trainval_idx])

                # Refit S branch
                scaler_sc = StandardScaler()
                X_sc_train_z = scaler_sc.fit_transform(X_sc[trainval_idx])
                X_sc_test_z = scaler_sc.transform(X_sc[test_idx])
                s_model = Ridge(alpha=result.level1_results["S"].selected_hyperparams.get("alpha", 1.0), fit_intercept=True)
                s_model.fit(X_sc_train_z, y_task[trainval_idx])
                s_test = s_model.predict(X_sc_test_z)

                # Control FP prediction
                ctrl_test = predict_fp_final(
                    X_fc[test_idx], X_fc_train_ctrl_z, alpha_ctrl, ctrl_cache,
                    fp_params, scaler_ctrl, ctrl_ym, ctrl_ystd,
                )

                # Apply LF1 weights
                w_fp = lf1_weights.get("FP", 0.0)
                w_s = lf1_weights.get("S", 0.0)
                lf1_ctrl_pred = w_fp * ctrl_test + w_s * s_test

                m = prediction_metrics(y_task[test_idx], lf1_ctrl_pred)
                elapsed = time.time() - t0

                all_control_rows.append({
                    "task": target_key,
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "prior_type": ctrl_type,
                    "pearson": m["pearson"],
                    "rmse": m["rmse"],
                    "mae": m["mae"],
                    "w_fp": w_fp,
                    "w_s": w_s,
                    "elapsed": elapsed,
                })

                print(f"    {ctrl_type}: pearson={m['pearson']:.4f} (w_fp={w_fp:.2f}, w_s={w_s:.2f}) [{elapsed:.1f}s]")

# Save results
ctrl_df = pd.DataFrame(all_control_rows)
ctrl_df.to_csv(OUT / "lf1_fixed_prior_swap_split_metrics.csv", index=False)

# Summary
summary_rows = []
for task in ["working_memory", "fluid_intelligence"]:
    for prior_type in ["unrelated", "shuffled", "random"]:
        subset = ctrl_df[(ctrl_df.task == task) & (ctrl_df.prior_type == prior_type)]
        summary_rows.append({
            "task": task,
            "prior_type": prior_type,
            "pearson_mean": float(subset["pearson"].mean()),
            "pearson_std": float(subset["pearson"].std(ddof=1)) if len(subset) > 1 else 0.0,
            "rmse_mean": float(subset["rmse"].mean()),
            "mae_mean": float(subset["mae"].mean()),
        })

pd.DataFrame(summary_rows).to_csv(OUT / "lf1_fixed_prior_swap_summary.csv", index=False)

# Compare with matched LF1
print("\n--- LF1 vs Controls Summary ---")
for task in ["working_memory", "fluid_intelligence"]:
    task_seeds = pd.read_csv(BASE / f"outputs/iclr/prior_aware_late_fusion/{task}/seed_metrics.csv")
    lf1_matched_mean = float(task_seeds[task_seeds.model == "LF1"]["pearson"].mean())
    
    print(f"\n  {task}: LF1 matched = {lf1_matched_mean:.4f}")
    for prior_type in ["unrelated", "shuffled", "random"]:
        subset = ctrl_df[(ctrl_df.task == task) & (ctrl_df.prior_type == prior_type)]
        ctrl_mean = float(subset["pearson"].mean())
        delta = lf1_matched_mean - ctrl_mean
        print(f"    vs {prior_type}: ctrl={ctrl_mean:.4f}, delta={delta:+.4f}")

# Determine if prior-aware beats controls
for task in ["working_memory", "fluid_intelligence"]:
    task_seeds = pd.read_csv(BASE / f"outputs/iclr/prior_aware_late_fusion/{task}/seed_metrics.csv")
    lf1_matched_mean = float(task_seeds[task_seeds.model == "LF1"]["pearson"].mean())
    
    beats_count = 0
    for prior_type in ["unrelated", "shuffled", "random"]:
        subset = ctrl_df[(ctrl_df.task == task) & (ctrl_df.prior_type == prior_type)]
        ctrl_mean = float(subset["pearson"].mean())
        if lf1_matched_mean > ctrl_mean:
            beats_count += 1
    
    print(f"\n  {task}: LF1 matched beats {beats_count}/3 prior controls")
    if beats_count >= 2:
        print(f"    → PASSES the 'beats at least 2/3 controls' gate")
    else:
        print(f"    → FAILS the 'beats at least 2/3 controls' gate")

print("\nDone.")
