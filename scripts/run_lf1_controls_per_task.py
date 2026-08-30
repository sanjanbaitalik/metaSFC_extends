"""LF1 fixed prior controls — fastest approach.
Re-run only outer splits to get matched LF1 weights, then run controls."""
from __future__ import annotations

import json
import sys
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
    RIDGE_GRID, TOP_K, DIAGONAL_EPSILON,
    fit_predict_ridge_sc_oof, fit_predict_fp_final, predict_fp_final,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import build_msancr_cache
from metascfc.models.iclr_backbones.network_constrained_ridge import build_edge_laplacian
from metascfc.experiments.msancr_refinement import load_roi_prior, upper_triangle_features

BASE = Path("/home/iemiedc2026/Documents/Sanjan/iclr")
OUT = BASE / "outputs/iclr/prior_aware_late_fusion_integrity_audit/lf1_fixed_prior_swaps"
OUT.mkdir(parents=True, exist_ok=True)

task_key = sys.argv[1]
cfg = yaml.safe_load((BASE / "configs/iclr/prior_aware_late_fusion.yaml").read_text())

print("Loading data...", flush=True)
fc_mats, sc_mats, y_all, subject_ids, groups = load_connectomes(cfg["data"])
n_rois = int(cfg.get("n_rois", fc_mats.shape[1]))
X_fc = upper_triangle_features(fc_mats)
X_sc = upper_triangle_features(sc_mats)

target_cfg = cfg["targets"][task_key]
y_task = np.asarray(np.load(target_cfg["label_path"], allow_pickle=False), dtype=np.float64).reshape(-1)

prior_cfg = cfg.get("priors", {}).get(task_key, {})
priors = {}
for pn in ["matched", "unrelated", "shuffled", "random"]:
    path = prior_cfg.get(pn)
    if path:
        priors[pn] = load_roi_prior(path, n_rois)

# Hardcoded FP hyperparams from original run
# Use typical values: lambda_fc=1.0, lambda_l=0.5, gamma=0.5
# Actually, we need the per-split matched params. Let me re-run the matched experiment
# but ONLY the outer split evaluation (skip prior controls)

# Run full experiment for matched only
from metascfc.experiments.prior_aware_late_fusion import run_late_fusion_experiment
print("Running matched experiment (no prior controls)...", flush=True)
result = run_late_fusion_experiment(
    X_fc, X_sc, y_task, {"matched": priors["matched"]},
    seeds=[0, 1, 2], n_outer_folds=5, n_inner_folds=3,
    ridge_grid=RIDGE_GRID,
    gamma_grid=[0.1, 0.25, 0.5, 1.0, 2.0],
    lambda_l_grid=[0.03, 0.1, 0.5, 1.0, 2.0, 5.0],
    n_rois=n_rois,
    output_dir=str(OUT / f"matched_{task_key}"),
    prior_controls=[],
)

all_rows = []
for sr in result["all_split_results"]:
    seed = sr.seed
    outer_fold = sr.outer_fold
    trainval_idx = sr.train_idx
    test_idx = sr.test_idx
    lf1_weights = sr.lf1_weights
    fp_params = sr.level1_results["FP"].selected_hyperparams
    s_alpha = sr.level1_results["S"].selected_hyperparams.get("alpha", 1.0)

    w_fp = lf1_weights.get("FP", 0.0)
    w_s = lf1_weights.get("S", 0.0)

    m_matched = prediction_metrics(y_task[test_idx], sr.lf1_test_pred)
    all_rows.append({
        "task": task_key, "seed": seed, "fold": outer_fold,
        "prior_type": "matched", "pearson": m_matched["pearson"],
        "rmse": m_matched["rmse"], "mae": m_matched["mae"],
        "w_fp": w_fp, "w_s": w_s,
    })

    scaler_sc = StandardScaler()
    X_sc_train_z = scaler_sc.fit_transform(X_sc[trainval_idx])
    X_sc_test_z = scaler_sc.transform(X_sc[test_idx])
    s_model = Ridge(alpha=s_alpha, fit_intercept=True)
    s_model.fit(X_sc_train_z, y_task[trainval_idx])
    s_test = s_model.predict(X_sc_test_z)

    for ctrl_type in ["unrelated", "shuffled", "random"]:
        ctrl_cache = build_msancr_cache(
            priors[ctrl_type], n_rois, gamma=0.5, lifting="prod",
            top_k=TOP_K, epsilon=DIAGONAL_EPSILON, weighting="binary",
            couple_modalities=False, normalize_laplacian="sym",
            edge_laplacian=build_edge_laplacian(n_rois, prior_scores=priors[ctrl_type], top_k=TOP_K, weighting="binary", couple_modalities=False, normalize="sym"),
        )
        t0 = time.time()
        alpha_ctrl, scaler_ctrl, cym, cystd = fit_predict_fp_final(
            X_fc, y_task, trainval_idx, ctrl_cache, fp_params,
        )
        X_fc_train_ctrl_z = scaler_ctrl.transform(X_fc[trainval_idx])
        fp_ctrl_test = predict_fp_final(
            X_fc[test_idx], X_fc_train_ctrl_z, alpha_ctrl, ctrl_cache,
            fp_params, scaler_ctrl, cym, cystd,
        )
        lf1_ctrl_pred = w_fp * fp_ctrl_test + w_s * s_test
        m = prediction_metrics(y_task[test_idx], lf1_ctrl_pred)
        all_rows.append({
            "task": task_key, "seed": seed, "fold": outer_fold,
            "prior_type": ctrl_type, "pearson": m["pearson"],
            "rmse": m["rmse"], "mae": m["mae"],
            "w_fp": w_fp, "w_s": w_s,
        })
        print(f"  S{seed}F{outer_fold} {ctrl_type}: {m['pearson']:.4f} [{time.time()-t0:.0f}s]", flush=True)

df = pd.DataFrame(all_rows)
df.to_csv(OUT / f"lf1_controls_{task_key}.csv", index=False)

print(f"\n--- {task_key} Summary ---")
for pt in ["matched", "unrelated", "shuffled", "random"]:
    subset = df[df.prior_type == pt]
    print(f"  {pt:12s}: pearson={subset['pearson'].mean():.4f}")

lf1_matched = df[df.prior_type == "matched"]["pearson"].mean()
beats = 0
for pt in ["unrelated", "shuffled", "random"]:
    ctrl_mean = df[df.prior_type == pt]["pearson"].mean()
    delta = lf1_matched - ctrl_mean
    if delta > 0:
        beats += 1
    print(f"  vs {pt}: ctrl={ctrl_mean:.4f}, delta={delta:+.4f}")
print(f"  Beats {beats}/3 controls")
