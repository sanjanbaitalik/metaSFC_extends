"""Preflight smoke test for LF1 final 10x5."""
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
    RIDGE_GRID, GAMMA_GRID, LAMBDA_L_GRID, TOP_K, DIAGONAL_EPSILON,
    fit_predict_ridge_sc_oof, fit_predict_ridge_fc_oof,
    fit_predict_fp_oof, fit_predict_fp_final, predict_fp_final,
    search_weights_2b,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import build_msancr_cache
from metascfc.models.iclr_backbones.network_constrained_ridge import build_edge_laplacian
from metascfc.experiments.msancr_refinement import load_roi_prior, upper_triangle_features

BASE = Path("/home/iemiedc2026/Documents/Sanjan/iclr")
OUT = BASE / "outputs/iclr/lf1_finalization_audit"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    cfg = yaml.safe_load((BASE / "configs/iclr/lf1_final_10x5.yaml").read_text())

    print("Loading data...")
    fc_mats, sc_mats, y_all, subject_ids, groups = load_connectomes(cfg["data"])
    n_rois = int(cfg.get("n_rois", fc_mats.shape[1]))
    X_fc = upper_triangle_features(fc_mats)
    X_sc = upper_triangle_features(sc_mats)

    preflight = {"checks": [], "all_pass": True}

    for target_key in ["working_memory", "fluid_intelligence"]:
        target_cfg = cfg["targets"][target_key]
        print(f"\n{'='*60}")
        print(f"Preflight: {target_cfg['name']}")
        print(f"{'='*60}")

        y_task = np.asarray(np.load(target_cfg["label_path"], allow_pickle=False), dtype=np.float64).reshape(-1)

        prior_cfg = cfg.get("priors", {}).get(target_key, {})
        priors = {}
        for pn in ["matched", "unrelated", "shuffled", "random"]:
            path = prior_cfg.get(pn)
            if path:
                priors[pn] = load_roi_prior(path, n_rois)

        # Smoke test: seed 0, fold 0
        seed = 0
        outer_fold = 0
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        trainval_idx, test_idx = next(iter(kf.split(np.arange(len(y_task)))))
        trainval_idx = np.asarray(trainval_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)

        t0 = time.time()
        print(f"  Seed 0, fold 0: train={len(trainval_idx)}, test={len(test_idx)}")

        fp_cache = build_msancr_cache(
            priors["matched"], n_rois, gamma=0.5, lifting="prod",
            top_k=TOP_K, epsilon=DIAGONAL_EPSILON, weighting="binary",
            couple_modalities=False, normalize_laplacian="sym",
            edge_laplacian=build_edge_laplacian(n_rois, prior_scores=priors["matched"], top_k=TOP_K, weighting="binary", couple_modalities=False, normalize="sym"),
        )

        # OOF
        s_oof, s_alpha, _, _, s_info = fit_predict_ridge_sc_oof(X_sc, y_task, trainval_idx, trainval_idx, RIDGE_GRID, 3, seed, outer_fold)
        f0_oof, f0_alpha, _, _, f0_info = fit_predict_ridge_fc_oof(X_fc, y_task, trainval_idx, trainval_idx, RIDGE_GRID, 3, seed, outer_fold)
        fp_oof, fp_info = fit_predict_fp_oof(X_fc, y_task, trainval_idx, trainval_idx, fp_cache, GAMMA_GRID, LAMBDA_L_GRID, RIDGE_GRID, 3, seed, outer_fold, n_rois)

        lf0_w, _ = search_weights_2b(y_task[trainval_idx], {"F0": f0_oof, "S": s_oof}, ["F0", "S"])
        lf1_w, _ = search_weights_2b(y_task[trainval_idx], {"FP": fp_oof, "S": s_oof}, ["FP", "S"])

        # Refit
        scaler_sc = StandardScaler()
        X_sc_train_z = scaler_sc.fit_transform(X_sc[trainval_idx])
        X_sc_test_z = scaler_sc.transform(X_sc[test_idx])
        s_model = Ridge(alpha=s_info["alpha"], fit_intercept=True)
        s_model.fit(X_sc_train_z, y_task[trainval_idx])
        s_test = s_model.predict(X_sc_test_z)

        scaler_fc = StandardScaler()
        X_fc_train_z = scaler_fc.fit_transform(X_fc[trainval_idx])
        X_fc_test_z = scaler_fc.transform(X_fc[test_idx])
        f0_model = Ridge(alpha=f0_info["alpha"], fit_intercept=True)
        f0_model.fit(X_fc_train_z, y_task[trainval_idx])
        f0_test = f0_model.predict(X_fc_test_z)

        alpha_fp, scaler_fp, fp_ym, fp_ystd = fit_predict_fp_final(X_fc, y_task, trainval_idx, fp_cache, fp_info.get("best_params", {}))
        X_fc_train_fp_z = scaler_fp.transform(X_fc[trainval_idx])
        fp_test = predict_fp_final(X_fc[test_idx], X_fc_train_fp_z, alpha_fp, fp_cache, fp_info.get("best_params", {}), scaler_fp, fp_ym, fp_ystd)

        lf0_pred = lf0_w["F0"] * f0_test + lf0_w["S"] * s_test
        lf1_pred = lf1_w["FP"] * fp_test + lf1_w["S"] * s_test
        a4_pred = 0.5 * (f0_test + s_test)

        m_lf0 = prediction_metrics(y_task[test_idx], lf0_pred)
        m_lf1 = prediction_metrics(y_task[test_idx], lf1_pred)
        m_a4 = prediction_metrics(y_task[test_idx], a4_pred)

        print(f"  LF0={m_lf0['pearson']:.4f} LF1={m_lf1['pearson']:.4f} A4={m_a4['pearson']:.4f} [{time.time()-t0:.0f}s]")

        # Check 1: finite predictions
        check1 = all(np.all(np.isfinite(p)) for p in [lf0_pred, lf1_pred, a4_pred])
        preflight["checks"].append({"task": target_key, "check": "finite_predictions", "pass": check1})

        # Check 2: OOF integrity (no NaN in OOF)
        check2 = all(np.all(np.isfinite(o)) for o in [s_oof, f0_oof, fp_oof])
        preflight["checks"].append({"task": target_key, "check": "oof_integrity", "pass": check2})

        # Check 3: weights valid
        check3 = all(lf1_w[k] >= 0 for k in lf1_w) and abs(sum(lf1_w.values()) - 1.0) < 1e-4
        preflight["checks"].append({"task": target_key, "check": "fusion_weights_valid", "pass": check3})

        # Check 4: matched LF1 reproduction
        # Run matched control using same weights/HPs
        from metascfc.experiments.prior_aware_late_fusion import evaluate_prior_swap
        ctrl_result = evaluate_prior_swap(
            X_fc, X_sc, y_task, seed, outer_fold,
            trainval_idx, test_idx, fp_cache,
            matched_weights=lf1_w,
            matched_fp_params=fp_info.get("best_params", {}),
            control_prior_type="matched",
            s_alpha=s_info["alpha"],
            f0_alpha=f0_info["alpha"],
        )
        check4 = abs(ctrl_result["test_pearson"] - m_lf1["pearson"]) < 1e-6
        preflight["checks"].append({"task": target_key, "check": "matched_control_reproduces", "pass": check4,
                                    "ordinary": m_lf1["pearson"], "control": ctrl_result["test_pearson"],
                                    "diff": abs(ctrl_result["test_pearson"] - m_lf1["pearson"])})
        if not check4:
            print(f"  WARNING: matched control={ctrl_result['test_pearson']:.6f} != ordinary={m_lf1['pearson']:.6f} (diff={abs(ctrl_result['test_pearson'] - m_lf1['pearson']):.2e})")

        # Check 5: finite metrics
        check5 = all(np.isfinite(m[k]) for m in [m_lf0, m_lf1, m_a4] for k in ["pearson", "rmse", "mae"])
        preflight["checks"].append({"task": target_key, "check": "finite_metrics", "pass": check5})

        # Check 6: no mod3
        check6 = not Path("src/metascfc/experiments/modification_3.py").exists()
        preflight["checks"].append({"task": target_key, "check": "no_mod3", "pass": check6})

    preflight["all_pass"] = all(c["pass"] for c in preflight["checks"])

    with open(OUT / "preflight_integrity.json", "w") as f:
        json.dump(preflight, f, indent=2, default=str)

    print(f"\nPreflight: {'ALL PASS' if preflight['all_pass'] else 'FAILURES DETECTED'}")
    for c in preflight["checks"]:
        print(f"  [{('PASS' if c['pass'] else 'FAIL')}] {c['task']} {c['check']}")

    return preflight["all_pass"]


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
