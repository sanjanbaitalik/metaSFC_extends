"""LF1-only final experiment runner — uses evaluate_outer_split for correctness."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from metascfc.benchmark_utils import prediction_metrics
from metascfc.experiments.prior_aware_late_fusion import (
    RIDGE_GRID, GAMMA_GRID, LAMBDA_L_GRID, TOP_K, DIAGONAL_EPSILON,
    evaluate_outer_split, fit_predict_fp_final, predict_fp_final,
    evaluate_prior_swap,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import build_msancr_cache
from metascfc.models.iclr_backbones.network_constrained_ridge import build_edge_laplacian


def run_lf1_experiment(
    X_fc, X_sc, y, priors, seeds, n_outer=5, n_inner=3,
    ridge_grid=RIDGE_GRID, gamma_grid=GAMMA_GRID, lambda_l_grid=LAMBDA_L_GRID,
    n_rois=116, output_dir=None,
):
    """Run LF0 + LF1 + A4 via evaluate_outer_split."""
    output_dir = Path(output_dir) if output_dir else Path("outputs/iclr/lf1_final_10x5")
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = []
    all_split_results = []
    t_start = time.time()

    for seed in seeds:
        kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
        for outer_fold, (trainval_idx, test_idx) in enumerate(kf.split(np.arange(len(y)))):
            trainval_idx = np.asarray(trainval_idx, dtype=int)
            test_idx = np.asarray(test_idx, dtype=int)
            t0 = time.time()

            fp_cache = build_msancr_cache(
                priors["matched"], n_rois, gamma=0.5, lifting="prod",
                top_k=TOP_K, epsilon=DIAGONAL_EPSILON, weighting="binary",
                couple_modalities=False, normalize_laplacian="sym",
                edge_laplacian=build_edge_laplacian(n_rois, prior_scores=priors["matched"], top_k=TOP_K, weighting="binary", couple_modalities=False, normalize="sym"),
            )

            result = evaluate_outer_split(
                X_fc, X_sc, y, seed, outer_fold,
                trainval_idx, test_idx, fp_cache,
                prior_type="matched",
                ridge_grid=ridge_grid, gamma_grid=gamma_grid,
                lambda_l_grid=lambda_l_grid, n_inner_folds=n_inner, n_rois=n_rois,
            )

            fp_params = result.level1_results["FP"].selected_hyperparams

            for model_name, prefix in [("A4", "a4"), ("LF0", "lf0"), ("LF1", "lf1")]:
                split_rows.append({
                    "seed": seed, "fold": outer_fold, "model": model_name,
                    "pearson": getattr(result, f"{prefix}_pearson"),
                    "rmse": getattr(result, f"{prefix}_rmse"),
                    "mae": getattr(result, f"{prefix}_mae"),
                    "w_fp": result.lf1_weights.get("FP", 0.0),
                    "w_s": result.lf1_weights.get("S", 0.0),
                    "w_f0": result.lf0_weights.get("F0", 0.0),
                    "s_alpha": result.level1_results["S"].selected_hyperparams.get("alpha", 1.0),
                    "f0_alpha": result.level1_results["F0"].selected_hyperparams.get("alpha", 1.0),
                    "fp_lambda_fc": fp_params.get("lambda_fc", 1.0),
                    "fp_lambda_l": fp_params.get("lambda_l", 0.0),
                    "fp_gamma": fp_params.get("gamma", 0.5),
                    "fp_lifting": fp_params.get("lifting", "prod"),
                })

            all_split_results.append(result)

            elapsed = time.time() - t0
            print(f"  S{seed}F{outer_fold}: LF0={result.lf0_pearson:.4f} LF1={result.lf1_pearson:.4f} A4={result.a4_pearson:.4f} [{elapsed:.0f}s]", flush=True)

    split_df = pd.DataFrame(split_rows)
    seed_df = split_df.groupby(["seed", "model"], as_index=False).agg(
        pearson=("pearson", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean"),
    )

    summary_rows = []
    for model in ["A4", "LF0", "LF1"]:
        ms = seed_df[seed_df.model == model]
        p = ms["pearson"].values
        summary_rows.append({
            "model": model,
            "pearson_mean": float(np.mean(p)), "pearson_median": float(np.median(p)),
            "pearson_std": float(np.std(p, ddof=1)),
            "rmse_mean": float(ms["rmse"].mean()), "mae_mean": float(ms["mae"].mean()),
        })

    elapsed_total = time.time() - t_start
    print(f"\nTotal: {elapsed_total:.0f}s", flush=True)

    split_df.to_csv(output_dir / "split_metrics.csv", index=False)
    seed_df.to_csv(output_dir / "seed_metrics.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output_dir / "summary_metrics.csv", index=False)

    return {"split_df": split_df, "seed_df": seed_df, "summary": summary_rows, "all_split_results": all_split_results}


def run_lf1_prior_controls(all_split_results, X_fc, X_sc, y, priors, n_rois=116, output_dir=None):
    """Run LF1 fixed prior controls using evaluate_prior_swap (fixed bug)."""
    output_dir = Path(output_dir) if output_dir else Path("outputs/iclr/lf1_final_10x5")
    output_dir.mkdir(parents=True, exist_ok=True)

    ctrl_rows = []
    for i, result in enumerate(all_split_results):
        seed = result.seed
        outer_fold = result.outer_fold
        trainval_idx = result.train_idx
        test_idx = result.test_idx
        lf1_weights = result.lf1_weights
        fp_params = result.level1_results["FP"].selected_hyperparams
        s_info = result.level1_results["S"].selected_hyperparams
        w_fp = lf1_weights.get("FP", 0.0)
        w_s = lf1_weights.get("S", 0.0)

        m_lf1 = prediction_metrics(y[test_idx], result.lf1_test_pred)
        ctrl_rows.append({"seed": seed, "fold": outer_fold, "prior_type": "matched",
                          "pearson": m_lf1["pearson"], "rmse": m_lf1["rmse"], "mae": m_lf1["mae"],
                          "w_fp": w_fp, "w_s": w_s})

        for ctrl_type in ["unrelated", "shuffled", "random"]:
            ctrl_cache = build_msancr_cache(
                priors[ctrl_type], n_rois, gamma=0.5, lifting="prod",
                top_k=TOP_K, epsilon=DIAGONAL_EPSILON, weighting="binary",
                couple_modalities=False, normalize_laplacian="sym",
                edge_laplacian=build_edge_laplacian(n_rois, prior_scores=priors[ctrl_type], top_k=TOP_K, weighting="binary", couple_modalities=False, normalize="sym"),
            )
            ctrl_result = evaluate_prior_swap(
                X_fc, X_sc, y, seed, outer_fold,
                trainval_idx, test_idx, ctrl_cache,
                matched_weights=lf1_weights,
                matched_fp_params=fp_params,
                control_prior_type=ctrl_type,
                s_alpha=s_info.get("alpha", 1.0),
                f0_alpha=result.level1_results["F0"].selected_hyperparams.get("alpha", 1.0),
            )
            ctrl_rows.append({"seed": seed, "fold": outer_fold, "prior_type": ctrl_type,
                              "pearson": ctrl_result["test_pearson"], "rmse": ctrl_result["test_rmse"],
                              "mae": ctrl_result["test_mae"],
                              "w_fp": ctrl_result["w_p"], "w_s": ctrl_result["w_s"]})

    ctrl_df = pd.DataFrame(ctrl_rows)
    ctrl_df.to_csv(output_dir / "fixed_prior_swap_split_metrics.csv", index=False)

    ctrl_seed = ctrl_df.groupby(["seed", "prior_type"], as_index=False).agg(
        pearson=("pearson", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean"),
    )
    ctrl_seed.to_csv(output_dir / "fixed_prior_swap_seed_metrics.csv", index=False)

    ctrl_summary = ctrl_df.groupby("prior_type", as_index=False).agg(
        pearson_mean=("pearson", "mean"), pearson_std=("pearson", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        rmse_mean=("rmse", "mean"), mae_mean=("mae", "mean"),
    )
    ctrl_summary.to_csv(output_dir / "fixed_prior_swap_summary.csv", index=False)

    return {"ctrl_df": ctrl_df, "ctrl_seed": ctrl_seed, "ctrl_summary": ctrl_summary}


def compute_statistics(seed_df, ctrl_summary, task_name):
    """Compute all inference statistics."""
    lf0 = seed_df[seed_df.model == "LF0"].sort_values("seed")["pearson"].values
    lf1 = seed_df[seed_df.model == "LF1"].sort_values("seed")["pearson"].values
    a4 = seed_df[seed_df.model == "A4"].sort_values("seed")["pearson"].values
    deltas = lf1 - lf0

    nonzero = np.abs(deltas) > 1e-10
    if nonzero.sum() >= 3:
        stat, p_wilcoxon = wilcoxon(deltas[nonzero])
    else:
        p_wilcoxon = 1.0

    rng = np.random.RandomState(42)
    n_boot = 10000
    boot_means = np.array([np.mean(deltas[rng.randint(0, len(deltas), len(deltas))]) for _ in range(n_boot)])
    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

    mean_d = np.mean(deltas)
    std_d = np.std(deltas, ddof=1)
    dz = mean_d / std_d if std_d > 0 else 0.0

    matched_mean = float(ctrl_summary[ctrl_summary.prior_type == "matched"]["pearson_mean"].iloc[0]) if len(ctrl_summary[ctrl_summary.prior_type == "matched"]) > 0 else 0.0
    beats = 0
    ctrl_means = {}
    for pt in ["unrelated", "shuffled", "random"]:
        cm = float(ctrl_summary[ctrl_summary.prior_type == pt]["pearson_mean"].iloc[0]) if len(ctrl_summary[ctrl_summary.prior_type == pt]) > 0 else 0.0
        ctrl_means[pt] = cm
        if matched_mean > cm:
            beats += 1

    large_margin = (np.mean(deltas) >= 0.010 and np.median(deltas) >= 0.010 and np.sum(deltas > 0) >= 7)

    return {
        "task": task_name,
        "lf0_mean": float(np.mean(lf0)), "lf0_std": float(np.std(lf0, ddof=1)),
        "lf1_mean": float(np.mean(lf1)), "lf1_std": float(np.std(lf1, ddof=1)),
        "a4_mean": float(np.mean(a4)), "a4_std": float(np.std(a4, ddof=1)),
        "delta_mean": float(mean_d), "delta_median": float(np.median(deltas)),
        "positive_seeds": int(np.sum(deltas > 0)),
        "n_seeds": len(deltas),
        "p_wilcoxon": float(p_wilcoxon),
        "ci_95": [float(ci_lo), float(ci_hi)],
        "cohen_dz": float(dz),
        "large_margin_consistency_gate": bool(large_margin),
        "ctrl_matched": matched_mean,
        "ctrl_unrelated": ctrl_means.get("unrelated", 0),
        "ctrl_shuffled": ctrl_means.get("shuffled", 0),
        "ctrl_random": ctrl_means.get("random", 0),
        "ctrl_beats": beats,
    }


def compute_repeated_cv_sensitivity(all_split_results, y_task):
    """Compute corrected repeated-k-fold sensitivity."""
    fold_deltas = []
    for result in all_split_results:
        fold_deltas.append(result.lf1_pearson - result.lf0_pearson)
    fold_deltas = np.array(fold_deltas)

    n = len(fold_deltas)
    k = 5
    mean_d = np.mean(fold_deltas)
    var_d = np.var(fold_deltas, ddof=1)
    corrected_var = (1.0 / k + 1.0 / (n / k)) * var_d
    se = np.sqrt(corrected_var) if corrected_var > 0 else 1e-10
    t_stat = mean_d / se
    from scipy.stats import t as t_dist
    p_cv = 2 * t_dist.sf(abs(t_stat), df=n - 1)
    return {"mean": float(mean_d), "std": float(np.sqrt(var_d)), "corrected_se": float(se), "t": float(t_stat), "p": float(p_cv)}


def compute_biomarker_stats(priors, y_task, n_rois, seeds, n_outer, n_inner, output_dir):
    """Compute biomarker statistics from FP branch coefficients."""
    from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
        build_msancr_cache, _solve_msancr_kernel,
    )
    from metascfc.experiments.prior_aware_late_fusion import fit_predict_fp_oof
    from metascfc.experiments.msancr_refinement import upper_triangle_features
    from metascfc.benchmark_utils import load_connectomes
    import yaml

    fc_mats, sc_mats, y_all, _, _ = load_connectomes(yaml.safe_load(open(Path("/home/iemiedc2026/Documents/Sanjan/iclr/configs/iclr/lf1_final_10x5.yaml"))["data"]))
    X_fc = upper_triangle_features(fc_mats)

    alignments = {"matched": [], "unrelated": [], "shuffled": [], "random": [], "no_prior": []}
    top10_jaccards = []
    rank_stabilities = []

    for seed in seeds[:3]:
        kf = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
        for outer_fold, (trainval_idx, test_idx) in enumerate(kf.split(np.arange(len(y_task)))):
            trainval_idx = np.asarray(trainval_idx, dtype=int)

            fp_cache = build_msancr_cache(
                priors["matched"], n_rois, gamma=0.5, lifting="prod",
                top_k=TOP_K, epsilon=DIAGONAL_EPSILON, weighting="binary",
                couple_modalities=False, normalize_laplacian="sym",
                edge_laplacian=build_edge_laplacian(n_rois, prior_scores=priors["matched"], top_k=TOP_K, weighting="binary", couple_modalities=False, normalize="sym"),
            )
            _, fp_info = fit_predict_fp_oof(
                X_fc, y_task, trainval_idx, trainval_idx, fp_cache,
                GAMMA_GRID, LAMBDA_L_GRID, RIDGE_GRID, n_inner, seed, outer_fold, n_rois,
            )
            fp_params = fp_info.get("best_params", {})

            scaler = StandardScaler()
            X_fc_train_z = scaler.fit_transform(X_fc[trainval_idx])

            alpha, _ = _solve_msancr_kernel(
                X_fc_train_z, np.zeros_like(X_fc_train_z),
                (y_task[trainval_idx] - y_task[trainval_idx].mean()) / max(y_task[trainval_idx].std(), 1e-8),
                fp_cache, fp_params.get("lambda_fc", 1.0), 1.0, fp_params.get("lambda_l", 0.0), fc_only=True,
            )
            fc_coeffs = alpha[:X_fc.shape[1]]

            n_fc = X_fc.shape[1]
            n_rois_actual = int((1 + np.sqrt(1 + 8 * n_fc)) / 2)
            roi_importance = np.zeros(n_rois_actual)
            idx = 0
            for i in range(n_rois_actual):
                for j in range(i + 1, n_rois_actual):
                    roi_importance[i] += abs(fc_coeffs[idx])
                    roi_importance[j] += abs(fc_coeffs[idx])
                    idx += 1
            roi_importance /= max(n_rois_actual - 1, 1)

            rho, _ = spearmanr(roi_importance, priors["matched"])
            alignments["matched"].append(float(rho))

            for ctrl_type in ["unrelated", "shuffled", "random"]:
                rho_ctrl, _ = spearmanr(roi_importance, priors[ctrl_type])
                alignments[ctrl_type].append(float(rho_ctrl))

            from sklearn.linear_model import Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_fc_train_z, y_task[trainval_idx])
            ridge_coeffs = ridge.coef_
            roi_ridge = np.zeros(n_rois_actual)
            idx = 0
            for i in range(n_rois_actual):
                for j in range(i + 1, n_rois_actual):
                    roi_ridge[i] += abs(ridge_coeffs[idx])
                    roi_ridge[j] += abs(ridge_coeffs[idx])
                    idx += 1
            roi_ridge /= max(n_rois_actual - 1, 1)
            rho_no_prior, _ = spearmanr(roi_ridge, priors["matched"])
            alignments["no_prior"].append(float(rho_no_prior))

            top10_fp = set(np.argsort(roi_importance)[-10:])
            top10_ridge = set(np.argsort(roi_ridge)[-10:])
            jaccard = len(top10_fp & top10_ridge) / len(top10_fp | top10_ridge) if len(top10_fp | top10_ridge) > 0 else 0.0
            top10_jaccards.append(float(jaccard))
            rank_stabilities.append(float(np.corrcoef(np.argsort(roi_importance), np.argsort(priors["matched"]))[0, 1]))

    result = {
        "no_prior_alignment": float(np.mean(alignments["no_prior"])),
        "matched_alignment": float(np.mean(alignments["matched"])),
        "unrelated_alignment": float(np.mean(alignments["unrelated"])),
        "shuffled_alignment": float(np.mean(alignments["shuffled"])),
        "random_alignment": float(np.mean(alignments["random"])),
        "rank_stability": float(np.mean(rank_stabilities)),
        "top10_jaccard": float(np.mean(top10_jaccards)),
        "matched_beats_negative_controls": float(np.mean(alignments["matched"])) > max(
            float(np.mean(alignments["unrelated"])),
            float(np.mean(alignments["shuffled"])),
            float(np.mean(alignments["random"])),
        ),
    }

    pd.DataFrame([result]).to_csv(Path(output_dir) / "biomarker_metrics.csv", index=False)
    return result
