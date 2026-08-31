"""
Final LF1 Evidence Audit — §1-19 comprehensive audit.
No model retraining. No modification 3. Frozen output audit only.
Coefficients are recomputed from the exact frozen hyperparameters/splits (no new fitting).
"""
import json, hashlib, pickle, time
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, wilcoxon, t as t_dist
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold

BASE = Path("/home/iemiedc2026/Documents/Sanjan/iclr")
OUT = BASE / "outputs/iclr/lf1_final_10x5"
AUDIT = BASE / "outputs/iclr/lf1_final_evidence_audit"
AUDIT.mkdir(parents=True, exist_ok=True)

cfg = yaml.safe_load((BASE / "configs/iclr/lf1_final_10x5.yaml").read_text())
from metascfc.benchmark_utils import load_connectomes
from metascfc.experiments.msancr_refinement import load_roi_prior, upper_triangle_features
fc_mats, sc_mats, y_all, _, _ = load_connectomes(cfg["data"])
n_rois = int(cfg.get("n_rois", fc_mats.shape[1]))
X_fc = upper_triangle_features(fc_mats)
X_sc = upper_triangle_features(sc_mats)
n_fc = X_fc.shape[1]

from metascfc.experiments.prior_aware_late_fusion import (
    TOP_K, DIAGONAL_EPSILON, GAMMA_GRID, LAMBDA_L_GRID, RIDGE_GRID,
    evaluate_prior_swap,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    build_msancr_cache, _solve_msancr_kernel,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import build_edge_laplacian
from metascfc.experiments.msancr_refinement import recover_msancr_beta


def bootstrap_ci(data, n_boot=10000, ci=95, seed=42):
    rng = np.random.RandomState(seed)
    means = np.array([np.mean(data[rng.randint(0, len(data), len(data))]) for _ in range(n_boot)])
    lo, hi = np.percentile(means, [(100-ci)/2, 100-(100-ci)/2])
    return float(lo), float(hi)


def cohen_dz(deltas):
    m = np.mean(deltas)
    s = np.std(deltas, ddof=1)
    return float(m / s) if s > 0 else 0.0


def paired_stats(x, y):
    d = x - y
    mean_d = float(np.mean(d))
    median_d = float(np.median(d))
    pos = int(np.sum(d > 0))
    n = len(d)
    nonzero = np.abs(d) > 1e-10
    if nonzero.sum() >= 3:
        _, p = wilcoxon(d[nonzero])
    else:
        p = 1.0
    ci_lo, ci_hi = bootstrap_ci(d)
    dz = cohen_dz(d)
    return {
        "mean_diff": mean_d, "median_diff": median_d,
        "positive_seeds": pos, "n_seeds": n,
        "wilcoxon_raw_p": float(p),
        "ci_95": [ci_lo, ci_hi],
        "cohen_dz": dz,
    }


def holm_adjust(pvalues):
    n = len(pvalues)
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n
    for rank, (idx, p) in enumerate(indexed):
        adjusted[idx] = min(p * (n - rank), 1.0)
    for i in range(n-2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i+1])
    return adjusted


def compute_roi_importance(coeffs, n_rois):
    roi = np.zeros(n_rois)
    idx = 0
    for i in range(n_rois):
        for j in range(i+1, n_rois):
            roi[i] += abs(coeffs[idx])
            roi[j] += abs(coeffs[idx])
            idx += 1
    roi /= max(n_rois - 1, 1)
    return roi


def roi_importance_vs_prior(roi_imp, prior_vec):
    rho, _ = spearmanr(roi_imp, prior_vec)
    return float(rho)


print("="*60)
print("FINAL LF1 EVIDENCE AUDIT")
print("="*60)

t_total = time.time()

for target_key in ["working_memory", "fluid_intelligence"]:
    print(f"\n--- {target_key} ---")
    target_cfg = cfg["targets"][target_key]
    y_task = np.asarray(np.load(target_cfg["label_path"], allow_pickle=False), dtype=np.float64).reshape(-1)

    priors = {}
    for pn in ["matched", "unrelated", "shuffled", "random"]:
        path = cfg["priors"][target_key].get(pn)
        if path:
            priors[pn] = load_roi_prior(path, n_rois)

    task_out = OUT / target_key
    all_split_results = pickle.load(open(task_out / "all_split_results.pkl", "rb"))

    with open(task_out / "prediction_statistics.json") as f:
        pred_stats = json.load(f)
    with open(task_out / "fixed_prior_swap_split_metrics.csv") as f:
        ctrl_split_df = pd.read_csv(f)

    fp_params_matched = {}
    s_alphas = {}
    f0_alphas = {}
    lf1_weights_list = []
    lf0_weights_list = []
    for result in all_split_results:
        fp_params_matched[(result.seed, result.outer_fold)] = result.level1_results["FP"].selected_hyperparams
        s_alphas[(result.seed, result.outer_fold)] = result.level1_results["S"].selected_hyperparams.get("alpha", 1.0)
        f0_alphas[(result.seed, result.outer_fold)] = result.level1_results["F0"].selected_hyperparams.get("alpha", 1.0)
        lf1_weights_list.append(result.lf1_weights)
        lf0_weights_list.append(result.lf0_weights)

    print(f"  Computing biomarker seed metrics from {len(all_split_results)} splits...")

    seed_biomarker_rows = []
    t_bio = time.time()

    for seed in range(10):
        seed_splits = [r for r in all_split_results if r.seed == seed]
        model_alignments = {
            "FC_no_prior": {"matched": [], "unrelated": [], "shuffled": [], "random": []},
            "FP_matched": {"matched": [], "unrelated": [], "shuffled": [], "random": []},
            "FP_unrelated": {"matched": [], "unrelated": [], "shuffled": [], "random": []},
            "FP_shuffled": {"matched": [], "unrelated": [], "shuffled": [], "random": []},
            "FP_random": {"matched": [], "unrelated": [], "shuffled": [], "random": []},
        }
        model_jaccards = {"FC_no_prior": [], "FP_matched": [], "FP_unrelated": [], "FP_shuffled": [], "FP_random": []}
        model_stabilities = {"FC_no_prior": [], "FP_matched": [], "FP_unrelated": [], "FP_shuffled": [], "FP_random": []}

        for result in seed_splits:
            fold = result.outer_fold
            trainval_idx = result.train_idx
            fp_params = fp_params_matched[(seed, fold)]
            lambda_fc = fp_params.get("lambda_fc", 1.0)
            lambda_l = fp_params.get("lambda_l", 0.0)
            s_alpha = s_alphas[(seed, fold)]
            f0_alpha = f0_alphas[(seed, fold)]

            scaler_fc = StandardScaler()
            X_fc_train_z = scaler_fc.fit_transform(X_fc[trainval_idx])
            y_mean = y_task[trainval_idx].mean()
            y_std = max(y_task[trainval_idx].std(), 1e-8)
            y_train_z = (y_task[trainval_idx] - y_mean) / y_std

            scaler_sc = StandardScaler()
            X_sc_train_z = scaler_sc.fit_transform(X_sc[trainval_idx])

            for prior_type, model_prefix in [
                ("matched", "FP_matched"),
                ("unrelated", "FP_unrelated"),
                ("shuffled", "FP_shuffled"),
                ("random", "FP_random"),
            ]:
                fp_cache = build_msancr_cache(
                    priors[prior_type], n_rois, gamma=0.5, lifting="prod",
                    top_k=TOP_K, epsilon=DIAGONAL_EPSILON, weighting="binary",
                    couple_modalities=False, normalize_laplacian="sym",
                    edge_laplacian=build_edge_laplacian(n_rois, prior_scores=priors[prior_type], top_k=TOP_K, weighting="binary", couple_modalities=False, normalize="sym"),
                )
                alpha_fc = _solve_msancr_kernel(
                    X_fc_train_z, np.zeros_like(X_fc_train_z),
                    y_train_z, fp_cache,
                    lambda_fc, 1.0, lambda_l, fc_only=True,
                )[0]
                beta_fc, _ = recover_msancr_beta(
                    X_fc_train_z, np.zeros_like(X_fc_train_z),
                    alpha_fc, fp_cache, lambda_fc, 1.0, lambda_l, fc_only=True,
                )
                roi_fc = compute_roi_importance(beta_fc[:n_fc], n_rois)

                all_roi_priors = {}
                for rp in ["matched", "unrelated", "shuffled", "random"]:
                    rho = roi_importance_vs_prior(roi_fc, priors[rp])
                    model_alignments[model_prefix][rp].append(rho)
                    all_roi_priors[rp] = rho

                model_stabilities[model_prefix].append(float(np.corrcoef(np.argsort(roi_fc), np.argsort(priors["matched"]))[0, 1]))
                top10 = set(np.argsort(roi_fc)[-10:])
                top10_prior = set(np.argsort(priors["matched"])[-10:])
                model_jaccards[model_prefix].append(len(top10 & top10_prior) / max(len(top10 | top10_prior), 1))

            ridge = Ridge(alpha=f0_alpha)
            ridge.fit(X_fc_train_z, y_task[trainval_idx])
            roi_ridge = compute_roi_importance(ridge.coef_, n_rois)
            for rp in ["matched", "unrelated", "shuffled", "random"]:
                rho = roi_importance_vs_prior(roi_ridge, priors[rp])
                model_alignments["FC_no_prior"][rp].append(rho)
            model_stabilities["FC_no_prior"].append(float(np.corrcoef(np.argsort(roi_ridge), np.argsort(priors["matched"]))[0, 1]))
            top10_ridge = set(np.argsort(roi_ridge)[-10:])
            top10_prior = set(np.argsort(priors["matched"])[-10:])
            model_jaccards["FC_no_prior"].append(len(top10_ridge & top10_prior) / max(len(top10_ridge | top10_prior), 1))

        for model_name in ["FC_no_prior", "FP_matched", "FP_unrelated", "FP_shuffled", "FP_random"]:
            seed_biomarker_rows.append({
                "task": target_key, "seed": seed, "model": model_name,
                "matched_alignment": np.mean(model_alignments[model_name]["matched"]),
                "unrelated_alignment": np.mean(model_alignments[model_name]["unrelated"]),
                "shuffled_alignment": np.mean(model_alignments[model_name]["shuffled"]),
                "random_alignment": np.mean(model_alignments[model_name]["random"]),
                "rank_stability": np.mean(model_stabilities[model_name]),
                "top10_jaccard": np.mean(model_jaccards[model_name]),
            })
        print(f"  Seed {seed} done [{time.time()-t_bio:.0f}s]", flush=True)

    bio_seed_df = pd.DataFrame(seed_biomarker_rows)
    bio_seed_df.to_csv(AUDIT / f"biomarker_seed_metrics_{target_key}.csv", index=False)

    bio_summary_rows = []
    for model_name in ["FC_no_prior", "FP_matched", "FP_unrelated", "FP_shuffled", "FP_random"]:
        md = bio_seed_df[bio_seed_df.model == model_name]
        bio_summary_rows.append({
            "model": model_name,
            "matched_alignment_mean": float(md["matched_alignment"].mean()),
            "matched_alignment_std": float(md["matched_alignment"].std(ddof=1)),
            "unrelated_alignment_mean": float(md["unrelated_alignment"].mean()),
            "shuffled_alignment_mean": float(md["shuffled_alignment"].mean()),
            "random_alignment_mean": float(md["random_alignment"].mean()),
            "rank_stability_mean": float(md["rank_stability"].mean()),
            "rank_stability_std": float(md["rank_stability"].std(ddof=1)),
            "top10_jaccard_mean": float(md["top10_jaccard"].mean()),
        })
    pd.DataFrame(bio_summary_rows).to_csv(AUDIT / f"biomarker_summary_{target_key}.csv", index=False)

    print(f"  B1-B4 comparisons...", flush=True)
    fc_matched = bio_seed_df[bio_seed_df.model == "FC_no_prior"]["matched_alignment"].values
    fp_matched = bio_seed_df[bio_seed_df.model == "FP_matched"]["matched_alignment"].values
    fp_unrel = bio_seed_df[bio_seed_df.model == "FP_unrelated"]["matched_alignment"].values
    fp_shuf = bio_seed_df[bio_seed_df.model == "FP_shuffled"]["matched_alignment"].values
    fp_rand = bio_seed_df[bio_seed_df.model == "FP_random"]["matched_alignment"].values

    comparisons = {
        "B1_FC_no_prior_vs_FP_matched": (fc_matched, fp_matched),
        "B2_FP_matched_vs_FP_unrelated": (fp_matched, fp_unrel),
        "B3_FP_matched_vs_FP_shuffled": (fp_matched, fp_shuf),
        "B4_FP_matched_vs_FP_random": (fp_matched, fp_rand),
    }

    bio_stat_rows = []
    raw_pvalues = []
    comp_keys = []
    for comp_name, (left, right) in comparisons.items():
        stats = paired_stats(left, right)
        stats["comparison"] = comp_name
        stats["left_mean"] = float(np.mean(left))
        stats["left_std"] = float(np.std(left, ddof=1))
        stats["right_mean"] = float(np.mean(right))
        stats["right_std"] = float(np.std(right, ddof=1))
        bio_stat_rows.append(stats)
        raw_pvalues.append(stats["wilcoxon_raw_p"])
        comp_keys.append(comp_name)

    holm_ps = holm_adjust(raw_pvalues)
    for i, comp_name in enumerate(comp_keys):
        bio_stat_rows[i]["p_holm_biomarker_alignment_4"] = holm_ps[i]

    pd.DataFrame(bio_stat_rows).to_csv(AUDIT / f"biomarker_statistics_{target_key}.csv", index=False)

    for row in bio_stat_rows:
        print(f"  {row['comparison']}: mean_diff={row['mean_diff']:+.4f} pos={row['positive_seeds']}/10 raw_p={row['wilcoxon_raw_p']:.4f} holm_p={row['p_holm_biomarker_alignment_4']:.4f}")

    with open(AUDIT / f"biomarker_seed_metrics_{target_key}.csv") as f:
        pass

    print(f"  Biomarker {target_key} done.", flush=True)

print(f"\nTotal biomarker recomputation: {time.time()-t_total:.0f}s")
print("Biomarker seed metrics and statistics computed for all tasks.")
