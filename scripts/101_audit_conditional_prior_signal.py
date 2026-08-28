#!/usr/bin/env python3
"""Conditional / Residual Prior-Signal Audit v2 (ICLR 2027).

Fully nested, leakage-safe implementation.  All variant/eta selection
occurs via inner cross-fitting within the outer-training set.  No
outer-test labels are used for any model selection.

Usage:
    python scripts/101_audit_conditional_prior_signal.py \
        --config configs/iclr/conditional_prior_signal_v2.yaml \
        --output-dir outputs/iclr/conditional_prior_signal_v2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr

from metascfc.benchmark_utils import iter_nested_splits, load_connectomes, prediction_metrics, save_json
from metascfc.diagnostics.conditional_prior_signal import (
    crossfit_ridge,
    crossfit_residual_branch_pca,
    crossfit_residual_branch_topk,
    crossfit_residual_branch_weighted,
    fast_top_fraction,
    fit_ridge_baseline,
    residual_enrichment,
    seed_level_stats,
    select_eta_and_evaluate,
)
from metascfc.diagnostics.prior_predictive_enrichment import holm_correction, roi_to_edge_prior


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


def _build_candidate_grid(cfg: Dict) -> List[Dict]:
    """Build the identical candidate grid used for ALL prior types."""
    candidates = []
    for tf in [float(f) for f in cfg["topk_fractions"]]:
        candidates.append({"variant": f"topk_{tf}", "top_fraction": tf})
    for gamma in [float(g) for g in cfg["weighted_gammas"]]:
        candidates.append({
            "variant": f"weighted_{gamma}",
            "gamma": gamma,
            "epsilon": float(cfg["weighted_epsilon"]),
        })
    for nc in [int(c) for c in cfg["pca_n_components"]]:
        candidates.append({"variant": f"pca_{nc}", "n_components": nc})
    return candidates


# ── main audit ───────────────────────────────────────────────────────────────

def run_audit(config: Dict, override_output_dir: Optional[str] = None) -> None:
    t0 = time.time()
    out_dir = Path(override_output_dir or config["output_dir"])
    fig_dir = Path(config["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    seeds = config["seeds"]
    n_folds = config["n_folds"]
    val_frac = config["val_fraction"]
    alpha_grid = [float(a) for a in config["ridge_alphas"]]
    lifting_rules = config["lifting_rules"]
    top_fracs = [float(f) for f in config["top_fractions"]]
    n_random = int(config["n_random_subsets"])
    n_inner = int(config["n_inner_folds"])
    eta_grid = [float(e) for e in config["eta_grid"]]
    epsilon = float(config["weighted_epsilon"])

    candidates = _build_candidate_grid(config)

    fc, sc, _, _, groups = load_connectomes(config["data"])
    n_rois = int(fc.shape[1])
    x_fc, n_edges = upper_triangle_features(fc)
    x_sc, _ = upper_triangle_features(sc)
    x_all = np.concatenate([x_fc, x_sc], axis=1)
    p = x_all.shape[1]

    # Collect all results
    all_split_rows: List[Dict] = []
    all_top_frac_rows: List[Dict] = []
    all_assoc_rows: List[Dict] = []
    all_incr_split_rows: List[Dict] = []
    all_baseline_rows: List[Dict] = []
    all_hyperparam_rows: List[Dict] = []
    all_inner_candidate_rows: List[Dict] = []
    all_inner_selected_rows: List[Dict] = []

    total_evals = 0

    for target_key, target_info in config["targets"].items():
        y_full = np.load(target_info["label_path"]).astype(np.float64).reshape(-1)
        if len(y_full) != len(fc):
            print(f"SKIP {target_key}: labels mismatch")
            continue

        tp = config["priors"][target_key]
        roi_priors: Dict[str, np.ndarray] = {"matched": load_roi_prior(tp["matched"], n_rois)}
        for alt in ["unrelated", "shuffled", "random"]:
            if tp.get(alt):
                roi_priors[alt] = load_roi_prior(tp[alt], n_rois)

        for seed in seeds:
            for fold, (_, _seed, tr, va, te) in enumerate(
                iter_nested_splits(y_full, [seed], n_folds, val_frac, groups)
            ):
                tv_idx = np.concatenate([tr, va])
                y_tv = y_full[tv_idx]
                y_te = y_full[te]

                # ── 1. Cross-fitted residuals on outer-trainval ────────────
                rng_cf = np.random.RandomState(seed * 10000 + fold * 100)
                cf_pred_tv, cf_res_tv = crossfit_ridge(
                    x_all, y_full, tv_idx, n_inner, alpha_grid, rng_cf)

                # ── 2. Baseline Ridge on outer-test ───────────────────────
                rng_bl = np.random.RandomState(seed * 10000 + fold * 100 + 1)
                pred_te_bl, pred_tv_bl_ins, bl_alpha = fit_ridge_baseline(
                    x_all, y_full, tv_idx, te, alpha_grid, rng_bl)

                baseline_metrics = prediction_metrics(y_te, pred_te_bl)
                all_baseline_rows.append({
                    "target": target_key, "seed": seed, "fold": fold,
                    "baseline_pearson": baseline_metrics["pearson"],
                    "baseline_rmse": baseline_metrics["rmse"],
                    "baseline_mae": baseline_metrics["mae"],
                    "ridge_alpha": bl_alpha,
                    "n_trainval": len(tv_idx), "n_test": len(te),
                })

                # ── 3. Residual enrichment diagnostics (per prior) ────────
                for pt_name, roi_prior in roi_priors.items():
                    for rule in lifting_rules:
                        ep = roi_to_edge_prior(roi_prior, n_rois, rule)
                        ep_full = np.concatenate([ep, ep])

                        pr_a, sr_a, m_e = residual_enrichment(
                            x_all[tv_idx], cf_res_tv, ep_full)

                        tf_res = fast_top_fraction(
                            ep_full, m_e, top_fracs, n_random,
                            np.random.RandomState(seed * 1000 + fold))
                        for trr in tf_res:
                            all_top_frac_rows.append({
                                "target": target_key, "prior_type": pt_name,
                                "lifting": rule, "seed": seed, "fold": fold,
                                **trr})

                        all_split_rows.append({
                            "target": target_key, "prior_type": pt_name,
                            "lifting": rule, "seed": seed, "fold": fold,
                            "residual_pearson": pr_a, "residual_spearman": sr_a,
                            "n_trainval": len(tv_idx),
                        })

                        for mod_name, sl, mod_ep in [
                            ("FC", slice(0, n_edges), ep),
                            ("SC", slice(n_edges, None), ep),
                        ]:
                            pr_mod, _, _ = residual_enrichment(
                                x_all[tv_idx, sl], cf_res_tv, mod_ep)
                            all_assoc_rows.append({
                                "target": target_key, "prior_type": pt_name,
                                "lifting": rule, "modality": mod_name,
                                "residual_pearson": pr_mod,
                                "seed": seed, "fold": fold,
                            })

                # ── 4. Fully nested variant/eta selection for ALL priors ───
                #    For each prior: run ALL candidates with OOF, select best,
                #    then refit on full trainval and predict on test.
                for pt_name, roi_prior in roi_priors.items():
                    rng_prior = np.random.RandomState(seed * 100000 + fold * 1000 + hash(pt_name) % 10000)
                    pt_ep = roi_to_edge_prior(roi_prior, n_rois, "mean")
                    pt_ep_full = np.concatenate([pt_ep, pt_ep])

                    best_inner_pearson = -float("inf")
                    best_candidate_idx = 0
                    best_eta = 0.0

                    candidate_oof_results: List[Dict] = []

                    for ci, cand in enumerate(candidates):
                        rng_cand = np.random.RandomState(rng_prior.randint(0, 2**31))

                        # Generate OOF residual branch predictions within outer-trainval
                        if cand["variant"].startswith("topk_"):
                            res = crossfit_residual_branch_topk(
                                x_all, cf_res_tv, tv_idx, te,
                                pt_ep_full, cand["top_fraction"],
                                alpha_grid, n_inner, rng_cand)
                        elif cand["variant"].startswith("weighted_"):
                            res = crossfit_residual_branch_weighted(
                                x_all, cf_res_tv, tv_idx, te,
                                pt_ep_full, cand["gamma"], cand["epsilon"],
                                alpha_grid, n_inner, rng_cand)
                        elif cand["variant"].startswith("pca_"):
                            res = crossfit_residual_branch_pca(
                                x_all, cf_res_tv, tv_idx, te,
                                pt_ep_full, cand["n_components"],
                                alpha_grid, n_inner, rng_cand)
                        else:
                            continue

                        # Select eta using OOF predictions on outer-trainval
                        # Use cross-fitted residuals for this
                        best_eta_local, best_pearson_local = 0.0, -float("inf")
                        for eta in eta_grid:
                            combined_oof = cf_pred_tv + eta * res["oof_pred_trainval"]
                            r_local = float(np.corrcoef(y_tv, combined_oof)[0, 1])
                            if r_local > best_pearson_local:
                                best_pearson_local = r_local
                                best_eta_local = eta

                        all_inner_candidate_rows.append({
                            "target": target_key, "seed": seed, "fold": fold,
                            "prior_type": pt_name, "variant": cand["variant"],
                            "inner_pearson": best_pearson_local,
                            "selected_eta": best_eta_local,
                        })

                        candidate_oof_results.append({
                            "candidate": cand,
                            "res": res,
                            "inner_pearson": best_pearson_local,
                            "inner_eta": best_eta_local,
                        })

                        if best_pearson_local > best_inner_pearson:
                            best_inner_pearson = best_pearson_local
                            best_candidate_idx = len(candidate_oof_results) - 1
                            best_eta = best_eta_local

                    # Select best candidate
                    if not candidate_oof_results:
                        continue

                    best = candidate_oof_results[best_candidate_idx]
                    best_cand = best["candidate"]
                    best_res = best["res"]

                    # Refit best variant on FULL outer-trainval, predict on test
                    rng_refit = np.random.RandomState(seed * 100000 + fold * 1000 + 999)
                    if best_cand["variant"].startswith("topk_"):
                        final = crossfit_residual_branch_topk(
                            x_all, cf_res_tv, tv_idx, te,
                            pt_ep_full, best_cand["top_fraction"],
                            alpha_grid, n_inner, rng_refit)
                    elif best_cand["variant"].startswith("weighted_"):
                        final = crossfit_residual_branch_weighted(
                            x_all, cf_res_tv, tv_idx, te,
                            pt_ep_full, best_cand["gamma"], best_cand["epsilon"],
                            alpha_grid, n_inner, rng_refit)
                    elif best_cand["variant"].startswith("pca_"):
                        final = crossfit_residual_branch_pca(
                            x_all, cf_res_tv, tv_idx, te,
                            pt_ep_full, best_cand["n_components"],
                            alpha_grid, n_inner, rng_refit)
                    else:
                        continue

                    # Final evaluation on outer-test
                    eta_result = select_eta_and_evaluate(
                        cf_pred_tv, final["oof_pred_trainval"], y_tv,
                        pred_te_bl, final["pred_test"], y_te, eta_grid)

                    all_hyperparam_rows.append({
                        "target": target_key, "seed": seed, "fold": fold,
                        "prior_type": pt_name,
                        "selected_variant": best_cand["variant"],
                        "selected_eta": eta_result["selected_eta"],
                        "selected_alpha": final["alpha"],
                        "inner_pearson": best_inner_pearson,
                        "inner_rmse": 0.0,
                        "inner_mae": 0.0,
                    })

                    all_inner_selected_rows.append({
                        "target": target_key, "seed": seed, "fold": fold,
                        "prior_type": pt_name,
                        "selected_variant": best_cand["variant"],
                        "selected_alpha": final["alpha"],
                        "selected_eta": eta_result["selected_eta"],
                        "inner_pearson": best_inner_pearson,
                    })

                    all_incr_split_rows.append({
                        "target": target_key, "seed": seed, "fold": fold,
                        "prior_type": pt_name,
                        "variant": best_cand["variant"],
                        "selected_eta": eta_result["selected_eta"],
                        "baseline_pearson": eta_result["baseline_pearson"],
                        "baseline_rmse": eta_result["baseline_rmse"],
                        "baseline_mae": eta_result["baseline_mae"],
                        "combined_pearson": eta_result["combined_pearson"],
                        "combined_rmse": eta_result["combined_rmse"],
                        "combined_mae": eta_result["combined_mae"],
                        "delta_pearson": eta_result["delta_pearson"],
                        "delta_rmse": eta_result["delta_rmse"],
                        "delta_mae": eta_result["delta_mae"],
                    })

                    total_evals += 1
                    n_done = len(all_incr_split_rows)
                    if n_done % 200 == 0:
                        print(f"  [{time.time()-t0:.0f}s] {target_key} seed={seed} "
                              f"fold={fold} prior={pt_name} evals={total_evals}")

    elapsed = time.time() - t0
    print(f"Main loop: {elapsed:.0f}s ({total_evals} prior evaluations)")

    # ── Save raw CSVs ──────────────────────────────────────────────────────
    pd.DataFrame(all_split_rows).to_csv(out_dir / "residual_edge_association.csv", index=False)
    pd.DataFrame(all_top_frac_rows).to_csv(out_dir / "residual_top_fraction_enrichment.csv", index=False)
    pd.DataFrame(all_assoc_rows).to_csv(out_dir / "modality_conditional_enrichment.csv", index=False)
    pd.DataFrame(all_baseline_rows).to_csv(out_dir / "crossfit_baseline_residual_metrics.csv", index=False)
    pd.DataFrame(all_incr_split_rows).to_csv(out_dir / "incremental_residual_prediction_split_metrics.csv", index=False)
    pd.DataFrame(all_hyperparam_rows).to_csv(out_dir / "selected_hyperparameters.csv", index=False)
    pd.DataFrame(all_inner_candidate_rows).to_csv(out_dir / "inner_candidate_metrics.csv", index=False)
    pd.DataFrame(all_inner_selected_rows).to_csv(out_dir / "inner_selected_configs.csv", index=False)

    if not all_incr_split_rows:
        print("No incremental results.")
        return

    # ── Seed-level aggregation (average folds within seed) ─────────────────
    incr_df = pd.DataFrame(all_incr_split_rows)
    metric_cols = [
        "delta_pearson", "delta_rmse", "delta_mae",
        "baseline_pearson", "combined_pearson",
        "baseline_rmse", "combined_rmse", "selected_eta",
    ]
    seed_agg = (
        incr_df.groupby(["target", "prior_type", "seed"])
        .agg({c: "mean" for c in metric_cols if c in incr_df.columns})
        .reset_index()
    )
    seed_agg.to_csv(out_dir / "incremental_residual_prediction_seed_metrics.csv", index=False)

    # ── Summary across seeds ──────────────────────────────────────────────
    summary_rows = []
    for (tgt, pt), grp in seed_agg.groupby(["target", "prior_type"]):
        row = {"target": tgt, "prior_type": pt, "n_seeds": len(grp)}
        for c in ["delta_pearson", "delta_rmse", "combined_pearson", "baseline_pearson"]:
            if c in grp.columns:
                row[f"mean_{c}"] = float(grp[c].mean())
                row[f"std_{c}"] = float(grp[c].std())
                row[f"median_{c}"] = float(grp[c].median())
        row["n_positive_delta_pearson"] = int((grp["delta_pearson"] > 0).sum())
        row["median_delta_pearson"] = float(grp["delta_pearson"].median())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "incremental_residual_prediction_summary.csv", index=False)

    # ── Residual enrichment comparison (matched vs controls) ──────────────
    split_df = pd.DataFrame(all_split_rows)
    ctrl_comparison = []
    if not split_df.empty:
        for tgt in split_df["target"].unique():
            for rule in lifting_rules:
                m_sub = split_df[(split_df.target == tgt) & (split_df.prior_type == "matched")
                                 & (split_df.lifting == rule)]
                for ctrl in ["unrelated", "shuffled", "random"]:
                    c_sub = split_df[(split_df.target == tgt) & (split_df.prior_type == ctrl)
                                     & (split_df.lifting == rule)]
                    m_seed = m_sub.groupby("seed")["residual_pearson"].mean().values
                    c_seed = c_sub.groupby("seed")["residual_pearson"].mean().values
                    n = min(len(m_seed), len(c_seed))
                    if n < 3:
                        continue
                    st = seed_level_stats(m_seed[:n], c_seed[:n], f"matched_vs_{ctrl}")
                    st["target"] = tgt
                    st["lifting"] = rule
                    ctrl_comparison.append(st)

    if ctrl_comparison:
        cc_df = pd.DataFrame(ctrl_comparison)
        pvals = cc_df["wilcoxon_p"].values
        if len(pvals) > 1:
            cc_df["wilcoxon_p_holm"] = list(holm_correction(pvals))
        else:
            cc_df["wilcoxon_p_holm"] = pvals
        cc_df.to_csv(out_dir / "prior_control_residual_comparison.csv", index=False)

    # ── Incremental statistics (matched vs controls for delta_pearson) ────
    stat_rows = []
    for tgt in seed_agg["target"].unique():
        for ctrl in ["unrelated", "shuffled", "random"]:
            m_sub = seed_agg[(seed_agg.target == tgt) & (seed_agg.prior_type == "matched")]
            c_sub = seed_agg[(seed_agg.target == tgt) & (seed_agg.prior_type == ctrl)]
            if m_sub.empty or c_sub.empty:
                continue
            common_seeds = sorted(set(m_sub.seed) & set(c_sub.seed))
            if len(common_seeds) < 3:
                continue
            m_vals = m_sub.set_index("seed").loc[common_seeds, "delta_pearson"].values
            c_vals = c_sub.set_index("seed").loc[common_seeds, "delta_pearson"].values
            st = seed_level_stats(m_vals, c_vals, f"matched_vs_{ctrl}")
            st["target"] = tgt
            stat_rows.append(st)

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        pvals_s = stat_df["wilcoxon_p"].values
        if len(pvals_s) > 1:
            stat_df["wilcoxon_p_holm"] = list(holm_correction(pvals_s))
        else:
            stat_df["wilcoxon_p_holm"] = pvals_s
        stat_df.to_csv(out_dir / "incremental_statistics.csv", index=False)

    # ── Modality decision ─────────────────────────────────────────────────
    mod_rows = []
    assoc_df = pd.DataFrame(all_assoc_rows)
    if not assoc_df.empty:
        for tgt in assoc_df["target"].unique():
            sub = assoc_df[(assoc_df.target == tgt) & (assoc_df.prior_type == "matched")]
            mod_scores = sub.groupby("modality")["residual_pearson"].mean().to_dict()
            best_mod = max(mod_scores, key=mod_scores.get) if mod_scores else "FC+SC"
            mod_rows.append({
                "target": tgt, "best_modality": best_mod,
                "FC_residual_pearson": mod_scores.get("FC", 0.0),
                "SC_residual_pearson": mod_scores.get("SC", 0.0),
                "FC_SC_residual_pearson": sum(mod_scores.values()) / max(len(mod_scores), 1),
            })
    mod_df = pd.DataFrame(mod_rows) if mod_rows else pd.DataFrame()

    # ── Per-task decisions ────────────────────────────────────────────────
    task_decisions = {}
    for tgt in seed_agg["target"].unique():
        tgt_seed = seed_agg[seed_agg.target == tgt]
        matched_seed = tgt_seed[tgt_seed.prior_type == "matched"]
        if matched_seed.empty:
            task_decisions[tgt] = {
                "error": "no matched prior results",
                "recommended_next_step": "rebuild_prior",
            }
            continue

        dp_vals = matched_seed["delta_pearson"].values
        median_dp = float(np.median(dp_vals))
        n_pos = int(np.sum(dp_vals > 0))
        n_seeds = len(dp_vals)
        mean_dp = float(np.mean(dp_vals))

        # Residual enrichment: does matched beat shuffled/random?
        beats_shuffled = False
        beats_random = False
        beats_unrelated = False
        residual_enrichment_positive = False

        if ctrl_comparison:
            cc_df = pd.DataFrame(ctrl_comparison)
            for ctrl_name in ["shuffled", "random", "unrelated"]:
                comp_label = f"matched_vs_{ctrl_name}"
                sub = cc_df[(cc_df.target == tgt) & (cc_df.label == comp_label)]
                if not sub.empty:
                    md = float(sub["mean_diff"].values[0])
                    if ctrl_name == "shuffled" and md > 0:
                        beats_shuffled = True
                    if ctrl_name == "random" and md > 0:
                        beats_random = True
                    if ctrl_name == "unrelated" and md > 0:
                        beats_unrelated = True

        if beats_shuffled or beats_random:
            residual_enrichment_positive = True

        # Check if unrelated gives higher incremental prediction
        unrelated_seed = tgt_seed[tgt_seed.prior_type == "unrelated"]
        unrelated_dp = float(unrelated_seed["delta_pearson"].median()) if not unrelated_seed.empty else 0.0

        # Compute best modality
        best_mod = "FC+SC"
        if not mod_df.empty:
            mod_row = mod_df[mod_df.target == tgt]
            if not mod_row.empty:
                best_mod = mod_row.iloc[0]["best_modality"]

        # Decision logic
        incremental_gain = median_dp >= 0.010 and n_pos >= 8
        moderate_gain = median_dp >= 0.005 and n_pos >= 7

        if incremental_gain and (beats_shuffled or beats_random):
            next_step = "adaptive_residual_ncr"
        elif moderate_gain and (beats_shuffled or beats_random):
            next_step = "anisotropic_or_residual_ncr"
        elif residual_enrichment_positive and median_dp < 0.005:
            next_step = "anisotropic_ncr"
        elif (not beats_unrelated) and unrelated_dp >= median_dp and unrelated_dp > 0:
            next_step = "improve_task_prior_matching"
        elif not residual_enrichment_positive and abs(median_dp) < 0.005:
            next_step = "rebuild_prior"
        elif moderate_gain:
            next_step = "anisotropic_or_residual_ncr"
        else:
            next_step = "rebuild_prior"

        task_decisions[tgt] = {
            "matched_prior_residual_enrichment": residual_enrichment_positive,
            "matched_beats_unrelated": beats_unrelated,
            "matched_beats_shuffled": beats_shuffled,
            "matched_beats_random": beats_random,
            "median_delta_pearson": round(median_dp, 6),
            "mean_delta_pearson": round(mean_dp, 6),
            "positive_delta_seeds": n_pos,
            "total_seeds": n_seeds,
            "best_residual_variant": str(matched_seed.iloc[0].get("variant", "unknown")),
            "best_modality": best_mod,
            "recommended_next_step": next_step,
        }

    # ── Overall decision (preserve task asymmetry) ────────────────────────
    task_recs = {k: v.get("recommended_next_step", "rebuild_prior") for k, v in task_decisions.items()}

    # Overall: strongest recommendation across tasks
    priority_order = [
        "adaptive_residual_ncr", "anisotropic_or_residual_ncr",
        "anisotropic_ncr", "improve_task_prior_matching", "rebuild_prior",
    ]
    overall_rec = "rebuild_prior"
    for pr in priority_order:
        if any(r == pr for r in task_recs.values()):
            overall_rec = pr
            break

    summary_decision = {
        "per_task": task_decisions,
        "overall_recommended_next_step": overall_rec,
        "task_recommendations": task_recs,
    }

    (out_dir / "summary_decision.json").write_text(
        json.dumps(summary_decision, indent=2) + "\n", encoding="utf-8")
    (out_dir / "task_decisions.json").write_text(
        json.dumps(task_decisions, indent=2) + "\n", encoding="utf-8")

    save_json({
        "seeds": seeds, "n_folds": n_folds, "val_fraction": val_frac,
        "lifting_rules": lifting_rules, "top_fractions": top_fracs,
        "n_random_subsets": n_random, "n_inner_folds": n_inner,
        "topk_fractions": [float(f) for f in config["topk_fractions"]],
        "weighted_gammas": [float(g) for g in config["weighted_gammas"]],
        "weighted_epsilon": epsilon,
        "pca_n_components": [int(c) for c in config["pca_n_components"]],
        "eta_grid": eta_grid,
        "n_prior_types": len(roi_priors),
        "n_candidates_per_prior": len(candidates),
        "elapsed_seconds": time.time() - t0,
    }, out_dir / "run_metadata.json")

    (out_dir / "COMPLETE").write_text("done", encoding="utf-8")

    # ── Figures ───────────────────────────────────────────────────────────
    _make_plots(seed_agg, split_df, fig_dir, eta_grid)

    elapsed_total = time.time() - t0
    print(f"Saved audit to {out_dir} ({elapsed_total:.0f}s)")
    print(f"Overall decision: {overall_rec}")
    print(f"Per-task decisions:")
    for tgt, rec in task_recs.items():
        print(f"  {tgt}: {rec}")

    # ── Validation report ─────────────────────────────────────────────────
    _print_validation_report(out_dir, seed_agg, candidates)


def _make_plots(seed_agg: pd.DataFrame, split_df: pd.DataFrame,
                fig_dir: Path, eta_grid: Sequence[float]) -> None:
    try:
        if not seed_agg.empty:
            for tgt in seed_agg["target"].unique():
                tgt_sub = seed_agg[seed_agg.target == tgt]
                matched = tgt_sub[tgt_sub.prior_type == "matched"]
                if matched.empty:
                    continue

                # Delta Pearson by prior type
                fig, ax = plt.subplots(figsize=(8, 5))
                prior_types = tgt_sub.prior_type.unique()
                for pt in prior_types:
                    ps = tgt_sub[tgt_sub.prior_type == pt]
                    vals = ps["delta_pearson"].values
                    ax.bar(np.arange(len(vals)) + list(prior_types).index(pt) * 0.2,
                           vals, 0.2, label=pt)
                ax.set_xlabel("Seed")
                ax.set_ylabel("Delta Pearson r")
                ax.set_title(f"Incremental gain: {tgt}")
                ax.legend()
                ax.axhline(0, color="gray", ls="--", lw=0.5)
                fig.tight_layout()
                fig.savefig(fig_dir / f"delta_pearson_matched_vs_controls_{tgt}.pdf", dpi=150)
                plt.close(fig)

                # Eta distribution
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(matched["selected_eta"].values, bins=len(eta_grid), edgecolor="black")
                ax.set_xlabel("Selected eta")
                ax.set_ylabel("Count")
                ax.set_title(f"eta selection: {tgt}")
                fig.tight_layout()
                fig.savefig(fig_dir / f"selected_eta_distribution_{tgt}.pdf", dpi=150)
                plt.close(fig)

        if not split_df.empty:
            for tgt in split_df["target"].unique():
                ms = split_df[(split_df.target == tgt) & (split_df.prior_type == "matched")]
                if ms.empty:
                    continue
                summary = ms.groupby("lifting")["residual_pearson"].mean().reset_index()
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.bar(summary["lifting"], summary["residual_pearson"])
                ax.set_title(f"Residual enrichment by lifting rule: {tgt}")
                fig.tight_layout()
                fig.savefig(fig_dir / f"residual_enrichment_matched_vs_controls_{tgt}.pdf", dpi=150)
                plt.close(fig)

    except Exception as e:
        print(f"Plot error: {e}")


def _print_validation_report(out_dir: Path, seed_agg: pd.DataFrame,
                              candidates: List[Dict]) -> None:
    print("\n" + "=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    # Check seed/fold existence
    split_file = out_dir / "incremental_residual_prediction_split_metrics.csv"
    if split_file.exists():
        df = pd.read_csv(split_file)
        n_seeds = df["seed"].nunique()
        n_folds = df["fold"].nunique()
        n_targets = df["target"].nunique()
        n_priors = df["prior_type"].nunique()
        print(f"  Seeds: {n_seeds}, Folds: {n_folds}, Targets: {n_targets}, Priors: {n_priors}")

        # Check all priors have same count
        counts = df.groupby(["target", "prior_type"]).size().reset_index(name="count")
        print(f"  Per-target-prior counts: {counts['count'].tolist()}")

        # Check for NaN
        nan_count = df.isna().sum().sum()
        print(f"  NaN values in split metrics: {nan_count}")

        # Check for duplicate rows
        dup = df.duplicated(subset=["target", "seed", "fold", "prior_type"]).sum()
        print(f"  Duplicate rows: {dup}")
    else:
        print("  WARNING: split metrics file not found")

    # Check v1 outputs untouched
    v1_dir = out_dir.parent / "conditional_prior_signal"
    if v1_dir.exists():
        v1_complete = v1_dir / "COMPLETE"
        if v1_complete.exists():
            print(f"  v1 outputs intact at {v1_dir}")

    print("=" * 60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/iclr/conditional_prior_signal.yaml")
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_audit(cfg, override_output_dir=args.output_dir)


if __name__ == "__main__":
    main()
