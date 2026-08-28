#!/usr/bin/env python3
"""Conditional / Residual Prior-Signal Audit (ICLR 2027).

Determines whether the matched prior contains predictive information
beyond ordinary FC+SC Ridge via cross-fitted residual analysis.

Usage:
    python scripts/101_audit_conditional_prior_signal.py \
        --config configs/iclr/conditional_prior_signal.yaml
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from metascfc.benchmark_utils import iter_nested_splits, load_connectomes, save_json
from metascfc.diagnostics.conditional_prior_signal import (
    compute_additive_metrics,
    crossfit_ridge,
    fast_top_fraction,
    fit_prior_pca_ridge,
    fit_prior_topk_ridge,
    fit_prior_weighted_ridge,
    fit_ridge_baseline,
    residual_enrichment,
    seed_level_stats,
)
from metascfc.diagnostics.prior_predictive_enrichment import (
    holm_correction,
    roi_to_edge_prior,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def upper_triangle_features(mats):
    iu = np.triu_indices(mats.shape[1], k=1)
    return mats[:, iu[0], iu[1]].astype(np.float64), len(iu[0])


def load_roi_prior(path, n_rois):
    df = pd.read_csv(path)
    if "roi_index" in df.columns:
        df = df.sort_values("roi_index")
    p = df["prior_score"].to_numpy(np.float64)
    if p.shape != (n_rois,):
        raise ValueError(f"Prior {path}: shape {p.shape} != ({n_rois},)")
    return p


# ── main audit ───────────────────────────────────────────────────────────────

def run_audit(config: Dict) -> None:
    t0 = time.time()
    out_dir = Path(config["output_dir"])
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
    topk_fracs = [float(f) for f in config["topk_fractions"]]
    weighted_gammas = [float(g) for g in config["weighted_gammas"]]
    weighted_eps = float(config["weighted_epsilon"])
    pca_ncomps = [int(c) for c in config["pca_n_components"]]
    eta_grid = [float(e) for e in config["eta_grid"]]

    fc, sc, _, _, groups = load_connectomes(config["data"])
    n_rois = int(fc.shape[1])
    x_fc, n_edges = upper_triangle_features(fc)
    x_sc, _ = upper_triangle_features(sc)
    x_all = np.concatenate([x_fc, x_sc], axis=1)

    split_rows: List[Dict] = []
    top_frac_rows: List[Dict] = []
    assoc_rows: List[Dict] = []
    incr_split_rows: List[Dict] = []
    baseline_rows: List[Dict] = []
    hyperparam_rows: List[Dict] = []

    for target_key, target_info in config["targets"].items():
        y_full = np.load(target_info["label_path"]).astype(np.float64).reshape(-1)
        if len(y_full) != len(fc):
            print(f"SKIP {target_key}: labels mismatch")
            continue

        tp = config["priors"][target_key]
        roi_priors: Dict[str, np.ndarray] = {
            "matched": load_roi_prior(tp["matched"], n_rois),
        }
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

                # ── 1. Cross-fitted residuals on trainval ──────────────────
                rng_cf = np.random.RandomState(seed * 10000 + fold * 100)
                cf_pred_tv, cf_res_tv = crossfit_ridge(
                    x_all, y_full, tv_idx, n_inner, alpha_grid, rng_cf)

                # ── 2. Baseline Ridge on test ─────────────────────────────
                rng_bl = np.random.RandomState(seed * 10000 + fold * 100 + 1)
                pred_te, pred_tv_ins, bl_alpha = fit_ridge_baseline(
                    x_all, y_full, tv_idx, te, alpha_grid, rng_bl)

                baseline_rows.append({
                    "target": target_key, "seed": seed, "fold": fold,
                    "baseline_pearson": float(pearsonr(y_te, pred_te).statistic)
                    if len(y_te) > 1 else 0.0,
                    "baseline_rmse": float(np.sqrt(np.mean((y_te - pred_te) ** 2))),
                    "baseline_mae": float(np.mean(np.abs(y_te - pred_te))),
                    "ridge_alpha": bl_alpha,
                    "n_trainval": len(tv_idx), "n_test": len(te),
                })

                # ── 3. Diagnostics A / C per prior type × lifting ─────────
                for pt_name, roi_prior in roi_priors.items():
                    for rule in lifting_rules:
                        ep = roi_to_edge_prior(roi_prior, n_rois, rule)
                        ep_full = np.concatenate([ep, ep])

                        # Diagnostic A — residual enrichment (on trainval)
                        pr_a, sr_a, m_e = residual_enrichment(
                            x_all[tv_idx], cf_res_tv, ep_full)

                        # Diagnostic C — top-fraction enrichment
                        tf_res = fast_top_fraction(
                            ep_full, m_e, top_fracs, n_random,
                            np.random.RandomState(seed * 1000 + fold))
                        for trr in tf_res:
                            top_frac_rows.append({
                                "target": target_key, "prior_type": pt_name,
                                "lifting": rule, "seed": seed, "fold": fold,
                                **trr})

                        split_rows.append({
                            "target": target_key, "prior_type": pt_name,
                            "lifting": rule, "seed": seed, "fold": fold,
                            "residual_pearson": pr_a,
                            "residual_spearman": sr_a,
                            "n_trainval": len(tv_idx),
                        })

                        # Modality-specific association
                        for mod_name, sl, mod_ep in [
                            ("FC", slice(0, n_edges), ep),
                            ("SC", slice(n_edges, None), ep),
                        ]:
                            pr_mod, _, _ = residual_enrichment(
                                x_all[tv_idx, sl], cf_res_tv, mod_ep)
                            assoc_rows.append({
                                "target": target_key, "prior_type": pt_name,
                                "lifting": rule, "modality": mod_name,
                                "residual_pearson": pr_mod,
                                "seed": seed, "fold": fold,
                            })

                # ── 4. Diagnostic D — prior-only residual models ──────────
                best_variant, best_delta = None, -float("inf")
                best_prior_pred_test = None
                best_prior_pred_tv = None
                best_hyper: Dict = {}

                rng_d = np.random.RandomState(seed * 10000 + fold * 100 + 2)
                matched_ep = roi_to_edge_prior(roi_priors["matched"], n_rois, "mean")
                matched_ep_full = np.concatenate([matched_ep, matched_ep])

                for tf in topk_fracs:
                    res = fit_prior_topk_ridge(
                        x_all, cf_res_tv, tv_idx, te,
                        matched_ep_full, tf, alpha_grid, rng=rng_d)
                    delta = compute_additive_metrics(
                        pred_te, res["pred_test"], y_te,
                        cf_pred_tv, res["pred_trainval"], y_tv, eta_grid)
                    if delta["delta_pearson"] > best_delta:
                        best_delta = delta["delta_pearson"]
                        best_variant = f"topk_{tf}"
                        best_prior_pred_test = res["pred_test"]
                        best_prior_pred_tv = res["pred_trainval"]
                        best_hyper = {"variant": "topk", "top_fraction": tf,
                                      "best_alpha": res["best_alpha"]}
                    incr_split_rows.append({
                        "target": target_key, "seed": seed, "fold": fold,
                        "prior_type": "matched", "variant": f"topk_{tf}",
                        **delta})

                for gamma in weighted_gammas:
                    res = fit_prior_weighted_ridge(
                        x_all, cf_res_tv, tv_idx, te,
                        matched_ep_full, gamma, weighted_eps, alpha_grid,
                        rng=np.random.RandomState(seed * 10000 + fold * 100 + 3))
                    delta = compute_additive_metrics(
                        pred_te, res["pred_test"], y_te,
                        cf_pred_tv, res["pred_trainval"], y_tv, eta_grid)
                    if delta["delta_pearson"] > best_delta:
                        best_delta = delta["delta_pearson"]
                        best_variant = f"weighted_{gamma}"
                        best_prior_pred_test = res["pred_test"]
                        best_prior_pred_tv = res["pred_trainval"]
                        best_hyper = {"variant": "weighted", "gamma": gamma,
                                      "best_alpha": res["best_alpha"]}
                    incr_split_rows.append({
                        "target": target_key, "seed": seed, "fold": fold,
                        "prior_type": "matched", "variant": f"weighted_{gamma}",
                        **delta})

                for nc in pca_ncomps:
                    res = fit_prior_pca_ridge(
                        x_all, cf_res_tv, tv_idx, te,
                        matched_ep_full, nc, alpha_grid,
                        rng=np.random.RandomState(seed * 10000 + fold * 100 + 4))
                    delta = compute_additive_metrics(
                        pred_te, res["pred_test"], y_te,
                        cf_pred_tv, res["pred_trainval"], y_tv, eta_grid)
                    if delta["delta_pearson"] > best_delta:
                        best_delta = delta["delta_pearson"]
                        best_variant = f"pca_{nc}"
                        best_prior_pred_test = res["pred_test"]
                        best_prior_pred_tv = res["pred_trainval"]
                        best_hyper = {"variant": "pca", "n_components": nc,
                                      "best_alpha": res["best_alpha"],
                                      "explained_variance": res["explained_variance"]}
                    incr_split_rows.append({
                        "target": target_key, "seed": seed, "fold": fold,
                        "prior_type": "matched", "variant": f"pca_{nc}",
                        **delta})

                hyperparam_rows.append({
                    "target": target_key, "seed": seed, "fold": fold,
                    "best_variant": best_variant, **best_hyper,
                })

                # ── 5. Diagnostic F — control priors incremental ──────────
                for ctrl_name in ["unrelated", "shuffled", "random"]:
                    if ctrl_name not in roi_priors:
                        continue
                    ctrl_ep = roi_to_edge_prior(roi_priors[ctrl_name], n_rois, "mean")
                    ctrl_ep_full = np.concatenate([ctrl_ep, ctrl_ep])
                    ctrl_res = fit_prior_topk_ridge(
                        x_all, cf_res_tv, tv_idx, te,
                        ctrl_ep_full, 0.10, alpha_grid,
                        rng=np.random.RandomState(seed * 10000 + fold * 100 + 5))
                    ctrl_delta = compute_additive_metrics(
                        pred_te, ctrl_res["pred_test"], y_te,
                        cf_pred_tv, ctrl_res["pred_trainval"], y_tv, eta_grid)
                    incr_split_rows.append({
                        "target": target_key, "seed": seed, "fold": fold,
                        "prior_type": ctrl_name, "variant": "topk_0.10",
                        **ctrl_delta})

                n_done = len(split_rows)
                if n_done % 200 == 0:
                    print(f"  [{time.time()-t0:.0f}s] {target_key} seed={seed} "
                          f"fold={fold} rows={n_done}")

    elapsed = time.time() - t0
    print(f"Main loop: {elapsed:.0f}s ({len(split_rows)} evaluations)")

    # ── Save raw CSVs ────────────────────────────────────────────────────────
    pd.DataFrame(split_rows).to_csv(out_dir / "residual_edge_association.csv", index=False)
    pd.DataFrame(top_frac_rows).to_csv(out_dir / "residual_top_fraction_enrichment.csv", index=False)
    pd.DataFrame(assoc_rows).to_csv(out_dir / "modality_conditional_enrichment.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(out_dir / "crossfit_baseline_residual_metrics.csv", index=False)
    pd.DataFrame(incr_split_rows).to_csv(out_dir / "incremental_residual_prediction_split_metrics.csv", index=False)
    pd.DataFrame(hyperparam_rows).to_csv(out_dir / "selected_hyperparameters.csv", index=False)

    if not incr_split_rows:
        print("No incremental results.")
        return

    # ── Seed-level aggregation ───────────────────────────────────────────────
    incr_df = pd.DataFrame(incr_split_rows)
    metric_cols = ["delta_pearson", "delta_rmse", "delta_mae",
                   "baseline_pearson", "combined_pearson",
                   "baseline_rmse", "combined_rmse", "selected_eta"]
    agg_dict = {}
    for c in metric_cols:
        if c in incr_df.columns:
            agg_dict[f"{c}_mean"] = (c, "mean")
            agg_df = True
    seed_agg = (
        incr_df.groupby(["target", "prior_type", "variant", "seed"])
        .agg(**{f"{c}_mean": (c, "mean") for c in metric_cols if c in incr_df.columns})
        .reset_index())
    seed_agg.to_csv(out_dir / "incremental_residual_prediction_seed_metrics.csv", index=False)

    # ── Summary across seeds ─────────────────────────────────────────────────
    summary_rows = []
    for (tgt, pt, var), grp in seed_agg.groupby(["target", "prior_type", "variant"]):
        row = {"target": tgt, "prior_type": pt, "variant": var, "n_seeds": len(grp)}
        for c in ["delta_pearson_mean", "delta_rmse_mean", "delta_mae_mean",
                   "combined_pearson_mean", "baseline_pearson_mean"]:
            if c in grp.columns:
                row[f"{c}_across_seeds_mean"] = float(grp[c].mean())
                row[f"{c}_across_seeds_std"] = float(grp[c].std())
        row["n_positive_delta_pearson"] = int((grp["delta_pearson_mean"] > 0).sum())
        row["median_delta_pearson"] = float(grp["delta_pearson_mean"].median())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "incremental_residual_prediction_summary.csv", index=False)

    # ── Residual enrichment comparison (matched vs controls) ─────────────────
    split_df = pd.DataFrame(split_rows)
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
                    st = seed_level_stats(m_seed[:n], c_seed[:n], ctrl)
                    st["target"] = tgt
                    st["lifting"] = rule
                    ctrl_comparison.append(st)

    if ctrl_comparison:
        cc_df = pd.DataFrame(ctrl_comparison)
        pvals = cc_df["wilcoxon_p"].values
        if len(pvals) > 1:
            cc_df["holm_p"] = list(holm_correction(pvals))
        else:
            cc_df["holm_p"] = pvals
        cc_df.to_csv(out_dir / "prior_control_residual_comparison.csv", index=False)

    # ── Statistics: matched vs no-prior (delta_pearson) ──────────────────────
    stat_rows = []
    best_variants = seed_agg[seed_agg.prior_type == "matched"].groupby(
        ["target", "variant"])["delta_pearson_mean"].mean().reset_index()
    for tgt in seed_agg["target"].unique():
        tgt_sub = seed_agg[(seed_agg.target == tgt) & (seed_agg.prior_type == "matched")]
        if tgt_sub.empty:
            continue
        best_var = (tgt_sub.groupby("variant")["delta_pearson_mean"].mean().idxmax()
                    if len(tgt_sub) > 0 else None)
        if best_var is None:
            continue
        bv_sub = tgt_sub[tgt_sub.variant == best_var]
        dp = bv_sub["delta_pearson_mean"].values
        dr = bv_sub["delta_rmse_mean"].values
        n_pos = int(np.sum(dp > 0))
        stat_rows.append({
            "target": tgt, "variant": best_var,
            "mean_delta_pearson": float(np.mean(dp)),
            "std_delta_pearson": float(np.std(dp)),
            "median_delta_pearson": float(np.median(dp)),
            "n_positive_seeds": n_pos,
            "n_seeds": len(dp),
        })

    # matched vs control priors (best variant)
    for tgt in seed_agg["target"].unique():
        matched_sub = seed_agg[(seed_agg.target == tgt) & (seed_agg.prior_type == "matched")]
        if matched_sub.empty:
            continue
        best_var = matched_sub.groupby("variant")["delta_pearson_mean"].mean().idxmax()
        m_vals = matched_sub[matched_sub.variant == best_var]["delta_pearson_mean"].values
        for ctrl in ["unrelated", "shuffled", "random"]:
            ctrl_sub = seed_agg[(seed_agg.target == tgt) & (seed_agg.prior_type == ctrl)]
            if ctrl_sub.empty:
                continue
            c_vals = ctrl_sub[ctrl_sub.variant == best_var]["delta_pearson_mean"].values
            n = min(len(m_vals), len(c_vals))
            if n < 3:
                continue
            st = seed_level_stats(m_vals[:n], c_vals[:n], f"matched_vs_{ctrl}")
            st["target"] = tgt
            st["variant"] = best_var
            stat_rows.append(st)

    if stat_rows:
        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(out_dir / "incremental_statistics.csv", index=False)

    # ── Decision ─────────────────────────────────────────────────────────────
    decision = _make_decision(seed_agg, ctrl_comparison, stat_rows,
                              best_variants, summary_df)

    (out_dir / "summary_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    save_json({
        "seeds": seeds, "n_folds": n_folds, "val_fraction": val_frac,
        "lifting_rules": lifting_rules, "top_fractions": top_fracs,
        "n_random_subsets": n_random, "n_inner_folds": n_inner,
        "topk_fractions": topk_fracs, "weighted_gammas": weighted_gammas,
        "pca_n_components": pca_ncomps, "eta_grid": eta_grid,
    }, out_dir / "run_metadata.json")
    (out_dir / "COMPLETE").write_text("done", encoding="utf-8")

    _make_plots(seed_agg, split_df, fig_dir)

    elapsed_total = time.time() - t0
    print(f"Saved audit to {out_dir} ({elapsed_total:.0f}s)")
    print(f"Decision: {json.dumps(decision, indent=2)}")


def _make_decision(seed_agg, ctrl_comparison, stat_rows, best_variants, summary_df):
    matched = seed_agg[seed_agg.prior_type == "matched"]
    if matched.empty:
        return {"recommended_next_step": "rebuild_prior", "error": "no matched results"}

    best_var = (matched.groupby("variant")["delta_pearson_mean"].mean().idxmax()
                if len(matched) > 0 else "unknown")

    best_sub = matched[matched.variant == best_var]
    dp_vals = best_sub["delta_pearson_mean"].values
    median_dp = float(np.median(dp_vals)) if len(dp_vals) > 0 else 0.0
    n_pos = int(np.sum(dp_vals > 0)) if len(dp_vals) > 0 else 0
    n_seeds = len(dp_vals)

    residual_enrichment_positive = False
    beats_shuffled = False
    beats_random = False
    beats_unrelated = False
    if ctrl_comparison:
        cc_df = pd.DataFrame(ctrl_comparison)
        for ctrl in ["shuffled", "random", "unrelated"]:
            sub = cc_df[cc_df.label == f"matched_vs_{ctrl}"]
            if not sub.empty:
                md = float(sub["mean_diff"].values[0])
                if ctrl == "shuffled" and md > 0:
                    beats_shuffled = True
                if ctrl == "random" and md > 0:
                    beats_random = True
                if ctrl == "unrelated" and md > 0:
                    beats_unrelated = True
    if beats_shuffled or beats_random:
        residual_enrichment_positive = True

    incremental_gain = median_dp >= 0.010 and n_pos >= 7

    best_modality = "FC+SC"

    # Check if unrelated beats matched
    unrelated_sub = seed_agg[(seed_agg.prior_type == "unrelated") & (seed_agg.variant == best_var)]
    if not unrelated_sub.empty:
        udp = unrelated_sub["delta_pearson_mean"].values
        if len(udp) > 0 and float(np.median(udp)) > median_dp:
            next_step = "improve_task_prior_matching"
        elif incremental_gain:
            if best_var.startswith("pca"):
                next_step = "low_rank_residual_ncr"
            else:
                next_step = "residual_ncr"
        else:
            next_step = "anisotropic_ncr"
    elif incremental_gain:
        if best_var.startswith("pca"):
            next_step = "low_rank_residual_ncr"
        else:
            next_step = "residual_ncr"
    elif residual_enrichment_positive:
        next_step = "anisotropic_ncr"
    else:
        next_step = "rebuild_prior"

    return {
        "matched_prior_residual_enrichment": residual_enrichment_positive,
        "matched_beats_unrelated_on_residual_signal": beats_unrelated,
        "matched_beats_shuffled_random": beats_shuffled or beats_random,
        "incremental_prediction_gain": incremental_gain,
        "median_delta_pearson": round(median_dp, 6),
        "positive_delta_seeds": n_pos,
        "total_seeds": n_seeds,
        "best_modality": best_modality,
        "best_residual_variant": best_var,
        "recommended_next_step": next_step,
    }


def _make_plots(seed_agg, split_df, fig_dir):
    try:
        if not seed_agg.empty:
            matched = seed_agg[seed_agg.prior_type == "matched"]
            if not matched.empty:
                for tgt in matched["target"].unique():
                    tm = matched[matched.target == tgt]
                    best_var = tm.groupby("variant")["delta_pearson_mean"].mean().idxmax()
                    bv = tm[tm.variant == best_var]
                    prior_types = seed_agg.prior_type.unique()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    x = np.arange(len(bv))
                    w = 0.8 / max(len(prior_types), 1)
                    for i, pt in enumerate(prior_types):
                        ps = seed_agg[(seed_agg.target == tgt) & (seed_agg.prior_type == pt)
                                      & (seed_agg.variant == best_var)]
                        if not ps.empty:
                            ax.bar(x + i * w, ps["delta_pearson_mean"].values, w, label=pt)
                    ax.set_xlabel("Seed")
                    ax.set_ylabel("Delta Pearson r")
                    ax.set_title(f"Incremental gain: {tgt} ({best_var})")
                    ax.legend()
                    ax.axhline(0, color="gray", ls="--", lw=0.5)
                    fig.tight_layout()
                    fig.savefig(fig_dir / f"incremental_delta_pearson_by_prior_{tgt}.pdf", dpi=150)
                    plt.close(fig)

            for tgt in matched["target"].unique():
                tm = matched[matched.target == tgt]
                best_var = tm.groupby("variant")["delta_pearson_mean"].mean().idxmax()
                bv = tm[tm.variant == best_var]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(bv["selected_eta_mean"].values, bins=len(config_eta_grid), edgecolor="black")
                ax.set_xlabel("Selected eta")
                ax.set_ylabel("Count")
                ax.set_title(f"eta selection: {tgt} ({best_var})")
                fig.tight_layout()
                fig.savefig(fig_dir / f"eta_selection_distribution_{tgt}.pdf", dpi=150)
                plt.close(fig)

        if not split_df.empty:
            matched_s = split_df[split_df.prior_type == "matched"]
            if not matched_s.empty:
                for tgt in matched_s["target"].unique():
                    ms = matched_s[matched_s.target == tgt]
                    summary = ms.groupby(["lifting", "prior_type"])["residual_pearson"].mean().reset_index()
                    fig, ax = plt.subplots(figsize=(8, 5))
                    pts = summary.prior_type.unique()
                    for i, pt in enumerate(pts):
                        ps = summary[summary.prior_type == pt]
                        ax.bar(np.arange(len(ps)) + i * 0.2, ps["residual_pearson"].values,
                               0.2, label=pt)
                    ax.set_title(f"Residual enrichment: {tgt}")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(fig_dir / f"matched_vs_control_residual_enrichment_{tgt}.pdf", dpi=150)
                    plt.close(fig)

    except Exception as e:
        print(f"Plot error: {e}")


config_eta_grid = [0.0, 0.25, 0.5, 0.75, 1.0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/iclr/conditional_prior_signal.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_audit(cfg)


if __name__ == "__main__":
    main()
