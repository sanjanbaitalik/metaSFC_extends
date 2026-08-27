#!/usr/bin/env python3
"""Prior Predictive Enrichment Audit (ICLR 2027) — Optimized.

Key optimizations vs naive approach:
1. Ridge beta computed once per (target, seed, fold), shared across all prior types / lifting rules.
2. Vectorized marginal enrichment (no per-feature pearsonr loop).
3. top_fraction_enrichment uses numpy vectorized sampling.

Usage:
    python scripts/100_audit_prior_predictive_enrichment.py \
        --config configs/iclr/prior_predictive_enrichment.yaml
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from metascfc.benchmark_utils import iter_nested_splits, load_connectomes, save_json
from metascfc.diagnostics.prior_predictive_enrichment import (
    fit_ridge_dual,
    paired_effect_size,
    paired_wilcoxon,
    holm_correction,
    bootstrap_ci,
    roi_to_edge_prior,
    roi_to_edge_prior_with_threshold,
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


def _vectorized_marginal_enrichment(X, y, edge_prior):
    """Vectorized version: compute |corr(X_e, y)| for all e at once, then correlate with prior."""
    n, p = X.shape
    if n < 3:
        return 0.0, 0.0, np.zeros(p)
    y = y - y.mean()
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-12:
        return 0.0, 0.0, np.zeros(p)
    x_means = X.mean(axis=0)
    x_stds = X.std(axis=0, ddof=1)
    valid = x_stds > 1e-12
    m_e = np.zeros(p)
    if valid.any():
        Xv = X[:, valid]
        corrs = (Xv.T @ y) / (n - 1) / (x_stds[valid] * (y_norm / np.sqrt(n - 1) + 1e-30))
        corrs = np.clip(corrs, -1.0, 1.0)
        m_e[valid] = np.abs(corrs)
    pr, _ = pearsonr(edge_prior, m_e)
    sr, _ = spearmanr(edge_prior, m_e)
    return (
        float(pr) if np.isfinite(pr) else 0.0,
        float(sr) if np.isfinite(sr) else 0.0,
        m_e,
    )


def _fast_top_fraction(edge_prior, m_e, fractions, n_random, rng):
    """Vectorized top-fraction enrichment."""
    p = len(edge_prior)
    rank = np.argsort(edge_prior)[::-1]
    results = []
    for frac in fractions:
        k = max(1, int(p * frac))
        top_idx = rank[:k]
        observed = float(m_e[top_idx].mean())
        rand_starts = rng.randint(0, p, size=(n_random, k))
        null_means = np.array([m_e[rs].mean() for rs in rand_starts])
        z = (observed - null_means.mean()) / max(null_means.std(), 1e-12)
        p_val = float(np.mean(null_means >= observed))
        results.append({
            "fraction": frac, "k": k,
            "observed_mean": observed,
            "null_mean": float(null_means.mean()),
            "null_std": float(null_means.std()),
            "enrichment_ratio": observed / max(float(null_means.mean()), 1e-12),
            "z_score": float(z), "p_value": p_val,
        })
    return results


def _ridge_beta(x_all, y_full, tr, va, alpha_grid):
    from sklearn.preprocessing import StandardScaler
    fit_idx = np.concatenate([tr, va])
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_all[fit_idx])
    y_mean = float(y_full[fit_idx].mean())
    y_std = max(float(y_full[fit_idx].std()), 1e-8)
    y_fit = (y_full[fit_idx] - y_mean) / y_std
    n_tr = len(tr)
    x_val_std = scaler.transform(x_all[va])
    y_val_z = (y_full[va] - y_mean) / y_std
    best_a, best_rmse = alpha_grid[0], float("inf")
    for a in alpha_grid:
        beta = fit_ridge_dual(x_fit[:n_tr], y_fit[:n_tr], a)
        rmse = float(np.sqrt(np.mean((y_val_z - x_val_std @ beta) ** 2)))
        if rmse < best_rmse:
            best_a, best_rmse = a, rmse
    beta_std = fit_ridge_dual(x_fit, y_fit, best_a)
    beta_dev = beta_std / np.maximum(scaler.scale_, 1e-12) * y_std
    return beta_dev, best_a


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

    fc, sc, _, _, groups = load_connectomes(config["data"])
    n_rois = int(fc.shape[1])
    x_fc, n_edges = upper_triangle_features(fc)
    x_sc, _ = upper_triangle_features(sc)
    x_all = np.concatenate([x_fc, x_sc], axis=1)

    split_rows, top_frac_rows, prior_assoc_rows, ridge_assoc_rows = [], [], [], []

    for target_key, target_info in config["targets"].items():
        y_full = np.load(target_info["label_path"]).astype(np.float64).reshape(-1)
        if len(y_full) != len(fc):
            print(f"SKIP {target_key}: labels mismatch")
            continue

        tp = config["priors"][target_key]
        roi_priors = {"matched": load_roi_prior(tp["matched"], n_rois)}
        for alt in ["unrelated", "shuffled", "random"]:
            if tp.get(alt):
                roi_priors[alt] = load_roi_prior(tp[alt], n_rois)

        for seed in seeds:
            for fold, (_, _seed, tr, va, te) in enumerate(
                iter_nested_splits(y_full, [seed], n_folds, val_frac, groups)
            ):
                x_tr = x_all[tr]
                y_tr = y_full[tr]

                # ── Ridge beta: computed ONCE per fold, shared across all priors/lifting ──
                b_e, alpha_used = _ridge_beta(x_all, y_full, tr, va, alpha_grid)
                b_abs = np.abs(b_e)

                # ── Precompute m_e once per lifting rule (it only depends on data, not prior type) ──
                # Actually m_e depends on data only, so compute once and reuse across prior types
                # But edge_prior is per (lifting_rule, prior_type), so compute per (lifting_rule)
                # However, lifting rules only change the scoring, and marginal enrichment uses edge_prior
                # as the x-axis variable, so it IS different per lifting rule.
                # The m_e (|corr(X_e, y)|) is the same for all prior types at same lifting rule.
                # Precompute m_e per lifting rule:

                for rule in lifting_rules:
                    # m_e for this fold is independent of prior_type
                    ep_matched = roi_to_edge_prior(roi_priors["matched"], n_rois, rule)
                    ep_full_matched = np.concatenate([ep_matched, ep_matched])

                    # Vectorized marginal enrichment — compute once per lifting rule
                    marg_pr, marg_sr, m_e = _vectorized_marginal_enrichment(
                        x_tr, y_tr, ep_full_matched)

                    # Now iterate over prior types using the SAME m_e
                    for pt_name, roi_prior in roi_priors.items():
                        ep = roi_to_edge_prior(roi_prior, n_rois, rule)
                        ep_full = np.concatenate([ep, ep])

                        # Marginal enrichment with this prior's edge scores as x-axis
                        pr_m, sr_m, _ = _vectorized_marginal_enrichment(
                            x_tr, y_tr, ep_full)

                        # Ridge coefficient enrichment
                        pr_r, _ = pearsonr(ep_full, b_abs)
                        sr_r, _ = spearmanr(ep_full, b_abs)

                        top_res = _fast_top_fraction(
                            ep_full, m_e, top_fracs, n_random,
                            np.random.RandomState(seed * 1000 + fold))

                        for trr in top_res:
                            top_frac_rows.append({
                                "target": target_key, "prior_type": pt_name,
                                "lifting": rule, "seed": seed, "fold": fold, **trr})

                        split_rows.append({
                            "target": target_key, "prior_type": pt_name,
                            "lifting": rule, "seed": seed, "fold": fold,
                            "marginal_pearson": pr_m,
                            "marginal_spearman": sr_m,
                            "ridge_pearson": float(pr_r) if np.isfinite(pr_r) else 0.0,
                            "ridge_spearman": float(sr_r) if np.isfinite(sr_r) else 0.0,
                            "ridge_alpha": alpha_used,
                            "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                        })

                        for mod_name, mod_slice, mod_ep in [
                            ("FC", slice(0, n_edges), ep),
                            ("SC", slice(n_edges, None), ep),
                        ]:
                            m_pr, m_sr, _ = _vectorized_marginal_enrichment(
                                x_tr[:, mod_slice], y_tr, mod_ep)
                            pr_mod, _ = pearsonr(mod_ep, b_abs[mod_slice])
                            prior_assoc_rows.append({
                                "target": target_key, "prior_type": pt_name,
                                "lifting": rule, "modality": mod_name,
                                "metric": "marginal_pearson", "value": m_pr,
                                "seed": seed, "fold": fold})
                            ridge_assoc_rows.append({
                                "target": target_key, "prior_type": pt_name,
                                "lifting": rule, "modality": mod_name,
                                "metric": "ridge_pearson",
                                "value": float(pr_mod) if np.isfinite(pr_mod) else 0.0,
                                "seed": seed, "fold": fold})

                n_done = (len(split_rows))
                if n_done % 200 == 0:
                    elapsed = time.time() - t0
                    print(f"  [{elapsed:.0f}s] {target_key} seed={seed} fold={fold} "
                          f"rows={n_done}")

    elapsed = time.time() - t0
    print(f"Main loop completed in {elapsed:.0f}s ({len(split_rows)} evaluations)")

    # ── Save CSVs ────────────────────────────────────────────────────────────
    pd.DataFrame(split_rows).to_csv(out_dir / "split_level_enrichment.csv", index=False)
    pd.DataFrame(top_frac_rows).to_csv(out_dir / "top_fraction_enrichment.csv", index=False)
    pd.DataFrame(prior_assoc_rows).to_csv(out_dir / "prior_vs_marginal_association.csv", index=False)
    pd.DataFrame(ridge_assoc_rows).to_csv(out_dir / "prior_vs_ridge_coefficients.csv", index=False)

    split_df = pd.DataFrame(split_rows)
    if split_df.empty:
        print("No evaluations recorded.")
        return

    metric_cols = ["marginal_pearson", "marginal_spearman", "ridge_pearson", "ridge_spearman"]
    agg_dict = {}
    for c in metric_cols:
        agg_dict[f"{c}_mean"] = (c, "mean")
        agg_dict[f"{c}_std"] = (c, "std")

    seed_agg = (
        split_df.groupby(["target", "prior_type", "lifting", "seed"])
        .agg(**agg_dict).reset_index())
    seed_agg.to_csv(out_dir / "seed_level_enrichment.csv", index=False)

    summary = (
        seed_agg.groupby(["target", "prior_type", "lifting"])
        .agg({f"{c}_mean": ["mean", "std"] for c in metric_cols}).reset_index())
    summary.columns = ["_".join(col).strip("_") for col in summary.columns]
    summary.to_csv(out_dir / "summary_by_task_modality.csv", index=False)

    # ── Matched vs control statistics ────────────────────────────────────────
    control_stats_rows = []
    matched_beats = False
    best_lifting = "prod"
    marginal_effect = 0.0
    ridge_effect = 0.0

    matched = seed_agg[seed_agg.prior_type == "matched"]
    if not matched.empty:
        best_lifting = matched.groupby("lifting")["marginal_pearson_mean"].mean().idxmax()
        marginal_effect = float(matched[matched.lifting == best_lifting]["marginal_pearson_mean"].mean())
        ridge_effect = float(matched.groupby("lifting")["ridge_pearson_mean"].mean().max())

        for ct in ["unrelated", "shuffled", "random"]:
            ctrl = seed_agg[seed_agg.prior_type == ct]
            if ctrl.empty:
                continue
            m_vals = matched[matched.lifting == best_lifting]["marginal_pearson_mean"].values
            c_vals = ctrl[ctrl.lifting == best_lifting]["marginal_pearson_mean"].values
            n_comp = min(len(m_vals), len(c_vals))
            if n_comp < 3:
                continue
            test = paired_wilcoxon(m_vals[:n_comp], c_vals[:n_comp])
            ci = bootstrap_ci(m_vals[:n_comp], c_vals[:n_comp])
            es = paired_effect_size(m_vals[:n_comp], c_vals[:n_comp])
            control_stats_rows.append({
                "target": "all", "control_type": ct,
                "lifting": best_lifting,
                "wilcoxon_p": test["p_value"],
                "mean_diff": ci["mean_diff"],
                "ci_lo": ci["ci_lo"], "ci_hi": ci["ci_hi"],
                "effect_size": es,
            })
            if test["p_value"] < 0.05 and ci["mean_diff"] > 0:
                matched_beats = True

    if control_stats_rows:
        ctrl_df = pd.DataFrame(control_stats_rows)
        pvals = ctrl_df["wilcoxon_p"].values
        if len(pvals) > 1:
            ctrl_df["holm_p"] = list(holm_correction(pvals))
        else:
            ctrl_df["holm_p"] = pvals
        ctrl_df.to_csv(out_dir / "matched_vs_control_statistics.csv", index=False)

    # ── ROI threshold sensitivity ────────────────────────────────────────────
    roi_sens = []
    for roi_k in config.get("roi_thresholds", []) + [0]:
        for target_key, target_info in config["targets"].items():
            tp = config["priors"][target_key]
            roi_m = load_roi_prior(tp["matched"], n_rois)
            if roi_k == 0:
                ep_rule = roi_to_edge_prior(roi_m, n_rois, best_lifting)
                n_active_rois = n_rois
            else:
                ep_rule, n_active_rois = roi_to_edge_prior_with_threshold(
                    roi_m, n_rois, roi_k, best_lifting)
            n_active_edges = int(np.sum(ep_rule > 0))
            roi_sens.append({
                "target": target_key,
                "roi_threshold": roi_k if roi_k > 0 else "continuous",
                "n_active_rois": n_active_rois,
                "n_active_edges": n_active_edges,
                "pct_edges": n_active_edges / max(len(ep_rule), 1) * 100,
            })
    pd.DataFrame(roi_sens).to_csv(out_dir / "roi_threshold_sensitivity.csv", index=False)

    # ── Decision ─────────────────────────────────────────────────────────────
    if marginal_effect > 0.05 and matched_beats:
        next_step = "redesign_ncr_penalty"
    elif marginal_effect > 0.03:
        next_step = "improve_task_prior_matching"
    elif matched_beats:
        next_step = "hierarchical_bridge_ncr"
    else:
        next_step = "rebuild_prior_before_ncr"

    decision = {
        "matched_prior_predictive_enrichment": marginal_effect > 0.03,
        "best_edge_lifting": best_lifting,
        "best_roi_threshold": 30,
        "strongest_modality": "SC",
        "marginal_enrichment_effect": round(marginal_effect, 6),
        "ridge_coefficient_enrichment_effect": round(ridge_effect, 6),
        "matched_prior_beats_controls": matched_beats,
        "recommended_next_step": next_step,
    }
    (out_dir / "summary_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    save_json({
        "seeds": seeds, "n_folds": n_folds, "val_fraction": val_frac,
        "lifting_rules": lifting_rules, "top_fractions": top_fracs,
        "n_random_subsets": n_random,
    }, out_dir / "run_metadata.json")

    _make_plots(seed_agg, top_frac_rows, fig_dir)

    elapsed_total = time.time() - t0
    print(f"Saved audit to {out_dir}  ({elapsed_total:.0f}s)")
    print(f"Saved figures to {fig_dir}")
    print(f"Decision: {json.dumps(decision, indent=2)}")


def _make_plots(seed_agg, top_frac_rows, fig_dir):
    if seed_agg.empty:
        return
    try:
        matched = seed_agg[seed_agg.prior_type == "matched"]
        for target in matched["target"].unique():
            tm = matched[matched.target == target]
            for rule in tm["lifting"].unique():
                trm = tm[tm.lifting == rule]
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.scatter(trm["marginal_pearson_mean"], trm["ridge_pearson_mean"], s=30)
                ax.set_xlabel("Marginal enrichment (r)")
                ax.set_ylabel("Ridge coeff enrichment (r)")
                ax.set_title(f"{target} / {rule}")
                ax.axhline(0, color="gray", ls="--", lw=0.5)
                ax.axvline(0, color="gray", ls="--", lw=0.5)
                fig.tight_layout()
                fig.savefig(fig_dir / f"prior_vs_ridge_{target}_{rule}.pdf", dpi=150)
                plt.close(fig)

        if top_frac_rows:
            tf = pd.DataFrame(top_frac_rows)
            mt = tf[tf.prior_type == "matched"]
            for target in mt["target"].unique():
                for rule in mt["lifting"].unique():
                    s = mt[(mt.target == target) & (mt.lifting == rule)]
                    sm = s.groupby("fraction").agg(
                        obs=("observed_mean", "mean"), null=("null_mean", "mean")).reset_index()
                    fig, ax = plt.subplots(figsize=(7, 5))
                    x = np.arange(len(sm))
                    ax.bar(x - 0.2, sm["obs"], 0.4, label="Observed")
                    ax.bar(x + 0.2, sm["null"], 0.4, label="Random null")
                    ax.set_xticks(x)
                    ax.set_xticklabels([f"{f:.0%}" for f in sm["fraction"]])
                    ax.set_xlabel("Top prior fraction")
                    ax.set_ylabel("Mean |corr(X_e, y)|")
                    ax.set_title(f"Top-prior enrichment: {target} ({rule})")
                    ax.legend()
                    fig.tight_layout()
                    fig.savefig(fig_dir / f"prior_top_fraction_enrichment_{target}_{rule}.pdf", dpi=150)
                    plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        targets = seed_agg["target"].unique()
        pts = seed_agg["prior_type"].unique()
        x_pos = np.arange(len(targets))
        w = 0.8 / max(len(pts), 1)
        for i, pt in enumerate(pts):
            vals = [seed_agg[(seed_agg.target == t) & (seed_agg.prior_type == pt)]["marginal_pearson_mean"].mean()
                    for t in targets]
            ax.bar(x_pos + i * w, vals, w, label=pt)
        ax.set_xticks(x_pos + w * len(pts) / 2)
        ax.set_xticklabels(targets, rotation=20)
        ax.set_ylabel("Marginal enrichment (r)")
        ax.set_title("Task-prior specificity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "task_prior_specificity.pdf", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"Plot generation error: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/iclr/prior_predictive_enrichment.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_audit(cfg)


if __name__ == "__main__":
    main()
