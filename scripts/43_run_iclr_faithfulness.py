#!/usr/bin/env python3
"""ICLR 2027 perturbation (faithfulness) protocol for Methods 2 and 3.

The test
--------
For every outer split of the shared nested-CV protocol:

    mask_TOP10_TRUE   = the 10 ROIs with the highest true WM prior score
    mask_TOP10_RANDOM = the 10 ROIs with the highest random prior score
    (both fixed across splits - the prior maps are fixed artifacts)

The split's model (Method 2 Meta-GAT or Method 3 Two-Stage KRR) is refit on
train+val with the *recorded* best hyperparameters of the headline runs
(scripts/41, scripts/42) and evaluated on the held-out test subjects with
all FC/SC connections incident to the masked ROIs removed.  The faithfulness
hypothesis:

    delta_rmse(mask_TOP10_TRUE) >> delta_rmse(mask_TOP10_RANDOM),
    delta_rmse(mask_TOP10_TRUE) >> delta_rmse(random-10),

i.e. the true prior forces the model to rely on biologically meaningful,
task-specific circuitry, so ablating it must hurt far more than ablating
random circuitry.

Learned-mask conditions (saliency top/bottom-10 from the headline-run
biomarkers) and 20 repeated random-10 masks are included for the standard
faithfulness comparisons (scripts/17 schema).  Degradation is defined as
delta_rmse = perturbed_rmse - original_rmse (positive = worse).

Outputs (outputs/aaai/faithfulness_iclr/):
  per_method/{method}/faithfulness_split_metrics.csv (resumable)
  per_method/{method}/faithfulness_long.csv
  per_method/{method}/attention/{split}.npz        (M2 attention biomarker)
  per_method/{method}/krr_saliency/{split}.npz     (M3 KRR gradient biomarker)
  faithfulness_all_split_metrics.csv / faithfulness_all_long.csv
  faithfulness_summary.csv (.tex)
  faithfulness_seed_level_metrics.csv
  faithfulness_seed_level_tests.csv (.tex)         (one-sample + paired, Holm)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_1samp, ttest_rel, wilcoxon

from metascfc.benchmark_utils import (
    atomic_write_csv,
    choose_device,
    iter_nested_splits,
    load_connectomes,
    prediction_metrics,
    save_json,
)
from metascfc.models.iclr_backbones import (
    MetaGATConfig,
    RefitKRRPredictor,
    RefitMetaGATPredictor,
    load_split_node_saliency,
    refit_krr_predictor,
    refit_meta_gat_predictor,
)


# ---------------------------------------------------------------------------
# Masking (raw connectome level, per model family)
# ---------------------------------------------------------------------------
def mask_connectomes_rois(
    fc: np.ndarray,
    sc: np.ndarray,
    roi_indices: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Zero all FC/SC connections incident to the selected ROIs.

    For the Meta-GAT representation (node features = connectome rows) this
    removes the masked ROIs as information sources; for the KRR
    representation (upper-triangle edge features) every edge incident to a
    masked ROI is zeroed.  Both remove exactly "all FC and SC connections
    incident to selected ROIs" in the model's feature space.
    """
    idx = np.unique(np.asarray(list(roi_indices), dtype=int))
    fc_out = np.array(fc, copy=True)
    sc_out = np.array(sc, copy=True)
    fc_out[:, idx, :] = 0.0
    fc_out[:, :, idx] = 0.0
    sc_out[:, idx, :] = 0.0
    sc_out[:, :, idx] = 0.0
    return fc_out, sc_out


def positive_degradation(original: Dict[str, float], perturbed: Dict[str, float], metric: str) -> float:
    if metric in {"rmse", "mae"}:
        return float(perturbed[metric] - original[metric])
    if metric == "pearson":
        return float(original[metric] - perturbed[metric])
    raise ValueError(metric)


# ---------------------------------------------------------------------------
# Statistical helpers (identical semantics to scripts/17)
# ---------------------------------------------------------------------------
def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def safe_wilcoxon_greater(values: np.ndarray) -> tuple[float, float]:
    try:
        result = wilcoxon(values, alternative="greater", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return 0.0, 1.0


def safe_wilcoxon_paired(a: np.ndarray, b: np.ndarray, alternative: str = "greater") -> tuple[float, float]:
    try:
        result = wilcoxon(np.asarray(a), np.asarray(b), alternative=alternative, zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return 0.0, 1.0


def bootstrap_ci(values: np.ndarray, n_bootstrap: int = 20000, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boots[i] = rng.choice(values, len(values), replace=True).mean()
    return np.quantile(boots, [0.025, 0.975])


def top_k_indices(prior: np.ndarray, k: int) -> np.ndarray:
    """Deterministic top-k set (ties handled by argpartition, as elsewhere)."""
    p = np.asarray(prior, dtype=float).reshape(-1)
    k = min(int(k), len(p))
    return np.argpartition(p, -k)[-k:]


def load_roi_prior(path: str | Path, n_rois: int) -> np.ndarray:
    df = pd.read_csv(path)
    if "roi_index" in df.columns:
        df = df.sort_values("roi_index")
    p = df["prior_score"].to_numpy(np.float64)
    if p.shape != (n_rois,):
        raise ValueError(f"Prior {path} has shape {p.shape}; expected {(n_rois,)}")
    p = np.clip(p, 0.0, None)
    if p.max() > p.min():
        p = (p - p.min()) / (p.max() - p.min())
    return p


# ---------------------------------------------------------------------------
# Per-family evaluation
# ---------------------------------------------------------------------------
def evaluate_masks_meta_gat(
    predictor: RefitMetaGATPredictor,
    fc_test: np.ndarray,
    sc_test: np.ndarray,
    y_test: np.ndarray,
    top_idx: np.ndarray,
    bottom_idx: np.ndarray,
    prior_true_idx: np.ndarray,
    prior_random_idx: np.ndarray,
    n_random: int,
    rng: np.random.Generator,
    n_rois: int,
) -> tuple[Dict[str, float], List[Dict[str, float]]]:
    """Evaluate original + masks for a refit Meta-GAT; return (row, long_rows)."""
    metrics = {m: prediction_metrics(y_test, predictor.predict(fc_test, sc_test))[m]
               for m in ("pearson", "rmse", "mae")}
    row = {"original_pearson": metrics["pearson"], "original_rmse": metrics["rmse"],
           "original_mae": metrics["mae"]}
    long_rows: List[Dict[str, float]] = []

    def eval_condition(name: str, roi_mask: Optional[np.ndarray], repeat: int = -1) -> Dict[str, float]:
        if roi_mask is None:
            fc_m, sc_m = fc_test, sc_test
        else:
            fc_m, sc_m = mask_connectomes_rois(fc_test, sc_test, roi_mask)
        res = {m: prediction_metrics(y_test, predictor.predict(fc_m, sc_m))[m]
               for m in ("pearson", "rmse", "mae")}
        long_rows.append({"condition": name, "repeat": repeat, **res})
        return res

    conditions = {
        "top": top_idx, "bottom": bottom_idx,
        "prior_true_top": prior_true_idx, "prior_random_top": prior_random_idx,
    }
    random_metrics: List[Dict[str, float]] = []
    for repeat in range(n_random):
        random_metrics.append(
            eval_condition("random", rng.choice(n_rois, size=len(prior_true_idx), replace=False), repeat)
        )

    evaluated = {}
    for name, mask in conditions.items():
        evaluated[name] = eval_condition(name, mask)
    random_mean = {m: float(np.mean([x[m] for x in random_metrics])) for m in ("pearson", "rmse", "mae")}
    random_std = {m: float(np.std([x[m] for x in random_metrics], ddof=1)) for m in ("pearson", "rmse", "mae")}

    for metric in ("pearson", "rmse", "mae"):
        row[f"random_{metric}_mean"] = random_mean[metric]
        row[f"random_{metric}_std"] = random_std[metric]
        for name in ("top", "bottom", "prior_true_top", "prior_random_top"):
            row[f"{name}_{metric}"] = float(evaluated[name][metric])
            row[f"delta_{metric}_{name}"] = positive_degradation(metrics, evaluated[name], metric)
        row[f"delta_{metric}_random"] = positive_degradation(metrics, random_mean, metric)
        row[f"gap_{metric}_top_vs_random"] = row[f"delta_{metric}_top"] - row[f"delta_{metric}_random"]
        row[f"gap_{metric}_top_vs_bottom"] = row[f"delta_{metric}_top"] - row[f"delta_{metric}_bottom"]
        row[f"gap_{metric}_prior_true_vs_random"] = row[f"delta_{metric}_prior_true_top"] - row[f"delta_{metric}_random"]
        row[f"gap_{metric}_prior_true_vs_prior_random"] = row[f"delta_{metric}_prior_true_top"] - row[f"delta_{metric}_prior_random_top"]
    return row, long_rows


def evaluate_masks_krr(
    predictor: RefitKRRPredictor,
    fc_test: np.ndarray,
    sc_test: np.ndarray,
    y_test: np.ndarray,
    top_idx: np.ndarray,
    bottom_idx: np.ndarray,
    prior_true_idx: np.ndarray,
    prior_random_idx: np.ndarray,
    n_random: int,
    rng: np.random.Generator,
    n_rois: int,
) -> tuple[Dict[str, float], List[Dict[str, float]]]:
    return evaluate_masks_meta_gat(
        predictor, fc_test, sc_test, y_test,
        top_idx, bottom_idx, prior_true_idx, prior_random_idx, n_random, rng, n_rois,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aaai/faithfulness_iclr.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--folds", nargs="*", type=int)
    args = ap.parse_args()

    cfg: Dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(cfg["output_dir"])
    if args.overwrite and out.exists():
        import shutil
        shutil.rmtree(out)
    (out / "per_method").mkdir(parents=True, exist_ok=True)

    device = choose_device(cfg.get("device", "auto"))
    print("Device:", device, flush=True)

    fc, sc, y, subject_ids, groups = load_connectomes(cfg["data"])
    n_rois = fc.shape[1]
    n_edges = int(np.triu_indices(n_rois, 1)[0].size)
    k = int(cfg.get("topk", 10))
    n_random = int(cfg.get("random_repeats", 20))

    prior_true = load_roi_prior(cfg["prior_true"], n_rois)
    prior_random = load_roi_prior(cfg["prior_random"], n_rois)
    prior_true_idx = top_k_indices(prior_true, k)
    prior_random_idx = top_k_indices(prior_random, k)
    print(
        f"Top-{k} TRUE  prior ROIs: {sorted(prior_true_idx.tolist())}",
        flush=True,
    )
    print(
        f"Top-{k} RANDOM prior ROIs: {sorted(prior_random_idx.tolist())} "
        f"(overlap with TRUE: {len(set(prior_true_idx) & set(prior_random_idx))})",
        flush=True,
    )

    seeds = args.seeds if args.seeds else [int(s) for s in cfg["seeds"]]
    n_folds = int(cfg.get("n_folds", 5))
    val_fraction = float(cfg.get("val_fraction", 0.15))

    all_rows: List[Dict] = []
    all_long: List[Dict] = []
    meta_gat_fixed: Dict = cfg["families"]["meta_gat"]
    krr_fixed: Dict = cfg["families"]["kernel_ridge"]

    for family in ("meta_gat", "kernel_ridge"):
        for method_id, mcfg in cfg["families"][family]["methods"].items():
            if args.methods and method_id not in args.methods:
                continue
            exp_out = out / "per_method" / method_id
            exp_out.mkdir(parents=True, exist_ok=True)
            (exp_out / "attention").mkdir(exist_ok=True)
            (exp_out / "krr_saliency").mkdir(exist_ok=True)
            split_csv = exp_out / "faithfulness_split_metrics.csv"
            long_csv = exp_out / "faithfulness_long.csv"
            done = (exp_out / "COMPLETE")
            if done.exists() and not args.overwrite:
                print(f"SKIP completed faithfulness run: {method_id}", flush=True)
                if split_csv.exists():
                    all_rows.extend(pd.read_csv(split_csv).to_dict("records"))
                if long_csv.exists():
                    all_long.extend(pd.read_csv(long_csv).to_dict("records"))
                continue

            results_df = pd.read_csv(mcfg["results_csv"])
            saliency_dir = Path(mcfg["saliency_dir"])
            rows: List[Dict] = []
            long_rows: List[Dict] = []
            completed = set()
            if split_csv.exists() and not args.overwrite:
                for rec in pd.read_csv(split_csv).to_dict("records"):
                    rows.append(rec)
                    completed.add((int(rec["seed"]), int(rec["fold"])))
            print(f"=== {family} / {method_id}: {mcfg['name']} ===", flush=True)

            for seed, fold, train_idx, val_idx, test_idx in iter_nested_splits(y, seeds, n_folds, val_fraction, groups):
                if args.folds and fold not in args.folds:
                    continue
                if (seed, fold) in completed:
                    print(f"SKIP {method_id} seed{seed:02d}_fold{fold:02d}", flush=True)
                    continue
                started = time.time()
                split_seed = seed * 1000 + fold
                fit_idx = np.concatenate([train_idx, val_idx])
                rng = np.random.default_rng(split_seed + 9173)

                if family == "meta_gat":
                    rec = results_df[(results_df.method_id == method_id) & (results_df.seed == seed) & (results_df.fold == fold)].iloc[0]
                    cfg_m2 = MetaGATConfig(
                        hidden=int(rec["best_hidden"]),
                        heads1=int(meta_gat_fixed["heads1"]),
                        heads2=int(meta_gat_fixed["heads2"]),
                        dropout=float(rec["best_dropout"]),
                        gamma_init=float(meta_gat_fixed["gamma_init"]),
                        learning_rate=float(rec["best_learning_rate"]),
                        weight_decay=float(meta_gat_fixed["weight_decay"]),
                        epochs=1,
                        patience=1,
                        min_epochs=1,
                        grad_clip=float(meta_gat_fixed["grad_clip"]),
                    )
                    saliency = load_split_node_saliency(saliency_dir, seed, fold)
                    predictor = refit_meta_gat_predictor(
                        fc, sc, y, fit_idx, cfg_m2, saliency, device,
                        n_epochs=int(rec["best_epoch"]),
                        top_percent=float(meta_gat_fixed["top_percent_sc"]),
                        seed=split_seed,
                    )
                    row, per_cond = evaluate_masks_meta_gat(
                        predictor, fc[test_idx], sc[test_idx], y[test_idx],
                        top_k_indices(saliency, k), np.argsort(saliency)[:k],
                        prior_true_idx, prior_random_idx, n_random, rng, n_rois,
                    )
                    np.savez(
                        exp_out / "attention" / f"seed{seed:02d}_fold{fold:02d}.npz",
                        node_attention_mass=predictor.attention_mass(fc[fit_idx], sc[fit_idx]),
                    )
                else:
                    rec = results_df[(results_df.method_id == method_id) & (results_df.seed == seed) & (results_df.fold == fold)].iloc[0]
                    saliency = load_split_node_saliency(saliency_dir, seed, fold)
                    predictor = refit_krr_predictor(
                        fc, sc, y, fit_idx, saliency,
                        alpha=float(rec["best_alpha"]),
                        gamma=float(rec["best_gamma"]),
                        gate_mode=str(krr_fixed.get("gate_mode", "product")),
                    )
                    row, per_cond = evaluate_masks_krr(
                        predictor, fc[test_idx], sc[test_idx], y[test_idx],
                        top_k_indices(saliency, k), np.argsort(saliency)[:k],
                        prior_true_idx, prior_random_idx, n_random, rng, n_rois,
                    )
                    np.savez(
                        exp_out / "krr_saliency" / f"seed{seed:02d}_fold{fold:02d}.npz",
                        node_saliency=predictor.gradient_node_saliency(fc[fit_idx], sc[fit_idx]),
                    )

                row.update({
                    "method_id": method_id, "method_name": mcfg["name"],
                    "method_family": family, "seed": seed, "fold": fold,
                    "split_id": f"seed{seed:02d}_fold{fold:02d}",
                    "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
                    "topk": k, "mask_mode": cfg.get("mask_mode", "both"),
                    "random_repeats": n_random, "device": str(device),
                    "runtime_seconds": time.time() - started,
                })
                rows.append(row)
                for cond in per_cond:
                    long_rows.append({"method_id": method_id, "seed": seed, "fold": fold, **cond})
                atomic_write_csv(pd.DataFrame(rows), split_csv)
                atomic_write_csv(pd.DataFrame(long_rows), long_csv)
                print(
                    method_id, row["split_id"],
                    f"orig_rmse={row['original_rmse']:.3f} "
                    f"delta_true_top={row['delta_rmse_prior_true_top']:+.3f} "
                    f"delta_random_top={row['delta_rmse_prior_random_top']:+.3f} "
                    f"delta_random={row['delta_rmse_random']:+.3f} "
                    f"gap_true_vs_random={row['gap_rmse_prior_true_vs_random']:+.3f}",
                    flush=True,
                )
            done.write_text("ok\n", encoding="utf-8")
            all_rows.extend(rows)
            all_long.extend(long_rows)

    if not all_rows:
        print("Nothing to do.")
        return

    split_df = pd.DataFrame(all_rows)
    long_df = pd.DataFrame(all_long)
    split_df.to_csv(out / "faithfulness_all_split_metrics.csv", index=False)
    long_df.to_csv(out / "faithfulness_all_long.csv", index=False)

    summary_rows = []
    for method_id, g in split_df.groupby("method_id"):
        srow = {
            "ID": method_id,
            "Method": g["method_name"].iloc[0],
            "Family": g["method_family"].iloc[0],
            "Seeds": g["seed"].nunique(),
            "Folds": g["fold"].nunique(),
            "Top-k": int(g["topk"].iloc[0]),
            "Original RMSE": g["original_rmse"].mean(),
        }
        for col in [c for c in g.columns if c.startswith("delta_") or c.startswith("gap_")]:
            srow[f"{col} Mean"] = g[col].mean()
            srow[f"{col} Std"] = g[col].std(ddof=1)
        summary_rows.append(srow)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "faithfulness_summary.csv", index=False)
    (out / "faithfulness_summary.tex").write_text(
        summary.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )

    # Seed-level inference (average the 5 folds within each seed).
    gap_cols = [c for c in split_df.columns if c.startswith("gap_")]
    delta_cols = [c for c in split_df.columns if c.startswith("delta_")]
    seed_df = split_df.groupby(["method_id", "seed"], as_index=False)[gap_cols + delta_cols].mean()
    seed_df.to_csv(out / "faithfulness_seed_level_metrics.csv", index=False)

    stat_rows: List[Dict] = []
    for method_id, g in seed_df.groupby("method_id"):
        for metric in gap_cols:
            values = g[metric].dropna().to_numpy(float)
            if len(values) < 3:
                continue
            t = ttest_1samp(values, popmean=0.0, alternative="greater")
            w_stat, w_p = safe_wilcoxon_greater(values)
            lo, hi = bootstrap_ci(values)
            stat_rows.append({
                "ID": method_id, "contrast": metric, "analysis_unit": "seed_mean_over_5_folds",
                "n_seeds": len(values), "mean_gap": values.mean(), "std_gap": values.std(ddof=1),
                "bootstrap95_low": lo, "bootstrap95_high": hi,
                "one_sample_t": float(t.statistic), "one_sample_t_p_greater": float(t.pvalue),
                "wilcoxon_W": w_stat, "wilcoxon_p_greater": w_p,
            })
        # Paired contrasts of the hypothesis (seed-level).
        d_true = g["delta_rmse_prior_true_top"].to_numpy(float)
        d_random_top = g["delta_rmse_prior_random_top"].to_numpy(float)
        d_random = g["delta_rmse_random"].to_numpy(float)
        for name, other in (("prior_random_top", d_random_top), ("random_mean", d_random)):
            w_stat, w_p = safe_wilcoxon_paired(d_true, other, "greater")
            t_result = ttest_rel(d_true, other, alternative="greater")
            stat_rows.append({
                "ID": method_id, "contrast": f"delta_rmse_prior_true_top > delta_rmse_{name}",
                "analysis_unit": "paired_seed_means", "n_seeds": len(d_true),
                "mean_gap": float(np.mean(d_true - other)),
                "std_gap": float(np.std(d_true - other, ddof=1)),
                "bootstrap95_low": float(np.nan), "bootstrap95_high": float(np.nan),
                "one_sample_t": float(t_result.statistic),
                "one_sample_t_p_greater": float(t_result.pvalue),
                "wilcoxon_W": w_stat, "wilcoxon_p_greater": w_p,
            })
    stats = pd.DataFrame(stat_rows)
    if not stats.empty:
        stats["one_sample_t_p_holm"] = holm_adjust(stats["one_sample_t_p_greater"].to_numpy())
        stats["wilcoxon_p_holm"] = holm_adjust(stats["wilcoxon_p_greater"].to_numpy())
        stats["significant_wilcoxon_holm_005"] = stats["wilcoxon_p_holm"] < 0.05
    stats.to_csv(out / "faithfulness_seed_level_tests.csv", index=False)
    (out / "faithfulness_seed_level_tests.tex").write_text(
        stats.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )
    save_json({
        "config": cfg, "n_subjects": len(y), "n_rois": n_rois, "n_edges": n_edges,
        "prior_true_topk_rois": prior_true_idx.tolist(),
        "prior_random_topk_rois": prior_random_idx.tolist(),
        "overlap_true_random_topk": int(len(set(prior_true_idx) & set(prior_random_idx))),
        "device": str(device),
    }, out / "run_metadata.json")
    print(f"Saved faithfulness outputs to {out}")


if __name__ == "__main__":
    main()