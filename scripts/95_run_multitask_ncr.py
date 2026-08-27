#!/usr/bin/env python3
"""Multi-Task NCR runner (ICLR 2027).

Runs dual-task NCR on HCP-YA with Qwen 3.8 27B Contrastive Priors:

    Method 1a: MT-NCR (independent) — each task solved by standard NCR,
               Jaccard overlap of top-k edges reported post-hoc.
    Method 1b: MT-NCR (joint l2,1) — IRLS joint penalty (lambda3 > 0).
    Baseline:  No-Prior Ridge — plain FC+SC ridge (lambda2 = 0).

Outputs (under --output-root, default outputs/iclr/mt_ncr):
    split_metrics.csv   one row per (method, target, seed, fold)
    summary.csv         aggregated mean +/- std
    run_metadata.json

Example:
    python scripts/95_run_multitask_ncr.py --seeds 0 --folds 0   # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from metascfc.benchmark_utils import (
    iter_nested_splits,
    load_connectomes,
    prediction_metrics,
    save_json,
)
from metascfc.models.iclr_backbones import (
    build_edge_laplacian,
    fit_predict_network_constrained,
    node_saliency_from_beta,
)
from metascfc.models.mt_ncr import (
    fit_predict_multitask_ncr,
    compute_biomarker_jaccard,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    factor_laplacian_eig,
)

DEFAULT_OUTPUT_ROOT = "outputs/iclr/mt_ncr"
DEFAULT_NCR_CONFIG = "configs/aaai/network_constrained_ridge.yaml"

FLUID_PRIOR = "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv"
WM_PRIOR = "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"


def load_roi_prior(path: str, n_rois: int) -> np.ndarray:
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--ncr-config", default=DEFAULT_NCR_CONFIG)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--folds", nargs="*", type=int)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--l3-grid", nargs="*", type=float, default=[0.0, 0.1, 1.0],
                    help="lambda3 (l2,1 strength) grid for joint mode")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg: Dict = yaml.safe_load(Path(args.ncr_config).read_text(encoding="utf-8"))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    n_folds = int(cfg.get("n_folds", 5))
    seeds = args.seeds if args.seeds else [int(s) for s in range(10)]

    fc, sc, _, _, groups = load_connectomes(cfg["data"])
    n_rois = int(fc.shape[1])
    iu = np.triu_indices(n_rois, k=1)
    x_edges = np.concatenate(
        [fc[:, iu[0], iu[1]], sc[:, iu[0], iu[1]]], axis=1
    ).astype(np.float64)

    y_fluid = np.load("inputs/dataset_SC/label_all.npy").astype(np.float64).reshape(-1)
    y_wm = np.load("inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy").astype(np.float64).reshape(-1)

    scores_fluid = load_roi_prior(FLUID_PRIOR, n_rois)
    scores_wm = load_roi_prior(WM_PRIOR, n_rois)

    top_k = int(cfg.get("top_k", 30))
    weighting = str(cfg.get("laplacian_weighting", "binary"))
    normalization = str(cfg.get("laplacian_normalization", "sym"))
    couple = bool(cfg.get("couple_modalities", False))

    lap_fluid = build_edge_laplacian(n_rois=n_rois, prior_scores=scores_fluid,
                                      top_k=top_k, weighting=weighting,
                                      couple_modalities=couple, normalize=normalization)
    lap_wm = build_edge_laplacian(n_rois=n_rois, prior_scores=scores_wm,
                                   top_k=top_k, weighting=weighting,
                                   couple_modalities=couple, normalize=normalization)
    lap_null = build_edge_laplacian(n_rois=n_rois,
                                     prior_adjacency=np.zeros((n_rois, n_rois)),
                                     weighting=weighting, couple_modalities=couple,
                                     normalize=normalization)

    eig_fluid = factor_laplacian_eig(lap_fluid)
    eig_wm = factor_laplacian_eig(lap_wm)
    eig_null = factor_laplacian_eig(lap_null)

    alpha1_grid = [float(a) for a in cfg["ridge_alphas"]]
    alpha2_grid = [float(a) for a in cfg["laplacian_alphas"]]

    split_csv = out_root / "split_metrics.csv"
    if args.overwrite and split_csv.exists():
        split_csv.unlink()
    rows = [] if not split_csv.exists() else pd.read_csv(split_csv).to_dict("records")
    completed = {(r["method"], int(r["seed"]), int(r["fold"])) for r in rows}

    selected_folds = set(args.folds) if args.folds else None
    total_new = 0

    for seed, fold, train_idx, val_idx, test_idx in iter_nested_splits(
        y_fluid, seeds, n_folds, val_fraction, groups
    ):
        if selected_folds is not None and fold not in selected_folds:
            continue
        split_id = f"seed{seed:02d}_fold{fold:02d}"
        split_key_seed = seed * 1000 + fold

        # --- Method 1a: MT-NCR (independent) ---
        method = "mt_ncr_independent"
        if (method, seed, fold) not in completed:
            started = time.time()
            try:
                res = fit_predict_multitask_ncr(
                    x_edges, y_fluid, y_wm, train_idx, val_idx, test_idx,
                    lap_fluid, lap_wm, alpha1_grid, alpha2_grid,
                    alpha3_grid=[0.0], eig_fluid=eig_fluid, eig_wm=eig_wm,
                    n_rois=n_rois)
                m_f = prediction_metrics(y_fluid[test_idx], res.pred_fluid)
                m_m = prediction_metrics(y_wm[test_idx], res.pred_wm)
                sal_f = node_saliency_from_beta(res.beta_fluid, n_rois)
                sal_m = node_saliency_from_beta(res.beta_wm, n_rois)
                for target_name, metrics in [("fluid_intelligence", m_f),
                                              ("working_memory", m_m)]:
                    rows.append({
                        "method": method, "target": target_name,
                        "seed": seed, "fold": fold, "split_id": split_id,
                        "n_train": len(train_idx), "n_val": len(val_idx),
                        "n_test": len(test_idx),
                        "best_lambda1": res.best_lambda1,
                        "best_lambda2": res.best_lambda2,
                        "best_lambda3": 0.0,
                        "runtime_seconds": time.time() - started,
                        **metrics,
                    })
                rows.append({
                    "method": method, "target": "jaccard_top10pct",
                    "seed": seed, "fold": fold, "split_id": split_id,
                    "jaccard": res.jaccard,
                })
                completed.add((method, seed, fold))
                total_new += 1
                pd.DataFrame(rows).to_csv(split_csv, index=False)
                print(f"{method} {split_id} "
                      f"fluid_r={m_f['pearson']:+.3f} wm_r={m_m['pearson']:+.3f} "
                      f"jaccard={res.jaccard:.3f}", flush=True)
            except Exception as exc:
                rows.append({"method": method, "seed": seed, "fold": fold,
                             "error": str(exc)})
                pd.DataFrame(rows).to_csv(split_csv, index=False)
                print(f"FAILED {method} {split_id}: {exc}", flush=True)

        # --- Method 1b: MT-NCR (joint l2,1) ---
        method = "mt_ncr_joint_l21"
        if (method, seed, fold) not in completed:
            started = time.time()
            try:
                res = fit_predict_multitask_ncr(
                    x_edges, y_fluid, y_wm, train_idx, val_idx, test_idx,
                    lap_fluid, lap_wm, alpha1_grid, alpha2_grid,
                    alpha3_grid=args.l3_grid, eig_fluid=eig_fluid, eig_wm=eig_wm,
                    n_rois=n_rois)
                m_f = prediction_metrics(y_fluid[test_idx], res.pred_fluid)
                m_m = prediction_metrics(y_wm[test_idx], res.pred_wm)
                for target_name, metrics in [("fluid_intelligence", m_f),
                                              ("working_memory", m_m)]:
                    rows.append({
                        "method": method, "target": target_name,
                        "seed": seed, "fold": fold, "split_id": split_id,
                        "n_train": len(train_idx), "n_val": len(val_idx),
                        "n_test": len(test_idx),
                        "best_lambda1": res.best_lambda1,
                        "best_lambda2": res.best_lambda2,
                        "best_lambda3": res.best_lambda3,
                        "runtime_seconds": time.time() - started,
                        **metrics,
                    })
                rows.append({
                    "method": method, "target": "jaccard_top10pct",
                    "seed": seed, "fold": fold, "split_id": split_id,
                    "jaccard": res.jaccard,
                })
                completed.add((method, seed, fold))
                total_new += 1
                pd.DataFrame(rows).to_csv(split_csv, index=False)
                print(f"{method} {split_id} "
                      f"fluid_r={m_f['pearson']:+.3f} wm_r={m_m['pearson']:+.3f} "
                      f"l3={res.best_lambda3} jaccard={res.jaccard:.3f}", flush=True)
            except Exception as exc:
                rows.append({"method": method, "seed": seed, "fold": fold,
                             "error": str(exc)})
                pd.DataFrame(rows).to_csv(split_csv, index=False)
                print(f"FAILED {method} {split_id}: {exc}", flush=True)

        # --- Baseline: No-Prior Ridge ---
        method = "no_prior_ridge"
        if (method, seed, fold) not in completed:
            started = time.time()
            try:
                pred_f, _, _, rmse_f, beta_f = fit_predict_network_constrained(
                    x_edges, y_fluid, train_idx, val_idx, test_idx,
                    lap_null, alpha1_grid, [0.0], laplacian_eig=eig_null)
                pred_m, _, _, rmse_m, beta_m = fit_predict_network_constrained(
                    x_edges, y_wm, train_idx, val_idx, test_idx,
                    lap_null, alpha1_grid, [0.0], laplacian_eig=eig_null)
                m_f = prediction_metrics(y_fluid[test_idx], pred_f)
                m_m = prediction_metrics(y_wm[test_idx], pred_m)
                for target_name, metrics in [("fluid_intelligence", m_f),
                                              ("working_memory", m_m)]:
                    rows.append({
                        "method": method, "target": target_name,
                        "seed": seed, "fold": fold, "split_id": split_id,
                        "n_train": len(train_idx), "n_val": len(val_idx),
                        "n_test": len(test_idx),
                        "best_lambda1": 0.0, "best_lambda2": 0.0,
                        "best_lambda3": 0.0,
                        "runtime_seconds": time.time() - started,
                        **metrics,
                    })
                completed.add((method, seed, fold))
                total_new += 1
                pd.DataFrame(rows).to_csv(split_csv, index=False)
                print(f"{method} {split_id} "
                      f"fluid_r={m_f['pearson']:+.3f} wm_r={m_m['pearson']:+.3f}",
                      flush=True)
            except Exception as exc:
                rows.append({"method": method, "seed": seed, "fold": fold,
                             "error": str(exc)})
                pd.DataFrame(rows).to_csv(split_csv, index=False)
                print(f"FAILED {method} {split_id}: {exc}", flush=True)

    # --- Aggregate summary ---
    df = pd.DataFrame(rows)
    if df.empty:
        print("No evaluations recorded.")
        return

    metric_cols = ["pearson", "rmse", "mae"]
    metric_agg = {}
    for c in metric_cols:
        if c in df.columns:
            metric_agg[f"{c}_mean"] = (c, "mean")
            metric_agg[f"{c}_std"] = (c, "std")

    agg = (
        df[df["target"].isin(["fluid_intelligence", "working_memory"])]
        .groupby(["method", "target"])
        .agg(n_splits=("fold", "size"), **metric_agg)
        .reset_index()
    )
    agg.to_csv(out_root / "summary.csv", index=False)

    save_json({
        "methods": ["mt_ncr_independent", "mt_ncr_joint_l21", "no_prior_ridge"],
        "targets": ["fluid_intelligence", "working_memory"],
        "seeds": sorted(seeds), "n_folds": n_folds,
        "val_fraction": val_fraction, "n_evaluations": int(len(df)),
        "ncr_config": args.ncr_config,
        "l3_grid": args.l3_grid,
    }, out_root / "run_metadata.json")
    print(f"Saved {len(df)} evaluations ({total_new} new) to {out_root}")


if __name__ == "__main__":
    main()
