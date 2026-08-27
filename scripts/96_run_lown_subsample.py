#!/usr/bin/env python3
"""Low-N data scarcity curves (ICLR 2027).

Demonstrates sample-efficient connectomics: how does NCR with a zero-shot
LLM contrastive prior perform as training set size N shrinks?

For each N in {50, 100, 200, 412}:
  1. Subsample N subjects (stratified by target, seed-controlled)
  2. Run single-task NCR for each target with Qwen 3.8 contrastive prior
  3. Run No-Prior Ridge (lambda2=0) as the ablation baseline
  4. Record mean test Pearson r over 3 seeds x 5 folds

Outputs (under --output-root, default outputs/iclr/lown_curve):
    lown_curve.csv   one row per (method, target, N, seed, fold)
    summary.csv      mean +/- std over seeds/folds per (method, target, N)

Example:
    python scripts/96_run_lown_subsample.py --seeds 0   # smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

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
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    factor_laplacian_eig,
)

DEFAULT_OUTPUT_ROOT = "outputs/iclr/lown_curve"
DEFAULT_NCR_CONFIG = "configs/aaai/network_constrained_ridge.yaml"

FLUID_PRIOR = "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv"
WM_PRIOR = "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"

NSIZES = [50, 100, 200, 412]


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


def subsample_indices(
    y: np.ndarray, n: int, seed: int, groups: np.ndarray | None = None
) -> np.ndarray:
    """Stratified subsample of n subjects (preserving target distribution)."""
    rng = np.random.RandomState(seed)
    all_idx = np.arange(len(y))
    if n >= len(y):
        return all_idx
    bins = pd.qcut(y, q=min(5, len(y)), labels=False, duplicates="drop")
    selected: List[int] = []
    for b in np.unique(bins):
        pool = all_idx[bins == b]
        k = max(1, int(round(n * len(pool) / len(y))))
        chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
        selected.extend(chosen.tolist())
    selected = np.array(selected)
    if len(selected) > n:
        selected = rng.choice(selected, size=n, replace=False)
    elif len(selected) < n:
        leftover = np.setdiff1d(all_idx, selected)
        extra = rng.choice(leftover, size=n - len(selected), replace=False)
        selected = np.concatenate([selected, extra])
    return np.sort(selected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--ncr-config", default=DEFAULT_NCR_CONFIG)
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--nsizes", nargs="*", type=int, default=NSIZES)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg: Dict = yaml.safe_load(Path(args.ncr_config).read_text(encoding="utf-8"))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    n_folds = int(cfg.get("n_folds", 5))
    seeds = args.seeds if args.seeds else [0, 1, 2]

    fc, sc, _, _, groups = load_connectomes(cfg["data"])
    n_rois = int(fc.shape[1])
    iu = np.triu_indices(n_rois, k=1)
    x_all = np.concatenate(
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

    csv_path = out_root / "lown_curve.csv"
    if args.overwrite and csv_path.exists():
        csv_path.unlink()
    rows = [] if not csv_path.exists() else pd.read_csv(csv_path).to_dict("records")
    completed = {
        (r["method"], r["target"], int(r["N"]), int(r["seed"]), int(r["fold"]))
        for r in rows
    }

    total_new = 0
    for n_size in args.nsizes:
        for target_name, y_target, lap_prior, eig_prior in [
            ("fluid_intelligence", y_fluid, lap_fluid, eig_fluid),
            ("working_memory", y_wm, lap_wm, eig_wm),
        ]:
            for seed in seeds:
                sub_idx = subsample_indices(y_target, n_size, seed)
                x_sub = x_all[sub_idx]
                y_sub = y_target[sub_idx]

                for fold_i, (si, fi, tr_idx, va_idx, te_idx) in enumerate(
                    iter_nested_splits(y_sub, [seed], n_folds,
                                       val_fraction)
                ):
                    for method, lap, eig in [
                        ("ncr_contrastive", lap_prior, eig_prior),
                        ("no_prior_ridge", lap_null, eig_null),
                    ]:
                        if (method, target_name, n_size, seed, fold_i) in completed:
                            continue
                        started = time.time()
                        try:
                            pred, _, _, _, beta = fit_predict_network_constrained(
                                x_sub, y_sub, tr_idx, va_idx, te_idx,
                                lap, alpha1_grid, alpha2_grid, laplacian_eig=eig)
                            metrics = prediction_metrics(y_sub[te_idx], pred)
                            rows.append({
                                "method": method, "target": target_name,
                                "N": n_size, "seed": seed, "fold": fold_i,
                                "n_train": len(tr_idx), "n_val": len(va_idx),
                                "n_test": len(te_idx),
                                "runtime_seconds": time.time() - started,
                                **metrics,
                            })
                            completed.add((method, target_name, n_size, seed, fold_i))
                            total_new += 1
                            pd.DataFrame(rows).to_csv(csv_path, index=False)
                            print(f"  {method}/{target_name} N={n_size} "
                                  f"seed{seed}_fold{fold_i} "
                                  f"r={metrics['pearson']:+.3f}", flush=True)
                        except Exception as exc:
                            rows.append({
                                "method": method, "target": target_name,
                                "N": n_size, "seed": seed, "fold": fold_i,
                                "error": str(exc),
                            })
                            pd.DataFrame(rows).to_csv(csv_path, index=False)
                            print(f"  FAILED {method}/{target_name} N={n_size}: {exc}",
                                  flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No evaluations recorded.")
        return

    eval_df = df[df["target"].isin(["fluid_intelligence", "working_memory"])
                 & df["pearson"].notna()].copy()
    agg = (
        eval_df.groupby(["method", "target", "N"])
        .agg(n_evals=("fold", "size"),
             pearson_mean=("pearson", "mean"),
             pearson_std=("pearson", "std"),
             rmse_mean=("rmse", "mean"),
             rmse_std=("rmse", "std"))
        .reset_index()
    )
    agg.to_csv(out_root / "summary.csv", index=False)

    save_json({
        "methods": ["ncr_contrastive", "no_prior_ridge"],
        "targets": ["fluid_intelligence", "working_memory"],
        "nsizes": args.nsizes, "seeds": sorted(seeds),
        "n_folds": n_folds, "val_fraction": val_fraction,
        "n_evaluations": int(len(df)),
    }, out_root / "run_metadata.json")
    print(f"Saved {len(df)} evaluations ({total_new} new) to {out_root}")


if __name__ == "__main__":
    main()
