#!/usr/bin/env python3
"""Leakage-free prior-weighted FC+SC Ridge baselines.

The method standardizes FC/SC edge features within the development split,
then scales each standardized feature by sqrt(w_ij), where
w_ij = epsilon + (1-epsilon) p_i p_j. Ordinary Ridge on the scaled features
is equivalent to a generalized Ridge penalty beta_ij^2 / w_ij.

Methods:
  PW_TRUE      true working-memory prior
  PW_SHUFFLED  anatomically shuffled control
  PW_RANDOM    random control

Both alpha and epsilon are selected only on the inner validation split.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from metascfc.benchmark_utils import (
    aggregate_split_metrics, atomic_write_csv, iter_nested_splits,
    load_connectomes, prediction_metrics, save_json, seed_level_metrics,
)


def upper_triangle_features(mats: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    iu = np.triu_indices(mats.shape[1], k=1)
    return mats[:, iu[0], iu[1]].astype(np.float32), iu


def load_roi_prior(path: str | Path, n_rois: int) -> np.ndarray:
    df = pd.read_csv(path)
    if "roi_index" in df.columns:
        df = df.sort_values("roi_index")
    if "prior_score" not in df.columns:
        raise ValueError(f"{path} lacks prior_score")
    p = df["prior_score"].to_numpy(np.float64)
    if p.shape != (n_rois,):
        raise ValueError(f"Prior {path} has shape {p.shape}; expected {(n_rois,)}")
    p = np.clip(p, 0.0, None)
    if p.max() > p.min():
        p = (p - p.min()) / (p.max() - p.min())
    return p


def edge_weight_vector(prior: np.ndarray, iu, epsilon: float, modalities: int = 2) -> np.ndarray:
    outer = np.outer(prior, prior)
    w = float(epsilon) + (1.0 - float(epsilon)) * outer[iu]
    w = np.clip(w, 1e-6, None)
    return np.tile(w, modalities).astype(np.float64)


def dual_predictions(x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, alpha: float) -> np.ndarray:
    scale = float(max(1, x_train.shape[1]))
    kernel = (x_train @ x_train.T) / scale
    eigvals, eigvecs = np.linalg.eigh(kernel)
    projected = eigvecs.T @ y_train
    dual = eigvecs @ (projected / (eigvals + float(alpha) + 1e-10))
    return ((x_eval @ x_train.T) / scale) @ dual


def select_and_fit(
    x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
    test_idx: np.ndarray, prior: np.ndarray, iu,
    alphas: Iterable[float], epsilons: Iterable[float],
) -> Tuple[np.ndarray, float, float, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx]).astype(np.float64, copy=False)
    x_val = scaler.transform(x[val_idx]).astype(np.float64, copy=False)
    y_mean, y_std = float(y[train_idx].mean()), float(y[train_idx].std())
    y_std = y_std if y_std >= 1e-8 else 1.0
    yz = (y[train_idx] - y_mean) / y_std

    best = (float("inf"), None, None)
    # The feature weights change with epsilon, so one eigendecomposition is
    # required per epsilon. It is then reused across the complete alpha grid.
    for epsilon in epsilons:
        scale_vec = np.sqrt(edge_weight_vector(prior, iu, epsilon))
        xt = x_train * scale_vec
        xv = x_val * scale_vec
        kernel_scale = float(max(1, xt.shape[1]))
        kernel = (xt @ xt.T) / kernel_scale
        eigvals, eigvecs = np.linalg.eigh(kernel)
        projected = eigvecs.T @ yz
        kernel_val = (xv @ xt.T) / kernel_scale
        for alpha in alphas:
            dual = eigvecs @ (projected / (eigvals + float(alpha) + 1e-10))
            pred_z = kernel_val @ dual
            rmse = prediction_metrics(y[val_idx], pred_z * y_std + y_mean)["rmse"]
            candidate = (rmse, float(alpha), float(epsilon))
            if candidate < best:
                best = candidate
    best_rmse, best_alpha, best_epsilon = best
    if best_alpha is None or best_epsilon is None:
        raise RuntimeError("No alpha/epsilon candidate was selected")

    fit_idx = np.concatenate([train_idx, val_idx])
    final_scaler = StandardScaler()
    x_fit = final_scaler.fit_transform(x[fit_idx]).astype(np.float64, copy=False)
    x_test = final_scaler.transform(x[test_idx]).astype(np.float64, copy=False)
    fit_mean, fit_std = float(y[fit_idx].mean()), float(y[fit_idx].std())
    fit_std = fit_std if fit_std >= 1e-8 else 1.0
    scale_vec = np.sqrt(edge_weight_vector(prior, iu, best_epsilon))
    pred_z = dual_predictions(
        x_fit * scale_vec, (y[fit_idx] - fit_mean) / fit_std,
        x_test * scale_vec, best_alpha,
    )
    return pred_z * fit_std + fit_mean, best_alpha, best_epsilon, best_rmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aaai/prior_weighted_ridge.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, help="Optional seed override for smoke/partial runs")
    ap.add_argument("--methods", nargs="*", help="Optional method-ID override")
    ap.add_argument("--folds", nargs="*", type=int, help="Optional fold-index override")
    args = ap.parse_args()
    cfg: Dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    thread_controller = threadpool_limits(limits=int(cfg.get("n_threads", 4)))

    out_dir = Path(cfg.get("output_dir", "outputs/aaai/prior_weighted_ridge"))
    complete = out_dir / "COMPLETE"
    if args.overwrite and out_dir.exists():
        import shutil; shutil.rmtree(out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    fc, sc, y, subject_ids, groups = load_connectomes(cfg["data"])
    x_fc, iu = upper_triangle_features(fc)
    x_sc, _ = upper_triangle_features(sc)
    x = np.concatenate([x_fc, x_sc], axis=1)
    selected_method_ids = args.methods if args.methods else list(cfg["methods"].keys())
    unknown = set(selected_method_ids) - set(cfg["methods"])
    if unknown:
        raise ValueError(f"Unknown method IDs: {sorted(unknown)}")
    priors = {
        method_id: (cfg["methods"][method_id]["name"], load_roi_prior(cfg["methods"][method_id]["path"], fc.shape[1]))
        for method_id in selected_method_ids
    }
    seeds = args.seeds if args.seeds is not None and len(args.seeds) else [int(s) for s in cfg["seeds"]]
    n_folds = int(cfg.get("n_folds", 5)); val_fraction = float(cfg.get("val_fraction", 0.15))
    alphas = [float(x) for x in cfg["ridge_alphas"]]
    epsilons = [float(x) for x in cfg["epsilon_grid"]]

    existing_path = out_dir / "split_metrics.csv"
    rows = [] if args.overwrite or not existing_path.exists() else pd.read_csv(existing_path).to_dict("records")
    completed = {(r["method_id"], int(r["seed"]), int(r["fold"])) for r in rows}

    selected_folds = set(args.folds) if args.folds else None
    for seed, fold, train_idx, val_idx, test_idx in iter_nested_splits(y, seeds, n_folds, val_fraction, groups):
        if selected_folds is not None and fold not in selected_folds:
            continue
        split_id = f"seed{seed:02d}_fold{fold:02d}"
        for method_id, (method_name, prior) in priors.items():
            if (method_id, seed, fold) in completed:
                print(f"SKIP {method_id} {split_id}")
                continue
            started = time.time()
            pred, alpha, epsilon, val_rmse = select_and_fit(
                x, y, train_idx, val_idx, test_idx, prior, iu, alphas, epsilons,
            )
            metrics = prediction_metrics(y[test_idx], pred)
            row = {
                "method_id": method_id, "method_name": method_name,
                "method_family": "prior_weighted_ridge", "seed": seed, "fold": fold,
                "split_id": split_id, "n_train": len(train_idx), "n_val": len(val_idx),
                "n_test": len(test_idx), "best_alpha": alpha, "best_epsilon": epsilon,
                "best_val_rmse": val_rmse, "runtime_seconds": time.time() - started,
                "group_aware": groups is not None, **metrics,
            }
            rows.append(row); completed.add((method_id, seed, fold))
            pd.DataFrame({
                "subject_index": test_idx, "subject_id": subject_ids[test_idx],
                "target": y[test_idx], "prediction": pred, "seed": seed, "fold": fold,
                "method_id": method_id,
            }).to_csv(out_dir / "predictions" / f"{method_id}_{split_id}.csv", index=False)
            atomic_write_csv(pd.DataFrame(rows), existing_path)
            print(method_id, split_id, json.dumps(metrics), f"alpha={alpha} epsilon={epsilon}")

    split_df = pd.DataFrame(rows).sort_values(["method_id", "seed", "fold"])
    summary = aggregate_split_metrics(split_df)
    seed_df = seed_level_metrics(split_df)
    atomic_write_csv(summary, out_dir / "summary.csv")
    atomic_write_csv(seed_df, out_dir / "seed_level_metrics.csv")
    latex = summary[["Method", "Pearson Mean", "Pearson Std", "RMSE Mean", "RMSE Std", "MAE Mean", "MAE Std"]].copy()
    latex["Pearson $\\uparrow$"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(latex.pop("Pearson Mean"), latex.pop("Pearson Std"))]
    latex["RMSE $\\downarrow$"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(latex.pop("RMSE Mean"), latex.pop("RMSE Std"))]
    latex["MAE $\\downarrow$"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(latex.pop("MAE Mean"), latex.pop("MAE Std"))]
    (out_dir / "summary.tex").write_text(latex.to_latex(index=False, escape=False), encoding="utf-8")
    save_json({"config": cfg, "seeds_run": seeds, "n_subjects": len(y), "n_rois": fc.shape[1]}, out_dir / "run_metadata.json")
    configured_methods = set(cfg["methods"].keys())
    configured_seeds = set(int(v) for v in cfg["seeds"])
    full = split_df[split_df.method_id.isin(configured_methods) & split_df.seed.isin(configured_seeds)]
    expected = len(configured_methods) * len(configured_seeds) * n_folds
    if len(full) == expected and not full.duplicated(["method_id", "seed", "fold"]).any():
        complete.write_text("ok\n", encoding="utf-8")
    print(f"Saved {len(split_df)} evaluations to {out_dir}")


if __name__ == "__main__":
    main()
