#!/usr/bin/env python3
"""Method 1 (ICLR 2027): Network-Constrained Prior-Laplacian Ridge.

Runs the exact repeated nested-CV protocol of the AAAI baselines (10 seeds,
5 outer folds, 15% inner validation, identical data loaders and splitters):

    minimize_β  ||y - Xβ||^2  +  λ1·||β||^2  +  λ2·β^T L_prior β

The ROI-level meta-analysis prior is converted into the quadratic penalty
L_prior by lifting the top-k prior-active ROI network to the line graph of
the connectome edge features (see
src/metascfc/models/iclr_backbones/network_constrained_ridge.py).  The
prior thus enters the *predictive inductive bias* directly (weight
smoothness over functionally coupled regions), not as a feature scaler.

Methods
-------
  NCR_TRUE      true working-memory prior (Method 1)
  NCR_SHUFFLED  anatomically shuffled control
  NCR_RANDOM    random control

Both λ1 (ridge) and λ2 (network-constraint strength) are selected only on
the inner validation split.  λ2 = 0.0 degenerates to the plain FC+SC Ridge
baseline (B3), which serves as an internal solver check.

Outputs (identical schema to scripts/27_run_prior_weighted_ridge.py so the
results drop into the prediction-benchmark aggregation):
  split_metrics.csv / summary.csv / seed_level_metrics.csv / summary.tex
  predictions/{method_id}_seed{ss}_fold{f}.csv
  saliency/{method_id}/seed{ss}_fold{f}.npz   (node_saliency, for the
      biomarker alignment / rank-stability pipeline of scripts/35)
  run_metadata.json / COMPLETE
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from threadpoolctl import threadpool_limits

from metascfc.benchmark_utils import (
    aggregate_split_metrics,
    atomic_write_csv,
    iter_nested_splits,
    load_connectomes,
    prediction_metrics,
    save_json,
    seed_level_metrics,
)
from metascfc.models.iclr_backbones import (
    EdgeLaplacian,
    build_edge_laplacian,
    fit_predict_network_constrained,
    node_saliency_from_beta,
)


def upper_triangle_features(mats: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Upper-triangle edge features; indices are identical to the AAAI scripts."""
    iu = np.triu_indices(mats.shape[1], k=1)
    return mats[:, iu[0], iu[1]].astype(np.float32), iu


def load_roi_prior(path: str | Path, n_rois: int) -> np.ndarray:
    """Load and min-max normalize an ROI-level prior (same contract as script 27)."""
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


def prior_alignment_metrics(
    saliency: np.ndarray,
    prior: np.ndarray,
    topk: int = 10,
) -> Dict[str, float]:
    """Biomarker alignment of a node-saliency vector with the prior."""
    r_pearson = float(pearsonr(saliency, prior).statistic)
    r_spearman = float(spearmanr(saliency, prior).statistic)
    if not np.isfinite(r_pearson):
        r_pearson = 0.0
    if not np.isfinite(r_spearman):
        r_spearman = 0.0
    k = min(topk, len(saliency))
    top_sal = set(np.argpartition(saliency, -k)[-k:].tolist())
    top_prior = set(np.argpartition(prior, -k)[-k:].tolist())
    union = top_sal | top_prior
    jaccard = len(top_sal & top_prior) / len(union) if union else 1.0
    return {
        "prior_alignment_pearson": r_pearson,
        "prior_alignment_spearman": r_spearman,
        "prior_alignment_top10_jaccard": float(jaccard),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aaai/network_constrained_ridge.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, help="Optional seed override for smoke/partial runs")
    ap.add_argument("--methods", nargs="*", help="Optional method-ID override")
    ap.add_argument("--folds", nargs="*", type=int, help="Optional fold-index override")
    args = ap.parse_args()
    cfg: Dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    thread_controller = threadpool_limits(limits=int(cfg.get("n_threads", 4)))

    out_dir = Path(cfg.get("output_dir", "outputs/aaai/network_constrained_ridge"))
    complete = out_dir / "COMPLETE"
    if args.overwrite and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "saliency").mkdir(parents=True, exist_ok=True)
    for method_id in cfg["methods"]:
        (out_dir / "saliency" / method_id).mkdir(parents=True, exist_ok=True)

    fc, sc, y, subject_ids, groups = load_connectomes(cfg["data"])
    n_rois = fc.shape[1]
    x_fc, iu = upper_triangle_features(fc)
    x_sc, _ = upper_triangle_features(sc)
    x = np.concatenate([x_fc, x_sc], axis=1)

    selected_method_ids = args.methods if args.methods else list(cfg["methods"].keys())
    unknown = set(selected_method_ids) - set(cfg["methods"])
    if unknown:
        raise ValueError(f"Unknown method IDs: {sorted(unknown)}")

    top_k = int(cfg.get("top_k", 30))
    weighting = str(cfg.get("laplacian_weighting", "binary"))
    normalization = str(cfg.get("laplacian_normalization", "sym"))
    couple_modalities = bool(cfg.get("couple_modalities", False))

    # The Laplacian depends only on the prior (not on the data fold), so it
    # is built once per method.  Isolated line-graph nodes (edges touching no
    # prior-active ROI) are handled inside build_edge_laplacian.
    laplacians: Dict[str, EdgeLaplacian] = {}
    priors: Dict[str, np.ndarray] = {}
    for method_id in selected_method_ids:
        prior = load_roi_prior(cfg["methods"][method_id]["path"], n_rois)
        priors[method_id] = prior
        laplacians[method_id] = build_edge_laplacian(
            n_rois=n_rois,
            prior_scores=prior,
            top_k=top_k,
            weighting=weighting,
            couple_modalities=couple_modalities,
            normalize=normalization,
        )
        print(
            f"{method_id}: {len(priors[method_id])} ROIs, "
            f"top-{top_k} prior-active, {laplacians[method_id].n_active} "
            f"active edge features ({laplacians[method_id].n_edges * 2} total)",
            flush=True,
        )

    seeds = args.seeds if args.seeds is not None and len(args.seeds) else [int(s) for s in cfg["seeds"]]
    n_folds = int(cfg.get("n_folds", 5))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    alpha1_grid = [float(a) for a in cfg["ridge_alphas"]]
    alpha2_grid = [float(a) for a in cfg["laplacian_alphas"]]
    topk_align = int(cfg.get("alignment_topk", 10))

    existing_path = out_dir / "split_metrics.csv"
    rows = [] if args.overwrite or not existing_path.exists() else pd.read_csv(existing_path).to_dict("records")
    completed = {(r["method_id"], int(r["seed"]), int(r["fold"])) for r in rows}

    selected_folds = set(args.folds) if args.folds else None
    for seed, fold, train_idx, val_idx, test_idx in iter_nested_splits(y, seeds, n_folds, val_fraction, groups):
        if selected_folds is not None and fold not in selected_folds:
            continue
        split_id = f"seed{seed:02d}_fold{fold:02d}"
        for method_id in selected_method_ids:
            if (method_id, seed, fold) in completed:
                print(f"SKIP {method_id} {split_id}")
                continue
            started = time.time()
            pred, best_a1, best_a2, best_val_rmse, beta_dev = fit_predict_network_constrained(
                x, y, train_idx, val_idx, test_idx,
                laplacians[method_id], alpha1_grid, alpha2_grid,
            )
            metrics = prediction_metrics(y[test_idx], pred)
            saliency = node_saliency_from_beta(beta_dev, n_rois)
            alignment = prior_alignment_metrics(saliency, priors[method_id], topk_align)
            row = {
                "method_id": method_id, "method_name": cfg["methods"][method_id]["name"],
                "method_family": "network_constrained_ridge", "seed": seed, "fold": fold,
                "split_id": split_id, "n_train": len(train_idx), "n_val": len(val_idx),
                "n_test": len(test_idx), "best_alpha1": best_a1, "best_alpha2": best_a2,
                "best_val_rmse": best_val_rmse, "runtime_seconds": time.time() - started,
                "group_aware": groups is not None, **metrics, **alignment,
            }
            rows.append(row)
            completed.add((method_id, seed, fold))
            pd.DataFrame({
                "subject_index": test_idx, "subject_id": subject_ids[test_idx],
                "target": y[test_idx], "prediction": pred, "seed": seed, "fold": fold,
                "method_id": method_id,
            }).to_csv(out_dir / "predictions" / f"{method_id}_{split_id}.csv", index=False)
            np.savez(
                out_dir / "saliency" / method_id / f"{split_id}.npz",
                node_saliency=saliency,
            )
            atomic_write_csv(pd.DataFrame(rows), existing_path)
            print(
                method_id, split_id, json.dumps(metrics),
                f"lambda1={best_a1} lambda2={best_a2}",
                flush=True,
            )

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
    save_json({
        "config": cfg,
        "seeds_run": seeds,
        "n_subjects": len(y),
        "n_rois": n_rois,
        "n_edge_features": x.shape[1],
        "active_edges": {m: int(l.n_active) for m, l in laplacians.items()},
    }, out_dir / "run_metadata.json")
    configured_methods = set(cfg["methods"].keys())
    configured_seeds = set(int(v) for v in cfg["seeds"])
    full = split_df[split_df.method_id.isin(configured_methods) & split_df.seed.isin(configured_seeds)]
    expected = len(configured_methods) * len(configured_seeds) * n_folds
    if len(full) == expected and not full.duplicated(["method_id", "seed", "fold"]).any():
        complete.write_text("ok\n", encoding="utf-8")
    print(f"Saved {len(split_df)} evaluations to {out_dir}")


if __name__ == "__main__":
    main()
