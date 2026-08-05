#!/usr/bin/env python3
"""Method 3 (ICLR 2027): Two-Stage Biomarker-Guided Kernel Ridge.

Stage 1 uses the *already computed* per-split node saliency of the AAAI
MS-Inter-GCN runs (E0 / E7 / E8 / E9) as an explicit feature-space
projector; Stage 2 trains a Kernel Ridge Regression (RBF) on the gated
[FC_upper | SC_upper] feature space:

    Stage 1 (per split): c in [0,1]^116 = node saliency of an AAAI E* model
    Stage 2 (per split): X_gated = X_std ⊙ [lift(c) | lift(c)],
                         K(s,t) = k(x_s ⊙ c, x_t ⊙ c) with k = RBF,
                         KRR trained on the gated space.

Methods
-------
  M3_E0       stage-1 biomarker: no-prior GCN saliency  (AAAI E0_baseline)
  M3_TRUE     stage-1 biomarker: true edge-prior GCN    (AAAI E7_edge_true)
  M3_SHUFFLED stage-1 biomarker: shuffled edge-prior    (AAAI E8_edge_shuffled)
  M3_RANDOM   stage-1 biomarker: random edge-prior      (AAAI E9_edge_random)

The (alpha, gamma) grid is selected only on the inner validation split; the
winner is refit on train+val and evaluated on the outer test split.  The
true/shuffled/random variants share identical grids, seeds, and folds.  The
Stage-1 gate saliency used per split is exported for the biomarker
alignment / rank stability pipeline.

Outputs (identical schema to scripts/40 and scripts/41):
  split_metrics.csv / summary.csv / seed_level_metrics.csv / summary.tex
  predictions/{method_id}_{split_id}.csv
  saliency/{method_id}/{split_id}.npz   (node_saliency = Stage-1 gate)
  run_metadata.json / COMPLETE
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr

from metascfc.benchmark_utils import (
    aggregate_split_metrics,
    atomic_write_csv,
    choose_device,
    iter_nested_splits,
    load_connectomes,
    prediction_metrics,
    save_json,
    seed_level_metrics,
)
from metascfc.models.iclr_backbones import fit_predict_two_stage_krr, load_split_node_saliency


def load_roi_prior(path: str | Path, n_rois: int) -> np.ndarray:
    """Load and min-max normalize an ROI-level prior (same contract as scripts 40/41)."""
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
    """Biomarker alignment of a node-saliency vector with the (true) prior."""
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
    ap.add_argument("--config", default="configs/aaai/two_stage_kernel_ridge.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int, help="Optional seed override for smoke/partial runs")
    ap.add_argument("--methods", nargs="*", help="Optional method-ID override")
    ap.add_argument("--folds", nargs="*", type=int, help="Optional fold-index override")
    args = ap.parse_args()

    cfg: Dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg.get("output_dir", "outputs/aaai/two_stage_kernel_ridge"))
    complete = out_dir / "COMPLETE"
    if args.overwrite and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (out_dir / "saliency").mkdir(parents=True, exist_ok=True)
    for method_id in cfg["methods"]:
        (out_dir / "saliency" / method_id).mkdir(parents=True, exist_ok=True)

    device = choose_device(cfg.get("device", "auto"))
    print("Device:", device)

    fc, sc, y, subject_ids, groups = load_connectomes(cfg["data"])
    n_rois = fc.shape[1]

    selected_method_ids = args.methods if args.methods else list(cfg["methods"].keys())
    unknown = set(selected_method_ids) - set(cfg["methods"])
    if unknown:
        raise ValueError(f"Unknown method IDs: {sorted(unknown)}")

    # True working-memory prior, used for the biomarker-alignment metric of
    # every method (so M3_E0 is not trivially aligned/diluted by design).
    prior_true = load_roi_prior(cfg.get("alignment_prior", ""), n_rois)

    methods_cfg: Dict[str, Dict] = cfg["methods"]
    stage1_dirs: Dict[str, Path] = {}
    for method_id in selected_method_ids:
        stage1_dirs[method_id] = Path(methods_cfg[method_id]["saliency_dir"])
        n_sal = len(list(stage1_dirs[method_id].glob("*.npz")))
        print(
            f"{method_id}: stage-1 saliency dir = {stage1_dirs[method_id]} "
            f"({n_sal} split files)", flush=True,
        )

    seeds = args.seeds if args.seeds is not None and len(args.seeds) else [int(s) for s in cfg["seeds"]]
    n_folds = int(cfg.get("n_folds", 5))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    topk_align = int(cfg.get("alignment_topk", 10))

    existing_path = out_dir / "split_metrics.csv"
    rows = [] if args.overwrite or not existing_path.exists() else pd.read_csv(existing_path).to_dict("records")
    completed = {(r["method_id"], int(r["seed"]), int(r["fold"])) for r in rows}

    fixed = {
        "gate_mode": str(cfg.get("gate_mode", "product")),
        "verbose": False,
    }

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
            saliency = load_split_node_saliency(stage1_dirs[method_id], seed, fold)
            pred, best_cfg, best_val_rmse, gate_saliency, n_params = fit_predict_two_stage_krr(
                fc, sc, y, train_idx, val_idx, test_idx,
                saliency=saliency,
                alpha_grid=[float(a) for a in cfg["alpha_grid"]],
                gamma_grid=[float(g) for g in cfg["gamma_grid"]],
                **fixed,
            )
            metrics = prediction_metrics(y[test_idx], pred)
            alignment = prior_alignment_metrics(gate_saliency, prior_true, topk_align)
            row = {
                "method_id": method_id, "method_name": methods_cfg[method_id]["name"],
                "method_family": "two_stage_kernel_ridge", "seed": seed, "fold": fold,
                "split_id": split_id, "n_train": len(train_idx), "n_val": len(val_idx),
                "n_test": len(test_idx),
                "best_alpha": best_cfg["alpha"], "best_gamma": best_cfg["gamma"],
                "best_kernel": best_cfg["kernel"], "gate_mode": best_cfg["gate_mode"],
                "best_val_rmse": best_val_rmse,
                "parameters": n_params, "runtime_seconds": time.time() - started,
                "device": str(device), "group_aware": groups is not None,
                **metrics, **alignment,
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
                node_saliency=gate_saliency,
            )
            atomic_write_csv(pd.DataFrame(rows), existing_path)
            print(
                method_id, split_id, json.dumps(metrics),
                f"cfg=alpha{best_cfg['alpha']}_gamma{best_cfg['gamma']} "
                f"val_rmse={best_val_rmse:.3f} params={n_params:,}",
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
        "n_edge_features": int(np.triu_indices(n_rois, 1)[0].size * 2),
        "n_candidates": len(cfg["alpha_grid"]) * len(cfg["gamma_grid"]),
        "device": str(device),
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