#!/usr/bin/env python3
"""Method 4 (ICLR 2027): LLM-Gated Cross-Modal Graph Attention Transformer.

Runs the exact repeated nested-CV protocol of Methods 1-3 (10 seeds, 5 outer
folds, 15% inner validation, identical data loaders and splitters):

    e_ij = (1 - alpha) * LeakyReLU(a^T [W_f h_i | W_s h_j]) + alpha * (p_i + p_j)

where p is the zero-shot LLM-generated semantic prior (scripts/
46_generate_llm_priors.py) and alpha = sigmoid(rho) is a learnable per-layer
bypass gate (adaptive prior routing; see
src/metascfc/models/llm_gated_transformer.py).  The learned alpha and the
converged Information Bottleneck metrics (I(X;Z), I(Z;Y)) are recorded per
split when --track-ib is passed.

Dual-task matrix: point --config at configs/iclr/llm_wm_prior.yaml (HCP
ListSort_Unadj working memory) or configs/iclr/llm_fluid_prior.yaml (HCP
PMAT24_A_CR fluid intelligence).

Methods
-------
  LLMT_TRUE      LLM semantic prior for the target task
  LLMT_SHUFFLED  anatomically shuffled control (--controls in script 46)
  LLMT_RANDOM    canonical random control

Outputs (identical schema to scripts/41_run_meta_gat.py):
  split_metrics.csv / summary.csv / seed_level_metrics.csv / summary.tex
  predictions/{method_id}_seed{ss}_fold{f}.csv
  saliency/{method_id}/seed{ss}_fold{f}.npz   (node_saliency)
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
from metascfc.metrics import IBEpochTracker
from metascfc.models.iclr_backbones import fit_predict_llm_gated


def load_roi_prior(path: str | Path, n_rois: int) -> np.ndarray:
    """Load and min-max normalize an ROI-level prior (same contract as script 40)."""
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


def prior_alignment_metrics(saliency: np.ndarray, prior: np.ndarray,
                            topk: int = 10) -> Dict[str, float]:
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
    ap.add_argument("--config", default="configs/iclr/llm_wm_prior.yaml")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seeds", nargs="*", type=int,
                    help="Optional seed override for smoke/partial runs")
    ap.add_argument("--methods", nargs="*", help="Optional method-ID override")
    ap.add_argument("--folds", nargs="*", type=int, help="Optional fold-index override")
    ap.add_argument("--ib-method", choices=("gaussian", "mine"), default="gaussian")
    ap.add_argument("--track-ib", action="store_true",
                    help="Record per-epoch/converged Information Bottleneck "
                         "metrics and the learned bypass alpha per split")
    args = ap.parse_args()

    cfg: Dict = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir = Path(cfg["output_dir"])
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
    print(f"Target measure: {cfg.get('target_measure', 'unknown')}")

    selected_method_ids = args.methods if args.methods else list(cfg["methods"].keys())
    unknown = set(selected_method_ids) - set(cfg["methods"])
    if unknown:
        raise ValueError(f"Unknown method IDs: {sorted(unknown)}")

    priors: Dict[str, np.ndarray] = {}
    for method_id in selected_method_ids:
        priors[method_id] = load_roi_prior(cfg["methods"][method_id]["path"], n_rois)
        print(
            f"{method_id}: {len(priors[method_id])} ROIs, "
            f"prior range [{priors[method_id].min():.3f}, {priors[method_id].max():.3f}]",
            flush=True,
        )

    seeds = args.seeds if args.seeds is not None and len(args.seeds) else [int(s) for s in cfg["seeds"]]
    n_folds = int(cfg.get("n_folds", 5))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    topk_align = int(cfg.get("alignment_topk", 10))

    existing_path = out_dir / "split_metrics.csv"
    rows = [] if args.overwrite or not existing_path.exists() else pd.read_csv(existing_path).to_dict("records")
    completed = {(r["method_id"], int(r["seed"]), int(r["fold"])) for r in rows}

    fixed = {
        "top_percent": float(cfg.get("top_percent_sc", 10.0)),
        "n_layers": int(cfg.get("n_layers", 2)),
        "heads": int(cfg.get("heads1", 4)),
        "weight_decay": float(cfg.get("weight_decay", 1e-4)),
        "epochs": int(cfg.get("epochs", 60)),
        "patience": int(cfg.get("patience", 15)),
        "min_epochs": int(cfg.get("min_epochs", 10)),
        "alpha_init": float(cfg.get("alpha_init", cfg.get("lambda_init", 0.5))),
        "grad_clip": float(cfg.get("grad_clip", 5.0)),
    }

    selected_folds = set(args.folds) if args.folds else None
    for seed, fold, train_idx, val_idx, test_idx in iter_nested_splits(y, seeds, n_folds, val_fraction, groups):
        if selected_folds is not None and fold not in selected_folds:
            continue
        split_id = f"seed{seed:02d}_fold{fold:02d}"
        split_seed = seed * 1000 + fold
        for method_id in selected_method_ids:
            if (method_id, seed, fold) in completed:
                print(f"SKIP {method_id} {split_id}")
                continue
            started = time.time()
            tracker = (IBEpochTracker(method=args.ib_method)
                       if args.track_ib else None)
            pred, best_cfg, best_val_rmse, best_epoch, saliency, n_params = fit_predict_llm_gated(
                fc, sc, y, train_idx, val_idx, test_idx,
                priors[method_id],
                hidden_grid=[float(h) for h in cfg["hidden_grid"]],
                dropout_grid=[float(d) for d in cfg["dropout_grid"]],
                lr_grid=[float(l) for l in cfg["lr_grid"]],
                device=device,
                seed=split_seed,
                ib_tracker=tracker,
                **fixed,
            )
            metrics = prediction_metrics(y[test_idx], pred)
            alignment = prior_alignment_metrics(saliency, priors[method_id], topk_align)
            row = {
                "method_id": method_id, "method_name": cfg["methods"][method_id]["name"],
                "method_family": "llm_gated_transformer", "seed": seed, "fold": fold,
                "split_id": split_id, "n_train": len(train_idx), "n_val": len(val_idx),
                "n_test": len(test_idx),
                "best_hidden": best_cfg["hidden"], "best_dropout": best_cfg["dropout"],
                "best_learning_rate": best_cfg["learning_rate"],
                "best_epoch": best_epoch, "best_val_rmse": best_val_rmse,
                "parameters": n_params, "runtime_seconds": time.time() - started,
                "device": str(device), "group_aware": groups is not None,
                **metrics, **alignment,
            }
            if tracker is not None:
                if tracker.final:
                    row["I_XZ_final"] = float(tracker.final["I_XZ"])
                    row["I_ZY_final"] = float(tracker.final["I_ZY"])
                    row["probe_r2_final"] = float(tracker.final["probe_r2"])
                if getattr(tracker, "alpha_final", None):
                    row["bypass_alpha_mean"] = float(np.mean(tracker.alpha_final))
                (out_dir / "ib_tracking").mkdir(parents=True, exist_ok=True)
                save_json(
                    {"config": {k: tracker.to_dict()[k] for k in ("noise_floor", "epochs", "I_XZ", "I_ZY")},
                     "alpha_final": tracker.alpha_final},
                    out_dir / "ib_tracking" / f"{method_id}_{split_id}.json",
                )
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
                f"cfg=hid{best_cfg['hidden']}_do{best_cfg['dropout']}_lr{best_cfg['learning_rate']}"
                f" epoch={best_epoch} val_rmse={best_val_rmse:.3f} params={n_params:,}",
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
        "target_measure": cfg.get("target_measure"),
        "n_subjects": len(y),
        "n_rois": n_rois,
        "n_candidates": len(cfg["hidden_grid"]) * len(cfg["dropout_grid"]) * len(cfg["lr_grid"]),
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
