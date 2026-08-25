#!/usr/bin/env python3
"""Validation-Selected Discrete Routing verification (post Gradient-Absorption).

The continuous bypass gate failed for a structural reason (Gradient
Absorption): the downstream learned branch can absorb any gate change, so the
rho-gradient is flat and noise-dominated - no initialization or
regularization weight fixes that.  The gate is therefore no longer learned:
alpha is a FIXED hyperparameter selected per split from a discrete grid by
inner-validation RMSE (leakage-free nested CV).

This script verifies the mechanism on the REAL data with the CONTRASTIVE
qwen3.8:27b priors (1 seed x 1 fold, fluid-intelligence target):

- matched prior     (LLM-Fluid contrastive, fluid target)
- mismatched prior  (LLM-WM contrastive, fluid target)

For each, it reports the selected alpha and test metrics.  Routing is
"expressed" when the two arms select DIFFERENT alphas in the predicted
direction (matched higher, mismatched lower).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from metascfc.benchmark_utils import (
    choose_device,
    iter_nested_splits,
    load_connectomes,
    prediction_metrics,
)
from metascfc.metrics import IBEpochTracker
from metascfc.models.iclr_backbones import fit_predict_llm_gated

DEFAULT_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def load_prior(path: str, n_rois: int) -> np.ndarray:
    df = pd.read_csv(path).sort_values("roi_index")
    p = df.prior_score.to_numpy(np.float64)
    p = np.clip(p, 0.0, None)
    return (p - p.min()) / (p.max() - p.min())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--target", default="fluid_intelligence",
                    choices=("fluid_intelligence", "working_memory"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--alpha-grid", type=float, nargs="+", default=DEFAULT_GRID)
    ap.add_argument("--matched-prior", default=None,
                    help="Matched-prior CSV (default: contrastive qwen3.8 fluid)")
    ap.add_argument("--mismatched-prior", default=None,
                    help="Mismatched-prior CSV (default: contrastive qwen3.8 WM)")
    ap.add_argument("--out-dir", default="outputs/iclr/discrete_routing_qwen3/alpha_check")
    args = ap.parse_args()

    matched_path = args.matched_prior or \
        "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv"
    mismatched_path = args.mismatched_prior or \
        "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print("Device:", device)
    print(f"Alpha grid: {args.alpha_grid}")

    fc, sc, y, _, groups = load_connectomes({
        "fc_path": "inputs/dataset_FC/FC_all.npy",
        "sc_path": "inputs/dataset_SC/SC_all.npy",
        "y_path": "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy"
        if args.target == "working_memory"
        else "inputs/dataset_SC/label_all.npy",
    })
    n_rois = fc.shape[1]
    seed, fold = args.seed, args.fold
    _, _, train_idx, val_idx, test_idx = next(
        iter_nested_splits(y, [seed], 5, 0.15, groups))
    print(f"Split seed{seed:02d}_fold{fold:02d}: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    cells = {
        "matched_llm_fluid": matched_path,
        "mismatched_llm_wm": mismatched_path,
    }
    results = {}
    for name, path in cells.items():
        print(f"\n=== {name} ({path}) ===", flush=True)
        prior = load_prior(path, n_rois)
        tracker = IBEpochTracker(method="mine", mine_steps=40)
        started = time.time()
        pred, cfg, val_rmse, best_epoch, saliency, n_params = fit_predict_llm_gated(
            fc, sc, y, train_idx, val_idx, test_idx, prior,
            hidden_grid=[16], dropout_grid=[0.2], lr_grid=[1e-3],
            device=device, n_layers=2, heads=4, weight_decay=1e-4,
            epochs=args.epochs, patience=12, min_epochs=10,
            alpha_grid=args.alpha_grid, grad_clip=5.0,
            seed=seed * 1000 + fold, ib_tracker=tracker,
        )
        metrics = prediction_metrics(y[test_idx], pred)
        selected = tracker.selected_alpha
        print(f"  SELECTED alpha = {selected}  "
              f"(grid {args.alpha_grid}; inner-val RMSE {val_rmse:.4f})")
        print(f"  test metrics: {metrics}")
        print(f"  IB: I_XZ={tracker.final['I_XZ']:.3f} "
              f"I_ZY={tracker.final['I_ZY']:.3f} "
              f"| best_epoch={best_epoch} | {time.time() - started:.0f}s")
        results[name] = {
            "prior_path": path,
            "selected_alpha": selected,
            "alpha_grid": args.alpha_grid,
            "metrics": metrics,
            "I_XZ_final": tracker.final["I_XZ"],
            "I_ZY_final": tracker.final["I_ZY"],
            "best_epoch": best_epoch, "n_params": n_params,
        }

    a_m = results["matched_llm_fluid"]["selected_alpha"]
    a_x = results["mismatched_llm_wm"]["selected_alpha"]
    print("\n================ VERDICT ================")
    print(f"matched    llm_fluid: selected alpha = {a_m}")
    print(f"mismatched llm_wm   : selected alpha = {a_x}")
    if a_m > a_x + 1e-9:
        print("-> Discrete routing EXPRESSED: validation selects a higher "
              "prior weight for the matched prior than for the mismatched one.")
    elif a_m == a_x:
        print("-> Both arms selected the same alpha at this split (one split "
              "is weak evidence; run the full matrix for the distribution).")
    else:
        print("-> Routing expressed in the OPPOSITE direction at this split.")

    (out_dir / "alpha_check.json").write_text(json.dumps({
        "seed": seed, "fold": fold, "target": args.target,
        "alpha_grid": args.alpha_grid, "results": results,
    }, indent=2))
    print(f"Saved {out_dir / 'alpha_check.json'}")


if __name__ == "__main__":
    main()
