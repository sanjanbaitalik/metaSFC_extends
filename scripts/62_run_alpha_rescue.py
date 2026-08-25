#!/usr/bin/env python3
"""Alpha-Rescue verification: does adaptive prior routing come alive?

Runs ONE seed x ONE fold of the LLM-gated transformer on the REAL fluid
intelligence target with the two REAL zero-shot LLM priors:

- ``llm_fluid``  (MATCHED prior:  LLM fluid-intelligence prior, fluid target)
- ``llm_wm``     (MISMATCHED prior: LLM working-memory prior, fluid target)

with the alpha-rescue fixes enabled:

1. ``alpha_init = 0.1`` - the gate starts by trusting the DATA, not the
   prior (sigmoid-logit parameterization, so alpha stays in (0, 1)).
2. Anti-dead-zone reward ``-1e-4 * |alpha - 0.5|`` per layer (sign-corrected:
   an additive +|alpha-0.5| penalty would pin the gate AT 0.5).
3. Per-epoch alpha trajectory logged through the IB tracker.

Expected qualitative outcome (printed to console): the MATCHED gate moves
UP from 0.1 (toward trusting the informative prior); the MISMATCHED gate
stays low / moves further down (bypassing the uninformative prior).

Results are saved to outputs/iclr/alpha_rescue/alpha_rescue.json.
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
    ap.add_argument("--alpha-inits", type=float, nargs="+", default=[0.1, 0.5],
                    help="Bypass-gate initializations to compare. 0.1 is the "
                         "mandated data-first init; 0.5 is the cold-start "
                         "control (prior has influence at init, so the MSE "
                         "gradient can actually steer the gate).")
    ap.add_argument("--out-dir", default="outputs/iclr/alpha_rescue")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print("Device:", device)

    fc, sc, y, _, groups = load_connectomes({
        "fc_path": "inputs/dataset_FC/FC_all.npy",
        "sc_path": "inputs/dataset_SC/SC_all.npy",
        "y_path": "inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy"
        if args.target == "working_memory"
        else "inputs/dataset_SC/label_all.npy",
    })
    n_rois = fc.shape[1]
    seed, fold = args.seed, args.fold
    train_idx, val_idx, test_idx = next(iter_nested_splits(
        y, [seed], 5, 0.15, groups))[2:]
    print(f"Split seed{seed:02d}_fold{fold:02d}: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    cells = {
        "matched_llm_fluid": "outputs/priors/llm/fluid_intelligence/roi_prior.csv",
        "mismatched_llm_wm": "outputs/priors/llm/working_memory/roi_prior.csv",
    }
    results = {}
    for name, path in cells.items():
        for alpha_init in args.alpha_inits:
            key = f"{name}_init{alpha_init:.2f}"
            print(f"\n=== {key} ({path}) ===", flush=True)
            prior = load_prior(path, n_rois)
            tracker = IBEpochTracker(method="mine", mine_steps=40)
            started = time.time()
            pred, cfg, val_rmse, best_epoch, saliency, n_params = fit_predict_llm_gated(
                fc, sc, y, train_idx, val_idx, test_idx, prior,
                hidden_grid=[16], dropout_grid=[0.2], lr_grid=[1e-3],
                device=device, n_layers=2, heads=4, weight_decay=1e-4,
                epochs=args.epochs, patience=12, min_epochs=10,
                alpha_init=alpha_init, alpha_explore_weight=1e-4, grad_clip=5.0,
                seed=seed * 1000 + fold, ib_tracker=tracker,
            )
            metrics = prediction_metrics(y[test_idx], pred)
            alphas_final = tracker.alpha_final or []
            traj = [float(np.mean(a)) for a in tracker.alpha_epochs]
            print(f"  alpha trajectory (mean over layers): "
                  f"{' -> '.join(f'{a:.3f}' for a in traj[:: max(1, len(traj)//8)])}")
            print(f"  FINAL alpha per layer: "
                  f"{[round(a, 4) for a in alphas_final]}  (init {alpha_init})")
            print(f"  test metrics: {metrics}")
            print(f"  IB: I_XZ={tracker.final['I_XZ']:.3f} "
                  f"I_ZY={tracker.final['I_ZY']:.3f} "
                  f"| val_rmse={val_rmse:.3f} best_epoch={best_epoch} "
                  f"| {time.time() - started:.0f}s")
            results[key] = {
                "prior_path": path, "alpha_init": alpha_init,
                "alpha_final_per_layer": alphas_final,
                "alpha_trajectory_mean": traj,
                "metrics": metrics,
                "I_XZ_final": tracker.final["I_XZ"],
                "I_ZY_final": tracker.final["I_ZY"],
                "best_epoch": best_epoch, "n_params": n_params,
            }

    print("\n================ VERDICT ================")
    for alpha_init in args.alpha_inits:
        m_key = f"matched_llm_fluid_init{alpha_init:.2f}"
        x_key = f"mismatched_llm_wm_init{alpha_init:.2f}"
        m = np.mean(results[m_key]["alpha_final_per_layer"])
        x = np.mean(results[x_key]["alpha_final_per_layer"])
        m0 = results[m_key]["alpha_trajectory_mean"][0]
        print(f"init={alpha_init:.2f}: matched {m0:.3f}->{m:.3f} | "
              f"mismatched {results[x_key]['alpha_trajectory_mean'][0]:.3f}->{x:.3f}")
        if m > x + 0.05:
            print("  -> routing expressed: matched gate trusts the prior more.")
        elif abs(m - x) <= 0.05 and abs(m - m0) <= 0.02:
            print("  -> gate QUASI-STATIONARY at this budget. Known causes: "
                  "(a) the learned branch absorbs gate changes, so the loss "
                  "is nearly flat in rho and its gradient is noise-dominated; "
                  "(b) if the two LLM priors are highly correlated there is "
                  "little match/mismatch contrast to express (check the "
                  "prior-overlap diagnostic below).")
        else:
            print("  -> routing NOT expressed at this budget.")
    # Prior-discriminability diagnostic: routing needs OPPOSING directions,
    # which near-identical priors cannot provide.
    wm = load_prior(cells["mismatched_llm_wm"], n_rois)
    fl = load_prior(cells["matched_llm_fluid"], n_rois)
    r = float(np.corrcoef(wm, fl)[0, 1])
    tw = set(np.argsort(wm)[-10:].tolist()); tf = set(np.argsort(fl)[-10:].tolist())
    print(f"\nPrior discriminability: pearson(LLM_WM, LLM_Fluid) = {r:.3f}; "
          f"top-10 overlap = {len(tw & tf)}/10")
    if r > 0.8:
        print("-> The two zero-shot LLM priors are near-duplicates: the "
              "match/mismatch contrast is weak regardless of the gate. "
              "Consider contrastive prompting or a Neurosynth-difference "
              "prior for the routing experiment.")

    (out_dir / "alpha_rescue.json").write_text(json.dumps({
        "seed": seed, "fold": fold, "target": args.target,
        "alpha_init": 0.1, "alpha_explore_weight": 1e-4,
        "epochs_budget": args.epochs, "results": results,
    }, indent=2))
    print(f"Saved {out_dir / 'alpha_rescue.json'}")


if __name__ == "__main__":
    main()
