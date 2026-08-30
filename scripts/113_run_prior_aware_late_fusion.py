"""Runner for Modification 2: Prior-Aware Late Fusion pilot."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from metascfc.benchmark_utils import load_connectomes, prediction_metrics
from metascfc.experiments.prior_aware_late_fusion import (
    RIDGE_GRID,
    GAMMA_GRID,
    LAMBDA_L_GRID,
    run_late_fusion_experiment,
    compute_late_fusion_decision,
)
from metascfc.experiments.msancr_refinement import load_roi_prior, upper_triangle_features


def main(config_path: str = "configs/iclr/prior_aware_late_fusion.yaml") -> None:
    cfg = yaml.safe_load(Path(config_path).read_text())
    output_dir = Path(cfg.get("output_dir", "outputs/iclr/prior_aware_late_fusion"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading HCP data...")
    fc_mats, sc_mats, y_all, subject_ids, groups = load_connectomes(cfg["data"])
    n_rois = int(cfg.get("n_rois", fc_mats.shape[1]))
    print(f"  FC: {fc_mats.shape}, SC: {sc_mats.shape}, y: {y_all.shape}")

    # Upper-triangle features
    X_fc = upper_triangle_features(fc_mats)
    X_sc = upper_triangle_features(sc_mats)
    print(f"  X_fc: {X_fc.shape}, X_sc: {X_sc.shape}")

    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2])]
    n_outer = int(cfg.get("n_outer_folds", 5))
    n_inner = int(cfg.get("n_inner_folds", 3))
    ridge_grid = [float(v) for v in cfg.get("ridge_grid", RIDGE_GRID)]
    gamma_grid = [float(v) for v in cfg.get("gamma_grid", GAMMA_GRID)]
    lambda_l_grid = [float(v) for v in cfg.get("lambda_laplacian_grid", LAMBDA_L_GRID)]
    prior_controls = cfg.get("prior_controls", ["unrelated", "shuffled", "random"])

    all_decisions = {}

    for target_key, target_cfg in cfg.get("targets", {}).items():
        print(f"\n{'='*60}")
        print(f"Running late fusion for: {target_cfg['name']}")
        print(f"{'='*60}")

        # Load target labels
        y_task = np.asarray(
            __import__("numpy").load(target_cfg["label_path"], allow_pickle=False),
            dtype=np.float64,
        ).reshape(-1)
        if len(y_task) != len(y_all):
            raise ValueError(f"Target length {len(y_task)} != subject count {len(y_all)}")

        # Load priors for this task
        prior_cfg = cfg.get("priors", {}).get(target_key, {})
        priors = {}
        for prior_name in ["matched", "unrelated", "shuffled", "random"]:
            path = prior_cfg.get(prior_name)
            if path:
                priors[prior_name] = load_roi_prior(path, n_rois)
            else:
                raise ValueError(f"Missing prior: {prior_name} for {target_key}")

        # Run experiment
        task_output_dir = output_dir / target_key
        result = run_late_fusion_experiment(
            X_fc, X_sc, y_task, priors,
            seeds=seeds,
            n_outer_folds=n_outer,
            n_inner_folds=n_inner,
            ridge_grid=ridge_grid,
            gamma_grid=gamma_grid,
            lambda_l_grid=lambda_l_grid,
            n_rois=n_rois,
            output_dir=str(task_output_dir),
            prior_controls=prior_controls,
        )

        # Compute decision
        decision = compute_late_fusion_decision(
            result["summary"], result["control_summary"], target_key,
        )
        all_decisions[target_key] = decision

        # Save decision
        (task_output_dir / "late_fusion_decision.json").write_text(
            json.dumps(decision, indent=2, default=str)
        )

        print(f"\nResults for {target_key}:")
        for row in result["summary"]:
            print(f"  {row['model']:6s}: Pearson={row['pearson_mean']:.4f} "
                  f"(median={row['pearson_median']:.4f}, std={row['pearson_std']:.4f})")
        print(f"  Decision: {decision['status']}")
        print(f"  Mean delta vs strongest no-prior: {decision['mean_delta_vs_strongest_no_prior']:+.4f}")

    # Cross-task decision
    wm_dec = all_decisions.get("working_memory", {})
    fl_dec = all_decisions.get("fluid_intelligence", {})
    wm_status = wm_dec.get("status", "FAILURE")
    fl_status = fl_dec.get("status", "FAILURE")

    if wm_status in ("LARGE_MARGIN_SUCCESS", "PROMISING") and fl_status in ("LARGE_MARGIN_SUCCESS", "PROMISING"):
        next_step = "full_late_fusion_10x5_both_tasks"
    elif wm_status in ("LARGE_MARGIN_SUCCESS", "PROMISING") and fl_status == "FAILURE":
        next_step = "consider_modification_3"
    elif fl_status in ("LARGE_MARGIN_SUCCESS", "PROMISING") and wm_status == "FAILURE":
        next_step = "human_review_before_full_runs"
    else:
        next_step = "consider_modification_3"

    cross_task = {
        "working_memory": wm_dec,
        "fluid_intelligence": fl_dec,
        "recommended_next_step": next_step,
    }
    (output_dir / "late_fusion_decision.json").write_text(
        json.dumps(cross_task, indent=2, default=str)
    )

    # Print completion report
    print(f"\n{'='*60}")
    print("MODIFICATION 2/3 — PRIOR-AWARE LATE FUSION COMPLETE")
    print(f"{'='*60}")

    for task_name, dec in [("WORKING MEMORY", wm_dec), ("FLUID INTELLIGENCE", fl_dec)]:
        print(f"\n{task_name}")
        print(f"  A4 = {dec.get('A4_pearson', 0):.4f}")
        print(f"  LF0 no-prior late fusion = {dec.get('LF0_pearson', 0):.4f}")
        print(f"  LF1 prior substitution = {dec.get('LF1_pearson', 0):.4f}")
        print(f"  LF2 prior augmentation = {dec.get('LF2_matched_pearson', 0):.4f}")
        print(f"  LF2 - strongest no-prior:")
        print(f"    mean ΔPearson = {dec.get('mean_delta_vs_strongest_no_prior', 0):+.4f}")
        print(f"    positive seeds = {dec.get('positive_seeds', 0)}/3")
        print(f"  Status: {dec.get('status', '')}")

    print(f"\nDecision: {next_step}")
    print("No Modification 3 implemented.")
    print("No post-hoc fusion tuning performed.")

    # Write COMPLETE marker
    (output_dir / "COMPLETE").write_text(
        f"Late fusion pilot complete.\n"
        f"Recommended next step: {next_step}\n"
        f"No Modification 3 implemented.\n"
    )


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/iclr/prior_aware_late_fusion.yaml"
    main(config_path)
