"""Main runner for LF1 final 10x5 experiment — corrected."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from metascfc.benchmark_utils import load_connectomes, prediction_metrics
from metascfc.experiments.msancr_refinement import load_roi_prior, upper_triangle_features
from metascfc.experiments.lf1_final_experiment import (
    run_lf1_experiment, run_lf1_prior_controls,
    compute_statistics, compute_repeated_cv_sensitivity,
    compute_biomarker_stats,
)

BASE = Path("/home/iemiedc2026/Documents/Sanjan/iclr")
OUT = BASE / "outputs/iclr/lf1_final_10x5"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    cfg = yaml.safe_load((BASE / "configs/iclr/lf1_final_10x5.yaml").read_text())

    print("Loading HCP data...")
    fc_mats, sc_mats, y_all, subject_ids, groups = load_connectomes(cfg["data"])
    n_rois = int(cfg.get("n_rois", fc_mats.shape[1]))
    X_fc = upper_triangle_features(fc_mats)
    X_sc = upper_triangle_features(sc_mats)
    print(f"  X_fc: {X_fc.shape}, X_sc: {X_sc.shape}")

    seeds = [int(s) for s in cfg.get("seeds", list(range(10)))]
    n_outer = int(cfg.get("outer_folds", 5))
    n_inner = int(cfg.get("inner_folds", 3))

    all_decisions = {}

    for target_key in ["working_memory", "fluid_intelligence"]:
        target_cfg = cfg["targets"][target_key]
        print(f"\n{'='*60}")
        print(f"Running LF1 final 10x5: {target_cfg['name']}")
        print(f"{'='*60}")

        y_task = np.asarray(np.load(target_cfg["label_path"], allow_pickle=False), dtype=np.float64).reshape(-1)

        prior_cfg = cfg.get("priors", {}).get(target_key, {})
        priors = {}
        for pn in ["matched", "unrelated", "shuffled", "random"]:
            path = prior_cfg.get(pn)
            if path:
                priors[pn] = load_roi_prior(path, n_rois)

        task_out = OUT / target_key
        result = run_lf1_experiment(
            X_fc, X_sc, y_task, priors, seeds,
            n_outer=n_outer, n_inner=n_inner,
            n_rois=n_rois, output_dir=str(task_out),
        )

        print(f"\nRunning prior controls for {target_key}...", flush=True)
        ctrl = run_lf1_prior_controls(
            result["all_split_results"], X_fc, X_sc, y_task, priors,
            n_rois=n_rois, output_dir=str(task_out),
        )

        stats = compute_statistics(result["seed_df"], ctrl["ctrl_summary"], target_key)

        print("Computing repeated-CV sensitivity...", flush=True)
        cv_sens = compute_repeated_cv_sensitivity(result["all_split_results"], y_task)
        stats["cv_sensitivity"] = cv_sens

        all_decisions[target_key] = stats

        print(f"\n--- {target_key} Results ---")
        print(f"  LF0: {stats['lf0_mean']:.4f} +/- {stats['lf0_std']:.4f}")
        print(f"  LF1: {stats['lf1_mean']:.4f} +/- {stats['lf1_std']:.4f}")
        print(f"  A4:  {stats['a4_mean']:.4f} +/- {stats['a4_std']:.4f}")
        print(f"  Delta: mean={stats['delta_mean']:+.4f} median={stats['delta_median']:+.4f}")
        print(f"  Positive: {stats['positive_seeds']}/{stats['n_seeds']}")
        print(f"  Wilcoxon p={stats['p_wilcoxon']:.4f}")
        print(f"  95% CI: [{stats['ci_95'][0]:.4f}, {stats['ci_95'][1]:.4f}]")
        print(f"  Cohen dz: {stats['cohen_dz']:.4f}")
        print(f"  CV sensitivity p={cv_sens['p']:.4f}")
        print(f"  Large-margin gate: {stats['large_margin_consistency_gate']}")
        print(f"  Controls: matched={stats['ctrl_matched']:.4f} beats {stats['ctrl_beats']}/3")

        with open(task_out / "prediction_statistics.json", "w") as f:
            json.dump(stats, f, indent=2, default=str)

    # Two-task Holm correction
    p_wm = all_decisions["working_memory"]["p_wilcoxon"]
    p_fl = all_decisions["fluid_intelligence"]["p_wilcoxon"]
    pvals = sorted([p_wm, p_fl])
    holm_pvals = [pvals[0] * 2, pvals[1] * 1]
    holm_pvals = [min(p, 1.0) for p in holm_pvals]
    holm_pvals[1] = max(holm_pvals[0], holm_pvals[1])

    all_decisions["working_memory"]["prediction_holm_p"] = float(holm_pvals[0]) if p_wm <= p_fl else float(holm_pvals[1])
    all_decisions["fluid_intelligence"]["prediction_holm_p"] = float(holm_pvals[1]) if p_wm <= p_fl else float(holm_pvals[0])

    # Biomarker
    print("\nComputing biomarker statistics...", flush=True)
    for target_key in ["working_memory", "fluid_intelligence"]:
        target_cfg = cfg["targets"][target_key]
        y_task = np.asarray(np.load(target_cfg["label_path"], allow_pickle=False), dtype=np.float64).reshape(-1)
        prior_cfg = cfg.get("priors", {}).get(target_key, {})
        priors = {}
        for pn in ["matched", "unrelated", "shuffled", "random"]:
            path = prior_cfg.get(pn)
            if path:
                priors[pn] = load_roi_prior(path, n_rois)
        task_out = OUT / target_key
        biom = compute_biomarker_stats(priors, y_task, n_rois, seeds, n_outer, n_inner, str(task_out))
        all_decisions[target_key]["biomarker"] = biom

    # Professor requirement
    prof = {}
    for tk in ["working_memory", "fluid_intelligence"]:
        s = all_decisions[tk]
        pred_improved = s["delta_mean"] > 0 and s["positive_seeds"] > s["n_seeds"] / 2
        bio_improved = s.get("biomarker", {}).get("matched_beats_negative_controls", False)
        prof[tk] = {
            "prediction_improved": pred_improved,
            "biomarker_improved": bio_improved,
            "prediction_raw_p": s["p_wilcoxon"],
            "prediction_holm_p": s["prediction_holm_p"],
            "prediction_cv_corrected_p": s["cv_sensitivity"]["p"],
        }

    both_pred = all(p["prediction_improved"] for p in prof.values())
    both_bio = all(p["biomarker_improved"] for p in prof.values())
    prof["overall"] = {
        "both_tasks_prediction_improved": both_pred,
        "both_tasks_biomarker_improved": both_bio,
        "professor_requirement_satisfied_both_tasks": both_pred and both_bio,
    }

    with open(OUT / "professor_requirement_summary.json", "w") as f:
        json.dump(prof, f, indent=2, default=str)

    with open(OUT / "final_decision.json", "w") as f:
        json.dump(all_decisions, f, indent=2, default=str)

    with open(OUT / "COMPLETE", "w") as f:
        f.write("done")
    with open(OUT / "FINAL_COMPLETE", "w") as f:
        f.write("done")

    # Print completion report
    print("\n" + "="*60)
    print("MODIFICATION 2 FINAL LF1 10x5 COMPLETE")
    print("="*60)
    for tk, label in [("working_memory", "WORKING MEMORY"), ("fluid_intelligence", "FLUID INTELLIGENCE")]:
        s = all_decisions[tk]
        print(f"\n{label}")
        print(f"  LF0 no-prior = {s['lf0_mean']:.4f}")
        print(f"  LF1 matched = {s['lf1_mean']:.4f}")
        print(f"  Delta Pearson mean = {s['delta_mean']:+.4f}")
        print(f"  Delta Pearson median = {s['delta_median']:+.4f}")
        print(f"  Positive seeds = {s['positive_seeds']}/{s['n_seeds']}")
        print(f"  Wilcoxon p = {s['p_wilcoxon']:.4f}")
        print(f"  Two-task Holm p = {s['prediction_holm_p']:.4f}")
        print(f"  Corrected repeated-CV p = {s['cv_sensitivity']['p']:.4f}")
        print(f"  95% CI = [{s['ci_95'][0]:.4f}, {s['ci_95'][1]:.4f}]")
        print(f"  Cohen dz = {s['cohen_dz']:.4f}")
        print(f"  Matched controls: unrelated={s['ctrl_unrelated']:.4f} shuffled={s['ctrl_shuffled']:.4f} random={s['ctrl_random']:.4f}")
        b = s.get("biomarker", {})
        print(f"  Biomarker:")
        print(f"    no-prior alignment = {b.get('no_prior_alignment', 'N/A'):.4f}")
        print(f"    matched alignment = {b.get('matched_alignment', 'N/A'):.4f}")
        print(f"    rank stability = {b.get('rank_stability', 'N/A'):.4f}")
        print(f"    top-10 Jaccard = {b.get('top10_jaccard', 'N/A'):.4f}")

    print(f"\nPROFESSOR REQUIREMENT")
    print(f"  Prediction improved for WM: {'YES' if prof['working_memory']['prediction_improved'] else 'NO'}")
    print(f"  Prediction improved for Fluid: {'YES' if prof['fluid_intelligence']['prediction_improved'] else 'NO'}")
    print(f"  Biomarker improved for WM: {'YES' if prof['working_memory']['biomarker_improved'] else 'NO'}")
    print(f"  Biomarker improved for Fluid: {'YES' if prof['fluid_intelligence']['biomarker_improved'] else 'NO'}")
    print(f"  Overall requirement satisfied for both tasks: {'YES' if prof['overall']['professor_requirement_satisfied_both_tasks'] else 'NO'}")
    print(f"\nNo Modification 3 implemented.")
    print(f"No post-hoc tuning performed.")


if __name__ == "__main__":
    main()
