#!/usr/bin/env python3
"""111 -- WM statistical sensitivity analysis. No model reruns."""
import argparse
import json
from pathlib import Path

from metascfc.benchmark_utils import save_json
from metascfc.experiments.msancr_cv_sensitivity import run_cv_sensitivity
from metascfc.experiments.msancr_sensitivity_interpretation import (
    build_sensitivity_interpretation,
    build_explicit_multiplicity_families,
)
from metascfc.experiments.msancr_final_inference import (
    build_seed_level_table,
)
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description="WM statistical sensitivity analysis")
    ap.add_argument("--final-dir", default="outputs/iclr/msancr_final_10x5")
    ap.add_argument("--report-dir", default="outputs/iclr/msancr_final_10x5/postfinal_reporting")
    args = ap.parse_args()

    final_dir = Path(args.final_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Part A1-A3: Corrected repeated-k-fold sensitivity...")
    cv_result = run_cv_sensitivity(final_dir, report_dir)
    print(f"  A3-vs-A4 corrected p = {cv_result['a3_a4_pearson_corrected_p']:.6f}")

    print("Part A4: Three-layer sensitivity interpretation...")
    interpretation = build_sensitivity_interpretation(final_dir, report_dir / "cv_dependence_sensitivity.json")
    save_json(interpretation, report_dir / "statistical_sensitivity_interpretation.json")

    print("Part A4b: Explicit multiplicity families...")
    base_df = pd.read_csv(final_dir / "split_metrics.csv")
    swap_df = pd.read_csv(final_dir / "prior_swap_split_metrics.csv")
    combined = pd.concat([base_df, swap_df], ignore_index=True)
    seed_df = build_seed_level_table(combined)
    if not {"wm_alignment", "rank_stability", "top10_jaccard"}.issubset(seed_df.columns):
        bio = pd.read_csv(final_dir / "biomarker_metrics.csv")
        seed_df = seed_df.merge(bio, on=["model_id", "prior_type", "seed"], how="left")

    multiplicity = build_explicit_multiplicity_families(seed_df)
    save_json(multiplicity, report_dir / "explicit_multiplicity_families.json")

    # Verify original outputs untouched
    for f in [
        "final_prediction_statistics.csv", "final_biomarker_statistics.csv",
        "final_hypothesis_decision.json", "final_statistical_summary.json",
        "FINAL_COMPLETE",
    ]:
        assert (final_dir / f).exists(), f"Original missing: {f}"

    print("\nWM STATISTICAL SENSITIVITY COMPLETE")
    print(f"  Primary Wilcoxon p = {interpretation['layer_1_designated_primary']['p_value']:.5f}")
    print(f"  Conservative Holm p = {interpretation['layer_2_conservative_multiplicity']['p_holm_all_five']:.5f}")
    print(f"  CV-corrected p = {interpretation['layer_3_cv_dependence_sensitivity']['corrected_p_two_sided']:.5f}")
    print(f"  All outputs in: {report_dir}")


if __name__ == "__main__":
    main()
