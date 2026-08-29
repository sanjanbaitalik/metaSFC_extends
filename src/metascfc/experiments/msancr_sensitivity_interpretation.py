"""Statistical sensitivity interpretation + multiplicity family naming (Part A4, A4b)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from metascfc.benchmark_utils import save_json, holm_adjust
from metascfc.experiments.msancr_final_inference import (
    MODEL_A3, MODEL_A2, MODEL_A4,
    COMPARISONS, PREDICTION_METRICS,
    paired_seed_values, paired_statistics,
    build_seed_level_table,
)


# ---------------------------------------------------------------------------
# Part A4b -- Explicit multiplicity-family naming
# ---------------------------------------------------------------------------

# Prior-specificity-only family (3 comparisons)
PRIOR_SPECIFICITY_COMPARISONS = [
    "A3_matched_vs_A3_unrelated_fixed",
    "A3_matched_vs_A3_shuffled_fixed",
    "A3_matched_vs_A3_random_fixed",
]

# Broad secondary-method family (4 comparisons, includes A2)
BROAD_SECONDARY_COMPARISONS = [
    "A3_matched_vs_A2",
    "A3_matched_vs_A3_unrelated_fixed",
    "A3_matched_vs_A3_shuffled_fixed",
    "A3_matched_vs_A3_random_fixed",
]


def build_explicit_multiplicity_families(
    seed_df: pd.DataFrame,
) -> dict[str, Any]:
    """Compute Holm for two explicit families + preserve all-five conservative."""
    all_rows = []
    for name, left_key, right_key in COMPARISONS:
        for metric in PREDICTION_METRICS:
            paired = paired_seed_values(seed_df, left_key, right_key, metric)
            stats = paired_statistics(
                paired.left.to_numpy(), paired.right.to_numpy(), metric
            )
            all_rows.append({"comparison": name, "metric": metric, **stats})

    all_df = pd.DataFrame(all_rows)

    # Family 1: Prior-specificity-only (3 comparisons)
    prior_spec_rows = all_df[all_df.comparison.isin(PRIOR_SPECIFICITY_COMPARISONS)]
    prior_spec_holm = {}
    for metric in PREDICTION_METRICS:
        mask = prior_spec_rows.metric == metric
        pvals = prior_spec_rows.loc[mask, "wilcoxon_p"].to_numpy(float)
        adjusted = holm_adjust(pvals)
        comps = prior_spec_rows.loc[mask, "comparison"].tolist()
        for comp, p_adj in zip(comps, adjusted):
            prior_spec_holm[f"{comp}|{metric}"] = float(p_adj)

    # Family 2: Broad secondary (4 comparisons)
    broad_rows = all_df[all_df.comparison.isin(BROAD_SECONDARY_COMPARISONS)]
    broad_holm = {}
    for metric in PREDICTION_METRICS:
        mask = broad_rows.metric == metric
        pvals = broad_rows.loc[mask, "wilcoxon_p"].to_numpy(float)
        adjusted = holm_adjust(pvals)
        comps = broad_rows.loc[mask, "comparison"].tolist()
        for comp, p_adj in zip(comps, adjusted):
            broad_holm[f"{comp}|{metric}"] = float(p_adj)

    # Family 3: Conservative all-five (preserved from original)
    all_comps_df = all_df.copy()
    allfive_holm = {}
    for metric in PREDICTION_METRICS:
        mask = all_comps_df.metric == metric
        pvals = all_comps_df.loc[mask, "wilcoxon_p"].to_numpy(float)
        adjusted = holm_adjust(pvals)
        comps = all_comps_df.loc[mask, "comparison"].tolist()
        for comp, p_adj in zip(comps, adjusted):
            allfive_holm[f"{comp}|{metric}"] = float(p_adj)

    return {
        "prior_specificity_3_holm": prior_spec_holm,
        "broad_secondary_4_holm": broad_holm,
        "conservative_all_five_holm": allfive_holm,
    }


# ---------------------------------------------------------------------------
# Part A4 -- Three-layer statistical sensitivity interpretation
# ---------------------------------------------------------------------------

def build_sensitivity_interpretation(
    final_output_dir: str | Path,
    cv_sensitivity_path: str | Path,
) -> dict[str, Any]:
    """Build three-layer interpretation: primary + conservative + CV-corrected."""
    final_output_dir = Path(final_output_dir)

    # Load original results
    orig_decision = json.loads(
        (final_output_dir / "final_hypothesis_decision.json").read_text()
    )
    orig_summary = json.loads(
        (final_output_dir / "final_statistical_summary.json").read_text()
    )

    # Load CV sensitivity
    cv_sens = json.loads(Path(cv_sensitivity_path).read_text())
    a3_a4_cv = cv_sens["comparisons"]["A3_matched_vs_A4"]["metrics"]["pearson"]

    # Load multiplicity families
    base_df = pd.read_csv(final_output_dir / "split_metrics.csv")
    swap_df = pd.read_csv(final_output_dir / "prior_swap_split_metrics.csv")
    combined = pd.concat([base_df, swap_df], ignore_index=True)
    seed_df = build_seed_level_table(combined)
    if not {"wm_alignment", "rank_stability", "top10_jaccard"}.issubset(seed_df.columns):
        bio = pd.read_csv(final_output_dir / "biomarker_metrics.csv")
        seed_df = seed_df.merge(bio, on=["model_id", "prior_type", "seed"], how="left")

    multiplicity = build_explicit_multiplicity_families(seed_df)

    # Extract key values
    primary_p = orig_decision["prediction_hypothesis"]["pearson_vs_A4"]["wilcoxon_p"]
    conservative_p = orig_summary["prediction_hypothesis"]["pearson_vs_A4"]["wilcoxon_p_holm_metric"]
    cv_corrected_p = a3_a4_cv["corrected_p_two_sided"]
    mean_delta = orig_decision["prediction_hypothesis"]["pearson_vs_A4"]["mean_diff"]

    # Prior-specificity for Pearson (A3 vs A3_unrelated)
    prior_unrelated_p = multiplicity["prior_specificity_3_holm"].get(
        "A3_matched_vs_A3_unrelated_fixed|pearson", float("nan")
    )
    prior_unrelated_holm = multiplicity["broad_secondary_4_holm"].get(
        "A3_matched_vs_A3_unrelated_fixed|pearson", float("nan")
    )

    interpretation = {
        "layer_1_designated_primary": {
            "test": "seed-aggregated paired Wilcoxon signed-rank",
            "contrast": "A3 matched vs A4 modality-specific Ridge",
            "metric": "Pearson",
            "p_value": float(primary_p),
            "significant_at_nominal_005": bool(primary_p < 0.05),
            "mean_delta_r": float(mean_delta),
            "note": "Predesignated primary analysis. No multiplicity correction needed within a single contrast.",
        },
        "layer_2_conservative_multiplicity": {
            "test": "Holm correction across five comparisons (primary + secondary + control)",
            "p_holm_all_five": float(conservative_p),
            "significant": bool(conservative_p < 0.05),
            "note": (
                "The five-comparison Holm sensitivity analysis groups the primary "
                "A3-vs-A4 contrast with four secondary/control comparisons.  This "
                "is a conservative upper bound on familywise error."
            ),
        },
        "layer_3_cv_dependence_sensitivity": {
            "test": "Bouckaert-Frank / Nadeau-Bengio corrected repeated-k-fold t-test",
            "corrected_p_two_sided": float(cv_corrected_p),
            "corrected_t": float(a3_a4_cv["corrected_t"]),
            "df": int(a3_a4_cv["df"]),
            "mean_difference": float(a3_a4_cv["mean_difference"]),
            "sd_fold_difference": float(a3_a4_cv["sd_fold_difference"]),
            "mean_test_train_ratio": float(a3_a4_cv["mean_test_train_ratio"]),
            "significant": bool(cv_corrected_p < 0.05),
            "note": (
                "Because repeated cross-validation partitions reuse the same "
                "participants, this corrected test inflates uncertainty for "
                "train/test overlap.  Conclusions from this sensitivity "
                "analysis are reported separately and do not trigger model retuning."
            ),
        },
        "multiplicity_families": {
            "prior_specificity_3": {
                "comparisons": ["A3 vs unrelated", "A3 vs shuffled", "A3 vs random"],
                "holm_adjusted_p_values": {
                "unrelated_pearson": float(prior_unrelated_p),
                "shuffled_pearson": multiplicity["prior_specificity_3_holm"].get(
                    "A3_matched_vs_A3_shuffled_fixed|pearson", float("nan")
                ),
                "random_pearson": multiplicity["prior_specificity_3_holm"].get(
                    "A3_matched_vs_A3_random_fixed|pearson", float("nan")
                ),
                },
                "note": "Holm within the 3 prior-identity controls only.",
            },
            "broad_secondary_4": {
                "comparisons": [
                    "A3 vs A2", "A3 vs unrelated", "A3 vs shuffled", "A3 vs random"
                ],
                "holm_adjusted_p_values": {
                "A2_pearson": multiplicity["broad_secondary_4_holm"].get(
                    "A3_matched_vs_A2|pearson", float("nan")
                ),
                "unrelated_pearson": float(prior_unrelated_holm),
                "shuffled_pearson": multiplicity["broad_secondary_4_holm"].get(
                    "A3_matched_vs_A3_shuffled_fixed|pearson", float("nan")
                ),
                "random_pearson": multiplicity["broad_secondary_4_holm"].get(
                    "A3_matched_vs_A3_random_fixed|pearson", float("nan")
                ),
                },
                "note": "Holm across the broader 4-comparison secondary/method family.",
            },
            "conservative_all_five": {
                "comparisons": [
                    "A3 vs A4", "A3 vs A2", "A3 vs unrelated", "A3 vs shuffled", "A3 vs random"
                ],
                "p_holm_all_five": float(conservative_p),
                "note": "Holm across all five comparisons. Preserved as-is from original analysis.",
            },
        },
        "recommended_interpretation": (
            "The designated primary seed-aggregated paired analysis showed a positive "
            f"A3-vs-A4 effect (mean Δr = {mean_delta:.4f}, p = {primary_p:.5f}).  "
            "Because repeated cross-validation partitions reuse the same "
            "participants, we additionally report a corrected repeated-k-fold sensitivity "
            f"analysis that inflates uncertainty for train/test overlap (corrected p = {cv_corrected_p:.5f}).  "
            "Conclusions from this sensitivity analysis are reported separately and do not "
            "trigger model retuning."
        ),
        "prohibited_claims": [
            "The corrected repeated-CV test is the one true test",
            "The primary Wilcoxon is the only valid test",
            "Both tests must be significant for the result to be valid",
            "The conservative Holm analysis was never planned",
        ],
    }

    return interpretation
