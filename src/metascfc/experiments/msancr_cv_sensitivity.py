"""Corrected repeated-k-fold sensitivity analysis for WM 10x5 results.

Implements the Bouckaert-Frank / Nadeau-Bengio corrected t-test using
the 50 paired outer-fold differences, accounting for overlap between
repeated cross-validation folds.

Zero model reruns.  Only reads frozen prediction artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist

from metascfc.benchmark_utils import save_json
from metascfc.experiments.msancr_final_inference import (
    MODEL_A3,
    MODEL_A2,
    MODEL_A4,
    N_FINAL_SEEDS,
    N_FOLDS,
    COMPARISONS,
    PREDICTION_METRICS,
    HIGHER_BETTER,
)

EXPECTED_FOLDS = N_FINAL_SEEDS * N_FOLDS  # 50


def corrected_repeated_cv_test(
    fold_differences: np.ndarray,
    n_test: np.ndarray,
    n_train: np.ndarray,
    r: int = N_FINAL_SEEDS,
    k: int = N_FOLDS,
) -> dict[str, Any]:
    """Nadeau-Bengio corrected repeated-k-fold t-test.

    Parameters
    ----------
    fold_differences : (50,) array of paired metric differences per outer fold.
        Positive = A3 is better (already oriented by caller).
    n_test : (50,) array of test-set sizes per fold.
    n_train : (50,) array of train-set sizes per fold.
    r : number of repeats (seeds).
    k : number of folds per repeat.

    Returns
    -------
    dict with corrected t, p, and diagnostic values.
    """
    n_resamples = len(fold_differences)
    assert n_resamples == r * k, f"Expected {r*k} fold differences; got {n_resamples}"

    d_bar = float(np.mean(fold_differences))
    s_d_sq = float(np.var(fold_differences, ddof=1))

    # Mean test/train ratio across folds
    ratios = n_test.astype(float) / n_train.astype(float)
    mean_ratio = float(np.mean(ratios))

    # Corrected standard error
    corrected_se = np.sqrt(
        (1.0 / (k * r) + mean_ratio) * s_d_sq
    )

    # t-statistic
    df = k * r - 1  # 49
    if corrected_se < 1e-15:
        t_stat = 0.0 if np.abs(d_bar) < 1e-15 else float(np.sign(d_bar) * np.inf)
    else:
        t_stat = d_bar / corrected_se

    # Two-sided p-value
    p_value = 2.0 * (1.0 - t_dist.cdf(np.abs(t_stat), df=df))

    return {
        "n_resamples": int(n_resamples),
        "k_folds": int(k),
        "r_repeats": int(r),
        "mean_difference": d_bar,
        "sd_fold_difference": float(np.sqrt(s_d_sq)),
        "mean_test_train_ratio": mean_ratio,
        "corrected_se": float(corrected_se),
        "corrected_t": float(t_stat),
        "df": int(df),
        "corrected_p_two_sided": float(p_value),
    }


def build_cv_dependence_sensitivity(
    final_output_dir: str | Path,
) -> dict[str, Any]:
    """Build corrected repeated-CV sensitivity for all comparisons x metrics."""
    final_output_dir = Path(final_output_dir)

    base_df = pd.read_csv(final_output_dir / "split_metrics.csv")
    swap_df = pd.read_csv(final_output_dir / "prior_swap_split_metrics.csv")
    combined = pd.concat([base_df, swap_df], ignore_index=True)

    # Get n_train/n_test from base (retuned_base) splits
    base_only = combined[combined.evaluation_type == "retuned_base"]

    results: dict[str, Any] = {"comparisons": {}}

    for comp_name, left_key, right_key in COMPARISONS:
        left_model, left_prior = left_key
        right_model, right_prior = right_key

        comp_results: dict[str, Any] = {"metrics": {}}

        for metric in PREDICTION_METRICS:
            higher_better = metric in HIGHER_BETTER

            # Collect 50 paired fold differences
            fold_diffs = []
            n_tests = []
            n_trains = []

            for seed in range(N_FINAL_SEEDS):
                for fold in range(N_FOLDS):
                    # Left
                    left_rows = combined[
                        (combined.seed == seed)
                        & (combined.fold == fold)
                        & (combined.model_id == left_model)
                        & (combined.prior_type == left_prior)
                    ]
                    # Right
                    right_rows = combined[
                        (combined.seed == seed)
                        & (combined.fold == fold)
                        & (combined.model_id == right_model)
                        & (combined.prior_type == right_prior)
                    ]

                    if left_rows.empty or right_rows.empty:
                        raise ValueError(
                            f"Missing data for {comp_name} seed={seed} fold={fold}"
                        )

                    left_val = float(left_rows[metric].iloc[0])
                    right_val = float(right_rows[metric].iloc[0])

                    # Positive = left is better
                    if higher_better:
                        diff = left_val - right_val
                    else:
                        diff = right_val - left_val

                    fold_diffs.append(diff)

                    # Get n_train/n_test from base rows
                    base_row = base_only[
                        (base_only.seed == seed)
                        & (base_only.fold == fold)
                        & (base_only.model_id == left_model)
                    ]
                    if not base_row.empty:
                        n_tests.append(int(base_row.n_test.iloc[0]))
                        n_trains.append(int(base_row.n_train.iloc[0]))
                    else:
                        # Fallback: get from right model
                        base_row = base_only[
                            (base_only.seed == seed)
                            & (base_only.fold == fold)
                            & (base_only.model_id == right_model)
                        ]
                        n_tests.append(int(base_row.n_test.iloc[0]))
                        n_trains.append(int(base_row.n_train.iloc[0]))

            fold_diffs = np.array(fold_diffs)
            n_tests = np.array(n_tests)
            n_trains = np.array(n_trains)

            test_result = corrected_repeated_cv_test(
                fold_diffs, n_tests, n_trains
            )

            comp_results["metrics"][metric] = {
                **test_result,
                "direction_positive_is_A3_better": True,
            }

        results["comparisons"][comp_name] = comp_results

    return results


def run_cv_sensitivity(
    final_output_dir: str | Path,
    report_output_dir: str | Path,
) -> dict[str, Any]:
    """Master function: compute sensitivity + write outputs."""
    report_output_dir = Path(report_output_dir)
    report_output_dir.mkdir(parents=True, exist_ok=True)

    sensitivity = build_cv_dependence_sensitivity(final_output_dir)

    # Flatten to CSV
    rows = []
    for comp_name, comp_data in sensitivity["comparisons"].items():
        for metric_name, metric_data in comp_data["metrics"].items():
            rows.append({
                "comparison": comp_name,
                "metric": metric_name,
                **metric_data,
            })
    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(report_output_dir / "cv_dependence_sensitivity.csv", index=False)
    save_json(sensitivity, report_output_dir / "cv_dependence_sensitivity.json")

    # Extract A3 vs A4 Pearson result for the interpretation
    a3_a4_pearson = sensitivity["comparisons"]["A3_matched_vs_A4"]["metrics"]["pearson"]

    return {
        "sensitivity": sensitivity,
        "a3_a4_pearson_corrected_p": a3_a4_pearson["corrected_p_two_sided"],
        "a3_a4_pearson_corrected_t": a3_a4_pearson["corrected_t"],
        "a3_a4_pearson_mean_diff": a3_a4_pearson["mean_difference"],
        "csv_path": str(report_output_dir / "cv_dependence_sensitivity.csv"),
    }
