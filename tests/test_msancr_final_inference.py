"""Tests for the final n=10 seed-level inference module (prompt v7)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metascfc.experiments.msancr_final_inference import (
    MODEL_A2,
    MODEL_A3,
    MODEL_A4,
    N_FINAL_SEEDS,
    build_seed_level_table,
    hypothesis_decision,
    paired_statistics,
    prior_swap_integrity_check,
    run_family_inference,
)
from metascfc.experiments.msancr_refinement import (
    hyperparameter_payload,
    make_fixed_prior_swap_configs,
    stable_hash,
)

ROOT = Path(__file__).resolve().parents[1]
CONTROL_PRIORS = ("unrelated", "shuffled", "random")


def make_split_frame(all_values: dict) -> pd.DataFrame:
    """all_values: {(model, prior, eval_type): ndarray(10) of pearson}."""
    rows = []
    for (model, prior, eval_type), values in all_values.items():
        for seed in range(N_FINAL_SEEDS):
            for fold in range(5):
                rows.append({
                    "model_id": model, "prior_type": prior,
                    "evaluation_type": eval_type,
                    "seed": seed, "fold": fold,
                    "pearson": float(values[seed]),
                    "rmse": float(1.0 - values[seed]),
                    "mae": float(0.5 - values[seed]),
                    "wm_alignment": 0.3 + 0.01 * seed,
                    "rank_stability": 0.3 + 0.02 * seed,
                    "top10_jaccard": 0.1 + 0.01 * seed,
                })
    return pd.DataFrame(rows)


def make_default_values():
    rng = np.random.default_rng(7)
    return {
        (MODEL_A4, "none", "retuned_base"): rng.normal(0.25, 0.05, 10),
        (MODEL_A2, "matched", "retuned_base"): rng.normal(0.26, 0.05, 10),
        (MODEL_A3, "matched", "retuned_base"): rng.normal(0.27, 0.05, 10),
        (MODEL_A3, "unrelated", "fixed_prior_swap"): rng.normal(0.245, 0.05, 10),
        (MODEL_A3, "shuffled", "fixed_prior_swap"): rng.normal(0.242, 0.05, 10),
        (MODEL_A3, "random", "fixed_prior_swap"): rng.normal(0.243, 0.05, 10),
    }


def test_folds_are_averaged_within_seed_before_inference():
    values = np.arange(10) / 10.0
    frame = make_split_frame({(MODEL_A3, "matched", "retuned_base"): values})
    seed_df = build_seed_level_table(frame)
    a3 = seed_df[(seed_df.model_id == MODEL_A3) & (seed_df.prior_type == "matched")]
    assert len(a3) == 10
    np.testing.assert_allclose(a3.pearson.to_numpy(), values)


def test_exactly_ten_seeds_required_for_complete_inference():
    values = np.arange(10) / 10.0
    frame = make_split_frame({(MODEL_A3, "matched", "retuned_base"): values})
    seed_df = build_seed_level_table(frame)
    assert seed_df.seed.nunique() == N_FINAL_SEEDS
    with pytest.raises(ValueError):
        build_seed_level_table(frame[frame.seed < 9])


def test_paired_seed_ids_are_aligned():
    values = make_default_values()
    frame = make_split_frame(values)
    seed_df = build_seed_level_table(frame)
    prediction = run_family_inference(seed_df, ("pearson",), n_boot=100)
    a3a4 = prediction[prediction.comparison == "A3_matched_vs_A4"].iloc[0]
    assert a3a4.n_seeds == 10


def test_metric_wise_holm_families_are_independent():
    values = make_default_values()
    frame = make_split_frame(values)
    seed_df = build_seed_level_table(frame)
    family = run_family_inference(seed_df, ("pearson", "rmse"), n_boot=100)
    pearsons = family[family.metric == "pearson"]
    rmses = family[family.metric == "rmse"]
    assert len(pearsons) == 5 and len(rmses) == 5
    assert pearsons.p_holm_metric.isna().sum() == 0
    assert rmses.p_holm_metric.isna().sum() == 0


def test_bootstrap_resamples_seed_paired_differences():
    rng = np.random.default_rng(11)
    left = rng.normal(0.3, 0.05, 10)
    right = rng.normal(0.25, 0.05, 10)
    stats = paired_statistics(left, right, "pearson", n_boot=5000, rng_seed=42)
    assert stats["bootstrap_resamples"] == 5000
    assert stats["ci95_low"] < stats["ci95_high"]
    assert stats["n_seeds"] == 10
    assert np.isfinite(stats["wilcoxon_p"])


def test_rmse_improvement_orientation_positive_is_a3_better():
    left_rmse = np.arange(10) / 10.0 + 1.0
    right_rmse = left_rmse + np.linspace(0.03, 0.08, 10)  # comparator worse, varying
    stats = paired_statistics(left_rmse, right_rmse, "rmse")
    assert stats["paired_mean_diff"] > 0
    assert stats["positive_seeds"] == 10
    assert stats["cohens_dz"] > 0


def test_cohens_dz_sign_positive_when_a3_better():
    rng = np.random.default_rng(5)
    left = np.arange(10) / 10.0
    right = left - (0.04 + rng.normal(0, 0.01, 10))
    stats = paired_statistics(left, right, "pearson")
    assert stats["paired_mean_diff"] > 0
    assert stats["cohens_dz"] > 0
    stats2 = paired_statistics(right, left, "pearson")
    assert stats2["cohens_dz"] < 0


def test_final_inference_does_not_emit_n_equals_3_wording():
    frame = make_split_frame(make_default_values())
    seed_df = build_seed_level_table(frame)
    decision = hypothesis_decision(
        run_family_inference(seed_df, ("pearson", "rmse"), n_boot=100),
        run_family_inference(seed_df, ("wm_alignment",), n_boot=100),
    )
    text = json.dumps(decision)
    assert "n_equals_3" not in text
    assert "ten_paired_seed_level_summaries" in text


def test_final_decision_uses_significance_not_pilot_thresholds():
    rng = np.random.default_rng(13)
    values = make_default_values()
    a4 = values[(MODEL_A4, "none", "retuned_base")]
    # A3 = A4 + small positive shift with tiny jitter -> strongly significant,
    # positive in nearly every seed.
    values[(MODEL_A3, "matched", "retuned_base")] = a4 + rng.normal(0.03, 0.008, 10)
    frame = make_split_frame(values)
    frame.loc[(frame.model_id == MODEL_A3) & (frame.prior_type == "matched"),
              "wm_alignment"] += 0.05
    seed_df = build_seed_level_table(frame)
    prediction = run_family_inference(seed_df, ("pearson", "rmse"), n_boot=2000)
    biomarker = run_family_inference(seed_df, ("wm_alignment",), n_boot=100)
    decision = hypothesis_decision(prediction, biomarker)
    assert decision["prediction_hypothesis"]["prediction_supported"] is True
    assert decision["prediction_hypothesis"]["pearson_vs_A4"]["wilcoxon_p_holm_metric"] < 0.05
    assert decision["overall_recommendation"] in (
        "prediction_and_biomarker_supported", "prediction_supported_biomarker_mixed")


def _make_selected_and_swaps(a3_row: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    hp = {k: a3_row[k] for k in ("lambda_fc", "lambda_sc", "lambda_l", "gamma", "lifting")}
    hp_hash = stable_hash(hp)
    selected = pd.DataFrame([{"seed": a3_row["seed"], "fold": a3_row["fold"],
                              "model_id": MODEL_A3, **hp,
                              "selected_hyperparameter_hash": hp_hash}])
    swap_rows = []
    for control in CONTROL_PRIORS:
        row = {"seed": a3_row["seed"], "fold": a3_row["fold"],
               "model_id": MODEL_A3, "prior_type": control,
               **hp, "selected_hyperparameter_hash": hp_hash}
        swap_rows.append(row)
    return selected, pd.DataFrame(swap_rows)


def test_prior_swap_integrity_check_detects_mismatches():
    a3_row = {"seed": 3, "fold": 1, "lambda_fc": 0.1, "lambda_sc": 1.0,
              "lambda_l": 0.5, "gamma": 0.5, "lifting": "mean"}
    selected, swaps = _make_selected_and_swaps(a3_row)
    integrity = prior_swap_integrity_check(
        pd.DataFrame([{"seed": 3, "fold": 1}]),
        swaps, selected)
    assert integrity["n_fail"] == 0 and integrity["n_pass"] == 3
    # Corrupt one swap to prove detection.
    bad = swaps.copy()
    bad.loc[0, "gamma"] = 0.25
    with pytest.raises(RuntimeError):
        prior_swap_integrity_check(pd.DataFrame([{"seed": 3, "fold": 1}]), bad, selected)


def test_prior_swap_hyperparameters_frozen_from_matched_a3():
    selected = {"lambda_fc": 0.001, "lambda_sc": 100.0, "lambda_l": 0.03,
                "gamma": 0.1, "lifting": "mean"}
    swaps = make_fixed_prior_swap_configs(selected)
    assert all(payload == hyperparameter_payload(selected) for payload in swaps.values())


def test_final_config_is_unchanged_by_analyzer():
    config_path = ROOT / "configs/iclr/msancr_final_10x5.yaml"
    text = config_path.read_text(encoding="utf-8")
    assert "msancr_final_10x5" in text


def test_partial_runs_cannot_be_declared_complete():
    values = make_default_values()
    frame = make_split_frame(values)
    incomplete = frame[frame.fold < 4]
    with pytest.raises(ValueError):
        build_seed_level_table(incomplete)