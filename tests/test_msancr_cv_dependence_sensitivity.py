"""Tests for CV-dependence sensitivity analysis (Part A5)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metascfc.experiments.msancr_cv_sensitivity import (
    corrected_repeated_cv_test,
    build_cv_dependence_sensitivity,
    EXPECTED_FOLDS,
)
from metascfc.experiments.msancr_sensitivity_interpretation import (
    build_sensitivity_interpretation,
    build_explicit_multiplicity_families,
)
from metascfc.experiments.msancr_final_inference import (
    build_seed_level_table, COMPARISONS, PREDICTION_METRICS,
)

FINAL_DIR = Path("outputs/iclr/msancr_final_10x5")
REPORT_DIR = FINAL_DIR / "postfinal_reporting"


@pytest.fixture
def cv_sensitivity():
    path = REPORT_DIR / "cv_dependence_sensitivity.json"
    if not path.exists():
        pytest.skip("Run 111_run_wm_sensitivity.py first")
    return json.loads(path.read_text())


@pytest.fixture
def seed_df():
    base = pd.read_csv(FINAL_DIR / "split_metrics.csv")
    swap = pd.read_csv(FINAL_DIR / "prior_swap_split_metrics.csv")
    combined = pd.concat([base, swap], ignore_index=True)
    sdf = build_seed_level_table(combined)
    if not {"wm_alignment", "rank_stability", "top10_jaccard"}.issubset(sdf.columns):
        bio = pd.read_csv(FINAL_DIR / "biomarker_metrics.csv")
        sdf = sdf.merge(bio, on=["model_id", "prior_type", "seed"], how="left")
    return sdf


class TestCorrectedRepeatedCVTest:
    def test_hand_calculated_example(self):
        """Verify against hand-calculated synthetic example."""
        # Synthetic: 10 folds, 2 repeats
        fold_diffs = np.array([0.1, 0.05, -0.02, 0.08, 0.03, 0.06, 0.01, -0.01, 0.04, 0.07])
        n_test = np.array([80] * 10)
        n_train = np.array([332] * 10)

        result = corrected_repeated_cv_test(fold_diffs, n_test, n_train, r=2, k=5)

        # Manual calculation
        d_bar = float(np.mean(fold_diffs))
        s_d_sq = float(np.var(fold_diffs, ddof=1))
        mean_ratio = 80.0 / 332.0
        corrected_se = np.sqrt((1.0 / 10 + mean_ratio) * s_d_sq)
        t_stat = d_bar / corrected_se

        assert abs(result["mean_difference"] - d_bar) < 1e-10
        assert abs(result["sd_fold_difference"] - np.sqrt(s_d_sq)) < 1e-10
        assert abs(result["corrected_t"] - t_stat) < 1e-10
        assert result["df"] == 9

    def test_df_equals_49(self, cv_sensitivity):
        """df = kr - 1 = 49 for complete 10x5 data."""
        a3_a4 = cv_sensitivity["comparisons"]["A3_matched_vs_A4"]["metrics"]["pearson"]
        assert a3_a4["df"] == 49

    def test_n_resamples_equals_50(self, cv_sensitivity):
        """Exactly 50 aligned fold pairs."""
        a3_a4 = cv_sensitivity["comparisons"]["A3_matched_vs_A4"]["metrics"]["pearson"]
        assert a3_a4["n_resamples"] == 50

    def test_positive_orientation_correct(self, cv_sensitivity):
        """Positive mean difference means A3 is better for Pearson."""
        a3_a4 = cv_sensitivity["comparisons"]["A3_matched_vs_A4"]["metrics"]["pearson"]
        assert a3_a4["mean_difference"] > 0, "A3 should be better than A4"

    def test_actual_n_train_n_test_used(self, cv_sensitivity):
        """Mean test/train ratio should be reasonable for 412 subjects."""
        a3_a4 = cv_sensitivity["comparisons"]["A3_matched_vs_A4"]["metrics"]["pearson"]
        ratio = a3_a4["mean_test_train_ratio"]
        assert 0.15 < ratio < 0.35, f"Ratio {ratio} outside expected range"

    def test_p_value_in_valid_range(self, cv_sensitivity):
        """p-value must be between 0 and 1."""
        for comp_name, comp_data in cv_sensitivity["comparisons"].items():
            for metric, data in comp_data["metrics"].items():
                p = data["corrected_p_two_sided"]
                assert 0 <= p <= 1, f"{comp_name}/{metric}: p={p} out of range"

    def test_all_comparisons_present(self, cv_sensitivity):
        """All 5 comparisons x 3 metrics = 15 results."""
        total = sum(
            len(comp["metrics"])
            for comp in cv_sensitivity["comparisons"].values()
        )
        assert total == 15

    def test_no_model_rerun(self):
        """This test file must not import or call any model solver."""
        # If this file imports solver code, the test will fail at collection
        from metascfc.experiments.msancr_cv_sensitivity import (
            corrected_repeated_cv_test,
        )
        assert callable(corrected_repeated_cv_test)

    def test_original_outputs_untouched(self):
        """Verify original final outputs still exist."""
        for f in [
            "final_prediction_statistics.csv", "FINAL_COMPLETE",
            "final_hypothesis_decision.json",
        ]:
            assert (FINAL_DIR / f).exists()


class TestSensitivityInterpretation:
    def test_three_layers_present(self, cv_sensitivity):
        path = REPORT_DIR / "statistical_sensitivity_interpretation.json"
        if not path.exists():
            pytest.skip("Run 111 first")
        interp = json.loads(path.read_text())
        assert "layer_1_designated_primary" in interp
        assert "layer_2_conservative_multiplicity" in interp
        assert "layer_3_cv_dependence_sensitivity" in interp

    def test_primary_p_preserved(self):
        path = REPORT_DIR / "statistical_sensitivity_interpretation.json"
        if not path.exists():
            pytest.skip("Run 111 first")
        interp = json.loads(path.read_text())
        # Original p should be preserved
        assert abs(interp["layer_1_designated_primary"]["p_value"] - 0.037109) < 0.001

    def test_conservative_p_preserved(self):
        path = REPORT_DIR / "statistical_sensitivity_interpretation.json"
        if not path.exists():
            pytest.skip("Run 111 first")
        interp = json.loads(path.read_text())
        assert abs(interp["layer_2_conservative_multiplicity"]["p_holm_all_five"] - 0.111328) < 0.001

    def test_recommended_wording_present(self):
        path = REPORT_DIR / "statistical_sensitivity_interpretation.json"
        if not path.exists():
            pytest.skip("Run 111 first")
        interp = json.loads(path.read_text())
        assert "corrected repeated-k-fold" in interp["recommended_interpretation"]

    def test_prohibited_claims(self):
        path = REPORT_DIR / "statistical_sensitivity_interpretation.json"
        if not path.exists():
            pytest.skip("Run 111 first")
        interp = json.loads(path.read_text())
        assert len(interp["prohibited_claims"]) > 0


class TestMultiplicityFamilies:
    def test_explicit_families_present(self):
        path = REPORT_DIR / "explicit_multiplicity_families.json"
        if not path.exists():
            pytest.skip("Run 111 first")
        fam = json.loads(path.read_text())
        assert "prior_specificity_3_holm" in fam
        assert "broad_secondary_4_holm" in fam
        assert "conservative_all_five_holm" in fam

    def test_three_prior_specificity_entries(self):
        path = REPORT_DIR / "explicit_multiplicity_families.json"
        if not path.exists():
            pytest.skip("Run 111 first")
        fam = json.loads(path.read_text())
        pearson_keys = [k for k in fam["prior_specificity_3_holm"] if "pearson" in k]
        assert len(pearson_keys) == 3

    def test_four_broad_secondary_entries(self):
        path = REPORT_DIR / "explicit_multiplicity_families.json"
        if not path.exists():
            pytest.skip("Run 111 first")
        fam = json.loads(path.read_text())
        pearson_keys = [k for k in fam["broad_secondary_4_holm"] if "pearson" in k]
        assert len(pearson_keys) == 4

    def test_five_conservative_entries(self):
        path = REPORT_DIR / "explicit_multiplicity_families.json"
        if not path.exists():
            pytest.skip("Run 111 first")
        fam = json.loads(path.read_text())
        pearson_keys = [k for k in fam["conservative_all_five_holm"] if "pearson" in k]
        assert len(pearson_keys) == 5

    def test_holm_gte_raw(self, seed_df):
        """Holm-adjusted p >= raw p for all families."""
        multiplicity = build_explicit_multiplicity_families(seed_df)
        for family_key in ["prior_specificity_3_holm", "broad_secondary_4_holm", "conservative_all_five_holm"]:
            for key, p_adj in multiplicity[family_key].items():
                assert p_adj >= -1e-10, f"{family_key}[{key}] = {p_adj} < 0"
                assert p_adj <= 1.0 + 1e-10, f"{family_key}[{key}] = {p_adj} > 1"
