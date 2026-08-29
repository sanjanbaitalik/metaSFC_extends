"""Tests for post-final reporting patch (Part A)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metascfc.experiments.msancr_postfinal_reporting import (
    build_corrected_boundary_metadata,
    verify_grid_execution,
    build_primary_prediction_contrast,
    build_secondary_family_statistics,
    build_conservative_sensitivity,
    build_prior_specificity_summary,
    build_paper_safe_interpretation,
    build_prediction_table_v2,
    build_biomarker_table_v2,
    REQUIRED_RIDGE_GRID,
    REQUIRED_GAMMA_GRID,
    REQUIRED_LAMBDA_L_GRID,
)
from metascfc.experiments.msancr_final_inference import (
    MODEL_A3, MODEL_A4, MODEL_A2, build_seed_level_table,
)
from metascfc.benchmark_utils import holm_adjust


FINAL_DIR = Path("outputs/iclr/msancr_final_10x5")
CONFIG_PATH = Path("configs/iclr/msancr_final_10x5.yaml")


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


@pytest.fixture
def selected_df():
    return pd.read_csv(FINAL_DIR / "selected_hyperparameters.csv")


class TestA1BoundaryMetadata:
    def test_grids_match_config(self, selected_df):
        result = build_corrected_boundary_metadata(CONFIG_PATH, selected_df)
        assert result["grids_source"] == "frozen_final_config"
        assert result["ridge_grid"] == REQUIRED_RIDGE_GRID
        assert result["gamma_grid"] == REQUIRED_GAMMA_GRID
        assert result["lambda_laplacian_grid"] == REQUIRED_LAMBDA_L_GRID

    def test_boundary_counts_non_negative(self, selected_df):
        result = build_corrected_boundary_metadata(CONFIG_PATH, selected_df)
        for k, v in result["boundary_counts"].items():
            assert v >= 0, f"{k} is negative: {v}"
            assert v <= 50, f"{k} exceeds 50: {v}"

    def test_no_duplicate_grid_endpoints(self, selected_df):
        result = build_corrected_boundary_metadata(CONFIG_PATH, selected_df)
        for grid_name in ("ridge_grid", "gamma_grid", "lambda_laplacian_grid"):
            grid = result[grid_name]
            assert len(grid) == len(set(grid)), f"Duplicate in {grid_name}: {grid}"

    def test_lambda_fc_in_config_grid(self, selected_df):
        result = build_corrected_boundary_metadata(CONFIG_PATH, selected_df)
        a3_vals = set(selected_df[selected_df.model_id == MODEL_A3].lambda_fc.unique())
        config_grid = set(result["ridge_grid"])
        assert a3_vals.issubset(config_grid), (
            f"A3 lambda_fc values {a3_vals} not subset of config grid {config_grid}"
        )

    def test_distribution_keys_complete(self, selected_df):
        result = build_corrected_boundary_metadata(CONFIG_PATH, selected_df)
        assert "lambda_fc" in result["selected_distributions"]
        assert "gamma" in result["selected_distributions"]
        assert "lifting" in result["selected_distributions"]


class TestA2GridExecution:
    def test_grid_execution_valid(self):
        result = verify_grid_execution(CONFIG_PATH, FINAL_DIR / "inner_cv_metrics.csv")
        assert result["overall_grid_execution_valid"], result

    def test_a3_uses_gamma_and_lifting(self):
        result = verify_grid_execution(CONFIG_PATH, FINAL_DIR / "inner_cv_metrics.csv")
        a3 = result["models"]["A3_msancr"]
        assert a3["grid_match"], a3
        assert len(a3["observed_gamma_values"]) > 0
        assert len(a3["observed_lifting_values"]) > 0

    def test_a4_grid_valid(self):
        result = verify_grid_execution(CONFIG_PATH, FINAL_DIR / "inner_cv_metrics.csv")
        a4 = result["models"]["A4_modality_ridge"]
        assert a4["grid_match"], a4

    def test_a2_no_gamma_by_design(self):
        result = verify_grid_execution(CONFIG_PATH, FINAL_DIR / "inner_cv_metrics.csv")
        a2 = result["models"]["A2_fc_laplacian"]
        assert a2["grid_match"], a2
        assert "does not use the full" in a2.get("note", "")

    def test_all_ridge_values_present(self):
        result = verify_grid_execution(CONFIG_PATH, FINAL_DIR / "inner_cv_metrics.csv")
        for model_id, model_audit in result["models"].items():
            assert not model_audit["missing_from_ridge_grid"], (
                f"{model_id} missing ridge values: {model_audit['missing_from_ridge_grid']}"
            )


class TestA3PrimaryContrast:
    def test_primary_has_required_fields(self, seed_df):
        primary = build_primary_prediction_contrast(seed_df)
        assert primary["contrast"] == "A3_matched_vs_A4"
        assert primary["metric"] == "pearson"
        assert primary["n_seeds"] == 10
        assert 0 < primary["p_primary"] < 1
        assert len(primary["ci95"]) == 2
        assert primary["ci95"][0] < primary["ci95"][1]
        assert primary["positive_seeds"] + primary["negative_seeds"] <= 10

    def test_primary_p_matches_original(self, seed_df):
        primary = build_primary_prediction_contrast(seed_df)
        original = pd.read_csv(FINAL_DIR / "final_prediction_statistics.csv")
        orig_row = original[
            (original.comparison == "A3_matched_vs_A4") & (original.metric == "pearson")
        ].iloc[0]
        assert abs(primary["p_primary"] - orig_row.wilcoxon_p) < 1e-10

    def test_positive_plus_negative_plus_zero_equals_10(self, seed_df):
        primary = build_primary_prediction_contrast(seed_df)
        assert primary["positive_seeds"] + primary["negative_seeds"] + primary.get("zero_seeds", 0) == 10


class TestA3SecondaryFamily:
    def test_secondary_excludes_primary(self, seed_df):
        secondary = build_secondary_family_statistics(seed_df)
        assert "A3_matched_vs_A4" not in secondary.comparison.values

    def test_secondary_has_four_comparisons(self, seed_df):
        secondary = build_secondary_family_statistics(seed_df)
        assert secondary.comparison.nunique() == 4

    def test_holm_adjustment_applied(self, seed_df):
        secondary = build_secondary_family_statistics(seed_df)
        assert secondary.p_holm_secondary.notna().all()
        # Holm-adjusted p >= raw p
        assert (secondary.p_holm_secondary >= secondary.wilcoxon_p - 1e-12).all()

    def test_three_metrics_per_comparison(self, seed_df):
        secondary = build_secondary_family_statistics(seed_df)
        for comp in secondary.comparison.unique():
            n_metrics = len(secondary[secondary.comparison == comp])
            assert n_metrics == 3, f"{comp} has {n_metrics} metrics"


class TestA3ConservativeSensitivity:
    def test_conservative_matches_original(self, seed_df):
        conservative = build_conservative_sensitivity(seed_df)
        original = pd.read_csv(FINAL_DIR / "final_prediction_statistics.csv")
        orig_a3_a4 = original[
            (original.comparison == "A3_matched_vs_A4") & (original.metric == "pearson")
        ].iloc[0]
        assert abs(conservative["p_raw"] - orig_a3_a4.wilcoxon_p) < 1e-10, (
            f"Raw p mismatch: {conservative['p_raw']} vs {orig_a3_a4.wilcoxon_p}"
        )
        assert abs(conservative["p_holm_all_five"] - orig_a3_a4.p_holm_metric) < 1e-10, (
            f"Holm p mismatch: {conservative['p_holm_all_five']} vs {orig_a3_a4.p_holm_metric}"
        )

    def test_holm_greater_equal_raw(self, seed_df):
        conservative = build_conservative_sensitivity(seed_df)
        assert conservative["p_holm_all_five"] >= conservative["p_raw"] - 1e-12

    def test_five_comparison_holm_not_significant(self, seed_df):
        conservative = build_conservative_sensitivity(seed_df)
        assert conservative["p_holm_all_five"] > 0.05


class TestA4PriorSpecificity:
    def test_directional_superiority(self, seed_df):
        spec = build_prior_specificity_summary(seed_df)
        assert spec["paper_safe_claims"]["matched_greater_than_all_swaps_directionally"]

    def test_secondary_holm_for_swaps(self, seed_df):
        spec = build_prior_specificity_summary(seed_df)
        for swap_name in ("unrelated", "shuffled", "random"):
            swap = spec["swaps"][swap_name]
            assert "raw_wilcoxon_p" in swap
            assert "p_holm_secondary" in swap
            assert swap["p_holm_secondary"] >= swap["raw_wilcoxon_p"] - 1e-12

    def test_no_false_claim_all_significant(self, seed_df):
        spec = build_prior_specificity_summary(seed_df)
        # The random swap should NOT be significant under conservative Holm
        assert not spec["paper_safe_claims"]["matched_significant_after_conservative_all_five_holm"]


class TestA5PaperInterpretation:
    def test_has_recommended_wording(self, seed_df):
        primary = build_primary_prediction_contrast(seed_df)
        conservative = build_conservative_sensitivity(seed_df)
        spec = build_prior_specificity_summary(seed_df)
        interp = build_paper_safe_interpretation(primary, conservative, spec)
        assert len(interp["recommended_wording"]) > 100
        assert "did not remain significant" in interp["recommended_wording"]

    def test_has_prohibited_wording(self, seed_df):
        primary = build_primary_prediction_contrast(seed_df)
        conservative = build_conservative_sensitivity(seed_df)
        spec = build_prior_specificity_summary(seed_df)
        interp = build_paper_safe_interpretation(primary, conservative, spec)
        assert len(interp["prohibited_wording"]) > 0
        for phrase in interp["prohibited_wording"]:
            assert phrase not in interp["recommended_wording"]

    def test_nominal_significant_but_holm_not(self, seed_df):
        primary = build_primary_prediction_contrast(seed_df)
        conservative = build_conservative_sensitivity(seed_df)
        spec = build_prior_specificity_summary(seed_df)
        interp = build_paper_safe_interpretation(primary, conservative, spec)
        assert interp["primary_prediction_contrast"]["supported_at_nominal_005"]
        assert not interp["conservative_familywise_sensitivity"]["supported"]


class TestA6LatexTables:
    def test_prediction_table_valid_latex(self, seed_df):
        original_pred = pd.read_csv(FINAL_DIR / "final_prediction_statistics.csv")
        tex = build_prediction_table_v2(seed_df, original_pred)
        assert r"\begin{tabular}" in tex
        assert r"\end{tabular}" in tex
        assert r"\dagger" in tex
        assert r"\star" in tex

    def test_biomarker_table_valid_latex(self, seed_df):
        original_bio = pd.read_csv(FINAL_DIR / "final_biomarker_statistics.csv")
        tex = build_biomarker_table_v2(seed_df, original_bio)
        assert r"\begin{tabular}" in tex
        assert r"\end{tabular}" in tex

    def test_prediction_table_has_dagger_footnote(self, seed_df):
        original_pred = pd.read_csv(FINAL_DIR / "final_prediction_statistics.csv")
        tex = build_prediction_table_v2(seed_df, original_pred)
        assert "designated primary" in tex
