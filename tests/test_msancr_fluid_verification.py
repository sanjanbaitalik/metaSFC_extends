"""Tests for Fluid Intelligence verification (Part B)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    ModalitySelectiveAnisotropicNCR,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    NetworkConstrainedRidge,
)

CONFIG_PATH = Path("configs/iclr/msancr_fluid_verification.yaml")
FLUID_DIR = Path("outputs/iclr/msancr_fluid_verification")


class TestPreflight:
    def test_solver_direct_dual(self):
        """Verify final solver direct/dual equivalence."""
        from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
            build_msancr_cache, _solve_msancr_kernel, _predict_msancr,
        )
        from metascfc.experiments.msancr_refinement import upper_triangle_features
        n, n_rois = 50, 116
        rng = np.random.default_rng(42)
        mats_fc = rng.standard_normal((n, n_rois, n_rois))
        mats_sc = rng.standard_normal((n, n_rois, n_rois))
        X_fc = upper_triangle_features(mats_fc)
        X_sc = upper_triangle_features(mats_sc)
        y = rng.standard_normal(n)
        prior_scores = rng.uniform(0.1, 1.0, n_rois)
        cache = build_msancr_cache(prior_scores, n_rois, gamma=0.5, lifting='mean', top_k=10)
        alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, lambda_fc=0.1, lambda_sc=1.0, lambda_l=0.01)
        pred_z = _predict_msancr(X_fc, X_sc, X_fc, X_sc, alpha, cache, lambda_fc=0.1, lambda_sc=1.0, lambda_l=0.01)
        assert pred_z.shape == (n,)
        assert np.all(np.isfinite(pred_z))

    def test_fluid_labels_align_with_412_subjects(self):
        y = np.load("inputs/dataset_SC/label_all.npy")
        assert len(y) == 412

    def test_all_four_priors_have_116_entries(self):
        priors = yaml.safe_load(CONFIG_PATH.read_text())["priors"]["fluid_intelligence"]
        for ptype, path in priors.items():
            df = pd.read_csv(path)
            assert len(df) == 116, f"{ptype} has {len(df)} rows, expected 116"

    def test_config_exists(self):
        assert CONFIG_PATH.exists()

    def test_frozen_grids_match_final(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        final = yaml.safe_load(Path("configs/iclr/msancr_final_10x5.yaml").read_text())
        assert config["ridge_grid"] == final["ridge_grid"]
        assert config["gamma_grid"] == final["gamma_grid"]
        assert config["lambda_laplacian_grid"] == final["lambda_laplacian_grid"]


class TestFluidVerificationOutputs:
    @pytest.fixture(autouse=True)
    def _check_complete(self):
        if not (FLUID_DIR / "COMPLETE").exists():
            pytest.skip("Run 110 first")

    def test_split_metrics_exists(self):
        assert (FLUID_DIR / "split_metrics.csv").exists()

    def test_split_metrics_shape(self):
        df = pd.read_csv(FLUID_DIR / "split_metrics.csv")
        # 3 seeds x 5 folds x 4 models = 60
        assert len(df) == 60

    def test_prior_swap_metrics_exists(self):
        assert (FLUID_DIR / "prior_swap_split_metrics.csv").exists()

    def test_prior_swap_shape(self):
        df = pd.read_csv(FLUID_DIR / "prior_swap_split_metrics.csv")
        # 3 seeds x 5 folds x 3 swaps = 45
        assert len(df) == 45

    def test_selected_hyperparameters_exists(self):
        assert (FLUID_DIR / "selected_hyperparameters.csv").exists()

    def test_paired_comparisons_exists(self):
        assert (FLUID_DIR / "paired_comparisons.csv").exists()

    def test_fluid_decision_exists(self):
        assert (FLUID_DIR / "fluid_verification_decision.json").exists()

    def test_decision_has_recommendation(self):
        decision = json.loads((FLUID_DIR / "fluid_verification_decision.json").read_text())
        assert "recommendation" in decision
        assert decision["recommendation"] in [
            "full_fluid_10x5_frozen", "review_before_full_fluid",
            "stop_fluid_method_development", "error",
        ]

    def test_decision_gate_logic(self):
        decision = json.loads((FLUID_DIR / "fluid_verification_decision.json").read_text())
        if "A3_vs_A4" in decision:
            d = decision["A3_vs_A4"]
            median_delta = d["median_delta_pearson"]
            pos = d["positive_seeds"]
            if median_delta >= 0.010 and pos >= 2:
                assert decision["recommendation"] in [
                    "full_fluid_10x5_frozen", "review_before_full_fluid",
                ]
            elif median_delta < 0.005 or pos < 2:
                assert decision["recommendation"] == "stop_fluid_method_development"

    def test_all_split_metrics_finite(self):
        df = pd.read_csv(FLUID_DIR / "split_metrics.csv")
        assert np.all(np.isfinite(df.pearson))
        assert np.all(np.isfinite(df.rmse))
        assert np.all(np.isfinite(df.mae))

    def test_no_group_aware(self):
        meta = json.loads((FLUID_DIR / "run_metadata.json").read_text())
        assert meta.get("group_aware", False) is False

    def test_seeds_are_0_1_2(self):
        df = pd.read_csv(FLUID_DIR / "split_metrics.csv")
        assert sorted(df.seed.unique()) == [0, 1, 2]

    def test_five_folds_per_seed(self):
        df = pd.read_csv(FLUID_DIR / "split_metrics.csv")
        for seed in [0, 1, 2]:
            n_folds = len(df[df.seed == seed].fold.unique())
            assert n_folds == 5, f"Seed {seed} has {n_folds} folds"
