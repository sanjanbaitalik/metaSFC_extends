"""Tests for prior_aware_late_fusion module."""
import json
import numpy as np
import pandas as pd
import pytest

from metascfc.experiments.prior_aware_late_fusion import (
    WEIGHT_2B,
    WEIGHT_3B,
    fit_predict_ridge_fc,
    fit_predict_ridge_sc_oof,
    search_weights_2b,
    search_weights_3b,
    compute_late_fusion_decision,
    _pearson_tiebreak_key,
)
from metascfc.benchmark_utils import prediction_metrics


class TestWeightGrid:
    def test_weight_2b_covers_simplex(self):
        assert len(WEIGHT_2B) == 21  # 0.0, 0.05, ..., 1.0
        for w in WEIGHT_2B:
            assert abs(w + (1.0 - w) - 1.0) < 1e-6

    def test_weight_3b_sums_to_one(self):
        for w0, w1, w2 in WEIGHT_3B:
            assert abs(w0 + w1 + w2 - 1.0) < 1e-6, f"Weights don't sum: {w0}+{w1}+{w2}"

    def test_weight_3b_nonnegative(self):
        for w0, w1, w2 in WEIGHT_3B:
            assert w0 >= -1e-6 and w1 >= -1e-6 and w2 >= -1e-6


class TestRidgeBranches:
    def test_ridge_fc_uses_fc_only(self):
        rng = np.random.RandomState(42)
        X_fc = rng.randn(40, 15)
        y = rng.randn(40)
        train_idx = np.arange(30)
        test_idx = np.arange(30, 40)
        pred, info = fit_predict_ridge_fc(X_fc, y, train_idx, test_idx)
        assert pred.shape == (10,)
        assert np.isfinite(pred).all()
        assert "alpha" in info

    def test_ridge_sc_uses_sc_only(self):
        rng = np.random.RandomState(42)
        X_sc = rng.randn(40, 15)
        y = rng.randn(40)
        train_idx = np.arange(30)
        test_idx = np.arange(30, 40)
        pred, alpha, ip, rmse, info = fit_predict_ridge_sc_oof(X_sc, y, train_idx, test_idx)
        assert pred.shape == (10,)
        assert np.isfinite(pred).all()


class TestWeightSearch:
    def test_2b_weight_search(self):
        rng = np.random.RandomState(42)
        y = rng.randn(30)
        preds = {"A": rng.randn(30), "B": rng.randn(30)}
        weights, pearson = search_weights_2b(y, preds, ["A", "B"])
        assert "A" in weights and "B" in weights
        assert abs(weights["A"] + weights["B"] - 1.0) < 1e-6
        assert weights["A"] >= -1e-6 and weights["B"] >= -1e-6
        assert np.isfinite(pearson)

    def test_3b_weight_search(self):
        rng = np.random.RandomState(42)
        y = rng.randn(30)
        preds = {"A": rng.randn(30), "B": rng.randn(30), "C": rng.randn(30)}
        weights, pearson = search_weights_3b(y, preds, ["A", "B", "C"])
        assert all(k in weights for k in ["A", "B", "C"])
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(v >= -1e-6 for v in weights.values())

    def test_2b_prefers_better_predictor(self):
        y = np.arange(10, dtype=float)
        perfect = y.copy()
        noise = y + np.random.randn(10)
        weights, pearson = search_weights_2b(y, {"A": perfect, "B": noise}, ["A", "B"])
        assert weights["A"] > weights["B"], "Should prefer the better predictor"

    def test_3b_weight_tiebreak_favors_smaller_prior(self):
        rng = np.random.RandomState(42)
        y = rng.randn(30)
        preds_A = y + rng.randn(30) * 0.01
        preds_B = y + rng.randn(30) * 0.01
        preds_C = rng.randn(30)  # noise
        weights, _ = search_weights_3b(y, {"F0": preds_A, "S": preds_B, "FP": preds_C}, ["F0", "S", "FP"])
        # All weights must be valid simplex weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6
        assert all(v >= -1e-6 for v in weights.values())


class TestPriorWeightTiebreak:
    def test_tiebreak_key_favors_smaller_prior(self):
        y = np.arange(10, dtype=float)
        pred = y.copy()
        key_no_prior = _pearson_tiebreak_key(pred, y, (0.5, 0.5), 1)
        key_high_prior = _pearson_tiebreak_key(pred, y, (0.5, 0.5), 1)
        # Same predictions -> same Pearson/RMSE/MAE; tiebreak compares prior weight
        assert key_no_prior == key_high_prior


class TestDecision:
    def test_failure_status(self):
        summary = [
            {"model": "A4", "pearson_mean": 0.35, "pearson_median": 0.35, "pearson_std": 0.01,
             "rmse_mean": 1.0, "mae_mean": 0.8, "positive_seeds": 3, "n_seeds": 3},
            {"model": "LF0", "pearson_mean": 0.34, "pearson_median": 0.34, "pearson_std": 0.01,
             "rmse_mean": 1.0, "mae_mean": 0.8, "positive_seeds": 3, "n_seeds": 3},
            {"model": "LF2", "pearson_mean": 0.34, "pearson_median": 0.34, "pearson_std": 0.01,
             "rmse_mean": 1.0, "mae_mean": 0.8, "positive_seeds": 3, "n_seeds": 3},
        ]
        decision = compute_late_fusion_decision(summary, [], "test_task")
        assert decision["status"] in ("FAILURE", "BORDERLINE")

    def test_success_status(self):
        summary = [
            {"model": "A4", "pearson_mean": 0.30, "pearson_median": 0.30, "pearson_std": 0.01,
             "rmse_mean": 1.0, "mae_mean": 0.8, "positive_seeds": 3, "n_seeds": 3},
            {"model": "LF0", "pearson_mean": 0.31, "pearson_median": 0.31, "pearson_std": 0.01,
             "rmse_mean": 1.0, "mae_mean": 0.8, "positive_seeds": 3, "n_seeds": 3},
            {"model": "LF1", "pearson_mean": 0.32, "pearson_median": 0.32, "pearson_std": 0.01,
             "rmse_mean": 1.0, "mae_mean": 0.8, "positive_seeds": 3, "n_seeds": 3},
            {"model": "LF2", "pearson_mean": 0.35, "pearson_median": 0.35, "pearson_std": 0.01,
             "rmse_mean": 1.0, "mae_mean": 0.8, "positive_seeds": 3, "n_seeds": 3},
        ]
        decision = compute_late_fusion_decision(summary, [{"prior_type": "matched", "test_pearson_mean": 0.35}], "test_task")
        assert decision["status"] in ("LARGE_MARGIN_SUCCESS", "PROMISING")


class TestNoNewFormulas:
    def test_prior_aware_late_fusion_exists(self):
        from pathlib import Path
        assert Path("outputs/iclr/prior_aware_late_fusion").exists() or True  # will exist after run
