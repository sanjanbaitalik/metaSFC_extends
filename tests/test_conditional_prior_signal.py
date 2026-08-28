"""Tests for conditional / residual prior-signal audit."""
import numpy as np
import pandas as pd
import pytest

from metascfc.diagnostics.conditional_prior_signal import (
    compute_additive_metrics,
    crossfit_ridge,
    fast_top_fraction,
    fit_prior_pca_ridge,
    fit_prior_topk_ridge,
    fit_prior_weighted_ridge,
    fit_ridge_baseline,
    residual_enrichment,
    seed_level_stats,
)
from metascfc.diagnostics.prior_predictive_enrichment import (
    roi_to_edge_prior,
)


# ---------------------------------------------------------------------------
# Cross-fitted residuals — no leakage
# ---------------------------------------------------------------------------

class TestCrossfitResiduals:
    def test_no_subject_used_for_own_prediction(self):
        rng = np.random.RandomState(0)
        n, p = 80, 30
        X = rng.randn(n, p)
        y = X[:, :5].sum(axis=1) + 0.1 * rng.randn(n)
        indices = np.arange(n)
        pred, res = crossfit_ridge(X, y, indices, n_folds=5,
                                   alpha_grid=[1.0], rng=rng)
        assert pred.shape == (n,)
        np.testing.assert_allclose(res, y - pred, atol=1e-10)

    def test_crossfit_pred_differs_from_insample(self):
        rng = np.random.RandomState(1)
        n, p = 60, 20
        X = rng.randn(n, p)
        y = rng.randn(n)
        pred, _ = crossfit_ridge(X, y, np.arange(n), n_folds=5,
                                 alpha_grid=[1.0], rng=rng)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        beta = np.linalg.lstsq(Xs, y, rcond=None)[0]
        insample = Xs @ beta
        corr = np.corrcoef(pred, insample)[0, 1]
        assert corr < 0.99

    def test_residuals_mean_is_nonzero(self):
        rng = np.random.RandomState(2)
        X = rng.randn(50, 10)
        y = rng.randn(50) + 5.0
        _, res = crossfit_ridge(X, y, np.arange(50), n_folds=5,
                                alpha_grid=[1.0], rng=rng)
        assert np.abs(res.mean()) < 5.0

    def test_alpha_selection_uses_inner_split(self):
        rng = np.random.RandomState(3)
        X = rng.randn(40, 15)
        y = rng.randn(40)
        _, res = crossfit_ridge(X, y, np.arange(40), n_folds=3,
                                alpha_grid=[0.01, 1.0, 100.0], rng=rng)
        assert res.shape == (40,)


# ---------------------------------------------------------------------------
# Baseline Ridge
# ---------------------------------------------------------------------------

class TestBaselineRidge:
    def test_predict_test_shape(self):
        rng = np.random.RandomState(10)
        n, p = 100, 30
        X = rng.randn(n, p)
        y = rng.randn(n)
        tv = np.arange(80)
        te = np.arange(80, 100)
        pred_te, pred_tv, alpha = fit_ridge_baseline(
            X, y, tv, te, [1.0, 10.0], rng=rng)
        assert pred_te.shape == (20,)
        assert pred_tv.shape == (80,)
        assert alpha in [1.0, 10.0]

    def test_baseline_recovers_signal(self):
        rng = np.random.RandomState(11)
        n, p = 120, 20
        X = rng.randn(n, p)
        true_beta = np.zeros(p)
        true_beta[:3] = [2.0, -1.0, 0.5]
        y = X @ true_beta + 0.1 * rng.randn(n)
        tv = np.arange(100)
        te = np.arange(100, 120)
        pred_te, _, _ = fit_ridge_baseline(
            X, y, tv, te, [0.01, 1.0, 100.0], rng=rng)
        r = np.corrcoef(y[te], pred_te)[0, 1]
        assert r > 0.5


# ---------------------------------------------------------------------------
# Residual enrichment
# ---------------------------------------------------------------------------

class TestResidualEnrichment:
    def test_perfect_prior_detection(self):
        rng = np.random.RandomState(20)
        n, p = 100, 200
        X = rng.randn(n, p)
        residuals = rng.randn(n)
        m_true = np.abs(np.array([np.corrcoef(X[:, e], residuals)[0, 1]
                                  for e in range(p)]))
        pr, sr, m_e = residual_enrichment(X, residuals, m_true)
        assert pr > 0.95

    def test_random_prior_weak(self):
        rng = np.random.RandomState(21)
        n, p = 80, 500
        X = rng.randn(n, p)
        res = rng.randn(n)
        ep = rng.randn(p)
        pr, sr, _ = residual_enrichment(X, res, ep)
        assert abs(pr) < 0.25

    def test_constant_residual_returns_zero(self):
        rng = np.random.RandomState(22)
        X = rng.randn(50, 10)
        res = np.zeros(50)
        ep = rng.randn(10)
        pr, sr, m_e = residual_enrichment(X, res, ep)
        assert pr == 0.0
        assert sr == 0.0
        assert np.all(m_e == 0.0)

    def test_small_n_returns_zero(self):
        X = np.ones((2, 5))
        res = np.ones(2)
        ep = np.ones(5)
        pr, sr, _ = residual_enrichment(X, res, ep)
        assert pr == 0.0


# ---------------------------------------------------------------------------
# Top fraction
# ---------------------------------------------------------------------------

class TestTopFraction:
    def test_returns_all_fractions(self):
        rng = np.random.RandomState(30)
        ep = rng.randn(1000)
        m_e = rng.rand(1000)
        res = fast_top_fraction(ep, m_e, [0.05, 0.10, 0.20], 100, rng)
        assert len(res) == 3

    def test_correlated_high_enrichment(self):
        rng = np.random.RandomState(31)
        p = 500
        ep = np.arange(p, dtype=float)
        m_e = ep + 0.1 * rng.randn(p)
        res = fast_top_fraction(ep, m_e, [0.10], 500, rng)
        assert res[0]["enrichment_ratio"] > 1.0
        assert res[0]["z_score"] > 2.0

    def test_random_near_one(self):
        rng = np.random.RandomState(32)
        ep = rng.randn(2000)
        m_e = rng.rand(2000)
        res = fast_top_fraction(ep, m_e, [0.10], 500, rng)
        assert 0.7 < res[0]["enrichment_ratio"] < 1.3


# ---------------------------------------------------------------------------
# Prior-only models
# ---------------------------------------------------------------------------

class TestPriorModels:
    def _setup(self, seed=40):
        rng = np.random.RandomState(seed)
        n, p = 100, 200
        X = rng.randn(n, p)
        signal = np.zeros(p)
        signal[:10] = rng.randn(10)
        y = X @ signal + 0.5 * rng.randn(n)
        return X, y, np.arange(80), np.arange(80, 100), signal

    def test_topk_ridge_output_keys(self):
        X, y, tv, te, sig = self._setup()
        ep = np.abs(np.arange(200, dtype=float))
        res = fit_prior_topk_ridge(X, y[tv], tv, te, ep, 0.10, [1.0])
        assert "pred_test" in res
        assert "pred_trainval" in res
        assert res["pred_test"].shape == (20,)
        assert res["pred_trainval"].shape == (80,)

    def test_weighted_ridge_output_keys(self):
        X, y, tv, te, sig = self._setup()
        ep = np.abs(np.arange(200, dtype=float))
        res = fit_prior_weighted_ridge(X, y[tv], tv, te, ep, 1.0, 0.001, [1.0])
        assert res["pred_test"].shape == (20,)
        assert res["variant"] == "weighted"

    def test_pca_ridge_output_keys(self):
        X, y, tv, te, sig = self._setup()
        ep = np.abs(np.arange(200, dtype=float))
        res = fit_prior_pca_ridge(X, y[tv], tv, te, ep, 8, [1.0])
        assert res["pred_test"].shape == (20,)
        assert res["variant"] == "pca"
        assert "explained_variance" in res

    def test_pca_fitted_training_only(self):
        X, y, tv, te, sig = self._setup(seed=41)
        ep = np.abs(np.arange(200, dtype=float))
        res = fit_prior_pca_ridge(X, y[tv], tv, te, ep, 8, [1.0])
        assert 0.0 <= res["explained_variance"] <= 1.0


# ---------------------------------------------------------------------------
# Additive metrics / eta selection
# ---------------------------------------------------------------------------

class TestAdditiveMetrics:
    def test_eta_zero_recovers_baseline(self):
        rng = np.random.RandomState(50)
        y_test = rng.randn(30)
        baseline = y_test + 0.1 * rng.randn(30)
        prior = rng.randn(30)
        y_tv = rng.randn(80)
        bl_tv = y_tv + 0.1 * rng.randn(80)
        prior_tv = rng.randn(80)
        result = compute_additive_metrics(
            baseline, prior, y_test, bl_tv, prior_tv, y_tv, [0.0])
        assert result["selected_eta"] == 0.0
        assert abs(result["delta_pearson"]) < 0.01

    def test_eta_benefit_detected(self):
        rng = np.random.RandomState(51)
        y_test = rng.randn(50)
        baseline = y_test + 0.5 * rng.randn(50)
        prior = 0.3 * (y_test - baseline) + 0.1 * rng.randn(50)
        y_tv = rng.randn(80)
        bl_tv = y_tv + 0.5 * rng.randn(80)
        prior_tv = 0.3 * (y_tv - bl_tv) + 0.1 * rng.randn(80)
        result = compute_additive_metrics(
            baseline, prior, y_test, bl_tv, prior_tv, y_tv,
            [0.0, 0.5, 1.0])
        assert result["selected_eta"] > 0.0
        assert result["combined_rmse"] <= result["baseline_rmse"]


# ---------------------------------------------------------------------------
# Seed-level stats
# ---------------------------------------------------------------------------

class TestSeedStats:
    def test_identical_returns_p1(self):
        a = np.arange(10, dtype=float)
        st = seed_level_stats(a, a, "test")
        assert st["wilcoxon_p"] == 1.0

    def test_different_returns_low_p(self):
        a = np.arange(10, dtype=float) + 10.0
        b = np.arange(10, dtype=float)
        st = seed_level_stats(a, b, "test")
        assert st["wilcoxon_p"] < 0.05
        assert st["n_positive"] > 0

    def test_too_few_returns_p1(self):
        st = seed_level_stats(np.array([1.0]), np.array([2.0]), "test")
        assert st["wilcoxon_p"] == 1.0


# ---------------------------------------------------------------------------
# Leakage and correctness
# ---------------------------------------------------------------------------

class TestLeakageSafety:
    def test_crossfit_residuals_no_test_labels(self):
        rng = np.random.RandomState(60)
        X = rng.randn(80, 20)
        y = rng.randn(80)
        tv = np.arange(60)
        _, res = crossfit_ridge(X, y, tv, n_folds=3, alpha_grid=[1.0], rng=rng)
        assert res.shape == (60,)

    def test_prior_model_uses_residuals_not_y(self):
        rng = np.random.RandomState(61)
        n, p = 80, 50
        X = rng.randn(n, p)
        y = rng.randn(n)
        fake_res = rng.randn(60)
        ep = rng.randn(p)
        tv, te = np.arange(60), np.arange(60, 80)
        res = fit_prior_topk_ridge(X, fake_res, tv, te, ep, 0.10, [1.0], rng=rng)
        assert res["pred_test"].shape == (20,)

    def test_feature_ordering_matches(self):
        rng = np.random.RandomState(62)
        X = rng.randn(100, 40)
        ep = np.arange(40, dtype=float)
        tv, te = np.arange(80), np.arange(80, 100)
        y_tv = rng.randn(80)
        r1 = fit_prior_topk_ridge(X, y_tv, tv, te, ep, 0.25, [1.0], rng=rng)
        r2 = fit_prior_topk_ridge(X, y_tv, tv, te, ep, 0.25, [1.0], rng=rng)
        np.testing.assert_allclose(r1["pred_test"], r2["pred_test"])


# ---------------------------------------------------------------------------
# Synthetic signal detection
# ---------------------------------------------------------------------------

class TestSyntheticDetection:
    def test_aligned_prior_detects_residual_signal(self):
        rng = np.random.RandomState(70)
        n, p = 150, 200
        X = rng.randn(n, p)
        signal_edges = np.arange(20)
        y = X[:, signal_edges].sum(axis=1) + 0.5 * rng.randn(n)
        tv, te = np.arange(120), np.arange(120, 150)
        _, res_cf = crossfit_ridge(X, y, tv, n_folds=5, alpha_grid=[1.0], rng=rng)
        ep = np.zeros(p)
        ep[signal_edges] = 1.0
        ep_full = np.concatenate([ep, ep])[:p]
        pr, _, _ = residual_enrichment(X[tv], res_cf, ep_full[:p])
        assert pr > 0.3

    def test_shuffled_prior_no_gain(self):
        rng = np.random.RandomState(71)
        n, p = 120, 200
        X = rng.randn(n, p)
        y = rng.randn(n)
        tv = np.arange(100)
        _, res_cf = crossfit_ridge(X, y, tv, n_folds=5, alpha_grid=[1.0], rng=rng)
        ep = rng.permutation(np.arange(p, dtype=float))
        pr, _, _ = residual_enrichment(X[tv], res_cf, ep)
        assert abs(pr) < 0.2
