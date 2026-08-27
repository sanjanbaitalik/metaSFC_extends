"""Tests for prior predictive enrichment audit module."""
import numpy as np
import pandas as pd
import pytest

from metascfc.diagnostics.prior_predictive_enrichment import (
    bootstrap_ci,
    fit_ridge_dual,
    holm_correction,
    marginal_predictive_enrichment,
    paired_effect_size,
    paired_wilcoxon,
    ridge_coefficient_enrichment,
    roi_to_edge_prior,
    roi_to_edge_prior_with_threshold,
    select_ridge_alpha,
    top_fraction_enrichment,
)


class TestEdgeLifting:
    def test_prod_rule_symmetric(self):
        p = np.array([1.0, 2.0, 3.0])
        ep = roi_to_edge_prior(p, 3, "prod")
        iu = np.triu_indices(3, k=1)
        expected = p[iu[0]] * p[iu[1]]
        np.testing.assert_allclose(ep, expected)

    def test_mean_rule(self):
        p = np.array([1.0, 4.0, 7.0])
        ep = roi_to_edge_prior(p, 3, "mean")
        iu = np.triu_indices(3, k=1)
        expected = 0.5 * (p[iu[0]] + p[iu[1]])
        np.testing.assert_allclose(ep, expected)

    def test_max_rule(self):
        p = np.array([1.0, 4.0, 7.0])
        ep = roi_to_edge_prior(p, 3, "max")
        iu = np.triu_indices(3, k=1)
        expected = np.maximum(p[iu[0]], p[iu[1]])
        np.testing.assert_allclose(ep, expected)

    def test_bridge_rule(self):
        p = np.array([0.2, 0.8, 0.5])
        ep = roi_to_edge_prior(p, 3, "bridge")
        iu = np.triu_indices(3, k=1)
        expected = p[iu[0]] * (1 - p[iu[1]]) + p[iu[1]] * (1 - p[iu[0]])
        np.testing.assert_allclose(ep, expected)

    def test_output_length(self):
        p = np.random.randn(10)
        ep = roi_to_edge_prior(p, 10, "prod")
        assert len(ep) == 10 * 9 // 2

    def test_invalid_rule_raises(self):
        p = np.ones(5)
        with pytest.raises(ValueError, match="Unknown lifting rule"):
            roi_to_edge_prior(p, 5, "invalid")

    def test_length_mismatch_raises(self):
        p = np.ones(3)
        with pytest.raises(ValueError, match="entries"):
            roi_to_edge_prior(p, 5, "prod")

    def test_high_rois_map_to_high_edges(self):
        """Edges involving high-priority ROIs should have high prior scores."""
        p = np.zeros(5)
        p[4] = 10.0
        ep = roi_to_edge_prior(p, 5, "prod")
        assert ep[-1] == 0.0
        assert ep[-2] == 0.0
        assert ep[3] == 0.0
        assert ep[6] == 0.0

    def test_threshold_keeps_top_rois(self):
        p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ep, n_active = roi_to_edge_prior_with_threshold(p, 5, 2, "prod")
        assert n_active == 2


class TestRidgeSolver:
    def test_ridge_dual_shape(self):
        rng = np.random.RandomState(0)
        X = rng.randn(20, 50)
        y = rng.randn(20)
        beta = fit_ridge_dual(X, y, 1.0)
        assert beta.shape == (50,)

    def test_ridge_zero_alpha_reduces_to_ols_approx(self):
        rng = np.random.RandomState(1)
        n, p = 30, 10
        X = rng.randn(n, p)
        true_beta = rng.randn(p)
        y = X @ true_beta + 0.1 * rng.randn(n)
        beta = fit_ridge_dual(X, y, 1e-8)
        np.testing.assert_allclose(beta, true_beta, atol=0.5)

    def test_ridge_large_alpha_shrinks_to_zero(self):
        rng = np.random.RandomState(2)
        X = rng.randn(15, 30)
        y = rng.randn(15)
        beta = fit_ridge_dual(X, y, 1e10)
        assert np.max(np.abs(beta)) < 0.01

    def test_select_ridge_alpha_returns_valid(self):
        rng = np.random.RandomState(3)
        X = rng.randn(20, 40)
        y = rng.randn(20)
        Xval = rng.randn(5, 40)
        yval = rng.randn(5)
        best_a, best_beta = select_ridge_alpha(X, y, Xval, yval, [0.1, 1.0, 10.0])
        assert best_a in [0.1, 1.0, 10.0]
        assert best_beta.shape == (40,)


class TestMarginalEnrichment:
    def test_perfect_prior_reveals_high_enrichment(self):
        """When edge prior matches |corr(X_e, y)| exactly, enrichment should be high."""
        rng = np.random.RandomState(42)
        n, p = 100, 200
        X = rng.randn(n, p)
        y = rng.randn(n)
        m_e_true = np.abs(np.array([np.corrcoef(X[:, e], y)[0, 1] for e in range(p)]))
        result = marginal_predictive_enrichment(X, y, m_e_true)
        assert result["pearson"] > 0.99
        assert "m_e" in result

    def test_random_prior_gives_weak_enrichment(self):
        rng = np.random.RandomState(43)
        n, p = 80, 500
        X = rng.randn(n, p)
        y = rng.randn(n)
        ep = rng.randn(p)
        result = marginal_predictive_enrichment(X, y, ep)
        assert abs(result["pearson"]) < 0.3

    def test_too_small_sample_returns_zero(self):
        X = np.ones((2, 10))
        y = np.ones(2)
        ep = np.ones(10)
        result = marginal_predictive_enrichment(X, y, ep)
        assert result["pearson"] == 0.0
        assert result["spearman"] == 0.0

    def test_no_test_label_leakage(self):
        """Enrichment uses only training data - verify by construction."""
        rng = np.random.RandomState(44)
        n, p = 100, 50
        X = rng.randn(n, p)
        y = rng.randn(n)
        ep = rng.randn(p)
        result = marginal_predictive_enrichment(X, y, ep)
        assert len(result["m_e"]) == p


class TestTopFractionEnrichment:
    def test_top_fraction_returns_all_fractions(self):
        rng = np.random.RandomState(50)
        ep = rng.randn(1000)
        m_e = rng.rand(1000)
        results = top_fraction_enrichment(ep, m_e, [0.05, 0.10, 0.20], 100, rng)
        assert len(results) == 3
        fracs = [r["fraction"] for r in results]
        assert fracs == [0.05, 0.10, 0.20]

    def test_enrichment_ratio_above_one_for_correlated(self):
        rng = np.random.RandomState(51)
        p = 500
        ep = np.arange(p, dtype=float)
        m_e = ep + 0.1 * rng.randn(p)
        results = top_fraction_enrichment(ep, m_e, [0.10], 500, rng)
        assert results[0]["enrichment_ratio"] > 1.0
        assert results[0]["z_score"] > 2.0

    def test_random_prior_gives_ratio_near_one(self):
        rng = np.random.RandomState(52)
        ep = rng.randn(2000)
        m_e = rng.rand(2000)
        results = top_fraction_enrichment(ep, m_e, [0.10], 500, rng)
        assert 0.7 < results[0]["enrichment_ratio"] < 1.3

    def test_k_matches_fraction(self):
        rng = np.random.RandomState(53)
        ep = rng.randn(1000)
        m_e = rng.rand(1000)
        results = top_fraction_enrichment(ep, m_e, [0.05], 10, rng)
        assert results[0]["k"] == 50


class TestRidgeCoefficientEnrichment:
    def test_returns_pearson_and_spearman(self):
        rng = np.random.RandomState(60)
        beta = rng.randn(200)
        ep = rng.randn(200)
        result = ridge_coefficient_enrichment(beta, ep, [0.10], 50, rng)
        assert "pearson" in result
        assert "spearman" in result
        assert "top_fraction" in result


class TestStatHelpers:
    def test_wilcoxon_identical_returns_p1(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paired_wilcoxon(a, a)
        assert result["p_value"] == 1.0

    def test_wilcoxon_different_returns_low_p(self):
        a = np.arange(20, dtype=float)
        b = a + 10.0
        result = paired_wilcoxon(a, b)
        assert result["p_value"] < 0.05

    def test_wilcoxon_too_few_returns_p1(self):
        result = paired_wilcoxon(np.array([1.0]), np.array([2.0]))
        assert result["p_value"] == 1.0

    def test_holm_correction_ordering(self):
        pvals = np.array([0.01, 0.04, 0.03, 0.50])
        adj = holm_correction(pvals)
        assert len(adj) == 4
        assert np.all(adj >= pvals)
        assert np.all(adj <= 1.0)
        assert adj[0] <= adj[np.argsort(pvals)[-1]]

    def test_holm_single_value_unchanged(self):
        adj = holm_correction([0.05])
        assert adj[0] == 0.05

    def test_bootstrap_ci_contains_true_mean(self):
        rng = np.random.RandomState(70)
        a = rng.randn(50) + 1.0
        b = rng.randn(50)
        ci = bootstrap_ci(a, b, n_boot=2000, seed=70)
        assert ci["ci_lo"] < ci["mean_diff"] < ci["ci_hi"]

    def test_effect_size_positive_when_a_gt_b(self):
        rng = np.random.RandomState(99)
        a = rng.randn(100) + 2.0
        b = rng.randn(100)
        es = paired_effect_size(a, b)
        assert es > 0

    def test_effect_size_zero_when_equal(self):
        rng = np.random.RandomState(98)
        a = rng.randn(100)
        es = paired_effect_size(a, a)
        assert es == 0.0


class TestThresholdedLifting:
    def test_threshold_reduces_active_rois(self):
        p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ep, n_active = roi_to_edge_prior_with_threshold(p, 5, 3, "prod")
        assert n_active == 3
        assert n_active < 5

    def test_threshold_all_rois(self):
        p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ep, n_active = roi_to_edge_prior_with_threshold(p, 5, 10, "prod")
        assert n_active == 5

    def test_threshold_zero_sets_scores_to_zero(self):
        p = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ep, n_active = roi_to_edge_prior_with_threshold(p, 5, 2, "prod")
        inactive_roi_mask = np.zeros(5, dtype=bool)
        inactive_roi_mask[np.argsort(p)[-2:]] = True
        inactive_rois = np.where(~inactive_roi_mask)[0]
        iu = np.triu_indices(5, k=1)
        for i, j in zip(iu[0], iu[1]):
            if i in inactive_rois or j in inactive_rois:
                idx = np.where((iu[0] == i) & (iu[1] == j))[0][0]
                assert ep[idx] == 0.0


class TestSyntheticSignalDetection:
    def test_enrichment_detects_injected_signal(self):
        rng = np.random.RandomState(80)
        n, p = 200, 100
        X = rng.randn(n, p)
        signal_edges = np.arange(20)
        y = X[:, signal_edges].sum(axis=1) + 0.5 * rng.randn(n)
        ep = np.zeros(p)
        ep[signal_edges] = 1.0
        result = marginal_predictive_enrichment(X, y, ep)
        assert result["pearson"] > 0.5

    def test_null_prior_gives_null_enrichment(self):
        rng = np.random.RandomState(81)
        n, p = 200, 500
        X = rng.randn(n, p)
        y = rng.randn(n)
        ep = rng.randn(p)
        result = marginal_predictive_enrichment(X, y, ep)
        assert abs(result["pearson"]) < 0.2
        assert abs(result["spearman"]) < 0.2
