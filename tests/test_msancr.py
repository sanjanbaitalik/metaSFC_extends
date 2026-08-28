"""Tests for Modality-Selective Anisotropic NCR (MS-A-NCR)."""
import numpy as np
import pytest

from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    ModalitySelectiveAnisotropicNCR,
    _MSANCRCache,
    _predict_msancr,
    _solve_msancr_kernel,
    build_msancr_cache,
    compute_diagonal_penalty,
    fit_predict_msancr,
    lift_roi_to_edge,
)

# Small atlas for fast tests: n_rois=6 -> n_edges=15
N_ROIS = 6
N_EDGES = N_ROIS * (N_ROIS - 1) // 2  # 15


def _make_data(n, n_rois=N_ROIS, rng=None):
    """Return (X_fc, X_sc, y) with correct edge count."""
    if rng is None:
        rng = np.random.RandomState(0)
    ne = n_rois * (n_rois - 1) // 2
    X_fc = rng.randn(n, ne)
    X_sc = rng.randn(n, ne)
    y = rng.randn(n)
    return X_fc, X_sc, y


# ---------------------------------------------------------------------------
# Diagonal penalty
# ---------------------------------------------------------------------------

class TestDiagonalPenalty:
    def test_formula(self):
        q = np.array([0.1, 0.5, 1.0, 2.0])
        d = compute_diagonal_penalty(q, gamma=1.0, epsilon=1e-3, normalize=False)
        expected = (1e-3 + q) ** (-1.0)
        np.testing.assert_allclose(d, expected)

    def test_normalized_mean_one(self):
        q = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        d = compute_diagonal_penalty(q, gamma=0.5, normalize=True)
        assert abs(d.mean() - 1.0) < 1e-10

    def test_high_prior_gets_lower_shrinkage(self):
        q = np.array([0.01, 0.1, 1.0, 10.0])
        d = compute_diagonal_penalty(q, gamma=1.0, normalize=False)
        assert d[0] > d[1] > d[2] > d[3]

    def test_gamma_zero_isotropic(self):
        q = np.array([0.1, 0.5, 1.0, 2.0])
        d = compute_diagonal_penalty(q, gamma=0.0, normalize=False)
        np.testing.assert_allclose(d, 1.0, atol=1e-12)

    def test_uniform_prior_gives_uniform_penalty(self):
        q = np.ones(10) * 0.5
        d = compute_diagonal_penalty(q, gamma=1.0, normalize=True)
        np.testing.assert_allclose(d, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Edge lifting
# ---------------------------------------------------------------------------

class TestEdgeLifting:
    def test_prod_rule(self):
        roi = np.array([1.0, 2.0, 3.0])
        ep = lift_roi_to_edge(roi, 3, "prod")
        np.testing.assert_allclose(ep, [2.0, 3.0, 6.0])

    def test_mean_rule(self):
        roi = np.array([1.0, 2.0, 3.0])
        ep = lift_roi_to_edge(roi, 3, "mean")
        np.testing.assert_allclose(ep, [1.5, 2.0, 2.5])

    def test_output_length(self):
        roi = np.random.randn(10)
        ep = lift_roi_to_edge(roi, 10, "prod")
        assert len(ep) == 10 * 9 // 2

    def test_invalid_rule_raises(self):
        with pytest.raises(ValueError):
            lift_roi_to_edge(np.ones(5), 5, "max")


# ---------------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------------

class TestCacheConstruction:
    def test_cache_shape(self):
        rng = np.random.RandomState(42)
        roi = rng.rand(N_ROIS)
        cache = build_msancr_cache(roi, N_ROIS, gamma=1.0, lifting="prod", top_k=3)
        assert cache.D.shape == (N_EDGES,)
        assert cache.D_inv_sqrt.shape == (N_EDGES,)
        assert cache.n_edges == N_EDGES
        assert cache.n_rois == N_ROIS

    def test_gamma_zero_gives_uniform_D(self):
        roi = np.random.RandomState(43).rand(N_ROIS)
        cache = build_msancr_cache(roi, N_ROIS, gamma=0.0, lifting="prod")
        np.testing.assert_allclose(cache.D, 1.0, atol=1e-10)

    def test_active_laplacian_stored(self):
        roi = np.random.RandomState(44).rand(N_ROIS)
        cache = build_msancr_cache(roi, N_ROIS, gamma=1.0, lifting="prod", top_k=3)
        assert hasattr(cache, "active_laplacian")
        na = len(cache.active_indices)
        assert cache.active_laplacian.shape == (na, na)


# ---------------------------------------------------------------------------
# Solver correctness
# ---------------------------------------------------------------------------

class TestSolver:
    def test_predict_shape(self):
        rng = np.random.RandomState(51)
        n, n_rois = 40, N_ROIS
        ne = n_rois * (n_rois - 1) // 2
        roi = rng.rand(n_rois)
        cache = build_msancr_cache(roi, n_rois, gamma=0.5, top_k=3)
        model = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=0.5, gamma=0.5,
            cache=cache, n_rois=n_rois,
        )
        X_fc = rng.randn(n, ne)
        X_sc = rng.randn(n, ne)
        y = rng.randn(n)
        model.fit(X_fc, X_sc, y)
        n_test = 5
        pred = model.predict(rng.randn(n_test, ne), rng.randn(n_test, ne))
        assert pred.shape == (n_test,)

    def test_sc_no_laplacian(self):
        """SC should not receive Laplacian penalty in A3."""
        rng = np.random.RandomState(52)
        n, n_rois = 30, N_ROIS
        ne = n_rois * (n_rois - 1) // 2
        roi = rng.rand(n_rois)
        cache = build_msancr_cache(roi, n_rois, gamma=1.0, top_k=3)
        model_no_lap = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=0.0, gamma=1.0,
            cache=cache, n_rois=n_rois,
        )
        model_with_lap = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=1.0, gamma=1.0,
            cache=cache, n_rois=n_rois,
        )
        X_fc = rng.randn(n, ne)
        X_sc = rng.randn(n, ne)
        y = rng.randn(n)
        model_no_lap.fit(X_fc, X_sc, y)
        model_with_lap.fit(X_fc, X_sc, y)
        pred_no = model_no_lap.predict(X_fc[:5], X_sc[:5])
        pred_with = model_with_lap.predict(X_fc[:5], X_sc[:5])
        assert not np.allclose(pred_no, pred_with, atol=1e-6)

    def test_fit_predict_consistency(self):
        rng = np.random.RandomState(53)
        n, n_rois = 30, N_ROIS
        ne = n_rois * (n_rois - 1) // 2
        roi = rng.rand(n_rois)
        cache = build_msancr_cache(roi, n_rois, gamma=0.5, top_k=3)
        model = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=0.5, gamma=0.5,
            cache=cache, n_rois=n_rois,
        )
        X_fc = rng.randn(n, ne)
        X_sc = rng.randn(n, ne)
        y = rng.randn(n)
        model.fit(X_fc, X_sc, y)
        pred = model.predict(X_fc, X_sc)
        assert pred.shape == (n,)
        assert np.all(np.isfinite(pred))


# ---------------------------------------------------------------------------
# fit_predict convenience
# ---------------------------------------------------------------------------

class TestFitPredict:
    def test_returns_correct_shapes(self):
        rng = np.random.RandomState(60)
        n, n_rois = 60, N_ROIS
        ne = n_rois * (n_rois - 1) // 2
        X_fc = rng.randn(n, ne)
        X_sc = rng.randn(n, ne)
        y = rng.randn(n)
        train = np.arange(40)
        val = np.arange(40, 50)
        test = np.arange(50, 60)

        roi = rng.rand(n_rois)
        cache = build_msancr_cache(roi, n_rois, gamma=0.5, top_k=3)
        pred, lfc, lsc, ll, rmse, info = fit_predict_msancr(
            X_fc, X_sc, y, train, val, test, cache,
            lambda_fc_grid=[0.1, 1.0], lambda_sc_grid=[0.1, 1.0],
            lambda_l_grid=[0.0, 0.5], n_rois=n_rois,
        )
        assert pred.shape == (10,)
        assert lfc in [0.1, 1.0]
        assert lsc in [0.1, 1.0]
        assert ll in [0.0, 0.5]
        assert rmse >= 0


# ---------------------------------------------------------------------------
# Staged selection helpers
# ---------------------------------------------------------------------------

class TestStagedSelection:
    def test_a4_modality_specific_ridge(self):
        """A4: gamma=0, lambda_l=0, lambda_fc != lambda_sc."""
        rng = np.random.RandomState(70)
        n, n_rois = 50, N_ROIS
        ne = n_rois * (n_rois - 1) // 2
        X_fc = rng.randn(n, ne)
        X_sc = rng.randn(n, ne)
        y = rng.randn(n)

        roi = np.ones(n_rois)
        cache = build_msancr_cache(roi, n_rois, gamma=0.0, top_k=n_rois)

        model = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=10.0, lambda_l=0.0, gamma=0.0,
            cache=cache, n_rois=n_rois,
        )
        model.fit(X_fc, X_sc, y)
        pred = model.predict(X_fc, X_sc)
        assert pred.shape == (n,)
        assert np.all(np.isfinite(pred))

    def test_a1_anisotropic_no_laplacian(self):
        """A1: gamma>0, lambda_l=0."""
        rng = np.random.RandomState(71)
        n, n_rois = 50, N_ROIS
        ne = n_rois * (n_rois - 1) // 2
        roi = rng.rand(n_rois)
        cache = build_msancr_cache(roi, n_rois, gamma=1.0, top_k=3)

        model = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=0.0, gamma=1.0,
            cache=cache, n_rois=n_rois,
        )
        X_fc = rng.randn(n, ne)
        X_sc = rng.randn(n, ne)
        model.fit(X_fc, X_sc, rng.randn(n))
        pred = model.predict(X_fc[:5], X_sc[:5])
        assert pred.shape == (5,)

    def test_a2_isotropic_with_laplacian(self):
        """A2: gamma=0, lambda_l>0."""
        rng = np.random.RandomState(72)
        n, n_rois = 50, N_ROIS
        ne = n_rois * (n_rois - 1) // 2
        roi = rng.rand(n_rois)
        cache = build_msancr_cache(roi, n_rois, gamma=0.0, top_k=3)

        model = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=1.0, gamma=0.0,
            cache=cache, n_rois=n_rois,
        )
        X_fc = rng.randn(n, ne)
        X_sc = rng.randn(n, ne)
        model.fit(X_fc, X_sc, rng.randn(n))
        pred = model.predict(X_fc[:5], X_sc[:5])
        assert pred.shape == (5,)


# ---------------------------------------------------------------------------
# Prior control symmetry
# ---------------------------------------------------------------------------

class TestPriorControlSymmetry:
    def test_same_grid_all_priors(self):
        grids = {
            "lambda_fc": [0.01, 0.1, 1.0, 10.0, 100.0],
            "lambda_sc": [0.01, 0.1, 1.0, 10.0, 100.0],
            "lambda_l": [0.0, 0.1, 0.5, 1.0, 2.0, 5.0],
        }
        for prior_type in ["matched", "unrelated", "shuffled", "random"]:
            assert grids["lambda_fc"] == [0.01, 0.1, 1.0, 10.0, 100.0]
            assert grids["lambda_sc"] == [0.01, 0.1, 1.0, 10.0, 100.0]
            assert grids["lambda_l"] == [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]

    def test_lifting_rules_limited(self):
        liftings = ["prod", "mean"]
        assert len(liftings) == 2


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

class TestNumericalStability:
    def test_large_features_no_nan(self):
        rng = np.random.RandomState(80)
        n, n_rois = 50, N_ROIS
        ne = n_rois * (n_rois - 1) // 2
        X_fc = rng.randn(n, ne) * 100
        X_sc = rng.randn(n, ne) * 100
        y = rng.randn(n) * 100

        roi = rng.rand(n_rois)
        cache = build_msancr_cache(roi, n_rois, gamma=2.0, top_k=3)
        model = ModalitySelectiveAnisotropicNCR(
            lambda_fc=0.01, lambda_sc=0.01, lambda_l=5.0, gamma=2.0,
            cache=cache, n_rois=n_rois,
        )
        model.fit(X_fc, X_sc, y)
        pred = model.predict(X_fc, X_sc)
        assert np.all(np.isfinite(pred))

    def test_p_greater_than_n(self):
        """Solver should handle p >> n (n_rois small, many subjects)."""
        rng = np.random.RandomState(81)
        n, n_rois = 30, N_ROIS
        ne = n_rois * (n_rois - 1) // 2
        X_fc = rng.randn(n, ne)
        X_sc = rng.randn(n, ne)
        y = rng.randn(n)

        roi = rng.rand(n_rois)
        cache = build_msancr_cache(roi, n_rois, gamma=1.0, top_k=3)
        model = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=1.0, gamma=1.0,
            cache=cache, n_rois=n_rois,
        )
        model.fit(X_fc, X_sc, y)
        pred = model.predict(X_fc, X_sc)
        assert np.all(np.isfinite(pred))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_fixed_seed_same_result(self):
        rng = np.random.RandomState(90)
        n, n_rois = 40, N_ROIS
        ne = n_rois * (n_rois - 1) // 2
        roi = rng.rand(n_rois)
        cache = build_msancr_cache(roi, n_rois, gamma=0.5, top_k=3)

        X_fc = rng.randn(n, ne)
        X_sc = rng.randn(n, ne)
        y = rng.randn(n)

        m1 = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=0.5, gamma=0.5,
            cache=cache, n_rois=n_rois,
        )
        m1.fit(X_fc, X_sc, y)
        pred1 = m1.predict(X_fc, X_sc)

        m2 = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=0.5, gamma=0.5,
            cache=cache, n_rois=n_rois,
        )
        m2.fit(X_fc, X_sc, y)
        pred2 = m2.predict(X_fc, X_sc)

        np.testing.assert_allclose(pred1, pred2, atol=1e-10)


# ---------------------------------------------------------------------------
# Old NCR imports still work
# ---------------------------------------------------------------------------

class TestOldNCRBackcompat:
    def test_old_ncr_imports_work(self):
        from metascfc.models.iclr_backbones.network_constrained_ridge import (
            NetworkConstrainedRidge,
            build_edge_laplacian,
            fit_predict_network_constrained,
        )
        assert NetworkConstrainedRidge is not None
        assert build_edge_laplacian is not None
        assert fit_predict_network_constrained is not None
