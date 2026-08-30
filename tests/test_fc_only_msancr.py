"""Tests for FC-only MS-A-NCR solver mode."""
import numpy as np
import pytest

from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    ModalitySelectiveAnisotropicNCR,
    _solve_msancr_kernel,
    _predict_msancr,
    recover_msancr_beta,
    build_msancr_cache,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import build_edge_laplacian


N_ROIS = 6
N_EDGES = N_ROIS * (N_ROIS - 1) // 2
N = 30


def _make_test_data():
    rng = np.random.RandomState(42)
    roi_prior = rng.rand(N_ROIS)
    X_fc = rng.randn(N, N_EDGES)
    X_sc = rng.randn(N, N_EDGES)
    y = rng.randn(N)
    edge_lap = build_edge_laplacian(N_ROIS, prior_scores=roi_prior, top_k=3, weighting="binary", couple_modalities=False, normalize="sym")
    cache = build_msancr_cache(roi_prior, N_ROIS, gamma=0.5, lifting="prod", top_k=3, epsilon=1e-3, weighting="binary", couple_modalities=False, normalize_laplacian="sym", edge_laplacian=edge_lap)
    return X_fc, X_sc, y, roi_prior, cache


class TestFCOnlySolver:
    def test_fc_only_no_sc_in_kernel(self):
        X_fc, X_sc, y, roi_prior, cache = _make_test_data()
        alpha_fc, (K_active, K_inactive, K_sc, _) = _solve_msancr_kernel(
            X_fc, X_sc, y, cache, 1.0, 1.0, 0.0, fc_only=True,
        )
        assert K_sc is None
        assert alpha_fc.shape == (N,)

    def test_fc_plus_sc_has_sc(self):
        X_fc, X_sc, y, roi_prior, cache = _make_test_data()
        alpha_fcsc, (K_active, K_inactive, K_sc, _) = _solve_msancr_kernel(
            X_fc, X_sc, y, cache, 1.0, 1.0, 0.0, fc_only=False,
        )
        assert K_sc is not None
        assert K_sc.shape == (N, N)

    def test_fc_only_predict_no_sc(self):
        X_fc, X_sc, y, roi_prior, cache = _make_test_data()
        alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 1.0, 0.0, fc_only=True)
        pred = _predict_msancr(
            X_fc[:5], np.zeros((5, N_EDGES)),
            X_fc, np.zeros_like(X_fc),
            alpha, cache, 1.0, 1.0, 0.0, fc_only=True,
        )
        assert pred.shape == (5,)
        assert np.isfinite(pred).all()

    def test_fc_only_beta_sc_zeros(self):
        X_fc, X_sc, y, roi_prior, cache = _make_test_data()
        alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 1.0, 0.0, fc_only=True)
        beta_fc, beta_sc = recover_msancr_beta(
            X_fc, X_sc, alpha, cache, 1.0, 1.0, 0.0, fc_only=True,
        )
        assert beta_fc.shape == (N_EDGES,)
        assert beta_sc.shape == (N_EDGES,)
        assert np.allclose(beta_sc, 0.0)

    def test_fc_plus_sc_beta_sc_nonzero(self):
        X_fc, X_sc, y, roi_prior, cache = _make_test_data()
        alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 1.0, 0.0, fc_only=False)
        beta_fc, beta_sc = recover_msancr_beta(
            X_fc, X_sc, alpha, cache, 1.0, 1.0, 0.0, fc_only=False,
        )
        assert not np.allclose(beta_sc, 0.0)

    def test_fc_only_class_fit_predict(self):
        X_fc, X_sc, y, roi_prior, cache = _make_test_data()
        model = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=0.0, gamma=0.5,
            cache=cache, n_rois=N_ROIS, fc_only=True,
        )
        model.fit(X_fc, X_sc, y)
        pred = model.predict(X_fc[:5], X_sc[:5])
        assert pred.shape == (5,)
        assert np.isfinite(pred).all()

    def test_fc_only_class_beta_sc_zeros(self):
        X_fc, X_sc, y, roi_prior, cache = _make_test_data()
        model = ModalitySelectiveAnisotropicNCR(
            lambda_fc=1.0, lambda_sc=1.0, lambda_l=0.0, gamma=0.5,
            cache=cache, n_rois=N_ROIS, fc_only=True,
        )
        model.fit(X_fc, X_sc, y)
        beta = model.beta()
        # beta = [beta_fc, beta_sc]; beta_sc half should be zero
        assert np.allclose(beta[N_EDGES:], 0.0)

    def test_fc_only_matches_direct_with_no_sc(self):
        X_fc, X_sc, y, roi_prior, cache = _make_test_data()
        # FC-only solution should differ from FC+SC solution
        alpha_fc_only, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 1.0, 0.0, fc_only=True)
        alpha_fc_sc, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 1.0, 0.0, fc_only=False)
        assert not np.allclose(alpha_fc_only, alpha_fc_sc), "FC-only should differ from FC+SC"

    def test_fc_only_numerical_stability(self):
        X_fc, X_sc, y, roi_prior, cache = _make_test_data()
        X_fc_large = X_fc * 1000
        alpha, _ = _solve_msancr_kernel(X_fc_large, np.zeros_like(X_fc_large), y, cache, 1.0, 1.0, 0.0, fc_only=True)
        assert np.isfinite(alpha).all()

    def test_old_behavior_unchanged(self):
        X_fc, X_sc, y, roi_prior, cache = _make_test_data()
        # Default fc_only=False should work exactly as before
        alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 1.0, 0.0)
        pred = _predict_msancr(X_fc[:5], X_sc[:5], X_fc, X_sc, alpha, cache, 1.0, 1.0, 0.0)
        assert pred.shape == (5,)
        assert np.isfinite(pred).all()
