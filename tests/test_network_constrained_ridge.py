"""Unit tests for Method 1 (Network-Constrained Prior-Laplacian Ridge).

Verifies the solver against the closed-form primal solution, the λ2 = 0
degeneration to plain Ridge, the structure of the prior Laplacian
(top-k thresholding, PSD-ness), and edge cases (isolated line-graph nodes).
"""
import numpy as np
import pytest
from scipy import sparse
from sklearn.preprocessing import StandardScaler

from metascfc.models.iclr_backbones import (
    NetworkConstrainedRidge,
    build_edge_laplacian,
    build_prior_adjacency,
    fit_predict_network_constrained,
    node_saliency_from_beta,
)


def make_toy(n_subjects: int = 40, n_rois: int = 12, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    n_edges = int(n_rois * (n_rois - 1) / 2)
    fc = rng.standard_normal((n_subjects, n_rois, n_rois))
    sc = rng.standard_normal((n_subjects, n_rois, n_rois))
    fc = 0.5 * (fc + fc.transpose(0, 2, 1))
    sc = 0.5 * (sc + sc.transpose(0, 2, 1))
    iu = np.triu_indices(n_rois, k=1)
    x = np.concatenate([fc[:, iu[0], iu[1]], sc[:, iu[0], iu[1]]], axis=1)
    true_beta = np.zeros(2 * n_edges)
    true_beta[:n_edges:7] = 0.5
    true_beta[n_edges::7] = -0.3
    y = x @ true_beta + 0.1 * rng.standard_normal(n_subjects)
    return x, y, n_rois


def test_prior_adjacency_is_complete_graph_on_top_k():
    rng = np.random.default_rng(1)
    prior = rng.uniform(0.0, 1.0, size=20)
    a = build_prior_adjacency(prior, top_k=5)
    assert a.shape == (20, 20)
    assert np.allclose(a, a.T)
    assert np.all(np.diag(a) == 0.0)
    active = np.where(a.sum(axis=0) > 0)[0]
    assert len(active) == 5
    assert np.allclose(a[np.ix_(active, active)], 1.0 - np.eye(5))


def test_edge_laplacian_structure_counts():
    n_rois = 116
    rng = np.random.default_rng(0)
    prior = rng.uniform(0.0, 1.0, size=n_rois)
    lap = build_edge_laplacian(n_rois=n_rois, prior_scores=prior, top_k=30)
    n_edges = int(n_rois * (n_rois - 1) / 2)
    assert lap.n_edges == n_edges
    # Active edges: within the 30-node clique (C(30,2)) plus edges between
    # the clique and the remaining 86 ROIs (30 x 86).
    assert lap.n_active == 435 + 30 * (n_rois - 30)
    assert lap.active_indices.dtype == np.int64
    # PSD: smallest eigenvalue of the Laplacian block must be ~ 0.
    assert np.linalg.eigvalsh(lap.active_laplacian)[0] > -1e-8
    # Full penalty matrix is sparse and block structured.
    full = lap.full_laplacian()
    assert isinstance(full, sparse.csr_matrix)
    assert full.shape == (2 * n_edges, 2 * n_edges)


def test_edge_laplacian_isolated_node_edge_case():
    # top_k=1: a single prior-active ROI has no active edges -> empty block.
    prior = np.zeros(8)
    prior[3] = 1.0
    lap = build_edge_laplacian(n_rois=8, prior_scores=prior, top_k=1)
    assert lap.n_active == 0
    assert lap.active_laplacian.shape == (0, 0)
    # All-zeros prior with a large top_k must not crash either.
    lap2 = build_edge_laplacian(n_rois=8, prior_scores=np.zeros(8), top_k=8)
    assert lap2.n_active == 8 * 7 // 2


@pytest.mark.parametrize("alpha1, alpha2", [(3.0, 0.0), (2.0, 1.5), (0.5, 5.0)])
def test_solver_matches_closed_form_primal(alpha1, alpha2):
    """The dual solver must reproduce the closed-form primal solution.

    The effective primal penalty is P = s·(λ1·I + λ2·L_full) with
    s = max(1, n_features), matching the dual-Ridge kernel convention of
    the AAAI baselines.  λ2 = 0 covers the plain Ridge degeneration.
    """
    rng = np.random.default_rng(2)
    n, n_rois = 30, 8
    n_edges = int(n_rois * (n_rois - 1) / 2)
    x = rng.standard_normal((n, 2 * n_edges))
    y = rng.standard_normal(n)
    prior = rng.uniform(0.0, 1.0, size=n_rois)
    lap = build_edge_laplacian(n_rois=n_rois, prior_scores=prior, top_k=4)
    model = NetworkConstrainedRidge(
        alpha1=alpha1, alpha2=alpha2, edge_laplacian=lap,
        n_rois=n_rois, standardize=False,
    )
    model.fit(x, y)
    y_z = (y - y.mean()) / y.std()
    scale = float(max(1, x.shape[1]))
    l_full = lap.full_laplacian().toarray()
    beta_ref = np.linalg.solve(
        x.T @ x + scale * (alpha1 * np.eye(2 * n_edges) + alpha2 * l_full),
        x.T @ y_z,
    )
    assert np.allclose(model.beta(), beta_ref, atol=1e-8)
    pred_ref = x @ beta_ref * y.std() + y.mean()
    assert np.allclose(model.predict(x), pred_ref, atol=1e-8)


def test_nested_fit_predict_matches_closed_form():
    """Nested selection + refit must equal the closed-form primal solution
    evaluated at the selected (λ1, λ2) on the train+val refit split."""
    x, y, n_rois = make_toy(n_subjects=50)
    prior = np.ones(n_rois)
    prior[:3] = 2.0
    lap = build_edge_laplacian(n_rois=n_rois, prior_scores=prior, top_k=3)
    n_subjects = len(y)
    train_idx = np.arange(0, 25)
    val_idx = np.arange(25, 35)
    test_idx = np.arange(35, n_subjects)
    pred, a1, a2, val_rmse, beta_dev = fit_predict_network_constrained(
        x, y, train_idx, val_idx, test_idx, lap,
        alpha1_grid=[0.1, 1.0, 10.0], alpha2_grid=[0.0, 1.0],
    )
    assert pred.shape == (n_subjects - 35,)
    assert np.isfinite(pred).all()
    assert a1 in (0.1, 1.0, 10.0)
    assert a2 in (0.0, 1.0)
    assert isinstance(val_rmse, float) and val_rmse >= 0.0

    fit_idx = np.concatenate([train_idx, val_idx])
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x[fit_idx])
    x_test = scaler.transform(x[test_idx])
    y_fit = y[fit_idx]
    m, s = float(y_fit.mean()), float(y_fit.std())
    scale = float(max(1, x.shape[1]))
    l_full = lap.full_laplacian().toarray()
    beta_ref = np.linalg.solve(
        x_fit.T @ x_fit + scale * (a1 * np.eye(x.shape[1]) + a2 * l_full),
        x_fit.T @ ((y_fit - m) / s),
    )
    pred_ref = x_test @ beta_ref * s + m
    assert np.allclose(pred, pred_ref, atol=1e-6)
    assert np.allclose(beta_dev, beta_ref, atol=1e-6)


def test_nested_fit_is_leakage_free_and_returns_saliency():
    """With λ2 = 0 the selected model must match plain ridge on the test set."""
    x, y, n_rois = make_toy(n_subjects=50)
    prior = np.ones(n_rois)
    lap = build_edge_laplacian(n_rois=n_rois, prior_scores=prior, top_k=5)
    train_idx = np.arange(0, 25)
    val_idx = np.arange(25, 35)
    test_idx = np.arange(35, 50)
    pred, a1, a2, val_rmse, beta_dev = fit_predict_network_constrained(
        x, y, train_idx, val_idx, test_idx, lap,
        alpha1_grid=[0.1, 1.0, 10.0], alpha2_grid=[0.0, 1.0],
    )
    assert pred.shape == (15,)
    assert np.isfinite(pred).all()
    assert a1 in (0.1, 1.0, 10.0)
    assert a2 in (0.0, 1.0)
    assert isinstance(val_rmse, float) and val_rmse >= 0.0
    sal = node_saliency_from_beta(beta_dev, n_rois)
    assert sal.shape == (n_rois,)
    assert np.isfinite(sal).all()
    assert 0.0 <= sal.min() and sal.max() <= 1.0 + 1e-12


def test_node_saliency_aggregates_both_modalities():
    n_rois = 6
    n_edges = 15
    beta = np.zeros(2 * n_edges)
    beta[0] = 1.0  # FC edge (0, 1)
    beta[n_edges + 5] = 2.0  # SC edge (1, 2)
    sal = node_saliency_from_beta(beta, n_rois)
    assert sal.shape == (n_rois,)
    assert sal[0] > 0.0 and sal[1] > 0.5 and sal[2] > 0.0
    assert np.isclose(sal.max(), 1.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
