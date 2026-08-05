"""Unit tests for Method 2 (Prior-Gated Graph Attention Network, Meta-GAT).

Verifies the prior gate (attention shifts toward prior-active ROIs), the
stable segment softmax (no overflow for large logits), the self-loop edge
case for isolated ROIs, the split-graph construction, the gradient-based
node saliency biomarker, and the full nested-CV protocol (leakage-free,
deterministic).
"""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from metascfc.models.iclr_backbones import (
    MetaGAT,
    MetaGATConfig,
    PriorGatedGATLayer,
    build_candidate_grid,
    build_split_graph,
    fit_predict_meta_gat,
    gradient_node_saliency,
)


def make_toy(
    n_subjects: int = 40, n_rois: int = 8, seed: int = 0,
) -> tuple:
    rng = np.random.default_rng(seed)
    fc = rng.standard_normal((n_subjects, n_rois, n_rois))
    sc = np.abs(rng.standard_normal((n_subjects, n_rois, n_rois)))
    sc = 0.5 * (sc + sc.transpose(0, 2, 1))
    y = fc.sum(axis=(1, 2)) + sc.sum(axis=(1, 2))
    return fc, sc, y, n_rois


def test_prior_gate_biases_attention():
    """A strong prior must force node 1 to route through prior-active node 0.

    Two-node complete graph (self-loops included), prior = [1, 0]: with a
    large gate, edge (0 -> 1) dominates node 1's incoming attention, so
    node 1's output collapses to ELU(W x_0) - the message of the
    prior-active region.  Without the gate it stays the pooled average.
    """
    n_nodes, in_dim, out_dim = 2, 4, 6
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((1, n_nodes, in_dim)).astype(np.float32))
    prior = np.array([1.0, 0.0])
    edge_src = np.array([0, 1, 0, 1], dtype=np.int64)
    edge_dst = np.array([0, 1, 1, 0], dtype=np.int64)

    def node1_self_error(gamma: float) -> float:
        layer = PriorGatedGATLayer(
            in_dim, out_dim, heads=1, n_nodes=n_nodes, prior=prior,
            edge_src=edge_src, edge_dst=edge_dst, gamma_init=gamma,
        )
        with torch.no_grad():
            layer.gamma.fill_(gamma)
            layer.eval()
            out = layer(x)[0, 1]  # node 1 output
            target = F.elu(x[0, 0] @ layer.W[0]).detach()
            assert torch.isfinite(out).all()
        return float(torch.abs(out - target).mean())

    err_off = node1_self_error(0.0)
    err_on = node1_self_error(12.0)
    assert err_on < 0.5 * err_off


def test_single_self_loop_closed_form():
    """A one-node graph has a single incoming edge: alpha = 1 exactly."""
    n_nodes, in_dim, out_dim = 1, 3, 5
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.standard_normal((2, n_nodes, in_dim)).astype(np.float32))
    layer = PriorGatedGATLayer(
        in_dim, out_dim, heads=2, n_nodes=n_nodes, prior=np.array([0.7]),
        edge_src=np.array([0]), edge_dst=np.array([0]), gamma_init=0.0,
    )
    with torch.no_grad():
        out = layer(x)
        # alpha = 1 -> out = concat_h ELU(W_h x).
        expected = torch.cat([F.elu(x @ layer.W[h]) for h in range(2)], dim=-1)
    assert np.allclose(out.numpy(), expected.numpy(), atol=1e-5)


def test_isolated_node_self_loop_finite():
    """Isolated ROIs (no structural edge) must still propagate: self-loop."""
    n_nodes, in_dim, out_dim = 6, 4, 5
    rng = np.random.default_rng(2)
    x = torch.from_numpy(rng.standard_normal((3, n_nodes, in_dim)).astype(np.float32))
    prior = rng.uniform(0.0, 1.0, size=n_nodes).astype(np.float32)
    # Edges only among {0, 1, 2}; nodes 3..5 are isolated (self-loops only).
    edge_src = np.array([0, 1, 1, 2, 3, 4, 5], dtype=np.int64)
    edge_dst = np.array([1, 0, 2, 1, 3, 4, 5], dtype=np.int64)
    layer = PriorGatedGATLayer(
        in_dim, out_dim, heads=3, n_nodes=n_nodes, prior=prior,
        edge_src=edge_src, edge_dst=edge_dst, gamma_init=10.0,
    )
    out = layer(x)
    assert torch.isfinite(out).all()


def test_build_split_graph_structure():
    fc, sc, y, n_rois = make_toy(n_subjects=12, n_rois=6)
    src, dst = build_split_graph(sc, top_percent=10.0)
    assert src.dtype == np.int64 and dst.dtype == np.int64
    assert src.shape == dst.shape and len(src) >= n_rois
    # Self-loops for every ROI.
    assert set(np.arange(n_rois)) <= set(src) and set(np.arange(n_rois)) <= set(dst)
    # Every edge is covered by the node set.
    assert src.max() < n_rois and dst.max() < n_rois
    # Deterministic.
    src2, dst2 = build_split_graph(sc, top_percent=10.0)
    assert np.array_equal(src, src2) and np.array_equal(dst, dst2)


def test_build_candidate_grid_cartesian():
    grid = build_candidate_grid([16, 32], [0.2, 0.5], [1e-3, 3e-3], epochs=10)
    assert len(grid) == 8
    assert {(c.hidden, c.dropout, c.learning_rate) for c in grid} == {
        (16, 0.2, 1e-3), (16, 0.2, 3e-3), (16, 0.5, 1e-3), (16, 0.5, 3e-3),
        (32, 0.2, 1e-3), (32, 0.2, 3e-3), (32, 0.5, 1e-3), (32, 0.5, 3e-3),
    }
    assert all(c.epochs == 10 for c in grid)


def test_meta_gat_fits_linear_signal():
    """The GAT must learn a target that is linear in the pooled features."""
    fc, sc, y, n_rois = make_toy(n_subjects=40, n_rois=6)
    y = y - y.mean()
    y = y / (y.std() + 1e-12)
    train_idx = np.arange(0, 30)
    x = np.concatenate([fc, sc], axis=2).astype(np.float32)
    x_flat = x[train_idx].reshape(len(train_idx), -1)
    mu, sd = x_flat.mean(axis=0), x_flat.std(axis=0)
    sd[sd < 1e-8] = 1.0
    x_t = torch.from_numpy(
        ((x.reshape(len(x), -1) - mu) / sd).reshape(x.shape).astype(np.float32)
    )
    y_t = torch.from_numpy(y.astype(np.float32))
    prior = np.ones(n_rois)
    prior[:3] = 2.0
    prior = (prior - prior.min()) / (prior.max() - prior.min())
    src, dst = build_split_graph(sc, top_percent=10.0)
    cfg = MetaGATConfig(hidden=16, heads1=2, heads2=1, dropout=0.0,
                        learning_rate=0.01, weight_decay=0.0, epochs=150,
                        patience=40, min_epochs=20, gamma_init=1.0)
    torch.manual_seed(0)
    model = MetaGAT(n_rois, x.shape[2], cfg, prior, src, dst)
    from metascfc.models.iclr_backbones.meta_gat import _train_fixed_epochs
    _train_fixed_epochs(model, x_t[train_idx], y_t[train_idx], cfg, 150, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        pred = model(x_t[train_idx]).numpy()
    r = np.corrcoef(pred, y[train_idx])[0, 1]
    assert r > 0.75


def test_fit_predict_meta_gat_protocol_and_determinism():
    fc, sc, y, n_rois = make_toy(n_subjects=50, n_rois=8)
    prior = np.ones(n_rois)
    prior[:3] = 2.0
    prior = (prior - prior.min()) / (prior.max() - prior.min())
    train_idx = np.arange(0, 25)
    val_idx = np.arange(25, 35)
    test_idx = np.arange(35, 50)
    kw = dict(
        fc=fc, sc=sc, y=y, train_idx=train_idx, val_idx=val_idx,
        test_idx=test_idx, prior=prior,
        hidden_grid=[8, 16], dropout_grid=[0.2], lr_grid=[0.01],
        device=torch.device("cpu"), top_percent=10.0, heads1=2, heads2=1,
        weight_decay=0.0, epochs=25, patience=8, min_epochs=4,
        gamma_init=1.0, grad_clip=5.0, seed=7,
    )
    pred1, cfg1, val_rmse1, epoch1, saliency1, params1 = fit_predict_meta_gat(**kw)
    assert pred1.shape == (15,)
    assert np.isfinite(pred1).all()
    assert cfg1["hidden"] in (8, 16) and cfg1["dropout"] == 0.2
    assert val_rmse1 >= 0.0 and epoch1 >= 1
    assert saliency1.shape == (n_rois,)
    assert np.isfinite(saliency1).all()
    assert saliency1.min() >= 0.0 and saliency1.max() <= 1.0 + 1e-12
    assert params1 > 0
    # Determinism: the same seed reproduces identical predictions.
    pred2, _, _, _, _, _ = fit_predict_meta_gat(**kw)
    assert np.allclose(pred1, pred2, atol=1e-6)


def test_gradient_saliency_shape_and_range():
    fc, sc, y, n_rois = make_toy(n_subjects=10, n_rois=5)
    x = torch.from_numpy(np.concatenate([fc, sc], axis=2).astype(np.float32))
    prior = np.linspace(0.0, 1.0, n_rois)
    src, dst = build_split_graph(sc, top_percent=10.0)
    cfg = MetaGATConfig(hidden=8, heads1=2, heads2=1, dropout=0.0,
                        learning_rate=1e-2, weight_decay=0.0, epochs=10,
                        patience=5, min_epochs=2, gamma_init=1.0)
    model = MetaGAT(n_rois, x.shape[2], cfg, prior, src, dst)
    sal = gradient_node_saliency(model, x, torch.device("cpu"))
    assert sal.shape == (n_rois,)
    assert np.isfinite(sal).all()
    assert sal.min() >= 0.0 and sal.max() <= 1.0 + 1e-12


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
