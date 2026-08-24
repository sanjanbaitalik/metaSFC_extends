"""Unit tests for Method 4 (LLM-Gated Cross-Modal Graph Attention Transformer).

Verifies the LLM-prior gate (attention shifts toward prior-active ROIs), the
stable segment softmax under extreme logits, residual/LayerNorm wiring of
the transformer blocks, the gradient saliency biomarker contract, and the
nested-CV entry point (leakage-free, deterministic) plus the faithfulness
refit predictor.
"""
import numpy as np
import pytest
import torch

from metascfc.models.iclr_backbones import (
    LLMPriorGatedGATLayer,
    LLMGatedConfig,
    LLMGatedTransformer,
    build_llm_gated_grid,
    fit_predict_llm_gated,
    llm_gated_node_saliency,
    refit_llm_gated_predictor,
)


def make_toy(n_subjects: int = 40, n_rois: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    fc = rng.standard_normal((n_subjects, n_rois, n_rois))
    sc = np.abs(rng.standard_normal((n_subjects, n_rois, n_rois)))
    sc = 0.5 * (sc + sc.transpose(0, 2, 1))
    y = fc.sum(axis=(1, 2)) + sc.sum(axis=(1, 2))
    return fc, sc, y, n_rois


def _toy_edges(n_nodes: int = 2):
    edge_src = np.array([0, 1, 0, 1], dtype=np.int64)
    edge_dst = np.array([0, 1, 1, 0], dtype=np.int64)
    return edge_src, edge_dst


def test_prior_gate_biases_attention():
    """At alpha -> 1 the routing becomes purely prior-driven and the node
    output is analytically determined by the prior-gate softmax."""
    n_nodes, in_dim, out_dim = 2, 4, 6
    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.standard_normal((1, n_nodes, in_dim)).astype(np.float32))
    prior = np.array([1.0, 0.0])
    edge_src, edge_dst = _toy_edges()
    half = in_dim // 2

    def make_layer(alpha_init: float) -> LLMPriorGatedGATLayer:
        torch.manual_seed(11)
        return LLMPriorGatedGATLayer(
            in_dim, out_dim, heads=1, n_nodes=n_nodes, prior=prior,
            edge_src=edge_src, edge_dst=edge_dst, alpha_init=alpha_init,
        )

    # Same seed -> identical learned weights; only the bypass differs.
    strong_layer, weak_layer = make_layer(0.999), make_layer(0.001)
    assert torch.allclose(strong_layer.W_fc, weak_layer.W_fc)
    with torch.no_grad():
        strong_layer.eval()
        weak_layer.eval()
        strong = strong_layer(x)[0, 1]
        weak = weak_layer(x)[0, 1]
    assert not torch.allclose(strong, weak)

    # alpha -> 1: incoming attention of node 1 = softmax over prior gates
    # {edge 0->1: p0+p1 = 1, edge 1->1: p1+p1 = 0} = softmax([1, 0]).
    w = torch.sigmoid(strong_layer.rho).detach()
    assert w > 0.99
    attn = torch.softmax(torch.tensor([w.item() * 1.0, w.item() * 0.0]), dim=0)
    proj = lambda xi: x[0, xi, :half] @ strong_layer.W_fc[0] \
        + x[0, xi, half:] @ strong_layer.W_sc[0]
    target = torch.nn.functional.elu(
        attn[0] * proj(0) + attn[1] * proj(1)
    ).detach()
    assert torch.allclose(strong, target, atol=1e-4)


def test_bypass_alpha_is_learnable_and_bounded():
    layer = LLMPriorGatedGATLayer(
        4, 4, heads=1, n_nodes=2, prior=np.ones(2),
        edge_src=_toy_edges()[0], edge_dst=_toy_edges()[1], alpha_init=0.7,
    )
    assert isinstance(layer.rho, torch.nn.Parameter)
    init = float(torch.sigmoid(layer.rho))
    assert init == pytest.approx(0.7, abs=1e-5)
    with torch.no_grad():
        layer.rho.fill_(3.0)
    assert layer.bypass_alpha == pytest.approx(1.0 / (1.0 + np.exp(-3.0)), abs=1e-5)
    with torch.no_grad():
        layer.rho.fill_(-8.0)
    assert layer.bypass_alpha < 1e-3  # mismatch regime: prior ignored


def test_stable_softmax_under_extreme_logits():
    """Huge features must not overflow the segment softmax."""
    n_nodes, in_dim, out_dim = 4, 6, 4
    rng = np.random.default_rng(1)
    x = torch.from_numpy(
        (rng.standard_normal((2, n_nodes, in_dim)) * 300).astype(np.float32)
    )
    layer = LLMPriorGatedGATLayer(
        in_dim, out_dim, heads=2, n_nodes=n_nodes, prior=rng.random(n_nodes),
        edge_src=np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64),
        edge_dst=np.array([0, 1, 2, 3, 1, 0, 3, 2], dtype=np.int64),
    )
    layer.eval()
    out = layer(x)
    assert out.shape == (2, n_nodes, out_dim * 2)
    assert torch.isfinite(out).all()
    attn = layer._last_attention
    assert torch.isfinite(attn).all()


def test_transformer_residual_shapes_and_depth():
    cfg = LLMGatedConfig(hidden=4, n_layers=3, heads=3, dropout=0.0)
    n_nodes, in_dim = 5, 8
    model = LLMGatedTransformer(
        n_nodes, in_dim, cfg, np.linspace(0, 1, n_nodes),
        np.arange(n_nodes * 2, dtype=np.int64) % n_nodes,
        np.tile(np.arange(n_nodes, dtype=np.int64), 2),
    )
    x = torch.randn(2, n_nodes, in_dim)
    model.eval()
    with torch.no_grad():
        pred = model(x)
    assert pred.shape == (2,)
    assert torch.isfinite(pred).all()
    d_model = cfg.hidden * cfg.heads
    assert model.input_proj.out_features == d_model
    # Every block preserves width -> residual connections are active.
    for block in model.blocks:
        assert block.norm1.normalized_shape[0] == d_model


def test_candidate_grid_cartesian_product():
    grid = build_llm_gated_grid(
        [8, 16], [0.2], [1e-3], n_layers=2, heads=4, weight_decay=1e-4,
        epochs=5, patience=2, min_epochs=1, alpha_init=0.5, grad_clip=5.0,
    )
    assert len(grid) == 2
    assert {c.hidden for c in grid} == {8, 16}


def test_fit_predict_deterministic_and_leakage_free():
    fc, sc, y, n_rois = make_toy()
    device = torch.device("cpu")
    prior = np.linspace(0.0, 1.0, n_rois)
    idx = np.arange(len(y))
    train_idx, val_idx, test_idx = idx[:24], idx[24:32], idx[32:]

    kwargs = dict(
        hidden_grid=[8], dropout_grid=[0.0], lr_grid=[1e-3], device=device,
        n_layers=2, heads=2, epochs=4, min_epochs=2, patience=10,
        top_percent=25.0,
    )
    pred_a, cfg_a, rmse_a, epoch_a, sal_a, n_params_a = fit_predict_llm_gated(
        fc, sc, y, train_idx, val_idx, test_idx, prior, seed=7, **kwargs
    )
    pred_b, cfg_b, rmse_b, epoch_b, sal_b, n_params_b = fit_predict_llm_gated(
        fc, sc, y, train_idx, val_idx, test_idx, prior, seed=7, **kwargs
    )
    assert np.allclose(pred_a, pred_b)
    assert cfg_a == cfg_b and rmse_a == pytest.approx(rmse_b)
    assert epoch_a == epoch_b and n_params_a == n_params_b

    assert pred_a.shape == (len(test_idx),)
    assert np.isfinite(pred_a).all()
    assert sal_a.shape == (n_rois,)
    assert sal_a.min() >= -1e-9 and abs(sal_a.max() - 1.0) < 1e-9
    assert n_params_a > 0


def test_fit_predict_logs_information_bottleneck():
    """The IB tracker records per-epoch metrics on the refit and converged
    values plus the learned bypass alphas at the end."""
    from metascfc.metrics import IBEpochTracker

    fc, sc, y, n_rois = make_toy()
    device = torch.device("cpu")
    prior = np.linspace(0.0, 1.0, n_rois)
    idx = np.arange(len(y))
    tracker = IBEpochTracker(noise_floor=0.05)
    fit_predict_llm_gated(
        fc, sc, y, idx[:24], idx[24:32], idx[32:], prior,
        hidden_grid=[8], dropout_grid=[0.0], lr_grid=[1e-3], device=device,
        n_layers=2, heads=2, epochs=4, min_epochs=2, patience=10,
        top_percent=25.0, seed=5, ib_tracker=tracker,
    )
    assert len(tracker.epochs) > 0
    assert len(tracker.I_XZ) == len(tracker.epochs) == len(tracker.I_ZY)
    assert all(np.isfinite(tracker.I_XZ)) and all(v > 0 for v in tracker.I_XZ)
    assert tracker.final is not None
    assert set(tracker.final) == {"I_XZ", "I_ZY", "probe_r2"}
    assert np.isfinite(tracker.final["probe_r2"])
    assert len(tracker.alpha_final) == 2  # one bypass per layer
    assert all(0.0 < a < 1.0 for a in tracker.alpha_final)


def test_refit_predictor_predict_and_attention_mass():
    fc, sc, y, n_rois = make_toy(n_subjects=30)
    device = torch.device("cpu")
    prior = np.linspace(0.0, 1.0, n_rois)
    fit_idx = np.arange(20)
    cfg = LLMGatedConfig(hidden=4, n_layers=1, heads=2, dropout=0.0,
                         epochs=3, min_epochs=1, patience=2)
    predictor = refit_llm_gated_predictor(
        fc, sc, y, fit_idx, cfg, prior, device,
        n_epochs=3, top_percent=25.0, seed=3,
    )
    pred = predictor.predict(fc[20:], sc[20:])
    assert pred.shape == (10,) and np.isfinite(pred).all()
    mass = predictor.attention_mass(fc[:5], sc[:5])
    assert mass.shape == (n_rois,)
    assert np.isfinite(mass).all()
    assert mass.sum() == pytest.approx(1.0, abs=1e-6)
    assert (mass >= 0).all()


def test_saliency_helper_matches_entry_point_contract():
    fc, sc, y, n_rois = make_toy(n_subjects=12)
    cfg = LLMGatedConfig(hidden=4, n_layers=1, heads=2, dropout=0.0,
                         epochs=2, min_epochs=1, patience=2)
    from metascfc.models.iclr_backbones.meta_gat import build_split_graph
    x = torch.randn(6, n_rois, 2 * n_rois)
    edges = build_split_graph(sc[:6].astype(np.float32), top_percent=25.0)
    model = LLMGatedTransformer(n_rois, 2 * n_rois, cfg,
                                np.linspace(0, 1, n_rois), *edges)
    sal = llm_gated_node_saliency(model, x, torch.device("cpu"))
    assert sal.shape == (n_rois,) and np.isfinite(sal).all()
