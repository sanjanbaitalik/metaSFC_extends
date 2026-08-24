"""Unit tests for the Information Bottleneck trackers.

Covers the Gaussian/VIB-proxy estimators (compression I(X;Z), predictive
I(Z;Y)), the epoch tracker, and their integration into both training paths:
per-epoch + converged logging in the LLM-gated transformer and converged
logging in the closed-form network-constrained ridge.
"""
import numpy as np
import pytest
import torch

from metascfc.metrics import (
    IBEpochTracker,
    compression_mi,
    information_bottleneck_metrics,
    predictive_mi,
)


def test_compression_mi_scales_with_latent_variance():
    rng = np.random.default_rng(0)
    flat = rng.normal(0.0, 0.1, size=(200, 4))
    rich = rng.normal(0.0, 10.0, size=(200, 4))
    i_flat = compression_mi(flat)
    i_rich = compression_mi(rich)
    assert i_rich > i_flat >= 0.0
    # Constant latent carries no information about X.
    assert compression_mi(np.ones((50, 3))) == pytest.approx(0.0)


def test_predictive_mi_monotone_in_r2():
    rng = np.random.default_rng(1)
    n = 300
    y = rng.normal(size=n)
    z_good = y[:, None] + 0.1 * rng.normal(size=(n, 1))
    z_noise = rng.normal(size=(n, 1))
    good = predictive_mi(z_good, y)
    noise = predictive_mi(z_noise, y)
    assert good["probe_r2"] > 0.9
    assert good["I_ZY"] > noise["I_ZY"] >= 0.0


def test_information_bottleneck_metrics_keys_and_validation():
    z = np.random.default_rng(2).normal(size=(100, 5))
    y = np.random.default_rng(3).normal(size=100)
    out = information_bottleneck_metrics(z, y)
    assert set(out) == {"I_XZ", "I_ZY", "probe_r2"}
    with pytest.raises(ValueError):
        compression_mi(np.array([np.inf, 0.0, 1.0]))
    with pytest.raises(ValueError):
        predictive_mi(z[:10], y[:11])


def test_epoch_tracker_accumulates():
    tracker = IBEpochTracker(noise_floor=0.02)
    rng = np.random.default_rng(4)
    for epoch in range(3):
        z = rng.normal(scale=1.0 + epoch, size=(40, 6))
        y = rng.normal(size=40)
        tracker.log_epoch(epoch, z, y)
    final = tracker.log_final(rng.normal(size=(40, 6)), rng.normal(size=40))
    assert set(final) == {"I_XZ", "I_ZY", "probe_r2"}
    d = tracker.to_dict()
    assert d["noise_floor"] == pytest.approx(0.02)
    assert d["epochs"] == [0, 1, 2]
    assert len(d["I_XZ"]) == len(d["I_ZY"]) == 3
    assert d["final"] is not None
    # Compression should grow as the latent spread grows across epochs.
    assert d["I_XZ"][2] > d["I_XZ"][0]


def test_network_constrained_ridge_logs_converged_ib():
    """NCR is closed-form; the tracker must still capture the converged
    I(X;Z) / I(Z;Y) of the latent z = X beta."""
    from metascfc.models.iclr_backbones import (
        build_edge_laplacian,
        fit_predict_network_constrained,
    )

    rng = np.random.default_rng(7)
    n_rois, n_subjects = 8, 40
    iu = np.triu_indices(n_rois, k=1)
    fc = rng.standard_normal((n_subjects, n_rois, n_rois))
    sc = np.abs(rng.standard_normal((n_subjects, n_rois, n_rois)))
    x = np.concatenate(
        [fc[:, iu[0], iu[1]], sc[:, iu[0], iu[1]]], axis=1
    ).astype(np.float64)
    y = x @ rng.normal(size=x.shape[1]) * 0.05 + rng.normal(size=n_subjects) * 0.01

    prior = np.linspace(0.0, 1.0, n_rois)
    laplacian = build_edge_laplacian(n_rois=n_rois, prior_scores=prior, top_k=3)
    idx = np.arange(n_subjects)
    tracker = IBEpochTracker()
    pred, a1, a2, val_rmse, beta = fit_predict_network_constrained(
        x, y, idx[:24], idx[24:32], idx[32:], laplacian,
        alpha1_grid=[0.1, 1.0], alpha2_grid=[0.0, 1.0],
        ib_tracker=tracker,
    )
    assert pred.shape == (8,) and np.isfinite(pred).all()
    assert tracker.final is not None
    assert tracker.final["I_XZ"] >= 0.0 and tracker.final["I_ZY"] >= 0.0
    assert 0.0 <= tracker.final["probe_r2"] <= 1.0
    # NCR's prior-trust dial: tau = lambda2/lambda1.
    assert len(tracker.alpha_final) == 1
    assert tracker.alpha_final[0] == pytest.approx(a2 / a1)


def test_transformer_tracker_matches_refit_epochs():
    from metascfc.models.iclr_backbones import fit_predict_llm_gated

    rng = np.random.default_rng(9)
    n_rois, n_subjects = 8, 40
    fc = rng.standard_normal((n_subjects, n_rois, n_rois))
    sc = np.abs(rng.standard_normal((n_subjects, n_rois, n_rois)))
    sc = 0.5 * (sc + sc.transpose(0, 2, 1))
    y = fc.sum(axis=(1, 2))
    tracker = IBEpochTracker()
    _, _, _, best_epoch, _, _ = fit_predict_llm_gated(
        fc, sc, y, np.arange(24), np.arange(24, 32), np.arange(32, 40),
        np.linspace(0, 1, n_rois),
        hidden_grid=[8], dropout_grid=[0.0], lr_grid=[1e-3],
        device=torch.device("cpu"), n_layers=1, heads=2,
        epochs=5, min_epochs=2, patience=10, top_percent=25.0,
        seed=3, ib_tracker=tracker,
    )
    # One logged epoch per fixed-epoch refit epoch.
    assert len(tracker.epochs) == best_epoch
