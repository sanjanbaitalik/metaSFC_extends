"""Behavioral tests for corrected MS-A-NCR and the v2 refinement path."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metascfc.experiments.msancr_refinement import (
    CONTROL_PRIORS,
    MODEL_A3,
    CacheFactory,
    candidate_id,
    evaluate_candidates,
    hyperparameter_payload,
    make_fixed_prior_swap_configs,
    make_inner_cv_splits,
    prepare_inner_folds,
    select_best_candidate,
    validate_refinement_config,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    ModalitySelectiveAnisotropicNCR,
    _predict_msancr,
    _solve_msancr_kernel,
    build_msancr_cache,
    recover_msancr_beta,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/iclr/msancr_refinement.yaml"


def synthetic(seed=0, n=24, n_rois=6):
    rng = np.random.default_rng(seed)
    n_edges = n_rois * (n_rois - 1) // 2
    return (
        rng.normal(size=(n, n_edges)),
        rng.normal(size=(n, n_edges)),
        rng.normal(size=n),
        rng.uniform(size=n_rois),
    )


def embedded_fc_laplacian(cache):
    result = np.zeros((cache.n_edges, cache.n_edges), dtype=np.float64)
    result[np.ix_(cache.active_indices, cache.active_indices)] = cache.active_laplacian
    return result


def direct_primal(X_fc, X_sc, y, cache, lambda_fc, lambda_sc, lambda_l):
    p = cache.n_edges
    X = np.concatenate([X_fc, X_sc], axis=1)
    penalty_fc = lambda_fc * np.diag(cache.D) + lambda_l * embedded_fc_laplacian(cache)
    penalty = np.zeros((2 * p, 2 * p), dtype=np.float64)
    penalty[:p, :p] = penalty_fc
    penalty[p:, p:] = lambda_sc * np.eye(p)
    return np.linalg.solve(
        X.T @ X + float(2 * p) * penalty,
        X.T @ y,
    )


@pytest.mark.parametrize(
    "gamma,lambda_l",
    [(0.0, 0.0), (1.0, 0.0), (0.0, 1.5), (1.0, 1.5)],
)
def test_optimized_dual_matches_direct_primal(gamma, lambda_l):
    X_fc, X_sc, y, prior = synthetic(seed=11)
    cache = build_msancr_cache(prior, 6, gamma=gamma, top_k=3)
    alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 0.7, 1.3, lambda_l)
    beta_fc, beta_sc = recover_msancr_beta(X_fc, X_sc, alpha, cache, 0.7, 1.3, lambda_l)
    beta_ref = direct_primal(X_fc, X_sc, y, cache, 0.7, 1.3, lambda_l)
    np.testing.assert_allclose(np.r_[beta_fc, beta_sc], beta_ref, atol=1e-7, rtol=1e-6)
    pred = _predict_msancr(X_fc, X_sc, X_fc, X_sc, alpha, cache, 0.7, 1.3, lambda_l)
    np.testing.assert_allclose(pred, np.c_[X_fc, X_sc] @ beta_ref, atol=1e-7, rtol=1e-6)


def test_full_fc_d_is_used_for_active_and_inactive_edges():
    X_fc, X_sc, y, prior = synthetic(seed=12)
    cache = build_msancr_cache(prior, 6, gamma=2.0, top_k=2)
    inactive = np.setdiff1d(np.arange(cache.n_edges), cache.active_indices)
    assert len(cache.active_indices) and len(inactive)
    assert np.ptp(cache.D[cache.active_indices]) > 1e-6
    assert np.ptp(cache.D[inactive]) > 1e-6
    alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 0.5, 2.0, 0.8)
    beta = np.r_[recover_msancr_beta(X_fc, X_sc, alpha, cache, 0.5, 2.0, 0.8)]
    beta_ref = direct_primal(X_fc, X_sc, y, cache, 0.5, 2.0, 0.8)
    np.testing.assert_allclose(beta, beta_ref, atol=1e-7, rtol=1e-6)


def test_inactive_nonuniform_d_changes_predictions_and_old_active_only_fails():
    X_fc, X_sc, y, prior = synthetic(seed=13)
    cache = build_msancr_cache(prior, 6, gamma=2.0, top_k=2)
    inactive = np.setdiff1d(np.arange(cache.n_edges), cache.active_indices)
    assert np.ptp(cache.D[inactive]) > 1e-6
    alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 1.0, 0.0)
    pred = _predict_msancr(X_fc, X_sc, X_fc, X_sc, alpha, cache, 1.0, 1.0, 0.0)

    # Reconstruct the old active-only objective: inactive D was forcibly one.
    old_D = cache.D.copy()
    old_D[inactive] = 1.0
    p = cache.n_edges
    X = np.c_[X_fc, X_sc]
    old_penalty = np.zeros((2 * p, 2 * p))
    old_penalty[:p, :p] = np.diag(old_D)
    old_penalty[p:, p:] = np.eye(p)
    old_beta = np.linalg.solve(X.T @ X + 2 * p * old_penalty, X.T @ y)
    full_beta = direct_primal(X_fc, X_sc, y, cache, 1.0, 1.0, 0.0)
    assert not np.allclose(old_beta, full_beta, atol=1e-6)
    assert not np.allclose(X @ old_beta, pred, atol=1e-6)
    np.testing.assert_allclose(X @ full_beta, pred, atol=1e-7, rtol=1e-6)


def test_gamma_zero_and_uniform_prior_are_isotropic():
    _, _, _, prior = synthetic(seed=14)
    gamma_zero = build_msancr_cache(prior, 6, gamma=0.0, top_k=3)
    uniform = build_msancr_cache(np.ones(6), 6, gamma=2.0, top_k=3)
    np.testing.assert_allclose(gamma_zero.D, 1.0, atol=1e-12)
    np.testing.assert_allclose(uniform.D, 1.0, atol=1e-12)


def test_lambda_l_zero_removes_laplacian_and_sc_never_receives_it():
    X_fc, X_sc, y, prior = synthetic(seed=15)
    cache = build_msancr_cache(prior, 6, gamma=1.0, top_k=3)
    # With FC identically zero, changing the cognitive Laplacian cannot affect SC.
    X_fc[:] = 0.0
    alpha0, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 0.8, 0.0)
    alpha5, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 0.8, 5.0)
    pred0 = _predict_msancr(X_fc, X_sc, X_fc, X_sc, alpha0, cache, 1.0, 0.8, 0.0)
    pred5 = _predict_msancr(X_fc, X_sc, X_fc, X_sc, alpha5, cache, 1.0, 0.8, 5.0)
    np.testing.assert_allclose(pred0, pred5, atol=1e-10)


def test_a4_same_solver_no_prior_recovery_and_exact_model_beta():
    X_fc, X_sc, y, _ = synthetic(seed=16)
    cache = build_msancr_cache(np.ones(6), 6, gamma=0.0, top_k=3)
    model = ModalitySelectiveAnisotropicNCR(
        lambda_fc=1.0, lambda_sc=1.0, lambda_l=0.0,
        gamma=0.0, cache=cache, n_rois=6,
    ).fit(X_fc, X_sc, y)
    X_fc_z = model.scaler_fc_.transform(X_fc)
    X_sc_z = model.scaler_sc_.transform(X_sc)
    y_z = (y - y.mean()) / y.std()
    beta_ref = direct_primal(X_fc_z, X_sc_z, y_z, cache, 1.0, 1.0, 0.0)
    np.testing.assert_allclose(model.beta(), beta_ref, atol=1e-7, rtol=1e-6)


def test_cache_execution_matches_no_compatibility_cache():
    X_fc, X_sc, y, prior = synthetic(seed=17)
    cache = build_msancr_cache(prior, 6, gamma=1.0, top_k=3)
    compatibility_cache = {0.0: (np.array([999.0]), np.array([[999.0]]))}
    a_cached, _ = _solve_msancr_kernel(
        X_fc, X_sc, y, cache, 1.0, 2.0, 0.0, eig_cache=compatibility_cache
    )
    a_fresh, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 2.0, 0.0)
    np.testing.assert_allclose(a_cached, a_fresh, atol=1e-12)
    assert list(compatibility_cache) == [0.0]


def load_actual_config():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_actual_runtime_config_has_all_controls_and_both_liftings():
    cfg = load_actual_config()
    validate_refinement_config(cfg)
    assert tuple(cfg["prior_controls"]) == CONTROL_PRIORS
    assert set(cfg["priors"]["working_memory"]) == {"matched", *CONTROL_PRIORS}
    assert set(cfg["lifting_rules"]) == {"prod", "mean"}
    for path in cfg["priors"]["working_memory"].values():
        assert (ROOT / path).exists()


@pytest.mark.parametrize("lifting", ["prod", "mean"])
def test_prod_and_mean_execute_through_actual_solver(lifting):
    X_fc, X_sc, y, prior = synthetic(seed=18)
    cache = build_msancr_cache(prior, 6, gamma=0.5, lifting=lifting, top_k=3)
    alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 1.0, 1.0, 0.5)
    assert np.isfinite(alpha).all()


def test_outer_test_never_appears_in_inner_scoring_and_oof_is_exact():
    y = np.linspace(0, 1, 30)
    outer_train = np.arange(24)
    outer_test = np.arange(24, 30)
    splits = make_inner_cv_splits(outer_train, y, seed=0, outer_fold=0, n_splits=3)
    validation = np.concatenate([val for _, val in splits])
    assert not any(np.intersect1d(outer_test, part).size for split in splits for part in split)
    np.testing.assert_array_equal(np.sort(validation), outer_train)
    assert len(np.unique(validation)) == len(outer_train)


def test_selection_uses_pearson_first_and_rmse_only_within_tolerance():
    summary = pd.DataFrame([
        {"candidate_id": "higher", "mean_pearson": 0.30, "mean_rmse": 20.0, "mean_mae": 10.0,
         "lambda_fc": 1.0, "lambda_sc": 1.0, "lambda_l": 1.0, "gamma": 1.0, "lifting": "prod"},
        {"candidate_id": "lower", "mean_pearson": 0.295, "mean_rmse": 1.0, "mean_mae": 1.0,
         "lambda_fc": 1.0, "lambda_sc": 1.0, "lambda_l": 1.0, "gamma": 1.0, "lifting": "prod"},
    ])
    assert select_best_candidate(summary)["candidate_id"] == "higher"
    summary.loc[1, "mean_pearson"] = 0.299
    assert select_best_candidate(summary)["candidate_id"] == "lower"


def test_prior_swaps_reuse_exact_matched_selected_hyperparameters():
    selected = {
        "lambda_fc": 0.1, "lambda_sc": 10.0, "lambda_l": 2.0,
        "gamma": 0.5, "lifting": "mean",
    }
    swaps = make_fixed_prior_swap_configs(selected)
    assert tuple(swaps) == CONTROL_PRIORS
    assert all(payload == hyperparameter_payload(selected) for payload in swaps.values())
    assert len({candidate_id({**payload, "cache_key": None}) for payload in swaps.values()}) == 1


def test_scaler_is_fitted_on_inner_training_only():
    rng = np.random.default_rng(19)
    X_fc = rng.normal(size=(12, 15))
    X_sc = rng.normal(size=(12, 15))
    X_fc[8:] += 100.0
    X_sc[8:] -= 100.0
    y = rng.normal(size=12)
    prepared = prepare_inner_folds(X_fc, X_sc, y, [(np.arange(8), np.arange(8, 12))])[0]
    np.testing.assert_allclose(prepared.scaler_fc_mean, X_fc[:8].mean(axis=0))
    np.testing.assert_allclose(prepared.scaler_sc_mean, X_sc[:8].mean(axis=0))
    assert not np.allclose(prepared.scaler_fc_mean, X_fc.mean(axis=0))


def test_actual_candidate_evaluation_uses_three_inner_folds():
    X_fc, X_sc, y, prior = synthetic(seed=20, n=30)
    splits = make_inner_cv_splits(np.arange(30), y, seed=0, outer_fold=0, n_splits=3)
    folds = prepare_inner_folds(X_fc, X_sc, y, splits)
    factory = CacheFactory({"matched": prior}, 6, 3, 1e-3, "binary", "sym")
    candidate = {
        "lambda_fc": 1.0, "lambda_sc": 1.0, "lambda_l": 0.5,
        "gamma": 0.5, "lifting": "prod",
        "cache_key": CacheFactory.key("matched", 0.5, "prod"),
    }
    fold_df, summary = evaluate_candidates(
        [candidate], folds, factory, "test", 0, 0, MODEL_A3, "matched"
    )
    assert len(fold_df) == 3
    assert summary.n_inner_folds.iloc[0] == 3
    assert fold_df.inner_val_indices_hash.nunique() == 3