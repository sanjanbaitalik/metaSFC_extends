"""Behavioral tests for the pre-full-run MS-A-NCR grid closure (prompt v6)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metascfc.experiments.msancr_grid_closure import (
    NEW_GAMMA_LOWER,
    NEW_LAMBDA_L_LOWER,
    NEW_RIDGE_LOWER,
    compute_boundary_report_v2,
    build_grid_closure_comparison,
    finalize_grids,
    freeze_final_config,
    grid_closure_status,
    make_grid_closure_decision,
)
from metascfc.experiments.msancr_refinement import (
    MODEL_A2,
    MODEL_A3,
    MODEL_A4,
    CacheFactory,
    hyperparameter_payload,
    make_fixed_prior_swap_configs,
    make_inner_cv_splits,
    prepare_inner_folds,
    select_best_candidate,
    tune_outer_split,
    validate_refinement_config,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    _predict_msancr,
    _solve_msancr_kernel,
    build_msancr_cache,
    recover_msancr_beta,
)

ROOT = Path(__file__).resolve().parents[1]
CLOSURE_CONFIG = ROOT / "configs/iclr/msancr_grid_closure.yaml"
OLD_REFINEMENT_DIR = ROOT / "outputs/iclr/msancr_refinement"


def load_closure_config():
    return yaml.safe_load(CLOSURE_CONFIG.read_text(encoding="utf-8"))


def test_production_config_contains_new_ridge_lower():
    cfg = load_closure_config()
    assert NEW_RIDGE_LOWER in [float(v) for v in cfg["ridge_grid"]]


def test_production_config_contains_new_gamma_lower():
    cfg = load_closure_config()
    assert NEW_GAMMA_LOWER in [float(v) for v in cfg["gamma_grid"]]


def test_production_config_contains_new_lambda_L_lower():
    cfg = load_closure_config()
    assert NEW_LAMBDA_L_LOWER in [float(v) for v in cfg["lambda_laplacian_grid"]]


def test_no_unrequested_grid_expansion_occurred():
    cfg = load_closure_config()
    validate_refinement_config(cfg)
    assert [float(v) for v in cfg["ridge_grid"]] == [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    assert [float(v) for v in cfg["gamma_grid"]] == [0.1, 0.25, 0.5, 1.0, 2.0]
    assert [float(v) for v in cfg["lambda_laplacian_grid"]] == [0.03, 0.1, 0.5, 1.0, 2.0, 5.0]
    # No second-stage expansion beyond the closure grid.
    assert [float(v) for v in cfg["ridge_expanded_grid"]] == [float(v) for v in cfg["ridge_grid"]]
    assert set(cfg["lifting_rules"]) == {"prod", "mean"}


def test_selection_remains_pearson_first():
    summary = pd.DataFrame([
        {"candidate_id": "higher_r", "mean_pearson": 0.30, "mean_rmse": 20.0, "mean_mae": 10.0,
         "lambda_fc": 1.0, "lambda_sc": 1.0, "lambda_l": 1.0, "gamma": 1.0, "lifting": "prod"},
        {"candidate_id": "lower_rmse", "mean_pearson": 0.295, "mean_rmse": 1.0, "mean_mae": 1.0,
         "lambda_fc": 1.0, "lambda_sc": 1.0, "lambda_l": 1.0, "gamma": 1.0, "lifting": "prod"},
    ])
    assert select_best_candidate(summary)["candidate_id"] == "higher_r"
    summary.loc[1, "mean_pearson"] = 0.299
    assert select_best_candidate(summary)["candidate_id"] == "lower_rmse"


def test_same_three_fold_inner_cv_is_configured():
    cfg = load_closure_config()
    assert int(cfg["n_inner_folds"]) == 3 and int(cfg["n_outer_folds"]) == 5
    assert [int(s) for s in cfg["seeds"]] == [0, 1, 2]
    y = np.linspace(0, 1, 30)
    splits = make_inner_cv_splits(np.arange(24), y, seed=0, outer_fold=0, n_splits=3)
    validation = np.concatenate([val for _, val in splits])
    np.testing.assert_array_equal(np.sort(validation), np.arange(24))


def synthetic(seed=0, n=30, n_rois=6):
    rng = np.random.default_rng(seed)
    n_edges = n_rois * (n_rois - 1) // 2
    return (
        rng.normal(size=(n, n_edges)),
        rng.normal(size=(n, n_edges)),
        rng.normal(size=n),
        rng.uniform(size=n_rois),
    )


def test_outer_test_labels_are_not_used_for_selection():
    X_fc, X_sc, y, prior = synthetic(seed=21)
    splits = make_inner_cv_splits(np.arange(24), y, seed=0, outer_fold=0, n_splits=3)
    factory = CacheFactory({"matched": prior}, 6, 3, 1e-3, "binary", "sym")
    cfg = {
        "ridge_grid": [0.1, 1.0], "ridge_expanded_grid": [0.1, 1.0],
        "lambda_laplacian_grid": [0.5], "lifting_rules": ["prod"], "gamma_grid": [0.5],
    }
    selected_a, _, _ = tune_outer_split(prepare_inner_folds(X_fc, X_sc, y, splits), factory, cfg, 0, 0)
    y_corrupted = y.copy()
    y_corrupted[24:] = 1000.0  # outer-test-only labels
    selected_b, _, _ = tune_outer_split(
        prepare_inner_folds(X_fc, X_sc, y_corrupted, splits), factory, cfg, 0, 0
    )
    for model in (MODEL_A4, MODEL_A3):
        assert hyperparameter_payload(selected_a[model]) == hyperparameter_payload(selected_b[model])


def test_fixed_control_swaps_reuse_matched_selected_hyperparameters():
    selected = {
        "lambda_fc": 0.001, "lambda_sc": 100.0, "lambda_l": 0.03,
        "gamma": 0.1, "lifting": "mean",
    }
    swaps = make_fixed_prior_swap_configs(selected)
    assert all(payload == hyperparameter_payload(selected) for payload in swaps.values())


def _closure_grid_frame(values_by_column):
    rows = []
    for i in range(15):
        rows.append({"seed": 0, "fold": i, **{c: v[i] for c, v in values_by_column.items()}})
    return pd.DataFrame(rows)


def test_boundary_flags_are_correctly_computed():
    cfg = load_closure_config()
    grids = finalize_grids(cfg)
    selected = pd.DataFrame([
        {"seed": 0, "fold": 0, "model_id": MODEL_A4, "lambda_fc": 0.001, "lambda_sc": 1.0,
         "lambda_l": 0.0, "gamma": 0.0, "lifting": "prod"},
        {"seed": 0, "fold": 0, "model_id": MODEL_A2, "lambda_fc": 1.0, "lambda_sc": 1.0,
         "lambda_l": 0.03, "gamma": 0.0, "lifting": "prod"},
        {"seed": 0, "fold": 0, "model_id": MODEL_A3, "lambda_fc": 0.5, "lambda_sc": 1.0,
         "lambda_l": 0.5, "gamma": 0.1, "lifting": "mean"},
        {"seed": 0, "fold": 1, "model_id": MODEL_A4, "lambda_fc": 1.0, "lambda_sc": 10.0,
         "lambda_l": 0.0, "gamma": 0.0, "lifting": "prod"},
        {"seed": 0, "fold": 1, "model_id": MODEL_A2, "lambda_fc": 1.0, "lambda_sc": 1.0,
         "lambda_l": 1.0, "gamma": 0.0, "lifting": "prod"},
        {"seed": 0, "fold": 1, "model_id": MODEL_A3, "lambda_fc": 1.0, "lambda_sc": 1.0,
         "lambda_l": 5.0, "gamma": 1.0, "lifting": "prod"},
    ])
    report = compute_boundary_report_v2(selected, grids)
    assert bool(report.loc[0, "A4_lambda_fc_boundary"]) and report.loc[0, "A4_lambda_fc_unresolved_lower"]
    assert not report.loc[0, "A4_lambda_sc_boundary"]
    assert report.loc[0, "A2_lambda_L_boundary"] and report.loc[0, "A2_lambda_L_unresolved_lower"]
    assert report.loc[0, "A3_gamma_boundary"] and report.loc[0, "A3_gamma_unresolved_lower"]
    assert not report.loc[0, "A3_lambda_L_boundary"]
    assert not report.loc[1, "A4_lambda_fc_boundary"]
    assert report.loc[1, "A3_lambda_L_boundary"] and not report.loc[1, "A3_lambda_L_unresolved_lower"]
    required = {
        "seed", "fold", "A4_lambda_fc", "A4_lambda_sc", "A4_lambda_fc_boundary",
        "A4_lambda_sc_boundary", "A2_lambda_L", "A2_lambda_L_boundary", "A3_lambda_fc",
        "A3_lambda_sc", "A3_gamma", "A3_lambda_L", "A3_lifting", "A3_lambda_fc_boundary",
        "A3_lambda_sc_boundary", "A3_gamma_boundary", "A3_lambda_L_boundary",
    }
    assert required.issubset(set(report.columns))


def test_grid_closed_threshold_is_correctly_computed():
    cfg = load_closure_config()
    grids = finalize_grids(cfg)
    ridge_min, gamma_min, lam_min = 0.001, 0.1, 0.03

    def frame(ridge_hits, gamma_hits, lam_hits):
        values = {
            "A4_lambda_fc": [ridge_min if i < ridge_hits else 1.0 for i in range(15)],
            "A3_lambda_fc": [1.0] * 15,
            "A3_gamma": [gamma_min if i < gamma_hits else 0.5 for i in range(15)],
            "A2_lambda_L": [lam_min if i < lam_hits else 0.5 for i in range(15)],
            "A3_lambda_L": [0.5] * 15,
        }
        return _closure_grid_frame(values)

    closed = grid_closure_status(frame(3, 3, 3), grids)
    assert closed["ridge_grid_closed"]["grid_closed"] is True
    assert closed["gamma_grid_closed"]["grid_closed"] is True
    assert closed["lambda_L_grid_closed"]["grid_closed"] is True
    unresolved = grid_closure_status(frame(4, 4, 4), grids)
    assert unresolved["ridge_grid_closed"]["grid_closed"] is False
    assert unresolved["gamma_grid_closed"]["grid_closed"] is False
    assert unresolved["lambda_L_grid_closed"]["grid_closed"] is False
    assert unresolved["ridge_grid_closed"]["threshold_closed_max_selections"] == 3


def test_freeze_final_config_only_after_go(tmp_path):
    cfg = load_closure_config()
    go_decision = {"recommended_next_step": "full_10x5_msancr"}
    stop_decision = {"recommended_next_step": "ct_mac_prior_rebuild"}
    assert freeze_final_config(stop_decision, cfg, tmp_path / "final.yaml") is None
    frozen = freeze_final_config(go_decision, cfg, tmp_path / "final.yaml")
    assert frozen is not None and frozen.exists()
    final = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    assert final["seeds"] == list(range(10))
    assert [float(v) for v in final["ridge_grid"]] == [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    assert [float(v) for v in final["gamma_grid"]] == [0.1, 0.25, 0.5, 1.0, 2.0]
    assert [float(v) for v in final["lambda_laplacian_grid"]] == [0.03, 0.1, 0.5, 1.0, 2.0, 5.0]
    assert [float(v) for v in final["ridge_expanded_grid"]] == [float(v) for v in final["ridge_grid"]]


def test_final_runner_is_prepared_but_never_invoked():
    script = ROOT / "scripts/106_run_msancr_final.py"
    assert script.exists()
    text = script.read_text(encoding="utf-8")
    assert 'if __name__ == "__main__":' in text
    assert "run_refinement(" in text
    assert "enforce_seed_gate=False" in text
    assert not (ROOT / "outputs/iclr/msancr_final_10x5").exists()


def test_old_refinement_outputs_are_not_overwritten():
    import json
    meta = json.loads((OLD_REFINEMENT_DIR / "run_metadata.json").read_text(encoding="utf-8"))
    assert meta["config_hash"] == "4bb1e3a6b9993cee"
    assert (OLD_REFINEMENT_DIR / "COMPLETE").exists()
    assert len(pd.read_csv(OLD_REFINEMENT_DIR / "seed_metrics.csv")) == 21
    closure_cfg = load_closure_config()
    assert Path(closure_cfg["output_dir"]).resolve() != OLD_REFINEMENT_DIR.resolve()


def test_direct_dual_solver_invariant_still_holds():
    X_fc, X_sc, y, prior = synthetic(seed=31, n=24)
    cache = build_msancr_cache(prior, 6, gamma=1.0, top_k=3)
    alpha, _ = _solve_msancr_kernel(X_fc, X_sc, y, cache, 0.5, 2.0, 1.0)
    beta_fc, beta_sc = recover_msancr_beta(X_fc, X_sc, alpha, cache, 0.5, 2.0, 1.0)
    pred = _predict_msancr(X_fc, X_sc, X_fc, X_sc, alpha, cache, 0.5, 2.0, 1.0)
    p = cache.n_edges
    X = np.c_[X_fc, X_sc]
    lap_full = np.zeros((p, p))
    lap_full[np.ix_(cache.active_indices, cache.active_indices)] = cache.active_laplacian
    penalty = np.zeros((2 * p, 2 * p))
    penalty[:p, :p] = 0.5 * np.diag(cache.D) + 1.0 * lap_full
    penalty[p:, p:] = 2.0 * np.eye(p)
    beta_ref = np.linalg.solve(X.T @ X + 2 * p * penalty, X.T @ y)
    np.testing.assert_allclose(np.r_[beta_fc, beta_sc], beta_ref, atol=1e-7, rtol=1e-6)
    np.testing.assert_allclose(pred, X @ beta_ref, atol=1e-7, rtol=1e-6)


def _paired_comparisons_frame(median, mean, positive, rmse):
    rows = [{
        "comparison": "A3 matched - A4", "n_seeds": 3,
        "mean_delta_pearson": mean, "median_delta_pearson": median,
        "positive_pearson": positive, "mean_delta_rmse": rmse, "mean_delta_mae": 0.0,
    }]
    for control in ("unrelated", "shuffled", "random"):
        rows.append({
            "comparison": f"A3 matched - A3 {control}-fixed", "n_seeds": 3,
            "mean_delta_pearson": 0.015, "median_delta_pearson": 0.015,
            "positive_pearson": 3, "mean_delta_rmse": -0.05, "mean_delta_mae": -0.05,
        })
    rows.append({
        "comparison": "A3 matched - A2", "n_seeds": 3,
        "mean_delta_pearson": 0.006, "median_delta_pearson": 0.006,
        "positive_pearson": 3, "mean_delta_rmse": 0.0, "mean_delta_mae": 0.0,
    })
    return pd.DataFrame(rows)


def _summary_frame():
    rows = []
    for model, prior, value in [
        (MODEL_A4, "none", 0.269), (MODEL_A2, "matched", 0.274), (MODEL_A3, "matched", 0.282),
    ]:
        rows.append({"model_id": model, "prior_type": prior, "pearson_mean": value,
                     "rmse_mean": 11.5, "mae_mean": 9.3, "n_seeds": 3})
    return pd.DataFrame(rows)


def test_decision_gate_thresholds_match_prompt_v6():
    grids = finalize_grids(load_closure_config())
    boundary = _closure_grid_frame({
        "A4_lambda_fc": [1.0] * 15, "A3_lambda_fc": [1.0] * 15,
        "A3_gamma": [0.5] * 15, "A2_lambda_L": [0.5] * 15, "A3_lambda_L": [0.5] * 15,
    })
    status = grid_closure_status(boundary, grids)
    go = make_grid_closure_decision(
        _summary_frame(), _paired_comparisons_frame(0.012, 0.012, 3, -0.05), 0.1, status
    )
    assert go["recommended_next_step"] == "full_10x5_msancr"
    borderline = make_grid_closure_decision(
        _summary_frame(), _paired_comparisons_frame(0.007, 0.007, 3, -0.05), 0.1, status
    )
    assert borderline["recommended_next_step"] == "review_before_full_run"
    stop = make_grid_closure_decision(
        _summary_frame(), _paired_comparisons_frame(0.003, 0.003, 1, 0.2), 0.1, status
    )
    assert stop["recommended_next_step"] == "ct_mac_prior_rebuild"


def test_comparison_is_descriptive_and_wellformed():
    def seed_frame(a4, a3):
        rows = []
        for seed in range(3):
            rows.append({"model_id": MODEL_A4, "prior_type": "none",
                         "evaluation_type": "retuned_base", "seed": seed, "pearson": a4[seed]})
            rows.append({"model_id": MODEL_A3, "prior_type": "matched",
                         "evaluation_type": "retuned_base", "seed": seed, "pearson": a3[seed]})
        return pd.DataFrame(rows)

    per_seed, agg = build_grid_closure_comparison(
        seed_frame([0.20, 0.25, 0.30], [0.21, 0.27, 0.31]),
        seed_frame([0.22, 0.26, 0.29], [0.24, 0.28, 0.31]),
    )
    assert list(per_seed.columns) == [
        "seed", "old_A4_pearson", "new_A4_pearson", "old_A3_pearson", "new_A3_pearson",
        "old_A3_minus_A4", "new_A3_minus_A4",
    ]
    assert len(per_seed) == 3
    np.testing.assert_allclose(
        per_seed.new_A3_minus_A4 - per_seed.old_A3_minus_A4,
        [0.01, 0.00, 0.01], atol=1e-12,
    )
    assert set(agg) >= {
        "delta_new_A4_vs_old_A4_mean", "delta_new_A3_vs_old_A3_mean",
        "change_in_A3_minus_A4_margin_mean",
    }




