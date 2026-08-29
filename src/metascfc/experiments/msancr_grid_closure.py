"""Pre-full-run grid closure for the frozen MS-A-NCR v2 refinement.

This module adds only hyperparameter-boundary closure on top of the frozen
refinement: explicit v2 boundary reporting, the 20%-of-splits grid-closed
criterion, the v6 decision gate, a descriptive old-vs-new comparison, and
freezing of the final 10x5 config.  The solver, splits, selection rule,
prior swaps, and biomarker definitions are reused unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd
import yaml

from metascfc.benchmark_utils import atomic_write_csv
from metascfc.experiments.msancr_refinement import (
    CONTROL_PRIORS,
    MODEL_A2,
    MODEL_A3,
    MODEL_A4,
    atomic_write_json,
    component_diagnostic,
    stable_hash,
)

FINAL_SEEDS = list(range(10))
NEW_RIDGE_LOWER = 0.001
NEW_GAMMA_LOWER = 0.1
NEW_LAMBDA_L_LOWER = 0.03


def finalize_grids(cfg: Mapping[str, Any]) -> dict[str, list[float]]:
    return {
        "ridge_grid_final": [float(v) for v in cfg["ridge_grid"]],
        "gamma_grid_final": [float(v) for v in cfg["gamma_grid"]],
        "lambda_laplacian_grid_final": [float(v) for v in cfg["lambda_laplacian_grid"]],
    }


def compute_boundary_report_v2(
    selected_df: pd.DataFrame,
    grids: Mapping[str, list[float]],
) -> pd.DataFrame:
    """Explicit min/max boundary flags for every outer split's final selection."""
    ridge = grids["ridge_grid_final"]
    gamma = grids["gamma_grid_final"]
    lam = grids["lambda_laplacian_grid_final"]
    rows = []
    for (seed, fold), group in selected_df.groupby(["seed", "fold"]):
        def one(model_id: str) -> pd.Series:
            selected = group[group.model_id == model_id]
            if len(selected) != 1:
                raise ValueError(f"Expected exactly one {model_id} row for seed={seed} fold={fold}")
            return selected.iloc[0]
        a4, a2, a3 = one(MODEL_A4), one(MODEL_A2), one(MODEL_A3)
        rows.append({
            "seed": int(seed),
            "fold": int(fold),
            "A4_lambda_fc": float(a4.lambda_fc),
            "A4_lambda_sc": float(a4.lambda_sc),
            "A4_lambda_fc_boundary": bool(float(a4.lambda_fc) in (min(ridge), max(ridge))),
            "A4_lambda_sc_boundary": bool(float(a4.lambda_sc) in (min(ridge), max(ridge))),
            "A2_lambda_L": float(a2.lambda_l),
            "A2_lambda_L_boundary": bool(float(a2.lambda_l) in (min(lam), max(lam))),
            "A3_lambda_fc": float(a3.lambda_fc),
            "A3_lambda_sc": float(a3.lambda_sc),
            "A3_gamma": float(a3.gamma),
            "A3_lambda_L": float(a3.lambda_l),
            "A3_lifting": str(a3.lifting),
            "A3_lambda_fc_boundary": bool(float(a3.lambda_fc) in (min(ridge), max(ridge))),
            "A3_lambda_sc_boundary": bool(float(a3.lambda_sc) in (min(ridge), max(ridge))),
            "A3_gamma_boundary": bool(float(a3.gamma) in (min(gamma), max(gamma))),
            "A3_lambda_L_boundary": bool(float(a3.lambda_l) in (min(lam), max(lam))),
            "A4_lambda_fc_unresolved_lower": bool(float(a4.lambda_fc) == min(ridge)),
            "A3_gamma_unresolved_lower": bool(float(a3.gamma) == min(gamma)),
            "A3_lambda_L_unresolved_lower": bool(float(a3.lambda_l) == min(lam)),
            "A2_lambda_L_unresolved_lower": bool(float(a2.lambda_l) == min(lam)),
        })
    return pd.DataFrame(rows).sort_values(["seed", "fold"]).reset_index(drop=True)


def grid_closure_status(
    boundary_df: pd.DataFrame,
    grids: Mapping[str, list[float]],
) -> dict[str, Any]:
    """Closed iff at most 20% of outer splits select the NEW lower boundary."""
    n = int(len(boundary_df))
    if n == 0:
        raise ValueError("Cannot evaluate grid closure on an empty boundary report")
    threshold = int(0.2 * n)

    def entry(count: int, boundary_value: float, description: str) -> dict[str, Any]:
        return {
            "description": description,
            "boundary_value": float(boundary_value),
            "n_boundary_selections": int(count),
            "n_outer_splits": n,
            "threshold_closed_max_selections": threshold,
            "grid_closed": bool(count <= threshold),
        }

    ridge_min = min(grids["ridge_grid_final"])
    gamma_min = min(grids["gamma_grid_final"])
    lam_min = min(grids["lambda_laplacian_grid_final"])
    ridge_hits = boundary_df[
        (boundary_df.A4_lambda_fc == ridge_min) | (boundary_df.A3_lambda_fc == ridge_min)
    ]
    gamma_hits = boundary_df[boundary_df.A3_gamma == gamma_min]
    lam_hits = boundary_df[
        (boundary_df.A2_lambda_L == lam_min) | (boundary_df.A3_lambda_L == lam_min)
    ]
    return {
        "closure_rule": "grid_closed = new-boundary selections <= 20% of outer splits",
        "ridge_grid_closed": entry(
            len(ridge_hits), ridge_min, "lambda_fc selected at the new 0.001 lower boundary"
        ),
        "gamma_grid_closed": entry(
            len(gamma_hits), gamma_min, "gamma selected at the new 0.1 lower boundary"
        ),
        "lambda_L_grid_closed": entry(
            len(lam_hits), lam_min, "lambda_L selected at the new 0.03 lower boundary"
        ),
    }


def make_grid_closure_decision(
    summary: pd.DataFrame,
    comparisons: pd.DataFrame,
    max_rmse_degradation: float,
    closure_status: Mapping[str, Any],
) -> dict[str, Any]:
    """v6 decision gate; n=3 results are descriptive only."""
    def row_for(label: str) -> pd.Series:
        row = comparisons[comparisons.comparison == label]
        if len(row) != 1:
            raise ValueError(f"Missing paired comparison: {label}")
        return row.iloc[0]

    main = row_for("A3 matched - A4")
    median_delta = float(main.median_delta_pearson)
    mean_delta = float(main.mean_delta_pearson)
    positive_seeds = int(main.positive_pearson)
    rmse_degradation = float(main.mean_delta_rmse)
    no_material_rmse_degradation = rmse_degradation <= float(max_rmse_degradation)

    specificity = {}
    for control in CONTROL_PRIORS:
        swap = row_for(f"A3 matched - A3 {control}-fixed")
        specificity[control] = {
            "mean_delta_pearson": float(swap.mean_delta_pearson),
            "median_delta_pearson": float(swap.median_delta_pearson),
            "matched_better_mean_or_median": bool(
                swap.mean_delta_pearson > 0 or swap.median_delta_pearson > 0
            ),
        }
    specificity_pass_count = sum(v["matched_better_mean_or_median"] for v in specificity.values())

    go = (
        median_delta >= 0.010
        and positive_seeds >= 2
        and mean_delta >= 0.008
        and no_material_rmse_degradation
        and specificity_pass_count >= 2
    )
    if go:
        recommendation = "full_10x5_msancr"
    elif 0.005 <= median_delta < 0.010 and positive_seeds >= 2:
        recommendation = "review_before_full_run"
    else:
        recommendation = "ct_mac_prior_rebuild"

    a2 = row_for("A3 matched - A2")
    return {
        "recommended_next_step": recommendation,
        "inference_status": "descriptive_only_n_equals_3_no_significance_claim",
        "gate": {
            "median_delta_pearson_vs_A4": median_delta,
            "mean_delta_pearson_vs_A4": mean_delta,
            "positive_seeds_vs_A4": positive_seeds,
            "min_seeds_individually_positive_required": 2,
            "min_seeds_individually_positive_met": bool(positive_seeds >= 2),
            "median_threshold_go": 0.010,
            "mean_threshold_go": 0.008,
            "borderline_median_lower": 0.005,
            "mean_delta_rmse_A3_minus_A4": rmse_degradation,
            "max_material_rmse_degradation": float(max_rmse_degradation),
            "no_material_rmse_degradation": no_material_rmse_degradation,
            "specificity_pass_count": int(specificity_pass_count),
            "specificity_required": 2,
            "go_all_requirements": bool(go),
        },
        "fixed_prior_swap_specificity": specificity,
        "new_A3_minus_A2": {
            "mean_delta_pearson": float(a2.mean_delta_pearson),
            "median_delta_pearson": float(a2.median_delta_pearson),
            "positive_seeds": int(a2.positive_pearson),
        },
        "grid_closure": dict(closure_status),
        "component_contribution": component_diagnostic(summary),
    }


def build_grid_closure_comparison(
    old_seed_df: pd.DataFrame,
    new_seed_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Descriptive per-seed old-vs-new comparison; never used for selection."""

    def extract(seed_df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        base = seed_df[seed_df.evaluation_type == "retuned_base"]
        a4 = base[base.model_id == MODEL_A4].set_index("seed")["pearson"]
        a3 = base[(base.model_id == MODEL_A3) & (base.prior_type == "matched")].set_index("seed")["pearson"]
        return a4, a3

    old_a4, old_a3 = extract(old_seed_df)
    new_a4, new_a3 = extract(new_seed_df)
    seeds = sorted(set(old_a4.index) & set(old_a3.index) & set(new_a4.index) & set(new_a3.index))
    if not seeds:
        raise ValueError("No common seeds between old and new refinement runs")
    rows = []
    for seed in seeds:
        rows.append({
            "seed": int(seed),
            "old_A4_pearson": float(old_a4.loc[seed]),
            "new_A4_pearson": float(new_a4.loc[seed]),
            "old_A3_pearson": float(old_a3.loc[seed]),
            "new_A3_pearson": float(new_a3.loc[seed]),
            "old_A3_minus_A4": float(old_a3.loc[seed] - old_a4.loc[seed]),
            "new_A3_minus_A4": float(new_a3.loc[seed] - new_a4.loc[seed]),
        })
    per_seed = pd.DataFrame(rows)
    aggregates = {
        "delta_new_A4_vs_old_A4_mean": float((new_a4.loc[seeds] - old_a4.loc[seeds]).mean()),
        "delta_new_A4_vs_old_A4_median": float((new_a4.loc[seeds] - old_a4.loc[seeds]).median()),
        "delta_new_A3_vs_old_A3_mean": float((new_a3.loc[seeds] - old_a3.loc[seeds]).mean()),
        "delta_new_A3_vs_old_A3_median": float((new_a3.loc[seeds] - old_a3.loc[seeds]).median()),
        "old_A3_minus_A4_mean": float((old_a3.loc[seeds] - old_a4.loc[seeds]).mean()),
        "old_A3_minus_A4_median": float((old_a3.loc[seeds] - old_a4.loc[seeds]).median()),
        "new_A3_minus_A4_mean": float((new_a3.loc[seeds] - new_a4.loc[seeds]).mean()),
        "new_A3_minus_A4_median": float((new_a3.loc[seeds] - new_a4.loc[seeds]).median()),
        "change_in_A3_minus_A4_margin_mean": float(
            (new_a3.loc[seeds] - new_a4.loc[seeds]).mean()
            - (old_a3.loc[seeds] - old_a4.loc[seeds]).mean()
        ),
        "change_in_A3_minus_A4_margin_median": float(
            (new_a3.loc[seeds] - new_a4.loc[seeds]).median()
            - (old_a3.loc[seeds] - old_a4.loc[seeds]).median()
        ),
    }
    return per_seed, aggregates


def freeze_final_config(
    decision: Mapping[str, Any],
    closure_cfg: Mapping[str, Any],
    path: Optional[str | Path] = None,
) -> Optional[Path]:
    """Freeze configs/iclr/msancr_final_10x5.yaml ONLY after a GO decision.

    This writes a config file; it never executes the final run.
    """
    if decision.get("recommended_next_step") != "full_10x5_msancr":
        return None
    repo_root = Path(__file__).resolve().parents[3]
    target = Path(path) if path is not None else repo_root / "configs/iclr/msancr_final_10x5.yaml"
    final = dict(closure_cfg)
    final["experiment_name"] = "msancr_final_10x5_working_memory"
    final["seeds"] = list(FINAL_SEEDS)
    final["output_dir"] = "outputs/iclr/msancr_final_10x5"
    final["figures_dir"] = "figures/iclr/msancr_final_10x5"
    # Frozen grids: identical to the closure run, no further expansion.
    final["ridge_expanded_grid"] = [float(v) for v in final["ridge_grid"]]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(final, sort_keys=False), encoding="utf-8")
    return target


def finalize_grid_closure(
    output_dir: str | Path,
    cfg: Mapping[str, Any],
    old_refinement_dir: str | Path,
    max_rmse_degradation: float,
) -> dict[str, Any]:
    """Write v2 boundary report, decision, comparison; freeze final config on GO."""
    import json

    output_dir = Path(output_dir)
    old_dir = Path(old_refinement_dir)
    summary = pd.read_csv(output_dir / "summary_metrics.csv")
    comparisons = pd.read_csv(output_dir / "paired_comparisons.csv")
    selected = pd.read_csv(output_dir / "selected_hyperparameters.csv")
    grids = finalize_grids(cfg)

    boundary = compute_boundary_report_v2(selected, grids)
    atomic_write_csv(boundary, output_dir / "boundary_selection_report_v2.csv")
    status = grid_closure_status(boundary, grids)
    decision = make_grid_closure_decision(summary, comparisons, max_rmse_degradation, status)

    old_seed = pd.read_csv(old_dir / "seed_metrics.csv")
    new_seed = pd.read_csv(output_dir / "seed_metrics.csv")
    per_seed, aggregates = build_grid_closure_comparison(old_seed, new_seed)
    atomic_write_csv(per_seed, output_dir / "grid_closure_comparison.csv")
    decision["old_vs_new_aggregate_changes"] = aggregates

    frozen = freeze_final_config(decision, cfg)
    decision["final_config_path"] = str(frozen) if frozen is not None else None
    decision["final_config_frozen"] = frozen is not None
    atomic_write_json(decision, output_dir / "grid_closure_decision.json")

    metadata_path = output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["grid_closure_decision_hash"] = stable_hash(decision)
    metadata["grid_closure_status"] = status
    metadata["grid_closure_complete"] = True
    atomic_write_json(metadata, metadata_path)
    return decision



