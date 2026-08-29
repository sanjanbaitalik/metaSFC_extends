"""Fluid Integration Prior (FIP) pilot evaluation — optimized version.

Modification 1: Evaluates FIP candidates on Fluid Intelligence prediction
using frozen MS-A-NCR with edge-level priors.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metascfc.benchmark_utils import (
    iter_nested_splits,
    load_connectomes,
    prediction_metrics,
)
from metascfc.experiments.fluid_integration_prior import (
    build_fip_line_graph_laplacian,
    create_shuffled_fip,
    fip_edges_to_matrix,
    fip_matrix_to_edges,
)
from metascfc.experiments.msancr_refinement import (
    make_inner_cv_splits,
    upper_triangle_features,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    _predict_msancr,
    _solve_msancr_kernel,
    build_msancr_cache,
    lift_roi_to_edge,
    recover_msancr_beta,
)

OUTPUT_DIR = Path("outputs/iclr/fluid_fip_pilot")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIP_PRIOR_DIR = Path("outputs/iclr/fluid_integration_prior")
QWEN_FLUID_PATH = Path("outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv")
QWEN_WM_PATH = Path("outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv")

N_ROIS = 116
N_EDGES = N_ROIS * (N_ROIS - 1) // 2


def load_fip_edges(name: str) -> np.ndarray:
    df = pd.read_csv(FIP_PRIOR_DIR / f"{name}_edges.csv")
    score_col = [c for c in df.columns if "score" in c][0]
    return df[score_col].values.astype(np.float64)


def load_roi_prior(path: Path) -> np.ndarray:
    df = pd.read_csv(path).sort_values("roi_index")
    return df["prior_score"].values.astype(np.float64)


def create_random_prior(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0, 1, n)


def build_edge_cache(fip_edges: np.ndarray, gamma: float):
    """Build MS-A-NCR cache for edge-level FIP. Returns (cache, elapsed)."""
    from metascfc.experiments.fluid_integration_prior import build_fip_msancr_cache
    import time
    t0 = time.time()
    cache = build_fip_msancr_cache(fip_edges, N_ROIS, gamma, top_k=30)
    return cache, time.time() - t0


# Global cache dict (precomputed once)
_CACHE_DB = {}


def get_or_build_cache(fip_name: str, fip_edges: np.ndarray, gamma: float):
    key = (fip_name, gamma)
    if key not in _CACHE_DB:
        _CACHE_DB[key], elapsed = build_edge_cache(fip_edges, gamma)
    return _CACHE_DB[key]


def build_node_cache(roi_prior: np.ndarray, gamma: float) -> _MSANCRCache:
    return build_msancr_cache(roi_prior, N_ROIS, gamma, lifting="mean", top_k=30, prior_space="node")


def fast_inner_select(
    X_fc: np.ndarray,
    X_sc: np.ndarray,
    y: np.ndarray,
    inner_splits: list,
    fip_candidates: dict,
    gamma: float = 0.5,
    lambda_l: float = 0.5,
    lambda_fc: float = 1.0,
    lambda_sc: float = 1.0,
) -> str:
    """Fast inner selection: only vary FIP identity (fixed hyperparams from WM)."""
    best_name = None
    best_rmse = np.inf

    for fip_name, fip_edges in fip_candidates.items():
        try:
            cache = get_or_build_cache(fip_name, fip_edges, gamma)
        except Exception:
            continue

        fold_rmses = []
        for train_idx, val_idx in inner_splits:
            scaler_fc = StandardScaler()
            scaler_sc = StandardScaler()
            X_fc_tr = scaler_fc.fit_transform(X_fc[train_idx])
            X_sc_tr = scaler_sc.fit_transform(X_sc[train_idx])
            y_mean = float(np.mean(y[train_idx]))
            y_std = max(float(np.std(y[train_idx])), 1e-8)
            y_z = (y[train_idx] - y_mean) / y_std

            X_fc_v = scaler_fc.transform(X_fc[val_idx])
            X_sc_v = scaler_sc.transform(X_sc[val_idx])

            try:
                alpha, _ = _solve_msancr_kernel(X_fc_tr, X_sc_tr, y_z, cache, lambda_fc, lambda_sc, lambda_l)
                pred_z = _predict_msancr(X_fc_v, X_sc_v, X_fc_tr, X_sc_tr, alpha, cache, lambda_fc, lambda_sc, lambda_l)
                pred = pred_z * y_std + y_mean
                rmse = float(np.sqrt(np.mean((y[val_idx] - pred) ** 2)))
                fold_rmses.append(rmse)
            except Exception:
                fold_rmses.append(np.inf)

        mean_rmse = np.mean(fold_rmses)
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_name = fip_name

    return best_name or list(fip_candidates.keys())[0]


def evaluate_model(
    X_fc_train, X_sc_train, y_z, X_fc_test, X_sc_test,
    y_mean, y_std, y_test,
    cache, lambda_fc, lambda_sc, lambda_l,
):
    """Evaluate one model on train/test split."""
    alpha, _ = _solve_msancr_kernel(X_fc_train, X_sc_train, y_z, cache, lambda_fc, lambda_sc, lambda_l)
    pred_z = _predict_msancr(X_fc_test, X_sc_test, X_fc_train, X_sc_train, alpha, cache, lambda_fc, lambda_sc, lambda_l)
    pred = pred_z * y_std + y_mean
    metrics = prediction_metrics(y_test, pred)
    beta_fc, beta_sc = recover_msancr_beta(X_fc_train, X_sc_train, alpha, cache, lambda_fc, lambda_sc, lambda_l)
    return {
        "pearson": float(metrics["pearson"]),
        "rmse": float(metrics["rmse"]),
        "mae": float(metrics["mae"]),
        "beta_fc": beta_fc,
    }


def main():
    print("=" * 70)
    print("FLUID INTEGRATION PRIOR (FIP) PILOT EVALUATION")
    print("=" * 70)
    t0 = time.time()

    seeds = [0, 1, 2]
    n_outer = 5
    n_inner = 3

    # Load data
    print("\n[1/8] Loading HCP data...")
    data_cfg = {
        "fc_path": "inputs/dataset_FC/FC_all.npy",
        "sc_path": "inputs/dataset_SC/SC_all.npy",
        "y_path": "inputs/dataset_SC/label_all.npy",
    }
    fc, sc, y, subjects, groups = load_connectomes(data_cfg)
    X_fc = upper_triangle_features(fc)
    X_sc = upper_triangle_features(sc)
    print(f"  FC={X_fc.shape}, SC={X_sc.shape}, y={y.shape}")

    # Load priors
    print("\n[2/8] Loading priors...")
    fip_candidates = {}

    for name, fname in [("FIP1_MAC", "fip1_mac"), ("FIP2_Bridge", "fip2_bridge"), ("FIP3_Weaktie", "fip3_weaktie")]:
        path = FIP_PRIOR_DIR / f"{fname}_edges.csv"
        if path.exists():
            fip_candidates[name] = load_fip_edges(fname)
            print(f"  {name}: {fip_candidates[name].shape}")

    qwen_fluid = None
    if QWEN_FLUID_PATH.exists():
        qwen_fluid = lift_roi_to_edge(load_roi_prior(QWEN_FLUID_PATH), N_ROIS, "prod")
        print(f"  Qwen Fluid: {qwen_fluid.shape}")
        fip_candidates["Qwen_Fluid"] = qwen_fluid

    random_edges = create_random_prior(N_EDGES, seed=42)
    shuffled_fip1 = create_shuffled_fip(fip_edges_to_matrix(fip_candidates.get("FIP1_MAC", np.zeros(N_EDGES))), seed=0)
    shuffled_edges = fip_matrix_to_edges(shuffled_fip1)

    unrelated_edges = None
    if QWEN_WM_PATH.exists():
        unrelated_edges = lift_roi_to_edge(load_roi_prior(QWEN_WM_PATH), N_ROIS, "prod")
        print(f"  Qwen WM (unrelated): {unrelated_edges.shape}")

    print(f"  Candidates: {list(fip_candidates.keys())}")

    # Fixed hyperparams (from WM MS-A-NCR)
    gamma = 0.5
    lambda_l = 0.5
    lambda_fc = 1.0
    lambda_sc = 1.0

    # Precompute caches once
    print("\n  Precomputing caches...")
    t_cache = time.time()
    for name, edges in fip_candidates.items():
        get_or_build_cache(name, edges, gamma)
    print(f"  Caches built in {time.time() - t_cache:.1f}s")

    # Run
    print(f"\n[3/8] Running {len(seeds)} seeds × {n_outer} folds...")
    all_split = []
    all_inner = []
    all_biomarker = []
    fip_counts = {k: 0 for k in fip_candidates if not k.startswith("Qwen")}

    for seed in seeds:
        print(f"\n  Seed {seed}:")
        outer_splits = list(iter_nested_splits(y, [seed], n_outer, 0.15, groups))

        for fold_i, (_, _, train_idx, val_idx, test_idx) in enumerate(outer_splits):
            trainval_idx = np.sort(np.concatenate([train_idx, val_idx])).astype(int)

            # Inner CV for FIP selection
            inner_splits = make_inner_cv_splits(
                trainval_idx, y, seed, fold_i, n_splits=n_inner, groups=groups,
            )

            # Only FIP candidates (not Qwen) for inner selection
            fip_only = {k: v for k, v in fip_candidates.items() if not k.startswith("Qwen")}
            selected_fip = fast_inner_select(
                X_fc, X_sc, y, inner_splits, fip_only,
                gamma=gamma, lambda_l=lambda_l,
            )
            fip_counts[selected_fip] = fip_counts.get(selected_fip, 0) + 1
            print(f"    Fold {fold_i}: selected={selected_fip}")

            all_inner.append({
                "seed": seed, "fold": fold_i,
                "selected_fip": selected_fip,
            })

            # Evaluate models
            scaler_fc = StandardScaler()
            scaler_sc = StandardScaler()
            X_fc_train = scaler_fc.fit_transform(X_fc[trainval_idx])
            X_sc_train = scaler_sc.fit_transform(X_sc[trainval_idx])
            X_fc_test = scaler_fc.transform(X_fc[test_idx])
            X_sc_test = scaler_sc.transform(X_sc[test_idx])
            y_mean = float(np.mean(y[trainval_idx]))
            y_std = max(float(np.std(y[trainval_idx])), 1e-8)
            y_z = (y[trainval_idx] - y_mean) / y_std

            models = {
                "B0_A4": {"type": "node", "prior": np.ones(N_ROIS)},
                "B1_A2": {"type": "node", "prior": np.ones(N_ROIS)},
                "FIP_selected": {"type": "edge", "edges": fip_candidates[selected_fip]},
                "FIP_shuffled": {"type": "edge", "edges": shuffled_edges},
                "FIP_random": {"type": "edge", "edges": random_edges},
            }
            if unrelated_edges is not None:
                models["FIP_unrelated"] = {"type": "edge", "edges": unrelated_edges}

            # Add individual FIPs
            for fname, fedges in fip_candidates.items():
                if not fname.startswith("Qwen") and f"FIP_{fname}" not in models:
                    models[f"B_{fname}"] = {"type": "edge", "edges": fedges}

            for model_name, model_info in models.items():
                try:
                    if model_info["type"] == "node":
                        cache = build_node_cache(model_info["prior"], gamma)
                    else:
                        cache = get_or_build_cache(model_name, model_info["edges"], gamma)

                    res = evaluate_model(
                        X_fc_train, X_sc_train, y_z, X_fc_test, X_sc_test,
                        y_mean, y_std, y[test_idx],
                        cache, lambda_fc, lambda_sc, lambda_l,
                    )

                    all_split.append({
                        "seed": seed, "fold": fold_i,
                        "model_id": model_name, **{k: v for k, v in res.items() if k != "beta_fc"},
                    })

                    if "FIP_selected" in model_name:
                        bio_corr = float(pearsonr(res["beta_fc"], fip_candidates[selected_fip]).statistic)
                        all_biomarker.append({
                            "seed": seed, "fold": fold_i,
                            "model_id": model_name,
                            "fip_alignment": bio_corr,
                        })

                except Exception as e:
                    all_split.append({
                        "seed": seed, "fold": fold_i,
                        "model_id": model_name, "error": str(e),
                    })

    # Save
    print("\n[4/8] Saving results...")
    split_df = pd.DataFrame(all_split)
    split_df.to_csv(OUTPUT_DIR / "split_metrics.csv", index=False)

    inner_df = pd.DataFrame(all_inner)
    inner_df.to_csv(OUTPUT_DIR / "inner_cv_metrics.csv", index=False)

    bio_df = pd.DataFrame(all_biomarker) if all_biomarker else pd.DataFrame()
    if len(bio_df) > 0:
        bio_df.to_csv(OUTPUT_DIR / "biomarker_metrics.csv", index=False)

    # Summary
    print("\n[5/8] Summary...")
    valid_split = split_df[~split_df.get("pearson", pd.Series(dtype=float)).isna()] if "pearson" in split_df.columns else split_df
    summary_rows = []
    for model_id in split_df["model_id"].unique():
        mdf = split_df[split_df["model_id"] == model_id]
        if "pearson" in mdf.columns and mdf["pearson"].notna().any():
            seed_means = mdf.groupby("seed")[["pearson", "rmse", "mae"]].mean()
            summary_rows.append({
                "model_id": model_id,
                "pearson_mean": float(seed_means["pearson"].mean()),
                "pearson_std": float(seed_means["pearson"].std()) if len(seed_means) > 1 else 0,
                "rmse_mean": float(seed_means["rmse"].mean()),
                "mae_mean": float(seed_means["mae"].mean()),
                "n_folds": len(mdf),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
    print(summary_df[["model_id", "pearson_mean", "rmse_mean"]].to_string(index=False))

    # Paired comparisons
    print("\n[6/8] Paired comparisons...")
    a4_seeds = split_df[split_df["model_id"] == "B0_A4"].groupby("seed")["pearson"].mean()
    comp_rows = []
    for model_id in split_df["model_id"].unique():
        if model_id == "B0_A4" or "error" in str(split_df[split_df["model_id"] == model_id].get("pearson", "")):
            continue
        model_seeds = split_df[split_df["model_id"] == model_id].groupby("seed")["pearson"].mean()
        common = sorted(set(a4_seeds.index) & set(model_seeds.index))
        if common:
            delta = model_seeds[common].values - a4_seeds[common].values
            comp_rows.append({
                "comparison": f"{model_id}_vs_A4",
                "mean_delta": float(delta.mean()),
                "median_delta": float(np.median(delta)),
                "positive_seeds": int(np.sum(delta > 0)),
                "n_seeds": len(common),
            })
    comp_df = pd.DataFrame(comp_rows)
    if len(comp_df) > 0:
        comp_df.to_csv(OUTPUT_DIR / "paired_comparisons.csv", index=False)
        print(comp_df.to_string(index=False))

    # Decision
    print("\n[7/8] Decision...")
    a4_p = summary_df[summary_df["model_id"] == "B0_A4"]["pearson_mean"].values[0] if "B0_A4" in summary_df["model_id"].values else 0

    # Find FIP-selected vs A4 comparison
    fip_sel_row = comp_df[comp_df["comparison"] == "FIP_selected_vs_A4"] if len(comp_df) > 0 else None
    fip_sel_p = summary_df[summary_df["model_id"] == "FIP_selected"]["pearson_mean"].values[0] if "FIP_selected" in summary_df["model_id"].values else 0

    mean_delta = float(fip_sel_row["mean_delta"].iloc[0]) if fip_sel_row is not None and len(fip_sel_row) > 0 else 0
    median_delta = float(fip_sel_row["median_delta"].iloc[0]) if fip_sel_row is not None and len(fip_sel_row) > 0 else 0
    pos_seeds = int(fip_sel_row["positive_seeds"].iloc[0]) if fip_sel_row is not None and len(fip_sel_row) > 0 else 0

    if median_delta >= 0.015 and mean_delta >= 0.012 and pos_seeds >= 3:
        next_step = "full_fluid_fip_10x5"
    elif median_delta >= 0.008 and pos_seeds >= 2:
        next_step = "full_fluid_fip_10x5"
    elif 0.005 <= median_delta < 0.008 and pos_seeds >= 2:
        next_step = "human_review"
    else:
        next_step = "consider_modification_2"

    fip_dist = {k: v / max(sum(fip_counts.values()), 1) for k, v in fip_counts.items()}

    decision = {
        "available_candidates": list(fip_candidates.keys()),
        "best_candidate_descriptive": max(fip_counts, key=fip_counts.get) if fip_counts else "none",
        "inner_selected_candidate_distribution": fip_dist,
        "A4_pearson": float(a4_p),
        "original_qwen_A3_pearson": float(
            summary_df[summary_df["model_id"] == "B_Qwen_Fluid"]["pearson_mean"].iloc[0]
            if "B_Qwen_Fluid" in summary_df["model_id"].values else 0
        ),
        "fip_selected_pearson": float(fip_sel_p),
        "mean_delta_pearson_vs_A4": mean_delta,
        "median_delta_pearson_vs_A4": median_delta,
        "positive_seeds_vs_A4": pos_seeds,
        "mean_delta_rmse_vs_A4": 0.0,
        "mean_delta_mae_vs_A4": 0.0,
        "matched_beats_shuffled": False,
        "matched_beats_random": False,
        "matched_beats_unrelated": False,
        "fip_biomarker_alignment": float(bio_df["fip_alignment"].mean()) if len(bio_df) > 0 else 0,
        "recommended_next_step": next_step,
    }

    print(json.dumps({k: v for k, v in decision.items() if k != "inner_selected_candidate_distribution"}, indent=2))

    with open(OUTPUT_DIR / "fip_decision.json", "w") as f:
        json.dump(decision, f, indent=2)

    elapsed = time.time() - t0
    metadata = {
        "seeds": seeds, "n_outer_folds": n_outer, "n_inner_folds": n_inner,
        "target": "fluid_intelligence", "elapsed_seconds": float(elapsed),
        "fip_selection_distribution": fip_dist,
        "recommendation": next_step,
    }
    with open(OUTPUT_DIR / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(OUTPUT_DIR / "COMPLETE", "w") as f:
        f.write("COMPLETE")

    print(f"\n[8/8] COMPLETE in {elapsed:.0f}s")
    print(f"  Results: {OUTPUT_DIR}")
    print(f"  Recommendation: {next_step}")


if __name__ == "__main__":
    main()
