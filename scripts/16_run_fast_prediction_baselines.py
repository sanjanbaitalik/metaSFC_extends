#!/usr/bin/env python3
"""Fast, leakage-free connectome prediction baselines for the AAAI study.

Implements four regression baselines using the exact repeated outer-fold and
inner-validation protocol used by E0--E10:

B0: training-fold mean predictor
B1: FC upper-triangle Ridge regression
B2: SC upper-triangle Ridge regression
B3: concatenated FC+SC upper-triangle Ridge regression

The ridge penalty is selected using only the inner validation split. Feature
standardization and target standardization are fitted on the inner training
partition only. Predictions are reported in the original target units.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


METHODS = {
    "B0": ("Mean Predictor", "mean"),
    "B1": ("FC Ridge", "fc"),
    "B2": ("SC Ridge", "sc"),
    "B3": ("FC+SC Ridge", "fusion"),
}


def load_yaml(path: str | Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def quantile_bins(y: np.ndarray, n_bins: int = 5):
    try:
        return pd.qcut(y, q=min(n_bins, len(y)), labels=False, duplicates="drop")
    except Exception:
        return None


def make_inner_split(
    trainval_idx: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    seed: int,
    groups: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr_local, va_local = next(
            splitter.split(trainval_idx, y[trainval_idx], groups[trainval_idx])
        )
        return trainval_idx[tr_local], trainval_idx[va_local]

    stratify = quantile_bins(y[trainval_idx])
    try:
        tr, va = train_test_split(
            trainval_idx,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        tr, va = train_test_split(
            trainval_idx,
            test_size=val_fraction,
            random_state=seed,
            shuffle=True,
        )
    return np.asarray(tr), np.asarray(va)


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) > 1 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        r = float(pearsonr(y_true, y_pred).statistic)
    else:
        r = 0.0
    return {
        "pearson": r if np.isfinite(r) else 0.0,
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def upper_triangle_features(mats: np.ndarray) -> np.ndarray:
    if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
        raise ValueError(f"Expected [subjects, ROI, ROI], got {mats.shape}")
    iu = np.triu_indices(mats.shape[1], k=1)
    return np.asarray(mats[:, iu[0], iu[1]], dtype=np.float32)


def build_feature_blocks(fc: np.ndarray, sc: np.ndarray) -> Dict[str, np.ndarray]:
    x_fc = upper_triangle_features(fc)
    x_sc = upper_triangle_features(sc)
    return {
        "fc": x_fc,
        "sc": x_sc,
        "fusion": np.concatenate([x_fc, x_sc], axis=1),
    }


def _dual_kernel_fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linear Ridge in dual form; efficient when subjects << features."""
    scale = float(max(1, x_train.shape[1]))
    kernel_train = (x_train @ x_train.T) / scale
    eigvals, eigvecs = np.linalg.eigh(kernel_train)
    projected = eigvecs.T @ y_train
    dual = eigvecs @ (projected / (eigvals + float(alpha) + 1e-10))
    kernel_eval = (x_eval @ x_train.T) / scale
    return kernel_eval @ dual


def fit_predict_ridge(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    alphas: Iterable[float],
) -> Tuple[np.ndarray, float, float]:
    """Nested linear Ridge using a fast dual/kernel implementation."""
    scaler = StandardScaler(copy=True)
    x_train = scaler.fit_transform(x[train_idx]).astype(np.float64, copy=False)
    x_val = scaler.transform(x[val_idx]).astype(np.float64, copy=False)

    y_mean = float(np.mean(y[train_idx]))
    y_std = float(np.std(y[train_idx]))
    if y_std < 1e-8:
        y_std = 1.0
    y_train_z = (y[train_idx] - y_mean) / y_std

    # Compute the eigendecomposition once and reuse it for the alpha grid.
    scale = float(max(1, x_train.shape[1]))
    kernel_train = (x_train @ x_train.T) / scale
    eigvals, eigvecs = np.linalg.eigh(kernel_train)
    projected = eigvecs.T @ y_train_z
    kernel_val = (x_val @ x_train.T) / scale

    best_alpha = None
    best_rmse = float("inf")
    for alpha in alphas:
        dual = eigvecs @ (projected / (eigvals + float(alpha) + 1e-10))
        val_pred = (kernel_val @ dual) * y_std + y_mean
        val_rmse = prediction_metrics(y[val_idx], val_pred)["rmse"]
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            best_alpha = float(alpha)

    assert best_alpha is not None
    # Refit on train+validation with scalers and target normalization fitted only
    # on that outer-fold development partition.
    fit_idx = np.concatenate([train_idx, val_idx])
    final_scaler = StandardScaler(copy=True)
    x_fit = final_scaler.fit_transform(x[fit_idx]).astype(np.float64, copy=False)
    x_test = final_scaler.transform(x[test_idx]).astype(np.float64, copy=False)
    y_fit_mean = float(np.mean(y[fit_idx]))
    y_fit_std = float(np.std(y[fit_idx]))
    if y_fit_std < 1e-8:
        y_fit_std = 1.0
    pred_z = _dual_kernel_fit_predict(
        x_fit, (y[fit_idx] - y_fit_mean) / y_fit_std, x_test, best_alpha
    )
    pred = pred_z * y_fit_std + y_fit_mean
    return np.asarray(pred), best_alpha, best_rmse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="configs/aaai/fast_prediction_baselines.yaml",
        help="Baseline YAML configuration.",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    # Avoid BLAS oversubscription on DGX/desktop systems. The returned controller
    # keeps the limit active for the lifetime of this process.
    thread_controller = threadpool_limits(limits=int(cfg.get("n_threads", 4)))
    out_dir = Path(cfg.get("output_dir", "outputs/aaai/prediction_baselines"))
    complete = out_dir / "COMPLETE"
    if complete.exists() and not args.overwrite:
        print(f"Complete baseline result exists at {out_dir}; use --overwrite to rerun.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)

    data_cfg = cfg["data"]
    fc = np.load(data_cfg["fc_path"]).astype(np.float32)
    sc = np.load(data_cfg["sc_path"]).astype(np.float32)
    y = np.load(data_cfg["y_path"]).astype(np.float64).reshape(-1)
    if fc.shape != sc.shape or len(y) != len(fc):
        raise ValueError(f"Data mismatch: FC={fc.shape}, SC={sc.shape}, y={y.shape}")

    subject_ids = np.arange(len(y)).astype(str)
    subjects_path = data_cfg.get("subjects_path")
    if subjects_path and Path(subjects_path).exists():
        sdf = pd.read_csv(subjects_path)
        col = "subject" if "subject" in sdf.columns else "Subject"
        if len(sdf) == len(y):
            subject_ids = sdf[col].astype(str).to_numpy()

    groups = None
    groups_path = data_cfg.get("groups_path")
    if groups_path:
        groups = np.load(groups_path, allow_pickle=True).reshape(-1)
        if len(groups) != len(y):
            raise ValueError("Group count does not match subject count")

    features = build_feature_blocks(fc, sc)
    selected_methods = cfg.get("methods", list(METHODS))
    unknown = set(selected_methods) - set(METHODS)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")

    seeds = [int(s) for s in cfg.get("seeds", list(range(10)))]
    n_folds = int(cfg.get("n_folds", 5))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    alphas = [float(a) for a in cfg.get("ridge_alphas", [0.01, 0.1, 1, 10, 100, 1000])]

    rows = []
    for seed in seeds:
        if groups is not None:
            splitter = GroupKFold(n_splits=n_folds)
            split_iter = splitter.split(np.arange(len(y)), y, groups)
        else:
            splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(np.arange(len(y)))

        for fold, (trainval_idx, test_idx) in enumerate(split_iter):
            trainval_idx = np.asarray(trainval_idx)
            test_idx = np.asarray(test_idx)
            split_seed = seed * 1000 + fold
            train_idx, val_idx = make_inner_split(
                trainval_idx, y, val_fraction, split_seed, groups=groups
            )
            split_id = f"seed{seed:02d}_fold{fold:02d}"

            for method_id in selected_methods:
                method_name, kind = METHODS[method_id]
                started = time.time()
                if kind == "mean":
                    # Match the final refit convention used by the ridge baselines.
                    fit_idx = np.concatenate([train_idx, val_idx])
                    pred = np.full(len(test_idx), np.mean(y[fit_idx]), dtype=float)
                    best_alpha = np.nan
                    best_val_rmse = np.nan
                else:
                    pred, best_alpha, best_val_rmse = fit_predict_ridge(
                        features[kind], y, train_idx, val_idx, test_idx, alphas
                    )

                metrics = prediction_metrics(y[test_idx], pred)
                row = {
                    "experiment_id": method_id,
                    "experiment_name": method_name,
                    "method_family": "fast_prediction_baseline",
                    "feature_source": kind,
                    "seed": seed,
                    "fold": fold,
                    "split_id": split_id,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                    "best_alpha": best_alpha,
                    "best_val_rmse": best_val_rmse,
                    "runtime_seconds": time.time() - started,
                    "group_aware": groups is not None,
                    **metrics,
                }
                rows.append(row)
                pd.DataFrame(
                    {
                        "subject_index": test_idx,
                        "subject_id": subject_ids[test_idx],
                        "target": y[test_idx],
                        "prediction": pred,
                        "seed": seed,
                        "fold": fold,
                        "experiment_id": method_id,
                    }
                ).to_csv(out_dir / "predictions" / f"{method_id}_{split_id}.csv", index=False)
                print(method_id, split_id, metrics, "alpha=", best_alpha)

            pd.DataFrame(rows).to_csv(out_dir / "prediction_baselines_split_metrics.csv", index=False)

    split_df = pd.DataFrame(rows)
    summary_rows = []
    for method_id, g in split_df.groupby("experiment_id", sort=False):
        summary_rows.append(
            {
                "ID": method_id,
                "Method": g["experiment_name"].iloc[0],
                "Feature Source": g["feature_source"].iloc[0],
                "N": int(round(g["n_test"].sum() / g["seed"].nunique())),
                "Seeds": int(g["seed"].nunique()),
                "Folds": int(g["fold"].nunique()),
                "Pearson Mean": g["pearson"].mean(),
                "Pearson Std": g["pearson"].std(ddof=1),
                "RMSE Mean": g["rmse"].mean(),
                "RMSE Std": g["rmse"].std(ddof=1),
                "MAE Mean": g["mae"].mean(),
                "MAE Std": g["mae"].std(ddof=1),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "prediction_baselines_summary.csv", index=False)

    latex = summary[["ID", "Method", "Pearson Mean", "Pearson Std", "RMSE Mean", "RMSE Std", "MAE Mean", "MAE Std"]].copy()
    latex["Pearson $\\uparrow$"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(latex.pop("Pearson Mean"), latex.pop("Pearson Std"))]
    latex["RMSE $\\downarrow$"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(latex.pop("RMSE Mean"), latex.pop("RMSE Std"))]
    latex["MAE $\\downarrow$"] = [f"{m:.3f} $\\pm$ {s:.3f}" for m, s in zip(latex.pop("MAE Mean"), latex.pop("MAE Std"))]
    (out_dir / "prediction_baselines_summary.tex").write_text(
        latex.to_latex(index=False, escape=False), encoding="utf-8"
    )

    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": cfg,
                "n_subjects": len(y),
                "n_rois": fc.shape[1],
                "n_evaluations_per_method": len(seeds) * n_folds,
                "note": "All preprocessing and hyperparameter selection are nested within the outer split.",
            },
            f,
            indent=2,
        )
    complete.write_text("ok\n", encoding="utf-8")
    print(f"Saved fast prediction baselines to {out_dir}")


if __name__ == "__main__":
    main()
