#!/usr/bin/env python3
"""Perturbation-based explanation faithfulness evaluation.

For every selected experiment and repeated outer split, the model is retrained
using the same nested-CV protocol as the primary AAAI experiments. Salient ROIs
are selected from the INNER VALIDATION set only. On the held-out outer test set,
all FC and SC connections incident to selected ROIs are removed. The resulting
prediction degradation is compared against bottom-k and repeated random-k masks.

A faithful explanation should satisfy:
  degradation(top-k learned ROIs) > degradation(random-k ROIs)
  degradation(top-k learned ROIs) > degradation(bottom-k learned ROIs)

The default run uses E0, E1 and E10 with three seeds, which is designed to
finish within roughly one day on the target GPU. More configs/seeds can be added
from the command line.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_1samp, wilcoxon
from torch.utils.data import DataLoader, Dataset

from metascfc.config import load_config
from metascfc.data.connectome_dataset import ConnectomeDataset, load_fc_sc_arrays
from metascfc.experiments import PriorGuidedTrainer
from metascfc.seed import set_seed


def load_primary_runner():
    path = Path(__file__).with_name("07_run_aaai_experiment.py")
    spec = importlib.util.spec_from_file_location("metascfc_primary_runner", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import primary runner from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module




class PerturbedGraphDataset(Dataset):
    """Graph-level ROI ablation that preserves the original thresholded graph.

    The selected ROI rows/columns are zeroed in connectivity-profile node
    features and all graph edges incident to those ROIs are removed. Other
    edges are left unchanged, avoiding the confound of recomputing graph
    thresholds after every perturbation.
    """

    def __init__(self, source: Dataset, roi_indices: Iterable[int], mode: str = "both"):
        self.source = source
        self.roi_indices = torch.as_tensor(sorted(set(int(x) for x in roi_indices)), dtype=torch.long)
        self.mode = mode

    def __len__(self):
        return len(self.source)

    def _mask_modality(self, item: Dict, prefix: str) -> None:
        x_key = f"{prefix}_x"
        ei_key = f"{prefix}_edge_index"
        ew_key = f"{prefix}_edge_weight"
        x = item[x_key].clone()
        idx = self.roi_indices
        x[idx, :] = 0.0
        if x.dim() == 2 and x.shape[1] >= int(idx.max().item()) + 1:
            x[:, idx] = 0.0
        item[x_key] = x

        edge_index = item[ei_key].clone()
        incident = torch.isin(edge_index[0], idx) | torch.isin(edge_index[1], idx)
        keep = ~incident
        item[ei_key] = edge_index[:, keep]
        if ew_key in item and item[ew_key] is not None:
            item[ew_key] = item[ew_key].clone()[keep]

    def __getitem__(self, index):
        original = self.source[index]
        item = {key: (value.clone() if torch.is_tensor(value) else value) for key, value in original.items()}
        if self.mode in {"both", "fc"}:
            self._mask_modality(item, "fc")
        if self.mode in {"both", "sc"}:
            self._mask_modality(item, "sc")
        return item


def mask_connectomes(
    fc: np.ndarray,
    sc: np.ndarray,
    roi_indices: Iterable[int],
    mode: str = "both",
) -> tuple[np.ndarray, np.ndarray]:
    """Remove all edges incident to selected ROIs in FC and/or SC."""
    roi_indices = np.unique(np.asarray(list(roi_indices), dtype=int))
    fc_out = np.array(fc, copy=True)
    sc_out = np.array(sc, copy=True)
    if mode in {"both", "fc"}:
        fc_out[:, roi_indices, :] = 0.0
        fc_out[:, :, roi_indices] = 0.0
    if mode in {"both", "sc"}:
        sc_out[:, roi_indices, :] = 0.0
        sc_out[:, :, roi_indices] = 0.0
    return fc_out, sc_out


def make_test_loader(R, base_dataset, test_idx, cfg, roi_indices=None, mode="both"):
    dataset = base_dataset
    if roi_indices is not None:
        dataset = PerturbedGraphDataset(base_dataset, roi_indices, mode=mode)
    return R.make_loader(dataset, np.asarray(test_idx), cfg, shuffle=False)


def positive_degradation(original: Dict[str, float], perturbed: Dict[str, float], metric: str) -> float:
    if metric in {"rmse", "mae"}:
        return float(perturbed[metric] - original[metric])
    if metric == "pearson":
        return float(original[metric] - perturbed[metric])
    raise ValueError(metric)


def bootstrap_ci(values: np.ndarray, n_bootstrap=20000, seed=42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    boots = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        boots[i] = rng.choice(values, len(values), replace=True).mean()
    return np.quantile(boots, [0.025, 0.975])




def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted

def safe_wilcoxon_greater(values: np.ndarray):
    try:
        result = wilcoxon(values, alternative="greater", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return 0.0, 1.0


def load_reference_roi(cfg: Dict, n_rois: int) -> Optional[np.ndarray]:
    ref = cfg.get("reference_prior", {}) or {}
    path = ref.get("roi_prior_path")
    if not path or not Path(path).exists():
        return None
    arr = pd.read_csv(path).sort_values("roi_index")["prior_score"].to_numpy(np.float32)
    if arr.shape != (n_rois,):
        raise ValueError(f"Reference ROI prior has shape {arr.shape}, expected {(n_rois,)}")
    return arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/aaai/E0_baseline.yaml",
            "configs/aaai/E1_node_true.yaml",
            "configs/aaai/E10_node_unrelated_visual.yaml",
        ],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--random_repeats", type=int, default=20)
    ap.add_argument("--mask_mode", choices=["both", "fc", "sc"], default="both")
    ap.add_argument("--out", default="outputs/aaai/faithfulness")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    R = load_primary_runner()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "per_experiment").mkdir(exist_ok=True)

    all_rows = []
    long_rows = []
    for config_path in args.configs:
        cfg = load_config(config_path)
        experiment_id = str(cfg.get("experiment_id", Path(config_path).stem))
        exp_out = out / "per_experiment" / experiment_id
        exp_out.mkdir(parents=True, exist_ok=True)
        complete = exp_out / "COMPLETE"
        if complete.exists() and not args.overwrite:
            print(f"Skipping completed faithfulness run: {experiment_id}")
            existing = exp_out / "faithfulness_split_metrics.csv"
            if existing.exists():
                all_rows.extend(pd.read_csv(existing).to_dict("records"))
            existing_long = exp_out / "faithfulness_long.csv"
            if existing_long.exists():
                long_rows.extend(pd.read_csv(existing_long).to_dict("records"))
            continue

        device_name = cfg.get("device", "auto")
        device = torch.device(
            "cuda" if device_name == "auto" and torch.cuda.is_available()
            else ("cpu" if device_name == "auto" else device_name)
        )
        fc, sc, y = load_fc_sc_arrays(
            cfg["data"]["fc_path"], cfg["data"]["sc_path"], cfg["data"]["y_path"]
        )
        y = np.asarray(y).reshape(-1)
        n_rois = int(cfg.get("roi_num", fc.shape[1]))
        base_dataset = R.GraphDataset(
            ConnectomeDataset(fc, sc, y),
            n_rois,
            cfg.get("top_percent_fc", 10.0),
            cfg.get("top_percent_sc", 10.0),
            cache=cfg.get("cache_graphs", True),
        )
        roi_prior, module_prior, edge_prior, mapping = R.load_priors(cfg, n_rois)
        reference_roi = load_reference_roi(cfg, n_rois)

        groups = None
        groups_path = cfg.get("data", {}).get("groups_path")
        if groups_path:
            groups = np.load(groups_path, allow_pickle=True).reshape(-1)

        n_folds = int(cfg.get("n_folds", 5))
        val_fraction = float(cfg.get("val_fraction", 0.15))
        selection_metric = cfg.get("selection_metric", "rmse")
        patience = int(cfg.get("early_stopping_patience", 20))
        min_epochs = int(cfg.get("min_epochs", 20))
        exp_rows = []
        exp_long = []

        for seed in args.seeds:
            if groups is not None:
                splitter = R.GroupKFold(n_splits=n_folds)
                split_iter = splitter.split(np.arange(len(y)), y, groups)
            else:
                splitter = R.KFold(n_splits=n_folds, shuffle=True, random_state=seed)
                split_iter = splitter.split(np.arange(len(y)))

            for fold, (trainval_idx, test_idx) in enumerate(split_iter):
                started = time.time()
                split_seed = seed * 1000 + fold
                set_seed(split_seed)
                train_idx, val_idx = R.make_inner_split(
                    np.asarray(trainval_idx), y, cfg.get("task", "regression"),
                    val_fraction, split_seed, groups=groups,
                )
                train_loader = R.make_loader(base_dataset, train_idx, cfg, shuffle=True)
                val_loader = R.make_loader(base_dataset, val_idx, cfg, shuffle=False)

                model = R.build_model(cfg, n_rois).to(device)
                trainer = PriorGuidedTrainer(model, cfg, device)
                trainer.set_priors(roi_prior, module_prior, edge_prior, mapping)
                if cfg.get("task") == "regression" and cfg.get("standardize_labels_within_fold", True):
                    train_targets = y[train_idx].astype(float)
                    trainer.set_target_scaler(train_targets.mean(), train_targets.std())
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=trainer.learning_rate, weight_decay=trainer.weight_decay
                )

                best = float("inf") if selection_metric in {"rmse", "mae", "loss"} else -float("inf")
                best_state = None
                best_epoch = -1
                wait = 0
                for epoch in range(trainer.n_epochs):
                    train_loss = trainer.train_epoch(train_loader, optimizer)
                    val_result = trainer.evaluate(val_loader)
                    value = float(val_result["metrics"].get(selection_metric, train_loss))
                    if R.is_better(selection_metric, value, best):
                        best = value
                        best_state = copy.deepcopy(model.state_dict())
                        best_epoch = epoch
                        wait = 0
                    else:
                        wait += 1
                    if epoch + 1 >= min_epochs and wait >= patience:
                        break
                if best_state is not None:
                    model.load_state_dict(best_state)

                # Salient ROIs are derived only from the inner validation set.
                val_result = trainer.evaluate(val_loader)
                val_aux = R.compute_aux(val_result, trainer, cfg, reference=None)
                saliency = val_aux.get("node_saliency")
                if saliency is None:
                    raise RuntimeError(f"No validation node saliency for {experiment_id}")
                saliency = np.asarray(saliency)
                k = min(args.topk, len(saliency))
                top_idx = np.argsort(saliency)[-k:]
                bottom_idx = np.argsort(saliency)[:k]
                prior_idx = np.argsort(reference_roi)[-k:] if reference_roi is not None else None

                original = trainer.evaluate(
                    make_test_loader(R, base_dataset, np.asarray(test_idx), cfg)
                )["metrics"]
                top = trainer.evaluate(
                    make_test_loader(R, base_dataset, np.asarray(test_idx), cfg, top_idx, args.mask_mode)
                )["metrics"]
                bottom = trainer.evaluate(
                    make_test_loader(R, base_dataset, np.asarray(test_idx), cfg, bottom_idx, args.mask_mode)
                )["metrics"]
                prior_top = None
                if prior_idx is not None:
                    prior_top = trainer.evaluate(
                        make_test_loader(R, base_dataset, np.asarray(test_idx), cfg, prior_idx, args.mask_mode)
                    )["metrics"]

                random_metrics = []
                rng = np.random.default_rng(split_seed + 9173)
                for repeat in range(args.random_repeats):
                    random_idx = rng.choice(n_rois, size=k, replace=False)
                    result = trainer.evaluate(
                        make_test_loader(R, base_dataset, np.asarray(test_idx), cfg, random_idx, args.mask_mode)
                    )["metrics"]
                    random_metrics.append(result)
                    exp_long.append(
                        {
                            "experiment_id": experiment_id,
                            "seed": seed,
                            "fold": fold,
                            "condition": "random",
                            "repeat": repeat,
                            **{m: float(result[m]) for m in ("pearson", "rmse", "mae")},
                        }
                    )
                random_mean = {
                    m: float(np.mean([x[m] for x in random_metrics]))
                    for m in ("pearson", "rmse", "mae")
                }
                random_std = {
                    m: float(np.std([x[m] for x in random_metrics], ddof=1))
                    for m in ("pearson", "rmse", "mae")
                }

                row = {
                    "experiment_id": experiment_id,
                    "experiment_name": cfg.get("experiment_name", experiment_id),
                    "seed": seed,
                    "fold": fold,
                    "split_id": f"seed{seed:02d}_fold{fold:02d}",
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "n_test": len(test_idx),
                    "best_epoch": best_epoch + 1,
                    "best_val_metric": best,
                    "topk": k,
                    "mask_mode": args.mask_mode,
                    "random_repeats": args.random_repeats,
                    "runtime_seconds": time.time() - started,
                }
                for metric in ("pearson", "rmse", "mae"):
                    row[f"original_{metric}"] = float(original[metric])
                    row[f"top_{metric}"] = float(top[metric])
                    row[f"bottom_{metric}"] = float(bottom[metric])
                    row[f"random_{metric}_mean"] = random_mean[metric]
                    row[f"random_{metric}_std"] = random_std[metric]
                    if prior_top is not None:
                        row[f"prior_top_{metric}"] = float(prior_top[metric])
                    row[f"delta_{metric}_top"] = positive_degradation(original, top, metric)
                    row[f"delta_{metric}_bottom"] = positive_degradation(original, bottom, metric)
                    row[f"delta_{metric}_random"] = positive_degradation(original, random_mean, metric)
                    if prior_top is not None:
                        row[f"delta_{metric}_prior_top"] = positive_degradation(original, prior_top, metric)
                    row[f"gap_{metric}_top_vs_random"] = row[f"delta_{metric}_top"] - row[f"delta_{metric}_random"]
                    row[f"gap_{metric}_top_vs_bottom"] = row[f"delta_{metric}_top"] - row[f"delta_{metric}_bottom"]

                for condition, values in (("original", original), ("top", top), ("bottom", bottom)):
                    exp_long.append(
                        {
                            "experiment_id": experiment_id,
                            "seed": seed,
                            "fold": fold,
                            "condition": condition,
                            "repeat": -1,
                            **{m: float(values[m]) for m in ("pearson", "rmse", "mae")},
                        }
                    )
                if prior_top is not None:
                    exp_long.append(
                        {
                            "experiment_id": experiment_id,
                            "seed": seed,
                            "fold": fold,
                            "condition": "reference_prior_top",
                            "repeat": -1,
                            **{m: float(prior_top[m]) for m in ("pearson", "rmse", "mae")},
                        }
                    )

                exp_rows.append(row)
                pd.DataFrame(exp_rows).to_csv(exp_out / "faithfulness_split_metrics.csv", index=False)
                pd.DataFrame(exp_long).to_csv(exp_out / "faithfulness_long.csv", index=False)
                print(experiment_id, row["split_id"], {
                    "delta_rmse_top": row["delta_rmse_top"],
                    "delta_rmse_random": row["delta_rmse_random"],
                    "gap": row["gap_rmse_top_vs_random"],
                })

        complete.write_text("ok\n", encoding="utf-8")
        all_rows.extend(exp_rows)
        long_rows.extend(exp_long)

    split_df = pd.DataFrame(all_rows)
    long_df = pd.DataFrame(long_rows)
    split_df.to_csv(out / "faithfulness_all_split_metrics.csv", index=False)
    long_df.to_csv(out / "faithfulness_all_long.csv", index=False)

    summary_rows = []
    for experiment_id, g in split_df.groupby("experiment_id"):
        row = {
            "ID": experiment_id,
            "Method": g["experiment_name"].iloc[0],
            "Seeds": g["seed"].nunique(),
            "Folds": g["fold"].nunique(),
            "Top-k": int(g["topk"].iloc[0]),
        }
        for col in [c for c in g.columns if c.startswith("delta_") or c.startswith("gap_")]:
            row[f"{col} Mean"] = g[col].mean()
            row[f"{col} Std"] = g[col].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "faithfulness_summary.csv", index=False)

    # Paper-ready seed-level inference: average folds within each seed first.
    gap_cols = [c for c in split_df.columns if c.startswith("gap_")]
    seed_df = split_df.groupby(["experiment_id", "seed"], as_index=False)[gap_cols].mean()
    seed_df.to_csv(out / "faithfulness_seed_level_metrics.csv", index=False)
    stat_rows = []
    for experiment_id, g in seed_df.groupby("experiment_id"):
        for metric in gap_cols:
            values = g[metric].dropna().to_numpy(float)
            if len(values) < 3:
                continue
            t = ttest_1samp(values, popmean=0.0, alternative="greater")
            w_stat, w_p = safe_wilcoxon_greater(values)
            lo, hi = bootstrap_ci(values)
            stat_rows.append(
                {
                    "ID": experiment_id,
                    "contrast": metric,
                    "analysis_unit": "seed_mean_over_5_folds",
                    "n_seeds": len(values),
                    "mean_gap": values.mean(),
                    "std_gap": values.std(ddof=1),
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "one_sample_t": float(t.statistic),
                    "one_sample_t_p_greater": float(t.pvalue),
                    "wilcoxon_W": w_stat,
                    "wilcoxon_p_greater": w_p,
                }
            )
    stats = pd.DataFrame(stat_rows)
    if not stats.empty:
        stats["one_sample_t_p_holm"] = holm_adjust(stats["one_sample_t_p_greater"].to_numpy())
        stats["wilcoxon_p_holm"] = holm_adjust(stats["wilcoxon_p_greater"].to_numpy())
        stats["significant_wilcoxon_holm_005"] = stats["wilcoxon_p_holm"] < 0.05
        stats["ci_excludes_zero"] = (stats["bootstrap95_low"] > 0) | (stats["bootstrap95_high"] < 0)
    stats.to_csv(out / "faithfulness_seed_level_tests.csv", index=False)
    (out / "faithfulness_seed_level_tests.tex").write_text(
        stats.to_latex(index=False, escape=False, float_format="%.4f"), encoding="utf-8"
    )
    (out / "faithfulness_summary.tex").write_text(
        summary.to_latex(index=False, escape=False, float_format="%.4f"), encoding="utf-8"
    )
    print(f"Saved faithfulness outputs to {out}")


if __name__ == "__main__":
    main()
