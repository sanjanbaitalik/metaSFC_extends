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

from metascfc.benchmark_utils import (
    choose_device,
    iter_nested_splits,
    load_connectomes,
    prediction_metrics,
)
from metascfc.models.iclr_backbones import (
    MetaGATConfig,
    load_split_node_saliency,
    refit_krr_predictor,
    refit_meta_gat_predictor,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    NetworkConstrainedRidge,
    build_edge_laplacian,
)

ICLR_FAITHFULNESS_SOURCE = Path("outputs/aaai/faithfulness_iclr/per_method")
NCR_OUTPUT_DIR = Path("outputs/aaai/network_constrained_ridge")
ICLR_METHOD_PREFIXES = ("NCR_", "M2_", "M3_")


def _load_43_helpers():
    path = Path(__file__).with_name("43_run_iclr_faithfulness.py")
    spec = importlib.util.spec_from_file_location("iclr_faithfulness_43", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import scripts/43 helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refit_iclr_masked(
    cfg: Dict,
    method_id: str,
    method_cfg: Dict,
    seeds: list[int],
    args,
) -> None:
    """Refit + mask-evaluate M2/M3 variants that scripts/43 did not cover
    (the shuffled-prior controls), writing the same per-method format that
    the conversion branch reads.  The reference prior is the WM meta-analysis
    map (uniform across all methods, identical to the AAAI E* protocol); the
    fixed random map is kept as the extra control."""
    F = _load_43_helpers()
    src = ICLR_FAITHFULNESS_SOURCE / method_id
    (src / "attention").mkdir(parents=True, exist_ok=True)
    (src / "krr_saliency").mkdir(parents=True, exist_ok=True)
    split_csv = src / "faithfulness_split_metrics.csv"
    long_csv = src / "faithfulness_long.csv"

    fc, sc, y, subject_ids, groups = load_connectomes(cfg["data"])
    y = np.asarray(y).reshape(-1)
    n_rois = int(fc.shape[1])
    n_folds = int(cfg.get("n_folds", 5))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    topk = int(args.topk)
    n_random = int(args.random_repeats)

    wm_prior = F.load_roi_prior("outputs/priors/working_memory/aal116/roi_prior.csv", n_rois)
    prior_true_idx = F.top_k_indices(wm_prior, topk)
    random_prior = F.load_roi_prior(
        "outputs/priors/random_prior/aal116/roi_prior.csv", n_rois
    )
    prior_random_idx = F.top_k_indices(random_prior, topk)

    fixed = cfg
    if method_id.startswith("M2_"):
        family = "meta_gat"
        headline = pd.read_csv(Path(cfg["output_dir"]) / "split_metrics.csv")
        headline = headline[headline.method_id == method_id].set_index(["seed", "fold"])
        device = choose_device(str(cfg.get("device", "auto")))
        saliency_dir = Path(cfg["output_dir"]) / "saliency" / method_id
        biomarker_dir = src / "attention"
        biomarker_key = "node_attention_mass"
    else:
        family = "kernel_ridge"
        headline = pd.read_csv(Path(cfg["output_dir"]) / "split_metrics.csv")
        headline = headline[headline.method_id == method_id].set_index(["seed", "fold"])
        device = None
        saliency_dir = Path(cfg["output_dir"]) / "saliency" / method_id
        biomarker_dir = src / "krr_saliency"
        biomarker_key = "node_saliency"

    rows = (
        []
        if not split_csv.exists()
        else pd.read_csv(split_csv).to_dict("records")
    )
    completed = {(int(r["seed"]), int(r["fold"])) for r in rows}
    long_rows = (
        []
        if not long_csv.exists()
        else pd.read_csv(long_csv).to_dict("records")
    )
    for seed, fold, train_idx, val_idx, test_idx in iter_nested_splits(
        y, seeds, n_folds, val_fraction, groups
    ):
        if (seed, fold) in completed:
            continue
        started = time.time()
        split_seed = seed * 1000 + fold
        fit_idx = np.concatenate([train_idx, val_idx])
        rec = headline.loc[(seed, fold)]
        saliency = load_split_node_saliency(saliency_dir, seed, fold)
        rng = np.random.default_rng(split_seed + 9173)

        if family == "meta_gat":
            predictor = refit_meta_gat_predictor(
                fc, sc, y, fit_idx,
                MetaGATConfig(
                    hidden=int(rec["best_hidden"]),
                    heads1=int(fixed.get("heads1", 4)),
                    heads2=int(fixed.get("heads2", 1)),
                    dropout=float(rec["best_dropout"]),
                    gamma_init=float(fixed.get("gamma_init", 1.0)),
                    learning_rate=float(rec["best_learning_rate"]),
                    weight_decay=float(fixed.get("weight_decay", 1e-4)),
                    epochs=int(fixed.get("epochs", 60)),
                    patience=int(fixed.get("patience", 15)),
                    min_epochs=int(fixed.get("min_epochs", 10)),
                    grad_clip=float(fixed.get("grad_clip", 5.0)),
                ),
                saliency, device,
                n_epochs=int(rec["best_epoch"]),
                top_percent=float(fixed.get("top_percent_sc", 10.0)),
                seed=split_seed,
            )
            row, per_cond = F.evaluate_masks_meta_gat(
                predictor, fc[test_idx], sc[test_idx], y[test_idx],
                F.top_k_indices(saliency, topk), np.argsort(saliency)[:topk],
                prior_true_idx, prior_random_idx, n_random, rng, n_rois,
            )
            np.savez(
                biomarker_dir / f"seed{seed:02d}_fold{fold:02d}.npz",
                node_attention_mass=predictor.attention_mass(fc[fit_idx], sc[fit_idx]),
            )
        else:
            predictor = refit_krr_predictor(
                fc, sc, y, fit_idx, saliency,
                alpha=float(rec["best_alpha"]),
                gamma=float(rec["best_gamma"]),
                gate_mode=str(fixed.get("gate_mode", "product")),
            )
            row, per_cond = F.evaluate_masks_krr(
                predictor, fc[test_idx], sc[test_idx], y[test_idx],
                F.top_k_indices(saliency, topk), np.argsort(saliency)[:topk],
                prior_true_idx, prior_random_idx, n_random, rng, n_rois,
            )
            np.savez(
                biomarker_dir / f"seed{seed:02d}_fold{fold:02d}.npz",
                node_saliency=predictor.gradient_node_saliency(fc[fit_idx], sc[fit_idx]),
            )

        rows.append({
            "method_id": method_id,
            "method_name": method_cfg["name"],
            "method_family": family,
            "seed": seed, "fold": fold,
            "split_id": f"seed{seed:02d}_fold{fold:02d}",
            "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
            "topk": topk, "random_repeats": n_random,
            "runtime_seconds": time.time() - started,
            **row,
        })
        for item in per_cond:
            long_rows.append({
                "method_id": method_id, "seed": seed, "fold": fold,
                **item,
            })
        pd.DataFrame(rows).to_csv(split_csv, index=False)
        pd.DataFrame(long_rows).to_csv(long_csv, index=False)
        print(
            f"[refit] {method_id} seed{seed:02d}_fold{fold:02d} "
            f"delta_true_top={row['delta_rmse_prior_true_top']:+.3f} "
            f"delta_random_top={row['delta_rmse_prior_random_top']:+.3f}",
            flush=True,
        )
    print(f"[refit] {method_id}: {len(rows)} splits computed", flush=True)


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


def load_wm_reference_roi(n_rois: int) -> np.ndarray:
    """The WM meta-analysis ROI prior, used as the uniform faithfulness
    reference for every ICLR method (identical to the AAAI protocol, where
    all E0-E10 experiments share the same reference_prior)."""
    return load_reference_roi({"reference_prior": {
        "roi_prior_path": "outputs/priors/working_memory/aal116/roi_prior.csv",
    }}, n_rois)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# ICLR (NCR / Meta-GAT / Two-stage KRR) faithfulness support
# ---------------------------------------------------------------------------

def load_roi_prior_minmax(path: str | Path, n_rois: int) -> np.ndarray:
    """Load and min-max normalize an ROI-level prior (same contract as 27/40)."""
    df = pd.read_csv(path)
    if "roi_index" in df.columns:
        df = df.sort_values("roi_index")
    values = df["prior_score"].to_numpy(np.float64).reshape(-1)
    if values.shape != (n_rois,):
        raise ValueError(f"ROI prior has shape {values.shape}, expected {(n_rois,)}")
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros(n_rois, dtype=np.float64)
    return (values - low) / (high - low)


def upper_triangle_features(mats: np.ndarray) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    iu = np.triu_indices(mats.shape[1], k=1)
    return mats[:, iu[0], iu[1]].astype(np.float32), iu


def load_split_node_saliency(directory: Path, seed: int, fold: int) -> np.ndarray:
    payload = np.load(directory / f"seed{seed:02d}_fold{fold:02d}.npz", allow_pickle=False)
    if "node_saliency" not in payload.files:
        raise KeyError(f"node_saliency missing from {directory}")
    return np.asarray(payload["node_saliency"], dtype=float).reshape(-1)


def _ncr_mask_columns(roi_indices: Iterable[int], n_rois: int, iu) -> np.ndarray:
    idx = np.unique(np.asarray(list(roi_indices), dtype=int))
    incident = np.isin(iu[0], idx) | np.isin(iu[1], idx)
    n_edges = len(iu[0])
    return np.concatenate([incident, incident])


def run_iclr_config(
    config_path: Path,
    args,
    out: Path,
) -> tuple[list[Dict], list[Dict]]:
    """Faithfulness for the ICLR method configs (NCR / M2 / M3).

    NCR (linear ridge): refitted per split with the recorded best (lambda1,
    lambda2) from the headline run, then evaluated under raw-connectivity
    ROI masks.  Meta-GAT / KRR: converted from the refit faithfulness run of
    scripts/43 (identical protocol; raw condition values are stored in the
    per-method long files).  The reference prior of every method is the WM
    meta-analysis map (uniform across TRUE/SHUFFLED/RANDOM variants),
    matching the AAAI E* semantics where all experiments share one
    reference_prior.
    """
    cfg = load_config(config_path)
    exp_rows: list[Dict] = []
    exp_long: list[Dict] = []

    fc, sc, y, subject_ids, groups = load_connectomes(cfg["data"])
    y = np.asarray(y).reshape(-1)
    n_rois = int(fc.shape[1])
    n_folds = int(cfg.get("n_folds", 5))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    x_fc, iu = upper_triangle_features(fc)
    x_sc, _ = upper_triangle_features(sc)
    x = np.concatenate([x_fc, x_sc], axis=1)
    n_edges = len(iu[0])
    seeds = list(args.seeds)

    for method_id, method_cfg in cfg["methods"].items():
        exp_out = out / "per_experiment" / method_id
        exp_out.mkdir(parents=True, exist_ok=True)
        complete = exp_out / "COMPLETE"
        if complete.exists() and not args.overwrite:
            print(f"Skipping completed faithfulness run: {method_id}")
            existing = exp_out / "faithfulness_split_metrics.csv"
            if existing.exists():
                exp_rows.extend(pd.read_csv(existing).to_dict("records"))
            existing_long = exp_out / "faithfulness_long.csv"
            if existing_long.exists():
                exp_long.extend(pd.read_csv(existing_long).to_dict("records"))
            continue

        method_rows: list[Dict] = []
        method_long: list[Dict] = []

        if method_id.startswith("NCR_"):
            prior = load_roi_prior_minmax(method_cfg["path"], n_rois)
            wm_prior = load_wm_reference_roi(n_rois)
            results_path = NCR_OUTPUT_DIR / "split_metrics.csv"
            headline = pd.read_csv(results_path)
            headline = headline[
                (headline.method_id == method_id)
            ].set_index(["seed", "fold"])
            laplacian = build_edge_laplacian(
                n_rois=n_rois,
                prior_scores=prior,
                top_k=int(cfg.get("top_k", 30)),
                weighting=str(cfg.get("laplacian_weighting", "binary")),
                couple_modalities=bool(cfg.get("couple_modalities", False)),
                normalize=str(cfg.get("laplacian_normalization", "sym")),
            )
            saliency_dir = NCR_OUTPUT_DIR / "saliency" / method_id
            for seed, fold, train_idx, val_idx, test_idx in iter_nested_splits(
                y, seeds, n_folds, val_fraction, groups
            ):
                started = time.time()
                split_seed = seed * 1000 + fold
                rec = headline.loc[(seed, fold)]
                fit_idx = np.concatenate([train_idx, val_idx])
                model = NetworkConstrainedRidge(
                    alpha1=float(rec["best_alpha1"]),
                    alpha2=float(rec["best_alpha2"]),
                    edge_laplacian=laplacian,
                    n_rois=n_rois,
                ).fit(x[fit_idx], y[fit_idx])
                saliency = load_split_node_saliency(saliency_dir, seed, fold)
                k = min(args.topk, len(saliency))
                top_idx = np.argsort(saliency)[-k:]
                bottom_idx = np.argsort(saliency)[:k]
                prior_idx = np.argsort(wm_prior)[-k:]

                def masked(x_test, roi_indices):
                    x_masked = x_test.copy()
                    x_masked[:, _ncr_mask_columns(roi_indices, n_rois, iu)] = 0.0
                    return model.predict(x_masked)

                x_test = x[test_idx]
                original = prediction_metrics(y[test_idx], model.predict(x_test))
                top = prediction_metrics(y[test_idx], masked(x_test, top_idx))
                bottom = prediction_metrics(y[test_idx], masked(x_test, bottom_idx))
                prior_top = prediction_metrics(y[test_idx], masked(x_test, prior_idx))
                random_metrics = []
                rng = np.random.default_rng(split_seed + 9173)
                for repeat in range(args.random_repeats):
                    random_idx = rng.choice(n_rois, size=k, replace=False)
                    result = prediction_metrics(y[test_idx], masked(x_test, random_idx))
                    random_metrics.append(result)
                    method_long.append({
                        "experiment_id": method_id,
                        "seed": seed, "fold": fold,
                        "condition": "random", "repeat": repeat,
                        **{m: float(result[m]) for m in ("pearson", "rmse", "mae")},
                    })
                random_mean = {m: float(np.mean([r[m] for r in random_metrics])) for m in ("pearson", "rmse", "mae")}
                random_std = {m: float(np.std([r[m] for r in random_metrics], ddof=1)) for m in ("pearson", "rmse", "mae")}

                row = _build_faithfulness_row(
                    method_id, method_cfg["name"], seed, fold,
                    train_idx, val_idx, test_idx, original, top, bottom,
                    prior_top, random_mean, random_std, k, args, time.time() - started,
                )
                method_rows.append(row)
                method_long.append({"experiment_id": method_id, "seed": seed, "fold": fold,
                                    "condition": "original", "repeat": -1, **original})
                method_long.append({"experiment_id": method_id, "seed": seed, "fold": fold,
                                    "condition": "top", "repeat": -1, **top})
                method_long.append({"experiment_id": method_id, "seed": seed, "fold": fold,
                                    "condition": "bottom", "repeat": -1, **bottom})
                method_long.append({"experiment_id": method_id, "seed": seed, "fold": fold,
                                    "condition": "reference_prior_top", "repeat": -1, **prior_top})
                (exp_out / "saliency").mkdir(exist_ok=True)
                np.savez(
                    exp_out / "saliency" / f"seed{seed:02d}_fold{fold:02d}.npz",
                    node_saliency=saliency,
                )
                pd.DataFrame(method_rows).to_csv(exp_out / "faithfulness_split_metrics.csv", index=False)
                pd.DataFrame(method_long).to_csv(exp_out / "faithfulness_long.csv", index=False)
                print(method_id, f"seed{seed:02d}_fold{fold:02d}", {
                    "delta_rmse_top": row["delta_rmse_top"],
                    "delta_rmse_prior_top": row["delta_rmse_prior_top"],
                    "delta_rmse_random": row["delta_rmse_random"],
                }, flush=True)

        elif method_id.startswith(("M2_", "M3_")):
            src = ICLR_FAITHFULNESS_SOURCE / method_id
            split_csv = src / "faithfulness_split_metrics.csv"
            long_csv = src / "faithfulness_long.csv"
            if not split_csv.exists() or not long_csv.exists():
                print(f"[refit] {method_id}: no scripts/43 outputs, refitting masks...", flush=True)
                _refit_iclr_masked(cfg, method_id, method_cfg, seeds, args)
            is_random = method_id.endswith("RANDOM")
            ref_key = "prior_true_top"
            other_ref_key = "prior_random_top"
            split_df = pd.read_csv(split_csv)
            split_df = split_df[split_df.seed.isin(seeds)]
            long_df = pd.read_csv(long_csv)
            long_df = long_df[long_df.seed.isin(seeds)]

            for _, r in split_df.iterrows():
                row = {
                    "experiment_id": method_id,
                    "experiment_name": method_cfg["name"],
                    "seed": int(r["seed"]), "fold": int(r["fold"]),
                    "split_id": f"seed{int(r['seed']):02d}_fold{int(r['fold']):02d}",
                    "n_train": int(r["n_train"]), "n_val": int(r["n_val"]),
                    "n_test": int(r["n_test"]),
                    "best_epoch": -1, "best_val_metric": float("nan"),
                    "topk": int(r["topk"]), "mask_mode": args.mask_mode,
                    "random_repeats": args.random_repeats,
                    "runtime_seconds": float(r.get("runtime_seconds", float("nan"))),
                }
                for metric in ("pearson", "rmse", "mae"):
                    row[f"original_{metric}"] = float(r[f"original_{metric}"])
                    row[f"top_{metric}"] = float(r[f"top_{metric}"])
                    row[f"bottom_{metric}"] = float(r[f"bottom_{metric}"])
                    row[f"random_{metric}_mean"] = float(r[f"random_{metric}_mean"])
                    row[f"random_{metric}_std"] = float(r[f"random_{metric}_std"])
                    row[f"prior_top_{metric}"] = float(r[f"{ref_key}_{metric}"])
                    row[f"delta_{metric}_top"] = float(r[f"delta_{metric}_top"])
                    row[f"delta_{metric}_bottom"] = float(r[f"delta_{metric}_bottom"])
                    row[f"delta_{metric}_random"] = float(r[f"delta_{metric}_random"])
                    row[f"delta_{metric}_prior_top"] = float(r[f"delta_{metric}_{ref_key}"])
                    row[f"gap_{metric}_top_vs_random"] = row[f"delta_{metric}_top"] - row[f"delta_{metric}_random"]
                    row[f"gap_{metric}_top_vs_bottom"] = row[f"delta_{metric}_top"] - row[f"delta_{metric}_bottom"]
                method_rows.append(row)

            cond_map = {"original": "original", "top": "top", "bottom": "bottom",
                        "random": "random", ref_key: "reference_prior_top"}
            for _, r in long_df.iterrows():
                condition = r["condition"]
                mapped = cond_map.get(condition, other_ref_key)
                if mapped not in cond_map.values():
                    continue
                method_long.append({
                    "experiment_id": method_id,
                    "seed": int(r["seed"]), "fold": int(r["fold"]),
                    "condition": mapped, "repeat": int(r["repeat"]),
                    **{m: float(r[m]) for m in ("pearson", "rmse", "mae")},
                })

            (exp_out / "saliency").mkdir(exist_ok=True)
            if method_id.startswith("M2_"):
                biomarker_dir = src / "attention"
                key = "node_attention_mass"
            else:
                biomarker_dir = src / "krr_saliency"
                key = "node_saliency"
            for payload in biomarker_dir.glob("seed*_fold*.npz"):
                seed, fold = int(payload.stem[4:6]), int(payload.stem[11:13])
                if seed not in seeds:
                    continue
                with np.load(payload, allow_pickle=False) as data:
                    np.savez(
                        exp_out / "saliency" / payload.name,
                        node_saliency=np.asarray(data[key], dtype=float).reshape(-1),
                    )

            pd.DataFrame(method_rows).to_csv(exp_out / "faithfulness_split_metrics.csv", index=False)
            pd.DataFrame(method_long).to_csv(exp_out / "faithfulness_long.csv", index=False)
            print(f"Converted {method_id}: {len(method_rows)} splits, {len(method_long)} long rows", flush=True)

        else:
            raise ValueError(f"Unknown ICLR method prefix for {method_id}")

        complete.write_text("ok\n", encoding="utf-8")
        exp_rows.extend(method_rows)
        exp_long.extend(method_long)

    return exp_rows, exp_long


def _build_faithfulness_row(
    experiment_id: str,
    experiment_name: str,
    seed: int,
    fold: int,
    train_idx,
    val_idx,
    test_idx,
    original: Dict,
    top: Dict,
    bottom: Dict,
    prior_top: Dict,
    random_mean: Dict,
    random_std: Dict,
    topk: int,
    args,
    runtime: float,
) -> Dict:
    row = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "seed": seed, "fold": fold,
        "split_id": f"seed{seed:02d}_fold{fold:02d}",
        "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
        "best_epoch": -1, "best_val_metric": float("nan"),
        "topk": topk, "mask_mode": args.mask_mode,
        "random_repeats": args.random_repeats,
        "runtime_seconds": runtime,
    }
    for metric in ("pearson", "rmse", "mae"):
        row[f"original_{metric}"] = float(original[metric])
        row[f"top_{metric}"] = float(top[metric])
        row[f"bottom_{metric}"] = float(bottom[metric])
        row[f"random_{metric}_mean"] = random_mean[metric]
        row[f"random_{metric}_std"] = random_std[metric]
        row[f"prior_top_{metric}"] = float(prior_top[metric])
        row[f"delta_{metric}_top"] = positive_degradation(original, top, metric)
        row[f"delta_{metric}_bottom"] = positive_degradation(original, bottom, metric)
        row[f"delta_{metric}_random"] = positive_degradation(original, random_mean, metric)
        row[f"delta_{metric}_prior_top"] = positive_degradation(original, prior_top, metric)
        row[f"gap_{metric}_top_vs_random"] = row[f"delta_{metric}_top"] - row[f"delta_{metric}_random"]
        row[f"gap_{metric}_top_vs_bottom"] = row[f"delta_{metric}_top"] - row[f"delta_{metric}_bottom"]
    return row


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
        method_ids = list(cfg.get("methods", {}).keys())
        if any(str(m).startswith(ICLR_METHOD_PREFIXES) for m in method_ids):
            exp_rows, exp_long = run_iclr_config(Path(config_path), args, out)
            all_rows.extend(exp_rows)
            long_rows.extend(exp_long)
            continue
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
