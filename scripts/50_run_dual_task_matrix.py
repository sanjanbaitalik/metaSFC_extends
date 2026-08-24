#!/usr/bin/env python3
"""Master Dual-Task Matrix runner (ICLR 2027 final experiments).

Executes the 2 x 4 x 2 Inductive-Bottleneck matrix with the exact nested-CV
protocol shared by all methods (default 10 seeds x 5 outer folds, 15% inner
validation, leakage-free):

    models   x priors          x targets
    llm_gated x llm_wm           x fluid_intelligence (PMAT24_A_CR)
    ncr       x llm_fluid        x working_memory     (ListSort_Unadj)
              x random_control
              x no_prior

Hypothesis: a target-matched prior acts as an informative projector and
beats the linear baseline; a mismatched prior over-constrains the hypothesis
space (Inductive Bottleneck) - visible as high I(X;Z) but low I(Z;Y), a
learned bypass gate alpha -> 0, and degraded prediction.

Outputs (under --output-root, default outputs/iclr/dual_task_matrix):
    split_metrics.csv   one row per (model, prior, target, seed, fold) with
                        pearson/rmse/mae, converged I_XZ / I_ZY / probe_r2,
                        learned bypass alpha (NCR: tau = lambda2/lambda1)
    summary.csv         16 aggregated rows (mean +/- std over splits)
    errors.jsonl        per-evaluation failures (matrix keeps running)
    run_metadata.json, COMPLETE

Example:
    python scripts/50_run_dual_task_matrix.py --seeds 0 --folds 0   # smoke
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from metascfc.benchmark_utils import (
    choose_device,
    iter_nested_splits,
    load_connectomes,
    prediction_metrics,
    save_json,
)
from metascfc.data import build_task_labels, resolve_target
from metascfc.metrics import IBEpochTracker
from metascfc.models.iclr_backbones import (
    build_edge_laplacian,
    fit_predict_llm_gated,
    fit_predict_network_constrained,
)

MODELS = ("llm_gated", "ncr")
PRIORS = ("llm_wm", "llm_fluid", "random_control", "no_prior")
TARGETS = ("fluid_intelligence", "working_memory")

DEFAULT_OUTPUT_ROOT = "outputs/iclr/dual_task_matrix"
DEFAULT_NCR_CONFIG = "configs/aaai/network_constrained_ridge.yaml"
DEFAULT_LLM_CONFIG = "configs/iclr/llm_fluid_prior.yaml"

LLM_PRIOR_PATHS = {
    "llm_wm": "outputs/priors/llm/working_memory/roi_prior.csv",
    "llm_fluid": "outputs/priors/llm/fluid_intelligence/roi_prior.csv",
}
RANDOM_PRIOR_PATH = "outputs/priors/random_prior/aal116/roi_prior.csv"

# Target -> label file.  The WM file is materialized on demand from the HCP
# behavior table via metascfc.data.build_task_labels.
def target_label_path(target: str) -> Path:
    if target == "fluid_intelligence":
        return Path("inputs/dataset_SC/label_all.npy")
    canonical = resolve_target(target)
    return Path(f"inputs/dataset_SC/task_labels/{canonical}/label_all.npy")


def load_roi_prior_optional(path: Optional[str], n_rois: int) -> np.ndarray:
    """Min-max normalized ROI scores; zeros encode 'no prior'."""
    if path is None:
        return np.zeros(n_rois, dtype=np.float64)
    df = pd.read_csv(path)
    if "roi_index" in df.columns:
        df = df.sort_values("roi_index")
    p = df["prior_score"].to_numpy(np.float64)
    if p.shape != (n_rois,):
        raise ValueError(f"Prior {path} has shape {p.shape}; expected {(n_rois,)}")
    p = np.clip(p, 0.0, None)
    if p.max() > p.min():
        p = (p - p.min()) / (p.max() - p.min())
    return p


def ensure_target_labels(target: str, behavior_csv: Optional[str]) -> Path:
    """Return the label file for a target, building WM labels if needed.

    Edge cases handled by the loader itself: subjects missing from the
    behavior table or with NaN measures raise an explicit error listing the
    offending subject IDs (never silently imputed).
    """
    path = target_label_path(target)
    if path.exists():
        return path
    if behavior_csv is None:
        raise FileNotFoundError(
            f"Labels for '{target}' not found at {path}. Generate them first:\n"
            f"  python -m metascfc.data.hcp_targets --behavior-csv "
            f"<HCP_unrestricted.csv> --targets {target}"
        )
    canonical = resolve_target(target)
    print(f"[labels] building '{canonical}' from {behavior_csv} ...", flush=True)
    return build_task_labels(
        "inputs/dataset_SC/hcp_subjects_used.csv", behavior_csv, target,
        out_dir=Path("inputs/dataset_SC/task_labels") / canonical,
    ) / "label_all.npy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    ap.add_argument("--ncr-config", default=DEFAULT_NCR_CONFIG)
    ap.add_argument("--llm-config", default=DEFAULT_LLM_CONFIG)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--behavior-csv", default=None,
                    help="HCP unrestricted behavioral CSV; required to "
                         "materialize ListSort_Unadj labels on first use")
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--folds", nargs="*", type=int, help="Optional fold filter")
    ap.add_argument("--models", nargs="*", choices=MODELS, default=list(MODELS))
    ap.add_argument("--priors", nargs="*", choices=PRIORS, default=list(PRIORS))
    ap.add_argument("--targets", nargs="*", choices=TARGETS, default=list(TARGETS))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--prior-overrides", default=None,
                    help="Optional JSON dict overriding prior paths, e.g. "
                         "'{\"llm_wm\": \"outputs/priors/working_memory/aal116/roi_prior.csv\"}' "
                         "(ablations / pipeline verification)")
    args = ap.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    seeds = args.seeds if args.seeds else [int(s) for s in range(10)]
    selected_folds = set(args.folds) if args.folds else None

    ncr_cfg: Dict = yaml.safe_load(Path(args.ncr_config).read_text(encoding="utf-8"))
    llm_cfg: Dict = yaml.safe_load(Path(args.llm_config).read_text(encoding="utf-8"))
    val_fraction = float(ncr_cfg.get("val_fraction", 0.15))
    n_folds = int(ncr_cfg.get("n_folds", 5))

    # ---- data: FC/SC are shared across targets; only y differs ----
    fc, sc, _, _, groups = load_connectomes(ncr_cfg["data"])
    n_rois = int(fc.shape[1])
    iu = np.triu_indices(n_rois, k=1)
    x_edges = np.concatenate(
        [fc[:, iu[0], iu[1]], sc[:, iu[0], iu[1]]], axis=1
    ).astype(np.float64)

    # ---- priors + NCR penalty cache (per prior; target-independent) ----
    prior_paths: Dict[str, Optional[str]] = {
        name: (LLM_PRIOR_PATHS.get(name, RANDOM_PRIOR_PATH) if name != "no_prior" else None)
        for name in args.priors
    }
    if args.prior_overrides:
        overrides = json.loads(args.prior_overrides)
        unknown = set(overrides) - set(PRIORS)
        if unknown:
            raise ValueError(f"Unknown prior names in --prior-overrides: {sorted(unknown)}")
        prior_paths.update({k: v for k, v in overrides.items()})
    laplacians: Dict[str, object] = {}
    eig_cache: Dict[str, object] = {}
    for name in args.priors:
        if name == "no_prior":
            laplacians[name] = build_edge_laplacian(
                n_rois=n_rois, prior_adjacency=np.zeros((n_rois, n_rois)),
                weighting=str(ncr_cfg.get("laplacian_weighting", "binary")),
                couple_modalities=bool(ncr_cfg.get("couple_modalities", False)),
                normalize=str(ncr_cfg.get("laplacian_normalization", "sym")),
            )
        else:
            scores = load_roi_prior_optional(prior_paths[name], n_rois)
            laplacians[name] = build_edge_laplacian(
                n_rois=n_rois, prior_scores=scores,
                top_k=int(ncr_cfg.get("top_k", 30)),
                weighting=str(ncr_cfg.get("laplacian_weighting", "binary")),
                couple_modalities=bool(ncr_cfg.get("couple_modalities", False)),
                normalize=str(ncr_cfg.get("laplacian_normalization", "sym")),
            )
    from metascfc.models.iclr_backbones.network_constrained_ridge import factor_laplacian_eig
    for name in args.priors:
        eig_cache[name] = factor_laplacian_eig(laplacians[name])
    prior_scores_cache: Dict[str, np.ndarray] = {
        name: load_roi_prior_optional(prior_paths[name], n_rois) for name in args.priors
    }

    ncr_fixed = dict(
        alpha1_grid=[float(a) for a in ncr_cfg["ridge_alphas"]],
        alpha2_grid=[float(a) for a in ncr_cfg["laplacian_alphas"]],
    )
    llm_fixed = dict(
        top_percent=float(llm_cfg.get("top_percent_sc", 10.0)),
        n_layers=int(llm_cfg.get("n_layers", 2)),
        heads=int(llm_cfg.get("heads1", 4)),
        weight_decay=float(llm_cfg.get("weight_decay", 1e-4)),
        epochs=int(llm_cfg.get("epochs", 60)),
        patience=int(llm_cfg.get("patience", 15)),
        min_epochs=int(llm_cfg.get("min_epochs", 10)),
        alpha_init=float(llm_cfg.get("alpha_init", 0.5)),
        grad_clip=float(llm_cfg.get("grad_clip", 5.0)),
    )

    split_csv = out_root / "split_metrics.csv"
    if args.overwrite and split_csv.exists():
        split_csv.unlink()
    rows = [] if not split_csv.exists() else pd.read_csv(split_csv).to_dict("records")
    completed = {
        (r["model"], r["prior"], r["target"], int(r["seed"]), int(r["fold"]))
        for r in rows
    }

    total = expected = 0
    for target in args.targets:
        try:
            label_path = ensure_target_labels(target, args.behavior_csv)
            y = np.load(label_path).astype(np.float64).reshape(-1)
            if len(y) != len(fc):
                raise ValueError(f"{target}: labels ({len(y)}) != subjects ({len(fc)})")
        except Exception as exc:  # noqa: BLE001 - skip the target, keep the matrix running
            msg = f"Target '{target}' unavailable: {exc}"
            print(msg, flush=True)
            with open(out_root / "errors.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"model": "*", "prior": "*", "target": target,
                                     "error": str(exc)}) + "\n")
            continue
        for model in args.models:
            for prior in args.priors:
                for seed, fold, train_idx, val_idx, test_idx in iter_nested_splits(
                    y, seeds, n_folds, val_fraction, groups
                ):
                    if selected_folds is not None and fold not in selected_folds:
                        continue
                    expected += 1
                    key = (model, prior, target, seed, fold)
                    if key in completed:
                        continue
                    split_id = f"seed{seed:02d}_fold{fold:02d}"
                    split_seed = seed * 1000 + fold
                    started = time.time()
                    tracker = IBEpochTracker()
                    row: Dict = {
                        "model": model, "prior": prior, "target": target,
                        "seed": seed, "fold": fold, "split_id": split_id,
                        "n_train": len(train_idx), "n_val": len(val_idx),
                        "n_test": len(test_idx),
                    }
                    try:
                        if model == "ncr":
                            pred, a1, a2, val_rmse, _ = fit_predict_network_constrained(
                                x_edges, y, train_idx, val_idx, test_idx,
                                laplacians[prior],
                                laplacian_eig=eig_cache[prior],
                                ib_tracker=tracker,
                                **ncr_fixed,
                            )
                            row.update({
                                "best_alpha1": float(a1), "best_alpha2": float(a2),
                                "best_val_rmse": float(val_rmse),
                            })
                        else:
                            pred, best_cfg, best_val_rmse, best_epoch, _, n_params = (
                                fit_predict_llm_gated(
                                    fc, sc, y, train_idx, val_idx, test_idx,
                                    prior_scores_cache[prior],
                                    hidden_grid=[float(h) for h in llm_cfg["hidden_grid"]],
                                    dropout_grid=[float(d) for d in llm_cfg["dropout_grid"]],
                                    lr_grid=[float(l) for l in llm_cfg["lr_grid"]],
                                    device=device, seed=split_seed,
                                    ib_tracker=tracker, **llm_fixed,
                                )
                            )
                            row.update({
                                "best_hidden": best_cfg["hidden"],
                                "best_dropout": best_cfg["dropout"],
                                "best_learning_rate": best_cfg["learning_rate"],
                                "best_epoch": best_epoch, "parameters": n_params,
                            })
                        metrics = prediction_metrics(y[test_idx], pred)
                        row.update(metrics)
                        row["runtime_seconds"] = time.time() - started
                        if tracker.final:
                            row["I_XZ_final"] = float(tracker.final["I_XZ"])
                            row["I_ZY_final"] = float(tracker.final["I_ZY"])
                            row["probe_r2_final"] = float(tracker.final["probe_r2"])
                        alphas = getattr(tracker, "alpha_final", None)
                        if alphas:
                            row["bypass_alpha"] = float(np.mean(alphas))
                        rows.append(row)
                        completed.add(key)
                        total += 1
                        pd.DataFrame(rows).to_csv(split_csv, index=False)
                        print(
                            f"{model}/{prior}/{target} {split_id} "
                            f"r={metrics['pearson']:+.3f} rmse={metrics['rmse']:.3f} "
                            f"I_XZ={row.get('I_XZ_final', float('nan')):.3f} "
                            f"I_ZY={row.get('I_ZY_final', float('nan')):.3f} "
                            f"alpha={row.get('bypass_alpha', float('nan')):.3f}",
                            flush=True,
                        )
                    except Exception as exc:  # noqa: BLE001 - keep the matrix running
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        with open(out_root / "errors.jsonl", "a", encoding="utf-8") as fh:
                            fh.write(json.dumps({**row, "traceback": traceback.format_exc()}) + "\n")
                        print(f"FAILED {key}: {exc}", flush=True)

    # ---- aggregate summary over the full grid ----
    df = pd.DataFrame(rows)
    if df.empty:
        print("No evaluations recorded.")
        return
    metric_cols = ["pearson", "rmse", "mae", "I_XZ_final", "I_ZY_final",
                   "probe_r2_final", "bypass_alpha"]
    agg = (
        df.groupby(["model", "prior", "target"])
        .agg(n_splits=("fold", "size"),
             **{f"{c}_mean": (c, "mean") for c in metric_cols if c in df.columns},
             **{f"{c}_std": (c, "std") for c in metric_cols if c in df.columns})
        .reset_index()
    )
    order_m = {m: i for i, m in enumerate(MODELS)}
    order_p = {p: i for i, p in enumerate(PRIORS)}
    order_t = {t: i for i, t in enumerate(TARGETS)}
    agg["_o"] = agg.model.map(order_m) * 100 + agg.prior.map(order_p) * 10 + agg.target.map(order_t)
    agg = agg.sort_values("_o").drop(columns="_o")
    agg.to_csv(out_root / "summary.csv", index=False)

    configured_seeds = set(int(s) for s in (args.seeds or range(10)))
    full_expected = (len(args.models) * len(args.priors) * len(args.targets)
                     * len(configured_seeds) * n_folds)
    complete = out_root / "COMPLETE"
    if len(df) >= full_expected and not (out_root / "errors.jsonl").exists():
        complete.write_text("ok\n", encoding="utf-8")
    save_json({
        "models": list(args.models), "priors": list(args.priors),
        "targets": list(args.targets), "seeds": sorted(configured_seeds),
        "n_folds": n_folds, "val_fraction": val_fraction,
        "n_evaluations": int(len(df)), "device": str(device),
        "ncr_config": args.ncr_config, "llm_config": args.llm_config,
        "ib_noise_floor": IBEpochTracker().noise_floor,
    }, out_root / "run_metadata.json")
    print(f"Saved {len(df)} evaluations ({total} new this run) to {out_root}")


if __name__ == "__main__":
    main()
