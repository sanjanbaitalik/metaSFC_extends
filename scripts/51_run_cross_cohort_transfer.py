#!/usr/bin/env python3
"""Cross-cohort zero-shot transfer for LLM-gated checkpoints (ICLR 2027).

Requested validation on the latest HCP cohorts (e.g. HCP-Development): load
the best-performing frozen checkpoint from the dual-task matrix
(``outputs/iclr/dual_task_matrix/checkpoints/``) and evaluate it - WITHOUT
any fine-tuning - on newly packed arrays of the target cohort (defaults:
``inputs/dataset_hcpd_FC/FC_all.npy``, ``inputs/dataset_hcpd_SC/SC_all.npy``).

The checkpoint carries everything needed to reproduce the feature pipeline
exactly (fit-partition scalers, target de-normalization, split structural
graph, prior, architecture config), so transfer is strictly frozen-feature.

Strict QC (hard errors, never silent):
- FC/SC must be [n, 116, 116] and match the checkpoint's n_rois exactly
  (AAL116); different parcellations are rejected, not adapted.
- Labels must be finite and one-per-subject; cohort size may differ freely.

Outputs (under --output-dir): transfer_metrics.csv / .json with Pearson r,
RMSE, MAE per evaluated checkpoint + per-subject predictions CSVs.

Examples
--------
    # Auto-pick the best fluid-intelligence checkpoint and transfer to HCP-D:
    python scripts/51_run_cross_cohort_transfer.py --auto-best --target fluid_intelligence

    # Explicit checkpoints:
    python scripts/51_run_cross_cohort_transfer.py \
        --checkpoints outputs/iclr/dual_task_matrix/checkpoints/llm_gated_llm_wm_working_memory_seed00_fold00.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from metascfc.benchmark_utils import prediction_metrics
from metascfc.models.iclr_backbones import LLMGatedConfig, LLMGatedTransformer

DEFAULT_MATRIX_ROOT = "outputs/iclr/dual_task_matrix"
DEFAULT_OUTPUT_DIR = "outputs/iclr/cross_cohort_transfer"
N_ROIS_AAL116 = 116


def load_checkpoint(path: str | Path) -> Dict:
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Run "
            "scripts/50_run_dual_task_matrix.py --save-checkpoints first."
        )
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if payload.get("model_family") != "llm_gated_transformer":
        raise ValueError(f"{ckpt_path} is not an llm_gated_transformer checkpoint")
    return payload


def pick_best_checkpoint(matrix_root: str | Path, target: str,
                         prior: str | None = None) -> Path:
    """Best llm_gated cell by mean Pearson in the dual-task matrix results."""
    split_csv = Path(matrix_root) / "split_metrics.csv"
    if not split_csv.exists():
        raise FileNotFoundError(
            f"{split_csv} not found - run scripts/50_run_dual_task_matrix.py first."
        )
    df = pd.read_csv(split_csv)
    df = df[(df.model == "llm_gated") & (df.target == target)]
    if prior:
        df = df[df.prior == prior]
    if df.empty:
        raise ValueError(f"No llm_gated rows for target='{target}' in {split_csv}")
    cell = (df.groupby(["prior", "target"], as_index=False)
              .agg(pearson_mean=("pearson", "mean"))
              .sort_values("pearson_mean", ascending=False)
              .iloc[0])
    stem = f"llm_gated_{cell.prior}_{cell.target}"
    matches = sorted((Path(matrix_root) / "checkpoints").glob(f"{stem}_seed*.pt"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint matching '{stem}_*.pt' under {matrix_root}/checkpoints; "
            "re-run the matrix with --save-checkpoints."
        )
    print(f"[auto-best] {cell.prior}/{cell.target} "
          f"pearson={cell.pearson_mean:.4f} -> {matches[0].name}")
    return matches[0]


def qc_new_cohort(fc: np.ndarray, sc: np.ndarray, y: np.ndarray,
                  n_rois_expected: int) -> None:
    """Strict dimension/finiteness checks for the transfer cohort."""
    if fc.ndim != 3 or sc.ndim != 3 or fc.shape != sc.shape:
        raise ValueError(
            f"FC/SC must be matched [subjects, ROI, ROI] arrays; got "
            f"FC={fc.shape}, SC={sc.shape}"
        )
    n_rois = fc.shape[1]
    if fc.shape[1] != fc.shape[2] or sc.shape[1] != sc.shape[2]:
        raise ValueError(f"Non-square matrices: FC={fc.shape}, SC={sc.shape}")
    if n_rois != N_ROIS_AAL116 or n_rois != n_rois_expected:
        raise ValueError(
            f"Cohort parcellation mismatch: new cohort has {n_rois} ROIs but "
            f"AAL116 requires {N_ROIS_AAL116} and the checkpoint was trained "
            f"with {n_rois_expected}. Repack the cohort with AAL116 "
            "(scripts/24_pack_hcp_arrays.py flow) before transfer."
        )
    y_vec = np.asarray(y, dtype=np.float64).reshape(-1)
    if len(y_vec) != len(fc):
        raise ValueError(f"Labels ({len(y_vec)}) != subjects ({len(fc)})")
    n_nan_fc = int((~np.isfinite(fc)).sum())
    n_nan_sc = int((~np.isfinite(sc)).sum())
    if n_nan_fc or n_nan_sc or not np.isfinite(y_vec).all():
        n_nan_y = int((~np.isfinite(y_vec)).sum())
        raise ValueError(
            f"New cohort contains non-finite values: FC={n_nan_fc}, "
            f"SC={n_nan_sc}, y={n_nan_y}. Clean/repack the arrays first."
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[],
                    help="Explicit checkpoint paths (.pt)")
    ap.add_argument("--auto-best", action="store_true",
                    help="Pick the best llm_gated checkpoint from the matrix")
    ap.add_argument("--matrix-root", default=DEFAULT_MATRIX_ROOT)
    ap.add_argument("--target", default=None,
                    help="Target filter for --auto-best (fluid_intelligence|working_memory)")
    ap.add_argument("--prior", default=None, help="Prior filter for --auto-best")
    ap.add_argument("--fc", default="inputs/dataset_hcpd_FC/FC_all.npy")
    ap.add_argument("--sc", default="inputs/dataset_hcpd_SC/SC_all.npy")
    ap.add_argument("--y", default="inputs/dataset_hcpd_SC/label_all.npy",
                    help="Behavioral target of the NEW cohort (same measure as training)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args()

    paths: List[Path] = [Path(p) for p in args.checkpoints]
    if args.auto_best:
        if len(paths) > 1:
            raise ValueError("Use either --auto-best or explicit --checkpoints, not both lists")
        paths.append(pick_best_checkpoint(args.matrix_root,
                                          target=args.target or "fluid_intelligence",
                                          prior=args.prior))
    if not paths:
        raise SystemExit("Nothing to evaluate: pass --auto-best or --checkpoints.")

    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- new-cohort data ----
    for required in (args.fc, args.sc, args.y):
        if not Path(required).exists():
            raise FileNotFoundError(
                f"Transfer cohort file missing: {required}. Pack the new "
                "cohort (AAL116) into inputs/dataset_hcpd_*/ first."
            )
    fc_new = np.load(args.fc)
    sc_new = np.load(args.sc)
    y_new = np.load(args.y).astype(np.float64).reshape(-1)

    rows: List[Dict] = []
    for path in paths:
        print(f"\n=== Zero-shot transfer: {path.name} ===", flush=True)
        ckpt = load_checkpoint(path)
        qc_new_cohort(fc_new, sc_new, y_new, int(ckpt["n_rois"]))

        cfg = LLMGatedConfig(**ckpt["config"])
        model = LLMGatedTransformer(
            int(ckpt["n_rois"]), int(ckpt["in_dim"]), cfg,
            ckpt["prior"], ckpt["edge_src"], ckpt["edge_dst"],
        ).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()

        x_mean, x_std = ckpt["x_mean"].reshape(1, -1), ckpt["x_std"].reshape(1, -1)
        preds = np.empty(len(fc_new), dtype=np.float64)
        with torch.no_grad():
            for start in range(0, len(fc_new), args.batch_size):
                sl = slice(start, start + args.batch_size)
                x = np.concatenate([fc_new[sl], sc_new[sl]], axis=2) \
                    .reshape(len(fc_new[sl]), -1).astype(np.float64)
                x = ((x - x_mean) / x_std) \
                    .reshape(len(fc_new[sl]), int(ckpt["n_rois"]), -1).astype(np.float32)
                preds[sl] = model(torch.from_numpy(x).to(device)).cpu().numpy() \
                    * float(ckpt["fit_std"]) + float(ckpt["fit_mean"])

        metrics = prediction_metrics(y_new, preds)
        metrics["pearson"] = float(pearsonr(y_new, preds).statistic)
        row = {
            "checkpoint": str(path),
            "checkpoint_name": path.stem,
            **metrics,
            "n_subjects_transfer": len(y_new),
        }
        rows.append(row)
        pd.DataFrame({
            "subject_index": np.arange(len(y_new)), "target": y_new,
            "zero_shot_prediction": preds,
        }).to_csv(out_dir / f"predictions_{path.stem}.csv", index=False)
        print(json.dumps(metrics, indent=2))

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "transfer_metrics.csv", index=False)
    (out_dir / "transfer_metrics.json").write_text(
        json.dumps({
            "cohort": {"fc": args.fc, "sc": args.sc, "y": args.y,
                       "n_subjects": int(len(y_new))},
            "results": rows,
        }, indent=2)
    )
    print(f"\nSaved transfer metrics for {len(rows)} checkpoint(s) to {out_dir}")


if __name__ == "__main__":
    main()
