#!/usr/bin/env python3
"""Preflight checks for the new prediction benchmark experiments."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from metascfc.benchmark_utils import load_connectomes


def img_parameter_estimate(n_rois: int, n_modules: int, graph_hidden: int, ratio: int) -> dict:
    flat = n_rois * n_rois
    hidden = max(1, flat // ratio)
    attention = flat * hidden + hidden + hidden * flat + flat
    interaction = n_modules * n_modules * (2 * n_rois + 1)
    gcn = n_rois * graph_hidden
    head = (2 * n_rois * graph_hidden) * 256 + 256 + 256 + 1
    total = attention + interaction + gcn + head
    return {"flat": flat, "attention_hidden": hidden, "attention": attention, "total": total}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge-config", default="configs/aaai/prior_weighted_ridge.yaml")
    ap.add_argument("--sota-config", default="configs/aaai/sota_graph_baselines.yaml")
    args = ap.parse_args()

    ridge_cfg = yaml.safe_load(Path(args.ridge_config).read_text(encoding="utf-8"))
    sota_cfg = yaml.safe_load(Path(args.sota_config).read_text(encoding="utf-8"))
    fc, sc, y, subject_ids, groups = load_connectomes(sota_cfg["data"])
    n, r, _ = fc.shape
    print(f"PASS data: subjects={n}, ROIs={r}, target={y.shape}, groups={'yes' if groups is not None else 'no'}")
    if n != 412:
        print(f"WARNING: expected the final 412-subject cohort, found {n}")

    module_path = Path(sota_cfg["module_map_path"])
    mdf = pd.read_csv(module_path).sort_values("roi_index")
    if len(mdf) != r or mdf["roi_index"].nunique() != r:
        raise ValueError("Module map must contain one unique row per ROI")
    n_modules = int(mdf["module_id"].nunique())
    print(f"PASS module map: {r} ROIs, {n_modules} modules")

    for method_id, spec in ridge_cfg["methods"].items():
        path = Path(spec["path"])
        df = pd.read_csv(path)
        if "prior_score" not in df or len(df) != r:
            raise ValueError(f"Invalid prior for {method_id}: {path}")
        if not np.isfinite(df["prior_score"]).all():
            raise ValueError(f"Non-finite prior values for {method_id}")
        print(f"PASS prior {method_id}: {path}")

    img = sota_cfg["models"]["IMG_GCN"]
    est = img_parameter_estimate(r, n_modules, int(img.get("graph_hidden", 16)), int(img.get("bottleneck_ratio", 2)))
    print(f"IMG-GCN full attention dimensions: {est['flat']} -> {est['attention_hidden']} -> {est['flat']}")
    print(f"IMG-GCN estimated parameters: {est['total']:,} ({est['attention']:,} in attention)")
    print(f"Approx. FP32 weights only: {est['total'] * 4 / 2**30:.2f} GiB")
    print(f"Approx. FP32 Adam training state (weights+grads+2 moments): {est['total'] * 16 / 2**30:.2f} GiB")

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print("CUDA device:", props.name)
        print(f"CUDA visible memory: {props.total_memory / 2**30:.1f} GiB")
    else:
        print("WARNING: CUDA is unavailable. Full IMG-GCN is intended for the DGX Spark GPU and will be slow on CPU.")

    expected = len(sota_cfg["seeds"]) * int(sota_cfg["n_folds"])
    print(f"Expected evaluations per graph method: {expected}")
    print(f"Expected prior-Ridge evaluations: {len(ridge_cfg['methods']) * expected}")
    print("PREFLIGHT PASSED")


if __name__ == "__main__":
    main()
