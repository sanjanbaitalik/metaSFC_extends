#!/usr/bin/env python3
"""Validate completeness and integrity of new benchmark outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def check_table(path: Path, expected_methods: set[str], seeds: set[int], n_folds: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"method_id", "method_name", "seed", "fold", "pearson", "rmse", "mae"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {sorted(missing)}")
    if df.duplicated(["method_id", "seed", "fold"]).any():
        raise ValueError(f"Duplicate method/seed/fold rows in {path}")
    if not np.isfinite(df[["pearson", "rmse", "mae"]].to_numpy(float)).all():
        raise ValueError(f"Non-finite metrics in {path}")
    for method in expected_methods:
        g = df[(df.method_id == method) & (df.seed.isin(seeds))]
        expected = len(seeds) * n_folds
        if len(g) != expected:
            raise ValueError(f"{method}: expected {expected} rows, found {len(g)}")
        if set(g.fold.astype(int)) != set(range(n_folds)):
            raise ValueError(f"{method}: incomplete fold indices")
    return df


def check_prediction_coverage(pred_dir: Path, method: str, seeds: set[int], n_folds: int, n_subjects: int) -> None:
    for seed in seeds:
        files = sorted(pred_dir.glob(f"{method}_seed{seed:02d}_fold*.csv"))
        if len(files) != n_folds:
            raise ValueError(f"{method} seed {seed}: expected {n_folds} prediction files, found {len(files)}")
        merged = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        if len(merged) != n_subjects or merged["subject_index"].nunique() != n_subjects:
            raise ValueError(f"{method} seed {seed}: outer-test predictions do not cover every subject exactly once")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge-config", default="configs/aaai/prior_weighted_ridge.yaml")
    ap.add_argument("--sota-config", default="configs/aaai/sota_graph_baselines.yaml")
    args = ap.parse_args()
    rcfg = yaml.safe_load(Path(args.ridge_config).read_text(encoding="utf-8"))
    scfg = yaml.safe_load(Path(args.sota_config).read_text(encoding="utf-8"))
    seeds = set(int(s) for s in scfg["seeds"]); n_folds = int(scfg["n_folds"])
    n_subjects = len(np.load(scfg["data"]["y_path"], allow_pickle=True).reshape(-1))

    ridge_dir = Path(rcfg["output_dir"]); sota_dir = Path(scfg["output_dir"])
    ridge_methods = set(rcfg["methods"]); sota_methods = set(scfg["models"])
    check_table(ridge_dir / "split_metrics.csv", ridge_methods, seeds, n_folds)
    check_table(sota_dir / "split_metrics.csv", sota_methods, seeds, n_folds)
    for method in ridge_methods:
        check_prediction_coverage(ridge_dir / "predictions", method, seeds, n_folds, n_subjects)
    for method in sota_methods:
        check_prediction_coverage(sota_dir / "predictions", method, seeds, n_folds, n_subjects)
    print("VALIDATION PASSED: all methods have complete, finite, nonduplicated 10x5 results")


if __name__ == "__main__":
    main()
