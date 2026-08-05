#!/usr/bin/env python3
"""ICLR experiment 3: paper-ready paired statistical tests.

Primary claims (paired at seed level, 5 folds averaged per seed, n=10):
  1. NCR_TRUE (network-constrained ridge) vs B3 (FC+SC Ridge) -- the coupling
     prior is no worse than the best fusion baseline.
  2. M3_E0 (two-stage KRR, no-prior biomarker) vs E1 (MetaSFC end-to-end GNN) --
     the two-stage projection vastly improves prediction.

Uses the same methodology as scripts/11_statistical_tests.py: paired t-test,
two-sided Wilcoxon signed-rank, bootstrap 95% CI of the improvement,
Cohen's d_z, and Holm correction (per metric and globally). For RMSE/MAE the
improvement is defined as B - A so that positive values always mean A is
better (higher pearson, lower rmse/mae).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

LOWER_BETTER = {"rmse", "mae"}

COMPARISONS = [
    ("NCR_TRUE", "B3"),
    ("M3_E0", "E1"),
]
METRICS = ["pearson", "rmse", "mae"]

DATA_SOURCES = {
    "NCR_TRUE": ("outputs/aaai/network_constrained_ridge/split_metrics.csv", "method_id"),
    "B3": ("outputs/aaai/prediction_baselines/prediction_baselines_split_metrics.csv", "experiment_id"),
    "M3_E0": ("outputs/aaai/two_stage_kernel_ridge/split_metrics.csv", "method_id"),
    "E1": ("outputs/aaai/final/E1_node_true/split_metrics.csv", "experiment_id"),
}


def holm_adjust(pvals) -> np.ndarray:
    p = np.asarray(list(pvals), dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def bootstrap_ci(diff: np.ndarray, n_bootstrap: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    boots = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        boots[i] = rng.choice(diff, len(diff), replace=True).mean()
    return np.quantile(boots, [0.025, 0.975])


def cohens_dz(diff: np.ndarray) -> float:
    diff = np.asarray(diff, dtype=float)
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 1e-12 else 0.0


def safe_wilcoxon(diff: np.ndarray):
    try:
        result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return 0.0, 1.0


def load_seed_level(experiment_id: str) -> pd.DataFrame:
    path, key = DATA_SOURCES[experiment_id]
    split_df = pd.read_csv(path)
    split_df = split_df[split_df[key] == experiment_id]
    missing = {"seed", "pearson", "rmse", "mae"}.difference(split_df.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    if split_df.seed.nunique() != 10:
        raise ValueError(f"{experiment_id} has {split_df.seed.nunique()} seeds; expected 10")
    return (
        split_df.groupby("seed", as_index=False)[["pearson", "rmse", "mae"]]
        .mean(numeric_only=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("outputs/aaai/statistics_iclr"))
    parser.add_argument("--bootstrap", type=int, default=20000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for a, b in COMPARISONS:
        merged = load_seed_level(a).merge(load_seed_level(b), on="seed", suffixes=("_a", "_b"))
        if len(merged) != 10:
            raise ValueError(f"{a} vs {b}: expected 10 paired seeds, got {len(merged)}")
        for metric in METRICS:
            x = merged[f"{metric}_a"].to_numpy(float)
            y = merged[f"{metric}_b"].to_numpy(float)
            improvement = (y - x) if metric in LOWER_BETTER else (x - y)
            t = ttest_rel(x, y)
            w_stat, w_p = safe_wilcoxon(improvement)
            lo, hi = bootstrap_ci(improvement, args.bootstrap)
            rows.append({
                "A": a, "B": b, "metric": metric,
                "analysis_unit": "seed_mean_over_5_folds",
                "n_pairs": len(merged),
                "A_mean": float(x.mean()), "A_std": float(x.std(ddof=1)),
                "B_mean": float(y.mean()), "B_std": float(y.std(ddof=1)),
                "improvement_mean": float(improvement.mean()),
                "bootstrap95_low": float(lo), "bootstrap95_high": float(hi),
                "paired_t": float(t.statistic), "paired_t_p": float(t.pvalue),
                "wilcoxon_W": w_stat, "wilcoxon_p": w_p,
                "cohens_dz": cohens_dz(improvement),
            })

    result = pd.DataFrame(rows)
    result["wilcoxon_p_holm_metric"] = np.nan
    for metric, idx in result.groupby("metric").groups.items():
        idx = list(idx)
        result.loc[idx, "wilcoxon_p_holm_metric"] = holm_adjust(
            result.loc[idx, "wilcoxon_p"].to_numpy(float)
        )
    result["wilcoxon_p_holm_global"] = holm_adjust(result["wilcoxon_p"])
    result["significant_wilcoxon_holm_005"] = result["wilcoxon_p_holm_metric"] < 0.05
    result["ci_excludes_zero"] = (
        (result["bootstrap95_low"] > 0) | (result["bootstrap95_high"] < 0)
    )

    result.to_csv(args.out / "seed_level_paired_statistical_tests_iclr.csv", index=False)
    (args.out / "seed_level_paired_statistical_tests_iclr.tex").write_text(
        result.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    print(f"\nSaved experiment-3 outputs to {args.out}")


if __name__ == "__main__":
    main()