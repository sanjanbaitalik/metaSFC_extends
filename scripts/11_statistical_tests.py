#!/usr/bin/env python3
"""Paper-ready paired statistical tests using seed-level means.

The original 50 seed-fold rows are not independent because folds within a seed
share the same repeated-CV realization and subjects recur across folds. This
script first averages all folds within each seed, yielding 10 paired units per
experiment. Paired t-tests, Wilcoxon tests, bootstrap confidence intervals,
effect sizes and Holm corrections are then computed on those seed-level units.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

LOWER_BETTER = {"rmse", "mae"}
DEFAULT_COMPARISONS = [
    ("E1", "E0"), ("E1", "E2"), ("E1", "E3"), ("E1", "E10"),
    ("E4", "E0"), ("E4", "E5"), ("E4", "E6"),
    ("E7", "E0"), ("E7", "E8"), ("E7", "E9"),
    ("NCR_TRUE", "B3"), ("NCR_TRUE", "E0"),
    ("M2_TRUE", "E0"), ("M2_TRUE", "E1"),
    ("M3_E0", "E0"), ("M3_E0", "E1"), ("M3_E0", "E7"),
]
DEFAULT_METRICS = [
    "pearson", "rmse", "mae",
    "reference_alignment_node_pearson",
    "reference_alignment_module_pearson",
    "reference_alignment_edge_diagonal_pearson",
    "alignment_node_pearson",
    "alignment_module_pearson",
    "alignment_edge_diagonal_pearson",
]
# Extra per-method split CSVs (ICLR methods and the fusion ridge baseline)
# merged into the shared split table.  The ICLR files use method_id/method_name
# columns; the prediction baselines use experiment_id/experiment_name.
EXTRA_SPLIT_SOURCES = [
    ("outputs/aaai/network_constrained_ridge/split_metrics.csv", "method_id"),
    ("outputs/aaai/meta_gat/split_metrics.csv", "method_id"),
    ("outputs/aaai/two_stage_kernel_ridge/split_metrics.csv", "method_id"),
    ("outputs/aaai/prediction_baselines/prediction_baselines_split_metrics.csv", "experiment_id"),
]


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


def holm_adjust(pvals: Iterable[float]) -> np.ndarray:
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


def seed_aggregate(df: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    available = [m for m in metrics if m in df.columns]
    return (
        df.groupby(["experiment_id", "seed"], as_index=False)[available]
        .mean(numeric_only=True)
    )


def safe_wilcoxon(diff: np.ndarray):
    try:
        result = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return 0.0, 1.0


def load_split_metrics(table_dir: Path) -> pd.DataFrame:
    split_df = pd.read_csv(table_dir / "all_split_metrics.csv")
    for path, key in EXTRA_SPLIT_SOURCES:
        csv_path = Path(path)
        if not csv_path.exists():
            print(f"[WARN] missing extra split source: {csv_path}")
            continue
        extra = pd.read_csv(csv_path)
        if "experiment_id" not in extra.columns:
            extra = extra.rename(columns={"method_id": "experiment_id"})
        if "experiment_name" not in extra.columns and "method_name" in extra.columns:
            extra = extra.rename(columns={"method_name": "experiment_name"})
        keep = [c for c in extra.columns if c in split_df.columns or c in {"experiment_id", "experiment_name", "seed", "fold"}]
        split_df = pd.concat([split_df, extra[keep]], ignore_index=True)
    return split_df.drop_duplicates(subset=["experiment_id", "seed", "fold"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table_dir", default="outputs/aaai/tables")
    ap.add_argument("--out", default="outputs/aaai/statistics")
    ap.add_argument("--bootstrap", type=int, default=20000)
    args = ap.parse_args()

    table_dir = Path(args.table_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split_df = load_split_metrics(table_dir)
    metrics = [m for m in DEFAULT_METRICS if m in split_df.columns]
    seed_df = seed_aggregate(split_df, metrics)
    seed_df.to_csv(out / "seed_level_method_metrics.csv", index=False)

    rows = []
    for a, b in DEFAULT_COMPARISONS:
        da = seed_df[seed_df.experiment_id == a]
        db = seed_df[seed_df.experiment_id == b]
        merged = da.merge(db, on="seed", suffixes=("_a", "_b"))
        if merged.empty:
            continue
        for metric in metrics:
            ca, cb = f"{metric}_a", f"{metric}_b"
            if ca not in merged or cb not in merged:
                continue
            x = merged[ca].to_numpy(float)
            y = merged[cb].to_numpy(float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if len(x) < 3:
                continue
            # Positive improvement means method A is better in every metric.
            improvement = (y - x) if metric in LOWER_BETTER else (x - y)
            t = ttest_rel(x, y)
            w_stat, w_p = safe_wilcoxon(improvement)
            lo, hi = bootstrap_ci(improvement, args.bootstrap)
            rows.append(
                {
                    "A": a,
                    "B": b,
                    "metric": metric,
                    "analysis_unit": "seed_mean_over_5_folds",
                    "n_pairs": len(improvement),
                    "A_mean": x.mean(),
                    "A_std": x.std(ddof=1),
                    "B_mean": y.mean(),
                    "B_std": y.std(ddof=1),
                    "improvement_mean": improvement.mean(),
                    "bootstrap95_low": lo,
                    "bootstrap95_high": hi,
                    "paired_t": float(t.statistic),
                    "paired_t_p": float(t.pvalue),
                    "wilcoxon_W": w_stat,
                    "wilcoxon_p": w_p,
                    "cohens_dz": cohens_dz(improvement),
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        # Correct separately within each metric family and also globally.
        result["paired_t_p_holm_global"] = holm_adjust(result["paired_t_p"])
        result["wilcoxon_p_holm_global"] = holm_adjust(result["wilcoxon_p"])
        result["paired_t_p_holm_metric"] = np.nan
        result["wilcoxon_p_holm_metric"] = np.nan
        for metric, idx in result.groupby("metric").groups.items():
            idx = list(idx)
            result.loc[idx, "paired_t_p_holm_metric"] = holm_adjust(result.loc[idx, "paired_t_p"])
            result.loc[idx, "wilcoxon_p_holm_metric"] = holm_adjust(result.loc[idx, "wilcoxon_p"])
        result["significant_wilcoxon_holm_005"] = result["wilcoxon_p_holm_metric"] < 0.05
        result["ci_excludes_zero"] = (
            (result["bootstrap95_low"] > 0) | (result["bootstrap95_high"] < 0)
        )

    csv_path = out / "seed_level_paired_statistical_tests.csv"
    tex_path = out / "seed_level_paired_statistical_tests.tex"
    result.to_csv(csv_path, index=False)

    paper_cols = [
        "A", "B", "metric", "n_pairs", "A_mean", "B_mean",
        "improvement_mean", "bootstrap95_low", "bootstrap95_high",
        "wilcoxon_p_holm_metric", "cohens_dz",
    ]
    paper = result[paper_cols].copy() if not result.empty else result
    tex_path.write_text(
        paper.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )
    # Preserve compatibility with existing downstream references.
    result.to_csv(out / "paired_statistical_tests.csv", index=False)
    (out / "paired_statistical_tests.tex").write_text(
        paper.to_latex(index=False, escape=False, float_format="%.4f"),
        encoding="utf-8",
    )
    print(result.to_string(index=False))
    print(f"Saved seed-level paper-ready statistics to {out}")


if __name__ == "__main__":
    main()
