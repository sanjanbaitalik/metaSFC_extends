#!/usr/bin/env python3
"""Combine deep methods, fast baselines, statistics and faithfulness into paper tables."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def format_pm(mean, std, digits=3):
    if pd.isna(mean):
        return "--"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def normalize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "Rmse Mean": "RMSE Mean", "Rmse Std": "RMSE Std",
        "Mae Mean": "MAE Mean", "Mae Std": "MAE Std",
    }
    return df.rename(columns=aliases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aaai_tables", default="outputs/aaai/tables")
    ap.add_argument("--baselines", default="outputs/aaai/prediction_baselines")
    ap.add_argument("--statistics", default="outputs/aaai/statistics")
    ap.add_argument("--faithfulness", default="outputs/aaai/faithfulness")
    ap.add_argument("--out", default="outputs/aaai/submission_tables")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    deep = normalize_prediction_columns(
        pd.read_csv(Path(args.aaai_tables) / "table1_prediction_performance.csv")
    )
    baseline_path = Path(args.baselines) / "prediction_baselines_summary.csv"
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    baseline = normalize_prediction_columns(baseline)

    deep_keep = deep[deep["ID"].isin(["E0", "E1", "E4", "E7", "E10"])].copy()
    deep_keep["Category"] = "FC-SC graph model"
    if not baseline.empty:
        baseline["Category"] = "Fast predictive baseline"
        all_pred = pd.concat([baseline, deep_keep], ignore_index=True, sort=False)
    else:
        all_pred = deep_keep
    order = {x: i for i, x in enumerate(["B0", "B1", "B2", "B3", "E0", "E1", "E4", "E7", "E10"])}
    all_pred = all_pred.sort_values("ID", key=lambda s: s.map(order).fillna(999))
    all_pred.to_csv(out / "paper_table_prediction_all.csv", index=False)

    tex = all_pred[["ID", "Method", "Pearson Mean", "Pearson Std", "RMSE Mean", "RMSE Std", "MAE Mean", "MAE Std"]].copy()
    tex["Pearson $\\uparrow$"] = [format_pm(m, s) for m, s in zip(tex.pop("Pearson Mean"), tex.pop("Pearson Std"))]
    tex["RMSE $\\downarrow$"] = [format_pm(m, s) for m, s in zip(tex.pop("RMSE Mean"), tex.pop("RMSE Std"))]
    tex["MAE $\\downarrow$"] = [format_pm(m, s) for m, s in zip(tex.pop("MAE Mean"), tex.pop("MAE Std"))]
    (out / "paper_table_prediction_all.tex").write_text(
        tex.to_latex(index=False, escape=False), encoding="utf-8"
    )

    # Copy concise seed-level statistics table.
    stats_path = Path(args.statistics) / "seed_level_paired_statistical_tests.csv"
    if stats_path.exists():
        stats = pd.read_csv(stats_path)
        stats.to_csv(out / "paper_table_seed_level_statistics.csv", index=False)
        concise_cols = [
            "A", "B", "metric", "n_pairs", "A_mean", "B_mean", "improvement_mean",
            "bootstrap95_low", "bootstrap95_high", "wilcoxon_p_holm_metric", "cohens_dz",
        ]
        concise = stats[[c for c in concise_cols if c in stats.columns]]
        (out / "paper_table_seed_level_statistics.tex").write_text(
            concise.to_latex(index=False, escape=False, float_format="%.4f"), encoding="utf-8"
        )

    faith_path = Path(args.faithfulness) / "faithfulness_summary.csv"
    if faith_path.exists():
        faith = pd.read_csv(faith_path)
        preferred = [
            "ID", "Method", "Seeds", "Folds", "Top-k",
            "delta_rmse_top Mean", "delta_rmse_random Mean",
            "gap_rmse_top_vs_random Mean", "gap_rmse_top_vs_bottom Mean",
            "delta_pearson_top Mean", "gap_pearson_top_vs_random Mean",
        ]
        faith = faith[[c for c in preferred if c in faith.columns]]
        # Keep controls first and place the primary/best MetaSFC row last.
        faith_order = {"E0": 0, "E10": 1, "E1": 2}
        if "ID" in faith.columns:
            faith["_paper_order"] = faith["ID"].map(faith_order).fillna(999)
            faith = faith.sort_values("_paper_order").drop(columns="_paper_order")
        faith.to_csv(out / "paper_table_faithfulness.csv", index=False)
        (out / "paper_table_faithfulness.tex").write_text(
            faith.to_latex(index=False, escape=False, float_format="%.4f"), encoding="utf-8"
        )

    print(f"Saved submission tables to {out}")


if __name__ == "__main__":
    main()
