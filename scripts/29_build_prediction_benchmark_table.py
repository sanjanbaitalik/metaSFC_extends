#!/usr/bin/env python3
"""Merge predictive benchmarks, bold the best means, and compare the best
predictor with every alternative by paired Wilcoxon signed-rank tests.

The statistical unit is the seed-level mean over the five outer folds.  The
best predictor is selected from the main-paper methods using a configurable
primary metric (Pearson by default).  Two-sided paired Wilcoxon tests are
Holm-adjusted separately within Pearson, RMSE, and MAE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metascfc.benchmark_utils import (
    aggregate_split_metrics,
    atomic_write_csv,
    paired_wilcoxon_vs_reference,
    seed_level_metrics,
)


METHOD_ORDER = [
    "B0", "B1", "B2", "B3",
    "PW_TRUE", "PW_SHUFFLED", "PW_RANDOM",
    "MGCN", "IMG_GCN", "MS_INTER", "METASFC",
]
MAIN_IDS = [
    "B0", "B1", "B2", "PW_TRUE", "B3",
    "MGCN", "IMG_GCN", "MS_INTER", "METASFC",
]
MAIN_FAMILIES = {
    "linear": ["B0", "B1", "B2", "PW_TRUE", "B3"],
    "graph": ["MGCN", "IMG_GCN", "MS_INTER", "METASFC"],
}
METRIC_SPECS = {
    "pearson": {"mean": "Pearson Mean", "std": "Pearson Std", "higher": True},
    "rmse": {"mean": "RMSE Mean", "std": "RMSE Std", "higher": False},
    "mae": {"mean": "MAE Mean", "std": "MAE Std", "higher": False},
}


def standardize_existing_fast(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).rename(
        columns={"experiment_id": "method_id", "experiment_name": "method_name"}
    )


def standardize_graph(path: Path, method_id: str, method_name: str) -> pd.DataFrame:
    df = pd.read_csv(path).copy()
    df["method_id"] = method_id
    df["method_name"] = method_name
    return df


def choose_best_method(summary: pd.DataFrame, metric: str) -> str:
    if metric not in METRIC_SPECS:
        raise ValueError(f"Unsupported primary metric: {metric}")
    spec = METRIC_SPECS[metric]
    values = pd.to_numeric(summary[spec["mean"]], errors="coerce")
    if values.isna().all():
        raise ValueError(f"No finite values are available for {metric}")
    index = values.idxmax() if spec["higher"] else values.idxmin()
    return str(summary.loc[index, "ID"])


def bold_best_metric_values(
    frame: pd.DataFrame,
    significance: pd.DataFrame | None = None,
    reference_id: str | None = None,
    tolerance: float = 1e-12,
) -> pd.DataFrame:
    """Format metrics, bold the numerical best, and mark significant differences.

    A superscript star marks a method whose paired seed-level value differs
    significantly from the best/reference method after metric-wise Holm
    correction. The reference itself is never starred.
    """
    display = frame[[
        "ID", "Method",
        "Pearson Mean", "Pearson Std",
        "RMSE Mean", "RMSE Std",
        "MAE Mean", "MAE Std",
    ]].copy()

    significant_pairs: set[tuple[str, str]] = set()
    if significance is not None and not significance.empty:
        for _, row in significance.iterrows():
            if bool(row.get("significant_holm_005", False)):
                significant_pairs.add((str(row["method_id"]), str(row["metric"])))

    formatted = display[["ID", "Method"]].copy()
    for metric, spec in METRIC_SPECS.items():
        means = pd.to_numeric(display[spec["mean"]], errors="raise")
        stds = pd.to_numeric(display[spec["std"]], errors="raise")
        best = means.max() if spec["higher"] else means.min()
        cells = []
        for method_id, mean, std in zip(display["ID"], means, stds):
            value = f"{mean:.3f} \\pm {std:.3f}"
            is_significant = (
                reference_id is not None
                and str(method_id) != str(reference_id)
                and (str(method_id), metric) in significant_pairs
            )
            star = "^{*}" if is_significant else ""
            if np.isclose(mean, best, rtol=0.0, atol=tolerance):
                cell = f"$\\mathbf{{{value}}}{star}$"
            else:
                cell = f"${value}{star}$"
            cells.append(cell)
        label = {
            "pearson": "Pearson $\\uparrow$",
            "rmse": "RMSE $\\downarrow$",
            "mae": "MAE $\\downarrow$",
        }[metric]
        formatted[label] = cells
    return formatted


def order_main_table(frame: pd.DataFrame, reference_id: str) -> pd.DataFrame:
    """Keep model families together and place the best model last in its family."""
    ordered_ids: list[str] = []
    present_ids = set(frame["ID"].astype(str))
    for ids in MAIN_FAMILIES.values():
        present = [method_id for method_id in ids if method_id in present_ids]
        if reference_id in present:
            present = [method_id for method_id in present if method_id != reference_id] + [reference_id]
        ordered_ids.extend(present)
    remaining = [method_id for method_id in frame["ID"] if method_id not in ordered_ids]
    ordered_ids.extend(remaining)
    order = {method_id: index for index, method_id in enumerate(ordered_ids)}
    out = frame.copy()
    out["_paper_order"] = out["ID"].map(order).fillna(999)
    return out.sort_values("_paper_order").drop(columns="_paper_order")


def write_prediction_tex(
    frame: pd.DataFrame,
    path: Path,
    caption: str,
    label: str,
    significance: pd.DataFrame | None = None,
    reference_id: str | None = None,
) -> None:
    display = bold_best_metric_values(
        frame,
        significance=significance,
        reference_id=reference_id,
    ).drop(columns="ID")
    tabular = display.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
    )
    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption} Bold denotes the best mean in each metric; "
        "$^{*}$ denotes a significant difference from the best predictor "
        "under a paired two-sided Wilcoxon signed-rank test with metric-wise "
        "Holm correction ($p_{\\mathrm{Holm}}<0.05$).}\n"
        f"\\label{{{label}}}\n"
        + tabular
        + "\\end{table*}\n"
    )
    path.write_text(tex, encoding="utf-8")


def format_pvalue(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if value < 0.001:
        return "$<0.001$"
    return f"{value:.3f}"


def write_wilcoxon_tex(
    stats: pd.DataFrame,
    summary: pd.DataFrame,
    reference_id: str,
    path: Path,
    metric: str = "pearson",
) -> None:
    subset = stats[stats.metric == metric].copy()
    name_map = summary.set_index("ID")["Method"].to_dict()
    reference_name = name_map[reference_id]

    reference_row = pd.DataFrame([{
        "Method": reference_name,
        "Mean": float(summary.loc[summary.ID == reference_id, METRIC_SPECS[metric]["mean"]].iloc[0]),
        "$\\Delta$ from best": 0.0,
        "$W$": "--",
        "$p_{\\mathrm{Holm}}$": "--",
        "Result": "Reference",
    }])

    rows = []
    for _, row in subset.iterrows():
        rows.append({
            "Method": row["method_name"],
            "Mean": row["method_mean"],
            "$\\Delta$ from best": row["method_mean"] - row["reference_mean"],
            "$W$": f"{row['wilcoxon_W']:.1f}",
            "$p_{\\mathrm{Holm}}$": format_pvalue(row["wilcoxon_p_holm_metric"]),
            "Result": (
                "Significant" if bool(row["significant_holm_005"])
                else "Not significant"
            ),
        })
    paper = pd.concat([reference_row, pd.DataFrame(rows)], ignore_index=True)
    paper["Mean"] = paper["Mean"].map(lambda x: f"{float(x):.3f}")
    paper["$\\Delta$ from best"] = paper["$\\Delta$ from best"].map(
        lambda x: f"{float(x):+.3f}"
    )
    tabular = paper.to_latex(
        index=False,
        escape=False,
        column_format="lccccc",
    )
    metric_name = {"pearson": "Pearson correlation", "rmse": "RMSE", "mae": "MAE"}[metric]
    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{Paired seed-level Wilcoxon signed-rank comparisons of "
        f"{metric_name} against the best predictor, {reference_name}. "
        "The five folds are averaged within each seed ($n=10$ paired seeds); "
        "two-sided $p$-values are Holm-adjusted across methods for this metric. "
        "A nonsignificant result does not establish equivalence.}}\n"
        f"\\label{{tab:prediction_wilcoxon_{metric}}}\n"
        + tabular
        + "\\end{table*}\n"
    )
    path.write_text(tex, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="outputs/aaai/paper_prediction_benchmark")
    ap.add_argument(
        "--primary-metric",
        choices=sorted(METRIC_SPECS),
        default="pearson",
        help="Metric used to identify the best predictor for paired comparisons.",
    )
    ap.add_argument(
        "--reference-id",
        default=None,
        help="Optional explicit best/reference method ID. By default it is selected from the main table.",
    )
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    required = [
        Path("outputs/aaai/prediction_baselines/prediction_baselines_split_metrics.csv"),
        Path("outputs/aaai/prior_weighted_ridge/split_metrics.csv"),
        Path("outputs/aaai/sota_graph_baselines/split_metrics.csv"),
        Path("outputs/aaai/final/E0_baseline/split_metrics.csv"),
        Path("outputs/aaai/final/E1_node_true/split_metrics.csv"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required result files:\n  " + "\n  ".join(missing)
        )

    sources = [
        standardize_existing_fast(required[0]),
        pd.read_csv(required[1]),
        pd.read_csv(required[2]),
        standardize_graph(required[3], "MS_INTER", "No-prior MS-Inter-GCN"),
        standardize_graph(required[4], "METASFC", "MetaSFC (ours)"),
    ]
    keep = [
        "method_id", "method_name", "seed", "fold", "split_id",
        "pearson", "rmse", "mae", "runtime_seconds",
    ]
    all_df = pd.concat([frame[keep] for frame in sources], ignore_index=True)

    duplicate = all_df.duplicated(["method_id", "seed", "fold"])
    if duplicate.any():
        raise ValueError(
            "Duplicate method/seed/fold rows:\n"
            + all_df.loc[duplicate, ["method_id", "seed", "fold"]].to_string(index=False)
        )

    counts = all_df.groupby("method_id").size()
    incomplete = counts[counts < 50]
    if len(incomplete):
        print("WARNING: incomplete methods:\n", incomplete.to_string())

    summary = aggregate_split_metrics(all_df)
    order_map = {method_id: index for index, method_id in enumerate(METHOD_ORDER)}
    summary["_order"] = summary.ID.map(order_map).fillna(999)
    summary = summary.sort_values("_order").drop(columns="_order")
    seed_df = seed_level_metrics(all_df)

    atomic_write_csv(all_df, out / "all_split_metrics.csv")
    atomic_write_csv(seed_df, out / "all_seed_level_metrics.csv")
    atomic_write_csv(summary, out / "paper_prediction_benchmark.csv")

    main_summary = summary[summary.ID.isin(MAIN_IDS)].copy()
    main_order_map = {method_id: index for index, method_id in enumerate(MAIN_IDS)}
    main_summary["_order"] = main_summary.ID.map(main_order_map)
    main_summary = main_summary.sort_values("_order").drop(columns="_order")
    extended_summary = summary.copy()
    atomic_write_csv(main_summary, out / "paper_prediction_benchmark_main.csv")
    atomic_write_csv(extended_summary, out / "paper_prediction_benchmark_extended.csv")

    if args.reference_id:
        reference_id = args.reference_id
        if reference_id not in set(main_summary.ID):
            raise ValueError(
                f"Requested reference {reference_id} is not present in the main table"
            )
    else:
        reference_id = choose_best_method(main_summary, args.primary_metric)

    best_row = main_summary[main_summary.ID == reference_id].iloc[0]
    best_metadata = {
        "reference_id": reference_id,
        "reference_name": best_row["Method"],
        "selection_metric": args.primary_metric,
        "selection_value": float(best_row[METRIC_SPECS[args.primary_metric]["mean"]]),
        "analysis_unit": "seed_mean_over_5_folds",
        "n_seeds": int(seed_df[seed_df.method_id == reference_id].seed.nunique()),
    }
    (out / "best_predictor.json").write_text(
        json.dumps(best_metadata, indent=2), encoding="utf-8"
    )

    main_seed_df = seed_df[seed_df.method_id.isin(MAIN_IDS)].copy()
    wilcoxon_stats = paired_wilcoxon_vs_reference(
        seed_df=main_seed_df,
        reference_id=reference_id,
        metrics=("pearson", "rmse", "mae"),
    )
    extended_wilcoxon_stats = paired_wilcoxon_vs_reference(
        seed_df=seed_df,
        reference_id=reference_id,
        metrics=("pearson", "rmse", "mae"),
    )
    atomic_write_csv(
        wilcoxon_stats, out / "best_vs_table1_wilcoxon_seed_level.csv"
    )
    atomic_write_csv(
        wilcoxon_stats[wilcoxon_stats.metric == args.primary_metric],
        out / f"best_vs_table1_wilcoxon_{args.primary_metric}.csv",
    )
    atomic_write_csv(
        extended_wilcoxon_stats,
        out / "best_vs_all_extended_wilcoxon_seed_level.csv",
    )

    # Place the best predictor last within its model family, as requested.
    main_summary = order_main_table(main_summary, reference_id)
    extended_summary = order_main_table(extended_summary, reference_id)
    atomic_write_csv(main_summary, out / "paper_prediction_benchmark_main_ordered.csv")
    atomic_write_csv(extended_summary, out / "paper_prediction_benchmark_extended_ordered.csv")

    write_prediction_tex(
        main_summary,
        out / "paper_prediction_benchmark_main.tex",
        "Cognitive prediction under identical repeated nested splits. "
        "Published graph architectures are reimplemented on the same "
        "participants, AAL116 connectomes, target, and partitions.",
        "tab:prediction_benchmark",
        significance=wilcoxon_stats,
        reference_id=reference_id,
    )
    write_prediction_tex(
        extended_summary,
        out / "paper_prediction_benchmark_extended.tex",
        "Extended cognitive-prediction comparison including shuffled and "
        "random prior-weighted Ridge controls.",
        "tab:prediction_benchmark_extended",
        significance=extended_wilcoxon_stats,
        reference_id=reference_id,
    )
    write_wilcoxon_tex(
        stats=wilcoxon_stats,
        summary=main_summary,
        reference_id=reference_id,
        path=out / "best_vs_table1_wilcoxon_pearson.tex",
        metric="pearson",
    )
    write_wilcoxon_tex(
        stats=wilcoxon_stats,
        summary=main_summary,
        reference_id=reference_id,
        path=out / "best_vs_table1_wilcoxon_rmse.tex",
        metric="rmse",
    )
    write_wilcoxon_tex(
        stats=wilcoxon_stats,
        summary=main_summary,
        reference_id=reference_id,
        path=out / "best_vs_table1_wilcoxon_mae.tex",
        metric="mae",
    )

    print("\nBest predictor")
    print(json.dumps(best_metadata, indent=2))
    print("\nMain prediction table")
    print(main_summary.to_string(index=False))
    print("\nWilcoxon signed-rank tests versus best predictor")
    print(wilcoxon_stats.to_string(index=False))
    print(f"\nSaved paper-ready outputs to {out}")


if __name__ == "__main__":
    main()
