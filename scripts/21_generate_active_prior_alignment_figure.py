#!/usr/bin/env python3
"""Generate Figure 3 using the active prior level for each experiment.

The script reads all E0--E10 split_metrics.csv files under
outputs/aaai/final, averages the five folds within each seed, and then
computes the mean and 95% t-confidence interval across seeds.

Active alignment metric:
    E0--E3, E10 : reference_alignment_node_pearson
    E4--E6      : reference_alignment_module_pearson
    E7--E9      : reference_alignment_edge_pearson

Outputs:
    figures/fig3_active_prior_alignment.pdf
    figures/fig3_active_prior_alignment.png
    outputs/aaai/tables/figure3_active_prior_alignment_seed_level.csv
    outputs/aaai/tables/figure3_active_prior_alignment_summary.csv
    outputs/aaai/tables/figure3_active_prior_alignment_summary.tex

Example:
    python scripts/21_generate_active_prior_alignment_figure.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPERIMENT_ORDER = [f"E{i}" for i in range(11)]

ACTIVE_LEVEL = {
    "E0": "node",
    "E1": "node",
    "E2": "node",
    "E3": "node",
    "E4": "module",
    "E5": "module",
    "E6": "module",
    "E7": "edge",
    "E8": "edge",
    "E9": "edge",
    "E10": "node",
}

COLUMN_CANDIDATES = {
    "node": [
        "reference_alignment_node_pearson",
        "reference_node_alignment_pearson",
        "ref_node_pearson",
    ],
    "module": [
        "reference_alignment_module_pearson",
        "reference_module_alignment_pearson",
        "ref_module_pearson",
    ],
    "edge": [
        "reference_alignment_edge_pearson",
        "reference_edge_alignment_pearson",
        "ref_edge_pearson",
    ],
}

# Two-sided 95% Student-t critical values indexed by degrees of freedom.
# Includes the expected df=9 for ten seeds.
T_CRITICAL_975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def experiment_sort_key(experiment_id: str) -> int:
    match = re.fullmatch(r"E(\d+)", str(experiment_id))
    return int(match.group(1)) if match else 10_000


def resolve_column(df: pd.DataFrame, level: str, source: Path) -> str:
    for candidate in COLUMN_CANDIDATES[level]:
        if candidate in df.columns:
            return candidate

    available = "\n  - ".join(sorted(df.columns))
    raise KeyError(
        f"Could not find a {level}-reference alignment column in {source}.\n"
        f"Tried: {COLUMN_CANDIDATES[level]}\n"
        f"Available columns:\n  - {available}"
    )


def infer_experiment_id(df: pd.DataFrame, path: Path) -> str:
    if "experiment_id" in df.columns:
        ids = df["experiment_id"].dropna().astype(str).unique().tolist()
        if len(ids) == 1:
            return ids[0]

    for part in reversed(path.parts):
        match = re.search(r"\b(E(?:10|[0-9]))\b", part)
        if match:
            return match.group(1)

    raise ValueError(f"Could not infer experiment ID from {path}")


def load_seed_level_metrics(results_root: Path) -> pd.DataFrame:
    files = sorted(results_root.rglob("split_metrics.csv"))
    if not files:
        raise FileNotFoundError(
            f"No split_metrics.csv files were found below {results_root}."
        )

    records: list[pd.DataFrame] = []
    found: set[str] = set()

    for path in files:
        df = pd.read_csv(path)
        experiment_id = infer_experiment_id(df, path)

        if experiment_id not in ACTIVE_LEVEL:
            continue
        if experiment_id in found:
            raise RuntimeError(
                f"More than one split_metrics.csv was found for "
                f"{experiment_id}. Remove stale duplicate result folders.\n"
                f"Second file: {path}"
            )

        required = {"seed", "fold"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        level = ACTIVE_LEVEL[experiment_id]
        metric_column = resolve_column(df, level, path)

        selected = df[["seed", "fold", metric_column]].copy()
        selected = selected.rename(columns={metric_column: "alignment"})
        selected["experiment_id"] = experiment_id
        selected["active_level"] = level
        selected["source_file"] = str(path)

        # Five folds are averaged within each seed. The seed is the unit
        # used for the confidence interval.
        seed_level = (
            selected.groupby(
                ["experiment_id", "active_level", "seed"],
                as_index=False,
            )
            .agg(
                alignment=("alignment", "mean"),
                n_folds=("fold", "nunique"),
            )
        )

        records.append(seed_level)
        found.add(experiment_id)

    missing_experiments = [
        experiment_id
        for experiment_id in EXPERIMENT_ORDER
        if experiment_id not in found
    ]
    if missing_experiments:
        raise FileNotFoundError(
            "Missing result files for: " + ", ".join(missing_experiments)
        )

    result = pd.concat(records, ignore_index=True)
    result["experiment_number"] = result["experiment_id"].map(
        experiment_sort_key
    )
    return result.sort_values(["experiment_number", "seed"]).drop(
        columns="experiment_number"
    )


def summarize(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for experiment_id in EXPERIMENT_ORDER:
        group = seed_level[
            seed_level["experiment_id"] == experiment_id
        ].copy()
        values = group["alignment"].dropna().to_numpy(dtype=float)
        n = len(values)

        if n < 2:
            raise ValueError(
                f"{experiment_id} has only {n} usable seed-level values."
            )

        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        sem = sd / np.sqrt(n)
        dfree = n - 1
        t_critical = T_CRITICAL_975.get(dfree, 1.96)
        ci_half_width = float(t_critical * sem)

        rows.append(
            {
                "experiment_id": experiment_id,
                "active_level": ACTIVE_LEVEL[experiment_id],
                "n_seeds": n,
                "mean_alignment": mean,
                "sd_across_seeds": sd,
                "sem": sem,
                "ci95_low": mean - ci_half_width,
                "ci95_high": mean + ci_half_width,
                "ci95_half_width": ci_half_width,
            }
        )

    return pd.DataFrame(rows)


def save_latex_table(summary: pd.DataFrame, output_path: Path) -> None:
    paper = summary.copy()
    paper["Prior level"] = paper["active_level"].map(
        {
            "node": "Node",
            "module": "Module",
            "edge": "Corresponding edge",
        }
    )
    paper["Alignment"] = paper.apply(
        lambda row: (
            f"{row['mean_alignment']:.3f} "
            f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]"
        ),
        axis=1,
    )
    paper = paper[
        ["experiment_id", "Prior level", "n_seeds", "Alignment"]
    ].rename(
        columns={
            "experiment_id": "ID",
            "n_seeds": "Seeds",
        }
    )

    latex = paper.to_latex(
        index=False,
        escape=False,
        column_format="llcl",
        caption=(
            "Task-specific reference alignment at each experiment's "
            "active prior level. Values are seed-level means with "
            "95\\% confidence intervals after averaging folds within "
            "each seed."
        ),
        label="tab:active_prior_alignment",
    )
    output_path.write_text(latex, encoding="utf-8")


def plot_figure(
    summary: pd.DataFrame,
    output_pdf: Path,
    output_png: Path,
    dpi: int,
) -> None:
    summary = summary.set_index("experiment_id").loc[
        EXPERIMENT_ORDER
    ].reset_index()

    x = np.arange(len(summary))
    means = summary["mean_alignment"].to_numpy(dtype=float)
    errors = summary["ci95_half_width"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8.5, 4.2))

    bars = ax.bar(
        x,
        means,
        yerr=errors,
        capsize=3,
        edgecolor="black",
        linewidth=0.7,
    )

    ax.axhline(0.0, linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["experiment_id"].tolist())
    ax.set_xlabel("Experiment")
    ax.set_ylabel("Reference alignment (Pearson $r$)")
    ax.set_title("Task-specific reference alignment")
    ax.grid(axis="y", alpha=0.25)

    # Add compact active-level labels above or below each bar.
    level_abbrev = {"node": "N", "module": "M", "edge": "CE"}
    for bar, mean, error, level in zip(
        bars,
        means,
        errors,
        summary["active_level"],
    ):
        if mean >= 0:
            y = mean + error + 0.035
            va = "bottom"
        else:
            y = mean - error - 0.035
            va = "top"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            level_abbrev[level],
            ha="center",
            va=va,
            fontsize=8,
        )

    ax.text(
        0.995,
        0.02,
        "N: node   M: module   CE: corresponding edge",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )

    lower = min(-0.85, float(np.min(means - errors)) - 0.10)
    upper = max(1.10, float(np.max(means + errors)) + 0.10)
    ax.set_ylim(lower, upper)

    fig.tight_layout()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("outputs/aaai/final"),
        help="Directory containing E0--E10 result folders.",
    )
    parser.add_argument(
        "--figure-pdf",
        type=Path,
        default=Path("figures/fig3_active_prior_alignment.pdf"),
    )
    parser.add_argument(
        "--figure-png",
        type=Path,
        default=Path("figures/fig3_active_prior_alignment.png"),
    )
    parser.add_argument(
        "--table-dir",
        type=Path,
        default=Path("outputs/aaai/tables"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    seed_level = load_seed_level_metrics(args.results_root)
    summary = summarize(seed_level)

    args.table_dir.mkdir(parents=True, exist_ok=True)
    seed_csv = (
        args.table_dir
        / "figure3_active_prior_alignment_seed_level.csv"
    )
    summary_csv = (
        args.table_dir
        / "figure3_active_prior_alignment_summary.csv"
    )
    summary_tex = (
        args.table_dir
        / "figure3_active_prior_alignment_summary.tex"
    )

    seed_level.to_csv(seed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    save_latex_table(summary, summary_tex)

    plot_figure(
        summary=summary,
        output_pdf=args.figure_pdf,
        output_png=args.figure_png,
        dpi=args.dpi,
    )

    print("\nActive-prior alignment summary")
    print(
        summary[
            [
                "experiment_id",
                "active_level",
                "n_seeds",
                "mean_alignment",
                "ci95_low",
                "ci95_high",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:")
    print(f"  {args.figure_pdf}")
    print(f"  {args.figure_png}")
    print(f"  {seed_csv}")
    print(f"  {summary_csv}")
    print(f"  {summary_tex}")


if __name__ == "__main__":
    main()
