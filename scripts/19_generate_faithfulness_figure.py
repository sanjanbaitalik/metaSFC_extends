#!/usr/bin/env python3
"""Generate the paper faithfulness perturbation figure.

Example:
    python scripts/19_generate_faithfulness_figure.py \
        --faithfulness-dir outputs/aaai/faithfulness \
        --output figures/faithfulness_perturbation.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_ORDER = ["E0", "E1", "E10"]
METHOD_LABELS = {
    "E0": "Unregularized",
    "E1": "Working-memory prior",
    "E10": "Visual prior",
}


def mean_ci95(values: np.ndarray) -> tuple[float, float]:
    """Return mean and two-sided 95% t interval half-width."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean = float(values.mean())
    if len(values) == 1:
        return mean, 0.0

    # Exact t critical for five seeds (df=4); normal approximation otherwise.
    t_critical = 2.7764451051977987 if len(values) == 5 else 1.96
    sem = values.std(ddof=1) / np.sqrt(len(values))
    return mean, float(t_critical * sem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--faithfulness-dir",
        type=Path,
        default=Path("outputs/aaai/faithfulness"),
        help="Directory containing faithfulness_all_split_metrics.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/faithfulness_perturbation.jpg"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    metrics_path = args.faithfulness_dir / "faithfulness_all_split_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing {metrics_path}. Run scripts/17_run_faithfulness.py first."
        )

    df = pd.read_csv(metrics_path)
    required = {
        "experiment_id",
        "seed",
        "fold",
        "delta_rmse_top",
        "delta_rmse_random",
        "delta_rmse_bottom",
        "gap_rmse_top_vs_random",
        "gap_rmse_top_vs_bottom",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for experiment_id in METHOD_ORDER:
        rows = df[df["experiment_id"] == experiment_id]
        if rows.empty:
            raise ValueError(f"No rows found for {experiment_id}.")
        if rows["seed"].nunique() < 5:
            print(
                f"[WARN] {experiment_id} has only "
                f"{rows['seed'].nunique()} seeds."
            )

    # Treat the seed as the analysis unit by averaging folds inside each seed.
    seed_df = (
        df.groupby(["experiment_id", "seed"], as_index=False)
        .agg(
            delta_rmse_top=("delta_rmse_top", "mean"),
            delta_rmse_random=("delta_rmse_random", "mean"),
            delta_rmse_bottom=("delta_rmse_bottom", "mean"),
            gap_rmse_top_vs_random=("gap_rmse_top_vs_random", "mean"),
            gap_rmse_top_vs_bottom=("gap_rmse_top_vs_bottom", "mean"),
        )
    )

    fig, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.7), constrained_layout=True
    )
    x = np.arange(len(METHOD_ORDER))

    conditions = [
        ("delta_rmse_top", "Learned top-10"),
        ("delta_rmse_random", "Random 10"),
        ("delta_rmse_bottom", "Learned bottom-10"),
    ]
    width = 0.24
    for index, (column, label) in enumerate(conditions):
        means, errors = [], []
        for method in METHOD_ORDER:
            values = seed_df.loc[
                seed_df["experiment_id"] == method, column
            ].to_numpy()
            mean, ci = mean_ci95(values)
            means.append(mean)
            errors.append(ci)

        axes[0].bar(
            x + (index - 1) * width,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            label=label,
            edgecolor="black",
            linewidth=0.6,
        )

    axes[0].axhline(0.0, linewidth=0.8, linestyle="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        [METHOD_LABELS[m] for m in METHOD_ORDER],
        rotation=12,
        ha="right",
    )
    axes[0].set_ylabel(
        r"Increase in RMSE after masking ($\Delta$RMSE)"
    )
    axes[0].set_title("(a) Predictive damage from ROI masking")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.25)

    gaps = [
        ("gap_rmse_top_vs_random", "Top − random"),
        ("gap_rmse_top_vs_bottom", "Top − bottom"),
    ]
    width = 0.32
    for index, (column, label) in enumerate(gaps):
        means, errors = [], []
        for method in METHOD_ORDER:
            values = seed_df.loc[
                seed_df["experiment_id"] == method, column
            ].to_numpy()
            mean, ci = mean_ci95(values)
            means.append(mean)
            errors.append(ci)

        axes[1].bar(
            x + (index - 0.5) * width,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            label=label,
            edgecolor="black",
            linewidth=0.6,
        )

    axes[1].axhline(0.0, linewidth=0.8, linestyle="--")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [METHOD_LABELS[m] for m in METHOD_ORDER],
        rotation=12,
        ha="right",
    )
    axes[1].set_ylabel(r"Faithfulness gap in $\Delta$RMSE")
    axes[1].set_title("(b) Top-ROI masking versus controls")
    axes[1].legend(frameon=False, fontsize=8)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Perturbation-based explanation faithfulness "
        "(top-$k$ = 10)",
        fontsize=12,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")

    # Also save a vector PDF beside the requested raster output.
    pdf_path = args.output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {args.output}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
