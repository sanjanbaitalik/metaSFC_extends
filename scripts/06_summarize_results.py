#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metascfc import visualize as vis


def collect_experiment_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        metrics_file = exp_dir / "metrics.json"
        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            data = json.load(f)

        config = data.get("config", {})
        row = {
            "experiment": exp_dir.name,
            "prior_type": config.get("prior_type", "unknown"),
            "lambda_node": config.get("lambda_node", 0.0),
            "lambda_module": config.get("lambda_module", 0.0),
            "lambda_edge": config.get("lambda_edge", 0.0),
        }
        for key, val in data.items():
            if key not in ("config", "fold_metrics", "fold_aux"):
                row[key] = val
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Summarize experiment results")
    parser.add_argument("--results_dir", type=str, default="outputs/experiments",
                        help="Directory containing experiment subdirectories")
    parser.add_argument("--out", type=str, default="outputs/summary",
                        help="Output directory for summary")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists():
        print(f"No results directory found at {results_dir}")
        return

    summary_df = collect_experiment_results(results_dir)
    if len(summary_df) == 0:
        print(f"No experiment results found in {results_dir}")
        return

    summary_path = out_dir / "experiment_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    print("\n=== Experiment Summary ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(summary_df.to_string(index=False))

    metric_cols = [c for c in summary_df.columns if c.endswith("_mean")]
    if "prior_type" in summary_df.columns and metric_cols:
        vis.set_style()
        for metric in metric_cols:
            if metric.replace("_mean", "_std") in summary_df.columns:
                fig_path = out_dir / f"{metric}_comparison.png"
                plot_data = summary_df[["prior_type", metric, metric.replace("_mean", "_std")]].copy()
                plot_data = plot_data.rename(columns={
                    "prior_type": "condition",
                    metric: "value",
                    metric.replace("_mean", "_std"): "std",
                })
                vis.plot_prediction_summary(
                    plot_data.set_index("condition"),
                    metric="value",
                    title=f"{metric} by Prior Type",
                    save_path=fig_path,
                )
                print(f"Saved {fig_path}")

    print(f"\nAll summaries in {out_dir}")


if __name__ == "__main__":
    main()
