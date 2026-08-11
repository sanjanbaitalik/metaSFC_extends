#!/usr/bin/env python3
"""Build Table 2 with seed-level SDs and paired Wilcoxon significance.

The main comparison uses E1 (true working-memory ROI prior) as the reference
against E0 (no prior), E2 (shuffled), E3 (random), and E10 (visual control).

Analysis units
--------------
* Working-memory alignment: mean over the five folds within each seed.
* Rank/Jaccard stability: mean pairwise similarity across the five fold-level
  saliency vectors within each seed (10 fold pairs per seed).

This yields 10 matched seed-level values per method and metric. Two-sided
paired Wilcoxon signed-rank tests are Holm-adjusted separately within each
metric family. A star in the LaTeX table marks a control that is significantly
worse than MetaSFC after correction. MetaSFC is bolded and placed last.
"""
from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr, t as student_t, wilcoxon

REFERENCE_ID = "E1"
METHODS = {
    "E0": ("E0_baseline", "No-prior MS-Inter-GCN"),
    "E2": ("E2_node_shuffled", "Shuffled-prior control"),
    "E3": ("E3_node_random", "Random-prior control"),
    "E10": ("E10_node_unrelated_visual", "Visual-prior control"),
    "E1": ("E1_node_true", "MetaSFC (ours)"),
}
METHOD_ORDER = ["E0", "E2", "E3", "E10", "E1"]
ICLR_METHODS = {
    "NCR_TRUE": ("NCR_TRUE", "Network-Constrained Ridge (true prior)"),
    "NCR_SHUFFLED": ("NCR_SHUFFLED", "Network-Constrained Ridge (shuffled prior)"),
    "NCR_RANDOM": ("NCR_RANDOM", "Network-Constrained Ridge (random prior)"),
    "M2_TRUE": ("M2_TRUE", "Meta-GAT attention mass (true prior)"),
    "M2_SHUFFLED": ("M2_SHUFFLED", "Meta-GAT attention mass (shuffled prior)"),
    "M2_RANDOM": ("M2_RANDOM", "Meta-GAT attention mass (random prior)"),
    "M3_TRUE": ("M3_TRUE", "Two-stage KRR gradient saliency (true prior)"),
    "M3_SHUFFLED": ("M3_SHUFFLED", "Two-stage KRR gradient saliency (shuffled prior)"),
    "M3_RANDOM": ("M3_RANDOM", "Two-stage KRR gradient saliency (random prior)"),
    "M3_E0": ("M3_E0", "Two-stage KRR gradient saliency (no-prior biomarker)"),
}
ICLR_FAITHFULNESS_DIR = Path("outputs/aaai/faithfulness/per_experiment")
METRICS = ["wm_alignment", "rank_stability", "top10_jaccard"]


def holm_adjust(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    value = float(spearmanr(a, b).statistic)
    return 0.0 if not np.isfinite(value) else value


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    if k <= 0 or k > len(a):
        raise ValueError(f"topk must be in [1, {len(a)}], got {k}")
    ia = set(np.argpartition(a, -k)[-k:].tolist())
    ib = set(np.argpartition(b, -k)[-k:].tolist())
    union = ia | ib
    return float(len(ia & ib) / len(union)) if union else 1.0


def parse_seed_fold(path: Path) -> tuple[int, int]:
    match = re.fullmatch(r"seed(\d+)_fold(\d+)", path.stem)
    if not match:
        raise ValueError(f"Unexpected saliency filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def load_seed_level_metrics(results_root: Path, topk: int) -> pd.DataFrame:
    rows: list[dict] = []

    for experiment_id, (folder_name, method_name) in METHODS.items():
        folder = results_root / folder_name
        split_path = folder / "split_metrics.csv"
        saliency_dir = folder / "saliency"
        if not split_path.exists():
            raise FileNotFoundError(split_path)
        if not saliency_dir.exists():
            raise FileNotFoundError(saliency_dir)

        split_df = pd.read_csv(split_path)
        required = {"seed", "fold", "reference_alignment_node_pearson"}
        missing = required.difference(split_df.columns)
        if missing:
            raise ValueError(f"{split_path} is missing {sorted(missing)}")

        alignment_by_seed = (
            split_df.groupby("seed", as_index=True)["reference_alignment_node_pearson"]
            .mean()
            .to_dict()
        )

        saliency_by_seed: dict[int, list[tuple[int, np.ndarray]]] = {}
        for path in sorted(saliency_dir.glob("seed*_fold*.npz")):
            seed, fold = parse_seed_fold(path)
            with np.load(path, allow_pickle=False) as payload:
                if "node_saliency" not in payload.files:
                    raise KeyError(f"node_saliency missing from {path}")
                vector = np.asarray(payload["node_saliency"], dtype=float).reshape(-1)
            saliency_by_seed.setdefault(seed, []).append((fold, vector))

        for seed in sorted(alignment_by_seed):
            fold_vectors = sorted(saliency_by_seed.get(seed, []), key=lambda item: item[0])
            if len(fold_vectors) != 5:
                raise ValueError(
                    f"{experiment_id} seed {seed} has {len(fold_vectors)} saliency folds; expected 5"
                )
            vectors = [vector for _, vector in fold_vectors]
            rank_values = []
            jaccard_values = []
            for left, right in combinations(vectors, 2):
                rank_values.append(safe_spearman(left, right))
                jaccard_values.append(topk_jaccard(left, right, topk))

            rows.append({
                "experiment_id": experiment_id,
                "method_name": method_name,
                "seed": int(seed),
                "wm_alignment": float(alignment_by_seed[seed]),
                "rank_stability": float(np.mean(rank_values)),
                "top10_jaccard": float(np.mean(jaccard_values)),
                "n_folds": 5,
                "n_fold_pairs": len(rank_values),
            })

    seed_df = pd.DataFrame(rows)
    for experiment_id in METHOD_ORDER:
        subset = seed_df[seed_df.experiment_id == experiment_id]
        if subset.seed.nunique() != 10:
            raise ValueError(
                f"{experiment_id} has {subset.seed.nunique()} seeds; expected 10"
            )
    return seed_df


def load_iclr_seed_level_metrics(configs: list[Path], topk: int) -> pd.DataFrame:
    """Seed-level wm_alignment / rank_stability / top10_jaccard for the ICLR
    methods, from the biomarker vectors exported by scripts/17 into
    outputs/aaai/faithfulness/per_experiment/<method>/saliency/."""
    rows: list[dict] = []
    method_ids: list[str] = []
    for config_path in configs:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        method_ids.extend(str(m) for m in cfg.get("methods", {}).keys())
    for method_id in method_ids:
        if method_id not in ICLR_METHODS:
            raise ValueError(f"Unknown ICLR method {method_id}")
        folder_name, method_name = ICLR_METHODS[method_id]
        saliency_dir = ICLR_FAITHFULNESS_DIR / folder_name / "saliency"
        if not saliency_dir.exists():
            raise FileNotFoundError(
                f"ICLR faithfulness biomarkers missing for {method_id}: {saliency_dir}"
            )
        saliency_by_seed: dict[int, list[tuple[int, np.ndarray]]] = {}
        for path in sorted(saliency_dir.glob("seed*_fold*.npz")):
            seed, fold = parse_seed_fold(path)
            with np.load(path, allow_pickle=False) as payload:
                if "node_saliency" not in payload.files:
                    raise KeyError(f"node_saliency missing from {path}")
                vector = np.asarray(payload["node_saliency"], dtype=float).reshape(-1)
            saliency_by_seed.setdefault(seed, []).append((fold, vector))
        for seed in sorted(saliency_by_seed):
            fold_vectors = sorted(saliency_by_seed[seed], key=lambda item: item[0])
            if len(fold_vectors) != 5:
                raise ValueError(
                    f"{method_id} seed {seed} has {len(fold_vectors)} folds; expected 5"
                )
            vectors = [vector for _, vector in fold_vectors]
            rank_values = []
            jaccard_values = []
            for left, right in combinations(vectors, 2):
                rank_values.append(safe_spearman(left, right))
                jaccard_values.append(topk_jaccard(left, right, topk))
            rows.append({
                "experiment_id": method_id,
                "method_name": method_name,
                "seed": int(seed),
                "wm_alignment": float(np.nan),
                "rank_stability": float(np.mean(rank_values)),
                "top10_jaccard": float(np.mean(jaccard_values)),
                "n_folds": len(fold_vectors),
                "n_fold_pairs": len(rank_values),
            })
    iclr_df = pd.DataFrame(rows)
    if iclr_df.empty:
        return iclr_df
    iclr_df["wm_alignment"] = iclr_df["wm_alignment"].astype(float)

    # WM alignment of each ICLR biomarker vs the working-memory prior, computed
    # per fold and averaged within a seed (Spearman, same as the E* metric).
    wm_prior = pd.read_csv("outputs/priors/working_memory/aal116/roi_prior.csv")
    wm_prior = wm_prior.sort_values("roi_index")["prior_score"].to_numpy(np.float64)
    alignment: dict[tuple[str, int], list[float]] = {}
    for method_id in method_ids:
        folder_name, _ = ICLR_METHODS[method_id]
        saliency_dir = ICLR_FAITHFULNESS_DIR / folder_name / "saliency"
        for path in sorted(saliency_dir.glob("seed*_fold*.npz")):
            seed, fold = parse_seed_fold(path)
            with np.load(path, allow_pickle=False) as payload:
                vector = np.asarray(payload["node_saliency"], dtype=float).reshape(-1)
            alignment.setdefault((method_id, seed), []).append(safe_spearman(vector, wm_prior))
    for (method_id, seed), values in alignment.items():
        iclr_df.loc[(iclr_df.experiment_id == method_id) & (iclr_df.seed == seed), "wm_alignment"] = float(np.mean(values))
    return iclr_df


def paired_tests(seed_df: pd.DataFrame) -> pd.DataFrame:
    reference = seed_df[seed_df.experiment_id == REFERENCE_ID].set_index("seed")
    rows: list[dict] = []
    for experiment_id in METHOD_ORDER:
        if experiment_id == REFERENCE_ID:
            continue
        current = seed_df[seed_df.experiment_id == experiment_id].set_index("seed")
        common = sorted(set(reference.index).intersection(current.index))
        for metric in METRICS:
            ref_values = reference.loc[common, metric].to_numpy(float)
            current_values = current.loc[common, metric].to_numpy(float)
            difference = ref_values - current_values
            if np.allclose(difference, 0.0):
                statistic, pvalue = 0.0, 1.0
            else:
                result = wilcoxon(
                    difference,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic, pvalue = float(result.statistic), float(result.pvalue)
            rows.append({
                "reference_id": REFERENCE_ID,
                "reference_name": METHODS[REFERENCE_ID][1],
                "method_id": experiment_id,
                "method_name": METHODS[experiment_id][1],
                "metric": metric,
                "analysis_unit": "seed_level",
                "n_pairs": len(common),
                "reference_mean": float(np.mean(ref_values)),
                "method_mean": float(np.mean(current_values)),
                "reference_advantage_mean": float(np.mean(difference)),
                "wilcoxon_W": statistic,
                "wilcoxon_p": pvalue,
            })

    iclr_ids = sorted(seed_df.experiment_id[seed_df.experiment_id.isin(ICLR_METHODS)].unique())
    for experiment_id in iclr_ids:
        current = seed_df[seed_df.experiment_id == experiment_id].set_index("seed")
        common = sorted(set(reference.index).intersection(current.index))
        for metric in METRICS:
            ref_values = reference.loc[common, metric].to_numpy(float)
            current_values = current.loc[common, metric].to_numpy(float)
            difference = ref_values - current_values
            if np.allclose(difference, 0.0):
                statistic, pvalue = 0.0, 1.0
            else:
                result = wilcoxon(
                    difference,
                    alternative="two-sided",
                    zero_method="wilcox",
                    method="auto",
                )
                statistic, pvalue = float(result.statistic), float(result.pvalue)
            rows.append({
                "reference_id": REFERENCE_ID,
                "reference_name": METHODS[REFERENCE_ID][1],
                "method_id": experiment_id,
                "method_name": ICLR_METHODS[experiment_id][1],
                "metric": metric,
                "analysis_unit": "seed_level",
                "n_pairs": len(common),
                "reference_mean": float(np.mean(ref_values)),
                "method_mean": float(np.mean(current_values)),
                "reference_advantage_mean": float(np.mean(difference)),
                "wilcoxon_W": statistic,
                "wilcoxon_p": pvalue,
            })

    tests = pd.DataFrame(rows)
    tests["wilcoxon_p_holm_metric"] = np.nan
    for metric, indices in tests.groupby("metric").groups.items():
        idx = list(indices)
        tests.loc[idx, "wilcoxon_p_holm_metric"] = holm_adjust(
            tests.loc[idx, "wilcoxon_p"].to_numpy(float)
        )
    tests["significant_holm_005"] = tests.wilcoxon_p_holm_metric < 0.05
    tests["reference_significantly_better"] = (
        tests.significant_holm_005 & (tests.reference_advantage_mean > 0)
    )
    return tests.sort_values(["metric", "wilcoxon_p_holm_metric", "method_id"])


def all_method_names() -> dict:
    names = {k: v[1] for k, v in METHODS.items()}
    names.update({k: v[1] for k, v in ICLR_METHODS.items()})
    return names


def summarize(seed_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        seed_df.groupby(["experiment_id", "method_name"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            wm_alignment_mean=("wm_alignment", "mean"),
            wm_alignment_sd=("wm_alignment", "std"),
            rank_stability_mean=("rank_stability", "mean"),
            rank_stability_sd=("rank_stability", "std"),
            top10_jaccard_mean=("top10_jaccard", "mean"),
            top10_jaccard_sd=("top10_jaccard", "std"),
        )
    )
    ids = list(METHOD_ORDER)
    ids.extend(sorted(seed_df.experiment_id[seed_df.experiment_id.isin(ICLR_METHODS)].unique()))
    order = {experiment_id: index for index, experiment_id in enumerate(ids)}
    summary["_order"] = summary.experiment_id.map(order)
    return summary.sort_values("_order").drop(columns="_order")


def format_pvalue(value: float) -> str:
    if value < 0.001:
        return "$<0.001$"
    return f"${value:.3f}$"


def write_tests_tex(tests: pd.DataFrame, path: Path) -> None:
    paper = tests.copy()
    paper["Metric"] = paper.metric.map({
        "wm_alignment": "WM alignment",
        "rank_stability": "Rank stability",
        "top10_jaccard": "Top-10 Jaccard",
    })
    paper["Control"] = paper.method_name
    paper["$W$"] = paper.wilcoxon_W.map(lambda value: f"{value:.1f}")
    paper["$p_{\\mathrm{Holm}}$"] = paper.wilcoxon_p_holm_metric.map(format_pvalue)
    paper["Result"] = np.where(
        paper.reference_significantly_better,
        "MetaSFC higher",
        "Not significant",
    )
    paper = paper[["Metric", "Control", "$W$", "$p_{\\mathrm{Holm}}$", "Result"]]
    tabular = paper.to_latex(index=False, escape=False, column_format="llccc")
    tex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Paired seed-level Wilcoxon signed-rank comparisons of "
        "MetaSFC with ROI-level controls. Five folds are summarized within each "
        "seed ($n=10$ paired seeds), and two-sided $p$-values are Holm-adjusted "
        "separately within each metric.}\n"
        "\\label{tab:biomarker_wilcoxon}\n"
        + tabular
        + "\\end{table*}\n"
    )
    path.write_text(tex, encoding="utf-8")


def write_main_table_tex(summary: pd.DataFrame, tests: pd.DataFrame, path: Path) -> None:
    significant = {
        (str(row.method_id), str(row.metric))
        for row in tests.itertuples()
        if bool(row.reference_significantly_better)
    }

    def cell(experiment_id: str, metric: str, mean: float, sd: float) -> str:
        value = f"{mean:.3f} \\pm {sd:.3f}"
        if experiment_id == REFERENCE_ID:
            return f"$\\mathbf{{{value}}}$"
        suffix = "^{*}" if (experiment_id, metric) in significant else ""
        return f"${value}{suffix}$"

    rows = []
    for row in summary.itertuples():
        rows.append({
            "Method": row.method_name,
            "WM alignment": cell(
                row.experiment_id, "wm_alignment",
                row.wm_alignment_mean, row.wm_alignment_sd,
            ),
            "Rank": cell(
                row.experiment_id, "rank_stability",
                row.rank_stability_mean, row.rank_stability_sd,
            ),
            "Jaccard": cell(
                row.experiment_id, "top10_jaccard",
                row.top10_jaccard_mean, row.top10_jaccard_sd,
            ),
        })
    display = pd.DataFrame(rows)
    tabular = display.to_latex(index=False, escape=False, column_format="lccc")
    tex = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\caption{Specificity and stability of ROI-level FC--SC coupling "
        "biomarkers. Values are mean $\\pm$ standard deviation over 10 seed-level "
        "summaries. MetaSFC is placed last and bolded. $^{*}$ marks a control "
        "significantly lower than MetaSFC under a paired two-sided Wilcoxon "
        "signed-rank test with metric-wise Holm correction "
        "($p_{\\mathrm{Holm}}<0.05$).}\n"
        "\\label{tab:alignment}\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        + tabular
        + "}\n"
        "\\end{table}\n"
    )
    path.write_text(tex, encoding="utf-8")



def write_stability_figure(seed_df: pd.DataFrame, output_pdf: Path, output_png: Path) -> None:
    labels = [METHODS[experiment_id][1] for experiment_id in METHOD_ORDER]
    iclr_ids = sorted(seed_df.experiment_id[seed_df.experiment_id.isin(ICLR_METHODS)].unique())
    ids = list(METHOD_ORDER) + iclr_ids
    short_labels = ["No prior", "Shuffled", "Random", "Visual", "MetaSFC"] + iclr_ids
    x = np.arange(len(ids))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
    for ax, metric, title, ylabel in [
        (axes[0], "rank_stability", "Saliency-rank stability", "Mean pairwise Spearman"),
        (axes[1], "top10_jaccard", "Top-10 overlap stability", "Mean pairwise Jaccard"),
    ]:
        means = []
        errors = []
        for experiment_id in ids:
            values = seed_df.loc[seed_df.experiment_id == experiment_id, metric].to_numpy(float)
            means.append(float(values.mean()))
            sem = float(values.std(ddof=1) / np.sqrt(len(values)))
            errors.append(float(student_t.ppf(0.975, len(values) - 1) * sem))
        ax.bar(x, means, yerr=errors, capsize=3, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=22, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight")
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, default=Path("outputs/aaai/final"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/aaai/biomarker_significance"))
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument(
        "--configs", nargs="+", type=Path, default=[],
        help="Optional ICLR method configs (network_constrained_ridge / meta_gat / "
             "two_stage_kernel_ridge) whose biomarkers are added to the tables.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_df = load_seed_level_metrics(args.results_root, args.topk)
    if args.configs:
        iclr_df = load_iclr_seed_level_metrics(args.configs, args.topk)
        seed_df = pd.concat([seed_df, iclr_df], ignore_index=True)
    summary = summarize(seed_df)
    tests = paired_tests(seed_df)

    seed_df.to_csv(args.output_dir / "biomarker_seed_level_metrics.csv", index=False)
    summary.to_csv(args.output_dir / "biomarker_table2_summary.csv", index=False)
    tests.to_csv(args.output_dir / "biomarker_wilcoxon_vs_metasfc.csv", index=False)
    write_main_table_tex(summary, tests, args.output_dir / "paper_table2_biomarker_significance.tex")
    write_tests_tex(tests, args.output_dir / "biomarker_wilcoxon_vs_metasfc.tex")
    write_stability_figure(
        seed_df,
        args.output_dir / "fig_biomarker_stability_seed_level.pdf",
        args.output_dir / "fig_biomarker_stability_seed_level.png",
    )

    print("\nTable 2 seed-level summary")
    print(summary.to_string(index=False))
    print("\nPaired Wilcoxon tests versus MetaSFC")
    print(tests.to_string(index=False))
    print(f"\nSaved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
