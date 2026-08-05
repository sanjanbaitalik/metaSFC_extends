#!/usr/bin/env python3
"""ICLR experiment 2: biomarker alignment (vs WM prior) and stability.

For every split of the refit faithfulness run we export, per method:
  * Meta-GAT (M2_TRUE / M2_RANDOM): mean-incident attention mass,
    outputs/aaai/faithfulness_iclr/per_method/<id>/attention/seed__fold__.npz
  * Two-Stage KRR (M3_TRUE / M3_RANDOM): gradient node saliency,
    outputs/aaai/faithfulness_iclr/per_method/<id>/krr_saliency/seed__fold__.npz

Analysis units are seed-level (five folds averaged within a seed, n=10):
  * wm_alignment  : Spearman(biomarker, WM roi prior) per fold, mean over folds
  * rank_stability: mean pairwise Spearman over the 10 fold pairs within a seed
  * top10_jaccard : mean pairwise top-10 Jaccard over the 10 fold pairs

MetaSFC (E1) is included as the reference anchor. Paired two-sided Wilcoxon
tests (M2_TRUE vs M2_RANDOM; M3_TRUE vs M3_RANDOM; and each against E1) are
Holm-adjusted within each metric family.
"""
from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

WM_PRIOR_PATH = Path("outputs/priors/working_memory/aal116/roi_prior.csv")
FAITHFULNESS_DIR = Path("outputs/aaai/faithfulness_iclr/per_method")
E1_FOLDER = Path("outputs/aaai/final/E1_node_true")

METHODS = {
    "M2_TRUE": ("meta_gat", "Meta-GAT attention mass (true prior)"),
    "M2_RANDOM": ("meta_gat", "Meta-GAT attention mass (random prior)"),
    "M3_TRUE": ("kernel_ridge", "Two-stage KRR gradient saliency (true prior)"),
    "M3_RANDOM": ("kernel_ridge", "Two-stage KRR gradient saliency (random prior)"),
    "E1": ("e1", "MetaSFC (ours, AAAI)"),
}
METHOD_ORDER = ["M2_TRUE", "M2_RANDOM", "M3_TRUE", "M3_RANDOM", "E1"]
N_SEEDS, N_FOLDS = 10, 5
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
        raise ValueError(f"Unexpected filename: {path.name}")
    return int(match.group(1)), int(match.group(2))


def load_biomarkers(method_id: str, topk: int) -> dict[tuple[int, int], np.ndarray]:
    kind = METHODS[method_id][0]
    if kind == "meta_gat":
        directory = FAITHFULNESS_DIR / method_id / "attention"
        key = "node_attention_mass"
    elif kind == "kernel_ridge":
        directory = FAITHFULNESS_DIR / method_id / "krr_saliency"
        key = "node_saliency"
    else:
        directory = E1_FOLDER / "saliency"
        key = "node_saliency"
    vectors: dict[tuple[int, int], np.ndarray] = {}
    for path in sorted(directory.glob("seed*_fold*.npz")):
        seed, fold = parse_seed_fold(path)
        with np.load(path, allow_pickle=False) as payload:
            if key not in payload.files:
                raise KeyError(f"{key} missing from {path}")
            vectors[(seed, fold)] = np.asarray(payload[key], dtype=float).reshape(-1)
    return vectors


def load_wm_prior() -> np.ndarray:
    prior = pd.read_csv(WM_PRIOR_PATH).sort_values("roi_index")
    if len(prior) != 116 or prior["prior_score"].isna().any():
        raise ValueError(f"WM prior must span all 116 AAL ROIs; got {len(prior)}")
    return prior["prior_score"].to_numpy(np.float64)


def load_seed_level_metrics(topk: int) -> pd.DataFrame:
    prior = load_wm_prior()
    rows: list[dict] = []
    for method_id in METHOD_ORDER:
        vectors = load_biomarkers(method_id, topk)
        alignment: dict[int, list[float]] = {}
        folds_by_seed: dict[int, list[tuple[int, np.ndarray]]] = {}
        for (seed, fold), vector in vectors.items():
            alignment.setdefault(seed, []).append(safe_spearman(vector, prior))
            folds_by_seed.setdefault(seed, []).append((fold, vector))
        for seed in range(N_SEEDS):
            if seed not in alignment:
                raise ValueError(f"{method_id} missing seed {seed} biomarkers")
            fold_values = sorted(folds_by_seed[seed], key=lambda item: item[0])
            if len(fold_values) != N_FOLDS:
                raise ValueError(
                    f"{method_id} seed {seed} has {len(fold_values)} folds; expected {N_FOLDS}"
                )
            sub = [v for _, v in fold_values]
            rank = []
            jacc = []
            for left, right in combinations(sub, 2):
                rank.append(safe_spearman(left, right))
                jacc.append(topk_jaccard(left, right, topk))
            rows.append({
                "method_id": method_id,
                "method_name": METHODS[method_id][1],
                "seed": int(seed),
                "wm_alignment": float(np.mean(alignment[seed])),
                "rank_stability": float(np.mean(rank)),
                "top10_jaccard": float(np.mean(jacc)),
            })
    seed_df = pd.DataFrame(rows)
    for method_id in METHOD_ORDER:
        if seed_df.loc[seed_df.method_id == method_id].seed.nunique() != N_SEEDS:
            raise ValueError(f"{method_id} does not span all {N_SEEDS} seeds")
    return seed_df


def pair_means(seed_df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return seed_df[seed_df.method_id == a].merge(
        seed_df[seed_df.method_id == b].rename(columns={c: f"{c}_{b}" for c in METRICS}),
        on="seed",
    )


def paired_tests(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for a, b in [("M2_TRUE", "M2_RANDOM"), ("M3_TRUE", "M3_RANDOM")]:
        merged = pair_means(seed_df, a, b)
        for metric in METRICS:
            x = merged[metric].to_numpy(float)
            y = merged[f"{metric}_{b}"].to_numpy(float)
            diff = x - y
            w_stat, w_p = 0.0, 1.0
            if not np.allclose(diff, 0.0):
                result = wilcoxon(
                    diff, alternative="two-sided", zero_method="wilcox", method="auto"
                )
                w_stat, w_p = float(result.statistic), float(result.pvalue)
            rows.append({
                "method_id": a, "vs_method_id": b,
                "metric": metric, "n_pairs": len(merged),
                f"{a}_mean": float(x.mean()), f"{a}_std": float(x.std(ddof=1)),
                f"{b}_mean": float(y.mean()), f"{b}_std": float(y.std(ddof=1)),
                "advantage_mean": float(np.mean(diff)),
                "wilcoxon_W": w_stat, "wilcoxon_p": w_p,
            })

    for a in ("M2_TRUE", "M3_TRUE"):
        b = "E1"
        merged = pair_means(seed_df, a, b)
        for metric in METRICS:
            x = merged[metric].to_numpy(float)
            y = merged[f"{metric}_{b}"].to_numpy(float)
            diff = x - y
            w_stat, w_p = 0.0, 1.0
            if not np.allclose(diff, 0.0):
                result = wilcoxon(
                    diff, alternative="two-sided", zero_method="wilcox", method="auto"
                )
                w_stat, w_p = float(result.statistic), float(result.pvalue)
            rows.append({
                "method_id": a, "vs_method_id": b,
                "metric": metric, "n_pairs": len(merged),
                f"{a}_mean": float(x.mean()), f"{a}_std": float(x.std(ddof=1)),
                f"{b}_mean": float(y.mean()), f"{b}_std": float(y.std(ddof=1)),
                "advantage_mean": float(np.mean(diff)),
                "wilcoxon_W": w_stat, "wilcoxon_p": w_p,
            })

    tests = pd.DataFrame(rows)
    tests["wilcoxon_p_holm_metric"] = np.nan
    for metric, idx in tests.groupby("metric").groups.items():
        idx = list(idx)
        tests.loc[idx, "wilcoxon_p_holm_metric"] = holm_adjust(
            tests.loc[idx, "wilcoxon_p"].to_numpy(float)
        )
    tests["significant_holm_005"] = tests.wilcoxon_p_holm_metric < 0.05
    return tests.reset_index(drop=True)


def summarize(seed_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        seed_df.groupby(["method_id", "method_name"], as_index=False)
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
    order = {method_id: index for index, method_id in enumerate(METHOD_ORDER)}
    summary["_order"] = summary.method_id.map(order)
    return summary.sort_values("_order").drop(columns="_order")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("outputs/aaai/faithfulness_iclr"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seed_df = load_seed_level_metrics(args.topk)
    tests = paired_tests(seed_df)
    summary = summarize(seed_df)

    seed_df.to_csv(args.out / "biomarker_seed_level_metrics.csv", index=False)
    tests.to_csv(args.out / "biomarker_wilcoxon_paired_tests.csv", index=False)
    summary.to_csv(args.out / "biomarker_summary.csv", index=False)
    (args.out / "biomarker_wilcoxon_paired_tests.tex").write_text(
        tests.to_latex(index=False, escape=False, float_format="%.3f"),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("\nPaired Wilcoxon (Holm per metric)")
    print(tests.to_string(index=False))
    print(f"\nSaved experiment-2 outputs to {args.out}")


if __name__ == "__main__":
    main()