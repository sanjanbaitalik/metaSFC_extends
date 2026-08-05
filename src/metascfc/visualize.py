from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def set_style() -> None:
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10


def plot_roi_prior(
    prior_df: pd.DataFrame,
    title: str = "ROI Prior Scores",
    top_k: Optional[int] = None,
    save_path: Optional[str | Path] = None,
) -> None:
    df = prior_df.sort_values("prior_score", ascending=False)
    if top_k is not None:
        df = df.head(top_k)

    plt.figure(figsize=(12, max(4, len(df) * 0.3)))
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(df)))
    plt.barh(range(len(df)), df["prior_score"].values, color=colors[::-1])
    plt.yticks(range(len(df)), df["roi_label"].values)
    plt.xlabel("Prior Score")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_module_prior(
    module_prior_df: pd.DataFrame,
    title: str = "Module Prior Scores",
    save_path: Optional[str | Path] = None,
) -> None:
    df = module_prior_df.sort_values("prior_score", ascending=False)
    plt.figure(figsize=(10, 5))
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(df)))
    plt.bar(df["module"].astype(str), df["prior_score"].values, color=colors)
    plt.xlabel("Module")
    plt.ylabel("Prior Score")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_edge_prior_heatmap(
    edge_prior: np.ndarray,
    title: str = "Edge Prior Matrix",
    save_path: Optional[str | Path] = None,
) -> None:
    plt.figure(figsize=(8, 7))
    sns.heatmap(edge_prior, cmap="Reds", square=True, cbar_kws={"shrink": 0.8})
    plt.title(title)
    plt.xlabel("ROI")
    plt.ylabel("ROI")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_mean_coupling_vector(
    coupling_vectors: np.ndarray,
    roi_labels: Optional[List[str]] = None,
    title: str = "Mean Coupling Vector",
    save_path: Optional[str | Path] = None,
) -> None:
    mean_cv = coupling_vectors.mean(axis=0)
    std_cv = coupling_vectors.std(axis=0)
    plt.figure(figsize=(14, 4))
    plt.errorbar(range(len(mean_cv)), mean_cv, yerr=std_cv, fmt="o-", capsize=3, markersize=3)
    plt.xlabel("ROI Index")
    plt.ylabel("Coupling Weight")
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_learned_vs_prior_scatter(
    learned: np.ndarray,
    prior: np.ndarray,
    title: str = "Learned Saliency vs Prior",
    save_path: Optional[str | Path] = None,
) -> None:
    plt.figure(figsize=(6, 5))
    plt.scatter(prior, learned, alpha=0.6, s=20)
    plt.xlabel("Prior Score")
    plt.ylabel("Learned Saliency")
    plt.title(title)
    z = np.polyfit(prior, learned, 1)
    p = np.poly1d(z)
    x_line = np.linspace(prior.min(), prior.max(), 100)
    plt.plot(x_line, p(x_line), "r--", alpha=0.8)
    corr = np.corrcoef(prior, learned)[0, 1]
    plt.text(0.05, 0.95, f"r = {corr:.3f}", transform=plt.gca().transAxes, va="top")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_fold_stability_heatmap(
    fold_metrics: Dict[str, List[float]],
    title: str = "Fold Stability",
    save_path: Optional[str | Path] = None,
) -> None:
    df = pd.DataFrame(fold_metrics)
    plt.figure(figsize=(10, max(3, df.shape[1] * 0.4)))
    sns.heatmap(df.T, annot=True, cmap="viridis", fmt=".3f", cbar_kws={"shrink": 0.6})
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_prediction_summary(
    results_df: pd.DataFrame,
    metric: str = "pearson",
    title: str = "Prediction Summary",
    save_path: Optional[str | Path] = None,
) -> None:
    plt.figure(figsize=(10, 5))
    conditions = results_df["condition"].unique() if "condition" in results_df.columns else results_df.index
    values = results_df[metric].values if metric in results_df.columns else results_df.values
    colors = plt.cm.Set2(np.linspace(0, 1, len(conditions)))
    plt.bar(range(len(conditions)), values, color=colors)
    plt.xticks(range(len(conditions)), conditions, rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
