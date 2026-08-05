from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import pandas as pd

from . import atlases


def load_meta_map(path: str | Path):
    return atlases.load_meta_map(path)


def resample_meta_to_atlas(meta_img, atlas_img):
    return atlases.resample_meta_to_atlas(meta_img, atlas_img)


def build_roi_prior(
    meta_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    labels_df: pd.DataFrame,
    mode: str = "mean_positive",
) -> pd.DataFrame:
    meta_resampled = resample_meta_to_atlas(meta_img, atlas_img)
    meta_data = meta_resampled.get_fdata()
    _, atlas_data, roi_indices = atlases.load_atlas(atlas_img)

    if "roi_index" not in labels_df.columns:
        raise ValueError("labels_df must have a 'roi_index' column")
    if "roi_label" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["roi_label"] = labels_df["roi_index"].astype(str)

    results = []
    for roi_idx in roi_indices:
        vals = atlases.extract_roi_values(meta_data, atlas_data, int(roi_idx))
        if mode == "mean":
            raw_score = float(np.mean(vals))
        elif mode == "mean_positive":
            pos = vals[vals > 0]
            raw_score = float(np.mean(pos)) if len(pos) > 0 else 0.0
        elif mode == "max":
            raw_score = float(np.max(vals))
        elif mode == "proportion_positive":
            raw_score = float(np.mean(vals > 0)) if len(vals) > 0 else 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

        row = labels_df[labels_df["roi_index"] == int(roi_idx)]
        if len(row) == 0:
            label = f"ROI_{int(roi_idx)}"
        else:
            label = row.iloc[0]["roi_label"]
        results.append({"roi_index": int(roi_idx), "roi_label": label, "raw_score": raw_score})

    prior_df = pd.DataFrame(results)
    prior_df["prior_score"] = normalize_prior(prior_df["raw_score"].values, method="minmax")
    return prior_df


def normalize_prior(
    scores: np.ndarray,
    method: str = "minmax",
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if method == "minmax":
        s_min, s_max = scores.min(), scores.max()
        if s_max - s_min < 1e-12:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)
    elif method == "zscore_positive":
        pos = scores[scores > 0]
        if len(pos) == 0:
            return np.zeros_like(scores)
        mu, std = pos.mean(), pos.std()
        if std < 1e-12:
            return np.where(scores > 0, 1.0, 0.0)
        z = (scores - mu) / std
        return np.maximum(z, 0.0)
    elif method == "rank":
        ranks = np.argsort(np.argsort(scores))
        return ranks.astype(float) / (len(ranks) - 1) if len(ranks) > 1 else np.ones_like(scores)
    elif method == "softmax":
        exp_s = np.exp(scores - scores.max())
        return exp_s / exp_s.sum()
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def threshold_prior(
    prior_df: pd.DataFrame,
    top_k: Optional[int] = None,
    percentile: Optional[float] = None,
) -> pd.DataFrame:
    df = prior_df.copy()
    scores = df["prior_score"].values
    mask = np.ones(len(scores), dtype=bool)

    if top_k is not None:
        threshold = np.sort(scores)[::-1][min(top_k, len(scores)) - 1] if top_k <= len(scores) else scores.min()
        mask = scores >= threshold

    if percentile is not None:
        thresh = np.percentile(scores, 100 - percentile)
        mask = mask & (scores >= thresh)

    df.loc[~mask, "prior_score"] = 0.0
    return df


def build_module_prior(
    roi_prior_df: pd.DataFrame,
    roi_to_module_df: pd.DataFrame,
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate ROI priors into module-level priors.

    roi_to_module_df should contain roi_index and module. If it also contains
    module_id, that ordering is preserved for downstream training.
    """
    required = {"roi_index", "module"}
    missing = required - set(roi_to_module_df.columns)
    if missing:
        raise ValueError(f"roi_to_module_df missing required columns: {sorted(missing)}")

    merged = roi_prior_df.merge(roi_to_module_df, on="roi_index", how="inner")
    if len(merged) != len(roi_prior_df):
        raise ValueError(
            f"ROI-to-module mapping covers {len(merged)} ROIs, but roi_prior has {len(roi_prior_df)} ROIs."
        )

    group_cols = ["module"]
    if "module_id" in merged.columns:
        group_cols = ["module_id", "module"]

    if agg == "mean":
        grouped = merged.groupby(group_cols, as_index=False)["prior_score"].mean()
    elif agg == "sum":
        grouped = merged.groupby(group_cols, as_index=False)["prior_score"].sum()
    elif agg == "max":
        grouped = merged.groupby(group_cols, as_index=False)["prior_score"].max()
    else:
        raise ValueError(f"Unknown aggregation: {agg}")

    grouped = grouped.rename(columns={"prior_score": "raw_score"})
    grouped["prior_score"] = normalize_prior(grouped["raw_score"].values, method="minmax")

    if "module_id" in grouped.columns:
        grouped = grouped.sort_values("module_id").reset_index(drop=True)
    else:
        grouped = grouped.sort_values("module").reset_index(drop=True)
    return grouped


def build_edge_prior(
    roi_prior_df: pd.DataFrame,
    method: str = "outer_product",
) -> np.ndarray:
    scores = roi_prior_df["prior_score"].values.astype(float)
    n = len(scores)

    if method == "outer_product":
        prior = np.outer(scores, scores)
    elif method == "average":
        prior = (scores[:, None] + scores[None, :]) / 2.0
    elif method == "min":
        prior = np.minimum(scores[:, None], scores[None, :])
    elif method == "max":
        prior = np.maximum(scores[:, None], scores[None, :])
    else:
        raise ValueError(f"Unknown edge prior method: {method}")

    p_min, p_max = prior.min(), prior.max()
    if p_max - p_min > 1e-12:
        prior = (prior - p_min) / (p_max - p_min)
    else:
        prior = np.zeros_like(prior)
    return prior


def save_prior_outputs(
    roi_prior_df: pd.DataFrame,
    module_prior_df: Optional[pd.DataFrame],
    edge_prior: np.ndarray,
    metadata: Dict,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    roi_prior_df.to_csv(output_dir / "roi_prior.csv", index=False)
    if module_prior_df is not None:
        module_prior_df.to_csv(output_dir / "module_prior.csv", index=False)
    np.save(str(output_dir / "edge_prior.npy"), edge_prior)

    import json
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)
