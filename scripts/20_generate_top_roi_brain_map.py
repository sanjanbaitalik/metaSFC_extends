#!/usr/bin/env python3
"""Generate a compact AAL116 glass-brain visualization for the MetaSFC paper.

The figure contains:
(a) external working-memory meta-analysis prior,
(b) mean E1 learned node saliency,
(c) spatial overlap between the prior and E1 saliency,
(d) mean E10 visual-prior learned saliency.

Example
-------
python scripts/20_generate_top_roi_brain_map.py \
  --atlas inputs/atlases/AAL116.nii.gz \
  --labels inputs/atlases/AAL116_labels.csv \
  --working-prior outputs/priors/working_memory/aal116/roi_prior.csv \
  --e1-saliency outputs/aaai/final/E1_node_true/all_node_saliency.npy \
  --e10-saliency outputs/aaai/final/E10_node_unrelated_visual/all_node_saliency.npy \
  --output figures/top_roi_brain_map.pdf \
  --topk 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import plotting


def minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def load_saliency(path: Path, n_rois: int) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.shape[-1] != n_rois:
        raise ValueError(
            f"{path} has shape {arr.shape}; expected last dimension {n_rois}."
        )
    arr = np.asarray(arr, dtype=np.float64).reshape(-1, n_rois)
    # Saliency magnitude is averaged across all seed-fold evaluations.
    return minmax(np.mean(np.abs(arr), axis=0))


def topk_mask(scores: np.ndarray, k: int) -> np.ndarray:
    if not 1 <= k <= len(scores):
        raise ValueError(f"topk must be in [1, {len(scores)}], got {k}.")
    keep = np.argpartition(scores, -k)[-k:]
    masked = np.zeros_like(scores)
    masked[keep] = scores[keep]
    return masked


def roi_scores_to_img(
    atlas_img: nib.Nifti1Image,
    roi_indices: np.ndarray,
    scores: np.ndarray,
) -> nib.Nifti1Image:
    atlas = np.asarray(atlas_img.get_fdata(), dtype=np.int32)
    out = np.zeros(atlas.shape, dtype=np.float32)
    for roi_id, score in zip(roi_indices, scores):
        out[atlas == int(roi_id)] = float(score)
    return nib.Nifti1Image(out, atlas_img.affine, atlas_img.header)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--working-prior", type=Path, required=True)
    parser.add_argument("--e1-saliency", type=Path, required=True)
    parser.add_argument("--e10-saliency", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/top_roi_brain_map.pdf"),
    )
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    for path in [
        args.atlas,
        args.labels,
        args.working_prior,
        args.e1_saliency,
        args.e10_saliency,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)

    labels = pd.read_csv(args.labels).sort_values("roi_index")
    required_label_cols = {"roi_index", "roi_label"}
    if not required_label_cols.issubset(labels.columns):
        raise ValueError(
            f"{args.labels} must contain {sorted(required_label_cols)}."
        )

    n_rois = len(labels)
    roi_indices = labels["roi_index"].to_numpy(dtype=int)

    prior_df = pd.read_csv(args.working_prior)
    if not {"roi_index", "prior_score"}.issubset(prior_df.columns):
        raise ValueError(
            f"{args.working_prior} must contain roi_index and prior_score."
        )
    prior_df = labels[["roi_index"]].merge(
        prior_df[["roi_index", "prior_score"]],
        on="roi_index",
        how="left",
        validate="one_to_one",
    )
    if prior_df["prior_score"].isna().any():
        raise ValueError("The working-memory prior is missing AAL116 ROIs.")

    working_prior = minmax(prior_df["prior_score"].to_numpy(dtype=float))
    e1 = load_saliency(args.e1_saliency, n_rois)
    e10 = load_saliency(args.e10_saliency, n_rois)

    # Continuous conjunction: high only where both external prior and E1 saliency are high.
    overlap = minmax(working_prior * e1)

    maps = [
        topk_mask(working_prior, args.topk),
        topk_mask(e1, args.topk),
        topk_mask(overlap, args.topk),
        topk_mask(e10, args.topk),
    ]
    titles = [
        "(a) Working-memory prior",
        "(b) MetaSFC learned saliency",
        "(c) Prior–saliency overlap",
        "(d) Visual-prior saliency",
    ]

    atlas_img = nib.load(str(args.atlas))
    imgs = [
        roi_scores_to_img(atlas_img, roi_indices, scores)
        for scores in maps
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.8))
    cmap = "inferno"

    for ax, img, title in zip(axes.ravel(), imgs, titles):
        plotting.plot_glass_brain(
            img,
            axes=ax,
            figure=fig,
            display_mode="ortho",
            plot_abs=False,
            threshold=1e-6,
            colorbar=False,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            black_bg=False,
            annotate=False,
        )
        ax.set_title(title, fontsize=10, pad=4)

    sm = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
    )
    cbar.set_label("Normalized ROI score", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.subplots_adjust(
        left=0.01,
        right=0.92,
        top=0.96,
        bottom=0.02,
        wspace=0.02,
        hspace=0.12,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")

    # Also save a high-resolution PNG beside the PDF.
    png_path = args.output.with_suffix(".png")
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Export the exact top-k ROI lists for reporting and audit.
    report = labels[["roi_index", "roi_label"]].copy()
    report["working_memory_prior"] = working_prior
    report["e1_saliency"] = e1
    report["prior_saliency_overlap"] = overlap
    report["e10_visual_saliency"] = e10
    report.to_csv(
        args.output.with_name("top_roi_brain_map_scores.csv"),
        index=False,
    )

    print(f"Saved: {args.output}")
    print(f"Saved: {png_path}")
    print(
        f"Saved: {args.output.with_name('top_roi_brain_map_scores.csv')}"
    )


if __name__ == "__main__":
    main()
