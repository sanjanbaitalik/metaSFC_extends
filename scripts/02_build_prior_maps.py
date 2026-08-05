#!/usr/bin/env python3
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from metascfc.config import load_config, get_output_dir
from metascfc.seed import set_seed
from metascfc.priors import (
    load_meta_map,
    build_roi_prior,
    threshold_prior,
    build_module_prior,
    build_edge_prior,
    save_prior_outputs,
)
from metascfc import visualize as vis
from metascfc.atlases import load_atlas, load_labels, load_roi_to_module


def main():
    parser = argparse.ArgumentParser(description="Build meta-analysis prior maps for FC-SC coupling")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    term = cfg["term"]
    atlas_name = cfg["atlas"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    meta_map_path = Path(cfg["meta_map_path"])
    atlas_path = Path(cfg["atlas_path"])
    labels_path = Path(cfg["labels_path"])
    roi_to_module_path = cfg.get("roi_to_module_path")

    print(f"Building priors for term: {term}, atlas: {atlas_name}")

    meta_img = load_meta_map(str(meta_map_path))
    atlas_img, atlas_data, roi_indices = load_atlas(str(atlas_path))
    labels_df = load_labels(str(labels_path))

    roi_prior = build_roi_prior(
        meta_img, atlas_img, labels_df,
        mode=cfg.get("roi_mode", "mean_positive"),
    )

    if cfg.get("threshold"):
        roi_prior = threshold_prior(
            roi_prior,
            top_k=cfg["threshold"].get("top_k"),
            percentile=cfg["threshold"].get("percentile"),
        )

    roi_prior["prior_score"] = roi_prior["prior_score"].values

    module_prior = None
    if roi_to_module_path:
        roi_to_module = load_roi_to_module(str(roi_to_module_path))
        module_prior = build_module_prior(
            roi_prior, roi_to_module,
            agg=cfg.get("module_agg", "mean"),
        )

    edge_prior = build_edge_prior(
        roi_prior,
        method=cfg.get("edge_prior_method", "outer_product"),
    )

    metadata = {
        "term": term,
        "atlas": atlas_name,
        "source_map_path": str(meta_map_path),
        "normalization_method": cfg.get("normalization", "minmax"),
        "thresholding": cfg.get("threshold"),
        "date_generated": datetime.now().isoformat(),
        "code_version": "metascfc-0.1.0",
    }

    save_prior_outputs(roi_prior, module_prior, edge_prior, metadata, output_dir)
    print(f"Priors saved to {output_dir}")

    if cfg.get("generate_figures", True):
        vis.set_style()
        vis.plot_roi_prior(
            roi_prior, title=f"ROI Prior: {term}",
            top_k=30, save_path=output_dir / "figures" / "roi_prior_barplot.png",
        )
        if module_prior is not None:
            vis.plot_module_prior(
                module_prior, title=f"Module Prior: {term}",
                save_path=output_dir / "figures" / "module_prior_barplot.png",
            )
        vis.plot_edge_prior_heatmap(
            edge_prior, title=f"Edge Prior: {term}",
            save_path=output_dir / "figures" / "edge_prior_heatmap.png",
        )
        print(f"Figures saved to {output_dir / 'figures'}")


if __name__ == "__main__":
    main()
