#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Prepare atlas metadata and synthetic test data")
    parser.add_argument("--atlas_name", type=str, default="aal116",
                        help="Atlas name (aal116, schaefer200, power264)")
    parser.add_argument("--n_rois", type=int, default=116,
                        help="Number of ROIs")
    parser.add_argument("--out_dir", type=str, default="inputs/atlases",
                        help="Output directory for atlas metadata")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic atlas NIfTI and labels")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_df = pd.DataFrame({
        "roi_index": list(range(1, args.n_rois + 1)),
        "roi_label": [f"ROI_{i:03d}" for i in range(1, args.n_rois + 1)],
    })
    labels_path = out_dir / f"{args.atlas_name}_labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"Saved labels to {labels_path}")

    if args.synthetic:
        roi_to_module = pd.DataFrame({
            "roi_index": list(range(1, args.n_rois + 1)),
            "module": np.random.randint(0, 7, size=args.n_rois),
        })
        mod_path = out_dir / f"{args.atlas_name}_yeo7_mapping.csv"
        roi_to_module.to_csv(mod_path, index=False)
        print(f"Saved module mapping to {mod_path}")

    print(f"Atlas metadata prepared for {args.atlas_name} ({args.n_rois} ROIs)")


if __name__ == "__main__":
    main()
