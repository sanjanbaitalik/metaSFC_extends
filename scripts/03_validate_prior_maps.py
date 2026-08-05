#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_prior_dir(prior_dir: Path) -> dict:
    results = {"prior_dir": str(prior_dir), "checks": [], "errors": [], "warnings": []}

    roi_csv = prior_dir / "roi_prior.csv"
    if not roi_csv.exists():
        results["errors"].append("Missing roi_prior.csv")
        return results
    results["checks"].append("roi_prior.csv exists")

    roi_df = pd.read_csv(roi_csv)
    n_rois = len(roi_df)

    if "roi_index" not in roi_df.columns:
        results["errors"].append("roi_prior.csv missing 'roi_index' column")
    if "prior_score" not in roi_df.columns:
        results["errors"].append("roi_prior.csv missing 'prior_score' column")

    scores = roi_df["prior_score"].values
    if np.any(np.isnan(scores)):
        results["errors"].append("Prior scores contain NaN")
    if np.any(np.isinf(scores)):
        results["errors"].append("Prior scores contain Inf")

    within_01 = (scores >= 0).all() and (scores <= 1).all()
    results["norm_check"] = "minmax [0,1]" if within_01 else "not minmax-normalized"

    top5 = roi_df.nlargest(5, "prior_score")
    results["top5_rois"] = top5[["roi_index", "roi_label", "prior_score"]].to_dict("records")

    edge_npy = prior_dir / "edge_prior.npy"
    if edge_npy.exists():
        results["checks"].append("edge_prior.npy exists")
        edge = np.load(str(edge_npy))
        if edge.shape != (n_rois, n_rois):
            results["errors"].append(f"Edge prior shape {edge.shape} != ({n_rois}, {n_rois})")
        else:
            results["checks"].append(f"Edge prior shape correct ({n_rois}x{n_rois})")
        if np.any(np.isnan(edge)):
            results["errors"].append("Edge prior contains NaN")
    else:
        results["warnings"].append("Missing edge_prior.npy")

    module_csv = prior_dir / "module_prior.csv"
    if module_csv.exists():
        module_df = pd.read_csv(module_csv)
        results["checks"].append(f"module_prior.csv exists with {len(module_df)} modules")
        results["module_scores"] = module_df["prior_score"].describe().to_dict()
    else:
        results["warnings"].append("Missing module_prior.csv (optional)")

    meta_json = prior_dir / "metadata.json"
    if meta_json.exists():
        with open(meta_json) as f:
            meta = json.load(f)
        results["metadata"] = meta
    else:
        results["warnings"].append("Missing metadata.json")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate generated prior maps")
    parser.add_argument("--prior_dir", type=str,
                        default="outputs/priors/fluid_intelligence/aal116",
                        help="Path to prior output directory")
    parser.add_argument("--out", type=str, default=None,
                        help="Path to save validation report JSON")
    args = parser.parse_args()

    prior_dir = Path(args.prior_dir)
    if not prior_dir.exists():
        print(f"ERROR: prior directory not found: {prior_dir}")
        return

    results = validate_prior_dir(prior_dir)
    n_errors = len(results["errors"])
    n_warnings = len(results["warnings"])

    print(f"\nValidation Report for: {prior_dir}")
    print(f"  Checks passed: {len(results['checks'])}")
    print(f"  Errors: {n_errors}")
    print(f"  Warnings: {n_warnings}")
    print(f"  Normalization: {results.get('norm_check', 'N/A')}")

    if results.get("top5_rois"):
        print("\n  Top 5 ROIs:")
        for r in results["top5_rois"]:
            print(f"    ROI {r['roi_index']} ({r['roi_label']}): {r['prior_score']:.4f}")

    if n_errors > 0:
        print("\n  ERRORS:")
        for e in results["errors"]:
            print(f"    - {e}")

    if n_warnings > 0:
        print("\n  WARNINGS:")
        for w in results["warnings"]:
            print(f"    - {w}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
