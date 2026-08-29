#!/usr/bin/env python3
"""110 -- Corrected Fluid Intelligence 3-seed verification.

Uses the generalized run_refinement with fluid_intelligence target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from metascfc.experiments.msancr_refinement import run_refinement


def main() -> None:
    ap = argparse.ArgumentParser(description="Fluid Intelligence 3-seed verification")
    ap.add_argument("--config", default="configs/iclr/msancr_fluid_verification.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    try:
        result = run_refinement(
            cfg,
            output_dir_override=cfg["output_dir"],
            figure_dir_override=cfg["figures_dir"],
            overwrite=True,
            enforce_seed_gate=True,
            target_key="fluid_intelligence",
            inference_status="descriptive_only_n_equals_3_no_significance_claim",
        )
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
