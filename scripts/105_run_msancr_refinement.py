#!/usr/bin/env python3
"""Run the corrected 3-seed Working-Memory MS-A-NCR refinement.

This script refuses to configure a 10-seed run. Use ``--seeds 0 --folds 0``
for the required preflight outer-split smoke; rerun without overrides to
resume and complete the 3 x 5 refinement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from threadpoolctl import threadpool_limits

from metascfc.experiments.msancr_refinement import run_refinement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/iclr/msancr_refinement.yaml")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--folds", nargs="*", type=int)
    parser.add_argument("--output-dir")
    parser.add_argument("--figures-dir")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    with threadpool_limits(limits=int(config.get("n_threads", 4))):
        decision = run_refinement(
            config,
            seeds_override=args.seeds if args.seeds else None,
            folds_override=args.folds if args.folds else None,
            output_dir_override=args.output_dir,
            figure_dir_override=args.figures_dir,
            overwrite=args.overwrite,
        )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()