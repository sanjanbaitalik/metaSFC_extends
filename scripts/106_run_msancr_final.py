#!/usr/bin/env python3
"""Final 10-seed x 5-fold MS-A-NCR runner (PREPARED, NOT EXECUTED).

The grid-closure step explicitly forbids launching this script.  When
review approves the final run, execute:

    python scripts/106_run_msancr_final.py --config configs/iclr/msancr_final_10x5.yaml

It reuses the frozen v2 refinement path unchanged and supports resume,
config-hash protection, atomic checkpoints, and seed/fold granularity.
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
    parser.add_argument("--config", default="configs/iclr/msancr_final_10x5.yaml")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--folds", nargs="*", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    with threadpool_limits(limits=int(config.get("n_threads", 4))):
        decision = run_refinement(
            config,
            seeds_override=args.seeds if args.seeds else None,
            folds_override=args.folds if args.folds else None,
            overwrite=args.overwrite,
            # The final run intentionally relaxes the 3-seed gate to 10 seeds.
            enforce_seed_gate=False,
        )
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
