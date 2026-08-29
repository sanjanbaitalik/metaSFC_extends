#!/usr/bin/env python3
"""Final 10-seed x 5-fold MS-A-NCR runner (authorized by prompt v7).

    python scripts/106_run_msancr_final.py --config configs/iclr/msancr_final_10x5.yaml

The one-fold preflight uses an isolated output directory so it cannot
contaminate the final directory, and supports atomic resume with seed/fold
granularity.  After all 50 outer folds complete, the final inference
analyzer runs automatically and writes FINAL_COMPLETE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml
from threadpoolctl import threadpool_limits

from metascfc.experiments.msancr_final_inference import run_final_inference
from metascfc.experiments.msancr_refinement import run_refinement


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/iclr/msancr_final_10x5.yaml")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--folds", nargs="*", type=int)
    parser.add_argument("--output-dir", default=None,
                        help="Isolated output (use outputs/iclr/msancr_final_smoke for the smoke)")
    parser.add_argument("--figures-dir", default=None)
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip auto-finalization (use for the isolated smoke and to avoid "
                             "premature FINAL_COMPLETE in the final dir)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.figures_dir:
        config["figures_dir"] = args.figures_dir
    with threadpool_limits(limits=int(config.get("n_threads", 4))):
        decision = run_refinement(
            config,
            seeds_override=args.seeds if args.seeds else None,
            folds_override=args.folds if args.folds else None,
            overwrite=args.overwrite,
            enforce_seed_gate=False,
            inference_status="deferred_to_final_inference_n10",
        )
    print(json.dumps(decision, indent=2))
    if not args.skip_inference:
        output_dir = Path(config["output_dir"])
        if not output_dir.is_absolute():
            output_dir = Path(__file__).resolve().parents[1] / output_dir
        if (output_dir / "FINAL_COMPLETE").exists():
            print("FINAL_COMPLETE already present; analyzer skipped.")
            return
        if not decision.get("status") == "complete" or not (output_dir / "COMPLETE").exists():
            print("Refinement not fully complete; final inference deferred.")
            return
        try:
            final_decision = run_final_inference(
                output_dir=output_dir,
                figure_dir=Path(config["figures_dir"]) if config["figures_dir"] else output_dir.parent / "figures",
                config_path=args.config,
                closure_dir="outputs/iclr/msancr_grid_closure",
                n_boot=10000,
            )
            print(json.dumps(final_decision, indent=2))
        except Exception as exc:  # noqa: BLE001 - surface for review; never silently pass
            print(f"FINAL INFERENCE ERROR: {exc}")
            raise


if __name__ == "__main__":
    main()
