#!/usr/bin/env python3
"""108 -- Post-final reporting patch. No model reruns."""
import argparse
from pathlib import Path

from metascfc.experiments.msancr_postfinal_reporting import run_postfinal_reporting


def main() -> None:
    ap = argparse.ArgumentParser(description="Post-final reporting: correct metadata + family-aware stats")
    ap.add_argument("--config", default="configs/iclr/msancr_final_10x5.yaml")
    ap.add_argument("--output-dir", default="outputs/iclr/msancr_final_10x5",
                    help="Directory containing the frozen 10x5 outputs")
    ap.add_argument("--report-dir", default="outputs/iclr/msancr_final_10x5/postfinal_reporting",
                    help="Where to write corrected reporting artifacts")
    args = ap.parse_args()

    results = run_postfinal_reporting(
        final_output_dir=args.output_dir,
        config_path=args.config,
        report_output_dir=args.report_dir,
    )
    print("POST-FINAL REPORTING COMPLETE")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
