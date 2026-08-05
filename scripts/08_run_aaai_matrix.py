#!/usr/bin/env python3
"""Run AAAI E0-E9 configs sequentially with resumability."""
import argparse, subprocess, sys
from pathlib import Path

ORDER = [f"E{i}" for i in range(10)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_dir", default="configs/aaai")
    ap.add_argument("--only", nargs="*", default=None, help="Subset such as E0 E1 E2")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    config_dir = Path(args.config_dir)
    chosen = set(args.only or ORDER)
    for eid in ORDER:
        if eid not in chosen:
            continue
        matches = sorted(config_dir.glob(f"{eid}_*.yaml"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected exactly one config for {eid}, found {matches}")
        cmd = [sys.executable, "scripts/07_run_aaai_experiment.py", "--config", str(matches[0])]
        if args.overwrite:
            cmd.append("--overwrite")
        print("\nRUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
