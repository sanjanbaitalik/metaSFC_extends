#!/usr/bin/env python3
"""Run a compact development-only lambda sweep.

Use this before final E0-E9 runs. Freeze the selected lambdas, then rerun the
final matrix. Sweep outputs must not be mixed into final test tables.
"""
import argparse, copy, subprocess, sys, tempfile, yaml, json
from pathlib import Path
import pandas as pd

DEFAULTS = {
    "node": [0.01, 0.05, 0.1],
    "module": [0.01, 0.05, 0.1],
    "edge": [0.001, 0.01, 0.05],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="True-prior config for the selected level")
    ap.add_argument("--prior_type", choices=["node", "module", "edge"], required=True)
    ap.add_argument("--lambdas", nargs="*", type=float)
    ap.add_argument("--out", default="outputs/aaai/lambda_sweeps")
    args = ap.parse_args()
    base = yaml.safe_load(Path(args.config).read_text())
    vals = args.lambdas or DEFAULTS[args.prior_type]
    out = Path(args.out) / args.prior_type
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for value in vals:
        cfg = copy.deepcopy(base)
        cfg["experiment_id"] = f"SWEEP_{args.prior_type}_{value:g}"
        cfg["experiment_name"] = cfg["experiment_id"]
        cfg["seeds"] = cfg.get("sweep_seeds", [0, 1, 2])
        cfg["n_folds"] = int(cfg.get("sweep_folds", 3))
        cfg["output_dir"] = str(out / f"lambda_{value:g}")
        cfg["lambda_node"] = value if args.prior_type == "node" else 0.0
        cfg["lambda_module"] = value if args.prior_type == "module" else 0.0
        cfg["lambda_edge"] = value if args.prior_type == "edge" else 0.0
        tmp = out / f"config_lambda_{value:g}.yaml"
        tmp.write_text(yaml.safe_dump(cfg, sort_keys=False))
        subprocess.run([sys.executable, "scripts/07_run_aaai_experiment.py", "--config", str(tmp)], check=True)
        m = json.loads((Path(cfg["output_dir"]) / "metrics.json").read_text())
        rows.append({"prior_type": args.prior_type, "lambda": value,
                     "pearson_mean": m.get("pearson_mean"), "rmse_mean": m.get("rmse_mean"),
                     "mae_mean": m.get("mae_mean"),
                     "alignment_mean": next((v for k,v in m.items() if k.startswith("alignment_") and k.endswith("pearson_mean")), None)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "lambda_sweep_summary.csv", index=False)
    best = df.sort_values(["rmse_mean", "alignment_mean"], ascending=[True, False]).iloc[0].to_dict()
    (out / "recommended_lambda.json").write_text(json.dumps(best, indent=2))
    print(df.to_string(index=False)); print("Recommended:", best)

if __name__ == "__main__":
    main()
