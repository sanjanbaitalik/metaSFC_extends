from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def check_fc(path):
    if not path.exists():
        return False, "missing"
    try:
        x = np.load(path)
        if x.shape != (116, 116):
            return False, f"bad shape {x.shape}"
        if not np.isfinite(x).all():
            return False, "non-finite values"
        if np.abs(x - x.T).max() > 1e-4:
            return False, "not symmetric"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def check_sc(path):
    if not path.exists():
        return False, "missing"
    try:
        x = np.loadtxt(path, delimiter=",")
        if x.shape != (116, 116):
            return False, f"bad shape {x.shape}"
        if not np.isfinite(x).all():
            return False, "non-finite values"
        if np.abs(x - x.T).max() > 1e-4:
            return False, "not symmetric"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_file", required=True)
    parser.add_argument("--out_csv", default=None)
    args = parser.parse_args()

    subjects = [
        s.strip()
        for s in Path(args.batch_file).read_text().splitlines()
        if s.strip()
    ]

    rows = []

    for sub in subjects:
        fc_path = Path(f"data/hcp/processed/fc/{sub}_fc.npy")
        sc_path = Path(f"data/hcp/processed/sc/{sub}/sc_116.csv")

        fc_ok, fc_msg = check_fc(fc_path)
        sc_ok, sc_msg = check_sc(sc_path)

        status = "ok" if fc_ok and sc_ok else "fail"

        rows.append({
            "subject": sub,
            "fc_ok": fc_ok,
            "fc_msg": fc_msg,
            "sc_ok": sc_ok,
            "sc_msg": sc_msg,
            "status": status,
        })

        print(sub, status, "| FC:", fc_msg, "| SC:", sc_msg)

    df = pd.DataFrame(rows)

    if args.out_csv is None:
        batch_name = Path(args.batch_file).stem
        args.out_csv = f"data/hcp/qc/{batch_name}_qc.csv"

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("\nSaved QC:", args.out_csv)
    print(df["status"].value_counts())


if __name__ == "__main__":
    main()