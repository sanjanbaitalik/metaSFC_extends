#!/usr/bin/env python3
"""
Preflight-check HCP raw files before SC processing.

Checks whether each subject has the raw files needed by scripts/23_build_hcp_sc_one_subject.sh.
Optionally re-downloads missing files from HCP OpenAccess S3 into the expected local layout.

Expected local layout:
  data/hcp/raw/{SUB}/Diffusion/data.nii.gz
  data/hcp/raw/{SUB}/Diffusion/bvecs
  data/hcp/raw/{SUB}/Diffusion/bvals
  data/hcp/raw/{SUB}/Diffusion/nodif_brain_mask.nii.gz
  data/hcp/raw/{SUB}/T1w/T1w_acpc_dc_restore.nii.gz
  data/hcp/raw/{SUB}/xfms/standard2acpc_dc.nii.gz
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

REQUIRED = {
    "dwi": {
        "local": "Diffusion/data.nii.gz",
        "s3": "T1w/Diffusion/data.nii.gz",
    },
    "bvecs": {
        "local": "Diffusion/bvecs",
        "s3": "T1w/Diffusion/bvecs",
    },
    "bvals": {
        "local": "Diffusion/bvals",
        "s3": "T1w/Diffusion/bvals",
    },
    "mask": {
        "local": "Diffusion/nodif_brain_mask.nii.gz",
        "s3": "T1w/Diffusion/nodif_brain_mask.nii.gz",
    },
    "t1": {
        "local": "T1w/T1w_acpc_dc_restore.nii.gz",
        "s3": "T1w/T1w_acpc_dc_restore.nii.gz",
    },
    "xfm": {
        "local": "xfms/standard2acpc_dc.nii.gz",
        "s3": "MNINonLinear/xfms/standard2acpc_dc.nii.gz",
    },
}


def read_subjects(path: Path) -> List[str]:
    subjects: List[str] = []
    for line in path.read_text().splitlines():
        sub = line.strip().replace("\r", "")
        if sub:
            subjects.append(sub)
    return subjects


def run(cmd: List[str], dry_run: bool = False) -> bool:
    print("[CMD]", " ".join(cmd))
    if dry_run:
        return True
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        print("[ERROR] command failed")
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        return False
    return True


def redownload_file(
    sub: str,
    item: str,
    raw_root: Path,
    bucket: str,
    profile: str,
    region: str,
    dry_run: bool,
) -> bool:
    local_rel = REQUIRED[item]["local"]
    s3_rel = REQUIRED[item]["s3"]
    dest = raw_root / sub / local_rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    bucket = bucket.rstrip("/")
    src = f"{bucket}/{sub}/{s3_rel}"

    cmd = [
        "aws", "s3", "cp", src, str(dest),
        "--profile", profile,
        "--region", region,
    ]
    return run(cmd, dry_run=dry_run)


def check_subject(sub: str, raw_root: Path) -> Tuple[bool, Dict[str, str]]:
    status: Dict[str, str] = {}
    ok_all = True
    for item, meta in REQUIRED.items():
        p = raw_root / sub / meta["local"]
        if p.exists() and p.is_file() and p.stat().st_size > 0:
            status[item] = "ok"
        elif p.exists() and p.is_file() and p.stat().st_size == 0:
            status[item] = "zero_bytes"
            ok_all = False
        else:
            status[item] = "missing"
            ok_all = False
    return ok_all, status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_file", required=True, help="Subject list to check")
    parser.add_argument("--raw_root", default="data/hcp/raw")
    parser.add_argument("--out_dir", default="data/hcp/qc/raw_preflight")
    parser.add_argument("--redownload_missing", action="store_true", help="Re-download missing files from HCP S3")
    parser.add_argument("--aws_profile", default="hcp")
    parser.add_argument("--aws_region", default="us-east-1")
    parser.add_argument("--aws_bucket", default="s3://hcp-openaccess/HCP_1200")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    batch_file = Path(args.batch_file)
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = read_subjects(batch_file)
    batch_name = batch_file.stem

    rows = []
    ready_before = []
    missing_before = []

    print(f"Checking {len(subjects)} subjects from {batch_file}")

    for sub in subjects:
        ok, status = check_subject(sub, raw_root)
        missing_items = [k for k, v in status.items() if v != "ok"]

        if ok:
            ready_before.append(sub)
        else:
            missing_before.append(sub)

        print(f"{sub}: {'READY' if ok else 'MISSING'}", ", ".join(f"{k}={v}" for k, v in status.items()))

        if args.redownload_missing and missing_items:
            print(f"  Attempting re-download for {sub}: {missing_items}")
            for item in missing_items:
                redownload_file(
                    sub=sub,
                    item=item,
                    raw_root=raw_root,
                    bucket=args.aws_bucket,
                    profile=args.aws_profile,
                    region=args.aws_region,
                    dry_run=args.dry_run,
                )

        ok_after, status_after = check_subject(sub, raw_root)
        row = {"subject": sub, "ready": ok_after}
        for item in REQUIRED:
            row[item] = status_after[item]
        row["missing_items"] = ";".join([k for k, v in status_after.items() if v != "ok"])
        rows.append(row)

    # Write CSV report
    report_csv = out_dir / f"{batch_name}_raw_preflight.csv"
    with report_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "ready", *REQUIRED.keys(), "missing_items"])
        writer.writeheader()
        writer.writerows(rows)

    ready_subjects = [r["subject"] for r in rows if r["ready"]]
    missing_subjects = [r["subject"] for r in rows if not r["ready"]]

    ready_txt = out_dir / f"{batch_name}_ready_for_sc.txt"
    missing_txt = out_dir / f"{batch_name}_missing_raw.txt"
    ready_txt.write_text("\n".join(ready_subjects) + ("\n" if ready_subjects else ""))
    missing_txt.write_text("\n".join(missing_subjects) + ("\n" if missing_subjects else ""))

    print("\n===== SUMMARY =====")
    print(f"Total subjects: {len(subjects)}")
    print(f"Ready for SC:   {len(ready_subjects)}")
    print(f"Still missing:  {len(missing_subjects)}")
    print(f"Report CSV:     {report_csv}")
    print(f"Ready list:     {ready_txt}")
    print(f"Missing list:   {missing_txt}")

    if missing_subjects:
        print("\nSubjects still missing raw files:")
        for sub in missing_subjects[:30]:
            missing = next(r["missing_items"] for r in rows if r["subject"] == sub)
            print(f"  {sub}: {missing}")
        if len(missing_subjects) > 30:
            print(f"  ... and {len(missing_subjects) - 30} more")


if __name__ == "__main__":
    main()
