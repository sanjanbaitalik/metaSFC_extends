#!/usr/bin/env python3
"""
Select additional unique HCP subjects excluding an existing pilot list.

Use case:
  You already processed/downloaded ~50 subjects and need additional subjects
  to build a final corpus of 500 subjects.

This script can obtain candidate HCP IDs in two ways:
  1. From HCP OpenAccess S3 using AWS CLI.
  2. From a local text file containing candidate subject IDs.

Outputs:
  - existing_unique_subjects.txt
  - additional_450_subjects.txt, or another requested count
  - subjects_final.txt
  - selection_report.json
  - optional machine-wise split files

Example:
  python scripts/38_select_additional_hcp_subjects.py \
    --existing_file data/hcp/manifest/subjects_pilot.txt \
    --additional_count 450 \
    --source aws \
    --aws_profile hcp \
    --out_dir data/hcp/manifest/corpus_500

If your existing file has duplicates and you want exactly 500 unique total subjects:
  python scripts/38_select_additional_hcp_subjects.py \
    --existing_file data/hcp/manifest/subjects_pilot.txt \
    --target_total 500 \
    --source aws \
    --aws_profile hcp \
    --out_dir data/hcp/manifest/corpus_500
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple


SUBJECT_RE = re.compile(r"^\d{6}$")


def read_subjects(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Subject file not found: {path}")

    subjects: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Keep only first whitespace-separated token, useful if a line has comments.
        s = s.split()[0].strip()
        if SUBJECT_RE.match(s):
            subjects.append(s)
        else:
            print(f"[WARN] Ignoring non-HCP-looking subject ID: {s!r}")

    return subjects


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def find_duplicates(items: Iterable[str]) -> dict:
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


def run_cmd(cmd: list[str]) -> str:
    print("[CMD]", " ".join(cmd))
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout


def collect_candidates_from_aws(profile: str, region: str, bucket: str) -> List[str]:
    """
    Reads subject folder prefixes from HCP OpenAccess S3.

    Expected AWS CLI output lines look like:
      PRE 100307/

    Important: the HCP_1200 prefix must end with '/'. Without the trailing slash,
    AWS CLI may list the parent prefix and only return 'PRE HCP_1200/'.
    """
    if not bucket.endswith("/"):
        bucket = bucket + "/"

    cmd = [
        "aws", "s3", "ls", bucket,
        "--profile", profile,
        "--region", region,
    ]
    stdout = run_cmd(cmd)

    candidates = []
    for line in stdout.splitlines():
        line = line.strip()
        # Usually: PRE 100307/
        parts = line.split()
        if not parts:
            continue
        token = parts[-1].rstrip("/")
        if SUBJECT_RE.match(token):
            candidates.append(token)

    candidates = sorted(set(candidates), key=lambda x: int(x))
    if not candidates:
        preview = "\n".join(stdout.splitlines()[:20])
        raise RuntimeError(
            "No 6-digit subject folders found from AWS listing.\n"
            f"Bucket used: {bucket}\n"
            "If you only see 'PRE HCP_1200/', rerun with '--aws_bucket s3://hcp-openaccess/HCP_1200/'.\n"
            f"First AWS output lines:\n{preview}"
        )
    return candidates


def collect_candidates_from_file(path: Path) -> List[str]:
    return sorted(set(read_subjects(path)), key=lambda x: int(x))


def filter_by_behavior(
    candidates: List[str],
    behavior_csv: Path,
    target_column: str,
) -> Tuple[List[str], dict]:
    import pandas as pd

    if not behavior_csv.exists():
        raise FileNotFoundError(f"Behavior CSV not found: {behavior_csv}")

    df = pd.read_csv(behavior_csv)

    subject_col = None
    for col in ["Subject", "subject", "Subject_ID", "subject_id"]:
        if col in df.columns:
            subject_col = col
            break

    if subject_col is None:
        raise ValueError(
            "Could not find subject column in behavior CSV. "
            "Expected one of: Subject, subject, Subject_ID, subject_id."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column {target_column!r} not found in behavior CSV. "
            f"Available columns include: {list(df.columns)[:30]}"
        )

    df[subject_col] = df[subject_col].astype(str)
    valid = set(
        df.loc[df[target_column].notna(), subject_col]
        .astype(str)
        .str.strip()
        .tolist()
    )

    before = len(candidates)
    filtered = [s for s in candidates if s in valid]

    report = {
        "behavior_csv": str(behavior_csv),
        "target_column": target_column,
        "subject_column": subject_col,
        "candidates_before_behavior_filter": before,
        "candidates_after_behavior_filter": len(filtered),
    }
    return filtered, report


def verify_required_s3_files(
    candidates: List[str],
    profile: str,
    region: str,
    bucket: str,
    limit_needed: int,
) -> List[str]:
    """
    Optional slow verification. It checks whether required HCP files exist for
    each subject and stops after enough valid subjects are found.

    Use only if you want a stricter list before downloading.
    """
    required_relpaths = [
        "MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz",
        "MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_hp2000_clean.nii.gz",
        "T1w/Diffusion/data.nii.gz",
        "T1w/Diffusion/bvals",
        "T1w/Diffusion/bvecs",
        "T1w/Diffusion/nodif_brain_mask.nii.gz",
        "T1w/T1w_acpc_dc_restore.nii.gz",
        "MNINonLinear/xfms/standard2acpc_dc.nii.gz",
    ]

    valid = []
    for i, sub in enumerate(candidates, start=1):
        print(f"[VERIFY] {i}/{len(candidates)} subject {sub}")
        ok = True
        for rel in required_relpaths:
            cmd = [
                "aws", "s3", "ls",
                f"{bucket}/{sub}/{rel}",
                "--profile", profile,
                "--region", region,
            ]
            try:
                out = run_cmd(cmd)
                if not out.strip():
                    ok = False
                    print(f"  [MISSING] {rel}")
                    break
            except subprocess.CalledProcessError:
                ok = False
                print(f"  [MISSING/ERROR] {rel}")
                break

        if ok:
            valid.append(sub)
            print(f"  [OK] {sub}; valid so far: {len(valid)}")

        if len(valid) >= limit_needed:
            break

    return valid


def write_list(path: Path, subjects: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(subjects) + "\n")


def write_machine_splits(subjects: List[str], out_dir: Path, machines: int) -> List[str]:
    split_paths = []
    if machines <= 0:
        return split_paths

    split_dir = out_dir / "machine_splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    batch_size = math.ceil(len(subjects) / machines)
    for i in range(machines):
        machine_id = chr(ord("A") + i)
        chunk = subjects[i * batch_size:(i + 1) * batch_size]
        p = split_dir / f"machine_{machine_id}_subjects.txt"
        write_list(p, chunk)
        split_paths.append(str(p))
        print(f"[SPLIT] Machine {machine_id}: {len(chunk)} subjects -> {p}")

    return split_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing_file", required=True,
                        help="Text file containing already downloaded/processed subject IDs.")
    parser.add_argument("--out_dir", default="data/hcp/manifest/corpus_500")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--additional_count", type=int,
                      help="Number of additional unique subjects to select, e.g., 450.")
    mode.add_argument("--target_total", type=int,
                      help="Final number of unique subjects desired, e.g., 500.")

    parser.add_argument("--source", choices=["aws", "file"], default="aws")
    parser.add_argument("--candidate_file", default=None,
                        help="Required if --source file. Text file with candidate HCP subject IDs.")

    parser.add_argument("--aws_profile", default="hcp")
    parser.add_argument("--aws_region", default="us-east-1")
    parser.add_argument("--aws_bucket", default="s3://hcp-openaccess/HCP_1200/")

    parser.add_argument("--behavior_csv", default=None,
                        help="Optional HCP behavioral CSV to filter subjects with available target labels.")
    parser.add_argument("--target_column", default="PMAT24_A_CR",
                        help="Behavior target column used when --behavior_csv is provided.")

    parser.add_argument("--verify_s3_files", action="store_true",
                        help="Slow but stricter: check required fMRI/DWI/T1w files on S3 before selecting.")

    parser.add_argument("--machines", type=int, default=0,
                        help="Optionally split the additional selected subjects across N machines.")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_raw = read_subjects(Path(args.existing_file))
    existing_unique = unique_preserve_order(existing_raw)
    duplicates = find_duplicates(existing_raw)

    if duplicates:
        print("[WARN] Existing file contains duplicate subjects:")
        for sub, count in duplicates.items():
            print(f"  {sub}: {count} times")

    if args.target_total is not None:
        needed = args.target_total - len(existing_unique)
        if needed <= 0:
            raise ValueError(
                f"target_total={args.target_total} is not larger than "
                f"existing unique count={len(existing_unique)}"
            )
    else:
        needed = args.additional_count

    print(f"Existing lines: {len(existing_raw)}")
    print(f"Existing unique subjects: {len(existing_unique)}")
    print(f"Additional subjects needed: {needed}")

    if args.source == "aws":
        candidates = collect_candidates_from_aws(
            profile=args.aws_profile,
            region=args.aws_region,
            bucket=args.aws_bucket,
        )
    else:
        if not args.candidate_file:
            raise ValueError("--candidate_file is required when --source file")
        candidates = collect_candidates_from_file(Path(args.candidate_file))

    behavior_report = {}
    if args.behavior_csv:
        candidates, behavior_report = filter_by_behavior(
            candidates=candidates,
            behavior_csv=Path(args.behavior_csv),
            target_column=args.target_column,
        )

    existing_set = set(existing_unique)
    additional_pool = [s for s in candidates if s not in existing_set]

    if args.verify_s3_files:
        additional_pool = verify_required_s3_files(
            additional_pool,
            profile=args.aws_profile,
            region=args.aws_region,
            bucket=args.aws_bucket,
            limit_needed=needed,
        )

    if len(additional_pool) < needed:
        raise RuntimeError(
            f"Not enough candidates. Need {needed}, available {len(additional_pool)}. "
            "Try removing filters or checking HCP/AWS access."
        )

    additional = additional_pool[:needed]
    final_subjects = existing_unique + additional

    existing_out = out_dir / "existing_unique_subjects.txt"
    additional_out = out_dir / f"additional_{needed}_subjects.txt"
    final_out = out_dir / f"subjects_{len(final_subjects)}_final.txt"

    write_list(existing_out, existing_unique)
    write_list(additional_out, additional)
    write_list(final_out, final_subjects)

    split_paths = write_machine_splits(additional, out_dir, args.machines)

    report = {
        "existing_file": str(args.existing_file),
        "existing_lines": len(existing_raw),
        "existing_unique_count": len(existing_unique),
        "duplicates_in_existing": duplicates,
        "source": args.source,
        "candidate_file": args.candidate_file,
        "aws_bucket": args.aws_bucket if args.source == "aws" else None,
        "behavior_filter": behavior_report,
        "additional_needed": needed,
        "additional_selected_count": len(additional),
        "final_unique_count": len(set(final_subjects)),
        "final_line_count": len(final_subjects),
        "outputs": {
            "existing_unique_subjects": str(existing_out),
            "additional_subjects": str(additional_out),
            "final_subjects": str(final_out),
            "machine_splits": split_paths,
        },
    }

    report_out = out_dir / "selection_report.json"
    report_out.write_text(json.dumps(report, indent=2))

    print("\nSaved:")
    print(" ", existing_out)
    print(" ", additional_out)
    print(" ", final_out)
    print(" ", report_out)

    if len(set(final_subjects)) != len(final_subjects):
        print("[WARN] Final list still contains duplicates. This should not happen.")
    else:
        print("[OK] Final list contains unique subjects only.")

    print(f"Final unique subject count: {len(set(final_subjects))}")

    if args.additional_count is not None and len(existing_unique) + args.additional_count != 500:
        print(
            "[NOTE] You requested a fixed additional_count. "
            f"Because existing unique count is {len(existing_unique)}, "
            f"your final unique count is {len(existing_unique) + args.additional_count}. "
            "If you want exactly 500 unique subjects, rerun with --target_total 500."
        )


if __name__ == "__main__":
    main()
