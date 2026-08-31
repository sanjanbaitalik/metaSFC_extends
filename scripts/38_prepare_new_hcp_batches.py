#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# HCP S1200 AWS configuration
# ---------------------------------------------------------------------

BUCKET = "hcp-openaccess"

# Exact files required by your current FC + SC preprocessing pipeline.
REQUIRED_KEYS = {
    "rest1_lr": (
        "HCP_1200/{sub}/MNINonLinear/Results/"
        "rfMRI_REST1_LR/"
        "rfMRI_REST1_LR_hp2000_clean.nii.gz"
    ),
    "rest1_rl": (
        "HCP_1200/{sub}/MNINonLinear/Results/"
        "rfMRI_REST1_RL/"
        "rfMRI_REST1_RL_hp2000_clean.nii.gz"
    ),
    "dwi": (
        "HCP_1200/{sub}/T1w/Diffusion/data.nii.gz"
    ),
    "bvals": (
        "HCP_1200/{sub}/T1w/Diffusion/bvals"
    ),
    "bvecs": (
        "HCP_1200/{sub}/T1w/Diffusion/bvecs"
    ),
    "dwi_mask": (
        "HCP_1200/{sub}/T1w/Diffusion/"
        "nodif_brain_mask.nii.gz"
    ),
    "t1w": (
        "HCP_1200/{sub}/T1w/"
        "T1w_acpc_dc_restore.nii.gz"
    ),
    "standard2acpc": (
        "HCP_1200/{sub}/MNINonLinear/xfms/"
        "standard2acpc_dc.nii.gz"
    ),
}


# ---------------------------------------------------------------------
# Subject-ID utilities
# ---------------------------------------------------------------------

def normalize_subject(value) -> str:
    """
    Normalize HCP subject IDs.

    Examples
    --------
    100206       -> 100206
    100206.0     -> 100206
    "100206 "    -> 100206
    """
    s = str(value).strip().replace("\r", "")

    if s.endswith(".0"):
        s = s[:-2]

    return s


def is_valid_hcp_subject(subject: str) -> bool:
    """
    Basic HCP-YA ID sanity check.

    HCP S1200 subject IDs are six-digit numeric strings.
    """
    return bool(
        re.fullmatch(r"\d{6}", subject)
    )


# ---------------------------------------------------------------------
# Existing AAAI cohort
# ---------------------------------------------------------------------

def read_existing_subjects(path: Path) -> list[str]:
    """
    Read the authoritative existing AAAI cohort.

    Expected default:
        inputs/dataset_SC/hcp_subjects_used.csv
    """
    df = pd.read_csv(
        path,
        dtype=str,
        low_memory=False,
    )

    candidates = [
        "subject",
        "Subject",
        "subject_id",
        "Subject_ID",
        "id",
    ]

    subject_col = None

    for col in candidates:
        if col in df.columns:
            subject_col = col
            break

    if (
        subject_col is None
        and len(df.columns) == 1
    ):
        subject_col = df.columns[0]

    if subject_col is None:
        raise ValueError(
            f"Could not identify subject column in {path}.\n"
            f"Columns: {list(df.columns)}"
        )

    subjects = [
        normalize_subject(x)
        for x in df[subject_col].dropna()
    ]

    subjects = [
        s
        for s in subjects
        if is_valid_hcp_subject(s)
    ]

    # De-duplicate while preserving file order.
    return list(dict.fromkeys(subjects))


# ---------------------------------------------------------------------
# Behavioral candidate pool
# ---------------------------------------------------------------------

def read_behavioral_candidates(
    path: Path,
) -> list[str]:
    """
    Read the full HCP-YA behavioral subject pool.

    IMPORTANT
    ---------
    This file is only used to obtain candidate IDs.

    A subject is NOT included in the final manifest unless every
    required FC/SC imaging object also exists in AWS.
    """
    print(
        f"Reading behavioral eligibility pool from: {path}"
    )

    df = pd.read_csv(
        path,
        low_memory=False,
        on_bad_lines="skip",
        dtype=str,
    )

    possible_cols = [
        "Subject",
        "subject",
        "Subject_ID",
        "subject_id",
    ]

    subject_col = None

    for col in possible_cols:
        if col in df.columns:
            subject_col = col
            break

    if subject_col is None:
        raise ValueError(
            "Could not find HCP subject ID column.\n"
            f"Available columns include: "
            f"{list(df.columns)[:30]}"
        )

    subjects = [
        normalize_subject(x)
        for x in df[subject_col].dropna()
    ]

    subjects = [
        s
        for s in subjects
        if is_valid_hcp_subject(s)
    ]

    # Preserve the original behavioral-file order.
    return list(dict.fromkeys(subjects))


# ---------------------------------------------------------------------
# AWS utilities
# ---------------------------------------------------------------------

def aws_base_command(
    profile: str | None,
) -> list[str]:
    """
    Construct the common AWS CLI prefix.

    Uses configured AWS credentials.
    """
    cmd = ["aws"]

    if profile:
        cmd += [
            "--profile",
            profile,
        ]

    return cmd


def verify_aws_configuration(
    profile: str | None,
) -> None:
    """
    Verify that AWS CLI exists and configured credentials are usable.
    """

    try:
        version = subprocess.run(
            ["aws", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "AWS CLI is not installed or is not on PATH."
        ) from exc

    if version.returncode != 0:
        raise RuntimeError(
            "`aws --version` failed."
        )

    cmd = (
        aws_base_command(profile)
        + [
            "sts",
            "get-caller-identity",
            "--output",
            "json",
        ]
    )

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "AWS credentials are configured incorrectly or "
            "are no longer valid.\n\n"
            f"STDERR:\n{result.stderr}"
        )

    print()
    print("AWS credential check: PASS")


def head_hcp_object(
    key: str,
    profile: str | None,
) -> tuple[bool, str | None]:
    """
    Check one exact HCP object using authenticated AWS credentials.

    Returns
    -------
    (True, None)
        object exists

    (False, "not_found")
        object genuinely does not exist

    Raises
    ------
    RuntimeError
        credentials/access/network/etc. failed.

    Important
    ---------
    We intentionally DO NOT use --no-sign-request.
    """

    cmd = (
        aws_base_command(profile)
        + [
            "s3api",
            "head-object",
            "--bucket",
            BUCKET,
            "--key",
            key,
            "--request-payer",
            "requester",
        ]
    )

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode == 0:
        return True, None

    stderr = (
        result.stderr
        or ""
    ).strip()

    stderr_lower = stderr.lower()

    # Genuine missing-object cases.
    if (
        "404" in stderr_lower
        or "not found" in stderr_lower
        or "nosuchkey" in stderr_lower
        or "no such key" in stderr_lower
    ):
        return False, "not_found"

    # Access problems must NOT be interpreted as a missing image.
    if (
        "accessdenied" in stderr_lower
        or "access denied" in stderr_lower
        or "403" in stderr_lower
        or "invalidaccesskeyid" in stderr_lower
        or "signaturedoesnotmatch" in stderr_lower
        or "expiredtoken" in stderr_lower
    ):
        raise RuntimeError(
            "\nAWS access failed while checking an HCP object.\n\n"
            f"Key:\n{key}\n\n"
            f"AWS error:\n{stderr}\n\n"
            "This is an access/credential problem, not an "
            "imaging-missing result."
        )

    # Any other AWS problem should also stop the workflow.
    raise RuntimeError(
        "\nUnexpected AWS error while checking HCP object.\n\n"
        f"Key:\n{key}\n\n"
        f"AWS error:\n{stderr}"
    )


def validate_hcp_access_with_known_subject(
    subject: str,
    profile: str | None,
) -> None:
    """
    Test AWS HCP access before checking hundreds of subjects.

    Uses one known subject from the original AAAI cohort.
    """

    print()
    print(
        "Testing authenticated HCP object access using "
        f"existing subject {subject}..."
    )

    # Test a small/ordinary object instead of the huge DWI image.
    key = REQUIRED_KEYS[
        "bvals"
    ].format(
        sub=subject
    )

    exists, _ = head_hcp_object(
        key,
        profile,
    )

    if not exists:
        raise RuntimeError(
            f"Known existing AAAI subject {subject} does not have:\n"
            f"{key}\n\n"
            "Before continuing, verify that the existing cohort "
            "and AWS bucket/path correspond to the same HCP release."
        )

    print("HCP AWS object access: PASS")


# ---------------------------------------------------------------------
# Subject imaging completeness
# ---------------------------------------------------------------------

def check_subject_requirements(
    subject: str,
    profile: str | None,
) -> tuple[bool, list[str]]:
    """
    Verify all files required by the current FC/SC pipeline.
    """

    missing: list[str] = []

    for requirement, template in (
        REQUIRED_KEYS.items()
    ):
        key = template.format(
            sub=subject
        )

        exists, _ = head_hcp_object(
            key,
            profile,
        )

        if not exists:
            missing.append(
                requirement
            )

    return (
        len(missing) == 0,
        missing,
    )


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def write_subject_file(
    path: Path,
    subjects: list[str],
) -> None:
    """
    Write one subject ID per line.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    content = (
        "\n".join(subjects)
        + ("\n" if subjects else "")
    )

    path.write_text(
        content,
        encoding="utf-8",
    )


def remove_stale_batch_files(
    out_dir: Path,
) -> None:
    """
    Prevent old manifests from mixing with a newly generated run.
    """

    for path in out_dir.glob(
        "machine_A_subjects_*.txt"
    ):
        path.unlink()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create new HCP-YA imaging-ready subject manifests "
            "while excluding the existing AAAI cohort."
        )
    )

    parser.add_argument(
        "--behavior-csv",
        type=Path,
        default=Path(
            "data/hcp/behavior/"
            "unrestricted_behavioral.csv"
        ),
        help=(
            "Full HCP-YA behavioral CSV. "
            "Used only as the candidate-ID pool."
        ),
    )

    parser.add_argument(
        "--existing",
        type=Path,
        default=Path(
            "inputs/dataset_SC/"
            "hcp_subjects_used.csv"
        ),
        help=(
            "Authoritative existing AAAI cohort."
        ),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "data/hcp/manifest/new_batches"
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--n-batches",
        type=int,
        default=None,
        help=(
            "Maximum number of batches. "
            "Example: --n-batches 3 selects "
            "at most 300 subjects."
        ),
    )

    parser.add_argument(
        "--aws-profile",
        type=str,
        default=None,
        help=(
            "Optional configured AWS profile. "
            "Leave unset to use the normal/default "
            "configured AWS credentials."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------

    if not args.behavior_csv.exists():
        raise FileNotFoundError(
            f"Behavior CSV not found:\n"
            f"{args.behavior_csv}"
        )

    if not args.existing.exists():
        raise FileNotFoundError(
            f"Existing cohort file not found:\n"
            f"{args.existing}"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0"
        )

    if (
        args.n_batches is not None
        and args.n_batches <= 0
    ):
        raise ValueError(
            "--n-batches must be > 0"
        )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # AWS authentication preflight
    # ---------------------------------------------------------

    verify_aws_configuration(
        args.aws_profile
    )

    # ---------------------------------------------------------
    # Read cohorts
    # ---------------------------------------------------------

    behavioral_subjects = (
        read_behavioral_candidates(
            args.behavior_csv
        )
    )

    existing_subjects = (
        read_existing_subjects(
            args.existing
        )
    )

    behavioral_set = set(
        behavioral_subjects
    )

    existing_set = set(
        existing_subjects
    )

    print()
    print(
        f"Behavioral candidate subjects: "
        f"{len(behavioral_subjects)}"
    )

    print(
        f"Existing AAAI cohort: "
        f"{len(existing_subjects)}"
    )

    if not existing_subjects:
        raise RuntimeError(
            "Existing cohort is empty."
        )

    # ---------------------------------------------------------
    # Verify HCP AWS access with one known existing subject.
    # ---------------------------------------------------------

    validate_hcp_access_with_known_subject(
        existing_subjects[0],
        args.aws_profile,
    )

    # ---------------------------------------------------------
    # Candidate pool:
    #
    # behavioral HCP-YA subjects
    # MINUS original AAAI 412.
    #
    # Imaging eligibility is determined below by exact HEAD checks.
    # ---------------------------------------------------------

    new_candidates = [
        sub
        for sub in behavioral_subjects
        if sub not in existing_set
    ]

    print()
    print(
        f"Subjects after excluding existing cohort: "
        f"{len(new_candidates)}"
    )

    # ---------------------------------------------------------
    # Save provenance files before AWS imaging filtering.
    # ---------------------------------------------------------

    write_subject_file(
        args.out_dir
        / "subjects_behavioral_all.txt",
        behavioral_subjects,
    )

    write_subject_file(
        args.out_dir
        / "subjects_existing_412_excluded.txt",
        existing_subjects,
    )

    write_subject_file(
        args.out_dir
        / "subjects_behavioral_after_412_exclusion.txt",
        new_candidates,
    )

    # ---------------------------------------------------------
    # Exact imaging completeness check.
    # ---------------------------------------------------------

    imaging_ready: list[str] = []
    rejected: list[dict[str, object]] = []

    print()
    print(
        "Checking exact FC/SC prerequisites..."
    )

    print(
        f"Candidates: {len(new_candidates)}"
    )

    print(
        f"Required AWS objects per subject: "
        f"{len(REQUIRED_KEYS)}"
    )

    print()

    total = len(
        new_candidates
    )

    for idx, subject in enumerate(
        new_candidates,
        start=1,
    ):

        ready, missing = (
            check_subject_requirements(
                subject,
                args.aws_profile,
            )
        )

        if ready:

            imaging_ready.append(
                subject
            )

            print(
                f"[{idx:4d}/{total}] "
                f"{subject}: READY"
            )

        else:

            rejected.append(
                {
                    "subject": subject,
                    "n_missing": len(
                        missing
                    ),
                    "missing": ";".join(
                        missing
                    ),
                }
            )

            print(
                f"[{idx:4d}/{total}] "
                f"{subject}: SKIP "
                f"({len(missing)} missing: "
                f"{','.join(missing)})"
            )

    # ---------------------------------------------------------
    # Save complete imaging-ready pool.
    # ---------------------------------------------------------

    write_subject_file(
        args.out_dir
        / "subjects_new_imaging_ready_all.txt",
        imaging_ready,
    )

    # ---------------------------------------------------------
    # Save exclusions.
    # ---------------------------------------------------------

    rejected_csv = (
        args.out_dir
        / "subjects_rejected_missing_imaging.csv"
    )

    with rejected_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "subject",
                "n_missing",
                "missing",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rejected
        )

    # ---------------------------------------------------------
    # Select N × batch-size if requested.
    # ---------------------------------------------------------

    selected = list(
        imaging_ready
    )

    total_available_ready = len(
        selected
    )

    if args.n_batches is not None:

        max_subjects = (
            args.n_batches
            * args.batch_size
        )

        selected = selected[
            :max_subjects
        ]

    # ---------------------------------------------------------
    # Final selected manifest.
    # ---------------------------------------------------------

    write_subject_file(
        args.out_dir
        / "subjects_selected_for_run.txt",
        selected,
    )

    # ---------------------------------------------------------
    # Ensure stale manifests cannot survive.
    # ---------------------------------------------------------

    remove_stale_batch_files(
        args.out_dir
    )

    # ---------------------------------------------------------
    # Split into 100-subject batches.
    # ---------------------------------------------------------

    batch_files: list[
        tuple[Path, int]
    ] = []

    for start in range(
        0,
        len(selected),
        args.batch_size,
    ):

        batch = selected[
            start:
            start + args.batch_size
        ]

        batch_idx = (
            start // args.batch_size
        ) + 1

        batch_path = (
            args.out_dir
            / (
                "machine_A_subjects_"
                f"{batch_idx:03d}.txt"
            )
        )

        write_subject_file(
            batch_path,
            batch,
        )

        batch_files.append(
            (
                batch_path,
                len(batch),
            )
        )

    # ---------------------------------------------------------
    # Integrity checks
    # ---------------------------------------------------------

    selected_set = set(
        selected
    )

    overlap = (
        selected_set
        & existing_set
    )

    duplicate_count = (
        len(selected)
        - len(selected_set)
    )

    missing_behavior = (
        selected_set
        - behavioral_set
    )

    # ---------------------------------------------------------
    # Final report
    # ---------------------------------------------------------

    print()
    print(
        "======================================"
    )

    print(
        "      HCP-YA MANIFEST SUMMARY"
    )

    print(
        "======================================"
    )

    print(
        f"Behavioral candidate pool:       "
        f"{len(behavioral_subjects)}"
    )

    print(
        f"Existing AAAI cohort:            "
        f"{len(existing_subjects)}"
    )

    print(
        f"Candidates after exclusion:      "
        f"{len(new_candidates)}"
    )

    print(
        f"Exact FC/SC imaging-ready:       "
        f"{total_available_ready}"
    )

    print(
        f"Rejected for missing imaging:    "
        f"{len(rejected)}"
    )

    print(
        f"Selected for this run:           "
        f"{len(selected)}"
    )

    print(
        f"Batch size:                      "
        f"{args.batch_size}"
    )

    print(
        f"Batches created:                 "
        f"{len(batch_files)}"
    )

    print(
        f"Overlap with existing cohort:    "
        f"{len(overlap)}"
    )

    print(
        f"Duplicate selected IDs:          "
        f"{duplicate_count}"
    )

    print(
        f"Selected without behavior:       "
        f"{len(missing_behavior)}"
    )

    print()

    for path, n_subjects in (
        batch_files
    ):
        print(
            f"{path}: "
            f"{n_subjects} subjects"
        )

    print()

    # ---------------------------------------------------------
    # Hard integrity failures
    # ---------------------------------------------------------

    if len(existing_subjects) != 412:
        print(
            "[WARNING] Expected the existing AAAI "
            "cohort to contain 412 subjects, but "
            f"found {len(existing_subjects)}."
        )

    if overlap:
        raise RuntimeError(
            f"{len(overlap)} selected subjects "
            "overlap with the existing AAAI cohort."
        )

    if duplicate_count:
        raise RuntimeError(
            "Duplicate selected subject IDs detected."
        )

    if missing_behavior:
        raise RuntimeError(
            "Selected subjects are missing behavioral metadata."
        )

    if len(selected) == 0:
        raise RuntimeError(
            "No imaging-ready new subjects were found."
        )

    print(
        "Manifest preparation: PASS"
    )


if __name__ == "__main__":
    main()