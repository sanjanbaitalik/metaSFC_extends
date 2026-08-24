"""Dual-task HCP behavioral target loading (ICLR 2027 dual-task matrix).

The ICLR 2027 pivot evaluates priors on TWO cognitive domains:

- Fluid intelligence  -> HCP ``PMAT24_A_CR``  (the existing AAAI target)
- Working memory      -> HCP ``ListSort_Unadj`` (HCP List Sorting working
  memory) or an n-back accuracy measure (e.g. ``WM_Task_2back_Acc``)

The FC/SC arrays are shared across tasks; only the label vector changes.
This module aligns a behavioral-measure column of the HCP behavior table
("unrestricted" CSV, one row per Subject) to the subject order of
``inputs/dataset_SC/hcp_subjects_used.csv`` and materializes per-task label
files in exactly the layout every existing loader expects:

    <out_dir>/
        label_all.npy        [N] raw behavioral scores (float64)
        label_metadata.json  {"target": ..., "mean": ..., "std": ..., ...}

so ``benchmark_utils.load_connectomes`` (and scripts 40-47, the AAAI loop,
and the faithfulness protocol) consume the new tasks without modification:
point the config's ``data.y_path`` at the generated file.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Friendly aliases -> canonical HCP behavioral measure columns.
KNOWN_TARGETS: Dict[str, str] = {
    "fluid_intelligence": "PMAT24_A_CR",
    "pmat": "PMAT24_A_CR",
    "working_memory": "ListSort_Unadj",
    "list_sorting_wm": "ListSort_Unadj",
    "listsort": "ListSort_Unadj",
    "wm_nback": "WM_Task_2back_Acc",
    "wm_nback_0back": "WM_Task_0back_Acc",
    "wm_nback_place": "WM_Task_Place_Acc",
    "wm_nback_face": "WM_Task_Face_Acc",
}

DEFAULT_SUBJECTS_CSV = "inputs/dataset_SC/hcp_subjects_used.csv"
DEFAULT_OUT_ROOT = "inputs/dataset_SC/task_labels"


def resolve_target(target: str) -> str:
    """Return the canonical HCP column name for ``target`` (alias-aware)."""
    if target in KNOWN_TARGETS.values():
        return target
    key = target.strip().lower()
    if key in KNOWN_TARGETS:
        return KNOWN_TARGETS[key]
    raise ValueError(
        f"Unknown target '{target}'. Known targets: "
        f"{sorted(set(KNOWN_TARGETS.values()))} or aliases {sorted(KNOWN_TARGETS)}"
    )


def _subject_column(df: pd.DataFrame, path: Path) -> str:
    for col in ("Subject", "subject", "SUBJECT", "subj"):
        if col in df.columns:
            return col
    raise ValueError(f"{path} has no subject ID column (looked for Subject/subject)")


def load_hcp_behavior(behavior_csv: str | Path) -> pd.DataFrame:
    """Load the HCP behavior table indexed by string subject IDs."""
    path = Path(behavior_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"HCP behavior table not found: {path}. Export the 'unrestricted' "
            "behavioral data CSV from the ConnectomeDB / BALSA and pass its path."
        )
    df = pd.read_csv(path, on_bad_lines='skip')
    col = _subject_column(df, path)
    df[col] = df[col].astype(str)
    return df.set_index(col)


def load_task_target(
    subjects_csv: str | Path,
    behavior_csv: str | Path,
    target: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Align one behavioral measure to the used-subject order.

    Returns (y, metadata_df).  Raises on missing subjects/column or NaN
    entries - silently imputing scores would corrupt every downstream CV.
    """
    canonical = resolve_target(target)
    subj_df = pd.read_csv(subjects_csv)
    scol = _subject_column(subj_df, Path(subjects_csv))
    subject_ids = subj_df[scol].astype(str)

    beh = load_hcp_behavior(behavior_csv)
    if canonical not in beh.columns:
        raise ValueError(
            f"Column '{canonical}' not found in {behavior_csv}. Available "
            f"columns include: {[c for c in beh.columns[:20]]}..."
        )
    try:
        values = beh.loc[subject_ids, canonical].to_numpy(np.float64)
    except KeyError as exc:
        missing = sorted(set(subject_ids) - set(beh.index))
        raise ValueError(
            f"{len(missing)} used subjects are absent from the behavior table "
            f"(e.g. {missing[:5]}...)"
        ) from exc

    nan_mask = ~np.isfinite(values)
    if nan_mask.any():
        bad = subject_ids[nan_mask].tolist()
        raise ValueError(
            f"{int(nan_mask.sum())} subjects have NaN '{canonical}' "
            f"(e.g. {bad[:5]}...). Exclude them from hcp_subjects_used.csv."
        )
    meta = pd.DataFrame({
        "subject_id": subject_ids,
        "target": canonical,
        "value": values,
    })
    return values, meta


def dual_target_qc(
    subject_ids: list[str] | pd.Series,
    behavior_csv: str | Path,
    targets: tuple[str, ...] = ("PMAT24_A_CR", "ListSort_Unadj"),
) -> tuple[list[str], Dict[str, object]]:
    """Intersection QC across multiple behavioral targets.

    Real HCP data has NaN measures for subjects who skipped a task
    (e.g. ``ListSort_Unadj``).  This returns the subjects that have finite
    values for EVERY requested target plus a structured drop log with the
    exact counts per exclusion reason - never silent imputation.

    Returns
    -------
    (kept_ids, log)
        log keys: ``n_input``, ``n_kept``, ``n_dropped``,
        ``dropped_missing_behavior_row``, ``dropped_nan_per_target``
        (dict target -> count), ``dropped_ids_by_reason``.
    """
    canonical_targets = [resolve_target(t) for t in targets]
    ids = [str(s) for s in subject_ids]
    beh = load_hcp_behavior(behavior_csv)
    for col in canonical_targets:
        if col not in beh.columns:
            raise ValueError(f"Column '{col}' not found in {behavior_csv}")

    dropped_rows = sorted(set(ids) - set(beh.index))
    present = [s for s in ids if s not in set(dropped_rows)]
    nan_counts: Dict[str, int] = {}
    nan_ids: Dict[str, list[str]] = {col: [] for col in canonical_targets}
    keep = np.ones(len(present), dtype=bool)
    for j, col in enumerate(canonical_targets):
        values = beh.loc[present, col].to_numpy(np.float64)
        mask = ~np.isfinite(values)
        nan_counts[col] = int(mask.sum())
        for k in np.where(mask)[0]:
            nan_ids[col].append(present[k])
        keep &= ~mask

    kept = [s for s, k in zip(present, keep) if k]
    log: Dict[str, object] = {
        "n_input": len(ids),
        "n_kept": len(kept),
        "n_dropped": len(ids) - len(kept),
        "dropped_missing_behavior_row": dropped_rows,
        "dropped_nan_per_target": nan_counts,
        "dropped_ids_by_reason": {
            "missing_row": dropped_rows,
            **{col: nan_ids[col] for col in canonical_targets},
        },
        "targets": canonical_targets,
    }
    return kept, log


def build_task_labels(
    subjects_csv: str | Path,
    behavior_csv: str | Path,
    target: str,
    out_dir: Optional[str | Path] = None,
) -> Path:
    """Write label_all.npy + label_metadata.json for one cognitive task."""
    canonical = resolve_target(target)
    y, _ = load_task_target(subjects_csv, behavior_csv, target)
    out_path = Path(out_dir) if out_dir else Path(DEFAULT_OUT_ROOT) / canonical
    out_path.mkdir(parents=True, exist_ok=True)
    np.save(out_path / "label_all.npy", y.astype(np.float64))
    (out_path / "label_metadata.json").write_text(json.dumps({
        "target": canonical,
        "globally_standardized": False,
        "mean": float(y.mean()),
        "std": float(y.std()),
        "n_subjects": int(len(y)),
    }, indent=2))
    print(f"Wrote {out_path / 'label_all.npy'} "
          f"[{len(y)} subjects, mean={y.mean():.3f}, std={y.std():.3f}]")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Materialize per-task HCP labels.")
    ap.add_argument("--behavior-csv", required=True,
                    help="HCP unrestricted behavioral data CSV")
    ap.add_argument("--subjects-csv", default=DEFAULT_SUBJECTS_CSV)
    ap.add_argument("--targets", nargs="+", required=True,
                    help="e.g. fluid_intelligence working_memory wm_nback "
                         "(aliases or raw HCP column names)")
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()
    for target in args.targets:
        build_task_labels(args.subjects_csv, args.behavior_csv, target,
                          out_dir=Path(args.out_root) / resolve_target(target))


if __name__ == "__main__":
    main()
