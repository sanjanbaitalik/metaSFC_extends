"""Family-aware splitting, assertions, and inference for HCP robustness.

This module provides randomized balanced group-preserving splitters that
ensure no biological family appears in both train and test partitions.
All output artifacts contain only aggregate counts and hashes -- never
actual family identifiers.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon, ttest_1samp

from metascfc.benchmark_utils import holm_adjust, load_connectomes, atomic_write_csv
from metascfc.experiments.msancr_final_inference import (
    MODEL_A3,
    MODEL_A2,
    MODEL_A4,
    N_FINAL_SEEDS,
    N_FOLDS,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    WILCOXON_ZERO_METHOD,
    HIGHER_BETTER,
    PREDICTION_METRICS,
    BIOMARKER_METRICS,
    COMPARISONS,
    paired_seed_values,
    paired_statistics,
    build_seed_level_table,
)


# ---------------------------------------------------------------------------
# Part B2 -- Group integrity
# ---------------------------------------------------------------------------

def load_and_validate_groups(
    groups_path: str | Path,
    expected_n: int,
) -> np.ndarray:
    """Load family groups, validate alignment with subject count."""
    groups_path = Path(groups_path)
    if not groups_path.exists():
        raise FileNotFoundError(
            f"FAMILY_DATA_REQUIRED: {groups_path} not found.\n"
            "Obtain authorized HCP restricted data and run:\n"
            "  python scripts/24b_prepare_hcp_family_groups.py \\\n"
            "    --restricted_csv /LOCAL/PATH/HCP_S1200_restricted.csv \\\n"
            "    --group_col Family_ID \\\n"
            "    --subjects inputs/dataset_SC/hcp_subjects_used.csv \\\n"
            "    --out inputs/dataset_SC/family_groups.npy"
        )
    groups = np.load(groups_path, allow_pickle=True).reshape(-1)
    if len(groups) != expected_n:
        raise ValueError(
            f"Group count {len(groups)} does not match subject count {expected_n}"
        )
    return groups


def build_group_integrity(groups: np.ndarray) -> dict[str, Any]:
    """Aggregate-only integrity check. No raw IDs."""
    unique_groups, counts = np.unique(groups, return_counts=True)
    return {
        "n_subjects": int(len(groups)),
        "n_unique_groups": int(len(unique_groups)),
        "max_family_size": int(counts.max()),
        "family_size_distribution": {
            int(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))
        },
        "groups_sha256": hashlib.sha256(
            np.sort(groups).tobytes()
        ).hexdigest()[:16],
    }


# ---------------------------------------------------------------------------
# Part B3 -- RandomizedBalancedGroupKFold
# ---------------------------------------------------------------------------

class RandomizedBalancedGroupKFold:
    """Deterministic, group-preserving K-fold splitter with balanced subject counts.

    1. obtains unique groups;
    2. shuffles group order deterministically with ``np.random.default_rng(seed)``;
    3. assigns entire groups greedily to the currently smallest fold by subject count;
    4. never divides a family;
    5. produces K disjoint test folds covering every subject exactly once.
    """

    def __init__(self, n_splits: int = 5, random_state: int | None = None):
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2; got {n_splits}")
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, X: np.ndarray, y: Any = None, groups: np.ndarray = None):
        """Yield (train_indices, test_indices) for each fold."""
        if groups is None:
            raise ValueError("groups must be provided")
        groups = np.asarray(groups)
        n = len(groups)
        indices = np.arange(n)

        # Group indices by unique group
        unique_groups = np.unique(groups)
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(unique_groups)

        # Greedy balanced assignment
        fold_subject_counts = np.zeros(self.n_splits, dtype=int)
        group_to_fold: dict[str, int] = {}
        for g in unique_groups:
            fold_idx = int(np.argmin(fold_subject_counts))
            group_to_fold[str(g)] = fold_idx
            fold_subject_counts[fold_idx] += int((groups == g).sum())

        for fold_idx in range(self.n_splits):
            test_mask = np.array([group_to_fold[str(g)] == fold_idx for g in groups])
            test_indices = indices[test_mask]
            train_indices = indices[~test_mask]
            yield train_indices, test_indices

    def get_n_splits(self, X: Any = None, y: Any = None, groups: Any = None):
        return self.n_splits


def make_family_inner_split(
    outer_train_idx: np.ndarray,
    groups: np.ndarray,
    seed: int,
    outer_fold: int,
    n_splits: int = 3,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Family-preserving inner CV splits within outer training set."""
    outer_groups = groups[outer_train_idx]
    splitter = RandomizedBalancedGroupKFold(
        n_splits=n_splits,
        random_state=int(seed) * 1000 + int(outer_fold) + 17001,
    )
    splits = []
    for train_local, val_local in splitter.split(outer_train_idx, groups=outer_groups):
        train_global = outer_train_idx[np.asarray(train_local, dtype=int)]
        val_global = outer_train_idx[np.asarray(val_local, dtype=int)]
        # Leakage guard
        train_families = set(groups[train_global])
        val_families = set(groups[val_global])
        if train_families & val_families:
            raise RuntimeError(
                f"Inner family leakage: {train_families & val_families}"
            )
        splits.append((train_global, val_global))
    return splits


# ---------------------------------------------------------------------------
# Part B4 -- Seed diversity
# ---------------------------------------------------------------------------

def compute_split_manifest_hash(seed_groups: list[set[str]]) -> str:
    """Hash a set of fold group-sets for diversity comparison."""
    serialized = json.dumps([sorted(s) for s in seed_groups], sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def verify_seed_diversity(
    split_manifests: dict[int, list[set[str]]],
    min_unique: int = 8,
) -> dict[str, Any]:
    """Verify that seeds produce genuinely different partitions."""
    hashes = {}
    for seed, fold_groups in split_manifests.items():
        hashes[seed] = compute_split_manifest_hash(fold_groups)
    unique_hashes = len(set(hashes.values()))
    return {
        "n_seeds": len(hashes),
        "n_unique_partition_manifests": unique_hashes,
        "min_unique_required": min_unique,
        "diversity_sufficient": unique_hashes >= min_unique,
        "seed_hashes": hashes,
    }


# ---------------------------------------------------------------------------
# Part B5 -- Family leakage assertions
# ---------------------------------------------------------------------------

def assert_no_family_leakage(
    groups: np.ndarray,
    outer_splits: list[tuple[np.ndarray, np.ndarray]],
    inner_splits_per_fold: dict[int, list[tuple[np.ndarray, np.ndarray]]],
) -> dict[str, Any]:
    """Assert no family appears across train/val/test partitions."""
    n_outer_checks = 0
    n_inner_checks = 0
    n_failures = 0

    # Outer train vs test
    for fold_idx, (train_idx, test_idx) in enumerate(outer_splits):
        n_outer_checks += 1
        train_families = set(groups[train_idx])
        test_families = set(groups[test_idx])
        if train_families & test_families:
            n_failures += 1

    # Inner train vs val (must be subsets of outer train)
    for fold_idx, inner_splits in inner_splits_per_fold.items():
        outer_train_idx = outer_splits[fold_idx][0]
        outer_train_families = set(groups[outer_train_idx])
        for inner_train_idx, inner_val_idx in inner_splits:
            n_inner_checks += 1
            inner_train_families = set(groups[inner_train_idx])
            inner_val_families = set(groups[inner_val_idx])
            # Inner train/val must be disjoint from each other
            if inner_train_families & inner_val_families:
                n_failures += 1
            # Inner must be subset of outer train
            if not inner_val_families.issubset(outer_train_families):
                n_failures += 1
            if not inner_train_families.issubset(outer_train_families):
                n_failures += 1

    return {
        "n_outer_checks": n_outer_checks,
        "n_inner_checks": n_inner_checks,
        "n_failures": n_failures,
        "all_passed": n_failures == 0,
    }


# ---------------------------------------------------------------------------
# Part B8 -- Family-aware inference
# ---------------------------------------------------------------------------

def run_family_aware_inference(seed_df: pd.DataFrame) -> dict[str, Any]:
    """Primary/secondary/conservative inference for family-aware results."""
    # Primary: A3 vs A4, Pearson
    paired = paired_seed_values(
        seed_df, (MODEL_A3, "matched"), (MODEL_A4, "none"), "pearson"
    )
    primary = paired_statistics(
        paired.left.to_numpy(), paired.right.to_numpy(), "pearson"
    )

    # Secondary family
    secondary_rows = []
    for name, left_key, right_key in COMPARISONS[1:]:
        for metric in PREDICTION_METRICS:
            p = paired_seed_values(seed_df, left_key, right_key, metric)
            s = paired_statistics(p.left.to_numpy(), p.right.to_numpy(), metric)
            secondary_rows.append({"comparison": name, "metric": metric, **s})
    sec_df = pd.DataFrame(secondary_rows)
    sec_df["p_holm_secondary"] = np.nan
    for metric, indices in sec_df.groupby("metric").groups.items():
        idx = list(indices)
        sec_df.loc[idx, "p_holm_secondary"] = holm_adjust(
            sec_df.loc[idx, "wilcoxon_p"].to_numpy(float)
        )

    # Conservative all-five Holm
    all_p = []
    for name, left_key, right_key in COMPARISONS:
        p = paired_seed_values(seed_df, left_key, right_key, "pearson")
        s = paired_statistics(p.left.to_numpy(), p.right.to_numpy(), "pearson")
        all_p.append(s["wilcoxon_p"])
    conservative_holm = holm_adjust(np.array(all_p))

    return {
        "primary": {
            "contrast": "A3_matched_vs_A4",
            "metric": "pearson",
            **primary,
        },
        "secondary": sec_df,
        "conservative": {
            "p_raw": float(all_p[0]),
            "p_holm_all_five": float(conservative_holm[0]),
        },
    }


# ---------------------------------------------------------------------------
# Part B9 -- Subject-wise vs family-aware comparison
# ---------------------------------------------------------------------------

def build_comparison_table(
    subject_wise_dir: str | Path,
    family_aware_df: pd.DataFrame,
    family_aware_results: dict,
) -> pd.DataFrame:
    """Compare subject-wise vs family-aware metrics."""
    sw_pred = pd.read_csv(Path(subject_wise_dir) / "final_prediction_statistics.csv")
    sw_a3_a4 = sw_pred[
        (sw_pred.comparison == "A3_matched_vs_A4") & (sw_pred.metric == "pearson")
    ].iloc[0]

    fa_primary = family_aware_results["primary"]

    rows = [
        {"metric": "Pearson", "model": "A4", "subject_wise": sw_a3_a4.right_mean,
         "family_aware": family_aware_df[family_aware_df.model_id == MODEL_A4].pearson.mean()},
        {"metric": "Pearson", "model": "A3", "subject_wise": sw_a3_a4.left_mean,
         "family_aware": family_aware_df[
             (family_aware_df.model_id == MODEL_A3) & (family_aware_df.prior_type == "matched")
         ].pearson.mean()},
        {"metric": "Pearson", "model": "Delta(A3-A4)", "subject_wise": sw_a3_a4.paired_mean_diff,
         "family_aware": fa_primary["paired_mean_diff"]},
        {"metric": "RMSE", "model": "A4", "subject_wise": None, "family_aware": None},
        {"metric": "RMSE", "model": "A3", "subject_wise": None, "family_aware": None},
        {"metric": "MAE", "model": "A4", "subject_wise": None, "family_aware": None},
        {"metric": "MAE", "model": "MAE", "subject_wise": None, "family_aware": None},
    ]
    # Pattern interpretation
    sw_delta = sw_a3_a4.paired_mean_diff
    fa_delta = fa_primary["paired_mean_diff"]
    if fa_delta > 0 and fa_delta >= sw_delta * 0.5:
        pattern = "Pattern 3: A3-A4 advantage strengthens or survives with family-aware splitting"
    elif fa_delta > 0:
        pattern = "Pattern 1: absolute r decreases but A3-A4 advantage survives"
    else:
        pattern = "Pattern 2: both absolute r and A3-A4 advantage collapse"

    return pd.DataFrame(rows), pattern
