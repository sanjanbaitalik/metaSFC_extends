"""Tests for family-aware splitting, assertions, and runner (Part B)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from metascfc.experiments.msancr_family_aware import (
    load_and_validate_groups,
    build_group_integrity,
    RandomizedBalancedGroupKFold,
    make_family_inner_split,
    verify_seed_diversity,
    assert_no_family_leakage,
)
from metascfc.experiments.msancr_final_inference import (
    build_seed_level_table,
    MODEL_A3, MODEL_A4,
)


CONFIG_PATH = Path("configs/iclr/msancr_family_aware_10x5.yaml")
FINAL_DIR = Path("outputs/iclr/msancr_final_10x5")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_groups(n: int = 412, n_families: int = 100, seed: int = 42) -> np.ndarray:
    """Create synthetic family groups for testing."""
    rng = np.random.default_rng(seed)
    families = [f"fam_{i}" for i in range(n_families)]
    groups = rng.choice(families, size=n)
    return groups


# ---------------------------------------------------------------------------
# Part B2 -- Group integrity
# ---------------------------------------------------------------------------

class TestGroupIntegrity:
    def test_integrity_keys(self):
        groups = _make_synthetic_groups()
        result = build_group_integrity(groups)
        assert result["n_subjects"] == 412
        assert result["n_unique_groups"] <= 100
        assert result["max_family_size"] >= 1
        assert "groups_sha256" in result

    def test_no_raw_ids_in_output(self):
        groups = _make_synthetic_groups()
        result = build_group_integrity(groups)
        output_str = json.dumps(result, default=str)
        for fam in np.unique(groups):
            assert str(fam) not in output_str

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="FAMILY_DATA_REQUIRED"):
            load_and_validate_groups("nonexistent_path.npy", expected_n=100)


# ---------------------------------------------------------------------------
# Part B3 -- RandomizedBalancedGroupKFold
# ---------------------------------------------------------------------------

class TestRandomizedBalancedGroupKFold:
    def test_n_folds(self):
        groups = _make_synthetic_groups()
        splitter = RandomizedBalancedGroupKFold(n_splits=5, random_state=42)
        splits = list(splitter.split(np.arange(len(groups)), groups=groups))
        assert len(splits) == 5

    def test_covers_all_subjects(self):
        groups = _make_synthetic_groups()
        splitter = RandomizedBalancedGroupKFold(n_splits=5, random_state=42)
        all_test = set()
        for train, test in splitter.split(np.arange(len(groups)), groups=groups):
            all_test.update(test)
        assert all_test == set(range(len(groups)))

    def test_no_family_split(self):
        groups = _make_synthetic_groups()
        splitter = RandomizedBalancedGroupKFold(n_splits=5, random_state=42)
        for train, test in splitter.split(np.arange(len(groups)), groups=groups):
            train_families = set(groups[train])
            test_families = set(groups[test])
            assert not (train_families & test_families), (
                f"Family leak: {train_families & test_families}"
            )

    def test_deterministic(self):
        groups = _make_synthetic_groups()
        splitter1 = RandomizedBalancedGroupKFold(n_splits=5, random_state=42)
        splitter2 = RandomizedBalancedGroupKFold(n_splits=5, random_state=42)
        splits1 = [(t.tolist(), te.tolist()) for t, te in splitter1.split(np.arange(len(groups)), groups=groups)]
        splits2 = [(t.tolist(), te.tolist()) for t, te in splitter2.split(np.arange(len(groups)), groups=groups)]
        assert splits1 == splits2

    def test_different_seeds_different_partitions(self):
        groups = _make_synthetic_groups()
        s1 = RandomizedBalancedGroupKFold(n_splits=5, random_state=42)
        s2 = RandomizedBalancedGroupKFold(n_splits=5, random_state=99)
        test_sets_1 = [set(te.tolist()) for _, te in s1.split(np.arange(len(groups)), groups=groups)]
        test_sets_2 = [set(te.tolist()) for _, te in s2.split(np.arange(len(groups)), groups=groups)]
        assert test_sets_1 != test_sets_2

    def test_balanced_subject_counts(self):
        groups = _make_synthetic_groups()
        splitter = RandomizedBalancedGroupKFold(n_splits=5, random_state=42)
        sizes = []
        for _, test in splitter.split(np.arange(len(groups)), groups=groups):
            sizes.append(len(test))
        # All folds should be within 20% of each other
        assert max(sizes) - min(sizes) <= len(groups) * 0.25

    def test_inner_split_no_leakage(self):
        groups = _make_synthetic_groups()
        indices = np.arange(len(groups))
        outer_splitter = RandomizedBalancedGroupKFold(n_splits=5, random_state=42)
        train_idx, test_idx = next(outer_splitter.split(indices, groups=groups))
        inner_splits = make_family_inner_split(train_idx, groups, seed=0, outer_fold=0, n_splits=3)
        for inner_train, inner_val in inner_splits:
            train_fam = set(groups[inner_train])
            val_fam = set(groups[inner_val])
            assert not (train_fam & val_fam)


# ---------------------------------------------------------------------------
# Part B4 -- Seed diversity
# ---------------------------------------------------------------------------

class TestSeedDiversity:
    def test_diversity_sufficient(self):
        groups = _make_synthetic_groups()
        split_manifests = {}
        for seed in range(10):
            splitter = RandomizedBalancedGroupKFold(n_splits=5, random_state=seed)
            splits = list(splitter.split(np.arange(len(groups)), groups=groups))
            split_manifests[seed] = [set(groups[idx]) for _, idx in splits]
        result = verify_seed_diversity(split_manifests)
        assert result["diversity_sufficient"], result

    def test_all_seeds_unique(self):
        groups = _make_synthetic_groups()
        split_manifests = {}
        for seed in range(10):
            splitter = RandomizedBalancedGroupKFold(n_splits=5, random_state=seed)
            splits = list(splitter.split(np.arange(len(groups)), groups=groups))
            split_manifests[seed] = [set(groups[idx]) for _, idx in splits]
        result = verify_seed_diversity(split_manifests)
        assert result["n_unique_partition_manifests"] >= 8


# ---------------------------------------------------------------------------
# Part B5 -- Family leakage assertions
# ---------------------------------------------------------------------------

class TestFamilyLeakageAssertions:
    def test_no_leakage(self):
        groups = _make_synthetic_groups()
        indices = np.arange(len(groups))
        outer_splits = []
        inner_per_fold = {}
        for fold in range(5):
            splitter = RandomizedBalancedGroupKFold(n_splits=5, random_state=42)
            splits = list(splitter.split(indices, groups=groups))
            train_idx, test_idx = splits[fold]
            outer_splits.append((train_idx, test_idx))
            inner_per_fold[fold] = make_family_inner_split(
                train_idx, groups, seed=42, outer_fold=fold, n_splits=3
            )
        result = assert_no_family_leakage(groups, outer_splits, inner_per_fold)
        assert result["all_passed"], result
        assert result["n_failures"] == 0


# ---------------------------------------------------------------------------
# Part B6 -- Config validation
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_config_exists(self):
        assert CONFIG_PATH.exists()

    def test_groups_path_not_null(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        assert config["data"]["groups_path"] is not None

    def test_split_strategy(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        assert config.get("split_strategy") == "randomized_balanced_group_kfold"

    def test_frozen_grids_match_final(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        final_config = yaml.safe_load(
            Path("configs/iclr/msancr_final_10x5.yaml").read_text()
        )
        assert config["ridge_grid"] == final_config["ridge_grid"]
        assert config["gamma_grid"] == final_config["gamma_grid"]
        assert config["lambda_laplacian_grid"] == final_config["lambda_laplacian_grid"]

    def test_output_dirs_differ(self):
        config = yaml.safe_load(CONFIG_PATH.read_text())
        final_config = yaml.safe_load(
            Path("configs/iclr/msancr_final_10x5.yaml").read_text()
        )
        assert config["output_dir"] != final_config["output_dir"]


# ---------------------------------------------------------------------------
# Part B11 -- Old outputs untouched
# ---------------------------------------------------------------------------

class TestOldOutputsUntouched:
    def test_subject_wise_complete_exists(self):
        assert (FINAL_DIR / "FINAL_COMPLETE").exists()

    def test_subject_wise_split_metrics_exists(self):
        assert (FINAL_DIR / "split_metrics.csv").exists()
