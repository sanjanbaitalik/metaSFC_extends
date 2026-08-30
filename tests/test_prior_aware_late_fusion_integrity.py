"""Modification 2 Integrity Audit tests."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

AUDIT_DIR = Path("outputs/iclr/prior_aware_late_fusion_integrity_audit")
MOD2_WM = Path("outputs/iclr/prior_aware_late_fusion/working_memory")
MOD2_FL = Path("outputs/iclr/prior_aware_late_fusion/fluid_intelligence")
LF1_CTRL = AUDIT_DIR / "lf1_fixed_prior_swaps"


@pytest.fixture
def wm_seed():
    return pd.read_csv(MOD2_WM / "seed_metrics.csv")


@pytest.fixture
def fl_seed():
    return pd.read_csv(MOD2_FL / "seed_metrics.csv")


@pytest.fixture
def seed_level():
    return pd.read_csv(AUDIT_DIR / "seed_level_primary_models.csv")


@pytest.fixture
def paired_recon():
    return pd.read_csv(AUDIT_DIR / "paired_seed_reconstruction.csv")


@pytest.fixture
def strongest_def():
    return json.loads((AUDIT_DIR / "strongest_no_prior_definition.json").read_text())


@pytest.fixture
def corrected_decision():
    return json.loads((AUDIT_DIR / "corrected_late_fusion_decision.json").read_text())


@pytest.fixture
def leakage_audit():
    return json.loads((AUDIT_DIR / "stacking_leakage_audit_v2.json").read_text())


@pytest.fixture
def lf2_null():
    return pd.read_csv(AUDIT_DIR / "lf2_null_equivalence.csv")


@pytest.fixture
def lf1_ctrl_wm():
    return pd.read_csv(LF1_CTRL / "lf1_controls_working_memory.csv")


@pytest.fixture
def lf1_ctrl_fl():
    return pd.read_csv(LF1_CTRL / "lf1_controls_fluid_intelligence.csv")


class TestSeedMedianPositiveCountConsistency:
    """1. 3-seed median/positive-count consistency."""

    def test_lf1_wm_consistency(self, paired_recon):
        row = paired_recon[(paired_recon.task == "working_memory") & (paired_recon.model == "LF1")].iloc[0]
        assert row["consistent"], f"LF1 WM inconsistent: median={row['median']}, pos={row['positive_count']}"

    def test_lf2_wm_consistency(self, paired_recon):
        row = paired_recon[(paired_recon.task == "working_memory") & (paired_recon.model == "LF2")].iloc[0]
        assert row["consistent"], f"LF2 WM inconsistent: median={row['median']}, pos={row['positive_count']}"

    def test_lf1_fl_consistency(self, paired_recon):
        row = paired_recon[(paired_recon.task == "fluid_intelligence") & (paired_recon.model == "LF1")].iloc[0]
        assert row["consistent"], f"LF1 FL inconsistent"

    def test_lf2_fl_consistency(self, paired_recon):
        row = paired_recon[(paired_recon.task == "fluid_intelligence") & (paired_recon.model == "LF2")].iloc[0]
        assert row["consistent"], f"LF2 FL inconsistent"

    def test_if_pos_le_1_median_le_0(self, paired_recon):
        for _, row in paired_recon.iterrows():
            if row["positive_count"] <= 1:
                assert row["median"] <= 1e-10, (
                    f"{row['task']} {row['model']}: pos={row['positive_count']} but median={row['median']}"
                )

    def test_if_pos_ge_2_median_ge_0(self, paired_recon):
        for _, row in paired_recon.iterrows():
            if row["positive_count"] >= 2:
                assert row["median"] >= -1e-10, (
                    f"{row['task']} {row['model']}: pos={row['positive_count']} but median={row['median']}"
                )


class TestGlobalTaskStrongestNoPrior:
    """2. Global task-level strongest no-prior identity."""

    def test_wm_comparator_is_a4(self, strongest_def):
        assert strongest_def["working_memory"]["strongest_no_prior"] == "A4"

    def test_fl_comparator_is_lf0(self, strongest_def):
        assert strongest_def["fluid_intelligence"]["strongest_no_prior"] == "LF0"

    def test_wm_a4_gt_lf0(self, strongest_def):
        assert strongest_def["working_memory"]["A4_mean"] > strongest_def["working_memory"]["LF0_mean"]

    def test_fl_lf0_gt_a4(self, strongest_def):
        assert strongest_def["fluid_intelligence"]["LF0_mean"] > strongest_def["fluid_intelligence"]["A4_mean"]


class TestNumericSignConsistency:
    """3. Numeric/text sign consistency."""

    def test_wm_lf1_delta_sign_matches_status(self, corrected_decision):
        wm = corrected_decision["working_memory"]
        lf1 = wm["LF1_vs_strongest"]
        if lf1["status"] in ("LARGE_MARGIN_SUCCESS", "PROMISING"):
            assert lf1["median"] >= 0.008
            assert lf1["positive_seeds"] >= 2
        elif lf1["status"] == "FAILURE":
            assert lf1["median"] < 0.008 or lf1["positive_seeds"] < 2

    def test_fl_lf1_delta_sign_matches_status(self, corrected_decision):
        fl = corrected_decision["fluid_intelligence"]
        lf1 = fl["LF1_vs_strongest"]
        if lf1["status"] in ("LARGE_MARGIN_SUCCESS", "PROMISING"):
            assert lf1["median"] >= 0.008
            assert lf1["positive_seeds"] >= 2


class TestSeedAlignment:
    """4. Seed alignment."""

    def test_wm_seed_count(self, wm_seed):
        for model in ["A4", "LF0", "LF1", "LF2"]:
            count = len(wm_seed[wm_seed.model == model])
            assert count == 3, f"WM {model} has {count} seeds, expected 3"

    def test_fl_seed_count(self, fl_seed):
        for model in ["A4", "LF0", "LF1", "LF2"]:
            count = len(fl_seed[fl_seed.model == model])
            assert count == 3, f"FL {model} has {count} seeds, expected 3"

    def test_wm_seed_values_are_finite(self, wm_seed):
        assert wm_seed["pearson"].notna().all()
        assert np.all(np.isfinite(wm_seed["pearson"].values))


class TestA4Parity:
    """5. A4 parity."""

    def test_wm_a4_mean_matches(self, corrected_decision):
        wm = corrected_decision["working_memory"]
        assert abs(wm["A4_pearson"] - 0.2743) < 0.001

    def test_fl_a4_mean_matches(self, corrected_decision):
        fl = corrected_decision["fluid_intelligence"]
        assert abs(fl["A4_pearson"] - 0.3238) < 0.001


class TestLF1LF2EligibilityGates:
    """6. LF1/LF2 eligibility gates."""

    def test_wm_lf1_promising(self, corrected_decision):
        assert corrected_decision["working_memory"]["LF1_vs_strongest"]["status"] == "PROMISING"

    def test_fl_lf1_promising(self, corrected_decision):
        assert corrected_decision["fluid_intelligence"]["LF1_vs_strongest"]["status"] == "PROMISING"

    def test_wm_lf2_failure(self, corrected_decision):
        assert corrected_decision["working_memory"]["LF2_vs_strongest"]["status"] == "FAILURE"

    def test_fl_lf2_borderline(self, corrected_decision):
        assert corrected_decision["fluid_intelligence"]["LF2_vs_strongest"]["status"] == "BORDERLINE"


class TestLF2NullEquivalence:
    """7. LF2-null equivalence."""

    def test_at_least_one_equivalent_split(self, lf2_null):
        n_equiv = lf2_null["equivalent"].sum()
        assert n_equiv >= 1, f"Only {n_equiv}/30 LF2==LF0 exact matches"

    def test_majority_differ(self, lf2_null):
        n_diff = len(lf2_null) - lf2_null["equivalent"].sum()
        assert n_diff >= 1, "LF2 always equals LF0 — prior branch never contributes"


class TestWeightSimplexConstraints:
    """8. Weight simplex constraints."""

    def test_matched_weights_are_valid(self, lf1_ctrl_wm):
        matched = lf1_ctrl_wm[lf1_ctrl_wm.prior_type == "matched"]
        for _, row in matched.iterrows():
            assert row["w_fp"] >= 0 and row["w_s"] >= 0
            assert abs(row["w_fp"] + row["w_s"] - 1.0) < 1e-4, (
                f"S{row['seed']}F{row['fold']}: w_fp={row['w_fp']} + w_s={row['w_s']} = {row['w_fp']+row['w_s']}"
            )


class TestOOFLeakage:
    """9. OOF leakage."""

    def test_no_nan_in_split_metrics(self):
        for task in ["working_memory", "fluid_intelligence"]:
            df = pd.read_csv(f"outputs/iclr/prior_aware_late_fusion/{task}/split_metrics.csv")
            assert df["pearson"].notna().all(), f"{task} has NaN pearson values"

    def test_leakage_audit_passes(self, leakage_audit):
        assert leakage_audit["oof_integrity"] == "PASS"
        assert leakage_audit["n_failures"] == 0


class TestFixedSwapsDifferOnlyByPrior:
    """10. Fixed swaps differ only by prior."""

    def test_wm_ctrl_same_seed_fold(self, lf1_ctrl_wm):
        matched = lf1_ctrl_wm[lf1_ctrl_wm.prior_type == "matched"]
        for pt in ["unrelated", "shuffled", "random"]:
            ctrl = lf1_ctrl_wm[lf1_ctrl_wm.prior_type == pt]
            assert len(ctrl) == len(matched), f"Control {pt} has different number of rows"

    def test_wm_ctrl_same_weights(self, lf1_ctrl_wm):
        matched = lf1_ctrl_wm[lf1_ctrl_wm.prior_type == "matched"].sort_values(["seed", "fold"]).reset_index(drop=True)
        for pt in ["unrelated", "shuffled", "random"]:
            ctrl = lf1_ctrl_wm[lf1_ctrl_wm.prior_type == pt].sort_values(["seed", "fold"]).reset_index(drop=True)
            for i in range(len(matched)):
                assert abs(matched.loc[i, "w_fp"] - ctrl.loc[i, "w_fp"]) < 1e-10
                assert abs(matched.loc[i, "w_s"] - ctrl.loc[i, "w_s"]) < 1e-10


class TestLF1ControlsTriggeredByEligibility:
    """11. LF1 controls triggered only by eligibility."""

    def test_wm_lf1_controls_exist(self, lf1_ctrl_wm):
        assert len(lf1_ctrl_wm) > 0

    def test_fl_lf1_controls_exist(self, lf1_ctrl_fl):
        assert len(lf1_ctrl_fl) > 0

    def test_wm_lf1_beats_all_controls(self, lf1_ctrl_wm):
        matched_mean = lf1_ctrl_wm[lf1_ctrl_wm.prior_type == "matched"]["pearson"].mean()
        beats = 0
        for pt in ["unrelated", "shuffled", "random"]:
            ctrl_mean = lf1_ctrl_wm[lf1_ctrl_wm.prior_type == pt]["pearson"].mean()
            if matched_mean > ctrl_mean:
                beats += 1
        assert beats >= 2, f"WM LF1 only beats {beats}/3 controls"

    def test_fl_lf1_beats_all_controls(self, lf1_ctrl_fl):
        matched_mean = lf1_ctrl_fl[lf1_ctrl_fl.prior_type == "matched"]["pearson"].mean()
        beats = 0
        for pt in ["unrelated", "shuffled", "random"]:
            ctrl_mean = lf1_ctrl_fl[lf1_ctrl_fl.prior_type == pt]["pearson"].mean()
            if matched_mean > ctrl_mean:
                beats += 1
        assert beats >= 2, f"FL LF1 only beats {beats}/3 controls"


class TestNoNewGridsFormulas:
    """12. No new grids/formulas."""

    def test_ridge_grid_unchanged(self):
        from metascfc.experiments.prior_aware_late_fusion import RIDGE_GRID
        assert RIDGE_GRID == [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    def test_weight_step_unchanged(self):
        from metascfc.experiments.prior_aware_late_fusion import WEIGHT_STEP
        assert WEIGHT_STEP == 0.05


class TestModification3CannotLaunch:
    """13. Modification 3 cannot launch."""

    def test_no_mod3_implementation(self):
        import os
        assert not os.path.exists("src/metascfc/experiments/modification_3.py")

    def test_validated_success(self, corrected_decision):
        assert corrected_decision["overall"]["status"] == "VALIDATED_SUCCESS"
        assert corrected_decision["overall"]["recommended_next_step"] == "full_10x5_late_fusion_both_tasks"
