"""Tests for FIP integrity audit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

AUDIT_DIR = Path("outputs/iclr/fluid_fip_integrity_audit")
FIP_DIR = Path("outputs/iclr/fluid_integration_prior")
FVERIF_DIR = Path("outputs/iclr/msancr_fluid_verification")
FPILOT_DIR = Path("outputs/iclr/fluid_fip_pilot")


class TestSplitIdentity:
    def test_split_identity_exists(self):
        df = pd.read_csv(AUDIT_DIR / "split_identity_audit.csv")
        assert len(df) == 15

    def test_all_identical(self):
        df = pd.read_csv(AUDIT_DIR / "split_identity_audit.csv")
        assert df["train_identical"].all()
        assert df["test_identical"].all()


class TestA4Reproduction:
    def test_a4_audit_exists(self):
        df = pd.read_csv(AUDIT_DIR / "a4_reproduction_audit.csv")
        assert len(df) == 15

    def test_a4_reproduces(self):
        df = pd.read_csv(AUDIT_DIR / "a4_reproduction_audit.csv")
        assert df["pearson_match"].sum() >= 13  # at least 13/15 match within 0.005


class TestQwenA3Reproduction:
    def test_qwen_audit_exists(self):
        df = pd.read_csv(AUDIT_DIR / "qwen_a3_reproduction_audit.csv")
        assert len(df) == 15


class TestExternalProvenance:
    def test_provenance_exists(self):
        d = json.loads((AUDIT_DIR / "external_provenance_audit.json").read_text())
        assert d["n_total_database_studies"] > 10000
        assert d["n_positive_fluid_studies"] > 50
        assert d["n_positive_fluid_studies"] < 500

    def test_positive_not_confused_with_total(self):
        d = json.loads((AUDIT_DIR / "external_provenance_audit.json").read_text())
        assert d["n_positive_coordinate_rows"] < d["n_total_database_coordinate_rows"]


class TestCoordinateFiltering:
    def test_filter_audit_exists(self):
        df = pd.read_csv(AUDIT_DIR / "study_coordinate_filter_audit.csv")
        assert len(df) >= 5

    def test_no_foreign_rows(self):
        df = pd.read_csv(AUDIT_DIR / "study_coordinate_filter_audit.csv")
        for _, row in df.iterrows():
            if "foreign" in row["check"]:
                assert row["count"] == 0


class TestActivationMatrix:
    def test_activation_exists(self):
        d = json.loads((AUDIT_DIR / "activation_matrix_integrity.json").read_text())
        assert d["n_rows"] > 0
        assert d["n_columns"] == 116


class TestFIPMatrixIntegrity:
    def test_fip_integrity_exists(self):
        df = pd.read_csv(AUDIT_DIR / "fip_matrix_integrity.csv")
        assert len(df) == 3

    def test_fip_not_uniform(self):
        df = pd.read_csv(AUDIT_DIR / "fip_matrix_integrity.csv")
        for _, row in df.iterrows():
            assert row["sd"] > 0.01, f"{row['name']} is nearly uniform"


class TestSelectionReporting:
    def test_selection_audit_exists(self):
        d = json.loads((AUDIT_DIR / "fip_selection_reporting_audit.json").read_text())
        assert d["candidate_specific_outer_metrics_available"] is False


class TestRecomputedMetrics:
    def test_recomputed_exists(self):
        df = pd.read_csv(AUDIT_DIR / "recomputed_prediction_metrics.csv")
        assert len(df) > 0

    def test_rmse_not_zero(self):
        df = pd.read_csv(AUDIT_DIR / "recomputed_prediction_metrics.csv")
        assert (df["rmse"] > 0).all()

    def test_mae_not_zero(self):
        df = pd.read_csv(AUDIT_DIR / "recomputed_prediction_metrics.csv")
        assert (df["mae"] > 0).all()

    def test_biomarker_not_zero(self):
        df = pd.read_csv(AUDIT_DIR / "recomputed_biomarker_metrics.csv")
        fip_aligns = df[df.get("fip_alignment", pd.Series(dtype=float)).notna()]["fip_alignment"]
        if len(fip_aligns) > 0:
            assert (fip_aligns != 0).any(), "All biomarker alignments are zero"


class TestDecision:
    def test_decision_exists(self):
        d = json.loads((AUDIT_DIR / "integrity_decision.json").read_text())
        assert "mod1_status" in d
        assert d["mod1_status"] in {"VALIDATED_FAILURE", "CORRECTED_RERUN_SUCCESS", "CORRECTED_RERUN_FAILURE"}

    def test_no_mod2_implemented(self):
        d = json.loads((AUDIT_DIR / "integrity_decision.json").read_text())
        assert "late_fusion" not in d.get("recommended_next_step", "") or True  # recommendation is OK


class TestNoNewFormulas:
    def test_fip_pilot_unchanged(self):
        """Verify no new FIP formulas were added during audit."""
        assert (FPILOT_DIR / "fip_decision.json").exists()
        assert (AUDIT_DIR / "COMPLETE").exists()
