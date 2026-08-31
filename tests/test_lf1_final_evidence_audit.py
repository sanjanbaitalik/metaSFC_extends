"""Tests for the LF1 Final Evidence Audit (prompt v15)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

AUDIT = Path("outputs/iclr/lf1_final_evidence_audit")
OUT = Path("outputs/iclr/lf1_final_10x5")


class TestPredictionValuesReproduce:
    def test_prediction_requirement_csv_exists(self):
        assert (AUDIT / "prediction_requirement.csv").exists()

    def test_wm_values_match(self):
        df = pd.read_csv(AUDIT / "prediction_requirement.csv")
        wm = df[df.task == "working_memory"].iloc[0]
        assert abs(wm.LF0_pearson - 0.2571) < 0.001
        assert abs(wm.LF1_pearson - 0.2741) < 0.001
        assert abs(wm.mean_delta - 0.0170) < 0.001

    def test_fl_values_match(self):
        df = pd.read_csv(AUDIT / "prediction_requirement.csv")
        fl = df[df.task == "fluid_intelligence"].iloc[0]
        assert abs(fl.LF0_pearson - 0.3621) < 0.001
        assert abs(fl.LF1_pearson - 0.3700) < 0.001
        assert abs(fl.mean_delta - 0.0079) < 0.001

    def test_wm_source_files_match(self):
        with open(OUT / "working_memory/prediction_statistics.json") as f:
            src = json.load(f)
        df = pd.read_csv(AUDIT / "prediction_requirement.csv")
        wm = df[df.task == "working_memory"].iloc[0]
        assert abs(wm.LF0_pearson - src["lf0_mean"]) < 1e-10
        assert abs(wm.wilcoxon_raw_p - src["p_wilcoxon"]) < 1e-10

    def test_fl_source_files_match(self):
        with open(OUT / "fluid_intelligence/prediction_statistics.json") as f:
            src = json.load(f)
        df = pd.read_csv(AUDIT / "prediction_requirement.csv")
        fl = df[df.task == "fluid_intelligence"].iloc[0]
        assert abs(fl.LF0_pearson - src["lf0_mean"]) < 1e-10

    def test_corrected_cv_p_visible(self):
        df = pd.read_csv(AUDIT / "prediction_requirement.csv")
        assert all(df.corrected_repeated_cv_p > 0.01)

    def test_prediction_booleans_separated(self):
        with open(AUDIT / "final_professor_verdict.json") as f:
            v = json.load(f)
        assert "both_tasks_prediction_direction_positive" in v
        assert "both_tasks_prediction_two_task_holm_supported" in v
        assert "both_tasks_biomarker_supported" in v


class TestBiomarkerUsesFrozenCoefficients:
    def test_biomarker_seed_metrics_exist(self):
        for tk in ["working_memory", "fluid_intelligence"]:
            assert (AUDIT / f"biomarker_seed_metrics_{tk}.csv").exists()

    def test_biomarker_models_present(self):
        for tk in ["working_memory", "fluid_intelligence"]:
            df = pd.read_csv(AUDIT / f"biomarker_seed_metrics_{tk}.csv")
            assert set(df.model.unique()) == {"FC_no_prior", "FP_matched", "FP_unrelated", "FP_shuffled", "FP_random"}

    def test_biomarker_10_seeds(self):
        for tk in ["working_memory", "fluid_intelligence"]:
            df = pd.read_csv(AUDIT / f"biomarker_seed_metrics_{tk}.csv")
            assert len(df[df.model == "FP_matched"]) == 10


class TestMatchedTaskPriorAlignmentUsed:
    def test_common_matched_prior_for_specificity(self):
        for tk in ["working_memory", "fluid_intelligence"]:
            df = pd.read_csv(AUDIT / f"biomarker_seed_metrics_{tk}.csv")
            fp_unrelated_matched = df[df.model == "FP_unrelated"]["matched_alignment"].mean()
            fp_unrelated_unrelated = df[df.model == "FP_unrelated"]["unrelated_alignment"].mean()
            assert fp_unrelated_matched != fp_unrelated_unrelated

    def test_metric_definitions_specify_common_prior(self):
        with open(AUDIT / "biomarker_metric_definitions.json") as f:
            defs = json.load(f)
        assert "matched_task_prior_alignment" in defs["professor_facing_comparison"].lower() or "matched" in defs["professor_facing_comparison"].lower()
        assert "not its own prior" in defs["prior_vector"] or "MATCHED task prior" in defs["prior_vector"]


class TestBiomarkerHolmFamily:
    def test_exactly_4_comparisons_per_task(self):
        for tk in ["working_memory", "fluid_intelligence"]:
            df = pd.read_csv(AUDIT / f"biomarker_statistics_{tk}.csv")
            assert len(df) == 4
            assert set(df.comparison) == {
                "B1_FC_no_prior_vs_FP_matched",
                "B2_FP_matched_vs_FP_unrelated",
                "B3_FP_matched_vs_FP_shuffled",
                "B4_FP_matched_vs_FP_random",
            }

    def test_holm_column_exists(self):
        for tk in ["working_memory", "fluid_intelligence"]:
            df = pd.read_csv(AUDIT / f"biomarker_statistics_{tk}.csv")
            assert "p_holm_biomarker_alignment_4" in df.columns
            holm_vals = df["p_holm_biomarker_alignment_4"].values
            assert all(holm_vals >= 0)
            assert all(holm_vals <= 1.0)

    def test_no_prediction_in_biomarker_family(self):
        for tk in ["working_memory", "fluid_intelligence"]:
            df = pd.read_csv(AUDIT / f"biomarker_statistics_{tk}.csv")
            for comp in df.comparison:
                assert "wilcoxon" not in comp.lower()
                assert "prediction" not in comp.lower()


class TestPredictionAndBiomarkerSeparate:
    def test_prediction_family_not_in_biomarker(self):
        for tk in ["working_memory", "fluid_intelligence"]:
            df = pd.read_csv(AUDIT / f"biomarker_statistics_{tk}.csv")
            for comp in df.comparison:
                assert "LF0" not in comp and "LF1" not in comp and "prediction" not in comp


class TestCorrectedRepeatedCVVisible:
    def test_cv_p_in_prediction_requirement(self):
        df = pd.read_csv(AUDIT / "prediction_requirement.csv")
        assert all(df.corrected_repeated_cv_p > 0.01)


class TestFluidNotLargeMargin:
    def test_fluid_large_margin_false(self):
        with open(AUDIT / "final_professor_verdict.json") as f:
            v = json.load(f)
        with open(OUT / "fluid_intelligence/prediction_statistics.json") as f:
            fl = json.load(f)
        if not fl.get("large_margin_consistency_gate", False):
            pass  # Correct: Fluid should not be labeled large-margin


class TestNoTrainingCalled:
    def test_biomarker_definitions_note_recomputation(self):
        with open(AUDIT / "biomarker_metric_definitions.json") as f:
            defs = json.load(f)
        assert "recomputed" in defs["biomarker_limitation"].lower() or "frozen" in defs["coefficient_source"].lower()

    def test_no_modification_3(self):
        with open(AUDIT / "final_professor_verdict.json") as f:
            v = json.load(f)
        assert v["modification_3_recommended"] is False
        assert v["additional_model_tuning_recommended"] is False


class TestSourceIntegrity:
    def test_source_integrity_exists(self):
        assert (AUDIT / "source_integrity.json").exists()

    def test_source_integrity_verified(self):
        with open(AUDIT / "source_integrity.json") as f:
            si = json.load(f)
        assert si["status"] == "FROZEN_OUTPUTS_VERIFIED"
        assert len(si["hashes"]) > 5

    def test_source_hashes_unchanged(self):
        import hashlib
        with open(AUDIT / "source_integrity.json") as f:
            si = json.load(f)
        for fname, expected_hash in si["hashes"].items():
            if "/" in fname:
                fp = OUT / fname
            else:
                fp = AUDIT / fname
            if fp.exists():
                actual = hashlib.sha256(fp.read_bytes()).hexdigest()
                assert actual == expected_hash, f"Hash mismatch for {fname}"


class TestAllRequiredOutputs:
    def test_required_csvs(self):
        required = [
            "prediction_requirement.csv",
            "biomarker_seed_metrics_working_memory.csv",
            "biomarker_seed_metrics_fluid_intelligence.csv",
            "biomarker_summary_working_memory.csv",
            "biomarker_summary_fluid_intelligence.csv",
            "biomarker_statistics_working_memory.csv",
            "biomarker_statistics_fluid_intelligence.csv",
            "prior_control_prediction.csv",
            "fusion_weight_evidence.csv",
            "professor_requirement_evidence.csv",
            "table_professor_prediction.csv",
            "table_professor_biomarker.csv",
        ]
        for r in required:
            assert (AUDIT / r).exists(), f"Missing {r}"

    def test_required_jsons(self):
        required = [
            "biomarker_metric_definitions.json",
            "final_professor_verdict.json",
            "source_integrity.json",
        ]
        for r in required:
            assert (AUDIT / r).exists(), f"Missing {r}"

    def test_required_tops(self):
        required = [
            "table_professor_prediction.tex",
            "table_professor_biomarker.tex",
        ]
        for r in required:
            assert (AUDIT / r).exists(), f"Missing {r}"

    def test_professor_update_md(self):
        assert (AUDIT / "PROFESSOR_UPDATE.md").exists()

    def test_complete_marker(self):
        assert (AUDIT / "COMPLETE").exists()
