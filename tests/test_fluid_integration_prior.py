"""Tests for Fluid Integration Prior (FIP) construction and evaluation."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from metascfc.experiments.fluid_integration_prior import (
    build_fip_line_graph_laplacian,
    build_fip_msancr_cache,
    build_study_roi_activation,
    compute_coactivation_statistics,
    compute_prior_similarity,
    construct_fip1_mac,
    construct_fip2_bridge,
    construct_fip3_weaktie,
    create_shuffled_fip,
    deterministic_background_sample,
    fip_edges_to_matrix,
    fip_matrix_to_edges,
    load_neurosynth_data,
    identify_fluid_studies,
    validate_fip_matrix,
)
from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    build_msancr_cache,
    lift_roi_to_edge,
)

N_ROIS = 116
N_EDGES = N_ROIS * (N_ROIS - 1) // 2
FIP_DIR = Path("outputs/iclr/fluid_integration_prior")
FIP_PILOT_DIR = Path("outputs/iclr/fluid_fip_pilot")


# --- Part 5: Pre-implementation audit ---

class TestPreimplementationAudit:
    def test_audit_file_exists(self):
        p = FIP_DIR / "preimplementation_audit.json"
        assert p.exists(), f"Missing {p}"
        audit = json.loads(p.read_text())
        assert audit["hcp_data_loaded"] is False
        assert audit["n_rois"] == 116

    def test_complete_prior_marker(self):
        assert (FIP_DIR / "COMPLETE_PRIOR").exists()


# --- Part 6: Term availability ---

class TestTermAvailability:
    def test_term_availability_exists(self):
        df = pd.read_csv(FIP_DIR / "term_availability.csv")
        assert "requested_term" in df.columns
        assert "available" in df.columns
        assert len(df) >= 5

    def test_no_fabricated_terms(self):
        df = pd.read_csv(FIP_DIR / "term_availability.csv")
        for _, row in df.iterrows():
            if row["available"] is False or str(row["available"]).lower() == "false":
                assert row["matched_feature"] == "" or pd.isna(row["matched_feature"])


# --- Part 7: Positive/background study sets ---

class TestStudySets:
    def test_study_selection_exists(self):
        df = pd.read_csv(FIP_DIR / "study_selection.csv")
        assert "set" in df.columns
        assert "study_id" in df.columns

    def test_sets_disjoint(self):
        df = pd.read_csv(FIP_DIR / "study_selection.csv")
        fluid_ids = set(df[df["set"] == "fluid"]["study_id"])
        bg_ids = set(df[df["set"] == "background"]["study_id"])
        assert len(fluid_ids & bg_ids) == 0, "Sets overlap!"

    def test_positive_and_background_nonempty(self):
        df = pd.read_csv(FIP_DIR / "study_selection.csv")
        assert (df["set"] == "fluid").sum() > 0
        assert (df["set"] == "background").sum() > 0

    def test_background_deterministic(self):
        bg = pd.DataFrame({"id": range(100)})
        s1 = deterministic_background_sample(bg, 20, n_repeats=3, base_seed=42)
        s2 = deterministic_background_sample(bg, 20, n_repeats=3, base_seed=42)
        for a, b in zip(s1, s2):
            assert np.array_equal(a["id"].values, b["id"].values)


# --- Part 8: Study-by-ROI activation ---

class TestActivationMatrix:
    def test_activation_matrix_exists(self):
        data = np.load(FIP_DIR / "study_roi_activation_matrix.npz")
        assert "positive" in data

    def test_activation_binary(self):
        data = np.load(FIP_DIR / "study_roi_activation_matrix.npz")
        pos = data["positive"]
        assert pos.shape[1] == N_ROIS
        assert set(np.unique(pos)).issubset({0.0, 1.0})

    def test_metadata_exists(self):
        meta = json.loads((FIP_DIR / "study_roi_activation_metadata.json").read_text())
        assert meta["n_rois"] == N_ROIS


# --- Part 9: Edge coactivation ---

class TestEdgeCoactivation:
    def test_coactivation_file_exists(self):
        df = pd.read_csv(FIP_DIR / "edge_coactivation_statistics.csv")
        assert len(df) == N_EDGES
        assert "lor_mean" in df.columns
        assert "delta_pmi" in df.columns

    def test_lor_finite(self):
        df = pd.read_csv(FIP_DIR / "edge_coactivation_statistics.csv")
        assert np.isfinite(df["lor_mean"]).all()

    def test_delta_pmi_finite(self):
        df = pd.read_csv(FIP_DIR / "edge_coactivation_statistics.csv")
        assert np.isfinite(df["delta_pmi"]).all()

    def test_lor_haldane_correction(self):
        rng = np.random.default_rng(42)
        pos = (rng.random((50, 10)) > 0.5).astype(float)
        bg = [(rng.random((50, 10)) > 0.5).astype(float)]
        result = compute_coactivation_statistics(pos, bg, n_rois=10)
        assert result["lor_mean"].shape == (45,)  # C(10,2)
        assert np.isfinite(result["lor_mean"]).all()


# --- Part 10-12: FIP matrix construction and validation ---

class TestFIPConstruction:
    def test_fip1_exists(self):
        mat = np.load(FIP_DIR / "fip1_mac_matrix.npy")
        val = validate_fip_matrix(mat, "FIP-1")
        assert val["valid"], val["issues"]

    def test_fip2_exists(self):
        mat = np.load(FIP_DIR / "fip2_bridge_matrix.npy")
        val = validate_fip_matrix(mat, "FIP-2")
        assert val["valid"], val["issues"]

    def test_fip3_exists(self):
        mat = np.load(FIP_DIR / "fip3_weaktie_matrix.npy")
        val = validate_fip_matrix(mat, "FIP-3")
        assert val["valid"], val["issues"]

    def test_fip1_shape(self):
        mat = np.load(FIP_DIR / "fip1_mac_matrix.npy")
        assert mat.shape == (N_ROIS, N_ROIS)

    def test_fip1_symmetric(self):
        mat = np.load(FIP_DIR / "fip1_mac_matrix.npy")
        assert np.allclose(mat, mat.T)

    def test_fip1_zero_diagonal(self):
        mat = np.load(FIP_DIR / "fip1_mac_matrix.npy")
        assert np.allclose(np.diag(mat), 0)

    def test_fip1_finite_bounded(self):
        mat = np.load(FIP_DIR / "fip1_mac_matrix.npy")
        assert np.isfinite(mat).all()
        assert mat.min() >= 0
        assert mat.max() <= 1

    def test_fip_edges_vectorization(self):
        mat = np.load(FIP_DIR / "fip1_mac_matrix.npy")
        edges = fip_matrix_to_edges(mat)
        mat2 = fip_edges_to_matrix(edges)
        assert np.allclose(mat, mat2)

    def test_fip_edges_csv_exists(self):
        df = pd.read_csv(FIP_DIR / "fip1_mac_edges.csv")
        assert len(df) == N_EDGES
        score_col = [c for c in df.columns if "score" in c][0]
        assert np.isfinite(df[score_col]).all()

    def test_edge_ordering_matches_regression(self):
        """FIP edges must be upper-triangle row-major, matching FC features."""
        df = pd.read_csv(FIP_DIR / "fip1_mac_edges.csv")
        assert np.array_equal(df["roi_i"].values, np.triu_indices(N_ROIS, k=1)[0])
        assert np.array_equal(df["roi_j"].values, np.triu_indices(N_ROIS, k=1)[1])


# --- Part 14: prior_space=edge bypasses node lifting ---

class TestPriorSpaceEdge:
    def test_edge_direct_bypasses_lifting(self):
        """FIP edges vector is used directly; no prod/mean lifting."""
        rng = np.random.default_rng(42)
        edge_prior = rng.uniform(0, 1, N_EDGES)
        cache = build_msancr_cache(
            edge_prior, N_ROIS, gamma=0.5, lifting="edge_direct",
            prior_space="edge", top_k=30,
        )
        assert cache.lifting == "edge_direct"
        assert cache.n_edges == N_EDGES

    def test_node_space_unchanged(self):
        rng = np.random.default_rng(42)
        roi_prior = rng.uniform(0, 1, N_ROIS)
        cache = build_msancr_cache(
            roi_prior, N_ROIS, gamma=0.5, lifting="mean",
            prior_space="node", top_k=30,
        )
        assert cache.lifting == "mean"

    def test_edge_cache_diagonal_valid(self):
        rng = np.random.default_rng(42)
        edge_prior = rng.uniform(0, 1, N_EDGES)
        cache = build_msancr_cache(
            edge_prior, N_ROIS, gamma=0.5, lifting="edge_direct",
            prior_space="edge", top_k=30,
        )
        assert np.isfinite(cache.D).all()
        assert np.all(cache.D > 0)


# --- Part 15: FIP line-graph Laplacian ---

class TestFIPLaplacian:
    def test_laplacian_symmetric_psd(self):
        rng = np.random.default_rng(42)
        edges = rng.uniform(0, 1, N_EDGES)
        el = build_fip_line_graph_laplacian(edges, N_ROIS, top_k=30)
        L = el.active_laplacian
        assert np.allclose(L, L.T, atol=1e-10), "Not symmetric"
        # PSD: all eigenvalues >= 0
        eigvals = np.linalg.eigvalsh(L)
        assert np.all(eigvals >= -1e-10), f"Not PSD: min eigenvalue = {eigvals.min()}"

    def test_laplacian_dimensions(self):
        rng = np.random.default_rng(42)
        edges = rng.uniform(0, 1, N_EDGES)
        el = build_fip_line_graph_laplacian(edges, N_ROIS, top_k=30)
        assert el.n_rois == N_ROIS
        assert el.n_edges == N_EDGES
        assert el.active_laplacian.shape == (el.n_active, el.n_active)

    def test_fip_cache_builds(self):
        rng = np.random.default_rng(42)
        edges = rng.uniform(0, 1, N_EDGES)
        cache = build_fip_msancr_cache(edges, N_ROIS, gamma=0.5, top_k=30)
        assert cache.n_active > 0
        assert np.isfinite(cache.D).all()


# --- Part 16-20: Prior similarity and controls ---

class TestPriorSimilarity:
    def test_similarity_file_exists(self):
        df = pd.read_csv(FIP_DIR / "prior_similarity.csv")
        assert "pearson" in df.columns
        assert "spearman" in df.columns

    def test_density_summary_exists(self):
        df = pd.read_csv(FIP_DIR / "prior_density_summary.csv")
        assert "mean" in df.columns
        assert "std" in df.columns

    def test_shuffled_preserves_invariants(self):
        rng = np.random.default_rng(42)
        mat = rng.uniform(0, 1, (N_ROIS, N_ROIS))
        mat = (mat + mat.T) / 2
        np.fill_diagonal(mat, 0)
        shuffled = create_shuffled_fip(mat, seed=0)
        assert shuffled.shape == mat.shape
        assert np.allclose(shuffled, shuffled.T), "Not symmetric"
        assert np.allclose(np.diag(shuffled), 0), "Diagonal not zero"
        assert np.isfinite(shuffled).all()
        # Same weight distribution
        assert np.allclose(np.sort(mat.ravel()), np.sort(shuffled.ravel()))


# --- Part 18-19: Existing outputs unchanged ---

class TestExistingOutputsUnchanged:
    def test_wm_final_outputs_unchanged(self):
        wm_dir = Path("outputs/iclr/msancr_final_10x5")
        assert (wm_dir / "COMPLETE").exists()
        assert (wm_dir / "FINAL_COMPLETE").exists()
        assert (wm_dir / "final_prediction_statistics.csv").exists()

    def test_fluid_verification_outputs_unchanged(self):
        fv_dir = Path("outputs/iclr/msancr_fluid_verification")
        assert (fv_dir / "COMPLETE").exists()
        assert (fv_dir / "fluid_verification_decision.json").exists()


# --- Part 20: FIP pilot outputs ---

class TestFIPPilotOutputs:
    def test_split_metrics(self):
        df = pd.read_csv(FIP_PILOT_DIR / "split_metrics.csv")
        assert "pearson" in df.columns
        assert "model_id" in df.columns
        assert len(df) > 0

    def test_summary_metrics(self):
        df = pd.read_csv(FIP_PILOT_DIR / "summary_metrics.csv")
        assert "pearson_mean" in df.columns
        assert len(df) > 0

    def test_inner_cv_metrics(self):
        df = pd.read_csv(FIP_PILOT_DIR / "inner_cv_metrics.csv")
        assert "selected_fip" in df.columns
        assert len(df) == 15  # 3 seeds × 5 folds

    def test_paired_comparisons(self):
        df = pd.read_csv(FIP_PILOT_DIR / "paired_comparisons.csv")
        assert "comparison" in df.columns
        assert len(df) > 0

    def test_fip_decision_json(self):
        d = json.loads((FIP_PILOT_DIR / "fip_decision.json").read_text())
        assert "recommended_next_step" in d
        assert "A4_pearson" in d
        assert "fip_selected_pearson" in d
        assert "positive_seeds_vs_A4" in d
        assert d["recommended_next_step"] in {
            "full_fluid_fip_10x5", "consider_modification_2", "human_review"
        }

    def test_complete_marker(self):
        assert (FIP_PILOT_DIR / "COMPLETE").exists()

    def test_no_auto_modification_2(self):
        """Decision code cannot automatically launch Modification 2."""
        d = json.loads((FIP_PILOT_DIR / "fip_decision.json").read_text())
        # Should never auto-run the next modification
        assert d["recommended_next_step"] != "auto_run_modification_2"


# --- Part 14 (additional): prior_space=edge is frozen for WM ---

class TestEdgePriorSpaceFrozen:
    def test_msancr_cache_accepts_edge_prior_space(self):
        """Verify prior_space parameter works."""
        rng = np.random.default_rng(42)
        edges = rng.uniform(0, 1, N_EDGES)
        cache = build_fip_msancr_cache(edges, N_ROIS, gamma=0.5, top_k=30)
        assert cache.n_edges == N_EDGES
        assert cache.n_rois == N_ROIS

    def test_node_prior_space_compatible(self):
        """Old node prior_space still works."""
        rng = np.random.default_rng(42)
        roi = rng.uniform(0, 1, N_ROIS)
        cache = build_msancr_cache(roi, N_ROIS, gamma=0.5, lifting="prod", prior_space="node")
        assert cache.n_edges == N_EDGES
