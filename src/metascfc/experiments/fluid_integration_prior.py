"""Fluid Integration Prior (FIP) construction and evaluation.

Modification 1: constructs edge-level priors from external Neurosynth
meta-analytic data, independent of HCP prediction data.

Three FIP candidates:
- FIP-1 (MAC): Meta-Analytic Coactivation Consensus
- FIP-2 (Bridge): Between-Network Bridge Prior (requires network mapping)
- FIP-3 (Weaktie): Distributed Weak/Intermediate Integration (requires network mapping)

Key invariants:
- prior construction NEVER loads HCP data
- positive/background external study sets are disjoint
- deterministic background sampling
- FIP matrices satisfy shape=116x116, symmetric, diagonal=0, finite, [0,1]
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
    compute_diagonal_penalty,
    lift_roi_to_edge,
)
from metascfc.models.iclr_backbones.network_constrained_ridge import (
    build_edge_laplacian,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FLUID_TERMS = [
    "fluid intelligence",
    "reasoning",
    "relational reasoning",
    "abstract reasoning",
    "matrix reasoning",
    "raven",
]

N_REPEATS = 100
EPSILON = 1e-6

# AAL116 coarse module mapping (8 modules)
COARSE_MODULE_CSV = "inputs/atlases/AAL116_coarse_modules.csv"


# ---------------------------------------------------------------------------
# External prior construction (NO HCP data)
# ---------------------------------------------------------------------------

def load_neurosynth_data(
    nimare_dir: str = "~/.nimare/neurosynth",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load Neurosynth coordinates and metadata from NiMARE cache."""
    nimare_path = Path(nimare_dir).expanduser()
    coord_file = nimare_path / "data-neurosynth_version-7_coordinates.tsv.gz"
    meta_file = nimare_path / "data-neurosynth_version-7_metadata.tsv.gz"

    if not coord_file.exists():
        raise FileNotFoundError(f"Neurosynth coordinates not found: {coord_file}")
    if not meta_file.exists():
        raise FileNotFoundError(f"Neurosynth metadata not found: {meta_file}")

    coords = pd.read_csv(coord_file, sep="\t", compression="gzip")
    meta = pd.read_csv(meta_file, sep="\t", compression="gzip")
    return coords, meta


def identify_fluid_studies(
    coords: pd.DataFrame,
    meta: pd.DataFrame,
    fluid_terms: List[str] = FLUID_TERMS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identify fluid intelligence and background cognitive studies.

    Returns (positive_df, background_df) where each has columns
    [id, x, y, z] and the sets are disjoint.
    """
    merged = coords.merge(meta[["id", "title"]], on="id", how="left")

    # Find fluid studies by title search
    fluid_mask = pd.Series(False, index=merged.index)
    for term in fluid_terms:
        fluid_mask |= merged["title"].str.contains(term, case=False, na=False)

    fluid_ids = set(merged.loc[fluid_mask, "id"].unique())

    # Find cognitive (non-fluid) background studies
    cog_terms = [
        "working memory", "attention", "perception", "language",
        "emotion", "motor", "visual", "auditory", "memory", "executive",
    ]
    cog_mask = pd.Series(False, index=merged.index)
    for term in cog_terms:
        cog_mask |= merged["title"].str.contains(term, case=False, na=False)

    cog_ids = set(merged.loc[cog_mask, "id"].unique())
    background_ids = cog_ids - fluid_ids

    positive_df = merged[merged["id"].isin(fluid_ids)][["id", "x", "y", "z"]].copy()
    background_df = merged[merged["id"].isin(background_ids)][["id", "x", "y", "z"]].copy()

    return positive_df, background_df


def deterministic_background_sample(
    background_df: pd.DataFrame,
    n_target: int,
    n_repeats: int = N_REPEATS,
    base_seed: int = 42,
) -> List[pd.DataFrame]:
    """Sample balanced background subsets deterministically.

    Returns list of n_repeats DataFrames, each with n_target studies
    sampled from background (with replacement if n_target > len).
    """
    bg_ids = background_df["id"].unique()
    n_bg = len(bg_ids)
    samples = []
    for r in range(n_repeats):
        rng = np.random.default_rng(base_seed + r)
        if n_bg >= n_target:
            chosen = rng.choice(bg_ids, size=n_target, replace=False)
        else:
            chosen = rng.choice(bg_ids, size=n_target, replace=True)
        sample_df = background_df[background_df["id"].isin(chosen)].copy()
        samples.append(sample_df)
    return samples


# ---------------------------------------------------------------------------
# Study-by-ROI activation matrix
# ---------------------------------------------------------------------------

def build_study_roi_activation(
    study_coords: pd.DataFrame,
    atlas_labels: pd.DataFrame,
    roi_radius: float = 6.0,
) -> Tuple[np.ndarray, Dict]:
    """Build binary study-by-ROI activation matrix.

    A[s,i] = 1 if study s has at least one peak within roi_radius mm of
    ROI i center (assumed MNI coordinates from AAL116_labels.csv).

    Parameters
    ----------
    study_coords : DataFrame with columns [id, x, y, z]
    atlas_labels : DataFrame with columns [roi_index, roi_label, x, y, z]
        or equivalent MNI center coordinates.

    Returns
    -------
    activation_matrix : (n_studies, n_rois) binary np.ndarray
    metadata : dict with study_ids and roi_labels
    """
    study_ids = study_coords["id"].unique()
    n_studies = len(study_ids)
    n_rois = len(atlas_labels)
    study_to_idx = {sid: i for i, sid in enumerate(study_ids)}

    activation = np.zeros((n_studies, n_rois), dtype=np.float64)

    roi_centers = atlas_labels[["x", "y", "z"]].values.astype(np.float64)

    for _, row in study_coords.iterrows():
        si = study_to_idx[row["id"]]
        coord = np.array([row["x"], row["y"], row["z"]])
        dists = np.sqrt(np.sum((roi_centers - coord) ** 2, axis=1))
        activation[si, dists <= roi_radius] = 1.0

    metadata = {
        "study_ids": study_ids.tolist(),
        "roi_labels": atlas_labels["roi_label"].tolist(),
        "n_studies": n_studies,
        "n_rois": n_rois,
        "roi_radius": roi_radius,
    }
    return activation, metadata


# ---------------------------------------------------------------------------
# Edge coactivation statistics
# ---------------------------------------------------------------------------

def compute_coactivation_statistics(
    positive_activation: np.ndarray,
    background_activations: List[np.ndarray],
    n_rois: int = 116,
    epsilon: float = EPSILON,
) -> Dict:
    """Compute LOR and PMI coactivation statistics.

    Fully vectorized implementation.

    Parameters
    ----------
    positive_activation : (n_pos, n_rois) binary
    background_activations : list of (n_bg, n_rois) binary arrays

    Returns
    -------
    dict with LOR_mean, LOR_sd, PMI_fluid, PMI_background, delta_PMI per edge
    """
    n_pos = positive_activation.shape[0]
    n_bg_list = [bg.shape[0] for bg in background_activations]

    # Upper triangle edge indices
    iu = np.triu_indices(n_rois, k=1)
    n_edges = len(iu[0])

    # --- Vectorized co-occurrence counts ---
    # cooccur[s, i, j] = 1 if study s activates both ROI i and j
    # Use einsum for fast outer product co-occurrence
    pos_cooccur = np.einsum('si,sj->ij', positive_activation, positive_activation)
    pos_both = pos_cooccur[iu]  # (n_edges,)

    # LOR across balanced background repetitions
    lor_values = np.zeros((len(background_activations), n_edges), dtype=np.float64)
    for r, bg_act in enumerate(background_activations):
        n_bg_r = bg_act.shape[0]
        bg_cooccur = np.einsum('si,sj->ij', bg_act, bg_act)
        bg_both = bg_cooccur[iu]  # (n_edges,)

        # Haldane-corrected LOR
        lor_values[r, :] = (
            np.log((pos_both + 0.5) / (n_pos - pos_both + 0.5))
            - np.log((bg_both + 0.5) / (n_bg_r - bg_both + 0.5))
        )

    lor_mean = np.mean(lor_values, axis=0)
    lor_sd = np.std(lor_values, axis=0, ddof=1) if len(background_activations) > 1 else np.zeros(n_edges)

    # --- PMI computation ---
    def _compute_pmi_matrix(act_matrix):
        """Compute PMI matrix from binary activation matrix."""
        n = act_matrix.shape[0]
        p_i = act_matrix.mean(axis=0)  # (n_rois,)
        # Co-occurrence probability
        p_ij = np.einsum('si,sj->ij', act_matrix, act_matrix) / n
        # PMI = log(P(i,j) / (P(i) * P(j)))
        denom = np.outer(p_i, p_i) + epsilon
        pmi = np.log((p_ij + epsilon) / denom)
        return pmi

    pmi_fluid_full = _compute_pmi_matrix(positive_activation)

    pmi_bg_full = np.zeros((n_rois, n_rois), dtype=np.float64)
    for bg_act in background_activations:
        pmi_bg_full += _compute_pmi_matrix(bg_act)
    pmi_bg_full /= len(background_activations)

    delta_pmi_full = pmi_fluid_full - pmi_bg_full

    # Extract upper triangle
    pmi_fluid = pmi_fluid_full[iu]
    pmi_bg = pmi_bg_full[iu]
    delta_pmi = delta_pmi_full[iu]

    return {
        "lor_mean": lor_mean,
        "lor_sd": lor_sd,
        "pmi_fluid": pmi_fluid,
        "pmi_background": pmi_bg,
        "delta_pmi": delta_pmi,
        "n_positive": n_pos,
        "n_background_mean": float(np.mean(n_bg_list)),
        "iu": iu,
    }


# ---------------------------------------------------------------------------
# FIP construction
# ---------------------------------------------------------------------------

def _rank_normalize_positive(values: np.ndarray) -> np.ndarray:
    """R_+: retain positive values, rank-normalize to [0,1], set non-positive to 0."""
    result = np.zeros_like(values, dtype=np.float64)
    positive_mask = values > 0
    if not positive_mask.any():
        return result
    pos_vals = values[positive_mask]
    # Rank normalization within positive values
    ranks = np.argsort(np.argsort(pos_vals)).astype(np.float64) + 1
    result[positive_mask] = ranks / (len(pos_vals) + 1)
    return result


def construct_fip1_mac(
    delta_pmi: np.ndarray,
    lor_mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """FIP-1: Meta-Analytic Coactivation Consensus.

    q_MAC = (R+(delta_PMI) + R+(LOR)) / 2, normalized to [0,1].
    """
    r_delta_pmi = _rank_normalize_positive(delta_pmi)
    r_lor = _rank_normalize_positive(lor_mean)
    fip1 = 0.5 * (r_delta_pmi + r_lor)
    # Normalize to [0,1]
    fmin, fmax = fip1.min(), fip1.max()
    if fmax > fmin:
        fip1 = (fip1 - fmin) / (fmax - fmin)
    else:
        fip1 = np.zeros_like(fip1)
    return fip1, None


def construct_fip2_bridge(
    fip1: np.ndarray,
    iu: Tuple[np.ndarray, np.ndarray],
    module_labels: np.ndarray,
    rho: float = 1.0,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """FIP-2: Between-Network Bridge Prior.

    q_bridge = q_MAC * [1 + rho * 1(network_i != network_j)]
    """
    n_rois = len(module_labels)
    net_i = module_labels[iu[0]]
    net_j = module_labels[iu[1]]
    between_network = (net_i != net_j).astype(np.float64)

    fip2 = fip1 * (1.0 + rho * between_network)
    fmin, fmax = fip2.min(), fip2.max()
    if fmax > fmin:
        fip2 = (fip2 - fmin) / (fmax - fmin)
    else:
        fip2 = np.zeros_like(fip2)
    return fip2, None


def construct_fip3_weaktie(
    fip1: np.ndarray,
    iu: Tuple[np.ndarray, np.ndarray],
    module_labels: np.ndarray,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """FIP-3: Distributed Weak/Intermediate Integration Prior.

    Between-network: q_weak = q_MAC * (1 - q_MAC)^0.5
    Within-network: q_weak = 0.5 * q_MAC
    """
    n_rois = len(module_labels)
    net_i = module_labels[iu[0]]
    net_j = module_labels[iu[1]]
    between_network = (net_i != net_j).astype(np.float64)

    fip3 = np.where(
        between_network == 1,
        fip1 * (1.0 - fip1) ** 0.5,
        0.5 * fip1,
    )
    fmin, fmax = fip3.min(), fip3.max()
    if fmax > fmin:
        fip3 = (fip3 - fmin) / (fmax - fmin)
    else:
        fip3 = np.zeros_like(fip3)
    return fip3, None


# ---------------------------------------------------------------------------
# FIP matrix validation
# ---------------------------------------------------------------------------

def validate_fip_matrix(
    fip_matrix: np.ndarray,
    name: str,
    n_rois: int = 116,
) -> Dict:
    """Validate FIP matrix invariants."""
    issues = []
    if fip_matrix.shape != (n_rois, n_rois):
        issues.append(f"shape={fip_matrix.shape}, expected ({n_rois},{n_rois})")
    if not np.allclose(fip_matrix, fip_matrix.T):
        issues.append("not symmetric")
    if not np.allclose(np.diag(fip_matrix), 0):
        issues.append("diagonal not zero")
    if not np.isfinite(fip_matrix).all():
        issues.append("contains NaN/Inf")
    if fip_matrix.min() < 0:
        issues.append(f"min={fip_matrix.min():.4f} < 0")
    if fip_matrix.max() > 1:
        issues.append(f"max={fip_matrix.max():.4f} > 1")

    return {
        "name": name,
        "valid": len(issues) == 0,
        "shape": fip_matrix.shape,
        "symmetric": np.allclose(fip_matrix, fip_matrix.T),
        "zero_diagonal": np.allclose(np.diag(fip_matrix), 0),
        "finite": np.isfinite(fip_matrix).all(),
        "min": float(fip_matrix.min()),
        "max": float(fip_matrix.max()),
        "mean": float(fip_matrix.mean()),
        "issues": issues,
    }


# ---------------------------------------------------------------------------
# Edge-level Laplacian for FIP (line-graph construction)
# ---------------------------------------------------------------------------

def build_fip_line_graph_laplacian(
    fip_edges: np.ndarray,
    n_rois: int = 116,
    top_k: int = 30,
    normalize: str = "sym",
):
    """Build line-graph Laplacian for edge-level FIP.

    Two FC features e=(i,j) and f=(k,l) are adjacent when they share a ROI.
    Weight w_ef = sqrt(q_e * q_f) when they share a node, 0 otherwise.

    Active edges are the top_k edges by FIP weight, matching the original
    MS-A-NCR convention.

    Returns an EdgeLaplacian compatible with MS-A-NCR.
    """
    from metascfc.models.iclr_backbones.network_constrained_ridge import EdgeLaplacian

    iu = np.triu_indices(n_rois, k=1)
    n_edges = len(iu[0])

    # Select top_k active edges by FIP weight
    top_k_actual = min(top_k, n_edges)
    active_indices = np.argsort(fip_edges)[-top_k_actual:] if top_k_actual > 0 else np.array([], dtype=int)
    active_set = set(active_indices.tolist())

    # Build adjacency: which active edges share a node?
    roi_to_active_edges = {}
    for e_idx in active_indices:
        i, j = iu[0][e_idx], iu[1][e_idx]
        roi_to_active_edges.setdefault(i, []).append(e_idx)
        roi_to_active_edges.setdefault(j, []).append(e_idx)

    # Build sparse adjacency weight matrix among active edges only
    from scipy import sparse

    rows, cols, vals = [], [], []
    for e_idx in active_indices:
        i, j = iu[0][e_idx], iu[1][e_idx]
        adjacent_edges = set(roi_to_active_edges.get(i, []) + roi_to_active_edges.get(j, []))
        adjacent_edges.discard(e_idx)

        for f_idx in adjacent_edges:
            w = np.sqrt(max(fip_edges[e_idx], 0) * max(fip_edges[f_idx], 0))
            if w > 0:
                rows.append(e_idx)
                cols.append(f_idx)
                vals.append(w)

    n_active = len(active_indices)
    # Map to local indices
    local_map = {e: i for i, e in enumerate(active_indices)}
    local_rows = [local_map[r] for r in rows]
    local_cols = [local_map[c] for c in cols]

    adj = sparse.csr_matrix((vals, (local_rows, local_cols)), shape=(n_active, n_active))
    adj = adj + adj.T
    adj.data /= 2.0

    # Degree matrix
    deg = np.array(adj.sum(axis=1)).ravel()
    deg_inv_sqrt = np.zeros_like(deg)
    nz = deg > 0
    deg_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])

    # Normalized Laplacian: L = I - D^-1/2 A D^-1/2
    if normalize == "sym":
        lap = sparse.eye(n_active) - sparse.diags(deg_inv_sqrt) @ adj @ sparse.diags(deg_inv_sqrt)
    else:
        lap = sparse.eye(n_active) - adj / np.maximum(deg[:, None], 1e-30)

    active_laplacian = lap.toarray()

    return EdgeLaplacian(
        n_rois=n_rois,
        n_edges=n_edges,
        active_indices=active_indices,
        active_laplacian=active_laplacian,
        top_k=top_k,
        weighting="fip_edge",
        couple_modalities=False,
    )


# ---------------------------------------------------------------------------
# FIP-to-MS-A-NCR cache (edge_direct lifting)
# ---------------------------------------------------------------------------

def build_fip_msancr_cache(
    fip_edges: np.ndarray,
    n_rois: int = 116,
    gamma: float = 0.5,
    top_k: int = 30,
    epsilon: float = 1e-3,
    normalize_laplacian: str = "sym",
):
    """Build MS-A-NCR cache from edge-level FIP (bypasses node lifting).

    This is the key function for prior_space='edge': the FIP edges vector
    is used directly as q_e for D(q;gamma), and a line-graph Laplacian
    is constructed from the FIP.
    """
    from metascfc.models.iclr_backbones.modality_selective_anisotropic_ncr import (
        _MSANCRCache,
    )

    edge_prior = np.asarray(fip_edges, dtype=np.float64).ravel()
    if len(edge_prior) != n_rois * (n_rois - 1) // 2:
        raise ValueError(
            f"fip_edges has {len(edge_prior)} entries; expected {n_rois*(n_rois-1)//2}"
        )

    D = compute_diagonal_penalty(edge_prior, gamma, epsilon, normalize=True)
    D_inv_sqrt = 1.0 / np.sqrt(np.maximum(D, 1e-30))

    edge_lap = build_fip_line_graph_laplacian(
        fip_edges, n_rois=n_rois, top_k=top_k, normalize=normalize_laplacian,
    )

    active = edge_lap.active_indices
    if len(active) > 0:
        d_active_inv_sqrt = D_inv_sqrt[active]
        whitened_laplacian = (
            d_active_inv_sqrt[:, None]
            * edge_lap.active_laplacian
            * d_active_inv_sqrt[None, :]
        )
        whitened_laplacian = 0.5 * (whitened_laplacian + whitened_laplacian.T)
        generalized_mu, generalized_u = np.linalg.eigh(whitened_laplacian)
        generalized_mu = np.clip(generalized_mu, 0.0, None)
    else:
        generalized_u = np.empty((0, 0), dtype=np.float64)
        generalized_mu = np.empty(0, dtype=np.float64)

    n_edges = n_rois * (n_rois - 1) // 2

    return _MSANCRCache(
        D=D,
        D_inv_sqrt=D_inv_sqrt,
        active_indices=active,
        D_active=D[edge_lap.active_indices] if len(active) > 0 else np.array([]),
        active_laplacian=edge_lap.active_laplacian,
        generalized_u=generalized_u,
        generalized_mu=generalized_mu,
        n_edges=n_edges,
        n_rois=n_rois,
        gamma=float(gamma),
        lifting="edge_direct",
    )


# ---------------------------------------------------------------------------
# Prior similarity audit
# ---------------------------------------------------------------------------

def compute_prior_similarity(
    prior_a: np.ndarray,
    prior_b: np.ndarray,
) -> Dict:
    """Compute similarity between two edge-level priors."""
    from scipy.stats import pearsonr, spearmanr

    a = np.asarray(prior_a, dtype=np.float64).ravel()
    b = np.asarray(prior_b, dtype=np.float64).ravel()
    assert len(a) == len(b)

    # Cosine similarity
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cosine = dot / (norm_a * norm_b + 1e-30)

    # Jaccard of top-10% and top-20%
    n = len(a)
    top10 = max(1, n // 10)
    top20 = max(1, n // 5)

    top_a_10 = set(np.argsort(a)[-top10:])
    top_b_10 = set(np.argsort(b)[-top10:])
    jaccard_10 = len(top_a_10 & top_b_10) / max(len(top_a_10 | top_b_10), 1)

    top_a_20 = set(np.argsort(a)[-top20:])
    top_b_20 = set(np.argsort(b)[-top20:])
    jaccard_20 = len(top_a_20 & top_b_20) / max(len(top_a_20 | top_b_20), 1)

    return {
        "pearson": float(pearsonr(a, b).statistic),
        "spearman": float(spearmanr(a, b).statistic),
        "cosine": float(cosine),
        "jaccard_top10": float(jaccard_10),
        "jaccard_top20": float(jaccard_20),
    }


# ---------------------------------------------------------------------------
# Shuffled FIP control
# ---------------------------------------------------------------------------

def create_shuffled_fip(
    fip_matrix: np.ndarray,
    seed: int = 0,
) -> np.ndarray:
    """Create shuffled FIP via ROI-label permutation.

    Preserves symmetry, zero diagonal, weight distribution, density.
    """
    rng = np.random.default_rng(seed)
    n = fip_matrix.shape[0]
    perm = rng.permutation(n)
    shuffled = fip_matrix[np.ix_(perm, perm)]
    return shuffled


def fip_matrix_to_edges(fip_matrix: np.ndarray, n_rois: int = 116) -> np.ndarray:
    """Extract upper-triangle edge vector from FIP matrix."""
    iu = np.triu_indices(n_rois, k=1)
    return fip_matrix[iu].copy()


def fip_edges_to_matrix(fip_edges: np.ndarray, n_rois: int = 116) -> np.ndarray:
    """Reconstruct symmetric FIP matrix from edge vector."""
    mat = np.zeros((n_rois, n_rois), dtype=np.float64)
    iu = np.triu_indices(n_rois, k=1)
    mat[iu] = fip_edges
    mat = mat + mat.T
    return mat
