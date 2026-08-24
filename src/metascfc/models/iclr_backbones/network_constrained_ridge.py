#!/usr/bin/env python3
"""Network-Constrained Prior-Laplacian Ridge (ICLR 2027, Method 1).

Objective
---------
    minimize_β   ||y - Xβ||^2  +  λ1·||β||^2  +  λ2·β^T L_prior β

where X = [FC_upper | SC_upper] is the per-subject concatenated
upper-triangle connectome edge-feature matrix and L_prior is a positive
semi-definite penalty matrix derived from the ROI-level meta-analysis
prior.  Unlike the AAAI prior-weighted Ridge (which *rescales* features by
the prior and thereby standardizes away its effect), this formulation ties
the prior directly to the predictive weights: weights of regions that the
prior declares functionally coupled are forced to be *smooth*.

ROI prior -> edge-feature penalty (line-graph lift)
---------------------------------------------------
The meta-analysis prior is a scalar score per ROI.  Thresholding the top-k
ROIs yields a binary "prior adjacency" A_prior over ROIs.  The design
matrix, however, lives in edge space: each feature is one (i, j)
connectivity edge.  We lift the ROI-level adjacency to the line graph of
the connectome by declaring two edges adjacent when they share a node that
is prior-active:

    e = (i, j) ~ f = (k, l)   <=>   |{i, j} ∩ {k, l}| = 1  and shared node ∈ S.

The Laplacian quadratic form on this line graph reads

    β^T L_line β  =  Σ_{e ~ f} (β_e - β_f)^2,

i.e. predictive weights of edges incident to functionally coupled
(prior-active) ROIs are smoothed toward one another, which is exactly the
"network-constrained regression" penalty (Li & Li, 2008) lifted to edge
features.  Edges that touch no prior-active ROI are isolated line-graph
nodes: their L row is zero and they are penalized only by the ridge term
(handled explicitly as an edge case).  Because FC and SC share the same
ROI space, the same active edge set / Laplacian block is applied to both
feature blocks; optionally (couple_modalities=True) the FC and SC copies of
an active edge are coupled so that cross-modality weights for the same ROI
pair are smoothed together.  The coupled penalty is made PSD by
construction: L = I⊗L0 + [[I, -I], [-I, I]].

Dual solve
----------
The normal equations are (X^T X + P) β = X^T y with the effective penalty
P = s·(λ1·I + λ2·L) where s = max(1, n_features) follows the dual-Ridge
kernel convention of the AAAI baselines.  By the Woodbury identity the
solution lives purely in the n_subjects x n_subjects dual space:

    α = (I + X P^-1 X^T)^-1 y,      β = P^-1 X^T α,
    ŷ_test = X_test P^-1 X^T α.

With the active-block eigen-decomposition L0 = U diag(μ) U^T (computed once
per prior, independent of the data fold) and the whitened designs
w = X_a U, the n x n matrix X P^-1 X^T collapses to

    X P^-1 X^T = (1/s) · [ Σ_j (w_j w_j^T)/(λ1 + λ2 μ_j)  +  X_i X_i^T/λ1 ],

where the second term is the ridge-only "inactive" feature block.  Because
1/(λ1 + λ2 μ) = (1/λ1)/(1 + τ μ) with τ = λ2/λ1, the inner n x n matrix
B(τ) + X_i X_i^T is eigen-decomposed once per distinct τ = λ2/λ1 and the
whole λ1 grid is then evaluated in O(n^2) per candidate.  This is exact,
leakage-free, and reproduces the plain FC+SC Ridge baseline at λ2 = 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.linalg import eigh as dense_eigh
from sklearn.preprocessing import StandardScaler


def build_prior_adjacency(
    prior_scores: np.ndarray,
    top_k: int = 30,
) -> np.ndarray:
    """Build the binary ROI-level prior adjacency from a meta-analysis map.

    Parameters
    ----------
    prior_scores : np.ndarray, shape (n_rois,)
        Non-negative ROI-level prior scores (e.g. the working-memory map).
    top_k : int
        Number of top-scoring ROIs that form the prior network.  The prior
        adjacency is the complete graph on this "prior-active" set (the
        prompt's "threshold the top 30 ROIs to form a prior adjacency
        matrix").

    Returns
    -------
    np.ndarray, shape (n_rois, n_rois)
        Symmetric, zero-diagonal adjacency matrix with A_ij = 1 iff both
        ROIs i and j are in the top-k set.

    Raises
    ------
    ValueError
        If top_k is smaller than 1 or prior_scores is empty.
    """
    p = np.asarray(prior_scores, dtype=np.float64).reshape(-1)
    n_rois = len(p)
    if n_rois == 0:
        raise ValueError("prior_scores must not be empty")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if top_k >= n_rois:
        top_k = n_rois

    # Ties are broken deterministically: argpartition returns the top-k set
    # (the exact set does not depend on the pivot ordering of equal scores).
    active = set(np.argpartition(p, -top_k)[-top_k:].tolist())
    adjacency = np.zeros((n_rois, n_rois), dtype=np.float64)
    iu = np.triu_indices(n_rois, k=1)
    for i, j in zip(iu[0], iu[1]):
        if i in active and j in active:
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
    return adjacency


def build_edge_laplacian(
    n_rois: int,
    prior_scores: Optional[np.ndarray] = None,
    prior_adjacency: Optional[np.ndarray] = None,
    top_k: int = 30,
    weighting: str = "binary",
    couple_modalities: bool = False,
    normalize: str = "sym",
) -> "EdgeLaplacian":
    """Build the edge-feature Laplacian penalty from an ROI prior.

    Exactly one of ``prior_scores`` and ``prior_adjacency`` must be given.

    Parameters
    ----------
    n_rois : int
        Atlas size (features are the upper-triangle edges of an n_rois x
        n_rois connectome).
    prior_scores : np.ndarray, optional
        ROI-level prior scores; ``build_prior_adjacency`` is applied with
        ``top_k``.
    prior_adjacency : np.ndarray, optional
        Precomputed ROI adjacency (e.g. from ``build_prior_adjacency``).
    top_k : int
        Prior-active ROI threshold (ignored if prior_adjacency is given).
    weighting : {"binary", "node_prior"}
        Line-graph edge weight between two adjacent edge features: constant
        1.0, or the prior score of their shared node.
    couple_modalities : bool
        Couple the FC and SC copies of each active edge (see EdgeLaplacian).
        The coupled penalty is PSD by construction.
    normalize : {"sym", "none"}
        "sym": symmetric-normalized Laplacian I - D^-1/2 A D^-1/2 (default,
        eigenvalues in [0, 2], scale-comparable with the ridge term);
        "none": unnormalized Laplacian D - A.

    Returns
    -------
    EdgeLaplacian
        Sparse-friendly representation of the penalty (active block only).
    """
    if (prior_scores is None) == (prior_adjacency is None):
        raise ValueError("Provide exactly one of prior_scores / prior_adjacency")
    if normalize not in ("sym", "none"):
        raise ValueError(f"normalize must be 'sym' or 'none', got {normalize!r}")
    if weighting not in ("binary", "node_prior"):
        raise ValueError(f"weighting must be 'binary' or 'node_prior', got {weighting!r}")
    if n_rois < 2:
        raise ValueError(f"n_rois must be >= 2, got {n_rois}")

    if prior_adjacency is None:
        prior_adjacency = build_prior_adjacency(np.asarray(prior_scores), top_k)
    adjacency = np.asarray(prior_adjacency, dtype=np.float64)
    if adjacency.shape != (n_rois, n_rois):
        raise ValueError(
            f"prior_adjacency shape {adjacency.shape} does not match n_rois={n_rois}"
        )
    if not np.allclose(adjacency, adjacency.T, atol=1e-12):
        raise ValueError("prior_adjacency must be symmetric")
    if prior_scores is None:
        top_k = int((np.abs(adjacency) > 0).sum(axis=0).max())
    p = (
        np.asarray(prior_scores, dtype=np.float64).reshape(-1)
        if prior_scores is not None
        else np.ones(n_rois, dtype=np.float64)
    )

    iu = np.triu_indices(n_rois, k=1)
    n_edges = int(iu[0].size)
    edge_nodes = np.stack([iu[0], iu[1]], axis=1)  # (n_edges, 2)

    # Active edges: edges with at least one endpoint in the prior-active set.
    active_set = np.where(np.abs(adjacency).sum(axis=0) > 0)[0]
    is_active = np.any(np.isin(edge_nodes, active_set), axis=1)
    active_indices = np.where(is_active)[0].astype(np.int64)

    n_active = len(active_indices)
    if n_active == 0:
        # No edge touches the prior-active set: the penalty is empty and the
        # method degenerates to plain ridge (λ2 has no effect).  Not an error.
        return EdgeLaplacian(
            active_indices=np.empty(0, dtype=np.int64),
            active_laplacian=np.zeros((0, 0), dtype=np.float64),
            n_edges=n_edges,
            n_rois=n_rois,
            top_k=top_k,
            weighting=weighting,
            couple_modalities=couple_modalities,
        )

    # Line-graph adjacency among active edges: two active edges are adjacent
    # iff they share exactly one node, and that node is prior-active.  For
    # each prior-active ROI u, every pair of distinct active edges incident
    # to u is an adjacency.
    pos = {int(u): i for i, u in enumerate(active_set)}
    edges_by_node: Dict[int, list] = {}
    for local, global_edge in enumerate(active_indices):
        u, v = int(edge_nodes[global_edge, 0]), int(edge_nodes[global_edge, 1])
        for node in (u, v):
            if node in pos:
                edges_by_node.setdefault(node, []).append(local)

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for node in active_set:
        incident = edges_by_node.get(node, [])
        w = float(p[node]) if weighting == "node_prior" else 1.0
        for a in range(len(incident)):
            for b in range(a + 1, len(incident)):
                rows.append(incident[a])
                cols.append(incident[b])
                rows.append(incident[b])
                cols.append(incident[a])
                data.append(w)
                data.append(w)

    line_adj = sparse.coo_matrix(
        (np.asarray(data, dtype=np.float64), (rows, cols)),
        shape=(n_active, n_active),
        dtype=np.float64,
    ).tocsr()

    degree = np.asarray(line_adj.sum(axis=1)).reshape(-1)
    if normalize == "none":
        lap = sparse.diags(degree) - line_adj
        lap = lap.toarray()
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_sqrt_deg = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        lap = (
            sparse.identity(n_active, format="csr")
            - sparse.diags(inv_sqrt_deg) @ line_adj @ sparse.diags(inv_sqrt_deg)
        )
        lap = lap.toarray()

    # Guard against tiny numerical asymmetry / negative eigenvalues.
    lap = 0.5 * (lap + lap.T)
    min_eig = np.linalg.eigvalsh(lap)[0]
    if min_eig < -1e-10:
        lap = lap - min_eig * np.eye(n_active)

    return EdgeLaplacian(
        active_indices=active_indices,
        active_laplacian=lap,
        n_edges=n_edges,
        n_rois=n_rois,
        top_k=top_k,
        weighting=weighting,
        couple_modalities=couple_modalities,
    )


@dataclass(frozen=True)
class EdgeLaplacian:
    """Precomputed edge-feature penalty matrix for one prior.

    Attributes
    ----------
    active_indices : np.ndarray, shape (n_active,)
        Indices (into the upper-triangle feature vector, length n_edges)
        of edges that touch at least one prior-active ROI.  These are the
        only features that carry a Laplacian penalty.
    active_laplacian : np.ndarray, shape (n_active, n_active)
        Dense symmetric positive semi-definite Laplacian block L0 over the
        active edges (one modality; the same block applies to both FC and
        SC).  Isolated line-graph nodes appear as all-zero rows (no
        penalty).  Empty (0 x 0) when no edge is active.
    n_edges : int
        Number of edge features per modality, n_rois * (n_rois - 1) / 2.
    n_rois : int
        Number of ROIs of the atlas.
    top_k : int
        Threshold used to define the prior-active ROI set.
    weighting : str
        "binary" or "node_prior" (adjacency weight = prior score of the
        shared node).
    couple_modalities : bool
        Whether the FC and SC copies of each active edge are coupled in the
        penalty (default False).  The coupled penalty is the PSD matrix
        I⊗L0 + [[I, -I], [-I, I]].
    """

    active_indices: np.ndarray
    active_laplacian: np.ndarray
    n_edges: int
    n_rois: int
    top_k: int
    weighting: str
    couple_modalities: bool

    @property
    def n_active(self) -> int:
        """Number of prior-active edge features (per modality)."""
        return int(self.active_indices.shape[0])

    def full_laplacian(self) -> sparse.csr_matrix:
        """Return the full (2*n_edges, 2*n_edges) sparse penalty matrix.

        Block structure over the FC and SC feature blocks, with the
        optional PSD cross-modality coupling I⊗L0 + [[I, -I], [-I, I]].
        """
        n_edges = self.n_edges
        n_active = self.n_active
        n_features = 2 * n_edges
        if n_active == 0:
            return sparse.csr_matrix((n_features, n_features), dtype=np.float64)
        active = self.active_indices
        if self.couple_modalities:
            eye = sparse.eye(n_active, dtype=np.float64, format="csr")
            block = sparse.bmat(
                [[self.active_laplacian + eye, -eye],
                 [-eye, self.active_laplacian + eye]],
                format="csr",
                dtype=np.float64,
            )
        else:
            block = sparse.csr_matrix(self.active_laplacian)
        lap = sparse.bmat(
            [[block, None], [None, block]], format="csr", dtype=np.float64
        )
        row_col = np.concatenate([active, active + n_edges])
        coo = lap.tocoo()
        return sparse.coo_matrix(
            (coo.data, (row_col[coo.row], row_col[coo.col])),
            shape=(n_features, n_features),
            dtype=np.float64,
        ).tocsr()


@dataclass(frozen=True)
class _LaplacianEig:
    """Eigen-decomposition of the active Laplacian block (one per prior).

    The FC and SC active blocks share the same Laplacian L0 = U diag(μ) U^T.
    With cross-modality coupling the penalty becomes the PSD matrix
    I⊗L0 + [[I, -I], [-I, I]] whose eigen-directions are the symmetric and
    anti-symmetric copies of each U-column with eigenvalues μ_j and μ_j + 2;
    this is used in closed form (no n_active-multiplication eigensolve).

    Attributes
    ----------
    u : np.ndarray, shape (n_active, n_active)
        Eigenvectors of the active Laplacian block L0.
    mu : np.ndarray, shape (n_active,)
        Eigenvalues of L0 (>= 0).
    n_active : int
    n_edges : int
    couple : bool
    """

    u: np.ndarray
    mu: np.ndarray
    n_active: int
    n_edges: int
    couple: bool


def factor_laplacian_eig(edge_laplacian: EdgeLaplacian) -> _LaplacianEig:
    """Compute the eigen-decomposition of the active Laplacian block.

    The result depends only on the prior, never on the data split, so it is
    computed once per prior and reused across all folds.
    """
    n_active = edge_laplacian.n_active
    if n_active == 0:
        return _LaplacianEig(
            u=np.empty((0, 0)), mu=np.empty(0), n_active=0,
            n_edges=edge_laplacian.n_edges,
            couple=edge_laplacian.couple_modalities,
        )
    mu, u = np.linalg.eigh(edge_laplacian.active_laplacian)
    # Guard against tiny negative eigenvalues from round-off.
    mu = np.clip(mu, 0.0, None)
    return _LaplacianEig(
        u=u, mu=mu, n_active=n_active, n_edges=edge_laplacian.n_edges,
        couple=edge_laplacian.couple_modalities,
    )


def _whitened_pairs(
    x_active: np.ndarray,
    eig: _LaplacianEig,
) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
    """Return [(z, mu, fc_sign, sc_sign), ...] defining X P^-1 X^T.

    ``x_active`` is the (n, 2*n_active) matrix of raw active feature columns
    (FC block followed by SC block).  Each returned pair contributes
    z diag(1/(λ1 + λ2 μ)) z^T to the n x n matrix X P^-1 X^T and
    fc_sign/sc_sign * U ((z^T α)/(λ1 + λ2 μ)) to the primal weights of the
    FC / SC active blocks.  Pair order is deterministic.
    """
    n_active = eig.n_active
    if n_active == 0:
        return []
    x_fc = x_active[:, :n_active]
    x_sc = x_active[:, n_active:]
    if not eig.couple:
        return [
            (x_fc @ eig.u, eig.mu, 1.0, 0.0),
            (x_sc @ eig.u, eig.mu, 0.0, 1.0),
        ]
    w = x_fc @ eig.u
    v = x_sc @ eig.u
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    return [
        ((w + v) * inv_sqrt2, eig.mu, 1.0, 1.0),
        ((w - v) * inv_sqrt2, eig.mu + 2.0, 1.0, -1.0),
    ]


def _denominator_grid(
    alpha1_grid: Sequence[float], alpha2_grid: Sequence[float]
) -> Dict[float, List[Tuple[float, float]]]:
    """Group (λ1, λ2) candidates by τ = λ2/λ1 for shared eigensolves.

    Returns a mapping τ -> [(λ1, λ2), ...].  Candidates with λ2 = 0 share
    τ = 0 (the plain-ridge limit, which also verifies the solver).
    """
    out: Dict[float, List[Tuple[float, float]]] = {}
    for alpha1 in alpha1_grid:
        for alpha2 in alpha2_grid:
            tau = 0.0 if alpha2 == 0.0 else alpha2 / alpha1
            out.setdefault(tau, []).append((alpha1, alpha2))
    return out


def _quadratic_form(
    pairs: List[Tuple[np.ndarray, np.ndarray, float, float]],
    alpha1: float,
    alpha2: float,
) -> np.ndarray:
    """Sum over pairs of z diag(1/(λ1 + λ2 μ)) z^T (n x n)."""
    n = pairs[0][0].shape[0] if pairs else 0
    out = np.zeros((n, n), dtype=np.float64)
    for z, mu, _, _ in pairs:
        denom = alpha1 + alpha2 * mu
        out += (z / denom) @ z.T
    return out


def _cross_quadratic(
    pairs: List[Tuple[np.ndarray, np.ndarray, float, float]],
    x_eval_active: np.ndarray,
    eig: _LaplacianEig,
    alpha1: float,
    alpha2: float,
) -> np.ndarray:
    """n_eval x n version of the quadratic form (for predictions).

    ``x_eval_active`` is the (n_eval, 2*n_active) raw active column matrix
    of the evaluation sample; the whitening rotation is re-applied with the
    same eigenbasis used to build ``pairs``.
    """
    if not pairs:
        return np.zeros((x_eval_active.shape[0], 0), dtype=np.float64)
    eval_pairs = _whitened_pairs(x_eval_active, eig)
    out = np.zeros((x_eval_active.shape[0], pairs[0][0].shape[0]), dtype=np.float64)
    for (z, mu, _, _), (z_eval, _, _, _) in zip(pairs, eval_pairs):
        denom = alpha1 + alpha2 * mu
        out += (z_eval / denom) @ z.T
    return out


def _primal_active_weights(
    pairs: List[Tuple[np.ndarray, np.ndarray, float, float]],
    eig: _LaplacianEig,
    alpha: np.ndarray,
    alpha1: float,
    alpha2: float,
    scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recover β on the FC and SC active feature blocks from the dual α.

    β_block = Σ_pairs sign * U diag(1/(λ1 + λ2 μ)) (z^T α) / scale.
    """
    n_active = eig.n_active
    if n_active == 0:
        return np.empty(0), np.empty(0)
    beta_fc = np.zeros(n_active)
    beta_sc = np.zeros(n_active)
    for z, mu, s_fc, s_sc in pairs:
        denom = alpha1 + alpha2 * mu
        proj = z.T @ alpha
        if s_fc != 0.0:
            beta_fc += s_fc * (eig.u @ (proj / denom))
        if s_sc != 0.0:
            beta_sc += s_sc * (eig.u @ (proj / denom))
    return beta_fc / scale, beta_sc / scale


def node_saliency_from_beta(
    beta: np.ndarray,
    n_rois: int,
    normalize: bool = True,
) -> np.ndarray:
    """Map a full edge-weight vector back to an ROI-level saliency map.

    The predictive weight vector lives on the concatenated FC/SC upper
    triangles.  This function aggregates |β| over all edges incident to
    each ROI (across both modalities), which yields the "biomarker" of a
    linear connectome predictor and is directly comparable with the prior
    (same pipeline as the AAAI node-saliency biomarker analyses).

    Parameters
    ----------
    beta : np.ndarray, shape (2 * n_edges,)
        Learned weights on standardized FC+SC edge features.
    n_rois : int
        Atlas size.
    normalize : bool
        Min-max normalize the saliency to [0, 1] (default True).

    Returns
    -------
    np.ndarray, shape (n_rois,)
        Node-level saliency vector.
    """
    beta = np.asarray(beta, dtype=np.float64).reshape(-1)
    n_edges = int(n_rois * (n_rois - 1) / 2)
    if beta.shape[0] != 2 * n_edges:
        raise ValueError(
            f"beta has {beta.shape[0]} entries; expected {2 * n_edges} "
            f"(2 modalities x {n_edges} edges for n_rois={n_rois})"
        )
    iu = np.triu_indices(n_rois, k=1)
    saliency = np.zeros(n_rois, dtype=np.float64)
    for block in (beta[:n_edges], beta[n_edges:]):
        np.add.at(saliency, iu[0], np.abs(block))
        np.add.at(saliency, iu[1], np.abs(block))
    if normalize:
        span = saliency.max() - saliency.min()
        if span > 1e-12:
            saliency = (saliency - saliency.min()) / span
        else:
            saliency = np.zeros_like(saliency)
    return saliency


class NetworkConstrainedRidge:
    """Linear Ridge with a network-constrained (graph-Laplacian) penalty.

    Solves  min_β ||y - Xβ||^2 + λ1||β||^2 + λ2 β^T L_prior β  in dual
    form.  ``L_prior`` is supplied as an :class:`EdgeLaplacian` built from
    the meta-analysis prior.  Features are standardized on the training
    data and predictions are returned in the original target units.

    Parameters
    ----------
    alpha1 : float
        Ridge penalty λ1 (applied to the standardized feature space).
    alpha2 : float
        Network-constraint strength λ2.
    edge_laplacian : EdgeLaplacian, optional
        Precomputed prior penalty.  If None, the method reduces to plain
        ridge (λ2 is ignored).
    n_rois : int
        Atlas size (used for saliency export).
    standardize : bool
        Standardize features on the training split (default True).
    """

    def __init__(
        self,
        alpha1: float = 1.0,
        alpha2: float = 1.0,
        edge_laplacian: Optional[EdgeLaplacian] = None,
        n_rois: int = 116,
        standardize: bool = True,
    ) -> None:
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.edge_laplacian = edge_laplacian
        self.n_rois = int(n_rois)
        self.standardize = standardize

        self.dual_coef_: Optional[np.ndarray] = None
        self.scaler_: Optional[StandardScaler] = None
        self.target_mean_ = 0.0
        self.target_std_ = 1.0
        self.pairs_: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        self.x_inactive_: Optional[np.ndarray] = None
        self.kernel_scale_ = 1.0
        self._eig: Optional[_LaplacianEig] = None
        self.n_features_ = 0

    def _active_all(self) -> np.ndarray:
        """Feature indices of the active FC+SC blocks (empty if no prior)."""
        if self.edge_laplacian is None or self.edge_laplacian.n_active == 0:
            return np.empty(0, dtype=np.int64)
        lap = self.edge_laplacian
        return np.concatenate([lap.active_indices, lap.active_indices + lap.n_edges])

    def _inactive_mask(self, n_features: int) -> np.ndarray:
        """Boolean mask of the ridge-only (inactive) feature columns."""
        mask = np.ones(n_features, dtype=bool)
        mask[self._active_all()] = False
        return mask

    def fit(self, x: np.ndarray, y: np.ndarray) -> "NetworkConstrainedRidge":
        """Fit the dual solution on (x, y).

        Parameters
        ----------
        x : np.ndarray, shape (n, 2 * n_edges)
            Concatenated FC+SC edge features (raw or standardized).
        y : np.ndarray, shape (n,)
            Target values.
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if self.standardize:
            self.scaler_ = StandardScaler()
            x = self.scaler_.fit_transform(x)
        self.target_mean_ = float(y.mean())
        y_std = float(y.std())
        self.target_std_ = y_std if y_std >= 1e-8 else 1.0
        y_z = (y - self.target_mean_) / self.target_std_

        self.n_features_ = x.shape[1]
        self.kernel_scale_ = float(max(1, x.shape[1]))
        self._eig = (
            factor_laplacian_eig(self.edge_laplacian)
            if self.edge_laplacian is not None
            else None
        )
        active_all = self._active_all()
        self.x_inactive_ = x[:, self._inactive_mask(x.shape[1])]
        self.pairs_ = _whitened_pairs(x[:, active_all], self._eig) if len(active_all) else []

        quad = _quadratic_form(self.pairs_, self.alpha1, self.alpha2)
        if quad.shape[0] == 0:  # no prior -> pure ridge (empty active block)
            quad = np.zeros((len(y_z), len(y_z)), dtype=np.float64)
        if self.x_inactive_.shape[1] > 0:
            quad = quad + (self.x_inactive_ @ self.x_inactive_.T) / self.alpha1
        quad = quad / self.kernel_scale_
        alpha = np.linalg.solve(np.eye(len(y)) + quad, y_z)
        self.dual_coef_ = alpha
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict targets for new subjects in the original target units."""
        if self.dual_coef_ is None:
            raise RuntimeError("predict() called before fit()")
        x = np.asarray(x, dtype=np.float64)
        if self.scaler_ is not None:
            x = self.scaler_.transform(x)
        pred_z = self._cross_quadratic_forms(x) @ self.dual_coef_
        return pred_z * self.target_std_ + self.target_mean_

    def _cross_quadratic_forms(self, x: np.ndarray) -> np.ndarray:
        """H_test = X_test P^-1 X_train^T (n_test x n_train), scaled."""
        active_all = self._active_all()
        if len(active_all):
            out = _cross_quadratic(
                self.pairs_, x[:, active_all], self._eig, self.alpha1, self.alpha2
            )
        else:
            out = np.zeros((x.shape[0], self.x_inactive_.shape[0]), dtype=np.float64)
        if self.x_inactive_.shape[1] > 0:
            mask = self._inactive_mask(x.shape[1])
            out = out + (x[:, mask] @ self.x_inactive_.T) / self.alpha1
        return out / self.kernel_scale_

    def beta(self) -> np.ndarray:
        """Recover the primal weight vector on standardized features."""
        if self.dual_coef_ is None:
            raise RuntimeError("beta() called before fit()")
        alpha = self.dual_coef_
        beta = np.zeros(self.n_features_, dtype=np.float64)
        if len(self.pairs_):
            lap = self.edge_laplacian
            beta_fc, beta_sc = _primal_active_weights(
                self.pairs_, self._eig, alpha, self.alpha1, self.alpha2,
                self.kernel_scale_,
            )
            beta[lap.active_indices] = beta_fc
            beta[lap.active_indices + lap.n_edges] = beta_sc
        if self.x_inactive_.shape[1] > 0:
            mask = self._inactive_mask(self.n_features_)
            beta[mask] = (self.x_inactive_.T @ alpha) / (self.alpha1 * self.kernel_scale_)
        return beta

    def node_saliency(self) -> np.ndarray:
        """ROI-level saliency of the learned weights (biomarker vector)."""
        return node_saliency_from_beta(self.beta(), self.n_rois)


def fit_predict_network_constrained(
    x: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    edge_laplacian: EdgeLaplacian,
    alpha1_grid: Iterable[float],
    alpha2_grid: Iterable[float],
    laplacian_eig: Optional[_LaplacianEig] = None,
    ib_tracker: Optional[object] = None,
) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
    """Nested hyperparameter selection for the network-constrained ridge.

    Implements the *exact* AAAI baseline protocol (scripts/16, scripts/27):
    feature standardization and target normalization are fitted on the inner
    training partition only, the (λ1, λ2) pair is selected on the inner
    validation partition, and the final model is refit on train+val before
    predicting the outer test split.

    Parameters
    ----------
    x : np.ndarray, shape (n, 2 * n_edges)
        Raw concatenated FC+SC edge features (all subjects).
    y : np.ndarray, shape (n,)
        Raw target scores.
    train_idx / val_idx / test_idx : np.ndarray
        Partition indices (leakage-free by construction).
    edge_laplacian : EdgeLaplacian
        Prior penalty (may be empty -> plain ridge).
    alpha1_grid / alpha2_grid : Iterable[float]
        λ1 / λ2 candidate grids.
    laplacian_eig : _LaplacianEig, optional
        Precomputed eigen-decomposition of the Laplacian (recomputed from
        edge_laplacian if omitted).
    ib_tracker : IBEpochTracker, optional
        The ridge solve is closed-form (no epochs); when a tracker is given,
        the converged Information Bottleneck metrics are computed once on the
        model's latent representation z = x_fit^std @ beta and stored in
        ``tracker.final``.

    Returns
    -------
    Tuple[np.ndarray, float, float, float, np.ndarray]
        (test predictions in original units, best λ1, best λ2,
        best validation RMSE, dev-fit beta on standardized features).
    """
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if laplacian_eig is None:
        laplacian_eig = factor_laplacian_eig(edge_laplacian)
    eig = laplacian_eig

    n_edges = eig.n_edges
    n_active = eig.n_active
    n_features = x.shape[1]
    scale = float(max(1, n_features))
    active_all = (
        np.concatenate([edge_laplacian.active_indices, edge_laplacian.active_indices + n_edges])
        if n_active > 0
        else np.empty(0, dtype=np.int64)
    )
    inactive_mask = np.ones(n_features, dtype=bool)
    inactive_mask[active_all] = False

    # ---- inner training partition (selection) ----
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x[train_idx]).astype(np.float64, copy=False)
    y_mean, y_std = float(y[train_idx].mean()), float(y[train_idx].std())
    y_std = y_std if y_std >= 1e-8 else 1.0
    y_z = (y[train_idx] - y_mean) / y_std

    if n_active == 0:
        pairs_train: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        x_inactive_train = x_train
    else:
        pairs_train = _whitened_pairs(x_train[:, active_all], eig)
        x_inactive_train = x_train[:, inactive_mask]
    quad_inactive_train = x_inactive_train @ x_inactive_train.T
    z_test_val = scaler.transform(x[val_idx]).astype(np.float64, copy=False)

    alpha1_grid = [float(a) for a in alpha1_grid]
    alpha2_grid = [float(a) for a in alpha2_grid]
    tau_groups = _denominator_grid(alpha1_grid, alpha2_grid)

    best = (float("inf"), None, None)
    for tau, candidates in tau_groups.items():
        # B(τ) = Σ_pairs z diag(1/(1 + τ μ)) z^T, shared by all λ1 in this τ.
        b_mat = np.zeros((len(y_z), len(y_z)), dtype=np.float64)
        for z, mu, _, _ in pairs_train:
            denom = 1.0 + tau * mu
            b_mat += (z / denom) @ z.T
        # H_test = [B_test(τ) + X_val_i X_train_i^T] / (λ1·s): both the
        # active-block and the ridge-only inactive-block cross terms must
        # enter the validation predictions (the refit path below mirrors
        # this).  B(τ) is the active part, A_test_i the inactive part.
        b_test = np.zeros((len(z_test_val), len(y_z)), dtype=np.float64)
        if n_active > 0:
            eval_pairs = _whitened_pairs(z_test_val[:, active_all], eig)
            for (z, mu, _, _), (z_eval, _, _, _) in zip(pairs_train, eval_pairs):
                denom = 1.0 + tau * mu
                b_test += (z_eval / denom) @ z.T
        if x_inactive_train.shape[1] > 0:
            b_test = b_test + z_test_val[:, inactive_mask] @ x_inactive_train.T
        c_vals, c_vecs = dense_eigh(b_mat + quad_inactive_train)
        for alpha1, alpha2 in candidates:
            # H = [B(τ) + A_i]/(λ1·s);  (I + H) α = y
            #  <=>  (C + λ1·s·I) sol = y,  α = λ1·s·sol,  ŷ = H_test α
            gamma = alpha1 * scale
            sol = c_vecs @ ((c_vecs.T @ y_z) / (c_vals + gamma))
            pred_z = b_test @ sol
            pred = pred_z * y_std + y_mean
            rmse_val = float(np.sqrt(np.mean((y[val_idx] - pred) ** 2)))
            candidate = (rmse_val, alpha1, alpha2)
            if candidate < best:
                best = candidate
    best_rmse, best_alpha1, best_alpha2 = best
    if best_alpha1 is None or best_alpha2 is None:
        raise RuntimeError("No (λ1, λ2) candidate was selected")

    # ---- refit on train + validation, predict outer test ----
    fit_idx = np.concatenate([train_idx, val_idx])
    final_scaler = StandardScaler()
    x_fit = final_scaler.fit_transform(x[fit_idx]).astype(np.float64, copy=False)
    x_test = final_scaler.transform(x[test_idx]).astype(np.float64, copy=False)
    fit_mean, fit_std = float(y[fit_idx].mean()), float(y[fit_idx].std())
    fit_std = fit_std if fit_std >= 1e-8 else 1.0
    y_fit_z = (y[fit_idx] - fit_mean) / fit_std

    if n_active == 0:
        pairs_fit: List[Tuple[np.ndarray, np.ndarray, float, float]] = []
        x_inactive_fit = x_fit
    else:
        pairs_fit = _whitened_pairs(x_fit[:, active_all], eig)
        x_inactive_fit = x_fit[:, inactive_mask]
    quad_fit = _quadratic_form(pairs_fit, best_alpha1, best_alpha2)
    if quad_fit.shape[0] == 0:  # no prior -> pure ridge (empty active block)
        quad_fit = np.zeros((len(y_fit_z), len(y_fit_z)), dtype=np.float64)
    quad_fit = quad_fit + (x_inactive_fit @ x_inactive_fit.T) / best_alpha1
    quad_fit = quad_fit / scale
    alpha = np.linalg.solve(quad_fit + np.eye(len(y_fit_z)), y_fit_z)
    pred_z = (
        _cross_quadratic(pairs_fit, x_test[:, active_all], eig, best_alpha1, best_alpha2) @ alpha
        if n_active > 0
        else np.zeros(len(x_test))
    )
    if x_inactive_fit.shape[1] > 0:
        pred_z = pred_z + ((x_test[:, inactive_mask] @ x_inactive_fit.T) @ alpha) / best_alpha1
    pred_z = pred_z / scale
    pred = pred_z * fit_std + fit_mean

    beta_dev = np.zeros(n_features, dtype=np.float64)
    if n_active > 0:
        beta_fc, beta_sc = _primal_active_weights(
            pairs_fit, eig, alpha, best_alpha1, best_alpha2, scale
        )
        beta_dev[edge_laplacian.active_indices] = beta_fc
        beta_dev[edge_laplacian.active_indices + n_edges] = beta_sc
    if x_inactive_fit.shape[1] > 0:
        beta_dev[inactive_mask] = (x_inactive_fit.T @ alpha) / (best_alpha1 * scale)

    if ib_tracker is not None:
        from metascfc.metrics import information_bottleneck_metrics
        latent = x_fit @ beta_dev  # scalar model representation z = X beta
        reference_var = float(x_fit.var(axis=0, ddof=1).mean())
        noise_floor = 0.05
        sigma_nu_sq = max(noise_floor, 1e-8) * max(reference_var, 1e-12)
        rate = float(0.5 * np.log1p(float(latent.var(ddof=1)) / sigma_nu_sq))
        probe = information_bottleneck_metrics(latent, y_fit_z, noise_floor=noise_floor)
        ib_tracker.final = {
            "I_XZ": rate,
            "I_ZY": probe["I_ZY"],
            "probe_r2": probe["probe_r2"],
        }
        ib_tracker.alpha_final = [float(best_alpha2 / best_alpha1)]

    return np.asarray(pred), best_alpha1, best_alpha2, best_rmse, beta_dev
