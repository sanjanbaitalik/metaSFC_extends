"""ICLR 2027 methodological backbones.

Method 1 (implemented)
----------------------
Network-Constrained Prior-Laplacian Ridge (linear/hybrid). The ROI-level
meta-analysis prior is converted into a quadratic penalty on the predictive
weights via a line-graph lift of the prior adjacency into edge-feature
space:  min_β ||y - Xβ||^2 + λ1||β||^2 + λ2 β^T L_prior β.

Method 2 (implemented)
----------------------
Prior-Gated Graph Attention Network (Meta-GAT): the prior is injected
directly into the attention logits, e_ij = LeakyReLU(a^T [W h_i || W h_j])
+ γ (p_i + p_j), with γ a learnable temperature per layer.

Method 3 (implemented)
----------------------
Two-Stage Biomarker-Guided Kernel Ridge: per-split Stage-1 node saliency
(AAAI GCN exports) is lifted to the connectome edge space and used as an
explicit feature-space projector; a Kernel Ridge (RBF) is trained on the
gated features.

See README.md -> "ICLR 2027 Methodological Extensions" and docs/.
"""

from .meta_gat import (
    MetaGAT,
    MetaGATConfig,
    PriorGatedGATLayer,
    build_candidate_grid,
    build_split_graph,
    fit_predict_meta_gat,
    gradient_node_saliency,
)
from .network_constrained_ridge import (
    EdgeLaplacian,
    NetworkConstrainedRidge,
    build_edge_laplacian,
    build_prior_adjacency,
    fit_predict_network_constrained,
    node_saliency_from_beta,
)
from .two_stage_kernel_ridge import (
    KRRConfig,
    extract_upper,
    fit_predict_two_stage_krr,
    lift_node_saliency_to_edges,
    load_split_node_saliency,
    upper_triangle_indices,
)

__all__ = [
    "EdgeLaplacian",
    "KRRConfig",
    "MetaGAT",
    "MetaGATConfig",
    "NetworkConstrainedRidge",
    "PriorGatedGATLayer",
    "build_candidate_grid",
    "build_edge_laplacian",
    "build_prior_adjacency",
    "build_split_graph",
    "extract_upper",
    "fit_predict_meta_gat",
    "fit_predict_network_constrained",
    "fit_predict_two_stage_krr",
    "gradient_node_saliency",
    "lift_node_saliency_to_edges",
    "load_split_node_saliency",
    "node_saliency_from_beta",
    "upper_triangle_indices",
]
