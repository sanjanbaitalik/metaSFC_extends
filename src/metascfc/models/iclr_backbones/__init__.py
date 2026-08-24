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

Method 4 (implemented, ICLR 2027 LLM-prior pivot)
-------------------------------------------------
LLM-Gated Cross-Modal Graph Attention Transformer: zero-shot LLM-generated
semantic priors (scripts/46) enter the attention logits of a transformer-
style graph attention stack, e_ij = LeakyReLU(a^T [W h_i || W h_j]) +
lambda * (p_i + p_j), with lambda a learnable per-layer temperature and
modality-specific FC/SC projections inside each head.

See README.md -> "ICLR 2027 Methodological Extensions" and docs/.
"""

from ..llm_gated_transformer import (
    LLMGatedConfig,
    LLMGatedTransformer,
    LLMGatedTransformerBlock,
    LLMPriorGatedGATLayer,
    RefitLLMGatedPredictor,
    build_candidate_grid as build_llm_gated_grid,
    fit_predict_llm_gated,
    gradient_node_saliency as llm_gated_node_saliency,
    refit_llm_gated_predictor,
)
from .meta_gat import (
    MetaGAT,
    MetaGATConfig,
    PriorGatedGATLayer,
    RefitMetaGATPredictor,
    build_candidate_grid,
    build_split_graph,
    fit_predict_meta_gat,
    gradient_node_saliency,
    refit_meta_gat_predictor,
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
    RefitKRRPredictor,
    extract_upper,
    fit_predict_two_stage_krr,
    lift_node_saliency_to_edges,
    load_split_node_saliency,
    refit_krr_predictor,
    upper_triangle_indices,
)

__all__ = [
    "EdgeLaplacian",
    "KRRConfig",
    "LLMGatedConfig",
    "LLMGatedTransformer",
    "LLMGatedTransformerBlock",
    "LLMPriorGatedGATLayer",
    "MetaGAT",
    "MetaGATConfig",
    "NetworkConstrainedRidge",
    "PriorGatedGATLayer",
    "RefitKRRPredictor",
    "RefitLLMGatedPredictor",
    "RefitMetaGATPredictor",
    "build_candidate_grid",
    "build_edge_laplacian",
    "build_llm_gated_grid",
    "build_prior_adjacency",
    "build_split_graph",
    "extract_upper",
    "fit_predict_llm_gated",
    "fit_predict_meta_gat",
    "fit_predict_network_constrained",
    "fit_predict_two_stage_krr",
    "gradient_node_saliency",
    "lift_node_saliency_to_edges",
    "llm_gated_node_saliency",
    "load_split_node_saliency",
    "node_saliency_from_beta",
    "refit_krr_predictor",
    "refit_llm_gated_predictor",
    "refit_meta_gat_predictor",
    "upper_triangle_indices",
]
