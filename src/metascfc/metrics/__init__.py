"""Metrics package for MetaSCFC.

Re-exports the classic prediction/alignment metrics (``core``) and the
Information Bottleneck trackers used by the ICLR 2027 dual-task matrix.
"""
from .core import (
    classification_metrics,
    compute_prediction_metrics,
    compute_prior_alignment_metrics,
    correlation_metrics,
    mae,
    pearson_corr,
    rmse,
    spearman_corr,
    topk_jaccard,
)
from .information_bottleneck import (
    IBEpochTracker,
    compression_mi,
    information_bottleneck_metrics,
    predictive_mi,
)

__all__ = [
    "IBEpochTracker",
    "classification_metrics",
    "compression_mi",
    "compute_prediction_metrics",
    "compute_prior_alignment_metrics",
    "correlation_metrics",
    "information_bottleneck_metrics",
    "mae",
    "pearson_corr",
    "predictive_mi",
    "rmse",
    "spearman_corr",
    "topk_jaccard",
]
