from typing import Dict, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr


def pearson_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return 0.0
    r, _ = pearsonr(y_true[mask], y_pred[mask])
    return float(r) if not np.isnan(r) else 0.0


def spearman_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 2:
        return 0.0
    r, _ = spearmanr(y_true[mask], y_pred[mask])
    return float(r) if not np.isnan(r) else 0.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return 0.0
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        sensitivity = 0.0
        specificity = 0.0

    auroc = 0.0
    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            auroc = roc_auc_score(y_true, y_prob)
        except Exception:
            auroc = 0.0

    return {
        "accuracy": float(acc),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "auroc": float(auroc),
    }


def correlation_metrics(
    learned: np.ndarray,
    prior: np.ndarray,
) -> Dict[str, float]:
    return {
        "pearson": pearson_corr(learned, prior),
        "spearman": spearman_corr(learned, prior),
    }


def topk_jaccard(
    learned: np.ndarray,
    prior: np.ndarray,
    k: int,
) -> float:
    topk_learned = set(np.argsort(learned)[-k:])
    topk_prior = set(np.argsort(prior)[-k:])
    intersection = topk_learned & topk_prior
    union = topk_learned | topk_prior
    return len(intersection) / len(union) if union else 0.0


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task: str = "regression",
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    if task == "classification":
        return classification_metrics(y_true, y_pred, y_prob)
    return {
        "pearson": pearson_corr(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
    }


def compute_prior_alignment_metrics(
    learned_saliency: np.ndarray,
    prior_scores: np.ndarray,
    topk: Optional[int] = None,
) -> Dict[str, float]:
    metrics = correlation_metrics(learned_saliency, prior_scores)
    if topk is not None:
        metrics["topk_jaccard"] = topk_jaccard(learned_saliency, prior_scores, k=topk)
    return metrics
