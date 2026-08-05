from typing import Optional

import torch
import torch.nn.functional as F


def pearson_corr_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = pred.flatten()
    target = target.flatten()
    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    r_num = (pred_centered * target_centered).sum()
    r_den = torch.sqrt((pred_centered ** 2).sum() * (target_centered ** 2).sum() + eps)
    r = r_num / r_den
    return 1.0 - r


def kl_prior_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = pred.flatten()
    target = target.flatten()
    p = F.softmax(pred, dim=0)
    t = F.softmax(target, dim=0)
    p = p.clamp(min=eps)
    t = t.clamp(min=eps)
    return (p * (p.log() - t.log())).sum()


def mse_prior_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(pred.flatten(), target.flatten())


def topk_overlap_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    k: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred_flat = pred.flatten()
    target_flat = target.flatten()

    pred_topk = torch.topk(pred_flat, k).indices
    target_topk = torch.topk(target_flat, k).indices

    pred_mask = torch.zeros_like(pred_flat)
    pred_mask[pred_topk] = 1.0
    target_mask = torch.zeros_like(target_flat)
    target_mask[target_topk] = 1.0

    intersection = (pred_mask * target_mask).sum()
    union = ((pred_mask + target_mask) > 0).sum() + eps
    return 1.0 - intersection / union


def task_loss_classification(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(pred, target)


def task_loss_regression(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(pred.squeeze(), target.squeeze())


def auxiliary_task_loss(
    fc_pred: Optional[torch.Tensor],
    sc_pred: Optional[torch.Tensor],
    target: torch.Tensor,
    alpha_aux: float = 0.1,
    task: str = "classification",
    eps: float = 1e-8,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=target.device)
    loss_fn = task_loss_classification if task == "classification" else task_loss_regression
    if fc_pred is not None:
        loss = loss + alpha_aux * loss_fn(fc_pred, target)
    if sc_pred is not None:
        loss = loss + alpha_aux * loss_fn(sc_pred, target)
    return loss


def total_prior_loss(
    node_loss: Optional[torch.Tensor] = None,
    module_loss: Optional[torch.Tensor] = None,
    edge_loss: Optional[torch.Tensor] = None,
    lambda_node: float = 0.0,
    lambda_module: float = 0.0,
    lambda_edge: float = 0.0,
) -> torch.Tensor:
    total = torch.tensor(0.0)
    if node_loss is not None and lambda_node > 0:
        total = total + lambda_node * node_loss
    if module_loss is not None and lambda_module > 0:
        total = total + lambda_module * module_loss
    if edge_loss is not None and lambda_edge > 0:
        total = total + lambda_edge * edge_loss
    return total
