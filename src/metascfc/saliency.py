from typing import Tuple

import torch


def normalize_torch_scores(x: torch.Tensor, eps: float = 1e-8, dim: int = -1) -> torch.Tensor:
    """Min-max normalize nonnegative scores along dim."""
    x = x.clamp(min=0)
    x_min = x.amin(dim=dim, keepdim=True)
    x_max = x.amax(dim=dim, keepdim=True)
    return torch.where((x_max - x_min) > eps, (x - x_min) / (x_max - x_min + eps), torch.zeros_like(x))


def node_saliency_from_coupling_vector(
    coupling_vector: torch.Tensor,
    mode: str = "abs",
) -> torch.Tensor:
    """
    Convert a corresponding-ROI FC-SC coupling vector into node saliency.

    Supports [N] or batched [B, N] tensors.
    """
    if mode == "abs":
        return coupling_vector.abs()
    if mode == "relu":
        return coupling_vector.clamp(min=0)
    if mode == "sigmoid":
        return torch.sigmoid(coupling_vector)
    if mode == "minmax_abs":
        v = coupling_vector.abs()
        dim = -1
        v_min = v.amin(dim=dim, keepdim=True)
        v_max = v.amax(dim=dim, keepdim=True)
        return torch.where((v_max - v_min) > 1e-8, (v - v_min) / (v_max - v_min + 1e-8), torch.zeros_like(v))
    raise ValueError(f"Unknown mode: {mode}")


def aggregate_node_saliency_from_interactions(
    O: torch.Tensor,
    mode: str = "mean_abs",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert an FC-SC interaction matrix into functional and structural node saliency.

    Supports unbatched [N, N] or batched [B, N, N] tensors.
    Returns two tensors of shape [N] or [B, N].
    """
    if O.dim() not in (2, 3):
        raise ValueError(f"O must be [N,N] or [B,N,N], got shape {tuple(O.shape)}")

    row_dim = -1  # structural dimension j in O[i, j]
    col_dim = -2  # functional dimension i in O[i, j]

    if mode == "mean_abs":
        fc_sal = O.abs().mean(dim=row_dim)
        sc_sal = O.abs().mean(dim=col_dim)
    elif mode == "sum_abs":
        fc_sal = O.abs().sum(dim=row_dim)
        sc_sal = O.abs().sum(dim=col_dim)
    elif mode == "max_abs":
        fc_sal = O.abs().max(dim=row_dim).values
        sc_sal = O.abs().max(dim=col_dim).values
    elif mode == "attention_mean":
        fc_sal = O.mean(dim=row_dim)
        sc_sal = O.mean(dim=col_dim)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return fc_sal, sc_sal


def aggregate_module_saliency_from_vector(
    coupling_vector: torch.Tensor,
    roi_to_module: torch.Tensor,
    num_modules: int,
    agg: str = "mean",
) -> torch.Tensor:
    """
    Aggregate corresponding-ROI coupling saliency [N] or [B,N] into module saliency.
    """
    if coupling_vector.dim() == 1:
        x = coupling_vector.unsqueeze(0)
        squeeze = True
    elif coupling_vector.dim() == 2:
        x = coupling_vector
        squeeze = False
    else:
        raise ValueError(f"coupling_vector must be [N] or [B,N], got {tuple(coupling_vector.shape)}")

    if roi_to_module.numel() != x.shape[-1]:
        raise ValueError(f"roi_to_module length {roi_to_module.numel()} != number of ROIs {x.shape[-1]}")

    out = []
    for m in range(num_modules):
        mask = roi_to_module == m
        if not torch.any(mask):
            val = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        elif agg == "mean":
            val = x[:, mask].mean(dim=1)
        elif agg == "sum":
            val = x[:, mask].sum(dim=1)
        elif agg == "max":
            val = x[:, mask].max(dim=1).values
        else:
            raise ValueError(f"Unknown module aggregation: {agg}")
        out.append(val)
    result = torch.stack(out, dim=1)
    return result.squeeze(0) if squeeze else result


def aggregate_module_saliency(
    O: torch.Tensor,
    roi_to_module: torch.Tensor,
    num_modules: int,
) -> torch.Tensor:
    """
    Aggregate [N,N] or [B,N,N] edge interaction saliency into [M,M] or [B,M,M].
    """
    if O.dim() == 2:
        x = O.unsqueeze(0)
        squeeze = True
    elif O.dim() == 3:
        x = O
        squeeze = False
    else:
        raise ValueError(f"O must be [N,N] or [B,N,N], got {tuple(O.shape)}")

    B, N, _ = x.shape
    if roi_to_module.numel() != N:
        raise ValueError(f"roi_to_module length {roi_to_module.numel()} != number of ROIs {N}")

    M = torch.zeros(N, num_modules, device=x.device, dtype=x.dtype)
    M[torch.arange(N, device=x.device), roi_to_module] = 1.0
    module_sal = torch.einsum("nm,bnp,pk->bmk", M, x.abs(), M)
    return module_sal.squeeze(0) if squeeze else module_sal


def coupling_vector_to_saliency_vector(
    coupling_vector: torch.Tensor,
    mode: str = "minmax_abs",
) -> torch.Tensor:
    return node_saliency_from_coupling_vector(coupling_vector, mode=mode)
