#!/usr/bin/env python3
"""LLM-Gated Cross-Modal Graph Attention Transformer (ICLR 2027, Method 4).

The ICLR 2027 pivot replaces the Neurosynth meta-analysis prior with
zero-shot LLM-generated semantic scores (scripts/46) and injects them into a
transformer-style graph attention stack.  For every directed edge (i -> j)
of the split structural graph:

    e_ij = LeakyReLU(a^T [W h_i || W h_j]) + lambda * (p_i + p_j)

where p is the min-max normalized LLM prior score and lambda is a *learnable*
per-layer temperature (init ``lambda_init``).  As in MetaGAT, the gate lives
outside the LeakyReLU in logit space so the prior cannot be washed out by the
learned attention; unlike Meta-GAT the layer is wrapped in transformer blocks

    h <- LayerNorm(h + PriorGatedAttention(h))
    h <- LayerNorm(h + FFN(h))

so deeper stacks remain trainable, and the FC/SC node features are projected
by modality-specific linear maps before concatenation ("cross-modal"
attention: each head can weight functional and structural evidence
differently while sharing one routing decision per edge).

Everything else mirrors Method 2 exactly: leakage-free split graph (row-wise
top-10% positive group-average SC of the inner training partition,
self-loops), standardized [FC_row | SC_row] node features, nested CV with
inner-validation selection by RMSE and fixed-epoch refit on train+val,
gradient node saliency biomarker, and a refit predictor exposing
``predict``/``attention_mass`` for the faithfulness protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metascfc.benchmark_utils import set_all_seeds
from metascfc.models.iclr_backbones.meta_gat import build_split_graph


# ---------------------------------------------------------------------------
# Hyperparameter container / candidate grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LLMGatedConfig:
    """Hyperparameters of one LLM-gated transformer candidate."""

    hidden: int = 16
    n_layers: int = 2
    heads: int = 4
    dropout: float = 0.2
    lambda_init: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    patience: int = 15
    min_epochs: int = 10
    grad_clip: float = 5.0
    ffn_mult: int = 2


def build_candidate_grid(
    hidden_grid: Sequence[int],
    dropout_grid: Sequence[float],
    lr_grid: Sequence[float],
    **fixed,
) -> List[LLMGatedConfig]:
    out: List[LLMGatedConfig] = []
    for hidden in hidden_grid:
        for dropout in dropout_grid:
            for lr in lr_grid:
                out.append(
                    LLMGatedConfig(hidden=int(hidden), dropout=float(dropout),
                                   learning_rate=float(lr), **fixed)
                )
    return out


# ---------------------------------------------------------------------------
# LLM-gated attention layer (the paper equation)
# ---------------------------------------------------------------------------
class LLMPriorGatedGATLayer(nn.Module):
    """Cross-modal graph attention whose logits are biased by the LLM prior.

        e_uv = LeakyReLU(a^T [W_f h_u | W_s h_v]) + lambda * (p_u + p_v)
        alpha_uv = softmax over incoming edges of v (stable segment softmax)

    W_f / W_s are modality-specific projections applied to the functional and
    structural halves of the node features (cross-modal); a is shared across
    modalities; lambda is a learnable scalar per layer.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int,
        n_nodes: int,
        prior: np.ndarray,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        lambda_init: float = 1.0,
        leaky: float = 0.2,
        dropout: float = 0.0,
        concat: bool = True,
    ) -> None:
        super().__init__()
        if heads < 1 or out_dim < 1 or in_dim < 2:
            raise ValueError("heads/out_dim must be >= 1 and in_dim >= 2")
        self.heads = heads
        self.concat = concat
        self.out_dim = out_dim
        self.leaky = float(leaky)
        self.n_modalities = 2
        if in_dim % self.n_modalities != 0:
            raise ValueError(f"in_dim must be even ([FC|SC]), got {in_dim}")
        half = in_dim // self.n_modalities

        # Modality-specific projections W_f, W_s and shared attention vector a.
        self.W_fc = nn.Parameter(torch.empty(heads, half, out_dim))
        self.W_sc = nn.Parameter(torch.empty(heads, half, out_dim))
        self.a = nn.Parameter(torch.empty(heads, 3 * out_dim))
        for w in (self.W_fc, self.W_sc):
            nn.init.xavier_uniform_(w)
        nn.init.xavier_uniform_(self.a.view(heads, 3 * out_dim))

        # Learnable prior temperature (one scalar per layer).
        self.lambda_gate = nn.Parameter(
            torch.tensor(float(lambda_init), dtype=torch.float32)
        )

        prior = np.asarray(prior, dtype=np.float32).reshape(-1)
        if prior.shape[0] != n_nodes:
            raise ValueError(f"prior has {prior.shape[0]} entries; expected {n_nodes}")
        self.register_buffer("prior", torch.from_numpy(prior))
        self.register_buffer("edge_src", torch.from_numpy(np.asarray(edge_src, dtype=np.int64)))
        self.register_buffer("edge_dst", torch.from_numpy(np.asarray(edge_dst, dtype=np.int64)))
        dst_onehot = np.zeros((len(edge_dst), n_nodes), dtype=np.float32)
        dst_onehot[np.arange(len(edge_dst)), edge_dst] = 1.0
        self.register_buffer("edge_dst_onehot", torch.from_numpy(dst_onehot))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_nodes, _ = x.shape
        device, dtype = x.device, x.dtype
        edge_src, edge_dst = self.edge_src, self.edge_dst
        n_edges = edge_src.shape[0]
        half = x.shape[-1] // 2

        # Gate: lambda * (p_src + p_dst), broadcast over batch and heads.
        gate = self.lambda_gate * (self.prior[edge_src] + self.prior[edge_dst])

        h_fc, h_sc = x[..., :half], x[..., half:]
        proj_fc = torch.einsum("bnd,hdq->bnhq", h_fc, self.W_fc)
        proj_sc = torch.einsum("bnd,hdq->bnhq", h_sc, self.W_sc)
        src = edge_src.reshape(1, -1, 1, 1).expand(batch, -1, self.heads, self.out_dim)
        dst = edge_dst.reshape(1, -1, 1, 1).expand(batch, -1, self.heads, self.out_dim)
        pair = torch.cat(
            [proj_fc.gather(1, src), proj_sc.gather(1, dst),
             0.5 * (proj_fc + proj_sc).gather(1, src)],
            dim=-1,
        )  # (batch, n_edges, heads, 3*out_dim): [W_f h_i | W_s h_j | pooled i]
        logit = torch.einsum("behd,hd->beh", pair, self.a)
        logit = F.leaky_relu(logit, negative_slope=self.leaky)
        logit = logit + gate.reshape(1, n_edges, 1)

        # Stable segment softmax over the incoming edges of each node.
        neg_inf = torch.finfo(dtype).min
        max_per_node = torch.full(
            (batch, self.heads, n_nodes), neg_inf, device=device, dtype=dtype
        )
        max_per_node = torch.index_reduce(
            max_per_node, 2, edge_dst, logit.transpose(1, 2),
            reduce="amax", include_self=False,
        ).nan_to_num(neg_inf)
        edge_index = edge_dst.reshape(1, 1, -1).expand(batch, self.heads, n_edges)
        exp_logit = (logit - max_per_node.gather(2, edge_index).transpose(1, 2)).exp()

        A = self.edge_dst_onehot
        sum_per_edge = torch.einsum(
            "bnh,en->beh", torch.einsum("beh,en->bnh", exp_logit, A), A
        )
        alpha = exp_logit / sum_per_edge.clamp_min(1e-8)
        self._last_alpha = alpha.detach()

        msg = alpha.unsqueeze(-1) * (
            proj_fc.gather(1, src) + proj_sc.gather(1, src)
        )
        agg = torch.einsum("behd,en->bnhd", msg, A)
        out = agg.reshape(batch, n_nodes, self.heads * self.out_dim) \
            if self.concat else agg.mean(dim=2)
        return self.dropout(F.elu(out))


# ---------------------------------------------------------------------------
# Transformer block + full predictor
# ---------------------------------------------------------------------------
class LLMGatedTransformerBlock(nn.Module):
    """Residual block at constant width: LN(h + gated attention) -> LN(h + FFN)."""

    def __init__(
        self,
        dim: int,
        out_dim: int,
        heads: int,
        n_nodes: int,
        prior: np.ndarray,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
        lambda_init: float,
        dropout: float,
        ffn_mult: int,
    ) -> None:
        super().__init__()
        self.attn = LLMPriorGatedGATLayer(
            in_dim=dim, out_dim=out_dim, heads=heads, n_nodes=n_nodes,
            prior=prior, edge_src=edge_src, edge_dst=edge_dst,
            lambda_init=lambda_init, dropout=dropout, concat=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        ffn_hidden = max(ffn_mult * dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)) if dropout > 0.0 else nn.Identity(),
            nn.Linear(ffn_hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x + self.attn(x))
        return self.norm2(h + self.ffn(h))


class LLMGatedTransformer(nn.Module):
    """Cross-modal LLM-gated graph attention transformer -> scalar prediction.

    Architecture: input projection to d_model = hidden * heads, then
    ``n_layers`` residual blocks (gated attention -> LN -> FFN -> LN), mean
    pooling over ROIs, and an MLP head.  Constant block width keeps the
    residual connections active at any depth.
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        config: LLMGatedConfig,
        prior: np.ndarray,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
    ) -> None:
        super().__init__()
        self.d_model = config.hidden * config.heads
        self.input_proj = nn.Linear(in_dim, self.d_model)
        self.blocks = nn.ModuleList([
            LLMGatedTransformerBlock(
                dim=self.d_model, out_dim=config.hidden, heads=config.heads,
                n_nodes=n_nodes, prior=prior, edge_src=edge_src,
                edge_dst=edge_dst, lambda_init=config.lambda_init,
                dropout=config.dropout, ffn_mult=config.ffn_mult,
            )
            for _ in range(config.n_layers)
        ])
        self.final_norm = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        self.readout = nn.Sequential(
            nn.Linear(self.d_model, self.d_model), nn.GELU(),
            nn.Linear(self.d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        pooled = self.final_norm(h).mean(dim=1)
        return self.readout(self.dropout(pooled)).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers (identical protocol to meta_gat.py)
# ---------------------------------------------------------------------------
def _train_with_early_stopping(model, x_train, y_train_z, x_val, y_val_raw,
                               y_val_mean, y_val_std, config, device):
    model = model.to(device)
    x_train, y_train_z = x_train.to(device), y_train_z.to(device)
    x_val, y_val_raw = x_val.to(device), y_val_raw.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    mse = nn.MSELoss()
    best_rmse, best_state, best_epoch, wait = float("inf"), None, 0, 0
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = mse(model(x_train), y_train_z)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if epoch + 1 < config.min_epochs:
            continue
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val) * y_val_std + y_val_mean
            rmse = float(torch.sqrt(torch.mean((pred_val - y_val_raw) ** 2)))
        if rmse < best_rmse - 1e-8:
            best_rmse, best_epoch, wait = rmse, epoch + 1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    return best_rmse, best_epoch


def _train_fixed_epochs(model, x_fit, y_fit_z, config, n_epochs, device):
    model = model.to(device)
    x_fit, y_fit_z = x_fit.to(device), y_fit_z.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    mse = nn.MSELoss()
    for _ in range(int(n_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = mse(model(x_fit), y_fit_z)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
    model.eval()


# ---------------------------------------------------------------------------
# Biomarker: gradient saliency (identical contract to meta_gat.py)
# ---------------------------------------------------------------------------
def gradient_node_saliency(model: LLMGatedTransformer, x_fit: torch.Tensor,
                           device: torch.device) -> np.ndarray:
    """Node saliency = mean |dy/dx_i| over fit subjects, min-max normalized."""
    model.eval()
    n_subjects, n_nodes, in_dim = x_fit.shape
    accum = torch.zeros((n_nodes, in_dim), device=device, dtype=torch.float32)
    for s in range(n_subjects):
        xi = x_fit[s:s + 1].to(device).requires_grad_(True)
        pred = model(xi)
        grad = torch.autograd.grad(pred.sum(), xi)[0][0].abs()
        accum += grad
    node_saliency = (accum / n_subjects).sum(dim=1).detach().cpu().numpy()
    span = node_saliency.max() - node_saliency.min()
    if span > 1e-12:
        node_saliency = (node_saliency - node_saliency.min()) / span
    else:
        node_saliency = np.zeros(n_nodes, dtype=np.float64)
    return node_saliency


# ---------------------------------------------------------------------------
# Nested cross-validation entry point (script-47 workhorse)
# ---------------------------------------------------------------------------
def fit_predict_llm_gated(
    fc: np.ndarray,
    sc: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    prior: np.ndarray,
    hidden_grid: Sequence[int],
    dropout_grid: Sequence[float],
    lr_grid: Sequence[float],
    device: torch.device,
    top_percent: float = 10.0,
    n_layers: int = 2,
    heads: int = 4,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    patience: int = 15,
    min_epochs: int = 10,
    lambda_init: float = 1.0,
    grad_clip: float = 5.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float], float, int, np.ndarray, int]:
    """Nested selection + refit for one outer split (leakage-free).

    Protocol identical to ``fit_predict_meta_gat``: standardize on inner
    train only, threshold the split graph on inner-train SC only, select the
    (hidden, dropout, lr) candidate by inner-validation RMSE, refit on
    train+val for the selected epoch budget, predict raw-unit test scores,
    export gradient node saliency.

    Returns
    -------
    (test predictions, best config dict, best val RMSE, best epoch,
     node saliency, number of parameters)
    """
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if seed is not None:
        set_all_seeds(seed)

    fc = np.asarray(fc, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    n_rois = fc.shape[1]
    x = np.concatenate([fc, sc], axis=2)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    def make_scaled(part_idx: np.ndarray, mean, std) -> torch.Tensor:
        scaled = ((x[part_idx].reshape(len(part_idx), -1) - mean) / std)
        return torch.from_numpy(
            scaled.reshape(len(part_idx), n_rois, -1).astype(np.float32)
        )

    train_flat = x[train_idx].reshape(len(train_idx), -1)
    mean, std = train_flat.mean(axis=0, keepdims=True), train_flat.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x_train = make_scaled(train_idx, mean, std)
    x_val = make_scaled(val_idx, mean, std)

    y_mean, y_std = float(y[train_idx].mean()), float(y[train_idx].std())
    y_std = y_std if y_std >= 1e-8 else 1.0
    y_train_z = torch.from_numpy(((y[train_idx] - y_mean) / y_std).astype(np.float32))
    y_val_raw = torch.from_numpy(y[val_idx].astype(np.float32))

    edge_src, edge_dst = build_split_graph(sc[train_idx], top_percent)
    candidates = build_candidate_grid(
        hidden_grid, dropout_grid, lr_grid, n_layers=n_layers, heads=heads,
        weight_decay=weight_decay, epochs=epochs, patience=patience,
        min_epochs=min_epochs, lambda_init=lambda_init, grad_clip=grad_clip,
    )
    best_rmse, best_cfg, best_epoch = float("inf"), None, 0
    for cfg in candidates:
        model = LLMGatedTransformer(n_rois, x.shape[2], cfg, prior, edge_src, edge_dst)
        val_rmse, epoch_used = _train_with_early_stopping(
            model, x_train, y_train_z, x_val, y_val_raw, y_mean, y_std, cfg, device,
        )
        if val_rmse < best_rmse - 1e-12:
            best_rmse, best_cfg, best_epoch = val_rmse, cfg, epoch_used
    if best_cfg is None:
        raise RuntimeError("No candidate configuration was selected")

    fit_idx = np.concatenate([train_idx, val_idx])
    fit_flat = x[fit_idx].reshape(len(fit_idx), -1)
    mean, std = fit_flat.mean(axis=0, keepdims=True), fit_flat.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x_fit = make_scaled(fit_idx, mean, std)
    x_test = make_scaled(test_idx, mean, std)
    fit_mean, fit_std = float(y[fit_idx].mean()), float(y[fit_idx].std())
    fit_std = fit_std if fit_std >= 1e-8 else 1.0
    y_fit_z = torch.from_numpy(((y[fit_idx] - fit_mean) / fit_std).astype(np.float32))

    edge_src, edge_dst = build_split_graph(sc[fit_idx], top_percent)
    final_model = LLMGatedTransformer(n_rois, x.shape[2], best_cfg, prior, edge_src, edge_dst)
    _train_fixed_epochs(final_model, x_fit, y_fit_z, best_cfg, best_epoch, device)
    final_model.eval()
    with torch.no_grad():
        pred = (final_model(x_test.to(device)).detach().cpu().numpy() * fit_std
                + fit_mean).astype(np.float64)

    saliency = gradient_node_saliency(final_model, x_fit, device)
    n_params = sum(p.numel() for p in final_model.parameters())
    best_cfg_dict = {
        "hidden": best_cfg.hidden, "dropout": best_cfg.dropout,
        "learning_rate": best_cfg.learning_rate, "n_layers": best_cfg.n_layers,
    }
    return pred, best_cfg_dict, best_rmse, best_epoch, saliency, n_params


# ---------------------------------------------------------------------------
# Refit predictor (faithfulness / perturbation protocol)
# ---------------------------------------------------------------------------
class RefitLLMGatedPredictor:
    """Trained LLM-gated transformer + scalers, ready for masking tests."""

    def __init__(self, model, x_mean, x_std, fit_mean, fit_std, n_rois, device):
        self.model = model
        self.x_mean = x_mean.reshape(1, -1)
        self.x_std = x_std.reshape(1, -1)
        self.fit_mean = float(fit_mean)
        self.fit_std = float(fit_std)
        self.n_rois = int(n_rois)
        self.device = device

    def _features(self, fc: np.ndarray, sc: np.ndarray) -> torch.Tensor:
        n = len(fc)
        x = np.concatenate([fc, sc], axis=2).reshape(n, -1).astype(np.float64)
        x = ((x - self.x_mean) / self.x_std).reshape(n, self.n_rois, -1).astype(np.float32)
        return torch.from_numpy(x).to(self.device)

    @torch.no_grad()
    def predict(self, fc: np.ndarray, sc: np.ndarray) -> np.ndarray:
        self.model.eval()
        pred = self.model(self._features(fc, sc)).detach().cpu().numpy()
        return (pred * self.fit_std + self.fit_mean).astype(np.float64)

    @torch.no_grad()
    def attention_mass(self, fc: np.ndarray, sc: np.ndarray) -> np.ndarray:
        """Mean attention coefficient per ROI over incident edges (L1-normalized)."""
        xt = self._features(fc, sc)
        self.model.eval()
        _ = self.model(xt)
        masses = []
        for block in self.model.blocks:
            layer = block.attn
            alpha = layer._last_alpha.mean(dim=(0, 2)).cpu().numpy()
            A_dst = layer.edge_dst_onehot.cpu().numpy()
            A_src = np.zeros_like(A_dst)
            A_src[np.arange(A_src.shape[0]), layer.edge_src.cpu().numpy()] = 1.0
            mass = ((A_dst.T @ alpha) + (A_src.T @ alpha)) / (
                A_dst.sum(axis=0) + A_src.sum(axis=0)
            )
            total = float(mass.sum())
            masses.append(mass / total if total > 1e-12 else mass)
        return np.asarray(np.mean(masses, axis=0), dtype=np.float64)


def refit_llm_gated_predictor(
    fc: np.ndarray,
    sc: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    config: LLMGatedConfig,
    prior: np.ndarray,
    device: torch.device,
    n_epochs: int,
    top_percent: float = 10.0,
    seed: Optional[int] = None,
) -> RefitLLMGatedPredictor:
    """Refit with a recorded configuration (faithfulness protocol entry)."""
    fit_idx = np.asarray(fit_idx, dtype=int)
    if seed is not None:
        set_all_seeds(seed)
    fc = np.asarray(fc, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    n_rois = fc.shape[1]
    x = np.concatenate([fc, sc], axis=2)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    flat = x[fit_idx].reshape(len(fit_idx), -1)
    mean, std = flat.mean(axis=0, keepdims=True), flat.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    x_fit = torch.from_numpy(
        ((x[fit_idx].reshape(len(fit_idx), -1) - mean) / std)
        .reshape(len(fit_idx), n_rois, -1).astype(np.float32)
    )
    fit_mean, fit_std = float(y[fit_idx].mean()), float(y[fit_idx].std())
    fit_std = fit_std if fit_std >= 1e-8 else 1.0
    y_fit_z = torch.from_numpy(((y[fit_idx] - fit_mean) / fit_std).astype(np.float32))

    edge_src, edge_dst = build_split_graph(sc[fit_idx], top_percent)
    model = LLMGatedTransformer(n_rois, x.shape[2], config, prior, edge_src, edge_dst)
    _train_fixed_epochs(model, x_fit, y_fit_z, config, int(n_epochs), device)
    model.eval()
    return RefitLLMGatedPredictor(model, mean, std, fit_mean, fit_std, n_rois, device)


__all__ = [
    "LLMGatedConfig",
    "LLMGatedTransformer",
    "LLMGatedTransformerBlock",
    "LLMPriorGatedGATLayer",
    "RefitLLMGatedPredictor",
    "build_candidate_grid",
    "fit_predict_llm_gated",
    "gradient_node_saliency",
    "refit_llm_gated_predictor",
]
