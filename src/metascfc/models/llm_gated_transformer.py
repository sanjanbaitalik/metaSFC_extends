#!/usr/bin/env python3
"""LLM-Gated Cross-Modal Graph Attention Transformer (ICLR 2027, Method 4).

The ICLR 2027 pivot replaces the Neurosynth meta-analysis prior with
zero-shot LLM-generated semantic scores (scripts/46) and injects them into a
transformer-style graph attention stack with *adaptive prior routing*.  For
every directed edge (i -> j) of the split structural graph:

    e_ij = (1 - alpha) * LeakyReLU(a^T [W_f h_i | W_s h_j])
         + alpha * (p_i + p_j)

where p is the min-max normalized LLM prior score and ``alpha = sigmoid(rho)``
is a learnable per-layer bypass gate (init ``alpha_init``): alpha -> 1 routes
information purely along the prior (trusting it), alpha -> 0 ignores it.
The learned alpha is logged per run and is expected to approach 1 for
target-matched priors and 0 for mismatched ones.

    h <- LayerNorm(h + PriorGatedAttention(h))
    h <- LayerNorm(h + FFN(h))

so deeper stacks remain trainable, and the FC/SC node features are projected
Everything else mirrors Method 2 exactly: leakage-free split graph (row-wise
top-10% positive group-average SC of the inner training partition,
self-loops), standardized [FC_row | SC_row] node features, nested CV with
inner-validation selection by RMSE and fixed-epoch refit on train+val,
gradient node saliency biomarker, and a refit predictor exposing
``predict``/``attention_mass`` for the faithfulness protocol.  Optional
Information Bottleneck tracking (per-epoch I(X;Z) / I(Z;Y) on the pooled
latent, plus the converged values and learned bypass alphas) is enabled by
passing an ``IBEpochTracker``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace as dataclasses_replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metascfc.benchmark_utils import set_all_seeds
from metascfc.metrics import IBEpochTracker
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
    # Validation-Selected Discrete Routing: alpha is a FIXED scalar
    # hyperparameter (no gradient flow), selected per split from a discrete
    # grid by inner-validation RMSE.  0.0 = pure data-driven attention
    # (bypass the prior), 1.0 = pure prior routing.
    alpha: float = 0.5
    # --- deprecated fields of the continuous-gate era (kept ONLY so that
    # --- existing checkpoints / configs deserialize); unused at runtime.
    alpha_init: float = 0.1
    alpha_explore_weight: float = 0.0
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
    """Cross-modal graph attention with adaptive prior routing (bypass gate).

        e_uv = (1 - alpha) * LeakyReLU(a^T [W_f h_u | W_s h_v] ...)
             + alpha * (p_u + p_v)
        alpha_uv = softmax over incoming edges of v (stable segment softmax)

    ``alpha`` is a FIXED scalar hyperparameter (Validation-Selected Discrete
    Routing): it receives no gradient flow and is chosen per split from a
    discrete grid by inner-validation RMSE.  alpha = 1 routes purely along
    the prior (trusts the LLM prior); alpha = 0 ignores it (pure learned
    attention).  This removes the Gradient-Absorption failure mode of the
    continuous gate, whose rho-gradient was flat because the learned branch
    can absorb any gate change.  W_f / W_s are modality-specific projections
    applied to the functional and structural halves of the node features
    (cross-modal); a is shared across modalities.
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
        alpha: float = 0.5,
        leaky: float = 0.2,
        dropout: float = 0.0,
        concat: bool = True,
    ) -> None:
        super().__init__()
        if heads < 1 or out_dim < 1 or in_dim < 2:
            raise ValueError("heads/out_dim must be >= 1 and in_dim >= 2")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
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

        # Fixed bypass gate (hyperparameter - NOT a Parameter, no gradients).
        self.alpha = float(alpha)

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

    @property
    def bypass_alpha(self) -> float:
        """The fixed bypass value alpha of this layer (Python float)."""
        return float(self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n_nodes, _ = x.shape
        device, dtype = x.device, x.dtype
        edge_src, edge_dst = self.edge_src, self.edge_dst
        n_edges = edge_src.shape[0]
        half = x.shape[-1] // 2

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
        learned_logit = F.leaky_relu(
            torch.einsum("behd,hd->beh", pair, self.a), negative_slope=self.leaky
        )

        # Adaptive prior routing: convex mix of learned attention and the
        # LLM-prior gate (p_src + p_dst), alpha learnable per layer.
        alpha = self.alpha  # fixed hyperparameter (no gradient flow)
        prior_gate = self.prior[edge_src] + self.prior[edge_dst]
        logit = (1.0 - alpha) * learned_logit + alpha * prior_gate.reshape(1, -1, 1)

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
        attn = exp_logit / sum_per_edge.clamp_min(1e-8)
        self._last_attention = attn.detach()

        msg = attn.unsqueeze(-1) * (
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
        alpha: float,
        dropout: float,
        ffn_mult: int,
    ) -> None:
        super().__init__()
        self.attn = LLMPriorGatedGATLayer(
            in_dim=dim, out_dim=out_dim, heads=heads, n_nodes=n_nodes,
            prior=prior, edge_src=edge_src, edge_dst=edge_dst,
            alpha=alpha, dropout=dropout, concat=True,
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
        self.alpha_explore_weight = float(config.alpha_explore_weight)
        self.input_proj = nn.Linear(in_dim, self.d_model)
        self.blocks = nn.ModuleList([
            LLMGatedTransformerBlock(
                dim=self.d_model, out_dim=config.hidden, heads=config.heads,
                n_nodes=n_nodes, prior=prior, edge_src=edge_src,
                edge_dst=edge_dst, alpha=config.alpha,
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Pooled penultimate representation Z (batch, d_model).

        The Information Bottleneck metrics are computed on this latent:
        I(X; Z) against the input features and I(Z; Y) against the target.
        """
        h = F.elu(self.input_proj(x))
        for block in self.blocks:
            h = block(h)
        return self.final_norm(h).mean(dim=1)

    def bypass_alphas(self) -> list[float]:
        """Fixed bypass alpha of every layer (identical; hyperparameter)."""
        return [block.attn.bypass_alpha for block in self.blocks]

    def alpha_explore_penalty(self) -> torch.Tensor:
        """Deprecated no-op from the continuous-gate era.

        With Validation-Selected Discrete Routing the gate receives no
        gradient flow, so there is nothing to regularize; kept so existing
        training loops and checkpoints remain compatible.
        """
        return torch.zeros((), device=next(self.parameters()).device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.encode(x)
        return self.readout(self.dropout(pooled)).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers (identical protocol to meta_gat.py)
# ---------------------------------------------------------------------------
def _train_with_early_stopping(model, x_train, y_train_z, x_val, y_val_raw,
                               y_val_mean, y_val_std, config, device,
                               ib_tracker=None):
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
        loss = mse(model(x_train), y_train_z) + model.alpha_explore_penalty()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if ib_tracker is not None:
            ib_tracker.log_alpha_epoch(epoch, model.bypass_alphas())
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


def _train_fixed_epochs(model, x_fit, y_fit_z, config, n_epochs, device,
                        ib_tracker=None):
    model = model.to(device)
    x_fit, y_fit_z = x_fit.to(device), y_fit_z.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    mse = nn.MSELoss()
    for epoch in range(int(n_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = mse(model(x_fit), y_fit_z) + model.alpha_explore_penalty()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
        if ib_tracker is not None:
            ib_tracker.log_alpha_epoch(epoch, model.bypass_alphas())
            model.eval()
            with torch.no_grad():
                latent = model.encode(x_fit)
            ib_tracker.log_epoch(epoch, latent.detach().cpu().numpy(),
                                 y_fit_z.detach().cpu().numpy(),
                                 x=x_fit.detach().cpu().numpy())
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
    alpha_grid: Optional[Sequence[float]] = None,
    grad_clip: float = 5.0,
    seed: Optional[int] = None,
    ib_tracker: Optional["IBEpochTracker"] = None,
    checkpoint_path: Optional[str | Path] = None,
) -> Tuple[np.ndarray, Dict[str, float], float, int, np.ndarray, int]:
    """Nested selection + refit for one outer split (leakage-free).

    Validation-Selected Discrete Routing protocol:

      1. Standardize on inner train only; threshold the split graph on
         inner-train SC only.
      2. Architecture selection: the (hidden, dropout, lr) candidates are
         trained with the gate at its neutral value alpha = 0.5 and scored
         by inner-validation RMSE.
      3. Alpha selection: with the selected architecture, one model per
         ``alpha_grid`` value (default {0.0, 0.25, 0.5, 0.75, 1.0}) is
         trained on the inner-training split; the alpha with the lowest
         inner-validation RMSE wins.  The gate is a fixed scalar - no
         gradient flow, hence no Gradient Absorption.
      4. The (architecture, alpha) winner is refit on train+val for its
         selected epoch budget, predicts the outer test split in raw units,
         and exports gradient node saliency.

    ``tracker.alpha_final`` holds the selected alpha per layer and
    ``tracker.selected_alpha`` the scalar itself.

    Returns
    -------
    (test predictions, best config dict, best val RMSE, best epoch,
     node saliency, number of parameters)
    """
    grid = list(alpha_grid) if alpha_grid is not None else [0.0, 0.25, 0.5, 0.75, 1.0]
    if not grid or not all(0.0 <= float(a) <= 1.0 for a in grid):
        raise ValueError(f"alpha_grid values must lie in [0, 1], got {grid}")
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
    # ---- stage 1: architecture selection at the neutral gate alpha = 0.5 ----
    candidates = build_candidate_grid(
        hidden_grid, dropout_grid, lr_grid, n_layers=n_layers, heads=heads,
        weight_decay=weight_decay, epochs=epochs, patience=patience,
        min_epochs=min_epochs, alpha=0.5, grad_clip=grad_clip,
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

    # ---- stage 2: discrete alpha selection with the selected architecture ----
    best_alpha, best_alpha_rmse, best_alpha_epoch = None, float("inf"), 0
    for alpha_candidate in grid:
        cfg = dataclasses_replace(best_cfg, alpha=float(alpha_candidate))
        model = LLMGatedTransformer(n_rois, x.shape[2], cfg, prior, edge_src, edge_dst)
        val_rmse, epoch_used = _train_with_early_stopping(
            model, x_train, y_train_z, x_val, y_val_raw, y_mean, y_std, cfg, device,
        )
        if val_rmse < best_alpha_rmse - 1e-12:
            best_alpha = float(alpha_candidate)
            best_alpha_rmse, best_alpha_epoch = val_rmse, epoch_used
    if best_alpha is None:
        raise RuntimeError("No alpha candidate was selected")
    best_cfg = dataclasses_replace(best_cfg, alpha=best_alpha)

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
    _train_fixed_epochs(final_model, x_fit, y_fit_z, best_cfg, best_epoch, device,
                        ib_tracker=ib_tracker)
    final_model.eval()
    if ib_tracker is not None:
        with torch.no_grad():
            latent_fit = final_model.encode(x_fit.to(device))
        ib_tracker.log_final(latent_fit.cpu().numpy(), y_fit_z.numpy(),
                             x=x_fit.numpy())
        ib_tracker.alpha_final = final_model.bypass_alphas()
        ib_tracker.selected_alpha = best_alpha
    with torch.no_grad():
        pred = (final_model(x_test.to(device)).detach().cpu().numpy() * fit_std
                + fit_mean).astype(np.float64)

    saliency = gradient_node_saliency(final_model, x_fit, device)
    if checkpoint_path is not None:
        # Frozen-inference checkpoint for cross-cohort zero-shot transfer:
        # weights plus EVERYTHING needed to reproduce the feature pipeline
        # (fit scalers, target de-normalization, split graph, prior, config).
        payload = {
            "format_version": 1,
            "model_family": "llm_gated_transformer",
            "state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()},
            "config": asdict(best_cfg),
            "n_rois": int(n_rois),
            "in_dim": int(x.shape[2]),
            "prior": np.asarray(prior, dtype=np.float64),
            "edge_src": edge_src,
            "edge_dst": edge_dst,
            "x_mean": mean.reshape(1, -1),
            "x_std": std.reshape(1, -1),
            "fit_mean": float(fit_mean),
            "fit_std": float(fit_std),
            "seed": int(seed) if seed is not None else None,
        }
        ckpt = Path(checkpoint_path)
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, ckpt)
    n_params = sum(p.numel() for p in final_model.parameters())
    best_cfg_dict = {
        "hidden": best_cfg.hidden, "dropout": best_cfg.dropout,
        "learning_rate": best_cfg.learning_rate, "n_layers": best_cfg.n_layers,
        "selected_alpha": best_alpha,
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
            alpha = layer._last_attention.mean(dim=(0, 2)).cpu().numpy()
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
