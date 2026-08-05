#!/usr/bin/env python3
"""Prior-Gated Graph Attention Network - Meta-GAT (ICLR 2027, Method 2).

The AAAI GCN diluted the meta-analysis prior through standard message
passing.  Meta-GAT injects the prior directly into the attention logits of a
graph attention network, turning the prior into a hard inductive bias that
routes information through task-relevant regions:

    e_ij = LeakyReLU(a^T [W h_i || W h_j]) + gamma * (p_i + p_j)

where p_i is the (min-max normalized) meta-analysis prior score of ROI i and
gamma is a learnable temperature parameter (one per layer).  The prior term
lives *outside* the LeakyReLU, in logit space, so a strong prior cannot be
washed out by the learned attention scores: if the prior says that regions i
and j are task-relevant, information is forced to flow along the (i, j) edge
and their messages dominate the node updates of the next layer.

Design (mirrors the AAAI protocol of scripts/28 and scripts/40)
---------------------------------------------------------------
- Subjects: identical cohort (FC_all.npy / SC_all.npy / label_all.npy).
- Graph topology: the structural connectome defines the graph; per split it
  is the row-wise top-10% (positive) thresholded *group average* of SC over
  the inner training partition only (leakage-free), symmetrized, with
  self-loops (so isolated ROIs still attend to themselves; the softmax is
  never over an empty edge set).
- Node features: per-subject [FC_row | SC_row] (232 dims), standardized on
  the inner training partition only.
- Nested CV: identical splitters (10 seeds x 5 folds, 15% inner validation);
  the (hidden, dropout, lr) candidate grid is selected on the inner
  validation split by RMSE; the winner is refit on train+val (scaler and
  graph recomputed on the refit partition) and evaluated on the outer test
  split.  Target z-normalization fitted on the training partition only.
- Biomarker: node-level saliency = mean |dy/dx_i| (gradient of the predicted
  score w.r.t. the node features, aggregated over both modalities), averaged
  over the fit subjects, min-max normalized to [0, 1] - the nonlinear
  analogue of |beta| in Method 1, directly comparable with the prior.

Edge cases
----------
- Attention softmax overflow: the logits are shifted by their per-node
  maximum before the exponential (stable segment softmax).
- Isolated ROIs (no structural edge): self-loops guarantee a non-empty
  neighborhood; the prior gate still applies to the self-loop logit.
- Non-finite loss: raises FloatingPointError instead of silently NaN-ing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metascfc.benchmark_utils import rowwise_topk_adjacency, set_all_seeds


# ---------------------------------------------------------------------------
# Hyperparameter container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MetaGATConfig:
    """Hyperparameters of one Meta-GAT candidate (selected by inner CV).

    Attributes
    ----------
    hidden : int
        Per-head hidden width of the first GAT layer.
    heads1 : int
        Attention heads of the first layer (outputs concatenated).
    heads2 : int
        Attention heads of the second layer (outputs averaged).
    dropout : float
        Dropout applied to the second-layer features and the readout.
    gamma_init : float
        Initial value of the learnable prior-gate temperature gamma.
    learning_rate : float
        Adam learning rate.
    weight_decay : float
        Adam weight decay.
    epochs : int
        Maximum number of training epochs (early stopping on validation).
    patience : int
        Epochs of no validation improvement before early stopping.
    min_epochs : int
        Minimum number of epochs before early stopping is considered.
    grad_clip : float
        Global gradient-norm clipping value.
    """

    hidden: int = 16
    heads1: int = 4
    heads2: int = 1
    dropout: float = 0.2
    gamma_init: float = 1.0
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    patience: int = 15
    min_epochs: int = 10
    grad_clip: float = 5.0


def build_candidate_grid(
    hidden_grid: Sequence[int],
    dropout_grid: Sequence[float],
    lr_grid: Sequence[float],
    **fixed: float,
) -> List[MetaGATConfig]:
    """Cartesian product of the tunable hyperparameters.

    The fixed keyword arguments (epochs, patience, weight_decay, ...) are
    shared by every candidate, matching the "identical hyperparameters for
    true/shuffled/random variants" safeguard of the AAAI protocol.
    """
    out: List[MetaGATConfig] = []
    for hidden in hidden_grid:
        for dropout in dropout_grid:
            for lr in lr_grid:
                out.append(
                    MetaGATConfig(hidden=int(hidden), dropout=float(dropout),
                                  learning_rate=float(lr), **fixed)
                )
    return out


# ---------------------------------------------------------------------------
# Graph construction (structural topology, leakage-free)
# ---------------------------------------------------------------------------
def build_split_graph(
    sc_fit: np.ndarray,
    top_percent: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Binary edge list of the split-level structural connectome graph.

    The graph is the row-wise top-``top_percent``% (positive) thresholded
    *group average* of SC over the given subjects (the inner training
    partition during selection, train+val at refit time - never the test
    subjects), symmetrized, with self-loops added for every ROI.

    Parameters
    ----------
    sc_fit : np.ndarray, shape (n_subjects, n_rois, n_rois)
        Structural connectomes of the subjects the graph is built from.
    top_percent : float
        Percentage of strongest positive entries retained per row
        (same semantics as ``rowwise_topk_adjacency``, the AAAI helper).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (edge_src, edge_dst), int64 arrays of shape (n_edges,) describing the
        directed edges of the graph (self-loops included).
    """
    sc_fit = np.asarray(sc_fit, dtype=np.float32)
    mean_sc = sc_fit.mean(axis=0, keepdims=True)  # (1, n_rois, n_rois)
    adj = rowwise_topk_adjacency(mean_sc, top_percent, positive_only=True)[0]
    binary = (adj > 0).astype(np.int64)
    np.fill_diagonal(binary, 0)
    binary = np.maximum(binary, binary.T)
    src, dst = np.nonzero(binary)
    n_rois = binary.shape[0]
    self_src = np.arange(n_rois, dtype=np.int64)
    edge_src = np.concatenate([self_src, src])
    edge_dst = np.concatenate([self_src, dst])
    return edge_src, edge_dst


# ---------------------------------------------------------------------------
# Prior-gated attention layer
# ---------------------------------------------------------------------------
class PriorGatedGATLayer(nn.Module):
    """Graph attention layer whose logits are biased by the prior.

    For every directed edge (u -> v) of the split graph:

        e_uv = LeakyReLU(a^T [W h_u || W h_v]) + gamma * (p_u + p_v)
        alpha_uv = exp(e_uv) / sum_{w in N(v)} exp(e_wv)     (per head)
        h'_v = sum_{u in N(v)} alpha_uv * W h_u              (per head)

    The gate term gamma * (p_u + p_v) is applied *after* the LeakyReLU, in
    logit space, so the prior biases the routing directly and cannot be
    re-absorbed by the learned attention scores.

    Parameters
    ----------
    in_dim : int
        Input feature width per node.
    out_dim : int
        Output width per head.
    heads : int
        Number of attention heads (independent W and a per head).
    n_nodes : int
        Number of ROIs.
    prior : np.ndarray, shape (n_nodes,)
        Min-max normalized ROI-level prior scores p.
    edge_src / edge_dst : np.ndarray, shape (n_edges,)
        Directed edge list of the split graph (self-loops included).
    gamma_init : float
        Initial value of the learnable temperature gamma.
    leaky : float
        Negative slope of LeakyReLU.
    dropout : float
        Dropout probability applied to the output features.
    concat : bool
        Concatenate head outputs (True) or average them (False).
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
        gamma_init: float = 1.0,
        leaky: float = 0.2,
        dropout: float = 0.0,
        concat: bool = True,
    ) -> None:
        super().__init__()
        if heads < 1:
            raise ValueError(f"heads must be >= 1, got {heads}")
        if out_dim < 1:
            raise ValueError(f"out_dim must be >= 1, got {out_dim}")
        self.heads = heads
        self.concat = concat
        self.out_dim = out_dim
        self.leaky = float(leaky)

        # Per-head feature transform W_h and attention vector a_h.
        self.W = nn.Parameter(
            torch.empty(heads, in_dim, out_dim, dtype=torch.float32)
        )
        self.a = nn.Parameter(
            torch.empty(heads, 2 * out_dim, dtype=torch.float32)
        )
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a.view(heads, 2 * out_dim))

        # Learnable prior-gate temperature (one scalar per layer).
        self.gamma = nn.Parameter(
            torch.tensor(float(gamma_init), dtype=torch.float32)
        )

        # Fixed split graph and prior (moved to the device with the model).
        prior = np.asarray(prior, dtype=np.float32).reshape(-1)
        if prior.shape[0] != n_nodes:
            raise ValueError(
                f"prior has {prior.shape[0]} entries; expected {n_nodes}"
            )
        self.register_buffer("prior", torch.from_numpy(prior))
        self.register_buffer(
            "edge_src", torch.from_numpy(np.asarray(edge_src, dtype=np.int64))
        )
        self.register_buffer(
            "edge_dst", torch.from_numpy(np.asarray(edge_dst, dtype=np.int64))
        )
        # One-hot (n_edges, n_nodes) of the destination nodes, used for the
        # sum reductions (torch.index_reduce only supports prod/mean/amax/amin,
        # not sum; einsum with this indicator is exact and differentiable).
        dst_onehot = np.zeros((len(edge_dst), n_nodes), dtype=np.float32)
        dst_onehot[np.arange(len(edge_dst)), edge_dst] = 1.0
        self.register_buffer("edge_dst_onehot", torch.from_numpy(dst_onehot))
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one prior-gated attention round.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_nodes, in_dim)
            Node features (standardized).

        Returns
        -------
        torch.Tensor
            Shape (batch, n_nodes, heads*out_dim) if concat else
            (batch, n_nodes, out_dim).
        """
        batch, n_nodes, _ = x.shape
        device, dtype = x.device, x.dtype
        edge_src, edge_dst = self.edge_src, self.edge_dst
        n_edges = edge_src.shape[0]

        # Gate (E,) broadcast over batch and heads: gamma * (p_src + p_dst).
        gate = self.gamma * (self.prior[edge_src] + self.prior[edge_dst])

        # Per-head transforms and attention logits.
        #   Wh : (batch, n_nodes, heads, out_dim)
        Wh = torch.einsum("bnd,hdq->bnhq", x, self.W)
        Wh_src = Wh[:, edge_src]  # (batch, n_edges, heads, out_dim)
        Wh_dst = Wh[:, edge_dst]
        #   logit : (batch, n_edges, heads)
        logit = torch.einsum(
            "behd,hd->beh", torch.cat([Wh_src, Wh_dst], dim=-1), self.a
        )
        logit = F.leaky_relu(logit, negative_slope=self.leaky)
        logit = logit + gate.reshape(1, n_edges, 1)

        # Stable segment softmax over the incoming edges of each node.
        # (max / sum reductions grouped by edge_dst; shifts prevent overflow).
        neg_inf = torch.finfo(dtype).min
        max_per_node = torch.full(
            (batch, self.heads, n_nodes), neg_inf, device=device, dtype=dtype
        )
        max_per_node = torch.index_reduce(
            max_per_node, 2, edge_dst, logit.transpose(1, 2),
            reduce="amax", include_self=False,
        )
        # Every node has at least its self-loop, but guard isolated cases.
        max_per_node = max_per_node.nan_to_num(neg_inf)
        edge_index = edge_dst.reshape(1, 1, -1).expand(
            batch, self.heads, n_edges
        )
        max_per_edge = max_per_node.gather(2, edge_index).transpose(1, 2)
        exp_logit = (logit - max_per_edge).exp()  # (batch, n_edges, heads)

        # Per-node sums via the destination one-hot indicator matrix A (E, N):
        #   sum_per_node[b, n, h] = sum_e A[e, n] * exp_logit[b, e, h].
        A = self.edge_dst_onehot  # (n_edges, n_nodes)
        sum_per_node = torch.einsum("beh,en->bnh", exp_logit, A)
        sum_per_edge = torch.einsum("bnh,en->beh", sum_per_node, A)
        alpha = exp_logit / sum_per_edge.clamp_min(1e-8)

        # Attention coefficients are cached (detached) for the biomarker
        # extraction (mean attention mass per ROI, faithfulness protocol).
        self._last_alpha = alpha.detach()

        # Message aggregation: h'_v += sum_{u in N(v)} alpha_uv * W h_u,
        # i.e. agg[b, n, h, d] = sum_e A[e, n] * msg[b, e, h, d].
        msg = alpha.unsqueeze(-1) * Wh_src  # (batch, n_edges, heads, out_dim)
        agg = torch.einsum("behd,en->bnhd", msg, A)

        if self.concat:
            out = agg.reshape(batch, n_nodes, self.heads * self.out_dim)
        else:
            out = agg.mean(dim=2)
        out = F.elu(out)
        return self.dropout(out)


# ---------------------------------------------------------------------------
# Full predictor
# ---------------------------------------------------------------------------
class MetaGAT(nn.Module):
    """Two-layer Prior-Gated GAT with mean-pooling readout.

    Computes a scalar prediction for one subject from its (standardized)
    per-ROI features [FC_row | SC_row] and the split structural graph.

    Parameters
    ----------
    n_nodes : int
        Number of ROIs.
    in_dim : int
        Input feature width per node (2 * n_rois when FC and SC rows are
        concatenated).
    config : MetaGATConfig
        Hyperparameters (including gamma_init of both layers).
    prior : np.ndarray, shape (n_nodes,)
        ROI-level prior scores p (min-max normalized).
    edge_src / edge_dst : np.ndarray, shape (n_edges,)
        Split-level directed edge list (self-loops included).
    """

    def __init__(
        self,
        n_nodes: int,
        in_dim: int,
        config: MetaGATConfig,
        prior: np.ndarray,
        edge_src: np.ndarray,
        edge_dst: np.ndarray,
    ) -> None:
        super().__init__()
        self.layer1 = PriorGatedGATLayer(
            in_dim=in_dim, out_dim=config.hidden, heads=config.heads1,
            n_nodes=n_nodes, prior=prior, edge_src=edge_src,
            edge_dst=edge_dst, gamma_init=config.gamma_init,
            dropout=config.dropout, concat=True,
        )
        self.layer2 = PriorGatedGATLayer(
            in_dim=config.hidden * config.heads1, out_dim=config.hidden,
            heads=config.heads2, n_nodes=n_nodes, prior=prior,
            edge_src=edge_src, edge_dst=edge_dst,
            gamma_init=config.gamma_init, dropout=config.dropout,
            concat=False,
        )
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0.0 else nn.Identity()
        self.readout = nn.Linear(config.hidden, config.hidden)
        self.head = nn.Linear(config.hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict a scalar score per subject.

        Parameters
        ----------
        x : torch.Tensor, shape (batch, n_nodes, in_dim)
            Standardized node features.

        Returns
        -------
        torch.Tensor, shape (batch,)
            Raw (unstandardized-space) prediction of the z-scored target.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        pooled = x.mean(dim=1)  # mean pooling over ROIs
        h = F.elu(self.readout(pooled))
        h = self.dropout(h)
        return self.head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
def _train_with_early_stopping(
    model: MetaGAT,
    x_train: torch.Tensor,
    y_train_z: torch.Tensor,
    x_val: torch.Tensor,
    y_val_raw: torch.Tensor,
    y_val_mean: float,
    y_val_std: float,
    config: MetaGATConfig,
    device: torch.device,
) -> Tuple[float, int]:
    """Train one candidate; return (best validation RMSE, best epoch).

    The validation RMSE is computed in the *raw* target units (the z-scored
    predictions are de-normalized with the training-partition statistics),
    matching the selection criterion of the AAAI baselines.
    """
    model = model.to(device)
    # Move the (CPU) data tensors to the training device - required for CUDA.
    x_train = x_train.to(device)
    y_train_z = y_train_z.to(device)
    x_val = x_val.to(device)
    y_val_raw = y_val_raw.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    mse = nn.MSELoss()
    best_rmse = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    wait = 0

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred_z = model(x_train)
        loss = mse(pred_z, y_train_z)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch + 1}: {loss.item()}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        if epoch + 1 < config.min_epochs:
            continue
        model.eval()
        with torch.no_grad():
            pred_val = model(x_val) * y_val_std + y_val_mean
            rmse = float(
                torch.sqrt(torch.mean((pred_val - y_val_raw) ** 2))
            )
        if rmse < best_rmse - 1e-8:
            best_rmse = rmse
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_epoch = epoch + 1
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("No validation checkpoint was selected")
    model.load_state_dict(best_state)
    return best_rmse, best_epoch


def _train_fixed_epochs(
    model: MetaGAT,
    x_fit: torch.Tensor,
    y_fit_z: torch.Tensor,
    config: MetaGATConfig,
    n_epochs: int,
    device: torch.device,
) -> None:
    """Refit a model for a fixed number of epochs (no validation involved)."""
    model = model.to(device)
    x_fit = x_fit.to(device)
    y_fit_z = y_fit_z.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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
# Biomarker: gradient saliency
# ---------------------------------------------------------------------------
def _saliency_one_pass(model: MetaGAT, x: torch.Tensor) -> torch.Tensor:
    """Saliency of one subject: |dy/dx| summed over the input features.

    Note: intentionally *not* under torch.no_grad() - the gradient of the
    predicted score w.r.t. the node features requires the autograd graph.
    """
    xi = x.requires_grad_(True)
    pred = model(xi)
    grad = torch.autograd.grad(pred.sum(), xi, retain_graph=False)[0][0]
    return grad.abs()


def gradient_node_saliency(
    model: MetaGAT,
    x_fit: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Node-level saliency = mean |dy/dx_i| over the fit subjects.

    For each subject the gradient of the predicted score w.r.t. its
    standardized node features is computed and its absolute value is
    aggregated over the input features (both the FC and SC rows, i.e. both
    modalities), then averaged over the subjects of the fit partition and
    min-max normalized to [0, 1].  This is the predictive-influence
    biomarker of the network - the nonlinear analogue of |beta| in Method 1 -
    directly comparable with the prior (prior_alignment_metrics).

    Parameters
    ----------
    model : MetaGAT
        Trained (refit) model, in eval mode.
    x_fit : torch.Tensor, shape (n_subjects, n_nodes, in_dim)
        Standardized node features of the fit partition.
    device : torch.device

    Returns
    -------
    np.ndarray, shape (n_nodes,)
        Min-max normalized saliency in [0, 1].
    """
    model.eval()
    n_subjects, n_nodes, in_dim = x_fit.shape
    accum = torch.zeros((n_nodes, in_dim), device=device, dtype=torch.float32)
    for s in range(n_subjects):
        accum += _saliency_one_pass(model, x_fit[s : s + 1].to(device))
    node_saliency = (accum / n_subjects).sum(dim=1).detach().cpu().numpy()
    span = node_saliency.max() - node_saliency.min()
    if span > 1e-12:
        node_saliency = (node_saliency - node_saliency.min()) / span
    else:
        node_saliency = np.zeros(n_nodes, dtype=np.float64)
    return node_saliency


# ---------------------------------------------------------------------------
# Nested cross-validation entry point (the script-41 workhorse)
# ---------------------------------------------------------------------------
def fit_predict_meta_gat(
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
    heads1: int = 4,
    heads2: int = 1,
    weight_decay: float = 1e-4,
    epochs: int = 60,
    patience: int = 15,
    min_epochs: int = 10,
    gamma_init: float = 1.0,
    grad_clip: float = 5.0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, float], float, int, np.ndarray, int]:
    """Nested selection + refit for one outer split (leakage-free).

    Protocol (identical to the AAAI baselines):
      1. Node features X = [FC_row | SC_row] and target y are standardized on
         the inner training partition only.
      2. The structural graph is thresholded from the group-average SC of the
         inner training partition only.
      3. Every (hidden, dropout, lr) candidate is trained with early stopping
         on the inner validation RMSE; the best candidate wins.
      4. The winner is refit on train+val (scaler and graph recomputed on the
         refit partition) for its selected epoch budget, then predicts the
         outer test subjects in raw target units.
      5. The node saliency biomarker is computed on the refit model.

    Parameters
    ----------
    fc / sc : np.ndarray, shape (n, n_rois, n_rois)
        Functional / structural connectomes of all subjects.
    y : np.ndarray, shape (n,)
        Raw target scores.
    train_idx / val_idx / test_idx : np.ndarray
        Partition indices from iter_nested_splits.
    prior : np.ndarray, shape (n_rois,)
        Min-max normalized ROI prior scores (true / shuffled / random).
    hidden_grid / dropout_grid / lr_grid : Sequence
        Candidate grids (cartesian product).
    device : torch.device
    top_percent : float
        Row-wise SC threshold for the split graph (default 10.0).
    heads1 / heads2 : int
        Attention head counts.
    weight_decay / epochs / patience / min_epochs / gamma_init / grad_clip :
        Fixed hyperparameters shared by all candidates.
    seed : int, optional
        If given, set_all_seeds(seed) is called first (determinism).

    Returns
    -------
    Tuple[np.ndarray, Dict[str, float], float, int, np.ndarray, int]
        (test predictions in raw units, best config dict, best validation
        RMSE, best epoch, node saliency, number of model parameters).
    """
    train_idx = np.asarray(train_idx, dtype=int)
    val_idx = np.asarray(val_idx, dtype=int)
    test_idx = np.asarray(test_idx, dtype=int)
    if seed is not None:
        set_all_seeds(seed)

    fc = np.asarray(fc, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    n_rois = fc.shape[1]
    x = np.concatenate([fc, sc], axis=2)  # (n, n_rois, 2 * n_rois)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    # ---- inner training partition (model selection) ----
    x_train_flat = x[train_idx].reshape(len(train_idx), -1)
    x_mean = x_train_flat.mean(axis=0, keepdims=True)
    x_std = x_train_flat.std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    x_train = torch.from_numpy(
        ((x[train_idx].reshape(len(train_idx), -1) - x_mean) / x_std)
        .reshape(len(train_idx), n_rois, -1).astype(np.float32)
    )
    x_val = torch.from_numpy(
        ((x[val_idx].reshape(len(val_idx), -1) - x_mean) / x_std)
        .reshape(len(val_idx), n_rois, -1).astype(np.float32)
    )
    y_train_mean, y_train_std = float(y[train_idx].mean()), float(y[train_idx].std())
    y_train_std = y_train_std if y_train_std >= 1e-8 else 1.0
    y_train_z = torch.from_numpy(
        ((y[train_idx] - y_train_mean) / y_train_std).astype(np.float32)
    )
    y_val_raw = torch.from_numpy(y[val_idx].astype(np.float32))

    edge_src, edge_dst = build_split_graph(sc[train_idx], top_percent)
    candidates = build_candidate_grid(
        hidden_grid, dropout_grid, lr_grid, heads1=heads1, heads2=heads2,
        weight_decay=weight_decay, epochs=epochs, patience=patience,
        min_epochs=min_epochs, gamma_init=gamma_init, grad_clip=grad_clip,
    )

    # ---- inner model selection on validation RMSE ----
    best_rmse = float("inf")
    best_cfg: Optional[MetaGATConfig] = None
    best_epoch = 0
    for cfg in candidates:
        model = MetaGAT(n_rois, x.shape[2], cfg, prior, edge_src, edge_dst)
        val_rmse, epoch_used = _train_with_early_stopping(
            model, x_train, y_train_z, x_val, y_val_raw,
            y_train_mean, y_train_std, cfg, device,
        )
        if val_rmse < best_rmse - 1e-12:
            best_rmse = val_rmse
            best_cfg = cfg
            best_epoch = epoch_used
    if best_cfg is None:
        raise RuntimeError("No candidate configuration was selected")

    # ---- refit on train + validation, predict the outer test split ----
    fit_idx = np.concatenate([train_idx, val_idx])
    x_fit_flat = x[fit_idx].reshape(len(fit_idx), -1)
    x_mean = x_fit_flat.mean(axis=0, keepdims=True)
    x_std = x_fit_flat.std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    x_fit = torch.from_numpy(
        ((x[fit_idx].reshape(len(fit_idx), -1) - x_mean) / x_std)
        .reshape(len(fit_idx), n_rois, -1).astype(np.float32)
    )
    x_test = torch.from_numpy(
        ((x[test_idx].reshape(len(test_idx), -1) - x_mean) / x_std)
        .reshape(len(test_idx), n_rois, -1).astype(np.float32)
    )
    fit_mean, fit_std = float(y[fit_idx].mean()), float(y[fit_idx].std())
    fit_std = fit_std if fit_std >= 1e-8 else 1.0
    y_fit_z = torch.from_numpy(((y[fit_idx] - fit_mean) / fit_std).astype(np.float32))

    edge_src, edge_dst = build_split_graph(sc[fit_idx], top_percent)
    final_model = MetaGAT(n_rois, x.shape[2], best_cfg, prior, edge_src, edge_dst)
    _train_fixed_epochs(final_model, x_fit, y_fit_z, best_cfg, best_epoch, device)
    final_model.eval()
    with torch.no_grad():
        pred = (
            final_model(x_test.to(device)).detach().cpu().numpy() * fit_std + fit_mean
        ).astype(np.float64)

    saliency = gradient_node_saliency(final_model, x_fit, device)
    n_params = sum(p.numel() for p in final_model.parameters())
    best_cfg_dict = {
        "hidden": best_cfg.hidden, "dropout": best_cfg.dropout,
        "learning_rate": best_cfg.learning_rate,
    }
    return pred, best_cfg_dict, best_rmse, best_epoch, saliency, n_params


# ---------------------------------------------------------------------------
# Refit predictor (faithfulness / perturbation protocol)
# ---------------------------------------------------------------------------
class RefitMetaGATPredictor:
    """Trained Meta-GAT with the split-level scalers, ready for masking tests.

    Wraps a model refit on the (train+val) partition together with the
    feature scaler (mean/std over the fit partition), the target
    de-normalization statistics, and the ROI count.  ``predict`` accepts
    *raw* connectomes of arbitrary subjects (e.g. with ROIs masked out) and
    returns raw-unit predictions.
    """

    def __init__(
        self,
        model: MetaGAT,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        fit_mean: float,
        fit_std: float,
        n_rois: int,
        device: torch.device,
    ) -> None:
        self.model = model
        self.x_mean = x_mean.reshape(1, -1)
        self.x_std = x_std.reshape(1, -1)
        self.fit_mean = float(fit_mean)
        self.fit_std = float(fit_std)
        self.n_rois = int(n_rois)
        self.device = device

    def predict(self, fc: np.ndarray, sc: np.ndarray) -> np.ndarray:
        """Predict raw-unit scores from raw (possibly perturbed) connectomes.

        Parameters
        ----------
        fc / sc : np.ndarray, shape (n_subjects, n_rois, n_rois)
            Raw connectomes (masked or unmasked).

        Returns
        -------
        np.ndarray, shape (n_subjects,)
        """
        n = len(fc)
        x = np.concatenate([fc, sc], axis=2).reshape(n, -1).astype(np.float64)
        x = ((x - self.x_mean) / self.x_std).reshape(
            n, self.n_rois, -1
        ).astype(np.float32)
        xt = torch.from_numpy(x).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred = self.model(xt).detach().cpu().numpy()
        return (pred * self.fit_std + self.fit_mean).astype(np.float64)

    def attention_mass(self, fc: np.ndarray, sc: np.ndarray) -> np.ndarray:
        """Mean attention coefficient per ROI over the given subjects/layers.

        For every layer the cached attention coefficients alpha (n_edges,
        heads) are averaged over subjects and heads -> (n_edges,).  Each
        ROI's score is the *mean* alpha over its incident edges (incoming,
        outgoing, and the self-loop).  The mean (not the sum) is used
        because the split graph is regular: every ROI has exactly the same
        number of incident edges, so sums are constant by construction.  The
        two layers are averaged and the vector is L1-normalized.

        Parameters
        ----------
        fc / sc : np.ndarray, shape (n_subjects, n_rois, n_rois)

        Returns
        -------
        np.ndarray, shape (n_rois,)
        """
        n = len(fc)
        x = np.concatenate([fc, sc], axis=2).reshape(n, -1).astype(np.float64)
        x = ((x - self.x_mean) / self.x_std).reshape(
            n, self.n_rois, -1
        ).astype(np.float32)
        xt = torch.from_numpy(x).to(self.device)
        self.model.eval()
        masses = []
        with torch.no_grad():
            _ = self.model(xt)
        for layer in (self.model.layer1, self.model.layer2):
            alpha = layer._last_alpha  # (batch, n_edges, heads)
            alpha_mean = alpha.mean(dim=(0, 2)).cpu().numpy()  # (n_edges,)
            A_dst = layer.edge_dst_onehot.cpu().numpy()  # (n_edges, n_nodes)
            A_src = np.zeros_like(A_dst)
            A_src[np.arange(A_src.shape[0]), layer.edge_src.cpu().numpy()] = 1.0
            in_sum = A_dst.T @ alpha_mean
            out_sum = A_src.T @ alpha_mean
            in_deg = A_dst.sum(axis=0)
            out_deg = A_src.sum(axis=0)
            mass = (in_sum + out_sum) / (in_deg + out_deg)
            total = float(mass.sum())
            masses.append(mass / total if total > 1e-12 else mass)
        return np.asarray(np.mean(masses, axis=0), dtype=np.float64)


def refit_meta_gat_predictor(
    fc: np.ndarray,
    sc: np.ndarray,
    y: np.ndarray,
    fit_idx: np.ndarray,
    config: MetaGATConfig,
    prior: np.ndarray,
    device: torch.device,
    n_epochs: int,
    top_percent: float = 10.0,
    seed: Optional[int] = None,
) -> RefitMetaGATPredictor:
    """Refit a Meta-GAT on a fit partition with a *given* configuration.

    Used by the faithfulness protocol: the configuration (and epoch budget)
    recorded by the main run (scripts/41) is reused so the perturbed
    evaluations are performed by the same model family that produced the
    headline results.  The scalers, the split graph, and the prior are the
    same objects the main refit used.

    Parameters
    ----------
    fc / sc : np.ndarray, shape (n, n_rois, n_rois)
    y : np.ndarray, shape (n,)
    fit_idx : np.ndarray
        Fit partition (train+val of the outer split).
    config : MetaGATConfig
        Recorded best configuration of the split.
    prior : np.ndarray, shape (n_rois,)
    device : torch.device
    n_epochs : int
        Recorded best epoch budget of the split (fixed-epoch refit).
    top_percent : float
    seed : int, optional
        If given, set_all_seeds(seed) is called first (determinism).
    """
    fit_idx = np.asarray(fit_idx, dtype=int)
    if seed is not None:
        set_all_seeds(seed)
    fc = np.asarray(fc, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    n_rois = fc.shape[1]
    x = np.concatenate([fc, sc], axis=2)
    y = np.asarray(y, dtype=np.float64).reshape(-1)

    x_fit_flat = x[fit_idx].reshape(len(fit_idx), -1)
    x_mean = x_fit_flat.mean(axis=0, keepdims=True)
    x_std = x_fit_flat.std(axis=0, keepdims=True)
    x_std[x_std < 1e-8] = 1.0
    x_fit = torch.from_numpy(
        ((x[fit_idx].reshape(len(fit_idx), -1) - x_mean) / x_std)
        .reshape(len(fit_idx), n_rois, -1).astype(np.float32)
    )
    fit_mean, fit_std = float(y[fit_idx].mean()), float(y[fit_idx].std())
    fit_std = fit_std if fit_std >= 1e-8 else 1.0
    y_fit_z = torch.from_numpy(((y[fit_idx] - fit_mean) / fit_std).astype(np.float32))

    edge_src, edge_dst = build_split_graph(sc[fit_idx], top_percent)
    model = MetaGAT(n_rois, x.shape[2], config, prior, edge_src, edge_dst)
    _train_fixed_epochs(model, x_fit, y_fit_z, config, int(n_epochs), device)
    model.eval()
    return RefitMetaGATPredictor(
        model, x_mean, x_std, fit_mean, fit_std, n_rois, device
    )


__all__ = [
    "MetaGAT",
    "MetaGATConfig",
    "PriorGatedGATLayer",
    "RefitMetaGATPredictor",
    "build_candidate_grid",
    "build_split_graph",
    "fit_predict_meta_gat",
    "gradient_node_saliency",
    "refit_meta_gat_predictor",
]
