from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .benchmark_utils import normalize_dense_adjacency_torch


class MGCNRegressor(nn.Module):
    """Paper-faithful dense implementation of D'Souza et al. (MIDL 2021).

    FC is treated as a signal on the subject-specific structural graph. The
    implementation follows Eqs. (2)--(3): row/column graph filters, node-wise
    pooling, graph-level projection, and a 256-128-30-1 regression head.
    """

    def __init__(self, n_rois: int = 116, channels: int = 32, graph_dim: int = 256):
        super().__init__()
        self.n_rois = n_rois
        self.channels = channels
        self.graph_dim = graph_dim
        self.row_filters = nn.Parameter(torch.empty(channels, n_rois))
        self.col_filters = nn.Parameter(torch.empty(channels, n_rois))
        self.channel_pool = nn.Parameter(torch.empty(channels, n_rois))
        self.embedding = nn.Parameter(torch.empty(graph_dim, n_rois))
        self.bias1 = nn.Parameter(torch.zeros(channels))
        self.bias2 = nn.Parameter(torch.zeros(1))
        self.bias3 = nn.Parameter(torch.zeros(graph_dim))
        self.head = nn.Sequential(
            nn.Linear(graph_dim, 128), nn.LeakyReLU(0.1),
            nn.Linear(128, 30), nn.LeakyReLU(0.1),
            nn.Linear(30, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for p in (self.row_filters, self.col_filters, self.channel_pool, self.embedding):
            nn.init.xavier_uniform_(p)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, fc: torch.Tensor, sc_adj: torch.Tensor) -> torch.Tensor:
        lap = normalize_dense_adjacency_torch(sc_adj, add_self_loops=True)
        l_fc = torch.bmm(lap, fc)
        fc_l = torch.bmm(fc, lap)
        row_term = torch.einsum("mn,bnj->bmj", self.row_filters, l_fc)
        col_term = torch.einsum("bij,mj->bmi", fc_l, self.col_filters)
        h1 = F.leaky_relu(
            row_term.unsqueeze(2) + col_term.unsqueeze(3) + self.bias1.view(1, -1, 1, 1),
            negative_slope=0.1,
        )
        pooled = torch.einsum("bmij,mj->bmi", h1, self.channel_pool)
        propagated = torch.einsum("bij,bmj->bmi", lap, pooled).sum(dim=1)
        h2 = F.leaky_relu(propagated + self.bias2, negative_slope=0.1)
        h2_graph = torch.bmm(lap, h2.unsqueeze(-1)).squeeze(-1)
        h3 = F.leaky_relu(torch.einsum("dn,bn->bd", self.embedding, h2_graph) + self.bias3, 0.1)
        return self.head(h3).squeeze(-1)


class IMGGCNRegressor(nn.Module):
    """Paper-faithful IMG-GCN reimplementation for AAL116 regression.

    It implements modularity-specific FC--SC interaction filters, the full
    N^2 bottleneck attention used in the MICCAI 2024 paper, a unified 2N-node
    graph, one dense GCN layer, flattening, and a 256-unit output head.
    """

    def __init__(
        self,
        module_ids: Sequence[int],
        node_feature_dim: int,
        graph_hidden: int = 16,
        bottleneck_ratio: int = 2,
        dropout: float = 0.5,
        attention_mode: str = "full",
        smoke_attention_dim: int = 64,
    ):
        super().__init__()
        module_ids = torch.as_tensor(module_ids, dtype=torch.long)
        if module_ids.ndim != 1:
            raise ValueError("module_ids must be a 1D ROI-to-module array")
        unique = torch.unique(module_ids, sorted=True)
        remap = torch.empty(int(module_ids.max()) + 1, dtype=torch.long)
        for new_id, old_id in enumerate(unique.tolist()):
            remap[old_id] = new_id
        module_ids = remap[module_ids]
        self.register_buffer("module_ids", module_ids)
        self.n_rois = int(len(module_ids))
        self.n_modules = int(module_ids.max().item() + 1)
        self.node_feature_dim = int(node_feature_dim)
        self.graph_hidden = int(graph_hidden)
        self.attention_mode = attention_mode

        self.interaction_weight = nn.Parameter(
            torch.empty(self.n_modules, self.n_modules, 2 * self.node_feature_dim)
        )
        self.interaction_bias = nn.Parameter(torch.zeros(self.n_modules, self.n_modules))
        flat_dim = self.n_rois * self.n_rois
        if attention_mode == "full":
            hidden = max(1, flat_dim // int(bottleneck_ratio))
        elif attention_mode == "smoke":
            hidden = min(flat_dim, int(smoke_attention_dim))
        else:
            raise ValueError("attention_mode must be 'full' or 'smoke'")
        self.attention = nn.Sequential(
            nn.Linear(flat_dim, hidden), nn.ReLU(), nn.Linear(hidden, flat_dim), nn.Sigmoid()
        )
        self.gcn_weight = nn.Parameter(torch.empty(self.node_feature_dim, self.graph_hidden))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * self.n_rois * self.graph_hidden, 256),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.interaction_weight)
        nn.init.xavier_uniform_(self.gcn_weight)
        for m in self.attention:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def _interactions(self, fc_x: torch.Tensor, sc_x: torch.Tensor) -> torch.Tensor:
        b, n, k = fc_x.shape
        t = fc_x.new_zeros((b, n, n))
        for mi in range(self.n_modules):
            idx_f = torch.where(self.module_ids == mi)[0]
            xf = fc_x.index_select(1, idx_f)
            for mj in range(self.n_modules):
                idx_s = torch.where(self.module_ids == mj)[0]
                xs = sc_x.index_select(1, idx_s)
                w = self.interaction_weight[mi, mj]
                wf, ws = w[:k], w[k:]
                sf = torch.einsum("bik,k->bi", xf, wf)
                ss = torch.einsum("bjk,k->bj", xs, ws)
                block = F.relu(sf.unsqueeze(2) + ss.unsqueeze(1) + self.interaction_bias[mi, mj])
                t[:, idx_f[:, None], idx_s[None, :]] = block
        return t

    def forward(
        self,
        fc_x: torch.Tensor,
        sc_x: torch.Tensor,
        fc_adj: torch.Tensor,
        sc_adj: torch.Tensor,
    ) -> torch.Tensor:
        t = self._interactions(fc_x, sc_x)
        attention = self.attention(t.flatten(1)).view_as(t)
        cross = t * attention
        upper = torch.cat([sc_adj, cross.transpose(1, 2)], dim=2)
        lower = torch.cat([cross, fc_adj], dim=2)
        unified_adj = torch.cat([upper, lower], dim=1)
        unified_x = torch.cat([sc_x, fc_x], dim=1)
        norm_adj = normalize_dense_adjacency_torch(unified_adj, add_self_loops=True)
        z = F.relu(torch.bmm(norm_adj, unified_x) @ self.gcn_weight)
        return self.head(z).squeeze(-1)
