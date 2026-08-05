from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from .fs_coupling_base import FSCouplingModel


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_weight=None, batch=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


class ClassifierHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // 2)
        self.fc2 = nn.Linear(in_channels // 2, num_classes)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x


class InteractiveModule(nn.Module):
    def __init__(self, roi_num: int, hidden_channels: int):
        super().__init__()
        self.fc = nn.Linear(hidden_channels * 2, 1)

    def forward(self, fc_emb: torch.Tensor, sc_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([fc_emb, sc_emb], dim=-1)
        coupling = torch.sigmoid(self.fc(x)).squeeze(-1)
        return coupling


class MSInterGCN(FSCouplingModel):
    def __init__(
        self,
        roi_num: int = 116,
        hidden_channels: int = 64,
        num_classes: int = 2,
        dropout: float = 0.3,
        task: str = "classification",
    ):
        super().__init__()
        self.roi_num = roi_num
        self.hidden_channels = hidden_channels
        self.task = task
        out_dim = num_classes if task == "classification" else 1

        self.fc_encoder = GCNEncoder(roi_num, hidden_channels, hidden_channels, dropout)
        self.sc_encoder = GCNEncoder(roi_num, hidden_channels, hidden_channels, dropout)

        self.fc_head = ClassifierHead(hidden_channels, out_dim, dropout)
        self.sc_head = ClassifierHead(hidden_channels, out_dim, dropout)

        self.interactive = InteractiveModule(roi_num, hidden_channels)

        # fc_pooled [H] + sc_pooled [H] + coupling vector [N]
        # + FC interactive features [N*H] + SC interactive features [N*H].
        combined_dim = hidden_channels * 2 + roi_num + 2 * roi_num * hidden_channels
        self.final_fc = nn.Linear(combined_dim, hidden_channels)
        self.final_head = ClassifierHead(hidden_channels, out_dim, dropout)

    def forward(
        self,
        fc_x: torch.Tensor,
        fc_edge_index: torch.Tensor,
        fc_edge_weight: Optional[torch.Tensor],
        sc_x: torch.Tensor,
        sc_edge_index: torch.Tensor,
        sc_edge_weight: Optional[torch.Tensor],
        batch_fc: Optional[torch.Tensor] = None,
        batch_sc: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        if batch_fc is None:
            batch_fc = torch.zeros(fc_x.shape[0], dtype=torch.long, device=fc_x.device)
        if batch_sc is None:
            batch_sc = torch.zeros(sc_x.shape[0], dtype=torch.long, device=sc_x.device)

        fc_emb = self.fc_encoder(fc_x, fc_edge_index, fc_edge_weight, batch_fc)
        sc_emb = self.sc_encoder(sc_x, sc_edge_index, sc_edge_weight, batch_sc)

        fc_pooled = global_mean_pool(fc_emb, batch_fc)
        sc_pooled = global_mean_pool(sc_emb, batch_sc)

        fc_pred = self.fc_head(fc_pooled)
        sc_pred = self.sc_head(sc_pooled)

        n_subjects = fc_pooled.shape[0]
        n_nodes_per_subject = fc_emb.shape[0] // n_subjects

        coupling_vectors = []
        for i in range(n_subjects):
            start = i * n_nodes_per_subject
            end = start + n_nodes_per_subject
            fc_i = fc_emb[start:end]
            sc_i = sc_emb[start:end]
            cv = self.interactive(fc_i, sc_i)
            coupling_vectors.append(cv)
        coupling_vector = torch.stack(coupling_vectors, dim=0)

        combined = torch.cat([fc_pooled, sc_pooled, coupling_vector], dim=1)
        combined_fc_feat = coupling_vector.unsqueeze(-1) * fc_pooled.unsqueeze(1)
        combined_sc_feat = coupling_vector.unsqueeze(-1) * sc_pooled.unsqueeze(1)
        combined_feat = torch.cat([
            combined,
            combined_fc_feat.view(n_subjects, -1),
            combined_sc_feat.view(n_subjects, -1),
        ], dim=1)

        if combined_feat.shape[1] != self.final_fc.in_features:
            raise RuntimeError(
                f"Unexpected combined feature dimension {combined_feat.shape[1]} != "
                f"initialized dimension {self.final_fc.in_features}. Check roi_num and batch construction."
            )

        h = F.relu(self.final_fc(combined_feat))
        y_pred = self.final_head(h)

        return {
            "y_pred": y_pred,
            "fc_pred": fc_pred,
            "sc_pred": sc_pred,
            "coupling_vector": coupling_vector,
            "extra": {
                "fc_emb": fc_emb,
                "sc_emb": sc_emb,
                "fc_pooled": fc_pooled,
                "sc_pooled": sc_pooled,
            },
        }
