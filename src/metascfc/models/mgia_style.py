import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from .fs_coupling_base import FSCouplingModel


class MGIAStyleFSCouplingModel(FSCouplingModel):
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

        self.fc_encoder = GCNConv(roi_num, hidden_channels)
        self.sc_encoder = GCNConv(roi_num, hidden_channels)

        self.interaction_mlp = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )

        self.fc_head = nn.Linear(hidden_channels, out_dim)
        self.sc_head = nn.Linear(hidden_channels, out_dim)

        combined_dim = hidden_channels * 2 + roi_num * roi_num
        self.final_fc = nn.Linear(combined_dim, hidden_channels)
        self.final_head = nn.Linear(hidden_channels, out_dim)

    def forward(self, fc_x, fc_edge_index, fc_edge_weight=None, sc_x=None, sc_edge_index=None,
                sc_edge_weight=None, batch_fc=None, batch_sc=None):
        if batch_fc is None:
            batch_fc = torch.zeros(fc_x.shape[0], dtype=torch.long, device=fc_x.device)
        if batch_sc is None:
            batch_sc = torch.zeros(sc_x.shape[0], dtype=torch.long, device=sc_x.device)

        fc_emb = self.fc_encoder(fc_x, fc_edge_index, fc_edge_weight)
        sc_emb = self.sc_encoder(sc_x, sc_edge_index, sc_edge_weight)

        fc_pooled = global_mean_pool(fc_emb, batch_fc)
        sc_pooled = global_mean_pool(sc_emb, batch_sc)

        n_subjects = fc_pooled.shape[0]
        n_nodes = self.roi_num

        O_list = []
        for i in range(n_subjects):
            start = i * n_nodes
            end = start + n_nodes
            fc_i = fc_emb[start:end]
            sc_i = sc_emb[start:end]

            fc_exp = fc_i.unsqueeze(1).expand(-1, n_nodes, -1)
            sc_exp = sc_i.unsqueeze(0).expand(n_nodes, -1, -1)
            pair_feat = torch.cat([fc_exp, sc_exp], dim=-1)
            O_i = self.interaction_mlp(pair_feat).squeeze(-1)
            O_list.append(O_i)

        O = torch.stack(O_list, dim=0)

        fc_pred = self.fc_head(fc_pooled)
        sc_pred = self.sc_head(sc_pooled)

        O_flat = O.view(n_subjects, -1)
        combined = torch.cat([fc_pooled, sc_pooled, O_flat], dim=1)
        h = F.relu(self.final_fc(combined))
        y_pred = self.final_head(h)

        return {
            "y_pred": y_pred,
            "fc_pred": fc_pred,
            "sc_pred": sc_pred,
            "O": O,
            # ROI-level saliency proxy obtained by averaging each ROI's row/column interactions.
            "coupling_vector": 0.5 * (O.abs().mean(dim=2) + O.abs().mean(dim=1)),
            "extra": {
                "fc_emb": fc_emb,
                "sc_emb": sc_emb,
                "fc_pooled": fc_pooled,
                "sc_pooled": sc_pooled,
            },
        }
