from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data


def connectivity_to_graph(
    matrix: np.ndarray,
    top_percent: float = 10.0,
    include_self_loops: bool = True,
    use_node_features: str = "connectivity_profile",
) -> Data:
    n = matrix.shape[0]
    flat = matrix.flatten()
    threshold = float(np.percentile(flat, 100 - top_percent))
    adj = (matrix >= threshold).astype(float)
    if include_self_loops:
        np.fill_diagonal(adj, 1.0)
    else:
        np.fill_diagonal(adj, 0.0)

    rows, cols = np.where(adj > 0)
    edge_index = torch.from_numpy(np.vstack([rows, cols])).long()
    edge_weight = torch.from_numpy(matrix[rows, cols]).float()

    if use_node_features == "connectivity_profile":
        x = torch.from_numpy(matrix).float()
    elif use_node_features == "degree":
        x = torch.from_numpy(adj.sum(axis=1, keepdims=True)).float()
    elif use_node_features == "eigenvector":
        try:
            from scipy.linalg import eigh
            vals, vecs = eigh(matrix)
            x = torch.from_numpy(vecs[:, -min(8, n):]).float()
        except Exception:
            x = torch.from_numpy(matrix).float()
    else:
        x = torch.from_numpy(matrix).float()

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    return data


def build_ms_inter_subject_graph(
    fc: np.ndarray,
    sc: np.ndarray,
    y: np.ndarray,
    roi_num: int = 116,
    top_percent_fc: float = 10.0,
    top_percent_sc: float = 10.0,
) -> Dict[str, torch.Tensor]:
    fc_graph = connectivity_to_graph(fc, top_percent=top_percent_fc)
    sc_graph = connectivity_to_graph(sc, top_percent=top_percent_sc)

    return {
        "fc_x": fc_graph.x,
        "fc_edge_index": fc_graph.edge_index,
        "fc_edge_weight": fc_graph.edge_weight,
        "sc_x": sc_graph.x,
        "sc_edge_index": sc_graph.edge_index,
        "sc_edge_weight": sc_graph.edge_weight,
        "y": torch.from_numpy(np.array(y)).float(),
    }


def collate_ms_inter_graphs(
    batch_graphs: list,
) -> Dict[str, torch.Tensor]:
    batch = {}
    for key in batch_graphs[0].keys():
        if key == "y":
            batch[key] = torch.stack([g[key] for g in batch_graphs])
        elif "edge_index" in key:
            offset = 0
            edges = []
            for i, g in enumerate(batch_graphs):
                ei = g[key].clone()
                ei = ei + offset
                offset += g["fc_x"].shape[0]
                edges.append(ei)
            batch[key] = torch.cat(edges, dim=1)
        elif "edge_weight" in key:
            batch[key] = torch.cat([g[key] for g in batch_graphs])
        elif "x" in key:
            batch[key] = torch.cat([g[key] for g in batch_graphs], dim=0)
    return batch
