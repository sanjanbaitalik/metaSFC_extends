#!/usr/bin/env python3
"""
Reproduce MS-Inter-GCN baseline. Loads config, sets up synthetic or real data,
trains the model, and saves outputs.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from metascfc.config import load_config, get_output_dir
from metascfc.seed import set_seed
from metascfc.data.connectome_dataset import ConnectomeDataset, SyntheticConnectomeDataset, load_fc_sc_arrays
from metascfc.data.graph_builders import build_ms_inter_subject_graph
from metascfc.models.ms_inter_gcn import MSInterGCN
from metascfc.experiments import PriorGuidedTrainer
from metascfc import metrics as metric_module
from metascfc import visualize as vis
from torch.utils.data import DataLoader, Dataset


class GraphDataset(Dataset):
    def __init__(self, source_dataset, roi_num: int = 116, top_percent_fc: float = 10.0, top_percent_sc: float = 10.0):
        self.source = source_dataset
        self.roi_num = roi_num
        self.top_percent_fc = top_percent_fc
        self.top_percent_sc = top_percent_sc

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        item = self.source[idx]
        fc = item["fc"].numpy()
        sc = item["sc"].numpy()
        y = item["y"].numpy() if torch.is_tensor(item["y"]) else item["y"]
        graph_data = build_ms_inter_subject_graph(
            fc, sc, y, roi_num=self.roi_num,
            top_percent_fc=self.top_percent_fc,
            top_percent_sc=self.top_percent_sc,
        )
        return graph_data


def collate_fn(batch):
    result = {}
    n_nodes_list = [b["fc_x"].shape[0] for b in batch]
    for key in batch[0].keys():
        if key == "y":
            result[key] = torch.stack([b[key] for b in batch])
        elif "edge_index" in key:
            offset = 0
            edges = []
            for i, b in enumerate(batch):
                ei = b[key].clone()
                ei = ei + offset
                offset += n_nodes_list[i]
                edges.append(ei)
            result[key] = torch.cat(edges, dim=1)
        elif "edge_weight" in key:
            result[key] = torch.cat([b[key] for b in batch])
        elif "x" in key:
            result[key] = torch.cat([b[key] for b in batch], dim=0)
    result["fc_batch"] = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(n_nodes_list))
    result["sc_batch"] = result["fc_batch"].clone()
    return result


def main():
    parser = argparse.ArgumentParser(description="Reproduce MS-Inter-GCN baseline")
    parser.add_argument("--config", type=str, default="configs/baseline_ms_inter_gcn.yaml",
                        help="Path to YAML config")
    parser.add_argument("--smoke", action="store_true",
                        help="Use smoke test config")
    args = parser.parse_args()

    config_path = "configs/baseline_ms_inter_gcn_smoke.yaml" if args.smoke else args.config
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    use_synthetic = cfg.get("use_synthetic_data", False)
    if use_synthetic:
        n_subjects = cfg.get("synthetic_n_subjects", 12)
        n_rois = cfg.get("synthetic_n_rois", 10)
        print(f"Generating synthetic data: {n_subjects} subjects, {n_rois} ROIs")
        synth_dataset = SyntheticConnectomeDataset(
            n_subjects=n_subjects, n_rois=n_rois,
            task=cfg.get("task", "regression"), seed=cfg.get("seed", 42),
        )
        graph_dataset = GraphDataset(synth_dataset, roi_num=n_rois)
    else:
        data_cfg = cfg["data"]
        fc, sc, y = load_fc_sc_arrays(data_cfg["fc_path"], data_cfg["sc_path"], data_cfg["y_path"])
        print(f"Loaded data: FC {fc.shape}, SC {sc.shape}, Y {y.shape}")
        base_dataset = ConnectomeDataset(fc, sc, y)
        graph_dataset = GraphDataset(base_dataset, roi_num=cfg.get("roi_num", 116))

    n_folds = cfg.get("n_folds", 5)
    n_rois = cfg.get("roi_num", 116)

    from sklearn.model_selection import KFold
    indices = np.arange(len(graph_dataset))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.get("seed", 42))

    all_metrics = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        set_seed(cfg.get("seed", 42) + fold)

        model = MSInterGCN(
            roi_num=n_rois,
            hidden_channels=cfg.get("hidden_channels", 64),
            num_classes=cfg.get("num_classes", 2),
            dropout=cfg.get("dropout", 0.3),
            task=cfg.get("task", "regression"),
        ).to(device)

        trainer = PriorGuidedTrainer(model, cfg, device)

        train_subset = torch.utils.data.Subset(graph_dataset, train_idx)
        val_subset = torch.utils.data.Subset(graph_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=cfg.get("batch_size", 8), collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=cfg.get("batch_size", 8), collate_fn=collate_fn)

        metrics, aux = trainer.train_and_evaluate(train_loader, val_loader)
        all_metrics.append(metrics)
        print(f"  Fold {fold + 1} metrics: {json.dumps(metrics, indent=2)}")

    output_dir = get_output_dir(cfg)
    aggregated = {
        "fold_metrics": all_metrics,
        "config": cfg,
    }
    metric_keys = all_metrics[0].keys()
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}")
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


if __name__ == "__main__":
    main()
