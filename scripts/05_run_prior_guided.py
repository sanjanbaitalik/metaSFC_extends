#!/usr/bin/env python3
"""
Run prior-guided FC-SC coupling experiments.

Supported baselines:
- ms_inter_gcn: corresponding-ROI FC_i-SC_i coupling vector [B,N]
- mgia_style: clean-room cross-ROI FC-SC interaction matrix [B,N,N]

For ms_inter_gcn, edge prior guidance uses only the diagonal/corresponding-edge
part of edge_prior.npy. For full cross-ROI edge-prior training, use mgia_style.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import KFold

from metascfc.config import load_config, get_output_dir
from metascfc.seed import set_seed
from metascfc.data.connectome_dataset import ConnectomeDataset, SyntheticConnectomeDataset, load_fc_sc_arrays
from metascfc.data.graph_builders import build_ms_inter_subject_graph
from metascfc.models.ms_inter_gcn import MSInterGCN
from metascfc.models.mgia_style import MGIAStyleFSCouplingModel
from metascfc.experiments import PriorGuidedTrainer
from metascfc import visualize as vis


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
        return build_ms_inter_subject_graph(
            fc, sc, y, roi_num=self.roi_num,
            top_percent_fc=self.top_percent_fc,
            top_percent_sc=self.top_percent_sc,
        )


def collate_fn(batch):
    result = {}
    n_nodes_list = [b["fc_x"].shape[0] for b in batch]
    for key in batch[0].keys():
        if key == "y":
            result[key] = torch.stack([b[key] for b in batch])
        elif "edge_index" in key:
            offset = 0
            edges = []
            for b in batch:
                ei = b[key].clone() + offset
                offset += b["fc_x"].shape[0]
                edges.append(ei)
            result[key] = torch.cat(edges, dim=1)
        elif "edge_weight" in key:
            result[key] = torch.cat([b[key] for b in batch])
        elif "x" in key:
            result[key] = torch.cat([b[key] for b in batch], dim=0)
    result["fc_batch"] = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(n_nodes_list))
    result["sc_batch"] = result["fc_batch"].clone()
    return result


def build_model(cfg, n_rois: int):
    baseline = cfg.get("baseline", "ms_inter_gcn")
    common = dict(
        roi_num=n_rois,
        hidden_channels=cfg.get("hidden_channels", 64),
        num_classes=cfg.get("num_classes", 2),
        dropout=cfg.get("dropout", 0.3),
        task=cfg.get("task", "regression"),
    )
    if baseline == "ms_inter_gcn":
        return MSInterGCN(**common)
    if baseline == "mgia_style":
        return MGIAStyleFSCouplingModel(**common)
    raise ValueError(f"Unknown baseline: {baseline}")


def load_roi_to_module_vector(path: str, n_rois: int) -> np.ndarray:
    df = pd.read_csv(path)
    if "roi_index" not in df.columns or "module_id" not in df.columns:
        raise ValueError("roi_to_module CSV must contain roi_index and module_id columns.")
    df = df.sort_values("roi_index")
    if len(df) != n_rois:
        raise ValueError(f"roi_to_module has {len(df)} rows, expected {n_rois}.")
    expected = np.arange(1, n_rois + 1)
    if not np.array_equal(df["roi_index"].to_numpy(), expected):
        raise ValueError("roi_to_module roi_index must be consecutive 1..N after sorting.")
    return df["module_id"].to_numpy(dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Run prior-guided FC-SC coupling experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument("--roi_prior_path", type=str, default=None, help="Override ROI prior CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    print(f"Baseline: {cfg.get('baseline', 'ms_inter_gcn')}")
    print(f"Prior type: {cfg['prior_type']}")
    print(f"Lambda node: {cfg.get('lambda_node', 0.0)}, module: {cfg.get('lambda_module', 0.0)}, edge: {cfg.get('lambda_edge', 0.0)}")

    use_synthetic = cfg.get("use_synthetic_data", False)
    if use_synthetic:
        n_subjects = cfg.get("synthetic_n_subjects", 20)
        n_rois = cfg.get("synthetic_n_rois", 10)
        synth_dataset = SyntheticConnectomeDataset(
            n_subjects=n_subjects, n_rois=n_rois,
            task=cfg.get("task", "regression"), seed=cfg.get("seed", 42),
        )
        graph_dataset = GraphDataset(synth_dataset, roi_num=n_rois)
    else:
        data_cfg = cfg["data"]
        fc, sc, y = load_fc_sc_arrays(data_cfg["fc_path"], data_cfg["sc_path"], data_cfg["y_path"])
        base_dataset = ConnectomeDataset(fc, sc, y)
        graph_dataset = GraphDataset(base_dataset, roi_num=cfg.get("roi_num", 116))

    n_rois = cfg.get("roi_num", 116)
    n_folds = cfg.get("n_folds", 5)
    indices = np.arange(len(graph_dataset))
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cfg.get("seed", 42))

    roi_prior = None
    module_prior = None
    edge_prior = None
    roi_to_module = None

    # Support both config styles:
    #   1) top-level: roi_prior_path: ...
    #   2) nested: prior: {roi_prior_path: ...}
    # Older configs used top-level keys, while the HCP configs often use nested keys.
    prior_cfg = cfg.get("prior", {}) or {}

    def _cfg_path(key, cli_override=None):
        return cli_override or cfg.get(key) or prior_cfg.get(key)

    prior_type = cfg.get("prior_type", "none")

    roi_prior_path = _cfg_path("roi_prior_path", args.roi_prior_path)
    if roi_prior_path and prior_type in ("node", "combined"):
        prior_df = pd.read_csv(roi_prior_path).sort_values("roi_index")
        roi_prior = prior_df["prior_score"].to_numpy(dtype=np.float32)
        print(f"Loaded ROI prior from {roi_prior_path}, shape {roi_prior.shape}")

    module_prior_path = _cfg_path("module_prior_path")
    if module_prior_path and prior_type in ("module", "combined"):
        mod_df = pd.read_csv(module_prior_path)
        if "module_id" in mod_df.columns:
            mod_df = mod_df.sort_values("module_id")
        module_prior = mod_df["prior_score"].to_numpy(dtype=np.float32)
        print(f"Loaded module prior from {module_prior_path}, shape {module_prior.shape}")

    roi_to_module_path = _cfg_path("roi_to_module_path")
    if roi_to_module_path and prior_type in ("module", "combined"):
        roi_to_module = load_roi_to_module_vector(roi_to_module_path, n_rois)
        print(f"Loaded ROI-to-module mapping from {roi_to_module_path}, shape {roi_to_module.shape}, modules={roi_to_module.max()+1}")

    edge_prior_path = _cfg_path("edge_prior_path")
    if edge_prior_path and prior_type in ("edge", "combined"):
        edge_prior = np.load(str(edge_prior_path)).astype(np.float32)
        print(f"Loaded edge prior from {edge_prior_path}, shape {edge_prior.shape}")
        if cfg.get("baseline", "ms_inter_gcn") == "ms_inter_gcn":
            print("NOTE: ms_inter_gcn has only corresponding-ROI coupling; edge loss will use diag(edge_prior).")

    # Fail loudly when a prior experiment is requested but the required prior was not loaded.
    if prior_type in ("node", "combined") and cfg.get("lambda_node", 0.0) > 0 and roi_prior is None:
        raise ValueError("Node-prior experiment requested, but ROI prior was not loaded. Check roi_prior_path or prior.roi_prior_path.")
    if prior_type in ("module", "combined") and cfg.get("lambda_module", 0.0) > 0:
        if module_prior is None:
            raise ValueError("Module-prior experiment requested, but module prior was not loaded. Check module_prior_path or prior.module_prior_path.")
        if roi_to_module is None:
            raise ValueError("Module-prior experiment requested, but ROI-to-module mapping was not loaded. Check roi_to_module_path or prior.roi_to_module_path.")
    if prior_type in ("edge", "combined") and cfg.get("lambda_edge", 0.0) > 0 and edge_prior is None:
        raise ValueError("Edge-prior experiment requested, but edge prior was not loaded. Check edge_prior_path or prior.edge_prior_path.")

    all_metrics = []
    all_aux = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
        print(f"\n=== Fold {fold + 1}/{n_folds} ===")
        set_seed(cfg.get("seed", 42) + fold)

        model = build_model(cfg, n_rois).to(device)
        trainer = PriorGuidedTrainer(model, cfg, device)
        trainer.set_priors(
            roi_prior=roi_prior,
            module_prior=module_prior,
            edge_prior=edge_prior,
            roi_to_module=roi_to_module,
        )

        train_subset = Subset(graph_dataset, train_idx)
        val_subset = Subset(graph_dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=cfg.get("batch_size", 8), collate_fn=collate_fn)
        val_loader = DataLoader(val_subset, batch_size=cfg.get("batch_size", 8), collate_fn=collate_fn)

        metrics, aux = trainer.train_and_evaluate(train_loader, val_loader)
        all_metrics.append(metrics)
        all_aux.append(aux)
        print(f"  Fold {fold + 1}: {json.dumps(metrics, indent=2)}")

    output_dir = get_output_dir(cfg)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)

    aggregated = {"fold_metrics": all_metrics, "fold_aux": all_aux, "config": cfg}

    metric_keys = all_metrics[0].keys()
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    # Flatten nested prior alignment dictionaries: node/module/edge scopes.
    flat_alignments = []
    for a in all_aux:
        d = {}
        for scope, vals in a.get("prior_alignment", {}).items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    d[f"prior_alignment_{scope}_{k}"] = float(v)
        flat_alignments.append(d)
    for key in sorted({k for d in flat_alignments for k in d.keys()}):
        values = [d[key] for d in flat_alignments if key in d]
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_std"] = float(np.std(values))

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2, default=str)

    vis.set_style()
    if roi_prior is not None and any(a.get("node_saliency") is not None for a in all_aux):
        learned = np.mean([a["node_saliency"] for a in all_aux if a.get("node_saliency") is not None], axis=0)
        vis.plot_learned_vs_prior_scatter(
            learned, roi_prior,
            title=f"Learned vs ROI Prior ({cfg['prior_type']}, lambda={cfg.get('lambda_node', 0.0)})",
            save_path=output_dir / "figures" / "learned_vs_roi_prior_scatter.png",
        )

    print(f"\nResults saved to {output_dir}")
    for key in metric_keys:
        values = [m[key] for m in all_metrics]
        print(f"  {key}: {np.mean(values):.4f} +/- {np.std(values):.4f}")


if __name__ == "__main__":
    main()
