#!/usr/bin/env python3
"""Repeated nested-CV runner for AAAI experiments.

This runner separates model selection (inner validation split) from final
outer-fold testing, repeats all folds across every requested seed, and saves
split-level predictions/saliency for paired statistics and stability analysis.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, GroupShuffleSplit, train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from metascfc.config import load_config
from metascfc.seed import set_seed
from metascfc.data.connectome_dataset import ConnectomeDataset, load_fc_sc_arrays
from metascfc.data.graph_builders import build_ms_inter_subject_graph
from metascfc.models.ms_inter_gcn import MSInterGCN
from metascfc.models.mgia_style import MGIAStyleFSCouplingModel
from metascfc.experiments import PriorGuidedTrainer
from metascfc import metrics as metric_module
from metascfc import saliency as saliency_module


class GraphDataset(Dataset):
    def __init__(self, source, roi_num=116, top_percent_fc=10.0, top_percent_sc=10.0, cache=True):
        self.source = source
        self.roi_num = roi_num
        self.top_percent_fc = top_percent_fc
        self.top_percent_sc = top_percent_sc
        self.cache = [None] * len(source) if cache else None

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        if self.cache is not None and self.cache[idx] is not None:
            return self.cache[idx]
        item = self.source[idx]
        result = build_ms_inter_subject_graph(
            item["fc"].numpy(), item["sc"].numpy(),
            item["y"].numpy() if torch.is_tensor(item["y"]) else item["y"],
            roi_num=self.roi_num,
            top_percent_fc=self.top_percent_fc,
            top_percent_sc=self.top_percent_sc,
        )
        result["subject_index"] = torch.tensor(idx, dtype=torch.long)
        if self.cache is not None:
            self.cache[idx] = result
        return result


def collate_fn(batch):
    result: Dict[str, Any] = {}
    n_nodes = [b["fc_x"].shape[0] for b in batch]
    for key in batch[0]:
        if key == "y":
            result[key] = torch.stack([b[key] for b in batch])
        elif key == "subject_index":
            result[key] = torch.stack([b[key] for b in batch])
        elif "edge_index" in key:
            edges, offset = [], 0
            for b in batch:
                edges.append(b[key].clone() + offset)
                offset += b["fc_x"].shape[0]
            result[key] = torch.cat(edges, dim=1)
        elif "edge_weight" in key:
            result[key] = torch.cat([b[key] for b in batch])
        elif key.endswith("_x"):
            result[key] = torch.cat([b[key] for b in batch], dim=0)
    result["fc_batch"] = torch.repeat_interleave(torch.arange(len(batch)), torch.tensor(n_nodes))
    result["sc_batch"] = result["fc_batch"].clone()
    return result


def build_model(cfg, n_rois):
    common = dict(
        roi_num=n_rois,
        hidden_channels=cfg.get("hidden_channels", 64),
        num_classes=cfg.get("num_classes", 2),
        dropout=cfg.get("dropout", 0.3),
        task=cfg.get("task", "regression"),
    )
    if cfg.get("baseline", "ms_inter_gcn") == "ms_inter_gcn":
        return MSInterGCN(**common)
    if cfg.get("baseline") == "mgia_style":
        return MGIAStyleFSCouplingModel(**common)
    raise ValueError(f"Unknown baseline: {cfg.get('baseline')}")


def load_priors(cfg, n_rois):
    prior_type = cfg.get("prior_type", "none")
    pcfg = cfg.get("prior", {}) or {}
    roi = module = edge = mapping = None
    if prior_type in ("node", "combined"):
        p = pcfg.get("roi_prior_path") or cfg.get("roi_prior_path")
        if not p:
            raise ValueError("roi_prior_path is required")
        df = pd.read_csv(p).sort_values("roi_index")
        roi = df["prior_score"].to_numpy(np.float32)
        if roi.shape != (n_rois,):
            raise ValueError(f"ROI prior shape {roi.shape}, expected {(n_rois,)}")
    if prior_type in ("module", "combined"):
        p = pcfg.get("module_prior_path") or cfg.get("module_prior_path")
        m = pcfg.get("roi_to_module_path") or cfg.get("roi_to_module_path")
        if not p or not m:
            raise ValueError("module_prior_path and roi_to_module_path are required")
        mdf = pd.read_csv(p)
        if "module_id" in mdf:
            mdf = mdf.sort_values("module_id")
        module = mdf["prior_score"].to_numpy(np.float32)
        mapdf = pd.read_csv(m).sort_values("roi_index")
        mapping = mapdf["module_id"].to_numpy(np.int64)
        if mapping.shape != (n_rois,):
            raise ValueError(f"ROI-module mapping shape {mapping.shape}, expected {(n_rois,)}")
    if prior_type in ("edge", "combined"):
        p = pcfg.get("edge_prior_path") or cfg.get("edge_prior_path")
        if not p:
            raise ValueError("edge_prior_path is required")
        edge = np.load(p).astype(np.float32)
        if edge.shape != (n_rois, n_rois):
            raise ValueError(f"Edge prior shape {edge.shape}, expected {(n_rois, n_rois)}")
    return roi, module, edge, mapping


def quantile_bins(y, n_bins=5):
    try:
        return pd.qcut(np.asarray(y).reshape(-1), q=min(n_bins, len(y)), labels=False, duplicates="drop")
    except Exception:
        return None


def make_inner_split(trainval_idx, y, task, val_fraction, seed, groups=None):
    labels = np.asarray(y)[trainval_idx]
    if groups is not None:
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr_local, va_local = next(gss.split(trainval_idx, labels, np.asarray(groups)[trainval_idx]))
        return np.asarray(trainval_idx)[tr_local], np.asarray(trainval_idx)[va_local]
    stratify = labels if task == "classification" else quantile_bins(labels)
    try:
        tr, va = train_test_split(trainval_idx, test_size=val_fraction, random_state=seed, shuffle=True, stratify=stratify)
    except ValueError:
        tr, va = train_test_split(trainval_idx, test_size=val_fraction, random_state=seed, shuffle=True)
    return np.asarray(tr), np.asarray(va)


def make_loader(dataset, idx, cfg, shuffle=False):
    return DataLoader(
        Subset(dataset, idx.tolist()),
        batch_size=cfg.get("batch_size", 8),
        shuffle=shuffle,
        num_workers=cfg.get("num_workers", 0),
        pin_memory=bool(cfg.get("pin_memory", True) and torch.cuda.is_available()),
        collate_fn=collate_fn,
        persistent_workers=bool(cfg.get("num_workers", 0) > 0),
    )


def compute_aux(eval_result, trainer, cfg, reference=None):
    aux: Dict[str, Any] = {"prior_alignment": {}}
    node_saliency = None
    if eval_result.get("coupling") is not None:
        cv_mean = eval_result["coupling"].mean(axis=0)
        node_saliency = saliency_module.coupling_vector_to_saliency_vector(
            torch.from_numpy(cv_mean), mode="minmax_abs"
        ).numpy()
    elif eval_result.get("O") is not None:
        O_mean = torch.from_numpy(eval_result["O"].mean(axis=0))
        fc_sal, sc_sal = saliency_module.aggregate_node_saliency_from_interactions(O_mean, mode="mean_abs")
        node_saliency = saliency_module.coupling_vector_to_saliency_vector(
            0.5 * (fc_sal + sc_sal), mode="minmax_abs"
        ).numpy()
    aux["node_saliency"] = node_saliency

    topk = int(cfg.get("topk", 10))
    if node_saliency is not None and trainer.roi_prior is not None:
        aux["prior_alignment"]["node"] = metric_module.compute_prior_alignment_metrics(
            node_saliency, trainer.roi_prior, topk=min(topk, len(node_saliency))
        )
    if node_saliency is not None and trainer.module_prior is not None and trainer.roi_to_module is not None:
        mod = saliency_module.aggregate_module_saliency_from_vector(
            torch.from_numpy(node_saliency).float(),
            torch.from_numpy(trainer.roi_to_module).long(),
            len(trainer.module_prior), agg=cfg.get("module_saliency_agg", "mean")
        ).numpy()
        aux["module_saliency"] = mod
        aux["prior_alignment"]["module"] = metric_module.compute_prior_alignment_metrics(
            mod, trainer.module_prior, topk=min(int(cfg.get("module_topk", 3)), len(mod))
        )
    if trainer.edge_prior is not None:
        if eval_result.get("O") is not None:
            edge_sal = np.abs(eval_result["O"]).mean(axis=0)
            aux["edge_saliency"] = edge_sal
            aux["prior_alignment"]["edge"] = metric_module.compute_prior_alignment_metrics(
                edge_sal.ravel(), trainer.edge_prior.ravel(), topk=min(int(cfg.get("edge_topk", 100)), edge_sal.size)
            )
        elif node_saliency is not None:
            diag = np.diag(trainer.edge_prior)
            aux["edge_saliency"] = node_saliency
            aux["prior_alignment"]["edge_diagonal"] = metric_module.compute_prior_alignment_metrics(
                node_saliency, diag, topk=min(topk, len(node_saliency))
            )
    if reference is not None and node_saliency is not None:
        aux["reference_alignment"] = {}
        if reference.get("roi") is not None:
            aux["reference_alignment"]["node"] = metric_module.compute_prior_alignment_metrics(
                node_saliency, reference["roi"], topk=min(topk, len(node_saliency)))
        if reference.get("module") is not None and reference.get("mapping") is not None:
            mod = aux.get("module_saliency")
            if mod is None:
                mod = saliency_module.aggregate_module_saliency_from_vector(
                    torch.from_numpy(node_saliency).float(), torch.from_numpy(reference["mapping"]).long(),
                    len(reference["module"]), agg=cfg.get("module_saliency_agg", "mean")).numpy()
            aux["reference_alignment"]["module"] = metric_module.compute_prior_alignment_metrics(
                mod, reference["module"], topk=min(int(cfg.get("module_topk", 3)), len(mod)))
        if reference.get("edge") is not None:
            if eval_result.get("O") is not None:
                edge_sal = np.abs(eval_result["O"]).mean(axis=0)
                aux["reference_alignment"]["edge"] = metric_module.compute_prior_alignment_metrics(
                    edge_sal.ravel(), reference["edge"].ravel(), topk=min(int(cfg.get("edge_topk",100)), edge_sal.size))
            else:
                aux["reference_alignment"]["edge_diagonal"] = metric_module.compute_prior_alignment_metrics(
                    node_saliency, np.diag(reference["edge"]), topk=min(topk,len(node_saliency)))
    return aux


def flatten_alignment(aux):
    out = {}
    for scope, vals in aux.get("prior_alignment", {}).items():
        for key, value in vals.items():
            out[f"alignment_{scope}_{key}"] = float(value)
    for scope, vals in aux.get("reference_alignment", {}).items():
        for key, value in vals.items():
            out[f"reference_alignment_{scope}_{key}"] = float(value)
    return out


def is_better(metric_name, value, best):
    return value < best if metric_name in {"rmse", "mae", "loss"} else value > best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)

    out_dir = Path(cfg["output_dir"])
    if (out_dir / "COMPLETE").exists() and not args.overwrite:
        print(f"Complete result already exists: {out_dir}. Use --overwrite to rerun.")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "saliency").mkdir(exist_ok=True)
    (out_dir / "predictions").mkdir(exist_ok=True)
    (out_dir / "checkpoints").mkdir(exist_ok=True)

    device_name = cfg.get("device", "auto")
    device = torch.device("cuda" if device_name == "auto" and torch.cuda.is_available() else ("cpu" if device_name == "auto" else device_name))
    print("Device:", device)

    fc, sc, y = load_fc_sc_arrays(cfg["data"]["fc_path"], cfg["data"]["sc_path"], cfg["data"]["y_path"])
    subject_ids = np.arange(len(y)).astype(str)
    subjects_path = cfg.get("data", {}).get("subjects_path", "inputs/dataset_SC/hcp_subjects_used.csv")
    if subjects_path and Path(subjects_path).exists():
        sdf = pd.read_csv(subjects_path)
        scol = "subject" if "subject" in sdf.columns else "Subject"
        if len(sdf) == len(y):
            subject_ids = sdf[scol].astype(str).to_numpy()
        else:
            print(f"WARNING: {subjects_path} has {len(sdf)} rows but arrays have {len(y)} subjects")
    groups = None
    groups_path = cfg.get("data", {}).get("groups_path")
    if groups_path:
        groups = np.load(groups_path, allow_pickle=True).reshape(-1)
        if len(groups) != len(y):
            raise ValueError(f"groups length {len(groups)} does not match subjects {len(y)}")
        print(f"Family/group-aware evaluation enabled: {len(np.unique(groups))} groups")
    elif cfg.get("require_group_split", False):
        raise ValueError("require_group_split=true but data.groups_path was not supplied")
    else:
        print("WARNING: no family/group IDs supplied; folds are subject-wise, not family-wise.")
    n_rois = int(cfg.get("roi_num", fc.shape[1]))
    dataset = GraphDataset(
        ConnectomeDataset(fc, sc, y), n_rois,
        cfg.get("top_percent_fc", 10.0), cfg.get("top_percent_sc", 10.0),
        cache=cfg.get("cache_graphs", True),
    )
    roi_prior, module_prior, edge_prior, mapping = load_priors(cfg, n_rois)
    rcfg = cfg.get("reference_prior", {}) or {}
    reference = {"roi": None, "module": None, "edge": None, "mapping": None}
    if rcfg:
        if rcfg.get("roi_prior_path"):
            reference["roi"] = pd.read_csv(rcfg["roi_prior_path"]).sort_values("roi_index")["prior_score"].to_numpy(np.float32)
        if rcfg.get("module_prior_path"):
            rdf = pd.read_csv(rcfg["module_prior_path"]); rdf = rdf.sort_values("module_id") if "module_id" in rdf else rdf
            reference["module"] = rdf["prior_score"].to_numpy(np.float32)
        if rcfg.get("edge_prior_path"):
            reference["edge"] = np.load(rcfg["edge_prior_path"]).astype(np.float32)
        if rcfg.get("roi_to_module_path"):
            reference["mapping"] = pd.read_csv(rcfg["roi_to_module_path"]).sort_values("roi_index")["module_id"].to_numpy(np.int64)

    seeds = [int(s) for s in cfg.get("seeds", [0, 1, 2, 3, 4])]
    n_folds = int(cfg.get("n_folds", 5))
    val_fraction = float(cfg.get("val_fraction", 0.15))
    selection_metric = cfg.get("selection_metric", "rmse" if cfg.get("task") == "regression" else "auroc")
    patience = int(cfg.get("early_stopping_patience", 20))
    min_epochs = int(cfg.get("min_epochs", 20))

    split_rows, all_saliency = [], []
    metadata = {
        "config": cfg, "n_subjects": len(dataset), "n_rois": n_rois,
        "seeds": seeds, "n_folds": n_folds, "device": str(device),
        "selection_metric": selection_metric,
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    for seed in seeds:
        if groups is not None:
            splitter = GroupKFold(n_splits=n_folds)
            split_iter = splitter.split(np.arange(len(y)), np.asarray(y).reshape(-1), groups)
        elif cfg.get("task") == "classification":
            splitter = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(np.arange(len(y)), np.asarray(y).reshape(-1))
        else:
            splitter = KFold(n_folds, shuffle=True, random_state=seed)
            split_iter = splitter.split(np.arange(len(y)))

        for fold, (trainval_idx, test_idx) in enumerate(split_iter):
            split_seed = seed * 1000 + fold
            set_seed(split_seed)
            train_idx, val_idx = make_inner_split(trainval_idx, y, cfg.get("task", "regression"), val_fraction, split_seed, groups=groups)
            train_loader = make_loader(dataset, train_idx, cfg, shuffle=True)
            val_loader = make_loader(dataset, val_idx, cfg, shuffle=False)
            test_loader = make_loader(dataset, np.asarray(test_idx), cfg, shuffle=False)

            model = build_model(cfg, n_rois).to(device)
            trainer = PriorGuidedTrainer(model, cfg, device)
            trainer.set_priors(roi_prior, module_prior, edge_prior, mapping)
            if cfg.get("task") == "regression" and cfg.get("standardize_labels_within_fold", True):
                train_targets = np.asarray(y).reshape(-1)[train_idx].astype(float)
                trainer.set_target_scaler(train_targets.mean(), train_targets.std())
            optimizer = torch.optim.Adam(model.parameters(), lr=trainer.learning_rate, weight_decay=trainer.weight_decay)

            best = float("inf") if selection_metric in {"rmse", "mae", "loss"} else -float("inf")
            best_state: Optional[Dict[str, torch.Tensor]] = None
            best_epoch, wait = -1, 0
            started = time.time()
            for epoch in range(trainer.n_epochs):
                train_loss = trainer.train_epoch(train_loader, optimizer)
                val_result = trainer.evaluate(val_loader)
                value = float(val_result["metrics"].get(selection_metric, train_loss))
                if is_better(selection_metric, value, best):
                    best, best_epoch, wait = value, epoch, 0
                    best_state = copy.deepcopy(model.state_dict())
                else:
                    wait += 1
                if epoch + 1 >= min_epochs and wait >= patience:
                    break
            if best_state is not None:
                model.load_state_dict(best_state)

            test_result = trainer.evaluate(test_loader)
            aux = compute_aux(test_result, trainer, cfg, reference=reference)
            split_id = f"seed{seed:02d}_fold{fold:02d}"
            row = {
                "experiment_id": cfg.get("experiment_id", cfg.get("experiment_name", out_dir.name)),
                "experiment_name": cfg.get("experiment_name", out_dir.name),
                "prior_type": cfg.get("prior_type", "none"),
                "prior_source": cfg.get("prior_source", "none"),
                "seed": seed, "fold": fold, "split_id": split_id,
                "n_train": len(train_idx), "n_val": len(val_idx), "n_test": len(test_idx),
                "best_epoch": best_epoch + 1, "best_val_metric": best,
                "runtime_seconds": time.time() - started,
                "target_train_mean": trainer.target_mean, "target_train_std": trainer.target_std,
                "group_aware": groups is not None,
                **{k: float(v) for k, v in test_result["metrics"].items()},
                **flatten_alignment(aux),
            }
            split_rows.append(row)

            pred_df = pd.DataFrame({
                "subject_index": np.asarray(test_idx),
                "subject_id": subject_ids[np.asarray(test_idx)],
                "target": np.asarray(test_result["targets"]).reshape(-1),
                "prediction": np.asarray(test_result["predictions"]).reshape(-1),
                "seed": seed, "fold": fold,
            })
            pred_df.to_csv(out_dir / "predictions" / f"{split_id}.csv", index=False)

            arrays = {}
            for key in ("node_saliency", "module_saliency", "edge_saliency"):
                if aux.get(key) is not None:
                    arrays[key] = np.asarray(aux[key])
            np.savez_compressed(out_dir / "saliency" / f"{split_id}.npz", **arrays)
            if aux.get("node_saliency") is not None:
                all_saliency.append(np.asarray(aux["node_saliency"]))
            if cfg.get("save_checkpoints", False):
                torch.save(model.state_dict(), out_dir / "checkpoints" / f"{split_id}.pt")

            pd.DataFrame(split_rows).to_csv(out_dir / "split_metrics.csv", index=False)
            print(split_id, json.dumps(row, indent=2))

    df = pd.DataFrame(split_rows)
    numeric = [c for c in df.columns if c not in {"experiment_id", "experiment_name", "prior_type", "prior_source", "split_id"} and pd.api.types.is_numeric_dtype(df[c])]
    summary = {"config": cfg, "n_subjects": len(dataset), "n_runs": len(df), "split_metrics": split_rows}
    for col in numeric:
        summary[f"{col}_mean"] = float(df[col].mean())
        summary[f"{col}_std"] = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    if all_saliency:
        np.save(out_dir / "all_node_saliency.npy", np.stack(all_saliency))
    (out_dir / "COMPLETE").write_text("ok\n")
    print(f"Completed {len(df)} outer test evaluations: {out_dir}")


if __name__ == "__main__":
    main()
