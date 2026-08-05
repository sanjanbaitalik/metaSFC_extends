#!/usr/bin/env python3
"""Repeated nested-CV reimplementations of IMG-GCN and M-GCN.

Both models use the exact subject cohort and outer/inner split generator used by
the MetaSFC study. Hyperparameters default to the original papers. Each split is
saved immediately, so interrupted runs can be resumed safely.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from metascfc.benchmark_utils import (
    aggregate_split_metrics, atomic_write_csv, binary_sc_adjacency, choose_device,
    iter_nested_splits, load_connectomes, prediction_metrics, rowwise_topk_adjacency,
    save_json, seed_level_metrics, set_all_seeds,
)
from metascfc.sota_baselines import IMGGCNRegressor, MGCNRegressor


class DenseConnectomeDataset(Dataset):
    def __init__(self, fc, sc, fc_adj, sc_adj, y, indices):
        self.fc = torch.from_numpy(fc[indices]).float()
        self.sc = torch.from_numpy(sc[indices]).float()
        self.fc_adj = torch.from_numpy(fc_adj[indices]).float()
        self.sc_adj = torch.from_numpy(sc_adj[indices]).float()
        self.y = torch.from_numpy(y[indices]).float()
        self.indices = torch.from_numpy(np.asarray(indices)).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.fc[idx], self.sc[idx], self.fc_adj[idx], self.sc_adj[idx], self.y[idx], self.indices[idx]


def make_loader(fc, sc, fc_adj, sc_adj, y, idx, batch_size, shuffle, workers):
    ds = DenseConnectomeDataset(fc, sc, fc_adj, sc_adj, y, idx)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available(), persistent_workers=workers > 0)


def build_model(method_id: str, spec: Dict, n_rois: int, module_ids: np.ndarray) -> torch.nn.Module:
    if method_id == "MGCN":
        return MGCNRegressor(n_rois=n_rois, channels=int(spec.get("channels", 32)), graph_dim=int(spec.get("graph_dim", 256)))
    if method_id == "IMG_GCN":
        return IMGGCNRegressor(
            module_ids=module_ids, node_feature_dim=n_rois,
            graph_hidden=int(spec.get("graph_hidden", 16)),
            bottleneck_ratio=int(spec.get("bottleneck_ratio", 2)),
            dropout=float(spec.get("dropout", 0.5)),
            attention_mode=str(spec.get("attention_mode", "full")),
            smoke_attention_dim=int(spec.get("smoke_attention_dim", 64)),
        )
    raise ValueError(f"Unknown model {method_id}")


def make_optimizer(model, method_id: str, spec: Dict):
    if method_id == "MGCN":
        opt = torch.optim.SGD(model.parameters(), lr=float(spec.get("learning_rate", 0.001)),
                              momentum=float(spec.get("momentum", 0.9)), weight_decay=float(spec.get("weight_decay", 0.001)))
        sch = torch.optim.lr_scheduler.StepLR(opt, step_size=int(spec.get("lr_step", 10)), gamma=float(spec.get("lr_gamma", 0.9)))
    else:
        opt = torch.optim.Adam(model.parameters(), lr=float(spec.get("learning_rate", 0.0005)),
                               weight_decay=float(spec.get("weight_decay", 0.001)))
        sch = None
    return opt, sch


def loss_fn(method_id: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = torch.mean((pred - target) ** 2)
    if method_id == "MGCN":
        return mse + torch.mean(torch.abs(pred - target))
    return torch.sqrt(mse + 1e-8)


def forward_model(model, method_id, batch, device, amp_enabled):
    fc, sc, fc_adj, sc_adj, y, indices = batch
    fc, sc, fc_adj, sc_adj, y = [x.to(device, non_blocking=True) for x in (fc, sc, fc_adj, sc_adj, y)]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled):
        if method_id == "MGCN": pred = model(fc, sc_adj)
        else: pred = model(fc, sc, fc_adj, sc_adj)
    return pred.float(), y.float(), indices.numpy()


@torch.no_grad()
def evaluate(model, method_id, loader, device, y_mean, y_std, amp_enabled):
    model.eval(); preds=[]; targets=[]; indices=[]
    for batch in loader:
        pred_z, y_raw, idx = forward_model(model, method_id, batch, device, amp_enabled)
        preds.append((pred_z.cpu().numpy() * y_std + y_mean).reshape(-1))
        targets.append(y_raw.cpu().numpy().reshape(-1)); indices.append(idx.reshape(-1))
    pred=np.concatenate(preds); target=np.concatenate(targets); index=np.concatenate(indices)
    return prediction_metrics(target, pred), pred, target, index


def train_one_split(method_id, spec, loaders, device, y_train, amp_enabled):
    train_loader, val_loader, test_loader = loaders
    model = build_model(method_id, spec, train_loader.dataset.fc.shape[1], spec["module_ids"]).to(device)
    optimizer, scheduler = make_optimizer(model, method_id, spec)
    y_mean, y_std = float(np.mean(y_train)), float(np.std(y_train)); y_std = y_std if y_std >= 1e-8 else 1.0
    epochs=int(spec.get("epochs", 40 if method_id == "MGCN" else 20))
    patience=int(spec.get("patience", epochs)); min_epochs=int(spec.get("min_epochs", 1))
    best_rmse=float("inf"); best_state=None; best_epoch=0; wait=0
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred_z, y_raw, _ = forward_model(model, method_id, batch, device, amp_enabled)
            target_z = (y_raw - y_mean) / y_std
            loss = loss_fn(method_id, pred_z, target_z)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss in {method_id} at epoch {epoch+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec.get("grad_clip", 5.0)))
            optimizer.step()
        if scheduler is not None: scheduler.step()
        val_metrics, _, _, _ = evaluate(model, method_id, val_loader, device, y_mean, y_std, amp_enabled)
        if val_metrics["rmse"] < best_rmse - 1e-8:
            best_rmse=val_metrics["rmse"]; best_state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_epoch=epoch+1; wait=0
        else: wait += 1
        if epoch + 1 >= min_epochs and wait >= patience: break
    if best_state is None: raise RuntimeError(f"No checkpoint selected for {method_id}")
    model.load_state_dict(best_state)
    metrics, pred, target, indices = evaluate(model, method_id, test_loader, device, y_mean, y_std, amp_enabled)
    return metrics, pred, target, indices, best_epoch, best_rmse, sum(p.numel() for p in model.parameters())


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aaai/sota_graph_baselines.yaml")
    ap.add_argument("--models", nargs="*", choices=["MGCN","IMG_GCN"])
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--folds", nargs="*", type=int, help="Optional fold-index override")
    ap.add_argument("--overwrite", action="store_true")
    args=ap.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out_dir=Path(cfg.get("output_dir","outputs/aaai/sota_graph_baselines")); complete=out_dir/"COMPLETE"
    if args.overwrite and out_dir.exists(): import shutil; shutil.rmtree(out_dir)
    (out_dir/"predictions").mkdir(parents=True,exist_ok=True)
    device=choose_device(cfg.get("device","auto")); print("Device:",device)
    fc,sc,y,subject_ids,groups=load_connectomes(cfg["data"])
    module_df=pd.read_csv(cfg["module_map_path"]).sort_values("roi_index")
    module_ids=module_df["module_id"].to_numpy(np.int64)
    if len(module_ids)!=fc.shape[1]: raise ValueError("Module map length does not match ROI count")
    print("Precomputing graph adjacencies...")
    fc_adj=rowwise_topk_adjacency(fc,float(cfg.get("top_percent_fc",10.0)),positive_only=True)
    img_sc_adj=rowwise_topk_adjacency(sc,float(cfg.get("top_percent_sc",10.0)),positive_only=True)
    mgcn_sc_adj=binary_sc_adjacency(sc)
    models=args.models if args.models else list(cfg["models"].keys())
    seeds=args.seeds if args.seeds else [int(s) for s in cfg["seeds"]]
    n_folds=int(cfg.get("n_folds",5)); val_fraction=float(cfg.get("val_fraction",0.15)); workers=int(cfg.get("num_workers",0))
    existing=out_dir/"split_metrics.csv"; rows=[] if args.overwrite or not existing.exists() else pd.read_csv(existing).to_dict("records")
    done={(r["method_id"],int(r["seed"]),int(r["fold"])) for r in rows}

    selected_folds=set(args.folds) if args.folds else None
    for seed,fold,train_idx,val_idx,test_idx in iter_nested_splits(y,seeds,n_folds,val_fraction,groups):
        if selected_folds is not None and fold not in selected_folds: continue
        split_seed=seed*1000+fold; split_id=f"seed{seed:02d}_fold{fold:02d}"
        for method_id in models:
            if (method_id,seed,fold) in done: print("SKIP",method_id,split_id); continue
            set_all_seeds(split_seed); spec=dict(cfg["models"][method_id]); spec["module_ids"]=module_ids
            batch_size=int(spec.get("batch_size",16 if method_id=="MGCN" else 4))
            sc_adj=mgcn_sc_adj if method_id=="MGCN" else img_sc_adj
            loaders=(
                make_loader(fc,sc,fc_adj,sc_adj,y,train_idx,batch_size,True,workers),
                make_loader(fc,sc,fc_adj,sc_adj,y,val_idx,batch_size,False,workers),
                make_loader(fc,sc,fc_adj,sc_adj,y,test_idx,batch_size,False,workers),
            )
            amp=bool(spec.get("amp",True) and device.type=="cuda")
            started=time.time()
            try:
                metrics,pred,target,index,best_epoch,best_val_rmse,n_params=train_one_split(
                    method_id,spec,loaders,device,y[train_idx],amp)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and method_id=="IMG_GCN" and batch_size>1:
                    import gc
                    gc.collect()
                    if device.type=="cuda": torch.cuda.empty_cache()
                    print(f"OOM with batch_size={batch_size}; retrying IMG_GCN with batch_size=1")
                    loaders=(
                        make_loader(fc,sc,fc_adj,sc_adj,y,train_idx,1,True,workers),
                        make_loader(fc,sc,fc_adj,sc_adj,y,val_idx,1,False,workers),
                        make_loader(fc,sc,fc_adj,sc_adj,y,test_idx,1,False,workers),
                    )
                    metrics,pred,target,index,best_epoch,best_val_rmse,n_params=train_one_split(
                        method_id,spec,loaders,device,y[train_idx],amp)
                    batch_size=1
                else: raise
            row={"method_id":method_id,"method_name":spec["name"],"method_family":"sota_graph_reimplementation",
                 "seed":seed,"fold":fold,"split_id":split_id,"n_train":len(train_idx),"n_val":len(val_idx),"n_test":len(test_idx),
                 "best_epoch":best_epoch,"best_val_rmse":best_val_rmse,"parameters":n_params,"batch_size_used":batch_size,
                 "runtime_seconds":time.time()-started,"device":str(device),"amp":amp,"group_aware":groups is not None,**metrics}
            rows.append(row);done.add((method_id,seed,fold));atomic_write_csv(pd.DataFrame(rows),existing)
            pd.DataFrame({"subject_index":index,"subject_id":subject_ids[index],"target":target,"prediction":pred,
                          "seed":seed,"fold":fold,"method_id":method_id}).to_csv(out_dir/"predictions"/f"{method_id}_{split_id}.csv",index=False)
            print(method_id,split_id,json.dumps(metrics),f"epoch={best_epoch} params={n_params:,}")
            del loaders
            if device.type=="cuda": torch.cuda.empty_cache()
    df=pd.DataFrame(rows).sort_values(["method_id","seed","fold"])
    atomic_write_csv(aggregate_split_metrics(df),out_dir/"summary.csv")
    atomic_write_csv(seed_level_metrics(df),out_dir/"seed_level_metrics.csv")
    summary=aggregate_split_metrics(df)
    latex=summary[["Method","Pearson Mean","Pearson Std","RMSE Mean","RMSE Std","MAE Mean","MAE Std"]].copy()
    latex["Pearson $\\uparrow$"]=[f"{m:.3f} $\\pm$ {s:.3f}" for m,s in zip(latex.pop("Pearson Mean"),latex.pop("Pearson Std"))]
    latex["RMSE $\\downarrow$"]=[f"{m:.3f} $\\pm$ {s:.3f}" for m,s in zip(latex.pop("RMSE Mean"),latex.pop("RMSE Std"))]
    latex["MAE $\\downarrow$"]=[f"{m:.3f} $\\pm$ {s:.3f}" for m,s in zip(latex.pop("MAE Mean"),latex.pop("MAE Std"))]
    (out_dir/"summary.tex").write_text(latex.to_latex(index=False,escape=False),encoding="utf-8")
    save_json({"config":cfg,"models_run":models,"seeds_run":seeds,"device":str(device),"n_subjects":len(y)},out_dir/"run_metadata.json")
    configured_models=set(cfg["models"].keys()); configured_seeds=set(int(v) for v in cfg["seeds"])
    current=df[df.method_id.isin(configured_models)&df.seed.isin(configured_seeds)]
    expected=len(configured_models)*len(configured_seeds)*n_folds
    if len(current)==expected and not current.duplicated(["method_id","seed","fold"]).any():
        complete.write_text("ok\n",encoding="utf-8")
    print(f"Saved {len(df)} evaluations to {out_dir}")

if __name__=="__main__": main()
