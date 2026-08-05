#!/usr/bin/env python3
"""Fast synthetic smoke test for new benchmark code; no PyG required."""
from __future__ import annotations
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from metascfc.benchmark_utils import binary_sc_adjacency, rowwise_topk_adjacency
from metascfc.sota_baselines import MGCNRegressor, IMGGCNRegressor


def main():
    rng=np.random.default_rng(7); b,n=6,12
    fc=rng.normal(size=(b,n,n)).astype(np.float32); fc=(fc+fc.transpose(0,2,1))/2
    sc=rng.random(size=(b,n,n)).astype(np.float32);sc=(sc+sc.transpose(0,2,1))/2;sc[sc<0.65]=0
    for m in list(fc)+list(sc): np.fill_diagonal(m,0)
    fc_adj=rowwise_topk_adjacency(fc,20,positive_only=True);sc_adj=rowwise_topk_adjacency(sc,20,positive_only=True)
    mgcn=MGCNRegressor(n_rois=n,channels=4,graph_dim=16)
    p=mgcn(torch.from_numpy(fc),torch.from_numpy(binary_sc_adjacency(sc)))
    assert p.shape==(b,) and torch.isfinite(p).all()
    modules=np.repeat(np.arange(3),4)
    img=IMGGCNRegressor(modules,n,graph_hidden=4,bottleneck_ratio=2,dropout=0.1,attention_mode="smoke",smoke_attention_dim=16)
    p2=img(torch.from_numpy(fc),torch.from_numpy(sc),torch.from_numpy(fc_adj),torch.from_numpy(sc_adj))
    assert p2.shape==(b,) and torch.isfinite(p2).all()
    loss=(p.mean()+p2.mean());loss.backward()
    print("PASS: M-GCN and IMG-GCN forward/backward smoke test")
    print("M-GCN params:",sum(x.numel() for x in mgcn.parameters()))
    print("IMG-GCN smoke params:",sum(x.numel() for x in img.parameters()))

if __name__=="__main__":main()
