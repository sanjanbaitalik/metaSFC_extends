import numpy as np
import torch

from metascfc.benchmark_utils import rowwise_topk_adjacency, binary_sc_adjacency
from metascfc.sota_baselines import MGCNRegressor, IMGGCNRegressor


def symmetric_data(b=4,n=12,seed=1):
    rng=np.random.default_rng(seed)
    fc=rng.normal(size=(b,n,n)).astype(np.float32);fc=(fc+fc.transpose(0,2,1))/2
    sc=rng.random(size=(b,n,n)).astype(np.float32);sc=(sc+sc.transpose(0,2,1))/2;sc[sc<.6]=0
    for x in fc: np.fill_diagonal(x,0)
    for x in sc: np.fill_diagonal(x,0)
    return fc,sc


def test_rowwise_adjacency_is_symmetric_and_finite():
    fc,_=symmetric_data();a=rowwise_topk_adjacency(fc,20,positive_only=True)
    assert np.allclose(a,a.transpose(0,2,1));assert np.isfinite(a).all()


def test_mgcn_forward_backward():
    fc,sc=symmetric_data();m=MGCNRegressor(12,channels=4,graph_dim=16)
    y=m(torch.from_numpy(fc),torch.from_numpy(binary_sc_adjacency(sc)))
    assert y.shape==(4,);y.mean().backward()


def test_img_gcn_forward_backward_smoke_attention():
    fc,sc=symmetric_data();fa=rowwise_topk_adjacency(fc,20);sa=rowwise_topk_adjacency(sc,20)
    m=IMGGCNRegressor(np.repeat(np.arange(3),4),12,graph_hidden=4,attention_mode='smoke',smoke_attention_dim=16)
    y=m(torch.from_numpy(fc),torch.from_numpy(sc),torch.from_numpy(fa),torch.from_numpy(sa))
    assert y.shape==(4,);y.mean().backward()


def test_paired_wilcoxon_vs_reference_uses_seed_pairs_and_metricwise_holm():
    import pandas as pd
    from metascfc.benchmark_utils import paired_wilcoxon_vs_reference

    rows = []
    # B3 is consistently better than WEAK, but nearly identical to PW_TRUE.
    for seed in range(10):
        rows.extend([
            {
                "method_id": "B3", "method_name": "FC+SC Ridge", "seed": seed,
                "pearson": 0.40 + 0.001 * seed,
                "rmse": 4.50 - 0.001 * seed,
                "mae": 3.80 - 0.001 * seed,
            },
            {
                "method_id": "PW_TRUE", "method_name": "Prior-weighted Ridge", "seed": seed,
                "pearson": 0.40 + 0.001 * seed + (0.0002 if seed % 2 else -0.0002),
                "rmse": 4.50 + (0.0002 if seed % 2 else -0.0002),
                "mae": 3.80 + (0.0002 if seed % 2 else -0.0002),
            },
            {
                "method_id": "WEAK", "method_name": "Weak model", "seed": seed,
                "pearson": 0.10 + 0.001 * seed,
                "rmse": 5.20 - 0.001 * seed,
                "mae": 4.40 - 0.001 * seed,
            },
        ])

    result = paired_wilcoxon_vs_reference(pd.DataFrame(rows), "B3")
    assert set(result["metric"]) == {"pearson", "rmse", "mae"}
    assert set(result["method_id"]) == {"PW_TRUE", "WEAK"}
    assert (result["n_pairs"] == 10).all()
    assert result["wilcoxon_p_holm_metric"].between(0, 1).all()
    weak = result[(result.method_id == "WEAK") & (result.metric == "pearson")].iloc[0]
    near = result[(result.method_id == "PW_TRUE") & (result.metric == "pearson")].iloc[0]
    assert weak["reference_advantage_mean"] > 0
    assert bool(weak["significant_holm_005"])
    assert not bool(near["significant_holm_005"])
