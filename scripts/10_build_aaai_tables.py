#!/usr/bin/env python3
"""Build AAAI-ready performance, alignment, stability and LaTeX tables."""
import argparse, json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

METHOD_ORDER = [f"E{i}" for i in range(10)]

def pairwise_stability(arr, topk=10):
    if arr is None or len(arr) < 2:
        return {"saliency_spearman_mean": np.nan, "saliency_jaccard_mean": np.nan}
    rs, js = [], []
    for i,j in combinations(range(len(arr)),2):
        r = spearmanr(arr[i], arr[j]).statistic
        rs.append(0.0 if np.isnan(r) else r)
        a=set(np.argsort(arr[i])[-topk:]); b=set(np.argsort(arr[j])[-topk:])
        js.append(len(a&b)/len(a|b))
    return {"saliency_spearman_mean": float(np.mean(rs)), "saliency_spearman_std": float(np.std(rs, ddof=1)),
            "saliency_jaccard_mean": float(np.mean(js)), "saliency_jaccard_std": float(np.std(js, ddof=1))}

def fmt(mean, std, digits=3):
    return "--" if pd.isna(mean) else f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--results", default="outputs/aaai/final"); ap.add_argument("--out", default="outputs/aaai/tables"); ap.add_argument("--topk", type=int, default=10); args=ap.parse_args()
    root=Path(args.results); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    perf=[]; align=[]; stability=[]; split_all=[]
    for d in root.iterdir() if root.exists() else []:
        p=d/"split_metrics.csv"
        if not p.exists(): continue
        df=pd.read_csv(p); split_all.append(df)
        first=df.iloc[0]; eid=str(first["experiment_id"])
        row={"ID":eid,"Method":first["experiment_name"],"Prior Type":first["prior_type"],"Prior Source":first["prior_source"],"N":int(df.n_test.sum()/df.seed.nunique()),"Seeds":df.seed.nunique(),"Folds":df.fold.nunique()}
        for m in ["pearson","rmse","mae"]:
            row[f"{m.capitalize()} Mean"]=df[m].mean(); row[f"{m.capitalize()} Std"]=df[m].std(ddof=1)
        perf.append(row)
        a={"ID":eid,"Method":first["experiment_name"]}
        for c in [x for x in df.columns if x.startswith("alignment_") or x.startswith("reference_alignment_")]:
            a[f"{c} Mean"]=df[c].mean(); a[f"{c} Std"]=df[c].std(ddof=1)
        align.append(a)
        sal_path=d/"all_node_saliency.npy"; arr=np.load(sal_path) if sal_path.exists() else None
        stability.append({"ID":eid,"Method":first["experiment_name"],**pairwise_stability(arr,args.topk)})
    if not perf: raise SystemExit(f"No results under {root}")
    key=lambda x: METHOD_ORDER.index(x) if x in METHOD_ORDER else 99
    perf_df=pd.DataFrame(perf).sort_values("ID",key=lambda s:s.map(key)); align_df=pd.DataFrame(align).sort_values("ID",key=lambda s:s.map(key)); stab_df=pd.DataFrame(stability).sort_values("ID",key=lambda s:s.map(key))
    perf_df.to_csv(out/"table1_prediction_performance.csv",index=False); align_df.to_csv(out/"table2_prior_alignment.csv",index=False); stab_df.to_csv(out/"table3_saliency_stability.csv",index=False)
    pd.concat(split_all,ignore_index=True).to_csv(out/"all_split_metrics.csv",index=False)

    latex=perf_df[["ID","Method","Pearson Mean","Pearson Std","Rmse Mean","Rmse Std","Mae Mean","Mae Std"]].copy()
    latex["Pearson $\\uparrow$"]=[fmt(a,b) for a,b in zip(latex.pop("Pearson Mean"),latex.pop("Pearson Std"))]
    latex["RMSE $\\downarrow$"]=[fmt(a,b) for a,b in zip(latex.pop("Rmse Mean"),latex.pop("Rmse Std"))]
    latex["MAE $\\downarrow$"]=[fmt(a,b) for a,b in zip(latex.pop("Mae Mean"),latex.pop("Mae Std"))]
    (out/"table1_prediction_performance.tex").write_text(latex.to_latex(index=False,escape=False,float_format="%.3f"))
    (out/"table2_prior_alignment.tex").write_text(align_df.to_latex(index=False,escape=False,float_format="%.3f"))
    (out/"table3_saliency_stability.tex").write_text(stab_df.to_latex(index=False,escape=False,float_format="%.3f"))
    print("Saved AAAI tables to",out)

if __name__=="__main__": main()
