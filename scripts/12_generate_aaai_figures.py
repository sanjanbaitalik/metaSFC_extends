#!/usr/bin/env python3
"""Generate publication-ready experiment figures from E0-E9 outputs."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER=[f"E{i}" for i in range(10)]
LABELS={"E0":"Baseline","E1":"Node-True","E2":"Node-Shuffled","E3":"Node-Random","E4":"Module-True","E5":"Module-Shuffled","E6":"Module-Random","E7":"Edge-True","E8":"Edge-Shuffled","E9":"Edge-Random"}

def save_bar(df,mean_col,std_col,ylabel,path,lower=False):
    df=df.copy(); df["ord"]=df.ID.map(lambda x:ORDER.index(x) if x in ORDER else 99); df=df.sort_values("ord")
    x=np.arange(len(df)); fig,ax=plt.subplots(figsize=(11,4.5)); ax.bar(x,df[mean_col],yerr=df[std_col],capsize=3)
    ax.set_xticks(x, [LABELS.get(v,v) for v in df.ID],rotation=35,ha="right"); ax.set_ylabel(ylabel); ax.axhline(0,linewidth=.8); fig.tight_layout(); fig.savefig(path,dpi=300); plt.close(fig)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--table_dir",default="outputs/aaai/tables"); ap.add_argument("--results",default="outputs/aaai/final"); ap.add_argument("--out",default="outputs/aaai/figures"); ap.add_argument("--labels",default="inputs/atlases/AAL116_labels.csv"); ap.add_argument("--topk",type=int,default=15); args=ap.parse_args()
    td=Path(args.table_dir); root=Path(args.results); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    perf=pd.read_csv(td/"table1_prediction_performance.csv")
    save_bar(perf,"Pearson Mean","Pearson Std","Pearson correlation",out/"fig2a_prediction_pearson.png")
    save_bar(perf,"Rmse Mean","Rmse Std","RMSE",out/"fig2b_prediction_rmse.png",True)
    save_bar(perf,"Mae Mean","Mae Std","MAE",out/"fig2c_prediction_mae.png",True)
    stab=pd.read_csv(td/"table3_saliency_stability.csv")
    save_bar(stab,"saliency_spearman_mean","saliency_spearman_std","Cross-run saliency Spearman",out/"fig3a_saliency_rank_stability.png")
    save_bar(stab,"saliency_jaccard_mean","saliency_jaccard_std","Top-k saliency Jaccard",out/"fig3b_saliency_topk_stability.png")

    # Alignment comparison, choosing the scope available for each experiment.
    split=pd.read_csv(td/"all_split_metrics.csv"); rows=[]
    for eid,g in split.groupby("experiment_id"):
        cols=[c for c in g if c.startswith("reference_alignment_") and c.endswith("_pearson")]
        if not cols:
            cols=[c for c in g if c.startswith("alignment_") and c.endswith("_pearson")]
        vals=pd.concat([g[c] for c in cols],ignore_index=True).dropna() if cols else pd.Series(dtype=float)
        if len(vals): rows.append({"ID":eid,"mean":vals.mean(),"std":vals.std(ddof=1)})
    if rows:
        adf=pd.DataFrame(rows); save_bar(adf,"mean","std","Learned-prior Pearson alignment",out/"fig4_control_prior_alignment.png")

    # Top ROI profile for E1 against E2/E3 when available.
    labels=pd.read_csv(args.labels) if Path(args.labels).exists() else pd.DataFrame({"roi_label":[str(i+1) for i in range(116)]})
    sal={}
    for eid in ["E1","E2","E3"]:
        matches=list(root.glob(f"{eid}_*/all_node_saliency.npy"))
        if matches: sal[eid]=np.load(matches[0]).mean(axis=0)
    if "E1" in sal:
        idx=np.argsort(sal["E1"])[-args.topk:][::-1]; y=np.arange(len(idx)); fig,ax=plt.subplots(figsize=(8,6))
        for eid,offset in [("E1",-.25),("E2",0),("E3",.25)]:
            if eid in sal: ax.barh(y+offset,sal[eid][idx],height=.23,label=LABELS[eid])
        ax.set_yticks(y,labels.iloc[idx].roi_label); ax.invert_yaxis(); ax.set_xlabel("Mean learned FC-SC saliency"); ax.legend(); fig.tight_layout(); fig.savefig(out/"fig5_top_roi_saliency_controls.png",dpi=300); plt.close(fig)
    print("Saved figures to",out)

if __name__=="__main__": main()
