#!/usr/bin/env python3
from pathlib import Path
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",default="outputs/aaai/figures/fig1_method_overview.png"); args=ap.parse_args(); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(13,4)); ax.set_xlim(0,13); ax.set_ylim(0,4); ax.axis("off")
    boxes=[(0.3,1.2,2.0,1.3,"Meta-analysis map\n(working memory)"),(2.8,1.2,2.0,1.3,"AAL116 projection\nROI / module / edge priors"),(5.3,2.1,2.0,1.1,"Functional connectome\nFC graph"),(5.3,.5,2.0,1.1,"Structural connectome\nSC graph"),(7.9,1.2,2.2,1.3,"Multimodal FC-SC\ncoupling network"),(10.7,1.2,2.0,1.3,"Prediction +\nprior-guided saliency")]
    for x,y,w,h,t in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.04",linewidth=1.5,facecolor="white")); ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=10)
    arrows=[((2.3,1.85),(2.8,1.85)),((4.8,1.85),(7.9,1.85)),((7.3,2.65),(7.9,2.15)),((7.3,1.05),(7.9,1.55)),((10.1,1.85),(10.7,1.85))]
    for a,b in arrows: ax.add_patch(FancyArrowPatch(a,b,arrowstyle="->",mutation_scale=14,linewidth=1.5))
    ax.text(6.4,3.55,"External neurobiological knowledge guides multimodal graph learning through soft regularization",ha="center",fontsize=11,fontweight="bold")
    fig.tight_layout(); fig.savefig(out,dpi=300,bbox_inches="tight"); plt.close(fig); print(out)
if __name__=="__main__": main()
