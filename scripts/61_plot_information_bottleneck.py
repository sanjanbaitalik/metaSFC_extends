#!/usr/bin/env python3
"""Information Bottleneck trade-off figure for the dual-task matrix.

Reads ``outputs/iclr/dual_task_matrix/summary.csv`` and plots the Inductive
Bottleneck plane:

    X = I(X; Z)  (compression / aggressive filtering)
    Y = I(Z; Y)  (predictive capacity)

One point per (model, prior, target) cell.  Color encodes the PRIOR type
(LLM-WM, LLM-Fluid, Random, No-Prior), marker shape encodes the TARGET task
(circle = Working Memory, square = Fluid Intelligence).  The mismatched
TRUE-prior cells (LLM-WM prior on Fluid, LLM-Fluid prior on WM) are expected
in the high-compression / low-prediction region - the shaded "Inductive
Bottleneck" zone - while matched cells and controls sit closer to the
high-prediction frontier.

Outputs
-------
    outputs/iclr/figures/ib_tradeoff.png   (300 dpi)
    outputs/iclr/figures/ib_tradeoff.pdf
    outputs/iclr/figures/ib_tradeoff.tex   (PGFPlots/TikZ snippet)
    prints a per-cell table to stdout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DEFAULT_SUMMARY = "outputs/iclr/dual_task_matrix/summary.csv"
DEFAULT_OUT_DIR = "outputs/iclr/figures"

PRIOR_STYLE = {
    "llm_wm": {"label": "True LLM prior (WM)", "color": "#1f77b4"},
    "llm_fluid": {"label": "True LLM prior (Fluid)", "color": "#d62728"},
    "random_control": {"label": "Random control", "color": "#7f7f7f"},
    "no_prior": {"label": "No prior", "color": "#ff7f0e"},
}
TARGET_STYLE = {
    "working_memory": {"label": "Working Memory", "marker": "o"},
    "fluid_intelligence": {"label": "Fluid Intelligence", "marker": "s"},
}
# True-prior cells whose domain does NOT match the target: the Inductive
# Bottleneck regime predicted by our hypothesis.
MISMATCHED_TRUE_CELLS = {
    ("llm_wm", "fluid_intelligence"),
    ("llm_fluid", "working_memory"),
}


def load_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"model", "prior", "target", "I_XZ_final_mean", "I_ZY_final_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing {sorted(missing)} - run "
            "scripts/50_run_dual_task_matrix.py first."
        )
    df = df[np.isfinite(df.I_XZ_final_mean) & np.isfinite(df.I_ZY_final_mean)]
    if df.empty:
        raise ValueError(f"{path} has no finite IB values to plot.")
    return df


def plot_ib_tradeoff(df: pd.DataFrame, out_dir: Path) -> None:
    x_med, y_med = df.I_XZ_final_mean.median(), df.I_ZY_final_mean.median()

    # Data-derived limits (computed BEFORE any artist is added, so the two
    # model families - e.g. linear ridge vs transformer - are never clipped).
    pad_x = 0.08 * (df.I_XZ_final_mean.max() - df.I_XZ_final_mean.min() + 1e-6)
    x_lo = max(0.0, df.I_XZ_final_mean.min() - pad_x)
    x_hi = df.I_XZ_final_mean.max() + pad_x
    y_lo = min(0.0, df.I_ZY_final_mean.min() - 0.10 * (df.I_ZY_final_mean.max() - df.I_ZY_final_mean.min()))
    y_hi = df.I_ZY_final_mean.max() + 0.22 * (df.I_ZY_final_mean.max() - df.I_ZY_final_mean.min())

    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Inductive Bottleneck zone: above-median compression AND below-median
    # predictive capacity (shaded reference region, not a hard threshold).
    ax.axvspan(x_med, x_hi, ymin=0.0,
               ymax=(y_med - y_lo) / (y_hi - y_lo),
               color="#d9534f", alpha=0.08, zorder=0)
    ax.text(x_med + 0.03 * (x_hi - x_med), y_lo + 0.05 * (y_hi - y_lo),
            "Inductive Bottleneck\n(high $I(X;Z)$, low $I(Z;Y)$)",
            fontsize=8.5, color="#a94442", style="italic", va="bottom")

    seen = set()
    for i, (_, row) in enumerate(df.iterrows()):
        prior, target = row.prior, row.target
        style = PRIOR_STYLE.get(prior)
        tstyle = TARGET_STYLE.get(target)
        if style is None or tstyle is None:
            continue
        label = style["label"] if prior not in seen else None
        seen.add(prior)
        is_mismatch = (prior, target) in MISMATCHED_TRUE_CELLS
        ax.scatter(
            row.I_XZ_final_mean, row.I_ZY_final_mean,
            s=150 if prior.startswith("llm_") else 90,
            marker=tstyle["marker"], color=style["color"],
            edgecolors="black" if is_mismatch else "none",
            linewidths=1.4, alpha=0.95, label=label, zorder=3,
        )

    # Cluster annotations (per-point micro-labels collide; the legend
    # already encodes color = prior, marker = target).
    ncr = df[df.model == "ncr"]
    llm = df[df.model == "llm_gated"]
    for sub, name, dy in ((ncr, "Network-constrained ridge (linear)", 1),
                          (llm, "LLM-gated transformer", -1)):
        if sub.empty:
            continue
        ax.annotate(name,
                    (sub.I_XZ_final_mean.mean(), sub.I_ZY_final_mean.mean()),
                    textcoords="offset points",
                    xytext=(0, 26 * dy), fontsize=8.5, color="dimgray",
                    ha="center", va="bottom" if dy > 0 else "top")

    # One clean callout per mismatched true-prior cell, on opposite sides.
    callout_offsets = {"llm_wm": (26, -46), "llm_fluid": (26, 30)}
    for prior, target in MISMATCHED_TRUE_CELLS:
        sel = df[(df.prior == prior) & (df.target == target)]
        if sel.empty:
            continue
        r = sel.iloc[0]
        ax.annotate(
            "Inductive Bottleneck\n(mismatched true prior)",
            (r.I_XZ_final_mean, r.I_ZY_final_mean),
            textcoords="offset points",
            xytext=callout_offsets.get(prior, (24, -40)),
            fontsize=8, color="#a94442", ha="left",
            arrowprops=dict(arrowstyle="->", color="#a94442", lw=1.2),
        )

    # Marker-shape legend for targets.
    for target, tstyle in TARGET_STYLE.items():
        ax.scatter([], [], marker=tstyle["marker"], color="black",
                   s=60, label=f"Target: {tstyle['label']}")

    ax.set_xlabel("$I(X; Z)$  — compression [nats]")
    ax.set_ylabel("$I(Z; Y)$  — predictive capacity [nats]")
    ax.set_title("Inductive Bottleneck: LLM semantic priors in the IB plane")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95, ncol=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "ib_tradeoff.png", dpi=300)
    fig.savefig(out_dir / "ib_tradeoff.pdf")
    plt.close(fig)


def write_tikz(df: pd.DataFrame, path: Path) -> None:
    """PGFPlots/TikZ snippet mirroring the matplotlib figure."""
    lines = [
        "% Generated by scripts/61_plot_information_bottleneck.py",
        "% Requires: \\usepackage{pgfplots} \\pgfplotsset{compat=1.17}",
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        "    width=0.85\\linewidth, height=0.65\\linewidth,",
        "    xlabel={$I(X;Z)$ --- compression [nats]},",
        "    ylabel={$I(Z;Y)$ --- predictive capacity [nats]},",
        "    title={Inductive Bottleneck: LLM semantic priors},",
        "    grid=major, legend pos=north east, legend style={font=\\scriptsize},",
        "    tick label style={font=\\scriptsize}, label style={font=\\small},",
        "]",
    ]
    for prior, pstyle in PRIOR_STYLE.items():
        sub = df[df.prior == prior]
        if sub.empty:
            continue
        hexc = pstyle["color"].lstrip("#")
        lines.append(f"\\definecolor{{{prior}}}{{HTML}}{{{hexc.upper()}}}")
        for target, tstyle in TARGET_STYLE.items():
            cell = sub[sub.target == target]
            if cell.empty:
                continue
            r = cell.iloc[0]
            mark = "*" if tstyle["marker"] == "o" else "square*"
            size = "4pt" if prior.startswith("llm_") else "3pt"
            lines.append(
                f"\\addplot+[only marks, mark={mark}, mark size={size}, "
                f"color={prior}, draw=black] coordinates {{"
                f"({r.I_XZ_final_mean:.4f},{r.I_ZY_final_mean:.4f})}};"
            )
            lines.append(
                f"\\addlegendentryexpanded{{{pstyle['label']}"
                f" \\,({tstyle['label'].split()[0]})}}"
            )
    lines += ["\\end{axis}", "\\end{tikzpicture}", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=DEFAULT_SUMMARY)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    df = load_summary(args.summary)
    cols = ["model", "prior", "target", "pearson_mean", "rmse_mean",
            "I_XZ_final_mean", "I_ZY_final_mean"]
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

    out_dir = Path(args.out_dir)
    plot_ib_tradeoff(df, out_dir)
    write_tikz(df, out_dir / "ib_tradeoff.tex")
    print(f"\nSaved IB trade-off figure to {out_dir}/ib_tradeoff.png "
          f"(+ .pdf, .tex)")


if __name__ == "__main__":
    main()
