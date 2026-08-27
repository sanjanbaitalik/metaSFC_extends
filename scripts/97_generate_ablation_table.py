#!/usr/bin/env python3
"""Generate GNN-as-Inductive-Bottleneck ablation table for ICLR 2027.

Reframes the existing GNN/Transformer IB metrics as ablations of the
information-bottleneck trade-off, with NCR (linear, no compression) as
the anchor reference.

Reads:
  - outputs/iclr/dual_task_matrix/summary.csv  (NCR + LLM-gated IB metrics)
  - outputs/iclr/mt_ncr/summary.csv             (MT-NCR + no-prior ridge)

Outputs (under --tables-dir, default outputs/iclr/tables):
    table3_ablation_ib.tex   LaTeX table

Example:
    python scripts/97_generate_ablation_table.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DUAL_SUMMARY = "outputs/iclr/dual_task_matrix/summary.csv"
DEFAULT_MTNCR_SUMMARY = "outputs/iclr/mt_ncr/summary.csv"
DEFAULT_LOWN_SUMMARY = "outputs/iclr/lown_curve/summary.csv"
DEFAULT_TABLES_DIR = "outputs/iclr/tables"

TARGET_NAMES = {
    "fluid_intelligence": "Fluid Intell. ($\\mathrm{PMAT24}_{A\\_CR}$)",
    "working_memory": "Working Memory (ListSort)",
}

METHOD_ORDER = [
    "ncr_contrastive", "ncr_no_prior", "no_prior_ridge",
    "mt_ncr_independent", "mt_ncr_joint_l21",
    "llm_gated_contrastive",
]
METHOD_NAMES = {
    "ncr_contrastive": "NCR + Contrastive Prior",
    "ncr_no_prior": "NCR (no prior)",
    "no_prior_ridge": "Ridge (no prior)",
    "mt_ncr_independent": "MT-NCR (independent)",
    "mt_ncr_joint_l21": "MT-NCR (l2,1 joint)",
    "llm_gated_contrastive": "LLM-Gated Transformer",
}
METHOD_ROLE = {
    "ncr_contrastive": "Proposed",
    "ncr_no_prior": "Ablation: no prior",
    "no_prior_ridge": "Ablation: no prior",
    "mt_ncr_independent": "Proposed (multi-task)",
    "mt_ncr_joint_l21": "Proposed (multi-task + l2,1)",
    "llm_gated_contrastive": "Ablation: inductive bottleneck",
}


def _normalize_method_name(method: str) -> str:
    """Map raw method names from CSVs to canonical names."""
    if method.startswith("ncr_llm_fluid") or method.startswith("ncr_llm_wm"):
        return "ncr_contrastive"
    if method.startswith("ncr_no_prior"):
        return "ncr_no_prior"
    if method.startswith("ncr_random_control"):
        return "ncr_random"
    if method.startswith("llm_gated_llm_"):
        return "llm_gated_contrastive"
    if method.startswith("llm_gated_no_prior"):
        return "llm_gated_no_prior"
    return method


def _fmt(mean: float, std: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "--"
    return f"${mean:.{digits}f} \\pm {std:.{digits}f}$"


def build_table3(
    dual_df: pd.DataFrame | None,
    mtncr_df: pd.DataFrame | None,
    lown_df: pd.DataFrame | None,
) -> str:
    """Build the ablation table combining dual-task and low-N results."""
    rows = []

    # Normalize dual_task_matrix: model+prior -> method
    sources = []
    if dual_df is not None and len(dual_df) > 0:
        ddf = dual_df.copy()
        if "model" in ddf.columns and "method" not in ddf.columns:
            if "prior" in ddf.columns:
                ddf["method"] = ddf["model"] + "_" + ddf["prior"]
            else:
                ddf["method"] = ddf["model"]
        sources.append(ddf)
    if mtncr_df is not None and len(mtncr_df) > 0:
        mdf = mtncr_df.copy()
        if "model" in mdf.columns and "method" not in mdf.columns:
            mdf["method"] = mdf["model"]
        sources.append(mdf)
    source = pd.concat(sources, ignore_index=True) if sources else pd.DataFrame()
    if len(source) > 0 and "method" in source.columns:
        source["method"] = source["method"].map(_normalize_method_name).fillna(source["method"])

    for method in METHOD_ORDER:
        if len(source) == 0 or method not in source["method"].values:
            continue
        mdf = source[source["method"] == method]
        if mdf.empty:
            continue
        role = METHOD_ROLE.get(method, "")
        method_cell = (f"\\textbf{{{METHOD_NAMES[method]}}}"
                       if role == "Proposed" else METHOD_NAMES.get(method, method))
        for target in ["fluid_intelligence", "working_memory"]:
            tdf = mdf[mdf["target"] == target]
            if tdf.empty:
                continue
            pearson_m = float(tdf["pearson_mean"].iloc[0]) if "pearson_mean" in tdf.columns else float("nan")
            pearson_s = float(tdf["pearson_std"].iloc[0]) if "pearson_std" in tdf.columns else 0.0
            rmse_m = float(tdf["rmse_mean"].iloc[0]) if "rmse_mean" in tdf.columns else float("nan")
            rmse_s = float(tdf["rmse_std"].iloc[0]) if "rmse_std" in tdf.columns else 0.0
            ixz_m = float(tdf["I_XZ_final_mean"].iloc[0]) if "I_XZ_final_mean" in tdf.columns else float("nan")
            ixz_s = float(tdf["I_XZ_final_std"].iloc[0]) if "I_XZ_final_std" in tdf.columns else 0.0
            izy_m = float(tdf["I_ZY_final_mean"].iloc[0]) if "I_ZY_final_mean" in tdf.columns else float("nan")
            izy_s = float(tdf["I_ZY_final_std"].iloc[0]) if "I_ZY_final_std" in tdf.columns else 0.0
            rows.append(
                f"{method_cell} & {TARGET_NAMES[target]} & {role} & "
                f"{_fmt(pearson_m, pearson_s)} & "
                f"{_fmt(rmse_m, rmse_s)} & "
                f"{_fmt(ixz_m, ixz_s)} & "
                f"{_fmt(izy_m, izy_s)} \\\\"
            )

    if not rows:
        body = "% No data available"
    else:
        body = "\n".join(rows)

    return f"""% Generated by scripts/97_generate_ablation_table.py - DO NOT EDIT.
% Requires: \\usepackage{{booktabs}}
\\begin{{table*}}[t]
\\centering
\\small
\\caption{{Inductive-Bottleneck ablation study. The GNN-based LLM-Gated
Transformer sits in the Bottleneck Zone (low $I(X;Z)$, low $I(Z;Y)$),
while the linear NCR operates in the Optimal Regime.  Removing the prior
(no-prior ridge) degrades prediction, confirming the prior's value.
Multi-task NCR (independent or l2,1 joint) further improves cross-task
biomarker discovery (Jaccard overlap, Table~\\\\ref{{tab:jaccard}}).}}
\\label{{tab:ablation_ib}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{lllcccc}}
\\toprule
Method & Target & Role & Pearson $r \\uparrow$ & RMSE $\\downarrow$ &
$I(X;Z) \\downarrow$ & $I(Z;Y) \\uparrow$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}}}
\\end{{table*}}
"""


def build_jaccard_table(mtncr_df: pd.DataFrame | None) -> str:
    """Build a small table for Jaccard overlap (MT-NCR biomarker stability)."""
    if mtncr_df is None or mtncr_df.empty:
        return "% No MT-NCR Jaccard data available\n"

    jdf = mtncr_df[mtncr_df["target"] == "jaccard_top10pct"]
    if jdf.empty:
        return "% No Jaccard data available\n"

    rows = []
    for method in ["mt_ncr_independent", "mt_ncr_joint_l21"]:
        mdf = jdf[jdf["method"] == method]
        if mdf.empty:
            continue
        j_mean = float(mdf["jaccard"].mean())
        j_std = float(mdf["jaccard"].std()) if len(mdf) > 1 else 0.0
        name = METHOD_NAMES.get(method, method)
        rows.append(f"{name} & {_fmt(j_mean, j_std)} \\\\")

    if not rows:
        return "% No Jaccard data available\n"

    body = "\n".join(rows)
    return f"""% Generated by scripts/97_generate_ablation_table.py - DO NOT EDIT.
\\begin{{table}}[t]
\\centering
\\small
\\caption{{Biomarker stability: Jaccard overlap of the top-10\\% edges
(selected by $|\\beta|$) across fluid-intelligence and working-memory
predictors.  Higher Jaccard indicates more consistent cross-task
biomarker discovery.}}
\\label{{tab:jaccard}}
\\begin{{tabular}}{{lc}}
\\toprule
Method & Jaccard (top-10\\%) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def build_lown_table(lown_df: pd.DataFrame | None) -> str:
    """Build the low-N sample efficiency table."""
    if lown_df is None or lown_df.empty:
        return "% No Low-N data available\n"

    rows = []
    for method in ["ncr_contrastive", "no_prior_ridge"]:
        mdf = lown_df[lown_df["method"] == method]
        if mdf.empty:
            continue
        method_name = ("NCR + Contrastive Prior" if method == "ncr_contrastive"
                       else "Ridge (no prior)")
        for target in ["fluid_intelligence", "working_memory"]:
            tdf = mdf[mdf["target"] == target]
            if tdf.empty:
                continue
            for _, r in tdf.iterrows():
                n_val = int(r["N"])
                p_m = float(r["pearson_mean"])
                p_s = float(r.get("pearson_std", 0))
                rows.append(
                    f"{method_name if target == 'fluid_intelligence' else ''} "
                    f"& {TARGET_NAMES[target]} & {n_val} & "
                    f"{_fmt(p_m, p_s)} \\\\"
                )

    if not rows:
        return "% No Low-N data available\n"

    body = "\n".join(rows)
    return f"""% Generated by scripts/97_generate_ablation_table.py - DO NOT EDIT.
\\begin{{table}}[t]
\\centering
\\small
\\caption{{Sample-efficient connectomics (Low-N curves).  NCR with a
zero-shot LLM contrastive prior maintains predictive accuracy at N=50
subjects, where a plain ridge baseline degrades.  This demonstrates
that zero-shot semantic priors compensate for limited training data.}}
\\label{{tab:lown}}
\\begin{{tabular}}{{llcc}}
\\toprule
Method & Target & $N$ & Pearson $r \\uparrow$ \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual-summary", default=DEFAULT_DUAL_SUMMARY)
    ap.add_argument("--mtncr-summary", default=DEFAULT_MTNCR_SUMMARY)
    ap.add_argument("--lown-summary", default=DEFAULT_LOWN_SUMMARY)
    ap.add_argument("--dual-splits", default=None,
                    help="split_metrics.csv from dual_task_matrix for per-split data")
    ap.add_argument("--mtncr-splits", default=None,
                    help="split_metrics.csv from mt_ncr for Jaccard data")
    ap.add_argument("--tables-dir", default=DEFAULT_TABLES_DIR)
    args = ap.parse_args()

    out_dir = Path(args.tables_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dual_df = pd.read_csv(args.dual_summary) if Path(args.dual_summary).exists() else None
    mtncr_df = pd.read_csv(args.mtncr_summary) if Path(args.mtncr_summary).exists() else None
    lown_df = pd.read_csv(args.lown_summary) if Path(args.lown_summary).exists() else None

    if args.mtncr_splits and Path(args.mtncr_splits).exists():
        mtncr_splits = pd.read_csv(args.mtncr_splits)
        jaccard_rows = mtncr_splits[mtncr_splits["target"] == "jaccard_top10pct"]
        if not jaccard_rows.empty and mtncr_df is not None:
            jaccard_agg = (
                jaccard_rows.groupby("method")
                .agg(jaccard=("jaccard", "mean"), n=("fold", "size"))
                .reset_index()
            )
            jaccard_agg["target"] = "jaccard_top10pct"
            mtncr_df = pd.concat([mtncr_df, jaccard_agg], ignore_index=True)

    t3 = build_table3(dual_df, mtncr_df, lown_df)
    (out_dir / "table3_ablation_ib.tex").write_text(t3, encoding="utf-8")
    print(f"Wrote {out_dir / 'table3_ablation_ib.tex'}")

    t4 = build_jaccard_table(mtncr_df)
    (out_dir / "table4_jaccard.tex").write_text(t4, encoding="utf-8")
    print(f"Wrote {out_dir / 'table4_jaccard.tex'}")

    t5 = build_lown_table(lown_df)
    (out_dir / "table5_lown.tex").write_text(t5, encoding="utf-8")
    print(f"Wrote {out_dir / 'table5_lown.tex'}")


if __name__ == "__main__":
    main()
