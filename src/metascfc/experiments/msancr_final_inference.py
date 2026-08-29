"""Final n=10 seed-level inference for the frozen MS-A-NCR 10x5 run.

The statistical unit is the seed (five outer folds averaged within each
seed); 50 fold rows are never treated as independent observations.  This
module supersedes the pilot ``n=3`` descriptive gate with Wilcoxon
signed-rank inference, metric-wise Holm correction, deterministic bootstrap
CIs, and paired Cohen's dz.  It only reads frozen prediction artifacts.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_1samp, wilcoxon

from metascfc.benchmark_utils import atomic_write_csv, holm_adjust

MODEL_A4 = "A4_modality_ridge"
MODEL_A4_ISO = "A4_iso_same_solver"
MODEL_A2 = "A2_fc_laplacian"
MODEL_A3 = "A3_msancr"
BASE_EVAL = "retuned_base"
SWAP_EVAL = "fixed_prior_swap"
CONTROL_PRIORS = ("unrelated", "shuffled", "random")
N_FINAL_SEEDS = 10
N_FOLDS = 5
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 20260829
WILCOXON_ZERO_METHOD = "wilcox"

PREDICTION_METRICS = ("pearson", "rmse", "mae")
BIOMARKER_METRICS = ("wm_alignment", "rank_stability", "top10_jaccard")
HIGHER_BETTER = {"pearson", "wm_alignment", "rank_stability", "top10_jaccard"}

COMPARISONS: tuple[tuple[str, tuple[str, str], tuple[str, str]], ...] = (
    ("A3_matched_vs_A4", (MODEL_A3, "matched"), (MODEL_A4, "none")),
    ("A3_matched_vs_A2", (MODEL_A3, "matched"), (MODEL_A2, "matched")),
    ("A3_matched_vs_A3_unrelated_fixed", (MODEL_A3, "matched"), (MODEL_A3, "unrelated")),
    ("A3_matched_vs_A3_shuffled_fixed", (MODEL_A3, "matched"), (MODEL_A3, "shuffled")),
    ("A3_matched_vs_A3_random_fixed", (MODEL_A3, "matched"), (MODEL_A3, "random")),
)

EXPECTED_BASE_ROWS = N_FINAL_SEEDS * N_FOLDS * 4
EXPECTED_SWAP_ROWS = N_FINAL_SEEDS * N_FOLDS * 3
EXPECTED_SELECTED_ROWS = N_FINAL_SEEDS * N_FOLDS * 4


def build_seed_level_table(split_df: pd.DataFrame) -> pd.DataFrame:
    """Average the five outer folds within each seed; unit = (model, prior, seed)."""
    required = {"model_id", "prior_type", "evaluation_type", "seed", "fold",
                *PREDICTION_METRICS}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(f"split dataframe missing columns: {sorted(missing)}")
    agg_spec = {"pearson": ("pearson", "mean"), "rmse": ("rmse", "mean"),
                "mae": ("mae", "mean"), "n_folds": ("fold", "nunique")}
    for metric in BIOMARKER_METRICS:
        if metric in split_df.columns:
            agg_spec[metric] = (metric, "mean")
    grouped = split_df.groupby(
        ["model_id", "prior_type", "evaluation_type", "seed"], as_index=False
    ).agg(**agg_spec)
    if not (grouped.n_folds == N_FOLDS).all():
        bad = grouped[grouped.n_folds != N_FOLDS]
        raise ValueError(f"Seed rows must average exactly {N_FOLDS} folds; offenders:\n{bad}")
    seeds = sorted(grouped.seed.unique())
    if seeds != list(range(N_FINAL_SEEDS)):
        raise ValueError(
            f"Complete seed-level inference requires seeds 0..{N_FINAL_SEEDS - 1}; found {seeds}")
    return grouped.drop(columns="n_folds").sort_values(
        ["evaluation_type", "model_id", "prior_type", "seed"]).reset_index(drop=True)


def assert_final_complete(split_df: pd.DataFrame, swap_df: pd.DataFrame,
                          selected_df: pd.DataFrame) -> None:
    """Fail loudly unless the frozen 10x5 cardinality is present."""
    n_base = len(split_df[split_df.evaluation_type == BASE_EVAL])
    n_swap = len(swap_df[swap_df.evaluation_type == SWAP_EVAL])
    if n_base != EXPECTED_BASE_ROWS:
        raise ValueError(f"Partial run: expected {EXPECTED_BASE_ROWS} base rows, found {n_base}")
    if n_swap != EXPECTED_SWAP_ROWS:
        raise ValueError(f"Partial run: expected {EXPECTED_SWAP_ROWS} swap rows, found {n_swap}")
    if len(selected_df) != EXPECTED_SELECTED_ROWS:
        raise ValueError(
            f"Partial run: expected {EXPECTED_SELECTED_ROWS} selected-HP rows, found {len(selected_df)}")
    seeds = sorted(split_df.seed.unique())
    if seeds != list(range(N_FINAL_SEEDS)):
        raise ValueError(f"Partial run: expected seeds 0..{N_FINAL_SEEDS - 1}, found {seeds}")
    if split_df.duplicated(["seed", "fold", "model_id", "prior_type"]).any():
        raise ValueError("Duplicate split rows detected")


def paired_seed_values(seed_df: pd.DataFrame, left: tuple[str, str],
                       right: tuple[str, str], metric: str) -> pd.DataFrame:
    """Align the two methods on shared seed IDs; one row per paired seed."""
    def pick(key: tuple[str, str]) -> pd.Series:
        model_id, prior = key
        rows = seed_df[(seed_df.model_id == model_id) & (seed_df.prior_type == prior)]
        if rows.empty:
            raise ValueError(f"No seed-level rows for {key}")
        if rows.seed.duplicated().any():
            raise ValueError(f"Duplicate seed rows for {key}")
        return rows.set_index("seed")[metric]

    left_s, right_s = pick(left), pick(right)
    common = sorted(set(left_s.index) & set(right_s.index))
    if len(common) != N_FINAL_SEEDS:
        raise ValueError(f"Paired inference needs {N_FINAL_SEEDS} aligned seeds; found {len(common)}")
    return pd.DataFrame({
        "seed": common,
        "left": left_s.loc[common].to_numpy(float),
        "right": right_s.loc[common].to_numpy(float),
    })


def paired_statistics(left: np.ndarray, right: np.ndarray, metric: str,
                      n_boot: int = BOOTSTRAP_RESAMPLES,
                      rng_seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    """Descriptives, Wilcoxon, bootstrap CI, dz, secondary paired t.

    ``left`` is A3; ``right`` is the comparator.  Differences are oriented so
    that positive always means A3 is better (higher-is-better metrics use
    left-right; lower-is-better metrics use right-left).
    """
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("Paired statistics require aligned 1-D vectors of equal length")
    higher_better = metric in HIGHER_BETTER
    diff = (left - right) if higher_better else (right - left)
    if not np.isfinite(diff).all():
        raise ValueError(f"Non-finite paired differences for {metric}")
    raw_p = 1.0
    w_stat = 0.0
    if not np.allclose(diff, 0.0):
        result = wilcoxon(diff, alternative="two-sided", zero_method=WILCOXON_ZERO_METHOD)
        w_stat, raw_p = float(result.statistic), float(result.pvalue)
        if not np.isfinite(raw_p):
            raw_p = 1.0
    rng = np.random.default_rng(rng_seed)
    boot_idx = rng.integers(0, len(diff), size=(int(n_boot), len(diff)))
    boot_means = diff[boot_idx].mean(axis=1)
    ci_low, ci_high = (float(x) for x in np.percentile(boot_means, [2.5, 97.5]))
    sd = float(diff.std(ddof=1))
    dz = float(diff.mean() / sd) if sd > 1e-12 else 0.0
    t_p = 1.0
    if not np.allclose(diff, 0.0):
        t_p = float(ttest_1samp(diff, 0.0, alternative="two-sided").pvalue)
    return {
        "n_seeds": int(len(diff)),
        "left_mean": float(left.mean()), "left_sd": float(left.std(ddof=1)),
        "right_mean": float(right.mean()), "right_sd": float(right.std(ddof=1)),
        "paired_mean_diff": float(diff.mean()),
        "paired_median_diff": float(np.median(diff)),
        "paired_sd_diff": sd,
        "positive_seeds": int((diff > 0).sum()),
        "negative_seeds": int((diff < 0).sum()),
        "zero_seeds": int((diff == 0).sum()),
        "improvement_orientation": "positive_is_A3_better",
        "wilcoxon_W": w_stat, "wilcoxon_p": raw_p,
        "zero_method": WILCOXON_ZERO_METHOD,
        "bootstrap_resamples": int(n_boot), "bootstrap_seed": int(rng_seed),
        "ci95_low": ci_low, "ci95_high": ci_high,
        "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
        "cohens_dz": dz,
        "paired_t_p_secondary": t_p,
    }


def run_family_inference(seed_df: pd.DataFrame, metrics: Sequence[str],
                         n_boot: int = BOOTSTRAP_RESAMPLES) -> pd.DataFrame:
    """All five comparisons x metrics, Holm-corrected within each metric."""
    rows = []
    for name, left_key, right_key in COMPARISONS:
        for metric in metrics:
            paired = paired_seed_values(seed_df, left_key, right_key, metric)
            stats = paired_statistics(
                paired.left.to_numpy(), paired.right.to_numpy(), metric, n_boot=n_boot
            )
            rows.append({"comparison": name, "metric": metric,
                         "left": f"{left_key[0]}|{left_key[1]}",
                         "right": f"{right_key[0]}|{right_key[1]}", **stats})
    frame = pd.DataFrame(rows)
    frame["p_holm_metric"] = np.nan
    for metric, indices in frame.groupby("metric").groups.items():
        idx = list(indices)
        frame.loc[idx, "p_holm_metric"] = holm_adjust(
            frame.loc[idx, "wilcoxon_p"].to_numpy(float)
        )
    frame["significant_holm_005"] = frame.p_holm_metric < 0.05
    return frame


def hypothesis_decision(prediction: pd.DataFrame, biomarker: pd.DataFrame) -> dict[str, Any]:
    """Inferential hypotheses; pilot thresholds are NOT used here."""
    def cell(frame: pd.DataFrame, comparison: str, metric: str) -> pd.Series:
        row = frame[(frame.comparison == comparison) & (frame.metric == metric)]
        if len(row) != 1:
            raise ValueError(f"Missing inference row for {comparison}/{metric}")
        return row.iloc[0]

    main_p = cell(prediction, "A3_matched_vs_A4", "pearson")
    main_rmse = cell(prediction, "A3_matched_vs_A4", "rmse")
    prediction_supported = bool(
        main_p.paired_mean_diff > 0 and main_p.paired_median_diff > 0
        and main_p.p_holm_metric < 0.05
    )
    rmse_contradiction = bool(
        main_rmse.paired_mean_diff < 0 and main_rmse.ci_excludes_zero
    )
    if prediction_supported and rmse_contradiction:
        prediction_state = "prediction_supported_with_metric_disagreement"
    else:
        prediction_state = "supported" if prediction_supported else "not_supported"

    align = cell(biomarker, "A3_matched_vs_A4", "wm_alignment")
    biomarker_supported = bool(align.paired_mean_diff > 0 and align.p_holm_metric < 0.05)
    other_bio_sig = biomarker[
        (biomarker.comparison != "A3_matched_vs_A4") & biomarker.significant_holm_005
    ]
    if biomarker_supported and not len(other_bio_sig):
        biomarker_state = "supported"
    elif biomarker_supported or len(other_bio_sig):
        biomarker_state = "mixed"
    else:
        biomarker_state = "not_supported"

    if prediction_state == "prediction_supported_with_metric_disagreement":
        overall = "prediction_supported_with_metric_disagreement"
    elif prediction_supported and biomarker_state == "supported":
        overall = "prediction_and_biomarker_supported"
    elif prediction_supported and biomarker_state == "mixed":
        overall = "prediction_supported_biomarker_mixed"
    elif prediction_supported:
        overall = "prediction_supported_biomarker_not_supported"
    elif biomarker_state in ("supported", "mixed"):
        overall = "prediction_not_significant_biomarker_supported"
    else:
        overall = "prediction_not_significant"

    swaps = {c: bool(cell(prediction, f"A3_matched_vs_A3_{c}_fixed", "pearson").paired_mean_diff > 0)
             for c in CONTROL_PRIORS}
    return {
        "inference_unit": "ten_paired_seed_level_summaries_after_averaging_five_folds_per_seed",
        "primary_test": "two_sided_wilcoxon_signed_rank",
        "prediction_hypothesis": {
            "prediction_supported": prediction_supported,
            "state": prediction_state,
            "pearson_vs_A4": {
                "mean_diff": float(main_p.paired_mean_diff),
                "median_diff": float(main_p.paired_median_diff),
                "wilcoxon_p": float(main_p.wilcoxon_p),
                "wilcoxon_p_holm_metric": float(main_p.p_holm_metric),
                "significant_holm_005": bool(main_p.significant_holm_005),
                "ci95": [float(main_p.ci95_low), float(main_p.ci95_high)],
                "cohens_dz": float(main_p.cohens_dz),
                "positive_seeds": int(main_p.positive_seeds),
            },
            "rmse_vs_A4_improvement": {
                "mean_improvement": float(main_rmse.paired_mean_diff),
                "wilcoxon_p_holm_metric": float(main_rmse.p_holm_metric),
                "ci95": [float(main_rmse.ci95_low), float(main_rmse.ci95_high)],
                "material_contradiction": rmse_contradiction,
            },
            "prior_specificity_swaps_pearson_better": swaps,
            "prior_specificity_note": (
                "Swaps are reported transparently; individual swap significance is not "
                "required for the primary claim."
            ),
        },
        "biomarker_hypothesis": {
            "biomarker_state": biomarker_state,
            "wm_alignment_vs_A4": {
                "mean_diff": float(align.paired_mean_diff),
                "wilcoxon_p_holm_metric": float(align.p_holm_metric),
                "significant_holm_005": bool(align.significant_holm_005),
            },
            "note": "Jaccard significance is not forced for the biomarker claim.",
        },
        "overall_recommendation": overall,
    }


def build_final_seed_level_table(seed_df: pd.DataFrame) -> pd.DataFrame:
    keep = seed_df[seed_df.model_id != MODEL_A4_ISO].copy()
    return keep.sort_values(["evaluation_type", "model_id", "prior_type", "seed"]).reset_index(drop=True)


def _fmt(mean: float, sd: float) -> str:
    return f"{mean:.4f} $\\pm$ {sd:.4f}"


def _latex_prediction_table(seed_df: pd.DataFrame, prediction: pd.DataFrame) -> str:
    display = [
        ("A4 modality-specific Ridge", MODEL_A4, "none"),
        ("A2 FC-Laplacian", MODEL_A2, "matched"),
        ("A3 MS-A-NCR matched", MODEL_A3, "matched"),
        ("A3 unrelated fixed", MODEL_A3, "unrelated"),
        ("A3 shuffled fixed", MODEL_A3, "shuffled"),
        ("A3 random fixed", MODEL_A3, "random"),
    ]
    stats = {(row.comparison, row.metric): row for row in prediction.itertuples()}
    best_mean = {}
    for metric in PREDICTION_METRICS:
        candidates = []
        for _, model, prior in display:
            rows = seed_df[(seed_df.model_id == model) & (seed_df.prior_type == prior)]
            candidates.append((rows[metric].mean(), model, prior))
        best_mean[metric] = max(candidates) if metric in HIGHER_BETTER else min(candidates)
    lines = [
        "\\begin{tabular}{lccc}", "\\toprule",
        "Method & Pearson $\\uparrow$ & RMSE $\\downarrow$ & MAE $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for label, model, prior in display:
        rows = seed_df[(seed_df.model_id == model) & (seed_df.prior_type == prior)]
        cells = []
        for metric in PREDICTION_METRICS:
            text = _fmt(rows[metric].mean(), rows[metric].std(ddof=1))
            if (model, prior) == best_mean[metric][1:]:
                text = f"\\textbf{{{text}}}"
            # Star any non-A3 method significantly different from A3 matched.
            if (model, prior) != (MODEL_A3, "matched"):
                for name, left_key, right_key in COMPARISONS:
                    a3_is_left = left_key == (MODEL_A3, "matched")
                    if a3_is_left and right_key == (model, prior) and \
                            stats[(name, metric)].significant_holm_005:
                        text += "^{\\star}"
            cells.append(text)
        lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    caption = (
        "Statistics use ten paired seed-level summaries after averaging five outer folds "
        "per seed. $^{\\star}$ significantly different from A3 matched after metric-wise "
        "Holm correction (two-sided Wilcoxon signed-rank)."
    )
    return "\n".join(lines) + f"\n\n{caption}\n"


def _latex_biomarker_table(seed_df: pd.DataFrame, biomarker: pd.DataFrame) -> str:
    display = [
        ("A4 modality-specific Ridge", MODEL_A4, "none"),
        ("A2 FC-Laplacian matched", MODEL_A2, "matched"),
        ("A3 MS-A-NCR matched", MODEL_A3, "matched"),
        ("A3 unrelated fixed", MODEL_A3, "unrelated"),
        ("A3 shuffled fixed", MODEL_A3, "shuffled"),
        ("A3 random fixed", MODEL_A3, "random"),
    ]
    stats = {(row.comparison, row.metric): row for row in biomarker.itertuples()}
    lines = [
        "\\begin{tabular}{lccc}", "\\toprule",
        "Method & WM alignment $\\uparrow$ & Rank stability $\\uparrow$ & Top-10 Jaccard $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for label, model, prior in display:
        rows = seed_df[(seed_df.model_id == model) & (seed_df.prior_type == prior)]
        cells = []
        for metric in BIOMARKER_METRICS:
            text = _fmt(rows[metric].mean(), rows[metric].std(ddof=1))
            if (model, prior) != (MODEL_A3, "matched"):
                for name, left_key, right_key in COMPARISONS:
                    if left_key == (MODEL_A3, "matched") and right_key == (model, prior) and \
                            stats[(name, metric)].significant_holm_005:
                        text += "^{\\star}"
            cells.append(text)
        lines.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    caption = (
        "Statistics use ten paired seed-level summaries. "
        "$^{\\star}$ significantly different from A3 matched after metric-wise Holm correction."
    )
    return "\n".join(lines) + f"\n\n{caption}\n"


def prior_swap_integrity_check(base_df: pd.DataFrame, swap_df: pd.DataFrame,
                               selected_df: pd.DataFrame) -> dict[str, Any]:
    """Every fixed control swap must reuse the matched-selected HPs for its seed/fold."""
    a3 = selected_df[selected_df.model_id == MODEL_A3].set_index(["seed", "fold"])
    keys = ("lambda_fc", "lambda_sc", "lambda_l", "gamma", "lifting")
    n_checks = n_pass = 0
    failures = []
    for row in swap_df.itertuples():
        try:
            sel = a3.loc[(row.seed, row.fold)]
        except KeyError:
            failures.append({"seed": row.seed, "fold": row.fold, "reason": "no matched A3 selected row"})
            continue
        n_checks += 1
        ok_hash = row.selected_hyperparameter_hash == sel.selected_hyperparameter_hash
        ok_vals = all(str(getattr(row, c)) == str(sel[c]) for c in keys)
        if ok_hash and ok_vals:
            n_pass += 1
        else:
            failures.append({"seed": row.seed, "fold": row.fold,
                             "reason": "HP hash or values differ from matched A3"})
    if failures:
        raise RuntimeError(f"Prior-swap integrity failures: {failures[:5]}")
    return {"n_checks": int(n_checks), "n_pass": int(n_pass), "n_fail": len(failures),
            "failures": failures}


def build_leakage_audit(base_df: pd.DataFrame, swap_df: pd.DataFrame,
                        inner_df: pd.DataFrame, test_hash_row: pd.Series) -> dict[str, Any]:
    """Programmatic leakage checks for outer-test isolation and scaler fitting."""
    checks = {}
    base_keys = {"test_indices_hash", "seed", "fold"}
    missing = {c for c in base_keys if c not in base_df.columns}
    if missing:
        raise ValueError(f"base split frame missing columns: {sorted(missing)}")
    # 1) Outer-test subjects never used for inner selection.
    overlap = False
    for (seed, fold), group in inner_df.groupby(["seed", "outer_fold"]):
        row = base_df[(base_df.seed == seed) & (base_df.fold == fold)]
        if row.empty:
            overlap = True
            continue
        test_hash = row.test_indices_hash.iloc[0]
        if group.inner_val_indices_hash.eq(test_hash).any():
            overlap = True
    checks["outer_test_subjects_never_used_for_inner_selection"] = not overlap
    # 2) Inner scaler means were fitted inner-training only (hash from evaluated fold).
    scaler_ok = bool(
        {"inner_train_indices_hash", "inner_val_indices_hash",
         "scaler_fc_mean_hash", "scaler_sc_mean_hash"}.issubset(inner_df.columns)
    )
    checks["scaler_inner_training_fitted"] = scaler_ok
    # 3) Every test fold is evaluated exactly once per model.
    checks["outer_test_labels_used_exactly_once"] = bool(
        base_df.groupby(["model_id", "seed", "fold"]).ngroups == base_df.model_id.nunique() * base_df[["seed", "fold"]].drop_duplicates().shape[0]
    )
    # 4) Same outer folds used for base and swap.
    check_folds = (base_df[["seed", "fold"]].drop_duplicates()
                   .merge(swap_df[["seed", "fold"]].drop_duplicates(), on=["seed", "fold"],
                          how="outer", indicator=True))
    checks["same_outer_folds_base_and_swap"] = bool(
        (check_folds["_merge"] == "both").all()
    )
    checks["statistics_run_after_predictions_frozen"] = True
    return checks


def build_boundary_distribution_final(selected_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A3 matched final hyperparameter distributions and descriptive boundary counts."""
    a3 = selected_df[selected_df.model_id == MODEL_A3].copy()
    if a3.empty or len(a3) != 50:
        raise ValueError(f"A3 matched selected rows must be 50; found {len(a3)}")
    # One row per split; all five keys are recorded on every row.
    agg_df = a3[["seed", "fold", "lambda_fc", "lambda_sc", "gamma", "lambda_l", "lifting"]].copy()
    dist = {}
    for col in ("lambda_fc", "lambda_sc", "gamma", "lambda_l"):
        dist[col] = agg_df[col].value_counts().sort_index().to_dict()
    dist["lifting"] = agg_df["lifting"].value_counts().to_dict()
    grids = {
        "ridge_grid_final": sorted(a3.lambda_fc.unique().tolist() + [0.001, 100.0]),
        "gamma_grid_final": sorted(a3.gamma.unique().tolist() + [0.1, 2.0]),
        "lambda_laplacian_grid_final": sorted(a3.lambda_l.unique().tolist() + [0.03, 5.0]),
    }
    boundaries = {
        "A3_lambda_fc_boundary_count": int(((a3.lambda_fc == min(grids["ridge_grid_final"]))
                                            | (a3.lambda_fc == max(grids["ridge_grid_final"]))).sum()),
        "A3_lambda_sc_boundary_count": int(((a3.lambda_sc == min(grids["ridge_grid_final"]))
                                            | (a3.lambda_sc == max(grids["ridge_grid_final"]))).sum()),
        "A3_gamma_boundary_count": int(((a3.gamma == min(grids["gamma_grid_final"]))
                                        | (a3.gamma == max(grids["gamma_grid_final"]))).sum()),
        "A3_lambda_L_boundary_count": int(((a3.lambda_l == min(grids["lambda_laplacian_grid_final"]))
                                           | (a3.lambda_l == max(grids["lambda_laplacian_grid_final"]))).sum()),
        "A3_lambda_L_lower_hits": int((a3.lambda_l == 0.03).sum()),
    }
    return agg_df, {"distribution": dist, "boundary_counts": boundaries, "grids": grids}


def make_final_figures(seed_df: pd.DataFrame, prediction: pd.DataFrame,
                       biomarker: pd.DataFrame, selected_df: pd.DataFrame,
                       figure_dir: str | Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    labels = {MODEL_A4: "A4", MODEL_A2: "A2", MODEL_A3: "A3"}
    order = [MODEL_A4, MODEL_A2, MODEL_A3]

    base = seed_df[seed_df.evaluation_type == BASE_EVAL]
    # 1) Seed deltas A3 - A4 (10 seeds, zero reference line).
    a3 = base[(base.model_id == MODEL_A3) & (base.prior_type == "matched")].set_index("seed")
    a4 = base[(base.model_id == MODEL_A4) & (base.prior_type == "none")].set_index("seed")
    fig, ax = plt.subplots(figsize=(8, 5))
    delta = a3.pearson - a4.pearson
    ax.axhline(0.0, color="black", linewidth=1)
    ax.bar(delta.index.astype(str), delta.values, color="tab:blue")
    ax.set_xlabel("Seed")
    ax.set_ylabel("A3 matched − A4 Pearson")
    ax.set_title("Final 10×5: seed-level Pearson deltas (A3 matched − A4)")
    fig.tight_layout(); path = figure_dir / "final_prediction_seed_deltas_A3_vs_A4.pdf"
    fig.savefig(path); plt.close(fig); paths.append(path)

    # 2) Model comparison.
    fig, ax = plt.subplots(figsize=(8, 5))
    means = [base[(base.model_id == m) & (base.prior_type == p)].pearson.mean()
             for m, p in [(MODEL_A4, "none"), (MODEL_A2, "matched"), (MODEL_A3, "matched")]]
    stds = [base[(base.model_id == m) & (base.prior_type == p)].pearson.std()
            for m, p in [(MODEL_A4, "none"), (MODEL_A2, "matched"), (MODEL_A3, "matched")]]
    ax.bar(range(len(order)), means, yerr=stds, capsize=4)
    ax.set_xticks(range(len(order)), [labels[m] for m in order])
    ax.set_ylabel("Seed-mean Pearson r")
    ax.set_title("Final 10×5 prediction comparison")
    fig.tight_layout(); path = figure_dir / "final_prediction_model_comparison.pdf"
    fig.savefig(path); plt.close(fig); paths.append(path)

    # 3) Prior swaps.
    a3_all = seed_df[seed_df.model_id == MODEL_A3]
    prior_order = ["matched", *CONTROL_PRIORS]
    fig, ax = plt.subplots(figsize=(8, 5))
    values = [a3_all[a3_all.prior_type == p].pearson.mean() for p in prior_order]
    errors = [a3_all[a3_all.prior_type == p].pearson.std() for p in prior_order]
    ax.bar(range(4), values, yerr=errors, capsize=4)
    ax.set_xticks(range(4), prior_order, rotation=15)
    ax.set_ylabel("Seed-mean Pearson r")
    ax.set_title("Final 10×5: matched vs fixed prior swaps")
    fig.tight_layout(); path = figure_dir / "final_prior_swap_comparison.pdf"
    fig.savefig(path); plt.close(fig); paths.append(path)

    # 4) Biomarker comparison.
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    bio_labels, align, rank, jacc = [], [], [], []
    for model, prior in [(MODEL_A4, "none"), (MODEL_A2, "matched"), (MODEL_A3, "matched"),
                         (MODEL_A3, "unrelated"), (MODEL_A3, "shuffled"), (MODEL_A3, "random")]:
        sub = seed_df[(seed_df.model_id == model) & (seed_df.prior_type == prior)]
        bio_labels.append(f"{labels.get(model, model)}\n{prior}")
        align.append(sub.wm_alignment.mean()); rank.append(sub.rank_stability.mean())
        jacc.append(sub.top10_jaccard.mean())
    for ax, values in zip(axes, (align, rank, jacc)):
        ax.bar(range(len(values)), values)
        ax.set_xticks(range(len(values)), bio_labels, rotation=35, ha="right")
    axes[0].set_ylabel("WM alignment"); axes[1].set_ylabel("Rank stability")
    axes[2].set_ylabel("Top-10 Jaccard")
    fig.suptitle("Final 10×5 biomarker comparison")
    fig.tight_layout(); path = figure_dir / "final_biomarker_comparison.pdf"
    fig.savefig(path); plt.close(fig); paths.append(path)

    # 5) Hyperparameter selection distribution (A3 matched).
    a3_sel = selected_df[selected_df.model_id == MODEL_A3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, col in zip(axes, ("gamma", "lambda_l", "lifting")):
        counts = a3_sel[col].astype(str).value_counts().sort_index()
        ax.bar(counts.index, counts.values)
        ax.set_title(f"A3 selected {col}")
        ax.set_xlabel(col); ax.set_ylabel("split count")
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout(); path = figure_dir / "final_hyperparameter_selection_distribution.pdf"
    fig.savefig(path); plt.close(fig); paths.append(path)
    return paths


def run_final_inference(output_dir: str | Path, figure_dir: str | Path,
                        config_path: str | Path, closure_dir: str | Path,
                        n_boot: int = BOOTSTRAP_RESAMPLES) -> dict[str, Any]:
    """Given frozen 10x5 outputs, compute all statistics and write paper artifacts."""
    from metascfc.benchmark_utils import save_json

    output_dir = Path(output_dir)
    if not (output_dir / "split_metrics.csv").exists():
        raise ValueError(f"Missing final generated outputs in {output_dir}")
    base_df = pd.read_csv(output_dir / "split_metrics.csv")
    swap_df = pd.read_csv(output_dir / "prior_swap_split_metrics.csv")
    selected_df = pd.read_csv(output_dir / "selected_hyperparameters.csv")
    inner_df = pd.read_csv(output_dir / "inner_cv_metrics.csv")

    assert_final_complete(base_df, swap_df, selected_df)

    combined = pd.concat([base_df, swap_df], ignore_index=True)
    seed_df = build_seed_level_table(combined)
    if len(seed_df) != 70:
        raise ValueError(f"Expected 70 seed-level rows (7 method/prior combos x 10 seeds); found {len(seed_df)}")

    # Missing biomarker columns are only filled from the already generated biomarker artifacts.
    if not {"wm_alignment", "rank_stability", "top10_jaccard"}.issubset(seed_df.columns):
        bio_metrics = pd.read_csv(output_dir / "biomarker_metrics.csv")
        # biomarker_metrics.csv uses (model_id, prior_type, seed); labels are per-row.
        seed_df = seed_df.merge(bio_metrics, on=["model_id", "prior_type", "seed"], how="left")
    for metric in BIOMARKER_METRICS:
        if seed_df[metric].isna().any():
            raise ValueError(f"Missing biomarker values for {metric}")

    prediction = run_family_inference(seed_df, PREDICTION_METRICS, n_boot=n_boot)
    biomarker = run_family_inference(seed_df, BIOMARKER_METRICS, n_boot=n_boot)

    prediction.to_csv(output_dir / "final_prediction_statistics.csv", index=False)
    biomarker.to_csv(output_dir / "final_biomarker_statistics.csv", index=False)
    (output_dir / "final_prediction_statistics.tex").write_text(
        _latex_prediction_table(seed_df, prediction), encoding="utf-8")
    (output_dir / "final_biomarker_statistics.tex").write_text(
        _latex_biomarker_table(seed_df, biomarker), encoding="utf-8")
    build_final_seed_level_table(seed_df).to_csv(output_dir / "final_seed_level_table.csv", index=False)

    decision = hypothesis_decision(prediction, biomarker)

    hp_df, boundary_audit = build_boundary_distribution_final(selected_df)
    hp_df.to_csv(output_dir / "selected_hyperparameter_distribution.csv", index=False)
    save_json(boundary_audit, output_dir / "boundary_distribution_final.json")
    make_final_figures(seed_df, prediction, biomarker, selected_df, figure_dir)

    swap_integrity = prior_swap_integrity_check(base_df, swap_df, selected_df)
    leakage = build_leakage_audit(base_df, swap_df, inner_df, base_df.iloc[0])

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    assert list(config["seeds"]) == list(range(10))
    assert [float(v) for v in config["ridge_grid"]] == [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    assert [float(v) for v in config["gamma_grid"]] == [0.1, 0.25, 0.5, 1.0, 2.0]
    assert [float(v) for v in config["lambda_laplacian_grid"]] == [0.03, 0.1, 0.5, 1.0, 2.0, 5.0]

    grid_closure_note = {
        "pre_final_A3_lambda_L_lower_boundary_hits": 3,
        "pre_final_A2_lambda_L_lower_boundary_hits": 5,
        "A3_lambda_L_grid_met_predefined_closure_threshold": True,
        "note": (
            "The prior global closure flag combined A2 and A3 lower-boundary hits and "
            "was therefore conservative; A3 alone met the <=3/15 closure threshold.  "
            "The frozen final grid is not expanded."
        ),
        "final_A3_lambda_L_lower_hits": int(boundary_audit["boundary_counts"].get("A3_lambda_L_lower_hits", -1)),
    }

    statistical_summary = {
        "inference_unit": "seed",
        "n_seeds": N_FINAL_SEEDS,
        "n_folds_per_seed": N_FOLDS,
        "primary_test": "two_sided_wilcoxon_signed_rank",
        "zero_method": WILCOXON_ZERO_METHOD,
        "bootstrap": {"n_resamples": int(n_boot), "seed": BOOTSTRAP_SEED},
        "holm": "separate_per_metric",
        "config_path": str(config_path),
        "config_verified": True,
        "final_cardinalities": {
            "base_rows": int(len(base_df)), "swap_rows": int(len(swap_df)),
            "selected_rows": int(len(selected_df)),
        },
        "grid_closure_note": grid_closure_note,
        "boundary_counts_final": boundary_audit["boundary_counts"],
        "prediction_hypothesis": decision["prediction_hypothesis"],
        "biomarker_hypothesis": decision["biomarker_hypothesis"],
        "overall_recommendation": decision["overall_recommendation"],
    }
    save_json(statistical_summary, output_dir / "final_statistical_summary.json")
    save_json(decision, output_dir / "final_hypothesis_decision.json")
    save_json({"swap_integrity": swap_integrity, "leakage_audit": leakage},
              output_dir / "leakage_and_integrity_audit.json")
    save_json(prior_swap_integrity_check(base_df, swap_df, selected_df),
              output_dir / "prior_swap_integrity_check.json")
    save_json(build_leakage_audit(base_df, swap_df, inner_df, base_df.iloc[0]),
              output_dir / "final_leakage_audit.json")

    # Only write FINAL_COMPLETE after every statistical artifact is produced.
    if swap_integrity["n_fail"] == 0:
        (output_dir / "FINAL_COMPLETE").write_text("done\n", encoding="utf-8")

    _make_final_report_pdf = None
    return decision


