from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Iterator, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, t as student_t, ttest_1samp, wilcoxon
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def load_numpy_flexible(path: str | Path, name: str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    try:
        arr = np.load(path, allow_pickle=False)
    except ValueError as exc:
        if "pickled" not in str(exc).lower() and "object" not in str(exc).lower():
            raise
        arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.asarray(arr.tolist())
    return np.asarray(arr)


def load_connectomes(data_cfg: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    fc = load_numpy_flexible(data_cfg["fc_path"], "FC").astype(np.float32)
    sc = load_numpy_flexible(data_cfg["sc_path"], "SC").astype(np.float32)
    y = load_numpy_flexible(data_cfg["y_path"], "labels").astype(np.float64).reshape(-1)
    if fc.ndim != 3 or sc.ndim != 3 or fc.shape != sc.shape:
        raise ValueError(f"Expected matched [subjects, ROI, ROI] arrays; FC={fc.shape}, SC={sc.shape}")
    if fc.shape[1] != fc.shape[2] or len(y) != len(fc):
        raise ValueError(f"Data mismatch: FC={fc.shape}, SC={sc.shape}, y={y.shape}")
    if not (np.isfinite(fc).all() and np.isfinite(sc).all() and np.isfinite(y).all()):
        raise ValueError("FC, SC, or labels contain NaN/Inf values")

    subject_ids = np.arange(len(y)).astype(str)
    subjects_path = data_cfg.get("subjects_path")
    if subjects_path and Path(subjects_path).exists():
        sdf = pd.read_csv(subjects_path)
        col = "subject" if "subject" in sdf.columns else "Subject"
        if col in sdf.columns and len(sdf) == len(y):
            subject_ids = sdf[col].astype(str).to_numpy()

    groups = None
    groups_path = data_cfg.get("groups_path")
    if groups_path:
        groups = load_numpy_flexible(groups_path, "groups").reshape(-1)
        if len(groups) != len(y):
            raise ValueError(f"Group count {len(groups)} does not match subject count {len(y)}")
    return fc, sc, y, subject_ids, groups


def quantile_bins(y: np.ndarray, n_bins: int = 5):
    try:
        return pd.qcut(y, q=min(n_bins, len(y)), labels=False, duplicates="drop")
    except Exception:
        return None


def make_inner_split(
    trainval_idx: np.ndarray,
    y: np.ndarray,
    val_fraction: float,
    seed: int,
    groups: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    trainval_idx = np.asarray(trainval_idx, dtype=int)
    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        tr_local, va_local = next(splitter.split(trainval_idx, y[trainval_idx], groups[trainval_idx]))
        return trainval_idx[tr_local], trainval_idx[va_local]
    stratify = quantile_bins(y[trainval_idx])
    try:
        tr, va = train_test_split(
            trainval_idx, test_size=val_fraction, random_state=seed,
            shuffle=True, stratify=stratify,
        )
    except ValueError:
        tr, va = train_test_split(
            trainval_idx, test_size=val_fraction, random_state=seed, shuffle=True,
        )
    return np.asarray(tr, dtype=int), np.asarray(va, dtype=int)


def iter_nested_splits(
    y: np.ndarray,
    seeds: Sequence[int],
    n_folds: int,
    val_fraction: float,
    groups: np.ndarray | None = None,
) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]]:
    indices = np.arange(len(y))
    for seed in seeds:
        if groups is not None:
            outer = GroupKFold(n_splits=n_folds)
            split_iter = outer.split(indices, y, groups)
        else:
            outer = KFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
            split_iter = outer.split(indices)
        for fold, (trainval_idx, test_idx) in enumerate(split_iter):
            split_seed = int(seed) * 1000 + int(fold)
            train_idx, val_idx = make_inner_split(
                np.asarray(trainval_idx), y, val_fraction, split_seed, groups,
            )
            yield int(seed), int(fold), train_idx, val_idx, np.asarray(test_idx, dtype=int)


def prediction_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) > 1 and np.std(y_true) > 1e-12 and np.std(y_pred) > 1e-12:
        r = float(pearsonr(y_true, y_pred).statistic)
    else:
        r = 0.0
    if not np.isfinite(r):
        r = 0.0
    return {
        "pearson": r,
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
    }


def choose_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def rowwise_topk_adjacency(mats: np.ndarray, top_percent: float, positive_only: bool = True) -> np.ndarray:
    """Return symmetric weighted adjacency retaining top-k entries per row.

    This follows the IMG-GCN graph description more closely than a global
    percentile: the highest fraction is retained for every ROI, then the graph
    is symmetrized. Diagonal entries are removed here and added during graph
    normalization.
    """
    mats = np.asarray(mats, dtype=np.float32)
    b, n, _ = mats.shape
    k = max(1, int(math.ceil((n - 1) * float(top_percent) / 100.0)))
    out = np.zeros_like(mats, dtype=np.float32)
    for s in range(b):
        m = mats[s].copy()
        np.fill_diagonal(m, -np.inf)
        if positive_only:
            score = np.where(m > 0, m, -np.inf)
        else:
            score = np.abs(m)
        idx = np.argpartition(score, kth=max(0, n - k), axis=1)[:, -k:]
        rows = np.arange(n)[:, None]
        vals = mats[s][rows, idx]
        if positive_only:
            vals = np.maximum(vals, 0.0)
        out[s][rows, idx] = vals
        out[s] = np.maximum(out[s], out[s].T)
        np.fill_diagonal(out[s], 0.0)
    return out


def binary_sc_adjacency(sc: np.ndarray) -> np.ndarray:
    out = (np.asarray(sc) > 0).astype(np.float32)
    for m in out:
        np.fill_diagonal(m, 0.0)
    return np.maximum(out, np.swapaxes(out, 1, 2))


def normalize_dense_adjacency_torch(adj: torch.Tensor, add_self_loops: bool = True) -> torch.Tensor:
    if adj.ndim != 3 or adj.shape[1] != adj.shape[2]:
        raise ValueError(f"Expected [B,N,N] adjacency, got {tuple(adj.shape)}")
    a = adj
    if add_self_loops:
        eye = torch.eye(a.shape[-1], device=a.device, dtype=a.dtype).unsqueeze(0)
        a = a + eye
    degree = a.sum(dim=-1).clamp_min(1e-8)
    inv_sqrt = degree.rsqrt()
    return inv_sqrt.unsqueeze(-1) * a * inv_sqrt.unsqueeze(-2)


def aggregate_split_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method_id, g in df.groupby("method_id", sort=False):
        rows.append({
            "ID": method_id,
            "Method": g["method_name"].iloc[0],
            "Seeds": int(g["seed"].nunique()),
            "Folds": int(g["fold"].nunique()),
            "Evaluations": int(len(g)),
            "Pearson Mean": float(g["pearson"].mean()),
            "Pearson Std": float(g["pearson"].std(ddof=1)),
            "RMSE Mean": float(g["rmse"].mean()),
            "RMSE Std": float(g["rmse"].std(ddof=1)),
            "MAE Mean": float(g["mae"].mean()),
            "MAE Std": float(g["mae"].std(ddof=1)),
            "Runtime Hours": float(g["runtime_seconds"].sum() / 3600.0),
        })
    return pd.DataFrame(rows)


def seed_level_metrics(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["method_id", "method_name", "seed"], as_index=False)
        .agg(pearson=("pearson", "mean"), rmse=("rmse", "mean"), mae=("mae", "mean"), folds=("fold", "nunique"))
    )


def t_confidence_interval(values: Iterable[float], confidence: float = 0.95) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if len(arr) < 2:
        return float(arr.mean()), float(arr.mean())
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / math.sqrt(len(arr)))
    crit = float(student_t.ppf((1 + confidence) / 2, df=len(arr) - 1))
    return mean - crit * sem, mean + crit * sem


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def paired_seed_tests(seed_df: pd.DataFrame, reference_id: str) -> pd.DataFrame:
    ref = seed_df[seed_df.method_id == reference_id].set_index("seed")
    if ref.empty:
        raise ValueError(f"Reference method {reference_id} is absent")
    rows = []
    for method_id in seed_df.method_id.unique():
        if method_id == reference_id:
            continue
        cur = seed_df[seed_df.method_id == method_id].set_index("seed")
        common = sorted(set(ref.index).intersection(cur.index))
        if len(common) < 2:
            continue
        for metric in ("pearson", "rmse", "mae"):
            a = cur.loc[common, metric].to_numpy(float)
            b = ref.loc[common, metric].to_numpy(float)
            diff = a - b if metric == "pearson" else b - a
            mean = float(diff.mean())
            sd = float(diff.std(ddof=1))
            dz = mean / sd if sd > 1e-12 else 0.0
            t_p = float(ttest_1samp(diff, 0.0, alternative="two-sided").pvalue)
            if np.allclose(diff, 0.0):
                w_p = 1.0
            else:
                try:
                    w_p = float(wilcoxon(diff, alternative="two-sided", zero_method="wilcox").pvalue)
                    if not np.isfinite(w_p):
                        w_p = 1.0
                except ValueError:
                    w_p = 1.0
            lo, hi = t_confidence_interval(diff)
            rows.append({
                "reference_id": reference_id,
                "method_id": method_id,
                "method_name": cur.method_name.iloc[0],
                "metric": metric,
                "n_seeds": len(common),
                "improvement_mean": mean,
                "ci95_low": lo,
                "ci95_high": hi,
                "cohens_dz": dz,
                "paired_t_p": t_p,
                "wilcoxon_p": w_p,
            })
    out = pd.DataFrame(rows)
    if not out.empty:
        out["paired_t_p_holm"] = holm_adjust(out["paired_t_p"].to_numpy())
        out["wilcoxon_p_holm"] = holm_adjust(out["wilcoxon_p"].to_numpy())
    return out



def paired_wilcoxon_vs_reference(
    seed_df: pd.DataFrame,
    reference_id: str,
    metrics: Sequence[str] = ("pearson", "rmse", "mae"),
) -> pd.DataFrame:
    """Compare one reference predictor with every alternative by paired seed.

    Each row of ``seed_df`` must represent one method/seed value after folds
    have already been averaged.  The returned difference is oriented so that
    a positive value always means that the reference method performs better:
    ``reference - method`` for higher-is-better metrics and ``method -
    reference`` for lower-is-better metrics.  Wilcoxon signed-rank tests are
    two-sided, paired by seed, and Holm-adjusted separately within each metric.

    A nonsignificant result means that the available repeated-CV evidence does
    not detect a difference; it is not an equivalence test.
    """
    required = {"method_id", "method_name", "seed", *metrics}
    missing = required.difference(seed_df.columns)
    if missing:
        raise ValueError(f"seed_df is missing columns: {sorted(missing)}")

    ref = seed_df[seed_df.method_id == reference_id].set_index("seed")
    if ref.empty:
        raise ValueError(f"Reference method {reference_id} is absent")
    if ref.index.duplicated().any():
        raise ValueError(f"Reference method {reference_id} has duplicate seed rows")

    lower_better = {"rmse", "mae"}
    rows = []
    for method_id in seed_df.method_id.drop_duplicates():
        if method_id == reference_id:
            continue
        cur = seed_df[seed_df.method_id == method_id].set_index("seed")
        if cur.index.duplicated().any():
            raise ValueError(f"Method {method_id} has duplicate seed rows")
        common = sorted(set(ref.index).intersection(cur.index))
        if len(common) < 2:
            continue

        for metric in metrics:
            ref_values = ref.loc[common, metric].to_numpy(float)
            method_values = cur.loc[common, metric].to_numpy(float)
            finite = np.isfinite(ref_values) & np.isfinite(method_values)
            ref_values = ref_values[finite]
            method_values = method_values[finite]
            if len(ref_values) < 2:
                continue

            # Positive advantage means the reference is better.
            advantage = (
                method_values - ref_values
                if metric in lower_better
                else ref_values - method_values
            )
            if np.allclose(advantage, 0.0):
                statistic, pvalue = 0.0, 1.0
            else:
                try:
                    result = wilcoxon(
                        advantage,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="auto",
                    )
                    statistic = float(result.statistic)
                    pvalue = float(result.pvalue)
                    if not np.isfinite(pvalue):
                        pvalue = 1.0
                except ValueError:
                    statistic, pvalue = 0.0, 1.0

            rows.append({
                "reference_id": reference_id,
                "reference_name": ref.method_name.iloc[0],
                "method_id": method_id,
                "method_name": cur.method_name.iloc[0],
                "metric": metric,
                "analysis_unit": "seed_mean_over_5_folds",
                "n_pairs": int(len(advantage)),
                "reference_mean": float(ref_values.mean()),
                "method_mean": float(method_values.mean()),
                "reference_advantage_mean": float(advantage.mean()),
                "reference_advantage_median": float(np.median(advantage)),
                "wilcoxon_W": statistic,
                "wilcoxon_p": pvalue,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["wilcoxon_p_holm_metric"] = np.nan
    for metric, indices in out.groupby("metric").groups.items():
        idx = list(indices)
        out.loc[idx, "wilcoxon_p_holm_metric"] = holm_adjust(
            out.loc[idx, "wilcoxon_p"].to_numpy(float)
        )
    out["significant_holm_005"] = out["wilcoxon_p_holm_metric"] < 0.05
    out["interpretation"] = np.where(
        out["significant_holm_005"],
        "significantly different from best",
        "no significant difference detected",
    )
    return out.sort_values(["metric", "wilcoxon_p_holm_metric", "method_id"])

def atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def save_json(obj: Dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
