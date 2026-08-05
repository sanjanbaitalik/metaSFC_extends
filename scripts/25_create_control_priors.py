from pathlib import Path
import json
import numpy as np
import pandas as pd


TRUE_PRIOR_DIR = Path("outputs/priors/working_memory/aal116")
MODULE_MAP_PATH = Path("inputs/atlases/AAL116_coarse_modules.csv")

OUT_SHUFFLED = Path("outputs/priors/working_memory_shuffled/aal116")
OUT_RANDOM = Path("outputs/priors/random_prior/aal116")


def normalize_minmax(x):
    x = np.asarray(x, dtype=np.float32)
    if x.max() - x.min() < 1e-8:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


def build_edge_prior(prior_scores):
    edge = np.outer(prior_scores, prior_scores).astype(np.float32)
    edge = normalize_minmax(edge)
    return edge


def build_module_prior(roi_df):
    if not MODULE_MAP_PATH.exists():
        print("[WARN] Module mapping not found. Skipping module_prior.csv")
        return None

    module_df = pd.read_csv(MODULE_MAP_PATH)

    if "module" not in module_df.columns:
        raise ValueError("AAL116_coarse_modules.csv must contain a 'module' column.")

    merged = roi_df.merge(module_df, on="roi_index", how="left")

    if merged["module"].isna().any():
        missing = merged[merged["module"].isna()]["roi_index"].tolist()
        raise ValueError(f"Missing module labels for ROI indices: {missing[:10]}")

    group_cols = ["module_id", "module"] if "module_id" in merged.columns else ["module"]
    mod = (merged.groupby(group_cols, as_index=False)["prior_score"]
           .mean().rename(columns={"prior_score": "raw_score"}))
    mod["prior_score"] = normalize_minmax(mod["raw_score"].values)
    if "module_id" in mod.columns:
        mod = mod.sort_values("module_id").reset_index(drop=True)
    return mod


def save_prior_set(out_dir, roi_df, name):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    roi_df = roi_df.copy()
    roi_df["prior_score"] = normalize_minmax(roi_df["prior_score"].values)

    roi_df.to_csv(out_dir / "roi_prior.csv", index=False)

    edge = build_edge_prior(roi_df["prior_score"].values)
    np.save(out_dir / "edge_prior.npy", edge)

    module_prior = build_module_prior(roi_df)
    if module_prior is not None:
        module_prior.to_csv(out_dir / "module_prior.csv", index=False)

    metadata = {
        "control_type": name,
        "source_prior": str(TRUE_PRIOR_DIR),
        "n_rois": int(len(roi_df)),
        "edge_shape": list(edge.shape),
        "note": "Control prior generated for negative-control experiments."
    }

    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved {name} prior to: {out_dir}")
    print("ROI prior:", out_dir / "roi_prior.csv")
    print("Module prior:", out_dir / "module_prior.csv")
    print("Edge prior:", out_dir / "edge_prior.npy")


def main():
    rng = np.random.default_rng(42)

    true_roi_path = TRUE_PRIOR_DIR / "roi_prior.csv"
    if not true_roi_path.exists():
        raise FileNotFoundError(f"Missing true prior: {true_roi_path}")

    roi_df = pd.read_csv(true_roi_path)

    required = {"roi_index", "roi_label", "prior_score"}
    missing = required - set(roi_df.columns)
    if missing:
        raise ValueError(f"roi_prior.csv missing columns: {missing}")

    # 1. Shuffled prior: same values, wrong ROI assignment
    shuffled_df = roi_df.copy()
    shuffled_df["prior_score"] = rng.permutation(shuffled_df["prior_score"].values)
    save_prior_set(OUT_SHUFFLED, shuffled_df, "shuffled_working_memory_prior")

    # 2. Random prior: independent random ROI scores
    random_df = roi_df.copy()
    random_df["prior_score"] = rng.random(len(random_df))
    save_prior_set(OUT_RANDOM, random_df, "random_prior")


if __name__ == "__main__":
    main()