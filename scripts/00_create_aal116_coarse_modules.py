#!/usr/bin/env python3
"""
Create a coarse anatomical ROI-to-module mapping for the reindexed AAL116 atlas.

This is a pragmatic prototype mapping for module-prior experiments. It is NOT a
replacement for a rigorously computed AAL-to-Yeo overlap mapping. Use it for the
first technical run, then replace it with a curated/Yeo-overlap mapping before
writing biological claims.
"""
from pathlib import Path

import pandas as pd


MODULE_ORDER = [
    "frontal_executive",
    "sensorimotor",
    "parietal_attention",
    "temporal_language_memory",
    "occipital_visual",
    "cingulo_insular_limbic",
    "subcortical",
    "cerebellar_vermis",
]
MODULE_TO_ID = {m: i for i, m in enumerate(MODULE_ORDER)}


def infer_module(label: str) -> str:
    x = label.lower()

    # Cerebellar and vermis labels should be caught before generic lobar rules.
    if "cerebel" in x or "vermis" in x:
        return "cerebellar_vermis"

    # Subcortical nuclei in AAL116.
    if any(k in x for k in ["thalamus", "caudate", "putamen", "pallidum", "hippocampus", "amygdala"]):
        return "subcortical"

    # Limbic / cingulo-insular / olfactory-orbitofrontal regions.
    if any(k in x for k in ["insula", "cingulum", "parahippocampal", "olfactory", "rectus"]):
        return "cingulo_insular_limbic"

    if any(k in x for k in ["calcarine", "cuneus", "lingual", "occipital", "fusiform"]):
        return "occipital_visual"

    if any(k in x for k in ["precentral", "postcentral", "rolandic", "supp_motor"]):
        return "sensorimotor"

    if any(k in x for k in ["parietal", "precuneus", "angular", "supramarginal"]):
        return "parietal_attention"

    if any(k in x for k in ["temporal", "heschl"]):
        return "temporal_language_memory"

    if "frontal" in x:
        return "frontal_executive"

    return "cingulo_insular_limbic"


def main() -> None:
    labels_path = Path("inputs/atlases/AAL116_labels.csv")
    out_path = Path("inputs/atlases/AAL116_coarse_modules.csv")

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing {labels_path}. Generate AAL116_labels.csv first.")

    labels = pd.read_csv(labels_path)
    if "roi_index" not in labels.columns or "roi_label" not in labels.columns:
        raise ValueError("AAL116_labels.csv must contain roi_index and roi_label columns.")

    rows = []
    for _, row in labels.sort_values("roi_index").iterrows():
        module = infer_module(str(row["roi_label"]))
        rows.append({
            "roi_index": int(row["roi_index"]),
            "roi_label": row["roi_label"],
            "module_id": MODULE_TO_ID[module],
            "module": module,
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(pd.DataFrame(rows)["module"].value_counts().sort_index())
    print("\nNOTE: This is a coarse anatomical mapping for prototyping only.")


if __name__ == "__main__":
    main()
