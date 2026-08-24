"""Pack processed HCP FC/SC matrices with RAW behavioral labels.

Do not globally z-score labels here. The AAAI runner fits label normalization on
training subjects only and applies it to validation/test predictions without
using held-out label statistics.

Dual-target QC (ICLR 2027): subjects missing EITHER behavioral target
(``label`` = PMAT24_A_CR fluid intelligence or ``listsort_unadj`` =
ListSort_Unadj working memory) are dropped BEFORE packing, with the exact
drop counts logged.  Both label arrays are materialized:
``inputs/dataset_SC/label_all.npy`` (PMAT) and
``inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy``.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

labels_df = pd.read_csv("data/hcp/processed/labels.csv")
required_cols = {"subject", "label", "listsort_unadj"}
missing_cols = required_cols - set(labels_df.columns)
if missing_cols:
    raise ValueError(
        f"labels.csv is missing {sorted(missing_cols)}; re-run "
        "scripts/21_prepare_hcp_labels.py (dual-target version) first."
    )

n_input = len(labels_df)
nan_mask = labels_df[["label", "listsort_unadj"]].isna().to_numpy().any(axis=1)
if nan_mask.any():
    dropped_ids = labels_df.loc[nan_mask, "subject"].astype(str).tolist()
    print(f"[QC] dropping {len(dropped_ids)} subjects with NaN behavioral "
          f"targets (e.g. {dropped_ids[:5]}...)")
    labels_df = labels_df.loc[~nan_mask].reset_index(drop=True)

subjects = labels_df["subject"].astype(str).tolist()
pmat_map = dict(zip(labels_df["subject"].astype(str), labels_df["label"]))
wm_map = dict(zip(labels_df["subject"].astype(str), labels_df["listsort_unadj"]))
fc_list, sc_list, y_list, wm_list, kept = [], [], [], [], []
n_dropped_shape = n_dropped_files = 0

for sub in subjects:
    fc_path = Path(f"data/hcp/processed/fc/{sub}_fc.npy")
    sc_path = Path(f"data/hcp/processed/sc/{sub}/sc_116.csv")
    if not fc_path.exists() or not sc_path.exists():
        print(f"[SKIP] missing FC/SC: {sub}"); n_dropped_files += 1; continue
    fc = np.load(fc_path).astype("float32")
    sc = np.loadtxt(sc_path, delimiter=",").astype("float32")
    if fc.shape != (116,116) or sc.shape != (116,116):
        print(f"[SKIP] bad shape {sub}: FC={fc.shape}, SC={sc.shape}")
        n_dropped_shape += 1
        continue
    fc=np.nan_to_num(fc); sc=np.nan_to_num(sc)
    sc=np.log1p(np.clip(sc,0,None))
    if sc.max()>0: sc=sc/sc.max()
    fc=(fc+fc.T)/2; sc=(sc+sc.T)/2
    np.fill_diagonal(fc,0); np.fill_diagonal(sc,0)
    fc_list.append(fc); sc_list.append(sc)
    y_list.append(float(pmat_map[sub])); wm_list.append(float(wm_map[sub]))
    kept.append(sub)

if not kept: raise RuntimeError("No subjects had both FC and SC")
FC_all=np.stack(fc_list).astype("float32"); SC_all=np.stack(sc_list).astype("float32")
label_all=np.asarray(y_list,dtype="float32"); wm_all=np.asarray(wm_list,dtype="float64")

print(f"[QC] input subjects: {n_input}; dropped {int(nan_mask.sum())} for NaN "
      f"behavioral targets, {n_dropped_files} for missing FC/SC files, "
      f"{n_dropped_shape} for wrong matrix dimensions; packed {len(kept)}.")

Path("inputs/dataset_FC").mkdir(parents=True,exist_ok=True); Path("inputs/dataset_SC").mkdir(parents=True,exist_ok=True)
np.save("inputs/dataset_FC/FC_all.npy",FC_all); np.save("inputs/dataset_SC/SC_all.npy",SC_all); np.save("inputs/dataset_SC/label_all.npy",label_all)
pd.DataFrame({"subject":kept,"label_raw":label_all}).to_csv("inputs/dataset_SC/hcp_subjects_used.csv",index=False)
Path("inputs/dataset_SC/label_metadata.json").write_text(json.dumps({"globally_standardized":False,"mean":float(label_all.mean()),"std":float(label_all.std()),"target":"PMAT24_A_CR"},indent=2))

# Dual-task: materialize the ListSort_Unadj labels in the standard layout so
# scripts/50 finds them without a separate behavior-CSV pass.
task_dir = Path("inputs/dataset_SC/task_labels/ListSort_Unadj"); task_dir.mkdir(parents=True, exist_ok=True)
np.save(task_dir / "label_all.npy", wm_all)
(task_dir / "label_metadata.json").write_text(json.dumps({
    "target": "ListSort_Unadj", "globally_standardized": False,
    "mean": float(wm_all.mean()), "std": float(wm_all.std()),
    "n_subjects": int(len(wm_all))}, indent=2))
print("FC",FC_all.shape,"SC",SC_all.shape,"Y",label_all.shape,"range",(label_all.min(),label_all.max()))
print("WM task labels:", wm_all.shape, "range", (wm_all.min(), wm_all.max()))
