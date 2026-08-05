"""Pack processed HCP FC/SC matrices with RAW behavioral labels.

Do not globally z-score labels here. The AAAI runner fits label normalization on
training subjects only and applies it to validation/test predictions without
using held-out label statistics.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

labels_df = pd.read_csv("data/hcp/processed/labels.csv")
subjects = labels_df["subject"].astype(str).tolist()
label_map = dict(zip(labels_df["subject"].astype(str), labels_df["label"]))
fc_list, sc_list, y_list, kept = [], [], [], []

for sub in subjects:
    fc_path = Path(f"data/hcp/processed/fc/{sub}_fc.npy")
    sc_path = Path(f"data/hcp/processed/sc/{sub}/sc_116.csv")
    if not fc_path.exists() or not sc_path.exists():
        print(f"[SKIP] missing FC/SC: {sub}"); continue
    fc = np.load(fc_path).astype("float32")
    sc = np.loadtxt(sc_path, delimiter=",").astype("float32")
    if fc.shape != (116,116) or sc.shape != (116,116):
        print(f"[SKIP] bad shape {sub}: FC={fc.shape}, SC={sc.shape}"); continue
    fc=np.nan_to_num(fc); sc=np.nan_to_num(sc)
    sc=np.log1p(np.clip(sc,0,None))
    if sc.max()>0: sc=sc/sc.max()
    fc=(fc+fc.T)/2; sc=(sc+sc.T)/2
    np.fill_diagonal(fc,0); np.fill_diagonal(sc,0)
    fc_list.append(fc); sc_list.append(sc); y_list.append(float(label_map[sub])); kept.append(sub)

if not kept: raise RuntimeError("No subjects had both FC and SC")
FC_all=np.stack(fc_list).astype("float32"); SC_all=np.stack(sc_list).astype("float32"); label_all=np.asarray(y_list,dtype="float32")
Path("inputs/dataset_FC").mkdir(parents=True,exist_ok=True); Path("inputs/dataset_SC").mkdir(parents=True,exist_ok=True)
np.save("inputs/dataset_FC/FC_all.npy",FC_all); np.save("inputs/dataset_SC/SC_all.npy",SC_all); np.save("inputs/dataset_SC/label_all.npy",label_all)
pd.DataFrame({"subject":kept,"label_raw":label_all}).to_csv("inputs/dataset_SC/hcp_subjects_used.csv",index=False)
Path("inputs/dataset_SC/label_metadata.json").write_text(json.dumps({"globally_standardized":False,"mean":float(label_all.mean()),"std":float(label_all.std()),"target":"PMAT24_A_CR"},indent=2))
print("FC",FC_all.shape,"SC",SC_all.shape,"Y",label_all.shape,"range",(label_all.min(),label_all.max()))
