from pathlib import Path
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker

ATLAS = "inputs/atlases/AAL116.nii.gz"
LABELS = "data/hcp/processed/labels.csv"
RAW_ROOT = Path("data/hcp/raw")
OUT_DIR = Path("data/hcp/processed/fc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

runs = [
    "rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz",
    "rfMRI_REST1_RL/rfMRI_REST1_RL_hp2000_clean.nii.gz",
]

df = pd.read_csv(LABELS)
subjects = df["subject"].astype(str).tolist()

masker = NiftiLabelsMasker(
    labels_img=ATLAS,
    standardize=True,
    detrend=True,
    t_r=0.72,
    verbose=1,
)

for sub in subjects:
    mats = []

    for run in runs:
        img_path = RAW_ROOT / sub / run

        if not img_path.exists():
            print(f"[WARN] missing fMRI: {img_path}")
            continue

        ts = masker.fit_transform(str(img_path))

        if ts.shape[1] != 116:
            print(f"[SKIP] wrong ROI count for {sub}, {run}: {ts.shape}")
            continue

        fc = np.corrcoef(ts.T)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)

        fc = np.clip(fc, -0.999999, 0.999999)
        fc = np.arctanh(fc)
        np.fill_diagonal(fc, 0.0)

        mats.append(fc.astype("float32"))

    if not mats:
        print(f"[SKIP] no usable FC for {sub}")
        continue

    fc_mean = np.mean(mats, axis=0).astype("float32")
    fc_mean = (fc_mean + fc_mean.T) / 2
    np.fill_diagonal(fc_mean, 0.0)

    np.save(OUT_DIR / f"{sub}_fc.npy", fc_mean)
    print("Saved FC:", sub, fc_mean.shape)