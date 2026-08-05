from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiLabelsMasker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_file", required=True)
    parser.add_argument("--atlas", default="inputs/atlases/AAL116.nii.gz")
    parser.add_argument("--labels", default="data/hcp/processed/labels.csv")
    parser.add_argument("--raw_root", default="data/hcp/raw")
    parser.add_argument("--out_dir", default="data/hcp/processed/fc")
    args = parser.parse_args()

    batch_subjects = [
        s.strip()
        for s in Path(args.batch_file).read_text().splitlines()
        if s.strip()
    ]

    labels_df = pd.read_csv(args.labels)
    valid_subjects = set(labels_df["subject"].astype(str))
    subjects = [s for s in batch_subjects if s in valid_subjects]

    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        "rfMRI_REST1_LR/rfMRI_REST1_LR_hp2000_clean.nii.gz",
        "rfMRI_REST1_RL/rfMRI_REST1_RL_hp2000_clean.nii.gz",
    ]

    masker = NiftiLabelsMasker(
        labels_img=args.atlas,
        standardize=True,
        detrend=True,
        t_r=0.72,
        verbose=1,
    )

    for sub in subjects:
        out_path = out_dir / f"{sub}_fc.npy"

        if out_path.exists():
            print(f"[SKIP] FC already exists: {sub}")
            continue

        mats = []

        for run in runs:
            img_path = raw_root / sub / run

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

        np.save(out_path, fc_mean)
        print("Saved FC:", sub, fc_mean.shape)


if __name__ == "__main__":
    main()