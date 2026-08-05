from pathlib import Path
import shutil

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import fetch_atlas_aal


def main():
    out_dir = Path("inputs/atlases")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use SPM12, not default 3v2, because we need the 116-region AAL atlas.
    aal = fetch_atlas_aal(version="SPM12")

    atlas_img = nib.load(aal.maps)
    atlas_data = atlas_img.get_fdata().astype(int)

    labels = list(aal.labels)
    indices = [int(x) for x in aal.indices]

    rows = []
    reindexed = np.zeros_like(atlas_data, dtype=np.int16)

    new_idx = 1
    for original_id, label in zip(indices, labels):
        if original_id == 0:
            continue

        mask = atlas_data == original_id
        if not np.any(mask):
            continue

        reindexed[mask] = new_idx
        rows.append({
            "roi_index": new_idx,
            "roi_label": label,
            "original_aal_id": original_id,
        })
        new_idx += 1

    out_img = nib.Nifti1Image(reindexed, atlas_img.affine, atlas_img.header)
    out_img.set_data_dtype(np.int16)

    atlas_out = out_dir / "AAL116.nii.gz"
    labels_out = out_dir / "AAL116_labels.csv"

    nib.save(out_img, atlas_out)
    pd.DataFrame(rows).to_csv(labels_out, index=False)

    print(f"Saved: {atlas_out}")
    print(f"Saved: {labels_out}")
    print(f"Number of ROIs: {len(rows)}")


if __name__ == "__main__":
    main()