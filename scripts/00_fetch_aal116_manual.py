from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

import nibabel as nib
import numpy as np
import pandas as pd


def safe_extract(tar, path: Path):
    path = path.resolve()
    for member in tar.getmembers():
        member_path = (path / member.name).resolve()
        if not str(member_path).startswith(str(path)):
            raise RuntimeError(f"Unsafe tar path detected: {member.name}")
    tar.extractall(path)


def main():
    out_dir = Path("inputs/atlases")
    tar_path = out_dir / "aal_for_SPM12.tar.gz"
    tmp_dir = out_dir / "_aal_spm12_tmp"

    if not tar_path.exists():
        raise FileNotFoundError(f"Missing file: {tar_path}")

    tmp_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        safe_extract(tar, tmp_dir)

    nii_candidates = list(tmp_dir.rglob("AAL.nii"))
    xml_candidates = list(tmp_dir.rglob("AAL.xml"))

    if not nii_candidates:
        raise FileNotFoundError("Could not find AAL.nii after extraction.")
    if not xml_candidates:
        raise FileNotFoundError("Could not find AAL.xml after extraction.")

    atlas_path = nii_candidates[0]
    xml_path = xml_candidates[0]

    atlas_img = nib.load(str(atlas_path))
    atlas_data = atlas_img.get_fdata().astype(int)

    root = ET.parse(xml_path).getroot()

    original_indices = []
    labels = []

    for label_node in root.iter("label"):
        idx = int(label_node.find("index").text)
        name = label_node.find("name").text
        original_indices.append(idx)
        labels.append(name)

    rows = []
    reindexed = np.zeros_like(atlas_data, dtype=np.int16)

    new_idx = 1
    for original_id, label in zip(original_indices, labels):
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

    nib.save(out_img, out_dir / "AAL116.nii.gz")
    pd.DataFrame(rows).to_csv(out_dir / "AAL116_labels.csv", index=False)

    print("Saved inputs/atlases/AAL116.nii.gz")
    print("Saved inputs/atlases/AAL116_labels.csv")
    print(f"Number of ROIs: {len(rows)}")

    if len(rows) != 116:
        print("WARNING: Expected 116 ROIs. Please inspect AAL116_labels.csv.")


if __name__ == "__main__":
    main()