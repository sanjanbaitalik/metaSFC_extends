from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image as nili


def load_atlas(path_or_img: str | Path | nib.Nifti1Image) -> Tuple[nib.Nifti1Image, np.ndarray, np.ndarray]:
    if isinstance(path_or_img, nib.Nifti1Image):
        img = path_or_img
    else:
        img = nib.load(str(path_or_img))
    data = img.get_fdata()
    roi_indices = np.unique(data).astype(int)
    roi_indices = roi_indices[roi_indices > 0]
    return img, data, roi_indices


def load_meta_map(path_or_img: str | Path | nib.Nifti1Image) -> nib.Nifti1Image:
    if isinstance(path_or_img, nib.Nifti1Image):
        return path_or_img
    return nib.load(str(path_or_img))


def resample_meta_to_atlas(
    meta_img: nib.Nifti1Image,
    atlas_img: nib.Nifti1Image,
    interpolation: str = "continuous",
) -> nib.Nifti1Image:
    return nili.resample_to_img(
        meta_img,
        atlas_img,
        interpolation=interpolation,
    )


def get_roi_mask(atlas_data: np.ndarray, roi_idx: int) -> np.ndarray:
    return atlas_data == roi_idx


def extract_roi_values(
    meta_data: np.ndarray,
    atlas_data: np.ndarray,
    roi_idx: int,
) -> np.ndarray:
    mask = get_roi_mask(atlas_data, roi_idx)
    return meta_data[mask]


def load_labels(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    elif path.suffix in (".tsv", ".txt"):
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unrecognized label file format: {path.suffix}")


def load_roi_to_module(path: str | Path) -> pd.DataFrame:
    return load_labels(path)
