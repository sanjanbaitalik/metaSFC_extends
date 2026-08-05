from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _coerce_to_numeric_array(arr, name: str) -> np.ndarray:
    """
    Convert trusted numpy/pickled object arrays into dense numeric arrays.

    This is needed because some FC/SC datasets are saved as object arrays or
    pickled lists of subject-level matrices.
    """
    if isinstance(arr, dict):
        raise ValueError(
            f"{name} loaded as a dict with keys {list(arr.keys())}. "
            "Please extract the correct array manually and save it as .npy."
        )

    if isinstance(arr, np.ndarray) and arr.dtype != object:
        return arr

    if isinstance(arr, np.ndarray) and arr.dtype == object:
        if arr.ndim == 0:
            arr = arr.item()
        else:
            try:
                return np.asarray(arr.tolist(), dtype=np.float32)
            except Exception:
                arr = list(arr)

    if isinstance(arr, (list, tuple)):
        try:
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            return np.stack([np.asarray(x, dtype=np.float32) for x in arr], axis=0)

    try:
        return np.asarray(arr, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"Could not convert {name} to a numeric numpy array.") from exc


def _load_numpy_flexible(path: str, name: str) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")

    try:
        arr = np.load(str(path), allow_pickle=False)
    except ValueError as exc:
        if "pickled" not in str(exc).lower():
            raise
        print(
            f"[WARN] {name} appears to contain pickled/object data. "
            f"Loading with allow_pickle=True because this is a trusted local file: {path}"
        )
        arr = np.load(str(path), allow_pickle=True)

    return _coerce_to_numeric_array(arr, name)


def load_fc_sc_arrays(
    fc_path: str,
    sc_path: str,
    y_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    fc = _load_numpy_flexible(fc_path, "FC")
    sc = _load_numpy_flexible(sc_path, "SC")
    y = _load_numpy_flexible(y_path, "labels")

    fc = np.asarray(fc, dtype=np.float32)
    sc = np.asarray(sc, dtype=np.float32)
    y = np.asarray(y)

    if y.ndim > 1 and y.shape[-1] == 1:
        y = y.squeeze(-1)

    if fc.ndim != 3:
        raise ValueError(f"FC must have shape [subjects, ROIs, ROIs], got {fc.shape}")
    if sc.ndim != 3:
        raise ValueError(f"SC must have shape [subjects, ROIs, ROIs], got {sc.shape}")
    if fc.shape != sc.shape:
        raise ValueError(f"FC and SC shapes must match, got FC={fc.shape}, SC={sc.shape}")
    if fc.shape[1] != fc.shape[2]:
        raise ValueError(f"FC matrices must be square, got {fc.shape}")
    if sc.shape[1] != sc.shape[2]:
        raise ValueError(f"SC matrices must be square, got {sc.shape}")
    if len(y) != fc.shape[0]:
        raise ValueError(f"Label count must match subject count, got y={len(y)}, subjects={fc.shape[0]}")

    if not np.isfinite(fc).all():
        raise ValueError("FC contains NaN or infinite values.")
    if not np.isfinite(sc).all():
        raise ValueError("SC contains NaN or infinite values.")
    if not np.isfinite(y.astype(np.float32)).all():
        raise ValueError("Labels contain NaN or infinite values.")

    print(f"Loaded FC shape: {fc.shape}")
    print(f"Loaded SC shape: {sc.shape}")
    print(f"Loaded label shape: {y.shape}")

    return fc, sc, y


class ConnectomeDataset(Dataset):
    def __init__(
        self,
        fc_arrays: np.ndarray,
        sc_arrays: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        self.fc = fc_arrays
        self.sc = sc_arrays
        self.y = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fc = torch.from_numpy(self.fc[idx]).float()
        sc = torch.from_numpy(self.sc[idx]).float()

        # Keep labels as float here. Classification loss will cast to long later.
        y = torch.as_tensor(self.y[idx]).float()

        item = {"fc": fc, "sc": sc, "y": y}
        if self.transform:
            item = self.transform(item)
        return item


class SyntheticConnectomeDataset(Dataset):
    def __init__(
        self,
        n_subjects: int = 20,
        n_rois: int = 116,
        task: str = "regression",
        seed: int = 42,
    ):
        rng = np.random.RandomState(seed)
        self.n_subjects = n_subjects
        self.n_rois = n_rois
        self.task = task

        fc = np.zeros((n_subjects, n_rois, n_rois), dtype=np.float32)
        sc = np.zeros((n_subjects, n_rois, n_rois), dtype=np.float32)

        for i in range(n_subjects):
            f = rng.randn(n_rois, n_rois)
            f = (f + f.T) / 2
            np.fill_diagonal(f, 1.0)
            fc[i] = f

            s = rng.rand(n_rois, n_rois)
            s = (s + s.T) / 2
            s = (s > 0.3).astype(float)
            np.fill_diagonal(s, 1.0)
            sc[i] = s

        if task == "regression":
            y = rng.randn(n_subjects).astype(np.float32)
        else:
            y = rng.randint(0, 2, size=n_subjects).astype(np.int64)

        self.fc = fc
        self.sc = sc
        self.y = y

    def __len__(self) -> int:
        return self.n_subjects

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "fc": torch.from_numpy(self.fc[idx]).float(),
            "sc": torch.from_numpy(self.sc[idx]).float(),
            "y": torch.tensor(self.y[idx]).float(),
        }