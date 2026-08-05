from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_upper_triangle_features_shape():
    module = load_script("16_run_fast_prediction_baselines.py")
    mats = np.arange(3 * 5 * 5, dtype=np.float32).reshape(3, 5, 5)
    feat = module.upper_triangle_features(mats)
    assert feat.shape == (3, 10)


def test_mask_connectomes_both_modalities():
    module = load_script("17_run_faithfulness.py")
    fc = np.ones((2, 5, 5), dtype=np.float32)
    sc = np.ones_like(fc)
    fc_masked, sc_masked = module.mask_connectomes(fc, sc, [1, 3], mode="both")
    for arr in (fc_masked, sc_masked):
        assert np.all(arr[:, [1, 3], :] == 0)
        assert np.all(arr[:, :, [1, 3]] == 0)
        assert arr[0, 0, 0] == 1


def test_positive_degradation_direction():
    module = load_script("17_run_faithfulness.py")
    original = {"rmse": 4.0, "mae": 3.0, "pearson": 0.3}
    perturbed = {"rmse": 4.5, "mae": 3.2, "pearson": 0.1}
    assert module.positive_degradation(original, perturbed, "rmse") == 0.5
    assert np.isclose(module.positive_degradation(original, perturbed, "mae"), 0.2)
    assert np.isclose(module.positive_degradation(original, perturbed, "pearson"), 0.2)


def test_perturbed_graph_dataset_preserves_nonincident_edges():
    module = load_script("17_run_faithfulness.py")

    class Toy:
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            import torch
            return {
                "fc_x": torch.ones(4, 4),
                "sc_x": torch.ones(4, 4),
                "fc_edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
                "sc_edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
                "fc_edge_weight": torch.ones(4),
                "sc_edge_weight": torch.ones(4),
                "y": torch.tensor(1.0),
            }

    ds = module.PerturbedGraphDataset(Toy(), [1], mode="both")
    item = ds[0]
    assert np.all(item["fc_x"][1].numpy() == 0)
    assert np.all(item["fc_x"][:, 1].numpy() == 0)
    assert not np.any(item["fc_edge_index"].numpy() == 1)
    # Edge 2->3 is not incident to ROI 1 and must be preserved.
    edges = {tuple(x) for x in item["fc_edge_index"].t().numpy().tolist()}
    assert (2, 3) in edges
