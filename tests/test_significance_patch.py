from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_biomarker_holm_is_monotone():
    module = load_module("biomarker", "scripts/35_build_biomarker_significance_table.py")
    raw = np.array([0.001, 0.02, 0.04, 0.8])
    adjusted = module.holm_adjust(raw)
    assert np.all(adjusted >= raw)
    assert np.all(adjusted <= 1.0)


def test_prediction_table_places_reference_last_in_linear_family():
    module = load_module("prediction", "scripts/29_build_prediction_benchmark_table.py")
    frame = pd.DataFrame({
        "ID": ["B0", "B1", "B2", "B3", "PW_TRUE", "MGCN", "METASFC"],
        "Method": ["B0", "B1", "B2", "B3", "PW", "M", "Meta"],
    })
    ordered = module.order_main_table(frame, "B3")
    ids = ordered.ID.tolist()
    assert ids[:5] == ["B0", "B1", "B2", "PW_TRUE", "B3"]
    assert ids[-1] == "METASFC"


def test_prediction_star_and_bold_formatting():
    module = load_module("prediction_format", "scripts/29_build_prediction_benchmark_table.py")
    frame = pd.DataFrame({
        "ID": ["B1", "B3"],
        "Method": ["FC Ridge", "FC+SC Ridge"],
        "Pearson Mean": [0.2, 0.37],
        "Pearson Std": [0.1, 0.08],
        "RMSE Mean": [4.8, 4.59],
        "RMSE Std": [0.3, 0.2],
        "MAE Mean": [4.0, 3.84],
        "MAE Std": [0.2, 0.2],
    })
    stats = pd.DataFrame({
        "method_id": ["B1", "B1", "B1"],
        "metric": ["pearson", "rmse", "mae"],
        "significant_holm_005": [True, True, True],
    })
    display = module.bold_best_metric_values(frame, stats, "B3")
    assert "^{*}" in display.loc[0, "Pearson $\\uparrow$"]
    assert "\\mathbf" in display.loc[1, "Pearson $\\uparrow$"]
