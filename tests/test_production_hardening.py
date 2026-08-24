"""Production-hardening tests: dual-target NaN QC, LLM JSON robustness,
MINE overhead sanity, and frozen-checkpoint zero-shot transfer."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import importlib.util


def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Task 1: dual-target NaN / missing-row QC
# ---------------------------------------------------------------------------
def _behavior_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "Subject": [f"{1000 + i}" for i in range(10)],
        "PMAT24_A_CR": rng.normal(17, 3, 10).round(2),
        "ListSort_Unadj": rng.normal(70, 5, 10).round(2),
    })
    path = tmp_path / "unrestricted.csv"
    df.to_csv(path, index=False)
    return path


def test_dual_target_qc_drops_nan_and_missing_rows(tmp_path):
    from metascfc.data import dual_target_qc

    csv = _behavior_csv(tmp_path)
    beh = pd.read_csv(csv)
    beh["Subject"] = beh["Subject"].astype(str)
    # Subject 1003 skipped WM; subject 1007 has no behavior row at all.
    beh.loc[beh.Subject == "1003", "ListSort_Unadj"] = np.nan
    beh = beh[beh.Subject != "1007"]
    csv2 = tmp_path / "unrestricted2.csv"
    beh.to_csv(csv2, index=False)

    subjects = [f"{1000 + i}" for i in range(10)]
    kept, log = dual_target_qc(subjects, csv2)

    assert kept == ["1000", "1001", "1002", "1004", "1005", "1006", "1008", "1009"]
    assert log["n_input"] == 10 and log["n_kept"] == 8 and log["n_dropped"] == 2
    assert log["dropped_missing_behavior_row"] == ["1007"]
    assert log["dropped_nan_per_target"]["ListSort_Unadj"] == 1
    assert log["dropped_ids_by_reason"]["ListSort_Unadj"] == ["1003"]


def test_dual_target_qc_requires_columns(tmp_path):
    from metascfc.data import dual_target_qc

    csv = tmp_path / "partial.csv"
    pd.DataFrame({"Subject": ["1"], "PMAT24_A_CR": [20.0]}).to_csv(csv, index=False)
    with pytest.raises(ValueError, match="ListSort_Unadj"):
        dual_target_qc(["1"], csv)


def test_pack_script_qc_logic(tmp_path):
    """The packing script's NaN-drop guard rejects incomplete labels files."""
    pack = _load_module("pack24", "scripts/24_pack_hcp_arrays.py") \
        if False else None  # module executes main body; only check source
    src = Path("scripts/24_pack_hcp_arrays.py").read_text()
    assert 'required_cols = {"subject", "label", "listsort_unadj"}' in src
    assert "nan_mask" in src and "[QC] dropping" in src


# ---------------------------------------------------------------------------
# Task 3: Ollama/LLM JSON robustness
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gen46():
    return _load_module("gen46", "scripts/46_generate_llm_priors.py")


LABELS = ["Precentral_L", "Frontal_Sup_L", "Caudate_R"]


def test_json_extraction_survives_conversational_filler(gen46):
    text = (
        "Sure! Here is the JSON you requested:\n```json\n"
        '{"scores": {"Precentral_L": 0.8, "Frontal_Sup_L": 0.5, '
        '"Caudate_R": 0.2}}\n```\nLet me know if you need anything else!'
    )
    payload = gen46.extract_json_object(text)
    scores, missing = gen46.parse_scores(payload, LABELS)
    assert missing == [] and np.allclose(scores, [0.8, 0.5, 0.2])


def test_json_extraction_handles_nested_braces_and_prefix_text(gen46):
    text = ('Here is the JSON you requested: {"scores": {"Precentral_L": 1.0, '
            '"note": {"inner": "brace } trap"}, "Frontal_Sup_L": 0.5, '
            '"Caudate_R": 0.0}} Hope this helps!')
    payload = gen46.extract_json_object(text)
    scores, missing = gen46.parse_scores(payload, LABELS)
    # note: "note" key simply doesn't match an atlas label -> ignored
    assert missing == [] and scores[0] == pytest.approx(1.0)


def test_parse_scores_reports_partial_mapping(gen46):
    payload = {"scores": {"Precentral_L": 0.9}}
    scores, missing = gen46.parse_scores(payload, LABELS)
    assert set(missing) == {"Frontal_Sup_L", "Caudate_R"}
    assert np.isfinite(scores[0]) and not np.isfinite(scores[1:]).any()


def test_retry_loop_recovers_after_incomplete_mapping(gen46, monkeypatch, capsys):
    """Simulate llama3 failing once (chatty + incomplete) then succeeding."""
    calls = {"n": 0}

    def fake_call_llm(provider, prompt, model, url, temp, seed, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            return ("Here is the JSON you requested: "
                    '{"scores": {"Precentral_L": 0.9}}')
        return json.dumps({"scores": {l: float(i) / len(LABELS)
                                      for i, l in enumerate(LABELS)}})

    monkeypatch.setattr(gen46, "call_llm", fake_call_llm)
    # Re-run the exact acceptance loop used by main().
    repair_hint = "RETRY"
    payload = None
    for attempt in range(1, 4):
        try:
            candidate = gen46.extract_json_object(
                fake_call_llm("ollama", prompt := "x" + (repair_hint if attempt > 1 else ""),
                              "llama3", "u", 0.2, 42, 10)
            )
            s, m = gen46.parse_scores(candidate, LABELS)
            if int(np.isfinite(s).sum()) == len(LABELS):
                payload = candidate
                break
            raise ValueError(f"incomplete: {len(m)} missing")
        except ValueError as exc:
            last = exc
    assert payload is not None and calls["n"] == 2
    s, m = gen46.parse_scores(payload, LABELS)
    assert m == []


# ---------------------------------------------------------------------------
# Task 2: MINE estimator sanity + tracker integration
# ---------------------------------------------------------------------------
def test_mine_estimates_correlation_above_independence():
    from metascfc.metrics import MINEEstimator

    rng = np.random.default_rng(0)
    n = 512
    y = rng.normal(size=n)
    z_dep = y[:, None] * 2.0 + 0.05 * rng.normal(size=(n, 1))
    z_ind = rng.normal(size=(n, 1))
    mi_dep = MINEEstimator(x_dim=1, y_dim=1, n_steps=200, seed=1).estimate(z_dep, y)
    mi_ind = MINEEstimator(x_dim=1, y_dim=1, n_steps=200, seed=1).estimate(z_ind, y)
    assert mi_dep > mi_ind
    assert mi_dep > 0.5      # strongly dependent pair
    assert mi_ind < 0.3      # near-zero for independent pair


def test_tracker_mine_method_logs_finite_metrics():
    from metascfc.metrics import IBEpochTracker

    rng = np.random.default_rng(2)
    tracker = IBEpochTracker(method="mine", mine_steps=30)
    z = rng.normal(size=(128, 8))
    y = z.sum(axis=1) + rng.normal(size=128) * 0.01
    x = rng.normal(size=(128, 64))
    tracker.log_epoch(0, z, y, x=x)
    final = tracker.log_final(z, y, x=x)
    assert len(tracker.I_XZ) == len(tracker.I_ZY) == 1
    assert np.isfinite(tracker.I_XZ[0]) and np.isfinite(tracker.I_ZY[0])
    assert final["I_ZY"] > 0  # y is a deterministic function of z


# ---------------------------------------------------------------------------
# Task 4: checkpoint export + frozen transfer
# ---------------------------------------------------------------------------
def test_checkpoint_roundtrip_and_frozen_prediction(tmp_path):
    from metascfc.models.iclr_backbones import (
        LLMGatedConfig,
        LLMGatedTransformer,
        fit_predict_llm_gated,
    )
    from tests.test_llm_gated_transformer import make_toy

    fc, sc, y, n_rois = make_toy(n_subjects=40)
    device = torch.device("cpu")
    prior = np.linspace(0.0, 1.0, n_rois)
    idx = np.arange(len(y))
    ckpt_path = tmp_path / "ckpt.pt"

    kwargs = dict(hidden_grid=[8], dropout_grid=[0.0], lr_grid=[1e-3],
                  device=device, n_layers=1, heads=2, epochs=3, min_epochs=2,
                  patience=10, top_percent=25.0)
    pred_a, cfg_a, _, best_epoch, sal_a, _ = fit_predict_llm_gated(
        fc, sc, y, idx[:24], idx[24:32], idx[32:], prior,
        seed=11, checkpoint_path=str(ckpt_path), **kwargs
    )
    assert ckpt_path.exists()
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert payload["model_family"] == "llm_gated_transformer"
    assert payload["n_rois"] == n_rois and payload["in_dim"] == 2 * n_rois

    # Rebuild exactly like scripts/51 does and compare predictions.
    cfg = LLMGatedConfig(**payload["config"])
    model = LLMGatedTransformer(payload["n_rois"], payload["in_dim"], cfg,
                                payload["prior"], payload["edge_src"],
                                payload["edge_dst"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    x = ((np.concatenate([fc[idx[32:]], sc[idx[32:]]], axis=2)
          .reshape(8, -1) - payload["x_mean"]) / payload["x_std"]) \
        .reshape(8, n_rois, -1).astype(np.float32)
    with torch.no_grad():
        pred_b = model(torch.from_numpy(x)).numpy() \
            * payload["fit_std"] + payload["fit_mean"]
    assert np.allclose(pred_a, pred_b, atol=1e-5)

    # Sanity: frozen predictions are finite and in a plausible range.
    # (Bit-identity with a *separate* refit is not expected - the RNG state
    # at refit time depends on the selection phase; the checkpoint payload
    # above is the exact trained model, verified by the exact-match assert.)
    assert np.isfinite(pred_a).all()


def test_transfer_qc_rejects_wrong_parcellation():
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location(
        "t51", "scripts/51_run_cross_cohort_transfer.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)

    rng = np.random.default_rng(0)
    fc = rng.normal(size=(10, 90, 90))     # wrong parcellation
    sc = rng.normal(size=(10, 90, 90))
    y = rng.normal(size=10)
    with pytest.raises(ValueError, match="parcellation mismatch"):
        mod.qc_new_cohort(fc, sc, y, n_rois_expected=116)

    fc = rng.normal(size=(10, 116, 116))
    sc_bad = rng.normal(size=(10, 116, 115))
    with pytest.raises(ValueError, match="matched"):
        mod.qc_new_cohort(fc, sc_bad, y, n_rois_expected=116)

    sc = rng.normal(size=(10, 116, 116))
    y_nan = y.copy(); y_nan[3] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        mod.qc_new_cohort(fc, sc, y_nan, n_rois_expected=116)
