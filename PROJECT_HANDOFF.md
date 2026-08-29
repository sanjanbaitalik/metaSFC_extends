# PROJECT_HANDOFF: MS-A-NCR Pilot for ICLR 2027

## Original Project Goal
Execute the final decisive pivot for an ICLR 2027 submission using **Modality-Selective Anisotropic Network-Constrained Ridge (MS-A-NCR)** — the first NCR redesign informed by diagnostic audits (Audit 100 & 101 v2) showing:
- Residual enrichment IS positive (matched prior beats all controls, p<0.05 Holm)
- But incremental prediction gain is negligible (median ΔPearson ≈ 0)
- Evidence: FC enrichment +0.019, SC -0.004 → prior should modulate FC shrinkage only

## Architecture & Design Decisions

### Core Method: MS-A-NCR
```
||y - Xβ||² + λ_FC β_FC^T D(q;γ) β_FC + λ_SC ||β_SC||² + λ_L β_FC^T L_q β_FC
```
- **D(q;γ)** = (ε + |q|)^{-γ}, normalized so mean(D)=1
- **L_q** = FC-only edge Laplacian from prior (top-k ROIs)
- SC receives ordinary Ridge; FC gets anisotropic + Laplacian
- Solved in dual Woodbury form (n×n kernel, n=412 subjects)

### Pilot Design (5 Models × 2 Targets × 4 Priors × 2 Liftings)
| Model | Description |
|-------|-------------|
| A0 | Standard Ridge (λ_FC = λ_SC) |
| A1 | Anisotropic FC + Ridge SC (γ>0, λ_L=0) |
| A2 | Laplacian FC + Ridge SC (γ=0, λ_L>0) |
| A3 | **Full MS-A-NCR** (γ>0, λ_L>0) |
| A4 | Modality-specific Ridge (γ=0, λ_L=0, λ_FC≠λ_SC) |

### Constraints (MUST PRESERVE)
- **DO NOT MODIFY** `src/metascfc/models/iclr_backbones/network_constrained_ridge.py` (existing NCR)
- **DO NOT MODIFY** existing benchmark outputs in `outputs/iclr/mt_ncr/`, `lown_curve/`, `tables/`
- Use existing **Qwen 3.8 27B Contrastive Priors** only
- Seed means = inferential unit; Wilcoxon + Holm correction required
- `load_connectomes()` returns `(fc, sc, y, subjects, groups)`
- `iter_nested_splits()` returns `(seed, fold, train_idx, val_idx, test_idx)`

## Implemented Phases

### Phase A: Discrete Routing Pivot ✅
- 57/57 tests passing

### Phase B: MT-NCR + Low-N + Ablation ✅
- `outputs/iclr/mt_ncr/`, `outputs/iclr/lown_curve/`, `outputs/iclr/tables/`

### Phase C: Diagnostic Audits ✅
| Audit | Status | Tests | Output | Recommendation |
|-------|--------|-------|--------|----------------|
| Audit 100 | Done | 35/35 | `outputs/iclr/prior_predictive_enrichment/` | `redesign_ncr_penalty` |
| Audit 101 v1 | Superseded | - | `outputs/iclr/conditional_prior_signal/` | `rebuild_prior` (buggy) |
| **Audit 101 v2** | **Done** | **30/30** | `outputs/iclr/conditional_prior_signal_v2/` | **`anisotropic_ncr`** |

### Phase D: MS-A-NCR Pilot ✅ (This Session)
- Core solver: `src/metascfc/models/iclr_backbones/modality_selective_anisotropic_ncr.py`
- Tests: `tests/test_msancr.py` (25/25 passing)
- Config: `configs/iclr/msancr_pilot.yaml`
- Runner: `scripts/104_run_msancr_pilot.py`
- Results: `outputs/iclr/msancr_pilot/`

## Files Created/Modified This Session

### New Files
| File | Purpose |
|------|---------|
| `src/metascfc/models/iclr_backbones/modality_selective_anisotropic_ncr.py` | MS-A-NCR solver (dual Woodbury, eig_cache) |
| `tests/test_msancr.py` | 25 unit tests (diagonal penalty, lifting, cache, solver, staged selection) |
| `configs/iclr/msancr_pilot.yaml` | Pilot config (seeds 0,1,2; 5 folds; minimal grid) |
| `scripts/104_run_msancr_pilot.py` | Pilot runner with pre-built caches, staged selection |

### Modified Files
| File | Change |
|------|--------|
| `src/metascfc/models/iclr_backbones/modality_selective_anisotropic_ncr.py` | Added `active_laplacian` to `_MSANCRCache`; fixed inactive FC block; added `eig_cache` parameter to `_solve_msancr_kernel`; fixed `_predict_msancr` inactive block |

## Important Commands

```bash
# Environment
source activate metascfc-hcp
# Python 3.11, conda env: metascfc-hcp

# Run tests (all pass)
python -m pytest tests/test_msancr.py -v
python -m pytest tests/test_conditional_prior_signal.py -v
python -m pytest tests/test_prior_predictive_enrichment.py -v

# Run pilot (4.5 hours)
python -u scripts/104_run_msancr_pilot.py --config configs/iclr/msancr_pilot.yaml

# Check outputs
cat outputs/iclr/msancr_pilot/pilot_decision.json
cat outputs/iclr/msancr_pilot/summary_metrics.csv

# Data paths (DO NOT CHANGE)
inputs/dataset_FC/FC_all.npy          # (412, 116, 116)
inputs/dataset_SC/SC_all.npy          # (412, 116, 116)
inputs/dataset_SC/label_all.npy       # Fluid Intelligence
inputs/dataset_SC/task_labels/ListSort_Unadj/label_all.npy  # Working Memory
outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv
outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv
outputs/priors/random_prior/aal116/roi_prior.csv
```

## Dependencies/Environment
```bash
conda activate metascfc-hcp
# Python 3.11
# Key packages: numpy, scipy, pandas, scikit-learn, pytorch, matplotlib, pyyaml
# GPU not required (CPU-only run)
```

## Bugs/Issues Discovered & Fixed

| Issue | Root Cause | Fix |
|-------|------------|-----|
| `_LaplacianEig` has no `active_laplacian` | `active_laplacian` lives on `EdgeLaplacian`, not eigendecomposed `_LaplacianEig` | Store `edge_lap.active_laplacian` in `_MSANCRCache` |
| Test dimension mismatch | Tests used `p=n_rois` features but solver expects `n_edges=n_rois*(n_rois-1)//2` | Rewrote tests with correct `n_edges=15` (n_rois=6) |
| Inactive FC block used D_inactive | Prior weighting should NOT apply to inactive edges (they get plain Ridge) | Changed to `(1/λ_FC) * X_inactive @ X_inactive.T` |
| Slow hyperparameter grid search | Eigendecomposition of 1105×1105 repeated for each λ_FC | Added `eig_cache` keyed by ratio `λ_L/λ_FC` (6.4× speedup) |
| Pilot process killed by OOM | Only 2GB free (run_fever.py + opencode consuming 115GB/124GB) | Ran with `setsid python -u ... > log 2>&1 &` |

## Failed Approaches & Why

| Attempt | Why It Failed |
|---------|---------------|
| `nohup python script.py > log 2>&1 &` | Conda env not preserved; output buffering hid progress |
| Background with `tee` | Pipe broke when shell exited |
| Full hyperparameter grid (4 γ × 2 lifting × 4 λ_L × 5 λ_FC × 5 λ_SC) | ~400 hours estimated |
| top_k=30 (n_active=3015) | Eigendecomposition O(n³) → 8s/solve vs 10s at top_k=10 |

## Current Working State

- **Pilot COMPLETE** (4.5 hours, 300 evaluations)
- Results at `outputs/iclr/msancr_pilot/`
- **Overall recommendation**: `refine_before_full_run`
  - Working Memory: `full_10x5_msancr` (ΔPearson=+0.017 vs A4, 2/3 seeds positive)
  - Fluid Intelligence: `ct_mac_prior_rebuild` (ΔPearson=+0.001 vs A4, not significant)
- Process: PID completed, `COMPLETE` marker file written

## Test Status

| Test File | Status |
|-----------|--------|
| `tests/test_msancr.py` | **25/25 PASS** |
| `tests/test_conditional_prior_signal.py` | **30/30 PASS** |
| `tests/test_prior_predictive_enrichment.py` | **35/35 PASS** |
| `tests/test_network_constrained_ridge.py` | **9/9 PASS** |
| Other tests | 9 CUDA OOM failures (pre-existing, unrelated) |

## Unresolved Problems

1. **Fluid Intelligence not benefiting** from MS-A-NCR (+0.001 vs A4). May need:
   - Different lifting rule (try `mean`)
   - Different top_k (10 → 15, 20)
   - Different λ_L grid
   - Cross-task prior mixing (current priors have 0.794 correlation)

2. **Only 3 seeds × 5 folds** — full run needs 10 seeds × 5 folds

3. **Random prior beats matched for A1 (Fluid)** — anisotropic penalty alone harms performance

## Exact Next Steps

### Option A: Targeted Working Memory Run (Recommended)
```bash
# Modify config for WM only, 10 seeds, expanded grid
# configs/iclr/msancr_wm_full.yaml:
#   seeds: [0,1,2,3,4,5,6,7,8,9]
#   targets: working_memory only
#   gamma_grid: [0.0, 0.5, 1.0, 2.0]
#   lambda_laplacian_grid: [0.0, 0.5, 1.0, 2.0, 5.0]
#   lambda_fc_grid: [0.01, 0.1, 1.0, 10.0, 100.0]
#   lambda_sc_grid: [0.01, 0.1, 1.0, 10.0, 100.0]
#   lifting_rules: [prod, mean]
#   pilot_prior_types: [matched, random, unrelated, shuffled]
python -u scripts/104_run_msancr_pilot.py --config configs/iclr/msancr_wm_full.yaml
```

### Option B: Investigate Fluid Intelligence Failure
```bash
# Add diagnostics to pilot script:
# - Compute per-seed, per-fold ΔPearson
# - Check if A3 overfits (val vs test gap)
# - Test gamma sweep with fixed λ_L
# - Try SC-weighted Laplacian (couple_modalities=True)
```

### Option C: Prior Rebuild (per Audit 101 v2)
If both tasks recommend `ct_mac_prior_rebuild`:
- Implement CT MAC prior construction from LLM
- Re-run Audit 100/101 with new priors
- Re-run MS-A-NCR pilot

## Assumptions Next Agent Must Preserve

1. **Never modify** `network_constrained_ridge.py` or Phase B outputs
2. **Always use** `load_connectomes()` and `iter_nested_splits()` from `benchmark_utils.py`
3. **Seed means are inferential unit** — no subject-level statistics
4. **Wilcoxon + Holm** for significance testing
5. **Dual Woodbury form** — never solve in primal (p=13340 ≫ n=412)
6. **StandardScaler on train only** — no leakage
7. **Nested CV** — inner val for hyperparameters, outer test for evaluation
8. **Qwen 3.8 27B Contrastive Priors** are fixed input
9. **AAL116 atlas** (116 ROIs, 6670 edges/modality)

## Repository State Summary
```
iclr/
├── configs/iclr/
│   ├── msancr_pilot.yaml          # Pilot config (minimal grid, 3 seeds)
│   ├── conditional_prior_signal_v2.yaml
│   └── prior_predictive_enrichment.yaml
├── scripts/
│   ├── 100_audit_prior_predictive_enrichment.py
│   ├── 101_audit_conditional_prior_signal.py
│   └── 104_run_msancr_pilot.py    # MS-A-NCR pilot runner
├── src/metascfc/
│   ├── models/iclr_backbones/
│   │   ├── network_constrained_ridge.py      # FROZEN
│   │   ├── modality_selective_anisotropic_ncr.py  # MS-A-NCR solver
│   │   └── meta_gat.py, llm_gated_transformer.py
│   ├── diagnostics/
│   │   ├── conditional_prior_signal.py       # Audit 101 v2
│   │   ├── prior_predictive_enrichment.py    # Audit 100
│   │   └── generalized_ridge.py              # Diagonal penalty helper
│   └── benchmark_utils.py                    # load_connectomes, iter_nested_splits
├── tests/
│   ├── test_msancr.py                  # 25/25 pass
│   ├── test_conditional_prior_signal.py # 30/30 pass
│   ├── test_prior_predictive_enrichment.py # 35/35 pass
│   └── test_network_constrained_ridge.py # 9/9 pass
└── outputs/iclr/
    ├── msancr_pilot/                    # PILOT COMPLETE
    ├── conditional_prior_signal_v2/     # Audit 101 v2
    ├── prior_predictive_enrichment/     # Audit 100
    ├── mt_ncr/                          # Phase B
    ├── lown_curve/                      # Phase B
    └── tables/                          # Phase B LaTeX
```

---
**Handoff Date**: 2026-08-29
**Last Pilot Run**: 2026-08-29 01:25 UTC (4.5 hours)
**Config Hash**: 18648422b187352a