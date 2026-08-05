# Method 3 — Two-Stage Biomarker-Guided Kernel Ridge (ICLR 2027)

**Status: IMPLEMENTED AND COMPLETE** (200/200 evaluations, 2026-08-05).

| Item | Value |
|---|---|
| Script | `scripts/42_run_two_stage_kernel_ridge.py` |
| Config | `configs/aaai/two_stage_kernel_ridge.yaml` |
| Core implementation | `src/metascfc/models/iclr_backbones/two_stage_kernel_ridge.py` |
| Output dir | `outputs/aaai/two_stage_kernel_ridge/` |
| Method IDs | `M3_E0`, `M3_TRUE`, `M3_SHUFFLED`, `M3_RANDOM` |
| Run log | `outputs/aaai/two_stage_kernel_ridge/experiment_log.txt` |

---

## 1. Method description

The highly stable MetaSFC saliency maps are used not as post-hoc
explanations but as an **explicit feature-space projector** for a secondary
strong predictor:

### Stage 1 — biomarker (saliency as projector)

Per (seed, fold) split, the min-max normalized node saliency
`c ∈ [0, 1]^116` of an *already trained* AAAI MS-Inter-GCN run is loaded
from its exported per-split artifact
(`saliency/seed{ss}_fold{ff}.npz` → `node_saliency`):

| Method ID | Stage-1 biomarker source | Prior used by stage-1 model |
|---|---|---|
| M3_E0 | `outputs/aaai/final/E0_baseline/saliency` | none (no-prior GCN) |
| M3_TRUE | `outputs/aaai/final/E7_edge_true/saliency` | true working-memory edge prior |
| M3_SHUFFLED | `outputs/aaai/final/E8_edge_shuffled/saliency` | shuffled control |
| M3_RANDOM | `outputs/aaai/final/E9_edge_random/saliency` | random control |

The E* saliency was computed on fit subjects only (AAAI protocol), so the
projector is leakage-free by construction, and the E* runs share the exact
same seeds/folds as every ICLR method (no retraining, no new stochasticity).

### Stage 2 — biomarker-constrained kernel regression

1. `X = [FC_upper | SC_upper]` (13340 edge features), standardized on the
   inner training partition only.
2. The node saliency is lifted to the edge space of the connectome upper
   triangle and broadcast over both the FC and SC blocks:

   ```
   gate_mode "product" (default):  g_ij = c_i · c_j
   gate_mode "sum":                g_ij = (c_i + c_j) / 2
   X_gated = X_std ⊙ [g | g]
   ```

   The product lift requires *both* endpoints of an edge to be salient
   (bilinear coupling); the sum lift is the arithmetic analogue.
3. Kernel Ridge Regression with an RBF kernel is trained on the gated
   features — this is exactly the subject-specific prior-weighted kernel
   `K(s, t) = k(x_s ⊙ c, x_t ⊙ c)`.

### 1.1 Hyperparameters (as frozen in the config)

| Parameter | Value(s) |
|---|---|
| `alpha_grid` (ridge) | [0.01, 0.1, 1, 10, 100] |
| `gamma_grid` (RBF width, sklearn convention) | [1e-5, 1e-4, 1e-3, 1e-2] → 20 candidates |
| `gate_mode` | product |
| `kernel` | rbf |
| `val_fraction` | 0.15 (inner), 5-fold outer, 10 seeds |

Selection: (alpha, gamma) on inner validation RMSE (raw target units);
winner refit on train+val (scaler recomputed on the refit partition; the
gate `c` stays fixed per split); test prediction in raw units.

### 1.2 Biomarker alignment (per split)

`prior_alignment_*` compares the **gate saliency** (the stage-1 biomarker
actually used) with the **true working-memory prior** (`alignment_prior`),
identically for all M3 methods — so the alignment values are directly
comparable (e.g., M3_E0's alignment measures how much the *no-prior* GCN's
biomarker already matches the WM prior).

### 1.3 Edge cases

- Degenerate (all-equal) saliency → all-ones gate (no gating), KRR still
  runs.
- Upper-triangle masking is exact (`np.triu_indices(k=1)`), same edge order
  as Method 1.
- The n×n Gram matrix (≤ 412) is solved exactly; sklearn/LAPACK only (no
  GPU kernels needed) — the runner still records the chosen device in
  `run_metadata.json` for consistency with the other methods.

---

## 2. Reproducing the run

Prerequisite: AAAI E0/E7/E8/E9 outputs with per-split saliency must exist
(they do in this repo: `outputs/aaai/final/E*_*/saliency/`).

```bash
conda activate metascfc-hcp
cd <repo>

# smoke test (1 seed, 1 fold, 1 method)
PYTHONPATH=src python scripts/42_run_two_stage_kernel_ridge.py --seeds 0 --methods M3_E0 --folds 0

# full run (10 seeds x 5 folds x 4 methods, resumable; --overwrite to reset)
PYTHONPATH=src python scripts/42_run_two_stage_kernel_ridge.py
```

Full run: 200 evaluations in ~2-4 min (per-split ≈ 1 s, numpy/sklearn).

---

## 3. Results (frozen, 2026-08-05)

From `outputs/aaai/two_stage_kernel_ridge/summary.csv` (50 evaluations per
method):

| Method | Pearson (mean±std) | RMSE (mean±std) | MAE (mean±std) |
|---|---|---|---|
| M3_E0 | 0.354 ± 0.097 | 4.619 ± 0.229 | 3.858 ± 0.196 |
| M3_TRUE | 0.282 ± 0.111 | 4.754 ± 0.296 | 4.009 ± 0.252 |
| M3_SHUFFLED | 0.356 ± 0.097 | 4.622 ± 0.226 | 3.857 ± 0.199 |
| M3_RANDOM | 0.344 ± 0.090 | 4.642 ± 0.251 | 3.894 ± 0.221 |

Biomarker alignment with the true WM prior (mean over splits):

| Method | align Pearson | align Spearman | top-10 Jaccard |
|---|---|---|---|
| M3_E0 | −0.390 | −0.393 | 0.016 |
| M3_TRUE | 0.727 | 0.678 | 0.419 |
| M3_SHUFFLED | −0.433 | −0.429 | 0.000 |
| M3_RANDOM | −0.082 | −0.080 | 0.016 |

### 3.1 Statistical comparison (paired, n = 50 splits)

| Contrast | ΔPearson | Wilcoxon p | wins |
|---|---|---|---|
| TRUE vs E0 | −0.073 | **< 0.001** | 12/50 |
| TRUE vs SHUFFLED | −0.074 | **< 0.001** | 11/50 |
| TRUE vs RANDOM | −0.062 | **< 0.001** | 12/50 |
| E0 vs SHUFFLED | −0.002 | 0.70 (n.s.) | 29/50 |
| E0 vs RANDOM | +0.010 | 0.17 (n.s.) | 28/50 |
| SHUFFLED vs RANDOM | +0.012 | 0.27 (n.s.) | 26/50 |

**Finding**: gating with the true-prior biomarker **significantly hurts**
PMAT prediction relative to the no-prior (E0), shuffled, and random
biomarkers (all p < 0.001).  The no-prior / shuffled / random gates are all
equivalent (≈ 0.35, all n.s.), and M3_E0 (0.354) matches the plain NCR
ridge baseline (0.364) — i.e., stage-2 KRR on the no-prior biomarker is a
strong predictor, but the working-memory prior is the *wrong* relevance
map for PMAT prediction.  Note M3_E0's biomarker is *anti*-aligned with the
WM prior (align Pearson −0.39): the no-prior GCN's own saliency selects
regions that differ from the WM prior, and those regions carry the
predictive signal.

---

## 4. Outputs produced

```text
outputs/aaai/two_stage_kernel_ridge/
  split_metrics.csv, summary.csv, seed_level_metrics.csv, summary.tex
  run_metadata.json, COMPLETE, experiment_log.txt
  predictions/{M3_*}_seedSS_foldFF.csv        (subject-level preds, raw units)
  saliency/{M3_*}/seedSS_foldFF.npz           (node_saliency = stage-1 gate, 116,)
```

---

## 5. Reproducibility checklist mapping

| Checklist item | Where satisfied |
|---|---|
| Model hyperparameters | §1.1 (grids + frozen config) |
| Number of runs | 50 splits × 4 methods = 200 (§3) |
| Prediction performance | §3 (Pearson/RMSE/MAE ± std) |
| Uncertainty / error bars | mean ± std; seed_level_metrics.csv |
| Data processing | Reproducibility Overview §2 |
| Train/val/test split | Overview §3 (identical nested CV) |
| Stage-1 source provenance | §1 (E* dirs, same splits, leakage-free) |
| Identical seeds across variants | Overview §3.1 |
| Total compute + resources | Overview §6 (~2-4 min, CPU/GPU agnostic) |
| Code/config availability | scripts/42 + config yaml + core module |
| Controls (no-prior/shuffled/random biomarker) | §3 (M3_E0/SHUFFLED/RANDOM) |
