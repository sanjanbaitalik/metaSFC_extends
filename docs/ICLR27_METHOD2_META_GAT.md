# Method 2 — Prior-Gated Graph Attention Network (Meta-GAT) (ICLR 2027)

**Status: IMPLEMENTED AND COMPLETE** (150/150 evaluations, CUDA, 2026-08-05).

| Item | Value |
|---|---|
| Script | `scripts/41_run_meta_gat.py` |
| Config | `configs/aaai/meta_gat.yaml` |
| Core implementation | `src/metascfc/models/iclr_backbones/meta_gat.py` |
| Tests | `tests/` (see §7) |
| Output dir | `outputs/aaai/meta_gat/` |
| Method IDs | `M2_TRUE`, `M2_SHUFFLED`, `M2_RANDOM` |
| Run log | `outputs/aaai/meta_gat/experiment_log.txt` |

---

## 1. Method description (mathematical formulation)

The AAAI GCN diluted the meta-analysis prior through standard message
passing.  Meta-GAT injects the prior **directly into the attention logits**,
making it a hard inductive bias that routes information through
task-relevant regions:

```
e_ij = LeakyReLU(aᵀ [W h_i || W h_j]) + γ·(p_i + p_j)
α_ij = exp(e_ij) / Σ_{u ∈ N(j)} exp(e_uj)          (per attention head)
h'_j = Σ_{u ∈ N(j)} α_uj · W h_u                    (per head)
```

- `p_i` = min-max normalized meta-analysis prior score of ROI i.
- `γ` = **learnable** temperature, one scalar per layer (initialized 1.0).
- The gate term sits *outside* the LeakyReLU, in logit space, so a strong
  prior cannot be washed out by the learned attention scores.

### 1.1 Model architecture

| Component | Specification |
|---|---|
| Input features | `x_i = [FC_row_i | SC_row_i]` ∈ ℝ^232 per ROI (116×2), standardized on the inner training partition only |
| Layer 1 | `PriorGatedGATLayer(in=232, out=hidden, heads=4, concat=True)` + ELU + dropout |
| Layer 2 | `PriorGatedGATLayer(in=hidden·4, out=hidden, heads=1, concat=False)` + ELU |
| Readout | mean pooling over ROIs → `Linear(hidden→hidden)` → ELU → dropout → `Linear(hidden→1)` |
| Loss | MSE on z-scored target; predictions de-normalized with training-partition stats |
| Optimizer | Adam (lr ∈ grid, weight_decay 1e-4), global grad-norm clip 5.0 |
| Regularization | dropout ∈ {0.2, 0.5}, early stopping on validation RMSE (patience 15, min 10 epochs, max 60) |

### 1.2 Graph construction (leakage-free)

- Topology: the **structural connectome** — per split, the row-wise top-10%
  (positive) threshold of the **group average SC over the inner training
  partition only** (`build_split_graph`), symmetrized, plus self-loops for
  every ROI (so isolated ROIs still attend to themselves; the softmax is
  never over an empty neighborhood).
- The graph is recomputed at refit time on the **train+val** partition.
- Test subjects never contribute to the graph or to any scaler.

### 1.3 Attention implementation notes (numerical stability)

- Stable segment softmax: per-node maximum over incoming logits computed
  with `torch.index_reduce(..., reduce="amax")`; logits shifted before the
  exponential.
- Per-node normalization via a fixed one-hot destination indicator
  `A ∈ {0,1}^(n_edges, n_nodes)` and exact, differentiable einsum reductions
  (`beh,en->bnh` and back) — `torch.index_reduce` has no sum reduction.
- Non-finite loss raises `FloatingPointError` instead of silently NaN-ing.

### 1.4 Hyperparameter selection (as frozen in the config)

| Parameter | Value(s) |
|---|---|
| `hidden_grid` | [16, 32] |
| `dropout_grid` | [0.2, 0.5] |
| `lr_grid` | [0.001, 0.003] → 8 candidates (cartesian product) |
| `heads1 / heads2` | 4 / 1 |
| `gamma_init` | 1.0 |
| `weight_decay` | 1e-4 |
| `epochs / patience / min_epochs` | 60 / 15 / 10 |
| `grad_clip` | 5.0 |
| `top_percent_sc` | 10.0 |
| `val_fraction` | 0.15 (inner), 5-fold outer, 10 seeds |

Candidate grid is selected **only on inner validation RMSE**; the winner is
refit on train+val for its selected epoch budget and evaluated on the outer
test split.

### 1.5 Biomarker (saliency)

`gradient_node_saliency`: node-level saliency = mean over fit subjects of
`|dy/dx_i|` (gradient of the predicted score w.r.t. the node features,
aggregated over both modalities), min-max normalized to [0, 1].  This is
the nonlinear analogue of |β| in Method 1 and is directly comparable with
the prior via the alignment metrics below.

---

## 2. Reproducing the run

```bash
conda activate metascfc-hcp
cd <repo>

# smoke test (1 seed, 1 fold, 1 method)
PYTHONPATH=src python scripts/41_run_meta_gat.py --seeds 0 --methods M2_TRUE --folds 0

# full run (10 seeds x 5 folds x 3 methods, resumable; --overwrite to reset)
PYTHONPATH=src python scripts/41_run_meta_gat.py
```

GPU notes (DGX Spark / NVIDIA GB10):
- `device: auto` in the config → CUDA when `torch.cuda.is_available()`.
- The runner was fixed to move all data tensors to the training device
  (`_train_with_early_stopping`, `_train_fixed_epochs`, and the test-time
  forward), which is required for CUDA execution.
- Per-split wall time: ≈ 19 s on GPU vs ≈ 580 s on CPU (≈ 30x speedup);
  full 150-split run ≈ 0.82 h on GPU.

---

## 3. Results (frozen, CUDA, 2026-08-05)

From `outputs/aaai/meta_gat/summary.csv` (50 evaluations per method):

| Method | Pearson (mean±std) | RMSE (mean±std) | MAE (mean±std) |
|---|---|---|---|
| M2_TRUE | 0.267 ± 0.093 | 4.821 ± 0.307 | 4.054 ± 0.274 |
| M2_SHUFFLED | 0.282 ± 0.100 | 4.846 ± 0.318 | 4.067 ± 0.273 |
| M2_RANDOM | 0.295 ± 0.097 | 4.784 ± 0.269 | 4.002 ± 0.245 |

Biomarker alignment (mean over splits, per method):

| Method | align Pearson | align Spearman | top-10 Jaccard |
|---|---|---|---|
| M2_TRUE | 0.800 | 0.716 | 0.173 |
| M2_SHUFFLED | 0.743 | 0.720 | 0.235 |
| M2_RANDOM | 0.626 | 0.773 | 0.147 |

### 3.1 Statistical comparison (paired, 50 splits)

| Contrast | ΔPearson | paired-t p | Wilcoxon p | wins |
|---|---|---|---|---|
| TRUE vs SHUFFLED | −0.015 | 0.104 | 0.169 | TRUE 19/50 |
| TRUE vs RANDOM | −0.028 | **0.005** | **0.003** | TRUE 16/50 |

**Finding**: the true working-memory prior does **not** improve prediction
over the controls; against the random prior the gap is statistically
significant in the *wrong* direction.  Saliency does align with the true
prior (align Pearson 0.800), so the biomarker story holds, but the
attention-gated inductive bias does not help PMAT prediction in this
nonlinear backbone (consistent with the Method 1 finding).

### 3.2 Selected hyperparameter usage

- `best_hidden` ∈ {16: 90 splits, 32: 60}; `best_dropout` ∈ {0.5: 93, 0.2:
  57}; `best_lr` ∈ {0.003: 80, 0.001: 70}.
- Model size ≈ 16,323 (hidden 16) / 35,203 (hidden 32) parameters.

---

## 4. Outputs produced

```text
outputs/aaai/meta_gat/
  split_metrics.csv, summary.csv, seed_level_metrics.csv, summary.tex
  run_metadata.json, COMPLETE, experiment_log.txt
  predictions/{M2_*}_seedSS_foldFF.csv        (subject-level preds, raw units)
  saliency/{M2_*}/seedSS_foldFF.npz           (node_saliency, [0,1], 116,)
```

---

## 5. Differences vs the AAAI protocol (deliberate, documented)

1. Graph topology: row-wise top-10% SC threshold (AAAI used a fixed
   sparsity pipeline); chosen for scale invariance and leakage-free
   refitting.
2. Node features are raw connectome rows, not GCN-encoded embeddings.
3. All hyperparameter selection is on inner validation RMSE; the AAAI grid
   used a fixed (hidden, dropout) per experiment.
4. Everything else (seeds, folds, inner split, refit protocol, target
   standardization) is identical.

---

## 6. Validation / edge cases tested

- Self-loops guarantee non-empty neighborhoods for isolated ROIs.
- Stable softmax prevents overflow on logits with large γ·(p_i+p_j).
- `λ`-style degeneracy check: the equivalent of "prior off" is
  γ → 0, in which case the layer reduces to standard GAT.
- Resumability: the runner skips completed (method, seed, fold) triples.
- GPU/CPU parity: the same config runs on both; CUDA results are the
  canonical ones for this report (deterministic-warning caveat documented
  in the overview).

---

## 7. Reproducibility checklist mapping

| Checklist item | Where satisfied |
|---|---|
| Model hyperparameters | §1.4 (grid + frozen values in config) |
| Number of runs | 50 splits × 3 methods = 150 (§3) |
| Prediction performance | §3 (Pearson/RMSE/MAE with std) |
| Uncertainty / error bars | mean ± std over seeds/splits; seed_level_metrics.csv |
| Data processing | Overview §2 (files, shapes, scaling) |
| Train/val/test split | Overview §3 (nested CV, exact sizes) |
| Identical seeds across variants | Overview §3.1 (set_all_seeds per split) |
| Total compute + resources | Overview §6 (GPU, wall time) |
| Code/config availability | This repo; scripts 41 + config yaml + core module |
| Controls (shuffled/random prior) | §3 (M2_SHUFFLED, M2_RANDOM) |
