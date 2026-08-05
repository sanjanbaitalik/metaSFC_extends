# Method 1 — Network-Constrained Prior-Laplacian Ridge (ICLR 2027)

**Status: IMPLEMENTED AND COMPLETE** (150/150 evaluations; resumable runner).

| Item | Value |
|---|---|
| Script | `scripts/40_run_network_constrained_ridge.py` |
| Config | `configs/aaai/network_constrained_ridge.yaml` |
| Core implementation | `src/metascfc/models/iclr_backbones/network_constrained_ridge.py` |
| Tests | `tests/test_network_constrained_ridge.py` |
| Output dir | `outputs/aaai/network_constrained_ridge/` |
| Method IDs | `NCR_TRUE`, `NCR_SHUFFLED`, `NCR_RANDOM` |

---

## 1. Method description (mathematical formulation)

The prior enters the predictive objective directly as a structured quadratic
penalty on the regression weights — not as a feature rescaler (which the
AAAI prior-weighted Ridge did and which standardizes away the prior's
effect):

```
min_β  ||y - Xβ||^2  +  λ1·||β||^2  +  λ2·βᵀ L_prior β
```

- `X = [FC_upper | SC_upper]`, shape (n, 13340): concatenated per-subject
  upper-triangle connectome edge features, standardized per split.
- `L_prior` is a PSD penalty matrix built from the ROI-level working-memory
  meta-analysis prior.

### 1.1 ROI prior → edge-space penalty (line-graph lift)

1. **Prior-active set S**: the top-`top_k = 30` ROIs by prior score
   (`build_prior_adjacency`; ties broken deterministically via
   `np.argpartition`).  The prior network is the complete graph on S.
2. **Line-graph lift**: two *edge features* e = (i, j), f = (k, l) are
   adjacent iff they share exactly one ROI and that shared ROI ∈ S.
   The Laplacian quadratic form on this line graph is

   ```
   βᵀ L_line β = Σ_{e ~ f} (β_e - β_f)²
   ```

   i.e. weights of edges incident to functionally coupled (prior-active)
   ROIs are smoothed together (network-constrained regression, Li & Li 2008,
   lifted to edge features).
3. **Active edge set**: 3015 edges per modality touch a prior-active ROI
   (identical for all three prior variants by construction).
4. **Edge case**: edges touching no prior-active ROI are isolated line-graph
   nodes (zero row in L) — penalized by ridge only.
5. **Optional modality coupling** (`couple_modalities=False` in the frozen
   config): couples FC/SC copies of each active edge with
   `L = I⊗L0 + [[I,-I],[-I,I]]`, PSD by construction.
6. Default normalization: symmetric (`laplacian_normalization: sym`),
   binary weighting (`laplacian_weighting: binary`).

### 1.2 Exact dual-space solver (Woodbury identity)

```
α = (I + X P⁻¹ Xᵀ)⁻¹ y,   β = P⁻¹ Xᵀ α,   ŷ_test = X_test P⁻¹ Xᵀ α,
P = s·(λ1·I + λ2·L),       s = max(1, n_features) = 13340
```

- `L0 = U diag(μ) Uᵀ` is eigen-decomposed **once per prior** (independent of
  the data fold); whitened designs w = X_a U.
- `X P⁻¹ Xᵀ = (1/s)·[ Σ_j (w_j w_jᵀ)/(λ1 + λ2·μ_j) + X_i X_iᵀ/λ1 ]`.
- Grouping by `τ = λ2/λ1`: `1/(λ1 + λ2 μ) = (1/λ1)/(1 + τ μ)`, so the n×n
  matrix is eigen-decomposed once per distinct τ and the whole λ1 grid is
  evaluated in O(n²) per candidate.
- `λ2 = 0` reproduces the plain FC+SC dual Ridge baseline exactly
  (verified by unit test).

### 1.3 Hyperparameters (as frozen in the config)

| Parameter | Value(s) |
|---|---|
| `ridge_alphas` (λ1 grid) | [0.01, 0.1, 1, 10, 100, 1000] |
| `laplacian_alphas` (λ2 grid) | [0.0, 0.5, 1, 2, 5, 10] |
| `top_k` | 30 |
| `laplacian_weighting` | binary |
| `laplacian_normalization` | sym |
| `couple_modalities` | false |
| `n_threads` | 4 |

Selection: (λ1, λ2) chosen on inner validation RMSE, winner refit on
train+val.

---

## 2. Reproducing the run

```bash
conda activate metascfc-hcp
cd <repo>

# smoke test (1 seed, 1 method)
PYTHONPATH=src python scripts/40_run_network_constrained_ridge.py --seeds 0 --methods NCR_TRUE

# full run (10 seeds x 5 folds x 3 methods, resumable)
PYTHONPATH=src python scripts/40_run_network_constrained_ridge.py
```

Deterministic closed-form solver → exact (no seed-dependent training noise
beyond fold assignment); still run with 10 seeds to stabilize the fold
partition statistics.  Prior variants share identical seeds/folds/grids.

---

## 3. Results (frozen, 2026-08)

From `outputs/aaai/network_constrained_ridge/summary.csv` (50 evaluations
per method):

| Method | Pearson (mean±std) | RMSE (mean±std) | MAE (mean±std) |
|---|---|---|---|
| NCR_TRUE | 0.364 ± 0.088 | 4.605 ± 0.242 | 3.861 ± 0.225 |
| NCR_SHUFFLED | 0.361 ± 0.087 | 4.619 ± 0.216 | 3.861 ± 0.197 |
| NCR_RANDOM | 0.364 ± 0.095 | 4.599 ± 0.216 | 3.842 ± 0.205 |

Reference baselines (same protocol): AAAI E0 GCN ≈ 0.151 (Pearson);
plain FC+SC Ridge = NCR with λ2 = 0.

**Finding**: the true prior does not improve prediction over the shuffled
and random controls (all ≈ 0.36); the prior's contribution in this linear
backbone is not predictive above chance-level priors, despite the penalty
being structurally meaningful.  See the summary discussion in the README
and the statistics produced by `scripts/15_finalize_aaai_results.py` for
significance testing.

## 4. Outputs produced

```text
outputs/aaai/network_constrained_ridge/
  split_metrics.csv, summary.csv, seed_level_metrics.csv, summary.tex
  run_metadata.json, COMPLETE
  predictions/{NCR_*}_{seedSS}_fold{FF}.csv
  saliency/{NCR_*}/{seedSS}_{foldFF}.npz
```
