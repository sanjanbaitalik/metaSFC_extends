# MetaSCFC: Meta-analysis-Guided FC-SC Coupling

MetaSCFC investigates whether external neuroimaging meta-analysis priors can improve the biological grounding, interpretability, and stability of multimodal functional-connectivity/structural-connectivity graph learning.

The current AAAI-27 experiment uses an MS-Inter-GCN-style corresponding-ROI FC-SC coupling model on HCP Young Adult data with AAL116 parcellation and PMAT fluid-intelligence prediction.

## Experiment matrix

- E0: baseline
- E1-E3: true/shuffled/random node priors
- E4-E6: true/shuffled/random module priors
- E7-E9: true/shuffled/random corresponding-edge priors

The AAAI runner performs repeated outer cross-validation, inner validation for model selection, training-fold-only label normalization, per-split prediction/saliency export, statistical testing, and publication table/figure generation.

## Quick start

```bash
conda create -n metascfc-aaai python=3.10 -y
conda activate metascfc-aaai
pip install -r requirements.txt
pip install -e .

python scripts/24_pack_hcp_arrays.py
python scripts/02_build_prior_maps.py --config configs/prior_aal116_working_with_modules.yaml
python scripts/25_create_control_priors.py
python scripts/14_preflight_aaai.py

# Tune lambdas before final experiments
python scripts/09_run_lambda_sweep.py --config configs/aaai/E1_node_true.yaml --prior_type node
python scripts/09_run_lambda_sweep.py --config configs/aaai/E4_module_true.yaml --prior_type module
python scripts/09_run_lambda_sweep.py --config configs/aaai/E7_edge_true.yaml --prior_type edge

# Freeze selected lambdas in E0-E9 configs, then run
python scripts/08_run_aaai_matrix.py

# Tables, statistics, figures
python scripts/15_finalize_aaai_results.py
```

## Critical evaluation safeguards

- Use raw PMAT labels in `label_all.npy`; standardization is fitted separately on each training fold.
- Use HCP family IDs and group-aware folds whenever accessible.
- True, shuffled, and random variants must use identical seeds, folds, hyperparameters, and model selection rules.
- Lambda tuning outputs are development results and must not be mixed into final E0-E9 tables.
- MS-Inter-GCN supports corresponding ROI edges only. Full cross-ROI edge claims require a validated cross-ROI model.

## Inputs

```text
inputs/dataset_FC/FC_all.npy            [N,116,116]
inputs/dataset_SC/SC_all.npy            [N,116,116]
inputs/dataset_SC/label_all.npy         [N], raw behavioral scores
inputs/dataset_SC/family_groups.npy     [N], optional but recommended
inputs/atlases/AAL116.nii.gz
inputs/atlases/AAL116_labels.csv
inputs/atlases/AAL116_coarse_modules.csv
inputs/meta_maps/working_memory_z.nii.gz
```

## Outputs

Each E0-E9 run writes:

```text
outputs/aaai/final/E*/
  run_metadata.json
  split_metrics.csv
  metrics.json
  predictions/
  saliency/
  all_node_saliency.npy
  COMPLETE
```

Final paper artifacts are generated under:

```text
outputs/aaai/tables/
outputs/aaai/statistics/
outputs/aaai/figures/
```

## Full instructions

See [AAAI27_RUNBOOK.md](AAAI27_RUNBOOK.md) for preprocessing handoff, family-aware splits, lambda tuning, E0-E9 execution, controls, tables, statistics, figures, and final checks.

### Method 3: Two-Stage Biomarker-Guided Kernel Ridge

The highly stable per-split saliency maps of the AAAI MS-Inter-GCN runs are
used not as explanations but as an *explicit feature-space projector*:

- Stage 1 (biomarker): per-split min-max normalized node saliency `c` of
  the AAAI runs (E0 no-prior / E7 true / E8 shuffled / E9 random edge-prior)
  is loaded from `outputs/aaai/final/E*/saliency/` — identical seeds/folds,
  leakage-free (saliency computed on fit subjects only).
- Stage 2 (predictor): `X_gated = X_std ⊙ [lift(c) | lift(c)]` with the
  edge lift `g_ij = c_i·c_j` (product, default) or `(c_i + c_j)/2` (sum),
  then Kernel Ridge Regression (RBF) on the gated features — the
  subject-specific prior-weighted kernel `K(s,t) = k(x_s ⊙ c, x_t ⊙ c)`.
  The (alpha, gamma) grid is selected on the inner validation split; the
  winner is refit on train+val; identical output schema to Methods 1-2.

Run (10 seeds x 5 folds x 4 methods, resumable):

```bash
python scripts/42_run_two_stage_kernel_ridge.py
# smoke test: python scripts/42_run_two_stage_kernel_ridge.py --seeds 0 --methods M3_E0 --folds 0
```

Outputs (same schema as Methods 1-2):

```text
outputs/aaai/two_stage_kernel_ridge/
  split_metrics.csv, summary.csv, seed_level_metrics.csv, summary.tex
  run_metadata.json, COMPLETE, experiment_log.txt
  predictions/{M3_*}_{split}.csv
  saliency/{M3_*}/{split}.npz   (node_saliency = stage-1 gate)
```

- Config: `configs/aaai/two_stage_kernel_ridge.yaml`
- Core: `src/metascfc/models/iclr_backbones/two_stage_kernel_ridge.py`
- Guide: `docs/ICLR27_METHOD3_TWO_STAGE_KRR.md`

## Reproducibility & implementation guides (ICLR 2027)

Extremely detailed, checklist-mapped implementation guides live in
[`docs/`](docs/):

- [`docs/ICLR27_REPRODUCIBILITY_OVERVIEW.md`](docs/ICLR27_REPRODUCIBILITY_OVERVIEW.md) — shared protocol: environment (DGX Spark, conda `metascfc-hcp`, PyTorch 2.13+cu130), data, nested-CV (10 seeds × 5 folds, 15% inner validation), evaluation metrics, output schema, compute budget.
- [`docs/ICLR27_METHOD1_NETWORK_CONSTRAINED_RIDGE.md`](docs/ICLR27_METHOD1_NETWORK_CONSTRAINED_RIDGE.md) — Method 1 (complete).
- [`docs/ICLR27_METHOD2_META_GAT.md`](docs/ICLR27_METHOD2_META_GAT.md) — Method 2 (complete, CUDA run).
- [`docs/ICLR27_METHOD3_TWO_STAGE_KRR.md`](docs/ICLR27_METHOD3_TWO_STAGE_KRR.md) — Method 3 (complete).

Run logs: `outputs/aaai/network_constrained_ridge/experiment_log.txt`,
`outputs/aaai/meta_gat/experiment_log.txt`.

## ICLR 2027 Methodological Extensions

### Method 1: Network-Constrained Prior-Laplacian Ridge

Instead of rescaling features with the prior (AAAI prior-weighted Ridge), the
prior enters the objective directly as a quadratic penalty on the predictive
weights:

```
min_β  ||y - Xβ||^2  +  λ1·||β||^2  +  λ2·βᵀ L_prior β
```

where X = [FC_upper | SC_upper] (13340 edge features) and L_prior is a PSD
matrix built from the ROI-level working-memory meta-analysis prior. The top-30
ROIs form a prior network; the penalty is lifted to edge space via the line
graph (two edges are adjacent when they share a prior-active ROI), smoothing
the weights of edges incident to functionally coupled regions
(3015 active edges per modality). `L = I⊗L0 + [[I,-I],[-I,I]]` optionally
couples the FC and SC copies of each active edge (PSD by construction);
the Laplacian is symmetric-normalized by default.

Solved exactly in the n_subjects dual space via the Woodbury identity:

```
α = (I + X P⁻¹ Xᵀ)⁻¹ y,   β = P⁻¹ Xᵀ α,   ŷ_test = X_test P⁻¹ Xᵀ α,
P = s·(λ1·I + λ2·L),      s = max(1, n_features)
```

Candidates (λ1, λ2) are grouped by τ = λ2/λ1 so each distinct τ needs a
single n×n eigensolve; the whole λ1 grid is then evaluated in O(n²) per
candidate. λ2 = 0 reproduces the plain FC+SC dual Ridge baseline exactly.

Run (10 seeds x 5 folds x 3 methods, resumable):

```bash
python scripts/40_run_network_constrained_ridge.py
# smoke test:  python scripts/40_run_network_constrained_ridge.py --seeds 0 --methods NCR_TRUE
```

Outputs (identical protocol to the AAAI baselines: inner-validation selection,
refit on train+val, per-split prediction/saliency export):

```text
outputs/aaai/network_constrained_ridge/
  split_metrics.csv, summary.csv, seed_level_metrics.csv, summary.tex
  run_metadata.json, COMPLETE
  predictions/{method}_{split}.csv
  saliency/{method}/{split}.npz
```

- Config: `configs/aaai/network_constrained_ridge.yaml`
- Core: `src/metascfc/models/iclr_backbones/network_constrained_ridge.py`
- Tests: `tests/test_network_constrained_ridge.py` (closed-form primal equivalence, λ2 = 0 degeneration)