# MetaSCFC: Meta-analysis-Guided FC-SC Coupling

MetaSCFC investigates whether external neuroimaging meta-analysis priors can improve the biological grounding, interpretability, and stability of multimodal functional-connectivity/structural-connectivity graph learning.

The current AAAI-27 experiment uses an MS-Inter-GCN-style corresponding-ROI FC-SC coupling model on HCP Young Adult data with AAL116 parcellation and PMAT fluid-intelligence prediction.

## ICLR 2027 pivot: LLM-generated semantic priors + dual-task matrix

The next phase replaces (and complements) the Neurosynth maps with
**zero-shot LLM semantic scores** and evaluates every method on **two HCP
tasks**: fluid intelligence (`PMAT24_A_CR`) and working memory
(`ListSort_Unadj`).

### Zero-shot LLM prior generator

```bash
# Working Memory prior (+ anatomically shuffled control), local llama3 via Ollama:
python scripts/46_generate_llm_priors.py --task "Working Memory" \
    --provider ollama --model llama3 --controls

# Fluid Intelligence prior, hosted gpt-4o-mini:
OPENAI_API_KEY=... python scripts/46_generate_llm_priors.py \
    --task "Fluid Intelligence" --provider openai --model gpt-4o-mini
```

The prompt lists all 116 AAL regions and requests continuous relevance
scores in [0, 1]; the response is parsed as strict JSON, clamped, min-max
normalized, and saved **in the exact Neurosynth schema**
(`roi_index, roi_label, raw_score, prior_score`) to
`outputs/priors/llm/{task_slug}/roi_prior.csv` plus a `provenance.json`
(model, seed, prompt hash). `--dry-run` prints the prompt without calling
the API; `--fill-missing` controls handling of unreturned regions (default:
fail loudly).

### Dual-task targets

```bash
python -m metascfc.data.hcp_targets --behavior-csv <HCP_unrestricted.csv> \
    --targets fluid_intelligence working_memory wm_nback
```

`src/metascfc/data/hcp_targets.py` aligns behavioral measures to the used
cohort order and writes per-task `label_all.npy` +
`label_metadata.json` under `inputs/dataset_SC/task_labels/<TARGET>/`, so
every existing loader (`benchmark_utils.load_connectomes`, the AAAI loop,
scripts 40-47) consumes the new tasks by simply pointing `data.y_path` at
the generated file. Known targets/aliases: `fluid_intelligence`
(`PMAT24_A_CR`), `working_memory` / `listsort` (`ListSort_Unadj`),
`wm_nback` (`WM_Task_2back_Acc`).

### Method 4: LLM-Gated Cross-Modal Graph Attention Transformer

New model in `src/metascfc/models/llm_gated_transformer.py`. The attention
logits are biased directly by the LLM prior:

```
e_ij = LeakyReLU(a^T [W_f h_i | W_s h_j] ...) + lambda * (p_i + p_j)
```

with modality-specific FC/SC projections inside each head (cross-modal) and
a learnable per-layer temperature lambda; layers are wrapped in residual
transformer blocks (LayerNorm + FFN). Nested-CV entry points mirror Method 2
exactly (`fit_predict_llm_gated`, `refit_llm_gated_predictor`,
gradient node saliency), so the faithfulness and biomarker-stability
pipelines work unchanged.

Run (10 seeds x 5 folds x 3 variants per task, resumable):

```bash
python scripts/47_run_llm_gated_transformer.py --config configs/iclr/llm_wm_prior.yaml
python scripts/47_run_llm_gated_transformer.py --config configs/iclr/llm_fluid_prior.yaml
# smoke test:
python scripts/47_run_llm_gated_transformer.py \
    --config configs/iclr/llm_fluid_prior.yaml --seeds 0 --methods LLMT_TRUE --folds 0
```

Outputs land in `outputs/iclr/<experiment>/` with the identical split /
summary / prediction / saliency schema as Methods 1-3.

## ICLR 2.0: LLM Priors & Dual-Task Matrix

Final experimental phase. Core hypothesis - the **Inductive Bottleneck
Phenomenon**: a cognitive prior whose domain matches the prediction target
acts as an informative projector and surpasses linear baselines; a
mismatched prior over-constrains the hypothesis space and degrades it.

### 1. Generate the zero-shot LLM priors (both tasks)

```bash
# Working Memory prior + anatomically shuffled control (local Ollama / llama3):
python scripts/46_generate_llm_priors.py --task "Working Memory" \
    --provider ollama --model llama3 --controls

# Fluid Intelligence prior (hosted OpenAI / gpt-4o-mini):
OPENAI_API_KEY=... python scripts/46_generate_llm_priors.py \
    --task "Fluid Intelligence" --provider openai --model gpt-4o-mini --controls
```

Both write Neurosynth-schema priors to
`outputs/priors/llm/{working_memory,fluid_intelligence}/roi_prior.csv`.

### 2. Run the dual-task matrix

```bash
# Full matrix (default): {llm_gated, ncr} x {llm_wm, llm_fluid,
# random_control, no_prior} x {fluid_intelligence, working_memory},
# 10 seeds x 5 folds, resumable:
python scripts/50_run_dual_task_matrix.py

# Materialize the ListSort_Unadj labels first (once):
python -m metascfc.data.hcp_targets --behavior-csv <HCP_unrestricted.csv> \
    --targets working_memory

# Smoke test (1 seed x 1 fold):
python scripts/50_run_dual_task_matrix.py --seeds 0 --folds 0
```

The unified result table is written to
`outputs/iclr/dual_task_matrix/summary.csv` - one row per cell of the
2 x 4 x 2 design with Pearson r / RMSE / MAE (mean +/- std), the converged
Information Bottleneck metrics, and the learned bypass gate.

### 3. Information Bottleneck tracking

`src/metascfc/metrics/information_bottleneck.py` logs two Tishby-style
quantities during training (Gaussian/VIB-proxy estimator; identical settings
across all cells so comparisons are meaningful):

- `I_XZ` - **compression**: mutual information between the input graphs X
  and the latent embedding Z (the pooled penultimate layer of the
  transformer; the scalar projection x^T beta for the ridge), estimated as a
  Gaussian-channel rate bound.
- `I_ZY` - **predictive capacity**: mutual information between Z and the
  target Y, estimated by inverting a linear probe's R^2.

Per-epoch curves are stored per split (`--track-ib` in script 47; always on
in script 50). The theory being tested: a *mismatched* prior yields high
I(X;Z) but low I(Z;Y) (aggressive filtering that destroys predictive
signal); a *matched* prior optimizes the trade-off.

### 4. Adaptive prior routing (bypass gate)

The LLM-gated attention uses a learnable convex mix instead of a fixed
additive bias:

```
e_ij = (1 - alpha) * LeakyReLU(a^T [W_f h_i | W_s h_j]) + alpha * (p_i + p_j)
```

with `alpha = sigmoid(rho)` learned per layer. The final value is recorded
per split (`bypass_alpha` column): alpha -> 1 means "trust the prior"
(matched), alpha -> 0 means "ignore the prior" (mismatched).

### 5. Production run on real HCP data

```bash
# (a) Dual-target label extraction + packing with intersection QC.
#     Subjects missing EITHER PMAT24_A_CR or ListSort_Unadj are dropped,
#     with exact per-reason drop counts logged:
python scripts/21_prepare_hcp_labels.py   # writes both target columns
python scripts/24_pack_hcp_arrays.py      # packs FC/SC + both label arrays

# (b) Generate the real LLM priors (local Ollama / llama3.1):
python scripts/46_generate_llm_priors.py --task "Working Memory" \
    --provider ollama --model llama3.1 --controls
python scripts/46_generate_llm_priors.py --task "Fluid Intelligence" \
    --provider ollama --model llama3.1 --controls
# Robust to conversational filler: JSON is recovered from the first balanced
# '{' via raw_decode, and incomplete AAL116 mappings are retried (3 attempts)
# with an escalating strict-JSON repair hint.

# (c) Full dual-task matrix with checkpoints and neural IB estimation:
python scripts/50_run_dual_task_matrix.py --save-checkpoints --ib-method mine
# (--ib-method gaussian, the default, uses the deterministic VIB proxy)

# (d) Zero-shot cross-cohort transfer (e.g. HCP-Development), frozen model:
python scripts/51_run_cross_cohort_transfer.py --auto-best \
    --target working_memory \
    --fc inputs/dataset_hcpd_FC/FC_all.npy \
    --sc inputs/dataset_hcpd_SC/SC_all.npy \
    --y inputs/dataset_hcpd_SC/label_all.npy
```

Transfer enforces strict AAL116 checks: the new cohort must be exactly
[n, 116, 116] and match the checkpoint's parcellation; different cohort
sizes are fine, wrong dimensions or non-finite values are hard errors.
Results land in `outputs/iclr/cross_cohort_transfer/transfer_metrics.csv`
(Pearson r / RMSE / MAE per checkpoint) plus per-subject prediction CSVs.

## ICLR Production Run (Real AAAI Data)

One command chains the entire definitive experiment on the real HCP-YA
cohort (`data/hcp/processed`, dual-target intersection QC):

```bash
bash scripts/60_run_real_iclr_pipeline.sh                 # full 10-seed run

# Faster single-seed verification pass (same stages):
SEEDS="0" bash scripts/60_run_real_iclr_pipeline.sh
```

Stages executed (all output appended to `outputs/iclr/production_run.log`):

1. **Repack with robust QC** - `scripts/21_prepare_hcp_labels.py` +
   `scripts/24_pack_hcp_arrays.py`. Subjects missing *either*
   `PMAT24_A_CR` or `ListSort_Unadj` are dropped before packing, with exact
   per-reason drop counts logged. On the current data drop: 1191
   behavior-complete subjects -> 779 without FC/SC files -> **412 packed**
   (the imaging cohort has complete ListSort scores; nothing was lost to
   NaNs). Both provenance CSVs are written:
   `data/hcp/behavior/targets_fluid_intelligence.csv` and
   `targets_working_memory.csv` (row-aligned to
   `inputs/dataset_SC/hcp_subjects_used.csv`).
2. **Real zero-shot priors** - `scripts/46_generate_llm_priors.py` with
   Ollama (`OLLAMA_MODEL`, default `qwen2.5:32b`), three-tier JSON recovery,
   shuffled controls included. No synthetic priors, no overrides.
3. **Dual-task matrix** - `scripts/50_run_dual_task_matrix.py` with
   `--ib-method mine` (neural MINE estimator active) and
   `--save-checkpoints` for later transfer experiments.
4. **IB figure** - `scripts/61_plot_information_bottleneck.py` renders the
   Inductive-Bottleneck plane ($I(X;Z)$ compression vs $I(Z;Y)$ prediction;
   color = prior type, marker = target task, mismatched true-prior cells
   annotated) to `outputs/iclr/figures/ib_tradeoff.{png,pdf,tex}` (TikZ
   snippet ready for the paper).

Config note: `configs/iclr/llm_wm_prior.yaml` /
`llm_fluid_prior.yaml` consume the packed arrays
(`inputs/dataset_FC|SC`) produced by stage 1 and the aligned label files it
writes (`task_labels/ListSort_Unadj/label_all.npy` for WM,
`label_all.npy` for fluid) - the loaders therefore always see the exact
intersection-QC cohort.

## Model Card / Provenance

- **Zero-shot semantic priors**: generated with the **`qwen2.5:32b`**
  foundation model served locally via **Ollama** (`localhost:11434`), from
  prompts listing all 116 AAL region labels with strict-JSON scoring
  instructions (temperature 0.2, seed 42, three-tier JSON recovery,
  116/116 regions returned). *This is the actual generation model recorded
  in every `outputs/priors/llm/*/provenance.json` (with prompt SHA-256);
  `qwen3.8:27b` is installed on the serving machine but was **not** used
  for the saved production priors.* No fine-tuning, no task training data
  shown to the model; the priors are pure zero-shot scores.
- **Prior discriminability caveat**: the WM and Fluid zero-shot priors
  correlate at r = 0.86 (6/10 top-10 overlap) - qwen2.5:32b produces
  substantially similar "generic cognitive" maps across tasks. This bounds
  the match/mismatch contrast available to adaptive prior routing (see
  `scripts/62_run_alpha_rescue.py`, which prints this diagnostic).
- **Information Bottleneck**: MINE neural estimation (small MLP critic,
  Donsker-Varadhan bound, max-shifted log-mean-exp) with the Gaussian/VIB
  proxy as deterministic fallback; `I(X;Z)` is estimated against a fixed
  random projection of the 26,680-dim edge features (JL sketch, identical
  across all cells). Implementation:
  `src/metascfc/metrics/information_bottleneck.py`.
- **Bypass gate ($\alpha$)**: `alpha = sigmoid(rho)` per layer, initialized
  at **0.1** (data-first), trained with a sign-corrected anti-dead-zone
  reward `-1e-4 * |alpha - 0.5|` (an *additive* `+|alpha-0.5|` penalty
  would pin the gate AT 0.5 - the dead-zone it is meant to escape).
  Per-epoch trajectories are stored in the IB tracker
  (`tracker.alpha_epochs`) and the per-split JSON logs. Verified outcome
  (`outputs/iclr/alpha_rescue/alpha_rescue.json`): the gate is
  quasi-stationary at the standard budget at both 0.1 and 0.5 inits - the
  loss surface along rho is flat (the learned branch absorbs gate changes)
  and the near-duplicate priors (r = 0.86) leave little routing contrast
  to express.
- **Environment**: DGX-class workstation (Arm64), conda env
  `metascfc-hcp`, PyTorch 2.13+cu130, CUDA for the transformer backbones;
  nested CV is CPU-parallel-friendly and fully deterministic per seed
  (`set_all_seeds`). Full protocol:
  `docs/ICLR27_REPRODUCIBILITY_OVERVIEW.md`.
- **Paper tables**: regenerate with
  `python scripts/70_generate_iclr_latex_tables.py` ->
  `outputs/iclr/tables/table{1,2}_*.tex` (Table 1 reads the dual-task
  summary; Table 2 re-reads `provenance.json` so the documented model can
  never drift from the artifacts).






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