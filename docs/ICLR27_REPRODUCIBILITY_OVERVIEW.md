# ICLR 2027 — Reproducibility Overview (Shared Protocol)

This document describes the *shared* experimental protocol of the ICLR 2027
methodological extensions (Methods 1-3) in the MetaSCFC project.  Every
method-specific guide references this document for anything that is common
across methods:

- [Method 1: Network-Constrained Prior-Laplacian Ridge](ICLR27_METHOD1_NETWORK_CONSTRAINED_RIDGE.md)
- [Method 2: Prior-Gated Graph Attention Network (Meta-GAT)](ICLR27_METHOD2_META_GAT.md)
- [Method 3: Two-Stage Biomarker-Guided Kernel Ridge](ICLR27_METHOD3_TWO_STAGE_KRR.md)
- [Post-Hoc Experiments: Faithfulness, Biomarkers, Statistics](ICLR27_POSTHOC_FAITHFULNESS_BIOMARKERS.md)

The protocol intentionally mirrors the AAAI-27 submission pipeline
(scripts/08_run_aaai_matrix.py) so that all ICLR numbers are directly
comparable with the AAAI baselines E0-E9.

---

## 1. Environment

| Item | Value |
|---|---|
| Host | DGX Spark (NVIDIA GB10, Grace-Blackwell unified memory, 128 GB) |
| GPU | NVIDIA GB10 (CUDA 13.0, driver 580.142) |
| Conda env | `metascfc-hcp` at `/home/iemiedc2026/miniconda3/envs/metascfc-hcp` |
| Python | 3.11.15 (conda-forge) |
| PyTorch | 2.13.0+cu130 (CUDA available: True; cuDNN 92000) |
| NumPy / SciPy / pandas | per `requirements.txt` of this repo |
| Package install | editable (`pip install -e .`), but see §2 caveat |

### 1.1 Resolving the correct package (important)

The conda env's editable install historically points at a *different*
checkout (`.../Sanjan/metascfc_new benchmark/src`), which does **not**
contain the `iclr_backbones` package.  All ICLR runs in this repository
must therefore launch with:

```bash
PYTHONPATH=src python scripts/41_run_meta_gat.py
```

`PYTHONPATH=src` makes `import metascfc` resolve to
`<repo>/src/metascfc` (verified: `print(metascfc.__file__)`).  To make the
env permanently point at this repo instead:

```bash
pip uninstall -y metascfc && pip install -e .
```

### 1.2 Determinism settings

- `set_all_seeds(seed)` (from `metascfc.benchmark_utils`) is called once per
  (seed, fold) split before any candidate is trained; it seeds Python,
  NumPy, and PyTorch and enables
  `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Note: on CUDA, `torch.index_reduce` (used in the Meta-GAT attention
  softmax) has no deterministic implementation; PyTorch emits a
  `warn_only=True` warning and the run continues.  All results reported in
  the method guides were produced on CUDA; CPU runs may differ in the last
  floating-point digit.

---

## 2. Data (identical across all methods and the AAAI baselines)

| File | Shape | Description |
|---|---|---|
| `inputs/dataset_FC/FC_all.npy` | (412, 116, 116) | Functional connectomes (HCP Young Adult, AAL116) |
| `inputs/dataset_SC/SC_all.npy` | (412, 116, 116) | Structural connectomes (HCP Young Adult, AAL116) |
| `inputs/dataset_SC/label_all.npy` | (412,) | Raw PMAT fluid-intelligence scores (no standardization) |
| `inputs/dataset_SC/hcp_subjects_used.csv` | (412,) | Subject IDs |
| `inputs/atlases/AAL116_labels.csv` | 116 | ROI labels |
| `inputs/atlases/AAL116_coarse_modules.csv` | 116 | Coarse module assignments |

- N = 412 subjects, N_ROIS = 116, upper-triangle edge features per modality
  = 116·115/2 = 6670 (13340 FC+SC combined).
- Family-group-aware folds: `group_aware = False` in every ICLR run listed
  in this repository's outputs (the group file was not passed by the AAAI
  handoff; `iter_nested_splits` falls back to `KFold(shuffle=True)` when
  `groups is None`).
- Target: **raw** PMAT scores.  Standardization is fitted **separately on
  each (inner) training partition only** and re-applied at test time.

### 2.1 Prior files (built by `scripts/02_build_prior_maps.py` and `scripts/25_create_control_priors.py`)

| Prior | Path | Description |
|---|---|---|
| True | `outputs/priors/working_memory/aal116/roi_prior.csv` | Voxel-level working-memory meta-analysis map (z-map) mapped to AAL116, min-max normalized to [0, 1] |
| Shuffled | `outputs/priors/working_memory_shuffled/aal116/roi_prior.csv` | True prior with ROI scores permuted (anatomically shuffled control) |
| Random | `outputs/priors/random_prior/aal116/roi_prior.csv` | Uniformly random scores in [0, 1] (non-anatomical control) |

The three variants share identical seeds/folds/grids; only the prior vector
differs.  Each run re-loads and min-max normalizes the prior again
(`load_roi_prior`), so the final vector is always in [0, 1].

---

## 3. Nested cross-validation protocol (identical across methods)

Driven by `metascfc.benchmark_utils.iter_nested_splits`:

```
for seed in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:          # 10 seeds
    outer = KFold(n_splits=5, shuffle=True, random_state=seed)
    for fold, (trainval, test) in enumerate(outer):
        split_seed = seed * 1000 + fold
        train, val = make_inner_split(trainval, y, 0.15, split_seed)
```

- Outer: 5-fold KFold (shuffled, per-seed random_state).
- Inner: 15% of trainval held out for model selection (`val_fraction = 0.15`).
- Per (seed, fold): train ≈ 279-280, val = 50, test = 82-83 subjects.
- Total: 10 seeds × 5 folds = **50 splits per method**.
- All hyperparameter selection happens **only on the inner validation
  split** (by RMSE in raw target units); the selected winner is **refit on
  train+val** (scalers, graph, priors recomputed on the refit partition)
  and evaluated once on the outer test split.
- Predictions and saliency are exported per split for downstream
  biomarker-alignment and rank-stability analysis.

### 3.1 Leakage safeguards

1. Feature standardization statistics: inner training partition only.
2. Target z-statistics: training (or refit) partition only.
3. Graph topology (Meta-GAT): thresholded from group-average SC of the
   inner training partition only (never test subjects).
4. Lambda / hyperparameter grids: selected on inner validation RMSE only.
5. Per-split model checkpointing: early stopping on validation RMSE.
6. True / shuffled / random variants: identical seeds, folds, grids, and
   selection rules.

---

## 4. Evaluation metrics (identical schema)

Per split, from `metascfc.benchmark_utils.prediction_metrics`:

- **Pearson** r between raw true and predicted PMAT (0 if degenerate).
- **RMSE**, **MAE** in raw PMAT units.

Biomarker alignment per split (Method 2; Method 1 exports the same fields):

- `prior_alignment_pearson`: Pearson between node saliency and prior.
- `prior_alignment_spearman`: Spearman.
- `prior_alignment_top10_jaccard`: Jaccard of the top-10 saliency ROIs vs
  the top-10 prior ROIs.

Aggregates: `aggregate_split_metrics` (mean ± std over the 50 splits) and
`seed_level_metrics` (mean per seed, then mean ± std over the 10 seeds),
written to `summary.csv` and `seed_level_metrics.csv` respectively.

---

## 5. Output schema (per method output dir)

```text
outputs/aaai/<method>/
  split_metrics.csv       one row per (method, seed, fold) evaluation
  summary.csv             mean±std over splits, per method
  seed_level_metrics.csv  seed-level aggregation
  summary.tex             LaTeX table of pearson/rmse/mae
  run_metadata.json       config + seeds + device + data sizes
  predictions/<method>_seedSS_foldFF.csv
  saliency/<method>/seedSS_foldFF.npz
  COMPLETE                marker written when the full matrix finished
```

The runner is **resumable**: completed (method, seed, fold) triples are
skipped on restart; `--overwrite` clears the output directory first.

---

## 6. Compute budget (as run for this report)

| Method | Device | Splits | Wall time | Per-split |
|---|---|---|---|---|
| Method 1 (NCR ridge) | CPU (4 threads) | 150 | ~0.32 h total | ~8 s |
| Method 2 (Meta-GAT) | CUDA (GB10) | 150 | ~0.82 h total | ~19 s (GPU) |
| Method 2 (Meta-GAT) | CPU | (abandoned partial) | — | ~580 s |
| Faithfulness refits (exp 1) | CUDA (GB10) | 200 | ~0.3 h total | ~30-45 s (Meta-GAT) |

Method 2 is ~30x faster on the GB10 GPU than on CPU, which is why the GPU
run is the canonical one.  The post-hoc faithfulness run (experiment 1)
refits each split once with the recorded best hyperparameters and evaluates
24 mask conditions per split (see the post-hoc guide).
