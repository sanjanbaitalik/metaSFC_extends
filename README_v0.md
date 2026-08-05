# MetaSCFC: Meta-analysis-Guided FC-SC Coupling

MetaSCFC investigates whether external neuroimaging meta-analysis priors can improve the biological grounding, interpretability, and stability of multimodal functional-connectivity/structural-connectivity graph learning.

The current AAAI-27 experiment uses an MS-Inter-GCN-style corresponding-ROI FC-SC coupling model on HCP Young Adult data with AAL116 parcellation and PMAT fluid-intelligence prediction.

## Note Regarding the Dataset

The complete code and processed dataset folder is approximately beyond the maximum file upload size limit. Because the supplementary-material submission system permits a maximum upload size of 50 MB, the full dataset could not be included in this archive.

To remain within the upload limit, we provide:

* the complete implementation and evaluation code;
* all configuration files needed to run the experiments;
* representative samples illustrating the format and structure of the data;
* data-loading and preprocessing scripts; and
* a data manifest describing the complete dataset.

The files in `sample_data/` are intended to demonstrate the expected input format and allow reviewers to inspect and test the execution pipeline. They are not the complete dataset and should not be used to reproduce the numerical results reported in the paper.

Subject to the conference’s anonymity and supplementary-material policies, the complete dataset and any remaining large files will be made publicly available upon publication.

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

## Repository

GitHub repository: `https://github.com/PLACEHOLDER/MetaSCFC`

Replace the placeholder before release. Do not commit HCP raw/restricted data or AWS credentials.

---

## Final AAAI additions: fast baselines, corrected statistics, and faithfulness

The following scripts were added for the final submission-stage evaluation.

### 1. Fast prediction baselines

Runs four leakage-free baselines using the same 10 seeds, 5 outer folds, and
inner-validation split as E0--E10:

- `B0`: training-fold mean predictor
- `B1`: FC upper-triangle Ridge regression
- `B2`: SC upper-triangle Ridge regression
- `B3`: concatenated FC+SC Ridge regression

```bash
python scripts/16_run_fast_prediction_baselines.py \
  --config configs/aaai/fast_prediction_baselines.yaml
```

Outputs:

```text
outputs/aaai/prediction_baselines/
  prediction_baselines_split_metrics.csv
  prediction_baselines_summary.csv
  prediction_baselines_summary.tex
  predictions/
  COMPLETE
```

The Ridge penalty is selected on the inner validation split only. Feature and
target scalers are fitted only on the corresponding training partition.

### 2. Seed-level paper-ready statistical tests

The corrected statistical analysis averages the five folds within each seed
before hypothesis testing. This yields 10 paired seed-level observations per
experiment rather than treating 50 dependent seed-fold rows as independent.

```bash
python scripts/11_statistical_tests.py
```

Outputs:

```text
outputs/aaai/statistics/
  seed_level_method_metrics.csv
  seed_level_paired_statistical_tests.csv
  seed_level_paired_statistical_tests.tex
```

The output includes paired t-tests, Wilcoxon signed-rank tests, bootstrap 95%
confidence intervals, Cohen's dz, and Holm corrections.

### 3. Perturbation-based explanation faithfulness

The default run retrains E0, E1, and E10 for three seeds and five folds. Salient
ROIs are selected from the inner validation set, and only the held-out outer
test set is perturbed. All FC and SC connections incident to selected ROIs are
removed.

```bash
python scripts/17_run_faithfulness.py \
  --configs \
    configs/aaai/E0_baseline.yaml \
    configs/aaai/E1_node_true.yaml \
    configs/aaai/E10_node_unrelated_visual.yaml \
  --seeds 0 1 2 \
  --topk 10 \
  --random_repeats 20 \
  --mask_mode both
```

For a wider supplementary run, add E4 and E7 or increase the seed list:

```bash
python scripts/17_run_faithfulness.py \
  --configs \
    configs/aaai/E0_baseline.yaml \
    configs/aaai/E1_node_true.yaml \
    configs/aaai/E4_module_true.yaml \
    configs/aaai/E7_edge_true.yaml \
    configs/aaai/E10_node_unrelated_visual.yaml \
  --seeds 0 1 2 3 4
```

A positive `gap_rmse_top_vs_random` means that masking the learned top-k ROIs
harms prediction more than masking random ROIs. A positive
`gap_rmse_top_vs_bottom` means top-k masking is more damaging than bottom-k
masking. These are the primary faithfulness quantities.

Outputs:

```text
outputs/aaai/faithfulness/
  faithfulness_all_split_metrics.csv
  faithfulness_summary.csv
  faithfulness_seed_level_metrics.csv
  faithfulness_seed_level_tests.csv
  faithfulness_seed_level_tests.tex
  per_experiment/
```

### 4. Final submission tables

After running the baselines, corrected statistics, and faithfulness analysis:

```bash
python scripts/18_build_submission_tables.py
```

Outputs:

```text
outputs/aaai/submission_tables/
  paper_table_prediction_all.csv
  paper_table_prediction_all.tex
  paper_table_seed_level_statistics.csv
  paper_table_seed_level_statistics.tex
  paper_table_faithfulness.csv
  paper_table_faithfulness.tex
```

### Recommended one-day execution order

```bash
python scripts/16_run_fast_prediction_baselines.py \
  --config configs/aaai/fast_prediction_baselines.yaml

python scripts/11_statistical_tests.py

python scripts/17_run_faithfulness.py \
  --configs configs/aaai/E0_baseline.yaml configs/aaai/E1_node_true.yaml configs/aaai/E10_node_unrelated_visual.yaml \
  --seeds 0 1 2 --topk 10 --random_repeats 20 --mask_mode both

python scripts/18_build_submission_tables.py
```
