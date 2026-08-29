# OpenCode Prompt — Post-Final Statistical Audit + Family-Aware HCP Robustness

## Role

You are working inside the existing `metaSFC_extends` repository after completion of the frozen Working-Memory MS-A-NCR 10-seed × 5-fold confirmatory experiment.

This is **not a model-development prompt**.

Do not redesign MS-A-NCR.
Do not change the prior.
Do not change the Working-Memory target.
Do not expand or retune any hyperparameter grid.
Do not modify the completed subject-wise predictions.
Do not overwrite any existing final outputs.

The next step has two tightly separated goals:

1. **repair/report two post-processing issues using the already-frozen final outputs, with zero model reruns;**
2. **prepare and, only if authorized HCP Family_ID data are locally available, run a family-aware robustness experiment using the exact frozen method.**

The family-aware experiment is a robustness/generalization analysis, not a new tuning opportunity.

---

# Current frozen subject-wise result

The completed subject-wise 10×5 experiment reports approximately:

```text
A4 modality-specific Ridge     Pearson = 0.2510
A2 FC-Laplacian                Pearson = 0.2495
A3 MS-A-NCR matched            Pearson = 0.2632
```

A3 matched vs A4:

```text
mean ΔPearson   = +0.01216
median ΔPearson = +0.01139
positive seeds  = 7/10

paired Wilcoxon raw p = 0.037109
Holm p across five Pearson comparisons = 0.111328

bootstrap 95% CI = [0.00445, 0.02038]
Cohen dz = 0.893
```

A3 also improves:

```text
RMSE mean improvement ≈ +0.064
MAE mean improvement  ≈ +0.073
```

and the matched prior has substantially stronger WM alignment.

Do not reinterpret or alter these numbers.

---

# Important findings from repository audit

Before implementing anything, independently verify these findings.

## Finding 1 — boundary metadata is reconstructed incorrectly

The actual final YAML contains:

```yaml
ridge_grid:
  - 0.001
  - 0.01
  - 0.1
  - 1.0
  - 10.0
  - 100.0

gamma_grid:
  - 0.1
  - 0.25
  - 0.5
  - 1.0
  - 2.0

lambda_laplacian_grid:
  - 0.03
  - 0.1
  - 0.5
  - 1.0
  - 2.0
  - 5.0
```

The actual `inner_cv_metrics.csv` should confirm that all of these values were searched.

However, `build_boundary_distribution_final()` currently reconstructs grids from selected values plus endpoints:

```python
sorted(a3.lambda_fc.unique().tolist() + [0.001, 100.0])
```

and analogous logic for gamma/lambda_L.

That can produce incorrect metadata such as duplicate endpoints and omit valid grid values that happened never to be selected.

This is a reporting bug only.

It must NOT trigger a rerun of the completed subject-wise experiment.

---

## Finding 2 — the final analysis mixes one explicitly primary comparison with four secondary/control comparisons in one Holm family

The code explicitly defines:

```text
Primary comparison:
A3 matched vs A4 modality-specific no-prior Ridge

Secondary mechanistic comparison:
A3 matched vs A2 FC-Laplacian

Prior-control comparisons:
A3 matched vs unrelated
A3 matched vs shuffled
A3 matched vs random
```

Yet the current inference function applies Holm jointly across all five comparisons for each metric.

The completed analysis must be preserved exactly as a conservative familywise sensitivity analysis.

Do NOT delete or replace the existing Holm result.

However, create a transparent reporting layer that distinguishes:

### Pre-specified primary contrast

```text
A3 matched vs A4, Pearson
```

Report its paired two-sided Wilcoxon p-value directly:

```text
p_primary = 0.037109...
```

No multiplicity correction is needed **within a family containing only this one designated primary contrast**.

### Secondary/exploratory family

Handle the remaining:

```text
A3 vs A2
A3 vs unrelated
A3 vs shuffled
A3 vs random
```

as secondary/control comparisons and apply Holm within that secondary family, separately per metric.

### Conservative sensitivity family

Retain the existing five-comparison Holm analysis exactly as:

```text
conservative_all_comparisons_holm
```

The paper-facing interpretation must show BOTH:

```text
Primary planned contrast:
significant at p = 0.037

Conservative Holm sensitivity across all five:
p_Holm = 0.111, not significant
```

Do not hide either result.

Do not rewrite history by claiming the conservative Holm analysis was never planned.

The result should be described as:

> the designated primary A3-vs-A4 contrast is significant, while significance does not survive the more conservative sensitivity correction that treats all five method/control comparisons as one family.

---

# Part A — Reporting-only patch

Create:

```text
src/metascfc/experiments/msancr_postfinal_reporting.py
scripts/108_finalize_msancr_reporting.py
tests/test_msancr_postfinal_reporting.py
```

Write NEW outputs to:

```text
outputs/iclr/msancr_final_10x5/postfinal_reporting/
```

Never overwrite the original:

```text
final_prediction_statistics.csv
final_biomarker_statistics.csv
final_hypothesis_decision.json
final_statistical_summary.json
boundary_distribution_final.json
```

---

# Part A1 — Correct grid metadata from config

Load the exact grids from:

```text
configs/iclr/msancr_final_10x5.yaml
```

Do not infer grids from selected values.

Create:

```text
corrected_boundary_distribution_final.json
```

containing:

```json
{
  "grids_source": "frozen_final_config",
  "ridge_grid": [],
  "gamma_grid": [],
  "lambda_laplacian_grid": [],
  "lifting_rules": [],
  "selected_distributions": {},
  "boundary_counts": {}
}
```

Boundary counts must compare selected A3 values against the actual config min/max.

Also include:

```text
lambda_fc lower/upper hits
lambda_sc lower/upper hits
gamma lower/upper hits
lambda_L lower/upper hits
```

No grid expansion is allowed.

---

# Part A2 — Verify actual grid execution

Read:

```text
inner_cv_metrics.csv
```

and programmatically verify that the intended candidates were actually evaluated.

Create:

```text
final_grid_execution_audit.json
```

For each model/stage, report observed candidate values.

Require:

```text
10.0 appears in actual Ridge candidate evaluations
0.001 appears
gamma {0.1,0.25,0.5,1,2} appear for A3
lambda_L {0.03,0.1,0.5,1,2,5} appear where appropriate
lifting {prod,mean} appear for A3
```

If a value is absent because a staged local-refinement stage does not logically use the full grid, distinguish that from an actual missing candidate.

Do not rerun models because of metadata-only discrepancies.

---

# Part A3 — Primary-versus-secondary statistical report

Using the already frozen 10 seed-level rows:

Create:

```text
primary_prediction_contrast.csv
secondary_prediction_statistics.csv
conservative_familywise_sensitivity.csv
```

## Primary prediction contrast

Only:

```text
A3 matched vs A4
metric = Pearson
```

Report:

```text
n_seeds
A3 mean ± SD
A4 mean ± SD
mean paired Δr
median paired Δr
positive seeds
negative seeds

two-sided paired Wilcoxon W
two-sided paired Wilcoxon p_primary

10,000-pair bootstrap 95% CI
Cohen dz
paired t-test p as secondary sensitivity
```

Do not change any frozen predictions.

---

## Secondary Pearson family

Apply Holm only across:

```text
A3 vs A2
A3 vs unrelated
A3 vs shuffled
A3 vs random
```

Produce:

```text
p_raw
p_holm_secondary
```

Do this separately for:

```text
Pearson
RMSE
MAE
```

---

## Conservative sensitivity

Copy/recompute the original five-comparison Holm family and verify exact numerical equality with the existing output.

Create a Boolean:

```text
matches_original_conservative_analysis = true
```

Fail loudly if the recomputed values differ beyond floating tolerance.

---

# Part A4 — Correct prior-specificity wording

The current data support:

```text
matched mean Pearson > unrelated
matched mean Pearson > shuffled
matched mean Pearson > random
```

But under the original five-comparison Pearson Holm family:

```text
unrelated: significant
shuffled: significant
random: not significant
```

Therefore do NOT write:

```text
"matched > all three controls, all significant after Holm"
```

for Pearson.

Create a paper-safe summary that distinguishes:

```text
directional superiority
raw significance
secondary-family Holm significance
five-comparison sensitivity Holm significance
```

---

# Part A5 — Paper-facing interpretation artifact

Create:

```text
paper_safe_interpretation.json
```

with fields:

```json
{
  "primary_prediction_contrast": {
    "supported_at_nominal_005": true,
    "p_primary": 0.0,
    "mean_delta_r": 0.0,
    "ci95": [0.0, 0.0],
    "cohens_dz": 0.0
  },
  "conservative_familywise_sensitivity": {
    "supported": false,
    "p_holm_all_five": 0.0
  },
  "recommended_wording": "",
  "prohibited_wording": []
}
```

Recommended wording should be scientifically transparent, e.g.:

```text
MS-A-NCR improved Working-Memory prediction relative to modality-specific Ridge
(Δr = ..., paired Wilcoxon p = ..., 95% CI ..., dz = ...).
This designated primary contrast did not remain significant under an additional
conservative Holm correction that grouped the primary comparison with four
secondary method/control comparisons (p_Holm = ...).
```

Do not use:

```text
"statistically significant after correction"
```

for A3 vs A4 Pearson.

---

# Part A6 — Revised LaTeX tables without changing numbers

Create NEW files:

```text
final_prediction_table_reporting_v2.tex
final_biomarker_table_reporting_v2.tex
```

Do not overwrite the old tables.

For the prediction table:

- bold best mean;
- retain Holm stars for secondary/control comparisons;
- mark A4 with a distinct symbol such as `\dagger` to indicate the designated primary A3-vs-A4 Wilcoxon p < 0.05;
- footnote clearly:
  ```text
  dagger = designated primary A3-vs-A4 paired Wilcoxon p < 0.05;
  star = significant after the stated secondary/conservative Holm procedure.
  ```

Do not visually imply that the dagger is Holm corrected.

---

# Part B — Family-aware HCP robustness

## Why this matters

The current final run explicitly has:

```text
groups_path: null
group_aware: false
```

HCP-YA contains twins and non-twin siblings.

The repository already contains:

```text
scripts/24b_prepare_hcp_family_groups.py
```

which requires an authorized HCP restricted CSV with:

```text
Subject
Family_ID
```

The family-aware experiment must keep all members of one biological family in the same train/validation/test partition.

This is the next major robustness test.

---

# Restricted-data safety — mandatory

HCP `Family_ID` is restricted information.

Do NOT:

- commit the restricted CSV;
- copy the restricted CSV into repository-tracked folders;
- print Family_ID values in logs;
- save subject-to-family mappings in output CSV/JSON;
- upload restricted family data to GitHub;
- create paper artifacts containing raw HCP IDs linked to family IDs.

Add/verify `.gitignore` coverage for at least:

```text
*restricted*.csv
family_groups.npy
family_groups*.npy
restricted_data/
private_hcp/
```

If `family_groups.npy` itself is considered restricted/derivative under the local data-use agreement, keep it local and ignored.

Only output:

```text
number of subjects
number of unique groups
max family size
split-integrity booleans
cryptographic hashes of split index sets
```

Never output actual family identifiers.

---

# Part B1 — Discover family data locally

Check, in this order:

```text
inputs/dataset_SC/family_groups.npy
```

or an explicit CLI argument:

```text
--restricted-csv /local/private/path/HCP_S1200_restricted.csv
```

If neither is present:

1. complete Part A;
2. prepare all family-aware code/config/tests;
3. do NOT fabricate groups;
4. do NOT download restricted data;
5. do NOT fall back to subject-wise splits;
6. stop with:
   ```text
   FAMILY_DATA_REQUIRED
   ```
7. print the local command the researcher should run after obtaining authorized HCP restricted access.

Do not ask for or expose restricted data in GitHub.

---

# Part B2 — Prepare aligned group vector

Reuse or harden:

```text
scripts/24b_prepare_hcp_family_groups.py
```

Requirements:

- exact subject order must match `hcp_subjects_used.csv`;
- every analysis subject must have a Family_ID;
- no duplicate subject rows;
- fail on missing Family_ID;
- save only local ignored:
  ```text
  inputs/dataset_SC/family_groups.npy
  ```
- log only aggregate counts.

Create:

```text
family_group_integrity.json
```

without raw IDs.

---

# Part B3 — IMPORTANT: fix seed-varying family-aware folds

Do NOT simply use the current:

```python
GroupKFold(n_splits=5)
```

inside the 10-seed loop.

`GroupKFold` without shuffling can produce identical outer folds for all ten seeds, which would make the ten seed summaries pseudo-replicates.

Implement a separate randomized family-preserving splitter for this robustness run.

Preferred if supported by installed sklearn:

```python
GroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=seed
)
```

ONLY if the installed sklearn version truly supports `shuffle` and `random_state`.

Otherwise implement:

```text
RandomizedBalancedGroupKFold
```

that:

1. obtains unique groups;
2. shuffles group order deterministically with `np.random.default_rng(seed)`;
3. assigns entire groups greedily to the currently smallest fold by subject count;
4. never divides a family;
5. produces five disjoint test folds covering every subject exactly once.

Do not use target labels to optimize the group allocation.

For inner 3-fold CV:

- use the same family-preserving randomized strategy;
- random state derived deterministically from:
  ```text
  seed
  outer_fold
  ```
- inner folds must contain only outer-training groups.

---

# Part B4 — Verify seed diversity

Before running models, create split manifests containing only hashes and counts.

For all ten seeds verify:

```text
outer split hash sets are not identical across all seeds
```

Require a reasonable diversity of partitions.

At minimum:

```text
number of unique 5-fold partition manifests >= 8/10
```

Prefer 10/10.

If all/most seeds are identical, stop and fix the splitter.

---

# Part B5 — Family leakage assertions

For every:

```text
seed
outer fold
inner fold
```

assert:

```text
train family set ∩ validation family set = empty
train family set ∩ test family set = empty
validation family set ∩ test family set = empty
```

Write aggregate checks only:

```text
n_outer_checks
n_inner_checks
n_failures
```

Require:

```text
n_failures = 0
```

No family IDs in output.

---

# Part B6 — Frozen model and grid

Use the exact completed subject-wise final method/config:

```text
Working Memory (ListSort_Unadj)
A4 modality-specific Ridge
A2 FC-Laplacian
A3 MS-A-NCR matched
fixed unrelated/shuffled/random swaps
```

Use exactly the frozen grids:

```yaml
ridge_grid:
  - 0.001
  - 0.01
  - 0.1
  - 1.0
  - 10.0
  - 100.0

gamma_grid:
  - 0.1
  - 0.25
  - 0.5
  - 1.0
  - 2.0

lambda_laplacian_grid:
  - 0.03
  - 0.1
  - 0.5
  - 1.0
  - 2.0
  - 5.0

lifting_rules:
  - prod
  - mean
```

No grid changes after seeing family-aware results.

No model changes.

No prior changes.

No target changes.

---

# Part B7 — Family-aware config and runner

Create:

```text
configs/iclr/msancr_family_aware_10x5.yaml
scripts/109_run_msancr_family_aware.py
```

Output:

```text
outputs/iclr/msancr_family_aware_10x5/
```

Figures:

```text
figures/iclr/msancr_family_aware_10x5/
```

Set:

```yaml
groups_path: inputs/dataset_SC/family_groups.npy
split_strategy: randomized_balanced_group_kfold
```

The family-aware runner must refuse to run if:

```text
groups_path is null
groups file missing
group alignment validation fails
```

No fallback.

---

# Part B8 — Family-aware inference

Use ten seed-level summaries after averaging five family-aware outer folds per seed.

Primary contrast:

```text
A3 matched vs A4
```

Report:

```text
mean ΔPearson
median ΔPearson
positive seeds / 10
paired Wilcoxon p_primary
bootstrap 95% CI
Cohen dz
```

For secondary/control comparisons use the same reporting structure created in Part A:

```text
primary contrast separate
secondary Holm family
conservative all-five Holm sensitivity
```

Do not move the goalposts if the family-aware result is weaker.

---

# Part B9 — Subject-wise vs family-aware robustness comparison

Create:

```text
subject_vs_family_aware_comparison.csv
```

Report:

```text
subject-wise A4 Pearson
family-aware A4 Pearson

subject-wise A3 Pearson
family-aware A3 Pearson

subject-wise Δ(A3-A4)
family-aware Δ(A3-A4)
```

Also RMSE and MAE.

Interpret separately:

### Pattern 1

```text
absolute r decreases but A3-A4 advantage survives
```

Interpretation:
family leakage inflated absolute predictability, but the prior-aware method advantage generalizes.

### Pattern 2

```text
both absolute r and A3-A4 advantage collapse
```

Interpretation:
the original predictive gain may depend partly on family-related similarity; do not claim robust generalization.

### Pattern 3

```text
A3-A4 advantage strengthens
```

Interpretation:
family-aware splitting strengthens the method-specific evidence.

Do not automatically prefer whichever result is more favorable.

---

# Part B10 — Family-aware biomarker analysis

Repeat the same seed-level:

```text
WM alignment
rank stability
top-10 Jaccard
```

for A3, A4, A2, and fixed prior swaps.

Because biomarkers are coefficient-derived from family-separated training sets, this is a valuable robustness test.

Use the same primary/secondary statistical structure.

---

# Part B11 — Expected output files

Create:

```text
outputs/iclr/msancr_family_aware_10x5/
```

with at least:

```text
family_group_integrity.json
family_split_integrity.json
family_split_diversity.json

split_metrics.csv
seed_metrics.csv
summary_metrics.csv
selected_hyperparameters.csv
inner_cv_metrics.csv

prior_swap_split_metrics.csv
prior_swap_seed_metrics.csv
prior_swap_summary.csv

family_aware_prediction_statistics.csv
family_aware_biomarker_statistics.csv

subject_vs_family_aware_comparison.csv

family_aware_hypothesis_summary.json
run_metadata.json
COMPLETE
FAMILY_AWARE_COMPLETE
```

Do not include raw family IDs.

---

# Part B12 — Tests

Add:

```text
tests/test_msancr_family_aware.py
```

Required tests:

1. no group appears in both outer train and test;
2. no group appears in inner train and validation;
3. five outer folds cover all subjects exactly once per seed;
4. families are never split;
5. 10 seed partition manifests are genuinely seed-varying;
6. deterministic output for the same seed;
7. different seeds normally generate different partitions;
8. group file must align exactly with subject array length/order;
9. missing group file causes hard failure;
10. no fallback to subject-wise CV;
11. no Family_ID values appear in output artifacts;
12. frozen model grids exactly match final config;
13. no post-hoc grid expansion;
14. fixed prior swaps reuse matched-selected hyperparameters;
15. primary comparison is A3 vs A4;
16. seed-level inference averages five folds first;
17. old subject-wise final outputs remain untouched.

Also run all existing MS-A-NCR tests.

---

# Part B13 — Preflight before expensive family-aware run

Before running 10×5:

1. run unit tests;
2. generate family split manifests only;
3. verify zero family leakage;
4. verify seed diversity;
5. run one smoke:
   ```text
   seed 0
   fold 0
   ```
   in a separate smoke output directory;
6. verify finite metrics and exact coefficient recovery;
7. only then launch full family-aware 10×5.

---

# Part B14 — Completion decision

Do not reduce the result to a binary "accepted/rejected".

Create:

```text
family_aware_hypothesis_summary.json
```

with:

```json
{
  "subject_wise_primary_contrast_nominal_significant": true,
  "subject_wise_conservative_holm_significant": false,

  "family_aware_primary_contrast_nominal_significant": false,
  "family_aware_conservative_holm_significant": false,

  "family_aware_mean_delta_pearson": 0.0,
  "family_aware_ci95": [0.0, 0.0],
  "family_aware_cohens_dz": 0.0,

  "prediction_advantage_survives_family_separation": false,
  "biomarker_alignment_survives_family_separation": false,

  "recommended_paper_claim": ""
}
```

Populate from actual results.

---

# Part C — No further tuning rule

After the family-aware 10×5 run:

Do NOT:

- add more seeds merely because p > 0.05;
- alter grids;
- change the target;
- switch priors based on test results;
- create new methods to rescue significance;
- analyze 50 folds as independent observations.

If the family-aware result is weaker, report it honestly.

The next stage after this prompt is manuscript-level synthesis, not more tuning, unless a genuine implementation bug is discovered.

---

# Commands

## Reporting-only stage

```bash
python scripts/108_finalize_msancr_reporting.py \
  --config configs/iclr/msancr_final_10x5.yaml \
  --output-dir outputs/iclr/msancr_final_10x5/postfinal_reporting
```

## If authorized restricted CSV is available locally

```bash
python scripts/24b_prepare_hcp_family_groups.py \
  --restricted_csv /PRIVATE/LOCAL/PATH/HCP_S1200_restricted.csv \
  --group_col Family_ID \
  --subjects inputs/dataset_SC/hcp_subjects_used.csv \
  --out inputs/dataset_SC/family_groups.npy
```

Then:

```bash
python scripts/109_run_msancr_family_aware.py \
  --config configs/iclr/msancr_family_aware_10x5.yaml
```

If restricted data are not locally available, complete Part A and preparation/tests for Part B, then stop with `FAMILY_DATA_REQUIRED`.

---

# Final completion report

Print:

```text
POST-FINAL REPORTING AUDIT
- actual frozen grid verified: PASS/FAIL
- corrected boundary metadata written
- primary A3-vs-A4 p = ...
- conservative all-five Holm p = ...
- secondary-family results summarized
- no model rerun performed

FAMILY-AWARE STATUS
- family groups available: YES/NO
- number of subjects: ...
- unique family groups: ...
- family leakage checks: PASS/FAIL
- unique seed partition manifests: .../10

If run completed:
- family-aware A4 Pearson = ...
- family-aware A3 Pearson = ...
- family-aware Δr = ...
- positive seeds = .../10
- primary Wilcoxon p = ...
- 95% CI = [...]
- Cohen dz = ...
- family-aware WM alignment = ...
- subject-wise vs family-aware interpretation = ...

No further hyperparameter tuning performed.
```
