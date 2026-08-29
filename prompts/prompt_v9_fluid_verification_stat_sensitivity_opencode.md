# OpenCode Prompt — Corrected Fluid Verification + Dependency-Aware Final Statistical Sensitivity

## Role

You are working inside the current `metaSFC_extends` repository after completion of:

- Audit 100
- Audit 101 v2
- MS-A-NCR pilot/refinement/grid closure
- final Working-Memory 10-seed × 5-fold MS-A-NCR experiment
- post-final reporting patch
- preparation of family-aware robustness code

HCP family data are restricted and are **not currently available**.

Therefore:

> Do NOT block the project waiting for family data.

The family-aware runner/config/tests must remain prepared and untouched for later use, but no family-aware experiment is possible in this step.

This prompt has two goals:

1. perform a **reporting-only dependency-aware statistical sensitivity analysis** on the already-frozen Working-Memory 10×5 results, with zero model reruns;
2. perform a **corrected Fluid Intelligence 3-seed verification** using the final frozen MS-A-NCR implementation.

This is not a model-development prompt.

Do not redesign MS-A-NCR.
Do not change any prior.
Do not expand any hyperparameter grid.
Do not tune using outer-test results.
Do not change the cohort.
Do not change the target definitions.

---

# Why the Fluid verification is necessary

The earlier Fluid MS-A-NCR pilot was run **before** the critical solver correction that:

- applied the anisotropic `D(q, gamma)` penalty to the full FC block;
- corrected inactive-edge penalties;
- corrected prediction kernels;
- corrected eigencache semantics;
- added exact primal coefficient recovery.

The definitive corrected solver was subsequently evaluated only on Working Memory.

Therefore the repository currently does **not** contain an apples-to-apples corrected evaluation of the final MS-A-NCR method on Fluid Intelligence.

Because the ICLR extension was conceived around two cognitive tasks:

```text
Fluid Intelligence
Working Memory
```

we need one corrected, frozen Fluid verification before deciding how to present task generality/specificity.

---

# PART A — Working-Memory dependency-aware statistical sensitivity

## Important statistical issue

The completed final analysis uses:

```text
10 seed-level summaries
```

where every seed is a different repeated 5-fold partition of the **same subjects**.

Those ten repetitions are useful for robustness, but they are not fully independent samples.

Do not delete or replace the existing:

```text
primary A3-vs-A4 Wilcoxon p = 0.037109
```

because that was the predesignated primary reporting analysis.

However, add a sensitivity analysis that explicitly accounts for overlap between repeated cross-validation folds.

---

# A1 — Corrected repeated k-fold t-test

Implement the Bouckaert-Frank / Nadeau-Bengio corrected repeated-k-fold test using the 50 paired outer-fold differences.

For:

```text
r = 10 repeats
k = 5 folds
n_resamples = 50
```

use:

\[
t =
\frac{\bar d}
{\sqrt{
\left(
\frac{1}{kr}
+
\frac{n_{test}}{n_{train}}
\right)
s_d^2
}}
\]

where:

- `d` = paired metric difference per outer fold;
- `s_d^2` = sample variance of the 50 paired fold differences;
- use the actual mean `n_test/n_train` ratio from `split_metrics.csv`;
- degrees of freedom = `kr - 1 = 49`.

Use a **two-sided** test for paper-facing sensitivity.

For Pearson:

```text
d = A3 - comparator
```

For RMSE/MAE:

```text
d = comparator - A3
```

so positive always means A3 is better.

---

# A2 — Comparisons

Compute corrected repeated-CV tests for:

```text
A3 matched vs A4
A3 matched vs A2
A3 matched vs unrelated fixed
A3 matched vs shuffled fixed
A3 matched vs random fixed
```

for:

```text
Pearson
RMSE
MAE
```

Do not apply these tests to biomarker metrics because those biomarker statistics are not naturally fold-level predictive losses in the same way.

---

# A3 — Output

Create:

```text
outputs/iclr/msancr_final_10x5/postfinal_reporting/cv_dependence_sensitivity.csv
outputs/iclr/msancr_final_10x5/postfinal_reporting/cv_dependence_sensitivity.json
```

Columns:

```text
comparison
metric
n_resamples
k_folds
r_repeats
mean_difference
sd_fold_difference
mean_test_train_ratio
corrected_se
corrected_t
df
corrected_p_two_sided
direction_positive_is_A3_better
```

---

# A4 — Do not alter the original conclusion silently

Update/create:

```text
statistical_sensitivity_interpretation.json
```

It must report all three layers separately:

### Layer 1 — designated primary analysis

```text
seed-aggregated paired Wilcoxon
A3 vs A4 Pearson p = existing value
```

### Layer 2 — conservative multiplicity sensitivity

```text
existing all-five Holm-adjusted value
```

### Layer 3 — CV-dependence sensitivity

```text
corrected repeated-k-fold t-test
```

Recommended interpretation structure:

```text
The designated primary seed-aggregated paired analysis showed a positive
A3-vs-A4 effect. Because repeated cross-validation partitions reuse the same
participants, we additionally report a corrected repeated-k-fold sensitivity
analysis that inflates uncertainty for train/test overlap. Conclusions from
this sensitivity analysis are reported separately and do not trigger model
retuning.
```

Do not claim the corrected repeated-CV test is "the one true test".

Do not hide it if it is nonsignificant.

---


# A4b — Resolve multiplicity-family naming ambiguity

The existing post-final outputs currently contain two different legitimate-but-distinct
secondary correction families:

### Prior-specificity-only family

```text
A3 matched vs unrelated
A3 matched vs shuffled
A3 matched vs random
```

Holm across these 3 comparisons answers:

> Is the matched prior superior to the three prior-identity controls?

### Broad secondary-method family

```text
A3 matched vs A2
A3 matched vs unrelated
A3 matched vs shuffled
A3 matched vs random
```

Holm across these 4 comparisons answers a broader secondary-method question.

Do not call both simply:

```text
secondary Holm
```

Create explicit columns/artifacts for both:

```text
p_holm_prior_specificity_3
p_holm_broad_secondary_4
```

where applicable.

Preserve the original five-comparison conservative sensitivity separately:

```text
p_holm_all_five
```

The paper must state the family name whenever an adjusted p-value is quoted.

Do not select one correction family post hoc based on significance.


# A5 — Tests

Create/update:

```text
tests/test_msancr_cv_dependence_sensitivity.py
```

Test:

1. exactly 50 aligned fold pairs for A3 vs A4;
2. seed/fold/test hash alignment;
3. positive orientation is correct;
4. actual per-fold n_train/n_test values are used;
5. formula matches a hand-calculated synthetic example;
6. df = 49 for complete 10×5 data;
7. no model rerun occurs;
8. original final outputs remain unchanged;
9. original nominal Wilcoxon and Holm values are preserved separately.

---

# PART B — Corrected Fluid Intelligence verification

## Scope

Run only:

```text
Fluid Intelligence (PMAT24_A_CR)
seeds = [0, 1, 2]
5 outer folds
3 inner folds
```

This is a verification run, not another tuning phase.

---

# B1 — Use the final frozen solver

Use the current corrected:

```text
src/metascfc/models/iclr_backbones/modality_selective_anisotropic_ncr.py
```

without mathematical modification.

The model remains:

\[
\min_{\beta}
\|y-X\beta\|_2^2
+
\lambda_{FC}\beta_{FC}^T D(q;\gamma)\beta_{FC}
+
\lambda_{SC}\|\beta_{SC}\|_2^2
+
\lambda_L \beta_{FC}^T L_q \beta_{FC}.
\]

Required behavior:

```text
D applies to ALL FC edges
Laplacian applies only to FC
SC receives ordinary modality-specific Ridge
exact dual solver
exact primal coefficient recovery
```

Do not modify these definitions.

---

# B2 — Frozen final grids

Use exactly the final Working-Memory grids:

```yaml
ridge_grid:
  - 0.001
  - 0.01
  - 0.1
  - 1.0
  - 10.0
  - 100.0

ridge_expanded_grid:
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

No new candidate may be added after observing Fluid results.

---

# B3 — Fluid target and priors

Use:

```text
target:
Fluid Intelligence (PMAT24_A_CR)

label:
inputs/dataset_SC/label_all.npy
```

Priors:

```text
matched:
outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv

unrelated:
outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv

shuffled:
outputs/priors/llm/fluid_intelligence_contrastive_qwen3_shuffled/roi_prior.csv

random:
outputs/priors/random_prior/aal116/roi_prior.csv
```

Verify paths and subject alignment before running.

---

# B4 — New files

Prefer creating a generic runner rather than copying large amounts of Working-Memory code.

Create:

```text
configs/iclr/msancr_fluid_verification.yaml
scripts/110_run_msancr_fluid_verification.py
tests/test_msancr_fluid_verification.py
```

If necessary, minimally generalize:

```text
src/metascfc/experiments/msancr_refinement.py
```

so it can accept a single configured target instead of hard-coding `"working_memory"`.

If generalizing existing code:

- preserve Working-Memory behavior bit-for-bit;
- all existing WM tests must still pass;
- do not modify solver mathematics;
- do not modify completed output files.

A cleaner option is a thin target-generic wrapper around the already-tested functions.

---

# B5 — Models

Run:

```text
A4 modality-specific no-prior Ridge
A4_iso same-solver sanity baseline
A2 FC-only Laplacian
A3 MS-A-NCR matched
```

and fixed matched-selected prior swaps:

```text
A3 unrelated
A3 shuffled
A3 random
```

Do not retune control priors.

For each split:

1. select A3 hyperparameters using matched prior only;
2. freeze:
   ```text
   lambda_fc
   lambda_sc
   gamma
   lambda_L
   lifting
   ```
3. substitute unrelated/shuffled/random priors;
4. refit on exactly the same outer-training data;
5. evaluate on exactly the same outer-test data.

---

# B6 — Selection protocol

Use exactly:

```text
3-fold inner CV
Pearson-first selection
Pearson tie tolerance = 0.002
RMSE second
MAE third
simplicity last
```

Train-only scaling.

No outer-test information in model selection.

Use the exact subject-wise split strategy currently used by the final Working-Memory experiment because family data are unavailable.

Record:

```text
group_aware = false
family_aware_status = deferred_restricted_data_unavailable
```

Do not claim family-independent generalization.

---

# B7 — Outputs

Write to:

```text
outputs/iclr/msancr_fluid_verification/
figures/iclr/msancr_fluid_verification/
```

Create at minimum:

```text
split_metrics.csv
seed_metrics.csv
summary_metrics.csv
inner_cv_metrics.csv
selected_hyperparameters.csv

prior_swap_split_metrics.csv
prior_swap_seed_metrics.csv
prior_swap_summary.csv

biomarker_metrics.csv
paired_comparisons.csv

fluid_verification_decision.json
run_metadata.json
COMPLETE
```

Save exact coefficient artifacts as in WM.

---

# B8 — Biomarker metrics

For Fluid A3 matched report:

```text
Fluid-prior alignment
rank stability
top-10 Jaccard
```

Compare descriptively against:

```text
A4
A2
unrelated
shuffled
random
```

Use exact recovered FC coefficients.

Do not invent a new biomarker definition.

---

# B9 — 3-seed interpretation only

This run has:

```text
n = 3 seeds
```

Do not make significance claims.

Report:

```text
mean ΔPearson
median ΔPearson
positive seeds / 3

mean ΔRMSE
mean ΔMAE
```

for:

```text
A3 vs A4
A3 vs A2
A3 vs unrelated
A3 vs shuffled
A3 vs random
```

---

# B10 — Decision gate

Create:

```text
fluid_verification_decision.json
```

## Strong enough for final Fluid 10×5

If:

```text
median ΔPearson A3-vs-A4 >= +0.010
positive seeds >= 2/3
mean ΔPearson >= +0.008
no material RMSE degradation
```

recommend:

```text
full_fluid_10x5_frozen
```

No further method refinement.

---

## Weak but directionally positive

If:

```text
+0.005 <= median ΔPearson < +0.010
positive seeds >= 2/3
```

recommend:

```text
review_before_full_fluid
```

Do not tune the method.

---

## Task-selective null

If:

```text
median ΔPearson < +0.005
```

or fewer than 2/3 seeds improve:

recommend:

```text
stop_fluid_method_development
```

Interpret this as possible task specificity:

```text
MS-A-NCR predictive benefit is stronger for Working Memory than Fluid Intelligence
under the present external priors.
```

Do not change grids or priors to rescue Fluid after this test.

---

# PART C — Cross-task synthesis artifact

After Fluid verification, create:

```text
outputs/iclr/cross_task_msancr_summary/
```

with:

```text
cross_task_prediction_summary.csv
cross_task_prior_signal_summary.csv
cross_task_biomarker_summary.csv
cross_task_interpretation.json
```

Combine:

### Working Memory

Use frozen final 10×5 values from:

```text
outputs/iclr/msancr_final_10x5/
```

### Fluid

Use the new corrected verification values.

### Prior-signal diagnostics

Pull relevant previously frozen Audit 101 v2 values showing FC-side residual enrichment for each task.

Do not rerun Audit 101.

---

# C1 — Cross-task question

Explicitly assess:

> Does the magnitude of MS-A-NCR benefit track the amount of task-specific FC prior signal identified before method development?

Report, without overclaiming causality:

```text
Fluid:
FC residual enrichment
A3-A4 prediction gain
prior alignment

Working Memory:
FC residual enrichment
A3-A4 prediction gain
prior alignment
```

This can become an important paper interpretation if WM is positive and Fluid is weak.

---

# PART D — Family-aware status

Do not attempt the family-aware experiment.

Keep existing:

```text
src/metascfc/experiments/msancr_family_aware.py
scripts/109_run_msancr_family_aware.py
configs/iclr/msancr_family_aware_10x5.yaml
tests/test_msancr_family_aware.py
```

unchanged unless there is a correctness bug.

Create/update a simple non-sensitive status artifact:

```text
outputs/iclr/family_aware_status.json
```

containing:

```json
{
  "status": "deferred",
  "reason": "HCP Family_ID is restricted and not available before the current study deadline",
  "family_aware_code_prepared": true,
  "family_aware_results_available": false,
  "paper_limitation_required": true
}
```

Do not include restricted IDs.

---

# PART E — No more tuning rule

After this prompt:

Do not automatically:

- redesign MS-A-NCR;
- add more hyperparameters;
- change priors;
- change Fluid target;
- add extra seeds to chase p-values;
- treat 50 folds as independent ordinary samples;
- fabricate family groups.

The next decision must be based on the corrected Fluid verification:

```text
either full frozen Fluid 10×5
or manuscript synthesis with task-specific findings
```

---

# Tests

Run:

```bash
pytest -q \
  tests/test_msancr.py \
  tests/test_msancr_refinement.py \
  tests/test_msancr_grid_closure.py \
  tests/test_msancr_postfinal_reporting.py \
  tests/test_msancr_cv_dependence_sensitivity.py \
  tests/test_msancr_fluid_verification.py
```

Also ensure the existing family-aware tests continue to pass even though no restricted data are available.

---

# Preflight

Before Fluid compute:

1. verify final solver direct/dual tests;
2. verify Fluid labels align with 412 subjects;
3. verify all four Fluid prior vectors have 116 entries;
4. run:
   ```text
   seed 0 / fold 0
   ```
   into:
   ```text
   outputs/iclr/msancr_fluid_verification_smoke/
   ```
5. verify finite metrics, exact coefficients, no leakage;
6. then run all:
   ```text
   seeds 0,1,2
   folds 0..4
   ```

---

# Completion report

Print:

```text
WORKING-MEMORY STATISTICAL SENSITIVITY
Primary seed-level Wilcoxon A3-vs-A4 p = ...
Conservative all-five Holm p = ...
Corrected repeated-5-fold×10 sensitivity p = ...
Mean ΔPearson = ...
Interpretation = ...

CORRECTED FLUID VERIFICATION
A4 Pearson = ...
A2 Pearson = ...
A3 matched Pearson = ...

A3-A4:
mean ΔPearson = ...
median ΔPearson = ...
positive seeds = .../3
mean ΔRMSE = ...
mean ΔMAE = ...

Matched prior swaps:
vs unrelated = ...
vs shuffled = ...
vs random = ...

Fluid biomarker alignment = ...
rank stability = ...
top-10 Jaccard = ...

FLUID DECISION:
full_fluid_10x5_frozen /
review_before_full_fluid /
stop_fluid_method_development

FAMILY-AWARE:
deferred — restricted Family_ID unavailable

No model redesign or post-hoc grid tuning performed.
