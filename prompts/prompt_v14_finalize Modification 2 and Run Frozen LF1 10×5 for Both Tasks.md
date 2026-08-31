# OpenCode Prompt — Finalize Modification 2 and Run Frozen LF1 10×5 for Both Tasks

## Objective

Modification 1 is closed as a validated failure.

Modification 2 has passed its integrity audit and identified the predeclared **LF1 prior-substitution late-fusion architecture** as the simplest and strongest candidate for both Working Memory and Fluid Intelligence.

This prompt authorizes **finalization of Modification 2 only**.

Do NOT implement Modification 3.

The goals are:

1. resolve the final LF1 matched fixed-control reproducibility discrepancy;
2. reconstruct the architecture-matched LF1-vs-LF0 pilot evidence correctly;
3. if and only if all predefined gates pass, freeze LF1;
4. run the definitive 10-seed × 5-fold LF1 experiment for both tasks;
5. compute prediction, prior-control, biomarker, and sensitivity statistics;
6. stop.

Do not redesign MS-A-NCR.
Do not change priors.
Do not expand grids.
Do not change fusion-weight resolution.
Do not use FIP.
Do not add new fusion architectures.
Do not automatically implement Modification 3.

---

## 1. Current corrected Modification-2 evidence

### Working Memory

```text
A4 early-fusion no-prior = 0.2743
LF0 no-prior late fusion = 0.2707
LF1 prior substitution   = 0.2848
LF2 prior augmentation   = 0.2840
```

LF1 vs A4:

```text
seed deltas = [-0.0004, +0.0234, +0.0083]
mean        = +0.0104
median      = +0.0083
positive    = 2/3
status      = PROMISING
```

The primary prior-effect comparison must now be:

```text
LF1 vs LF0
```

because LF0 and LF1 have the same two-branch late-fusion architecture and differ in whether the FC branch uses the task prior.

Their aggregate difference is approximately:

```text
0.2848 - 0.2707 = +0.0141
```

but reconstruct the actual three seed-level deltas from source rows before any full run.

---

### Fluid Intelligence

```text
A4 early-fusion no-prior = 0.3238
LF0 no-prior late fusion = 0.3634
LF1 prior substitution   = 0.3794
LF2 prior augmentation   = 0.3764
```

LF1 vs LF0:

```text
seed deltas = [+0.0113, +0.0308, +0.0056]
mean        = +0.0159
median      = +0.0113
positive    = 3/3
status      = PROMISING
```

LF1 is therefore the final Modification-2 candidate.

---

# 2. Final LF1 architecture

LF1 contains:

```text
FP = FC-only matched-prior MS-A-NCR
S  = SC-only ordinary Ridge
```

with:

```text
yhat = w_prior * yhat_FP + w_sc * yhat_S

w_prior >= 0
w_sc >= 0
w_prior + w_sc = 1
```

Fusion weights must be learned only from outer-training OOF predictions.

Primary no-prior comparator:

```text
LF0
```

with:

```text
F0 = FC-only ordinary Ridge
S  = SC-only ordinary Ridge
```

and the identical two-branch convex fusion architecture.

Thus:

```text
LF1 vs LF0
```

is the primary test of whether **adding the prior improves prediction**.

A4 is a secondary benchmark only.

---

# 3. Resolve the final fixed-control discrepancy

The integrity audit reports:

```text
WM ordinary LF1 matched = 0.2848
WM fixed-control matched = 0.2795
```

This is not acceptable for a final control experiment.

The matched member of the fixed-prior-swap block must reproduce ordinary LF1 matched when it uses:

```text
same task
same seed
same fold
same train/test subjects
same FP hyperparameters
same SC prediction
same fusion weights
same matched prior
same preprocessing
same fitting code
```

Fluid already nearly reproduces:

```text
ordinary LF1 = 0.3794
control matched = 0.3793
```

Audit WM at prediction level.

Create:

```text
outputs/iclr/lf1_finalization_audit/
```

and:

```text
lf1_matched_reproduction.csv
```

with:

```text
task
seed
fold

ordinary_pearson
control_matched_pearson

ordinary_rmse
control_matched_rmse

ordinary_mae
control_matched_mae

ordinary_lambda_fp
control_lambda_fp

ordinary_gamma
control_gamma

ordinary_lambda_L
control_lambda_L

ordinary_lifting
control_lifting

ordinary_w_prior
control_w_prior

ordinary_w_sc
control_w_sc

prediction_max_abs_diff

same_hyperparameters
same_weights
same_predictions
```

Require:

```text
same_predictions = true
```

within numerical tolerance.

---

# 4. Allowed fixes

## Case A — reporting/aggregation only

If predictions actually match but summaries differ:

```text
fix reporting only
do not rerun predictive models
```

## Case B — fixed-control path differs

If the fixed-control code uses a different:

```text
scaler
prior model
hyperparameters
fusion weights
refit path
fold
```

fix only the control path.

Rerun only the 3-seed LF1 fixed-control block.

Do not rerun LF0/LF1 primary models unnecessarily.

## Case C — ordinary LF1 correctness issue

If ordinary LF1 contains leakage or incorrect model selection:

```text
STOP
```

Fix correctness and rerun the exact same 3-seed LF1/LF0 pilot.

Do not launch 10×5 until the corrected pilot passes the frozen gate.

---

# 5. Reconstruct LF1-vs-LF0 seed-level evidence

Create:

```text
outputs/iclr/lf1_finalization_audit/lf1_vs_lf0_seed_gate.csv
```

for both tasks.

Columns:

```text
task
seed

LF0_pearson
LF1_pearson
delta_pearson

LF0_rmse
LF1_rmse
rmse_improvement

LF0_mae
LF1_mae
mae_improvement
```

Define:

```text
rmse_improvement = LF0_rmse - LF1_rmse
mae_improvement  = LF0_mae - LF1_mae
```

so positive means LF1 is better.

This reconstruction is mandatory for Working Memory.

---

# 6. Pre-final 3-seed gate

Before launching 10×5, BOTH tasks must satisfy:

```text
median ΔPearson(LF1 - LF0) >= +0.008
mean ΔPearson >= +0.008
positive seeds >= 2/3
no material mean RMSE degradation
```

Additionally, corrected LF1 matched controls must show:

```text
matched mean Pearson >
at least 2 of:
  unrelated
  shuffled
  random
```

for each task.

If either task fails:

```text
STOP
recommended_next_step = human_review_or_modification_3
```

Do not launch 10×5.

---

# 7. Freeze the final LF1 configuration

Only if the previous gate passes, create:

```text
configs/iclr/lf1_final_10x5.yaml
```

with:

```yaml
tasks:
  - working_memory
  - fluid_intelligence

seeds:
  - 0
  - 1
  - 2
  - 3
  - 4
  - 5
  - 6
  - 7
  - 8
  - 9

outer_folds: 5
inner_folds: 3

fusion_model: LF1
fusion_weight_step: 0.05

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

Use the existing task-matched Qwen priors.

Do not use FIP.

Do not expand anything.

---

# 8. Final models

For both tasks run:

```text
LF0 — no-prior late fusion
LF1 — matched-prior substitution late fusion
A4  — early-fusion modality-specific Ridge
```

LF0 is the primary no-prior comparator.

A4 is a secondary benchmark.

Because historical A4 experiments used different train/validation conventions, run A4 under the **same outer partitions and current Modification-2 evaluation convention**.

Label it clearly, e.g.:

```text
A4_latefusion_protocol
```

Do not mix it silently with historical A4 results.

---

# 9. Final prior controls

For LF1 evaluate:

```text
matched
unrelated
shuffled
random
```

For every outer split:

1. tune matched FP inside outer training;
2. choose matched LF1 fusion weights from outer-training OOF predictions;
3. freeze:
   ```text
   FP hyperparameters
   LF1 weights
   SC branch
   ```
4. change only prior identity;
5. refit the FP branch;
6. evaluate the same outer-test subjects.

Require:

```text
ordinary matched LF1 == fixed-control matched LF1
```

numerically.

Write a hard integrity check.

---

# 10. Leakage-safe stacking

For every outer split:

### Level-1 selection

Inside outer_train only:

```text
SC-only Ridge
FC-only Ridge
FC-only matched-prior MS-A-NCR
```

must be selected using inner CV only.

### Level-2 training

Generate one OOF Level-1 prediction for every outer-training subject.

A subject's stacking feature may never come from a model trained on that subject.

Use only these OOF predictions for fusion-weight selection.

### Outer-test evaluation

Once branch HPs and fusion weights are frozen:

```text
refit branches on full outer_train
predict outer_test once
combine using frozen weights
```

No outer-test labels may influence selection.

---

# 11. Final statistical unit

The full experiment is:

```text
10 seeds × 5 outer folds
```

For primary inference:

```text
average 5 folds within each seed
obtain 10 seed-level values
use those 10 aligned values for paired inference
```

Never treat the 50 folds as ordinary independent observations.

---

# 12. Primary prediction hypothesis

For each task independently:

```text
LF1 matched > LF0 no-prior
```

Primary metric:

```text
Pearson r
```

Report:

```text
LF0 mean ± SD
LF1 mean ± SD

mean paired ΔPearson
median paired ΔPearson
positive seeds / 10

two-sided paired Wilcoxon p
95% paired bootstrap CI
Cohen dz
```

This is the primary answer to:

> Does adding the task prior improve prediction?

---

# 13. Primary multiplicity family

There are exactly two primary prediction hypotheses:

```text
WM: LF1 vs LF0
Fluid: LF1 vs LF0
```

Apply Holm correction across these **two** p-values.

Report both:

```text
raw p
two-task Holm p
```

Do not mix A4 or prior-control hypotheses into this primary family.

---

# 14. Secondary comparison families

Per task evaluate:

```text
LF1 vs A4
LF1 matched vs unrelated
LF1 matched vs shuffled
LF1 matched vs random
```

Use clearly named secondary/control Holm families.

Do not call all adjusted values merely `Holm p`.

---

# 15. Repeated-CV dependence sensitivity

For:

```text
LF1 vs LF0
```

calculate the repository's corrected repeated-k-fold sensitivity using the 50 aligned fold-level differences.

Report three distinct inferential views:

```text
seed-level paired Wilcoxon
two-task primary Holm
corrected repeated-CV sensitivity
```

Do not replace one with another.

---

# 16. Error metrics

For LF1-vs-LF0 report:

```text
RMSE improvement = LF0_RMSE - LF1_RMSE
MAE improvement  = LF0_MAE - LF1_MAE
```

Positive means LF1 is better.

Report seed-level paired statistics as secondary outcomes.

If Pearson improves while error metrics materially worsen, flag the disagreement.

---

# 17. Large-margin consistency gate

For each final task report:

```text
mean ΔPearson >= +0.010
median ΔPearson >= +0.010
positive seeds >= 7/10
```

Store:

```text
large_margin_consistency_gate = true/false
```

Do not alter this after seeing results.

---

# 18. Biomarker evaluation

LF1 itself has no meaningful single fused coefficient vector.

The biomarker must come from:

```text
FP — matched-prior FC-only MS-A-NCR branch
```

Use exact FC coefficients.

For each task report:

```text
task-prior alignment
rank stability
top-10 Jaccard
```

Compare against:

```text
FC-only no-prior Ridge
FP unrelated
FP shuffled
FP random
```

Also report:

```text
LF1 prior fusion-weight distribution
```

Do not combine SC and FC coefficients into an artificial fused biomarker.

---

# 19. Biomarker statistics

Use the ten seed-level summaries.

Report:

```text
mean ± SD
paired Wilcoxon
Holm correction within biomarker family
bootstrap 95% CI
Cohen dz
```

Prediction and biomarker families must remain separate.

---

# 20. Professor-requirement artifact

Create:

```text
outputs/iclr/lf1_final_10x5/professor_requirement_summary.json
```

For each task include:

```text
task
no_prior_model = LF0
prior_model = LF1

no_prior_pearson
prior_pearson
delta_pearson
median_delta_pearson
positive_seeds

prediction_raw_p
prediction_two_task_holm_p
prediction_cv_corrected_p

prior_alignment_no_prior
prior_alignment_matched

matched_beats_negative_controls
biomarker_supported

professor_prediction_requirement
professor_biomarker_requirement
```

Overall fields:

```text
both_tasks_prediction_improved
both_tasks_biomarker_improved
professor_requirement_satisfied_both_tasks
```

Compute these honestly.

Do not hard-code `true`.

---

# 21. Required final outputs

Use:

```text
outputs/iclr/lf1_final_10x5/
figures/iclr/lf1_final_10x5/
```

Required artifacts:

```text
preflight_integrity.json

split_metrics.csv
seed_metrics.csv
summary_metrics.csv

level1_selected_hyperparameters.csv
level1_oof_integrity.csv

selected_fusion_weights.csv
fusion_weight_distribution.csv

fixed_prior_swap_split_metrics.csv
fixed_prior_swap_seed_metrics.csv
fixed_prior_swap_summary.csv
fixed_prior_swap_integrity.json

prediction_statistics.csv
prediction_statistics.json

biomarker_metrics.csv
biomarker_statistics.csv

repeated_cv_sensitivity.csv
two_task_primary_holm.csv

professor_requirement_summary.json
final_decision.json
run_metadata.json

COMPLETE
FINAL_COMPLETE
```

---

# 22. Paper-ready tables

Create:

```text
table_lf1_prediction.csv
table_lf1_prediction.tex

table_lf1_prior_controls.csv
table_lf1_prior_controls.tex

table_lf1_biomarkers.csv
table_lf1_biomarkers.tex
```

Prediction table rows per task:

```text
A4
LF0
LF1 matched
```

Columns:

```text
Pearson mean ± SD
RMSE mean ± SD
MAE mean ± SD
```

---

# 23. Figures

Create at minimum:

```text
wm_lf1_seed_deltas.pdf
fluid_lf1_seed_deltas.pdf

wm_lf1_vs_lf0.pdf
fluid_lf1_vs_lf0.pdf

lf1_prior_weight_distribution.pdf
lf1_prior_control_comparison.pdf
lf1_biomarker_comparison.pdf
```

---

# 24. Tests

Add:

```text
tests/test_lf1_finalization.py
tests/test_lf1_final_10x5.py
```

Test:

1. ordinary LF1 matched == fixed-control matched;
2. LF1 and LF0 use identical outer splits;
3. LF1 and LF0 have equal two-branch fusion complexity;
4. only the FC branch changes in prior use;
5. every stacking-training prediction is OOF;
6. fusion weights use outer-training data only;
7. fusion weights are nonnegative and sum to one;
8. no outer-test leakage;
9. fixed controls reuse matched HPs/weights;
10. prior identity is the only control difference;
11. 10 seeds × 5 folds complete for both tasks;
12. inference aggregates folds within seed;
13. two-task Holm family contains exactly the WM and Fluid LF1-vs-LF0 Pearson hypotheses;
14. repeated-CV sensitivity remains separate;
15. no Modification 3 executes;
16. no post-hoc grid expansion;
17. historical outputs remain unchanged.

Run all existing late-fusion and MS-A-NCR tests too.

---

# 25. Smoke test and launch gate

Before final compute:

1. resolve WM matched-control discrepancy;
2. reconstruct LF1-vs-LF0 seed deltas for both tasks;
3. verify both pass the pre-final gate;
4. run all tests;
5. run:
   ```text
   WM seed 0 / fold 0
   Fluid seed 0 / fold 0
   ```
   in separate smoke outputs;
6. verify OOF integrity;
7. verify matched fixed-control prediction == ordinary LF1 matched;
8. verify finite predictions, metrics, and coefficient artifacts.

ONLY if every check passes, execute the final run.

Suggested command:

```bash
python scripts/114_run_lf1_final_10x5.py \
  --config configs/iclr/lf1_final_10x5.yaml
```

This prompt authorizes that final 10×5 run only after preflight passes.

If any invariant fails:

```text
STOP
```

and report it.

---

# 26. Final result states

Allowed:

```text
BOTH_TASKS_PRIOR_PREDICTION_AND_BIOMARKER_SUPPORTED
BOTH_TASKS_PREDICTION_IMPROVED_BIOMARKER_MIXED
WM_SUPPORTED_FLUID_WEAK
FLUID_SUPPORTED_WM_WEAK
PREDICTION_EFFECTS_NOT_ROBUST
```

Do not implement Modification 3 automatically under any result.

---

# 27. Completion report

Print:

```text
MODIFICATION 2 FINAL LF1 10×5 COMPLETE

WORKING MEMORY

LF0 no-prior = ...
LF1 matched = ...

ΔPearson mean = ...
ΔPearson median = ...
positive seeds = .../10

Wilcoxon p = ...
two-task Holm p = ...
corrected repeated-CV p = ...

95% CI = [...]
Cohen dz = ...

Matched controls:
unrelated = ...
shuffled = ...
random = ...

Biomarker:
no-prior alignment = ...
matched alignment = ...
rank stability = ...
top-10 Jaccard = ...

FLUID INTELLIGENCE

LF0 no-prior = ...
LF1 matched = ...

ΔPearson mean = ...
ΔPearson median = ...
positive seeds = .../10

Wilcoxon p = ...
two-task Holm p = ...
corrected repeated-CV p = ...

95% CI = [...]
Cohen dz = ...

Matched controls:
unrelated = ...
shuffled = ...
random = ...

Biomarker:
no-prior alignment = ...
matched alignment = ...
rank stability = ...
top-10 Jaccard = ...

PROFESSOR REQUIREMENT

Prediction improved for WM: YES/NO
Prediction improved for Fluid: YES/NO

Biomarker improved for WM: YES/NO
Biomarker improved for Fluid: YES/NO

Overall requirement satisfied for both tasks: YES/NO

No Modification 3 implemented.
No post-hoc tuning performed.
```

---

# FINAL NON-NEGOTIABLE INSTRUCTION

This prompt finalizes **Modification 2 only**.

Do not implement Modification 3.

Do not modify the architecture, priors, grids, or statistical hypotheses after seeing the final 10×5 results.

If final results are weaker than the 3-seed pilot, report them honestly and STOP.