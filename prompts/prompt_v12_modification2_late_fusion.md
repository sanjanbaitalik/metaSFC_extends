# OpenCode Prompt — Modification 2 Only: Leakage-Safe Prior-Aware Late Fusion

## Role

You are working inside the current `metaSFC_extends` repository.

Modification 1 (Fluid Integration Prior / FIP) has been integrity-audited and is a validated failure.

This prompt authorizes **Modification 2 only**:

> leakage-safe late fusion / stacking of modality-specific predictions with a prior-aware FC branch.

Do NOT implement Modification 3.
Do NOT redesign MS-A-NCR.
Do NOT rebuild the priors.
Do NOT use the failed FIP priors in the primary Modification-2 model.
Do NOT expand the frozen MS-A-NCR grids.
Do NOT automatically launch any 10×5 experiment.

Stop after the 3-seed pilot and wait for review.

---

# 1. Scientific hypothesis

The current early-fusion model learns jointly from:

```text
FC + SC
```

inside one generalized Ridge objective.

The diagnostic work showed:

- task priors act primarily on FC;
- SC remains a strong no-prior predictor;
- forcing prior-aware FC and SC into one joint coefficient geometry may suppress complementary information;
- Fluid has a strong no-prior signal but weak prior utility;
- Working Memory has stronger prior-aware FC utility.

Hypothesis:

> Keep FC, SC, and prior-aware FC predictions separate until the prediction level, then combine them using a strictly leakage-safe second-stage model.

This allows:
- SC to retain its strong data-driven signal;
- ordinary FC to retain signal that the prior may suppress;
- prior-aware FC to contribute only when it adds complementary predictive information.

---

# 2. Literature-aligned design principle

Use a two-level stacking design:

```text
Level 1:
modality/prior-specific predictors

Level 2:
small fusion model trained only from out-of-fold Level-1 predictions
```

Never train the fusion layer using in-sample Level-1 predictions.

---

# 3. Targets

Run both:

```text
Working Memory
Fluid Intelligence
```

Use the exact frozen target files and subject cohorts from the current repository.

Do not redefine either phenotype.

---

# 4. Pilot scope

Use only:

```text
seeds = [0,1,2]
outer folds = 5
inner folds = 3
```

Reuse the existing deterministic outer splits used in the corrected MS-A-NCR evaluations wherever possible.

Do not launch 10×5 in this prompt.

---

# 5. Priors

Use only the original task-matched Qwen priors:

```text
Working Memory:
working_memory_contrastive_qwen3

Fluid:
fluid_intelligence_contrastive_qwen3
```

Controls:

```text
unrelated task prior
shuffled matched prior
random prior
```

Do NOT use FIP1/FIP2/FIP3 as the primary branch.

Modification 1 is closed.

---

# 6. Level-1 branch models

Implement exactly three predictive branch types.

## Branch S — SC-only Ridge

Input:

```text
SC upper-triangle features only
```

Model:

```text
ordinary Ridge
```

Tune alpha using the frozen Ridge grid inside inner CV.

---

## Branch F0 — FC-only no-prior Ridge

Input:

```text
FC upper-triangle features only
```

Model:

```text
ordinary Ridge
```

Tune alpha using the same frozen Ridge grid.

---

## Branch FP — FC-only prior-aware MS-A-NCR

Use FC only.

Objective:

\[
\min_{\beta_F}
\|y-X_F\beta_F\|_2^2
+
\lambda_F\beta_F^T D(q;\gamma)\beta_F
+
\lambda_L\beta_F^T L_q\beta_F.
\]

No SC block exists in this branch.

Use the original matched task prior.

Requirements:

```text
D applies to all FC edges
Laplacian is FC-only
exact generalized-Ridge / dual solver
exact coefficient recovery
```

Use the already-corrected MS-A-NCR machinery.

Prefer adding an explicit:

```text
fc_only = true
```

mode rather than duplicating solver code.

Old FC+SC behavior must remain unchanged.

---

# 7. Frozen Level-1 hyperparameter grids

## Ridge

```yaml
ridge_grid:
  - 0.001
  - 0.01
  - 0.1
  - 1.0
  - 10.0
  - 100.0
```

## Prior-aware FC branch

```yaml
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

No expansion.

---

# 8. Leakage-safe Level-1 tuning

For each outer split:

```text
outer_train
outer_test
```

Use only outer_train for all Level-1 hyperparameter selection.

Use the existing 3-fold inner CV.

For every Level-1 branch:

1. fit scaler on inner-training only;
2. fit candidate model on inner-training;
3. predict inner-validation;
4. aggregate mean validation Pearson across 3 folds;
5. select hyperparameters using:
   - Pearson first;
   - RMSE tie-break;
   - MAE tie-break;
   - simplicity last.

No outer-test information may enter branch selection.

---

# 9. Generate Level-1 OOF predictions for stacking

After selecting each branch's hyperparameters inside outer_train, produce one out-of-fold prediction for every outer-training subject.

For branch b:

\[
\hat y^{OOF}_{b,i}
\]

must come from a model that did not train on subject i.

Branches:

```text
S
F0
FP-matched
```

These OOF predictions are the only features permitted for Level-2 fusion training.

Never use in-sample fitted predictions to learn stacking weights.

---

# 10. Level-2 fusion models

Implement exactly three predefined fusion models.

## LF0 — No-prior late fusion

Input Level-1 predictions:

```text
F0
S
```

Prediction:

\[
\hat y =
w_F\hat y_{F0}
+
w_S\hat y_S.
\]

Constraints:

```text
w_F >= 0
w_S >= 0
w_F + w_S = 1
```

This is the primary no-prior late-fusion baseline.

---

## LF1 — Prior-substitution late fusion

Input:

```text
FP-matched
S
```

Prediction:

\[
\hat y =
w_P\hat y_{FP}
+
w_S\hat y_S.
\]

Constraints:

```text
w_P >= 0
w_S >= 0
w_P + w_S = 1
```

Same fusion complexity as LF0.

This cleanly tests whether replacing the FC branch with prior-aware FC improves prediction.

---

## LF2 — Prior-augmentation late fusion

Input:

```text
F0
S
FP-matched
```

Prediction:

\[
\hat y =
w_F\hat y_{F0}
+
w_S\hat y_S
+
w_P\hat y_{FP}.
\]

Constraints:

```text
w_F >= 0
w_S >= 0
w_P >= 0

w_F + w_S + w_P = 1
```

This tests whether prior-aware FC contains complementary signal beyond ordinary FC and SC.

This is the main Modification-2 candidate.

---

# 11. Fusion weight search

Use a deterministic simplex grid.

For 2 branches:

```text
weights in increments of 0.05
```

For 3 branches:

```text
weights in increments of 0.05
sum exactly to 1
```

Select weights using outer-training OOF predictions only.

Primary Level-2 metric:

```text
Pearson
```

Tie-breakers:

```text
RMSE
MAE
smaller prior weight
```

The final tie-break intentionally favors the simpler/no-prior explanation.

Do not optimize fusion weights on outer-test labels.

---

# 12. Refit for outer-test evaluation

After Level-1 branch hyperparameters and Level-2 weights are frozen:

1. refit each required Level-1 branch on the complete outer-training set;
2. generate predictions on outer-test;
3. combine using the frozen Level-2 weights;
4. evaluate once.

No test-set refitting or recalibration.

---

# 13. Required baseline anchors

Also report existing-style:

```text
A4 modality-specific early-fusion no-prior Ridge
A3 early-fusion matched MS-A-NCR
```

using the exact corresponding outer split.

The primary late-fusion comparisons are:

```text
LF1 vs LF0
LF2 vs LF0
LF2 vs A4
```

The strongest no-prior comparator for success gating is:

```text
max(LF0, A4)
```

computed at seed-summary level using predefined model identities, not chosen per outer-test fold.

---

# 14. Prior-identity controls for LF2

Construct:

```text
LF2-matched
LF2-unrelated
LF2-shuffled
LF2-random
```

All use:

```text
F0
S
prior-aware FC branch
```

with the same three-branch architecture.

For strict prior-identity testing:

1. select FP matched branch hyperparameters inside outer_train;
2. select LF2 matched fusion weights from matched OOF predictions;
3. freeze:
   - branch hyperparameters;
   - fusion weights;
4. replace only the prior identity;
5. refit the prior-aware FC branch;
6. evaluate the same outer-test fold.

Call this:

```text
fixed-weight prior swap
```

This is mandatory.

---

# 15. Optional control-weight adaptation

If computationally cheap, also provide a secondary control analysis where:

- prior branch hyperparameters remain fixed from matched;
- fusion weights are re-selected inside outer_train for each control prior.

Call this:

```text
control-adaptive-weight sensitivity
```

This is secondary.

Do not let it replace the mandatory fixed-weight swap.

---

# 16. Null-branch architecture control

Implement:

```text
LF2-null
```

using Level-1 inputs:

```text
F0
S
zero prediction branch
```

with the same three-weight simplex machinery as LF2.

This checks whether simply adding a third stacking coefficient changes results.

It should collapse functionally to LF0.

Verify numerically.

---

# 17. Stacking diagnostics

For every outer split save:

```text
selected w_F
selected w_S
selected w_P

inner OOF Pearson
outer-test Pearson

prior weight > 0?
prior weight >= 0.10?
```

Report distributions separately for WM and Fluid.

A useful prior should receive nonzero weight because inner CV selects it, not because the architecture forces it.

---

# 18. Main scientific questions

For each task answer:

## Q1
Does late fusion itself beat early-fusion A4?

```text
LF0 vs A4
```

## Q2
Does replacing no-prior FC with prior-aware FC help?

```text
LF1 vs LF0
```

## Q3
Does adding prior-aware FC as a complementary branch help?

```text
LF2 vs LF0
```

## Q4
Is any LF2 gain task-prior-specific?

```text
LF2 matched vs unrelated/shuffled/random
```

---

# 19. Pilot metrics

For every task/model report:

```text
Pearson
RMSE
MAE
```

Aggregate:

```text
5 folds -> seed mean
3 seeds -> descriptive summary
```

Do NOT perform inferential significance claims with n=3.

Report:

```text
mean
median
SD
mean paired delta
median paired delta
positive seeds / 3
```

---

# 20. Large-margin success definition

For each task define the strongest no-prior comparator as:

```text
strongest_no_prior = max(A4, LF0)
```

using mean seed-level Pearson for descriptive identification.

## Large-margin success

```text
median ΔPearson(LF2-matched - strongest_no_prior) >= +0.015
mean ΔPearson >= +0.012
positive seeds = 3/3
no material RMSE degradation
```

AND matched must outperform at least two of:

```text
unrelated
shuffled
random
```

under fixed-weight swaps.

---

# 21. Promising success

```text
median ΔPearson >= +0.008
positive seeds >= 2/3
```

with no meaningful RMSE degradation.

---

# 22. Failure / null

```text
median ΔPearson < +0.005
```

or:

```text
positive seeds <= 1/3
```

means no useful Modification-2 gain for that task.

Do not tune fusion further.

---

# 23. Cross-task decision

Create task-level recommendations.

Allowed outcomes:

## Both WM and Fluid succeed

```text
recommended_next_step = full_late_fusion_10x5_both_tasks
```

Do NOT start automatically.

## Fluid succeeds, WM does not improve further

```text
recommended_next_step = human_review_before_full_runs
```

## WM succeeds, Fluid remains null

```text
recommended_next_step = consider_modification_3
```

## Neither task improves

```text
recommended_next_step = consider_modification_3
```

## Borderline results

```text
recommended_next_step = human_review
```

Do NOT implement Modification 3 automatically.

---

# 24. Biomarker policy

Modification 2 is primarily a prediction-integration experiment.

Do not invent a new fused biomarker by averaging coefficient maps across incompatible branches.

For biomarker reporting:

- retain the prior-aware FP branch's exact FC coefficients;
- compute the existing:
  - prior alignment;
  - rank stability;
  - top-10 Jaccard;
- additionally report the fusion prior weight `w_P`.

Interpretation:

```text
biomarker = FP branch
predictive contribution = fusion weight
```

Do not claim the stack itself has a single coefficient biomarker.

---

# 25. Files to add

Prefer:

```text
src/metascfc/models/iclr_backbones/fc_only_msancr.py
src/metascfc/experiments/prior_aware_late_fusion.py

scripts/113_run_prior_aware_late_fusion.py
configs/iclr/prior_aware_late_fusion.yaml

tests/test_fc_only_msancr.py
tests/test_prior_aware_late_fusion.py
```

If FC-only behavior can be added safely to the existing solver via configuration, avoid duplicating solver mathematics.

All old tests must continue passing.

---

# 26. Output directory

Use:

```text
outputs/iclr/prior_aware_late_fusion/
figures/iclr/prior_aware_late_fusion/
```

Never overwrite prior experiments.

---

# 27. Required outputs

Create:

```text
split_metrics.csv
seed_metrics.csv
summary_metrics.csv

level1_inner_cv_metrics.csv
level1_selected_hyperparameters.csv
level1_oof_integrity.csv

fusion_weight_search.csv
selected_fusion_weights.csv
fusion_weight_distribution.csv

fixed_prior_swap_split_metrics.csv
fixed_prior_swap_seed_metrics.csv
fixed_prior_swap_summary.csv

control_adaptive_weight_summary.csv  # if run

biomarker_metrics.csv
paired_comparisons.csv

late_fusion_decision.json
run_metadata.json
COMPLETE
```

---

# 28. Leakage audit

Create:

```text
stacking_leakage_audit.json
```

Verify programmatically:

```text
each outer-training subject's Level-1 stacking feature is OOF
no Level-1 model predicts a stacking-training subject it trained on
outer-test labels never enter branch selection
outer-test labels never enter fusion-weight selection
outer-test predictions are generated only after all selections are frozen
```

Require:

```text
n_failures = 0
```

before COMPLETE.

---

# 29. Tests

At minimum verify:

1. FC-only MS-A-NCR matches the mathematical direct solution on synthetic data;
2. SC features are never used by FP;
3. F0 uses FC only;
4. S uses SC only;
5. every stacking-training prediction is OOF;
6. no outer-test leakage;
7. LF0 weights are nonnegative and sum to 1;
8. LF1 weights are nonnegative and sum to 1;
9. LF2 weights are nonnegative and sum to 1;
10. LF2-null reproduces LF0 within tolerance;
11. matched/unrelated/shuffled/random use identical architecture;
12. fixed prior swaps reuse matched-selected fusion weights;
13. control priors do not alter F0/S predictions;
14. frozen task priors are used correctly;
15. failed FIP priors are not used in the primary experiment;
16. prior weight can be exactly zero;
17. tie-break favors smaller prior weight;
18. old MS-A-NCR results remain untouched;
19. Modification 3 cannot be launched automatically.

Run all relevant existing MS-A-NCR tests.

---

# 30. Preflight

Before the 3-seed pilot:

1. validate FC-only solver;
2. run unit tests;
3. run WM seed 0/fold 0 smoke;
4. run Fluid seed 0/fold 0 smoke;
5. verify OOF stacking integrity;
6. verify LF2-null ~= LF0;
7. verify matched fixed-swap controls;
8. only then run both tasks × 3 seeds × 5 folds.

---

# 31. Decision artifact

Create:

```text
outputs/iclr/prior_aware_late_fusion/late_fusion_decision.json
```

with:

```json
{
  "working_memory": {
    "A4_pearson": 0.0,
    "LF0_pearson": 0.0,
    "LF1_pearson": 0.0,
    "LF2_matched_pearson": 0.0,
    "strongest_no_prior_pearson": 0.0,
    "mean_delta_vs_strongest_no_prior": 0.0,
    "median_delta_vs_strongest_no_prior": 0.0,
    "positive_seeds": 0,
    "median_prior_weight": 0.0,
    "matched_beats_controls": 0,
    "status": ""
  },
  "fluid": {
    "A4_pearson": 0.0,
    "LF0_pearson": 0.0,
    "LF1_pearson": 0.0,
    "LF2_matched_pearson": 0.0,
    "strongest_no_prior_pearson": 0.0,
    "mean_delta_vs_strongest_no_prior": 0.0,
    "median_delta_vs_strongest_no_prior": 0.0,
    "positive_seeds": 0,
    "median_prior_weight": 0.0,
    "matched_beats_controls": 0,
    "status": ""
  },
  "recommended_next_step": ""
}
```

---

# 32. Completion report

Print:

```text
MODIFICATION 2/3 — PRIOR-AWARE LATE FUSION COMPLETE

WORKING MEMORY
A4 = ...
LF0 no-prior late fusion = ...
LF1 prior substitution = ...
LF2 prior augmentation = ...

LF2 - strongest no-prior:
mean ΔPearson = ...
median ΔPearson = ...
positive seeds = .../3
median prior fusion weight = ...

matched - unrelated = ...
matched - shuffled = ...
matched - random = ...

FLUID
A4 = ...
LF0 no-prior late fusion = ...
LF1 prior substitution = ...
LF2 prior augmentation = ...

LF2 - strongest no-prior:
mean ΔPearson = ...
median ΔPearson = ...
positive seeds = .../3
median prior fusion weight = ...

matched - unrelated = ...
matched - shuffled = ...
matched - random = ...

Decision:
full_late_fusion_10x5_both_tasks /
consider_modification_3 /
human_review

No Modification 3 implemented.
No post-hoc fusion tuning performed.
```

---

# FINAL NON-NEGOTIABLE INSTRUCTION

This prompt authorizes **Modification 2 only**.

After `late_fusion_decision.json` and `COMPLETE` are created:

STOP.

Do not implement Modification 3.
Do not launch any 10×5 run automatically.
Wait for review.
