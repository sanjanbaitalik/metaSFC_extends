# OpenCode Prompt — Modification 2 Integrity Audit and LF1/LF2 Eligibility Review

## Objective

Audit Modification 2 before any Modification 3. Do not implement Modification 3, redesign MS-A-NCR, change priors, expand grids, or launch 10x5.

The current summary is internally inconsistent and must be reconstructed from raw split/seed outputs.

### Reported Working Memory
A4=0.2743, LF0=0.2707, LF1=0.2848, LF2=0.2840.
LF2-vs-strongest-no-prior: mean +0.0096, median +0.0096, positive seeds 1/3.

### Reported Fluid
A4=0.3238, LF0=0.3634, LF1=0.3794, LF2=0.3764.
LF2-vs-strongest-no-prior: mean +0.0130, median +0.0130, positive seeds 1/3.

With exactly 3 seed deltas, median>0 is mathematically impossible if only 1/3 deltas are positive. Also, 'LF0 beats A4 for both tasks' contradicts WM (0.2707<0.2743), and Fluid numbers imply LF0-A4=+0.0396, not +0.005. Baseline A4 values also differ from earlier frozen 3-seed evaluations.

## 1. Inspect
Inspect:
- src/metascfc/models/iclr_backbones/fc_only_msancr.py
- src/metascfc/experiments/prior_aware_late_fusion.py
- scripts/113_run_prior_aware_late_fusion.py
- configs/iclr/prior_aware_late_fusion.yaml
- tests/test_fc_only_msancr.py
- tests/test_prior_aware_late_fusion.py
- outputs/iclr/prior_aware_late_fusion/
- outputs/iclr/msancr_grid_closure/
- outputs/iclr/msancr_fluid_verification/

Write audit outputs to:
outputs/iclr/prior_aware_late_fusion_integrity_audit/

Never overwrite the original pilot.

## 2. Split identity
For WM and Fluid, seeds 0-2 and folds 0-4, compare train/test index hashes against the canonical prior corrected experiment.

WM reference: outputs/iclr/msancr_grid_closure/ (or exact canonical corrected 3-seed WM split manifest).
Fluid reference: outputs/iclr/msancr_fluid_verification/.

Create split_identity_audit.csv:
task,seed,fold,reference_train_hash,mod2_train_hash,reference_test_hash,mod2_test_hash,train_identical,test_identical.

## 3. A4 reproduction
Verify same:
- FC+SC input
- scaler
- target
- outer splits
- 3-fold inner CV
- ridge grids
- Pearson-first selection
- tie tolerance 0.002
- RMSE/MAE tie-breaking

Create a4_reproduction_audit.csv with old/new metrics and selected lambda_fc/lambda_sc for all 30 task-seed-fold rows.

Explain every baseline difference.

## 4. Recompute metrics independently
Do not trust late_fusion_decision.json.

From raw split-level outputs reconstruct:
A4, A3, LF0, LF1, LF2, LF2-null.

Aggregate:
5 folds -> seed mean.
3 seed means -> descriptive task summary.

Create:
- recomputed_split_metrics.csv
- recomputed_seed_metrics.csv
- recomputed_summary_metrics.csv

## 5. Mandatory seed-level table
Create seed_level_primary_models.csv:
task,seed,A4_pearson,LF0_pearson,LF1_pearson,LF2_pearson,
LF1_minus_A4,LF1_minus_LF0,LF2_minus_A4,LF2_minus_LF0,
plus oriented RMSE and MAE differences.

## 6. Define strongest no-prior correctly
Per task:
- compute three-seed mean Pearson for A4
- compute three-seed mean Pearson for LF0
- choose ONE global comparator identity, A4 or LF0
- use that same comparator for all 3 seed deltas

Do NOT switch comparator per seed or fold.

Create strongest_no_prior_definition.json.

## 7. Mathematical consistency
For LF1 and LF2 vs strongest no-prior compute from the same 3 seed deltas:
mean, median, positive_count, negative_count, zero_count.

Invariant:
if positive_count<=1, median must be <=0 (within tolerance).
if positive_count>=2, median must be >=0 unless ties explain zero.

Create paired_seed_reconstruction.csv.
Fail tests on inconsistency.

## 8. LF1 eligibility review
LF1 was PREDECLARED and is descriptively best in both tasks. It must receive a fair review.

Compute:
LF1 vs LF0
LF1 vs A4
LF1 vs strongest no-prior
LF2 vs LF0
LF2 vs A4
LF2 vs strongest no-prior

for each task.

Apply the same gates to LF1:

Large:
median delta >= +0.015
mean delta >= +0.012
positive seeds = 3/3
no material RMSE degradation

Promising:
median delta >= +0.008
positive seeds >= 2/3
no material RMSE degradation

Weak:
median delta < +0.005 OR positive seeds <=1/3

Do not alter thresholds.

## 9. Fusion weights
Verify all LF0/LF1/LF2 weights:
- nonnegative
- sum exactly to 1 within tolerance
- lie on 0.05 grid

Report LF1 prior and SC weight distributions.
Report LF2 FC, SC, prior weights.

For LF1 report:
fraction prior_weight=0
fraction prior_weight>=0.10
median prior_weight.

Create fusion_weight_integrity.csv and fusion_weight_distribution_recomputed.csv.

## 10. Leakage
Independently verify every Level-2 training feature is OOF.
Verify outer-test labels never influence Level-1 HP selection or fusion weights.

Create stacking_leakage_audit_v2.json.
Require n_failures=0.

## 11. LF2-null
Verify LF2-null equals LF0 per split within tolerance.
Create lf2_null_equivalence.csv.
If not, this is a material implementation bug.

## 12. LF2 fixed swaps
Verify matched/unrelated/shuffled/random:
- same seed/fold
- same F0
- same S
- same matched-selected fusion weights
- same prior-branch HPs
- only prior identity changes

Create fixed_prior_swap_integrity.csv.

## 13. LF1 controls only if eligible
If and only if:
- integrity checks pass;
- LF1 meets PROMISING gate for a task;
- LF1 fixed prior controls were not already run;

then complete Modification 2 with fixed-weight LF1 controls for that task:
LF1 matched / unrelated / shuffled / random.

Freeze matched-selected FP hyperparameters and matched-selected LF1 weights; change only prior identity and refit FP branch.

Do not retune controls.

Save under:
outputs/iclr/prior_aware_late_fusion_integrity_audit/lf1_fixed_prior_swaps/

## 14. Correctness-only rerun policy
A) Reporting bug only: fix reporting; no model rerun.
B) Baseline/model-selection bug: fix only bug and rerun exact Modification-2 3-seed pilot once to outputs/iclr/prior_aware_late_fusion_v2/.
C) Leakage/fusion bug: fix only correctness and rerun exact Modification 2 once.

No new formulas, grids, priors, or fusion resolution.

## 15. Final decision
If both tasks have a valid LF1 or LF2:
median delta >= +0.008,
positive seeds >=2/3,
no material RMSE degradation,
and proposed prior-aware architecture beats at least 2/3 prior controls,
then recommended_next_step=full_10x5_late_fusion_both_tasks.

If one task passes and one does not:
recommended_next_step=consider_modification_3.

If neither LF1 nor LF2 passes:
recommended_next_step=modification_3.

If ambiguous:
recommended_next_step=human_review.

If BOTH LF1 and LF2 qualify, prefer simpler LF1 unless LF2 median delta is >=0.005 larger with equal/better consistency.

Do NOT launch 10x5.
Do NOT implement Modification 3.

## 16. Biomarkers
For whichever prior-aware branch remains eligible, use exact FP FC coefficients for:
prior alignment, rank stability, top-10 Jaccard.
Do not invent fused coefficients.

## 17. Required outputs
Create:
- source_snapshot.json
- split_identity_audit.csv
- a4_reproduction_audit.csv
- recomputed_split_metrics.csv
- recomputed_seed_metrics.csv
- recomputed_summary_metrics.csv
- seed_level_primary_models.csv
- strongest_no_prior_definition.json
- paired_seed_reconstruction.csv
- fusion_weight_integrity.csv
- fusion_weight_distribution_recomputed.csv
- stacking_leakage_audit_v2.json
- lf2_null_equivalence.csv
- fixed_prior_swap_integrity.csv
- lf1_eligibility.json
- lf2_eligibility.json
- corrected_late_fusion_decision.json
- audit_findings.json
- COMPLETE

## 18. Tests
Add tests/test_prior_aware_late_fusion_integrity.py covering:
1. 3-seed median/positive-count consistency
2. global task-level strongest no-prior identity
3. numeric/text sign consistency
4. seed alignment
5. A4 parity
6. LF1/LF2 eligibility gates
7. LF2-null equivalence
8. weight simplex constraints
9. OOF leakage
10. fixed swaps differ only by prior
11. LF1 controls triggered only by eligibility
12. no new grids/formulas
13. Modification 3 cannot launch

Run all existing late-fusion/MS-A-NCR tests.

## Completion report
Print:

MODIFICATION 2 INTEGRITY AUDIT COMPLETE

Protocol:
WM split identity = .../15
Fluid split identity = .../15
OOF leakage = PASS/FAIL
LF2-null = PASS/FAIL

BASELINES
WM reference A4 = ...
WM Mod2 A4 = ...
cause if different = ...

Fluid reference A4 = ...
Fluid Mod2 A4 = ...
cause if different = ...

WORKING MEMORY
A4 = ...
LF0 = ...
LF1 = ...
LF2 = ...
strongest no-prior = ...

LF1 vs strongest seed deltas = [...]
mean = ...
median = ...
positive = .../3
status = ...

LF2 vs strongest seed deltas = [...]
mean = ...
median = ...
positive = .../3
status = ...

FLUID
A4 = ...
LF0 = ...
LF1 = ...
LF2 = ...
strongest no-prior = ...

LF1 vs strongest seed deltas = [...]
mean = ...
median = ...
positive = .../3
status = ...

LF2 vs strongest seed deltas = [...]
mean = ...
median = ...
positive = .../3
status = ...

LF1 fixed controls if triggered:
matched-unrelated = ...
matched-shuffled = ...
matched-random = ...

FINAL MODIFICATION-2 STATUS:
VALIDATED_SUCCESS / VALIDATED_PARTIAL / VALIDATED_FAILURE / CORRECTED_RERUN_REQUIRED

Recommended next step:
full_10x5_late_fusion_both_tasks / modification_3 / consider_modification_3 / human_review

NO MODIFICATION 3 IMPLEMENTED.
NO 10x5 RUN LAUNCHED.
