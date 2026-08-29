# Cline / GLM 5.3 Flash Prompt — MS-A-NCR Pre-Full-Run Grid Closure

## Role

You are working inside the existing `metaSFC_extends` repository.

This is **not** a redesign task.

The MS-A-NCR v2 solver, objective, Working-Memory target, data cohort, outer splits, 3-fold inner CV, Pearson-first selection rule, fixed prior-swap logic, and biomarker extraction are now considered **frozen** unless a genuine correctness bug is discovered.

Your task is only to close the remaining hyperparameter-boundary issue before the final 10-seed × 5-fold experiment.

Do **not** launch the final 10×5 run in this step.

Because this will be executed using GLM 5.3 Flash inside Cline, be explicit, conservative, and verify every assumption from the repository before editing.

---

# Current validated refinement result

The corrected 3-seed Working-Memory refinement produced:

```text
A4 modality-specific Ridge:  mean Pearson = 0.268983
A2 FC-Laplacian:             mean Pearson = 0.274184
A3 corrected MS-A-NCR:       mean Pearson = 0.281756
```

A3 matched versus A4:

```text
mean ΔPearson   = +0.012773
median ΔPearson = +0.012906
positive seeds  = 3/3

mean ΔRMSE      = -0.079278
mean ΔMAE       = -0.088660
```

A3 also beats A2:

```text
mean ΔPearson   = +0.007572
median ΔPearson = +0.005940
positive seeds  = 3/3
```

Fixed matched-selected prior swaps:

```text
matched - unrelated : +0.027226 mean ΔPearson
matched - shuffled  : +0.019139
matched - random    : +0.017069
```

Thus the method passed the requested predictive gate.

---

# Why another tiny step is needed

The inner-CV hyperparameter audit still shows unresolved boundaries.

For A4:

```text
final lambda_fc = 0.01
```

in 4/15 outer splits, where `0.01` is the minimum expanded value.

Those same four A3 selections also use:

```text
lambda_fc = 0.01
```

Additionally, A3 selects:

```text
lambda_L = 0.1
```

in 5/15 splits, where 0.1 is the current minimum.

A3 selects:

```text
gamma = 0.25
```

in 3/15 splits, where 0.25 is the current minimum.

This means the final 10×5 experiment should not be launched with the present truncated search grid.

This is an **inner-CV boundary closure**, not outer-test hyperparameter tuning.

---

# Files to inspect first

Inspect:

```text
configs/iclr/msancr_refinement.yaml
src/metascfc/experiments/msancr_refinement.py
src/metascfc/models/iclr_backbones/modality_selective_anisotropic_ncr.py
scripts/105_run_msancr_refinement.py

outputs/iclr/msancr_refinement/
```

Read specifically:

```text
boundary_selection_report.csv
selected_hyperparameters.csv
inner_cv_metrics.csv
seed_metrics.csv
paired_comparisons.csv
refinement_decision.json
```

Confirm the boundary counts from the actual files before modifying anything.

Write:

```text
outputs/iclr/msancr_grid_closure/precheck.json
```

with the confirmed counts.

---

# Non-negotiable preservation constraints

Do not change:

```text
Working-Memory target
subject cohort
outer split identities
seeds 0,1,2
5 outer folds
3-fold inner CV
Pearson-first selection
Pearson tie tolerance = 0.002
RMSE/MAE tie-break order
feature standardization
MS-A-NCR solver mathematics
full-FC anisotropic D
FC-only Laplacian
SC ordinary Ridge
prior files
prior lifting formulas
fixed prior-swap definition
biomarker definitions
```

Do not touch:

```text
network_constrained_ridge.py
Audit 100
Audit 101 v1/v2
previous msancr_refinement outputs
```

---

# New output directory

Use:

```text
outputs/iclr/msancr_grid_closure/
```

and:

```text
figures/iclr/msancr_grid_closure/
```

Never overwrite:

```text
outputs/iclr/msancr_refinement/
```

---

# Step 1 — Expand only boundaries justified by the existing INNER-CV results

Use the following final candidate values.

## Modality Ridge grid

Use:

```yaml
ridge_grid_final:
  - 0.001
  - 0.01
  - 0.1
  - 1.0
  - 10.0
  - 100.0
```

Do not add arbitrary intermediate values.

The sole new Ridge value is:

```text
0.001
```

because the previous minimum `0.01` was selected in 4/15 splits.

---

## Gamma grid

Use:

```yaml
gamma_grid_final:
  - 0.1
  - 0.25
  - 0.5
  - 1.0
  - 2.0
```

The sole new value is:

```text
0.1
```

because the previous minimum `0.25` was selected in 3/15 A3 splits.

---

## Laplacian grid

Use:

```yaml
lambda_laplacian_grid_final:
  - 0.03
  - 0.1
  - 0.5
  - 1.0
  - 2.0
  - 5.0
```

The sole new value is:

```text
0.03
```

because the previous minimum `0.1` was selected in 5/15 A3 splits.

---

## Lifting rules

Keep exactly:

```yaml
lifting_rules:
  - prod
  - mean
```

Do not add `max`, `bridge`, or any new prior transformation.

---

# Step 2 — Keep staged selection

Preserve the staged selection logic.

## Stage A

Select:

```text
lambda_fc
lambda_sc
```

for A4 using the final Ridge grid.

Use the same 3-fold inner CV and Pearson-first criterion.

---

## Stage B

With selected Stage-A modality penalties fixed, select A3 prior geometry:

```text
lifting
gamma
lambda_L
```

using:

```text
2 liftings × 5 gamma × 6 lambda_L = 60 configurations
```

This is acceptable.

A2 selects lambda_L from the same final Laplacian grid with gamma = 0.

---

## Stage C

Perform the same local coordinate refinement around the selected A4:

```text
one lower neighbor
selected value
one upper neighbor
```

for `lambda_fc` and `lambda_sc`.

Do not reopen a full 6×6 grid after Stage B.

---

# Step 3 — Boundary reporting must be explicit

For every outer split report whether the final selected value equals:

```text
minimum or maximum lambda_fc
minimum or maximum lambda_sc
minimum or maximum gamma
minimum or maximum lambda_L
```

Create:

```text
boundary_selection_report_v2.csv
```

with columns:

```text
seed
fold

A4_lambda_fc
A4_lambda_sc
A4_lambda_fc_boundary
A4_lambda_sc_boundary

A2_lambda_L
A2_lambda_L_boundary

A3_lambda_fc
A3_lambda_sc
A3_gamma
A3_lambda_L
A3_lifting

A3_lambda_fc_boundary
A3_lambda_sc_boundary
A3_gamma_boundary
A3_lambda_L_boundary
```

Do not recursively expand beyond the values specified in this prompt.

If the new minimum is selected, flag it as:

```text
unresolved_lower_boundary = true
```

but do not add another smaller value.

This experiment exists to determine whether such unresolved boundaries remain.

---

# Step 4 — Re-run exactly the same 3-seed WM refinement

Run:

```text
Working Memory only
seeds = [0,1,2]
5 outer folds
3-fold inner CV
```

Models:

```text
A4 modality-specific no-prior Ridge
A2 FC-only Laplacian NCR
A3 corrected MS-A-NCR matched
```

Also run the same fixed-hyperparameter A3 prior swaps:

```text
unrelated
shuffled
random
```

The control swaps must reuse each split's matched-selected:

```text
lambda_fc
lambda_sc
gamma
lambda_L
lifting
```

exactly.

---

# Step 5 — Do not use old outer-test results in the new selection

The old `msancr_refinement` outer-test values may be used only for AFTER-RUN comparison.

They must never influence:

```text
candidate ranking
grid pruning
hyperparameter selection
prior selection
```

The new grid values are already fixed by this prompt based solely on prior inner-CV boundary evidence.

---

# Step 6 — Same solver and coefficients

Use the current corrected MS-A-NCR solver unchanged.

Verify again:

```text
dual prediction == recovered primal coefficient prediction
```

for every final outer model within existing numerical tolerance.

Do not return to the old `X^T alpha` saliency proxy.

Use exact recovered coefficients.

---

# Step 7 — Compare old refinement versus grid closure

Create:

```text
grid_closure_comparison.csv
```

containing per seed:

```text
old_A4_pearson
new_A4_pearson
old_A3_pearson
new_A3_pearson

old_A3_minus_A4
new_A3_minus_A4
```

Also report aggregate changes:

```text
Δ new A4 versus old A4
Δ new A3 versus old A3
change in A3-A4 margin
```

This comparison is descriptive only.

Do not select between old/new configurations using outer performance; the new grid is the final intended grid regardless.

---

# Step 8 — Primary outputs

Create:

```text
outputs/iclr/msancr_grid_closure/
```

with:

```text
precheck.json
split_metrics.csv
seed_metrics.csv
summary_metrics.csv

inner_cv_metrics.csv
selected_hyperparameters.csv
boundary_selection_report_v2.csv

prior_swap_split_metrics.csv
prior_swap_seed_metrics.csv
prior_swap_summary.csv

biomarker_metrics.csv
paired_comparisons.csv
grid_closure_comparison.csv

grid_closure_decision.json
run_metadata.json
COMPLETE
```

---

# Step 9 — Decision criteria

Create:

```text
grid_closure_decision.json
```

## GO to full 10×5

Recommend:

```text
full_10x5_msancr
```

if:

```text
new A3 matched median ΔPearson vs A4 >= +0.010
positive seeds >= 2/3
mean ΔPearson >= +0.008
no material RMSE degradation
```

and:

```text
matched > at least 2/3 fixed prior swaps
```

Also require that the result is not entirely explained by one seed:

```text
at least 2 seeds must individually have ΔPearson > 0
```

---

## Borderline

If:

```text
+0.005 <= median ΔPearson < +0.010
```

with at least 2/3 positive seeds:

```text
recommended_next_step = review_before_full_run
```

Do not perform another hyperparameter search automatically.

---

## Stop

If:

```text
median ΔPearson < +0.005
```

or fewer than 2/3 seeds improve:

```text
recommended_next_step = ct_mac_prior_rebuild
```

---

# Step 10 — Boundary interpretation for the final run

The decision JSON must separately state:

```text
ridge_grid_closed
gamma_grid_closed
lambda_L_grid_closed
```

Definitions:

### Grid closed

`true` if no more than 20% of outer splits select the new lower/upper boundary for that parameter.

With 15 splits:

```text
<= 3/15 boundary selections
```

counts as closed.

### Grid unresolved

If:

```text
>= 4/15
```

select the new boundary, set:

```text
grid_closed = false
```

Do not automatically expand further.

Instead note it for final review.

---

# Step 11 — Freeze a final 10×5 config, but DO NOT run it

If the GO gate passes, create:

```text
configs/iclr/msancr_final_10x5.yaml
```

containing:

```yaml
target: working_memory
seeds: [0,1,2,3,4,5,6,7,8,9]
n_outer_folds: 5
n_inner_folds: 3

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

Include all other frozen settings from the grid-closure run.

Do **not** execute this config.

---

# Step 12 — Prepare the final-run script without executing

Create if needed:

```text
scripts/106_run_msancr_final.py
```

It should be able to run:

```bash
python scripts/106_run_msancr_final.py \
  --config configs/iclr/msancr_final_10x5.yaml
```

But this prompt explicitly forbids launching it.

The script must support:

```text
resume
config hash
atomic checkpoints
seed/fold granularity
```

---

# Step 13 — Tests

Add/update:

```text
tests/test_msancr_grid_closure.py
```

Test:

1. production config contains `0.001` Ridge;
2. production config contains gamma `0.1`;
3. production config contains lambda_L `0.03`;
4. no other unrequested grid expansion occurred;
5. selection remains Pearson-first;
6. same 3-fold inner CV is used;
7. outer-test labels are not used for selection;
8. fixed control swaps reuse matched-selected hyperparameters exactly;
9. boundary flags are correctly computed;
10. grid-closed threshold is correctly computed;
11. final 10×5 config is generated only after GO;
12. final runner is not automatically invoked;
13. old refinement outputs are not overwritten;
14. direct/dual solver invariants still pass;
15. all existing MS-A-NCR refinement tests still pass.

Run:

```bash
pytest -q \
  tests/test_msancr.py \
  tests/test_msancr_refinement.py \
  tests/test_msancr_grid_closure.py
```

and relevant NCR/generalized-Ridge tests.

---

# Step 14 — Runtime optimization

Reuse correct prior-independent caches where mathematically valid.

Do not modify solver mathematics for speed.

Cache keys must include every quantity that affects the cached object.

Maintain config-hash protection.

Do not reuse old hyperparameter selections.

---

# Step 15 — Completion report

At completion print:

```text
Old A4 Pearson
New A4 Pearson

Old A3 Pearson
New A3 Pearson

New A3 - A4:
  mean ΔPearson
  median ΔPearson
  positive seeds
  mean ΔRMSE
  mean ΔMAE

New A3 - A2

Matched - unrelated
Matched - shuffled
Matched - random

Selected gamma distribution
Selected lambda_L distribution
Selected lifting distribution
Selected lambda_fc/lambda_sc distribution

Boundary counts after expansion

Biomarker:
  WM alignment
  rank stability
  top-10 Jaccard

Final recommendation
```

If any result differs materially from the previous refinement, explain why.

---

# Non-negotiable final instruction

Do not launch the 10-seed × 5-fold experiment.

Stop after:

```text
grid closure
decision artifact
final config preparation
final runner preparation
```

and wait for review.
