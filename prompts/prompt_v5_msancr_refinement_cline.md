# Cline Prompt — MS-A-NCR v2: Correctness Patch + Targeted Working-Memory Refinement

## Role and operating mode

You are acting as the senior research/code agent for an ICLR 2027 extension of MetaSFC.

Use strong repository-level reasoning before editing anything. Do not blindly follow this prompt if the existing implementation contradicts an assumption: inspect the relevant code and outputs first, document the discrepancy, then implement the scientifically correct interpretation.

Work autonomously, but keep the experiment narrow and efficient.

Do **not** launch the final 10-seed × 5-fold experiment in this step.

---

# Scientific context

The previous 3-seed MS-A-NCR pilot produced:

```text
Fluid Intelligence:
A0 standard Ridge            r ≈ 0.328
A4 modality-specific Ridge   r ≈ 0.357
A2 FC-Laplacian              r ≈ 0.358
A3 MS-A-NCR                  r ≈ 0.358
A3 - A4                      ≈ +0.001

Working Memory:
A0 standard Ridge            r ≈ 0.224
A4 modality-specific Ridge   r ≈ 0.255
A2 FC-Laplacian              r ≈ 0.258
A3 MS-A-NCR                  r ≈ 0.272
A3 - A4                      ≈ +0.017
median A3 - A4               ≈ +0.028
positive seeds               2/3
```

The Working-Memory result is promising.

However, repository inspection found several reasons **not to run 10×5 yet**:

1. `pilot_prior_types` currently contains only:
   ```text
   matched
   random
   ```
   so unrelated and shuffled controls were not actually run.

2. The implementation computes `D(q, gamma)` for every FC edge, but the solver applies the anisotropic diagonal only to **Laplacian-active FC edges**. Inactive FC edges receive ordinary isotropic Ridge:
   ```python
   K_inactive_fc = (1 / lambda_fc) X_inactive X_inactive^T
   ```
   This does not match the stated MS-A-NCR objective, where `D(q,gamma)` should cover the full FC block.

3. The pilot config only tested:
   ```text
   gamma = {0, 1}
   lambda_L = {0, 1}
   lifting = {prod}
   ```
   so the method-level prior hyperparameters were barely searched.

4. Hyperparameters are currently selected by validation RMSE even though Pearson correlation is the primary cognitive-prediction endpoint and the intended pilot specification used Pearson as the primary selection metric.

5. Hyperparameter selection uses one 15% validation split. With a richer grid this is likely too noisy.

6. A0 uses sklearn Ridge while A4/A1/A2/A3 use the custom dual formulation with a feature-count normalization. Therefore A0 and A4 do not provide a strict same-solver isotropic invariant. The primary prior claim must continue to use the strongest no-prior comparator A4, but solver equivalence should be explicitly checked.

7. Existing tests contain superficial assertions about configured grids but do not test the actual config/run path.

The goal of this step is to correct these issues and determine whether the Working-Memory gain survives under a cleaner specification.

---

# Main decision

Focus this refinement on:

```text
Working Memory only
seeds = [0, 1, 2]
outer folds = 5
```

Do not spend compute on Fluid Intelligence in this refinement.

If the corrected WM result passes the gate defined below, the next step will be a full 10×5 run.

If it fails, we will not keep tuning MS-A-NCR indefinitely; the next direction will be prior reconstruction / CT-MAC.

---

# Files to inspect first

Before changing code, inspect at minimum:

```text
src/metascfc/models/iclr_backbones/modality_selective_anisotropic_ncr.py
src/metascfc/models/iclr_backbones/network_constrained_ridge.py
scripts/104_run_msancr_pilot.py
configs/iclr/msancr_pilot.yaml
tests/test_msancr.py

outputs/iclr/msancr_pilot/
outputs/iclr/conditional_prior_signal_v2/
```

Also inspect the actual split utility used by:

```text
metascfc.benchmark_utils.iter_nested_splits
```

and confirm exactly how train/validation/test indices are formed.

Write a short internal audit summary to:

```text
outputs/iclr/msancr_refinement/audit_before_changes.json
```

containing the issues you confirmed and any assumptions you had to revise.

---

# Part 1 — Correct the MS-A-NCR objective

The intended full model is:

\[
\min_{\beta}
\|y-X\beta\|_2^2
+
\lambda_{FC}\beta_{FC}^{T}D(q;\gamma)\beta_{FC}
+
\lambda_{SC}\|\beta_{SC}\|_2^2
+
\lambda_L\beta_{FC}^{T}L_q\beta_{FC}.
\]

Where:

```text
X = [X_FC | X_SC]
```

and:

\[
D_{ee}(q;\gamma)
=
(\epsilon + q_e)^{-\gamma}.
\]

Normalize:

\[
D \leftarrow D / mean(D).
\]

Use:

```text
epsilon = 1e-3
```

High-prior FC edges must receive less shrinkage.

---

# Part 2 — Apply anisotropy to the FULL FC block

This is mandatory.

The current solver only uses `D_active` for Laplacian-active edges and ordinary Ridge for inactive edges.

Correct it so that:

```text
all FC features
```

receive their corresponding diagonal prior penalty.

Conceptually:

\[
P_{FC}
=
\lambda_{FC}D
+
\lambda_L L_{embedded}.
\]

`L_embedded` is zero outside the Laplacian-active edge subset.

Thus:

- active FC edges:
  ```text
  lambda_fc * D_e + Laplacian contribution
  ```

- inactive FC edges:
  ```text
  lambda_fc * D_e
  ```

NOT:

```text
lambda_fc * 1
```

unless `gamma = 0`.

This correction must apply consistently in both fitting and prediction kernels.

---

# Part 3 — Efficient exact solver

Preserve an exact generalized-Ridge solution.

Do not switch to SGD.

For inactive FC features, because the penalty is diagonal:

\[
K_{inactive}
=
X_{inactive}
\operatorname{diag}\left(
\frac{1}{\lambda_{FC}D_e}
\right)
X_{inactive}^{T}.
\]

For the active FC block:

\[
P_{active}
=
\lambda_{FC}D_{active}
+
\lambda_L L_{active}.
\]

Use a stable eigendecomposition / solve.

SC remains:

\[
P_{SC}=\lambda_{SC}I.
\]

Preserve the p >> n dual/Woodbury strategy.

---

# Part 4 — Validate the mathematical implementation against a direct primal solution

Add a small synthetic test where p is small enough to compute:

\[
\hat\beta_{direct}
=
(X^TX+P)^{-1}X^Ty.
\]

Compare predictions from the optimized dual solver against the direct primal solution.

Require tight numerical agreement, e.g.:

```text
atol <= 1e-6
rtol <= 1e-5
```

Test cases must include:

1. gamma = 0, lambda_L = 0
2. gamma > 0, lambda_L = 0
3. gamma = 0, lambda_L > 0
4. gamma > 0, lambda_L > 0
5. inactive edges with non-uniform D

The last test must fail under the old implementation and pass under the corrected one.

---

# Part 5 — Add a same-solver isotropic baseline

Create a baseline:

```text
A4_iso
```

or equivalent descriptive name.

It must use the **same custom dual solver path** as MS-A-NCR with:

```text
gamma = 0
lambda_L = 0
lambda_fc = lambda_sc
uniform prior
```

This is a solver sanity baseline.

It should numerically agree with the corresponding generalized-Ridge solution.

Do not necessarily replace A0 in tables; A0 can remain as the historical sklearn Ridge baseline.

The scientifically important no-prior comparator remains:

```text
A4 modality-specific Ridge
```

because it is stronger.

---

# Part 6 — Use all four prior identities for A3

For the refined A3 evaluation include:

```text
matched
unrelated
shuffled
random
```

The actual files already exist in the config.

Do not silently omit any control.

For the main method comparison, tune the matched prior normally.

For the strongest causal/ablation-style prior-identity check, also run a **prior-swap evaluation**:

1. select A3 hyperparameters using the matched prior and inner CV;
2. freeze:
   ```text
   lambda_fc
   lambda_sc
   gamma
   lambda_L
   lifting
   ```
3. substitute:
   ```text
   unrelated
   shuffled
   random
   ```
4. refit on the same training subjects;
5. evaluate on the same outer test fold.

This directly asks whether the learned gain comes from the matched prior identity rather than from hyperparameter search freedom.

Save both:

```text
retuned-control results
fixed-hyperparameter prior-swap results
```

if runtime permits.

If runtime is a concern, the fixed-hyperparameter prior-swap is mandatory; fully retuned controls are secondary.

---

# Part 7 — Expand only the method-level prior grid

Do not perform an enormous Cartesian search.

Use:

```yaml
lifting_rules:
  - prod
  - mean

gamma_grid:
  - 0.25
  - 0.5
  - 1.0
  - 2.0

lambda_laplacian_grid:
  - 0.1
  - 0.5
  - 1.0
  - 2.0
  - 5.0
```

For A3, gamma must be > 0 and lambda_L > 0.

For A2, gamma = 0.

For A1, lambda_L = 0.

---

# Part 8 — Efficient staged tuning

Use a staged strategy to control runtime.

## Stage A — select A4 no-prior modality penalties

Using inner CV, select:

```text
lambda_fc
lambda_sc
```

from the existing validated Ridge grid.

Recommended grid:

```text
0.1
1.0
10.0
```

unless repository evidence strongly supports restoring:

```text
0.01
100.0
```

If boundary selections are frequent, automatically expand one step and document it.

---

## Stage B — select A3 prior geometry with A4 modality penalties fixed

With `lambda_fc/lambda_sc` fixed to Stage-A values, select:

```text
lifting
gamma
lambda_L
```

using the grid above.

This is only:

```text
2 × 4 × 5 = 40
```

prior-geometry configurations.

---

## Stage C — local coordinate refinement

After selecting the best:

```text
lifting
gamma
lambda_L
```

allow a small local refinement of `lambda_fc/lambda_sc`.

Test at most:

```text
one lower grid neighbor
selected value
one higher grid neighbor
```

for each modality.

Do not reopen the full Cartesian product.

---

# Part 9 — Replace single 15% validation selection with 3-fold inner CV

For this refinement use:

```text
3-fold inner CV
```

within every outer-training partition.

The outer test fold remains untouched.

All preprocessing/scaling must be fitted independently inside each inner training fold.

Use deterministic splits based on outer seed/fold.

If groups/family IDs are available, use them consistently; otherwise preserve the existing subject-wise limitation and record it.

---

# Part 10 — Primary model-selection metric must be Pearson

Select hyperparameters using:

```text
mean inner-validation Pearson r
```

across the 3 inner folds.

Tie-breakers:

```text
1. lower mean RMSE
2. lower mean MAE
3. simpler configuration
```

Define a numerical tie tolerance, e.g.:

```text
abs(delta Pearson) < 0.002
```

before applying tie-breakers.

Use the exact same selection rule for A4 and A3.

Do not use outer-test performance for selection.

---

# Part 11 — Models to run in this refinement

Working Memory only.

Run:

```text
B0 / A4:
modality-specific no-prior Ridge

B1 / A2:
FC-only Laplacian NCR + ordinary SC Ridge

B2 / A3:
corrected full-FC MS-A-NCR with matched prior
```

Also run:

```text
B2 prior-swap:
unrelated
shuffled
random
```

with matched-selected hyperparameters frozen.

A1 anisotropic-only is optional because the first pilot already showed it was weak.

If cheap, retain it as a mechanistic ablation; otherwise skip it in this targeted refinement.

---

# Part 12 — Preserve biomarker outputs

For corrected matched A3 save the FC coefficient vector and derived ROI saliency for every outer split.

Report:

```text
prior alignment
rank stability
top-10 Jaccard
```

using existing project definitions.

Also compute them for:

```text
A4
A3 unrelated
A3 shuffled
A3 random
```

where technically meaningful.

Do not run the expensive full perturbation-faithfulness experiment yet.

Prediction must pass the gate first.

---

# Part 13 — Outputs

Write to a new directory:

```text
outputs/iclr/msancr_refinement/
```

Do not overwrite:

```text
outputs/iclr/msancr_pilot/
```

Create at minimum:

```text
audit_before_changes.json
split_metrics.csv
seed_metrics.csv
summary_metrics.csv

inner_cv_metrics.csv
selected_hyperparameters.csv
boundary_selection_report.csv

prior_swap_split_metrics.csv
prior_swap_seed_metrics.csv
prior_swap_summary.csv

biomarker_metrics.csv
paired_comparisons.csv

refinement_decision.json
run_metadata.json
COMPLETE
```

Figures:

```text
figures/iclr/msancr_refinement/
```

Create:

```text
wm_model_comparison.pdf
wm_seed_delta_vs_A4.pdf
wm_matched_vs_prior_swaps.pdf
wm_selected_gamma_lambdaL.pdf
wm_biomarker_stability.pdf
```

---

# Part 14 — Required paired comparisons

At seed level report:

```text
A3 matched - A4
A3 matched - A2
A3 matched - A3 unrelated-fixed
A3 matched - A3 shuffled-fixed
A3 matched - A3 random-fixed
```

For n=3 this is descriptive only.

Report:

```text
mean delta Pearson
median delta Pearson
positive seeds / 3

mean delta RMSE
median delta RMSE

mean delta MAE
median delta MAE
```

Do not make significance claims with only three seeds.

---

# Part 15 — Hard decision gate

Create:

```text
refinement_decision.json
```

with the following logic.

## GO — full 10×5

Proceed if corrected matched A3 satisfies BOTH:

```text
median delta Pearson vs A4 >= +0.010
positive seeds >= 2/3
```

AND preferably:

```text
mean delta Pearson >= +0.010
```

AND there is no material average RMSE degradation.

Additionally, at least two of these should hold:

```text
matched > unrelated-fixed
matched > shuffled-fixed
matched > random-fixed
```

in mean/median seed delta.

Output:

```text
recommended_next_step = full_10x5_msancr
```

---

## BORDERLINE — one final tiny refinement

If:

```text
+0.005 <= median delta Pearson vs A4 < +0.010
```

with >= 2/3 positive seeds:

```text
recommended_next_step = one_final_small_refinement
```

Do not start 10×5 automatically.

---

## STOP MS-A-NCR

If:

```text
median delta Pearson vs A4 < +0.005
```

or only 1/3 or 0/3 seeds improve:

```text
recommended_next_step = ct_mac_prior_rebuild
```

Do not continue hyperparameter fishing.

---

# Part 16 — Explicitly diagnose component contribution

Report whether:

```text
A3 > A2 > A4
```

or another ordering occurs.

Interpret:

### If A3 > A2 > A4

```text
anisotropy + Laplacian are complementary
```

### If A3 ~= A2 > A4

```text
Laplacian drives the gain; simplify method
```

### If A2 ~= A4 and A3 > A4

```text
interaction between anisotropy and Laplacian drives gain
```

### If A4 >= A3

```text
prior does not improve prediction under corrected implementation
```

This diagnostic must be written to the decision JSON.

---

# Part 17 — Tests

Update/create:

```text
tests/test_msancr_refinement.py
```

Tests must verify actual implementation behavior, not hard-coded example lists.

Required tests:

1. full FC D is used for inactive and active FC edges;
2. inactive non-uniform D changes predictions;
3. old active-only implementation would fail the direct-primal equivalence test;
4. optimized dual equals direct primal on small synthetic data;
5. gamma=0 gives isotropic FC penalty;
6. lambda_L=0 removes Laplacian contribution;
7. uniform prior reduces anisotropy to isotropic shrinkage;
8. SC never receives the cognitive Laplacian;
9. A4 same-solver no-prior recovery;
10. actual runtime config contains matched/unrelated/shuffled/random when controls are requested;
11. prod and mean lifting both execute;
12. outer-test indices never appear in inner hyperparameter scoring;
13. 3-fold inner CV produces exactly one OOF validation prediction per inner validation subject where applicable;
14. selection uses Pearson first;
15. RMSE is only a tie-breaker;
16. prior-swap controls reuse the exact matched-selected hyperparameters;
17. scaler is fitted inner-training only;
18. all existing MS-A-NCR tests continue to pass or are updated only where the old behavior was mathematically incorrect.

Run:

```bash
pytest -q tests/test_msancr.py tests/test_msancr_refinement.py
```

Also run relevant existing NCR/generalized-Ridge tests.

---

# Part 18 — Runtime strategy

This is running through Cline with a strong reasoning model, but compute time still matters.

Optimize implementation before launching the experiment:

- cache prior liftings;
- cache Laplacians;
- cache eigendecompositions when mathematically valid;
- cache standardized inner-fold data;
- reuse A4 Stage-A results;
- do not recompute fixed prior-independent quantities;
- implement resume by target/seed/fold/model/prior;
- use atomic CSV/JSON checkpoints.

Do not use invalid caching keys.

In particular, if an eigendecomposition depends on absolute `lambda_fc` as well as `lambda_L/lambda_fc`, make sure cached eigenvalues are not incorrectly reused across scaled penalty matrices.

Add a test for cache correctness against no-cache execution.

---

# Part 19 — Before launching the expensive run

Cline must first execute a synthetic/small smoke test.

Then run exactly:

```text
Working Memory
seed = 0
fold = 0
```

for:

```text
A4
A2
A3 matched
A3 random fixed-prior-swap
```

Verify:

- no leakage;
- finite metrics;
- direct/dual solver checks pass;
- selected configs are sensible;
- output schemas are correct.

Only then launch:

```text
Working Memory
seeds = [0,1,2]
5 outer folds
```

---

# Part 20 — Completion report

At completion, print a concise report containing:

```text
A4 Pearson
A2 Pearson
A3 matched Pearson

A3 - A4 mean delta
A3 - A4 median delta
positive seeds

matched - unrelated fixed
matched - shuffled fixed
matched - random fixed

best gamma
best lambda_L
best lifting

biomarker alignment/stability summary

decision
```

Do not merely print “all tasks complete”.

Explain any deviation from the requested design.

---

# Files to return for review

Return at minimum:

```text
outputs/iclr/msancr_refinement/refinement_decision.json
outputs/iclr/msancr_refinement/summary_metrics.csv
outputs/iclr/msancr_refinement/seed_metrics.csv
outputs/iclr/msancr_refinement/paired_comparisons.csv
outputs/iclr/msancr_refinement/selected_hyperparameters.csv
outputs/iclr/msancr_refinement/prior_swap_summary.csv
outputs/iclr/msancr_refinement/biomarker_metrics.csv
outputs/iclr/msancr_refinement/audit_before_changes.json
```

Also return the full output directory if practical.

---

# Non-negotiable instruction

Do **not** launch the final 10-seed × 5-fold run automatically even if the refinement passes.

Stop after the 3-seed refinement, write the decision, and wait for review.
