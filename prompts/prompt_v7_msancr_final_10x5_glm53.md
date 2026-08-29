# Cline / GLM 5.3 Flash Prompt — Final MS-A-NCR 10×5 Run + Proper Seed-Level Inference

## Role

You are operating inside the existing `metaSFC_extends` repository.

This is the **final confirmatory experiment** for the Working-Memory MS-A-NCR method.

The method and hyperparameter search space are now frozen.

Do not redesign the model.
Do not expand grids.
Do not alter priors.
Do not alter targets.
Do not alter outer splits.
Do not alter inner-CV selection.
Do not alter solver mathematics.

Your job is:

1. make one small post-processing/inference correction for the 10-seed setting;
2. run validation/smoke checks;
3. launch the frozen 10-seed × 5-fold experiment;
4. compute proper seed-level inferential statistics;
5. produce final paper-ready result artifacts.

This prompt **authorizes the final 10×5 run** after the preflight checks pass.

---

# Scientific state before this run

The closed-grid 3-seed Working-Memory refinement produced:

```text
A4 modality-specific Ridge          Pearson = 0.268134
A2 FC-Laplacian                     Pearson = 0.272738
A3 corrected MS-A-NCR matched       Pearson = 0.279587
```

A3 matched vs A4:

```text
mean ΔPearson   = +0.011452
median ΔPearson = +0.012906
positive seeds  = 3/3

mean ΔRMSE      = -0.093605
mean ΔMAE       = -0.098447
```

A3 matched vs A2:

```text
mean ΔPearson   = +0.006848
median ΔPearson = +0.004078
positive seeds  = 3/3
```

Matched-selected fixed prior swaps:

```text
matched - unrelated = +0.026126 mean ΔPearson
matched - shuffled  = +0.016310
matched - random    = +0.015725
```

Therefore the predictive pilot gate has passed.

---

# Important interpretation of the remaining lambda_L boundary

The grid-closure artifact currently reports:

```text
lambda_L_grid_closed = false
```

because it combines lower-boundary hits from:

```text
A2 + A3
```

For the proposed A3 model itself:

```text
lambda_L = 0.03 in exactly 3/15 splits
```

which is exactly the stated closure threshold:

```text
<= 3/15
```

Therefore:

> Do not expand lambda_L again.

A2 is an ablation, not the proposed final model.

Before the full run, correct/report this distinction in metadata only.

Do not change the frozen search grid.

---

# Frozen final configuration

Use the existing:

```text
configs/iclr/msancr_final_10x5.yaml
```

Expected core settings:

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

Verify rather than silently overwrite.

If the config differs materially, stop and report the discrepancy.

---

# Frozen method definition

A3 MS-A-NCR remains:

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
D(q;gamma) applies to the full FC block
L_q applies only to FC
SC receives ordinary Ridge
```

The exact corrected dual/generalized-Ridge solver is frozen.

Do not modify it unless a preflight test reveals a genuine correctness failure.

---

# Part 1 — Fix final-run decision/inference logic BEFORE launching

The current final runner calls the refinement pipeline with:

```python
enforce_seed_gate=False
```

but the downstream `make_refinement_decision()` still contains language and thresholds designed for:

```text
n = 3
```

including:

```text
"inference_status": "descriptive_only_n_equals_3_no_significance_claim"
```

and a positive-seed threshold of 2.

This must not be used for the final paper interpretation.

Do one of the following:

## Preferred

Create a separate final inference module:

```text
src/metascfc/experiments/msancr_final_inference.py
```

and optionally:

```text
scripts/107_analyze_msancr_final.py
```

Do not disturb the validated refinement logic.

The final runner may call this new analyzer automatically after all 50 outer folds complete.

---

# Part 2 — Statistical unit

For every method/prior:

1. average the five outer folds within each seed;
2. obtain exactly 10 seed-level values;
3. perform inference on the 10 paired seed values.

Do **not** treat 50 folds as independent statistical observations.

Primary inferential unit:

```text
n = 10 seeds
```

---

# Part 3 — Final prediction comparisons

The primary method comparison is:

```text
A3 matched MS-A-NCR
vs
A4 modality-specific no-prior Ridge
```

Secondary mechanistic comparison:

```text
A3 matched
vs
A2 FC-Laplacian
```

Prior-specificity comparisons:

```text
A3 matched
vs
A3 unrelated-fixed

A3 matched
vs
A3 shuffled-fixed

A3 matched
vs
A3 random-fixed
```

The fixed swaps must use the exact matched-selected hyperparameters for each outer split, as already implemented.

Do not retune control priors in the final confirmatory experiment.

---

# Part 4 — Prediction statistics

For each comparison and metric:

```text
Pearson
RMSE
MAE
```

compute:

### Descriptives

```text
left mean
left SD
right mean
right SD

paired mean difference
paired median difference
paired SD

positive-difference seeds
negative-difference seeds
zero seeds
```

For RMSE/MAE, additionally provide a directional field where positive means **A3 is better**:

\[
improvement_{RMSE}=RMSE_{right}-RMSE_{A3}
\]

\[
improvement_{MAE}=MAE_{right}-MAE_{A3}.
\]

Avoid sign ambiguity.

---

# Part 5 — Wilcoxon signed-rank inference

Use:

```text
two-sided paired Wilcoxon signed-rank test
```

on the 10 seed-level paired values.

Report:

```text
W
raw p
```

Handle exact zeros deterministically using an explicitly documented SciPy `zero_method`.

Prefer:

```python
zero_method="wilcox"
```

unless repository constraints require otherwise.

---

# Part 6 — Holm correction

Apply Holm correction separately for each prediction metric across the five final comparisons:

```text
A3 vs A4
A3 vs A2
A3 vs unrelated
A3 vs shuffled
A3 vs random
```

Thus:

```text
5 Pearson hypotheses
5 RMSE hypotheses
5 MAE hypotheses
```

Do not mix Pearson/RMSE/MAE into one Holm family.

Report:

```text
p_holm_metric
significant_holm_005
```

The primary A3-vs-A4 raw p-value should also be retained explicitly.

---

# Part 7 — Bootstrap paired confidence intervals

For every paired comparison calculate a 95% bootstrap CI for the **mean seed-level paired difference**.

Use:

```text
>= 10,000 bootstrap resamples
deterministic RNG seed
resample paired seed differences
```

Report:

```text
ci95_low
ci95_high
ci_excludes_zero
```

Do not bootstrap the 50 fold rows independently.

---

# Part 8 — Effect sizes

Report paired Cohen's dz:

\[
d_z
=
\frac{\overline{d}}
{s_d}
\]

where `d` is the paired seed-level difference.

For RMSE/MAE use the improvement-oriented sign convention so:

```text
positive dz = A3 better
```

Also report the ordinary left-minus-right dz if useful internally, but paper-facing outputs should use consistent positive-is-better direction.

---

# Part 9 — Optional paired t-test as sensitivity analysis

You may additionally report:

```text
paired t-test p
```

as a secondary sensitivity statistic.

It must not replace Wilcoxon as the primary inferential test.

If included, label it clearly as secondary.

---

# Part 10 — Biomarker inference

Use the already generated seed-level biomarker metrics:

```text
WM prior alignment
rank stability
top-10 Jaccard
```

Primary biomarker comparisons:

```text
A3 matched vs A4
A3 matched vs A2
A3 matched vs unrelated-fixed
A3 matched vs shuffled-fixed
A3 matched vs random-fixed
```

Use the same:

```text
10 seed-level paired observations
two-sided Wilcoxon
metric-wise Holm correction
bootstrap 95% CI
Cohen dz
```

Apply Holm separately for:

```text
WM alignment family
rank stability family
top-10 Jaccard family
```

Do not infer using fold-pair rows as if independent.

---

# Part 11 — Final hypothesis interpretation

The final output must separately assess:

## Prediction hypothesis

Does matched A3 improve over the strongest no-prior A4?

Define:

```text
prediction_supported = true
```

only if:

```text
mean/median Pearson difference > 0
and
Wilcoxon p_holm_metric < 0.05
```

for the A3-vs-A4 Pearson comparison.

Also report RMSE/MAE evidence but do not require both for the Boolean Pearson-based prediction hypothesis unless the data indicate a contradictory degradation.

If Pearson is significant but RMSE materially worsens, flag:

```text
prediction_supported_with_metric_disagreement
```

instead of silently declaring success.

---

## Prior specificity hypothesis

Assess whether matched A3 is superior to:

```text
unrelated
shuffled
random
```

Do not require all three to be individually significant for the primary prediction claim.

Report them transparently.

---

## Biomarker hypothesis

Assess whether matched A3 improves:

```text
WM alignment
and/or rank stability
```

relative to A4 and controls.

Do not force Jaccard significance if it is not present.

---

# Part 12 — Final recommendation states

Create one of:

```text
prediction_and_biomarker_supported
prediction_supported_biomarker_mixed
prediction_not_significant_biomarker_supported
prediction_not_significant
```

Do not use pilot thresholds after the final n=10 run.

Inferential evidence now supersedes the 3-seed descriptive gate.

---

# Part 13 — Preflight tests

Before launching the final run, execute:

```bash
pytest -q \
  tests/test_msancr.py \
  tests/test_msancr_refinement.py \
  tests/test_msancr_grid_closure.py \
  tests/test_network_constrained_ridge.py \
  tests/test_conditional_prior_signal.py
```

Add:

```text
tests/test_msancr_final_inference.py
```

Tests must verify:

1. folds are averaged within seed before inference;
2. exactly 10 seeds are expected for complete final inference;
3. paired Wilcoxon uses aligned seed IDs;
4. metric-wise Holm families are independent;
5. bootstrap resamples seed-level paired differences;
6. RMSE/MAE improvement orientation is correct;
7. Cohen dz sign is positive when A3 is better;
8. final inference does not emit the old `n_equals_3` wording;
9. final decision uses statistical significance, not pilot thresholds;
10. no outer-test values influence hyperparameter selection;
11. prior-swap hyperparameters remain frozen from matched A3;
12. final config is unchanged by the analyzer;
13. partial/resumed runs cannot be declared complete.

---

# Part 14 — One-fold preflight execution

Before the expensive run, execute:

```bash
python scripts/106_run_msancr_final.py \
  --config configs/iclr/msancr_final_10x5.yaml \
  --seeds 0 \
  --folds 0
```

using a temporary output directory if necessary so that it cannot contaminate the final directory.

Better option:

add:

```text
--output-dir outputs/iclr/msancr_final_smoke
```

if the runner does not already support it.

Verify:

```text
finite prediction metrics
selected HPs from inner CV
all coefficient artifacts
all three prior swaps
dual/primal consistency
config hash
```

Then delete or preserve the smoke directory separately.

Do not mix smoke rows into the final output.

---

# Part 15 — Launch the final 10×5 run

After preflight passes, execute:

```bash
python scripts/106_run_msancr_final.py \
  --config configs/iclr/msancr_final_10x5.yaml
```

This prompt explicitly authorizes this command.

Do not modify the config after the run begins.

Use resume support if interrupted.

Never use `--overwrite` on a partially valid expensive run unless a confirmed config/correctness issue requires restarting.

---

# Part 16 — Expected final cardinalities

With:

```text
10 seeds × 5 folds
```

expect:

### Retuned base methods

There are four existing base rows per split:

```text
A4
A4_iso
A2
A3 matched
```

Expected:

```text
50 × 4 = 200 base split rows
```

### Fixed prior swaps

Three swaps per split:

```text
unrelated
shuffled
random
```

Expected:

```text
50 × 3 = 150 swap rows
```

### Selected hyperparameters

Four selected base models per split:

```text
50 × 4 = 200 selected HP rows
```

Check actual repository model counts and fail loudly if these expectations no longer match.

---

# Part 17 — Required final outputs

Under:

```text
outputs/iclr/msancr_final_10x5/
```

retain existing pipeline outputs and additionally create:

```text
final_prediction_statistics.csv
final_prediction_statistics.tex

final_biomarker_statistics.csv
final_biomarker_statistics.tex

final_seed_level_table.csv

final_hypothesis_decision.json
final_statistical_summary.json

selected_hyperparameter_distribution.csv
boundary_distribution_final.csv

FINAL_COMPLETE
```

Do not replace the existing `COMPLETE`; add `FINAL_COMPLETE` only after statistical analysis is complete.

---

# Part 18 — Final paper-ready prediction table

Create a concise LaTeX table containing at least:

```text
A4 modality-specific Ridge
A2 FC-Laplacian
A3 MS-A-NCR matched
A3 unrelated fixed
A3 shuffled fixed
A3 random fixed
```

Columns:

```text
Pearson mean ± SD
RMSE mean ± SD
MAE mean ± SD
```

Use seed-level mean ± SD for the final inference table.

Bold:

```text
best mean
```

Add a star only where a method is significantly different from A3 matched after metric-wise Holm correction.

Make the caption explicit:

```text
Statistics use ten paired seed-level summaries after averaging five outer folds per seed.
```

---

# Part 19 — Final paper-ready biomarker table

Rows:

```text
A4
A2 matched
A3 matched
A3 unrelated
A3 shuffled
A3 random
```

Columns:

```text
WM-prior alignment
rank stability
top-10 Jaccard
```

Use:

```text
seed-level mean ± SD
```

Bold the best favorable mean.

Use stars for significant difference from A3 matched after metric-wise Holm correction.

Do not star A3 itself.

---

# Part 20 — Final figures

Generate:

```text
figures/iclr/msancr_final_10x5/
```

At minimum:

```text
final_prediction_seed_deltas_A3_vs_A4.pdf
final_prediction_model_comparison.pdf
final_prior_swap_comparison.pdf
final_biomarker_comparison.pdf
final_hyperparameter_selection_distribution.pdf
```

The A3-vs-A4 seed-delta plot should show all 10 seed-level Pearson differences and a zero reference line.

---

# Part 21 — Final hyperparameter audit

After all 50 splits, report distributions of:

```text
lambda_fc
lambda_sc
gamma
lambda_L
lifting
```

for A3 matched.

Also report final boundary counts.

Important:

Do not reopen the grid even if some final 10×5 splits hit a boundary.

This is now the frozen confirmatory experiment.

Boundary counts are descriptive limitations only.

No post-hoc grid expansion is allowed after seeing final test results.

---

# Part 22 — Final prior-swap requirement

Confirm in code and metadata that each fixed control swap uses the exact matched-selected hyperparameters from the same:

```text
seed
fold
```

The control must differ only in prior identity.

Write:

```text
prior_swap_integrity_check.json
```

with:

```text
n_checks
n_pass
n_fail
```

Require:

```text
n_fail = 0
```

before `FINAL_COMPLETE`.

---

# Part 23 — Final leakage audit

Write:

```text
final_leakage_audit.json
```

Verify:

```text
outer-test subjects never used for inner selection
all scalers fit inner-training only during HP selection
outer-test labels used only once for final evaluation
statistics performed only after all predictions are frozen
```

This should be programmatically checked where possible.

---

# Part 24 — Correct the grid-closure metadata wording

Do not change the actual grid.

In final metadata distinguish:

```text
A3_lambda_L_lower_boundary_hits = 3/15 in pre-final closure
A2_lambda_L_lower_boundary_hits = 5/15
```

State:

```text
A3 grid met the predefined <=3/15 closure threshold.
The prior global closure flag combined A2 and A3 and was therefore conservative.
```

Do not rewrite or delete historical grid-closure artifacts.

---

# Part 25 — Completion report to print

At the very end print a human-readable report:

```text
FINAL MS-A-NCR 10×5 COMPLETE

Prediction:
A4 Pearson = ...
A2 Pearson = ...
A3 Pearson = ...

A3 vs A4:
mean ΔPearson = ...
median ΔPearson = ...
positive seeds = .../10
Wilcoxon p = ...
Holm p = ...
95% CI = [...]
Cohen dz = ...

RMSE:
...

MAE:
...

Prior swaps:
matched vs unrelated ...
matched vs shuffled ...
matched vs random ...

Biomarkers:
WM alignment ...
rank stability ...
top-10 Jaccard ...

Prediction hypothesis: SUPPORTED / NOT SUPPORTED
Biomarker hypothesis: SUPPORTED / MIXED / NOT SUPPORTED

Overall:
prediction_and_biomarker_supported / ...

Final output directory:
...
```

Do not hide negative or nonsignificant results.

---

# Non-negotiable rules

1. No method redesign.
2. No grid expansion.
3. No target changes.
4. No outer-split changes.
5. No post-hoc selection from final test metrics.
6. No treating 50 folds as independent.
7. No rerun with new settings because final significance is disappointing.
8. Preserve all prior pilot/refinement/grid-closure outputs.
9. Stop only after `FINAL_COMPLETE` and all validation checks pass, or after a genuine error requiring human review.
