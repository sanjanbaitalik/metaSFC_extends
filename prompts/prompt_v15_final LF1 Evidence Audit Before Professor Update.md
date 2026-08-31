# OpenCode Prompt — Final LF1 Evidence Audit Before Professor Update

## Objective

The final LF1 10-seed × 5-fold experiment is complete for Working Memory and Fluid Intelligence.

This is **NOT Modification 3**.

Do NOT:
- train or tune any new model;
- change LF0/LF1;
- change priors;
- add seeds;
- change hyperparameter grids;
- alter fusion weights;
- rerun the final prediction experiment unless a genuine correctness bug is discovered;
- attempt to improve the reported numbers.

The sole goal is to determine whether the final frozen evidence rigorously satisfies Prof. Rajapakse's requirement:

> Adding prior should improve both task prediction and biomarker discovery.

The prediction side is frozen. The main purpose is to rigorously audit the biomarker-discovery side and package a professor-facing conclusion.

---

# 1. Freeze and verify final prediction values

Read all values from:

```text
outputs/iclr/lf1_final_10x5/
```

Do not hard-code values.

Expected approximately:

## Working Memory

```text
LF0 no-prior      = 0.2571
LF1 matched       = 0.2741
mean ΔPearson     = +0.0170
median ΔPearson   = +0.0149
positive seeds    = 8/10
Wilcoxon p        = 0.0137
two-task Holm p   = 0.027
Cohen dz          = 0.951
95% CI            = [0.007, 0.028]
```

## Fluid Intelligence

```text
LF0 no-prior      = 0.3621
LF1 matched       = 0.3700
mean ΔPearson     = +0.0079
median ΔPearson   = +0.0105
positive seeds    = 7/10
Wilcoxon p        = 0.0488
two-task Holm p   = 0.049
Cohen dz          = 0.743
95% CI            = [0.002, 0.014]
```

Corrected repeated-CV sensitivity values are approximately:

```text
WM    ≈ 0.51
Fluid ≈ 0.52
```

These conservative sensitivity results must remain reported.

Create:

```text
outputs/iclr/lf1_final_evidence_audit/prediction_requirement.csv
```

Columns:

```text
task
LF0_pearson
LF1_pearson
mean_delta
median_delta
positive_seeds
wilcoxon_raw_p
two_task_holm_p
bootstrap_ci_low
bootstrap_ci_high
cohen_dz
corrected_repeated_cv_p
```

---

# 2. Keep prediction conclusions separated

Create separate booleans:

```text
prediction_direction_supported
prediction_seed_level_inference_supported
prediction_two_task_holm_supported
prediction_repeated_cv_sensitivity_supported
```

Do not collapse these into one generic:

```text
prediction_significant
```

For example, it is valid for:

```text
prediction_two_task_holm_supported = true
prediction_repeated_cv_sensitivity_supported = false
```

These represent different assumptions about dependence.

---

# 3. Biomarker alignment alone is NOT sufficient

The FP branch is explicitly regularized toward the cognitive prior.

Therefore high matched alignment such as:

```text
WM    ≈ 0.713
Fluid ≈ 0.853
```

cannot by itself establish improved biomarker discovery.

The final biomarker claim must combine:

1. matched-vs-no-prior alignment;
2. matched-vs-unrelated specificity;
3. matched-vs-shuffled specificity;
4. matched-vs-random specificity;
5. rank stability;
6. top-k reproducibility;
7. optionally existing frozen faithfulness evidence if available without retraining.

Do not invent new biomarker metrics merely to obtain favorable results.

---

# 4. Reconstruct biomarker evidence from frozen coefficients only

Use exact FC coefficient artifacts already saved from the final LF1 run.

Do NOT refit models.

For each:

```text
task = working_memory / fluid_intelligence
seed = 0..9
```

derive one seed-level biomarker summary after aggregating the five outer folds according to the repository's existing biomarker procedure.

Required models:

```text
FC-only no-prior Ridge
FP matched
FP unrelated
FP shuffled
FP random
```

Required metrics:

```text
matched-task-prior alignment
self-prior alignment
rank stability
top-10 Jaccard
```

If top-20 Jaccard is already supported by existing code, report it as secondary.

Do not add an entirely new metric.

Create:

```text
biomarker_seed_metrics.csv
biomarker_summary.csv
```

---

# 5. Critical alignment audit

Document exactly how alignment is computed.

Create:

```text
biomarker_metric_definitions.json
```

Include:

```text
coefficient source
signed vs absolute coefficients
prior vector used
similarity/correlation statistic
fold aggregation
seed aggregation
top-k definition
```

For fair prior-specificity testing, compute all coefficient maps against the same:

```text
matched task prior
```

Thus distinguish:

```text
self_prior_alignment
matched_task_prior_alignment
```

The professor-facing comparison must primarily use:

```text
matched_task_prior_alignment
```

across:

```text
no-prior
matched
unrelated
shuffled
random
```

Otherwise each control can appear aligned merely because it is measured against its own prior.

---

# 6. Primary biomarker comparisons

For each task define:

### B1
```text
FP matched vs FC-only no-prior
```

on matched-task-prior alignment.

### B2
```text
FP matched vs FP unrelated
```

### B3
```text
FP matched vs FP shuffled
```

### B4
```text
FP matched vs FP random
```

Use the same ten aligned seed summaries.

---

# 7. Biomarker statistics

For B1-B4 calculate:

```text
left mean ± SD
right mean ± SD

paired mean difference
paired median difference
positive seeds / 10

two-sided paired Wilcoxon raw p

10,000-pair bootstrap 95% CI

Cohen dz
```

Apply Holm across exactly B1-B4 **within each task**.

Call the adjusted result:

```text
p_holm_biomarker_alignment_4
```

Do not include prediction hypotheses in this family.

---

# 8. Rank stability

For:

```text
FC no-prior
FP matched
FP unrelated
FP shuffled
FP random
```

calculate the existing rank-stability statistic.

Report:

```text
mean ± SD
```

for ten seed summaries.

The matched prior should ideally have stability:

```text
>= no-prior
```

or at least not show a material collapse.

Do not claim improved biomarker discovery if alignment increases only because the model is forced toward the prior while rankings become highly unstable.

---

# 9. Top-k reproducibility

Using the repository's existing definition, report:

```text
top-10 Jaccard
```

for all models.

Interpret stability and relevance jointly.

A shuffled/random prior can sometimes produce highly stable but biologically irrelevant rankings.

Therefore:

```text
high stability alone != biomarker validity
```

The useful pattern is:

```text
high matched-task alignment
+
reasonable rank stability
+
reasonable top-k reproducibility
+
matched > negative controls
```

---

# 10. Optional frozen biomarker faithfulness

Only if all required artifacts already exist and the repository already has a validated faithfulness procedure:

evaluate frozen top-ranked matched FP biomarkers using existing:

```text
deletion
masking
or fixed top-k retained-feature analysis
```

Rules:

- no model reselection;
- no hyperparameter tuning;
- no selecting k from test performance;
- use only k thresholds already defined elsewhere in the repository;
- compare matched ranking against no-prior/shuffled/random.

If this cannot be done without retraining or introducing new choices:

```text
biomarker_faithfulness_status = unavailable
```

and skip it.

Do not reopen experiments merely for this analysis.

---

# 11. Final prediction prior-control audit

The final report says:

```text
matched beats 3/3 controls
```

for both tasks.

Extract the exact 10-seed control statistics.

Create:

```text
prior_control_prediction.csv
```

For:

```text
matched vs unrelated
matched vs shuffled
matched vs random
```

report:

```text
matched mean Pearson
control mean Pearson

mean delta
median delta
positive seeds

raw paired p
Holm p across the 3 control comparisons
bootstrap CI
Cohen dz
```

Do not reduce this evidence to only:

```text
3/3
```

---

# 12. Fusion-weight evidence

Extract final LF1 fusion weights for all 50 outer splits of each task.

Create:

```text
fusion_weight_evidence.csv
```

Report:

```text
mean prior-aware FC weight
median prior-aware FC weight
SD

fraction w_prior = 0
fraction w_prior >= 0.10
fraction w_prior >= 0.25

SC-weight distribution
```

Interpretation:

> A consistently non-trivial prior-aware FC weight chosen entirely from training OOF predictions provides mechanistic evidence that the prior-aware branch contributes useful predictive information.

Do not claim causality from fusion weights alone.

---

# 13. Large-margin terminology

Use the predeclared descriptive gate:

```text
mean ΔPearson >= +0.010
median ΔPearson >= +0.010
positive seeds >= 7/10
```

Create:

```text
wm_large_margin
fluid_large_margin
```

Expected from current numbers:

```text
WM:
likely true

Fluid:
likely false because mean ΔPearson ≈ +0.0079
```

Do NOT describe Fluid as a "large-margin improvement" if it fails the predefined gate.

Recommended language for Fluid if final evidence remains unchanged:

```text
consistent positive improvement
```

not:

```text
large improvement
```

---

# 14. Professor requirement matrix

Create:

```text
professor_requirement_evidence.csv
```

Rows:

```text
WM prediction
WM biomarker
Fluid prediction
Fluid biomarker
```

Columns:

```text
requirement
primary_evidence
effect
raw_p
adjusted_p
sensitivity_result
control_evidence
status
recommended_wording
caveat
```

Allowed statuses:

```text
SATISFIED
SATISFIED_WITH_STATISTICAL_SENSITIVITY
PARTIALLY_SATISFIED
NOT_SATISFIED
```

---

# 15. Biomarker requirement success rule

Set:

```text
biomarker_supported = true
```

for a task only if:

1. FP-matched mean matched-task-prior alignment > FC-no-prior;
2. FP-matched beats at least 2/3 unrelated/shuffled/random controls in matched-task-prior alignment;
3. matched-vs-no-prior alignment has:
   ```text
   Holm-adjusted p < 0.05
   ```
   OR a clearly positive bootstrap CI with substantial effect;
4. rank stability does not materially collapse;
5. top-10 Jaccard is non-trivial and not clearly inferior to all controls.

Do not hard-code this result.

---

# 16. Final professor verdict

Create:

```text
final_professor_verdict.json
```

Fields:

```json
{
  "wm_prediction": "",
  "wm_biomarker": "",
  "fluid_prediction": "",
  "fluid_biomarker": "",

  "both_tasks_prediction_direction_positive": false,
  "both_tasks_prediction_two_task_holm_supported": false,
  "both_tasks_biomarker_supported": false,

  "professor_core_requirement_satisfied": false,

  "modification_3_recommended": false,
  "additional_model_tuning_recommended": false,

  "ready_to_email_professor": false,
  "ready_for_manuscript_drafting": false,

  "main_claim": "",
  "main_prediction_caveat": "",
  "main_biomarker_caveat": ""
}
```

Unless a genuine correctness bug is found:

```text
modification_3_recommended = false
additional_model_tuning_recommended = false
```

Do not recommend Modification 3 merely to increase effect size.

---

# 17. Professor-facing update

Create:

```text
PROFESSOR_UPDATE.md
```

Keep it concise and data-driven.

Suggested structure:

```text
We addressed the earlier concern that the no-prior model remained strongest by
moving to leakage-safe prior-aware late fusion.

Using the architecture-matched no-prior late-fusion baseline, the matched prior
improved Working-Memory prediction from ... to ... and Fluid Intelligence from
... to ....

The improvements were positive in .../10 and .../10 repeated-CV seeds,
respectively. The two task-level paired tests remained significant after Holm
correction across the two primary hypotheses.

For biomarker recovery, the matched prior-aware FC branch increased
task-prior alignment from ... to ... for WM and from ... to ... for Fluid,
while also outperforming unrelated/shuffled/random controls and maintaining
... stability/reproducibility.

A conservative repeated-CV dependence correction for prediction was
nonsignificant, so we will report this statistical sensitivity explicitly.
```

Do NOT state:

```text
significant under every statistical test
large improvement for both tasks
biomarkers are biologically validated
```

unless source evidence truly supports those claims.

---

# 18. Paper-ready summary tables

Create:

```text
table_professor_prediction.csv
table_professor_prediction.tex

table_professor_biomarker.csv
table_professor_biomarker.tex
```

Prediction columns:

```text
Task
LF0
LF1
Delta r
Positive Seeds
Raw p
Two-task Holm p
Corrected repeated-CV p
95% CI
dz
```

Biomarker columns:

```text
Task
No-prior alignment
Matched alignment
Unrelated alignment
Shuffled alignment
Random alignment
Rank stability
Top-10 Jaccard
```

---

# 19. Source-integrity freeze

Before and after the audit hash:

```text
predictions
coefficients
selected hyperparameters
fusion weights
split metrics
seed metrics
final config
```

No frozen model artifact may change.

Create:

```text
source_integrity.json
```

If any frozen artifact changes:

```text
FROZEN_OUTPUT_DRIFT_DETECTED
```

and STOP.

---

# 20. Tests

Add:

```text
tests/test_lf1_final_evidence_audit.py
```

Verify:

1. prediction values reproduce final source files;
2. biomarker metrics use only frozen coefficients;
3. common matched-task-prior alignment is used for specificity comparisons;
4. biomarker Holm family contains exactly B1-B4 per task;
5. prediction and biomarker statistical families remain separate;
6. corrected repeated-CV sensitivity remains visible;
7. Fluid is not labeled large-margin unless its predefined gate passes;
8. no training/fitting function is called;
9. no Modification 3 is invoked;
10. frozen source hashes remain unchanged.

Run all relevant existing LF1 tests.

---

# 21. Output directory

Use:

```text
outputs/iclr/lf1_final_evidence_audit/
```

Required:

```text
prediction_requirement.csv

biomarker_metric_definitions.json
biomarker_seed_metrics.csv
biomarker_summary.csv
biomarker_statistics.csv

prior_control_prediction.csv
fusion_weight_evidence.csv

professor_requirement_evidence.csv
final_professor_verdict.json
PROFESSOR_UPDATE.md

table_professor_prediction.csv
table_professor_prediction.tex
table_professor_biomarker.csv
table_professor_biomarker.tex

source_integrity.json

COMPLETE
```

---

# 22. Completion report

Print:

```text
FINAL LF1 EVIDENCE AUDIT COMPLETE

WORKING MEMORY

Prediction:
LF0 = ...
LF1 = ...
Delta r = ...
positive seeds = .../10
raw p = ...
two-task Holm p = ...
corrected repeated-CV p = ...
dz = ...

Biomarker:
no-prior matched-task alignment = ...
matched alignment = ...
unrelated = ...
shuffled = ...
random = ...
rank stability = ...
top-10 Jaccard = ...

biomarker supported = YES/NO

FLUID INTELLIGENCE

Prediction:
LF0 = ...
LF1 = ...
Delta r = ...
positive seeds = .../10
raw p = ...
two-task Holm p = ...
corrected repeated-CV p = ...
dz = ...

Biomarker:
no-prior matched-task alignment = ...
matched alignment = ...
unrelated = ...
shuffled = ...
random = ...
rank stability = ...
top-10 Jaccard = ...

biomarker supported = YES/NO

PROFESSOR CORE REQUIREMENT SATISFIED = YES/NO

Modification 3 recommended = NO unless a genuine correctness failure was found.
Ready to email professor = YES/NO
Ready for manuscript drafting = YES/NO

No model retraining or post-hoc tuning performed.
```

# FINAL NON-NEGOTIABLE INSTRUCTION

Do not implement Modification 3.

Do not retrain or tune models.

This is a frozen-output evidence audit only.

After completion, STOP for human review.