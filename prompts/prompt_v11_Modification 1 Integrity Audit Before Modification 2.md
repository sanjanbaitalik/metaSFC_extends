# OpenCode Prompt — Modification 1 Integrity Audit Before Modification 2

## Objective

Audit the completed Fluid Integration Prior (FIP) experiment before deciding whether to proceed to Modification 2.

This is **not Modification 2**.

Do not implement late fusion.
Do not implement multi-task learning.
Do not add new FIP formulas.
Do not expand grids.
Do not redesign MS-A-NCR.
Do not alter Working-Memory outputs.

If Modification 1 is confirmed correct, stop with:

```text
MOD1_VALIDATED_FAILURE
```

and recommend Modification 2.

If a genuine correctness bug materially affected the result, fix **only that bug** and rerun the same predeclared 3-seed FIP experiment once.

---

## Known inconsistencies requiring investigation

### 1. A4 baseline does not reproduce

Previous corrected Fluid verification:

```text
A4 Pearson ≈ 0.3449
Original Qwen A3 ≈ 0.3412
```

Modification 1:

```text
A4 Pearson = 0.3593
Original Qwen A3 = 0.3411
```

The Qwen A3 essentially reproduces while supposedly unchanged A4 moves by approximately +0.0144.

Determine exactly why.

Under identical subjects, splits, target, preprocessing, grids, and model implementation, A4 must reproduce numerically.

---

### 2. RMSE and MAE deltas are suspicious

The completion report gives:

```text
mean ΔRMSE = 0.0
mean ΔMAE = 0.0
```

despite FIP-selected losing approximately 0.056 Pearson.

Determine whether these are:

```text
real computed values
placeholder defaults
missing fields
aggregation bugs
wrong-column reads
```

Numeric zero must never represent unavailable data.

---

### 3. Biomarker outputs appear incomplete

Reported:

```text
FIP alignment = 0.0
rank stability = N/A
top-10 Jaccard = N/A
```

Determine whether exact coefficient artifacts exist.

If coefficients exist, recompute the frozen biomarker metrics without retraining.

If they do not exist, report:

```text
unavailable_without_rerun
```

Do not encode missing biomarkers as zero.

---

### 4. Neurosynth provenance appears inconsistent

The completion report refers to roughly:

```text
73 studies
~500k peaks
```

The ~500k figure is approximately the scale of the **entire Neurosynth database**, not a 73-study subset.

Verify that FIP construction did not accidentally:

```text
select 73 positive studies
but use coordinate rows from the full database
```

or otherwise mix full-database and selected-subset statistics.

---

# Part 1 — Inspect implementation

Inspect:

```text
src/metascfc/priors/fluid_integration_prior.py
scripts/111_build_fluid_integration_prior.py
scripts/112_run_fluid_fip_pilot.py

configs/iclr/fluid_integration_prior.yaml
configs/iclr/fluid_fip_pilot.yaml

tests/test_fluid_integration_prior.py
tests/test_fluid_fip_pilot.py

outputs/iclr/fluid_integration_prior/
outputs/iclr/fluid_fip_pilot/
outputs/iclr/msancr_fluid_verification/
```

Write all audit outputs to:

```text
outputs/iclr/fluid_fip_integrity_audit/
```

Do not overwrite original Modification-1 outputs.

---

# Part 2 — Exact outer-split identity

For all:

```text
seeds = 0,1,2
folds = 0..4
```

compare the corrected Fluid verification against the FIP experiment.

Create:

```text
split_identity_audit.csv
```

Columns:

```text
seed
fold
old_train_hash
fip_train_hash
old_test_hash
fip_test_hash
train_identical
test_identical
```

If the experiment claimed to reuse the corrected Fluid splits, require:

```text
15/15 identical
```

If not identical, explain exactly why.

---

# Part 3 — A4 reproduction audit

Compare the exact A4 implementation/configuration used in:

```text
outputs/iclr/msancr_fluid_verification/
```

and:

```text
outputs/iclr/fluid_fip_pilot/
```

Verify:

```text
same model implementation
same target
same label vector
same preprocessing
same train-only scaler
same outer splits
same inner 3-fold CV
same Pearson-first selection
same 0.002 Pearson tie tolerance
same lambda_fc grid
same lambda_sc grid
same aggregation
```

Create:

```text
a4_reproduction_audit.csv
```

with per-split:

```text
seed
fold
old_pearson
new_pearson
old_rmse
new_rmse
old_mae
new_mae

old_lambda_fc
new_lambda_fc
old_lambda_sc
new_lambda_sc
```

If protocol/splits are identical, predictions should agree to numerical tolerance.

Identify the precise reason for the current aggregate difference.

---

# Part 4 — Original Qwen A3 reproduction

Perform the same split-level comparison for original Qwen Fluid A3.

Create:

```text
qwen_a3_reproduction_audit.csv
```

Do not rely only on the aggregate 0.3412 vs 0.3411 similarity.

---

# Part 5 — External Neurosynth provenance audit

Report distinctly:

```text
n_total_database_studies
n_total_database_coordinate_rows

n_positive_fluid_studies
n_background_studies_available
n_background_studies_per_repeat

n_positive_coordinate_rows
n_background_coordinate_rows_used_per_repeat
```

Do not describe the total database coordinate count as the number of peaks belonging to the selected Fluid studies.

Create:

```text
external_provenance_audit.json
```

---

# Part 6 — Coordinate filtering integrity

For every selected positive study ID:

verify that every coordinate row entering the positive study-by-ROI matrix belongs to one of the selected positive IDs.

For every background repetition:

verify that every coordinate row belongs to a study selected for that repetition.

Explicitly test:

```text
PMID vs experiment-ID confusion
string vs integer ID mismatch
NaN study IDs
duplicate IDs
article-level vs experiment-level collapse
```

Create:

```text
study_coordinate_filter_audit.csv
```

Require:

```text
positive_foreign_coordinate_rows = 0
background_foreign_coordinate_rows = 0
```

---

# Part 7 — Study-by-ROI matrix integrity

Verify:

```text
rows correspond exactly to selected studies/experiments
columns = 116
no unselected-study rows
no unexplained missing selected studies
```

Report:

```text
n_rows
n_columns
density
mean active ROIs per study
median active ROIs per study
max active ROIs per study
zero-ROI studies
```

Flag implausible activation density.

---

# Part 8 — FIP matrix audit

For each:

```text
FIP1_MAC
FIP2_Bridge
FIP3_Weaktie
```

report:

```text
shape
symmetry error
diagonal maximum
minimum
maximum
mean
SD
nonzero fraction
effective number of weighted edges
top-1% mass
top-5% mass
top-10% mass
```

Create:

```text
fip_matrix_integrity.csv
```

Determine whether any FIP is:

```text
nearly uniform
pathologically sparse
numerically degenerate
```

---

# Part 9 — FIP-selection reporting audit

The current report says:

```text
FIP-1 Pearson = 0.3036
FIP-2 = N/A but selected in 40%
FIP-3 = N/A
FIP-selected Pearson = 0.3036
```

Clarify what was actually outer-tested.

Create:

```text
fip_selection_reporting_audit.json
```

Report:

```text
selection count FIP1
selection count FIP2
selection count FIP3
```

If only the inner-selected candidate was outer-tested, do not report candidate-specific outer-test Pearson values.

Use:

```text
candidate_specific_outer_metrics_available = false
```

instead.

---

# Part 10 — Reconstruct prediction metrics

Using frozen predictions or split metrics, recompute:

```text
Pearson
RMSE
MAE
```

for:

```text
A4
original Qwen A3
FIP-selected
```

Aggregate:

```text
five folds -> seed mean
three seed means -> descriptive summary
```

Create:

```text
recomputed_prediction_metrics.csv
recomputed_seed_metrics.csv
```

Verify all existing reported values.

Numeric placeholder zeros are forbidden.

---

# Part 11 — Reconstruct biomarker metrics if possible

If exact coefficient artifacts already exist, compute without retraining:

```text
FIP alignment
original Fluid-prior alignment
rank stability
top-10 Jaccard
```

Create:

```text
recomputed_biomarker_metrics.csv
```

If unavailable:

```text
biomarker_status = unavailable_without_rerun
```

Do not substitute zero.

---

# Part 12 — Decision

## Case A — implementation and external filtering are correct

If FIP-selected still clearly loses to A4:

```text
mod1_status = VALIDATED_FAILURE
recommended_next_step = modification_2_late_fusion
```

Do not rerun FIP.

---

## Case B — material evaluation bug found

Examples:

```text
wrong A4 implementation
different outer splits
incorrect inner selection
incorrect metric aggregation
outer-test leakage
```

Fix only the bug.

Rerun the exact same original Modification-1 protocol once.

Write corrected outputs to:

```text
outputs/iclr/fluid_fip_pilot_v2/
```

Do not overwrite v1.

---

## Case C — material prior-construction bug found

Examples:

```text
all-database coordinates entered positive studies
study IDs filtered incorrectly
background coordinates leaked into positive set
study-by-ROI matrix built from wrong studies
```

Fix only the correctness issue.

Rebuild exactly the same:

```text
FIP1
FIP2
FIP3
```

No new formulas.

Rerun exactly the same 3-seed experiment once.

---

# Part 13 — Original decision thresholds remain frozen

If a correctness-only rerun occurs:

### Success

```text
median ΔPearson(FIP-selected - A4) >= +0.008
positive seeds >= 2/3
no material RMSE degradation
```

→

```text
full_fluid_fip_10x5
```

### Failure

```text
median ΔPearson < +0.005
or positive seeds <= 1/3
```

→

```text
modification_2_late_fusion
```

Do not modify these thresholds.

---

# Part 14 — Required outputs

Create:

```text
outputs/iclr/fluid_fip_integrity_audit/
```

containing:

```text
audit_findings.json
split_identity_audit.csv
a4_reproduction_audit.csv
qwen_a3_reproduction_audit.csv

external_provenance_audit.json
study_coordinate_filter_audit.csv
fip_matrix_integrity.csv
fip_selection_reporting_audit.json

recomputed_prediction_metrics.csv
recomputed_seed_metrics.csv
recomputed_biomarker_metrics.csv  # only if available

integrity_decision.json
COMPLETE
```

---

# Part 15 — Tests

Add:

```text
tests/test_fluid_fip_integrity.py
```

Test at least:

1. positive study IDs filter coordinates exactly;
2. unselected coordinates cannot enter the positive matrix;
3. background rows belong only to the sampled background IDs;
4. dtype mismatches cannot silently break ID filtering;
5. A4 reproduction detects split/config divergence;
6. RMSE/MAE cannot silently default to zero;
7. missing biomarkers serialize as null/NA, not zero;
8. candidate-specific performance cannot be reported unless outer-tested;
9. no new FIP formulas/grids are introduced;
10. Modification 2 is not launched.

Run all existing FIP and MS-A-NCR tests.

---

# Completion report

Print:

```text
MODIFICATION 1 INTEGRITY AUDIT COMPLETE

A4:
previous corrected Fluid = ...
Modification 1 = ...
cause of discrepancy = ...

Qwen A3 reproduction = ...

External data:
total Neurosynth studies = ...
total coordinate rows = ...
positive Fluid studies = ...
positive coordinate rows actually used = ...
background studies/repeat = ...
coordinate filtering = PASS/FAIL

FIP-selected corrected:
Pearson = ...
ΔPearson vs A4 = ...
ΔRMSE = ...
ΔMAE = ...

Biomarker status = ...

Modification 1 status:
VALIDATED_FAILURE /
CORRECTED_RERUN_SUCCESS /
CORRECTED_RERUN_FAILURE

Recommended next step:
full_fluid_fip_10x5 /
modification_2_late_fusion /
human_review

NO MODIFICATION 2 IMPLEMENTED.
```

STOP.