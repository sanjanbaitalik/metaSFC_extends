# OpenCode Prompt — Modification 1 Only: Fluid Integration Prior (FIP) for Frozen MS-A-NCR

## Role

You are working inside the current `metaSFC_extends` repository.

This is **Modification 1 of a maximum of three pre-declared final exploratory modifications** before reporting results to Prof. Rajapakse.

This prompt authorizes **Modification 1 only**.

Do NOT implement Modification 2.
Do NOT implement Modification 3.
Do NOT redesign MS-A-NCR.
Do NOT alter Working-Memory results.
Do NOT change the Fluid target.
Do NOT expand the frozen MS-A-NCR hyperparameter grids.
Do NOT automatically continue to another experiment after this one.

After Modification 1 completes, **STOP and wait for human review**.

---

# 1. Scientific Motivation

Current corrected results:

## Working Memory

```text
A4 modality-specific Ridge    r ≈ 0.2510
A3 matched MS-A-NCR           r ≈ 0.2632
mean Δr                       ≈ +0.0122
```

Working Memory also showed strong FC-side prior residual structure and substantially improved biomarker alignment.

## Fluid Intelligence

Corrected verification:

```text
A4 modality-specific Ridge    r ≈ 0.3449
A2 FC-Laplacian               r ≈ 0.3448
A3 matched MS-A-NCR           r ≈ 0.3412

A3 - A4 mean Δr               ≈ -0.0037
A3 - A4 median Δr             ≈ -0.0022
positive seeds                = 1/3
```

Audit 101 v2 also showed substantially weaker task-specific FC residual prior structure for Fluid than for Working Memory.

Therefore:

> Do not further tune the existing Fluid node prior.

The hypothesis of this modification is:

> Fluid Intelligence may depend more strongly on distributed and between-network integration than on a broadly lifted node-activation prior. An externally derived, edge-level **Fluid Integration Prior** may provide a more appropriate inductive bias for the already-frozen MS-A-NCR model.

Only the **Fluid prior** changes.

The prediction architecture remains frozen.

---

# 2. Name

Call the new prior family:

```text
FIP
Fluid Integration Prior
```

This is a **prior-construction experiment**, not a new prediction model.

---

# 3. Central Question

> Can an externally constructed edge-level Fluid Integration Prior provide sufficiently task-specific predictive structure for frozen MS-A-NCR to outperform the strongest no-prior Fluid baseline?

---

# 4. Strict External-Prior Rule

FIP construction must be completely independent of HCP prediction data.

Do NOT use during prior construction:

```text
HCP PMAT labels
HCP ListSort labels
HCP FC
HCP SC
HCP outer folds
HCP inner folds
HCP validation performance
HCP test performance
HCP Ridge coefficients
HCP MS-A-NCR coefficients
HCP residual correlations
Audit-101 HCP feature-target correlations
```

External prior construction must finish **before** any HCP prediction experiment starts.

HCP data may subsequently be used only for the predefined evaluation.

---

# 5. Repository Audit First

Before modifying code, inspect:

```text
src/metascfc/
outputs/priors/
outputs/iclr/conditional_prior_signal_v2/
outputs/iclr/msancr_fluid_verification/
outputs/iclr/msancr_final_10x5/
configs/iclr/
```

Determine:

1. how the current Qwen Fluid node prior is loaded;
2. how AAL116 FC edge ordering is defined;
3. how node priors are currently lifted to edge priors;
4. what Neurosynth/NiMARE resources already exist locally;
5. whether an AAL116-to-large-scale-network mapping exists locally;
6. whether study-level coordinates or study-by-ROI information already exist.

Create:

```text
outputs/iclr/fluid_integration_prior/preimplementation_audit.json
```

Do not silently download arbitrary substitute datasets.

If a critical external resource is unavailable, fail clearly and state exactly what is missing.

---

# 6. External Fluid Vocabulary

Start with the predefined terms:

```yaml
fluid_terms:
  - fluid intelligence
  - reasoning
  - relational reasoning
  - abstract reasoning
  - matrix reasoning
  - raven
```

Inspect the actual available Neurosynth/NiMARE vocabulary.

Create:

```text
term_availability.csv
```

with:

```text
requested_term
available
matched_feature
notes
```

Never fabricate unavailable terms.

Never silently replace an unavailable term with an unrelated concept.

---

# 7. Positive and Background External Study Sets

Construct:

```text
positive_studies
```

from studies associated with the available Fluid vocabulary.

Construct:

```text
background_cognitive_studies
```

from cognitive studies not included in the positive Fluid set.

Requirements:

```text
positive and background sets disjoint
remove experiments without usable coordinates
record study counts
do not use HCP information
```

Because the groups may be highly imbalanced, use deterministic balanced background sampling:

```text
n_background_per_repeat = n_positive
n_repeats = 100
```

or an equivalent valid balanced meta-analytic subtraction procedure already supported by the installed NiMARE version.

Record all external provenance.

---

# 8. Study-by-ROI Activation Matrix

Construct:

\[
A_{si}
\]

where:

```text
s = external study/experiment
i = AAL116 ROI
```

Preferred definition:

```text
A[s,i] = 1
```

if external study `s` reports at least one activation focus inside ROI `i`.

If the repository already contains a validated modeled-activation-map approach, reuse it instead.

Whichever method is used must be fixed before HCP evaluation.

Save:

```text
study_roi_activation_matrix.npz
study_roi_activation_metadata.json
```

---

# 9. Edge-Level Fluid-Specific Coactivation

For every AAL116 pair:

\[
e=(i,j)
\]

calculate external Fluid-specific coactivation.

Implement both of the following.

## 9.1 Coactivation Log-Odds

For Fluid-positive and background studies:

```text
c_pos = studies activating both ROI i and ROI j
c_bg  = background studies activating both ROI i and ROI j
```

Use Haldane correction:

\[
LOR_{ij}
=
\log
\frac{c^+_{ij}+0.5}
{N^+-c^+_{ij}+0.5}
-
\log
\frac{c^-_{ij}+0.5}
{N^--c^-_{ij}+0.5}.
\]

Aggregate across balanced background repetitions.

Save:

```text
LOR_mean
LOR_sd
```

per edge.

---

## 9.2 Base-Rate-Corrected Coactivation

Within each study group calculate:

\[
PMI_{ij}
=
\log
\frac{P(i,j)+\epsilon}
{(P(i)+\epsilon)(P(j)+\epsilon)}.
\]

Use:

```text
epsilon = 1e-6
```

unless a fixed numerically safer value is necessary.

Then calculate:

\[
\Delta PMI_{ij}
=
PMI^{Fluid}_{ij}
-
PMI^{Background}_{ij}.
\]

This correction is essential because generic cognitive hubs should not automatically receive high Fluid-specific prior weights.

---

# 10. Construct Exactly Three Predefined FIP Candidates

Do **not** create additional formulas after seeing HCP performance.

## FIP-1 — Meta-Analytic Coactivation Consensus

Create:

\[
q^{MAC}_{ij}
=
\frac{
R_+(\Delta PMI_{ij})
+
R_+(LOR_{ij})
}{2},
\]

where \(R_+\) means:

```text
retain positive task-specific values
rank-normalize them to [0,1]
set non-positive values to zero
```

Normalize the final matrix to [0,1].

This is the primary FIP candidate.

---

## FIP-2 — Between-Network Bridge Prior

Use FIP-1 plus a validated external AAL116 network assignment.

For `(i,j)`:

\[
q^{bridge}_{ij}
=
q^{MAC}_{ij}
\left[
1+
\rho\mathbf{1}(network_i\neq network_j)
\right].
\]

Fix:

```text
rho = 1.0
```

Do not tune it.

Normalize to [0,1].

### Critical restriction

If a reliable AAL116 network mapping does not already exist:

```text
do not invent one
do not derive networks from HCP
do not manually assign ROIs
```

Instead set:

```text
fip2_status = unavailable_network_mapping
```

and skip FIP-2.

---

## FIP-3 — Distributed Weak/Intermediate Integration Prior

If the required network mapping exists, construct:

For between-network edges:

\[
q^{weak}_{ij}
=
q^{MAC}_{ij}
(1-q^{MAC}_{ij})^{0.5}.
\]

For within-network edges:

\[
q^{weak}_{ij}
=
0.5q^{MAC}_{ij}.
\]

Normalize to [0,1].

The exponent is fixed:

```text
0.5
```

Do not tune it.

If the required network mapping is unavailable, skip FIP-3 rather than redefining it.

---

# 11. FIP Matrix Invariants

Every available FIP must satisfy:

```text
shape = 116 × 116
symmetric = true
diagonal = 0
finite values only
minimum >= 0
maximum <= 1
```

Create the exact upper-triangle vector using the same edge ordering as the FC regression features.

Save:

```text
fip1_mac_matrix.npy
fip1_mac_edges.csv

fip2_bridge_matrix.npy
fip2_bridge_edges.csv

fip3_weaktie_matrix.npy
fip3_weaktie_edges.csv
```

only where available.

---

# 12. External Prior Stability

Across the balanced-background repetitions record per edge:

```text
score_mean
score_sd
positive_repeat_fraction
```

Optional:

```text
empirical p
FDR-adjusted p
```

may be stored as diagnostic metadata.

Do not hard-threshold the primary continuous prior based on HCP prediction.

---

# 13. Prior Similarity Audit Before HCP Evaluation

Compare:

```text
original Qwen Fluid prior
Qwen Working-Memory prior
FIP-1
FIP-2 if available
FIP-3 if available
```

Report:

```text
Pearson correlation
Spearman correlation
cosine similarity
top-10% Jaccard
top-20% Jaccard
```

Write:

```text
prior_similarity.csv
prior_density_summary.csv
```

This is descriptive only.

Do not choose a FIP because it simply has lower similarity with the WM prior.

---

# 14. Add Explicit Prior-Space Support

Current MS-A-NCR supports node priors.

Add explicit configuration:

```yaml
prior_space: node
```

and:

```yaml
prior_space: edge
```

For:

```text
prior_space = edge
```

the supplied FIP directly provides:

```text
q_e
```

for FC anisotropic shrinkage.

Do NOT apply `prod` or `mean` node lifting again.

Preserve existing:

```text
prior_space = node
```

behavior exactly.

All previous Working-Memory tests must continue passing.

---

# 15. Edge-Level Laplacian for FIP

For edge-level FIP, use a line-graph construction.

Two FC connectome features `e` and `f` are adjacent when the underlying ROI edges share one ROI.

Use:

\[
w_{ef}
=
\sqrt{q_e q_f}
\]

when they share a node.

Otherwise:

\[
w_{ef}=0.
\]

Construct the symmetric normalized Laplacian using the same convention as frozen MS-A-NCR.

Do NOT use HCP FC correlations to define this graph.

The new prior must remain entirely external.

---

# 16. Fluid Pilot Evaluation

Once FIP construction is completed and frozen, evaluate:

```text
target = PMAT24_A_CR / Fluid Intelligence
seeds = [0,1,2]
outer folds = 5
inner folds = 3
```

Reuse the exact subject-wise splits from the corrected Fluid verification whenever possible.

Do not create more favorable outer splits.

Family-aware CV remains deferred because restricted Family_ID data are unavailable.

---

# 17. Frozen MS-A-NCR

Do not redesign the final method.

Use:

\[
\min_\beta
\|y-X\beta\|_2^2
+
\lambda_{FC}\beta_{FC}^{T}D(q;\gamma)\beta_{FC}
+
\lambda_{SC}\|\beta_{SC}\|_2^2
+
\lambda_L\beta_{FC}^{T}L_q\beta_{FC}.
\]

Requirements:

```text
D applies to ALL FC features
Laplacian is FC-only
SC uses modality-specific ordinary Ridge
exact dual solver
exact primal coefficient recovery
```

---

# 18. Frozen Hyperparameter Grids

Use exactly:

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

For old node priors:

```text
lifting = prod or mean
```

For FIP:

```text
lifting = edge_direct
```

No grid expansion.

---

# 19. Models to Evaluate

Run:

```text
B0 = A4 modality-specific no-prior Ridge
B1 = A2 FC-Laplacian
B2 = original Qwen Fluid MS-A-NCR
B3 = FIP-1 MS-A-NCR
B4 = FIP-2 MS-A-NCR if available
B5 = FIP-3 MS-A-NCR if available
```

Also create:

```text
FIP-selected
```

as described below.

---

# 20. Inner-Selected FIP

Within each outer split, candidate prior identity may be treated as an inner-CV categorical hyperparameter.

Candidate set:

```text
FIP-1
FIP-2 if available
FIP-3 if available
```

Selection must occur **inside outer-training data only**.

Selection metric:

```text
mean inner Pearson
```

Use the same tie-breaking logic as frozen MS-A-NCR.

Then freeze:

```text
FIP identity
lambda_fc
lambda_sc
gamma
lambda_L
```

and evaluate once on the outer-test fold.

Outer-test performance must never influence prior identity selection.

---

# 21. Search Fairness

Every available FIP candidate must receive the identical:

```text
lambda_fc
lambda_sc
gamma
lambda_L
```

search space.

Do not allocate more search freedom to any candidate.

---

# 22. Fixed Prior Controls

For each outer split:

1. select FIP identity and hyperparameters using matched FIP inside inner CV;
2. freeze all selected settings;
3. replace only the prior identity with controls;
4. refit;
5. evaluate on the same outer test fold.

Controls:

```text
shuffled selected-FIP prior
random prior
Working-Memory prior as unrelated
```

Do not retune controls.

---

# 23. Shuffled FIP

Use an ROI-label permutation applied consistently to rows and columns of the symmetric FIP matrix.

Preserve:

```text
symmetry
zero diagonal
weight distribution
density
graph topology degree structure as far as ROI relabeling permits
```

Use deterministic RNG seeds.

Do not independently shuffle the edge-vector values.

---

# 24. Primary Comparison

Primary exploratory comparison:

```text
FIP-selected
vs
A4 modality-specific no-prior Ridge
```

Secondary:

```text
FIP-selected vs original Qwen Fluid A3
FIP-selected vs A2
FIP-selected vs shuffled
FIP-selected vs random
FIP-selected vs unrelated
```

---

# 25. Evaluation Metrics

Per outer split:

```text
Pearson
RMSE
MAE
```

Average five folds within each seed.

With three seeds:

```text
descriptive only
no inferential significance claims
```

Report:

```text
mean
median
SD
paired mean delta
paired median delta
positive seeds / 3
```

---

# 26. Biomarker Evaluation

Using exact recovered FC coefficients calculate:

```text
FIP alignment
original Fluid-prior alignment
rank stability
top-10 Jaccard
```

for:

```text
A4
A2
original Qwen A3
each FIP candidate
FIP-selected
FIP-selected shuffled
FIP-selected random
FIP-selected unrelated
```

Descriptive only with n=3.

Do not make causal biomarker claims.

---

# 27. Output Directories

Prior outputs:

```text
outputs/iclr/fluid_integration_prior/
```

Prediction outputs:

```text
outputs/iclr/fluid_fip_pilot/
```

Figures:

```text
figures/iclr/fluid_fip_pilot/
```

Do not overwrite any existing result.

---

# 28. Required Prior Outputs

Create at least:

```text
preimplementation_audit.json
term_availability.csv
study_selection.csv
study_roi_activation_matrix.npz
study_roi_activation_metadata.json
edge_coactivation_statistics.csv

fip1_mac_matrix.npy
fip1_mac_edges.csv

fip2_bridge_matrix.npy
fip2_bridge_edges.csv

fip3_weaktie_matrix.npy
fip3_weaktie_edges.csv

prior_similarity.csv
prior_density_summary.csv
run_metadata.json
COMPLETE_PRIOR
```

Skip FIP-2/FIP-3 files if their prerequisites are unavailable.

---

# 29. Required Prediction Outputs

Create:

```text
split_metrics.csv
seed_metrics.csv
summary_metrics.csv
inner_cv_metrics.csv
selected_hyperparameters.csv
fip_selection_distribution.csv

prior_control_split_metrics.csv
prior_control_seed_metrics.csv
prior_control_summary.csv

biomarker_metrics.csv
paired_comparisons.csv

fip_decision.json
run_metadata.json
COMPLETE
```

---

# 30. Decision Gates

## Large-margin success

If:

```text
median ΔPearson(FIP-selected - A4) >= +0.015
mean ΔPearson >= +0.012
positive seeds = 3/3
no material mean RMSE degradation
```

set:

```text
recommended_next_step = full_fluid_fip_10x5
```

Do NOT start it automatically.

---

## Promising success

If:

```text
median ΔPearson >= +0.008
positive seeds >= 2/3
no material RMSE degradation
```

set:

```text
recommended_next_step = full_fluid_fip_10x5
```

Do NOT start it automatically.

---

## Borderline

If:

```text
0.005 <= median ΔPearson < 0.008
positive seeds >= 2/3
```

set:

```text
recommended_next_step = human_review
```

Do not tune FIP further.

---

## Weak / failure

If:

```text
median ΔPearson < +0.005
```

or:

```text
positive seeds <= 1/3
```

set:

```text
recommended_next_step = consider_modification_2
```

Do NOT implement Modification 2.

---

# 31. Decision JSON

Create:

```text
outputs/iclr/fluid_fip_pilot/fip_decision.json
```

containing:

```json
{
  "available_candidates": [],
  "best_candidate_descriptive": "",
  "inner_selected_candidate_distribution": {},

  "A4_pearson": 0.0,
  "original_qwen_A3_pearson": 0.0,
  "fip_selected_pearson": 0.0,

  "mean_delta_pearson_vs_A4": 0.0,
  "median_delta_pearson_vs_A4": 0.0,
  "positive_seeds_vs_A4": 0,

  "mean_delta_rmse_vs_A4": 0.0,
  "mean_delta_mae_vs_A4": 0.0,

  "matched_beats_shuffled": false,
  "matched_beats_random": false,
  "matched_beats_unrelated": false,

  "fip_biomarker_alignment": 0.0,

  "recommended_next_step": ""
}
```

---

# 32. Tests

Create:

```text
tests/test_fluid_integration_prior.py
tests/test_fluid_fip_pilot.py
```

At minimum verify:

1. prior construction never loads HCP data;
2. positive/background external study sets are disjoint;
3. background sampling is deterministic;
4. LOR implementation is correct;
5. PMI and ΔPMI are correct;
6. FIP matrices are symmetric;
7. diagonal is zero;
8. all FIP values are finite and [0,1];
9. FIP FC edge ordering exactly matches regression features;
10. `prior_space=edge` bypasses node lifting;
11. old `prior_space=node` behavior is unchanged;
12. FIP Laplacian is symmetric PSD within numerical tolerance;
13. FIP identity is selected using inner CV only;
14. outer-test labels never affect FIP selection;
15. candidates receive identical search grids;
16. fixed controls reuse matched-selected settings;
17. shuffled FIP preserves matrix invariants;
18. Working-Memory final outputs remain unchanged;
19. original Fluid verification outputs remain unchanged;
20. decision code cannot automatically launch Modification 2.

Run all relevant existing MS-A-NCR tests as well.

---

# 33. Preflight

Before the complete pilot:

1. construct external priors;
2. report available/missing terms;
3. validate FIP matrices;
4. inspect similarity/density outputs;
5. run unit tests;
6. run Fluid `seed=0, fold=0` into a separate smoke output;
7. verify finite metrics, no leakage, correct coefficient recovery, and inner-only FIP selection;
8. then run all 3 seeds × 5 outer folds.

---

# 34. Completion Report

Print:

```text
MODIFICATION 1/3 — FLUID INTEGRATION PRIOR COMPLETE

Available FIP candidates:
...

A4 no-prior Pearson = ...
Original Qwen Fluid A3 Pearson = ...

FIP-1 Pearson = ...
FIP-2 Pearson = ... / unavailable
FIP-3 Pearson = ... / unavailable

FIP-selected Pearson = ...

FIP-selected vs A4:
mean ΔPearson = ...
median ΔPearson = ...
positive seeds = .../3
mean ΔRMSE = ...
mean ΔMAE = ...

Fixed controls:
matched - shuffled = ...
matched - random = ...
matched - unrelated = ...

Biomarker:
FIP alignment = ...
rank stability = ...
top-10 Jaccard = ...

Decision:
full_fluid_fip_10x5 /
consider_modification_2 /
human_review

No Modification 2 implemented.
No Modification 3 implemented.
No post-hoc FIP tuning performed.
```

---

# FINAL NON-NEGOTIABLE INSTRUCTION

This prompt authorizes **Modification 1 only**.

After:

```text
fip_decision.json
COMPLETE
```

are created:

**STOP.**

Do not implement late fusion.

Do not implement multi-task MS-A-NCR.

Do not automatically run Fluid 10×5.

Wait for the results to be reviewed before any next step.