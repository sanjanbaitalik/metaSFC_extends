# Prompt — MS-A-NCR Pilot: Modality-Selective Anisotropic Network-Constrained Ridge

## Objective

Implement and evaluate a **3-seed pilot** of:

```text
MS-A-NCR
Modality-Selective Anisotropic Network-Constrained Ridge
```

for the ICLR extension of MetaSFC.

Do **not** run the full 10-seed experiment yet.

The purpose of this stage is to test one specific hypothesis emerging from Audit 101 v2:

> The external cognitive prior contains useful conditional structure primarily in **functional connectivity (FC)**, while applying the same prior to SC is likely counterproductive. The prior should therefore modulate FC shrinkage and FC network smoothing, while SC remains primarily ordinary whole-brain predictive signal.

This is the first actual NCR redesign after the diagnostic audits.

---

## Evidence motivating the design

Audit 101 v2 showed:

### Residual enrichment is primarily FC-side

Representative matched-prior residual-enrichment values:

```text
Fluid:
  FC product  ~ +0.019
  SC product  ~ -0.004

  FC mean     ~ +0.009
  SC mean     ~ -0.009

Working memory:
  FC product  ~ +0.063
  SC product  ~ -0.032

  FC mean     ~ +0.068
  SC mean     ~ -0.034
```

The residual-additive branch did not meaningfully improve prediction:

```text
Fluid median ΔPearson ~ 0
WM median ΔPearson    ~ 0
```

and inner selection frequently chose:

```text
eta = 0
```

Therefore:

- do not implement residual NCR;
- do not rebuild the prior yet;
- do not apply the prior symmetrically to FC and SC;
- test whether **anisotropic FC regularization** converts prior signal into improved held-out prediction.

---

# Core model

Let:

\[
X = [X_{FC}, X_{SC}]
\]

and:

\[
\beta =
\begin{bmatrix}
\beta_{FC}\\
\beta_{SC}
\end{bmatrix}.
\]

The primary MS-A-NCR objective is:

\[
\min_\beta
\|y-X\beta\|_2^2
+
\lambda_{FC}
\beta_{FC}^{\top}D(q;\gamma)\beta_{FC}
+
\lambda_{SC}\|\beta_{SC}\|_2^2
+
\lambda_L
\beta_{FC}^{\top}L_q\beta_{FC}.
\]

Where:

```text
D(q; gamma)
```

is the prior-dependent diagonal shrinkage matrix and:

```text
L_q
```

is the existing prior-derived FC edge Laplacian.

No prior-derived Laplacian should be applied to SC in the primary model.

---

# Anisotropic diagonal penalty

For FC edge `e` with continuous prior score `q_e`:

\[
d_e = (\epsilon + q_e)^{-\gamma}.
\]

Use:

```text
epsilon = 1e-3
```

Then normalize:

\[
D \leftarrow \frac{D}{mean(diag(D))}
\]

so that average penalty strength remains comparable across prior variants and gamma settings.

Interpretation:

```text
high q_e -> lower d_e -> less shrinkage
low q_e  -> higher d_e -> more shrinkage
```

This must be implemented as a **true generalized Ridge penalty**.

Do not implement this by scaling input features and then standardizing them away.

---

# Continuous prior lifting

Use the existing continuous ROI prior.

Test only these two primary edge liftings:

## Product

\[
q_{ij}^{prod} = p_i p_j
\]

## Mean

\[
q_{ij}^{mean} = \frac{p_i+p_j}{2}.
\]

Do not include `max` or `bridge` in the primary pilot grid.

The lifting rule may be selected inside inner CV.

---

# FC-only Laplacian

Construct/use the existing line-graph Laplacian from the prior for FC features only.

The global penalty matrix should conceptually be:

\[
P =
\begin{bmatrix}
\lambda_{FC} D(q;\gamma) + \lambda_L L_q & 0\\
0 & \lambda_{SC} I
\end{bmatrix}.
\]

Do not couple FC and SC coefficients inside the regularization matrix in this pilot.

The predictive design matrix remains concatenated:

```text
X = [FC | SC]
```

so the model still learns from both modalities jointly.

---

# Required model variants

Implement and compare exactly the following five model families.

## A0 — Standard no-prior FC+SC Ridge

Objective:

\[
\|y-X\beta\|^2
+
\lambda\|\beta\|_2^2.
\]

This is the primary anchor.

Reuse the existing implementation/results where possible, but ensure identical pilot splits.

---

## A1 — FC anisotropic Ridge + ordinary SC Ridge

No Laplacian.

\[
\|y-X\beta\|^2
+
\lambda_{FC}
\beta_{FC}^{\top}D(q;\gamma)\beta_{FC}
+
\lambda_{SC}\|\beta_{SC}\|^2.
\]

Purpose:

> isolate the effect of prior-dependent anisotropic shrinkage.

---

## A2 — FC-only Laplacian NCR + ordinary SC Ridge

Use:

```text
gamma = 0
```

so the diagonal FC penalty is isotropic.

\[
\|y-X\beta\|^2
+
\lambda_{FC}\|\beta_{FC}\|^2
+
\lambda_{SC}\|\beta_{SC}\|^2
+
\lambda_L\beta_{FC}^{\top}L_q\beta_{FC}.
\]

Purpose:

> isolate the effect of FC-only network smoothing.

---

## A3 — MS-A-NCR

Full proposed pilot model:

\[
\|y-X\beta\|^2
+
\lambda_{FC}
\beta_{FC}^{\top}D(q;\gamma)\beta_{FC}
+
\lambda_{SC}\|\beta_{SC}\|^2
+
\lambda_L\beta_{FC}^{\top}L_q\beta_{FC}.
\]

Purpose:

> test the combination of anisotropic shrinkage and network smoothing.

---

## A4 — Modality-specific no-prior Ridge

No external prior.

\[
\|y-X\beta\|^2
+
\lambda_{FC}\|\beta_{FC}\|^2
+
\lambda_{SC}\|\beta_{SC}\|^2.
\]

Purpose:

> determine whether any A3 improvement is truly prior-driven or merely due to giving FC and SC separate regularization strengths.

This baseline is mandatory.

---

# Primary hypothesis

The key comparison is:

```text
A3 matched prior
vs
max(A0, A4)
```

If A3 beats only A0 but not A4, then the gain is mainly due to modality-specific regularization rather than the prior.

A successful prior claim requires:

```text
A3 matched > A4
```

in a consistent held-out direction.

---

# Prior controls

For A1, A2, and A3 evaluate:

```text
matched prior
unrelated prior
shuffled prior
random prior
```

using exactly the same hyperparameter grids.

Do not give the matched prior a larger search space.

Control construction must reuse the existing prior-control utilities from Audits 100/101 wherever possible.

---

# Pilot scope

Run only:

```text
seeds = [0, 1, 2]
folds = 5
```

Total:

```text
3 seeds x 5 outer folds
```

for each target.

Use the exact same subject cohort and split definitions as the current ICLR experiments.

Do not regenerate split assignments.

---

# Targets

Run the same targets used in Audit 101 v2:

```text
Fluid Intelligence
Working Memory
```

Do not change target definitions in this pilot.

---

# Files to add

Prefer:

```text
src/metascfc/models/iclr_backbones/modality_selective_anisotropic_ncr.py
scripts/104_run_msancr_pilot.py
configs/iclr/msancr_pilot.yaml
tests/test_msancr.py
```

If the existing NCR solver can be cleanly generalized, modifying:

```text
src/metascfc/models/iclr_backbones/network_constrained_ridge.py
```

is acceptable only if:

- old NCR behavior remains exactly reproducible;
- all old tests continue passing;
- the new behavior is behind explicit configuration flags.

Do not silently alter existing NCR defaults.

---

# Solver requirement

The model should remain an exact generalized-Ridge / NCR solution.

Do not replace the current exact solver with SGD.

The penalty is:

\[
P =
\begin{bmatrix}
P_{FC} & 0\\
0 & P_{SC}
\end{bmatrix}
\]

with:

\[
P_{FC}
=
\lambda_{FC}D
+
\lambda_L L_q
\]

and:

\[
P_{SC}
=
\lambda_{SC}I.
\]

Solve:

\[
\hat\beta
=
(X^\top X + P)^{-1}X^\top y
\]

using a numerically stable formulation appropriate for:

```text
p >> n
```

Prefer reusing the repository's Woodbury / dual solver.

---

# Stable transformed formulation

Where useful, factor:

\[
P_{FC}
=
D^{1/2}
\left(
\lambda_{FC}I
+
\lambda_L
D^{-1/2}L_qD^{-1/2}
\right)
D^{1/2}.
\]

Define:

\[
X'_{FC}
=
X_{FC}D^{-1/2}.
\]

Then use:

\[
L'
=
D^{-1/2}L_qD^{-1/2}
\]

inside the existing spectral/Woodbury strategy.

Add numerical safeguards for extremely small or large penalty values.

---

# Feature standardization

Standardize FC and SC using training-set statistics only.

The prior must enter only through:

```text
D
L_q
```

not through feature rescaling that is later cancelled.

For every outer/inner split:

- fit scaler on training partition only;
- transform validation/test using that scaler;
- never fit scaling on held-out data.

---

# Hyperparameter grid

Keep the pilot intentionally small.

## Anisotropy

```yaml
gamma:
  - 0.0
  - 0.25
  - 0.5
  - 1.0
  - 2.0
```

`gamma = 0` must reduce the diagonal prior penalty to isotropic FC Ridge.

---

## Laplacian strength

```yaml
lambda_laplacian:
  - 0.0
  - 0.1
  - 0.5
  - 1.0
  - 2.0
  - 5.0
```

---

## FC/SC Ridge strengths

Use a compact existing Ridge-like grid, for example:

```yaml
lambda_fc:
  - 0.01
  - 0.1
  - 1.0
  - 10.0
  - 100.0

lambda_sc:
  - 0.01
  - 0.1
  - 1.0
  - 10.0
  - 100.0
```

If the repository already has a validated Ridge grid, reuse that grid rather than creating a conflicting one.

---

# Search-space control

Avoid an unnecessary Cartesian explosion.

Use staged inner selection if needed.

Recommended:

## Stage 1

For A4:

```text
select lambda_fc
select lambda_sc
```

## Stage 2

For A1:

```text
reuse/refine around selected lambda_fc/lambda_sc
select lifting
select gamma
```

## Stage 3

For A2:

```text
reuse/refine lambda_fc/lambda_sc
select lifting
select lambda_laplacian
```

## Stage 4

For A3:

```text
reuse/refine candidate values
select lifting
select gamma
select lambda_laplacian
```

All stages must remain inner-CV only.

Do not use outer-test performance to prune the search.

---

# Inner-CV model selection

Use the existing nested CV logic.

Primary selection metric:

```text
Pearson r
```

Tie-breakers:

```text
1. lower RMSE
2. lower MAE
3. simpler model
```

Suggested simplicity preference:

```text
smaller gamma
smaller lambda_laplacian
product/mean deterministic order
```

No outer-test labels may enter model or hyperparameter selection.

---

# Required invariant checks

Verify numerically:

### A0 recovery

If:

```text
lambda_fc = lambda_sc
gamma = 0
lambda_laplacian = 0
```

A1/A3 machinery should reduce to ordinary FC+SC Ridge up to numerical tolerance.

### A4 recovery

If:

```text
gamma = 0
lambda_laplacian = 0
lambda_fc != lambda_sc
```

the model should equal modality-specific Ridge.

### No-prior control

If prior is uniform:

```text
q_e = constant
```

anisotropic shrinkage must reduce to isotropic shrinkage after D normalization.

---

# Evaluation metrics

Report per outer split:

```text
Pearson
RMSE
MAE
```

Also save:

```text
selected lifting
selected gamma
selected lambda_fc
selected lambda_sc
selected lambda_laplacian
```

For A3 additionally report:

```text
mean D_fc
min D_fc
max D_fc
condition number / numerical diagnostics
```

---

# Primary output directory

Write to:

```text
outputs/iclr/msancr_pilot/
```

Do not overwrite existing NCR or Audit outputs.

Create:

```text
split_metrics.csv
seed_metrics.csv
summary_metrics.csv
selected_hyperparameters.csv
prior_control_metrics.csv
paired_comparisons.csv
pilot_decision.json
run_metadata.json
COMPLETE
```

---

# Required summary table

Create a table like:

```text
target
model_id
prior_type
pearson_mean
pearson_std
rmse_mean
rmse_std
mae_mean
mae_std
delta_r_vs_A0
delta_r_vs_A4
positive_seed_count_vs_A4
```

---

# Pairwise comparisons

For the pilot, do not overemphasize significance because there are only 3 seeds.

Still compute descriptive paired differences:

```text
A1 matched - A4
A2 matched - A4
A3 matched - A4

A3 matched - A3 unrelated
A3 matched - A3 shuffled
A3 matched - A3 random
```

Report:

```text
mean paired delta
median paired delta
positive seed count
```

Do not make inferential claims from n=3.

---

# Pilot decision logic

Create:

```text
outputs/iclr/msancr_pilot/pilot_decision.json
```

with per-task decisions.

Example:

```json
{
  "fluid": {
    "best_model": "A3",
    "best_prior": "matched",
    "delta_r_vs_A0": 0.0,
    "delta_r_vs_A4": 0.0,
    "positive_seeds_vs_A4": 0,
    "recommended_next_step": ""
  },
  "working_memory": {
    "best_model": "A3",
    "best_prior": "matched",
    "delta_r_vs_A0": 0.0,
    "delta_r_vs_A4": 0.0,
    "positive_seeds_vs_A4": 0,
    "recommended_next_step": ""
  },
  "overall_recommendation": ""
}
```

---

# Gate criteria

## Strong success

If:

```text
A3 matched delta_r_vs_strongest_no_prior >= +0.015
```

and improvement is positive in:

```text
3/3 seeds
```

or nearly all seed-level aggregates,

then:

```text
recommended_next_step = full_10x5_msancr
```

Freeze the architecture before the full run.

---

## Promising

If:

```text
+0.008 <= delta_r < +0.015
```

with consistent direction across seeds:

```text
recommended_next_step = one_targeted_msancr_refinement
```

Do not immediately run 10 seeds.

---

## Weak

If:

```text
delta_r < +0.005
```

or gains are inconsistent:

```text
recommended_next_step = ct_mac_prior_rebuild
```

Do not continue tuning anisotropic NCR extensively.

---

## Diagnostic outcomes

If:

```text
A1 > A2
and
A1 ~= A3
```

then anisotropic shrinkage is doing the work and Laplacian smoothing is unnecessary.

Recommendation:

```text
simplify_to_anisotropic_ridge
```

If:

```text
A2 > A1
```

then FC-only network smoothing matters more than anisotropy.

If:

```text
A3 > A1 and A3 > A2
```

then the two components are complementary.

If:

```text
A4 ~= A3
```

then any gain is mainly due to separate FC/SC penalties rather than the prior.

This must be explicitly flagged.

---

# Biomarker diagnostics

Because the final method must improve both prediction and biomarker discovery, preserve biomarker diagnostics for A3.

At minimum report for matched A3:

```text
prior alignment
rank stability
top-k Jaccard
```

using the same seed-level definitions already established in the AAAI/ICLR pipeline.

Compare against:

```text
A0 / no-prior
A4
A3 shuffled
A3 random
A3 unrelated
```

This pilot does not require a full faithfulness rerun unless prediction shows promise.

---

# Do not change

Do not modify:

```text
behavioral targets
HCP cohort
outer split identities
current Qwen prior
Audit 100
Audit 101 v1/v2
MT-NCR
GNN models
CT-MAC prior
final ICLR tables
```

Do not rebuild the prior in this step.

---

# Tests

Create:

```text
tests/test_msancr.py
```

Test at minimum:

1. D(q, gamma) formula;
2. D normalized to mean 1;
3. high-prior edge receives lower shrinkage;
4. gamma=0 gives isotropic FC diagonal penalty;
5. uniform prior gives isotropic FC penalty;
6. FC and SC feature blocks are mapped correctly;
7. SC receives no prior Laplacian in A3;
8. A0 recovery;
9. A4 recovery;
10. A1 has lambda_laplacian=0;
11. A2 has gamma=0;
12. A3 allows both components;
13. no outer-test leakage in model selection;
14. scaler fit only on training partitions;
15. matched/unrelated/shuffled/random use identical grids;
16. solver matches direct small-matrix solution on synthetic data;
17. solver remains numerically stable for p > n;
18. deterministic outputs for fixed seeds;
19. old NCR tests remain passing.

Run:

```bash
pytest -q tests/test_msancr.py
```

and all relevant existing NCR tests.

---

# Execution command

Support:

```bash
python scripts/104_run_msancr_pilot.py \
  --config configs/iclr/msancr_pilot.yaml
```

Default config must use:

```yaml
seeds: [0, 1, 2]
folds: 5
```

---

# Runtime / resumability

This may be computationally expensive.

Implement resumability at:

```text
target / model / prior / seed / fold
```

level.

Write intermediate split-level results atomically.

On restart:

- skip complete valid results;
- recompute missing/invalid results only.

Do not silently reuse outputs generated under a different config hash.

Store a config hash in metadata.

---

# Completion validation

Before writing `COMPLETE`, verify:

```text
all expected target/model/prior/seed/fold combinations exist
no duplicate rows
no NaN/Inf metrics
identical split IDs across methods
same subjects across paired comparisons
same search grid across priors
all hyperparameters selected inner-only
config hash matches run metadata
```

Print a concise validation report.

---

# Files to return for review

Return:

```text
outputs/iclr/msancr_pilot/pilot_decision.json
outputs/iclr/msancr_pilot/summary_metrics.csv
outputs/iclr/msancr_pilot/seed_metrics.csv
outputs/iclr/msancr_pilot/paired_comparisons.csv
outputs/iclr/msancr_pilot/selected_hyperparameters.csv
outputs/iclr/msancr_pilot/prior_control_metrics.csv
```

Also return the complete output directory if practical.

Do not run the full 10-seed experiment until these pilot results are reviewed.
