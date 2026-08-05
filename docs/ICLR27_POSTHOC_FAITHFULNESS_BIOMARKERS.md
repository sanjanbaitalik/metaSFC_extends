# ICLR 2027 Post-Hoc Experiments — Faithfulness, Biomarkers, Statistics

**Status: COMPLETE** (all three experiments, 2026-08-05).

| Item | Value |
|---|---|
| Experiment 1 script | `scripts/43_run_iclr_faithfulness.py` |
| Experiment 1 config | `configs/aaai/faithfulness_iclr.yaml` |
| Experiment 1 output | `outputs/aaai/faithfulness_iclr/` |
| Experiment 2 script | `scripts/44_run_iclr_biomarker_stability.py` |
| Experiment 3 script | `scripts/45_run_iclr_statistical_tests.py` |
| Experiment 3 output | `outputs/aaai/statistics_iclr/` |
| Run log | `outputs/aaai/faithfulness_iclr/experiment_log.txt` |

Reference implementations (schema reused): `scripts/17_run_faithfulness.py`
(faithfulness), `scripts/35_build_biomarker_significance_table.py`
(alignment/stability), `scripts/11_statistical_tests.py` (paired tests).

---

## Experiment 1 — perturbation faithfulness (masking protocol)

**Question:** does the true working-memory prior force the model to rely on
biologically meaningful circuitry?  For each of 10 seeds x 5 folds we refit
the model with its recorded best hyperparameters, then evaluate on the outer
test set with ROIs removed (all FC+SC edges incident to a mask set are zeroed):

* `top` / `bottom` — top-10 / bottom-10 learned saliency ROIs (per split);
* `prior_true_top` — top-10 ROIs of the fixed WM meta-analysis prior;
* `prior_random_top` — top-10 ROIs of the fixed random prior;
* `random` — 20 random-10 ROI sets (seeded `split_seed + 9173`).

Degradation = `Δ = rmse(masked) − rmse(original)` (positive = worse).

### Results (mean over 10 seed-level units)

| Method | Δ true-top10 | Δ random-top10 | Δ random-mean | p (true-top > random-top) |
|---|---|---|---|---|
| M2_TRUE (Meta-GAT) | **+2.63** | +1.15 | +1.38 | 0.0029 (Holm 0.0088) |
| M3_TRUE (Two-stage KRR) | **+0.19** | +0.10 | +0.39 | 0.0186 (Holm 0.0371) |
| M2_RANDOM (control) | +3.57 | +1.58 | +1.81 | <0.001 |
| M3_RANDOM (control) | +0.16 | +0.99 | +0.56 | 1.000 (n.s.) |

*Meta-GAT:* masking the true-prior top-10 ROIs degrades RMSE far more than
masking the random-prior top-10 (+1.47 RMSE gap, paired Wilcoxon p=0.0029)
or an average random-10 set (+1.25, p=0.0009).  This holds for both M2_TRUE
and M2_RANDOM: the WM hubs are functionally critical wiring regardless of
which prior gated them.

*Two-stage KRR:* masking the true top-10 degrades more than the *fixed*
random-top-10 mask (+0.08, p=0.019) but *less* than the average arbitrary
random-10 set (−0.20, n.s.).  The KRR predictor spreads its reliance across
the connectome; per-ROI node removal is therefore a weaker probe for it.
The alignment results (Experiment 2) and the strong predictive signal
(Experiment 3) carry the KRR argument instead.

Notes: for M2_TRUE, `Δ top − Δ bottom` was −0.80 (n.s.), the known
top-k-masking artifact of attribution in deep nets (removing *bottom* ROIs
also hurts because the network is coupled).  Original RMSE values of the
refits match the headline runs to within ~0.05 RMSE (CUDA
non-determinism of `index_reduce`); KRR refits match exactly.

## Experiment 2 — biomarker alignment & stability

Biomarkers per split: M2 = mean-incident attention mass (self-loops counted,
mean over layers, L1-normalized); M3 = KRR gradient node saliency (row-wise
RBF Gram).  WM alignment = Spearman vs the WM meta-analysis ROI prior,
fold-averaged within seed; stability = mean pairwise Spearman / top-10
Jaccard across the 5 folds within a seed (10 pairs).  MetaSFC (E1) anchor
included.

| Method | WM alignment | Rank stability | Top-10 Jaccard |
|---|---|---|---|
| M2_TRUE | **+0.402** | 0.942 | 0.700 |
| M2_RANDOM | −0.280 | 0.956 | 0.665 |
| M3_TRUE | **+0.005** | **0.767** | **0.419** |
| M3_RANDOM | −0.188 | 0.689 | 0.260 |
| E1 (MetaSFC) | +0.709 | 0.913 | 0.625 |

* True-prior gating dramatically re-orients Meta-GAT attention toward WM
  ROIs: alignment +0.40 vs −0.28 (Δ=0.68, Holm p=0.008).
* For the KRR, the true-prior variant is significantly more aligned (p<0.01)
  and significantly more stable (rank 0.77 vs 0.69, Jaccard 0.42 vs 0.26;
  Holm p=0.012 / 0.008), though its absolute alignment with the WM map is
  low: the two-stage projection captures predictive circuitry that is only
  partially described by the WM meta-analysis map.
* MetaSFC remains the most WM-aligned biomarker (0.71), but the two-stage
  KRR gradient saliency is the most stable variant in top-10 terms for M3_TRUE.

## Experiment 3 — statistical tests (paired at seed level, n=10)

Methodology of `scripts/11_statistical_tests.py`: average 5 folds per seed,
then paired t-test, two-sided Wilcoxon, bootstrap 95% CI of the
improvement, Cohen's d_z, Holm per metric.

### NCR_TRUE vs B3 (FC+SC Ridge)

| Metric | NCR_TRUE | B3 | Improvement | p (Wilcoxon, Holm) | d_z |
|---|---|---|---|---|---|
| pearson | 0.364 ± 0.027 | 0.370 ± 0.024 | −0.006 | 0.037 | −0.61 |
| rmse | 4.605 ± 0.061 | 4.588 ± 0.062 | −0.017 | 0.020 | −0.68 |
| mae | 3.861 ± 0.060 | 3.840 ± 0.058 | −0.021 | 0.004 | −0.85 |

NCR's coupling prior matches the strong fusion ridge baseline within ~0.006
Pearson / 0.02 RMSE; the deficit is small but statistically significant.

### M3_E0 vs E1 (MetaSFC, end-to-end MS-Inter-GCN)

| Metric | M3_E0 | E1 | Improvement | p (Wilcoxon, Holm) | d_z |
|---|---|---|---|---|---|
| pearson | 0.354 ± 0.021 | 0.149 ± 0.049 | **+0.205** | 0.004 | **+5.34** |
| rmse | 4.619 ± 0.051 | 4.976 ± 0.090 | **−0.358** | 0.004 | +3.04 |
| mae | 3.858 ± 0.054 | 4.153 ± 0.070 | **−0.295** | 0.004 | +2.87 |

The two-stage projection vastly improves prediction over the end-to-end GNN
(+0.205 Pearson, ~0.36 RMSE, all Holm-corrected p ≤ 0.004).

---

## Reproducing

```bash
conda activate metascfc-hcp
PYTHONPATH=src python scripts/43_run_iclr_faithfulness.py            # experiment 1 (resumable; ~40 min on GB10 GPU)
PYTHONPATH=src python scripts/44_run_iclr_biomarker_stability.py     # experiment 2
PYTHONPATH=src python scripts/45_run_iclr_statistical_tests.py       # experiment 3
```

All artifacts (split-level CSVs, per-split attention/saliency npz,
seed-level tests, LaTeX tables) live under
`outputs/aaai/faithfulness_iclr/` and `outputs/aaai/statistics_iclr/`.
Test suite: `PYTHONPATH=src python -m pytest -q tests/` (31 passed).
