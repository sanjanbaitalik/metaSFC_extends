#!/usr/bin/env bash
# =============================================================================
# DEFINITIVE RUN: Validation-Selected Discrete Routing + Contrastive Qwen3.8.
#
# Pivots away from the continuous learnable gate (Gradient Absorption: the
# learned branch absorbs gate changes -> flat rho-gradient -> quasi-stationary
# alpha).  alpha is now a FIXED hyperparameter selected per split from
# {0.0, 0.25, 0.5, 0.75, 1.0} by inner-validation RMSE (leakage-free).
#
# - Priors: CONTRASTIVE qwen3.8:27b maps (outputs/priors/llm/*_contrastive_qwen3)
#   via --prior-overrides; the qwen2.5:32b zero-shot priors and all earlier
#   artifacts remain untouched.
# - Results: outputs/iclr/discrete_routing_qwen3/ (summary.csv carries
#   selected_alpha_mean / selected_alpha_std per cell).
#
# Environment overrides: SEEDS (default "0 1 2 3 4 5 6 7 8 9"), IB_METHOD
# (default mine), LLM_CONFIG, DEVICE.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source /home/iemiedc2026/miniconda3/etc/profile.d/conda.sh
conda activate metascfc-hcp
export PYTHONPATH="$ROOT/src"

SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"
IB_METHOD="${IB_METHOD:-mine}"
LLM_CONFIG="${LLM_CONFIG:-configs/iclr/llm_fluid_prior.yaml}"
OUT="$ROOT/outputs/iclr/discrete_routing_qwen3"
LOG="$OUT/run.log"
mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo "[$(date '+%F %T')] Discrete-routing matrix (contrastive qwen3.8:27b)"
echo "  seeds=($SEEDS)  ib=$IB_METHOD  llm_config=$LLM_CONFIG"
echo "=============================================================="

# Sanity: the contrastive priors must exist (scripts/71 generates them).
for f in outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv \
         outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv; do
    [ -f "$f" ] || { echo "ERROR: contrastive prior missing: $f" >&2
                     echo "Run scripts/71_run_qwen3_contrastive.sh first." >&2
                     exit 1; }
done

# shellcheck disable=SC2086
python scripts/50_run_dual_task_matrix.py \
    --output-root "$OUT" \
    --llm-config "$LLM_CONFIG" \
    --ib-method "$IB_METHOD" \
    --save-checkpoints \
    --seeds $SEEDS \
    --prior-overrides '{"llm_wm": "outputs/priors/llm/working_memory_contrastive_qwen3/roi_prior.csv", "llm_fluid": "outputs/priors/llm/fluid_intelligence_contrastive_qwen3/roi_prior.csv"}'

python scripts/70_generate_iclr_latex_tables.py \
    --summary "$OUT/summary.csv" \
    --splits "$OUT/split_metrics.csv" \
    --tables-dir "$OUT/tables" \
    --include-contrastive || echo "[warn] table generation failed (matrix summary still saved)"

echo "=============================================================="
echo "[$(date '+%F %T')] Discrete-routing matrix COMPLETE."
echo "  summary : $OUT/summary.csv (selected_alpha_mean/std columns)"
echo "  tables  : $OUT/tables/"
echo "=============================================================="
