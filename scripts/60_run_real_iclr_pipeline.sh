#!/usr/bin/env bash
# =============================================================================
# ICLR 2027 definitive production run on the real AAAI HCP-YA dataset.
#
# Chains: robust dual-target repacking -> zero-shot LLM priors (Ollama) ->
# dual-task matrix with MINE Information-Bottleneck tracking -> IB figure.
#
# NO synthetic data and NO --prior-overrides are used anywhere.
# All output is appended to outputs/iclr/production_run.log, including the
# per-reason subject-drop log from the packing QC.
#
# Environment overrides:
#   OLLAMA_MODEL   LLM used for prior generation   (default qwen2.5:32b)
#   IB_METHOD      gaussian | mine                 (default mine)
#   SEEDS          quoted seed list for the matrix (default "0 1 2 3 4 5 6 7 8 9")
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source /home/iemiedc2026/miniconda3/etc/profile.d/conda.sh
conda activate metascfc-hcp
export PYTHONPATH="$ROOT/src"

OLLAMA_MODEL="${OLLAMA_MODEL:-qwen2.5:32b}"
IB_METHOD="${IB_METHOD:-mine}"
SEEDS="${SEEDS:-0 1 2 3 4 5 6 7 8 9}"

LOG="$ROOT/outputs/iclr/production_run.log"
mkdir -p outputs/iclr outputs/priors/llm outputs/iclr/figures
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo "[$(date '+%F %T')] ICLR production run"
echo "  model=$OLLAMA_MODEL  ib=$IB_METHOD  seeds=($SEEDS)"
echo "=============================================================="

# ---------------------------------------------------------------------------
echo "[1/4] Repack arrays with dual-target intersection QC (real data)"
# ---------------------------------------------------------------------------
python scripts/21_prepare_hcp_labels.py
python scripts/24_pack_hcp_arrays.py    # logs exact drop counts per reason

# ---------------------------------------------------------------------------
echo "[2/4] Generate REAL zero-shot LLM priors via Ollama ($OLLAMA_MODEL)"
# ---------------------------------------------------------------------------
if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null; then
    echo "ERROR: Ollama is not reachable at localhost:11434." >&2
    exit 1
fi
if ! ollama list | grep -q "$OLLAMA_MODEL"; then
    echo "Model $OLLAMA_MODEL not present locally - pulling..."
    ollama pull "$OLLAMA_MODEL"
fi
python scripts/46_generate_llm_priors.py --task "Working Memory" \
    --provider ollama --model "$OLLAMA_MODEL" --controls --retries 3
python scripts/46_generate_llm_priors.py --task "Fluid Intelligence" \
    --provider ollama --model "$OLLAMA_MODEL" --controls --retries 3

# Sanity: priors must exist before the matrix starts.
for f in outputs/priors/llm/working_memory/roi_prior.csv \
         outputs/priors/llm/fluid_intelligence/roi_prior.csv; do
    [ -f "$f" ] || { echo "ERROR: prior missing: $f" >&2; exit 1; }
done

# ---------------------------------------------------------------------------
echo "[3/4] Dual-task matrix (MINE IB tracking, checkpoints enabled)"
# ---------------------------------------------------------------------------
# shellcheck disable=SC2086
python scripts/50_run_dual_task_matrix.py \
    --ib-method "$IB_METHOD" --save-checkpoints --seeds $SEEDS

# ---------------------------------------------------------------------------
echo "[4/4] Information Bottleneck trade-off figure + TikZ snippet"
# ---------------------------------------------------------------------------
python scripts/61_plot_information_bottleneck.py

echo "=============================================================="
echo "[$(date '+%F %T')] Production run COMPLETE."
echo "  summary : outputs/iclr/dual_task_matrix/summary.csv"
echo "  figures : outputs/iclr/figures/ib_tradeoff.{png,pdf,tex}"
echo "  log     : $LOG"
echo "=============================================================="
