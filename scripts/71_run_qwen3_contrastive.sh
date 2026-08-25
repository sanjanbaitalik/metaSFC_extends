#!/usr/bin/env bash
# =============================================================================
# Contrastive Prompting + Qwen 3.8 27B execution (Semantic-Blurring fix).
#
# 1. Generates DISCRIMINATIVE zero-shot priors for both tasks with
#    qwen3.8:27b under --contrastive (domain-exclusion criteria), saved to
#    SEPARATE slugs so the qwen2.5:32b production priors and the 80-evaluation
#    dual-task matrix remain untouched.
# 2. Reports the prior-discriminability diagnostic (the old pair correlated
#    at r = 0.86 - the root cause of the alpha gate's quasi-stationarity).
# 3. Runs the targeted 1-seed Alpha-Rescue with the new priors, writing to
#    outputs/iclr/contrastive_qwen3/alpha_rescue/.
# 4. Regenerates the LaTeX tables (Table 2 reads model + prompting method
#    dynamically from provenance.json).
#
# Environment overrides: OLLAMA_MODEL (default qwen3.8:27b), SEED, FOLD.
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# shellcheck disable=SC1091
source /home/iemiedc2026/miniconda3/etc/profile.d/conda.sh
conda activate metascfc-hcp
export PYTHONPATH="$ROOT/src"

MODEL="${OLLAMA_MODEL:-qwen3.8:27b}"
SEED="${SEED:-0}"
FOLD="${FOLD:-0}"
WM_SLUG="working_memory_contrastive_qwen3"
FL_SLUG="fluid_intelligence_contrastive_qwen3"
OUT="$ROOT/outputs/iclr/contrastive_qwen3"
LOG="$OUT/run.log"
mkdir -p "$OUT" outputs/priors/llm
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo "[$(date '+%F %T')] Contrastive prompting run - model=$MODEL"
echo "=============================================================="

if ! curl -s --max-time 3 http://localhost:11434/api/tags >/dev/null; then
    echo "ERROR: Ollama is not reachable at localhost:11434." >&2
    exit 1
fi
if ! ollama list | grep -q "$MODEL"; then
    echo "Model $MODEL not present locally - pulling..."
    ollama pull "$MODEL"
fi

echo "[1/4] Contrastive prior generation ($MODEL)"
python scripts/46_generate_llm_priors.py --task "Working Memory" \
    --provider ollama --model "$MODEL" --contrastive --controls --retries 3 \
    --slug "$WM_SLUG"
python scripts/46_generate_llm_priors.py --task "Fluid Intelligence" \
    --provider ollama --model "$MODEL" --contrastive --controls --retries 3 \
    --slug "$FL_SLUG"

echo "[2/4] Prior-discriminability diagnostic (Semantic-Blurring check)"
python - "$WM_SLUG" "$FL_SLUG" <<'PY'
import sys
import numpy as np
import pandas as pd

wm = pd.read_csv(f"outputs/priors/llm/{sys.argv[1]}/roi_prior.csv").sort_values("roi_index").prior_score.to_numpy()
fl = pd.read_csv(f"outputs/priors/llm/{sys.argv[2]}/roi_prior.csv").sort_values("roi_index").prior_score.to_numpy()
r = float(np.corrcoef(wm, fl)[0, 1])
tw = set(np.argsort(wm)[-10:].tolist()); tf = set(np.argsort(fl)[-10:].tolist())
print(f"Contrastive prior correlation: r = {r:.3f} (qwen2.5:32b zero-shot baseline was 0.862)")
print(f"Top-10 overlap: {len(tw & tf)}/10 (baseline 6/10)")
if r > 0.8:
    print("WARNING: contrastive prompting did NOT break the correlation; "
          "the alpha gate will still lack contrast.")
else:
    print("Semantic Blurring BROKEN: the priors are discriminative - "
          "the alpha gate now has opposing directions to learn.")
PY

echo "[3/4] Targeted Alpha-Rescue (seed $SEED, fold $FOLD, contrastive priors)"
python scripts/62_run_alpha_rescue.py --seed "$SEED" --fold "$FOLD" \
    --matched-prior "outputs/priors/llm/$FL_SLUG/roi_prior.csv" \
    --mismatched-prior "outputs/priors/llm/$WM_SLUG/roi_prior.csv" \
    --out-dir "$OUT/alpha_rescue"

echo "[4/4] Regenerate LaTeX tables (provenance-driven rows)"
python scripts/70_generate_iclr_latex_tables.py --include-contrastive

echo "=============================================================="
echo "[$(date '+%F %T')] Contrastive run COMPLETE."
echo "  priors : outputs/priors/llm/{$WM_SLUG,$FL_SLUG}/"
echo "  rescue : $OUT/alpha_rescue/alpha_rescue.json"
echo "  tables : outputs/iclr/tables/"
echo "=============================================================="
