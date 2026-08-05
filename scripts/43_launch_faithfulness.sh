#!/bin/bash
# Launch ICLR faithfulness run (experiment 1). Detached, resumable.
LOG=/home/iemiedc2026/Documents/Sanjan/iclr/outputs/aaai/faithfulness_iclr/experiment_log.txt
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] LAUNCH scripts/43_run_iclr_faithfulness.py (resume; methods M2_TRUE M2_RANDOM M3_TRUE M3_RANDOM, seeds 0-9, folds 0-4)"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] env: $(which python) | $(python -c 'import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())')"
} >> "$LOG" 2>&1
cd /home/iemiedc2026/Documents/Sanjan/iclr || exit 1
# shellcheck source=/dev/null
source /home/iemiedc2026/miniconda3/etc/profile.d/conda.sh
conda activate metascfc-hcp
export PYTHONPATH=/home/iemiedc2026/Documents/Sanjan/iclr/src
python -u scripts/43_run_iclr_faithfulness.py >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE exit=$?" >> "$LOG" 2>&1
