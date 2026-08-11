#!/bin/bash
# Launch ICLR faithfulness (scripts/17) for NCR + M2 + M3 configs, seeds 0-4.
LOG=/home/iemiedc2026/Documents/Sanjan/iclr/outputs/aaai/faithfulness/experiment_log_iclr.txt
# shellcheck source=/dev/null
source /home/iemiedc2026/miniconda3/etc/profile.d/conda.sh
conda activate metascfc-hcp
cd /home/iemiedc2026/Documents/Sanjan/iclr || exit 1
export PYTHONPATH=/home/iemiedc2026/Documents/Sanjan/iclr/src
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] LAUNCH 17_run_faithfulness (ICLR configs, seeds 0-4, topk 10, random_repeats 20, mask_mode both, overwrite)"
} >> "$LOG" 2>&1
python -u scripts/17_run_faithfulness.py \
  --configs configs/aaai/network_constrained_ridge.yaml \
            configs/aaai/meta_gat.yaml \
            configs/aaai/two_stage_kernel_ridge.yaml \
  --seeds 0 1 2 3 4 --topk 10 --random_repeats 20 --mask_mode both --overwrite \
  >> "$LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE exit=$?" >> "$LOG" 2>&1
