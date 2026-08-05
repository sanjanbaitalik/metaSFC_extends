#!/usr/bin/env bash
set -euo pipefail

BATCH_FILE=${1:-data/hcp/manifest/batches/batch_01.txt}

while read -r SUB; do
    if [[ -z "$SUB" ]]; then
        continue
    fi

    echo ""
    echo "===== SC processing subject ${SUB} ====="

    OUT="data/hcp/processed/sc/${SUB}/sc_116.csv"

    if [[ -f "$OUT" ]]; then
        echo "[SKIP] SC already exists: ${SUB}"
        continue
    fi

    bash scripts/23_build_hcp_sc_one_subject.sh "$SUB"

done < "$BATCH_FILE"