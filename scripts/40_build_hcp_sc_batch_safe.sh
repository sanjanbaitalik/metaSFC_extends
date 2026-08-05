#!/usr/bin/env bash
set -uo pipefail

BATCH_FILE="${1:-}"

if [[ -z "$BATCH_FILE" ]]; then
    echo "[ERROR] Missing batch file."
    echo "Usage: bash scripts/40_build_hcp_sc_batch_safe.sh data/hcp/qc/raw_preflight/machine_A_subjects_ready_for_sc.txt"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

if [[ ! -f "$BATCH_FILE" ]]; then
    echo "[ERROR] Batch file not found: $BATCH_FILE"
    exit 1
fi

LOG_DIR="data/hcp/qc/sc_logs"
mkdir -p "$LOG_DIR"
SUCCESS_LOG="$LOG_DIR/sc_success_$(basename "$BATCH_FILE" .txt).txt"
FAILED_LOG="$LOG_DIR/sc_failed_$(basename "$BATCH_FILE" .txt).txt"
: > "$SUCCESS_LOG"
: > "$FAILED_LOG"

echo "Project root: $PROJECT_ROOT"
echo "Batch file:   $BATCH_FILE"
echo "Success log:  $SUCCESS_LOG"
echo "Failed log:   $FAILED_LOG"

while IFS= read -r SUB || [[ -n "$SUB" ]]; do
    SUB="$(echo "$SUB" | tr -d '\r' | xargs)"
    [[ -z "$SUB" ]] && continue

    echo ""
    echo "===== SC processing subject ${SUB} ====="

    OUT="data/hcp/processed/sc/${SUB}/sc_116.csv"
    if [[ -f "$OUT" ]]; then
        echo "[SKIP] SC already exists: ${SUB}"
        echo "$SUB SKIPPED_EXISTS" >> "$SUCCESS_LOG"
        continue
    fi

    # Pre-check the exact files required by scripts/23_build_hcp_sc_one_subject.sh
    REQUIRED=(
        "data/hcp/raw/${SUB}/Diffusion/data.nii.gz"
        "data/hcp/raw/${SUB}/Diffusion/bvecs"
        "data/hcp/raw/${SUB}/Diffusion/bvals"
        "data/hcp/raw/${SUB}/Diffusion/nodif_brain_mask.nii.gz"
        "data/hcp/raw/${SUB}/T1w/T1w_acpc_dc_restore.nii.gz"
        "data/hcp/raw/${SUB}/xfms/standard2acpc_dc.nii.gz"
    )

    MISSING=0
    for f in "${REQUIRED[@]}"; do
        if [[ ! -s "$f" ]]; then
            echo "[MISSING] $f"
            MISSING=1
        fi
    done

    if [[ "$MISSING" -eq 1 ]]; then
        echo "[SKIP] Missing raw files for ${SUB}."
        echo "$SUB MISSING_RAW" >> "$FAILED_LOG"
        continue
    fi

    bash scripts/23_build_hcp_sc_one_subject.sh "$SUB"
    STATUS=$?

    if [[ "$STATUS" -eq 0 && -f "$OUT" ]]; then
        echo "[OK] ${SUB}"
        echo "$SUB OK" >> "$SUCCESS_LOG"
    else
        echo "[FAIL] ${SUB}, exit code $STATUS"
        echo "$SUB FAILED_EXIT_${STATUS}" >> "$FAILED_LOG"
    fi

done < "$BATCH_FILE"

echo ""
echo "===== SC BATCH DONE ====="
echo "Successes: $(wc -l < "$SUCCESS_LOG")"
echo "Failures:  $(wc -l < "$FAILED_LOG")"
echo "Success log: $SUCCESS_LOG"
echo "Failed log:  $FAILED_LOG"
