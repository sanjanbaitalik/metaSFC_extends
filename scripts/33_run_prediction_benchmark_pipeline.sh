#!/usr/bin/env bash
# Resumable paper-ready benchmark pipeline for MetaSFC.
# Usage: bash scripts/33_run_prediction_benchmark_pipeline.sh [all|preflight|ridge|mgcn|img|validate|tables]
set -euo pipefail

STAGE="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs/new_prediction_benchmarks

run_logged() {
  local name="$1"; shift
  echo "[$(date -Iseconds)] START $name"
  "$@" 2>&1 | tee "logs/new_prediction_benchmarks/${name}.log"
  echo "[$(date -Iseconds)] DONE  $name"
}

preflight() {
  python -c "import metascfc; print('metascfc import: OK')"
  run_logged smoke_test python scripts/30_smoke_test_new_benchmarks.py
  run_logged unit_tests python -m pytest -q tests/test_new_prediction_benchmarks.py
  run_logged preflight python scripts/31_preflight_prediction_benchmarks.py
}

ridge() {
  run_logged prior_weighted_ridge \
    python scripts/27_run_prior_weighted_ridge.py \
      --config configs/aaai/prior_weighted_ridge.yaml
}

mgcn() {
  run_logged mgcn \
    python scripts/28_run_sota_graph_baselines.py \
      --config configs/aaai/sota_graph_baselines.yaml \
      --models MGCN
}

img() {
  run_logged img_gcn \
    python scripts/28_run_sota_graph_baselines.py \
      --config configs/aaai/sota_graph_baselines.yaml \
      --models IMG_GCN
}

validate() {
  run_logged validate_outputs python scripts/32_validate_prediction_benchmark_outputs.py
}

tables() {
  run_logged build_tables python scripts/29_build_prediction_benchmark_table.py
}

case "$STAGE" in
  all)
    preflight
    ridge
    mgcn
    img
    validate
    tables
    ;;
  preflight) preflight ;;
  ridge) ridge ;;
  mgcn) mgcn ;;
  img) img ;;
  validate) validate ;;
  tables) tables ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    echo "Valid stages: all, preflight, ridge, mgcn, img, validate, tables" >&2
    exit 2
    ;;
esac
