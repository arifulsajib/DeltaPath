#!/usr/bin/env bash
# Run the full benchmark for one city: pre-training, every downstream task, efficiency.
#
#   bash scripts/run_all.sh chengdu 0
#
# Arg 1: city (chengdu | xian).  Arg 2: CUDA device index (-1 for CPU).
set -euo pipefail

CITY="${1:-chengdu}"
CUDA="${2:-0}"

export SETTINGS_CACHE_DIR="${SETTINGS_CACHE_DIR:-settings/cache}"
export MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-saved_model}"
export PRED_SAVE_DIR="${PRED_SAVE_DIR:-predictions}"

run () { echo; echo "########## $1 ##########"; python main.py -s "$1" --cuda "${CUDA}"; }

# 1. Both pre-training stages, saved once and reused by every downstream task.
run "${CITY}_pretrain"

# 2. Destination prediction (road segment + GPS) and arrival time estimation,
#    each with and without encoder fine-tuning.
for task in dp_road dp_gps ate; do
    run "${CITY}_${task}"
    run "${CITY}_${task}_noft"
done

# 3. Similar trajectory search (parameters frozen after pre-training).
run "${CITY}_sts"

# 4. Efficiency (model size, embedding time, length scaling).
python scripts/benchmark_efficiency.py -s "${CITY}_sts" --cuda "${CUDA}" \
    --out "${PRED_SAVE_DIR}/${CITY}_efficiency.json"

echo; echo "All ${CITY} experiments finished. Metrics are under ${PRED_SAVE_DIR}."
