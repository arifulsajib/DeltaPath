#!/usr/bin/env bash
# Hyper-parameter, loss-weight and training-set-size studies,
# all on Chengdu destination prediction (road segment).
#
#   bash scripts/run_sweeps.sh 0
set -euo pipefail

CUDA="${1:-0}"

for cfg in settings/sweep/*.json; do
    name="sweep/$(basename "${cfg}" .json)"
    echo; echo "########## ${name} ##########"
    python main.py -s "${name}" --cuda "${CUDA}"
done

echo; echo "All sweeps finished."
