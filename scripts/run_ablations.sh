#!/usr/bin/env bash
# Run every ablation variant on Chengdu (see scripts/gen_settings.py for the list).
#
#   bash scripts/run_ablations.sh 0
#
# NOTE: the MG-Trans-DP / MG-Trans-DS variants need the alternative compressed
# corpora, so run `python data.py -s chengdu_sts --ablation-compression` first.
set -euo pipefail

CUDA="${1:-0}"
VARIANTS=(v_gdn2 v_transformer wo_purpose wo_kd wo_compress mg_trans_dp mg_trans_ds wo_filter)

for v in "${VARIANTS[@]}"; do
    echo; echo "########## ablation ${v} / pre-training ##########"
    python main.py -s "ablation/${v}_pretrain" --cuda "${CUDA}"
    for task in dp_road dp_gps ate; do
        echo; echo "########## ablation ${v} / ${task} ##########"
        python main.py -s "ablation/${v}_${task}" --cuda "${CUDA}"
    done
    echo; echo "########## ablation ${v} / sts ##########"
    python main.py -s "ablation/${v}_sts" --cuda "${CUDA}"
done

echo; echo "All ablations finished."
