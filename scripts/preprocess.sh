#!/usr/bin/env bash
# Build every preprocessing artefact for one city.
#
#   bash scripts/preprocess.sh chengdu
#   bash scripts/preprocess.sh xian
#
# Expects ./raw_data/<city>.h5 (keys: trips, pois, road_info),
#         ./raw_data/<city>_{train,valid,test}.h5 (key: trips),
#         ./raw_data/<city>_{road,poi}_embed.npy
set -euo pipefail

CITY="${1:-chengdu}"

echo "==> Preprocessing ${CITY}: trajectories, road graph, rule-based compression, STS labels"
python data.py -s "${CITY}_sts" --ablation-compression

echo "==> Done. Artefacts are under ${PROCESSED_DATA_DIR:-./processed_data}"
