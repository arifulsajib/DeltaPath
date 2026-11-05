#!/usr/bin/env bash
# End-to-end smoke test on a synthetic toy city (CPU, a few minutes).
# Exercises preprocessing, both pre-training stages, fine-tuning, every
# evaluation path and the architectural variants.
#
#   bash scripts/smoke_test.sh
set -euo pipefail
cd "$(dirname "$0")/.."

python scripts/make_toy_data.py --out samples --city toy
python data.py -s local_test_search
python main.py -s local_test --cuda -1           # pre-train + destination prediction
python main.py -s local_test_search --cuda -1    # similar trajectory search
python main.py -s local_test_variants --cuda -1  # GDN-2 / Transformer / no-compression variants

echo; echo "Smoke test passed."
