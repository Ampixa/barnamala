#!/usr/bin/env bash
# Train one config across 5 seeds: ./scripts/run_seeds.sh configs/student.yaml
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="$1"
for seed in 0 1 2 3 4; do
    echo "=== $CONFIG seed $seed ==="
    .venv/bin/python -m devnet.train --config "$CONFIG" --seed "$seed"
done
