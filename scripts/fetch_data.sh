#!/usr/bin/env bash
# Fetch the harmful/harmless prompt splits from andyrdt/refusal_direction (Apache-2.0).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/refusal_direction
for s in harmful_train harmful_val harmful_test harmless_train harmless_val harmless_test; do
  curl -sfL -o "data/refusal_direction/$s.json" \
    "https://raw.githubusercontent.com/andyrdt/refusal_direction/main/dataset/splits/$s.json"
done
echo "fetched $(ls data/refusal_direction | wc -l) files"
