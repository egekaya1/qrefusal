#!/usr/bin/env bash
# Build every MLX format used in the paper for one model.
# Usage: scripts/convert.sh <hf-id> <short-name>
#   scripts/convert.sh Qwen/Qwen2.5-3B-Instruct qwen2.5-3b
set -euo pipefail
HF="$1"; NAME="$2"; OUT="models"
mkdir -p "$OUT"

python -m mlx_lm convert --hf-path "$HF" --mlx-path "$OUT/$NAME-bf16" --dtype bfloat16
# affine round-to-nearest: 8-bit null, the default 4-bit build, and a finer group size
for cfg in "8 64" "4 64" "4 32"; do
  set -- $cfg
  python -m mlx_lm convert --hf-path "$HF" --mlx-path "$OUT/$NAME-${1}bit-g${2}" \
    -q --q-bits "$1" --q-group-size "$2"
done
python -m mlx_lm convert --hf-path "$HF" --mlx-path "$OUT/$NAME-mixed_4_6" -q --quant-predicate mixed_4_6
python -m mlx_lm convert --hf-path "$HF" --mlx-path "$OUT/$NAME-mxfp4" -q --q-mode mxfp4
# calibrated quantizers in the same format (4-bit, group 64, embedding/fallback layers included)
mlx_lm.awq --model "$OUT/$NAME-bf16" --mlx-path "$OUT/$NAME-awq" --bits 4 --group-size 64 \
  --embed-bits 4 --embed-group-size 64 --num-samples 128 --sequence-length 512
mlx_lm.gptq --model "$OUT/$NAME-bf16" --mlx-path "$OUT/$NAME-gptq" --bits 4 --group-size 64 \
  --fallback-bits 4 --fallback-group-size 64 --num-samples 128 --sequence-length 512
du -sh "$OUT"/"$NAME"-*
