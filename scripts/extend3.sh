#!/usr/bin/env bash
# Monday buffer: rebuild the failed Qwen2.5 GPTQ with the default 6-bit fallback, and
# rerun the 0.5B run-on rows that are still 3-prompt smoke-test leftovers.
# Starts after extend2.sh. Nothing here overwrites an existing result.
set -u
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml
J=mlx-community/Qwen2.5-14B-Instruct-4bit
log() { echo "[$(date)] $*" | tee -a logs/extend3.log; }
run() { local tag="$1"; shift; log "start $tag"; "$@" >> "logs/extend3_${tag%% *}.log" 2>&1; log "end $tag exit=$?"; }

until grep -q "extend2 done" logs/extend2.log 2>/dev/null && ! pgrep -f "scripts/extend2.sh" >/dev/null; do sleep 120; done
log "extend3 start"

# --- 1. GPTQ rebuild -------------------------------------------------------------------
# The registered build (4-bit fallback layers) lost about 9.3 nats per token. This rebuild with
# mlx_lm's default 6-bit fallback tests whether the fallback is the cause (it is not; Appendix A).
[ -d models/qwen2.5-3b-gptq6 ] || run "gptq6 build" \
  mlx_lm.gptq --model models/qwen2.5-3b-bf16 --mlx-path models/qwen2.5-3b-gptq6 \
  --bits 4 --group-size 64 --num-samples 128 --sequence-length 512

# Capability first: if this build is also broken, the NLL says so before we spend an hour judging.
run "gptq6 nll"    python runs/c1_capability.py --name qwen2.5-3b --configs gptq6
run "gptq6 gen"    python runs/a2_depth.py      --name qwen2.5-3b --configs gptq6 --ks 8
run "gptq6 judge"  python runs/r1b_judge.py     --name qwen2.5-3b --source a2/k8 --rubric v2 --judge $J --configs gptq6
run "gptq6 m1"     python runs/m1_direction.py  --name qwen2.5-3b --cell forced --builds gptq6

# --- 2. 0.5B run-on at full size -------------------------------------------------------
# bf16, 4bit-g64 and 4bit-g64-dequant are 2-line smoke-test leftovers; mxfp4 rows are already
# at the full 200. Move the stubs aside (do not delete) so the runner regenerates them.
mkdir -p results/r5/qwen2.5-0.5b/_smoketest
for f in bf16 4bit-g64 4bit-g64-dequant; do
  p="results/r5/qwen2.5-0.5b/$f.jsonl"
  if [ -f "$p" ] && [ "$(wc -l < "$p")" -lt 50 ]; then
    mv "$p" "results/r5/qwen2.5-0.5b/_smoketest/$f.jsonl"
    log "set aside stub $f ($(wc -l < "results/r5/qwen2.5-0.5b/_smoketest/$f.jsonl") lines)"
  fi
done
run "r5 full qwen2.5-0.5b" python runs/r5_runon.py --name qwen2.5-0.5b \
  --configs bf16,4bit-g64 --dequant 4bit-g64 --n 200

log "extend3 done"
