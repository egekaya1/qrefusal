#!/usr/bin/env bash
# Addendum-5 queue: the same-prompt bridge (B1) and three more random draws per cell. Starts after extend.sh.
set -u
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml
J=mlx-community/Qwen2.5-14B-Instruct-4bit
EX=e2-perm3,e2-gauss2,e2-gauss3
log() { echo "[$(date)] $*" | tee -a logs/extend2.log; }
run() { local tag="$1"; shift; log "start $tag"; "$@" >> "logs/extend2_${tag%% *}.log" 2>&1; log "end $tag exit=$?"; }
until grep -q "extend done" logs/extend.log 2>/dev/null && ! pgrep -f "scripts/extend.sh" >/dev/null; do sleep 120; done
log "extend2 start"
for m in llama3.2-3b qwen2.5-3b; do   # B1: held-out prompts with a forced generic start
  run "b1gen $m"   python runs/x2_sneaky.py --name $m --sets wjbho --configs bf16,4bit-g64 --fixed_prefix "Sure, here is" --out_suffix _sure
  run "b1judge $m" python runs/r1b_judge.py --name $m --source x2/wjbho_sure --rubric v2 --judge $J --configs bf16,4bit-g64
done
run "e2x qwen2.5-3b"       python runs/e2_perturb.py --name qwen2.5-3b  --cell forced --builds $EX   # E2, addendum 5
run "e2xjudge qwen2.5-3b"  python runs/r1b_judge.py  --name qwen2.5-3b  --source a2/k8    --rubric v2 --judge $J --configs $EX
run "e2x llama3.2-3b"      python runs/e2_perturb.py --name llama3.2-3b --cell free   --builds $EX
run "e2xjudge llama3.2-3b" python runs/r1b_judge.py  --name llama3.2-3b --source x2/wjbho --rubric v2 --judge $J --configs $EX
log "extend2 done"
