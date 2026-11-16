#!/usr/bin/env bash
# Dense W+E on every prompt of each control cell (it ran on the first 150 only).
# Answers whether the dense build reproduces the real 4-bit shift on the full cell,
# not only its likelihood and first token. Declared deviation; reported in Appendix A.
# The 150-prompt outputs are moved aside, not deleted.
set -u
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml
J=mlx-community/Qwen2.5-14B-Instruct-4bit
log() { echo "[$(date)] $*" | tee -a logs/extend4.log; }
run() { local tag="$1"; shift; log "start $tag"; "$@" >> "logs/extend4_${tag%% *}.log" 2>&1; log "end $tag exit=$?"; }
keep() {  # move the 150-prompt dense outputs out of the way
  local src="results/$1/$2" dst="results/_superseded/dense_n150/$1/$2"
  mkdir -p "$dst"
  for f in e2-dense.jsonl judge_v2_e2-dense.jsonl; do
    [ -f "$src/$f" ] && [ ! -f "$dst/$f" ] && mv "$src/$f" "$dst/$f"
  done
}
log "extend4 start"
keep a2/k8 qwen2.5-3b
run "dense qwen2.5-3b" python runs/e2_perturb.py --name qwen2.5-3b --cell forced --builds e2-dense --n_dense 562
run "densejudge qwen2.5-3b" python runs/r1b_judge.py --name qwen2.5-3b --source a2/k8 --rubric v2 --judge $J --configs e2-dense
keep x2/wjbho llama3.2-3b
run "dense llama3.2-3b" python runs/e2_perturb.py --name llama3.2-3b --cell free --builds e2-dense --n_dense 400
run "densejudge llama3.2-3b" python runs/r1b_judge.py --name llama3.2-3b --source x2/wjbho --rubric v2 --judge $J --configs e2-dense
log "extend4 done"
