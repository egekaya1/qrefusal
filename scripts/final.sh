#!/usr/bin/env bash
# Final queue (Nov 13-15).
# Steps are resumable: finished output files are skipped on rerun.
set -u
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml
J=mlx-community/Qwen2.5-14B-Instruct-4bit
E2B=e2-dense,e2-mirror,e2-perm1,e2-perm2,e2-gauss
log() { echo "[$(date)] $*" | tee -a logs/final.log; }
run() { local tag="$1"; shift; log "start $tag"; "$@" >> "logs/final_${tag%% *}.log" 2>&1; log "end $tag exit=$?"; }
e2() {  # C3/E2, only once the perturbation script has passed its tests (marker written by hand after validation)
  if [ -f logs/e2_validated ]; then
    run "e2 qwen2.5-3b" python runs/e2_perturb.py --name qwen2.5-3b --cell forced
    run "e2judge qwen2.5-3b" python runs/r1b_judge.py --name qwen2.5-3b --source a2/k8 --rubric v2 --judge $J --configs $E2B
    run "e2 llama3.2-3b" python runs/e2_perturb.py --name llama3.2-3b --cell free
    run "e2judge llama3.2-3b" python runs/r1b_judge.py --name llama3.2-3b --source x2/wjbho --rubric v2 --judge $J --configs $E2B
  else log "skip e2: logs/e2_validated missing"; fi
}
log "final start"
run "e0 qwen2.5-3b"  python runs/e0_determinism.py --name qwen2.5-3b   # C0/E0
run "e0 llama3.2-3b" python runs/e0_determinism.py --name llama3.2-3b  # C0/E0
run "e1gen llama3.2-3b" python runs/x2_sneaky.py --name llama3.2-3b --sets wjbho --configs bf16,8bit-g64,4bit-g64,mxfp4  # C2/E1
run "e1judge llama3.2-3b" python runs/r1b_judge.py --name llama3.2-3b --source x2/wjbho --rubric v2 --judge $J --configs bf16,8bit-g64,4bit-g64,mxfp4  # C2/E1
for m in qwen2.5-3b qwen3-4b; do
  run "e1gen $m" python runs/x2_sneaky.py --name $m --sets wjbho --configs bf16,8bit-g64,4bit-g64  # C2/E1
  run "e1judge $m" python runs/r1b_judge.py --name $m --source x2/wjbho --rubric v2 --judge $J --configs bf16,8bit-g64,4bit-g64  # C2/E1
done
run "e4gen qwen2.5-3b" python runs/a2_depth.py --name qwen2.5-3b --configs bf16,4bit-g64 --ks 8 --sample 150 --max_tokens 384 --out_suffix _len384  # C1/E4
run "e4judge qwen2.5-3b" python runs/r1b_judge.py --name qwen2.5-3b --source a2/k8_len384 --rubric v2 --judge $J --configs bf16,4bit-g64  # C1/E4
run "e4gen llama3.2-3b" python runs/x2_sneaky.py --name llama3.2-3b --sets wjbho --configs bf16,4bit-g64 --sample 150 --max_tokens 384 --out_suffix _len384  # C2/E4
run "e4judge llama3.2-3b" python runs/r1b_judge.py --name llama3.2-3b --source x2/wjbho_len384 --rubric v2 --judge $J --configs bf16,4bit-g64  # C2/E4
run "s8gen qwen2.5-3b" python runs/a2_depth.py --name qwen2.5-3b --configs bf16,4bit-g64 --fixed_prefix "Sure, here is" --tag sure  # C1/S8
run "s8judge qwen2.5-3b" python runs/r1b_judge.py --name qwen2.5-3b --source a2/sure --rubric v2 --judge $J --configs bf16,4bit-g64  # C1/S8
e2  # C3/E2
run "j3 llama3.2-3b" python runs/j2_ensemble.py --judge lg --name llama3.2-3b --source x2/wjbho --configs bf16,4bit-g64 --all  # C2/J3
for m in qwen2.5-3b llama3.2-3b; do  # C0/E3
  run "e3gen $m r3" python runs/a2_depth.py --name $m --configs bf16,4bit-g64 --ks 8 --temp 0.7 --top_p 0.8 --top_k 20 --run 3
  run "e3gen $m r1q" python runs/a2_depth.py --name $m --configs 4bit-g64 --ks 8 --temp 0.7 --top_p 0.8 --top_k 20 --run 1
  run "e3judge $m r3" python runs/r1b_judge.py --name $m --source a2/k8_t0.7_r3 --rubric v2 --judge $J --configs bf16,4bit-g64
  run "e3judge $m r1q" python runs/r1b_judge.py --name $m --source a2/k8_t0.7_r1 --rubric v2 --judge $J --configs 4bit-g64
done
for m in llama3.2-3b qwen3-4b; do  # C1/S8
  run "s8gen $m" python runs/a2_depth.py --name $m --configs bf16,4bit-g64 --fixed_prefix "Sure, here is" --tag sure
  run "s8judge $m" python runs/r1b_judge.py --name $m --source a2/sure --rubric v2 --judge $J --configs bf16,4bit-g64
done
e2  # C3/E2 second chance: picks up whatever is missing if validation landed late
log "final done"
