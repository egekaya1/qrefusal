#!/usr/bin/env bash
# Extension queue (PREREG addendum 4). Starts when scripts/final.sh has finished. Resumable.
set -u
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml
J=mlx-community/Qwen2.5-14B-Instruct-4bit
log() { echo "[$(date)] $*" | tee -a logs/extend.log; }
run() { local tag="$1"; shift; log "start $tag"; "$@" >> "logs/extend_${tag%% *}.log" 2>&1; log "end $tag exit=$?"; }
until grep -q "final done" logs/final.log 2>/dev/null && ! pgrep -f "scripts/final.sh" >/dev/null; do sleep 120; done
log "extend start"
# S9: AWQ and GPTQ builds, same format as the default build (4-bit, group 64, embedding/fallback 4-bit g64)
for m in qwen2.5-3b llama3.2-3b; do
  [ -d models/$m-awq ]  || run "awq $m"  mlx_lm.awq  --model models/$m-bf16 --mlx-path models/$m-awq  --bits 4 --group-size 64 --embed-bits 4 --embed-group-size 64 --num-samples 128 --sequence-length 512   # S9
  [ -d models/$m-gptq ] || run "gptq $m" mlx_lm.gptq --model models/$m-bf16 --mlx-path models/$m-gptq --bits 4 --group-size 64 --fallback-bits 4 --fallback-group-size 64 --num-samples 128 --sequence-length 512   # S9
done
run "s9gen qwen2.5-3b"   python runs/a2_depth.py  --name qwen2.5-3b  --configs awq,gptq --ks 8                                     # S9 forced
run "s9judge qwen2.5-3b" python runs/r1b_judge.py --name qwen2.5-3b  --source a2/k8    --rubric v2 --judge $J --configs awq,gptq   # S9
run "s9gen llama3.2-3b"  python runs/x2_sneaky.py --name llama3.2-3b --sets wjbho --configs awq,gptq                               # S9 free
run "s9judge llama3.2-3b" python runs/r1b_judge.py --name llama3.2-3b --source x2/wjbho --rubric v2 --judge $J --configs awq,gptq  # S9
for m in qwen2.5-3b llama3.2-3b; do run "s9nll $m" python runs/c1_capability.py --name $m --configs awq,gptq; done                # S9 capability
# E2 extension: dose-response along the real rounding direction
run "e2s qwen2.5-3b"      python runs/e2_perturb.py --name qwen2.5-3b  --cell forced --builds e2-scale0.5,e2-scale2               # E2x
run "e2sjudge qwen2.5-3b" python runs/r1b_judge.py  --name qwen2.5-3b  --source a2/k8    --rubric v2 --judge $J --configs e2-scale0.5,e2-scale2
run "e2s llama3.2-3b"     python runs/e2_perturb.py --name llama3.2-3b --cell free   --builds e2-scale0.5,e2-scale2               # E2x
run "e2sjudge llama3.2-3b" python runs/r1b_judge.py --name llama3.2-3b --source x2/wjbho --rubric v2 --judge $J --configs e2-scale0.5,e2-scale2
# M1: refusal-direction projections (forward passes only)
run "m1 qwen2.5-3b forced" python runs/m1_direction.py --name qwen2.5-3b  --cell forced    # M1 primary cell
run "m1 llama3.2-3b free"  python runs/m1_direction.py --name llama3.2-3b --cell free      # M1 primary cell
run "m1 qwen2.5-3b free"   python runs/m1_direction.py --name qwen2.5-3b  --cell free   --builds bf16,8bit-g64,4bit-g64,mxfp4,awq,gptq   # M1 cross-cell
run "m1 llama3.2-3b forced" python runs/m1_direction.py --name llama3.2-3b --cell forced --builds bf16,8bit-g64,4bit-g64,mxfp4,awq,gptq   # M1 cross-cell
# Reporting: per-matrix error table
for m in qwen2.5-3b llama3.2-3b qwen3-4b; do run "err $m" python runs/e2_errors.py --name $m; done
# S5 extension: run-on in a second Qwen2.5 size
[ -d models/qwen2.5-0.5b-mxfp4 ] || run "conv05 mxfp4" python -m mlx_lm convert --hf-path models/qwen2.5-0.5b-bf16 --mlx-path models/qwen2.5-0.5b-mxfp4 -q --q-mode mxfp4
run "r5 qwen2.5-0.5b" python runs/r5_runon.py --name qwen2.5-0.5b --configs bf16,4bit-g64,mxfp4 --dequant mxfp4   # S5
# Qwen3 through the controls (descriptive)
run "e2 qwen3-4b"      python runs/e2_perturb.py --name qwen3-4b --cell forced
run "e2judge qwen3-4b" python runs/r1b_judge.py  --name qwen3-4b --source a2/k8 --rubric v2 --judge $J --configs e2-dense,e2-mirror,e2-perm1,e2-perm2,e2-gauss
# G1: second runtime, last
run "g1 qwen2.5-3b"      python runs/g1_gguf.py --name qwen2.5-3b  --cell forced     # G1
run "g1judge qwen2.5-3b" python runs/r1b_judge.py --name qwen2.5-3b  --source a2/k8    --rubric v2 --judge $J --configs gguf-bf16,gguf-q8_0,gguf-q4km
run "g1 llama3.2-3b"     python runs/g1_gguf.py --name llama3.2-3b --cell free       # G1
run "g1judge llama3.2-3b" python runs/r1b_judge.py --name llama3.2-3b --source x2/wjbho --rubric v2 --judge $J --configs gguf-bf16,gguf-q8_0,gguf-q4km
log "extend done"
