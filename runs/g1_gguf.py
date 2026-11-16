"""G1 (PREREG addendum 4): the same cells in a second runtime, llama.cpp (GGUF).

Builds per model, all made locally from our bf16 checkpoint with a pinned llama.cpp
converter (tools/llama.cpp-src, commit 0ec4027) and Homebrew llama-quantize, no imatrix:
  gguf-bf16   reference (paired within the runtime)
  gguf-q8_0   8-bit null
  gguf-q4km   Q4_K_M, the most common community 4-bit GGUF format
Generation goes through llama-server with the prompt sent as TOKEN IDS from the same HF
tokenizer and chat template as the MLX runs, so prompts and prefills are token-identical
by construction; the server's evaluated-token count is checked against what was sent.
Greedy (temperature 0, top_k 1), 128 tokens, same row schema as a2/x2 so the same judge
and analysis apply.

Usage: python runs/g1_gguf.py --name qwen2.5-3b --cell forced
Writes models/gguf/<name>-<q>.gguf and results/{a2/k8|x2/wjbho}/<name>/gguf-<q>.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

from e2_perturb import cell_dest, cell_items  # noqa: E402
from qrefusal.data import chat_ids, reply_ids  # noqa: E402
from qrefusal.metrics import is_broken, is_refusal  # noqa: E402

CONVERT = ROOT.parent / "tools" / "llama.cpp-src" / "convert_hf_to_gguf.py"
QUANTS = {"gguf-bf16": None, "gguf-q8_0": "Q8_0", "gguf-q4km": "Q4_K_M"}
PORT = 8737


def ensure_ggufs(name):
    d = ROOT / "models" / "gguf"
    d.mkdir(parents=True, exist_ok=True)
    ref = d / f"{name}-gguf-bf16.gguf"
    if not ref.exists():
        tmp = ref.with_suffix(".tmp")
        subprocess.run([sys.executable, str(CONVERT), str(ROOT / "models" / f"{name}-bf16"),
                        "--outtype", "bf16", "--outfile", str(tmp)], check=True)
        tmp.rename(ref)
    for b, q in QUANTS.items():
        f = d / f"{name}-{b}.gguf"
        if q and not f.exists():
            tmp = f.with_suffix(".tmp")
            subprocess.run(["llama-quantize", str(ref), str(tmp), q], check=True)
            tmp.rename(f)
    return {b: d / f"{name}-{b}.gguf" for b in QUANTS}


def post(path, payload):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def serve(model_path):
    p = subprocess.Popen(["llama-server", "-m", str(model_path), "--port", str(PORT), "-ngl", "99",
                          "-c", "4096", "--parallel", "1", "--no-webui"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(240):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "ok":
                    return p
        except Exception:
            time.sleep(1)
    p.kill()
    raise RuntimeError("llama-server did not become healthy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cell", choices=["forced", "free"], required=True)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from mlx_lm.utils import load_tokenizer  # tokenizer only, no weights
    tok = load_tokenizer(ROOT / "models" / f"{args.name}-bf16")
    items = cell_items(args.name, args.cell)[: args.limit]
    files = ensure_ggufs(args.name)
    for b, f in files.items():
        dest = cell_dest(args.name, args.cell, b)
        if dest.exists():
            continue
        srv = serve(f)
        try:
            rows, t0 = [], time.time()
            for it in items:
                ids = chat_ids(tok, it["request"]) + it["prefix_ids"]
                r = post("/completion", {"prompt": ids, "n_predict": args.max_tokens, "temperature": 0.0,
                                         "top_k": 1, "cache_prompt": False, "seed": 0})
                n_eval = r.get("tokens_evaluated", r.get("timings", {}).get("prompt_n"))
                assert n_eval in (None, len(ids)), f"server evaluated {n_eval} tokens, sent {len(ids)}"
                gen = r["content"]
                reason = "stop" if r.get("stopped_eos") or r.get("stop_type") in ("eos", "word") else "length"
                pre = tok.decode(it["prefix_ids"]) if it["prefix_ids"] else ""
                sep = " " if pre and gen and not pre[-1].isspace() and gen[0].isalnum() else ""
                text = pre + sep + gen
                row = {"kind": "harmful", "i": it["i"], "request": it["request"], "reply": text, "finish_reason": reason,
                       "refused": is_refusal(text), "broken": is_broken(text, reply_ids(tok, text)), "config": b}
                if args.cell == "forced":
                    row.update({"k": 8, "prefix": pre, "prefix_ids": it["prefix_ids"], "temp": 0.0, "run": 0})
                else:
                    row["style"] = "wjb_heldout"
                rows.append(row)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(json.dumps(r) for r in rows))
            print(f"{args.name} {args.cell} {b}: {len(rows)} in {time.time() - t0:.0f}s", flush=True)
        finally:
            srv.terminate()
            srv.wait(timeout=30)


if __name__ == "__main__":
    main()
