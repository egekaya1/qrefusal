"""E0 (claim C0): is greedy generation on this machine deterministic?

Regenerates the first --n prompts of the forced-start setting (k = 8, bf16) and compares
each reply token-for-token with the stored a2/k8 bf16 reply. Rung 0 of the null ladder:
if reruns differ, every text-identity number is read relative to this rate.
Prints and stores counts only; no reply text is printed.

Usage: python runs/e0_determinism.py --name qwen2.5-3b --n 100
Writes results/e0/<name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrefusal.data import chat_ids, greedy_reply, reply_ids  # noqa: E402
from qrefusal.models import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_tokens", type=int, default=128)
    args = ap.parse_args()
    if (ROOT / "results" / "e0" / f"{args.name}.json").exists():
        print(f"skip {args.name}: already done", flush=True)
        return
    stored = {r["i"]: r for r in map(json.loads, (ROOT / "results/a2/k8" / args.name / "bf16.jsonl").read_text().splitlines())}
    items = [json.loads(l) for l in (ROOT / "results/e7" / args.name / "prefixes_full.jsonl").read_text().splitlines()]
    items = [it for it in items if it["i"] in stored][: args.n]
    model, tok = load(str(ROOT / "models" / f"{args.name}-bf16"))
    same = same32 = 0
    for it in items:
        s = stored[it["i"]]
        ids = chat_ids(tok, it["request"]) + s["prefix_ids"]
        gen, _ = greedy_reply(model, tok, ids, args.max_tokens)
        a, b = reply_ids(tok, s["reply"][len(s["prefix"]):].lstrip()), reply_ids(tok, gen.lstrip())
        same += a == b
        same32 += a[:32] == b[:32]
    res = {"n": len(items), "identical": same / len(items), "identical_first32": same32 / len(items)}
    out = ROOT / "results" / "e0"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(res, indent=1))
    print(args.name, res, flush=True)


if __name__ == "__main__":
    main()
