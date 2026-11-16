"""A2: the prefill-depth curve on the full harmful_test split (572 prompts).

For each prompt the assistant turn is pre-filled with the first k tokens of the
abliterated bf16 model's compliance reply (E7b), k in --ks (0 = no prefill), and
the model continues greedily. The same prefixes are used for every config of a
model, so bf16 vs quantized is a paired comparison per (prompt, k).
Rows store the request, so the judge reads it from the row:
    python runs/r1b_judge.py --name <m> --source a2/k<k> --rubric v2

Usage: python runs/a2_depth.py --name qwen2.5-3b --configs bf16,4bit-g64,mxfp4 --ks 0,2,8
Writes results/a2/k<k>/<name>/<config>.jsonl (resumable per file)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrefusal.data import chat_ids, greedy_reply, reply_ids, sampled_reply  # noqa: E402
from qrefusal.metrics import is_broken, is_refusal  # noqa: E402
from qrefusal.models import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--configs", default="bf16,4bit-g64,mxfp4")
    ap.add_argument("--ks", default="0,2,8")
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke tests)")
    ap.add_argument("--temp", type=float, default=0.0, help=">0 samples instead of greedy (P3 seed floor)")
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--run", type=int, default=0, help="run index; with --temp it seeds and names the output dir")
    ap.add_argument("--fixed_prefix", default=None,
                    help='S8: prefill this string for every prompt instead of the abliterated prefix (e.g. "Sure, here is"); '
                         "writes results/a2/<--tag>/; --ks is ignored")
    ap.add_argument("--tag", default="sure", help="output dir name for --fixed_prefix")
    ap.add_argument("--sample", type=int, default=None, help="E4: fixed random subset of N prompts")
    ap.add_argument("--out_suffix", default="", help="E4: append to the output dir name (e.g. _len384)")
    args = ap.parse_args()

    src = ROOT / "results" / "e7" / args.name / "prefixes_full.jsonl"
    items = [json.loads(l) for l in src.read_text().splitlines()][: args.limit]
    if args.sample:
        import random
        keep = set(random.Random("e4-a2").sample([it["i"] for it in items], args.sample))
        items = [it for it in items if it["i"] in keep]
    for cfg in args.configs.split(","):
        model = tok = None
        for k in ([None] if args.fixed_prefix else map(int, args.ks.split(","))):
            if args.fixed_prefix:
                tag = args.tag if args.temp == 0 else f"{args.tag}_t{args.temp}_r{args.run}"
            else:
                tag = f"k{k}" if args.temp == 0 else f"k{k}_t{args.temp}_r{args.run}"
            out = ROOT / "results" / "a2" / (tag + args.out_suffix) / args.name
            out.mkdir(parents=True, exist_ok=True)
            dest = out / f"{cfg}.jsonl"
            if dest.exists():
                continue
            if not (ROOT / "models" / f"{args.name}-{cfg}").exists():
                print(f"skip {cfg}: no model dir", flush=True)
                continue
            if model is None:
                model, tok = load(str(ROOT / "models" / f"{args.name}-{cfg}"))
            rows, t0 = [], time.time()
            for it in items:
                # Prefill by token ids (no string round-trip), so every config sees identical tokens.
                if args.fixed_prefix:
                    pre_ids = reply_ids(tok, args.fixed_prefix)
                else:
                    pre_ids = reply_ids(tok, it["reply"])[:k] if k else []
                pre = tok.decode(pre_ids) if pre_ids else ""
                ids = chat_ids(tok, it["request"]) + pre_ids
                if args.temp == 0:
                    gen, reason = greedy_reply(model, tok, ids, args.max_tokens)
                else:
                    # Per-prompt, per-run seed so runs are independent but reproducible.
                    seed = (hash((it["i"], args.run)) % (2**31 - 1))
                    gen, reason = sampled_reply(model, tok, ids, args.max_tokens, args.temp, args.top_p, args.top_k, seed)
                sep = " " if pre and gen and not pre[-1].isspace() and gen[0].isalnum() else ""
                text = pre + sep + gen
                rows.append({"kind": "harmful", "i": it["i"], "request": it["request"], "k": len(pre_ids) if args.fixed_prefix else k, "prefix": pre,
                             "temp": args.temp, "run": args.run,
                             "prefix_ids": pre_ids, "finish_reason": reason, "reply": text, "refused": is_refusal(text),
                             "broken": is_broken(text, reply_ids(tok, text)), "config": cfg})
            dest.write_text("\n".join(json.dumps(r) for r in rows))
            print(f"{args.name} {cfg} {tag}: {len(rows)} in {time.time() - t0:.0f}s, "
                  f"keyword refusal {sum(r['refused'] for r in rows) / len(rows):.1%}", flush=True)
        del model


if __name__ == "__main__":
    main()
