"""X2: free-start prompts (no prefill), bf16 vs quantized builds.

Prompt sets:
  sorry  SORRY-Bench (Xie et al. 2024): the same --n_base requests in plain form
         and in each --styles rewrite, so blatant vs disguised is paired by intent.
  wjb    WildJailbreak eval (Jiang et al. 2024): --n_wjb adversarial_harmful prompts
         (in-the-wild jailbreak tactics), plus all 210 adversarial_benign prompts
         as an over-refusal control. This is the 200-prompt pilot.
  wjbho  The confirmatory free-start set: --n_heldout further adversarial_harmful
         prompts, disjoint by text from the pilot (PREREG addendum 3).
Greedy, --max_tokens tokens. Rows store the request, so the judge reads it from the row:
    python runs/r1b_judge.py --name <m> --source x2/<set> --rubric v2

Usage: python runs/x2_sneaky.py --name qwen2.5-3b --sets sorry,wjb
Writes results/x2/<set>/<name>/<config>.jsonl (resumable per file)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrefusal.data import chat_ids, greedy_reply, reply_ids  # noqa: E402
from qrefusal.metrics import is_broken, is_refusal  # noqa: E402
from qrefusal.models import load  # noqa: E402

STYLES = "role_play,evidence-based_persuasion,technical_terms,misspellings"


def sorry_items(n_base, styles, seed=0):
    d = ROOT / "data" / "sorry"
    base = [json.loads(l) for l in (d / "question.jsonl").read_text().splitlines()]
    ids = sorted(random.Random(seed).sample([q["question_id"] for q in base], n_base))
    items = []
    for style in ["base", *styles]:
        f = d / ("question.jsonl" if style == "base" else f"question_{style}.jsonl")
        qs = {q["question_id"]: q for q in map(json.loads, f.read_text().splitlines())}
        for qid in ids:
            q = qs[qid]
            items.append({"kind": "harmful", "request": q["turns"][0], "style": style,
                          "question_id": qid, "category": q["category"]})
    return items


def wjb_items(n, seed=0):
    csv.field_size_limit(sys.maxsize)
    with open(ROOT / "data" / "wildjailbreak" / "eval" / "eval.tsv") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    harm = [r for r in rows if r["data_type"] == "adversarial_harmful"]
    benign = [r for r in rows if r["data_type"] == "adversarial_benign"]
    harm = random.Random(seed).sample(harm, n)
    return ([{"kind": "harmful", "request": r["adversarial"], "style": "wjb"} for r in harm]
            + [{"kind": "benign", "request": r["adversarial"], "style": "wjb_benign"} for r in benign])


def wjb_heldout_items(n, used=200, seed=1):
    """E1 (PREREG addendum 3): n adversarial_harmful prompts disjoint BY TEXT from the
    `used` pilot prompts drawn by wjb_items(used). No benign rows (the pilot has them)."""
    csv.field_size_limit(sys.maxsize)
    with open(ROOT / "data" / "wildjailbreak" / "eval" / "eval.tsv") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    pilot = {it["request"] for it in wjb_items(used) if it["kind"] == "harmful"}
    pool, seen = [], set()
    for r in rows:
        t = r["adversarial"]
        if r["data_type"] == "adversarial_harmful" and t not in pilot and t not in seen:
            seen.add(t)
            pool.append(t)
    pick = random.Random(seed).sample(pool, n)
    assert not set(pick) & pilot
    return [{"kind": "harmful", "request": t, "style": "wjb_heldout"} for t in pick]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--configs", default="bf16,4bit-g64,mxfp4")
    ap.add_argument("--sets", default="sorry,wjb")
    ap.add_argument("--n_base", type=int, default=120)
    ap.add_argument("--styles", default=STYLES)
    ap.add_argument("--n_wjb", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only (smoke tests)")
    ap.add_argument("--n_heldout", type=int, default=400)
    ap.add_argument("--sample", type=int, default=None,
                    help="E4: a fixed random subset of N prompts (ids keep their full-set index)")
    ap.add_argument("--out_suffix", default="", help="E4: write results/x2/<set><suffix>/ instead")
    ap.add_argument("--fixed_prefix", default=None, help="B1: prefill this string (as token ids) on every prompt")
    args = ap.parse_args()

    sets = {"sorry": lambda: sorry_items(args.n_base, args.styles.split(",")),
            "wjb": lambda: wjb_items(args.n_wjb),
            "wjbho": lambda: wjb_heldout_items(args.n_heldout)}
    for cfg in args.configs.split(","):
        model = tok = None
        for s in args.sets.split(","):
            out = ROOT / "results" / "x2" / (s + args.out_suffix) / args.name
            out.mkdir(parents=True, exist_ok=True)
            dest = out / f"{cfg}.jsonl"
            if dest.exists():
                continue
            if not (ROOT / "models" / f"{args.name}-{cfg}").exists():
                print(f"skip {cfg}: no model dir", flush=True)
                continue
            if model is None:
                model, tok = load(str(ROOT / "models" / f"{args.name}-{cfg}"))
            items = [{**it, "i": i} for i, it in enumerate(sets[s]())][: args.limit]
            if args.sample:
                keep = set(random.Random(f"e4-{s}").sample(range(len(items)), args.sample))
                items = [it for it in items if it["i"] in keep]
            rows, t0 = [], time.time()
            for it in items:
                i = it["i"]
                pre_ids = reply_ids(tok, args.fixed_prefix) if args.fixed_prefix else []
                gen, reason = greedy_reply(model, tok, chat_ids(tok, it["request"]) + pre_ids, args.max_tokens)
                pre = tok.decode(pre_ids) if pre_ids else ""
                sep = " " if pre and gen and not pre[-1].isspace() and gen[0].isalnum() else ""
                text = pre + sep + gen
                rows.append({**it, "i": i, "reply": text, "finish_reason": reason, "refused": is_refusal(text),
                             "broken": is_broken(text, reply_ids(tok, text)), "config": cfg,
                             **({"prefix": pre, "prefix_ids": pre_ids} if pre_ids else {})})
            dest.write_text("\n".join(json.dumps(r) for r in rows))
            harm = [r for r in rows if r["kind"] == "harmful"]
            print(f"{args.name} {cfg} {s}: {len(rows)} in {time.time() - t0:.0f}s, keyword refusal on harmful "
                  f"{sum(r['refused'] for r in harm) / max(1, len(harm)):.1%}", flush=True)
        del model


if __name__ == "__main__":
    main()
