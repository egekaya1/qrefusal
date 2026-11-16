"""C1: capability control. Is any format simply a broken model?

Two measures per format, both teacher-forced (no generation):
  nll   mean per-token negative log-likelihood of bf16's own replies to harmless
        prompts. Scored on the SAME text for every format, so it is comparable.
  bpw   bits per weight, from the on-disk weight bytes (sanity column for tables).
A format whose refusal behaviour changes but whose NLL is close to bf16's is not
"broken"; a format with a large NLL jump is.

Usage: python runs/c1_capability.py --name qwen2.5-3b --configs bf16,8bit-g64,4bit-g64,4bit-g32,mixed_4_6,mxfp4
Writes results/c1/<name>.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrefusal.data import chat_ids, reply_ids, split  # noqa: E402
from qrefusal.hooks import forward  # noqa: E402
from qrefusal.models import load, weight_bytes  # noqa: E402


def n_params(model):
    """Number of stored parameter elements (quantized weights count as packed uint32)."""
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--configs", default="bf16,8bit-g64,4bit-g64,4bit-g32,mixed_4_6,mxfp4")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_reply_tokens", type=int, default=128)
    args = ap.parse_args()

    out = ROOT / "results" / "c1"
    out.mkdir(parents=True, exist_ok=True)
    dest = out / f"{args.name}.json"
    res = json.loads(dest.read_text()) if dest.exists() else {}

    # Reference text: bf16's own replies to harmless prompts (from R1).
    r1 = ROOT / "results" / "r1" / args.name / "bf16.jsonl"
    rows = [json.loads(l) for l in r1.read_text().splitlines()]
    rows = [r for r in rows if r["kind"] == "harmless" and not r["broken"]][: args.n]
    prompts = split("harmless_test", 300)
    print(f"{args.name}: scoring {len(rows)} harmless replies", flush=True)

    for cfg in args.configs.split(","):
        if cfg in res and isinstance(res.get(cfg), dict):
            continue
        path = ROOT / "models" / f"{args.name}-{cfg}"
        if not path.exists():
            print(f"skip {cfg}: no model dir", flush=True)
            continue
        model, tok = load(str(path))
        t0, tot_nll, tot_tok = time.time(), 0.0, 0
        for r in rows:
            p = chat_ids(tok, prompts[r["i"]])
            c = reply_ids(tok, r["reply"])[: args.max_reply_tokens]
            if not c:
                continue
            logits, _ = forward(model, mx.array([p + c]))
            lp = logits[0, len(p) - 1: len(p) - 1 + len(c)].astype(mx.float32)
            lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            tgt = mx.array(c)
            tot_nll += -float(mx.take_along_axis(lp, tgt[:, None], axis=1).sum())
            tot_tok += len(c)
        # Quantized weights are stored packed, so count logical parameters from bf16 once
        # and use that as the denominator for every format's bits-per-weight.
        if cfg == "bf16":
            res["_n_params_logical"] = n_params(model)
        ref = res.get("_n_params_logical")
        res[cfg] = {"nll_per_token": tot_nll / tot_tok, "n_replies": len(rows), "n_tokens": tot_tok,
                    "weight_bytes": weight_bytes(model),
                    "bits_per_weight": (8 * weight_bytes(model) / ref) if ref else None}
        bpw = res[cfg]["bits_per_weight"]
        print(f"  {cfg:10s} nll/token {res[cfg]['nll_per_token']:.4f}  bpw {bpw:.2f} " if bpw else
              f"  {cfg:10s} nll/token {res[cfg]['nll_per_token']:.4f}  bpw n/a "
              f"({time.time() - t0:.0f}s)", flush=True)
        dest.write_text(json.dumps(res, indent=1))
        del model
    if "bf16" in res:
        base = res["bf16"]["nll_per_token"]
        print("delta vs bf16:", {c: round(v["nll_per_token"] - base, 4)
                                 for c, v in res.items() if isinstance(v, dict)})


if __name__ == "__main__":
    main()
