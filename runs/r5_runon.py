"""R5: the MXFP4 turn-ending failure -- prevalence, and is it the weights or the runtime?

Qwen2.5-3B in MXFP4 sometimes fails to emit its end-of-turn token: it runs past the
end of its reply and writes a fabricated next user turn. This script measures that
on BENIGN prompts (so no harmful text is involved) and separates two explanations:

  prevalence  per format: share of replies that hit the token cap without emitting
              a stop token ("no_stop"), and share containing a role marker ("marker").
  dequant     the MXFP4 model's weights are dequantized back to bf16 floats and the
              same prompts are rerun. The arithmetic is then ordinary bf16 matmul,
              so if the failure persists it is the weight VALUES; if it disappears,
              it is MLX's quantized kernel (a runtime defect, not a research finding).

Usage: python runs/r5_runon.py --name qwen2.5-3b --configs bf16,8bit-g64,4bit-g64,mxfp4 --dequant mxfp4
Writes results/r5/<name>/<config>.jsonl and summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrefusal.data import chat_ids, greedy_reply, split  # noqa: E402
from qrefusal.models import load  # noqa: E402

# Literal role markers: a reply containing one has written chat structure of its own.
MARKER = re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|start_header_id\|>|<\|eot_id\|>")


def dequantize_in_place(model):
    """Replace every QuantizedLinear with an equivalent bf16 nn.Linear.

    Same mathematical weights, ordinary dense arithmetic: this isolates weight
    values from the quantized matmul kernel.
    """
    import mlx.nn as nn

    def convert(module):
        for name, child in list(module.children().items()):
            if isinstance(child, nn.QuantizedLinear):
                w = mx.dequantize(child.weight, child.scales, child.biases,
                                  group_size=child.group_size, bits=child.bits,
                                  mode=getattr(child, "mode", "affine"))
                lin = nn.Linear(w.shape[1], w.shape[0], bias=hasattr(child, "bias"))
                lin.weight = w.astype(mx.bfloat16)
                if hasattr(child, "bias"):
                    lin.bias = child.bias
                setattr(module, name, lin)
            elif isinstance(child, nn.QuantizedEmbedding):
                # Tied embeddings: the output layer is embed_tokens.as_linear, so this
                # must be dequantized too or the logits still use the quantized kernel.
                w = mx.dequantize(child.weight, child.scales, child.biases,
                                  group_size=child.group_size, bits=child.bits,
                                  mode=getattr(child, "mode", "affine"))
                emb = nn.Embedding(w.shape[0], w.shape[1])
                emb.weight = w.astype(mx.bfloat16)
                setattr(module, name, emb)
            elif isinstance(child, (dict, list)) or hasattr(child, "children"):
                if hasattr(child, "children"):
                    convert(child)
                elif isinstance(child, list):
                    for c in child:
                        if hasattr(c, "children"):
                            convert(c)

    convert(model)
    mx.eval(model.parameters())
    left = [n for n, m in model.named_modules()
            if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding))]
    assert not left, f"quantized modules remain after dequantize: {left[:5]}"
    return model


def run(model, tok, prompts, max_tokens):
    rows = []
    for i, text in prompts:
        reply, reason = greedy_reply(model, tok, chat_ids(tok, text), max_tokens)
        rows.append({"i": i, "request": text, "reply": reply, "finish_reason": reason,
                     "no_stop": reason != "stop", "marker": bool(MARKER.search(reply)),
                     "n_chars": len(reply)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--configs", default="bf16,8bit-g64,4bit-g64,4bit-g32,mixed_4_6,mxfp4")
    ap.add_argument("--dequant", default="", help="comma list of configs to also run dequantized to bf16")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--max_tokens", type=int, default=512)
    args = ap.parse_args()

    out = ROOT / "results" / "r5" / args.name
    out.mkdir(parents=True, exist_ok=True)
    prompts = list(enumerate(split("harmless_test", args.n, seed=11)))

    jobs = [(c, False) for c in args.configs.split(",") if c]
    jobs += [(c, True) for c in args.dequant.split(",") if c]
    sfile = out / "summary.json"
    summary = json.loads(sfile.read_text()) if sfile.exists() else {}  # keep configs not rerun
    for cfg, deq in jobs:
        tag = f"{cfg}-dequant" if deq else cfg
        dest = out / f"{tag}.jsonl"
        if dest.exists():
            rows = [json.loads(l) for l in dest.read_text().splitlines()]
        else:
            path = ROOT / "models" / f"{args.name}-{cfg}"
            if not path.exists():
                print(f"skip {tag}: no model dir", flush=True)
                continue
            model, tok = load(str(path))
            if deq:
                dequantize_in_place(model)
            t0 = time.time()
            rows = run(model, tok, prompts, args.max_tokens)
            dest.write_text("\n".join(json.dumps(r) for r in rows))
            print(f"{tag}: {len(rows)} in {time.time() - t0:.0f}s", flush=True)
            del model
        n = len(rows)
        summary[tag] = {"n": n, "no_stop": sum(r["no_stop"] for r in rows) / n,
                        "marker": sum(r["marker"] for r in rows) / n}
        print(f"  {tag:18s} no_stop {summary[tag]['no_stop']:.1%}  role-marker {summary[tag]['marker']:.1%}", flush=True)
        (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
