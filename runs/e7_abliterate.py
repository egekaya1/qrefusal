"""E7: compliance continuations from an abliterated bf16 model.

Removes the refusal direction from the bf16 model's weights (Arditi et al.:
project it out of every matrix that writes to the residual stream, plus the
embeddings), picks the layer whose direction removes refusal best on train
prompts, then writes greedy compliance replies for the harmful test prompts.
These replies give the contrastive metric:
    CRCL = logp(bf16 refusal reply) - logp(compliance reply)

Usage: python runs/e7_abliterate.py --name qwen2.5-3b
Writes results/e7/<name>/compliance.jsonl and choice.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrefusal.data import chat_text, split  # noqa: E402
from qrefusal.hooks import _inner, n_layers  # noqa: E402
from qrefusal.metrics import is_broken, is_refusal  # noqa: E402
from qrefusal.models import load  # noqa: E402

GREEDY = make_sampler(temp=0.0)


def abliterate(model, r: np.ndarray):
    """Remove unit direction r from residual writes IN PLACE; returns restore().

    Only the edited matrices are kept as a backup (embeddings, o_proj, down_proj),
    instead of deep-copying the model, so a 4B bf16 model fits in 16 GB.
    Note: these models tie embeddings, so the embed edit also edits the unembedding.
    """
    inner = _inner(model)
    u = mx.array(r / np.linalg.norm(r)).astype(mx.float32)
    backup = [(inner.embed_tokens, inner.embed_tokens.weight)]
    for layer in inner.layers:
        backup += [(layer.self_attn.o_proj, layer.self_attn.o_proj.weight), (layer.mlp.down_proj, layer.mlp.down_proj.weight)]

    def out_proj(w):  # w: [d_model, d_in], output space is the residual
        wf = w.astype(mx.float32)
        return (wf - mx.outer(u, u @ wf)).astype(w.dtype)

    e = inner.embed_tokens.weight.astype(mx.float32)  # [vocab, d_model]
    inner.embed_tokens.weight = (e - mx.outer(e @ u, u)).astype(inner.embed_tokens.weight.dtype)
    for layer in inner.layers:
        layer.self_attn.o_proj.weight = out_proj(layer.self_attn.o_proj.weight)
        layer.mlp.down_proj.weight = out_proj(layer.mlp.down_proj.weight)
        mx.eval(layer.self_attn.o_proj.weight, layer.mlp.down_proj.weight)
    mx.eval(inner.embed_tokens.weight)

    def restore():
        for mod, w in backup:
            mod.weight = w
    return restore


def reply(model, tok, ins, n):
    prompt = chat_text(tok, ins)
    return generate(model, tok, prompt=prompt, max_tokens=n, sampler=GREEDY, verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--n_select", type=int, default=32)
    ap.add_argument("--gen", type=int, default=64)
    args = ap.parse_args()

    out = ROOT / "results" / "e7" / args.name
    out.mkdir(parents=True, exist_ok=True)
    d = np.load(ROOT / "results" / "r2" / args.name / "directions.npy")  # [L, d]
    f, tok = load(str(ROOT / "models" / f"{args.name}-bf16"))
    L = n_layers(f)

    # Pick the layer whose direction best removes refusal, keeping text coherent.
    sel = split("harmful_train", args.n_select, seed=7)
    harmless = split("harmless_train", 16, seed=7)
    best = None
    for l in range(L // 4, 3 * L // 4):
        restore = abliterate(f, d[l])
        refusals = sum(is_refusal(reply(f, tok, s, 24)) for s in sel) / len(sel)
        broken = sum(is_broken(t := reply(f, tok, s, 32), tok.encode(t)) for s in harmless) / len(harmless)
        restore()
        print(f"layer {l}: refusal {refusals:.2f} broken {broken:.2f}", flush=True)
        if broken <= 0.05 and (best is None or refusals < best[1]):
            best = (l, refusals)
    (out / "choice.json").write_text(json.dumps({"layer": best[0], "train_refusal": best[1]}))
    print("chosen layer", best, flush=True)

    abliterate(f, d[best[0]])
    m = f
    prompts = split("harmful_test", 300)
    rows = []
    for i, ins in enumerate(prompts):
        text = reply(m, tok, ins, args.gen)
        rows.append({"i": i, "reply": text, "refused": is_refusal(text), "broken": is_broken(text, tok.encode(text))})
    (out / "compliance.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(f"compliance replies: {len(rows)}, still refused {sum(r['refused'] for r in rows) / len(rows):.1%}", flush=True)


if __name__ == "__main__":
    main()
