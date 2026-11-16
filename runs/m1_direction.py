"""M1 (PREREG addendum 4): a representation-level account of each build's shift.

For every build of a model (real quantized builds and the E2 control builds), and on the
same prompts as the behavioural cells, project the residual stream (resid_pre, last
position, every layer) onto that model's refusal direction: the bf16 diff-in-means
direction harmful - harmless (results/r2/<name>/directions.npy, Arditi et al.), unit-
normalised per layer. Higher projection = more refusal-like internal state.

The quantity reported is the per-prompt change against bf16, at the layer chosen for
abliteration (results/e7/<name>/choice.json) and as a per-layer profile. Forward passes
only; no generation, no text is written.

Cells:
  forced  a2 k=8 prompts with the stored bf16 prefix tokens (the state the model continues from)
  free    E1 held-out WildJailbreak prompts, chat template only

Usage: python runs/m1_direction.py --name qwen2.5-3b --cell forced
Writes results/m1/<name>/<cell>/<build>.npy  (float32 [n_prompts, n_layers]) and ids.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

from e2_perturb import BUILDS, EXTRA, SCALED, build, cell_items  # noqa: E402
from qrefusal.data import chat_ids  # noqa: E402
from qrefusal.hooks import HookSpec, forward, n_layers  # noqa: E402
from qrefusal.models import load  # noqa: E402

REAL = ("bf16", "8bit-g64", "4bit-g32", "mixed_4_6", "4bit-g64", "mxfp4", "awq", "gptq")


def projections(model, tok, items, units):
    L = n_layers(model)
    keys = {(l, "resid_pre") for l in range(L)}
    U = mx.array(units)  # [L, d]
    out = np.zeros((len(items), L), dtype=np.float32)
    for j, it in enumerate(items):
        ids = chat_ids(tok, it["request"]) + it["prefix_ids"]
        _, store = forward(model, mx.array([ids]), HookSpec(capture=keys))
        acts = mx.stack([store[(l, "resid_pre")][0, -1] for l in range(L)]).astype(mx.float32)  # [L, d]
        p = (acts * U).sum(axis=1)
        mx.eval(p)
        out[j] = np.array(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cell", choices=["forced", "free"], required=True)
    ap.add_argument("--builds", default=",".join(REAL + BUILDS + SCALED + EXTRA))
    ap.add_argument("--limit", type=int, default=None, help="smoke tests: first N prompts, writes to a _smoke dir")
    args = ap.parse_args()

    d = np.load(ROOT / "results" / "r2" / args.name / "directions.npy").astype(np.float32)
    norms = np.linalg.norm(d, axis=1, keepdims=True)
    # Layer 0's last-position residual is the same template token for every prompt, so its
    # diff-in-means direction is zero; such layers get a zero vector (projection 0), not NaN.
    units = np.where(norms > 1e-6, d / np.maximum(norms, 1e-6), 0.0).astype(np.float32)
    items = cell_items(args.name, args.cell)[: args.limit]
    out = ROOT / "results" / "m1" / args.name / (args.cell + ("_smoke" if args.limit else ""))
    out.mkdir(parents=True, exist_ok=True)
    (out / "ids.json").write_text(json.dumps([it["i"] for it in items]))
    for b in args.builds.split(","):
        dest = out / f"{b}.npy"
        if dest.exists():
            continue
        t0 = time.time()
        if b.startswith("e2-"):
            model, tok, _ = build(args.name, b)
        else:
            path = ROOT / "models" / f"{args.name}-{b}"
            if not path.exists():
                print(f"skip {b}: no model dir", flush=True)
                continue
            model, tok = load(str(path))
        P = projections(model, tok, items, units)
        np.save(dest, P)
        print(f"{args.name} {args.cell} {b}: {P.shape} in {time.time() - t0:.0f}s", flush=True)
        del model


if __name__ == "__main__":
    main()
