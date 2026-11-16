"""Per-matrix rounding error of each 4-bit build (PREREG addendum 4; reporting only).

For every matrix the default build quantizes: relative error ||E||/||W|| for 4bit-g64 (RTN),
MXFP4, AWQ and GPTQ, and the cosine between each method's error and the RTN error.
Answers "where does the error go" (per layer and module type) and "how different are the
errors of methods with the same bits per weight". Weights only; no prompts.

Usage: python runs/e2_errors.py --name qwen2.5-3b
Writes results/e2/<name>/errors.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

from e2_perturb import CHUNK, _get, dequant, quantized_paths  # noqa: E402
from qrefusal.models import load  # noqa: E402

OTHERS = ("mxfp4", "awq", "gptq")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args()
    out = ROOT / "results" / "e2" / args.name / "errors.json"
    W_model, _ = load(str(ROOT / "models" / f"{args.name}-bf16"))
    rtn, _ = load(str(ROOT / "models" / f"{args.name}-4bit-g64"))
    paths = quantized_paths(rtn)
    res = {p: {} for p in paths}
    for other in (None,) + OTHERS:
        if other is not None:
            d = ROOT / "models" / f"{args.name}-{other}"
            if not d.exists():
                print(f"skip {other}: no model dir", flush=True)
                continue
            om, _ = load(str(d))
        for p in paths:
            W = _get(W_model, p).weight
            Q = dequant(_get(rtn, p))
            if other is not None:
                m = _get(om, p)
                if not hasattr(m, "scales"):
                    continue
                Q2 = dequant(m)
            w2 = e2 = o2 = dot = 0.0
            for r0 in range(0, W.shape[0], CHUNK):
                Wc = W[r0:r0 + CHUNK].astype(mx.float32)
                E = Q[r0:r0 + CHUNK] - Wc
                if other is None:
                    vals = [(Wc * Wc).sum(), (E * E).sum()]
                else:
                    E2 = Q2[r0:r0 + CHUNK] - Wc
                    vals = [(E2 * E2).sum(), (E * E2).sum(), (E * E).sum()]
                mx.eval(*vals)
                if other is None:
                    w2 += float(vals[0]); e2 += float(vals[1])
                else:
                    o2 += float(vals[0]); dot += float(vals[1]); e2 += float(vals[2])
            if other is None:
                res[p]["w_norm"] = math.sqrt(w2)
                res[p]["rel_rtn"] = math.sqrt(e2 / w2)
            else:
                res[p][f"rel_{other}"] = math.sqrt(o2) / res[p]["w_norm"]
                res[p][f"cos_rtn_{other}"] = dot / math.sqrt(e2 * o2) if e2 > 0 and o2 > 0 else None
        if other is not None:
            del om
        print(f"{args.name}: {other or 'rtn'} done", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"-> {out.relative_to(ROOT)} ({len(paths)} matrices)", flush=True)


if __name__ == "__main__":
    main()
