"""R2: refusal-direction geometry, bf16 vs quantized.

Per layer: cosine of each config's own diff-in-means direction with bf16's,
principal angles of the top-3 subspace, and how strongly harmful test prompts
project onto the bf16 unit direction (H-write check).

Usage: python runs/r2.py --name qwen2.5-3b --configs bf16,4bit-g64,3bit-g64
Writes results/r2/<name>/geometry.json and bf16 directions (directions.npy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrefusal.data import split  # noqa: E402
from qrefusal.directions import acts_at, cosine_per_layer, diff_in_means, principal_angles_deg, subspace, unit  # noqa: E402
from qrefusal.models import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--configs", required=True, help="comma list; must start with bf16")
    ap.add_argument("--n_train", type=int, default=128)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--where", default="last", choices=["last", "post"])
    args = ap.parse_args()

    out = ROOT / "results" / "r2" / args.name
    out.mkdir(parents=True, exist_ok=True)
    tr_h, tr_s = split("harmful_train", args.n_train), split("harmless_train", args.n_train)
    te_h, te_s = split("harmful_test", args.n_test), split("harmless_test", args.n_test)

    ref = None
    results = {}
    for cfg in args.configs.split(","):
        model, tok = load(str(ROOT / "models" / f"{args.name}-{cfg}"))
        H, S = acts_at(model, tok, tr_h, where=args.where), acts_at(model, tok, tr_s, where=args.where)
        TH, TS = acts_at(model, tok, te_h, where=args.where), acts_at(model, tok, te_s, where=args.where)
        d = diff_in_means(H, S)
        if cfg == "bf16":
            ref = {"d": d, "H": H, "S": S}
            np.save(out / "directions.npy", d)
        u = unit(ref["d"])  # bf16 unit directions [L, d]
        L = d.shape[0]
        results[cfg] = {
            "cos_to_bf16": cosine_per_layer(d, ref["d"]).tolist(),
            "angles_top3_deg": [principal_angles_deg(subspace(H, S, l), subspace(ref["H"], ref["S"], l)).tolist() for l in range(L)],
            "proj_harmful_mean": [(TH[:, l] @ u[l]).mean().item() for l in range(L)],
            "proj_harmless_mean": [(TS[:, l] @ u[l]).mean().item() for l in range(L)],
            "norm_d": np.linalg.norm(d, axis=-1).tolist(),
        }
        # Layer 0 is degenerate at the last token: every prompt ends in the same
        # template token, so harmful and harmless embeddings are identical.
        print(cfg, "min cos (layers 1+)", round(min(results[cfg]["cos_to_bf16"][1:]), 4), flush=True)
        del model
    (out / f"geometry_{args.where}.json").write_text(json.dumps(results))


if __name__ == "__main__":
    main()
