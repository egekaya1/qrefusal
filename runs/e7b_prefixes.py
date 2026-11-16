"""E7b: short compliance prefixes for every harmful_test prompt (all 572).

Reuses the abliteration layer E7 chose (results/e7/<name>/choice.json) and writes
the first --gen tokens of the abliterated bf16 model's greedy reply. These are the
prefill sources for the depth curve (A2); only the first k <= 8 tokens are used.

Usage: python runs/e7b_prefixes.py --name qwen2.5-3b
Writes results/e7/<name>/prefixes_full.jsonl  ({"i": full-split index, "request", "reply"})
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

from e7_abliterate import abliterate, reply  # noqa: E402
from qrefusal.data import split  # noqa: E402
from qrefusal.metrics import is_refusal  # noqa: E402
from qrefusal.models import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--gen", type=int, default=16)
    args = ap.parse_args()
    out = ROOT / "results" / "e7" / args.name
    layer = json.loads((out / "choice.json").read_text())["layer"]
    d = np.load(ROOT / "results" / "r2" / args.name / "directions.npy")
    f, tok = load(str(ROOT / "models" / f"{args.name}-bf16"))
    abliterate(f, d[layer])
    m = f
    seen, uniq = set(), []
    for i, ins in enumerate(split("harmful_test")):  # 572 rows, 562 unique prompts: keep first occurrence
        if ins not in seen:
            seen.add(ins)
            uniq.append((i, ins))
    rows = [{"i": i, "request": ins, "reply": reply(m, tok, ins, args.gen)} for i, ins in uniq]
    (out / "prefixes_full.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print(f"{args.name}: {len(rows)} prefixes from layer {layer}, refusal-like {sum(is_refusal(r['reply']) for r in rows) / len(rows):.1%}", flush=True)


if __name__ == "__main__":
    main()
