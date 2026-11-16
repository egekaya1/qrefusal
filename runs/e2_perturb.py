"""E2 (claim C3, PREREG addendum 3): matched-perturbation controls.

Is the direction of a 4-bit build's refusal shift the model's own lean, or set by the
particular rounding? Per weight matrix, E = dequantize(4bit-g64) - W (float32, tied
embedding included). Each build is a dense bf16 model:

  e2-dense   W + E          must reproduce the real 4-bit build (validation gate)
  e2-mirror  W - E          same error, opposite sign
  e2-perm1/2 W + P(E)       E shuffled within each quantization group, per row (seeds 1, 2)
  e2-gauss   W + G          Gaussian, per-row norm equal to E's row norm (seed 1)

Norms, biases and every unquantized parameter are untouched.

Gate (before any build of a model is generated): dense W + E matches the real 4bit-g64
build in teacher-forced NLL within --nll_tol and in first-token argmax on 200 cell
prompts in at least --agree_min. Every build's NLL and per-matrix relative norm
||P|| / ||E|| are recorded; a norm outside 5% is a hard error.

Cells:
  forced  a2 k = 8 prompts with the stored bf16 prefix tokens  -> results/a2/k8/<name>/<build>.jsonl
  free    E1 held-out WildJailbreak prompts, no prefill          -> results/x2/wjbho/<name>/<build>.jsonl
e2-dense runs on the first --n_dense prompts only. Outputs are resumable per build.

Usage:
  python runs/e2_perturb.py --name qwen2.5-3b --cell forced
  python runs/e2_perturb.py --name qwen2.5-0.5b --cell free --test     (validation suite, no generation)
Writes results/e2/<name>/{real.json, gate.json, builds.json}
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))

from qrefusal.data import chat_ids, greedy_reply, reply_ids, split  # noqa: E402
from qrefusal.metrics import is_broken, is_refusal  # noqa: E402
from qrefusal.models import load  # noqa: E402

BUILDS = ("e2-dense", "e2-mirror", "e2-perm1", "e2-perm2", "e2-gauss")
SCALED = ("e2-scale0.5", "e2-scale2")
EXTRA = ("e2-perm3", "e2-gauss2", "e2-gauss3")  # addendum 5: more independent random draws  # addendum 4: W + aE, a dose-response along the real rounding direction


def scale_of(kind):
    """Expected ||P|| / ||E|| for a build kind."""
    return float(kind.split("scale")[1]) if "scale" in kind else 1.0
QUANT = (nn.QuantizedLinear, nn.QuantizedEmbedding)
CHUNK = 8192  # rows per chunk, to bound float32 transients on 16 GB


def _get(module, path):
    for part in path.split("."):
        module = module[int(part)] if part.isdigit() else getattr(module, part)
    return module


def quantized_paths(qmodel):
    return [p for p, m in qmodel.named_modules() if isinstance(m, QUANT)]


def dequant(m):
    return mx.dequantize(m.weight, m.scales, getattr(m, "biases", None), group_size=m.group_size,
                         bits=m.bits, mode=getattr(m, "mode", "affine")).astype(mx.float32)


def perturbation(kind, E, group, seed):
    """Return the perturbation P (float32, same shape as E) for one chunk of rows."""
    if kind == "e2-dense":
        return E
    if kind == "e2-mirror":
        return -E
    if kind.startswith("e2-scale"):
        return scale_of(kind) * E
    rows, cols = E.shape
    key = mx.random.key(seed)
    if kind.startswith("e2-perm"):
        g = E.reshape(rows, cols // group, group)
        order = mx.argsort(mx.random.uniform(shape=g.shape, key=key), axis=-1)
        return mx.take_along_axis(g, order, axis=-1).reshape(rows, cols)
    if kind.startswith("e2-gauss"):
        G = mx.random.normal(shape=E.shape, key=key)
        scale = mx.sqrt((E * E).sum(axis=1, keepdims=True)) / mx.maximum(mx.sqrt((G * G).sum(axis=1, keepdims=True)), 1e-12)
        return G * scale
    raise ValueError(kind)


def build(name, kind):
    """Load bf16 and 4bit-g64, write W + P into the bf16 model in place, drop the 4-bit model.
    Returns (model, tok, norms) with norms = {path: ||P|| / ||E||}."""
    model, tok = load(str(ROOT / "models" / f"{name}-bf16"))
    qmodel, _ = load(str(ROOT / "models" / f"{name}-4bit-g64"))
    seed = {"e2-perm1": 1, "e2-perm2": 2, "e2-perm3": 3, "e2-gauss": 1, "e2-gauss2": 2, "e2-gauss3": 3}.get(kind, 0)
    norms = {}
    for j, path in enumerate(quantized_paths(qmodel)):
        qm, dm = _get(qmodel, path), _get(model, path)
        Q = dequant(qm)
        W = dm.weight
        assert W.shape == Q.shape, (path, W.shape, Q.shape)
        out, e2, p2 = [], 0.0, 0.0
        for r0 in range(0, W.shape[0], CHUNK):
            Wc = W[r0:r0 + CHUNK].astype(mx.float32)
            E = Q[r0:r0 + CHUNK] - Wc
            P = perturbation(kind, E, qm.group_size, seed * 100003 + j * 7919 + r0)
            new = (Wc + P).astype(mx.bfloat16)
            mx.eval(new, (E * E).sum(), (P * P).sum())
            e2 += float((E * E).sum())
            p2 += float((P * P).sum())
            out.append(new)
        dm.weight = mx.concatenate(out, axis=0) if len(out) > 1 else out[0]
        mx.eval(dm.weight)
        norms[path] = math.sqrt(p2 / e2) if e2 > 0 else 1.0
        del Q, out
    del qmodel
    bad = {p: r for p, r in norms.items() if abs(r - scale_of(kind)) > 0.05 * scale_of(kind)}
    assert not bad, f"{kind}: realized norm outside 5% of ||E||: {list(bad.items())[:3]}"
    return model, tok, norms


# ---- cells -------------------------------------------------------------------------------

def cell_items(name, cell):
    """List of {"i", "request", "prefix_ids"} in a fixed order."""
    if cell == "forced":
        stored = [json.loads(l) for l in (ROOT / "results/a2/k8" / name / "bf16.jsonl").read_text().splitlines()]
        return [{"i": r["i"], "request": r["request"], "prefix_ids": r["prefix_ids"]} for r in stored]
    from x2_sneaky import wjb_heldout_items
    return [{"i": i, "request": it["request"], "prefix_ids": []} for i, it in enumerate(wjb_heldout_items(400))]


def cell_dest(name, cell, kind):
    sub = "a2/k8" if cell == "forced" else "x2/wjbho"
    return ROOT / "results" / sub / name / f"{kind}.jsonl"


def generate(model, tok, items, kind, cell, max_tokens):
    rows = []
    for it in items:
        pre = tok.decode(it["prefix_ids"]) if it["prefix_ids"] else ""
        gen, reason = greedy_reply(model, tok, chat_ids(tok, it["request"]) + it["prefix_ids"], max_tokens)
        sep = " " if pre and gen and not pre[-1].isspace() and gen[0].isalnum() else ""
        text = pre + sep + gen
        row = {"kind": "harmful", "i": it["i"], "request": it["request"], "reply": text, "finish_reason": reason,
               "refused": is_refusal(text), "broken": is_broken(text, reply_ids(tok, text)), "config": kind}
        if cell == "forced":
            row.update({"k": 8, "prefix": pre, "prefix_ids": it["prefix_ids"], "temp": 0.0, "run": 0})
        else:
            row["style"] = "wjb_heldout"
        rows.append(row)
    return rows


# ---- gate quantities (teacher-forced, not text identity) -----------------------------------

def nll_texts(name, tok, n=200):
    """(prompt_ids, continuation_ids) pairs: bf16's own harmless replies (as in C1), or, for a
    model without R1 results, a fixed set of short harmless continuations generated by bf16."""
    r1 = ROOT / "results" / "r1" / name / "bf16.jsonl"
    prompts = split("harmless_test", 300)
    if r1.exists():
        rows = [json.loads(l) for l in r1.read_text().splitlines()]
        rows = [r for r in rows if r["kind"] == "harmless" and not r["broken"]][:n]
        return [(chat_ids(tok, prompts[r["i"]]), reply_ids(tok, r["reply"])[:128]) for r in rows]
    return None


def nll(model, pairs):
    tot, cnt = 0.0, 0
    for p, c in pairs:
        if not c:
            continue
        logits = model(mx.array([p + c]))[0, len(p) - 1: len(p) - 1 + len(c)].astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tot += -float(mx.take_along_axis(lp, mx.array(c)[:, None], axis=1).sum())
        cnt += len(c)
    return tot / cnt


def first_tokens(model, tok, items):
    out = []
    for it in items:
        ids = chat_ids(tok, it["request"]) + it["prefix_ids"]
        out.append(int(mx.argmax(model(mx.array([ids]))[0, -1])))
    return out


def real_reference(name, cell, items, pairs, out):
    """NLL and first-token argmax of the real 4bit-g64 build (quantized kernel), cached."""
    f = out / f"real_{cell}.json"
    if f.exists():
        return json.loads(f.read_text())
    model, tok = load(str(ROOT / "models" / f"{name}-4bit-g64"))
    ref = {"first": first_tokens(model, tok, items), "nll": nll(model, pairs)}
    bmodel = None
    del model
    bmodel, _ = load(str(ROOT / "models" / f"{name}-bf16"))
    ref["nll_bf16"] = nll(bmodel, pairs)
    del bmodel
    f.write_text(json.dumps(ref))
    return ref


def make_pairs(name, tok, items):
    pairs = nll_texts(name, tok)
    if pairs is None:  # small test models: score the bf16 model's own greedy replies to 50 harmless prompts
        model, _ = load(str(ROOT / "models" / f"{name}-bf16"))
        prompts = split("harmless_test", 50)
        pairs = []
        for p in prompts:
            ids = chat_ids(tok, p)
            text, _ = greedy_reply(model, tok, ids, 96)
            pairs.append((ids, reply_ids(tok, text)))
        del model
    return pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cell", choices=["forced", "free"], required=True)
    ap.add_argument("--builds", default=",".join(BUILDS))
    ap.add_argument("--n_dense", type=int, default=150)
    ap.add_argument("--n_gate", type=int, default=200)
    ap.add_argument("--nll_tol", type=float, default=0.005)
    ap.add_argument("--agree_min", type=float, default=0.98)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--test", action="store_true", help="validation suite only: gate + every build's norms and NLL")
    args = ap.parse_args()

    out = ROOT / "results" / "e2" / args.name
    out.mkdir(parents=True, exist_ok=True)
    m, tok = load(str(ROOT / "models" / f"{args.name}-bf16"))
    del m  # tokenizer only; never hold a second model through the builds
    items = cell_items(args.name, args.cell)
    gate_items = items[: args.n_gate]
    pairs = make_pairs(args.name, tok, items)
    ref = real_reference(args.name, args.cell, gate_items, pairs, out)
    print(f"{args.name}: real 4bit-g64 NLL {ref['nll']:.4f} (bf16 {ref['nll_bf16']:.4f})", flush=True)

    info = json.loads((out / "builds.json").read_text()) if (out / "builds.json").exists() else {}
    gate_f = out / f"gate_{args.cell}.json"
    for kind in args.builds.split(","):
        dest = cell_dest(args.name, args.cell, kind)
        if dest.exists() and not args.test and kind in info:
            print(f"skip {kind}: {dest.name} exists", flush=True)
            continue
        t0 = time.time()
        model, tok, norms = build(args.name, kind)
        b_nll = nll(model, pairs)
        rec = {"nll": b_nll, "nll_gap": b_nll - ref["nll_bf16"], "real_nll_gap": ref["nll"] - ref["nll_bf16"],
               "norm_ratio_min": min(norms.values()), "norm_ratio_max": max(norms.values()), "n_matrices": len(norms)}
        if kind == "e2-dense":
            ft = first_tokens(model, tok, gate_items)
            agree = sum(a == b for a, b in zip(ft, ref["first"])) / len(ft)
            ok = abs(b_nll - ref["nll"]) <= args.nll_tol and agree >= args.agree_min
            gate = {"pass": ok, "nll_dense": b_nll, "nll_real": ref["nll"], "nll_diff": b_nll - ref["nll"],
                    "first_token_agree": agree, "n": len(ft), "nll_tol": args.nll_tol, "agree_min": args.agree_min}
            gate_f.write_text(json.dumps(gate, indent=1))
            print(f"GATE {'PASS' if ok else 'FAIL'}: NLL dense {b_nll:.4f} vs real {ref['nll']:.4f} "
                  f"(diff {b_nll - ref['nll']:+.4f}), first-token agreement {agree:.1%} on {len(ft)}", flush=True)
            if not ok:
                del model
                sys.exit(2)
        elif not args.test:
            g = json.loads(gate_f.read_text()) if gate_f.exists() else {}
            if not g.get("pass"):
                print(f"stop: gate not passed for {args.name}/{args.cell}; run e2-dense first", flush=True)
                sys.exit(2)
        if not kind.startswith(("e2-scale", "e2-dense", "e2-mirror")) and rec["nll_gap"] > 2 * rec["real_nll_gap"]:
            rec["not_matched_in_damage"] = True
        info[kind] = rec
        (out / "builds.json").write_text(json.dumps(info, indent=1))
        print(f"  {kind:10s} NLL {b_nll:.4f} (gap {rec['nll_gap']:+.4f}; real {rec['real_nll_gap']:+.4f}) "
              f"norm ratio [{rec['norm_ratio_min']:.4f}, {rec['norm_ratio_max']:.4f}] over {len(norms)} matrices "
              f"({time.time() - t0:.0f}s)", flush=True)
        if not args.test:
            run_items = items[: args.n_dense] if kind == "e2-dense" else items
            t1 = time.time()
            rows = generate(model, tok, run_items, kind, args.cell, args.max_tokens)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text("\n".join(json.dumps(r) for r in rows))
            print(f"  {kind}: {len(rows)} generated in {time.time() - t1:.0f}s -> {dest.relative_to(ROOT)}", flush=True)
        del model


if __name__ == "__main__":
    main()
