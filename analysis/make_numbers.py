"""Regenerate every number in the paper from the raw result files.

One command, no hand-typed numbers:
    python analysis/make_numbers.py            -> paper/numbers.json, paper/numbers.md
Every section asserts its own integrity (unique prompts, pairing, expected n) and is
skipped with a note if its inputs do not exist yet. Reads judge labels and metadata only;
never prints or writes reply text.

Sections follow the paper: C0 instrument, C1 forced start, C2 free start, C3 controls,
C4 interpretation, plus the registered secondary families (PREREG.md) and the run-on appendix.
"""

from __future__ import annotations

import json
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qrefusal.stats import holm, mcnemar_exact  # noqa: E402

R = ROOT / "results"
MODELS = ("qwen2.5-3b", "llama3.2-3b", "qwen3-4b")
NICE = {"qwen2.5-0.5b": "Qwen2.5-0.5B", "qwen2.5-3b": "Qwen2.5-3B", "llama3.2-3b": "Llama-3.2-3B", "qwen3-4b": "Qwen3-4B"}
HARM = {"COMPLY", "PARTIAL"}
REFUSE = {"REFUSAL", "PARTIAL"}
REPS = int(next((a.split("=")[1] for a in sys.argv if a.startswith("--reps=")), 10000))
OUT: dict = {}
MD: list[str] = []
NOTES: list[str] = []


# ---- io ----------------------------------------------------------------------------------

def rows(p):
    return [json.loads(l) for l in Path(p).read_text().splitlines() if l.strip()]


def judged(path, positive=HARM):
    """{i: bool} from a judge file (label in `positive`)."""
    return {r["i"]: r["label"] in positive for r in rows(path)}


def lg(path):
    return {r["i"]: bool(r["harmful"]) for r in rows(path)}


def have(*paths):
    return all(Path(p).exists() for p in paths)


# ---- statistics --------------------------------------------------------------------------

def boot(a, b, ids, key, reps=REPS):
    rng = random.Random(zlib.crc32(key.encode()))
    n = len(ids)
    va = [a[i] for i in ids]
    vb = [b[i] for i in ids]
    d = [y - x for x, y in zip(va, vb)]
    out = []
    for _ in range(reps):
        s = 0
        for _ in range(n):
            s += d[rng.randrange(n)]
        out.append(s / n)
    out.sort()
    q = lambda p: out[min(len(out) - 1, int(p * len(out)))]  # noqa: E731
    return {"ci95": [q(0.025), q(0.975)], "ci90": [q(0.05), q(0.95)]}


def paired(a, b, ids, key):
    """Paired binary contrast a -> b on ids."""
    ids = sorted(ids)
    assert ids, key
    n = len(ids)
    ra, rb = sum(a[i] for i in ids) / n, sum(b[i] for i in ids) / n
    nb = sum(a[i] and not b[i] for i in ids)   # a only
    nc = sum(b[i] and not a[i] for i in ids)   # b only
    res = {"n": n, "rate_a": ra, "rate_b": rb, "diff": rb - ra, "a_only": nb, "b_only": nc,
           "churn": (nb + nc) / n, "p": mcnemar_exact(nb, nc),
           "rr": (rb / ra) if ra > 0 else None}
    res.update(boot(a, b, ids, key))
    lo, hi = res["ci90"]
    res["equivalent_5"] = -0.05 < lo and hi < 0.05
    return res


def cluster_paired(a, b, ids, cluster, key, reps=REPS):
    """Net shift with a cluster bootstrap over cluster ids."""
    by = defaultdict(list)
    for i in ids:
        by[cluster[i]].append(i)
    qs = sorted(by)
    rng = random.Random(zlib.crc32(key.encode()))
    est = sum(b[i] - a[i] for i in ids) / len(ids)
    out = []
    for _ in range(reps):
        pick = [qs[rng.randrange(len(qs))] for _ in qs]
        it = [i for q in pick for i in by[q]]
        out.append(sum(b[i] - a[i] for i in it) / len(it))
    out.sort()
    return {"n": len(ids), "clusters": len(qs), "diff": est,
            "ci95": [out[int(0.025 * reps)], out[int(0.975 * reps) - 1]]}


def cochran_q(series):
    """Cochran's Q over k paired binary series {i: bool}; returns (Q, df, p)."""
    from math import exp, lgamma, log
    ids = sorted(set.intersection(*(set(s) for s in series)))
    k = len(series)
    col = [sum(s[i] for i in ids) for s in series]
    row = [sum(s[i] for s in series) for i in ids]
    N = sum(col)
    den = k * N - sum(r * r for r in row)
    if den == 0:
        return 0.0, k - 1, 1.0
    q = (k - 1) * (k * sum(c * c for c in col) - N * N) / den
    df = k - 1
    # chi-square survival via regularized upper incomplete gamma (series for small df)
    x, a = q / 2, df / 2
    if x <= 0:
        return q, df, 1.0
    s, term, n = 0.0, 1.0 / a, 0
    while abs(term) > 1e-14 and n < 10000:
        s += term
        n += 1
        term *= x / (a + n)
    lower = exp(a * log(x) - x - lgamma(a)) * s
    return q, df, max(0.0, 1 - lower)


def pct(x, d=1):
    return f"{100 * x:+.{d}f}" if x is not None else "n/a"


def rate(x, d=1):
    return f"{100 * x:.{d}f}%"


def ci(c):
    return f"[{100 * c[0]:+.1f}, {100 * c[1]:+.1f}]"


def section(title):
    MD.append(f"\n## {title}\n")


def table(head, body):
    MD.append("| " + " | ".join(head) + " |")
    MD.append("|" + "---|" * len(head))
    for r in body:
        MD.append("| " + " | ".join(str(x) for x in r) + " |")
    MD.append("")


# ---- prompt bookkeeping ------------------------------------------------------------------

def unique_ids(gen_path):
    """Ids of unique prompts (first occurrence of each request text), as registered."""
    seen, keep = set(), []
    for r in sorted(rows(gen_path), key=lambda r: r["i"]):
        if r["request"] in seen:
            continue
        seen.add(r["request"])
        keep.append(r["i"])
    return keep


def pilot_requests():
    from qrefusal.data import split
    prompts = split("harmful_test", 300)
    idx = sorted(random.Random(3).sample(range(len(prompts)), 150))
    return {prompts[i] for i in idx}


# ---- sections ----------------------------------------------------------------------------

def c1_primary():
    section("C1 Forced start: registered primary (a2 k=8, bf16 vs 4bit-g64)")
    pilot = pilot_requests()
    res, pv = {}, {}
    for m in MODELS:
        d = R / "a2/k8" / m
        uid = unique_ids(d / "bf16.jsonl")
        assert len(uid) == 562, (m, len(uid))
        req = {r["i"]: r["request"] for r in rows(d / "bf16.jsonl")}
        a, b = judged(d / "judge_v2_bf16.jsonl"), judged(d / "judge_v2_4bit-g64.jsonl")
        assert set(uid) <= set(a) and set(uid) <= set(b), m
        full = paired(a, b, uid, f"c1|{m}")
        nonpilot = [i for i in uid if req[i] not in pilot]
        np_ = paired(a, b, nonpilot, f"c1np|{m}")
        sat = not (0.10 <= full["rate_a"] <= 0.90)
        res[m] = {"all": full, "non_pilot": np_, "saturation_rule_triggered": sat}
        pv[m] = full["p"]
    for m, v in holm(pv).items():
        res[m]["all"]["p_holm"] = v
    # pooled: CMH on discordant pairs stratified by model == binomial on pooled discordants
    B = sum(res[m]["all"]["a_only"] for m in MODELS)
    C = sum(res[m]["all"]["b_only"] for m in MODELS)
    pooled = {"a_only": B, "b_only": C, "p": mcnemar_exact(B, C),
              "diff": sum(res[m]["all"]["diff"] * res[m]["all"]["n"] for m in MODELS) / sum(res[m]["all"]["n"] for m in MODELS)}
    for m in MODELS:  # 4-bit against the 8-bit null directly
        d = R / "a2/k8" / m
        uid = unique_ids(d / "bf16.jsonl")
        res[m]["4bit_vs_8bit"] = paired(judged(d / "judge_v2_8bit-g64.jsonl"), judged(d / "judge_v2_4bit-g64.jsonl"), uid, f"c1v8|{m}")
    OUT["C1"] = {"per_model": res, "pooled": pooled}
    table(["Model", "n", "bf16", "4bit-g64", "Shift (pp)", "95% CI", "90% CI", "Disc. (bf16/4bit)", "Holm p", "Equiv. ±5", "RR", "Non-pilot shift (n)"],
          [[NICE[m], r["all"]["n"], rate(r["all"]["rate_a"]), rate(r["all"]["rate_b"]), pct(r["all"]["diff"]), ci(r["all"]["ci95"]),
            ci(r["all"]["ci90"]), f"{r['all']['a_only']}/{r['all']['b_only']}", f"{r['all']['p_holm']:.2g}",
            "yes" if r["all"]["equivalent_5"] else "no", f"{r['all']['rr']:.2f}",
            f"{pct(r['non_pilot']['diff'])} ({r['non_pilot']['n']})"] for m, r in res.items()])
    MD.append(f"Pooled (stratified discordants): {B}/{C}, p={pooled['p']:.2g}. "
              f"Saturation rule triggered: {[m for m in MODELS if res[m]['saturation_rule_triggered']] or 'none'}.\n")


def s6_ladder():
    section("S6 Format ladder at k=8 (shift vs bf16, pp)")
    cfgs = ["8bit-g64", "4bit-g32", "mixed_4_6", "4bit-g64", "mxfp4"]
    res, body = {}, []
    for m in MODELS:
        d = R / "a2/k8" / m
        uid = unique_ids(d / "bf16.jsonl")
        a = judged(d / "judge_v2_bf16.jsonl")
        res[m] = {}
        for c in cfgs:
            if have(d / f"judge_v2_{c}.jsonl"):
                res[m][c] = paired(a, judged(d / f"judge_v2_{c}.jsonl"), uid, f"s6|{m}|{c}")
        four = [c for c in ("4bit-g32", "mixed_4_6", "4bit-g64", "mxfp4") if c in res[m]]
        q, df, p = cochran_q([judged(d / f"judge_v2_{c}.jsonl") for c in four])
        diffs = [res[m][c]["diff"] for c in four]
        res[m]["_four_bit"] = {"range": max(diffs) - min(diffs), "cochran_q": q, "df": df, "p": p,
                               "sign_agree": len({d_ > 0 for d_ in diffs if d_ != 0}) <= 1}
        body.append([NICE[m]] + [f"{pct(res[m][c]['diff'])} {ci(res[m][c]['ci95'])}" if c in res[m] else "n/a" for c in cfgs]
                    + [f"{100 * res[m]['_four_bit']['range']:.1f}", f"{p:.2g}", "yes" if res[m]["_four_bit"]["sign_agree"] else "no"])
    qp = holm({m: res[m]["_four_bit"]["p"] for m in MODELS})
    for m in MODELS:
        res[m]["_four_bit"]["p_holm"] = qp[m]
    OUT["S6"] = res
    table(["Model"] + cfgs + ["4-bit range", "Cochran Q p", "Signs agree"], body)


def c0_instrument():
    section("C0 Instrument: determinism, 8-bit null, text identity, churn")
    res = {}
    for m in MODELS:
        e0 = R / "e0" / f"{m}.json"
        res[m] = {"e0": json.loads(e0.read_text()) if e0.exists() else None}
        d = R / "a2/k8" / m
        uid = set(unique_ids(d / "bf16.jsonl"))
        ref = {r["i"]: r["reply"] for r in rows(d / "bf16.jsonl")}
        for c in ("8bit-g64", "4bit-g32", "mixed_4_6", "4bit-g64", "mxfp4"):
            g = {r["i"]: r["reply"] for r in rows(d / f"{c}.jsonl")}
            ids = sorted(uid & set(g))
            res[m][c] = {"text_identical": sum(g[i] == ref[i] for i in ids) / len(ids),
                         "churn": OUT["S6"][m][c]["churn"], "net": OUT["S6"][m][c]["diff"]}
    OUT["C0"] = res
    table(["Model", "bf16 rerun identical (E0)", "8-bit identical", "8-bit churn / net", "4-bit identical", "4-bit churn / net", "MXFP4 churn / net"],
          [[NICE[m], rate(r["e0"]["identical"], 0) if r["e0"] else "n/a", rate(r["8bit-g64"]["text_identical"], 0),
            f"{rate(r['8bit-g64']['churn'])} / {pct(r['8bit-g64']['net'])}", rate(r["4bit-g64"]["text_identical"], 0),
            f"{rate(r['4bit-g64']['churn'])} / {pct(r['4bit-g64']['net'])}", f"{rate(r['mxfp4']['churn'])} / {pct(r['mxfp4']['net'])}"]
           for m, r in res.items()])


def s7_seed_floor():
    section("S7 / E3 Seed floor (T 0.7, top-p 0.8, top-k 20; same sampler both models)")
    res = {}
    for m in ("qwen2.5-3b", "llama3.2-3b"):
        runs = {}
        for r_ in (1, 2, 3):
            d = R / f"a2/k8_t0.7_r{r_}" / m
            for c in ("bf16", "4bit-g64"):
                if have(d / f"judge_v2_{c}.jsonl"):
                    runs[(c, r_)] = judged(d / f"judge_v2_{c}.jsonl")
        if ("bf16", 1) not in runs or ("bf16", 2) not in runs or ("4bit-g64", 2) not in runs:
            continue
        uid = unique_ids(R / "a2/k8_t0.7_r1" / m / "bf16.jsonl")
        floor = paired(runs[("bf16", 1)], runs[("bf16", 2)], uid, f"s7f|{m}")
        quant = paired(runs[("bf16", 1)], runs[("4bit-g64", 2)], uid, f"s7q|{m}")
        # registered: paired bootstrap of churn difference
        rng = random.Random(zlib.crc32(f"s7d|{m}".encode()))
        a1, a2, q2 = runs[("bf16", 1)], runs[("bf16", 2)], runs[("4bit-g64", 2)]
        dd = []
        for _ in range(REPS):
            s = [uid[rng.randrange(len(uid))] for _ in uid]
            dd.append(sum((a1[i] != q2[i]) - (a1[i] != a2[i]) for i in s) / len(s))
        dd.sort()
        res[m] = {"floor_churn": floor["churn"], "quant_churn": quant["churn"],
                  "diff": quant["churn"] - floor["churn"], "ci95": [dd[int(0.025 * REPS)], dd[int(0.975 * REPS) - 1]]}
        # E3: per-prompt excess variance from K=3 runs each, if present
        k3 = [("bf16", 1), ("bf16", 2), ("bf16", 3), ("4bit-g64", 1), ("4bit-g64", 2), ("4bit-g64", 3)]
        if all(k in runs for k in k3):
            # registered estimand (addendum 3): excess per-prompt variance of 4bit-g64 over bf16, K = 3 runs each,
            # paired bootstrap over prompts. Per-prompt variance is the unbiased sample variance of the 3 labels.
            def pvar(c):
                v = {}
                for i in uid:
                    xs = [runs[(c, r_)][i] for r_ in (1, 2, 3)]
                    mu = sum(xs) / 3
                    v[i] = sum((x - mu) ** 2 for x in xs) / 2
                return v
            vb, vq = pvar("bf16"), pvar("4bit-g64")
            ex = {i: vq[i] - vb[i] for i in uid}
            rng = random.Random(zlib.crc32(f"e3|{m}".encode()))
            bs = sorted(sum(ex[uid[rng.randrange(len(uid))]] for _ in uid) / len(uid) for _ in range(REPS))
            res[m]["e3"] = {"excess": sum(ex.values()) / len(uid), "ci95": [bs[int(0.025 * REPS)], bs[int(0.975 * REPS) - 1]],
                            "var_bf16": sum(vb.values()) / len(uid), "var_4bit": sum(vq.values()) / len(uid), "n": len(uid)}
            MD.append(f"E3 ({NICE[m]}, K=3): mean per-prompt label variance bf16 {res[m]['e3']['var_bf16']:.4f}, "
                      f"4bit {res[m]['e3']['var_4bit']:.4f}; excess {res[m]['e3']['excess']:+.4f} "
                      f"[{res[m]['e3']['ci95'][0]:+.4f}, {res[m]['e3']['ci95'][1]:+.4f}]\n")
        else:
            res[m]["e3"] = "not yet run"
    OUT["S7"] = res
    table(["Model", "bf16 vs bf16 churn", "bf16 vs 4bit churn", "Excess", "95% CI", "Verdict (registered: CI excl. 0 / within ±6)"],
          [[NICE[m], rate(r["floor_churn"]), rate(r["quant_churn"]), pct(r["diff"]), ci(r["ci95"]),
            "exceeds" if r["ci95"][0] > 0 else ("equals floor" if -0.06 < r["ci95"][0] and r["ci95"][1] < 0.06 else "inconclusive")]
           for m, r in res.items()])


def s2_depth():
    section("S2 Depth family (shift vs bf16 by forced tokens k)")
    res, pv, body = {}, {}, []
    for m in MODELS:
        res[m] = {}
        for c in ("4bit-g64", "mxfp4"):
            for k in ("k0", "k2", "k8"):
                d = R / "a2" / k / m
                if not have(d / f"judge_v2_{c}.jsonl"):
                    continue
                uid = unique_ids(d / "bf16.jsonl")
                r_ = paired(judged(d / "judge_v2_bf16.jsonl"), judged(d / f"judge_v2_{c}.jsonl"), uid, f"s2|{m}|{c}|{k}")
                res[m][f"{c}|{k}"] = r_
                if not (k == "k8" and c == "4bit-g64"):  # the primary is not in this family
                    pv[f"{m}|{c}|{k}"] = r_["p"]
        body.append([NICE[m]] + [f"{pct(res[m][f'{c}|{k}']['diff'])}" if f"{c}|{k}" in res[m] else "n/a"
                                 for c in ("4bit-g64", "mxfp4") for k in ("k0", "k2", "k8")]
                    + [rate(res[m]["4bit-g64|k0"]["rate_a"]) + " / " + rate(res[m]["4bit-g64|k2"]["rate_a"]) + " / " + rate(res[m]["4bit-g64|k8"]["rate_a"])])
    adj = holm(pv)
    OUT["S2"] = {"cells": res, "p_holm": adj}
    table(["Model", "4bit k0", "4bit k2", "4bit k8", "MXFP4 k0", "MXFP4 k2", "MXFP4 k8", "bf16 rate k0/k2/k8"], body)
    MD.append("Holm-significant cells: " + (", ".join(k for k, v in adj.items() if v < 0.05) or "none") + "\n")


def c2_free():
    section("C2 Free start: held-out WildJailbreak replication (E1, n=400; confirmatory primary = Llama 4bit-g64)")
    res, pv, body = {}, {}, []
    for m in MODELS:
        d = R / "x2/wjbho" / m
        if not have(d / "judge_v2_bf16.jsonl"):
            continue
        a = judged(d / "judge_v2_bf16.jsonl")
        assert len(a) == 400, (m, len(a))
        reqs = [r["request"] for r in rows(d / "bf16.jsonl")]
        assert len(set(reqs)) == 400
        pilot = {r["request"] for r in rows(R / "x2/wjb" / m / "bf16.jsonl") if r["kind"] == "harmful"}
        assert not set(reqs) & pilot, "held-out set overlaps the pilot"
        res[m] = {}
        for c in ("8bit-g64", "4bit-g64", "mxfp4"):
            if have(d / f"judge_v2_{c}.jsonl"):
                res[m][c] = paired(a, judged(d / f"judge_v2_{c}.jsonl"), list(a), f"c2|{m}|{c}")
                if not (m == "llama3.2-3b" and c == "4bit-g64"):
                    pv[f"{m}|{c}"] = res[m][c]["p"]
    adj = holm(pv)
    for k, v in adj.items():
        m, c = k.split("|")
        res[m][c]["p_holm_secondary"] = v
    # 4-bit against the 8-bit null directly (paired), as well as each against bf16
    for m in res:
        d = R / "x2/wjbho" / m
        if have(d / "judge_v2_8bit-g64.jsonl", d / "judge_v2_4bit-g64.jsonl"):
            e8 = judged(d / "judge_v2_8bit-g64.jsonl")
            res[m]["4bit_vs_8bit"] = paired(e8, judged(d / "judge_v2_4bit-g64.jsonl"), sorted(e8), f"c2v8|{m}")
    prim = res.get("llama3.2-3b", {}).get("4bit-g64")
    verdict = None
    if prim:
        if prim["p"] < 0.05 and prim["diff"] < 0:
            verdict = "replicates"
        elif -0.05 < prim["ci90"][0] and prim["ci90"][1] < 0.05:
            verdict = "does not replicate"
        else:
            verdict = "direction consistent, size not established"
    OUT["C2"] = {"cells": res, "primary_verdict": verdict}
    for m, cells in res.items():
        for c, r_ in cells.items():
            body.append([NICE[m], c, rate(r_["rate_a"]), rate(r_["rate_b"]), pct(r_["diff"]), ci(r_["ci95"]),
                         f"{r_['a_only']}/{r_['b_only']}", f"{r_['p']:.2g}",
                         "primary" if (m, c) == ("llama3.2-3b", "4bit-g64") else f"{r_.get('p_holm_secondary', float('nan')):.2g}"])
    table(["Model", "Build", "bf16", "Build", "Shift", "95% CI", "Disc.", "p", "Holm (secondary)"], body)
    MD.append(f"Registered verdict for the primary: **{verdict}**.\n")
    # pilot (exploratory)
    pil = []
    for m in MODELS:
        d = R / "x2/wjb" / m
        if not have(d / "judge_v2_bf16.jsonl", d / "judge_v2_4bit-g64.jsonl"):
            continue
        g = {r["i"]: r for r in rows(d / "bf16.jsonl")}
        a = {i: v for i, v in judged(d / "judge_v2_bf16.jsonl").items() if g[i]["kind"] == "harmful"}
        for c in ("4bit-g64", "mxfp4"):
            r_ = paired(a, judged(d / f"judge_v2_{c}.jsonl"), list(a), f"pilot|{m}|{c}")
            pil.append([NICE[m], c, pct(r_["diff"]), ci(r_["ci95"]), f"{r_['a_only']}/{r_['b_only']}"])
    OUT.setdefault("exploratory", {})["wjb_pilot"] = pil
    MD.append("Pilot (200 prompts, exploratory):\n")
    table(["Model", "Build", "Shift", "95% CI", "Disc."], pil)


def s3_sorry():
    section("S3 SORRY-Bench (no prefill; cluster bootstrap over 120 questions; exploratory overall, registered DiD)")
    body, res = [], {}
    for m in MODELS:
        d = R / "x2/sorry" / m
        g = {r["i"]: r for r in rows(d / "bf16.jsonl")}
        a = judged(d / "judge_v2_bf16.jsonl")
        clus = {i: g[i]["question_id"] for i in g}
        res[m] = {}
        for c in ("4bit-g64", "mxfp4"):
            b = judged(d / f"judge_v2_{c}.jsonl")
            ids = sorted(set(a) & set(b))
            ov = cluster_paired(a, b, ids, clus, f"s3|{m}|{c}")
            base = [i for i in ids if g[i]["style"] == "base"]
            rew = [i for i in ids if g[i]["style"] != "base"]
            # DiD with the same cluster bootstrap
            by = defaultdict(list)
            for i in ids:
                by[clus[i]].append(i)
            qs = sorted(by)
            rng = random.Random(zlib.crc32(f"s3did|{m}|{c}".encode()))
            gap = lambda it: sum(b[i] - a[i] for i in it) / len(it)  # noqa: E731
            did = gap(rew) - gap(base)
            bs = []
            for _ in range(REPS):
                pick = [qs[rng.randrange(len(qs))] for _ in qs]
                it = [i for q in pick for i in by[q]]
                bb = [i for i in it if g[i]["style"] == "base"]
                rr = [i for i in it if g[i]["style"] != "base"]
                bs.append(gap(rr) - gap(bb))
            bs.sort()
            res[m][c] = {"overall": ov, "did": did, "did_ci95": [bs[int(0.025 * REPS)], bs[int(0.975 * REPS) - 1]]}
            body.append([NICE[m], c, pct(ov["diff"]), ci(ov["ci95"]), pct(did), ci(res[m][c]["did_ci95"])])
    OUT["S3"] = res
    table(["Model", "Build", "Overall shift", "95% cluster CI", "DiD (rewrites - base)", "95% cluster CI"], body)


def c4_benign():
    section("C4 / S4 Benign prompts: refusal (REFUSAL or PARTIAL, xstest rubric)")
    body, res = [], {}
    for m in MODELS:
        res[m] = {}
        d = R / "x1" / m
        g = {r["i"]: r for r in rows(d / "bf16.jsonl")}
        safe = [i for i, r in g.items() if r["safe"] is True]
        assert len(safe) == 250, (m, len(safe))
        a = judged(d / "judge_xstest_bf16.jsonl", REFUSE)
        d2 = R / "x2/wjb" / m
        a2 = judged(d2 / "judge_xstest_bf16.jsonl", REFUSE)
        for c in ("4bit-g64", "mxfp4"):
            x = paired(a, judged(d / f"judge_xstest_{c}.jsonl", REFUSE), safe, f"s4|{m}|{c}")
            w = paired(a2, judged(d2 / f"judge_xstest_{c}.jsonl", REFUSE), list(a2), f"wjbb|{m}|{c}")
            res[m][c] = {"xstest_safe": x, "wjb_benign": w}
            body.append([NICE[m], c, rate(x["rate_a"]), pct(x["diff"]) + " " + ci(x["ci95"]), f"{x['p']:.2g}",
                         rate(w["rate_a"]), pct(w["diff"]) + " " + ci(w["ci95"]), f"{w['p']:.2g}"])
    pv = {f"{m}|{c}": res[m][c]["xstest_safe"]["p"] for m in MODELS for c in ("4bit-g64", "mxfp4")}
    for k, v in holm(pv).items():
        m, c = k.split("|")
        res[m][c]["xstest_safe"]["p_holm"] = v
    OUT["C4_benign"] = res
    table(["Model", "Build", "XSTest safe: bf16 refusal", "Shift [95% CI]", "p", "WJB benign: bf16 refusal", "Shift [95% CI]", "p"], body)


def c4_capability():
    section("C4 Capability: teacher-forced NLL per token on bf16's own harmless replies")
    body, res = [], {}
    for m in MODELS:
        c1 = json.loads((R / "c1" / f"{m}.json").read_text())
        base = c1["bf16"]["nll_per_token"]
        res[m] = {c: {"bpw": v["bits_per_weight"], "nll": v["nll_per_token"], "dnll": v["nll_per_token"] - base}
                  for c, v in c1.items() if isinstance(v, dict)}
        for c in ("bf16", "8bit-g64", "4bit-g32", "mixed_4_6", "4bit-g64", "mxfp4"):
            v = res[m][c]
            body.append([NICE[m], c, f"{v['bpw']:.2f}", f"{v['nll']:.4f}", f"{v['dnll']:+.4f}"])
    OUT["C4_capability"] = res
    table(["Model", "Format", "Bits/weight", "NLL/token", "Change vs bf16"], body)


def robustness():
    section("Robustness: length (E4, 150 random prompts) and generic prefix (S8)")
    res, body = {}, []
    for tag, m, short, long_ in (("E4qwen", "qwen2.5-3b", "a2/k8", "a2/k8_len384"),
                                 ("E4llama", "llama3.2-3b", "x2/wjbho", "x2/wjbho_len384")):
        dl = R / long_ / m
        if not have(dl / "judge_v2_4bit-g64.jsonl"):
            continue
        sub = sorted(judged(dl / "judge_v2_bf16.jsonl"))
        ds = R / short / m
        s = paired(judged(ds / "judge_v2_bf16.jsonl"), judged(ds / "judge_v2_4bit-g64.jsonl"), sub, f"e4s|{m}")
        l_ = paired(judged(dl / "judge_v2_bf16.jsonl"), judged(dl / "judge_v2_4bit-g64.jsonl"), sub, f"e4l|{m}")
        cut = sum(r["finish_reason"] != "stop" for r in rows(dl / "bf16.jsonl")) / len(sub)
        res[tag] = {"128": s, "384": l_, "bf16_still_cut_at_384": cut,
                    "verdict": "within 3 points" if abs(l_["diff"] - s["diff"]) <= 0.03 else ("shrinks" if abs(l_["diff"]) < abs(s["diff"]) else "grows")}
        body.append([{"E4qwen": "Qwen2.5 forced k=8", "E4llama": "Llama free (E1)"}[tag], len(sub), pct(s["diff"]) + " " + ci(s["ci95"]), pct(l_["diff"]) + " " + ci(l_["ci95"]), rate(cut, 0), res[tag]["verdict"]])
    s8, pv = {}, {}
    for m in MODELS:
        d = R / "a2/sure" / m
        if have(d / "judge_v2_4bit-g64.jsonl"):
            uid = unique_ids(d / "bf16.jsonl")
            s8[m] = paired(judged(d / "judge_v2_bf16.jsonl"), judged(d / "judge_v2_4bit-g64.jsonl"), uid, f"s8|{m}")
            pv[m] = s8[m]["p"]
    for m, v in holm(pv).items():
        s8[m]["p_holm"] = v
    res["S8"] = s8
    b1 = {}
    for m in ("llama3.2-3b", "qwen2.5-3b"):  # B1: same held-out prompts, forced generic start
        d = R / "x2/wjbho_sure" / m
        if have(d / "judge_v2_bf16.jsonl", d / "judge_v2_4bit-g64.jsonl"):
            a_ = judged(d / "judge_v2_bf16.jsonl")
            b1[m] = paired(a_, judged(d / "judge_v2_4bit-g64.jsonl"), sorted(a_), f"b1|{m}")
    res["B1"] = b1
    if b1:
        table(["B1 held-out prompts, forced 'Sure, here is'", "n", "bf16", "4bit", "Shift [95% CI]", "p"],
              [[NICE[m], r["n"], rate(r["rate_a"]), rate(r["rate_b"]), pct(r["diff"]) + " " + ci(r["ci95"]), f"{r['p']:.2g}"] for m, r in b1.items()])
    OUT["robustness"] = res
    table(["Check", "n", "Shift at 128 tok", "Shift at 384 tok", "bf16 cut at 384", "Verdict"], body)
    table(["S8 'Sure, here is'", "n", "bf16", "4bit", "Shift [95% CI]", "p", "Holm p (models run)"],
          [[NICE[m], r["n"], rate(r["rate_a"]), rate(r["rate_b"]), pct(r["diff"]) + " " + ci(r["ci95"]), f"{r['p']:.2g}", f"{r['p_holm']:.2g}"]
           for m, r in s8.items()])
    if len(s8) < 3:
        NOTES.append(f"S8 run for {len(s8)} of 3 models so far; Holm is over the models run.")


def judges_and_humans():
    section("Label validity: Llama Guard on the primary, and the two-phase human labels")
    body, res = [], {}
    for m in MODELS:
        d = R / "a2/k8" / m
        if not have(d / "judge_lg_bf16.jsonl", d / "judge_lg_4bit-g64.jsonl"):
            continue
        a, b = lg(d / "judge_lg_bf16.jsonl"), lg(d / "judge_lg_4bit-g64.jsonl")
        uid = [i for i in unique_ids(d / "bf16.jsonl") if i in a and i in b]
        r_ = paired(a, b, uid, f"lg|{m}")
        res[m] = {"llama_guard": r_}
        body.append([NICE[m], r_["n"], pct(r_["diff"]) + " " + ci(r_["ci95"]), f"{r_['p']:.2g}"])
    # human labels, design-weighted (l2_pair_sheet.score logic)
    lab_f = R / "labels/h1/labels_h1.json"
    if lab_f.exists():
        lab = json.loads(lab_f.read_text())
        key = json.loads((R / "labels/h1/key.json").read_text())
        items, sizes = key["items"], key["stratum_sizes"]
        done = {k: v for k, v in lab.items() if "A" in v and "B" in v}
        for m in MODELS:
            cnt = defaultdict(int)
            for k in done:
                if items[k]["model"] == m:
                    cnt[items[k]["stratum"]] += 1
            if not cnt:
                continue
            w = {s: sizes[f"{m}|{s}"] / cnt[s] for s in cnt}
            net = tot = agree = n_rep = 0.0
            cm = {"bf16": defaultdict(float), "4bit-g64": defaultdict(float)}
            for k in done:
                it = items[k]
                if it["model"] != m:
                    continue
                h = {it["A_cfg"]: lab[k]["A"] in HARM, it["B_cfg"]: lab[k]["B"] in HARM}
                net += w[it["stratum"]] * (h["4bit-g64"] - h["bf16"])
                tot += w[it["stratum"]]
                agree += (h[it["A_cfg"]] == it["judge_A"]) + (h[it["B_cfg"]] == it["judge_B"])
                n_rep += 2
                for side in ("A", "B"):
                    cfg, jh = it[f"{side}_cfg"], it[f"judge_{side}"]
                    cm[cfg][(jh, h[cfg])] += w[it["stratum"]]
            def stats(c):
                tot_ = sum(c.values())
                po = (c[(True, True)] + c[(False, False)]) / tot_
                pj = (c[(True, True)] + c[(True, False)]) / tot_
                ph = (c[(True, True)] + c[(False, True)]) / tot_
                pe = pj * ph + (1 - pj) * (1 - ph)
                jpos, jneg = c[(True, True)] + c[(True, False)], c[(False, True)] + c[(False, False)]
                return {"agreement": po, "kappa": (po - pe) / (1 - pe) if pe < 1 else None,
                        "ppv": c[(True, True)] / jpos if jpos else None, "npv": c[(False, False)] / jneg if jneg else None,
                        "judge_rate": pj, "human_rate": ph}
            # R4 consequence: human-label discordant-pair analysis. Within each stratum, count pairs that
            # humans split toward the 4-bit build (b) or toward bf16 (a); exact McNemar on the judge-discordant
            # stratum (a census), and a stratified bootstrap of the design-weighted net shift.
            per = defaultdict(list)
            for k in done:
                it = items[k]
                if it["model"] == m:
                    h = {it["A_cfg"]: lab[k]["A"] in HARM, it["B_cfg"]: lab[k]["B"] in HARM}
                    per[it["stratum"]].append(int(h["4bit-g64"]) - int(h["bf16"]))
            hb = {st: (sum(1 for x in v if x > 0), sum(1 for x in v if x < 0)) for st, v in per.items()}
            N = sum(sizes[f"{m}|{st}"] for st in per)
            rng_h = random.Random(zlib.crc32(f"h1boot|{m}".encode()))
            boots = []
            for _ in range(REPS):
                tot_b = 0.0
                for st, v in per.items():
                    smp = [v[rng_h.randrange(len(v))] for _ in v]
                    tot_b += sizes[f"{m}|{st}"] * sum(smp) / len(smp)
                boots.append(tot_b / N)
            boots.sort()
            bD, aD = hb.get("D", (0, 0))
            from scipy.stats import binomtest
            pD = binomtest(bD, bD + aD, 0.5).pvalue if bD + aD else None
            disc = {"stratum_split": {st: {"to_4bit": b_, "to_bf16": a_} for st, (b_, a_) in hb.items()},
                    "D_to_4bit": bD, "D_to_bf16": aD, "D_p": pD,
                    "net_ci95": [boots[int(0.025 * REPS)], boots[int(0.975 * REPS) - 1]],
                    "net_ci90": [boots[int(0.05 * REPS)], boots[int(0.95 * REPS) - 1]]}
            both = defaultdict(float)
            for cfg in cm:
                for k_, v in cm[cfg].items():
                    both[k_] += v
            res.setdefault(m, {})["human"] = {"pairs": sum(cnt.values()), "strata": dict(cnt),
                                              "net_design_weighted": net / tot, "judge_agreement_unweighted": agree / n_rep,
                                              "weighted": stats(both), "by_format": {c: stats(cm[c]) for c in cm}, "discordant": disc}
    # J3: Llama Guard on the held-out free-start Llama cell (unsafe rate, bf16 vs 4bit-g64)
    d = R / "x2/wjbho" / "llama3.2-3b"
    if have(d / "judge_lg_bf16.jsonl", d / "judge_lg_4bit-g64.jsonl"):
        a, b = lg(d / "judge_lg_bf16.jsonl"), lg(d / "judge_lg_4bit-g64.jsonl")
        r_ = paired(a, b, sorted(set(a) & set(b)), "lg|wjbho|llama")
        res.setdefault("free_llama", {})["llama_guard"] = r_
        body.append(["Llama-3.2-3B (free start, held out)", r_["n"], pct(r_["diff"]) + " " + ci(r_["ci95"]), f"{r_['p']:.2g}"])
    OUT["labels"] = res
    table(["Model", "n", "Llama Guard shift [95% CI]", "p"], body)
    table(["Model", "Human pairs", "Human net shift (dw)", "Agreement (dw)", "Kappa (dw)", "PPV bf16 / 4bit", "NPV bf16 / 4bit", "Judge rate / human rate (dw)"],
          [[NICE[m], r["human"]["pairs"], pct(r["human"]["net_design_weighted"]), rate(r["human"]["weighted"]["agreement"]),
            f"{r['human']['weighted']['kappa']:.2f}",
            " / ".join(rate(r["human"]["by_format"][c]["ppv"], 0) for c in ("bf16", "4bit-g64")),
            " / ".join(rate(r["human"]["by_format"][c]["npv"], 0) for c in ("bf16", "4bit-g64")),
            f"{rate(r['human']['weighted']['judge_rate'])} / {rate(r['human']['weighted']['human_rate'])}"]
           for m, r in res.items() if "human" in r])
    NOTES.append("R4 judge-failure rule: design-weighted kappa < 0.60, or a per-format PPV difference whose CI excludes 0.")
    NOTES.append("Human-label net shift is a design-weighted point estimate; do not quote it as an effect size.")


def truncation():
    """How the 128-token reply cap interacts with the judge and the human labels."""
    section("Truncation: the reply cap, judge-human disagreement, and a complete-replies-only shift")
    res, body = {}, []
    key = json.loads((R / "labels/h1/key.json").read_text())["items"]
    ege_f = R / "labels/h1/labels_h1.json"
    ege = json.loads(ege_f.read_text()) if ege_f.exists() else {}
    dis = {True: [0, 0], False: [0, 0]}
    for m in MODELS:
        g = {c: {r["i"]: r for r in rows(R / "a2/k8" / m / f"{c}.jsonl")} for c in ("bf16", "4bit-g64")}
        j = {c: judged(R / "a2/k8" / m / f"judge_v2_{c}.jsonl") for c in ("bf16", "4bit-g64")}
        ids = sorted(set(j["bf16"]) & set(j["4bit-g64"]))
        trunc = {c: sum(1 for i in ids if g[c][i]["finish_reason"] == "length") / len(ids) for c in g}
        comp = [i for i in ids if all(g[c][i]["finish_reason"] == "stop" for c in g)]
        r = {"trunc_share": trunc, "n_complete": len(comp),
             "all": paired(j["bf16"], j["4bit-g64"], ids, f"trunc|{m}|all")}
        if len(comp) >= 20:
            r["complete"] = paired(j["bf16"], j["4bit-g64"], comp, f"trunc|{m}|comp")
        res[m] = r
        body.append([NICE[m], rate(trunc["4bit-g64"]), len(comp), pct(r["all"]["diff"]),
                     (pct(r["complete"]["diff"]) + " " + ci(r["complete"]["ci95"])) if "complete" in r else "n too small"])
        for hid, it in key.items():
            if it["model"] != m or hid not in ege or "A" not in ege[hid]:
                continue
            for sd in ("A", "B"):
                t = g[it[f"{sd}_cfg"]][it["i"]]["finish_reason"] == "length"
                dis[t][0] += (ege[hid][sd] in HARM) != it[f"judge_{sd}"]
                dis[t][1] += 1
    res["disagreement"] = {"truncated": dis[True], "complete": dis[False]}
    OUT["TRUNC"] = res
    table(["Model", "4-bit replies truncated", "complete pairs", "shift (all)", "shift (complete only)"], body)
    for k_, (d, n) in (("truncated", dis[True]), ("complete", dis[False])):
        if n:
            MD.append(f"- judge-human disagreement on {k_} replies: {d}/{n} = {100 * d / n:.1f}%")
    MD.append("")


def c3_controls():
    section("C3 Matched-perturbation controls (E2)")
    cells = {"Qwen2.5 forced": ("qwen2.5-3b", "a2/k8", True), "Llama free": ("llama3.2-3b", "x2/wjbho", False)}
    res = {}
    for tag, (m, src, uniq) in cells.items():
        d = R / src / m
        info_f = R / "e2" / m / "builds.json"
        gate_f = R / "e2" / m / f"gate_{'forced' if uniq else 'free'}.json"
        if not gate_f.exists():
            NOTES.append(f"C3 {tag}: gate not run yet.")
            continue
        gate = json.loads(gate_f.read_text())
        info = json.loads(info_f.read_text()) if info_f.exists() else {}
        a = judged(d / "judge_v2_bf16.jsonl")
        ids = unique_ids(d / "bf16.jsonl") if uniq else sorted(a)
        real = judged(d / "judge_v2_4bit-g64.jsonl")
        ALL = ("e2-dense", "e2-mirror", "e2-perm1", "e2-perm2", "e2-perm3", "e2-gauss", "e2-gauss2", "e2-gauss3",
               "e2-scale0.5", "e2-scale2")
        got = {k: judged(d / f"judge_v2_{k}.jsonl") for k in ALL if have(d / f"judge_v2_{k}.jsonl")}
        r = {"gate": gate, "builds": info, "real": paired(a, real, ids, f"c3|{m}|real")}
        if "e2-dense" in got:
            sub = sorted(set(got["e2-dense"]) & set(ids))
            r["dense_vs_real"] = paired(real, got["e2-dense"], sub, f"c3|{m}|dense")
            r["dense"] = paired(a, got["e2-dense"], sub, f"c3|{m}|denseshift")
        for k in ALL[1:]:
            if k in got:
                r[k] = paired(a, got[k], ids, f"c3|{m}|{k}")
        # Registered verdicts: "summary3" uses the addendum-3 draws (permuted 1-2, Gaussian 1);
        # "summary" uses all six random draws of addendum 5 once they exist.
        three = ("e2-perm1", "e2-perm2", "e2-gauss")
        six = ("e2-perm1", "e2-perm2", "e2-perm3", "e2-gauss", "e2-gauss2", "e2-gauss3")
        for tag_, pool in (("summary3", three), ("summary", six)):
            if "e2-mirror" in got and all(k in got for k in pool):
                r[tag_] = verdict_for(a, real, got, pool, ids, info, r, m)
        # registered (addendum 4): paired-bootstrap OLS slope of shift on a over a in {-1, 0.5, 1, 2}
        dose = [(-1.0, got.get("e2-mirror")), (0.5, got.get("e2-scale0.5")), (1.0, real), (2.0, got.get("e2-scale2"))]
        if all(v is not None for _, v in dose):
            xs = [x for x, _ in dose]
            mx_ = sum(xs) / len(xs)
            sxx = sum((x - mx_) ** 2 for x in xs)
            def slope(sample):
                ys = [sum(v[i] - a[i] for i in sample) / len(sample) for _, v in dose]
                my = sum(ys) / len(ys)
                return sum((x - mx_) * (y - my) for x, y in zip(xs, ys)) / sxx
            rng_d = random.Random(zlib.crc32(f"dose|{m}".encode()))
            bs = sorted(slope([ids[rng_d.randrange(len(ids))] for _ in ids]) for _ in range(REPS))
            r["dose_slope"] = {"slope": slope(ids), "ci95": [bs[int(0.025 * REPS)], bs[int(0.975 * REPS) - 1]]}
        res[tag] = r
    OUT["C3"] = res
    for tag, r in res.items():
        MD.append(f"**{tag}** -- gate: {'PASS' if r['gate']['pass'] else 'FAIL'} (NLL diff {r['gate']['nll_diff']:+.4f}, "
                  f"first-token agreement {rate(r['gate']['first_token_agree'], 1)})\n")
        body = [["real 4bit-g64", pct(r["real"]["diff"]), ci(r["real"]["ci95"]), ""]]
        if "dense" in r:
            body.append([f"dense W+E (n={r['dense']['n']})", pct(r["dense"]["diff"]), ci(r["dense"]["ci95"]),
                         f"vs real: churn {rate(r['dense_vs_real']['churn'])}"])
        for k in ALL[1:]:
            if k in r:
                body.append([k, pct(r[k]["diff"]), ci(r[k]["ci95"]), f"NLL gap {r['builds'].get(k, {}).get('nll_gap', float('nan')):+.4f}"])
        table(["Build", "Shift vs bf16", "95% CI", "Note"], body)
        for tag_, label in (("summary3", "three draws (addendum 3)"), ("summary", "six draws (addendum 5)")):
            if tag_ in r:
                s = r[tag_]
                MD.append(f"{label}: random-control {pct(s['point']['random'])} {ci(s['ci95']['random'])}; mirror {pct(s['point']['mirror'])} "
                          f"{ci(s['ci95']['mirror'])}; real - random {pct(s['point']['real_minus_random'])} {ci(s['ci95']['real_minus_random'])}; "
                          f"even {pct(s['point']['even'])} {ci(s['ci95']['even'])}; odd {pct(s['point']['odd'])} {ci(s['ci95']['odd'])}; "
                          f"draw spread [{pct(s['draw_min'])}, {pct(s['draw_max'])}]. **Verdict: {s['verdict']}.**\n")
        if "summary3" not in r:
            NOTES.append(f"C3 {tag}: not all control builds judged yet.")


def verdict_for(a, real, got, pool, ids, info, r, m):
    """Registered verdict rule (addendum 3) for one pool of random controls."""
    excl = [k for k in pool if info.get(k, {}).get("not_matched_in_damage")]
    rnd = [k for k in pool if k not in excl]
    rng = random.Random(zlib.crc32(f"c3boot|{m}|{len(pool)}".encode()))
    n = len(ids)
    series = [("real", real), ("e2-mirror", got["e2-mirror"])] + [(k, got[k]) for k in rnd]
    stats = defaultdict(list)
    for _ in range(REPS):
        s = [ids[rng.randrange(n)] for _ in ids]
        sh = {k: sum(v[i] - a[i] for i in s) / n for k, v in series}
        rc = sum(sh[k] for k in rnd) / len(rnd)
        stats["mirror"].append(sh["e2-mirror"])
        stats["random"].append(rc)
        stats["real_minus_random"].append(sh["real"] - rc)
        stats["real_minus_mirror"].append(sh["real"] - sh["e2-mirror"])
        stats["even"].append((sh["real"] + sh["e2-mirror"]) / 2)
        stats["odd"].append((sh["real"] - sh["e2-mirror"]) / 2)
    civ = {k: [sorted(v)[int(0.025 * REPS)], sorted(v)[int(0.975 * REPS) - 1]] for k, v in stats.items()}
    pt = {"mirror": r["e2-mirror"]["diff"], "random": sum(r[k]["diff"] for k in rnd) / len(rnd),
          "even": (r["real"]["diff"] + r["e2-mirror"]["diff"]) / 2, "odd": (r["real"]["diff"] - r["e2-mirror"]["diff"]) / 2}
    pt["real_minus_random"] = r["real"]["diff"] - pt["random"]
    pt["real_minus_mirror"] = r["real"]["diff"] - pt["mirror"]
    sgn = 1 if r["real"]["diff"] > 0 else -1
    excl0 = lambda c: c[0] > 0 or c[1] < 0  # noqa: E731
    lean = pt["random"] * sgn > 0 and excl0(civ["random"]) and not (pt["mirror"] * sgn < 0 and excl0(civ["mirror"]))
    rounding = (pt["mirror"] * sgn < 0 and excl0(civ["mirror"])) or (not excl0(civ["random"]) and excl0(civ["real_minus_random"]))
    verdict = "unresolved" if lean == rounding else ("model's lean" if lean else "rounding-specific")
    draws = [r[k]["diff"] for k in rnd]
    return {"point": pt, "ci95": civ, "random_controls_used": rnd, "excluded_not_matched": excl, "verdict": verdict,
            "draw_min": min(draws), "draw_max": max(draws)}


def s9_quantizers_runtime():
    section("S9 / G1 Other quantizers in the same format, and a second runtime")
    cells = {"Qwen2.5 forced": ("qwen2.5-3b", "a2/k8", True), "Llama free": ("llama3.2-3b", "x2/wjbho", False)}
    res, body = {}, []
    for tag, (m, src, uniq) in cells.items():
        d = R / src / m
        res[tag] = {}
        mlx_ref = judged(d / "judge_v2_bf16.jsonl")
        ids = unique_ids(d / "bf16.jsonl") if uniq else sorted(mlx_ref)
        for b, ref_name in (("4bit-g64", "bf16"), ("awq", "bf16"), ("gptq", "bf16"), ("gptq6", "bf16"),  # gptq6: deviation, outside Holm
                            ("gguf-q8_0", "gguf-bf16"), ("gguf-q4km", "gguf-bf16")):
            if not have(d / f"judge_v2_{b}.jsonl", d / f"judge_v2_{ref_name}.jsonl"):
                continue
            ref = judged(d / f"judge_v2_{ref_name}.jsonl")
            cur = judged(d / f"judge_v2_{b}.jsonl")
            sub = [i for i in ids if i in ref and i in cur]
            res[tag][b] = paired(ref, cur, sub, f"s9|{m}|{b}")
            res[tag][b]["reference"] = ref_name
            body.append([tag, b, ref_name, pct(res[tag][b]["diff"]) + " " + ci(res[tag][b]["ci95"]), f"{res[tag][b]['p']:.2g}"])
        if have(d / "judge_v2_gguf-bf16.jsonl"):
            g = judged(d / "judge_v2_gguf-bf16.jsonl")
            sub = [i for i in ids if i in g]
            res[tag]["runtime_bf16"] = paired(mlx_ref, g, sub, f"s9rt|{m}")  # MLX bf16 -> llama.cpp bf16
            body.append([tag, "gguf-bf16", "MLX bf16", pct(res[tag]["runtime_bf16"]["diff"]), f"{res[tag]['runtime_bf16']['p']:.2g}"])
        # bits per weight of the GGUF builds, from file size against the same model's bf16 GGUF
        ref16 = ROOT / "models" / "gguf" / f"{m}-gguf-bf16.gguf"
        if ref16.exists():
            for b in ("q8_0", "q4km"):
                f = ROOT / "models" / "gguf" / f"{m}-gguf-{b}.gguf"
                if f.exists():
                    res[tag][f"gguf-{b}"] = res[tag].get(f"gguf-{b}", {})
                    res[tag][f"gguf-{b}"]["bpw"] = 16.0 * f.stat().st_size / ref16.stat().st_size
    # Registered families: S9 (AWQ, GPTQ) and G1 (GGUF), Holm within each, not pooled.
    # The Qwen2.5 GPTQ builds fail the capability check and are excluded (Appendix A).
    for fam in (("awq", "gptq"), ("gguf-q8_0", "gguf-q4km")):
        pv = {f"{t}|{b}": r["p"] for t, c in res.items() for b, r in c.items()
              if b in fam and "p" in r and not (t.startswith("Qwen2.5") and b.startswith("gptq"))}
        for k, v in holm(pv).items():
            t, b = k.split("|")
            res[t][b]["p_holm"] = v
    OUT["S9"] = res
    if body:
        table(["Cell", "Build", "Reference", "Shift [95% CI]", "p"], body)
    else:
        NOTES.append("S9/G1: no AWQ, GPTQ or GGUF cells judged yet.")


def m1_direction():
    section("M1 Refusal-direction projection (change vs bf16 at the abliteration layer)")
    import numpy as np
    res, body = {}, []
    for tag, (m, cell) in {"Qwen2.5 forced": ("qwen2.5-3b", "forced"), "Llama free": ("llama3.2-3b", "free"),
                           "Qwen2.5 free": ("qwen2.5-3b", "free"), "Llama forced": ("llama3.2-3b", "forced")}.items():
        d = R / "m1" / m / cell
        if not (d / "bf16.npy").exists():
            continue
        L = json.loads((R / "e7" / m / "choice.json").read_text())["layer"]
        base = np.load(d / "bf16.npy")
        res[tag] = {"layer": L, "builds": {}}
        for f in sorted(d.glob("*.npy")):
            b = f.stem
            if b == "bf16":
                continue
            delta = np.load(f) - base
            x = delta[:, L]
            rng = np.random.default_rng(zlib.crc32(f"m1|{m}|{cell}|{b}".encode()))
            bs = np.sort([x[rng.integers(0, len(x), len(x))].mean() for _ in range(REPS)])
            res[tag]["builds"][b] = {"mean": float(x.mean()), "ci95": [float(bs[int(0.025 * REPS)]), float(bs[int(0.975 * REPS) - 1])],
                                     "profile": delta.mean(axis=0).tolist()}
        # across-build association with the behavioural shift (cells with judged builds)
        beh = {}
        for b in res[tag]["builds"]:
            # behavioural shift from the same cell only: forced -> ladder (S6), free -> held-out cells (C2)
            same_cell = (OUT.get("S6", {}).get(m, {}).get(b) if cell == "forced"
                         else OUT.get("C2", {}).get("cells", {}).get(m, {}).get(b))
            for cand in (OUT.get("C3", {}).get(tag, {}).get(b), same_cell, OUT.get("S9", {}).get(tag, {}).get(b)):
                if isinstance(cand, dict) and "diff" in cand:
                    beh[b] = cand["diff"]
                    break
        # the Qwen2.5 GPTQ build is broken (NLL about 30x bf16) and is kept out of the association
        broken = {"gptq", "gptq6"} if m == "qwen2.5-3b" else set()  # both fail the capability check (Appendix A)
        common = [b for b in beh if b in res[tag]["builds"] and b not in broken]
        if len(common) >= 4:
            from scipy.stats import spearmanr  # optional dependency, only for this summary
            rho, pval = spearmanr([res[tag]["builds"][b]["mean"] for b in common], [beh[b] for b in common])
            res[tag]["spearman"] = {"rho": float(rho), "p": float(pval), "builds": common}
        for b, v in res[tag]["builds"].items():
            body.append([tag, b, f"{v['mean']:+.3f}", f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]", pct(beh[b]) if b in beh else ""])
    OUT["M1"] = res
    if body:
        table(["Cell", "Build", "Mean change in projection", "95% CI", "Behavioural shift"], body)
        for tag, r in res.items():
            if "spearman" in r:
                MD.append(f"{tag}: across {len(r['spearman']['builds'])} builds, Spearman rho = {r['spearman']['rho']:+.2f} (p = {r['spearman']['p']:.2g}).\n")
    else:
        NOTES.append("M1: no projections yet.")


def s5_runon():
    section("S5 Run-on on benign prompts (role marker in the reply; 200 harmless prompts, 512 tokens)")
    body = []
    res = {}
    for m in MODELS + ("qwen2.5-0.5b",):
        f = R / "r5" / m / "summary.json"
        if f.exists():
            s = json.loads(f.read_text())
            res[m] = s
            body.append([NICE[m]] + [f"{k}: {rate(v['marker'])}" for k, v in s.items()])
    OUT["S5"] = res
    for b in body:
        MD.append("- " + " | ".join(b))
    MD.append("")


# ---- appendix tables (written as LaTeX so no number in the appendix is typed by hand) -------

EXTRA: dict = {}
GEN = ROOT / "paper" / "tex" / "generated"


def _tex(name, lines):
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / f"{name}.tex").write_text("% generated by analysis/make_numbers.py -- do not edit\n" + "\n".join(lines) + "\n")


def _pp(x):
    return f"{100 * x:+.1f}".replace("-", "$-$")


def _ci(c):
    return f"[{_pp(c[0])}, {_pp(c[1])}]"


def _p(p):
    return "$" + _fmt_p(p) + "$" if p is not None else "--"


def _cellrow(lab, r, holm_p=None):
    return (f"{lab} & {100 * r['rate_a']:.1f} & {100 * r['rate_b']:.1f} & {_pp(r['diff'])} {_ci(r['ci95'])} & "
            f"{r['a_only']}/{r['b_only']} & {_p(r['p'])}" + (f" & {_p(holm_p)}" if holm_p is not None else " & --") + r" \\")


TEXNAME = {"qwen2.5-3b": r"\qwen", "llama3.2-3b": r"\llama", "qwen3-4b": r"\qwenthree", "qwen2.5-0.5b": "Qwen2.5-0.5B"}
TEXBUILD = {"4bit-g64": "4-bit", "mxfp4": "MXFP4", "8bit-g64": "8-bit", "4bit-g32": "4-bit g32", "mixed_4_6": "mixed 4/6",
            "bf16": "bf16", "mxfp4-dequant": "MXFP4 deq.", "4bit-g64-dequant": "4-bit, dequantized"}


def appendix_tables():
    head = [r"\begin{tabular}{llrrlrll}", r"\toprule",
            r"Model & Cell & bf16 \% & build \% & Shift [95\% CI] & Disc. & $p$ & Holm $p$ \\", r"\midrule"]
    # S2 depth family
    S2 = OUT.get("S2", {})
    body = []
    for m in MODELS:
        for c in ("4bit-g64", "mxfp4"):
            for k in ("k0", "k2", "k8"):
                r = S2.get("cells", {}).get(m, {}).get(f"{c}|{k}")
                if r and not (c == "4bit-g64" and k == "k8"):  # the primary, reported in the main text
                    body.append(_cellrow(f"{TEXNAME[m]} & {TEXBUILD[c]}, $k={k[1:]}$", r, S2["p_holm"].get(f"{m}|{c}|{k}")))
        body.append(r"\addlinespace")
    _tex("tab_depth", head + body[:-1] + [r"\bottomrule", r"\end{tabular}"])
    # benign prompts
    B = OUT.get("C4_benign", {})
    body = []
    for m in MODELS:
        for c in ("4bit-g64", "mxfp4"):
            for fam, lab in (("xstest_safe", "XSTest safe"), ("wjb_benign", "WJB benign")):
                r = B.get(m, {}).get(c, {}).get(fam)
                if r:
                    body.append(_cellrow(f"{TEXNAME[m]} & {TEXBUILD[c]}, {lab}", r, r.get("p_holm")))
        body.append(r"\addlinespace")
    _tex("tab_benign", head + body[:-1] + [r"\bottomrule", r"\end{tabular}"])
    # SORRY-Bench, cluster intervals
    S3 = OUT.get("S3", {})
    body = [r"\begin{tabular}{llll}", r"\toprule",
            r"Model & Build & Overall shift [95\% cluster CI] & Rewrites $-$ base [95\% cluster CI] \\", r"\midrule"]
    for m in MODELS:
        for c in ("4bit-g64", "mxfp4"):
            r = S3.get(m, {}).get(c)
            if r:
                body.append(f"{TEXNAME[m]} & {TEXBUILD[c]} & {_pp(r['overall']['diff'])} {_ci(r['overall']['ci95'])} & "
                            f"{_pp(r['did'])} {_ci(r['did_ci95'])} \\\\")
    _tex("tab_sorry", body + [r"\bottomrule", r"\end{tabular}"])
    # free-start pilot, rebuilt as paired results so counts are exact
    body = []
    for m in MODELS:
        d = R / "x2/wjb" / m
        if not have(d / "judge_v2_bf16.jsonl", d / "judge_v2_4bit-g64.jsonl"):
            continue
        g = {r["i"]: r for r in rows(d / "bf16.jsonl")}
        a = {i: v for i, v in judged(d / "judge_v2_bf16.jsonl").items() if g[i]["kind"] == "harmful"}
        for c in ("4bit-g64", "mxfp4"):
            if have(d / f"judge_v2_{c}.jsonl"):
                r = paired(a, judged(d / f"judge_v2_{c}.jsonl"), list(a), f"pilot|{m}|{c}")
                body.append(_cellrow(f"{TEXNAME[m]} & {TEXBUILD[c]}", r))
                EXTRA[f"pilot.{SHORT[m]}.{c}.diff".replace("_", "")] = f"{100 * r['diff']:+.1f}"
                EXTRA[f"pilot.{SHORT[m]}.{c}.n".replace("_", "")] = str(r["n"])
    _tex("tab_pilot", [h.replace("Model & Cell &", "Model & Build &") for h in head] + body + [r"\bottomrule", r"\end{tabular}"])
    # controls: NLL gap, realized norm ratio, shift
    C3 = OUT.get("C3", {})
    cellmodel = {"Qwen2.5 forced": "qwen2.5-3b", "Llama free": "llama3.2-3b"}
    body = [r"\begin{tabular}{lllrl}", r"\toprule",
            r"Cell & Build & $\Delta$NLL/token & Norm ratio & Shift [95\% CI] \\", r"\midrule"]
    for tag, m in cellmodel.items():
        bj = R / "e2" / m / "builds.json"
        if not bj.exists() or tag not in C3:
            continue
        meta = json.loads(bj.read_text())
        real_gap = next(iter(meta.values()))["real_nll_gap"]
        real = C3[tag]['real']
        if tag == "Llama free":  # same contrast as the main text; take its interval so the two match
            real = OUT["C2"]["cells"]["llama3.2-3b"]["4bit-g64"]
        body.append(f"{tag} & real 4-bit & {real_gap:+.3f} & -- & {_pp(real['diff'])} {_ci(real['ci95'])} \\\\")
        for b in ("e2-dense", "e2-mirror", "e2-perm1", "e2-perm2", "e2-perm3", "e2-gauss", "e2-gauss2", "e2-gauss3", "e2-scale0.5", "e2-scale2"):
            if b not in meta:
                continue
            x = meta[b]
            nr = (f"{x['norm_ratio_min']:.3f}" if abs(x["norm_ratio_max"] - x["norm_ratio_min"]) < 5e-4
                  else f"{x['norm_ratio_min']:.3f}--{x['norm_ratio_max']:.3f}")
            sh = C3[tag].get(b) or (C3[tag].get("dense") if b == "e2-dense" else None)
            sh_s = f"{_pp(sh['diff'])} {_ci(sh['ci95'])}" if sh and "ci95" in sh else "pending"
            lab = {"e2-dense": r"dense $W+E$", "e2-mirror": r"mirror $W-E$", "e2-scale0.5": r"$W+\tfrac12E$",
                   "e2-scale2": r"$W+2E$", "e2-perm1": "permuted 1", "e2-perm2": "permuted 2", "e2-perm3": "permuted 3",
                   "e2-gauss": "Gaussian 1", "e2-gauss2": "Gaussian 2", "e2-gauss3": "Gaussian 3"}[b]
            body.append(f" & {lab} & {x['nll_gap']:+.3f} & {nr} & {sh_s} \\\\")
            EXTRA[f"ctrl.{SHORT[tag]}.{b}.nr".replace("_", "")] = nr
        gate = next((json.loads(f.read_text()) for f in (R / "e2" / m).glob("gate_*.json")), None)
        if gate:
            EXTRA[f"ctrl.{SHORT[tag]}.gate.n"] = str(gate["n"])
        body.append(r"\addlinespace")
    _tex("tab_controls", body[:-1] + [r"\bottomrule", r"\end{tabular}"])
    # per-matrix error structure of each quantizer, against the default RTN build
    import statistics as st
    body = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
            r" & \multicolumn{3}{c}{median relative error} & \multicolumn{3}{c}{median cosine with RTN error} \\",
            r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
            r"Model & RTN & MXFP4 & GPTQ & MXFP4 & GPTQ & AWQ \\", r"\midrule"]
    for m in ("qwen2.5-3b", "llama3.2-3b"):
        f = R / "e2" / m / "errors.json"
        if not f.exists():
            continue
        e = json.loads(f.read_text()).values()
        med = lambda k: st.median(v[k] for v in e if v.get(k) is not None)  # noqa: E731
        vals = {k: med(k) for k in ("rel_rtn", "rel_mxfp4", "rel_gptq", "cos_rtn_mxfp4", "cos_rtn_gptq", "cos_rtn_awq", "rel_awq")}
        body.append(f"{TEXNAME[m]} & {vals['rel_rtn']:.3f} & {vals['rel_mxfp4']:.3f} & {vals['rel_gptq']:.3f} & "
                    f"{vals['cos_rtn_mxfp4']:.2f} & {vals['cos_rtn_gptq']:.2f} & {vals['cos_rtn_awq']:.2f} \\\\")
        for k, v in vals.items():
            EXTRA[f"err.{SHORT[m]}.{k}".replace("_", "")] = f"{v:.3f}" if k.startswith("rel") else f"{v:.2f}"
        EXTRA[f"err.{SHORT[m]}.nmat"] = str(len(list(e)))
    _tex("tab_errors", body + [r"\bottomrule", r"\end{tabular}"])
    # human-label strata
    key = R / "labels/h1/key.json"
    if key.exists():
        k = json.loads(key.read_text())
        cnt = defaultdict(int)
        for v in k["items"].values():
            cnt[f"{v['model']}|{v['stratum']}"] += 1
        body = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
                r" & \multicolumn{2}{c}{D (judge splits)} & \multicolumn{2}{c}{P (both harmful)} & \multicolumn{2}{c}{N (neither)} \\",
                r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
                r"Model & pop. & labelled & pop. & labelled & pop. & labelled \\", r"\midrule"]
        for m in MODELS:
            body.append(f"{TEXNAME[m]} & " + " & ".join(f"{k['stratum_sizes'][f'{m}|{s}']} & {cnt[f'{m}|{s}']}" for s in "DPN") + r" \\")
        _tex("tab_strata", body + [r"\bottomrule", r"\end{tabular}"])
        EXTRA["human.totalpairs"] = str(len(k["items"]))
        EXTRA["human.totalreplies"] = str(2 * len(k["items"]))
    # run-on prevalence, one row per model
    S5 = OUT.get("S5", {})
    order = ("bf16", "8bit-g64", "4bit-g32", "mixed_4_6", "4bit-g64", "mxfp4", "mxfp4-dequant")
    body = [r"\begin{tabular}{l" + "r" * len(order) + "}", r"\toprule",
            "Model & " + " & ".join(TEXBUILD[c] for c in order) + r" \\", r"\midrule"]
    for m in MODELS + ("qwen2.5-0.5b",):
        cells = []
        for c in order:
            v = S5.get(m, {}).get(c)
            cells.append("--" if v is None else (f"{100 * v['marker']:.1f}" + ("" if v["n"] == 200 else f" ($n$={v['n']})")))
        body.append(f"{TEXNAME[m]} & " + " & ".join(cells) + r" \\")
    _tex("tab_runon", body + [r"\bottomrule", r"\end{tabular}"])
    # E3 scalars
    for m, r in OUT.get("S7", {}).items():
        e3 = r.get("e3")
        if isinstance(e3, dict):
            b = f"e3.{SHORT[m]}"
            EXTRA[b + ".varbf"] = f"{e3['var_bf16']:.3f}"
            EXTRA[b + ".varq"] = f"{e3['var_4bit']:.3f}"
            EXTRA[b + ".excess"] = f"{e3['excess']:+.3f}".replace("-", "$-$")
            EXTRA[b + ".ci"] = f"[{e3['ci95'][0]:+.3f}, {e3['ci95'][1]:+.3f}]".replace("-", "$-$")
            EXTRA[b + ".n"] = str(e3["n"])
    # checkpoint fingerprints: sha256 over the bf16 safetensors shards, cached (hashing 6 GB takes a while)
    import hashlib
    cache = R / "stats" / "model_hashes.json"
    hashes = json.loads(cache.read_text()) if cache.exists() else {}
    for m in MODELS:
        if m in hashes:
            continue
        d = ROOT / "models" / f"{m}-bf16"
        shards = sorted(d.glob("*.safetensors"))
        if not shards:
            continue
        h = hashlib.sha256()
        for sh in shards:
            with open(sh, "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(blk)
        hashes[m] = h.hexdigest()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(hashes, indent=1))
    for m, v in hashes.items():
        EXTRA[f"hash.{SHORT[m]}"] = v[:12]
    # logged machine time: union of every queue step's [start, end] interval, one machine, one model at a time
    import re as _re
    from datetime import datetime
    iv, opened = [], {}
    pat = _re.compile(r"^\[(.+?)\] (start|end) (.+?)( exit=\d+)?$")
    for f in sorted((ROOT / "logs").glob("*.log")):
        for line in f.read_text(errors="ignore").splitlines():
            mm = pat.match(line.strip())
            if not mm:
                continue
            try:
                t = datetime.strptime(mm.group(1).replace(" MST", ""), "%a %b %d %H:%M:%S %Y")
            except ValueError:
                continue
            if mm.group(2) == "start":
                opened[(f.name, mm.group(3))] = t
            elif (f.name, mm.group(3)) in opened:
                iv.append((opened.pop((f.name, mm.group(3))), t))
    iv.sort()
    tot, cur = 0.0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur:
                tot += (cur[1] - cur[0]).total_seconds()
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur:
        tot += (cur[1] - cur[0]).total_seconds()
    EXTRA["compute.hours"] = f"{tot / 3600:.0f}"
    EXTRA["compute.steps"] = str(len(iv))


# ---- LaTeX export -------------------------------------------------------------------------

SHORT = {"free_llama": "freellama", "Qwen2.5 forced": "qwenforced", "Llama free": "llamafree", "Qwen2.5 free": "qwenfree", "Llama forced": "llamaforced",
         "qwen2.5-0.5b": "qwensmall", "qwen2.5-3b": "qwen", "llama3.2-3b": "llama", "qwen3-4b": "qwenthree"}


def _fmt_p(p):
    if p is None:
        return "n/a"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}\\times10^{{{int(e)}}}"
    return f"{p:.2g}" if p < 0.1 else f"{p:.2f}"


def latex_macros():
    """Flatten every paired result in OUT into \\N{path.field} macros (paper/tex/numbers.tex)."""
    flat = {}

    def key(parts):
        return ".".join(SHORT.get(str(x), str(x)) for x in parts).replace("_", "")

    def walk(node, parts):
        if isinstance(node, dict) and {"diff", "ci95"} <= set(node):
            k = key(parts)
            flat[k + ".diff"] = f"{100 * node['diff']:+.1f}"
            flat[k + ".absdiff"] = f"{abs(100 * node['diff']):.1f}"
            flat[k + ".ci"] = f"[{100 * node['ci95'][0]:+.1f}, {100 * node['ci95'][1]:+.1f}]"
            if "ci90" in node:
                flat[k + ".cininety"] = f"[{100 * node['ci90'][0]:+.1f}, {100 * node['ci90'][1]:+.1f}]"
            for f in ("rate_a", "rate_b", "churn"):
                if f in node and node[f] is not None:
                    flat[k + "." + f.replace("_", "")] = f"{100 * node[f]:.1f}"
            for f in ("n", "a_only", "b_only", "clusters"):
                if f in node:
                    flat[k + "." + f.replace("_", "")] = str(node[f])
            for f in ("p", "p_holm", "p_holm_secondary"):
                if node.get(f) is not None:
                    flat[k + "." + f.replace("_", "")] = _fmt_p(node[f])
            if node.get("rr") is not None:
                flat[k + ".rr"] = f"{node['rr']:.2f}"
        if isinstance(node, dict) and node.get("bpw") is not None:
            flat[key(parts) + ".bpw"] = f"{node['bpw']:.2f}"
        if isinstance(node, dict):
            for kk, v in node.items():
                walk(v, parts + [kk])
    walk(OUT, [])
    flat.update(EXTRA)
    # a few non-paired scalars used in text
    for m in MODELS:
        cap = OUT.get("C4_capability", {}).get(m, {})
        for c, v in cap.items():
            flat[f"cap.{SHORT[m]}.{c}.dnll".replace("_", "")] = f"{v['dnll']:+.3f}"
            flat[f"cap.{SHORT[m]}.{c}.bpw".replace("_", "")] = f"{v['bpw']:.2f}"
        e0 = OUT.get("C0", {}).get(m, {}).get("e0")
        if e0:
            flat[f"e0.{SHORT[m]}.identical"] = f"{100 * e0['identical']:.0f}"
        for c in ("8bit-g64", "4bit-g64", "mxfp4"):
            c0 = OUT.get("C0", {}).get(m, {}).get(c)
            if c0:
                flat[f"c0.{SHORT[m]}.{c}.identical"] = f"{100 * c0['text_identical']:.0f}"
        h = OUT.get("labels", {}).get(m, {}).get("human")
        if h:
            flat[f"human.{SHORT[m]}.pairs"] = str(h["pairs"])
            flat[f"human.{SHORT[m]}.net"] = f"{100 * h['net_design_weighted']:+.1f}"
            flat[f"human.{SHORT[m]}.agree"] = f"{100 * h['judge_agreement_unweighted']:.0f}"
            flat[f"human.{SHORT[m]}.agreedw"] = f"{100 * h['weighted']['agreement']:.0f}"
            flat[f"human.{SHORT[m]}.kappa"] = f"{h['weighted']['kappa']:.2f}"
            dsc = h.get("discordant")
            if dsc:
                flat[f"human.{SHORT[m]}.dtofour"] = str(dsc["D_to_4bit"])
                flat[f"human.{SHORT[m]}.dtobf"] = str(dsc["D_to_bf16"])
                flat[f"human.{SHORT[m]}.dp"] = _fmt_p(dsc["D_p"])
                flat[f"human.{SHORT[m]}.netci"] = f"[{100 * dsc['net_ci95'][0]:+.1f}, {100 * dsc['net_ci95'][1]:+.1f}]"
                flat[f"human.{SHORT[m]}.netcininety"] = f"[{100 * dsc['net_ci90'][0]:+.1f}, {100 * dsc['net_ci90'][1]:+.1f}]"
            flat[f"human.{SHORT[m]}.judgerate"] = f"{100 * h['weighted']['judge_rate']:.1f}"
            flat[f"human.{SHORT[m]}.humanrate"] = f"{100 * h['weighted']['human_rate']:.1f}"
            for c in ("bf16", "4bit-g64"):
                bf = h["by_format"][c]
                flat[f"human.{SHORT[m]}.{c}.ppv"] = f"{100 * bf['ppv']:.0f}"
                flat[f"human.{SHORT[m]}.{c}.npv"] = f"{100 * bf['npv']:.0f}"
    pool = OUT.get("C1", {}).get("pooled")
    if pool:
        flat["C1.pooled.diff"] = f"{100 * pool['diff']:+.1f}"
        flat["C1.pooled.p"] = _fmt_p(pool["p"])
        flat["C1.pooled.aonly"], flat["C1.pooled.bonly"] = str(pool["a_only"]), str(pool["b_only"])
    for m, r in OUT.get("S7", {}).items():
        flat[f"S7.{SHORT[m]}.floor"] = f"{100 * r['floor_churn']:.1f}"
        flat[f"S7.{SHORT[m]}.quant"] = f"{100 * r['quant_churn']:.1f}"
    for k_, v in OUT.get("S2", {}).get("p_holm", {}).items():
        mm, c, kk = k_.split("|")
        flat[f"S2.holm.{SHORT[mm]}.{c}.{kk}"] = _fmt_p(v)
    for m in MODELS:
        fb = OUT.get("S6", {}).get(m, {}).get("_four_bit")
        if fb:
            flat[f"S6.{SHORT[m]}.range"] = f"{100 * fb['range']:.1f}"
            flat[f"S6.{SHORT[m]}.qp"] = _fmt_p(fb["p_holm"])
    for m, summ in OUT.get("S5", {}).items():
        for c, v in summ.items():
            flat[f"s5.{SHORT[m]}.{c}".replace("_", "").replace("-", "")] = f"{100 * v['marker']:.1f}"
    # C3 verdict summaries and per-build damage; M1 projections
    tr = OUT.get("TRUNC", {})
    for mm, rr in tr.items():
        if mm == "disagreement":
            continue
        flat[f"trunc.{SHORT[mm]}.share"] = f"{100 * rr['trunc_share']['4bit-g64']:.0f}"
        flat[f"trunc.{SHORT[mm]}.ncomplete"] = str(rr["n_complete"])
        if "complete" in rr:
            flat[f"trunc.{SHORT[mm]}.completediff"] = f"{100 * rr['complete']['diff']:+.1f}"
            flat[f"trunc.{SHORT[mm]}.completeci"] = f"[{100 * rr['complete']['ci95'][0]:+.1f}, {100 * rr['complete']['ci95'][1]:+.1f}]"
    dg = tr.get("disagreement")
    if dg:
        for k_ in ("truncated", "complete"):
            d, n = dg[k_]
            if n:
                flat[f"trunc.dis.{k_}"] = f"{100 * d / n:.1f}"
                flat[f"trunc.dis.{k_}.n"] = str(n)
    for tag, r in OUT.get("C3", {}).items():
        t = SHORT.get(tag, tag)
        for sname in ("summary3", "summary"):
            sm = r.get(sname)
            if not sm:
                continue
            for k_, v in sm["point"].items():
                flat[f"C3.{t}.{sname}.{k_}".replace("_", "")] = f"{100 * v:+.1f}"
                flat[f"C3.{t}.{sname}.{k_}.ci".replace("_", "")] = f"[{100 * sm['ci95'][k_][0]:+.1f}, {100 * sm['ci95'][k_][1]:+.1f}]"
            flat[f"C3.{t}.{sname}.drawmin"] = f"{100 * sm['draw_min']:+.1f}"
            flat[f"C3.{t}.{sname}.drawmax"] = f"{100 * sm['draw_max']:+.1f}"
            flat[f"C3.{t}.{sname}.verdict"] = sm["verdict"]
        for b, info in r.get("builds", {}).items():
            if isinstance(info, dict) and info.get("nll_gap") is not None:
                flat[f"C3.{t}.{b}.nllgap"] = f"{info['nll_gap']:+.3f}"
                if info.get("real_nll_gap") is not None:
                    flat[f"C3.{t}.realnllgap"] = f"{info['real_nll_gap']:+.3f}"
        if r.get("dose_slope"):
            ds = r["dose_slope"]
            flat[f"C3.{t}.slope"] = f"{100 * ds['slope']:+.1f}"
            flat[f"C3.{t}.slope.ci"] = f"[{100 * ds['ci95'][0]:+.1f}, {100 * ds['ci95'][1]:+.1f}]"
    for tag, r in OUT.get("M1", {}).items():
        t = SHORT.get(tag, tag)
        flat[f"M1.{t}.layer"] = str(r["layer"])
        for b, v in r["builds"].items():
            flat[f"M1.{t}.{b}.mean"] = f"{v['mean']:+.2f}"
            flat[f"M1.{t}.{b}.ci"] = f"[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
        if "spearman" in r:
            flat[f"M1.{t}.rho"] = f"{r['spearman']['rho']:+.2f}"
            flat[f"M1.{t}.rhop"] = _fmt_p(r["spearman"]["p"])
            flat[f"M1.{t}.nbuilds"] = str(len(r["spearman"]["builds"]))
    flat["humanpairs.total"] = str(sum(OUT.get("labels", {}).get(m, {}).get("human", {}).get("pairs", 0) for m in MODELS))
    if OUT.get("C2", {}).get("primary_verdict"):
        flat["c2.verdict"] = OUT["C2"]["primary_verdict"]
    for tag, r in OUT.get("C3", {}).items():
        t = "qwen" if tag.startswith("Qwen") else "llama"
        flat[f"c3.{t}.gate.nlldiff"] = f"{r['gate']['nll_diff']:+.4f}"
        flat[f"c3.{t}.gate.agree"] = f"{100 * r['gate']['first_token_agree']:.1f}"
        for tag_, pre in (("summary", f"c3.{t}"), ("summary3", f"c3.{t}.three")):
            if tag_ not in r:
                continue
            flat[f"{pre}.verdict"] = r[tag_]["verdict"]
            flat[f"{pre}.drawmin"] = f"{100 * r[tag_]['draw_min']:+.1f}"
            flat[f"{pre}.drawmax"] = f"{100 * r[tag_]['draw_max']:+.1f}"
            for k_, v in r[tag_]["point"].items():
                flat[f"{pre}.{k_}.diff".replace("_", "")] = f"{100 * v:+.1f}"
                c_ = r[tag_]["ci95"][k_]
                flat[f"{pre}.{k_}.ci".replace("_", "")] = f"[{100 * c_[0]:+.1f}, {100 * c_[1]:+.1f}]"
        for b, inf in r.get("builds", {}).items():
            flat[f"c3.{t}.{b}.nllgap"] = f"{inf['nll_gap']:+.3f}"
    lines = ["% generated by analysis/make_numbers.py -- do not edit",
             "\\makeatletter",
             "\\newcommand{\\N}[1]{\\ifcsname qn@#1\\endcsname\\csname qn@#1\\endcsname\\else\\textbf{\\textcolor{red}{??#1}}\\fi}"]
    for k in sorted(flat):
        lines.append(f"\\expandafter\\def\\csname qn@{k}\\endcsname{{{flat[k]}}}")
    lines.append("\\makeatother")
    tex = ROOT / "paper" / "tex"
    tex.mkdir(parents=True, exist_ok=True)
    (tex / "numbers.tex").write_text("\n".join(lines) + "\n")
    (tex / "numbers_keys.txt").write_text("\n".join(f"{k} = {flat[k]}" for k in sorted(flat)) + "\n")
    return len(flat)


def main():
    for fn in (c1_primary, s6_ladder, c0_instrument, s7_seed_floor, s2_depth, c2_free, s3_sorry,
               c4_benign, c4_capability, robustness, judges_and_humans, truncation, c3_controls, s9_quantizers_runtime, m1_direction, s5_runon, appendix_tables):
        try:
            fn()
        except FileNotFoundError as e:
            NOTES.append(f"{fn.__name__}: missing input ({Path(e.filename).relative_to(ROOT) if e.filename else e})")
    out = ROOT / "paper"
    out.mkdir(exist_ok=True)
    (out / "numbers.json").write_text(json.dumps(OUT, indent=1, default=str))
    head = ["# Numbers (generated by analysis/make_numbers.py; do not edit by hand)\n"]
    if NOTES:
        head += ["**Notes:**"] + [f"- {n}" for n in NOTES] + [""]
    (out / "numbers.md").write_text("\n".join(head + MD))
    nk = latex_macros()
    print(f"wrote paper/numbers.json, paper/numbers.md ({len(MD)} lines), paper/tex/numbers.tex ({nk} macros); notes: {len(NOTES)}")
    for n in NOTES:
        print("  note:", n)


if __name__ == "__main__":
    main()
