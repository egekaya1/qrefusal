"""L2b: follow-up labelling sheet (H2).

Two jobs in one blind sheet, shuffled together so the annotator cannot tell them apart:

  new      100 further concordant pairs (strata P and N), which the pre-registration's
           judge-agreement rule asks for. The H1 sample is enriched for pairs the judge
           splits, so agreement estimated on it alone is pessimistic by construction.
  repeat   every H1 pair in which the annotator called a reply harmful where the judge
           did not, shown again with the A/B order re-randomised. This is an
           intra-annotator consistency check on the one cell that drives the
           human-label effect size, and the cell an independent third labelling
           found least settled.

Usage: python runs/l2b_followup.py
Score: python runs/l2b_followup.py --score results/labels/h2/labels_h2.json
"""

from __future__ import annotations

import argparse
import html
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runs.l2_pair_sheet import A_CFG, B_CFG, HARM, LABELS, MODELS, PAGE, rows  # noqa: E402


def build(batch="h2", source="a2/k8", n_new=100, n_repeat=60, seed=7):
    rng = random.Random(seed)
    h1 = json.loads((ROOT / "results/labels/h1/key.json").read_text())
    h1_items, h1_sizes = h1["items"], h1["stratum_sizes"]
    h1_lab = json.loads((ROOT / "results/labels/h1/labels_h1.json").read_text())

    per_model_new = n_new // (2 * len(MODELS))  # per model, per concordant stratum
    picked, sizes = [], {}
    for m in MODELS:
        base = ROOT / "results" / source / m
        gen = {c: rows(base / f"{c}.jsonl") for c in (A_CFG, B_CFG)}
        jud = {c: {i: r["label"] in HARM for i, r in rows(base / f"judge_v2_{c}.jsonl").items()}
               for c in (A_CFG, B_CFG)}
        ids = sorted(set(jud[A_CFG]) & set(jud[B_CFG]))
        used = {it["i"] for it in h1_items.values() if it["model"] == m}
        groups = {"P": [i for i in ids if jud[A_CFG][i] and jud[B_CFG][i] and i not in used],
                  "N": [i for i in ids if not jud[A_CFG][i] and not jud[B_CFG][i] and i not in used]}
        for s, members in groups.items():
            sizes[f"{m}|{s}"] = h1_sizes[f"{m}|{s}"]
            for i in rng.sample(members, min(per_model_new, len(members))):
                swap = rng.random() < 0.5
                ca, cb = (B_CFG, A_CFG) if swap else (A_CFG, B_CFG)
                picked.append({"kind": "new", "model": m, "stratum": s, "i": i, "A_cfg": ca, "B_cfg": cb,
                               "request": gen[A_CFG][i]["request"], "prefix": gen[A_CFG][i]["prefix"],
                               "A": gen[ca][i]["reply"], "B": gen[cb][i]["reply"],
                               "judge_A": jud[ca][i], "judge_B": jud[cb][i], "h1_id": None})
        # repeats: pairs where the annotator said harmful and the judge did not
        for hid, it in h1_items.items():
            if it["model"] != m or hid not in h1_lab or "A" not in h1_lab[hid]:
                continue
            if not any(h1_lab[hid][sd] in HARM and not it[f"judge_{sd}"] for sd in ("A", "B")):
                continue
            swap = rng.random() < 0.5
            ca, cb = (it["B_cfg"], it["A_cfg"]) if swap else (it["A_cfg"], it["B_cfg"])
            picked.append({"kind": "repeat", "model": m, "stratum": it["stratum"], "i": it["i"],
                           "A_cfg": ca, "B_cfg": cb,
                           "request": gen[A_CFG][it["i"]]["request"], "prefix": gen[A_CFG][it["i"]]["prefix"],
                           "A": gen[ca][it["i"]]["reply"], "B": gen[cb][it["i"]]["reply"],
                           "judge_A": jud[ca][it["i"]], "judge_B": jud[cb][it["i"]], "h1_id": hid})
    # Cap the repeats, keeping each model's share, so the sheet stays an hour's work.
    reps = [x for x in picked if x["kind"] == "repeat"]
    rng.shuffle(reps)
    keep = {id(x) for x in reps[:n_repeat]}
    picked = [x for x in picked if x["kind"] == "new" or id(x) in keep]
    rng.shuffle(picked)
    out = ROOT / "results" / "labels" / batch
    out.mkdir(parents=True, exist_ok=True)
    data = [{"id": f"{batch}-{j:03d}", "request": x["request"], "prefix": x["prefix"], "A": x["A"], "B": x["B"]}
            for j, x in enumerate(picked)]
    key = {"items": {f"{batch}-{j:03d}": {k: x[k] for k in
                                          ("kind", "model", "stratum", "i", "A_cfg", "B_cfg",
                                           "judge_A", "judge_B", "h1_id")}
                     for j, x in enumerate(picked)},
           "stratum_sizes": sizes, "source": source, "contrast": [A_CFG, B_CFG]}
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (out / "sheet.html").write_text(PAGE.replace("BATCH", html.escape(batch))
                                    .replace("DATA", json.dumps(data).replace("</", "<\\/"))
                                    .replace("LABELSJS", json.dumps(list(LABELS))))
    print(f"{len(picked)} pairs ({2 * len(picked)} replies) -> {out / 'sheet.html'}")
    print(" kinds:", dict(Counter(x["kind"] for x in picked)))
    print(" per stratum:", dict(Counter(f"{x['model']}|{x['stratum']}|{x['kind']}" for x in picked)))


def score(path):
    lab = json.loads(Path(path).read_text())
    key = json.loads((Path(path).parent / "key.json").read_text())
    items = key["items"]
    h1_lab = json.loads((ROOT / "results/labels/h1/labels_h1.json").read_text())
    h1_items = json.loads((ROOT / "results/labels/h1/key.json").read_text())["items"]
    done = {k: v for k, v in lab.items() if "A" in v and "B" in v}
    print(f"{len(done)} pairs fully labelled")

    agree = n = 0
    for k in done:
        it = items[k]
        if it["kind"] != "new":
            continue
        for sd in ("A", "B"):
            agree += (lab[k][sd] in HARM) == it[f"judge_{sd}"]
            n += 1
    if n:
        print(f"new concordant pairs: judge-human reply agreement {agree / n:.1%} ({agree}/{n})")

    same = tot = 0
    flip = defaultdict(int)
    for k in done:
        it = items[k]
        if it["kind"] != "repeat" or it["h1_id"] not in h1_lab:
            continue
        old, h1it = h1_lab[it["h1_id"]], h1_items[it["h1_id"]]
        by_cfg_old = {h1it["A_cfg"]: old["A"] in HARM, h1it["B_cfg"]: old["B"] in HARM}
        for sd in ("A", "B"):
            now = lab[k][sd] in HARM
            then = by_cfg_old[it[f"{sd}_cfg"]]
            same += now == then
            tot += 1
            if now != then:
                flip["harmful -> not" if then else "not -> harmful"] += 1
    if tot:
        print(f"repeats: self-consistency {same / tot:.1%} ({same}/{tot}); changed: {dict(flip)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", default=None)
    ap.add_argument("--n_new", type=int, default=100)
    ap.add_argument("--n_repeat", type=int, default=60)
    a = ap.parse_args()
    score(a.score) if a.score else build(n_new=a.n_new, n_repeat=a.n_repeat)
