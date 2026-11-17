"""Export every judge and human label without any reply text.

Writes labels/judge_labels.csv, labels/llamaguard_labels.csv and labels/human_labels.csv.
Rows are keyed by setting, model, build and prompt index `i` (the index into the prompt
split that runs/ uses). No request, prefix or reply text is written.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "labels"
SKIP = ("_superseded", "backup", "_old")


def judge_files(prefix):
    for f in sorted(R.rglob(f"{prefix}*.jsonl")):
        rel = f.relative_to(R)
        if any(s in str(rel) for s in SKIP):
            continue
        setting, model = "/".join(rel.parts[:-2]), rel.parts[-2]
        tag, build = f.stem.split("_", 2)[1], f.stem.split("_", 2)[2]
        yield setting, model, tag, build, f


def main():
    OUT.mkdir(exist_ok=True)
    n = 0
    with open(OUT / "judge_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "model", "rubric", "build", "i", "label", "parsed"])
        for setting, model, tag, build, f in (*judge_files("judge_v"), *judge_files("judge_xstest_")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                w.writerow([setting, model, tag, build, r["i"], r["label"], r.get("parsed")])
                n += 1
    m = 0
    with open(OUT / "llamaguard_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["setting", "model", "build", "i", "unsafe"])
        for setting, model, _, build, f in judge_files("judge_lg_"):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                w.writerow([setting, model, build, r["i"], r["harmful"]])
                m += 1
    key = json.loads((R / "labels/h1/key.json").read_text())["items"]
    lab = json.loads((R / "labels/h1/labels_h1.json").read_text())
    with open(OUT / "human_labels.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pair", "model", "stratum", "i", "build_A", "build_B", "human_A", "human_B", "judge_harmful_A", "judge_harmful_B"])
        for k in sorted(lab):
            it, v = key[k], lab[k]
            if "A" in v and "B" in v:
                w.writerow([k, it["model"], it["stratum"], it["i"], it["A_cfg"], it["B_cfg"], v["A"], v["B"], it["judge_A"], it["judge_B"]])
    print(f"judge rows {n}, Llama Guard rows {m}, human pairs {len(lab)}")


if __name__ == "__main__":
    main()
