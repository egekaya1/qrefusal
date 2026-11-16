"""Paper figures, drawn only from paper/numbers.json (run analysis/make_numbers.py first).

    python analysis/figures.py   -> paper/tex/figures/{forest,ladder,controls}.pdf

Encoding: shipped builds are blue circles, control perturbations orange squares (shape
carries the distinction too, so it survives greyscale and colour-vision deficiency).
Filled marker = a registered confirmatory test. Shaded band = the registered +/-5 pp
equivalence margin. Palette validated with the dataviz validator (blue/orange, light).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N = json.loads((ROOT / "paper" / "numbers.json").read_text())
OUT = ROOT / "paper" / "tex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#f0efec"
NICE = {"qwen2.5-3b": "Qwen2.5-3B", "llama3.2-3b": "Llama-3.2-3B", "qwen3-4b": "Qwen3-4B"}
BUILD = {"8bit-g64": "8-bit", "4bit-g32": "4-bit g32", "mixed_4_6": "mixed 4/6", "4bit-g64": "4-bit g64",
         "mxfp4": "MXFP4", "e2-mirror": "mirror", "e2-perm1": "permuted 1", "e2-perm2": "permuted 2",
         "e2-gauss": "Gaussian 1", "e2-gauss2": "Gaussian 2", "e2-gauss3": "Gaussian 3", "e2-perm3": "permuted 3",
         "e2-scale0.5": "0.5 x error", "e2-scale2": "2 x error", "e2-dense": "dense W+E",
         "awq": "AWQ", "gptq": "GPTQ", "gguf-q8_0": "GGUF Q8_0", "gguf-q4km": "GGUF Q4_K_M"}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0, "pdf.fonttype": 42})


def forest(ax, rows, title, xlim):
    """rows: list of (label, diff, ci95, kind['ship'|'ctrl'|'head'], confirmatory)."""
    ax.axvspan(-5, 5, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=INK2, lw=0.6, zorder=1)
    y, ticks, labels = 0, [], []
    for lab, d, ci, kind, conf in rows:
        if kind == "head":
            y -= 0.35
            ax.text(xlim[0] + 0.3, y, lab, fontsize=8, fontweight="bold", color=INK, va="center")
            y -= 1
            continue
        col, mk = (BLUE, "o") if kind == "ship" else (ORANGE, "s")
        ax.plot([100 * ci[0], 100 * ci[1]], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
        ax.plot(100 * d, y, marker=mk, ms=6.5, mfc=col if conf else "white", mec=col, mew=1.2, zorder=3)
        ticks.append(y)
        labels.append(lab)
        y -= 1
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y + 0.4, 0.6)
    ax.set_xlim(*xlim)
    ax.set_xlabel("shift vs bf16 (percentage points)")
    ax.set_title(title, fontsize=8.5, loc="left", color=INK)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def fig_forest():
    S6, C2, C3 = N["S6"], N["C2"]["cells"], N.get("C3", {})
    A = []
    for m in ("qwen2.5-3b", "llama3.2-3b", "qwen3-4b"):
        A.append((NICE[m], 0, None, "head", False))
        for c in ("8bit-g64", "4bit-g32", "mixed_4_6", "4bit-g64", "mxfp4"):
            r = S6[m][c]
            A.append((BUILD[c], r["diff"], r["ci95"], "ship", c == "4bit-g64"))
        if m == "qwen2.5-3b":
            for k in ("e2-mirror", "e2-perm1", "e2-perm2", "e2-gauss"):
                r = C3.get("Qwen2.5 forced", {}).get(k)
                if r:
                    A.append((BUILD[k], r["diff"], r["ci95"], "ctrl", False))
    B = []
    for m in ("llama3.2-3b", "qwen2.5-3b", "qwen3-4b"):
        B.append((NICE[m], 0, None, "head", False))
        for c in ("8bit-g64", "4bit-g64", "mxfp4"):
            r = C2.get(m, {}).get(c)
            if r:
                B.append((BUILD[c], r["diff"], r["ci95"], "ship", (m, c) == ("llama3.2-3b", "4bit-g64")))
        if m == "llama3.2-3b":
            for k in ("e2-mirror", "e2-perm1", "e2-perm2", "e2-gauss"):
                r = C3.get("Llama free", {}).get(k)
                if r:
                    B.append((BUILD[k], r["diff"], r["ci95"], "ctrl", False))
    h = max(len(A), len(B)) * 0.17 + 0.6
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, h))
    forest(a, A, "A  Forced start (8 tokens prefilled, n = 562)", (-14, 16))
    forest(b, B, "B  Free start (held-out jailbreaks, n = 400)", (-24, 12))
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", color=BLUE, mfc="white", mew=1.3, lw=1.6, label="shipped build"),
           Line2D([], [], marker="s", color=ORANGE, mfc="white", mew=1.3, lw=1.6, label="control perturbation"),
           Line2D([], [], marker="o", color=BLUE, mfc=BLUE, lw=0, label="confirmatory test")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "forest.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_ladder():
    C0, S6 = N["C0"], N["S6"]
    builds = ["8bit-g64", "4bit-g32", "mixed_4_6", "4bit-g64", "mxfp4"]
    fig, axes = plt.subplots(2, 3, figsize=(6.3, 2.9), sharey="row")
    for j, m in enumerate(("qwen2.5-3b", "llama3.2-3b", "qwen3-4b")):
        e0 = C0[m].get("e0")
        xs = ["rerun"] + [BUILD[b] for b in builds]
        txt = [0 if e0 else None] + [100 * (1 - C0[m][b]["text_identical"]) if b in C0[m] else None for b in builds]
        ch = [0 if e0 else None] + [100 * S6[m][b]["churn"] for b in builds]
        net = [0 if e0 else None] + [abs(100 * S6[m][b]["diff"]) for b in builds]
        for i, ax in enumerate(axes[:, j]):
            vals = txt if i == 0 else ch
            for k, v in enumerate(vals):
                if v is None:
                    ax.text(k, 1, "n/a", ha="center", fontsize=6, color=INK2)
                    continue
                if v == 0:
                    ax.text(k, 1.5, "0", ha="center", fontsize=6.5, color=INK2)
                ax.bar(k, v, width=0.7, color=BLUE, alpha=0.35 if i == 1 else 1, lw=0)
                if i == 1:
                    ax.bar(k, net[k], width=0.7, color=BLUE, lw=0)
            ax.set_xticks(range(len(xs)), xs if i == 1 else [""] * len(xs), rotation=45, ha="right", fontsize=6.5)
            ax.grid(axis="y", color=GRID, lw=0.5)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0, j].set_title(NICE[m], fontsize=8.5, color=INK)
    axes[0, 0].set_ylabel("replies differing\nfrom bf16 (%)")
    axes[1, 0].set_ylabel("label churn (%)\n(dark: |net shift|)")
    fig.tight_layout()
    fig.savefig(OUT / "ladder.pdf", bbox_inches="tight")
    plt.close(fig)


DOSE = [(-1.0, "e2-mirror"), (0.5, "e2-scale0.5"), (1.0, "real"), (2.0, "e2-scale2")]
CELLCOL = {"Qwen2.5 forced": BLUE, "Llama free": ORANGE}
CELLMK = {"Qwen2.5 forced": "o", "Llama free": "s"}


def dose_panel(ax, C3):
    """Shift against the multiple of the real rounding error, W + aE."""
    ax.axhline(0, color=INK2, lw=0.6, zorder=1)
    ax.axvline(0, color=GRID, lw=0.6, zorder=1)
    for tag, r in C3.items():
        pts = [(a, r[k]) for a, k in DOSE if k in r]
        if len(pts) < 3:
            continue
        col, mk = CELLCOL.get(tag, INK2), CELLMK.get(tag, "o")
        xs = [0.0] + [a for a, _ in pts]
        ys = [0.0] + [100 * v["diff"] for _, v in pts]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs, ys = [xs[i] for i in order], [ys[i] for i in order]
        ax.plot(xs, ys, color=col, lw=1.1, zorder=2, alpha=0.75)
        for a, v in pts:
            lo, hi = 100 * v["ci95"][0], 100 * v["ci95"][1]
            ax.plot([a, a], [lo, hi], color=col, lw=1.4, solid_capstyle="round", zorder=3)
            ax.plot(a, 100 * v["diff"], marker=mk, ms=5.5, mfc=col, mec=col, mew=1.1, zorder=4)
        ax.plot(0, 0, marker=mk, ms=5.5, mfc="white", mec=col, mew=1.1, zorder=4)
        sl = r.get("dose_slope")
        if sl:
            lab = f"{NICE2.get(tag, tag)}  {100 * sl['slope']:+.1f} pp".replace("-", "\u2212")
            ax.plot([], [], color=col, marker=mk, ms=5, lw=1.1, label=lab)
    ax.set_xticks([-1, 0, 0.5, 1, 2], ["$-E$", "bf16", r"$\frac{1}{2}E$", "$E$", "$2E$"])
    ax.set_xlabel("weights at $W + aE$")
    ax.set_ylabel("shift vs bf16 (pp)")
    ax.legend(frameon=False, fontsize=6.8, loc="lower left", handlelength=1.6,
              title="slope per unit of error", title_fontsize=6.8, alignment="left")
    ax.grid(color=GRID, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


NICE2 = {"Qwen2.5 forced": "Qwen2.5 forced", "Llama free": "Llama free"}
RND = ("e2-perm1", "e2-perm2", "e2-perm3", "e2-gauss", "e2-gauss2", "e2-gauss3")


def fig_dose():
    C3 = N.get("C3", {})
    if not any("dose_slope" in r for r in C3.values()):
        return False
    fig, ax = plt.subplots(figsize=(3.05, 2.05))
    dose_panel(ax, C3)
    fig.tight_layout()
    fig.savefig(OUT / "dose.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


def fig_controls():
    """Appendix: every control build per cell, beside other quantizers' real errors."""
    C3 = N.get("C3", {})
    S9 = N.get("S9", {})
    cells = [(t, r) for t, r in C3.items() if "summary" in r or "summary3" in r]
    if not cells:
        return False
    fig, axes = plt.subplots(1, len(cells), figsize=(3.15 * len(cells), 2.6), squeeze=False)
    for ax, (tag, r), letter in zip(axes[0], cells, "AB"):
        s = r.get("summary") or r["summary3"]
        rows = [("real 4-bit", r["real"]["diff"], r["real"]["ci95"], "ship", True),
                ("mirror $W-E$", r["e2-mirror"]["diff"], r["e2-mirror"]["ci95"], "ctrl", False)]
        rows += [(BUILD.get(k, k), r[k]["diff"], r[k]["ci95"], "ctrl", False) for k in RND if k in r]
        rows.append(("random mean", s["point"]["random"], s["ci95"]["random"], "ctrl", True))
        for b in ("awq", "gguf-q4km"):
            v = S9.get(tag, {}).get(b)
            if v and "ci95" in v:
                rows.append((BUILD.get(b, b), v["diff"], v["ci95"], "ship", False))
        lo = min(100 * x[2][0] for x in rows) - 2
        hi = max(100 * x[2][1] for x in rows) + 2
        forest(ax, rows, f"{letter}  {tag}: {s['verdict']}", (min(lo, -6), max(hi, 6)))
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], marker="o", color=BLUE, mfc="white", mew=1.3, lw=1.6, label="real quantized build"),
           Line2D([], [], marker="s", color=ORANGE, mfc="white", mew=1.3, lw=1.6, label="matched perturbation"),
           Line2D([], [], marker="o", color=BLUE, mfc=BLUE, lw=0, label="registered test")]
    fig.legend(handles=leg, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "controls.pdf", bbox_inches="tight")
    plt.close(fig)
    return True


if __name__ == "__main__":
    fig_forest()
    fig_ladder()
    print("forest.pdf, ladder.pdf written;", "dose.pdf written;" if fig_dose() else "dose pending;",
          "controls.pdf written" if fig_controls() else "controls pending")
