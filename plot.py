"""Draws the transformation grids, the shape diagrams and the matmul speed chart from results.json."""

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon

R = json.load(open("results.json"))
BLUE, RED, GRAY = "#2563eb", "#dc2626", "#9ca3af"


def transforms() -> None:
    fig, axes = plt.subplots(1, len(R["transforms"]), figsize=(14, 3.9))
    lines = np.linspace(-2, 2, 9)
    t = np.linspace(-2, 2, 50)
    for ax, tr in zip(axes, R["transforms"]):
        M = np.array(tr["matrix"])
        for c in lines:  # transformed grid lines
            for pts in (np.stack([np.full_like(t, c), t]), np.stack([t, np.full_like(t, c)])):
                p = M @ pts
                ax.plot(p[0], p[1], color=GRAY, lw=0.6)
        sq = M @ np.array([[0, 1, 1, 0], [0, 0, 1, 1]])
        ax.add_patch(Polygon(sq.T, closed=True, fc="#fde68a", ec="#b45309", alpha=0.8))
        for vec, col, lab in ((M[:, 0], BLUE, "î"), (M[:, 1], RED, "ĵ")):
            ax.annotate("", xy=vec, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=col, lw=2))
            ax.text(*(vec * 1.12), lab, color=col, fontsize=12, ha="center", va="center")
        ax.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal", xticks=[], yticks=[])
        ax.set_title(f"{tr['name']}\ndet = {tr['det']:g}, rank = {tr['rank']}", fontsize=10)
    fig.suptitle("A matrix moves the basis vectors î, ĵ; the unit square's area scales by det", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig("assets/transformations.png", dpi=150)


def box(ax, x, y, w, h, label, sub, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#374151", lw=1.2))
    ax.text(x + w / 2, y + h / 2 + 0.12, label, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.2, sub, ha="center", va="center", fontsize=9, color="#374151")


def batched_matmul() -> None:
    d = R["dims"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set(xlim=(0, 11), ylim=(0, 3.4))
    ax.axis("off")
    box(ax, 0.3, 0.8, 2.6, 1.8, "x", f"(B, T, C) = ({d['B']}, {d['T']}, {d['C']})", "#dbeafe")
    ax.text(3.25, 1.7, "@", fontsize=22, ha="center", va="center")
    box(ax, 3.6, 0.8, 2.2, 1.8, "W", f"(C, H) = ({d['C']}, {d['H']})", "#fee2e2")
    ax.text(6.15, 1.7, "=", fontsize=22, ha="center", va="center")
    box(ax, 6.5, 0.8, 2.6, 1.8, "y", f"(B, T, H) = ({d['B']}, {d['T']}, {d['H']})", "#dcfce7")
    ax.text(9.3, 2.3, "Rule:", fontsize=10, fontweight="bold")
    ax.text(9.3, 1.85, "inner dims must match\n(C = C), they disappear;", fontsize=9, va="top")
    ax.text(9.3, 1.1, "leading dims (B, T)\nare carried along.", fontsize=9, va="top")
    ax.annotate("", xy=(4.0, 0.55), xytext=(2.6, 0.55), arrowprops=dict(arrowstyle="<->", color=RED))
    ax.text(3.3, 0.3, "C must match", color=RED, ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig("assets/batched_matmul.png", dpi=150)


def attention() -> None:
    steps = R["attention_shapes"]
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.set(xlim=(0, 12.4), ylim=(0, 3.2))
    ax.axis("off")
    order = [0, 1, 4, 5, 6]  # x, q, scores, weights, out (k and v shown as side notes)
    xs = np.linspace(0.2, 10.2, len(order))
    colors = ["#dbeafe", "#e0e7ff", "#fef3c7", "#fde68a", "#dcfce7"]
    for i, (idx, x) in enumerate(zip(order, xs)):
        s = steps[idx]
        box(ax, x, 1.0, 2.0, 1.3, s["step"], s["shape"], colors[i])
        ax.text(x + 1.0, 0.7, s["meaning"], ha="center", fontsize=8, color="#374151", wrap=True)
        if i:
            ax.annotate("", xy=(x - 0.02, 1.65), xytext=(xs[i - 1] + 2.02, 1.65), arrowprops=dict(arrowstyle="-|>", color=GRAY))
    ax.text(xs[1] + 1.0, 2.65, f"k, v: also {steps[2]['shape']}", ha="center", fontsize=9, color="#4b5563")
    ax.text(xs[2] + 1.0, 2.65, "(T × T): cost grows with T²", ha="center", fontsize=9, color=RED)
    fig.suptitle("Shapes through one causal self-attention head", fontsize=11)
    fig.tight_layout()
    fig.savefig("assets/attention_shapes.png", dpi=150)


def speed() -> None:
    rows = R["matmul_benchmark"]
    n = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.semilogy(n, [r["loops_ms"] for r in rows], "o-", label="pure Python triple loop")
    ax.semilogy(n, [r["numpy_ms"] for r in rows], "s-", label="NumPy  A @ B  (BLAS)")
    for r in rows:
        ax.annotate(f"{r['speedup']:.0f}×", (r["n"], (r["loops_ms"] * r["numpy_ms"]) ** 0.5), ha="center", fontsize=9, color="#4b5563")
    ax.set(title="Same result, very different speed (n×n matmul)", xlabel="n", ylabel="time [ms] (log)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig("assets/matmul_loops_vs_numpy.png", dpi=150)


if __name__ == "__main__":
    transforms()
    batched_matmul()
    attention()
    speed()
