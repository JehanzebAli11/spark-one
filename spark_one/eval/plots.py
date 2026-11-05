"""Static figures for eval reports (matplotlib, Agg)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax, title):
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def reliability(panels: list[tuple[str, list[dict], float]], path, suptitle: str):
    """panels: (title, bins, ece). Bars = accuracy per confidence bin; diagonal = perfect."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.4), squeeze=False)
    for ax, (title, bins, e) in zip(axes[0], panels):
        _style(ax, f"{title}\nECE {e:.3f}")
        ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle="--", label="perfect")
        xs = [(b["lo"] + b["hi"]) / 2 for b in bins if b["n"]]
        accs = [b["acc"] for b in bins if b["n"]]
        ax.bar(xs, accs, width=0.09, color=SERIES[0], label="accuracy", edgecolor=SURFACE, linewidth=1)
        confs = [b["conf"] for b in bins if b["n"]]
        ax.plot(xs, confs, "o", color=SERIES[1], markersize=4, label="mean confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("confidence (top-1 probability)", color=INK2, fontsize=8)
    axes[0][0].set_ylabel("accuracy", color=INK2, fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle(suptitle, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def coverage(curves: dict[str, list[dict]], path, title: str):
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    _style(ax, title)
    for i, (name, pts) in enumerate(curves.items()):
        if not pts:
            continue
        ax.plot([p["coverage"] for p in pts], [p["accuracy"] for p in pts], color=SERIES[i % 8],
                linewidth=2, label=name)
        ax.annotate(name, (pts[-1]["coverage"], pts[-1]["accuracy"]), fontsize=7, color=INK2,
                    xytext=(4, 0), textcoords="offset points", va="center")
    ax.set_xlabel("coverage (fraction answered, most confident first)", color=INK2, fontsize=8)
    ax.set_ylabel("accuracy on answered", color=INK2, fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def slice_bars(rows: list[tuple[str, float, float]], path, title: str):
    """rows: (slice name, accuracy, ece) -> two side-by-side panels on their own axes."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2))
    names = [r[0] for r in rows]
    for ax, k, lab in ((axes[0], 1, "accuracy"), (axes[1], 2, "ECE (lower is better)")):
        _style(ax, lab)
        vals = [r[k] for r in rows]
        ax.barh(names, vals, color=SERIES[0] if k == 1 else SERIES[1], height=0.6)
        for y, v in enumerate(vals):
            ax.text(v, y, f" {v:.3f}", va="center", fontsize=7, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(0, max(1.0 if k == 1 else 0.0, max(vals) * 1.25 if vals else 1))
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def training_curves(logs: dict[str, list[dict]], path, title: str):
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    _style(ax, title)
    for i, (name, rows) in enumerate(logs.items()):
        ax.plot([r["step"] for r in rows], [r["ce"] for r in rows], color=SERIES[i % 8], linewidth=2,
                label=f"{name} (ground-truth log loss)")
    ax.set_xlabel("step", color=INK2, fontsize=8)
    ax.set_ylabel("log loss on answer tokens", color=INK2, fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
