"""Figures for the walk-forward study and the backtest.

    cd python
    python make_walkforward_figures.py --result ../data/wf_5bp.json

Reads the JSON `walkforward.py --out` writes, so every number plotted is the one
that run produced. Writes PNGs to `assets/`.

Three questions, in order:
  1. Which of the two predictions survives a month it was not trained on?
  2. What barrier would the measured hit rate need in order to pay for itself?
  3. What does the strategy actually earn, net of costs?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets"

# Same palette as make_figures.py, so the two sets read as one document.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, PLUM = "#2a78d6", "#eb6834", "#1baf7a", "#8b5fbf"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.labelcolor": INK_2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "grid.color": GRID, "grid.linewidth": 0.8, "figure.dpi": 160,
})


def _clean(ax, axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis=axis, linestyle="-", alpha=0.9)
    ax.grid(axis="x" if axis == "y" else "y", visible=False)
    ax.tick_params(length=0)


def figure_walkforward(data: dict, path: Path) -> None:
    """Gate vs side, fold by fold. One transfers across months; one does not."""
    folds = data["folds"]
    names = [f["fold"]["name"].split("-> ")[-1] for f in folds]
    x = np.arange(len(folds))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.1))

    # -- left: the gate --
    series = [
        ("XGBoost, 7 features", BLUE, [f["gate"]["trained_auc"] for f in folds]),
        ("Realized variance, untrained", ORANGE, [f["gate"]["rv_auc"] for f in folds]),
    ]
    width = 0.36
    for i, (label, color, vals) in enumerate(series):
        pos = x + (i - 0.5) * width
        ax1.bar(pos, vals, width=width - 0.03, color=color, label=label)
        for xi, v in zip(pos, vals):
            ax1.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
    ax1.axhline(0.5, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax1.text(-0.45, 0.508, "coin flip", color=INK_2, fontsize=8.5)
    ax1.set_xticks(x, names, fontsize=9.5, color=INK)
    ax1.set_ylim(0.45, 0.88)
    ax1.set_ylabel("Test ROC-AUC", fontsize=9)
    _clean(ax1)
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper left",
               bbox_to_anchor=(0, 1.0))
    ax1.set_title("Gate: will any barrier be touched?", fontsize=10.5, color=INK,
                  pad=10, loc="left", fontweight="bold")

    # -- right: the side --
    models = list(folds[0]["side_auc"])
    colors = dict(zip(models, (AQUA, PLUM, BLUE, ORANGE)))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        vals = [f["side_auc"][model] for f in folds]
        pos = x + (i - (len(models) - 1) / 2) * width
        ax2.bar(pos, vals, width=width - 0.02, color=colors[model], label=model)
    ax2.axhline(0.5, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax2.set_xticks(x, names, fontsize=9.5, color=INK)
    ax2.set_ylim(0.45, 0.88)
    ax2.set_ylabel("Test ROC-AUC", fontsize=9)
    _clean(ax2)
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="upper left", ncol=1)
    ax2.set_title("Side: which barrier first?", fontsize=10.5, color=INK,
                  pad=10, loc="left", fontweight="bold")

    fig.suptitle(
        "Volatility forecasts survive a month they were not trained on. Direction does not.",
        fontsize=11.5, color=INK, x=0.008, ha="left", y=1.03, fontweight="bold",
    )
    fig.text(0.008, -0.02, "Each fold trains only on months before its test month "
             f"({data['config']['scheme']}, {data['config']['train_months']} training months).",
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_economics(data: dict, path: Path) -> None:
    """The barrier a given hit rate needs before it pays for its own costs."""
    cost = data["config"]["breakeven_barrier_bps"]
    barrier = data["config"]["barrier_bps"]

    hits = np.linspace(0.505, 0.75, 400)
    required = cost / (2 * hits - 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(hits * 100, required, color=INK, linewidth=2)
    ax.fill_between(hits * 100, required, 1e4, color=AQUA, alpha=0.12)
    ax.fill_between(hits * 100, 0, required, color=ORANGE, alpha=0.12)

    ax.axhline(barrier, color=ORANGE, linewidth=1.4, linestyle=(0, (4, 3)))
    ax.text(74.5, barrier * 1.12, f"the barrier actually used: {barrier:.0f} bp",
            fontsize=8.5, color=ORANGE, ha="right")

    # `resolved_hit_rate`, not `hit_rate`: the break-even identity assumes every
    # trade ends at +B or -B, so the denominator has to exclude positions closed
    # by the clock. Mixing timeouts in drives h below 0.5 and the "required
    # barrier" negative, which is arithmetic nonsense rather than a result.
    measured = [(r.get("resolved_hit_rate", float("nan")) * 100, r["model"])
                for r in data["pooled"] if r["gate"] == "none" and r["n_trades"] > 100]
    measured = [m for m in measured if np.isfinite(m[0])]
    lo, hi = ax.get_xlim()
    for hit, _ in measured:
        if lo <= hit <= hi:
            ax.plot([hit], [barrier], marker="o", markersize=7, color=BLUE,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5)
    if measured:
        best = max(h for h, _ in measured)
        required = cost / (2 * best / 100 - 1) if best > 50 else float("inf")
        if lo <= best <= hi:
            ax.annotate(f"measured: {best:.1f}%", xy=(best, barrier),
                        xytext=(best + 2.5, barrier * 4.0), fontsize=9, color=INK,
                        arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))
        note = (f"Expected value per trade is B(2h-1) - c. At c = {cost:.1f} bp, the measured "
                f"{best:.1f}% hit rate needs a {required:,.0f} bp barrier."
                if np.isfinite(required) else
                f"Expected value per trade is B(2h-1) - c. The measured hit rate of "
                f"{best:.1f}% is at or below a coin flip, so no barrier makes it profitable.")
    else:
        note = f"Expected value per trade is B(2h-1) - c, with c = {cost:.1f} bp."

    ax.set_yscale("log")
    ax.set_xlim(50.5, 75)
    ax.set_ylim(1, 3000)
    ax.set_xlabel("Hit rate: how often the predicted barrier is touched first (%)", fontsize=9)
    ax.set_ylabel("Barrier needed to break even (bp, log scale)", fontsize=9)
    ax.set_yticks([1, 3, 10, 30, 100, 300, 1000, 3000],
                  ["1", "3", "10", "30", "100", "300", "1000", "3000"])
    _clean(ax)
    ax.text(63, 700, "profitable", fontsize=10, color=AQUA, fontweight="bold")
    ax.text(52.5, 2.2, "unprofitable", fontsize=10, color=ORANGE, fontweight="bold")
    ax.set_title(
        f"A {cost:.1f} bp round trip sets the price of being right",
        fontsize=11.5, color=INK, pad=12, loc="left", fontweight="bold",
    )
    fig.text(0.008, -0.02, note, fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_backtest(data: dict, path: Path) -> None:
    """Gross vs net per trade. The point is that gross is zero, not that net is bad.

    An earlier version of this figure plotted net alone and produced a dozen
    identical -6 bp bars, which says only that the cost is the cost. Splitting
    gross out is what shows where the money went: nowhere, because none was made.
    """
    rows = [r for r in data["pooled"] if r["n_trades"] > 100]
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    cost = data["config"]["breakeven_barrier_bps"]

    # One row per model: whichever gate configuration did best on net.
    best = {m: max((r for r in rows if r["model"] == m), key=lambda r: r["net_bps"])
            for m in models}

    # Two panels, because the two quantities differ by three orders of magnitude:
    # on a scale that shows -6 bp, a gross of +0.006 bp is a sub-pixel bar and
    # reads as missing data rather than as the finding.
    labels = [m.replace("Logistic Regression", "Logistic") for m in models]
    x = np.arange(len(models))
    gross = [best[m]["gross_bps"] for m in models]
    net = [best[m]["net_bps"] for m in models]
    span = max(0.05, max(abs(g) for g in gross) * 1.8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    ax1.bar(x, gross, width=0.5, color=AQUA)
    for xi, v in zip(x, gross):
        ax1.text(xi, v + (span * 0.06 if v >= 0 else -span * 0.06), f"{v:+.3f}",
                 ha="center", va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
    ax1.axhline(0, color=INK, linewidth=1.2)
    ax1.set_ylim(-span, span)
    ax1.set_xticks(x, labels, fontsize=9)
    ax1.set_ylabel("Basis points per trade", fontsize=9)
    _clean(ax1)
    ax1.set_title("Gross, before any cost", fontsize=10.5, color=INK,
                  pad=10, loc="left", fontweight="bold")

    ax2.bar(x, net, width=0.5, color=ORANGE)
    for xi, v in zip(x, net):
        ax2.text(xi, v - cost * 0.045, f"{v:.2f}", ha="center", va="top",
                 fontsize=9, color=INK)
    ax2.axhline(0, color=INK, linewidth=1.2)
    # The round-trip line is labelled in the panel title rather than annotated on
    # the axes: three bars leave no clear space for a caption at that height.
    ax2.axhline(-cost, color=MUTED, linewidth=1.1, linestyle=(0, (4, 3)))
    ax2.set_ylim(-cost * 1.3, cost * 0.12)
    ax2.set_xticks(x, labels, fontsize=9)
    _clean(ax2)
    ax2.set_title(f"Net, after the {cost:.1f} bp round trip (dashed)", fontsize=10.5,
                  color=INK, pad=10, loc="left", fontweight="bold")

    fig.suptitle("Gross P&L is zero, so net P&L is exactly minus the cost",
                 fontsize=11.5, color=INK, x=0.008, ha="left", y=1.03, fontweight="bold")
    fig.text(0.008, -0.02,
             "Pooled out-of-sample months, one position at a time, best gate "
             "configuration per model. Trades: "
             + ", ".join(f"{label} {best[m]['n_trades']:,}" for label, m in zip(labels, models)),
             fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--result", required=True, help="JSON from walkforward.py --out")
    parser.add_argument("--out", default=str(ASSETS))
    args = parser.parse_args()

    data = json.loads(Path(args.result).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, fn in (
        ("walkforward_auc.png", figure_walkforward),
        ("economics.png", figure_economics),
        ("backtest.png", figure_backtest),
    ):
        fn(data, out / name)
        print(f"  wrote {out / name}")


if __name__ == "__main__":
    main()
