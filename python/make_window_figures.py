"""
The lookback comparison: one figure across the short, mid and long window runs.

    cd python
    python make_window_figures.py --results ../data/wf_short.json \\
                                  ../data/wf_mid.json ../data/wf_long.json

Reads the JSON `walkforward.py --out` writes, one file per lookback, and plots
them against each other. Every number drawn is one those runs produced; nothing
here recomputes a model.

The question the figure exists to answer is whether five minutes of history was
the binding constraint. Three panels, in the order the argument runs:

  1. Does knowing *whether* a move is coming improve with a longer lookback?
  2. Does knowing *which way* improve with it?
  3. Does any of that reach the P&L?

Encoding note: the three lookbacks are an **ordered** quantity, not three
unrelated categories, so they are drawn as one hue getting darker rather than as
three colours. Longer lookback, darker bar.
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

# Same surface and ink as make_figures.py / make_walkforward_figures.py, so the
# four figures read as one document.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"

# The lookback ramp: one hue, light to dark, ordered by how much history the model
# reads. Checked for colour-vision separation (worst adjacent pair ΔE 17.2 under
# protanopia, 17.7 with normal vision) rather than eyeballed. The lightest step
# sits at 2.2:1 against the surface, below the 3:1 bar, so every bar in this
# figure carries a printed value and the README repeats the numbers in a table —
# the figure is never the only place a reader can get them.
SCALE_COLORS = {"short": "#79b0e8", "mid": "#2a78d6", "long": "#17497f"}
SCALE_ORDER = ("short", "mid", "long")
SCALE_LABELS = {"short": "5 minutes", "mid": "2 hours", "long": "24 hours"}

# Shortened for an axis; the full names are what walkforward.py emits.
MODEL_LABELS = {
    "Logistic Regression (OFI)": "Logistic\n(OFI)",
    "Logistic Regression (7f)": "Logistic\n(7 feat.)",
    "XGBoost": "XGBoost",
    "PatchTST": "PatchTST",
}
MODEL_ORDER = tuple(MODEL_LABELS)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "figure.dpi": 160,
    }
)


def _clean(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="-", alpha=0.9)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def load_runs(paths: list[str]) -> dict[str, dict]:
    """`{scale: result}`, keyed by the scale each run recorded in its own config.

    Keyed off the file's contents rather than its name, so a mislabelled path
    cannot silently plot a 2-hour run in the 5-minute slot.
    """
    runs: dict[str, dict] = {}
    for path in paths:
        data = json.loads(Path(path).read_text())
        config = data["config"]
        scale = config.get("window_scale") or f"{config['window']}s"
        if scale in runs:
            raise SystemExit(f"Two results claim the {scale} lookback; one of them is {path}.")
        runs[scale] = data
    unknown = [s for s in runs if s not in SCALE_ORDER]
    if unknown:
        raise SystemExit(
            f"Unknown lookback scale(s) {unknown}; this figure draws "
            f"{list(SCALE_ORDER)}. Re-run walkforward.py with --window-scale."
        )
    return {s: runs[s] for s in SCALE_ORDER if s in runs}


def _pooled(run: dict, model: str, gate: str, quantile: float) -> dict | None:
    for row in run["pooled"]:
        if (row["model"], row["gate"]) == (model, gate) and np.isclose(
            row["gate_quantile"], quantile
        ):
            return row
    return None


def _grouped_bars(ax, categories, series, value_of, fmt, missing_note="not trained"):
    """Grouped bars with one group per category and one bar per lookback.

    `value_of(scale, category)` returns a float or None; None leaves the slot
    visibly empty rather than closing the gap, because a model that has not been
    trained at a lookback yet is information, not an absence.
    """
    x = np.arange(len(categories), dtype=float)
    width = 0.82 / max(len(series), 1)
    for i, scale in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        for j, category in enumerate(categories):
            value = value_of(scale, category)
            if value is None:
                # Axes fraction for y, data coordinates for x. Drawn at a data
                # y of 0 instead, a panel whose axis starts at 0.47 would place
                # this marker thousands of pixels below the plot and drag the
                # saved bounding box down with it.
                ax.text(
                    x[j] + offset,
                    0.04,
                    missing_note,
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=6,
                    color=MUTED,
                )
                continue
            # 2px of surface between neighbouring fills, so the group reads as
            # three marks rather than one striped block.
            ax.bar(
                x[j] + offset,
                value,
                width=width * 0.9,
                color=SCALE_COLORS[scale],
                label=SCALE_LABELS[scale] if j == 0 else None,
                zorder=3,
            )
            ax.annotate(
                fmt(value),
                (x[j] + offset, value),
                textcoords="offset points",
                xytext=(0, 2 if value >= 0 else -9),
                ha="center",
                fontsize=6.6,
                color=INK_2,
            )
    ax.set_xticks(x)
    return x


def figure_windows(runs: dict[str, dict], path: Path, gate: str, quantile: float) -> None:
    scales = list(runs)
    months = [f["fold"]["name"].split("-> ")[-1] for f in next(iter(runs.values()))["folds"]]

    # Margins are set explicitly rather than by `tight_layout` or `constrained`:
    # the caption is four lines of figure-level text below the axes, which both
    # automatic layouts either ignore or collapse the axes trying to fit.
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13.2, 5.3))
    fig.subplots_adjust(left=0.055, right=0.995, top=0.80, bottom=0.30, wspace=0.24)

    # -- 1. the gate: will any barrier be reached inside the hour? --
    def gate_auc(scale, month):
        for fold in runs[scale]["folds"]:
            if fold["fold"]["name"].endswith(month):
                return fold["gate"]["trained_auc"]
        return None

    _grouped_bars(ax1, months, scales, gate_auc, lambda v: f"{v:.3f}")
    ax1.set_xticklabels(months)
    ax1.axhline(0.5, color=INK_2, linewidth=0.9, linestyle="--", zorder=4)
    ax1.set_ylim(0.45, max(0.85, ax1.get_ylim()[1]))
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title("Whether a move is coming", fontsize=10, color=INK, loc="left", pad=10)
    _clean(ax1)

    # -- 2. the side: which barrier first? --
    models = [m for m in MODEL_ORDER if any(m in runs[s]["folds"][0]["side_auc"] for s in scales)]

    def side_auc(scale, model):
        values = [
            fold["side_auc"][model]
            for fold in runs[scale]["folds"]
            if model in fold["side_auc"] and not np.isnan(fold["side_auc"][model])
        ]
        return float(np.mean(values)) if values else None

    _grouped_bars(ax2, models, scales, side_auc, lambda v: f"{v:.3f}")
    ax2.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
    ax2.axhline(0.5, color=INK_2, linewidth=0.9, linestyle="--", zorder=4)
    ax2.set_ylim(0.47, max(0.58, ax2.get_ylim()[1]))
    ax2.set_ylabel("ROC-AUC, mean over folds")
    ax2.set_title("Which way it goes", fontsize=10, color=INK, loc="left", pad=10)
    _clean(ax2)

    # -- 3. what reaches the P&L --
    def gross(scale, model):
        row = _pooled(runs[scale], model, gate, quantile)
        return row["gross_bps"] if row else None

    _grouped_bars(ax3, models, scales, gross, lambda v: f"{v:+.2f}")
    ax3.set_xticklabels([MODEL_LABELS[m] for m in models], fontsize=8)
    ax3.axhline(0.0, color=INK_2, linewidth=0.9, zorder=4)
    ax3.set_ylabel("gross P&L per trade (bp)")
    ax3.set_title("What reaches the P&L", fontsize=10, color=INK, loc="left", pad=10)
    # Headroom for the value labels, which sit above the bar they annotate.
    lo, hi = ax3.get_ylim()
    ax3.set_ylim(lo - 0.12 * (hi - lo), hi + 0.14 * (hi - lo))
    _clean(ax3)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="lookback",
        loc="upper right",
        bbox_to_anchor=(0.995, 0.99),
        ncol=len(scales),
        frameon=False,
        fontsize=8.5,
        title_fontsize=8.5,
    )

    fig.suptitle(
        "Does a longer lookback help?",
        fontsize=12,
        color=INK,
        x=0.006,
        y=0.965,
        ha="left",
        fontweight="bold",
    )

    selection = (
        "every window, one position at a time"
        if gate == "none"
        else f"the {'realized-variance' if gate == 'rv' else 'trained'} gate's top "
        f"{(1 - quantile) * 100:.0f}%"
    )
    trades = ", ".join(
        f"{SCALE_LABELS[s]} {row['n_trades']:,}"
        for s in scales
        if (row := _pooled(runs[s], models[0], gate, quantile))
    )
    fig.text(
        0.006,
        0.205,
        "Same 60 bp target, same one-hour deadline, same folds and the same execution model\n"
        "throughout; only the lookback changes. Panels 1 and 2 are the three out-of-sample\n"
        f"months. Panel 3 is pooled across them, trading {selection}\n"
        f"({trades}). An empty slot marks a model not yet trained at that lookback.",
        fontsize=8,
        color=MUTED,
        va="top",
        linespacing=1.5,
    )
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="one walkforward.py --out JSON per lookback, in any order",
    )
    parser.add_argument("--gate", default="rv", choices=("none", "rv", "trained"))
    parser.add_argument(
        "--gate-quantile",
        type=float,
        default=0.9,
        help="which threshold of that gate to draw in the P&L panel",
    )
    parser.add_argument("--out", default=str(ASSETS))
    args = parser.parse_args()

    runs = load_runs(args.results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "window_comparison.png"
    figure_windows(runs, path, args.gate, args.gate_quantile)
    print(f"  wrote {path}  ({', '.join(runs)})")


if __name__ == "__main__":
    main()
