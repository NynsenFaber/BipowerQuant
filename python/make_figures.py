"""Regenerate the README figures from a checkpoint and a bar cache.

    cd python
    python make_figures.py --weights ../weights/patchtst_BTCUSDT_2026-05_full.pt \
                           --bars-cache ../data/bars_full.npz

Every number plotted is recomputed here from the held-out test split — nothing is
hard-coded — so a figure cannot silently drift away from the result it claims to
show. Writes PNGs to `assets/`.

The three figures answer three questions, in order:
  1. Does any trained model out-rank a single untrained feature?
  2. What did switching to the triple-barrier target actually change?
  3. Where is the model confident, and does it hold across volatility regimes?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# macOS OpenMP: both halves of the rule from benchmark_inference.py apply here,
# because this script is the one place that fits XGBoost *and* runs torch in a
# single process. xgboost must be imported first, AND torch must be pinned to one
# thread — "xgboost first, no pinning" deadlocks inside torch's first tensor copy
# after xgboost has opened an OpenMP region, with no error and no CPU burn.
import xgboost  # noqa: F401  — must precede torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import sequence_matrix as seq  # noqa: E402

# benchmark_inference owns the safe import order and refuses to load if torch beat
# it, so it must come before the first `import torch` here, not after.
from benchmark_inference import _assert_openmp_safe, fit_tabular_baselines  # noqa: E402

_assert_openmp_safe()

import torch  # noqa: E402

torch.set_num_threads(1)  # the other half of the rule — without this it deadlocks

from patchtst_model import WindowBatcher, load_checkpoint, predict_proba  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ASSETS = REPO_ROOT / "assets"

# --- palette (validated: light surface #fcfcfb, all-pairs CVD ΔE 9.2) ---------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

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


def _clean(ax, x_grid=True):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if x_grid else "y", linestyle="-", alpha=0.9)
    ax.grid(axis="y" if x_grid else "x", visible=False)
    ax.tick_params(length=0)


def compute(
    weights: Path | None, bars_cache: Path, device: str, batch_size: int, label_mode: str
) -> dict:
    """Score every available model on the identical held-out windows.

    `weights=None` draws the figures for the tabular models alone. That is the
    useful default before a PatchTST checkpoint exists: the label-contrast figure
    needs no network at all, and the other two are still readable without it.
    """
    bars = seq.load_bars(bars_cache)

    if weights is not None:
        model, payload = load_checkpoint(weights, map_location=device)
        meta = payload.get("data_meta", {})
    else:
        model, meta = None, {"label_mode": label_mode}
    ds = seq.build_sequence_dataset(bars, **seq.dataset_kwargs_from(meta))

    price = bars["price"]
    test = ds.splits["test"]
    full_channels = seq.build_channels(bars, "full")
    X_train = seq.build_tabular_features(full_channels, price, ds.splits["train"], ds.window)
    X_test = seq.build_tabular_features(full_channels, price, test, ds.window)

    y_test = ds.labels("test").astype(np.int8)
    barrier = ds.meta["barrier"]

    # The label-flip control. Under the triple barrier the "other" label is the
    # side the path would have been given had the barriers been read backwards —
    # which is just 1 - y, so the flip test is degenerate and unnecessary: a
    # volatility forecast cannot score on a label whose two classes require the
    # same move. It is only informative for the old compound target, where the
    # magnitude and the sign were bundled into one class.
    end = test + ds.window - 1
    forward = price[end + ds.horizon] / price[end] - 1.0
    flipped_label = (forward < -barrier).astype(np.int8)
    flip_is_informative = ds.meta.get("label_mode") == "fee_threshold"

    logistic, booster = fit_tabular_baselines(X_train, ds.labels("train"))

    probs = {
        "Logistic Regression": logistic.predict_proba(X_test[:, [3]])[:, 1],
        "XGBoost": booster.predict_proba(X_test)[:, 1],
    }
    if model is not None:
        model.to(device)
        batcher = WindowBatcher(ds.channels, test, y_test, seq_len=ds.window, device=device)
        probs["PatchTST"] = predict_proba(model, batcher, batch_size=batch_size)

    rv = X_test[:, 0].astype(np.float64)
    singles = {
        "Realized variance": rv,
        "|5m return|": np.abs(X_test[:, 4].astype(np.float64)),
        "Order flow imbalance": X_test[:, 3].astype(np.float64),
        "5m return": X_test[:, 4].astype(np.float64),
    }
    decile = np.clip((np.argsort(np.argsort(rv)) * 10) // rv.size, 0, 9)

    def safe_auc(truth: np.ndarray, score: np.ndarray) -> float:
        # A decile can be single-class on a small slice; report NaN, not 0.5.
        if truth.size == 0 or truth.sum() in (0, truth.size):
            return float("nan")
        return float(roc_auc_score(truth, score))

    def profile(score: np.ndarray) -> dict:
        return {
            "auc": safe_auc(y_test, score),
            "auc_flipped": safe_auc(flipped_label, score),
            "deciles": [safe_auc(y_test[decile == d], score[decile == d]) for d in range(10)],
            # Keys are percent-as-string so the cache round-trips through JSON.
            "topk": {
                str(pct): float(y_test[np.argsort(-score)[: int(pct * score.size / 100)]].mean())
                for pct in (1, 5, 10, 25, 50)
            },
        }

    return {
        "models": {name: profile(p) for name, p in probs.items()},
        "singles": {name: profile(s) for name, s in singles.items()},
        "label_contrast": label_contrast(bars, full_channels, meta),
        "base_rate": float(y_test.mean()),
        "n_test": int(test.size),
        "label_mode": ds.meta.get("label_mode", "triple_barrier"),
        "barrier": float(barrier),
        "horizon": int(ds.horizon),
        "touch_rate": float(ds.meta.get("touch_rate", float("nan"))),
        "flip_is_informative": bool(flip_is_informative),
    }


# Single features are free to score, so the two labellings can be compared on the
# same held-out period without a second pass of network inference.
CONTRAST_FEATURES = {
    "Realized variance": 0,
    "Bipower variation": 1,
    "Jumps": 2,
    "Order flow imbalance": 3,
    "5m return": 4,
}


def label_contrast(bars: dict, full_channels: np.ndarray, meta: dict) -> dict:
    """Untrained single features scored under both target definitions.

    This is the figure that explains why the target was changed. The old target
    (`forward_return > +5bp`) is a compound event — a large move happened *and*
    it went up — so a pure volatility estimate ranks it well without any
    directional skill at all. The triple-barrier target asks only which side is
    touched first, and both of its classes require the same 5 bp move, so the
    same volatility estimate collapses to a coin flip on it.
    """
    out = {}
    for mode in ("fee_threshold", "triple_barrier"):
        kwargs = {**seq.dataset_kwargs_from(meta), "label_mode": mode}
        ds = seq.build_sequence_dataset(bars, **kwargs)
        test = ds.splits["test"]
        X = seq.build_tabular_features(full_channels, bars["price"], test, ds.window)
        y = ds.labels("test").astype(np.int8)
        out[mode] = {
            "features": {
                name: float(roc_auc_score(y, X[:, col].astype(np.float64)))
                for name, col in CONTRAST_FEATURES.items()
            },
            "n": int(test.size),
            "base_rate": float(y.mean()),
        }
    return out


# --- figures ------------------------------------------------------------------


def figure_auc(data: dict, path: Path) -> None:
    """Headline AUC: trained models against untrained single features."""
    rows = [(n, d["auc"], "model") for n, d in data["models"].items()]
    rows += [(n, d["auc"], "single") for n, d in data["singles"].items()]
    rows.sort(key=lambda r: r[1])
    best_model = max(d["auc"] for d in data["models"].values())
    best_single = max(d["auc"] for d in data["singles"].values())

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    colors = [BLUE if kind == "model" else ORANGE for *_, kind in rows]
    y = np.arange(len(rows))
    ax.barh(y, [r[1] for r in rows], height=0.62, color=colors)
    ax.axvline(0.5, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    ax.text(0.5, len(rows) - 0.25, "  coin flip", color=INK_2, fontsize=8.5, va="center")

    for i, (_, value, _) in enumerate(rows):
        # Nudge labels that would otherwise be struck through by the 0.5 rule.
        x = value + 0.006
        if abs(x - 0.5) < 0.012:
            x = 0.506
        ax.text(x, i, f"{value:.4f}", va="center", fontsize=9, color=INK)

    ax.set_yticks(y, [r[0] for r in rows], fontsize=9.5, color=INK)
    lo = min(0.45, min(r[1] for r in rows) - 0.02)
    ax.set_xlim(lo, max(0.62, max(r[1] for r in rows) + 0.05))
    ax.set_xlabel(f"Test ROC-AUC on the trained target ({_target_label(data)})", fontsize=9)
    _clean(ax)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=BLUE),
        plt.Rectangle((0, 0), 1, 1, color=ORANGE),
    ]
    ax.legend(
        handles,
        ["Trained model", "Single feature, no training"],
        frameon=False,
        fontsize=8.5,
        loc="lower right",
        labelcolor=INK_2,
    )
    # Titled from the data, so restyling cannot leave a stale claim behind. The
    # margin threshold is deliberate: on a signal this small, "the trained model
    # is 0.003 ahead" is not a result, and a title that reads like one would be
    # the same mistake the old fee-threshold headline made.
    margin = best_model - best_single
    if margin <= 0:
        verdict = "Every trained model is out-ranked by a single untrained feature"
    elif margin < 0.01:
        verdict = "Training barely out-ranks the best single untrained feature"
    else:
        verdict = "Training now buys something no single feature does"
    ax.set_title(verdict, fontsize=11.5, color=INK, pad=12, loc="left", fontweight="bold")
    fig.text(
        0.008, 0.005,
        f"{data['n_test']:,} held-out windows · BTC/USDT May 2026 · base rate {data['base_rate']:.2%}",
        fontsize=8, color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_decomposition(data: dict, path: Path) -> None:
    """What the target change did: the same features, scored under both labels."""
    contrast = data["label_contrast"]
    old, new = contrast["fee_threshold"], contrast["triple_barrier"]
    names = list(CONTRAST_FEATURES)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(names))
    width = 0.36
    for i, (label, color, block) in enumerate(
        [
            (f"Old target: 1m return > +{data['barrier']:.2%}", ORANGE, old),
            ("Triple barrier: which side is touched first", BLUE, new),
        ]
    ):
        vals = [block["features"][n] for n in names]
        pos = x + (i - 0.5) * width
        ax.bar(pos, vals, width=width - 0.03, color=color, label=label)
        for xi, v in zip(pos, vals):
            ax.text(xi, v + 0.006, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)

    ax.axhline(0.5, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    ax.text(-0.48, 0.505, "coin flip", color=INK_2, fontsize=8.5)
    ax.set_xticks(x, [n.replace(" ", "\n", 1) for n in names], fontsize=9, color=INK)
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylim(0.45, max(0.80, max(old["features"].values()) + 0.05))
    ax.set_ylabel("Test ROC-AUC, no training", fontsize=9)
    _clean(ax, x_grid=False)
    ax.legend(
        frameon=False, fontsize=8.5, labelcolor=INK_2, ncol=2,
        loc="lower center", bbox_to_anchor=(0.5, 1.005), handlelength=1.2,
        columnspacing=1.6, borderpad=0,
    )
    ax.set_title(
        "The old target could be forecast by volatility alone; the new one cannot",
        fontsize=11.5, color=INK, pad=30, loc="left", fontweight="bold",
    )
    fig.text(
        0.008, 0.005,
        f"Same held-out period. Old target: {old['n']:,} windows, {old['base_rate']:.1%} positive.  "
        f"Triple barrier: {new['n']:,} windows that touched a barrier, {new['base_rate']:.1%} positive.",
        fontsize=8, color=MUTED,
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_residual(data: dict, path: Path) -> None:
    """Where the models are confident, and what is left after volatility."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    palette = {"PatchTST": BLUE, "XGBoost": ORANGE, "Logistic Regression": AQUA}
    colors = {n: c for n, c in palette.items() if n in data["models"]}

    pcts = ["1", "5", "10", "25", "50"]
    xs = np.arange(len(pcts))
    for name, color in colors.items():
        vals = [data["models"][name]["topk"][p] for p in pcts]
        ax1.plot(xs, vals, marker="o", markersize=6, linewidth=2, color=color, label=name)
    ax1.axhline(data["base_rate"], color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax1.text(0.02, data["base_rate"] + 0.012,
             f"base rate {data['base_rate']:.3f}", color=INK_2, fontsize=8.5)
    ax1.set_xticks(xs, [f"top {p}%" for p in pcts], fontsize=9)
    ax1.set_ylabel("Precision", fontsize=9)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    top = max(v for n in colors for v in data["models"][n]["topk"].values())
    ax1.set_ylim(0, max(0.40, top * 1.25))
    _clean(ax1, x_grid=False)
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="upper right")
    ax1.set_title("Precision in the confident tail", fontsize=10.5, color=INK,
                  pad=10, loc="left", fontweight="bold")

    all_deciles = []
    for name, color in colors.items():
        d = data["models"][name]["deciles"]
        all_deciles += [v for v in d if not np.isnan(v)]
        ax2.plot(range(10), d, marker="o", markersize=5, linewidth=2, color=color,
                 label=f"{name} (mean {np.nanmean(d):.3f})")
    ax2.axhline(0.5, color=INK_2, linewidth=1.2, linestyle=(0, (4, 3)))
    ax2.set_xticks(range(10), [str(d) for d in range(10)], fontsize=9)
    # Plain ASCII: the bundled sans has no arrow glyph and renders tofu.
    ax2.set_xlabel("Realized-variance decile (quiet to violent)", fontsize=9)
    ax2.set_ylabel("ROC-AUC within decile", fontsize=9)
    span = max(0.06, max(abs(v - 0.5) for v in all_deciles) * 1.3) if all_deciles else 0.1
    ax2.set_ylim(0.5 - span, 0.5 + span)
    _clean(ax2, x_grid=False)
    ax2.legend(frameon=False, fontsize=8.5, labelcolor=INK_2, loc="lower right")
    ax2.set_title("...and whether it holds across volatility regimes",
                  fontsize=10.5, color=INK, pad=10, loc="left", fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_dataset(bars: dict, path: Path) -> dict:
    """What the tape looks like, and what the labelling scheme does to it.

    Returns the summary statistics it plotted, so the README table and the
    figure cannot disagree about the same month.
    """
    price, ts = bars["price"], bars["ts"]
    meta = bars["meta"]
    n = price.size
    days = (ts[-1] - ts[0]) / 86400.0

    side = seq.triple_barrier_labels(price, seq.HORIZON, seq.BARRIER)
    starts = seq.valid_window_starts(n)
    side_w = side[starts + seq.WINDOW_SIZE - 1]

    # Calendar months, and which of them a walk-forward run holds out. Shading a
    # 70/10/20 split here would contradict how the models are actually validated.
    import walkforward as wf

    months = wf.month_blocks(ts)
    try:
        folds = wf.build_folds(ts, "anchored", 3)
        test_months = {f.name.split("-> ")[-1] for f in folds}
    except SystemExit:  # too few months to form a fold; just show the calendar
        test_months = set()

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(9.4, 6.6), gridspec_kw={"height_ratios": [2.1, 1.25, 1.0]}
    )

    # -- price, with each month marked and the held-out ones highlighted --
    day = (ts - ts[0]) / 86400.0
    step = max(n // 12000, 1)  # ~12k points is past the resolution of the figure
    ax1.plot(day[::step], price[::step], color=INK, linewidth=0.9)
    for label, lo, hi in months:
        held_out = label in test_months
        color = ORANGE if held_out else BLUE
        ax1.axvspan(day[lo], day[min(hi, n - 1)], color=color, alpha=0.12 if held_out else 0.05)
        ax1.text((day[lo] + day[min(hi, n - 1)]) / 2, price.max() * 1.005,
                 label[-2:] + ("  (test)" if held_out else ""),
                 ha="center", va="bottom", fontsize=8, color=color if held_out else MUTED)
    ax1.set_ylabel("BTC/USDT", fontsize=9)
    ax1.set_ylim(price.min() * 0.99, price.max() * 1.05)
    _clean(ax1, x_grid=False)
    ax1.set_title(
        f"{len(months)} month{'s' if len(months) != 1 else ''} — {n:,} one-second bars "
        f"over {days:.0f} days"
        + (f"; {len(test_months)} held out by walk-forward" if test_months else ""),
        fontsize=11.5, color=INK, pad=10, loc="left", fontweight="bold",
    )

    # -- activity: how much of the tape is actually empty --
    hourly = bars["n_trades"][: (n // 3600) * 3600].reshape(-1, 3600)
    ax2.fill_between(np.arange(hourly.shape[0]) / 24.0, hourly.sum(axis=1),
                     color=INK_2, linewidth=0, alpha=0.85)
    ax2.set_ylabel("trades / hour", fontsize=9)
    _clean(ax2, x_grid=False)
    ax2.set_title(
        f"{meta['traded_seconds']:,} seconds carry a trade; "
        f"{meta['empty_seconds']:,} ({meta['empty_seconds'] / n:.1%}) carry none "
        "and are forward-filled",
        fontsize=9.5, color=INK_2, pad=8, loc="left",
    )

    # -- what the triple barrier does to the population --
    counts = [(side_w == 1).sum(), (side_w == -1).sum(), (side_w == 0).sum()]
    labels = [
        f"upper barrier first\n(y = 1)",
        f"lower barrier first\n(y = 0)",
        f"neither, within {seq.HORIZON}s\n(dropped)",
    ]
    left = 0.0
    # The two barrier segments are ~10% wide each, so their captions would collide
    # at a shared baseline; stagger them instead of shrinking the type.
    for value, label, color, drop in zip(counts, labels, (BLUE, ORANGE, GRID), (0.44, 1.02, 0.44)):
        share = value / side_w.size
        ax3.barh([0], [share], left=left, height=0.55, color=color)
        ax3.text(left + share / 2, 0, f"{share:.1%}", ha="center", va="center",
                 fontsize=9.5, color=INK if color is GRID else SURFACE, fontweight="bold")
        ax3.text(left + share / 2, -drop, label, ha="center", va="top",
                 fontsize=8, color=INK_2)
        left += share
    ax3.set_xlim(0, 1)
    ax3.set_ylim(-1.75, 0.5)
    ax3.axis("off")
    ax3.set_title(
        f"Triple barrier at +/-{seq.BARRIER:.2%} over {seq.HORIZON}s, "
        f"applied to all {side_w.size:,} windows",
        fontsize=9.5, color=INK_2, pad=8, loc="left",
    )

    ax2.set_xlabel("day of the sample", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    return {
        "source": meta["source"],
        "n_bars": int(n),
        "days": float(days),
        "traded_seconds": int(meta["traded_seconds"]),
        "empty_seconds": int(meta["empty_seconds"]),
        "n_trades": int(bars["n_trades"].sum()),
        "volume_btc": float(bars["qty"].sum()),
        "price_range": [float(price.min()), float(price.max())],
        "n_windows": int(side_w.size),
        "upper_first": int(counts[0]),
        "lower_first": int(counts[1]),
        "vertical": int(counts[2]),
    }


def _target_label(data: dict) -> str:
    if data.get("label_mode") == "fee_threshold":
        return f"forward {data['horizon']}s return > +{data['barrier']:.2%}"
    return f"which +/-{data['barrier']:.2%} barrier is touched first, within {data['horizon']}s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--weights",
        default=None,
        help="PatchTST checkpoint; omit to draw the tabular models only",
    )
    parser.add_argument("--bars-cache", help=".npz written by sequence_matrix.save_bars")
    parser.add_argument(
        "--label-mode",
        default="triple_barrier",
        choices=seq.LABEL_MODES,
        help="ignored when --weights is given: the checkpoint's own recipe wins",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--out", default=str(ASSETS))
    parser.add_argument(
        "--scores-cache",
        default=None,
        help="reuse/write the scored metrics as JSON, so restyling a figure does not "
        "re-run 535k windows of inference",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache = Path(args.scores_cache) if args.scores_cache else None
    if cache is not None and cache.exists():
        print(f"Reusing scores from {cache}")
        data = json.loads(cache.read_text())
    else:
        if not args.bars_cache:
            raise SystemExit("--bars-cache is required unless --scores-cache already exists")
        weights = Path(args.weights) if args.weights else None
        if weights is None:
            print("No --weights given: drawing the tabular models only.")
        print("Scoring every model on the held-out split ...")
        data = compute(
            weights, Path(args.bars_cache), args.device, args.batch_size, args.label_mode
        )
        if cache is not None:
            cache.write_text(json.dumps(data, indent=2))
            print(f"  scores cached to {cache}")

    if args.bars_cache:
        stats = figure_dataset(seq.load_bars(args.bars_cache), out / "dataset.png")
        print(f"  wrote {out / 'dataset.png'}")
        print("  " + json.dumps(stats))

    for name, fn in (
        ("auc_comparison.png", figure_auc),
        ("volatility_vs_direction.png", figure_decomposition),
        ("residual_signal.png", figure_residual),
    ):
        fn(data, out / name)
        print(f"  wrote {out / name}")


if __name__ == "__main__":
    main()
