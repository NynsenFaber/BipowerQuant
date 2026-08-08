"""
Walk-forward evaluation across months, with a gate/side split and a backtest.

This module exists to answer the two caveats the single-month study ended on.

**Sample.** One month is one regime. Folds here are built from calendar months:
each fold trains on months strictly before its test month and is scored on that
month alone, so every reported number is a genuine forecast. Three schemes:

    anchored  train on everything before the test month (default)
    rolling   train on a fixed-width window of recent months
    holdout   one split: first half trains, second half tests

Anchored is the default because it is what a desk actually does — you do not
throw away last year's data — and because it produces one estimate per test
month instead of a single number that cannot be checked for drift.

**Population.** The side model is trained only on windows where a barrier is
touched, which is unknowable at decision time. Scoring it on that same subset
silently conditions on the outcome. The fix is a **gate**: a second predictor,
using only past information, that decides *whether* a window is worth trading,
while the side model decides *which way*. The pair is then evaluated on the
whole population, including the windows the gate declines — which is what makes
the result a strategy rather than a subset statistic.

Two gates are compared, and the comparison is the point:

    rv        realized variance, thresholded at a percentile. No training, no
              parameters. The single-month study established that a rolling sum
              of squared returns forecasts barrier-touching at ~0.75 AUC.
    trained   XGBoost on the 7-feature matrix, target `barrier_touched`.

If the trained gate does not beat the rolling sum, the honest architecture is
the rolling sum, and this says so.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import backtest as bt
import sequence_matrix as seq

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMES = ("anchored", "rolling", "holdout")


# --- folds --------------------------------------------------------------------


@dataclass
class Fold:
    name: str
    train_lo: int  # bar indices, half-open [lo, hi)
    train_hi: int
    test_lo: int
    test_hi: int

    def as_dict(self) -> dict:
        return asdict(self)


def month_blocks(ts: np.ndarray) -> list[tuple[str, int, int]]:
    """`(label, start_bar, end_bar)` per calendar month, half-open at the end.

    Boundaries are found by binary search on the first instant of each month
    rather than by formatting 16M timestamps, which would dominate the runtime
    of everything downstream.
    """
    stamps = np.asarray(ts, dtype=np.int64)
    blocks, lo = [], 0
    current = datetime.fromtimestamp(int(stamps[0]), UTC)
    while lo < stamps.size:
        year = current.year + (current.month == 12)
        month = current.month % 12 + 1
        next_month = int(datetime(year, month, 1, tzinfo=UTC).timestamp())
        hi = int(np.searchsorted(stamps, next_month, side="left"))
        blocks.append((current.strftime("%Y-%m"), lo, hi))
        if hi >= stamps.size:
            break
        lo, current = hi, datetime.fromtimestamp(int(stamps[hi]), UTC)
    return blocks


def slice_months(bars: dict, labels: list[str]) -> dict:
    """A new bar dict holding only the named `YYYY-MM` months, in order.

    The point is target *selection*. Choosing a barrier and a horizon by looking
    at how they behave on the months a model is later tested on would import the
    answer through the target definition — a subtler version of the population
    problem this whole module exists to fix, and one no amount of purging would
    catch. Restricting the selection series to the training months makes the
    choice reproducible from information available before the test period starts.

    Raises on a label that is not present, rather than silently selecting less
    data than asked for.
    """
    blocks = {label: (lo, hi) for label, lo, hi in month_blocks(bars["ts"])}
    missing = [label for label in labels if label not in blocks]
    if missing:
        raise SystemExit(f"month(s) {missing} not in the bars; have {sorted(blocks)}")

    # Contiguity is not a nicety here. Splicing non-adjacent months would leave a
    # price discontinuity at the seam, and a triple barrier reads that jump as a
    # genuine first passage — manufacturing touches out of a calendar gap.
    spans = sorted(blocks[label] for label in labels)
    if any(lo != previous_hi for (lo, _), (_, previous_hi) in zip(spans[1:], spans[:-1])):
        raise SystemExit(f"months {sorted(labels)} are not contiguous; the seam would fake a jump")

    keep = np.concatenate([np.arange(lo, hi) for lo, hi in spans])
    out = {key: bars[key][keep] for key in ("ts", "price", "qty", "n_trades", "ofi")}
    out["meta"] = {
        **bars["meta"],
        "n_bars": int(keep.size),
        "first_ts": int(out["ts"][0]),
        "last_ts": int(out["ts"][-1]),
        "months": sorted(labels),
    }
    return out


def build_folds(ts: np.ndarray, scheme: str = "anchored", train_months: int = 3) -> list[Fold]:
    blocks = month_blocks(ts)
    if len(blocks) <= train_months:
        raise SystemExit(
            f"{len(blocks)} month(s) of data cannot support {train_months} training "
            "months plus a test month. Add data or lower --train-months."
        )

    folds = []
    if scheme == "holdout":
        cut = len(blocks) // 2
        folds.append(
            Fold(
                name=f"{blocks[0][0]}..{blocks[cut - 1][0]} -> {blocks[cut][0]}..{blocks[-1][0]}",
                train_lo=blocks[0][1],
                train_hi=blocks[cut - 1][2],
                test_lo=blocks[cut][1],
                test_hi=blocks[-1][2],
            )
        )
        return folds

    for i in range(train_months, len(blocks)):
        first = 0 if scheme == "anchored" else i - train_months
        folds.append(
            Fold(
                name=f"{blocks[first][0]}..{blocks[i - 1][0]} -> {blocks[i][0]}",
                train_lo=blocks[first][1],
                train_hi=blocks[i - 1][2],
                test_lo=blocks[i][1],
                test_hi=blocks[i][2],
            )
        )
    return folds


# --- per-fold model fitting ---------------------------------------------------


def _xgb(y, seed=42, **kw):
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return xgb.XGBClassifier(
        n_estimators=kw.get("n_estimators", 200),
        max_depth=kw.get("max_depth", 5),
        learning_rate=kw.get("learning_rate", 0.05),
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=32,
        scale_pos_weight=(neg / pos) if pos else 1.0,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=1,
    )


def _logistic():
    return make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)
    )


def _safe_auc(y, p) -> float:
    y = np.asarray(y)
    if y.size == 0 or y.sum() in (0, y.size):
        return float("nan")
    return float(roc_auc_score(y, p))


def run_fold(
    bars: dict,
    fold: Fold,
    X: np.ndarray,
    starts: np.ndarray,
    y_side: np.ndarray,
    touched: np.ndarray,
    window: int,
    horizon: int,
    barrier: float,
    purge: int,
    patchtst_probs: dict | None = None,
    train_stride: int = 1,
    side_models: tuple[str, ...] | None = None,
) -> dict:
    """Fit gate and side models on the fold's training months, score its test month.

    `train_stride` thins the *training* windows only. Adjacent windows share
    `window - 1` bars, so at stride 1 a month contributes far fewer independent
    observations than rows; dropping to every Nth costs little and makes a
    barrier sweep tractable. The test month is always scored at stride 1.
    """
    train = (starts >= fold.train_lo) & (starts < fold.train_hi - purge)
    test = (starts >= fold.test_lo) & (starts < fold.test_hi)

    if train_stride > 1:
        thin = np.zeros_like(train)
        idx = np.flatnonzero(train)[::train_stride]
        thin[idx] = True
        train = thin

    X_tr, X_te = X[train], X[test]
    touched_tr, touched_te = touched[train], touched[test]
    side_tr, side_te = y_side[train], y_side[test]
    starts_te = starts[test]

    out = {"fold": fold.as_dict(), "n_train": int(train.sum()), "n_test": int(test.sum())}

    # -- gate: will any barrier be touched? Trained on every window. --
    gate_model = _xgb(touched_tr)
    gate_model.fit(X_tr, touched_tr)
    p_gate = gate_model.predict_proba(X_te)[:, 1]
    rv_gate = X_te[:, 0].astype(np.float64)  # realized variance, untrained

    # Gate *thresholds* are quantiles of the TRAINING scores, never the test
    # scores. Taking "the top 10% of this month" would need the month's score
    # distribution in advance, which is a look-ahead — mild, since these
    # distributions are stable, but the whole point of this file is that the
    # traded population is selected from information available at the time.
    out["gate_train_scores"] = {
        "trained": gate_model.predict_proba(X_tr)[:, 1],
        "rv": X_tr[:, 0].astype(np.float64),
    }
    out["gate"] = {
        "trained_auc": _safe_auc(touched_te, p_gate),
        "rv_auc": _safe_auc(touched_te, rv_gate),
        "touch_rate": float(touched_te.mean()),
    }

    # -- side: which barrier first? Trained only where one was touched. --
    fit_rows = touched_tr.astype(bool)
    available = {
        "Logistic Regression (OFI)": (_logistic(), [3]),
        "Logistic Regression (7f)": (_logistic(), list(range(X.shape[1]))),
        "XGBoost": (_xgb(side_tr[fit_rows]), list(range(X.shape[1]))),
    }
    # A sweep over barriers and horizons re-fits everything per cell, so it asks
    # for XGBoost alone; the full walk-forward wants the whole comparison.
    wanted = (
        available
        if side_models is None
        else {k: v for k, v in available.items() if k in side_models}
    )
    p_side = {}
    for name, (model, cols) in wanted.items():
        model.fit(X_tr[fit_rows][:, cols], side_tr[fit_rows])
        p_side[name] = model.predict_proba(X_te[:, cols])[:, 1]

    if patchtst_probs and fold.name in patchtst_probs:
        stored = patchtst_probs[fold.name]
        if stored.size == starts_te.size:
            p_side["PatchTST"] = stored
        else:
            print(
                f"  ! PatchTST probabilities for {fold.name} have {stored.size:,} rows, "
                f"the fold has {starts_te.size:,}; skipping."
            )

    touched_rows = touched_te.astype(bool)
    out["side_auc"] = {
        name: _safe_auc(side_te[touched_rows], p[touched_rows]) for name, p in p_side.items()
    }
    out["probabilities"] = p_side
    out["starts_test"] = starts_te
    out["p_gate"] = p_gate
    out["rv_gate"] = rv_gate
    out["touched_test"] = touched_te
    return out


# --- the strategy grid --------------------------------------------------------


def backtest_fold(
    bars: dict,
    fold_result: dict,
    window: int,
    horizon: int,
    barrier: float,
    costs: bt.Costs,
    execution: bt.Execution,
    precomputed: tuple,
    gate_quantiles: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9),
    side_margin: float = 0.0,
) -> list[dict]:
    """Price every (model, gate, threshold) combination on this fold's test month."""
    starts = fold_result["starts_test"]
    rows = []

    gates = {
        "none": (None, None),
        "rv": (fold_result["rv_gate"], fold_result["gate_train_scores"]["rv"]),
        "trained": (fold_result["p_gate"], fold_result["gate_train_scores"]["trained"]),
    }

    for model_name, p_up in fold_result["probabilities"].items():
        for gate_name, (gate_score, train_score) in gates.items():
            for q in gate_quantiles:
                if gate_name == "none" and q > 0:
                    continue
                if gate_name != "none" and q == 0:
                    continue
                if gate_score is None:
                    threshold, keep_frac = -np.inf, 1.0
                else:
                    # Cut set on the training months; `keep_frac` then reports what
                    # share of the test month actually clears it, which will not be
                    # exactly 1 - q when the regime shifts. That difference is
                    # information, not an error.
                    threshold = float(np.quantile(train_score, q))
                    keep_frac = float((gate_score >= threshold).mean())

                entry_bars, direction = bt.signals_from_probabilities(
                    starts,
                    window,
                    p_up,
                    long_threshold=0.5 + side_margin,
                    short_threshold=0.5 - side_margin,
                    gate=gate_score,
                    gate_threshold=threshold,
                )
                if entry_bars.size == 0:
                    continue
                result = bt.simulate(
                    bars,
                    entry_bars,
                    direction,
                    barrier=barrier,
                    horizon=horizon,
                    costs=costs,
                    execution=execution,
                    bootstrap=0,
                    precomputed=precomputed,
                )
                rows.append(
                    {
                        "fold": fold_result["fold"]["name"],
                        "model": model_name,
                        "gate": gate_name,
                        "gate_quantile": q,
                        "gate_keep_frac": keep_frac,
                        **{
                            k: v
                            for k, v in result.summary.items()
                            if k not in ("costs", "execution")
                        },
                        "daily_day": result.daily["day"],
                        "daily_pnl_bps": result.daily["pnl_bps"],
                    }
                )
    return rows


def pool_daily(rows: list[dict]) -> dict:
    """Pool per-fold daily P&L into one out-of-sample track per configuration.

    Every fold's test month is disjoint from every other's, so concatenating
    their daily series gives a single continuous out-of-sample equity curve —
    which is the only Sharpe worth quoting. A mean of per-fold Sharpes would
    weight a quiet month the same as a violent one and hide the drift between
    them.
    """
    pooled: dict[tuple, dict] = {}
    for row in rows:
        key = (row["model"], row["gate"], row["gate_quantile"])
        entry = pooled.setdefault(
            key,
            {
                "day": [],
                "pnl": [],
                "trades": 0,
                "wins": 0.0,
                "gross": 0.0,
                "net": 0.0,
                "hold": 0.0,
                "folds": 0,
                "keep": [],
                "resolved": 0.0,
                "resolved_wins": 0.0,
            },
        )
        entry["day"].append(row["daily_day"])
        entry["pnl"].append(row["daily_pnl_bps"])
        entry["trades"] += row["n_trades"]
        entry["wins"] += row["hit_rate"] * row["n_trades"]
        entry["gross"] += row["gross_bps"] * row["n_trades"]
        entry["net"] += row["net_bps"] * row["n_trades"]
        entry["hold"] += row["mean_hold_s"] * row["n_trades"]
        entry["keep"].append(row["gate_keep_frac"])
        entry["folds"] += 1
        entry["cost_bps"] = row["cost_bps"]
        # Tracked separately from `wins`: the break-even arithmetic needs the hit
        # rate among trades that actually reached a barrier, not among all trades.
        n_resolved = row["resolved_share"] * row["n_trades"]
        entry["resolved"] += n_resolved
        entry["resolved_wins"] += row["resolved_hit_rate"] * n_resolved

    out = {}
    for key, entry in pooled.items():
        day = np.concatenate(entry["day"])
        pnl = np.concatenate(entry["pnl"])
        order = np.argsort(day)
        day, pnl = day[order], pnl[order]
        n = max(entry["trades"], 1)
        sd = float(pnl.std(ddof=1)) if pnl.size > 1 else 0.0
        sharpe = float(pnl.mean() / sd * np.sqrt(bt.TRADING_DAYS)) if sd > 0 else float("nan")

        equity = np.cumsum(pnl) * bt.BPS
        peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))

        ci = None
        if pnl.size > 5:
            rng = np.random.default_rng(11)
            draws = rng.integers(0, pnl.size, (1000, pnl.size))
            sample = pnl[draws]
            sd_b = sample.std(axis=1, ddof=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                sh = sample.mean(axis=1) / sd_b * np.sqrt(bt.TRADING_DAYS)
            sh = sh[np.isfinite(sh)]
            if sh.size:
                ci = [float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))]

        out[key] = {
            "model": key[0],
            "gate": key[1],
            "gate_quantile": key[2],
            "gate_keep_frac": float(np.mean(entry["keep"])),
            "n_trades": entry["trades"],
            "trades_per_day": entry["trades"] / max(pnl.size, 1),
            "hit_rate": entry["wins"] / n,
            "resolved_share": entry["resolved"] / n,
            "resolved_hit_rate": (
                entry["resolved_wins"] / entry["resolved"] if entry["resolved"] else float("nan")
            ),
            "gross_bps": entry["gross"] / n,
            "cost_bps": entry["cost_bps"],
            "net_bps": entry["net"] / n,
            "mean_hold_s": entry["hold"] / n,
            "total_return": float(pnl.sum() * bt.BPS),
            "max_drawdown": float((np.concatenate(([0.0], equity)) - peak).min()),
            "sharpe": sharpe,
            "sharpe_ci": ci,
            "n_days": int(pnl.size),
            "n_folds": entry["folds"],
        }
    return out


# --- driver -------------------------------------------------------------------


def load_bars(paths: list[str]) -> dict:
    if len(paths) == 1:
        return seq.load_bars(paths[0])
    return seq.load_bar_caches(paths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bars", nargs="+", required=True, help="one or more .npz bar caches")
    parser.add_argument("--scheme", default="anchored", choices=SCHEMES)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument(
        "--train-stride",
        type=int,
        default=1,
        help="thin the training windows; test is always stride 1",
    )
    parser.add_argument("--barrier", type=float, default=seq.BARRIER)
    parser.add_argument("--horizon", type=int, default=seq.HORIZON)
    parser.add_argument("--window", type=int, default=seq.WINDOW_SIZE)
    parser.add_argument("--taker-fee-bps", type=float, default=2.5)
    parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=bt.MEASURED_SLIPPAGE_BPS,
        help=f"per marketable side; the default is measured from the tape at "
        f"{bt.MEASURED_SLIPPAGE_SIZE_BTC} BTC, not assumed",
    )
    parser.add_argument(
        "--half-spread-bps",
        type=float,
        default=None,
        help="default: estimate from the tape with the Roll estimator",
    )
    parser.add_argument("--entry", default="taker", choices=("taker", "maker"))
    parser.add_argument("--exit", default="taker", choices=("taker", "maker"))
    parser.add_argument("--queue-ahead-btc", type=float, default=1.0)
    parser.add_argument(
        "--side-margin", type=float, default=0.0, help="only trade when |p - 0.5| exceeds this"
    )
    parser.add_argument(
        "--patchtst-probs",
        default=None,
        help=".npz of per-fold PatchTST probabilities, keyed by fold name",
    )
    parser.add_argument("--out", default=None, help="write the full result as JSON")
    args = parser.parse_args()

    print(f"1. Loading {len(args.bars)} bar cache(s) ...")
    bars = load_bars(args.bars)
    meta = bars["meta"]
    print(f"   {meta['n_bars']:,} bars | {meta.get('source', '?')}")

    print("2. Building features and labels ...")
    full_channels = seq.build_channels(bars, "full")
    starts = seq.valid_window_starts(bars["price"].size, args.window, args.horizon)
    X = seq.build_tabular_features(full_channels, bars["price"], starts, args.window)
    del full_channels  # ~750 MB at six months; the features are all we need now

    precomputed = bt.barrier_arrays(bars["price"], args.horizon, args.barrier)
    side, exit_offset, defined = precomputed
    label_bar = starts + args.window - 1
    y_side = (side[label_bar] > 0).astype(np.int8)
    touched = ((side[label_bar] != 0) & defined[label_bar]).astype(np.int8)
    print(f"   {starts.size:,} windows | barrier touched in {touched.mean():.2%}")

    # What this target pays, before any model: the share of positions that reach a
    # horizontal barrier, and the magnitude captured when one is reached — the
    # barrier plus the overshoot through it, since touches are detected on a
    # 1-second grid. Recorded in the result so the figures and the README quote a
    # number this run produced rather than one transcribed from a sweep.
    resolved = touched.astype(bool)
    exit_bar = label_bar + exit_offset[label_bar]
    captured_bps = float(
        (np.abs(bars["price"][exit_bar] / bars["price"][label_bar] - 1.0) / bt.BPS)[resolved].mean()
    )

    folds = build_folds(bars["ts"], args.scheme, args.train_months)
    print(f"3. {len(folds)} {args.scheme} fold(s):")
    for f in folds:
        print(f"   {f.name}")

    patchtst_probs = None
    if args.patchtst_probs:
        raw = np.load(args.patchtst_probs, allow_pickle=False)
        patchtst_probs = {k: raw[k] for k in raw.files}
        print(f"   PatchTST probabilities supplied for: {', '.join(patchtst_probs)}")

    costs = bt.Costs(
        taker_fee_bps=args.taker_fee_bps,
        maker_fee_bps=args.maker_fee_bps,
        half_spread_bps=args.half_spread_bps,
        slippage_bps=args.slippage_bps,
    )
    if costs.half_spread_bps is None:
        costs.half_spread_bps = bt.roll_half_spread_bps(bars["price"])
        print(f"   Roll half-spread estimate: {costs.half_spread_bps:.4f} bp")
    execution = bt.Execution(entry=args.entry, exit=args.exit, queue_ahead_btc=args.queue_ahead_btc)
    breakeven = bt.breakeven_barrier_bps(costs, execution)
    needed = bt.required_hit_rate(float(touched.mean()), captured_bps, costs, execution)
    print(
        f"   round trip {breakeven:.2f} bp vs barrier {args.barrier / bt.BPS:.2f} bp"
        f"  ->  {'oracle can profit' if args.barrier / bt.BPS > breakeven else 'NEGATIVE by construction'}"
    )
    print(
        f"   resolves {touched.mean():.1%} of windows, capturing {captured_bps:.2f} bp when it does"
    )
    print(
        f"   => the side model needs a {needed:.1%} hit rate to break even"
        f"  ->  {'reachable' if needed < 0.65 else 'NOT REACHABLE by any model'}"
    )

    purge = args.window + args.horizon - 1
    all_rows, fold_results = [], []
    for fold in folds:
        print(f"\n4. Fold {fold.name}")
        result = run_fold(
            bars,
            fold,
            X,
            starts,
            y_side,
            touched,
            args.window,
            args.horizon,
            args.barrier,
            purge,
            patchtst_probs,
            args.train_stride,
        )
        print(f"   train {result['n_train']:,} | test {result['n_test']:,}")
        print(
            f"   gate AUC  trained {result['gate']['trained_auc']:.4f} | "
            f"RV {result['gate']['rv_auc']:.4f} | touch rate {result['gate']['touch_rate']:.2%}"
        )
        for name, auc in result["side_auc"].items():
            print(f"   side AUC  {name:<26} {auc:.4f}")
        all_rows += backtest_fold(
            bars,
            result,
            args.window,
            args.horizon,
            args.barrier,
            costs,
            execution,
            precomputed,
            side_margin=args.side_margin,
        )
        fold_results.append(
            {
                "fold": result["fold"],
                "gate": result["gate"],
                "side_auc": result["side_auc"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
            }
        )

    print("\n5. Pooled out-of-sample backtest")
    pooled = pool_daily(all_rows)
    rows = sorted(
        pooled.values(), key=lambda r: -(r["sharpe"] if np.isfinite(r["sharpe"]) else -99)
    )
    header = (
        f"{'model':<26} {'gate':<12} {'keep':>6} {'trades':>9} {'hit':>7} "
        f"{'hit|res':>8} {'gross':>8} {'net':>8} {'Sharpe':>8} {'95% CI':>18}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        ci = f"[{r['sharpe_ci'][0]:.2f}, {r['sharpe_ci'][1]:.2f}]" if r["sharpe_ci"] else "-"
        gate = r["gate"] if r["gate"] == "none" else f"{r['gate']}@{r['gate_quantile']:g}"
        print(
            f"{r['model']:<26} {gate:<12} {r['gate_keep_frac']:>6.1%} {r['n_trades']:>9,} "
            f"{r['hit_rate']:>7.2%} {r['resolved_hit_rate']:>8.2%} "
            f"{r['gross_bps']:>8.3f} {r['net_bps']:>8.3f} "
            f"{r['sharpe']:>8.2f} {ci:>18}"
        )
    print("\nhit     = predicted barrier touched first, over all trades")
    print("hit|res = the same, over trades that reached a horizontal barrier at all")
    print("          (the second is what B(2h-1) > c is about)")

    # A model missing from some folds is pooled over fewer months than the rest,
    # which makes its row quietly incomparable — say so rather than let it sit in
    # the same table looking equivalent.
    partial = {r["model"] for r in rows if r["n_folds"] < len(folds)}
    for model in sorted(partial):
        covered = max(r["n_folds"] for r in rows if r["model"] == model)
        print(
            f"\n!  {model} is pooled over {covered} of {len(folds)} folds — its row is not "
            "comparable to the others."
        )

    if args.out:
        payload = {
            "config": {
                "bars": args.bars,
                "scheme": args.scheme,
                "train_months": args.train_months,
                "barrier_bps": args.barrier / bt.BPS,
                "horizon": args.horizon,
                "window": args.window,
                "costs": asdict(costs),
                "execution": asdict(execution),
                "breakeven_barrier_bps": breakeven,
                "side_margin": args.side_margin,
                # What the target costs, independent of any model. The economics
                # figure draws its break-even curve from these three.
                "target": {
                    "resolved_share": float(touched.mean()),
                    "captured_bps": captured_bps,
                    "required_hit_rate": needed,
                },
            },
            "folds": fold_results,
            "pooled": [{k: v for k, v in r.items()} for r in rows],
        }
        Path(args.out).write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
