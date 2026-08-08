"""
Barrier/horizon sweep: which target is worth trying to predict?

The project's original barrier was 5 bp because that is one taker round trip in
fees. The arithmetic usually quoted against that choice —

    E = B (2h - 1) - c

— says it can never pay, since `E < 0` for every `h` when `B < c`. True, but it
is the wrong correction, and acting on it makes the target *worse*. That form
assumes every trade ends at `+B` or `-B`. Most do not: at a 60-second deadline
roughly 60% of positions are closed by the clock, earning about nothing and
paying the full round trip anyway. With `rho` for the share that reaches a
horizontal barrier and `G` for the magnitude actually captured when one does
(the barrier plus the overshoot through it):

    E = rho * G * (2h - 1) - c        =>    h* = 1/2 (1 + c / (rho * G))

`h*` is the hit rate a side model must reach for the target to break even, and it
is what this sweep minimises. The `rho` term is the whole story:

* At a fixed horizon, `rho * G` has an interior maximum. Widening the barrier
  grows `G` roughly linearly and collapses `rho` faster. On BTC/USDT at 60
  seconds that maximum sits near 5 bp — so raising the barrier to 6 bp to "cover
  the round trip" raises `h*` from 1.58 to 1.61. Both are impossible; the second
  is more impossible.
* Only a longer deadline grows `rho * G`. Barrier and horizon are one choice, not
  two, and the horizon is the binding half.

Two stages, both restricted to the training months with `--months`, so the target
is chosen from information available before the test period begins:

    # 1. label geometry alone — no model, seconds per cell
    python sweep_barriers.py --bars ../data/bars_6m.npz --geometry-only \
        --months 2026-01 2026-02 2026-03 \
        --barriers 5 10 20 30 40 60 100 --horizons 60 300 900 1800 3600 7200

    # 2. does the achievable hit rate survive the shortlist? fits XGBoost
    python sweep_barriers.py --bars ../data/bars_6m.npz \
        --months 2026-01 2026-02 2026-03 --train-months 2 \
        --barriers 20 40 60 --horizons 900 1800 3600 --train-stride 10

Stage 1 says what a target *costs*. Stage 2 says whether a model can pay it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import backtest as bt
import sequence_matrix as seq
import walkforward as wf


def barrier_geometry(
    price: np.ndarray,
    rv_gate: np.ndarray,
    starts: np.ndarray,
    window: int,
    horizon: int,
    barrier: float,
    costs: bt.Costs,
    execution: bt.Execution,
    gate_quantile: float = 0.9,
) -> dict:
    """What one (barrier, horizon) target costs, before any model is fitted.

    Everything here is a property of the price path and the label definition, so
    it is measurable on training months alone and cannot be improved by a better
    model — which is exactly what makes it a selection criterion. Returns the
    resolved share `rho`, the captured magnitude `G`, the break-even hit rate
    `h*` both over all windows and over the windows an untrained realized-variance
    gate would admit, and the holding period that sets how many non-overlapping
    trades a day can hold.

    `h*` is reported twice because the gate changes the traded population and
    therefore the arithmetic: selecting the top decile of realized variance
    roughly doubles the share of positions that reach a barrier rather than
    timing out, which is a larger effect on `h*` than anything the side model does.
    """
    side, exit_offset, defined = seq.triple_barrier(price, horizon=horizon, barrier=barrier)
    label_bar = starts + window - 1

    resolved = (side[label_bar] != 0) & defined[label_bar]
    exit_bar = label_bar + exit_offset[label_bar]
    magnitude = np.abs(price[exit_bar] / price[label_bar] - 1.0) / bt.BPS

    admitted = rv_gate >= np.quantile(rv_gate, gate_quantile)
    captured = float(magnitude[resolved].mean()) if resolved.any() else 0.0
    rho_all = float(resolved.mean())
    rho_gated = float(resolved[admitted].mean()) if admitted.any() else 0.0
    hold = float(exit_offset[label_bar][admitted].mean()) if admitted.any() else float("nan")

    return {
        "barrier_bps": barrier / bt.BPS,
        "horizon_s": horizon,
        "resolved_share": rho_all,
        "resolved_share_gated": rho_gated,
        "captured_bps": captured,
        "overshoot_pct": captured / (barrier / bt.BPS) - 1.0,
        "required_hit_rate": bt.required_hit_rate(rho_all, captured, costs, execution),
        "required_hit_rate_gated": bt.required_hit_rate(rho_gated, captured, costs, execution),
        "mean_hold_s": hold,
        "trades_per_day": bt.SECONDS_PER_DAY / hold if hold and np.isfinite(hold) else float("nan"),
        "gate_quantile": gate_quantile,
    }


def evaluate_cell(
    bars: dict,
    folds: list[wf.Fold],
    window: int,
    horizon: int,
    barrier: float,
    costs: bt.Costs,
    execution: bt.Execution,
    train_stride: int,
    gate_quantiles: tuple[float, ...],
) -> dict:
    """One (barrier, horizon) cell: walk forward, then price the result."""
    price = bars["price"]
    starts = seq.valid_window_starts(price.size, window, horizon)
    channels = seq.build_channels(bars, "full")
    X = seq.build_tabular_features(channels, price, starts, window)
    del channels

    precomputed = bt.barrier_arrays(price, horizon, barrier)
    side, _, defined = precomputed
    label_bar = starts + window - 1
    y_side = (side[label_bar] > 0).astype(np.int8)
    touched = ((side[label_bar] != 0) & defined[label_bar]).astype(np.int8)

    purge = window + horizon - 1
    rows, fold_stats = [], []
    for fold in folds:
        result = wf.run_fold(
            bars,
            fold,
            X,
            starts,
            y_side,
            touched,
            window,
            horizon,
            barrier,
            purge,
            None,
            train_stride,
            side_models=("XGBoost",),
        )
        rows += wf.backtest_fold(
            bars,
            result,
            window,
            horizon,
            barrier,
            costs,
            execution,
            precomputed,
            gate_quantiles=gate_quantiles,
        )
        fold_stats.append(
            {
                "fold": result["fold"]["name"],
                "gate": result["gate"],
                "side_auc": result["side_auc"],
            }
        )

    pooled = wf.pool_daily(rows)
    return {
        "barrier_bps": barrier / bt.BPS,
        "horizon_s": horizon,
        "touch_rate": float(touched.mean()),
        "breakeven_bps": bt.breakeven_barrier_bps(costs, execution),
        "folds": fold_stats,
        "pooled": list(pooled.values()),
    }


def run_geometry(
    bars: dict, args: argparse.Namespace, costs: bt.Costs, execution: bt.Execution
) -> None:
    """Stage 1: price every target in the grid, fit nothing, print the ranking."""
    price = bars["price"]
    longest = max(args.horizons)

    # One window population, sized by the longest horizon, so every cell is scored
    # on identical decision bars. Scoring each cell on its own `valid_window_starts`
    # would give the short horizons more windows and quietly compare populations
    # rather than targets.
    starts = seq.valid_window_starts(price.size, args.window, longest)
    rv_gate = seq.rolling_realized_variance(price, starts, args.window)
    print(f"{starts.size:,} common windows (sized by the {longest}s horizon)\n")

    cells = []
    for horizon in args.horizons:
        for barrier_bps in args.barriers:
            cells.append(
                barrier_geometry(
                    price,
                    rv_gate,
                    starts,
                    args.window,
                    horizon,
                    barrier_bps * bt.BPS,
                    costs,
                    execution,
                    gate_quantile=max(args.gate_quantiles),
                )
            )
            print(".", end="", flush=True)
    print("\n")

    header = (
        f"{'horizon':>8} {'barrier':>8} {'resolved':>9} {'res|gate':>9} {'captured':>9} "
        f"{'over':>6} {'h*':>7} {'h*|gate':>8} {'hold':>7} {'trades/d':>9}"
    )
    print("=" * len(header))
    print("STAGE 1: what each target costs, before any model  (h* = hit rate needed to break even)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for cell in cells:
        print(
            f"{cell['horizon_s']:>7}s {cell['barrier_bps']:>7.0f}b "
            f"{cell['resolved_share']:>9.1%} {cell['resolved_share_gated']:>9.1%} "
            f"{cell['captured_bps']:>8.2f}b {cell['overshoot_pct']:>5.0%} "
            f"{cell['required_hit_rate']:>7.3f} {cell['required_hit_rate_gated']:>8.3f} "
            f"{cell['mean_hold_s']:>6.0f}s {cell['trades_per_day']:>9.0f}"
        )

    # The selection rule, applied. `B > c` is the oracle condition; the trade floor
    # keeps a cell from winning on h* alone while firing too rarely to measure.
    breakeven = bt.breakeven_barrier_bps(costs, execution)
    eligible = [
        c
        for c in cells
        if c["barrier_bps"] > breakeven
        and c["trades_per_day"] >= args.min_trades_per_day
        and np.isfinite(c["required_hit_rate_gated"])
    ]
    print(
        f"\n{len(eligible)} of {len(cells)} cells clear the filters "
        f"(barrier > {breakeven:.1f} bp, >= {args.min_trades_per_day:g} trades/day)"
    )
    if eligible:
        print("\nshortlist, by the hit rate the side model would have to reach:")
        for cell in sorted(eligible, key=lambda c: c["required_hit_rate_gated"])[:8]:
            print(
                f"   {cell['barrier_bps']:>4.0f} bp / {cell['horizon_s']:>5}s  "
                f"h* {cell['required_hit_rate_gated']:.3f} gated, "
                f"{cell['required_hit_rate']:.3f} ungated  |  "
                f"{cell['trades_per_day']:>4.0f} trades/day, "
                f"oracle nets {cell['captured_bps'] - breakeven:+.1f} bp"
            )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "stage": "geometry",
                    "months": args.months,
                    "window": args.window,
                    "costs": asdict(costs),
                    "execution": asdict(execution),
                    "breakeven_bps": breakeven,
                    "min_trades_per_day": args.min_trades_per_day,
                    "cells": cells,
                },
                indent=2,
                default=float,
            )
        )
        print(f"\nWrote {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bars", nargs="+", required=True)
    parser.add_argument(
        "--barriers",
        nargs="+",
        type=float,
        default=[5, 10, 20, 40],
        help="barrier half-widths in basis points",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[60, 300, 900],
        help="vertical barriers in seconds",
    )
    parser.add_argument("--window", type=int, default=seq.WINDOW_SIZE)
    parser.add_argument(
        "--months",
        nargs="+",
        default=None,
        metavar="YYYY-MM",
        help="restrict the series to these calendar months; use the TRAINING months, "
        "so the target is not chosen by looking at the period it is tested on",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="stage 1: report what each target costs (rho, G, h*) without fitting anything",
    )
    parser.add_argument("--scheme", default="anchored", choices=wf.SCHEMES)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--taker-fee-bps", type=float, default=2.5)
    parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    parser.add_argument(
        "--slippage-bps",
        type=float,
        default=bt.MEASURED_SLIPPAGE_BPS,
        help=f"per marketable side; the default is measured from the tape at "
        f"{bt.MEASURED_SLIPPAGE_SIZE_BTC} BTC, not assumed",
    )
    parser.add_argument("--entry", default="taker", choices=("taker", "maker"))
    parser.add_argument("--exit", default="taker", choices=("taker", "maker"))
    parser.add_argument("--gate-quantiles", nargs="+", type=float, default=[0.0, 0.9])
    parser.add_argument(
        "--min-trades-per-day",
        type=float,
        default=50.0,
        help="floor on non-overlapping trades a day; a target that fires too rarely "
        "cannot be told apart from noise however good its h* looks",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    bars = wf.load_bars(args.bars)
    if args.months:
        bars = wf.slice_months(bars, args.months)
    print(f"{bars['meta']['n_bars']:,} bars | {bars['meta'].get('source', '?')[:60]}...")
    if args.months:
        print(f"selection restricted to {', '.join(sorted(args.months))}")

    spread = bt.estimate_half_spread_bps(bars["price"])
    costs = bt.Costs(
        taker_fee_bps=args.taker_fee_bps,
        maker_fee_bps=args.maker_fee_bps,
        half_spread_bps=spread["half_spread_bps"],
        slippage_bps=args.slippage_bps,
    )
    execution = bt.Execution(entry=args.entry, exit=args.exit)
    breakeven = bt.breakeven_barrier_bps(costs, execution)
    print(f"half-spread {spread['half_spread_bps']:.4f} bp via {spread['method']}")
    print(f"round trip  {breakeven:.2f} bp  ({args.entry} in / {args.exit} out)")
    print(f"=> a perfect predictor needs barrier > {breakeven:.2f} bp, which is necessary")
    print("   and nowhere near sufficient; the column that matters below is h*.")

    if args.geometry_only:
        run_geometry(bars, args, costs, execution)
        return

    folds = wf.build_folds(bars["ts"], args.scheme, args.train_months)
    print(f"\n{len(folds)} {args.scheme} folds, train stride {args.train_stride}")

    cells = []
    for horizon in args.horizons:
        for barrier_bps in args.barriers:
            print(f"\n--- barrier {barrier_bps:g} bp | horizon {horizon}s ---", flush=True)
            cell = evaluate_cell(
                bars,
                folds,
                args.window,
                horizon,
                barrier_bps * bt.BPS,
                costs,
                execution,
                args.train_stride,
                tuple(args.gate_quantiles),
            )
            best = max(
                (r for r in cell["pooled"] if np.isfinite(r["sharpe"])),
                key=lambda r: r["sharpe"],
                default=None,
            )
            side_aucs = [f["side_auc"].get("XGBoost", float("nan")) for f in cell["folds"]]
            print(
                f"    touch rate {cell['touch_rate']:.2%} | "
                f"XGB side AUC {np.nanmean(side_aucs):.4f}"
            )
            if best:
                print(
                    f"    best: {best['model']} gate={best['gate']}@{best['gate_quantile']:g} "
                    f"| {best['n_trades']:,} trades | hit {best['hit_rate']:.2%} "
                    f"| gross {best['gross_bps']:.2f} bp | net {best['net_bps']:.2f} bp "
                    f"| Sharpe {best['sharpe']:.2f}"
                )
            cells.append(cell)

    print("\n" + "=" * 100)
    print("SWEEP SUMMARY  (best configuration per cell, XGBoost side model)")
    print("=" * 100)
    header = (
        f"{'barrier':>8} {'horizon':>8} {'touch':>7} {'AUC':>7} {'trades':>9} "
        f"{'hit':>7} {'gross':>8} {'net':>8} {'Sharpe':>8}"
    )
    print(header)
    print("-" * len(header))
    for cell in cells:
        xgb_rows = [
            r for r in cell["pooled"] if r["model"] == "XGBoost" and np.isfinite(r["sharpe"])
        ]
        best = max(xgb_rows, key=lambda r: r["sharpe"], default=None)
        auc = np.nanmean([f["side_auc"].get("XGBoost", float("nan")) for f in cell["folds"]])
        if best is None:
            print(
                f"{cell['barrier_bps']:>7.0f}b {cell['horizon_s']:>7}s "
                f"{cell['touch_rate']:>7.1%} {auc:>7.4f} {'-':>9}"
            )
            continue
        print(
            f"{cell['barrier_bps']:>7.0f}b {cell['horizon_s']:>7}s {cell['touch_rate']:>7.1%} "
            f"{auc:>7.4f} {best['n_trades']:>9,} {best['hit_rate']:>7.2%} "
            f"{best['gross_bps']:>8.2f} {best['net_bps']:>8.2f} {best['sharpe']:>8.2f}"
        )

    if args.out:
        Path(args.out).write_text(
            json.dumps(
                {
                    "costs": asdict(costs),
                    "execution": asdict(execution),
                    "breakeven_bps": breakeven,
                    "spread": spread,
                    "scheme": args.scheme,
                    "train_months": args.train_months,
                    "train_stride": args.train_stride,
                    "cells": cells,
                },
                indent=2,
                default=float,
            )
        )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
