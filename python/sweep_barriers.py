"""
Barrier/horizon sweep: where, if anywhere, does the edge clear its costs?

The single-month study set the barrier at 5 bp because that is one taker round
trip. That choice makes the strategy unprofitable *by construction*, and the
arithmetic is worth writing down before any model is fitted.

A position that wins captures `B` basis points and one that loses gives back `B`,
so with hit rate `h` and round-trip cost `c` the expectation per trade is

    E = B (2h - 1) - c

which is positive only when `B > c / (2h - 1)`. At the measured `h ~ 0.55` and a
6 bp taker round trip that needs `B > 60 bp` — twelve times the barrier the
project has been using. At `h = 0.60` it needs 30 bp; at `h = 0.65`, 20 bp.

But `h` is not free either. Widening the barrier without lengthening the horizon
simply means fewer windows resolve at all, and the ones that do resolve at the
vertical barrier contribute a small return minus the full cost. So barrier and
horizon have to move together, and whether `h` holds up as they do is an
empirical question. This sweep answers it.

    python sweep_barriers.py --bars ../data/bars_6m.npz \
        --barriers 5 10 20 40 --horizons 60 300 900 --train-stride 10
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
    parser.add_argument("--scheme", default="anchored", choices=wf.SCHEMES)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--train-stride", type=int, default=10)
    parser.add_argument("--taker-fee-bps", type=float, default=2.5)
    parser.add_argument("--maker-fee-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.5)
    parser.add_argument("--entry", default="taker", choices=("taker", "maker"))
    parser.add_argument("--exit", default="taker", choices=("taker", "maker"))
    parser.add_argument("--gate-quantiles", nargs="+", type=float, default=[0.0, 0.9])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    bars = wf.load_bars(args.bars)
    print(f"{bars['meta']['n_bars']:,} bars | {bars['meta'].get('source', '?')[:60]}...")

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
    print(f"=> a perfect predictor needs barrier > {breakeven:.2f} bp;")
    for h in (0.55, 0.60, 0.65):
        print(f"   at hit rate {h:.0%}, breakeven barrier is {breakeven / (2 * h - 1):.1f} bp")

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
