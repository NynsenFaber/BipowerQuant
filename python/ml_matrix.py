"""
Tabular feature matrix for the Logistic Regression and XGBoost baselines.

Each 5-minute lookback window is collapsed into 7 scalars by the C++ engine
(`bipower_core`) and labelled by the triple-barrier method. `sequence_matrix.py`
owns the two things both pipelines must agree on — the 1-second bar grid and the
labels — so the tabular and sequence models are guaranteed to be solving the
identical problem on the identical windows rather than approximately so.

The bar grid is **complete**: every second in the period gets a bar, trade-less
seconds carrying a forward-filled price and zero volume. That matters more than
it sounds like it should. This module previously used Polars' `group_by_dynamic`,
which emits a bar only for seconds that contain a trade — and on BTC/USDT ~15% of
seconds contain none, so a "60-bar horizon" was really a horizon of an arbitrary
70-odd seconds that varied with how busy the market was. Under a triple barrier,
where the vertical barrier *is* the horizon, that would make the label itself
depend on activity.
"""

from __future__ import annotations

import numpy as np
import bipower_core  # type: ignore

import sequence_matrix as seq
from sequence_matrix import (  # re-exported so callers have one place to import from
    BARRIER,
    EPSILON,
    HORIZON,
    WINDOW_SIZE,
)

FEATURE_NAMES = seq.TABULAR_FEATURE_NAMES


def rolling_metrics(bars: dict, window: int = WINDOW_SIZE) -> dict:
    """Run the C++ sliding-window engine over a complete 1-second bar grid.

    The engine signs order flow with a per-bar `is_buyer_maker` boolean and adds
    the whole bar quantity, so it is fed `|ofi|` with the sign carried in the
    flag. Its `order_flow_imbalance` output is then exactly the tick-level netted
    signed volume `sequence_matrix` computes, instead of a per-second majority
    vote that discards offsetting trades inside the same second.
    """
    signed_volume = bars["ofi"]
    return bipower_core.calculate_rolling_metrics(
        np.ascontiguousarray(bars["price"], dtype=np.float64),
        np.ascontiguousarray(np.abs(signed_volume), dtype=np.float64),
        np.ascontiguousarray(signed_volume < 0, dtype=bool),
        window,
    )


def build_training_matrix(
    bars: dict,
    window: int = WINDOW_SIZE,
    horizon: int = HORIZON,
    barrier: float = BARRIER,
    label_mode: str = "triple_barrier",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(X, y, starts)` for every window the labelling scheme actually resolves.

    `starts` is the bar index each retained window begins at, which is what lets
    a caller split chronologically and purge the boundaries the same way the
    sequence pipeline does.
    """
    price = bars["price"]
    metrics = rolling_metrics(bars, window)

    starts = seq.valid_window_starts(price.size, window=window, horizon=horizon)
    y_all, usable = seq.build_targets(
        bars, horizon=horizon, barrier=barrier, label_mode=label_mode
    )

    # The engine emits one row per window start, so its output aligns 1:1 with
    # `starts` once the trailing windows without a defined label are dropped.
    realized_variance = np.asarray(metrics["realized_variance"])[starts]
    bipower = np.asarray(metrics["bipower_variation"])[starts]
    jumps = np.asarray(metrics["jump_component"])[starts]
    order_flow = np.asarray(metrics["order_flow_imbalance"])[starts]

    # Log return across the same window the engine consumed: window k spans bars
    # [k, k + window - 1], so this aligns 1:1 with the metrics above.
    log_price = np.log(price)
    return_5m = log_price[starts + window - 1] - log_price[starts]

    # Order flow normalised by the continuous volatility of the window. BPV is a
    # variance, so sqrt(BPV) puts OFI on a per-unit-of-risk scale.
    vol_adjusted_ofi = order_flow / (np.sqrt(bipower) + EPSILON)

    # Jumps are magnitude-only by construction (max(RV - BPV, 0)); signing them
    # with the lookback direction says whether the shock hit an up- or downtrend.
    signed_jumps = jumps * np.sign(return_5m)

    X = np.column_stack(
        (
            realized_variance,
            bipower,
            jumps,
            order_flow,
            return_5m,
            vol_adjusted_ofi,
            signed_jumps,
        )
    )

    keep = usable[starts + window - 1]
    return X[keep], y_all[starts + window - 1][keep], starts[keep]


def build_from_csv(
    csv_path: str,
    hours: float | None = None,
    skip_hours: float = 0.0,
    cache: str | None = None,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Stream a raw trades CSV straight to `(X, y, starts, bars)`.

    Cache handling — including the refusal to reuse a cache built over a
    different slice — is `sequence_matrix.load_or_build_bars`'s job, so both
    pipelines fail the same way on the same mistake.
    """
    bars = seq.load_or_build_bars(csv_path, hours=hours, skip_hours=skip_hours, cache=cache)
    X, y, starts = build_training_matrix(bars, **kwargs)
    return X, y, starts, bars


if __name__ == "__main__":
    import argparse

    from data_feeder import FILE_PATH

    parser = argparse.ArgumentParser(description="Smoke-test the tabular matrix builder.")
    parser.add_argument("--csv", default=FILE_PATH)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--cache", default=None, help="optional .npz bar cache")
    parser.add_argument("--label-mode", default="triple_barrier", choices=seq.LABEL_MODES)
    args = parser.parse_args()

    print(f"Streaming {args.hours} hours from {args.csv} ...")
    X, y, starts, bars = build_from_csv(
        args.csv, hours=args.hours, cache=args.cache, label_mode=args.label_mode
    )

    print("✅ Matrix built successfully")
    print(f"X matrix shape: {X.shape}")
    print(f"y target shape: {y.shape}")
    print(
        f"Windows resolved by a horizontal barrier: {len(y):,} "
        f"({len(y) / max(len(bars['price']) - WINDOW_SIZE - HORIZON + 1, 1):.2%} of the population)"
    )
    print(f"Upper barrier first (y = 1): {int(y.sum()):,} / {len(y):,} ({y.mean():.2%})")
