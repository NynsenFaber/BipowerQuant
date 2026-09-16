"""The 7 tabular features, computed by the C++ engine.

This is the **cross-check path**, and that is the whole reason it exists as a
separate module. The same seven features are computed twice, two different ways:

* here, by `bipower_core`'s sliding-window C++ loop, which walks each window
  explicitly and accumulates the sums term by term;
* in `sequence_matrix.build_tabular_features`, by NumPy cumulative sums written
  from the formulas in the README rather than transcribed from that loop.

`tests/test_pipeline.py::test_the_two_feature_paths_agree` asserts the two match.
Two independent implementations of the same formulas agreeing is evidence the
formulas were implemented correctly; one implementation agreeing with itself is
not. The production pipelines (`walkforward.py`, `sweep_barriers.py`) all use the
NumPy path, because it is vectorised across every window at once and needs no
compiled extension — which is also what lets them run unchanged inside Colab.

Both paths take their bar grid and their labels from `sequence_matrix.py`, so a
disagreement can only come from the feature arithmetic itself.

The bar grid is **complete**: every second in the period gets a bar, trade-less
seconds carrying a forward-filled price and zero volume. That matters more than
it sounds like it should. This module previously used Polars' `group_by_dynamic`,
which emits a bar only for seconds that contain a trade — and on BTC/USDT ~11% of
seconds contain none, so a "60-bar horizon" was really a horizon of an arbitrary
70-odd seconds that varied with how busy the market was. Under a triple barrier,
where the vertical barrier *is* the horizon, that would make the label itself
depend on activity.
"""

from __future__ import annotations

import numpy as np

import bipower_core  # type: ignore
import sequence_matrix as seq
from sequence_matrix import BARRIER, EPSILON, HORIZON, WINDOW_SIZE


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
    y_all, usable = seq.build_targets(bars, horizon=horizon, barrier=barrier, label_mode=label_mode)

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
