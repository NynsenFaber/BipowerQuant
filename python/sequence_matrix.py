"""
Sequence dataset builder for the PatchTST baseline.

The tabular models (`train_baseline.py`, `train_xgboost.py`) collapse each
5-minute lookback window into 7 scalars via the C++ engine. PatchTST instead
consumes the *raw sequence*: the same 300 one-second bars, as M parallel
univariate channels, patched into tokens.

This module therefore deliberately depends on nothing but Polars and NumPy —
no `bipower_core`, no torch — so the exact same code runs inside a Google Colab
runtime (where the C++ extension is not compiled) and locally.

Target definition, lookback length and horizon mirror `ml_matrix.py` bar for bar
so the PatchTST numbers are read against the same problem the trees solved.

The one deliberate difference: bars are placed on a **complete** 1-second grid
(empty seconds carry a forward-filled price and zero volume), so a "60-bar"
horizon is always literally 60 seconds. `ml_matrix.py` uses `group_by_dynamic`,
which silently drops trade-less seconds — see Phase 5 of TODO.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from data_feeder import CSV_SCHEMA

# --- Problem definition (kept identical to ml_matrix.py) ----------------------

# Minimum forward return required to call a move "profitable" (5 bp round-trip).
FEE_THRESHOLD = 0.0005

# 5-minute lookback window, in 1-second bars. This is PatchTST's sequence length L.
WINDOW_SIZE = 300

# 1-minute forward prediction horizon, in 1-second bars.
HORIZON = 60

# pi / 2, the Bipower Variation scale factor (mirrors PI_FACTOR in math_engine.hpp).
PI_FACTOR = 1.5707963267948966

# Guards the volatility normalisation against flat windows (mirrors ml_matrix.py).
EPSILON = 1e-8

# The tabular feature set, in the order ml_matrix.py stacks it.
TABULAR_FEATURE_NAMES = [
    "Realized Variance",
    "Bipower Variation",
    "Jumps",
    "Order Flow Imbalance",
    "5m Return",
    "Vol-Adjusted OFI",
    "Signed Jumps",
]

# --- Channel layout ----------------------------------------------------------

# Per-bar quantities fed to the model as independent channels. Channels 1-3 are
# exactly the per-bar increments the C++ engine sums into RV, BPV and OFI, so
# PatchTST sees the un-aggregated version of the tabular features: attention over
# patches can learn a time-weighted, non-linear alternative to a plain sum.
#
# Verified numerically: a plain rolling sum of `realized_var` / `bipower` over 300
# bars reproduces bipower_core's RV / BPV to ~1e-15, up to the leading-edge terms
# the C++ loop skips (it starts its returns at i > 0 and its pairs at i > 1).
CHANNEL_NAMES = [
    "log_return",    # r_t = log p_t - log p_{t-1}
    "realized_var",  # r_t^2                   -> window sum = RV
    "bipower",       # (pi/2) |r_t| |r_{t-1}|  -> window sum = BPV
    "ofi",           # signed traded volume    -> window sum = OFI
    "log_volume",    # log1p(total qty in the bar)
    "log_trades",    # log1p(number of trades in the bar)
]
N_CHANNELS = len(CHANNEL_NAMES)


# --- Second-bar construction -------------------------------------------------


def _time_divisor(sample_timestamp: int) -> int:
    """Number of raw time units per second in a Binance trades file.

    The archives have shipped timestamps in milliseconds and (more recently)
    microseconds, so infer it from the magnitude rather than hard-coding it.
    """
    if sample_timestamp > 1e14:
        return 1_000_000  # microseconds
    if sample_timestamp > 1e11:
        return 1_000  # milliseconds
    return 1  # seconds


def _scan_trades(csv_path: str | Path) -> pl.LazyFrame:
    # Explicit projection so Polars only materialises the 4 columns we consume.
    return pl.scan_csv(csv_path, has_header=False, schema=CSV_SCHEMA).select(
        "time", "price", "qty", "is_buyer_maker"
    )


def _collect(lazy: pl.LazyFrame, streaming: bool) -> pl.DataFrame:
    if not streaming:
        return lazy.collect()
    try:
        return lazy.collect(engine="streaming")
    except TypeError:  # Polars < 1.0 spelled it differently
        return lazy.collect(streaming=True)


def _aggregate_to_seconds(lazy: pl.LazyFrame, divisor: int, streaming: bool) -> pl.DataFrame:
    """Fold raw ticks into one row per second that contains at least one trade."""
    grouped = (
        lazy.with_columns((pl.col("time") // divisor).alias("ts"))
        .group_by("ts")
        .agg(
            [
                # sort_by makes "last trade of the second" independent of the
                # order the streaming engine happens to feed rows in.
                pl.col("price").sort_by("time").last().alias("price"),
                pl.col("qty").sum().alias("qty"),
                pl.len().alias("n_trades"),
                # Aggressive buys (is_buyer_maker == False) add, sells subtract.
                pl.when(pl.col("is_buyer_maker"))
                .then(-pl.col("qty"))
                .otherwise(pl.col("qty"))
                .sum()
                .alias("ofi"),
            ]
        )
    )
    return _collect(grouped, streaming).sort("ts")


def load_second_bars(
    csv_path: str | Path,
    hours: float | None = None,
    skip_hours: float = 0.0,
    streaming: bool = True,
    chunk_hours: float | None = None,
) -> dict:
    """Stream a raw Binance trades CSV into a complete 1-second bar series.

    One pass over the file turns ~72M ticks into ~2.7M bars (a month of BTC/USDT is
    ~110 MB in memory, ~30 MB compressed on disk), which is what makes repeated
    training runs cheap: build this once, cache it with `save_bars`, never touch the
    CSV again.

    Args:
        csv_path: path to the decompressed Binance `*-trades-*.csv`.
        hours: keep only the first N hours of the file (None = the whole file).
            `hours=8` reproduces the window population `train_xgboost.py` used.
        skip_hours: drop this many hours from the start before applying `hours`.
        streaming: use the Polars streaming engine (keeps peak RAM flat).
        chunk_hours: if set, aggregate in slices of this many hours instead of a
            single pass. Slower (one CSV scan per slice) but bounds memory hard.

    Returns:
        dict of NumPy arrays on a gap-free 1-second grid: `ts`, `price`, `qty`,
        `n_trades`, `ofi`, plus a `meta` dict.
    """
    csv_path = Path(csv_path)
    lazy = _scan_trades(csv_path)

    first_ts = lazy.select(pl.col("time").min()).collect().item()
    if first_ts is None:
        raise ValueError(f"No rows found in {csv_path}")
    divisor = _time_divisor(int(first_ts))

    start_raw = int(first_ts) + int(skip_hours * 3600 * divisor)
    end_raw = None if hours is None else start_raw + int(hours * 3600 * divisor)

    def slice_of(lo: int, hi: int | None) -> pl.LazyFrame:
        window = lazy.filter(pl.col("time") >= lo)
        return window if hi is None else window.filter(pl.col("time") < hi)

    if chunk_hours is None:
        bars = _aggregate_to_seconds(slice_of(start_raw, end_raw), divisor, streaming)
    else:
        step = int(chunk_hours * 3600 * divisor)
        last_raw = end_raw if end_raw is not None else int(
            lazy.select(pl.col("time").max()).collect().item()
        ) + 1
        pieces = []
        cursor = start_raw
        while cursor < last_raw:
            stop = min(cursor + step, last_raw)
            piece = _aggregate_to_seconds(slice_of(cursor, stop), divisor, streaming)
            if len(piece) > 0:
                pieces.append(piece)
            cursor = stop
        if not pieces:
            raise ValueError(f"No rows in the requested time range of {csv_path}")
        bars = pl.concat(pieces).sort("ts")

    if len(bars) == 0:
        raise ValueError(f"No rows in the requested time range of {csv_path}")

    grid = _to_regular_grid(bars, source=csv_path.name)
    # Recorded so a checkpoint can name the exact slice it was trained on.
    grid["meta"]["hours"] = hours
    grid["meta"]["skip_hours"] = skip_hours
    return grid


def _to_regular_grid(bars: pl.DataFrame, source: str) -> dict:
    """Scatter sparse second-bars onto a gap-free grid, forward-filling price."""
    ts = bars["ts"].to_numpy().astype(np.int64)
    grid = np.arange(ts[0], ts[-1] + 1, dtype=np.int64)
    slot = (ts - ts[0]).astype(np.int64)

    price = np.full(grid.shape, np.nan, dtype=np.float64)
    price[slot] = bars["price"].to_numpy()

    # Forward-fill: carry the index of the last observed price across the gaps.
    observed = np.where(~np.isnan(price), np.arange(price.size), 0)
    np.maximum.accumulate(observed, out=observed)
    price = price[observed]

    def scatter(name: str) -> np.ndarray:
        out = np.zeros(grid.shape, dtype=np.float64)
        out[slot] = bars[name].to_numpy()
        return out

    # A trade-less second is a real event, not missing data: no volume, no flow.
    return {
        "ts": grid,
        "price": price,
        "qty": scatter("qty"),
        "n_trades": scatter("n_trades"),
        "ofi": scatter("ofi"),
        "meta": {
            "source": source,
            "n_bars": int(grid.size),
            "first_ts": int(grid[0]),
            "last_ts": int(grid[-1]),
            "traded_seconds": int(ts.size),
            "empty_seconds": int(grid.size - ts.size),
        },
    }


def save_bars(bars: dict, path: str | Path) -> Path:
    """Cache a bar series to a compact `.npz` (a month is ~130 MB)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ts=bars["ts"],
        price=bars["price"],
        qty=bars["qty"],
        n_trades=bars["n_trades"],
        ofi=bars["ofi"],
        meta=np.array(json.dumps(bars["meta"])),
    )
    return path


def load_bars(path: str | Path) -> dict:
    """Reload a `save_bars` cache."""
    raw = np.load(path, allow_pickle=False)
    bars = {k: raw[k] for k in ("ts", "price", "qty", "n_trades", "ofi")}
    bars["meta"] = json.loads(str(raw["meta"]))
    return bars


# --- Channels, targets, windows ----------------------------------------------


def build_channels(bars: dict) -> np.ndarray:
    """Turn the bar series into the (n_bars, M) channel matrix PatchTST reads."""
    price = bars["price"]
    log_p = np.log(price)

    # r[0] = 0: the first bar has no predecessor, and every window that actually
    # gets used starts at least WINDOW_SIZE bars into the series anyway.
    r = np.diff(log_p, prepend=log_p[0])
    abs_r = np.abs(r)
    prev_abs_r = np.concatenate(([0.0], abs_r[:-1]))

    channels = np.column_stack(
        (
            r,
            r * r,
            PI_FACTOR * abs_r * prev_abs_r,
            bars["ofi"],
            np.log1p(bars["qty"]),
            np.log1p(bars["n_trades"]),
        )
    )
    assert channels.shape[1] == N_CHANNELS
    return np.ascontiguousarray(channels, dtype=np.float32)


def build_tabular_features(
    channels: np.ndarray,
    price: np.ndarray,
    starts: np.ndarray,
    window: int = WINDOW_SIZE,
) -> np.ndarray:
    """The 7 features `ml_matrix.py` builds, derived from the same channel matrix.

    This exists so the tabular baselines can be scored and timed on *exactly* the
    windows PatchTST sees — same bars, same grid, same splits — instead of the
    near-but-not-identical population `group_by_dynamic` produces.

    The window offsets reproduce the C++ loop precisely: `math_engine.hpp` starts
    its returns at i > 0 and its bipower pairs at i > 1, while OFI covers all bars.
    """
    cumulative = {
        col: np.concatenate(([0.0], np.cumsum(channels[:, col], dtype=np.float64)))
        for col in (1, 2, 3)
    }

    def window_sum(col: int, first_offset: int) -> np.ndarray:
        c = cumulative[col]
        return c[starts + window] - c[starts + first_offset]

    realized_variance = window_sum(1, 1)
    bipower = window_sum(2, 2)
    order_flow = window_sum(3, 0)
    jumps = np.maximum(realized_variance - bipower, 0.0)

    log_price = np.log(price)
    return_5m = log_price[starts + window - 1] - log_price[starts]
    vol_adjusted_ofi = order_flow / (np.sqrt(bipower) + EPSILON)
    signed_jumps = jumps * np.sign(return_5m)

    return np.column_stack(
        (
            realized_variance,
            bipower,
            jumps,
            order_flow,
            return_5m,
            vol_adjusted_ofi,
            signed_jumps,
        )
    ).astype(np.float32)


def build_targets(
    bars: dict, horizon: int = HORIZON, fee_threshold: float = FEE_THRESHOLD
) -> np.ndarray:
    """`y[i] = 1` iff the return from bar i to bar i+horizon clears the fee.

    Entries within `horizon` of the end are undefined and left at 0; no window
    returned by `valid_window_starts` ever labels against them.
    """
    price = bars["price"]
    y = np.zeros(price.size, dtype=np.int8)
    if price.size > horizon:
        forward_return = price[horizon:] / price[:-horizon] - 1.0
        y[:-horizon] = (forward_return > fee_threshold).astype(np.int8)
    return y


def valid_window_starts(
    n_bars: int, window: int = WINDOW_SIZE, horizon: int = HORIZON
) -> np.ndarray:
    """Start indices of every window with a full lookback and a defined label.

    Window k covers bars [k, k + window - 1] and is labelled by the forward move
    out of its **last** bar — the same alignment `ml_matrix.py` produces.
    """
    n_windows = n_bars - window - horizon + 1
    if n_windows <= 0:
        raise ValueError(
            f"Need at least {window + horizon} bars to form one window, got {n_bars}"
        )
    return np.arange(n_windows, dtype=np.int64)


def labels_for(starts: np.ndarray, y: np.ndarray, window: int = WINDOW_SIZE) -> np.ndarray:
    """Label each window by the bar it ends on."""
    return y[starts + window - 1]


def chronological_split(
    starts: np.ndarray,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    purge: int = WINDOW_SIZE + HORIZON - 1,
    train_stride: int = 1,
    val_stride: int = 1,
) -> dict[str, np.ndarray]:
    """Split windows in time order, purging the overlap at each boundary.

    Consecutive windows share 299 of their 300 bars, so a naive cut leaks: the
    last training windows are labelled by price moves that fall *inside* the
    first validation window's lookback. Dropping `purge = window + horizon - 1`
    windows before each boundary removes that overlap entirely.

    `train_stride > 1` thins the training windows (they are ~99.7% autocorrelated
    at stride 1). `val_stride` does the same for validation, which is only used
    for early stopping — on a full month, scoring every one of ~270k validation
    windows every epoch costs more than the epoch does.

    **Test is always kept at stride 1**, so the reported metric covers every
    window in the held-out period.
    """
    n = starts.size
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    if train_end - purge <= 0 or val_end - purge <= train_end:
        raise ValueError(
            f"{n} windows is too few to purge {purge} at each boundary; "
            "stream more hours or lower the purge."
        )
    return {
        "train": starts[: train_end - purge][::train_stride],
        "val": starts[train_end : val_end - purge][::val_stride],
        "test": starts[val_end:],
    }


@dataclass
class SequenceDataset:
    """Everything a PatchTST run needs, with no window ever materialised."""

    channels: np.ndarray  # (n_bars, M) float32
    y: np.ndarray  # (n_bars,) int8
    splits: dict[str, np.ndarray]  # split name -> window start indices
    window: int = WINDOW_SIZE
    horizon: int = HORIZON
    meta: dict = field(default_factory=dict)

    def labels(self, split: str) -> np.ndarray:
        return labels_for(self.splits[split], self.y, self.window)

    def positive_rate(self, split: str) -> float:
        labels = self.labels(split)
        return float(labels.mean()) if labels.size else float("nan")

    def summary(self) -> str:
        lines = [
            f"bars: {self.channels.shape[0]:,} x {self.channels.shape[1]} channels "
            f"({', '.join(CHANNEL_NAMES)})",
            f"window: {self.window} bars | horizon: {self.horizon} bars | "
            f"threshold: {self.meta.get('fee_threshold', FEE_THRESHOLD):.4%}",
        ]
        for name in ("train", "val", "test"):
            if name in self.splits:
                n = self.splits[name].size
                lines.append(f"{name:>5}: {n:>9,} windows | positive rate {self.positive_rate(name):.2%}")
        return "\n".join(lines)


def build_sequence_dataset(
    bars: dict,
    window: int = WINDOW_SIZE,
    horizon: int = HORIZON,
    fee_threshold: float = FEE_THRESHOLD,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    train_stride: int = 1,
    val_stride: int = 1,
) -> SequenceDataset:
    """Assemble channels, labels and purged chronological splits from bar data."""
    channels = build_channels(bars)
    y = build_targets(bars, horizon=horizon, fee_threshold=fee_threshold)
    starts = valid_window_starts(channels.shape[0], window=window, horizon=horizon)
    purge = window + horizon - 1
    splits = chronological_split(
        starts,
        train_frac=train_frac,
        val_frac=val_frac,
        purge=purge,
        train_stride=train_stride,
        val_stride=val_stride,
    )

    meta = dict(bars.get("meta", {}))
    meta.update(
        {
            "window": window,
            "horizon": horizon,
            "fee_threshold": fee_threshold,
            "channels": list(CHANNEL_NAMES),
            "train_frac": train_frac,
            "val_frac": val_frac,
            "train_stride": train_stride,
            "val_stride": val_stride,
            "purge": purge,
            "n_windows": int(starts.size),
            "split_sizes": {k: int(v.size) for k, v in splits.items()},
        }
    )
    return SequenceDataset(
        channels=channels, y=y, splits=splits, window=window, horizon=horizon, meta=meta
    )


def build_from_csv(
    csv_path: str | Path,
    hours: float | None = None,
    skip_hours: float = 0.0,
    cache: str | Path | None = None,
    **dataset_kwargs,
) -> SequenceDataset:
    """One-shot CSV -> dataset, reusing `cache` when it covers the same slice."""
    if cache is not None and Path(cache).exists():
        bars = load_bars(cache)
        cached = (bars["meta"].get("hours"), bars["meta"].get("skip_hours", 0.0))
        if cached != (hours, skip_hours):
            # Silently scoring a different slice than the caller asked for is the
            # one failure that would look like a valid result.
            raise ValueError(
                f"Bar cache {cache} covers hours={cached[0]}, skip_hours={cached[1]}, "
                f"but hours={hours}, skip_hours={skip_hours} was requested. "
                "Delete the cache or point at a different file."
            )
    else:
        bars = load_second_bars(csv_path, hours=hours, skip_hours=skip_hours)
        if cache is not None:
            save_bars(bars, cache)
    return build_sequence_dataset(bars, **dataset_kwargs)


if __name__ == "__main__":
    import argparse

    from data_feeder import FILE_PATH

    parser = argparse.ArgumentParser(description="Smoke-test the sequence builder.")
    parser.add_argument("--csv", default=FILE_PATH)
    parser.add_argument("--hours", type=float, default=8.0)
    parser.add_argument("--cache", default=None, help="optional .npz bar cache")
    args = parser.parse_args()

    print(f"Streaming {args.hours} hours from {args.csv} ...")
    dataset = build_from_csv(args.csv, hours=args.hours, cache=args.cache)
    print("✅ Sequence dataset built")
    print(dataset.summary())
    print(
        f"empty (trade-less) seconds filled: {dataset.meta['empty_seconds']:,} "
        f"of {dataset.meta['n_bars']:,}"
    )
