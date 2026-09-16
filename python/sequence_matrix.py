"""The shared foundation: raw CSV -> 1-second bars -> windows, labels, channels.

**Everything downstream reads its data through this module**, which is the point.
The tabular models collapse each 5-minute lookback window into 7 scalars; PatchTST
consumes the *raw sequence*, the same 300 one-second bars as parallel univariate
channels. Because both take the bar grid, the window starts and the labels from
here, the two approaches are provably solving the identical problem on the
identical windows — which is what makes the comparison in README §5 a comparison
of models rather than of preprocessing.

Four things live here, in this order:

1. **Ingestion** (`load_second_bars`) — ticks folded into a complete 1-second
   grid, plus `save_bars` / `load_bars` / `concat_bars` for caching and splicing.
2. **Features** (`build_channels`, `build_tabular_features`) — the per-bar
   channels PatchTST reads, and the 7 window statistics the tabular models read.
3. **Labels** (`triple_barrier`, `build_targets`) — the vectorised triple barrier,
   and the two questions derived from it (which side, and whether either side).
4. **Splits** (`valid_window_starts`, `chronological_split`, `SequenceDataset`) —
   purged chronological splits, so training never sees its own test period.

This module deliberately depends on nothing but Polars and NumPy — no
`bipower_core`, no torch — so the exact same code runs inside a Google Colab
runtime, where the C++ extension is not compiled, as it does locally.

Bars are placed on a **complete** 1-second grid (empty seconds carry a
forward-filled price and zero volume), so a "60-bar" horizon is always literally
60 seconds. On BTC/USDT ~11% of seconds contain no trade at all, so a builder
that emits bars only for traded seconds — as `ml_matrix.py` originally did —
gives a horizon of an arbitrary wall-clock length. Since the vertical barrier
*is* the horizon, that would make the label itself depend on how busy the market
happened to be.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from data_feeder import CSV_SCHEMA

# --- Problem definition ------------------------------------------------------

# Half-width of the horizontal barriers, as a simple return, and the deadline the
# side must be decided by. These two are ONE choice, not two, and they were picked
# together by `sweep_barriers.py --geometry-only` on the training months alone.
#
# The criterion is the hit rate a side model must reach to break even,
#
#     h* = 1/2 (1 + c / (rho * G))
#
# where `c` is the round trip, `rho` the share of positions that reach a
# horizontal barrier rather than timing out, and `G` the magnitude captured when
# one is reached. See `backtest.required_hit_rate` for why the more familiar
# `B(2h-1) - c` is wrong and how badly.
#
# The project ran for a long time at 5 bp / 60 s, chosen because 5 bp is two taker
# fees. That target needed h* = 1.42 — literally unreachable — and the reason is
# not that 5 bp sat below its 6 bp round trip by a basis point. It is that a
# 60-second deadline caps `rho * G` at ~2.8 bp however the barrier is set: widen
# it and positions stop resolving as fast as the payoff grows. At 60 seconds
# every barrier from 5 bp to 100 bp needs h* > 0.88, gate or no gate.
#
# 60 bp / 3600 s minimises h* over a 48-cell grid, subject to the barrier clearing
# the round trip and at least 50 non-overlapping trades a day. Measured on
# Jan-Mar 2026: 36.5% of windows resolve, capturing 61.9 bp when they do, for
# h* = 0.550 on the realized-variance-gated population and 0.614 ungated.
# 40 bp / 3600 s is within 0.015 of that and was not chosen — the rule was fixed
# before the grid was run, and re-picking afterwards is how a target gets fitted.
BARRIER = 0.0060
HORIZON = 3600

# 5-minute lookback window, in 1-second bars. This is PatchTST's sequence length L.
# Unchanged when the horizon moved, so that the only difference between the old
# results and the new ones is the target. That leaves the lookback short relative
# to the deadline (1:12); `sweep_barriers.py --window 900` is the experiment that
# asks whether it matters, and §5 of the README reports what it found.
WINDOW_SIZE = 300

# pi / 2, the Bipower Variation scale factor (mirrors PI_FACTOR in math_engine.hpp).
PI_FACTOR = 1.5707963267948966

# Guards the volatility normalisation against flat windows (mirrors ml_matrix.py).
EPSILON = 1e-8

# The tabular feature set, in the order build_tabular_features stacks it.
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

# Per-bar quantities fed to the model as independent channels.
#
# "raw" is the default and the point of the sequence baseline: give the network
# the two primitive observables — the signed price increment and the signed
# traded volume — and let attention build whatever aggregate it wants out of
# them. `realized_var` and `bipower` are deterministic pointwise functions of
# `log_return` (r^2 and (pi/2)|r||r_prev|), so handing them over adds no
# information the model could not compute in its first layer; it only spends
# channel width. Two channels instead of six buys the depth back.
#
# "full" is the original six-channel layout, kept so checkpoints trained with it
# still load and score. Channels 1-3 are exactly the per-bar increments the C++
# engine sums into RV, BPV and OFI — verified numerically to ~1e-15 against
# `bipower_core`, up to the leading-edge terms the C++ loop skips.
CHANNEL_SETS = {
    "raw": [
        "log_return",  # r_t = log p_t - log p_{t-1}
        "ofi",  # signed traded volume    -> window sum = OFI
    ],
    "full": [
        "log_return",
        "realized_var",  # r_t^2                   -> window sum = RV
        "bipower",  # (pi/2) |r_t| |r_{t-1}|  -> window sum = BPV
        "ofi",
        "log_volume",  # log1p(total qty in the bar)
        "log_trades",  # log1p(number of trades in the bar)
    ],
}
DEFAULT_CHANNEL_SET = "raw"
CHANNEL_NAMES = CHANNEL_SETS[DEFAULT_CHANNEL_SET]
N_CHANNELS = len(CHANNEL_NAMES)

# Column layout of the "full" set, which `build_tabular_features` indexes into.
FULL_CHANNEL_NAMES = CHANNEL_SETS["full"]


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


def _row_batches(csv_path: Path, batch_size: int):
    """Yield fixed-size row batches of the four columns we consume.

    Polars renamed this capability mid-1.x: `LazyFrame.collect_batches` is the
    current spelling and `read_csv_batched` the deprecated one. Colab installs
    whatever is newest, so both are tried rather than pinning a version and
    hoping. The modern path also projects to four columns before reading, which
    is why it is preferred and not merely tolerated.
    """
    columns = ["time", "price", "qty", "is_buyer_maker"]
    lazy = pl.scan_csv(csv_path, has_header=False, schema=CSV_SCHEMA).select(columns)

    if hasattr(lazy, "collect_batches"):
        # `lazy=True` is load-bearing, not a micro-optimisation: the default
        # collects the entire result and *then* slices it into batches, which for
        # a 13 GB month is precisely the allocation this function exists to
        # avoid. Measured on one month: 1.3 GB streaming vs 5.3 GB eager.
        try:
            yield from lazy.collect_batches(chunk_size=batch_size, lazy=True)
        except TypeError:  # polars without the `lazy` keyword
            yield from lazy.collect_batches(chunk_size=batch_size)
        return

    reader = pl.read_csv_batched(  # polars < 1.35
        csv_path,
        has_header=False,
        new_columns=list(CSV_SCHEMA),
        schema_overrides=list(CSV_SCHEMA.values()),
        batch_size=batch_size,
    )
    while True:
        batches = reader.next_batches(1)
        if not batches:
            return
        for frame in batches:
            yield frame.select(columns)


def _aggregate_batched(
    csv_path: Path,
    divisor: int,
    start_raw: int,
    end_raw: int | None,
    batch_size: int,
) -> dict:
    """Fold a trades CSV into per-second bars with hard-bounded memory.

    Why this exists rather than one `group_by`: the lazy path materialises its
    aggregation through Polars' streaming engine, and whether that engine can
    actually stream `pl.col("price").sort_by("time").last()` depends on the
    Polars version. When it cannot, it silently falls back to collecting the
    whole frame — 185 M rows for the largest month here — and the process dies
    on a 12 GB runtime. That failure is version-dependent, invisible until it
    happens, and was what killed the Colab notebook.

    This path reads fixed-size row batches and scatters each into preallocated
    per-second accumulators, so peak memory is `O(seconds in the month)` plus one
    batch regardless of file size or Polars version. `np.bincount` does the sums;
    the last price in a second is simply the last batch row that writes to that
    slot, which is correct because Binance archives are ordered by trade id, and
    therefore by time.
    """
    n_slots = None
    price = qty = trades = ofi = None
    seen_any = False

    for frame in _row_batches(csv_path, batch_size):
        time_raw = frame["time"].to_numpy()

        keep = time_raw >= start_raw
        if end_raw is not None:
            keep &= time_raw < end_raw
        if not keep.any():
            # Archives are time-ordered, so once we are past the window we are done.
            if end_raw is not None and time_raw[0] >= end_raw:
                break
            continue

        seconds = (time_raw[keep] // divisor) - (start_raw // divisor)
        if n_slots is None:
            # The end of the requested window, or of the file, decides the size.
            span = (
                (end_raw - start_raw) // divisor + 1
                if end_raw is not None
                else int(seconds.max()) + 1
            )
            n_slots = int(span)
            price = np.full(n_slots, np.nan, dtype=np.float64)
            qty = np.zeros(n_slots, dtype=np.float64)
            trades = np.zeros(n_slots, dtype=np.float64)
            ofi = np.zeros(n_slots, dtype=np.float64)
        elif end_raw is None and seconds.max() >= n_slots:
            grow = int(seconds.max()) + 1
            price = np.concatenate([price, np.full(grow - n_slots, np.nan)])
            qty = np.concatenate([qty, np.zeros(grow - n_slots)])
            trades = np.concatenate([trades, np.zeros(grow - n_slots)])
            ofi = np.concatenate([ofi, np.zeros(grow - n_slots)])
            n_slots = grow

        seen_any = True
        batch_qty = frame["qty"].to_numpy()[keep]
        maker = frame["is_buyer_maker"].to_numpy()[keep]

        price[seconds] = frame["price"].to_numpy()[keep]  # last write per second wins
        qty += np.bincount(seconds, weights=batch_qty, minlength=n_slots)[:n_slots]
        trades += np.bincount(seconds, minlength=n_slots)[:n_slots]
        ofi += np.bincount(
            seconds, weights=np.where(maker, -batch_qty, batch_qty), minlength=n_slots
        )[:n_slots]
        del frame

    if not seen_any:
        raise ValueError(f"No rows in the requested time range of {csv_path}")

    # Trim trailing slots the file never reached, then forward-fill the gaps.
    observed_slots = np.flatnonzero(~np.isnan(price))
    last = int(observed_slots[-1]) + 1
    price, qty, trades, ofi = price[:last], qty[:last], trades[:last], ofi[:last]

    traded = int(observed_slots.size)
    filled = np.where(~np.isnan(price), np.arange(price.size), 0)
    np.maximum.accumulate(filled, out=filled)
    price = price[filled]

    first_second = start_raw // divisor
    return {
        "ts": np.arange(first_second, first_second + price.size, dtype=np.int64),
        "price": price,
        "qty": qty.astype(np.float32),
        "n_trades": trades.astype(np.float32),
        "ofi": ofi.astype(np.float32),
        "meta": {
            "source": csv_path.name,
            "n_bars": int(price.size),
            "first_ts": int(first_second),
            "last_ts": int(first_second + price.size - 1),
            "traded_seconds": traded,
            "empty_seconds": int(price.size - traded),
        },
    }


def load_second_bars(
    csv_path: str | Path,
    hours: float | None = None,
    skip_hours: float = 0.0,
    streaming: bool = True,
    chunk_hours: float | None = None,
    engine: str = "batched",
    batch_size: int = 4_000_000,
) -> dict:
    """Stream a raw Binance trades CSV into a complete 1-second bar series.

    One pass over the file turns ~72M ticks into ~2.7M bars (a month of BTC/USDT is
    ~110 MB in memory, ~30 MB compressed on disk), which is what makes repeated
    training runs cheap: build this once, cache it with `save_bars`, never touch the
    CSV again.

    Args:
        csv_path: path to the decompressed Binance `*-trades-*.csv`.
        hours: keep only the first N hours of the file (None = the whole file).
            Mainly for quick experiments; the study reads whole months.
        skip_hours: drop this many hours from the start before applying `hours`.
        streaming: use the Polars streaming engine (`engine="lazy"` only).
        chunk_hours: if set, aggregate in slices of this many hours instead of a
            single pass. Slower (one CSV scan per slice) but bounds memory hard.
        engine: `"batched"` (default) reads fixed-size row batches and scatters
            them into preallocated per-second accumulators, so peak memory is
            bounded by the *month*, not the file, whatever Polars decides to do.
            `"lazy"` is the original single `group_by`, kept because it is a
            useful cross-check — the two agree exactly.
        batch_size: rows per batch for `engine="batched"`. 4 M rows is ~200 MB.

    Returns:
        dict of NumPy arrays on a gap-free 1-second grid: `ts`, `price`, `qty`,
        `n_trades`, `ofi`, plus a `meta` dict.
    """
    csv_path = Path(csv_path)
    lazy = _scan_trades(csv_path)

    # A truncated or zero-byte download is the common way to get here, and Polars
    # reports it as an internal `NoDataError` rather than something that names the
    # file. Both routes to "this file has no ticks" get the same message.
    try:
        first_ts = lazy.select(pl.col("time").min()).collect().item()
    except pl.exceptions.NoDataError as empty:
        raise ValueError(f"No rows found in {csv_path}") from empty
    if first_ts is None:
        raise ValueError(f"No rows found in {csv_path}")
    divisor = _time_divisor(int(first_ts))

    start_raw = int(first_ts) + int(skip_hours * 3600 * divisor)
    end_raw = None if hours is None else start_raw + int(hours * 3600 * divisor)

    if engine == "batched":
        if chunk_hours is not None:
            raise ValueError(
                "chunk_hours applies to engine='lazy' only; the batched "
                "engine is already bounded by construction."
            )
        grid = _aggregate_batched(csv_path, divisor, start_raw, end_raw, batch_size)
        grid["meta"]["hours"] = hours
        grid["meta"]["skip_hours"] = skip_hours
        return grid
    if engine != "lazy":
        raise ValueError(f"engine must be 'batched' or 'lazy', got {engine!r}")

    def slice_of(lo: int, hi: int | None) -> pl.LazyFrame:
        window = lazy.filter(pl.col("time") >= lo)
        return window if hi is None else window.filter(pl.col("time") < hi)

    if chunk_hours is None:
        bars = _aggregate_to_seconds(slice_of(start_raw, end_raw), divisor, streaming)
    else:
        step = int(chunk_hours * 3600 * divisor)
        last_raw = (
            end_raw
            if end_raw is not None
            else int(lazy.select(pl.col("time").max()).collect().item()) + 1
        )
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
        # float32 for the volume-like series: they are quantities of order 1e0-1e3
        # that only ever get summed into float64 accumulators downstream, so the
        # extra 4 bytes a bar buys nothing and costs ~190 MB across six months.
        # `price` stays float64 — at $1e5 a float32 ULP is one tick, which is the
        # same size as the thing being predicted.
        out = np.zeros(grid.shape, dtype=np.float32)
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


def build_channels(bars: dict, channel_set: str = DEFAULT_CHANNEL_SET) -> np.ndarray:
    """Turn the bar series into the (n_bars, M) channel matrix PatchTST reads.

    `channel_set` selects a layout from `CHANNEL_SETS`; a list of channel names
    is also accepted, which is what lets a checkpoint rebuild the exact matrix it
    was trained on from its recorded `data_meta["channels"]`.
    """
    names = CHANNEL_SETS[channel_set] if isinstance(channel_set, str) else list(channel_set)

    # Built lazily and written straight into the output column, one at a time.
    # The obvious version — a dict of all six arrays, then column_stack — costs
    # ~0.8 GB of peak on six months to produce a 0.125 GB two-channel matrix,
    # because it materialises four float64 series nobody asked for and then
    # copies the lot. On a 12 GB Colab that headroom is worth having.
    def log_returns() -> np.ndarray:
        # r[0] = 0: the first bar has no predecessor, and every window that
        # actually gets used starts WINDOW_SIZE bars into the series anyway.
        log_p = np.log(bars["price"])
        return np.diff(log_p, prepend=log_p[0])

    def bipower_terms() -> np.ndarray:
        abs_r = np.abs(log_returns())
        out = np.empty_like(abs_r)
        out[0] = 0.0
        np.multiply(abs_r[1:], abs_r[:-1], out=out[1:])
        return np.multiply(out, PI_FACTOR, out=out)

    builders = {
        "log_return": log_returns,
        "realized_var": lambda: np.square(log_returns()),
        "bipower": bipower_terms,
        "ofi": lambda: bars["ofi"],
        "log_volume": lambda: np.log1p(bars["qty"]),
        "log_trades": lambda: np.log1p(bars["n_trades"]),
    }
    unknown = [n for n in names if n not in builders]
    if unknown:
        raise ValueError(f"Unknown channel(s) {unknown}; known: {sorted(builders)}")

    channels = np.empty((bars["price"].size, len(names)), dtype=np.float32)
    for col, name in enumerate(names):
        channels[:, col] = builders[name]()
    return channels


def build_tabular_features(
    channels: np.ndarray,
    price: np.ndarray,
    starts: np.ndarray,
    window: int = WINDOW_SIZE,
) -> np.ndarray:
    """The 7 features `ml_matrix.py` builds, derived from the same bar grid.

    This exists so the tabular baselines can be scored and timed on *exactly* the
    windows PatchTST sees — same bars, same grid, same splits, same labels.

    `channels` must be the **"full"** layout (`build_channels(bars, "full")`),
    since this indexes its `realized_var` / `bipower` / `ofi` columns by position.

    The window offsets reproduce the C++ loop precisely: `math_engine.hpp` starts
    its returns at i > 0 and its bipower pairs at i > 1, while OFI covers all bars.
    """
    if channels.shape[1] != len(FULL_CHANNEL_NAMES):
        raise ValueError(
            f"build_tabular_features needs the {len(FULL_CHANNEL_NAMES)}-column "
            f'"full" channel layout, got {channels.shape[1]} columns. '
            'Call build_channels(bars, "full").'
        )
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

    # float64, unlike the channel matrix: these columns span ~20 orders of
    # magnitude (RV is ~1e-7, OFI is ~1e2) and feed trees rather than a network,
    # so there is no reason to pay float32's rounding. 30 MB for a month.
    #
    # It still does not reproduce `bipower_core` bit for bit. Differencing a
    # 2.7M-term cumulative sum to recover a 300-term window sum cancels most of
    # the significant digits, leaving ~5e-8 relative error against the C++ loop's
    # direct summation. That is invisible in the features and just visible in the
    # result: XGBoost's split decisions amplify it into ~0.002 of test AUC, well
    # inside the bootstrap interval but enough that this path and `ml_matrix.py`
    # print different third decimals. This NumPy path is the one `walkforward.py`
    # runs, so it is the one the README quotes; `ml_matrix.py` exists to check it.
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
    )


def rolling_realized_variance(
    price: np.ndarray, starts: np.ndarray, window: int = WINDOW_SIZE
) -> np.ndarray:
    """Column 0 of `build_tabular_features`, without building the other six.

    This is the untrained gate: a rolling sum of squared 1-second log returns over
    the lookback. `build_tabular_features` computes it as part of the seven, from
    a shared set of cumulative sums; a caller that wants only the gate — a barrier
    sweep scoring dozens of (barrier, horizon) cells, say — would otherwise pay
    for a six-column channel matrix and a seven-column feature matrix to read one
    of them. The offsets mirror that function exactly, including the `+ 1` that
    matches the C++ loop's `i > 0` start, and a test asserts the two agree.
    """
    log_p = np.log(np.asarray(price, dtype=np.float64))
    squared = np.square(np.diff(log_p, prepend=log_p[0]))
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    return cumulative[starts + window] - cumulative[starts + 1]


# The two questions the project asks, and the only two labels it builds. They
# come from the *same* triple-barrier pass: the side label reads which barrier
# was touched, the gate label reads whether either was.
LABEL_MODES = ("triple_barrier", "barrier_touched")


def triple_barrier(
    price: np.ndarray, horizon: int = HORIZON, barrier: float = BARRIER
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lopez de Prado's triple barrier, vectorised over every bar at once.

    From each bar `i`, walk the next `horizon` bars and record which of three
    barriers the path touches first:

        upper  (profit) : log p_{i+k} - log p_i >  log(1 + barrier)
        lower  (loss)   : log p_{i+k} - log p_i < -log(1 + barrier)
        vertical (time) : neither, within k <= horizon

    Returns `(side, exit_offset, defined)`:

    * `side` — int8, `+1` upper first, `-1` lower first, `0` vertical barrier.
    * `exit_offset` — int32, bars from `i` until the position closes: the first
      touch, or `horizon` when the vertical barrier ends it. This is what a
      backtest needs and what a plain label cannot supply, since holding period
      is what turns a hit rate into a Sharpe.
    * `defined` — bool, False for the last `horizon` bars, whose forward path is
      truncated by the end of the series and whose label would be drawn from
      partial information.

    Cost is `horizon` vectorised passes over the series rather than a per-bar
    loop — ~0.3 s for a month of 1-second bars, against hours for the naive form.
    """
    log_p = np.log(np.asarray(price, dtype=np.float64))
    n = log_p.size
    theta = np.log1p(barrier)

    never = np.int32(horizon + 1)
    first_up = np.full(n, never, dtype=np.int32)
    first_down = np.full(n, never, dtype=np.int32)

    # Descending k, so the last write to any bar is its *smallest* touching k.
    for k in range(horizon, 0, -1):
        forward = log_p[k:] - log_p[:-k]
        first_up[: n - k][forward > theta] = k
        first_down[: n - k][forward < -theta] = k

    side = np.zeros(n, dtype=np.int8)
    side[first_up < first_down] = 1
    side[first_down < first_up] = -1

    exit_offset = np.minimum(np.minimum(first_up, first_down), horizon).astype(np.int32)

    defined = np.ones(n, dtype=bool)
    defined[max(n - horizon, 0) :] = False
    side[~defined] = 0
    return side, exit_offset, defined


def triple_barrier_labels(
    price: np.ndarray, horizon: int = HORIZON, barrier: float = BARRIER
) -> np.ndarray:
    """Just the side, for callers that do not need exit timing."""
    return triple_barrier(price, horizon=horizon, barrier=barrier)[0]


def build_targets(
    bars: dict,
    horizon: int = HORIZON,
    barrier: float = BARRIER,
    label_mode: str = "triple_barrier",
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar `(y, usable)`, where `y[i]` labels a window *ending* on bar i.

    `label_mode="triple_barrier"` (default) asks **which side gets touched
    first**: `y = 1` if the upper barrier is hit before the lower one, `y = 0` if
    the lower comes first, and `usable = False` where neither is reached inside
    the horizon. Both classes therefore require the *same* `barrier`-sized move,
    which is what stops a volatility forecast from scoring on the label: magnitude
    is constant across the two classes by construction, so only sign is left to
    predict. The price is that the windows resolving on the vertical barrier drop
    out of the population — at the current 60 bp / 3600 s target, 63% of them.

    `label_mode="barrier_touched"` is the **gate**: `y = 1` if *either* horizontal
    barrier is reached inside the horizon, defined on every window rather than a
    subset. This is the volatility question, and it is deliberately the easy one —
    pairing a gate that predicts *whether* a window is tradeable with a side model
    that predicts *which way* is what lets the pair be evaluated on the whole
    population instead of on a subset chosen with hindsight. `walkforward.py` is
    what fits the pair and scores them together.

    Entries within `horizon` of the end are never usable; no window returned by
    `valid_window_starts` labels against them in any case.
    """
    price = bars["price"]

    if label_mode == "triple_barrier":
        side, _, defined = triple_barrier(price, horizon=horizon, barrier=barrier)
        return (side > 0).astype(np.int8), (side != 0) & defined

    if label_mode == "barrier_touched":
        side, _, defined = triple_barrier(price, horizon=horizon, barrier=barrier)
        return (side != 0).astype(np.int8), defined

    raise ValueError(f"label_mode must be one of {LABEL_MODES}, got {label_mode!r}")


def valid_window_starts(
    n_bars: int, window: int = WINDOW_SIZE, horizon: int = HORIZON
) -> np.ndarray:
    """Start indices of every window with a full lookback and a defined label.

    Window k covers bars [k, k + window - 1] and is labelled by the forward move
    out of its **last** bar — the same alignment `ml_matrix.py` produces.
    """
    n_windows = n_bars - window - horizon + 1
    if n_windows <= 0:
        raise ValueError(f"Need at least {window + horizon} bars to form one window, got {n_bars}")
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
        channels = self.meta.get("channels", CHANNEL_NAMES)
        mode = self.meta.get("label_mode", "triple_barrier")
        lines = [
            f"bars: {self.channels.shape[0]:,} x {self.channels.shape[1]} channels "
            f"({', '.join(channels)})",
            f"window: {self.window} bars | horizon: {self.horizon} bars | "
            f"label: {mode} at +/-{self.meta.get('barrier', BARRIER):.4%}",
        ]
        if mode == "triple_barrier":
            lines.append(
                f"vertical-barrier windows dropped: "
                f"{1.0 - self.meta.get('touch_rate', float('nan')):.2%} of the population"
            )
        for name in ("train", "val", "test"):
            if name in self.splits:
                n = self.splits[name].size
                lines.append(
                    f"{name:>5}: {n:>9,} windows | positive rate {self.positive_rate(name):.2%}"
                )
        return "\n".join(lines)


def build_sequence_dataset(
    bars: dict,
    window: int = WINDOW_SIZE,
    horizon: int = HORIZON,
    barrier: float = BARRIER,
    label_mode: str = "triple_barrier",
    channel_set: str | list[str] = DEFAULT_CHANNEL_SET,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    train_stride: int = 1,
    val_stride: int = 1,
) -> SequenceDataset:
    """Assemble channels, labels and purged chronological splits from bar data.

    Order of operations matters and is deliberate: windows are split in time,
    then purged at the boundaries, then thinned by stride, and only then filtered
    down to the ones the triple barrier actually labels. Splitting first keeps
    the boundaries at fixed points in time regardless of how many windows survive
    the filter, so the purge argument (`window + horizon - 1`) stays exact.
    """
    channels = build_channels(bars, channel_set)
    y, usable = build_targets(bars, horizon=horizon, barrier=barrier, label_mode=label_mode)
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

    # A window is labelled by the bar it ends on, so that is where `usable` is read.
    splits = {name: s[usable[s + window - 1]] for name, s in splits.items()}
    empty = [name for name, s in splits.items() if s.size == 0]
    if empty:
        raise ValueError(
            f"Split(s) {empty} have no labelled windows left. With "
            f"label_mode={label_mode!r} and barrier={barrier:.4%}, no path in that "
            "period touched a horizontal barrier — widen the horizon, narrow the "
            "barrier, or stream more tape."
        )

    channel_names = CHANNEL_SETS[channel_set] if isinstance(channel_set, str) else list(channel_set)
    meta = dict(bars.get("meta", {}))
    meta.update(
        {
            "window": window,
            "horizon": horizon,
            "barrier": barrier,
            "label_mode": label_mode,
            "channels": list(channel_names),
            "channel_set": channel_set if isinstance(channel_set, str) else "custom",
            "train_frac": train_frac,
            "val_frac": val_frac,
            "train_stride": train_stride,
            "val_stride": val_stride,
            "purge": purge,
            "n_windows": int(starts.size),
            "touch_rate": float(usable[starts + window - 1].mean()),
            "split_sizes": {k: int(v.size) for k, v in splits.items()},
        }
    )
    return SequenceDataset(
        channels=channels, y=y, splits=splits, window=window, horizon=horizon, meta=meta
    )


# --- Evaluation helper -------------------------------------------------------
#
# Lives here rather than in `patchtst_model.py` because it must be importable
# without torch: the tabular trainers need it, and so does a Colab cell running
# before the model is built. It is NumPy-only for the same reason as the rest of
# this module.


def block_bootstrap_auc(
    y_true: np.ndarray,
    score: np.ndarray,
    n_boot: int = 400,
    block: int = WINDOW_SIZE + HORIZON,
    seed: int = 7,
) -> dict:
    """Moving-block bootstrap confidence interval for ROC-AUC.

    An i.i.d. bootstrap is wrong here and flatteringly so. Consecutive windows
    share 299 of their 300 bars and their labels are driven by overlapping
    forward paths, so resampling single windows treats ~360 correlated
    observations as 360 independent ones and returns an interval several times
    too narrow. Resampling contiguous blocks of `window + horizon` windows keeps
    each block internally intact, so the interval reflects the number of
    genuinely independent episodes in the split rather than its row count.

    Returns the interval plus `p_le_half`, the share of resamples at or below
    0.5 — the number to read when asking whether an edge exists at all.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score)
    n = y_true.size
    if n <= block:
        raise ValueError(f"Need more than {block} windows to block-bootstrap, got {n}")

    rng = np.random.default_rng(seed)
    n_blocks = max(1, n // block)
    offsets = np.arange(block)
    samples = np.empty(n_boot)
    for i in range(n_boot):
        picks = rng.integers(0, n - block, n_blocks)
        idx = (picks[:, None] + offsets).ravel()
        truth = y_true[idx]
        # A resample that happens to be single-class leaves AUC undefined.
        samples[i] = _roc_auc(truth, score[idx]) if 0 < truth.sum() < truth.size else np.nan

    samples = samples[~np.isnan(samples)]
    return {
        "lo": float(np.percentile(samples, 2.5)),
        "hi": float(np.percentile(samples, 97.5)),
        "p_le_half": float((samples <= 0.5).mean()),
        "n_boot": int(samples.size),
        "block": int(block),
    }


def _roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, score))


def dataset_kwargs_from(data_meta: dict) -> dict:
    """Recover the dataset recipe a checkpoint recorded, for `build_sequence_dataset`.

    Every consumer of a checkpoint needs this, and getting a default wrong here
    means silently scoring a model against a window population it never saw — a
    failure with no symptom, since the wrong population still produces a number.
    Every default below is therefore this module's own current default, so a key
    the checkpoint omitted resolves to what the code would have used anyway.
    """
    return {
        "window": data_meta.get("window", WINDOW_SIZE),
        "horizon": data_meta.get("horizon", HORIZON),
        "barrier": data_meta.get("barrier", BARRIER),
        "label_mode": data_meta.get("label_mode", "triple_barrier"),
        "channel_set": data_meta.get("channels", DEFAULT_CHANNEL_SET),
        "train_frac": data_meta.get("train_frac", 0.7),
        "val_frac": data_meta.get("val_frac", 0.1),
        "train_stride": data_meta.get("train_stride", 1),
        "val_stride": data_meta.get("val_stride", 1),
    }


def concat_bars(parts: list[dict]) -> dict:
    """Splice consecutive bar series into one continuous 1-second grid.

    Months arrive as separate archives but the market does not restart between
    them, so a walk-forward run over six months wants one series, not six. Parts
    are sorted by their first timestamp and any gap between them is filled the
    same way a trade-less second inside a month is: forward-filled price, zero
    volume, zero flow. Overlaps are an error rather than something to silently
    de-duplicate — they mean the same tape was passed twice.
    """
    if not parts:
        raise ValueError("concat_bars needs at least one bar series")
    if len(parts) == 1:
        return parts[0]

    parts = sorted(parts, key=lambda b: int(b["ts"][0]))
    for earlier, later in zip(parts, parts[1:]):
        if int(later["ts"][0]) <= int(earlier["ts"][-1]):
            raise ValueError(
                f"Bar series overlap: one ends at {earlier['ts'][-1]}, the next starts "
                f"at {later['ts'][0]}. Passing the same month twice would double-count it."
            )

    first, last = int(parts[0]["ts"][0]), int(parts[-1]["ts"][-1])
    grid = np.arange(first, last + 1, dtype=np.int64)
    out = {
        "ts": grid,
        "price": np.full(grid.size, np.nan, dtype=np.float64),
        "qty": np.zeros(grid.size, dtype=np.float32),
        "n_trades": np.zeros(grid.size, dtype=np.float32),
        "ofi": np.zeros(grid.size, dtype=np.float32),
    }
    for part in parts:
        lo = int(part["ts"][0]) - first
        hi = lo + part["ts"].size
        for key in ("price", "qty", "n_trades", "ofi"):
            out[key][lo:hi] = part[key]

    # Carry the last observed price across any inter-part gap.
    missing = np.isnan(out["price"])
    if missing.any():
        observed = np.where(~missing, np.arange(out["price"].size), 0)
        np.maximum.accumulate(observed, out=observed)
        out["price"] = out["price"][observed]
        if np.isnan(out["price"][0]):
            raise ValueError("The earliest bar series starts with no price")

    sources = [p.get("meta", {}).get("source", "?") for p in parts]
    out["meta"] = {
        "source": " + ".join(sources),
        "sources": sources,
        "n_bars": int(grid.size),
        "first_ts": first,
        "last_ts": last,
        "traded_seconds": int(sum(p.get("meta", {}).get("traded_seconds", 0) for p in parts)),
        "empty_seconds": int(
            grid.size - sum(p.get("meta", {}).get("traded_seconds", 0) for p in parts)
        ),
        "gap_seconds": int(missing.sum()),
        "hours": None,
        "skip_hours": 0.0,
        "part_bounds": [[int(p["ts"][0]), int(p["ts"][-1])] for p in parts],
    }
    return out


def load_bar_caches(paths: list[str | Path]) -> dict:
    """Load several `save_bars` caches and splice them into one series.

    Loads one month at a time and drops it as soon as it has been copied into the
    output, rather than holding all of them alongside the result. On six months
    that is the difference between ~1.6 GB of peak and ~1.0 GB — worth having on
    a 12 GB Colab runtime, where this runs immediately before the model does.
    """
    paths = list(paths)
    if len(paths) == 1:
        return load_bars(paths[0])

    # Two cheap passes over metadata first, so the output can be preallocated and
    # the parts never coexist.
    spans = []
    for path in paths:
        with np.load(path, allow_pickle=False) as raw:
            ts = raw["ts"]
            spans.append((int(ts[0]), int(ts[-1]), path))
    spans.sort()
    for (_, earlier_end, a), (later_start, _, b) in zip(spans, spans[1:]):
        if later_start <= earlier_end:
            raise ValueError(
                f"Bar series overlap: {Path(a).name} ends at {earlier_end}, "
                f"{Path(b).name} starts at {later_start}. Passing the same month "
                "twice would double-count it."
            )

    first, last = spans[0][0], spans[-1][1]
    grid = np.arange(first, last + 1, dtype=np.int64)
    out = {
        "ts": grid,
        "price": np.full(grid.size, np.nan, dtype=np.float64),
        "qty": np.zeros(grid.size, dtype=np.float32),
        "n_trades": np.zeros(grid.size, dtype=np.float32),
        "ofi": np.zeros(grid.size, dtype=np.float32),
    }
    sources, traded, bounds = [], 0, []
    for start, end, path in spans:
        part = load_bars(path)
        lo = start - first
        for key in ("price", "qty", "n_trades", "ofi"):
            out[key][lo : lo + part["ts"].size] = part[key]
        sources.append(part.get("meta", {}).get("source", Path(path).name))
        traded += int(part.get("meta", {}).get("traded_seconds", 0))
        bounds.append([start, end])
        del part

    missing = np.isnan(out["price"])
    gaps = int(missing.sum())
    if gaps:
        observed = np.where(~missing, np.arange(out["price"].size), 0)
        np.maximum.accumulate(observed, out=observed)
        out["price"] = out["price"][observed]
        del observed
        if np.isnan(out["price"][0]):
            raise ValueError("The earliest bar series starts with no price")
    del missing

    out["meta"] = {
        "source": " + ".join(sources),
        "sources": sources,
        "n_bars": int(grid.size),
        "first_ts": first,
        "last_ts": last,
        "traded_seconds": traded,
        "empty_seconds": int(grid.size - traded),
        "gap_seconds": gaps,
        "hours": None,
        "skip_hours": 0.0,
        "part_bounds": bounds,
    }
    return out


def load_or_build_bars(
    csv_path: str | Path,
    hours: float | None = None,
    skip_hours: float = 0.0,
    cache: str | Path | None = None,
) -> dict:
    """Bars for the requested slice, from `cache` if it holds that exact slice.

    The mismatch check is the point of this function. Reusing a month-long cache
    for a `--hours 8` request would report a full-month result under an 8-hour
    label, which is the one failure mode that produces a plausible-looking wrong
    number instead of an error. Every entry point that accepts both `--hours` and
    a cache path goes through here.
    """
    if cache is not None and Path(cache).exists():
        bars = load_bars(cache)
        cached = (bars["meta"].get("hours"), bars["meta"].get("skip_hours", 0.0))
        if cached != (hours, skip_hours):
            raise ValueError(
                f"Bar cache {cache} covers hours={cached[0]}, skip_hours={cached[1]}, "
                f"but hours={hours}, skip_hours={skip_hours} was requested. "
                "Delete the cache, point at a different file, or drop the --hours flag."
            )
        return bars

    bars = load_second_bars(csv_path, hours=hours, skip_hours=skip_hours)
    if cache is not None:
        save_bars(bars, cache)
    return bars
