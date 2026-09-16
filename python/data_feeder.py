"""The shape of a raw Binance trades CSV, and one lazy reader over it.

The archives at https://data.binance.vision are headerless, so nothing in the
file says what its columns mean — the schema below *is* that knowledge, and it
lives here so `sequence_matrix.py` and `backtest.py` cannot disagree about it.

Everything here is deliberately thin. Folding trades into 1-second bars is
`sequence_matrix.load_second_bars`'s job, and pricing them is `backtest.py`'s;
this module only opens the file.
"""

from __future__ import annotations

import polars as pl

# Column order is fixed by the archive format, not chosen here. Types are given
# explicitly because Polars' inference would read `trade_id` as Int64 on one
# month and Float64 on another depending on where the values happen to land,
# which silently changes the dtype of every downstream array.
CSV_SCHEMA = {
    "trade_id": pl.Int64,  # unique transaction identifier
    "price": pl.Float64,  # execution price in USDT
    "qty": pl.Float64,  # amount of BTC traded
    "quote_qty": pl.Float64,  # value of the trade in USDT (price * qty)
    "time": pl.Int64,  # Unix timestamp; milliseconds in some months, micro in others
    "is_buyer_maker": pl.Boolean,  # True = the taker was selling, so sell pressure
    "is_best_match": pl.Boolean,  # legacy routing field, unused
}

# The month the scripts read when no `--csv` is given. A default, not a constant:
# every entry point takes a path.
FILE_PATH = "../data/BTCUSDT-trades-2026-05.csv"


def get_lazy_feeder(path: str) -> pl.LazyFrame:
    """A lazy scan of one monthly archive, with `time` cast to a datetime.

    Lazy rather than eager because a single month is ~9 GB of CSV: returning a
    `LazyFrame` lets the caller push its filters and aggregations *into* the scan,
    so Polars reads only the columns and rows that survive them. Collecting this
    without narrowing it first will exhaust memory, which is what
    `sequence_matrix.py`'s batched reader exists to avoid.

    Note the cast assumes microseconds. `sequence_matrix._time_divisor` is what
    detects the unit per file; callers that need that robustness go through it.
    """
    return pl.scan_csv(path, has_header=False, schema=CSV_SCHEMA).with_columns(
        pl.col("time").cast(pl.Datetime("us"))
    )
