"""Shared fixtures, and the macOS OpenMP mitigation the whole suite depends on.

Everything here is synthetic and deterministic. The real tape is 52 GB and lives
outside the repository, so a test that needed it could not run in CI — and a test
that *did* run against it would be checking a market rather than a function. The
series built here are small enough to reason about by hand and seeded so a
failure reproduces.

`python/` is put on the import path by `pythonpath` in `pyproject.toml`, not by a
`sys.path` mutation here.
"""

from __future__ import annotations

# --- macOS OpenMP: import order is load-bearing -------------------------------
# torch and xgboost each ship their own libomp on macOS, and mixing them in one
# process deadlocks or segfaults. `benchmark_inference.py` documents the measured
# matrix; the short version is that xgboost must load first *and* torch must stay
# single-threaded. Neither half alone is enough.
#
# This belongs in conftest rather than in each test module because pytest imports
# conftest before it collects anything, which is the only hook that can fix the
# order for a suite whose modules import torch and xgboost in whatever sequence
# their own dependencies happen to need. Without it the walk-forward integration
# tests take the interpreter down with a SIGSEGV on macOS runners.
import xgboost  # noqa: F401  — must precede torch; see above

import torch  # noqa: E402

torch.set_num_threads(1)

from datetime import datetime, UTC  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

# 2026-01-01 00:00:00 UTC, the first second of the real dataset's first month.
EPOCH = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp())


def make_bars(
    n_bars: int = 2_000,
    start_ts: int = EPOCH,
    seed: int = 0,
    volatility: float = 5e-5,
    start_price: float = 70_000.0,
    empty_every: int | None = None,
) -> dict:
    """A bar dict shaped exactly like `sequence_matrix.load_bars` returns.

    A geometric random walk on a gap-free 1-second grid. `volatility` is the
    per-bar log-return standard deviation, and the default is calibrated rather
    than arbitrary: it puts ~34% of 60-second windows through a 5 bp barrier,
    against 39.7% in the real six months. That matters because both degenerate
    regimes hide real bugs — at high volatility every window resolves, so the
    vertical barrier is never exercised and overshoot swamps the barrier width;
    at low volatility almost nothing resolves and the side label is nearly empty.

    `empty_every` marks every n-th bar trade-less — forward-filled price, zero
    volume, zero flow — which is how `_to_regular_grid` represents the ~11% of
    seconds in the real tape that carry no trade.
    """
    rng = np.random.default_rng(seed)

    log_returns = rng.normal(0.0, volatility, n_bars)
    log_returns[0] = 0.0
    price = start_price * np.exp(np.cumsum(log_returns))

    qty = rng.gamma(2.0, 0.5, n_bars).astype(np.float32)
    n_trades = rng.poisson(45, n_bars).astype(np.float32)
    # OFI is signed net volume, so it straddles zero rather than being positive.
    ofi = (rng.normal(0.0, 1.0, n_bars) * qty).astype(np.float32)

    if empty_every:
        blank = np.zeros(n_bars, dtype=bool)
        blank[::empty_every] = True
        blank[0] = False  # the first bar must carry a price to forward-fill from
        qty[blank] = 0.0
        n_trades[blank] = 0.0
        ofi[blank] = 0.0
        # Forward-fill the price across the trade-less seconds.
        observed = np.where(~blank, np.arange(n_bars), 0)
        np.maximum.accumulate(observed, out=observed)
        price = price[observed]

    ts = np.arange(start_ts, start_ts + n_bars, dtype=np.int64)
    traded = int(n_bars - (blank.sum() if empty_every else 0))
    return {
        "ts": ts,
        "price": price,
        "qty": qty,
        "n_trades": n_trades,
        "ofi": ofi,
        "meta": {
            "source": f"synthetic(seed={seed})",
            "n_bars": n_bars,
            "first_ts": int(ts[0]),
            "last_ts": int(ts[-1]),
            "traded_seconds": traded,
            "empty_seconds": n_bars - traded,
        },
    }


@pytest.fixture
def bars() -> dict:
    """2,000 bars: enough for windows at the test sizes, small enough to be fast."""
    return make_bars()


@pytest.fixture
def long_bars() -> dict:
    """Enough bars to form windows at the production window/horizon (300/60)."""
    return make_bars(n_bars=5_000, seed=1)


@pytest.fixture
def bars_factory():
    """`make_bars` itself, for tests that need several series or custom shapes."""
    return make_bars
