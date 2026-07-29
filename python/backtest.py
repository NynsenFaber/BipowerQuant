"""
Event-driven backtest for triple-barrier signals, with realistic execution.

An ROC-AUC says how well a model *ranks* windows. It says nothing about whether
the ranking survives fees, the spread, slippage, or the fact that a signal you
cannot get filled on is worth nothing. This module answers the question AUC
cannot: **what Sharpe does the flagged subset actually earn?**

What is modelled
----------------
* **One position at a time.** Windows advance every second, so a naive backtest
  that opens a position on every flagged window counts the same price move
  hundreds of times and reports a Sharpe inflated by roughly the square root of
  that overlap. Signals arriving while a position is open are dropped.
* **Fees**, per side, taker and maker separately.
* **The spread.** Estimated from the tape with the Roll (1984) serial-covariance
  estimator rather than assumed, since on BTC/USDT it turns out to be small
  enough that assuming a "typical" value would dominate the result.
* **Slippage** on marketable orders, as an explicit per-side parameter.
* **Queue position** for passive entries: a limit order joins a FIFO queue and
  only fills once the volume ahead of it trades. This is what makes maker
  execution *not* free — you are filled precisely when the market is coming
  toward you, and not filled when it runs away.
* **Overshoot.** Barriers are detected on a 1-second grid, so the exit price is
  the actual price at the touching bar, not the barrier level. Stops therefore
  pay the gap through the level, which is the realistic direction of the error.

What is not modelled
--------------------
No L2 book. Binance's public trade archive has no depth, so the volume queued
ahead of a passive order is unobservable and is a **parameter**
(`queue_ahead_btc`), not a measurement. Treat the maker results as a sensitivity
analysis over that parameter rather than as a prediction, and read the taker
results as the number that does not depend on an assumption.

Partial fills, funding, latency, exchange downtime and market impact beyond the
slippage term are all out of scope.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np

import sequence_matrix as seq

BPS = 1e-4
SECONDS_PER_DAY = 86400
TRADING_DAYS = 365  # crypto trades continuously; no 252-day convention here


# --- cost and execution assumptions ------------------------------------------


@dataclass
class Costs:
    """Per-side trading costs, in basis points of notional.

    Defaults are a Binance spot VIP tier: 2.5 bp taker, 0 bp maker. The project's
    5 bp "round trip" is exactly `2 x taker_fee_bps`, which is worth staring at —
    see `breakeven_barrier_bps`.
    """

    taker_fee_bps: float = 2.5
    maker_fee_bps: float = 0.0
    half_spread_bps: float | None = None  # None -> estimate from the tape
    slippage_bps: float = 0.5  # extra adverse move on a marketable order

    def round_trip_bps(self, entry: str, exit_: str) -> float:
        def side(kind: str) -> float:
            if kind == "maker":
                return self.maker_fee_bps
            return self.taker_fee_bps + (self.half_spread_bps or 0.0) + self.slippage_bps

        return side(entry) + side(exit_)


@dataclass
class Execution:
    """How orders reach the market."""

    entry: str = "taker"  # "taker" | "maker"
    exit: str = "taker"  # stops must be marketable; a target could be passive
    queue_ahead_btc: float = 1.0  # volume queued ahead of a passive order
    max_wait_s: int = 5  # cancel an unfilled limit after this many seconds
    size_btc: float = 0.1  # our own order size, added to the queue we must clear


@dataclass
class BacktestResult:
    trades: dict  # per-trade arrays
    summary: dict
    daily: dict

    def format(self) -> str:
        s = self.summary
        lines = [
            f"trades              {s['n_trades']:>12,}   "
            f"(from {s['n_signals']:,} signals; {s['n_blocked']:,} blocked by an open "
            f"position, {s['n_unfilled']:,} unfilled)",
            f"hit rate            {s['hit_rate']:>12.2%}   (all trades)",
            f"  among resolved    {s['resolved_hit_rate']:>12.2%}   "
            f"({s['resolved_share']:.1%} reached a horizontal barrier)",
            f"gross per trade     {s['gross_bps']:>12.3f} bp",
            f"cost per trade      {s['cost_bps']:>12.3f} bp",
            f"net per trade       {s['net_bps']:>12.3f} bp",
            f"mean holding time   {s['mean_hold_s']:>12.1f} s",
            f"total net return    {s['total_return']:>12.2%}",
            f"max drawdown        {s['max_drawdown']:>12.2%}",
            f"Sharpe (annualised) {s['sharpe']:>12.2f}",
            f"days traded         {s['n_days']:>12,}",
        ]
        if s.get("sharpe_ci"):
            lo, hi = s["sharpe_ci"]
            lines.append(f"Sharpe 95% CI       [{lo:>6.2f}, {hi:>6.2f}]")
        return "\n".join(lines)


# --- the spread ---------------------------------------------------------------


TICK_SIZE_USDT = 0.01  # BTCUSDT spot price increment on Binance


def estimate_half_spread_bps(price: np.ndarray, block: int = SECONDS_PER_DAY) -> dict:
    """Effective half-spread in bp, with the method that produced it.

    Two estimators, because on this instrument the first one usually fails and
    silently returning 0 would understate costs:

    * **Roll (1984).** If trades alternate between bid and ask around an
      efficient price, consecutive price changes acquire a negative serial
      covariance of `-s^2/4`, so `s = 2*sqrt(-cov)`. It needs no quote data,
      which is the point — the archive has none.
    * **Tick floor.** Half of one price increment, `0.01 USDT`. This is the
      smallest spread the exchange can quote, so it is a hard lower bound.

    On BTC/USDT the Roll estimator almost always returns nothing usable: at
    ~$77,000 one tick is **0.0013 bp**, several orders of magnitude below the
    1-second return noise, so the bid-ask bounce is invisible and the daily
    serial covariance comes out positive (trend, not bounce). The honest
    reading is not "the spread is zero" but "the spread is at the tick floor and
    too small to measure this way" — which is the answer anyway, since 0.0013 bp
    against a 2.5 bp fee is a rounding error.
    """
    log_p = np.log(np.asarray(price, dtype=np.float64))
    estimates = []
    for lo in range(0, max(log_p.size - block, 0), block):
        diffs = np.diff(log_p[lo : lo + block])
        diffs = diffs[diffs != 0.0]  # trade-less seconds would dilute the covariance
        if diffs.size < 100:
            continue
        cov = np.cov(diffs[:-1], diffs[1:])[0, 1]
        if cov < 0:
            estimates.append(2.0 * np.sqrt(-cov))

    tick_floor = 0.5 * TICK_SIZE_USDT / float(np.median(price)) / BPS
    if not estimates:
        return {"half_spread_bps": tick_floor, "method": "tick floor (Roll unresolvable)",
                "roll_blocks": 0, "tick_floor_bps": tick_floor}

    roll = float(np.median(estimates)) / 2.0 / BPS
    return {"half_spread_bps": max(roll, tick_floor),
            "method": "Roll" if roll >= tick_floor else "tick floor (Roll below it)",
            "roll_blocks": len(estimates), "tick_floor_bps": tick_floor}


def roll_half_spread_bps(price: np.ndarray, block: int = SECONDS_PER_DAY) -> float:
    """Just the number, for callers that do not need the provenance."""
    return estimate_half_spread_bps(price, block)["half_spread_bps"]


def breakeven_barrier_bps(costs: Costs, execution: Execution) -> float:
    """Barrier width at which a *perfect* predictor earns exactly zero.

    A winning trade captures `barrier` and pays the round trip, so the strategy
    cannot make money at any accuracy unless `barrier > round_trip`. This is a
    property of the target definition, not of the model, and it is the first
    thing to check before reading any Sharpe.
    """
    return costs.round_trip_bps(execution.entry, execution.exit)


# --- passive fills ------------------------------------------------------------


def _maker_fill(
    price: np.ndarray,
    ofi: np.ndarray,
    bar: int,
    direction: int,
    execution: Execution,
) -> tuple[int, float] | None:
    """Simulate a passive limit order joining the queue at the current price.

    A long posts a bid at the current price and needs *aggressive sellers* to
    lift the queue in front of it; a short posts an ask and needs aggressive
    buyers. `ofi` carries net signed volume per second, so the relevant flow is
    its negative part for a bid and its positive part for an ask.

    Three outcomes, in the order they are checked each second:

    1. **Traded through.** The price moves past our limit, which means everything
       resting at our level was consumed — including us. Filled at our limit,
       and immediately on the wrong side of the move. This is adverse selection,
       and it is the cost that makes maker execution not free.
    2. **Queue cleared.** Enough same-direction aggressive volume arrived at our
       level. Filled at our limit.
    3. **Timed out or ran away.** Cancelled; the signal is discarded rather than
       chased, since chasing turns a maker fill into a taker fill.

    Returns `(fill_bar, fill_price)` or None.
    """
    limit_price = price[bar]
    needed = execution.queue_ahead_btc + execution.size_btc
    filled_ahead = 0.0

    for offset in range(1, execution.max_wait_s + 1):
        t = bar + offset
        if t >= price.size:
            return None

        if direction > 0:
            if price[t] < limit_price:  # traded through our bid
                return t, limit_price
            if price[t] > limit_price:  # market ran away
                return None
            filled_ahead += max(-ofi[t], 0.0)  # aggressive sells hitting the bid
        else:
            if price[t] > limit_price:  # traded through our ask
                return t, limit_price
            if price[t] < limit_price:
                return None
            filled_ahead += max(ofi[t], 0.0)  # aggressive buys lifting the ask

        if filled_ahead >= needed:
            return t, limit_price

    return None


# --- the simulation -----------------------------------------------------------


def barrier_arrays(
    price: np.ndarray, horizon: int = seq.HORIZON, barrier: float = seq.BARRIER
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`seq.triple_barrier`, exposed so a sweep can compute it once.

    Each call is 60 vectorised passes over the whole series — a couple of seconds
    on six months. A strategy grid evaluates dozens of configurations against the
    *same* barrier, so recomputing it inside `simulate` made the grid quadratic
    in nothing useful.
    """
    return seq.triple_barrier(price, horizon=horizon, barrier=barrier)


def simulate(
    bars: dict,
    entry_bars: np.ndarray,
    direction: np.ndarray,
    barrier: float = seq.BARRIER,
    horizon: int = seq.HORIZON,
    costs: Costs | None = None,
    execution: Execution | None = None,
    bootstrap: int = 400,
    seed: int = 7,
    precomputed: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> BacktestResult:
    """Run the signals through the execution model and price the result.

    Args:
        bars: the bar dict from `sequence_matrix`.
        entry_bars: bar indices at which a signal fires, ascending.
        direction: +1 to go long, -1 to go short, one per entry.
        barrier / horizon: the triple barrier the position is managed against.
        costs / execution: assumptions; defaults are documented on the classes.
        precomputed: `(side, exit_offset, defined)` from `barrier_arrays`, to
            avoid recomputing them for every configuration in a sweep.

    Positions never overlap: a signal arriving before the previous position has
    closed is dropped, and counted in `n_blocked`.
    """
    costs = costs or Costs()
    execution = execution or Execution()
    price, ts, ofi = bars["price"], bars["ts"], bars["ofi"]

    if costs.half_spread_bps is None:
        costs = Costs(**{**asdict(costs), "half_spread_bps": roll_half_spread_bps(price)})

    side, exit_offset, defined = (
        precomputed if precomputed is not None
        else barrier_arrays(price, horizon=horizon, barrier=barrier)
    )

    entry_bars = np.asarray(entry_bars, dtype=np.int64)
    direction = np.asarray(direction, dtype=np.int64)
    order = np.argsort(entry_bars, kind="stable")
    entry_bars, direction = entry_bars[order], direction[order]

    taker_side = costs.taker_fee_bps + costs.half_spread_bps + costs.slippage_bps
    entry_cost_bps = costs.maker_fee_bps if execution.entry == "maker" else taker_side
    exit_cost_bps = costs.maker_fee_bps if execution.exit == "maker" else taker_side
    round_trip_bps = entry_cost_bps + exit_cost_bps

    n_signals = int(entry_bars.size)
    is_taker = execution.entry != "maker"

    # Preallocated, because the flagged set can run to millions of windows and
    # appending to Python lists that many times dominates the simulation.
    taken_entry = np.empty(n_signals, dtype=np.int64)
    taken_dir = np.empty(n_signals, dtype=np.int64)
    fill_price = np.empty(n_signals, dtype=np.float64)
    n_taken = n_blocked = n_unfilled = 0
    free_at = -1
    limit = price.size - horizon - 1

    for i in range(n_signals):
        signal_bar = int(entry_bars[i])
        if signal_bar < free_at:
            n_blocked += 1
            continue
        if signal_bar > limit or not defined[signal_bar]:
            continue

        want = int(direction[i])
        if is_taker:
            entry_bar, entry_price = signal_bar, price[signal_bar]
        else:
            fill = _maker_fill(price, ofi, signal_bar, want, execution)
            if fill is None:
                n_unfilled += 1
                continue
            entry_bar, entry_price = fill
            if entry_bar > limit:
                continue

        taken_entry[n_taken] = entry_bar
        taken_dir[n_taken] = want
        fill_price[n_taken] = entry_price
        n_taken += 1
        free_at = entry_bar + int(exit_offset[entry_bar]) + 1

    taken_entry = taken_entry[:n_taken]
    taken_dir = taken_dir[:n_taken]
    fill_price = fill_price[:n_taken]

    # Everything after the selection loop is vectorised: the sequential part is
    # only "may I open a position here", which depends on the previous exit.
    hold = exit_offset[taken_entry].astype(np.int64)
    exit_bar = taken_entry + hold
    gross_bps = taken_dir * (price[exit_bar] / fill_price - 1.0) / BPS
    trades = {
        "entry_bar": taken_entry,
        "exit_bar": exit_bar,
        "entry_ts": ts[taken_entry].astype(np.int64),
        "exit_ts": ts[exit_bar].astype(np.int64),
        "direction": taken_dir,
        "gross_bps": gross_bps,
        "net_bps": gross_bps - round_trip_bps,
        "hold_s": hold,
        "won": (side[taken_entry].astype(np.int64) == taken_dir).astype(np.int64),
        # A position closed by the vertical barrier is neither a win nor a loss:
        # it exits at whatever the price happens to be. Keeping it separate matters
        # because the break-even arithmetic B(2h-1) > c assumes h is the hit rate
        # among *resolved* trades; mixing timeouts into that denominator would
        # understate h and overstate the barrier the strategy needs.
        "resolved": (side[taken_entry] != 0).astype(np.int64),
    }
    summary, daily = _summarise(
        trades,
        n_signals=n_signals,
        n_blocked=n_blocked,
        n_unfilled=n_unfilled,
        round_trip_bps=round_trip_bps,
        costs=costs,
        execution=execution,
        barrier=barrier,
        bootstrap=bootstrap,
        seed=seed,
    )
    return BacktestResult(trades=trades, summary=summary, daily=daily)


def _summarise(
    trades: dict,
    n_signals: int,
    n_blocked: int,
    n_unfilled: int,
    round_trip_bps: float,
    costs: Costs,
    execution: Execution,
    barrier: float,
    bootstrap: int,
    seed: int,
) -> tuple[dict, dict]:
    net = trades["net_bps"]
    base = {
        "n_signals": n_signals,
        "n_trades": int(net.size),
        "n_blocked": n_blocked,
        "n_unfilled": n_unfilled,
        "round_trip_bps": round_trip_bps,
        "breakeven_barrier_bps": breakeven_barrier_bps(costs, execution),
        "barrier_bps": barrier / BPS,
        "half_spread_bps": costs.half_spread_bps,
        "costs": asdict(costs),
        "execution": asdict(execution),
    }
    if net.size == 0:
        return {**base, "hit_rate": float("nan"), "resolved_hit_rate": float("nan"),
                "resolved_share": float("nan"), "gross_bps": float("nan"),
                "cost_bps": round_trip_bps, "net_bps": float("nan"),
                "mean_hold_s": float("nan"), "total_return": 0.0,
                "max_drawdown": 0.0, "sharpe": float("nan"), "n_days": 0,
                "sharpe_ci": None}, {"day": np.empty(0), "pnl_bps": np.empty(0)}

    # Daily P&L: the honest aggregation level. Per-trade Sharpe would depend on
    # how often the strategy happens to fire, which is not a property of the edge.
    day = trades["exit_ts"] // SECONDS_PER_DAY
    days, inverse = np.unique(day, return_inverse=True)
    daily_bps = np.bincount(inverse, weights=net, minlength=days.size)

    mean, sd = float(daily_bps.mean()), float(daily_bps.std(ddof=1)) if days.size > 1 else 0.0
    sharpe = float(mean / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")

    equity = np.cumsum(daily_bps) * BPS
    peak = np.maximum.accumulate(np.concatenate(([0.0], equity)))
    drawdown = float((np.concatenate(([0.0], equity)) - peak).min())

    sharpe_ci = None
    if bootstrap and days.size > 5:
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, days.size, (bootstrap, days.size))
        sample = daily_bps[draws]
        sd_b = sample.std(axis=1, ddof=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            sh = sample.mean(axis=1) / sd_b * np.sqrt(TRADING_DAYS)
        sh = sh[np.isfinite(sh)]
        if sh.size:
            sharpe_ci = [float(np.percentile(sh, 2.5)), float(np.percentile(sh, 97.5))]

    resolved = trades["resolved"].astype(bool)
    summary = {
        **base,
        "hit_rate": float(trades["won"].mean()),
        "resolved_share": float(resolved.mean()),
        "resolved_hit_rate": float(trades["won"][resolved].mean()) if resolved.any() else float("nan"),
        "gross_bps": float(trades["gross_bps"].mean()),
        "cost_bps": round_trip_bps,
        "net_bps": float(net.mean()),
        "mean_hold_s": float(trades["hold_s"].mean()),
        "total_return": float(net.sum() * BPS),
        "max_drawdown": drawdown,
        "sharpe": sharpe,
        "sharpe_ci": sharpe_ci,
        "n_days": int(days.size),
        "trades_per_day": float(net.size / max(days.size, 1)),
    }
    return summary, {"day": days, "pnl_bps": daily_bps}


def signals_from_probabilities(
    starts: np.ndarray,
    window: int,
    p_up: np.ndarray,
    long_threshold: float = 0.5,
    short_threshold: float | None = None,
    gate: np.ndarray | None = None,
    gate_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn model output into `(entry_bars, direction)` for `simulate`.

    A window starting at `s` is decided on its **last** bar, `s + window - 1`,
    which is the earliest moment its features exist — so that, and not the
    window start, is when an order could be sent.

    `gate` is the meta-label: the probability that this window is tradeable at
    all. Windows below `gate_threshold` are not traded regardless of how
    confident the side model is, which is what lets a model trained on
    barrier-touching windows be deployed on a population that has not been
    filtered with hindsight.
    """
    short_threshold = 1.0 - long_threshold if short_threshold is None else short_threshold

    take_long = p_up >= long_threshold
    take_short = p_up <= short_threshold
    active = take_long | take_short
    if gate is not None:
        active &= gate >= gate_threshold

    decision_bar = starts[active] + window - 1
    direction = np.where(take_long[active], 1, -1).astype(np.int64)
    return decision_bar, direction
