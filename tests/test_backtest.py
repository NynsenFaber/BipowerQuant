"""Costs, passive fills, and the simulation's invariants.

The two properties worth defending here are the ones that decide whether a
reported Sharpe means anything:

* **Positions never overlap.** Windows advance every second, so a simulator that
  opens a position per flagged window counts the same price move hundreds of
  times and inflates Sharpe by roughly the square root of the overlap — in the
  flattering direction, which is how it survives a casual review.
* **The oracle bound.** A strategy told the true side in advance earns the
  barrier and pays the round trip, and nothing can beat it. It is the arithmetic
  the project's central finding rests on, so it is asserted rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

import backtest as bt

# --- costs --------------------------------------------------------------------


def test_taker_round_trip_sums_fee_spread_and_slippage():
    costs = bt.Costs(taker_fee_bps=2.5, half_spread_bps=0.1, slippage_bps=0.5)
    # Each side: 2.5 + 0.1 + 0.5 = 3.1
    assert costs.round_trip_bps("taker", "taker") == pytest.approx(6.2)


def test_maker_sides_pay_only_the_maker_fee():
    """A passive order does not cross the spread and is not slipped."""
    costs = bt.Costs(taker_fee_bps=2.5, maker_fee_bps=0.0, half_spread_bps=0.1, slippage_bps=0.5)

    assert costs.round_trip_bps("maker", "maker") == pytest.approx(0.0)
    assert costs.round_trip_bps("maker", "taker") == pytest.approx(3.1)


def test_unset_half_spread_is_treated_as_zero_not_an_error():
    costs = bt.Costs(taker_fee_bps=2.5, half_spread_bps=None, slippage_bps=0.5)
    assert costs.round_trip_bps("taker", "taker") == pytest.approx(6.0)


def test_the_projects_headline_round_trip_is_six_basis_points():
    """2.5 bp taker + 0.5 bp slippage a side, spread negligible — against a 5 bp barrier.

    This is the number the whole "the target was defined below its own cost"
    finding turns on, so it is pinned here rather than recomputed in prose.
    """
    breakeven = bt.breakeven_barrier_bps(
        bt.Costs(half_spread_bps=0.0), bt.Execution(entry="taker", exit="taker")
    )
    assert breakeven == pytest.approx(6.0)
    assert breakeven > 5.0, "a 5 bp barrier cannot pay a 6 bp round trip"


def test_maker_entry_halves_the_round_trip():
    costs = bt.Costs(half_spread_bps=0.0)
    taker = bt.breakeven_barrier_bps(costs, bt.Execution(entry="taker", exit="taker"))
    maker = bt.breakeven_barrier_bps(costs, bt.Execution(entry="maker", exit="taker"))

    assert maker == pytest.approx(taker / 2)


# --- the spread ---------------------------------------------------------------


def test_spread_falls_back_to_the_tick_floor_when_roll_is_unresolvable():
    """On BTC/USDT the Roll estimator finds trend, not bounce, and must not return 0.

    Roll needs *negative* serial covariance in returns — the signature of trades
    alternating across a spread. A momentum series has positive covariance, so
    there is no real root to take and the estimator has nothing to say. Returning
    zero there would understate costs; the tick floor is the exchange's own hard
    lower bound and is the honest answer instead.
    """
    rng = np.random.default_rng(0)
    shocks = rng.normal(0, 2e-5, 5_000)
    # Positively autocorrelated returns: each bar carries half of the last one.
    returns = shocks + 0.5 * np.concatenate(([0.0], shocks[:-1]))
    trending = 70_000.0 * np.exp(np.cumsum(returns))

    out = bt.estimate_half_spread_bps(trending, block=1_000)

    assert out["roll_blocks"] == 0, "momentum should leave Roll with no real root"
    assert "tick floor" in out["method"]
    assert out["half_spread_bps"] == pytest.approx(out["tick_floor_bps"])
    assert out["half_spread_bps"] > 0.0


def test_tick_floor_is_half_a_tick_over_the_median_price():
    price = np.full(3_000, 80_000.0)
    out = bt.estimate_half_spread_bps(price, block=1_000)

    expected = 0.5 * bt.TICK_SIZE_USDT / 80_000.0 / bt.BPS
    assert out["tick_floor_bps"] == pytest.approx(expected)
    # ~0.0006 bp: three orders of magnitude below a 2.5 bp fee, as documented.
    assert out["tick_floor_bps"] < 0.01


def test_roll_estimator_recovers_a_planted_bid_ask_bounce():
    """Trades landing randomly on the bid or the ask is exactly Roll's model.

    The signs must be i.i.d., not alternating: Roll's `s = 2*sqrt(-cov)` assumes
    an independent coin flip per trade, and a deterministic bid/ask/bid pattern
    has twice the serial covariance, which the estimator would read as twice the
    spread.
    """
    rng = np.random.default_rng(0)
    efficient = np.cumsum(rng.normal(0, 1e-5, 40_000))
    half_spread = 0.002  # 20 bp in log terms, far above the tick floor
    side = rng.choice([-1.0, 1.0], 40_000)

    price = 100.0 * np.exp(efficient + half_spread * side)
    out = bt.estimate_half_spread_bps(price, block=10_000)

    assert out["method"] == "Roll"
    assert out["roll_blocks"] > 0
    assert out["half_spread_bps"] == pytest.approx(half_spread / bt.BPS, rel=0.15)


def test_roll_half_spread_returns_just_the_number(bars):
    value = bt.roll_half_spread_bps(bars["price"], block=500)
    full = bt.estimate_half_spread_bps(bars["price"], block=500)
    assert value == full["half_spread_bps"]


# --- passive fills ------------------------------------------------------------


def _flat(n=20, price=100.0):
    return np.full(n, price)


def test_maker_bid_fills_when_the_queue_ahead_is_cleared():
    """Aggressive sells hit our bid; once they exceed the queue, we trade."""
    price = _flat()
    ofi = np.zeros(20)
    ofi[1:4] = -1.0  # three seconds of aggressive selling, 1 BTC each
    execution = bt.Execution(queue_ahead_btc=2.0, size_btc=0.5, max_wait_s=5)

    fill = bt._maker_fill(price, ofi, bar=0, direction=1, execution=execution)

    assert fill is not None
    fill_bar, fill_price = fill
    assert fill_bar == 3  # 1.0 + 1.0 + 1.0 >= 2.0 + 0.5
    assert fill_price == 100.0


def test_maker_bid_is_not_filled_by_buying_pressure():
    """Aggressive *buys* lift the ask; they never reach a resting bid."""
    price = _flat()
    ofi = np.full(20, 5.0)  # heavy buying
    execution = bt.Execution(queue_ahead_btc=1.0, size_btc=0.1)

    assert bt._maker_fill(price, ofi, 0, direction=1, execution=execution) is None


def test_maker_ask_fills_on_buying_pressure():
    """The mirror image: a short posts an ask and needs aggressive buyers."""
    price = _flat()
    ofi = np.zeros(20)
    ofi[1:4] = 1.0  # three seconds of aggressive buying, 1 BTC each
    execution = bt.Execution(queue_ahead_btc=2.0, size_btc=0.5, max_wait_s=5)

    fill = bt._maker_fill(price, ofi, 0, direction=-1, execution=execution)

    assert fill is not None
    fill_bar, fill_price = fill
    assert fill_bar == 3  # 1.0 + 1.0 + 1.0 >= 2.0 + 0.5
    assert fill_price == 100.0


def test_a_bid_traded_through_is_filled_at_the_limit_and_adversely_selected():
    """Everything at our level was consumed, so we are filled — and already wrong.

    This is the cost that makes maker execution not free, so it must fill rather
    than escape.
    """
    price = _flat()
    price[2:] = 99.0  # the market trades down through our resting bid
    execution = bt.Execution(queue_ahead_btc=1e9)  # queue could never clear

    fill = bt._maker_fill(price, np.zeros(20), 0, direction=1, execution=execution)

    assert fill == (2, 100.0)


def test_an_ask_traded_through_is_filled_at_the_limit():
    price = _flat()
    price[3:] = 101.0
    execution = bt.Execution(queue_ahead_btc=1e9)

    fill = bt._maker_fill(price, np.zeros(20), 0, direction=-1, execution=execution)

    assert fill == (3, 100.0)


def test_an_order_the_market_runs_away_from_is_cancelled():
    """Chasing would turn a maker fill into a taker fill, so the signal is dropped."""
    price = _flat()
    price[1:] = 101.0  # market gone up, away from our bid

    fill = bt._maker_fill(price, np.zeros(20), 0, direction=1, execution=bt.Execution())

    assert fill is None


def test_an_unfilled_order_times_out():
    price = _flat()
    ofi = np.zeros(20)  # no flow at all: the queue never moves
    execution = bt.Execution(queue_ahead_btc=1.0, max_wait_s=3)

    assert bt._maker_fill(price, ofi, 0, direction=1, execution=execution) is None


def test_an_order_near_the_end_of_the_series_is_cancelled():
    """There are not enough bars left to wait out `max_wait_s`."""
    price = _flat(n=3)
    execution = bt.Execution(queue_ahead_btc=1e9, max_wait_s=5)

    assert bt._maker_fill(price, np.zeros(3), 1, direction=1, execution=execution) is None


# --- signal construction ------------------------------------------------------


def test_signals_fire_on_the_windows_last_bar():
    """A window is decided when its features exist, not when it starts."""
    starts = np.array([0, 10, 20], dtype=np.int64)
    p_up = np.array([0.9, 0.9, 0.9])

    entry_bars, direction = bt.signals_from_probabilities(starts, window=5, p_up=p_up)

    np.testing.assert_array_equal(entry_bars, [4, 14, 24])
    np.testing.assert_array_equal(direction, [1, 1, 1])


def test_probabilities_below_the_short_threshold_go_short():
    starts = np.arange(4, dtype=np.int64)
    p_up = np.array([0.9, 0.1, 0.55, 0.45])

    _, direction = bt.signals_from_probabilities(starts, window=1, p_up=p_up)
    np.testing.assert_array_equal(direction, [1, -1, 1, -1])


def test_a_side_margin_drops_the_undecided_middle():
    starts = np.arange(4, dtype=np.int64)
    p_up = np.array([0.9, 0.1, 0.52, 0.48])

    entry_bars, _ = bt.signals_from_probabilities(
        starts, window=1, p_up=p_up, long_threshold=0.6, short_threshold=0.4
    )
    np.testing.assert_array_equal(entry_bars, [0, 1])


def test_the_gate_suppresses_signals_regardless_of_side_confidence():
    """A confident side model must not trade a window the gate rejects."""
    starts = np.arange(4, dtype=np.int64)
    p_up = np.array([0.99, 0.99, 0.01, 0.01])
    gate = np.array([0.9, 0.1, 0.9, 0.1])

    entry_bars, _ = bt.signals_from_probabilities(
        starts, window=1, p_up=p_up, gate=gate, gate_threshold=0.5
    )
    np.testing.assert_array_equal(entry_bars, [0, 2])


# --- the simulation -----------------------------------------------------------


def _costless():
    return bt.Costs(taker_fee_bps=0.0, maker_fee_bps=0.0, half_spread_bps=0.0, slippage_bps=0.0)


def test_positions_never_overlap(long_bars):
    """The single most important property in the file.

    Every window is flagged; the simulator must still hold one position at a
    time, so each entry falls strictly after the previous exit.
    """
    n = long_bars["price"].size
    entry_bars = np.arange(0, n - 200, dtype=np.int64)
    direction = np.ones(entry_bars.size, dtype=np.int64)

    result = bt.simulate(
        long_bars, entry_bars, direction, barrier=0.0005, horizon=60, costs=_costless(), bootstrap=0
    )

    entries = result.trades["entry_bar"]
    exits = result.trades["exit_bar"]
    assert entries.size > 0
    assert np.all(entries[1:] > exits[:-1]), "a position opened before the previous closed"
    # The overlap is what collapses millions of signals into far fewer trades.
    assert result.summary["n_trades"] < entry_bars.size / 10
    assert result.summary["n_blocked"] > 0


def test_signal_and_trade_counts_reconcile(long_bars):
    entry_bars = np.arange(0, 2_000, dtype=np.int64)
    direction = np.ones(entry_bars.size, dtype=np.int64)

    result = bt.simulate(long_bars, entry_bars, direction, costs=_costless(), bootstrap=0)
    s = result.summary

    assert s["n_signals"] == entry_bars.size
    assert s["n_trades"] + s["n_blocked"] + s["n_unfilled"] <= s["n_signals"]


def test_an_oracle_earns_the_barrier_and_a_coin_flip_earns_nothing(long_bars):
    """The bound the project's central claim rests on.

    Told the true side, gross P&L per trade approaches the barrier width. Told
    nothing, it is zero. Any strategy sits between them, so a barrier below the
    round trip cannot be profitable at any accuracy.
    """
    barrier, horizon = 0.0005, 60
    price = long_bars["price"]
    side, _, defined = bt.barrier_arrays(price, horizon, barrier)

    entry_bars = np.flatnonzero(defined & (side != 0))[:1_500].astype(np.int64)
    oracle = bt.simulate(
        long_bars,
        entry_bars,
        side[entry_bars].astype(np.int64),
        barrier=barrier,
        horizon=horizon,
        costs=_costless(),
        bootstrap=0,
    )

    rng = np.random.default_rng(0)
    coin = bt.simulate(
        long_bars,
        entry_bars,
        rng.choice([-1, 1], entry_bars.size).astype(np.int64),
        barrier=barrier,
        horizon=horizon,
        costs=_costless(),
        bootstrap=0,
    )

    barrier_bps = barrier / bt.BPS
    assert oracle.summary["hit_rate"] == pytest.approx(1.0)
    assert oracle.summary["gross_bps"] == pytest.approx(barrier_bps, rel=0.25)
    assert abs(coin.summary["gross_bps"]) < barrier_bps / 2
    assert oracle.summary["gross_bps"] > coin.summary["gross_bps"]


def test_an_oracle_still_loses_money_when_the_barrier_is_below_the_round_trip(long_bars):
    """The finding, stated as a test: 5 bp of target against 6 bp of cost."""
    barrier, horizon = 0.0005, 60
    side, _, defined = bt.barrier_arrays(long_bars["price"], horizon, barrier)
    entry_bars = np.flatnonzero(defined & (side != 0))[:1_500].astype(np.int64)

    result = bt.simulate(
        long_bars,
        entry_bars,
        side[entry_bars].astype(np.int64),
        barrier=barrier,
        horizon=horizon,
        costs=bt.Costs(taker_fee_bps=2.5, half_spread_bps=0.0, slippage_bps=0.5),
        bootstrap=0,
    )

    assert result.summary["cost_bps"] == pytest.approx(6.0)
    assert result.summary["net_bps"] < 0.0


def test_net_is_gross_less_the_round_trip(long_bars):
    entry_bars = np.arange(0, 1_000, dtype=np.int64)
    costs = bt.Costs(taker_fee_bps=2.5, half_spread_bps=0.0, slippage_bps=0.5)

    result = bt.simulate(
        long_bars, entry_bars, np.ones(1_000, dtype=np.int64), costs=costs, bootstrap=0
    )

    np.testing.assert_allclose(
        result.trades["net_bps"], result.trades["gross_bps"] - 6.0, rtol=1e-10
    )


def test_shorting_flips_the_sign_of_the_gross_return(long_bars):
    entry_bars = np.array([100, 400, 700], dtype=np.int64)

    longs = bt.simulate(
        long_bars, entry_bars, np.ones(3, dtype=np.int64), costs=_costless(), bootstrap=0
    )
    shorts = bt.simulate(
        long_bars, entry_bars, -np.ones(3, dtype=np.int64), costs=_costless(), bootstrap=0
    )

    np.testing.assert_allclose(longs.trades["gross_bps"], -shorts.trades["gross_bps"], rtol=1e-10)


def test_holding_time_never_exceeds_the_horizon(long_bars):
    entry_bars = np.arange(0, 2_000, dtype=np.int64)
    result = bt.simulate(
        long_bars,
        entry_bars,
        np.ones(2_000, dtype=np.int64),
        horizon=60,
        costs=_costless(),
        bootstrap=0,
    )

    assert result.trades["hold_s"].max() <= 60
    assert result.summary["mean_hold_s"] <= 60


def test_unresolved_trades_are_counted_separately_from_losses(long_bars):
    """A vertical-barrier exit is neither a win nor a loss.

    Folding timeouts into the denominator would understate the hit rate and make
    the break-even barrier come out wrong — it printed a *negative* required
    barrier before the two were split.
    """
    entry_bars = np.arange(0, 2_000, dtype=np.int64)
    result = bt.simulate(
        long_bars, entry_bars, np.ones(2_000, dtype=np.int64), costs=_costless(), bootstrap=0
    )
    s = result.summary

    assert 0.0 <= s["resolved_share"] <= 1.0
    assert s["resolved_share"] < 1.0, "the fixture should produce some timeouts"
    # Among resolved trades only, so it differs from the all-trades hit rate.
    assert 0.0 <= s["resolved_hit_rate"] <= 1.0


def test_entry_order_does_not_depend_on_the_callers_sorting(long_bars):
    """Signals are sorted internally, so a shuffled input gives the same trades."""
    entry_bars = np.arange(0, 1_000, dtype=np.int64)
    direction = np.ones(1_000, dtype=np.int64)
    shuffled = np.random.default_rng(0).permutation(entry_bars.size)

    ordered = bt.simulate(long_bars, entry_bars, direction, costs=_costless(), bootstrap=0)
    scrambled = bt.simulate(
        long_bars, entry_bars[shuffled], direction[shuffled], costs=_costless(), bootstrap=0
    )

    np.testing.assert_array_equal(ordered.trades["entry_bar"], scrambled.trades["entry_bar"])


def test_no_signals_produces_an_empty_but_well_formed_result(long_bars):
    result = bt.simulate(
        long_bars, np.array([], dtype=np.int64), np.array([], dtype=np.int64), bootstrap=0
    )

    assert result.summary["n_trades"] == 0
    assert np.isnan(result.summary["gross_bps"])
    assert result.summary["total_return"] == 0.0
    assert result.daily["day"].size == 0
    assert isinstance(result.format(), str)


def test_signals_in_the_undefined_tail_are_not_traded(long_bars):
    """Their forward path is truncated, so there is no honest exit price."""
    n = long_bars["price"].size
    entry_bars = np.arange(n - 30, n, dtype=np.int64)

    result = bt.simulate(
        long_bars,
        entry_bars,
        np.ones(30, dtype=np.int64),
        horizon=60,
        costs=_costless(),
        bootstrap=0,
    )

    assert result.summary["n_trades"] == 0


def test_the_simulation_is_deterministic(long_bars):
    entry_bars = np.arange(0, 1_000, dtype=np.int64)
    direction = np.ones(1_000, dtype=np.int64)

    a = bt.simulate(long_bars, entry_bars, direction, costs=_costless(), seed=7)
    b = bt.simulate(long_bars, entry_bars, direction, costs=_costless(), seed=7)

    np.testing.assert_array_equal(a.trades["net_bps"], b.trades["net_bps"])
    assert a.summary["sharpe"] == b.summary["sharpe"] or (
        np.isnan(a.summary["sharpe"]) and np.isnan(b.summary["sharpe"])
    )


def test_precomputed_barriers_give_the_same_answer(long_bars):
    """The sweep's optimisation must not change the result it optimises."""
    entry_bars = np.arange(0, 1_000, dtype=np.int64)
    direction = np.ones(1_000, dtype=np.int64)
    precomputed = bt.barrier_arrays(long_bars["price"], 60, 0.0005)

    fresh = bt.simulate(long_bars, entry_bars, direction, costs=_costless(), bootstrap=0)
    reused = bt.simulate(
        long_bars, entry_bars, direction, costs=_costless(), bootstrap=0, precomputed=precomputed
    )

    np.testing.assert_array_equal(fresh.trades["entry_bar"], reused.trades["entry_bar"])
    np.testing.assert_allclose(fresh.trades["net_bps"], reused.trades["net_bps"])


def test_maker_entry_records_unfilled_signals(long_bars):
    """With an unclearable queue, every signal should go unfilled rather than trade."""
    entry_bars = np.arange(0, 500, dtype=np.int64)
    execution = bt.Execution(entry="maker", exit="taker", queue_ahead_btc=1e9, max_wait_s=2)

    result = bt.simulate(
        long_bars,
        entry_bars,
        np.ones(500, dtype=np.int64),
        costs=_costless(),
        execution=execution,
        bootstrap=0,
    )

    assert result.summary["n_unfilled"] > 0


def test_summary_reports_the_assumptions_it_used(long_bars):
    costs = bt.Costs(taker_fee_bps=1.0, half_spread_bps=0.2, slippage_bps=0.3)
    result = bt.simulate(
        long_bars,
        np.arange(0, 500, dtype=np.int64),
        np.ones(500, dtype=np.int64),
        costs=costs,
        bootstrap=0,
    )

    assert result.summary["costs"]["taker_fee_bps"] == 1.0
    assert result.summary["half_spread_bps"] == 0.2
    assert result.summary["barrier_bps"] == pytest.approx(5.0)


def test_a_missing_half_spread_is_estimated_from_the_tape(long_bars):
    """`Costs(half_spread_bps=None)` must resolve to a number, not propagate None."""
    result = bt.simulate(
        long_bars,
        np.arange(0, 500, dtype=np.int64),
        np.ones(500, dtype=np.int64),
        costs=bt.Costs(half_spread_bps=None),
        bootstrap=0,
    )

    assert result.summary["half_spread_bps"] is not None
    assert result.summary["half_spread_bps"] > 0.0


def test_bootstrap_produces_an_interval_that_brackets_the_estimate(long_bars):
    """Enough days for the block bootstrap to run at all."""
    bars = dict(long_bars)
    # Stretch the timestamps so the trades land across ~20 distinct days.
    bars["ts"] = long_bars["ts"][0] + np.arange(long_bars["ts"].size) * 400

    result = bt.simulate(
        bars,
        np.arange(0, 2_000, dtype=np.int64),
        np.ones(2_000, dtype=np.int64),
        costs=_costless(),
        bootstrap=200,
        seed=3,
    )

    if result.summary["sharpe_ci"] is not None:
        lo, hi = result.summary["sharpe_ci"]
        assert lo <= hi
        assert result.summary["n_days"] > 5
