"""CSV ingestion: raw ticks in, a gap-free 1-second bar grid out.

This is the first step of everything, and the one with the most ways to be
quietly wrong. The bar grid must have a row for *every* second, including the
~11% that carry no trade, because the horizon is counted in bars: if trade-less
seconds were dropped, a "60-bar horizon" would span 60-80 seconds of wall clock
depending on how busy the market was, and the triple barrier's deadline would
depend on activity — which is precisely the thing the label is supposed to be
independent of.

Both aggregation engines are tested against each other. `engine="lazy"` was the
original single `group_by`; `engine="batched"` replaced it because Polars would
silently fall back to collecting a 185 M-row frame. They must agree exactly, or
the replacement changed the data rather than the memory profile.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import sequence_matrix as seq


def write_trades_csv(path, rows, time_unit: int = 1_000):
    """Write a headerless Binance-format trades CSV.

    `rows` is a list of `(second_offset, price, qty, is_buyer_maker)`. Columns
    follow the real archive layout, including the three this project never reads
    — dropping them here would not test the projection that skips them.
    """
    base_second = 1_767_225_600  # 2026-01-01T00:00:00Z
    frame = pl.DataFrame(
        {
            "trade_id": list(range(len(rows))),
            "price": [float(r[1]) for r in rows],
            "qty": [float(r[2]) for r in rows],
            "quote_qty": [float(r[1]) * float(r[2]) for r in rows],
            "time": [(base_second + int(r[0])) * time_unit for r in rows],
            "is_buyer_maker": [bool(r[3]) for r in rows],
            "is_best_match": [True] * len(rows),
        },
        schema=seq.CSV_SCHEMA,
    )
    frame.write_csv(path, include_header=False)
    return path


@pytest.fixture
def trades_csv(tmp_path):
    """Four seconds of trades with a deliberate hole at second 2."""
    rows = [
        (0, 100.0, 1.0, False),  # aggressive buy
        (0, 100.5, 2.0, True),  # aggressive sell, same second
        (1, 101.0, 1.5, False),
        # second 2 has no trades at all
        (3, 102.0, 3.0, True),
    ]
    return write_trades_csv(tmp_path / "trades.csv", rows)


# --- the time unit ------------------------------------------------------------


def test_time_divisor_infers_the_archives_unit():
    """Binance has shipped seconds, then milliseconds, then microseconds."""
    assert seq._time_divisor(1_767_225_600) == 1
    assert seq._time_divisor(1_767_225_600_000) == 1_000
    assert seq._time_divisor(1_767_225_600_000_000) == 1_000_000


@pytest.mark.parametrize("time_unit", [1, 1_000, 1_000_000])
def test_bars_are_identical_whatever_the_timestamp_unit(tmp_path, time_unit):
    """The unit is a file format detail and must not reach the bar grid."""
    rows = [(0, 100.0, 1.0, False), (1, 101.0, 2.0, True), (2, 100.5, 1.0, False)]
    path = write_trades_csv(tmp_path / f"t{time_unit}.csv", rows, time_unit=time_unit)

    bars = seq.load_second_bars(path)

    np.testing.assert_array_equal(bars["ts"], [1_767_225_600, 1_767_225_601, 1_767_225_602])
    np.testing.assert_allclose(bars["price"], [100.0, 101.0, 100.5])


# --- the bar grid -------------------------------------------------------------


def test_every_second_gets_a_bar_including_the_empty_one(trades_csv):
    bars = seq.load_second_bars(trades_csv)

    assert bars["ts"].size == 4
    np.testing.assert_array_equal(np.diff(bars["ts"]), 1)
    assert bars["meta"]["traded_seconds"] == 3
    assert bars["meta"]["empty_seconds"] == 1


def test_an_empty_second_forward_fills_price_and_zeroes_flow(trades_csv):
    """A trade-less second is a real event: no volume, no flow, price unchanged."""
    bars = seq.load_second_bars(trades_csv)

    assert bars["price"][2] == bars["price"][1]
    assert bars["qty"][2] == 0.0
    assert bars["n_trades"][2] == 0.0
    assert bars["ofi"][2] == 0.0


def test_a_bars_price_is_the_last_trade_in_that_second(trades_csv):
    """Two trades land in second 0; the bar closes at the later one."""
    bars = seq.load_second_bars(trades_csv)
    assert bars["price"][0] == pytest.approx(100.5)


def test_quantities_and_trade_counts_are_summed_within_a_second(trades_csv):
    bars = seq.load_second_bars(trades_csv)

    assert bars["qty"][0] == pytest.approx(3.0)  # 1.0 + 2.0
    assert bars["n_trades"][0] == pytest.approx(2.0)


def test_order_flow_nets_offsetting_trades_inside_a_second(trades_csv):
    """1.0 bought aggressively, 2.0 sold aggressively -> -1.0 net.

    Netting at tick level rather than taking a per-second majority is what makes
    OFI a flow measure instead of a sign count.
    """
    bars = seq.load_second_bars(trades_csv)
    assert bars["ofi"][0] == pytest.approx(-1.0)


def test_order_flow_signs_follow_the_taker(tmp_path):
    path = write_trades_csv(
        tmp_path / "signs.csv",
        [
            (0, 100.0, 2.0, False),  # buyer is taker -> aggressive buy -> +qty
            (1, 100.0, 3.0, True),  # buyer is maker -> aggressive sell -> -qty
        ],
    )
    bars = seq.load_second_bars(path)

    assert bars["ofi"][0] == pytest.approx(2.0)
    assert bars["ofi"][1] == pytest.approx(-3.0)


def test_metadata_describes_the_series(trades_csv):
    meta = seq.load_second_bars(trades_csv)["meta"]

    assert meta["source"] == trades_csv.name
    assert meta["n_bars"] == 4
    assert meta["first_ts"] == 1_767_225_600
    assert meta["last_ts"] == 1_767_225_603


# --- engine equivalence -------------------------------------------------------


def test_the_two_engines_produce_identical_bars(tmp_path):
    """`batched` replaced `lazy` for memory reasons only — not for its output."""
    rng = np.random.default_rng(0)
    rows = [
        (int(second), 100.0 + rng.normal(0, 0.5), float(rng.gamma(2.0)), bool(rng.random() < 0.5))
        for second in np.repeat(np.arange(200), 3)
    ]
    path = write_trades_csv(tmp_path / "both.csv", rows)

    batched = seq.load_second_bars(path, engine="batched")
    lazy = seq.load_second_bars(path, engine="lazy")

    for key in ("ts", "price", "qty", "n_trades", "ofi"):
        np.testing.assert_allclose(
            batched[key], lazy[key], rtol=1e-9, err_msg=f"engines disagree on {key}"
        )


def test_a_small_batch_size_does_not_change_the_result(tmp_path):
    """Batch boundaries must not fall inside a second and split its aggregation."""
    rows = [(int(s), 100.0 + s * 0.01, 1.0, s % 2 == 0) for s in np.repeat(np.arange(50), 4)]
    path = write_trades_csv(tmp_path / "chunked.csv", rows)

    whole = seq.load_second_bars(path, batch_size=10_000_000)
    tiny = seq.load_second_bars(path, batch_size=7)  # forces many ragged batches

    for key in ("ts", "price", "qty", "n_trades", "ofi"):
        np.testing.assert_allclose(
            whole[key], tiny[key], rtol=1e-9, err_msg=f"batch size changed {key}"
        )


def test_chunked_lazy_aggregation_matches_a_single_pass(tmp_path):
    rows = [(int(s), 100.0 + s * 0.01, 1.0, False) for s in np.repeat(np.arange(400), 2)]
    path = write_trades_csv(tmp_path / "hours.csv", rows)

    single = seq.load_second_bars(path, engine="lazy")
    chunked = seq.load_second_bars(path, engine="lazy", chunk_hours=0.05)

    np.testing.assert_allclose(single["price"], chunked["price"])
    np.testing.assert_allclose(single["ofi"], chunked["ofi"])


# --- slicing ------------------------------------------------------------------


def test_hours_limits_the_span_that_is_read(tmp_path):
    rows = [(int(s), 100.0, 1.0, False) for s in range(7_200)]  # two hours
    path = write_trades_csv(tmp_path / "long.csv", rows)

    bars = seq.load_second_bars(path, hours=1.0)

    assert bars["ts"].size == 3_600
    assert bars["meta"]["hours"] == 1.0


def test_skip_hours_drops_the_start(tmp_path):
    rows = [(int(s), 100.0 + s, 1.0, False) for s in range(7_200)]
    path = write_trades_csv(tmp_path / "skip.csv", rows)

    bars = seq.load_second_bars(path, skip_hours=1.0)

    assert bars["ts"][0] == 1_767_225_600 + 3_600
    assert bars["meta"]["skip_hours"] == 1.0


def test_hours_and_skip_hours_compose(tmp_path):
    rows = [(int(s), 100.0, 1.0, False) for s in range(10_800)]  # three hours
    path = write_trades_csv(tmp_path / "both_slices.csv", rows)

    bars = seq.load_second_bars(path, hours=1.0, skip_hours=1.0)

    assert bars["ts"][0] == 1_767_225_600 + 3_600
    assert bars["ts"].size == 3_600


# --- failure modes ------------------------------------------------------------


def test_an_empty_csv_is_rejected(tmp_path):
    path = write_trades_csv(tmp_path / "empty.csv", [])

    with pytest.raises(ValueError, match="No rows"):
        seq.load_second_bars(path)


def test_an_unknown_engine_is_rejected(trades_csv):
    with pytest.raises(ValueError, match="engine"):
        seq.load_second_bars(trades_csv, engine="nonsense")


def test_chunk_hours_is_rejected_for_the_batched_engine(trades_csv):
    """It would be meaningless: the batched engine is already memory-bounded."""
    with pytest.raises(ValueError, match="chunk_hours"):
        seq.load_second_bars(trades_csv, engine="batched", chunk_hours=1.0)


# --- caching round trip -------------------------------------------------------


def test_build_then_cache_then_reload_is_lossless(trades_csv, tmp_path):
    """The path every training run takes: CSV once, `.npz` thereafter."""
    built = seq.load_second_bars(trades_csv)
    reloaded = seq.load_bars(seq.save_bars(built, tmp_path / "cache.npz"))

    for key in ("ts", "price", "qty", "n_trades", "ofi"):
        np.testing.assert_array_equal(reloaded[key], built[key])


def test_load_or_build_bars_uses_the_cache_the_second_time(trades_csv, tmp_path):
    cache = tmp_path / "cache.npz"

    first = seq.load_or_build_bars(trades_csv, cache=cache)
    assert cache.exists()

    # Point the CSV somewhere that does not exist: a second call must not read it.
    second = seq.load_or_build_bars(tmp_path / "gone.csv", cache=cache)
    np.testing.assert_array_equal(first["price"], second["price"])


def test_load_or_build_bars_refuses_a_cache_of_a_different_slice(trades_csv, tmp_path):
    """The failure mode worth an error: a full-month cache answering an 8-hour ask.

    It would report a month-long result under an 8-hour label — plausible-looking
    and wrong, which is worse than a crash.
    """
    cache = tmp_path / "cache.npz"
    seq.load_or_build_bars(trades_csv, cache=cache)  # cached with hours=None

    with pytest.raises(ValueError, match="hours"):
        seq.load_or_build_bars(trades_csv, hours=8.0, cache=cache)
