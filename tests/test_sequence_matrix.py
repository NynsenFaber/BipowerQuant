"""Labels, channels, windows and bar splicing.

The triple barrier gets the most attention here because every number in the
project is downstream of it: it defines the label the models fit, the population
they are scored on, and — through `exit_offset` — how long the backtest holds a
position. A sign error in it would not crash anything, it would just quietly
change what the project claims to have measured.
"""

from __future__ import annotations

import numpy as np
import pytest

import sequence_matrix as seq

# --- the triple barrier -------------------------------------------------------


def test_monotone_rise_touches_the_upper_barrier_first():
    # +10 bp a bar, so the 5 bp barrier is cleared on the very first step.
    price = 100.0 * np.exp(np.arange(50) * 1e-3)
    side, exit_offset, defined = seq.triple_barrier(price, horizon=10, barrier=0.0005)

    assert np.all(side[defined] == 1)
    np.testing.assert_array_equal(exit_offset[defined], 1)


def test_monotone_fall_touches_the_lower_barrier_first():
    price = 100.0 * np.exp(-np.arange(50) * 1e-3)
    side, exit_offset, defined = seq.triple_barrier(price, horizon=10, barrier=0.0005)

    assert np.all(side[defined] == -1)
    np.testing.assert_array_equal(exit_offset[defined], 1)


def test_flat_price_resolves_on_the_vertical_barrier():
    """No horizontal barrier is reachable, so every bar times out at `horizon`."""
    price = np.full(50, 100.0)
    side, exit_offset, defined = seq.triple_barrier(price, horizon=10, barrier=0.0005)

    assert np.all(side == 0)
    np.testing.assert_array_equal(exit_offset[defined], 10)


def test_exit_offset_records_the_first_touch_not_the_largest():
    """The barrier is a *first passage* — a later, bigger move must not overwrite it.

    The implementation walks k downwards so the last write wins; this pins that
    the last write is the smallest touching k rather than the largest.
    """
    # Flat, then +6 bp at k=3, then a much larger +50 bp at k=6.
    price = np.array([100.0, 100.0, 100.0, 100.06, 100.06, 100.06, 100.5, 100.5, 100.5])
    side, exit_offset, _ = seq.triple_barrier(price, horizon=8, barrier=0.0005)

    assert side[0] == 1
    assert exit_offset[0] == 3


def test_the_nearer_barrier_wins_when_both_are_touched():
    """Down first at k=1, up later at k=2: the label is the one reached first."""
    price = np.array([100.0, 99.9, 100.1, 100.1, 100.1])
    side, exit_offset, _ = seq.triple_barrier(price, horizon=4, barrier=0.0005)

    assert side[0] == -1
    assert exit_offset[0] == 1


def test_the_final_horizon_bars_are_undefined():
    """Their forward path is truncated by the end of the series."""
    price = 100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 3e-4, 100)))
    horizon = 12
    side, _, defined = seq.triple_barrier(price, horizon=horizon, barrier=0.0005)

    assert np.all(defined[: 100 - horizon])
    assert not np.any(defined[100 - horizon :])
    # An undefined bar must not carry a side, or it would be trained on.
    assert np.all(side[~defined] == 0)


def test_exit_offset_never_exceeds_the_horizon(bars):
    _, exit_offset, _ = seq.triple_barrier(bars["price"], horizon=60, barrier=0.0005)
    assert exit_offset.max() <= 60
    assert exit_offset.min() >= 0


def test_a_wider_barrier_resolves_no_more_windows(bars):
    """Monotonicity: raising the target cannot make more paths reach it."""
    resolved = []
    for barrier in (0.0002, 0.0005, 0.002):
        side, _, defined = seq.triple_barrier(bars["price"], horizon=60, barrier=barrier)
        resolved.append(int(((side != 0) & defined).sum()))

    assert resolved == sorted(resolved, reverse=True)


def test_a_longer_horizon_resolves_no_fewer_windows(bars):
    """The converse: more time cannot make a reachable barrier unreachable.

    Compared over the bars that are defined under the *longest* horizon, since a
    longer horizon also truncates more of the tail — counting over the whole
    series would measure that truncation rather than the resolution rate.
    """
    horizons = (10, 30, 120)
    common = bars["price"].size - max(horizons)

    resolved = []
    for horizon in horizons:
        side, _, _ = seq.triple_barrier(bars["price"], horizon=horizon, barrier=0.0005)
        resolved.append(int((side[:common] != 0).sum()))

    assert resolved == sorted(resolved)


def test_barrier_is_symmetric_under_price_inversion(bars):
    """Inverting the price must flip every side and leave the timing untouched.

    The barrier is defined on log returns, so 1/p mirrors the path exactly. This
    catches an asymmetry between the upper and lower comparison — the kind of bug
    that would manufacture a directional edge out of nothing.
    """
    side, exit_offset, defined = seq.triple_barrier(bars["price"], horizon=30, barrier=0.001)
    flipped_side, flipped_exit, flipped_defined = seq.triple_barrier(
        1.0 / bars["price"], horizon=30, barrier=0.001
    )

    np.testing.assert_array_equal(flipped_side, -side)
    np.testing.assert_array_equal(flipped_exit, exit_offset)
    np.testing.assert_array_equal(flipped_defined, defined)


def test_triple_barrier_labels_returns_just_the_side(bars):
    side, _, _ = seq.triple_barrier(bars["price"], horizon=30, barrier=0.0005)
    np.testing.assert_array_equal(seq.triple_barrier_labels(bars["price"], 30, 0.0005), side)


# --- label modes --------------------------------------------------------------


def test_triple_barrier_mode_keeps_only_resolved_windows(bars):
    y, usable = seq.build_targets(bars, horizon=30, barrier=0.0005, label_mode="triple_barrier")
    side, _, defined = seq.triple_barrier(bars["price"], horizon=30, barrier=0.0005)

    np.testing.assert_array_equal(usable, (side != 0) & defined)
    np.testing.assert_array_equal(y[usable], (side[usable] > 0).astype(np.int8))


def test_barrier_touched_mode_is_defined_on_every_window(bars):
    """The gate's population is every window, which is the point of the gate."""
    y, usable = seq.build_targets(bars, horizon=30, barrier=0.0005, label_mode="barrier_touched")
    _, _, defined = seq.triple_barrier(bars["price"], horizon=30, barrier=0.0005)

    np.testing.assert_array_equal(usable, defined)
    assert set(np.unique(y)).issubset({0, 1})


def test_gate_is_the_union_of_the_two_side_classes(bars):
    """`barrier_touched` must be exactly "the side label exists"."""
    _, side_usable = seq.build_targets(
        bars, horizon=30, barrier=0.0005, label_mode="triple_barrier"
    )
    gate_y, gate_usable = seq.build_targets(
        bars, horizon=30, barrier=0.0005, label_mode="barrier_touched"
    )

    np.testing.assert_array_equal(gate_y.astype(bool) & gate_usable, side_usable)


def test_fee_threshold_mode_is_a_forward_return_comparison(bars):
    y, usable = seq.build_targets(bars, horizon=30, barrier=0.0005, label_mode="fee_threshold")
    price = bars["price"]
    expected = (price[30:] / price[:-30] - 1.0) > 0.0005

    np.testing.assert_array_equal(y[:-30].astype(bool), expected)
    assert np.all(usable[:-30])
    assert not np.any(usable[-30:])


def test_unknown_label_mode_is_rejected(bars):
    with pytest.raises(ValueError, match="label_mode"):
        seq.build_targets(bars, label_mode="nonsense")


# --- windows and splits -------------------------------------------------------


def test_valid_window_starts_leaves_room_for_lookback_and_label():
    starts = seq.valid_window_starts(1_000, window=300, horizon=60)

    assert starts[0] == 0
    assert starts.size == 1_000 - 300 - 60 + 1
    # The last window's label must still be fully inside the series.
    assert starts[-1] + 300 - 1 + 60 <= 999


def test_valid_window_starts_rejects_a_series_that_is_too_short():
    with pytest.raises(ValueError, match="at least"):
        seq.valid_window_starts(100, window=300, horizon=60)


def test_labels_for_reads_the_windows_last_bar():
    y = np.arange(20, dtype=np.int8)
    starts = np.array([0, 5, 10])
    np.testing.assert_array_equal(seq.labels_for(starts, y, window=4), [3, 8, 13])


def test_chronological_split_is_ordered_and_disjoint():
    starts = np.arange(10_000, dtype=np.int64)
    splits = seq.chronological_split(starts, train_frac=0.7, val_frac=0.1, purge=359)

    assert splits["train"][-1] < splits["val"][0] < splits["test"][0]
    assert not set(splits["train"]) & set(splits["val"])
    assert not set(splits["val"]) & set(splits["test"])


def test_chronological_split_purges_the_boundary_overlap():
    """The gap at each boundary must be at least `purge` windows wide.

    Without it the last training windows are labelled by price moves that fall
    inside the first validation window's lookback — the leak the purge exists to
    remove.
    """
    starts = np.arange(10_000, dtype=np.int64)
    purge = 359
    splits = seq.chronological_split(starts, train_frac=0.7, val_frac=0.1, purge=purge)

    assert splits["val"][0] - splits["train"][-1] >= purge
    assert splits["test"][0] - splits["val"][-1] >= purge


def test_chronological_split_strides_train_and_val_but_not_test():
    starts = np.arange(10_000, dtype=np.int64)
    dense = seq.chronological_split(starts, purge=359)
    strided = seq.chronological_split(starts, purge=359, train_stride=10, val_stride=5)

    assert strided["train"].size == pytest.approx(dense["train"].size / 10, rel=0.02)
    assert strided["val"].size == pytest.approx(dense["val"].size / 5, rel=0.05)
    # Test is always dense, so the reported metric covers every held-out window.
    np.testing.assert_array_equal(strided["test"], dense["test"])


def test_chronological_split_rejects_too_few_windows():
    with pytest.raises(ValueError, match="too few"):
        seq.chronological_split(np.arange(100, dtype=np.int64), purge=359)


# --- channels -----------------------------------------------------------------


def test_build_channels_shape_and_dtype(bars):
    channels = seq.build_channels(bars, "raw")

    assert channels.shape == (bars["price"].size, 2)
    assert channels.dtype == np.float32


def test_raw_channels_are_log_return_and_ofi(bars):
    channels = seq.build_channels(bars, "raw")
    log_price = np.log(bars["price"])

    assert seq.CHANNEL_SETS["raw"] == ["log_return", "ofi"]
    # r[0] = 0 by construction: the first bar has no predecessor.
    assert channels[0, 0] == 0.0
    np.testing.assert_allclose(channels[1:, 0], np.diff(log_price), rtol=1e-5)
    np.testing.assert_allclose(channels[:, 1], bars["ofi"], rtol=1e-5)


def test_full_channels_are_pointwise_functions_of_the_raw_two(bars):
    """The claim in §5.3 that four of the six channels are redundant.

    If this fails, dropping them from the model was not free after all.
    """
    full = seq.build_channels(bars, "full")
    names = seq.CHANNEL_SETS["full"]
    log_return = full[:, names.index("log_return")].astype(np.float64)

    np.testing.assert_allclose(
        full[:, names.index("realized_var")], log_return**2, rtol=1e-4, atol=1e-12
    )
    expected_bipower = np.empty_like(log_return)
    expected_bipower[0] = 0.0
    expected_bipower[1:] = seq.PI_FACTOR * np.abs(log_return[1:]) * np.abs(log_return[:-1])
    np.testing.assert_allclose(
        full[:, names.index("bipower")], expected_bipower, rtol=1e-4, atol=1e-12
    )
    np.testing.assert_allclose(full[:, names.index("log_volume")], np.log1p(bars["qty"]), rtol=1e-5)


def test_build_channels_accepts_an_explicit_name_list(bars):
    """A checkpoint rebuilds its matrix from the names it recorded, not a set name."""
    channels = seq.build_channels(bars, ["ofi", "log_return"])

    assert channels.shape[1] == 2
    # Order follows the request, so a checkpoint's column layout is reproduced.
    np.testing.assert_allclose(channels[:, 0], bars["ofi"], rtol=1e-5)


def test_unknown_channel_name_is_rejected(bars):
    with pytest.raises(ValueError, match="Unknown channel"):
        seq.build_channels(bars, ["log_return", "not_a_channel"])


# --- tabular features ---------------------------------------------------------


def test_tabular_features_require_the_full_layout(bars):
    two_column = seq.build_channels(bars, "raw")
    starts = np.arange(100, dtype=np.int64)

    with pytest.raises(ValueError, match="full"):
        seq.build_tabular_features(two_column, bars["price"], starts, window=50)


def test_tabular_features_match_a_direct_window_sum(bars):
    """The cumulative-sum shortcut against the loop it replaces."""
    window = 40
    channels = seq.build_channels(bars, "full")
    starts = np.arange(200, dtype=np.int64)
    features = seq.build_tabular_features(channels, bars["price"], starts, window)

    names = seq.CHANNEL_SETS["full"]
    rv_col = channels[:, names.index("realized_var")].astype(np.float64)
    ofi_col = channels[:, names.index("ofi")].astype(np.float64)

    for row, start in enumerate(starts[:20]):
        # RV sums the window's returns, which begin one bar in.
        assert features[row, 0] == pytest.approx(rv_col[start + 1 : start + window].sum(), rel=1e-6)
        # OFI covers every bar of the window.
        assert features[row, 3] == pytest.approx(ofi_col[start : start + window].sum(), rel=1e-5)


def test_tabular_feature_identities_hold(bars):
    """Jumps, vol-adjusted OFI and signed jumps are defined off the other columns."""
    window = 50
    channels = seq.build_channels(bars, "full")
    starts = np.arange(300, dtype=np.int64)
    f = seq.build_tabular_features(channels, bars["price"], starts, window)

    rv, bpv, jumps, ofi, ret_5m, vol_ofi, signed = (f[:, i] for i in range(7))

    assert np.all(jumps >= 0.0)
    np.testing.assert_allclose(jumps, np.maximum(rv - bpv, 0.0), rtol=1e-12)
    np.testing.assert_allclose(vol_ofi, ofi / (np.sqrt(bpv) + seq.EPSILON), rtol=1e-10)
    np.testing.assert_allclose(signed, jumps * np.sign(ret_5m), rtol=1e-10)

    log_price = np.log(bars["price"])
    np.testing.assert_allclose(
        ret_5m, log_price[starts + window - 1] - log_price[starts], rtol=1e-10
    )


def test_tabular_feature_count_matches_the_documented_names():
    assert len(seq.TABULAR_FEATURE_NAMES) == 7


# --- caching and splicing -----------------------------------------------------


def test_save_and_load_bars_roundtrip(bars, tmp_path):
    path = seq.save_bars(bars, tmp_path / "bars.npz")
    reloaded = seq.load_bars(path)

    for key in ("ts", "price", "qty", "n_trades", "ofi"):
        np.testing.assert_array_equal(reloaded[key], bars[key])
    assert reloaded["meta"]["source"] == bars["meta"]["source"]
    assert reloaded["meta"]["n_bars"] == bars["meta"]["n_bars"]


def test_save_bars_creates_missing_parent_directories(bars, tmp_path):
    path = seq.save_bars(bars, tmp_path / "nested" / "deeper" / "bars.npz")
    assert path.exists()


def test_concat_bars_splices_a_continuous_grid(bars_factory):
    first = bars_factory(n_bars=500, start_ts=1_000_000, seed=1)
    second = bars_factory(n_bars=400, start_ts=1_000_500, seed=2)
    joined = seq.concat_bars([first, second])

    assert joined["ts"].size == 900
    np.testing.assert_array_equal(joined["ts"], np.arange(1_000_000, 1_000_900))
    np.testing.assert_array_equal(joined["price"][:500], first["price"])
    np.testing.assert_array_equal(joined["price"][500:], second["price"])
    assert joined["meta"]["gap_seconds"] == 0


def test_concat_bars_forward_fills_a_gap_between_months(bars_factory):
    """April ends at ...598 and May starts at ...600 in the real archive.

    The missing second is filled the way a trade-less second inside a month is:
    the last price carried forward, no volume, no flow.
    """
    first = bars_factory(n_bars=100, start_ts=1_000_000, seed=1)
    second = bars_factory(n_bars=100, start_ts=1_000_105, seed=2)  # 5-second hole
    joined = seq.concat_bars([first, second])

    assert joined["meta"]["gap_seconds"] == 5
    gap = slice(100, 105)
    np.testing.assert_allclose(joined["price"][gap], first["price"][-1])
    np.testing.assert_array_equal(joined["qty"][gap], 0.0)
    np.testing.assert_array_equal(joined["ofi"][gap], 0.0)


def test_concat_bars_sorts_parts_by_time(bars_factory):
    first = bars_factory(n_bars=100, start_ts=1_000_000, seed=1)
    second = bars_factory(n_bars=100, start_ts=1_000_100, seed=2)

    out_of_order = seq.concat_bars([second, first])
    in_order = seq.concat_bars([first, second])

    np.testing.assert_array_equal(out_of_order["price"], in_order["price"])


def test_concat_bars_rejects_overlapping_series(bars_factory):
    """An overlap means the same tape was passed twice — double-counting, silently."""
    first = bars_factory(n_bars=200, start_ts=1_000_000, seed=1)
    overlapping = bars_factory(n_bars=200, start_ts=1_000_100, seed=2)

    with pytest.raises(ValueError, match="overlap"):
        seq.concat_bars([first, overlapping])


def test_concat_bars_passes_a_single_part_through(bars):
    assert seq.concat_bars([bars]) is bars


def test_concat_bars_rejects_an_empty_list():
    with pytest.raises(ValueError, match="at least one"):
        seq.concat_bars([])


def test_load_bar_caches_matches_concat_bars(bars_factory, tmp_path):
    """The streaming loader and the in-memory splice must agree exactly.

    `load_bar_caches` exists only to use less peak memory; any difference in the
    series it produces would be a bug rather than a trade-off.
    """
    parts = [
        bars_factory(n_bars=300, start_ts=1_000_000, seed=1),
        bars_factory(n_bars=300, start_ts=1_000_305, seed=2),  # with a gap
        bars_factory(n_bars=300, start_ts=1_000_700, seed=3),
    ]
    paths = [seq.save_bars(p, tmp_path / f"part_{i}.npz") for i, p in enumerate(parts)]

    streamed = seq.load_bar_caches(paths)
    spliced = seq.concat_bars(parts)

    np.testing.assert_array_equal(streamed["ts"], spliced["ts"])
    np.testing.assert_allclose(streamed["price"], spliced["price"])
    np.testing.assert_allclose(streamed["ofi"], spliced["ofi"])
    assert streamed["meta"]["n_bars"] == spliced["meta"]["n_bars"]


def test_load_bar_caches_reads_a_lone_path(bars, tmp_path):
    path = seq.save_bars(bars, tmp_path / "one.npz")
    loaded = seq.load_bar_caches([path])
    np.testing.assert_array_equal(loaded["price"], bars["price"])


def test_load_bar_caches_rejects_overlapping_caches(bars_factory, tmp_path):
    a = seq.save_bars(bars_factory(n_bars=200, start_ts=1_000_000), tmp_path / "a.npz")
    b = seq.save_bars(bars_factory(n_bars=200, start_ts=1_000_100), tmp_path / "b.npz")

    with pytest.raises(ValueError, match="overlap"):
        seq.load_bar_caches([a, b])


def test_load_bar_caches_is_order_independent(bars_factory, tmp_path):
    a = seq.save_bars(bars_factory(n_bars=200, start_ts=1_000_000, seed=1), tmp_path / "a.npz")
    b = seq.save_bars(bars_factory(n_bars=200, start_ts=1_000_200, seed=2), tmp_path / "b.npz")

    np.testing.assert_allclose(
        seq.load_bar_caches([a, b])["price"], seq.load_bar_caches([b, a])["price"]
    )


# --- trade-less seconds -------------------------------------------------------


def test_empty_seconds_carry_a_forward_filled_price_and_no_flow(bars_factory):
    """~11% of real seconds have no trade; they are events, not missing data."""
    bars = bars_factory(n_bars=200, empty_every=5, seed=4)
    blank = np.zeros(200, dtype=bool)
    blank[::5] = True
    blank[0] = False

    assert np.all(bars["qty"][blank] == 0.0)
    assert np.all(bars["ofi"][blank] == 0.0)
    np.testing.assert_allclose(bars["price"][blank], bars["price"][np.flatnonzero(blank) - 1])


def test_channels_are_finite_when_seconds_are_empty(bars_factory):
    """`log1p(0)` and a zero return are both fine; a NaN here would poison training."""
    bars = bars_factory(n_bars=500, empty_every=4, seed=5)
    channels = seq.build_channels(bars, "full")

    assert np.all(np.isfinite(channels))
