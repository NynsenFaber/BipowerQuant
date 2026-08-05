"""The tabular feature matrix and the per-fold row selection.

`fold_rows` gets the most attention: it is where the purge between training and
validation actually happens, and a leak there would inflate every PatchTST number
in the README without failing anything. The rest covers `ml_matrix`, which is the
C++ engine's only caller, and the throughput/stride arithmetic that decides
whether an overnight run fits its budget.
"""

from __future__ import annotations

import numpy as np
import pytest

import patchtst_folds as pf
import sequence_matrix as seq
import walkforward as wf

ml_matrix = pytest.importorskip(
    "ml_matrix", reason="needs the C++ extension; run `python build.py`"
)


# --- the tabular matrix -------------------------------------------------------


def test_training_matrix_has_one_row_per_resolved_window(bars):
    X, y, starts = ml_matrix.build_training_matrix(bars, window=50, horizon=20)

    assert X.shape[0] == y.size == starts.size
    assert X.shape[1] == len(seq.TABULAR_FEATURE_NAMES) == 7
    assert set(np.unique(y)).issubset({0, 1})


def test_training_matrix_drops_windows_with_no_side(bars):
    """The side model has nothing to learn from a window that never resolved."""
    X, _, starts = ml_matrix.build_training_matrix(bars, window=50, horizon=20)
    side, _, defined = seq.triple_barrier(bars["price"], horizon=20, barrier=seq.BARRIER)

    label_bar = starts + 50 - 1
    assert np.all(side[label_bar] != 0)
    assert np.all(defined[label_bar])
    # Some windows must have been dropped, or the filter is not being exercised.
    assert X.shape[0] < seq.valid_window_starts(bars["price"].size, 50, 20).size


def test_the_gate_label_keeps_every_window(bars):
    """`barrier_touched` is defined everywhere, so nothing is filtered out."""
    _, y, starts = ml_matrix.build_training_matrix(
        bars, window=50, horizon=20, label_mode="barrier_touched"
    )

    assert starts.size == seq.valid_window_starts(bars["price"].size, 50, 20).size
    assert 0 < y.mean() < 1, "the gate label should not be degenerate on this fixture"


def test_training_matrix_is_finite(bars):
    X, _, _ = ml_matrix.build_training_matrix(bars, window=50, horizon=20)
    assert np.all(np.isfinite(X))


def test_rolling_metrics_emits_one_row_per_window_start(bars):
    metrics = ml_matrix.rolling_metrics(bars, window=50)
    expected = bars["price"].size - 50 + 1

    assert all(len(v) == expected for v in metrics.values())


def test_rolling_metrics_recovers_the_signed_flow(bars):
    """`ofi` is fed in as (|qty|, sign) and must come back out unchanged."""
    metrics = ml_matrix.rolling_metrics(bars, window=1_000)
    engine_ofi = np.asarray(metrics["order_flow_imbalance"])

    direct = np.convolve(bars["ofi"].astype(np.float64), np.ones(1_000), mode="valid")
    np.testing.assert_allclose(engine_ofi, direct, rtol=1e-6, atol=1e-4)


def test_the_two_feature_paths_agree(bars):
    """`ml_matrix` (C++ loop) against `build_tabular_features` (cumulative sums).

    They are separate implementations of the same seven features and the README
    quotes the C++ one, so a divergence beyond the documented ~5e-8 cancellation
    error would mean the two halves of the project describe different windows.
    """
    window, horizon = 50, 20
    X_cpp, _, starts = ml_matrix.build_training_matrix(bars, window=window, horizon=horizon)

    channels = seq.build_channels(bars, "full")
    X_py = seq.build_tabular_features(channels, bars["price"], starts, window)

    # Columns 0-3 are window sums; 4-6 are derived from them.
    np.testing.assert_allclose(X_cpp[:, 0], X_py[:, 0], rtol=1e-5)
    np.testing.assert_allclose(X_cpp[:, 1], X_py[:, 1], rtol=1e-5)
    np.testing.assert_allclose(X_cpp[:, 4], X_py[:, 4], rtol=1e-9)


# --- per-fold row selection ---------------------------------------------------


def _fold(train_lo=0, train_hi=10_000, test_lo=10_000, test_hi=12_000):
    return wf.Fold(name="test", train_lo=train_lo, train_hi=train_hi,
                   test_lo=test_lo, test_hi=test_hi)


def _touched(n, every=3):
    mask = np.zeros(n, dtype=bool)
    mask[::every] = True
    return mask


def test_fold_rows_splits_train_val_and_test():
    starts = np.arange(12_000, dtype=np.int64)
    train, val, test = pf.fold_rows(_fold(), starts, _touched(12_000), purge=359)

    assert train.size and val.size and test.size
    # Positions index into `starts`, so they must stay in range.
    assert train.max() < starts.size and test.max() < starts.size


def test_validation_is_the_last_slice_of_training_not_a_random_sample():
    """Early stopping on a random split selects against forward generalisation.

    Time is the only direction that matters here, so validation has to sit after
    everything the model trained on.
    """
    starts = np.arange(12_000, dtype=np.int64)
    train, val, _ = pf.fold_rows(_fold(), starts, _touched(12_000), purge=359)

    assert starts[train].max() < starts[val].min()


def test_a_purge_separates_training_from_validation():
    """Without it, training labels are drawn from moves inside a validation lookback."""
    starts = np.arange(12_000, dtype=np.int64)
    purge = 359
    train, val, _ = pf.fold_rows(_fold(), starts, _touched(12_000), purge=purge)

    assert starts[val].min() - starts[train].max() >= purge


def test_training_rows_stop_short_of_the_test_month():
    """The boundary every out-of-sample number in the README depends on."""
    starts = np.arange(12_000, dtype=np.int64)
    fold = _fold(train_hi=10_000, test_lo=10_000)
    train, val, test = pf.fold_rows(fold, starts, _touched(12_000), purge=359)

    assert starts[train].max() < fold.train_hi - 359
    assert starts[val].max() < fold.train_hi
    assert starts[test].min() >= fold.test_lo


def test_training_and_validation_keep_only_resolved_windows():
    """A window with no side has nothing to teach a side model."""
    starts = np.arange(12_000, dtype=np.int64)
    touched = _touched(12_000, every=4)
    train, val, _ = pf.fold_rows(_fold(), starts, touched, purge=359)

    assert np.all(touched[starts[train]])
    assert np.all(touched[starts[val]])


def test_the_test_split_keeps_every_window_resolved_or_not():
    """The gate may let an unresolved window through, so the backtest needs a score.

    This is why `train_folds` scores all of a month rather than its resolved subset.
    """
    starts = np.arange(12_000, dtype=np.int64)
    touched = _touched(12_000, every=4)
    _, _, test = pf.fold_rows(_fold(), starts, touched, purge=359)

    assert not np.all(touched[starts[test]])
    assert test.size == 2_000


def test_val_frac_controls_the_split_size():
    starts = np.arange(12_000, dtype=np.int64)
    touched = _touched(12_000)

    _, small, _ = pf.fold_rows(_fold(), starts, touched, purge=359, val_frac=0.05)
    _, large, _ = pf.fold_rows(_fold(), starts, touched, purge=359, val_frac=0.30)

    assert small.size < large.size


def test_fold_rows_never_returns_an_empty_training_split():
    """A tiny fold must still yield something rather than an empty tensor."""
    starts = np.arange(1_000, dtype=np.int64)
    fold = _fold(train_lo=0, train_hi=600, test_lo=600, test_hi=1_000)
    train, _, _ = pf.fold_rows(fold, starts, _touched(1_000), purge=359)

    assert train.size >= 1


# --- the time budget ----------------------------------------------------------


def test_stride_grows_until_the_projection_fits():
    """Halving the budget cannot lower the stride the projection needs."""
    strides = [
        pf.stride_for_budget(10_000_000, epochs=10, rate=1_500.0, budget_hours=budget)[0]
        for budget in (8.0, 4.0, 2.0, 1.0)
    ]
    assert strides == sorted(strides)


def test_a_generous_budget_uses_the_minimum_stride():
    stride, hours = pf.stride_for_budget(10_000, epochs=1, rate=10_000.0, budget_hours=100.0)

    assert stride == 2
    assert hours < 100.0


def test_the_stride_is_capped_so_a_run_never_thins_to_nothing():
    """Past the cap the run is reported as not fitting rather than striding further."""
    stride, hours = pf.stride_for_budget(
        10_000_000_000, epochs=50, rate=1.0, budget_hours=0.001, maximum=64
    )

    assert stride == 64
    assert hours > 0.001, "an unfittable job must project over its budget, not under"


def test_the_projection_scales_with_the_work():
    """Twice the rows at the same stride is twice the projected time."""
    _, small = pf.stride_for_budget(1_000_000, 10, 1_500.0, 1e9)
    _, large = pf.stride_for_budget(2_000_000, 10, 1_500.0, 1e9)

    assert large == pytest.approx(2 * small)


def test_the_projection_includes_the_validation_overhead():
    """Validation costs ~1.25x the training pass and is not free."""
    _, hours = pf.stride_for_budget(3_600_000, epochs=1, rate=1_000.0, budget_hours=1e9)

    # 3.6M rows / stride 2 / 1000 per s = 1800 s = 0.5 h, times the 1.25 overhead.
    assert hours == pytest.approx(0.5 * pf.VAL_OVERHEAD, rel=1e-6)
