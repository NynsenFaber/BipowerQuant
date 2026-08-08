"""Fold construction and cross-fold pooling.

Folds are where a leak would be invisible and fatal: every AUC in the README is
conditioned on a training set that must contain nothing from its test month. The
tests here assert that boundary directly rather than trusting the arithmetic that
produces it, and they assert the calendar edge cases (December wrapping into
January, a month that is entirely absent) that a six-month sample never exercises.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import walkforward as wf


def month_timestamps(months: list[str], per_month: int = 200) -> np.ndarray:
    """A sorted timestamp array covering whole calendar months.

    `month_blocks` finds boundaries by binary search, so it needs the stamps
    sorted but not contiguous — a few hundred a month stands in for the 2.6 M the
    real tape carries, and keeps the tests instant.
    """
    stamps = []
    for label in months:
        year, month = (int(part) for part in label.split("-"))
        start = datetime(year, month, 1, tzinfo=UTC)
        end = datetime(year + (month == 12), month % 12 + 1, 1, tzinfo=UTC)
        span = (end - start).total_seconds()
        step = span / per_month
        stamps += [int((start + timedelta(seconds=i * step)).timestamp()) for i in range(per_month)]
    return np.array(sorted(stamps), dtype=np.int64)


SIX_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]


# --- month blocks -------------------------------------------------------------


def test_month_blocks_finds_every_month():
    blocks = wf.month_blocks(month_timestamps(SIX_MONTHS))

    assert [label for label, _, _ in blocks] == SIX_MONTHS


def test_month_blocks_partition_the_series_without_gaps_or_overlaps():
    ts = month_timestamps(SIX_MONTHS)
    blocks = wf.month_blocks(ts)

    assert blocks[0][1] == 0
    assert blocks[-1][2] == ts.size
    for (_, _, end), (_, next_start, _) in zip(blocks, blocks[1:]):
        assert end == next_start


def test_month_blocks_wraps_december_into_the_next_january():
    """The `month % 12 + 1` year rollover, which a Jan-June sample never hits."""
    blocks = wf.month_blocks(month_timestamps(["2026-11", "2026-12", "2027-01"]))

    assert [label for label, _, _ in blocks] == ["2026-11", "2026-12", "2027-01"]


def test_month_blocks_handles_a_single_month():
    blocks = wf.month_blocks(month_timestamps(["2026-03"]))

    assert len(blocks) == 1
    assert blocks[0][0] == "2026-03"


def test_month_blocks_skips_a_month_with_no_data():
    """A missing archive must not silently shift later months onto wrong labels."""
    blocks = wf.month_blocks(month_timestamps(["2026-01", "2026-04"]))

    assert [label for label, _, _ in blocks] == ["2026-01", "2026-04"]


# --- month slicing, for choosing a target on training data only ----------------


def month_bars(months: list[str], per_month: int = 200) -> dict:
    """A bar dict spanning whole months, shaped like `sequence_matrix.load_bars`."""
    ts = month_timestamps(months, per_month)
    n = ts.size
    return {
        "ts": ts,
        "price": np.linspace(70_000.0, 71_000.0, n),
        "qty": np.ones(n, dtype=np.float32),
        "n_trades": np.full(n, 4.0, dtype=np.float32),
        "ofi": np.zeros(n, dtype=np.float32),
        "meta": {
            "source": "synthetic",
            "n_bars": n,
            "first_ts": int(ts[0]),
            "last_ts": int(ts[-1]),
        },
    }


def test_slice_months_keeps_only_the_named_months():
    sliced = wf.slice_months(month_bars(SIX_MONTHS), ["2026-01", "2026-02", "2026-03"])

    assert [label for label, _, _ in wf.month_blocks(sliced["ts"])] == [
        "2026-01",
        "2026-02",
        "2026-03",
    ]
    assert sliced["meta"]["n_bars"] == sliced["ts"].size
    assert sliced["meta"]["months"] == ["2026-01", "2026-02", "2026-03"]


def test_slice_months_slices_every_series_together():
    """A bar dict whose arrays disagree on length would corrupt every downstream label."""
    bars = month_bars(SIX_MONTHS)
    sliced = wf.slice_months(bars, ["2026-02", "2026-03"])

    lengths = {key: sliced[key].size for key in ("ts", "price", "qty", "n_trades", "ofi")}
    assert len(set(lengths.values())) == 1
    assert sliced["price"][0] == bars["price"][wf.month_blocks(bars["ts"])[1][1]]


def test_slice_months_rejects_a_month_that_is_not_there():
    with pytest.raises(SystemExit, match="2026-09"):
        wf.slice_months(month_bars(SIX_MONTHS), ["2026-01", "2026-09"])


def test_slice_months_rejects_a_gap_that_would_fake_a_jump():
    """Splicing January onto March leaves a price seam a triple barrier reads as a move."""
    with pytest.raises(SystemExit, match="contiguous"):
        wf.slice_months(month_bars(SIX_MONTHS), ["2026-01", "2026-03"])


def test_slice_months_accepts_the_months_in_any_order():
    out_of_order = wf.slice_months(month_bars(SIX_MONTHS), ["2026-03", "2026-01", "2026-02"])
    in_order = wf.slice_months(month_bars(SIX_MONTHS), ["2026-01", "2026-02", "2026-03"])

    np.testing.assert_array_equal(out_of_order["ts"], in_order["ts"])


# --- fold construction --------------------------------------------------------


def test_anchored_folds_match_the_documented_schedule():
    """Jan-Mar -> Apr, Jan-Apr -> May, Jan-May -> Jun, exactly as the README says."""
    folds = wf.build_folds(month_timestamps(SIX_MONTHS), "anchored", train_months=3)

    assert [f.name for f in folds] == [
        "2026-01..2026-03 -> 2026-04",
        "2026-01..2026-04 -> 2026-05",
        "2026-01..2026-05 -> 2026-06",
    ]


def test_anchored_training_windows_expand_from_a_fixed_start():
    folds = wf.build_folds(month_timestamps(SIX_MONTHS), "anchored", train_months=3)

    assert all(f.train_lo == 0 for f in folds)
    assert [f.train_hi for f in folds] == sorted(f.train_hi for f in folds)


def test_rolling_training_windows_slide_instead_of_expanding():
    folds = wf.build_folds(month_timestamps(SIX_MONTHS), "rolling", train_months=3)

    assert [f.train_lo for f in folds] == sorted(f.train_lo for f in folds)
    assert folds[0].train_lo < folds[-1].train_lo, "a rolling window must move"
    assert folds[0].name == "2026-01..2026-03 -> 2026-04"


@pytest.mark.parametrize("scheme", ["anchored", "rolling", "holdout"])
def test_training_always_ends_before_testing_begins(scheme):
    """The property every reported number depends on: no future in the training set."""
    folds = wf.build_folds(month_timestamps(SIX_MONTHS), scheme, train_months=3)

    assert folds, "a scheme must produce at least one fold"
    for fold in folds:
        assert fold.train_hi <= fold.test_lo, f"{fold.name} trains on its own test period"
        assert fold.train_lo < fold.train_hi
        assert fold.test_lo < fold.test_hi


def test_test_months_are_disjoint_across_folds():
    """Pooling concatenates them into one out-of-sample track, so they must not overlap."""
    folds = wf.build_folds(month_timestamps(SIX_MONTHS), "anchored", train_months=3)

    for earlier, later in zip(folds, folds[1:]):
        assert earlier.test_hi <= later.test_lo


def test_holdout_splits_the_sample_in_half():
    folds = wf.build_folds(month_timestamps(SIX_MONTHS), "holdout", train_months=3)

    assert len(folds) == 1
    assert folds[0].name == "2026-01..2026-03 -> 2026-04..2026-06"


def test_more_months_give_more_folds():
    ts = month_timestamps(SIX_MONTHS)
    assert len(wf.build_folds(ts, "anchored", train_months=3)) == 3
    assert len(wf.build_folds(ts, "anchored", train_months=4)) == 2
    assert len(wf.build_folds(ts, "anchored", train_months=5)) == 1


def test_too_few_months_is_rejected_with_an_actionable_message():
    ts = month_timestamps(["2026-01", "2026-02"])

    with pytest.raises(SystemExit, match="train-months"):
        wf.build_folds(ts, "anchored", train_months=3)


def test_exactly_enough_months_is_still_too_few():
    """Three training months and nothing left to test on is not a fold."""
    with pytest.raises(SystemExit):
        wf.build_folds(month_timestamps(SIX_MONTHS[:3]), "anchored", train_months=3)


def test_fold_serialises_to_a_dict():
    fold = wf.build_folds(month_timestamps(SIX_MONTHS), "anchored", 3)[0]
    payload = fold.as_dict()

    assert payload["name"] == fold.name
    assert set(payload) == {"name", "train_lo", "train_hi", "test_lo", "test_hi"}


# --- helpers ------------------------------------------------------------------


def test_auc_of_a_single_class_split_is_nan_not_a_half():
    """A degenerate split has no AUC; reporting 0.5 would look like a coin flip."""
    assert np.isnan(wf._safe_auc(np.ones(10), np.random.default_rng(0).random(10)))
    assert np.isnan(wf._safe_auc(np.zeros(10), np.random.default_rng(0).random(10)))
    assert np.isnan(wf._safe_auc(np.array([]), np.array([])))


def test_auc_of_a_perfect_ranking_is_one():
    y = np.array([0, 0, 1, 1])
    assert wf._safe_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_xgb_reweights_the_positive_class_from_the_training_split():
    """`scale_pos_weight` is recomputed per fold rather than assumed."""
    balanced = wf._xgb(np.array([0, 1] * 50))
    skewed = wf._xgb(np.array([0] * 90 + [1] * 10))

    assert balanced.scale_pos_weight == pytest.approx(1.0)
    assert skewed.scale_pos_weight == pytest.approx(9.0)


def test_xgb_survives_a_split_with_no_positives():
    """Division by a zero positive count would otherwise take the run down."""
    assert wf._xgb(np.zeros(50)).scale_pos_weight == 1.0


# --- pooling ------------------------------------------------------------------


def _row(
    fold,
    model="xgb",
    gate="none",
    q=0.0,
    n_trades=100,
    days=(0, 1, 2),
    pnl=(1.0, -2.0, 3.0),
    hit=0.5,
    gross=0.01,
    net=-5.99,
):
    return {
        "fold": fold,
        "model": model,
        "gate": gate,
        "gate_quantile": q,
        "gate_keep_frac": 1.0,
        "n_trades": n_trades,
        "hit_rate": hit,
        "resolved_share": 0.4,
        "resolved_hit_rate": 0.51,
        "gross_bps": gross,
        "net_bps": net,
        "cost_bps": 6.0,
        "mean_hold_s": 42.0,
        "daily_day": np.array(days),
        "daily_pnl_bps": np.array(pnl),
    }


def test_pooling_groups_by_model_gate_and_threshold():
    rows = [
        _row("f1", model="xgb"),
        _row("f2", model="xgb"),
        _row("f1", model="logistic"),
    ]
    pooled = wf.pool_daily(rows)

    assert len(pooled) == 2
    assert pooled[("xgb", "none", 0.0)]["n_folds"] == 2
    assert pooled[("logistic", "none", 0.0)]["n_folds"] == 1


def test_pooling_concatenates_daily_series_across_folds():
    rows = [
        _row("f1", days=(0, 1), pnl=(1.0, 2.0), n_trades=10),
        _row("f2", days=(2, 3), pnl=(3.0, 4.0), n_trades=10),
    ]
    pooled = wf.pool_daily(rows)[("xgb", "none", 0.0)]

    assert pooled["n_days"] == 4
    assert pooled["n_trades"] == 20
    assert pooled["total_return"] == pytest.approx(10.0 * 1e-4)


def test_pooled_averages_are_weighted_by_trade_count():
    """A fold with ten times the trades must carry ten times the weight."""
    rows = [
        _row("f1", n_trades=100, net=-1.0, days=(0,), pnl=(1.0,)),
        _row("f2", n_trades=900, net=-11.0, days=(1,), pnl=(1.0,)),
    ]
    pooled = wf.pool_daily(rows)[("xgb", "none", 0.0)]

    # (100*-1 + 900*-11) / 1000
    assert pooled["net_bps"] == pytest.approx(-10.0)


def test_pooling_sorts_days_so_the_equity_curve_is_chronological():
    rows = [
        _row("late", days=(5, 6), pnl=(1.0, 1.0)),
        _row("early", days=(0, 1), pnl=(-5.0, -5.0)),
    ]
    pooled = wf.pool_daily(rows)[("xgb", "none", 0.0)]

    # Drawdown is path-dependent: unsorted, the two losses would come last and
    # the reported drawdown would be wrong.
    assert pooled["max_drawdown"] == pytest.approx(-10.0 * 1e-4)


def test_pooling_counts_folds_so_a_partial_model_is_visible():
    """A model present in fewer folds is pooled over fewer months and is not comparable."""
    rows = [_row("f1", model="patchtst"), _row("f1", model="xgb"), _row("f2", model="xgb")]
    pooled = wf.pool_daily(rows)

    assert pooled[("patchtst", "none", 0.0)]["n_folds"] == 1
    assert pooled[("xgb", "none", 0.0)]["n_folds"] == 2


def test_pooling_an_empty_list_returns_nothing():
    assert wf.pool_daily([]) == {}


def test_resolved_hit_rate_is_nan_when_nothing_resolved():
    row = _row("f1")
    row["resolved_share"] = 0.0
    pooled = wf.pool_daily([row])[("xgb", "none", 0.0)]

    assert np.isnan(pooled["resolved_hit_rate"])
