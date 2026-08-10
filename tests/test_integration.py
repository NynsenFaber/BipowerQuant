"""The two orchestration loops, end to end on synthetic bars.

Everything here is marked `slow` — each test fits real models — but they are the
only tests that exercise how the pieces are wired together rather than what each
piece computes. That distinction is not academic: the `train_folds` signature
regression these tests now cover passed every unit test in the suite and still
took the whole training run down on the first call, because nothing had ever
invoked it the way `train_patchtst_local.py` does.

Configurations are deliberately degenerate in size (2 encoder layers, 32-bar
lookback, 2 epochs). A test that reproduced a published number would need the
real tape and hours of compute; these check that the loop runs, that its outputs
are shaped and aligned as the callers assume, and that its crash-safety
guarantees hold.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import TEST_BARRIER, TEST_HORIZON

import backtest as bt
import patchtst_folds as pf
import patchtst_model
import sequence_matrix as seq
import walkforward as wf
from patchtst_model import PatchTSTConfig
from patchtst_train import TrainConfig

pytestmark = pytest.mark.slow

# The lookback is shrunk to keep the model tiny; the barrier and horizon are the
# fixtures' own (`conftest.TEST_*`) rather than production's 60 bp over 3600 s.
# An hour-long horizon would leave a 12,000-bar fixture with almost no labelled
# windows, and a 60 bp barrier at any test-sized horizon resolves nothing — which
# would not fail, it would empty the side-model population and leave these tests
# asserting on nothing. What carries over is the touch rate, ~33% either way.
WINDOW = 32
HORIZON = TEST_HORIZON
BARRIER = TEST_BARRIER


@pytest.fixture
def fold_bars(bars_factory) -> dict:
    """Enough bars for two folds with a real purge between the splits."""
    return bars_factory(n_bars=12_000, seed=11)


def two_folds() -> list[wf.Fold]:
    """Hand-built folds, so the test does not need months of timestamps.

    `build_folds` derives these from the calendar and is tested separately; what
    matters here is that `train_folds` consumes the shape it produces.
    """
    return [
        wf.Fold(name="a -> b", train_lo=0, train_hi=5_000, test_lo=5_000, test_hi=8_000),
        wf.Fold(name="b -> c", train_lo=0, train_hi=8_000, test_lo=8_000, test_hi=11_800),
    ]


def tiny_model_config() -> PatchTSTConfig:
    return PatchTSTConfig(
        n_channels=2,
        seq_len=WINDOW,
        patch_len=8,
        stride=4,
        d_model=16,
        n_heads=2,
        n_layers=2,
        d_ff=32,
    )


def tiny_train_config() -> TrainConfig:
    return TrainConfig(epochs=2, batch_size=64, eval_batch_size=128, amp=False)


# --- the PatchTST fold loop ---------------------------------------------------


def test_train_folds_scores_every_test_window(fold_bars):
    """Each fold's probability vector must cover its whole test month.

    Not just the resolved windows: the gate may let an unresolved one through,
    and the backtest needs a side for every window the selection admits.
    """
    folds = two_folds()
    probabilities, meta = pf.train_folds(
        fold_bars,
        folds,
        window=WINDOW,
        horizon=HORIZON,
        barrier=BARRIER,
        config=tiny_model_config(),
        train_config=tiny_train_config(),
        train_stride=4,
        val_stride=4,
        device=torch.device("cpu"),
        verbose=False,
    )

    starts = seq.valid_window_starts(fold_bars["price"].size, WINDOW, HORIZON)
    for fold in folds:
        n_test = int(((starts >= fold.test_lo) & (starts < fold.test_hi)).sum())
        assert probabilities[fold.name].shape == (n_test,)
        assert np.all((probabilities[fold.name] >= 0.0) & (probabilities[fold.name] <= 1.0))
        assert meta["folds"][fold.name]["n_test"] == n_test


def test_train_folds_reports_metadata_the_callers_read(fold_bars):
    """`train_patchtst_local.py` prints these keys and writes them to JSON."""
    _, meta = pf.train_folds(
        fold_bars,
        two_folds(),
        window=WINDOW,
        horizon=HORIZON,
        barrier=BARRIER,
        config=tiny_model_config(),
        train_config=tiny_train_config(),
        train_stride=4,
        val_stride=4,
        device=torch.device("cpu"),
        verbose=False,
    )

    assert set(meta) >= {
        "folds",
        "train_stride",
        "elapsed_hours",
        "config",
        "train_config",
        "device",
    }
    for row in meta["folds"].values():
        assert set(row) >= {"roc_auc", "n_test", "n_resolved", "best_epoch", "epochs_run"}
        assert row["n_resolved"] <= row["n_test"]


def test_on_fold_complete_fires_once_per_fold_with_that_folds_probabilities(fold_bars):
    """The crash-safety hook: a fold's work must be persistable as soon as it ends.

    A regression test for a real failure — the callback was documented and passed
    by `train_patchtst_local.py` but not accepted by `train_folds`, so every run
    died at the first fold boundary with a TypeError.
    """
    seen: list[tuple[str, np.ndarray]] = []
    folds = two_folds()

    probabilities, _ = pf.train_folds(
        fold_bars,
        folds,
        window=WINDOW,
        horizon=HORIZON,
        barrier=BARRIER,
        config=tiny_model_config(),
        train_config=tiny_train_config(),
        train_stride=4,
        val_stride=4,
        device=torch.device("cpu"),
        verbose=False,
        on_fold_complete=lambda name, probs: seen.append((name, probs.copy())),
    )

    assert [name for name, _ in seen] == [f.name for f in folds]
    for name, probs in seen:
        np.testing.assert_array_equal(probs, probabilities[name])


def test_train_folds_writes_one_checkpoint_per_fold(fold_bars, tmp_path):
    folds = two_folds()
    pf.train_folds(
        fold_bars,
        folds,
        window=WINDOW,
        horizon=HORIZON,
        barrier=BARRIER,
        config=tiny_model_config(),
        train_config=tiny_train_config(),
        train_stride=4,
        val_stride=4,
        device=torch.device("cpu"),
        verbose=False,
        checkpoint_dir=tmp_path,
    )

    assert len(list(tmp_path.glob("patchtst_*.pt"))) == len(folds)


def test_checkpoints_record_the_fold_they_were_trained_on(fold_bars, tmp_path):
    """Without it, a directory of weights cannot be matched back to its months."""
    from patchtst_model import load_checkpoint

    pf.train_folds(
        fold_bars,
        two_folds()[:1],
        window=WINDOW,
        horizon=HORIZON,
        barrier=BARRIER,
        config=tiny_model_config(),
        train_config=tiny_train_config(),
        train_stride=4,
        val_stride=4,
        device=torch.device("cpu"),
        verbose=False,
        checkpoint_dir=tmp_path,
    )

    _, payload = load_checkpoint(next(tmp_path.glob("patchtst_*.pt")))
    data_meta = payload["data_meta"]

    assert data_meta["fold"]["name"] == "a -> b"
    assert data_meta["label_mode"] == "triple_barrier"
    assert data_meta["window"] == WINDOW
    assert data_meta["channels"] == ["log_return", "ofi"]


def test_a_time_budget_picks_a_stride_from_a_measured_probe(fold_bars):
    """The overnight guarantee: throughput is measured, not assumed."""
    _, meta = pf.train_folds(
        fold_bars,
        two_folds()[:1],
        window=WINDOW,
        horizon=HORIZON,
        barrier=BARRIER,
        config=tiny_model_config(),
        train_config=tiny_train_config(),
        train_stride=None,
        time_budget_hours=1.0,
        val_stride=4,
        device=torch.device("cpu"),
        verbose=False,
    )

    assert meta["train_stride"] >= 2
    assert np.isfinite(meta["projected_hours"])


# --- the fold loop at another lookback ----------------------------------------

# Two base windows (600 s), so the patch geometry `PatchTSTConfig.for_window`
# derives is the real one — patch 32, stride 16, the same 37 tokens the study's
# three scales all produce — rather than the hand-picked tiny geometry above. The
# encoder is still shrunk, because what is under test is the loop, not the model.
WIDE_WINDOW = 2 * seq.WINDOW_SCALES["short"]


def wide_model_config() -> PatchTSTConfig:
    return PatchTSTConfig.for_window(
        WIDE_WINDOW, n_channels=2, d_model=16, n_heads=2, n_layers=1, d_ff=32
    )


def test_the_fold_loop_runs_at_a_lookback_it_was_not_written_for(fold_bars):
    """Nothing in `train_folds` is pinned to the 5-minute window.

    The whole study rests on running this loop unchanged at three lookbacks, so a
    hardcoded 300 anywhere in it would not be a bug in one run — it would silently
    make two of the three runs describe the wrong window.
    """
    cfg = wide_model_config()
    assert (cfg.patch_len, cfg.stride, cfg.num_patches) == (32, 16, 37)

    folds = two_folds()
    probabilities, meta = pf.train_folds(
        fold_bars,
        folds,
        window=WIDE_WINDOW,
        horizon=HORIZON,
        barrier=BARRIER,
        config=cfg,
        train_config=tiny_train_config(),
        train_stride=4,
        val_stride=4,
        device=torch.device("cpu"),
        verbose=False,
    )

    starts = seq.valid_window_starts(fold_bars["price"].size, WIDE_WINDOW, HORIZON)
    for fold in folds:
        n_test = int(((starts >= fold.test_lo) & (starts < fold.test_hi)).sum())
        assert probabilities[fold.name].shape == (n_test,)

    assert meta["window"] == WIDE_WINDOW
    assert meta["window_scale"] == f"{WIDE_WINDOW}s"


def test_a_config_that_disagrees_with_the_window_is_refused(fold_bars):
    """The batcher gathers `window` bars; the model patches `seq_len` of them.

    A mismatch is not an error anywhere downstream — it is a model reading a
    window of the wrong length and scoring perfectly plausible probabilities from
    it. So it is refused at the door.
    """
    with pytest.raises(ValueError, match="seq_len"):
        pf.train_folds(
            fold_bars,
            two_folds()[:1],
            window=WIDE_WINDOW,
            horizon=HORIZON,
            barrier=BARRIER,
            config=tiny_model_config(),  # seq_len = 32, not 600
            train_config=tiny_train_config(),
            train_stride=4,
            device=torch.device("cpu"),
            verbose=False,
        )


def test_an_unaffordable_batch_is_clamped_before_the_run_starts(fold_bars, monkeypatch):
    """Discovering this as an OOM twenty minutes into a fold is the failure mode.

    The patched tensor grows with the patch length, so the batch size the project
    has always used is 288x more memory at the 24-hour lookback than at the
    5-minute one. The budget is squeezed here rather than building a fixture with
    a day of bars in it.
    """
    cfg = wide_model_config()
    per_window = cfg.n_channels * cfg.num_patches * cfg.patch_len
    monkeypatch.setattr(patchtst_model, "PATCH_TENSOR_BUDGET", per_window * 40)

    _, meta = pf.train_folds(
        fold_bars,
        two_folds()[:1],
        window=WIDE_WINDOW,
        horizon=HORIZON,
        barrier=BARRIER,
        config=cfg,
        train_config=tiny_train_config(),  # asks for 64
        train_stride=4,
        val_stride=4,
        device=torch.device("cpu"),
        verbose=False,
    )

    # Asked for 64 and 128; the budget allows 40, and the clamp rounds down to a
    # power of two so two runs at different budgets stay comparable.
    assert meta["train_config"]["batch_size"] == 32
    assert meta["train_config"]["eval_batch_size"] == 32


# --- the walk-forward loop ----------------------------------------------------


def _fold_inputs(bars):
    full = seq.build_channels(bars, "full")
    starts = seq.valid_window_starts(bars["price"].size, WINDOW, HORIZON)
    X = seq.build_tabular_features(full, bars["price"], starts, WINDOW)
    precomputed = bt.barrier_arrays(bars["price"], HORIZON, BARRIER)
    side, _, defined = precomputed
    label_bar = starts + WINDOW - 1
    y_side = (side[label_bar] > 0).astype(np.int8)
    touched = ((side[label_bar] != 0) & defined[label_bar]).astype(np.int8)
    return X, starts, y_side, touched, precomputed


def test_run_fold_fits_a_gate_and_every_side_model(fold_bars):
    X, starts, y_side, touched, _ = _fold_inputs(fold_bars)

    result = wf.run_fold(
        fold_bars,
        two_folds()[0],
        X,
        starts,
        y_side,
        touched,
        WINDOW,
        HORIZON,
        BARRIER,
        purge=WINDOW + HORIZON - 1,
        patchtst_probs=None,
        train_stride=8,
    )

    assert result["n_train"] > 0 and result["n_test"] > 0
    assert set(result["gate"]) >= {"trained_auc", "rv_auc", "touch_rate"}
    assert result["probabilities"], "no side models were fitted"
    for name, probs in result["probabilities"].items():
        assert probs.shape == result["starts_test"].shape, name
        assert np.all((probs >= 0.0) & (probs <= 1.0)), name


def test_supplied_patchtst_probabilities_join_the_model_grid(fold_bars):
    """The integration point: an `.npz` from `train_folds` scored in the same backtest."""
    X, starts, y_side, touched, _ = _fold_inputs(fold_bars)
    fold = two_folds()[0]
    n_test = int(((starts >= fold.test_lo) & (starts < fold.test_hi)).sum())
    supplied = {fold.name: np.full(n_test, 0.6, dtype=np.float32)}

    result = wf.run_fold(
        fold_bars,
        fold,
        X,
        starts,
        y_side,
        touched,
        WINDOW,
        HORIZON,
        BARRIER,
        purge=WINDOW + HORIZON - 1,
        patchtst_probs=supplied,
        train_stride=8,
    )

    patchtst = [k for k in result["probabilities"] if "patchtst" in k.lower()]
    assert patchtst, f"PatchTST missing from {list(result['probabilities'])}"


def test_misaligned_probabilities_are_skipped_loudly_not_scored(fold_bars, capsys):
    """A length mismatch means the vector describes a different window population.

    Scoring it anyway would silently compare one month's predictions against
    another's labels. The run continues — a bad `.npz` should not cost the whole
    grid — but PatchTST is dropped from this fold and the reason is printed, so
    the partial-coverage warning at the end of `main()` can flag the result as
    not comparable.
    """
    X, starts, y_side, touched, _ = _fold_inputs(fold_bars)
    fold = two_folds()[0]

    result = wf.run_fold(
        fold_bars,
        fold,
        X,
        starts,
        y_side,
        touched,
        WINDOW,
        HORIZON,
        BARRIER,
        purge=WINDOW + HORIZON - 1,
        patchtst_probs={fold.name: np.full(7, 0.5, dtype=np.float32)},
        train_stride=8,
    )

    assert not [k for k in result["probabilities"] if "patchtst" in k.lower()]
    assert "skipping" in capsys.readouterr().out


def test_backtest_fold_prices_every_model_gate_and_threshold(fold_bars):
    X, starts, y_side, touched, precomputed = _fold_inputs(fold_bars)
    result = wf.run_fold(
        fold_bars,
        two_folds()[0],
        X,
        starts,
        y_side,
        touched,
        WINDOW,
        HORIZON,
        BARRIER,
        purge=WINDOW + HORIZON - 1,
        patchtst_probs=None,
        train_stride=8,
    )

    rows = wf.backtest_fold(
        fold_bars,
        result,
        WINDOW,
        HORIZON,
        BARRIER,
        bt.Costs(half_spread_bps=0.0),
        bt.Execution(),
        precomputed,
    )

    assert rows
    # Derived, not written out: the round trip follows `Costs` defaults, and the
    # point of the assertion is that every row was priced with the costs it was
    # handed rather than that the default happens to be some particular number.
    expected_cost = bt.breakeven_barrier_bps(bt.Costs(half_spread_bps=0.0), bt.Execution())
    for row in rows:
        assert row["n_trades"] >= 0
        assert row["cost_bps"] == pytest.approx(expected_cost)
        assert "daily_pnl_bps" in row


def test_the_full_grid_pools_into_comparable_rows(fold_bars):
    """run_fold -> backtest_fold -> pool_daily, the sequence `main()` runs."""
    X, starts, y_side, touched, precomputed = _fold_inputs(fold_bars)
    rows = []
    for fold in two_folds():
        result = wf.run_fold(
            fold_bars,
            fold,
            X,
            starts,
            y_side,
            touched,
            WINDOW,
            HORIZON,
            BARRIER,
            purge=WINDOW + HORIZON - 1,
            patchtst_probs=None,
            train_stride=8,
        )
        rows += wf.backtest_fold(
            fold_bars,
            result,
            WINDOW,
            HORIZON,
            BARRIER,
            bt.Costs(half_spread_bps=0.0),
            bt.Execution(),
            precomputed,
        )

    pooled = wf.pool_daily(rows)

    assert pooled
    assert max(row["n_folds"] for row in pooled.values()) == 2


# --- the sequence dataset -----------------------------------------------------


def test_build_sequence_dataset_produces_purged_ordered_splits(fold_bars):
    dataset = seq.build_sequence_dataset(
        fold_bars, window=WINDOW, horizon=HORIZON, barrier=BARRIER, train_frac=0.7, val_frac=0.1
    )

    train, val, test = (dataset.splits[k] for k in ("train", "val", "test"))
    assert train.max() < val.min() < test.min()
    assert val.min() - train.max() >= WINDOW + HORIZON - 1
    assert isinstance(dataset.summary(), str)


def test_dataset_labels_and_positive_rate_line_up(fold_bars):
    dataset = seq.build_sequence_dataset(fold_bars, window=WINDOW, horizon=HORIZON, barrier=BARRIER)

    for split in ("train", "val", "test"):
        labels = dataset.labels(split)
        assert labels.size == dataset.splits[split].size
        assert 0.0 <= dataset.positive_rate(split) <= 1.0


def test_dataset_channels_follow_the_requested_layout(fold_bars):
    dataset = seq.build_sequence_dataset(
        fold_bars, window=WINDOW, horizon=HORIZON, barrier=BARRIER, channel_set="full"
    )

    assert dataset.channels.shape[1] == len(seq.CHANNEL_SETS["full"])


# --- bootstrap intervals ------------------------------------------------------


def test_block_bootstrap_brackets_a_genuine_edge_above_a_coin_flip():
    rng = np.random.default_rng(0)
    y = (rng.random(4_000) < 0.5).astype(np.int8)
    score = y * 0.3 + rng.random(4_000) * 0.7  # a genuine but noisy edge

    out = seq.block_bootstrap_auc(y, score, n_boot=200, block=360)

    assert out["lo"] < out["hi"]
    assert out["lo"] > 0.5, "a real edge should keep the interval clear of chance"
    assert out["p_le_half"] == pytest.approx(0.0)
    assert out["n_boot"] > 0 and out["block"] == 360


def test_block_bootstrap_needs_more_windows_than_one_block():
    with pytest.raises(ValueError, match="block-bootstrap"):
        seq.block_bootstrap_auc(
            np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), n_boot=10, block=360
        )


def test_block_bootstrap_reports_no_edge_for_a_random_score():
    """`p_le_half` is the number to read when asking whether an edge exists at all."""
    rng = np.random.default_rng(1)
    y = (rng.random(4_000) < 0.5).astype(np.int8)

    out = seq.block_bootstrap_auc(y, rng.random(4_000), n_boot=200, block=360)

    assert out["p_le_half"] > 0.1


def test_block_bootstrap_intervals_widen_with_the_block_length():
    """The whole argument for using blocks: i.i.d. resampling is too narrow.

    Consecutive windows share almost all of their bars, so resampling them singly
    treats hundreds of correlated observations as independent.
    """
    rng = np.random.default_rng(2)
    y = (rng.random(4_000) < 0.5).astype(np.int8)
    score = y * 0.2 + rng.random(4_000) * 0.8

    narrow = seq.block_bootstrap_auc(y, score, n_boot=300, block=1, seed=5)
    wide = seq.block_bootstrap_auc(y, score, n_boot=300, block=360, seed=5)

    assert (wide["hi"] - wide["lo"]) > (narrow["hi"] - narrow["lo"])
