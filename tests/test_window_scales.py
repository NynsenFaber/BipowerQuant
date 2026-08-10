"""The three lookbacks, and the property that makes comparing them mean anything.

The study asks one question — is five minutes of history too little to decide an
hour-long trade? — by running the identical pipeline at 5 minutes, 2 hours and 24
hours. That answer is only worth reading if the three runs differ in the lookback
*and nothing else*, so the invariants that guarantee it are asserted here rather
than argued in the README:

* every scale patches to the **same 37 tokens**, so attention cost, head width and
  therefore model capacity do not drift with the lookback;
* the longer scales **discard nothing** — every second still reaches the network,
  through a wider patch embedding rather than a coarser grid;
* the window count and the purge both move with the lookback in the way the fold
  arithmetic assumes.

The failure these defend against is quiet. A 24-hour model that patched at P=16
would be a 10,800-token Transformer: it would train, it would score, and any
difference from the 5-minute model would be uninterpretable — capacity, compute
and lookback would all have changed at once.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import patchtst_folds as pf
import sequence_matrix as seq
import walkforward as wf
from patchtst_model import PatchTSTClassifier, PatchTSTConfig, batch_size_for

# --- the registry -------------------------------------------------------------


def test_the_three_scales_are_the_documented_lengths():
    assert seq.WINDOW_SCALES == {"short": 300, "mid": 7200, "long": 86400}


def test_every_scale_is_a_whole_multiple_of_the_short_one():
    """The patch geometry only holds the token count constant at whole multiples.

    This is the constraint that picked 7200 and 86400 rather than, say, 5400 —
    see `PatchTSTConfig.for_window`.
    """
    base = seq.WINDOW_SCALES["short"]
    for name, window in seq.WINDOW_SCALES.items():
        assert window % base == 0, f"{name} ({window}) is not a multiple of {base}"


def test_the_default_window_is_still_the_short_scale():
    """Every caller that never heard of the scales keeps the original behaviour."""
    assert seq.WINDOW_SIZE == seq.WINDOW_SCALES[seq.DEFAULT_SCALE] == 300


def test_window_for_and_scale_of_round_trip():
    for name, window in seq.WINDOW_SCALES.items():
        assert seq.window_for(name) == window
        assert seq.scale_of(window) == name


def test_window_for_rejects_an_unknown_scale():
    with pytest.raises(ValueError, match="Unknown window scale"):
        seq.window_for("medium")


def test_scale_of_names_an_unregistered_window_by_its_length():
    """A one-off `--window 1800` must not collide with a named run's output."""
    assert seq.scale_of(1800) == "1800s"


# --- equal capacity across scales ---------------------------------------------


def config_at(scale: str, **overrides) -> PatchTSTConfig:
    return PatchTSTConfig.for_window(seq.window_for(scale), n_channels=2, **overrides)


@pytest.mark.parametrize("scale", ["short", "mid", "long"])
def test_every_scale_patches_to_the_same_token_count(scale):
    """37 tokens at 5 minutes, at 2 hours and at 24 hours."""
    assert config_at(scale).num_patches == 37


def test_the_patch_geometry_scales_with_the_lookback():
    assert (config_at("short").patch_len, config_at("short").stride) == (16, 8)
    assert (config_at("mid").patch_len, config_at("mid").stride) == (384, 192)
    assert (config_at("long").patch_len, config_at("long").stride) == (4608, 2304)


def test_for_window_rejects_a_window_that_is_not_a_whole_multiple():
    with pytest.raises(ValueError, match="not a multiple"):
        PatchTSTConfig.for_window(1000)


def test_for_window_suggests_the_nearest_usable_window():
    """The message has to be actionable — the caller cannot derive 900 themselves."""
    with pytest.raises(ValueError, match="900"):
        PatchTSTConfig.for_window(1000)


def test_only_the_patch_embedding_changes_size_with_the_lookback():
    """Encoder, positional embedding and head are identical at every scale.

    If anything else moved, a difference between the three runs could be a
    difference in capacity rather than in how much history the model read.
    """
    shapes = {}
    for scale in ("short", "mid", "long"):
        model = PatchTSTClassifier(config_at(scale))
        shapes[scale] = {
            name: tuple(p.shape)
            for name, p in model.named_parameters()
            if not name.startswith("value_embedding.")
        }

    assert shapes["short"] == shapes["mid"] == shapes["long"]


def test_the_patch_embedding_is_the_only_thing_that_grows():
    """And it grows exactly linearly in the lookback, being `Linear(P, d_model)`."""
    widths = {
        scale: PatchTSTClassifier(config_at(scale)).value_embedding.in_features
        for scale in ("short", "mid", "long")
    }
    assert widths == {"short": 16, "mid": 384, "long": 4608}


@pytest.mark.parametrize("scale", ["short", "mid", "long"])
def test_a_forward_pass_runs_at_every_scale(scale):
    cfg = config_at(scale, d_model=16, n_heads=2, n_layers=1, d_ff=16)
    model = PatchTSTClassifier(cfg).eval()
    with torch.no_grad():
        out = model(torch.randn(2, cfg.n_channels, cfg.seq_len))

    assert out.shape == (2,)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("scale", ["short", "mid", "long"])
def test_a_longer_lookback_discards_nothing(scale):
    """Every bar of the window reaches the model, at every scale.

    The alternative way to make a 24-hour lookback affordable is to coarsen the
    grid — 288-second bars instead of 1-second ones — and that would make the long
    model blind to the intra-minute ordering the short model's whole edge rests on.
    This pins the claim that the scales are *nested*: the long model sees
    everything the short one sees plus a day of context, so a difference between
    them is attributable to the extra history alone.

    Patching an index ramp makes the check direct — every index must survive.
    """
    cfg = config_at(scale)
    model = PatchTSTClassifier(cfg)
    ramp = torch.arange(cfg.seq_len, dtype=torch.float32).reshape(1, 1, cfg.seq_len)

    seen = np.unique(model.patchify(ramp).numpy())

    np.testing.assert_array_equal(seen, np.arange(cfg.seq_len, dtype=np.float32))


# --- batch sizing -------------------------------------------------------------


def test_a_batch_that_fits_is_left_exactly_as_asked():
    """Only the long scale is clamped; the other two keep the study's batch size."""
    assert batch_size_for(config_at("short"), base=512) == 512
    assert batch_size_for(config_at("mid"), base=512) == 512


def test_the_long_lookback_forces_a_smaller_batch():
    """A 4,608-bar patch is 288x the short scale's, and the patched tensor with it."""
    assert batch_size_for(config_at("long"), base=512) < 512


def test_the_clamped_batch_stays_within_the_budget():
    from patchtst_model import PATCH_TENSOR_BUDGET

    cfg = config_at("long")
    batch = batch_size_for(cfg, base=512)

    assert batch * cfg.n_channels * cfg.num_patches * cfg.patch_len <= PATCH_TENSOR_BUDGET


def test_the_batch_never_falls_below_the_floor():
    """A pathological config must still produce a runnable batch, not zero."""
    cfg = PatchTSTConfig.for_window(86400, n_channels=64)

    assert batch_size_for(cfg, base=512, floor=8) >= 8


# --- what the lookback does to the window population --------------------------


@pytest.mark.parametrize("scale", ["short", "mid", "long"])
def test_the_window_count_falls_by_exactly_the_lookback(scale):
    """A window needs `window` bars behind it and `horizon` ahead of its last bar."""
    n_bars, horizon = 200_000, seq.HORIZON
    window = seq.window_for(scale)

    starts = seq.valid_window_starts(n_bars, window, horizon)

    assert starts.size == n_bars - window - horizon + 1
    assert starts[0] == 0
    assert starts[-1] + window - 1 + horizon == n_bars - 1


def test_a_lookback_longer_than_the_series_is_refused_by_name():
    """The 24-hour scale needs a day of bars before it can form one window.

    Loud rather than empty: a run that silently produced no windows would fail
    much later, inside a model fit, with an error about an empty array.
    """
    with pytest.raises(ValueError, match="90000 bars"):
        seq.valid_window_starts(1_000, seq.window_for("long"), seq.HORIZON)


def test_the_purge_grows_with_the_lookback_and_still_separates_the_folds():
    """`fold_rows` must keep training clear of the test month at any lookback.

    The purge is `window + horizon - 1`, so at the 24-hour scale it is a full day
    rather than five minutes. A purge that failed to scale would leave training
    labels decided by price moves inside the test month — which fails nothing, it
    just raises every number the run reports.
    """
    n_bars, horizon = 400_000, 3_600
    fold = wf.Fold(name="a -> b", train_lo=0, train_hi=250_000, test_lo=250_000, test_hi=400_000)

    for scale in ("short", "mid"):
        window = seq.window_for(scale)
        starts = seq.valid_window_starts(n_bars, window, horizon)
        touched = np.ones(starts.size, dtype=bool)
        purge = window + horizon - 1

        train_idx, val_idx, test_idx = pf.fold_rows(fold, starts, touched, purge)

        last_train_bar = starts[train_idx].max() + window - 1 + horizon
        assert last_train_bar < fold.test_lo, f"{scale}: a training label reaches the test month"
        assert starts[val_idx].min() >= starts[train_idx].max() + purge
        assert starts[test_idx].min() >= fold.test_lo


# --- confidence intervals ------------------------------------------------------


def test_the_bootstrap_block_defaults_to_the_short_lookbacks_reach():
    """Left at its default, the block describes the 5-minute window only.

    Consecutive windows are correlated for `window + horizon` rows, so the block
    has to follow the lookback. A 24-hour run scored at the default block would
    treat rows that overlap for 90,000 as independent after 3,899 — an interval
    several times too narrow, in the direction that makes a result look real.
    Every caller in `python/` passes its own; this pins the fallback.
    `test_integration.py` covers the widening itself.
    """
    rng = np.random.default_rng(3)
    y = (rng.random(20_000) < 0.5).astype(np.int8)
    score = rng.random(20_000)

    default = seq.block_bootstrap_auc(y, score, n_boot=60, seed=1)
    explicit = seq.block_bootstrap_auc(
        y, score, n_boot=60, block=seq.WINDOW_SCALES["short"] + seq.HORIZON, seed=1
    )

    assert default == explicit
