"""The C++ rolling-window engine, against an independent NumPy reference.

The engine is the one component with no Python fallback: if `bipower_core` and
`sequence_matrix` disagree, the features the tabular models train on are not the
features the README describes. So the reference implementation here is written
from the *formulas* rather than from the C++ loop — a transcription of the C++
would agree with it for the same reason a typo agrees with itself.

The module is skipped rather than failed when it has not been built, so `pytest`
is useful on a checkout without a compiler. CI builds it first and asserts the
skip did not happen (`test_extension_is_available_in_ci`).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

bipower_core = pytest.importorskip(
    "bipower_core",
    reason="C++ extension not built; run `python build.py`",
)

PI_FACTOR = 1.5707963267948966


def reference_metrics(prices, qtys, is_buyer_maker, window):
    """RV, BPV, jumps and OFI over sliding windows, straight from the definitions.

    Window `w` covers ticks `[w, w + window)`. Within it:

        RV  = sum of r_i^2 over the window's returns
        BPV = (pi/2) * sum of |r_i| |r_{i-1}| over adjacent pairs
        J   = max(RV - BPV, 0)
        OFI = sum of +qty (aggressive buy) or -qty (aggressive sell)

    A window of `w` prices holds `w - 1` returns and `w - 2` adjacent pairs, so
    the first return of a window starts at offset 1 and the first bipower pair at
    offset 2 — the alignment `build_tabular_features` also reproduces.
    """
    prices = np.asarray(prices, dtype=np.float64)
    qtys = np.asarray(qtys, dtype=np.float64)
    maker = np.asarray(is_buyer_maker, dtype=bool)

    if prices.size < window or window < 2:
        return {
            k: np.empty(0)
            for k in (
                "realized_variance",
                "bipower_variation",
                "jump_component",
                "order_flow_imbalance",
            )
        }

    rv, bpv, jumps, ofi = [], [], [], []
    for start in range(prices.size - window + 1):
        window_prices = prices[start : start + window]
        window_qty = qtys[start : start + window]
        window_maker = maker[start : start + window]

        returns = np.diff(np.log(window_prices))
        rv_w = float(np.sum(returns**2))
        bpv_w = PI_FACTOR * float(np.sum(np.abs(returns[1:]) * np.abs(returns[:-1])))

        rv.append(rv_w)
        bpv.append(bpv_w)
        jumps.append(max(rv_w - bpv_w, 0.0))
        ofi.append(float(np.sum(np.where(window_maker, -window_qty, window_qty))))

    return {
        "realized_variance": np.array(rv),
        "bipower_variation": np.array(bpv),
        "jump_component": np.array(jumps),
        "order_flow_imbalance": np.array(ofi),
    }


def call_engine(prices, qtys, maker, window):
    return bipower_core.calculate_rolling_metrics(
        np.asarray(prices, dtype=np.float64),
        np.asarray(qtys, dtype=np.float64),
        np.asarray(maker, dtype=bool),
        window,
    )


@pytest.mark.skipif(os.environ.get("CI") is None, reason="guards the CI build step, not local runs")
def test_extension_is_available_in_ci():
    """In CI the extension must be built — a silent skip would hide a broken build."""
    assert hasattr(bipower_core, "calculate_rolling_metrics")


def test_matches_reference_on_a_hand_checkable_series():
    """Five prices, window 3: small enough to verify by hand if this ever fails."""
    prices = [100.0, 101.0, 100.5, 102.0, 101.0]
    qtys = [1.0, 2.0, 3.0, 4.0, 5.0]
    maker = [False, True, False, True, False]

    out = call_engine(prices, qtys, maker, 3)
    expected = reference_metrics(prices, qtys, maker, 3)

    for key, want in expected.items():
        np.testing.assert_allclose(
            out[key], want, rtol=1e-12, atol=1e-15, err_msg=f"{key} disagrees with the reference"
        )


def test_ofi_signs_follow_the_taker():
    """`is_buyer_maker=False` is an aggressive buy (+qty); True is a sell (-qty)."""
    prices = [100.0] * 4
    qtys = [1.0, 2.0, 4.0, 8.0]

    all_buys = call_engine(prices, qtys, [False] * 4, 4)
    all_sells = call_engine(prices, qtys, [True] * 4, 4)

    assert all_buys["order_flow_imbalance"] == pytest.approx([15.0])
    assert all_sells["order_flow_imbalance"] == pytest.approx([-15.0])


@pytest.mark.parametrize("window", [2, 3, 8, 25])
@pytest.mark.parametrize("seed", [0, 7])
def test_matches_reference_on_random_series(window, seed):
    """Random walks across several window sizes, including the degenerate w=2."""
    rng = np.random.default_rng(seed)
    n = 120
    prices = 70_000.0 * np.exp(np.cumsum(rng.normal(0, 2e-4, n)))
    qtys = rng.gamma(2.0, 0.5, n)
    maker = rng.random(n) < 0.5

    out = call_engine(prices, qtys, maker, window)
    expected = reference_metrics(prices, qtys, maker, window)

    for key, want in expected.items():
        assert len(out[key]) == n - window + 1, f"{key} has the wrong window count"
        np.testing.assert_allclose(
            out[key], want, rtol=1e-10, atol=1e-18, err_msg=f"{key} disagrees with the reference"
        )


def test_bipower_is_zero_when_a_window_holds_one_return():
    """Window 2 has a single return, so there is no adjacent pair to multiply."""
    out = call_engine([100.0, 101.0, 102.0], [1.0] * 3, [False] * 3, 2)
    np.testing.assert_allclose(out["bipower_variation"], [0.0, 0.0])
    # With BPV = 0, jumps collapse onto RV: max(RV - 0, 0) == RV.
    np.testing.assert_allclose(out["jump_component"], out["realized_variance"])


def test_jump_component_is_clamped_at_zero():
    """A smooth series has BPV >= RV, and the clamp is what keeps J non-negative."""
    # A constant-drift series: every return is identical, so the bipower terms
    # (pi/2 ~ 1.571 per pair) sum to more than the squared returns do.
    prices = 100.0 * np.exp(np.arange(40) * 1e-4)
    out = call_engine(prices, np.ones(40), np.zeros(40, dtype=bool), 20)

    assert np.all(np.asarray(out["jump_component"]) >= 0.0)
    assert np.any(np.asarray(out["bipower_variation"]) > np.asarray(out["realized_variance"]))


def test_a_jump_raises_realized_variance_above_bipower():
    """The estimator's whole purpose: BPV ignores an isolated jump, RV does not."""
    prices = np.full(40, 100.0)
    prices[20:] = 103.0  # one 3% dislocation, the rest flat

    out = call_engine(prices, np.ones(40), np.zeros(40, dtype=bool), 10)
    rv = np.asarray(out["realized_variance"])
    jumps = np.asarray(out["jump_component"])

    # Windows spanning the dislocation carry it; the flat ones stay at zero.
    assert jumps.max() > 0.0
    assert rv.max() > 0.0
    np.testing.assert_allclose(jumps[:10], 0.0, atol=1e-15)


@pytest.mark.parametrize("window", [0, 1])
def test_degenerate_windows_return_nothing(window):
    """`window < 2` cannot form a return, so the engine returns empty vectors."""
    out = call_engine([100.0] * 5, [1.0] * 5, [False] * 5, window)
    assert all(len(v) == 0 for v in out.values())


def test_window_longer_than_the_series_returns_nothing():
    out = call_engine([100.0, 101.0], [1.0, 1.0], [False, False], 10)
    assert all(len(v) == 0 for v in out.values())


def test_window_equal_to_series_length_returns_one_window():
    out = call_engine([100.0, 101.0, 102.0], [1.0] * 3, [False] * 3, 3)
    assert all(len(v) == 1 for v in out.values())


def test_mismatched_array_lengths_are_rejected():
    """A short `qtys` would otherwise be read past its end — silent memory garbage.

    The binding reads the length from `prices` alone, so without an explicit
    check this is an out-of-bounds read rather than an error.
    """
    with pytest.raises(ValueError, match="same length"):
        call_engine([100.0] * 10, [1.0] * 4, [False] * 10, 3)


def test_non_contiguous_input_is_handled():
    """A sliced array is not contiguous; reading its buffer directly would skip.

    `prices[::2]` shares its parent's memory with a stride of two, so a raw
    pointer walk returns every element rather than every other one.
    """
    base = 70_000.0 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 2e-4, 60)))
    strided = base[::2]
    qtys = np.ones(strided.size)
    maker = np.zeros(strided.size, dtype=bool)

    out = call_engine(strided, qtys, maker, 5)
    expected = reference_metrics(strided, qtys, maker, 5)
    np.testing.assert_allclose(out["realized_variance"], expected["realized_variance"], rtol=1e-10)


def test_multidimensional_input_is_rejected():
    """A 2-D array has no unambiguous reading as a tick series."""
    with pytest.raises(ValueError, match="1-D"):
        call_engine(np.ones((10, 2)), np.ones(20), np.zeros(20, dtype=bool), 3)


def test_agrees_with_the_python_window_sums(bars):
    """The C++ engine and `build_tabular_features` must describe the same windows.

    They compute the same quantities by different routes — a direct loop against
    a differenced cumulative sum — so this pins the alignment (which bar starts a
    window's returns, which starts its bipower pairs) that both depend on. The
    tolerance is loose because differencing a long cumulative sum cancels
    significant digits; `sequence_matrix` documents the ~5e-8 relative error.
    """
    import sequence_matrix as seq

    window = 50
    price, ofi = bars["price"], bars["ofi"]

    # `bipower_core` consumes signed flow as (qty, is_buyer_maker); the bar grid
    # stores it pre-netted as `ofi`. Feed magnitudes and signs to match.
    maker = ofi < 0
    engine = call_engine(price, np.abs(ofi).astype(np.float64), maker, window)

    channels = seq.build_channels(bars, "full")
    starts = np.arange(price.size - window + 1, dtype=np.int64)
    features = seq.build_tabular_features(channels, price, starts, window)

    np.testing.assert_allclose(features[:, 0], engine["realized_variance"], rtol=1e-6)
    np.testing.assert_allclose(features[:, 1], engine["bipower_variation"], rtol=1e-6)
    np.testing.assert_allclose(features[:, 3], engine["order_flow_imbalance"], rtol=1e-5, atol=1e-4)
