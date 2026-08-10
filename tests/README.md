# Tests

```bash
uv sync --group test
uv run pytest                      # everything, ~4 s
uv run pytest -m "not slow"        # skip the tests that fit models
uv run pytest --cov                # with coverage
uv run pytest tests/test_backtest.py -k oracle -v
```

Everything is synthetic and seeded. The real tape is 52 GB and lives outside the
repository, so a test that needed it could not run in CI — and one that *did* run
against it would be checking a market rather than a function.

## Layout

| File | Covers | Notes |
| :--- | :--- | :--- |
| `conftest.py` | fixtures | also fixes the macOS OpenMP import order for the whole suite |
| `test_math_engine.py` | `bipower_core` (C++) | against a NumPy reference written from the formulas |
| `test_sequence_matrix.py` | labels, channels, splits, splicing | the triple barrier gets the most attention |
| `test_ingestion.py` | CSV → 1-second bars | both aggregation engines, checked against each other |
| `test_backtest.py` | costs, queue fills, `simulate` | non-overlap and the oracle bound |
| `test_walkforward.py` | folds and pooling | calendar edges a six-month sample never hits |
| `test_patchtst.py` | model, batching, checkpoints | tiny configs; wiring, not results |
| `test_pipeline.py` | `ml_matrix`, fold rows, stride budget | where the purge actually happens |
| `test_window_scales.py` | the short/mid/long lookbacks | the equal-capacity property the comparison rests on |
| `test_integration.py` | `train_folds`, `run_fold` end to end | all `slow`; the only tests of how the pieces connect |

## Two things worth knowing before adding a test

**The bar fixture is calibrated, not arbitrary.** `make_bars` defaults to a
per-bar volatility that puts ~34% of 60-second windows through the 5 bp barrier,
against 39.7% in the real six months. Both degenerate regimes hide bugs: at high
volatility every window resolves, so the vertical barrier is never exercised and
overshoot swamps the barrier width; at low volatility almost nothing resolves and
the side label is nearly empty. If you change `volatility`, check what it does to
the touch rate first.

**Import order in `conftest.py` is load-bearing.** On macOS, torch and xgboost
ship separate OpenMP runtimes; xgboost must be imported first *and* torch pinned
to one thread, or the interpreter segfaults partway through the walk-forward
tests. `benchmark_inference.py` documents the measured failure matrix. Ruff is
configured not to sort that block — see `per-file-ignores` in `pyproject.toml`.

## Why these properties

Most of the suite checks arithmetic. A few tests defend claims the README makes,
and those are the ones to keep working if the code moves under them:

* **The barrier is symmetric under price inversion.** An asymmetry between the
  upper and lower comparison would manufacture a directional edge out of nothing
  — the exact failure this project exists to rule out.
* **Positions never overlap.** A backtest that opens one position per flagged
  window counts the same price move hundreds of times and reports a Sharpe
  inflated by roughly √overlap, in the flattering direction.
* **Training ends before testing begins**, asserted on every fold and every
  scheme. A leak here fails nothing; it just raises every AUC in §3.
* **An oracle still loses money.** 5 bp of target against a 6 bp round trip is
  the project's central finding, so it is asserted rather than argued.
* **`train_folds` accepts `on_fold_complete`.** A regression test: the callback
  was documented and passed by `train_patchtst_local.py` but not accepted by
  `train_folds`, which passed every unit test and still killed the run at the
  first fold boundary. Integration tests exist for that class of failure.
* **The three lookbacks are the same model.** Every scale patches to 37 tokens
  and every parameter except the patch embedding has an identical shape, so a
  difference between the runs is attributable to how much history each read. A
  24-hour model patched at the 5-minute geometry would be a 10,800-token
  Transformer — it would train, it would score, and the comparison would be
  meaningless without anything failing.
* **A longer lookback discards nothing.** Patching an index ramp and checking
  every index survives. The cheap way to afford a 24-hour window is a coarser bar
  grid, which would make the long model blind to exactly the intra-minute ordering
  the short model's edge comes from — and would leave "more history did not help"
  unable to be told apart from "less resolution hurt".
