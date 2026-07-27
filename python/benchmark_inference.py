"""
Inference latency benchmark: Logistic Regression vs XGBoost vs PatchTST.

All three models are fitted and timed on **exactly the same windows** — the same
1-second grid, the same splits — so neither the latency nor the accuracy column is
comparing different populations.

Two numbers matter and they answer different questions:

* **batch = 1 latency** is the trading number. It is what a signal costs when a
  window closes and you have to decide before the next tick.
* **batched throughput** is the research number: how long a backtest or a sweep
  over a month of windows takes.

They rank the models differently, which is the point of reporting both.

What is excluded: the per-bar channel construction (log returns, signed volume)
that both model families need upstream and that a live system would maintain
incrementally. What is *included* for the tabular models is the extra rolling
aggregation that turns 300 bars into 7 scalars, since PatchTST does not pay it —
it reads the window directly and normalises inside the forward pass.

**Every CPU model is pinned to one thread** (`threads=1`). Two reasons:

1. It is the honest comparison. "XGBoost on 6 cores" against "PatchTST on 6
   cores" against a NumPy dot product that never threads at this size measures
   the machine as much as the model. One thread each measures the models.
2. On macOS, torch and xgboost ship separate OpenMP runtimes and the process
   deadlocks the first time torch runs a threaded op after xgboost has run.
   `torch.set_num_threads(1)` is the only mitigation that actually holds —
   capping xgboost's threads does not.

Pass `threads=0` to use every core instead; on macOS that will hang, so it is
only safe on Linux (Colab included).
"""

from __future__ import annotations

import sys as _sys

# --- macOS OpenMP: import order is load-bearing ------------------------------
# torch and xgboost each ship their own libomp on macOS. Measured on this repo:
#
#   torch imported first  + torch.set_num_threads(1)  -> SIGSEGV
#   torch imported first  (no pinning)                -> SIGSEGV
#   xgboost imported first + torch.set_num_threads(1) -> OK
#   xgboost imported first (no pinning)               -> deadlock in torch
#
# So both are needed: xgboost must load first, and torch must stay single
# threaded. Hence this import leads the file, ahead of everything that pulls in
# torch. If the caller already imported torch we cannot fix it after the fact,
# so flag it instead of letting the process die halfway through a benchmark.
_TORCH_IMPORTED_FIRST = _sys.platform == "darwin" and "torch" in _sys.modules

import xgboost  # noqa: E402  — must precede torch; see above

import time  # noqa: E402
from typing import Callable  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import sequence_matrix as seq  # noqa: E402
from patchtst_model import PatchTSTClassifier  # noqa: E402


def _assert_openmp_safe() -> None:
    if _TORCH_IMPORTED_FIRST:
        raise RuntimeError(
            "On macOS, torch was imported before benchmark_inference, which makes "
            "xgboost crash the process. Import benchmark_inference first, run it in "
            "a fresh interpreter (`python benchmark_inference.py ...`), or set "
            "OMP_NUM_THREADS=1 in the environment before starting Python. "
            "Linux (including Colab) is unaffected."
        )

MILLISECOND = 1e3
MICROSECOND = 1e6


# --- timing core -------------------------------------------------------------


def timed(
    fn: Callable[[], object],
    warmup: int = 10,
    repeats: int = 100,
    sync: Callable[[], None] | None = None,
) -> dict:
    """Median / p99 / mean wall time of `fn`, in milliseconds.

    `sync` is called after each repetition so asynchronous device work is charged
    to the repetition that queued it rather than to whichever one happens to
    observe it.
    """
    for _ in range(warmup):
        fn()
    if sync:
        sync()

    samples = np.empty(repeats)
    for i in range(repeats):
        started = time.perf_counter()
        fn()
        if sync:
            sync()
        samples[i] = (time.perf_counter() - started) * MILLISECOND

    return {
        "median_ms": float(np.median(samples)),
        "p99_ms": float(np.percentile(samples, 99)),
        "mean_ms": float(samples.mean()),
        "min_ms": float(samples.min()),
    }


def _sync_for(device: torch.device) -> Callable[[], None] | None:
    if device.type == "cuda":
        return torch.cuda.synchronize
    if device.type == "mps":
        return torch.mps.synchronize
    return None


# --- the tabular baselines ---------------------------------------------------


def tabular_features_one_window(
    channels: np.ndarray, prices: np.ndarray, start: int, window: int
) -> tuple:
    """The 7 features for a *single* window — what a live system pays per tick.

    `seq.build_tabular_features` is vectorised across every window at once, so
    timing it on one window would charge that window for a cumulative sum over
    the entire series. This is the honest per-window cost: three sums over the
    300 bars in the lookback. Offsets mirror the vectorised version exactly.
    """
    realized_variance = float(channels[start + 1 : start + window, 1].sum())
    bipower = float(channels[start + 2 : start + window, 2].sum())
    order_flow = float(channels[start : start + window, 3].sum())
    jumps = max(realized_variance - bipower, 0.0)
    return_5m = float(np.log(prices[start + window - 1]) - np.log(prices[start]))
    return (
        realized_variance,
        bipower,
        jumps,
        order_flow,
        return_5m,
        order_flow / (np.sqrt(bipower) + seq.EPSILON),
        jumps * np.sign(return_5m),
    )


def fit_tabular_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    seed: int = 42,
    threads: int = 1,
):
    """Fit the two baselines exactly as `train_baseline.py` / `train_xgboost.py` do."""
    from sklearn.linear_model import LogisticRegression

    # train_baseline.py uses the OFI column alone (index 3 of the 7-feature matrix).
    logistic = LogisticRegression(class_weight="balanced", max_iter=1000)
    logistic.fit(X_train[:, [3]], y_train)

    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if positives == 0:
        raise ValueError("No positive windows in the training split.")

    booster = xgboost.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=negatives / positives,
        eval_metric="logloss",
        random_state=seed,
        nthread=threads,
    )
    booster.fit(X_train, y_train)
    return logistic, booster


# --- the benchmark -----------------------------------------------------------


def run_benchmark(
    dataset: seq.SequenceDataset,
    model: PatchTSTClassifier,
    prices: np.ndarray,
    device: torch.device | str = "cpu",
    batch_size: int = 1024,
    single_repeats: int = 200,
    batch_repeats: int = 20,
    also_cpu: bool = True,
    threads: int = 1,
    fit_kwargs: dict | None = None,
) -> dict:
    """Fit the tabular baselines, then time all three models on the test split.

    `prices` is `bars["price"]` — the raw level series, which the channel matrix
    does not carry (it stores log *returns*) but the tabular features need.

    `threads` pins every CPU model to the same core count; see the module
    docstring for why the default is 1 and why raising it hangs on macOS.
    """
    _assert_openmp_safe()
    device = torch.device(device)
    window = dataset.window
    rows: list[dict] = []

    # Set once, not restored: torch's thread count is global, and putting it back
    # would re-arm the macOS deadlock for anything the caller runs afterwards.
    if threads:
        torch.set_num_threads(threads)

    # -- features, on precisely the windows PatchTST is scored on --
    train_starts, test_starts = dataset.splits["train"], dataset.splits["test"]

    probe_start = int(test_starts[0])
    prep = timed(
        lambda: tabular_features_one_window(dataset.channels, prices, probe_start, window),
        warmup=20,
        repeats=single_repeats,
    )

    X_train = seq.build_tabular_features(dataset.channels, prices, train_starts, window)
    bulk_started = time.perf_counter()
    X_test = seq.build_tabular_features(dataset.channels, prices, test_starts, window)
    prep["amortised_ms"] = (time.perf_counter() - bulk_started) * MILLISECOND / len(test_starts)
    y_train = dataset.labels("train")
    y_test = dataset.labels("test")

    logistic, booster = fit_tabular_baselines(
        X_train, y_train, threads=threads, **(fit_kwargs or {})
    )
    raw_booster = booster.get_booster()

    # -- accuracy, as a by-product of having to fit them anyway --
    from sklearn.metrics import roc_auc_score

    logistic_prob = logistic.predict_proba(X_test[:, [3]])[:, 1]
    xgb_prob = booster.predict_proba(X_test)[:, 1]

    # -- latency: one window --
    one_tabular = np.ascontiguousarray(X_test[:1])
    one_ofi = np.ascontiguousarray(X_test[:1, [3]])
    coefficients = logistic.coef_.ravel().astype(np.float64)
    intercept = float(logistic.intercept_[0])

    rows.append(
        {
            "model": "Logistic Regression (NumPy)",
            "detail": "1 feature, 2 params",
            **timed(
                lambda: 1.0 / (1.0 + np.exp(-(one_ofi @ coefficients + intercept))),
                repeats=single_repeats,
            ),
            "batch": 1,
        }
    )
    rows.append(
        {
            "model": "Logistic Regression (sklearn)",
            "detail": "predict_proba",
            **timed(lambda: logistic.predict_proba(one_ofi), repeats=single_repeats),
            "batch": 1,
        }
    )
    rows.append(
        {
            "model": "XGBoost (inplace_predict)",
            "detail": f"{booster.n_estimators} trees, depth {booster.max_depth}",
            **timed(lambda: raw_booster.inplace_predict(one_tabular), repeats=single_repeats),
            "batch": 1,
        }
    )
    rows.append(
        {
            "model": "XGBoost (sklearn)",
            "detail": "predict_proba",
            **timed(lambda: booster.predict_proba(one_tabular), repeats=single_repeats),
            "batch": 1,
        }
    )

    # -- latency: PatchTST --
    # One batcher per device, reused: rebuilding it would re-upload the whole
    # channel matrix (64 MB for a month) on every timed configuration.
    batchers: dict[str, "WindowBatcher"] = {}

    def patchtst_row(target: torch.device, count: int, repeats: int) -> dict:
        from patchtst_model import WindowBatcher

        model.to(target).eval()
        key = target.type
        if key not in batchers:
            batchers[key] = WindowBatcher(
                dataset.channels, test_starts, y_test, window, target
            )
        x, _ = batchers[key].gather(torch.arange(count, device=target))
        sync = _sync_for(target)

        def forward():
            with torch.inference_mode():
                return model(x)

        return {
            "model": f"PatchTST ({target.type})",
            "detail": f"{model.n_parameters():,} params, {model.cfg.num_patches} patches",
            **timed(forward, warmup=min(10, max(3, repeats // 4)), repeats=repeats, sync=sync),
            "batch": count,
        }

    rows.append(patchtst_row(device, 1, single_repeats))
    if also_cpu and device.type != "cpu":
        rows.append(patchtst_row(torch.device("cpu"), 1, max(single_repeats // 4, 20)))

    # -- throughput: a full batch --
    batch_tabular = np.ascontiguousarray(X_test[:batch_size])
    batch_ofi = np.ascontiguousarray(X_test[:batch_size, [3]])
    batch_rows = [
        {
            "model": "Logistic Regression (NumPy)",
            **timed(
                lambda: 1.0 / (1.0 + np.exp(-(batch_ofi @ coefficients + intercept))),
                repeats=batch_repeats,
            ),
        },
        {
            "model": "XGBoost (inplace_predict)",
            **timed(lambda: raw_booster.inplace_predict(batch_tabular), repeats=batch_repeats),
        },
        patchtst_row(device, batch_size, batch_repeats),
    ]
    if also_cpu and device.type != "cpu":
        batch_rows.append(patchtst_row(torch.device("cpu"), batch_size, max(batch_repeats // 2, 5)))
    for row in batch_rows:
        row["batch"] = batch_size
        row["per_window_us"] = row["median_ms"] * MILLISECOND / batch_size
        row["windows_per_s"] = batch_size / (row["median_ms"] / MILLISECOND)

    model.to(device)
    return {
        "single": rows,
        "batched": batch_rows,
        "feature_prep": prep,
        "batch_size": batch_size,
        "device": str(device),
        "threads": torch.get_num_threads() if threads else 0,
        "accuracy": {
            "Logistic Regression": float(roc_auc_score(y_test, logistic_prob)),
            "XGBoost": float(roc_auc_score(y_test, xgb_prob)),
        },
        "test_windows": int(len(test_starts)),
    }


def format_benchmark(results: dict) -> str:
    """Render the benchmark as two tables plus the context needed to read them."""
    lines = []
    threads = results.get("threads", 1)
    lines.append(
        f"CPU models pinned to {threads} thread{'s' if threads != 1 else ''}"
        f"  |  test split: {results['test_windows']:,} windows"
    )
    lines.append("")
    lines.append("Single-window latency  (the trading number)")
    lines.append(f"{'model':<32} {'detail':<28} {'median':>9} {'p99':>9}")
    lines.append("-" * 82)
    for row in results["single"]:
        lines.append(
            f"{row['model']:<32} {row.get('detail', ''):<28} "
            f"{row['median_ms']:>8.3f}ms {row['p99_ms']:>8.3f}ms"
        )

    batch_size = results["batch_size"]
    lines.append("")
    lines.append(f"Batched throughput  (batch = {batch_size:,}, the backtest number)")
    lines.append(f"{'model':<32} {'batch total':>13} {'per window':>13} {'windows/s':>14}")
    lines.append("-" * 82)
    for row in results["batched"]:
        lines.append(
            f"{row['model']:<32} {row['median_ms']:>11.3f}ms "
            f"{row['per_window_us']:>11.2f}us {row['windows_per_s']:>14,.0f}"
        )

    prep = results["feature_prep"]
    lines.append("")
    lines.append("Feature preparation  (300 bars -> 7 scalars; only the tabular models pay it)")
    lines.append(
        f"  one window, on demand : {prep['median_ms']:.4f}ms  "
        f"-> add this to the Logistic Regression and XGBoost latencies above"
    )
    lines.append(
        f"  vectorised, amortised : {prep['amortised_ms']:.4f}ms per window across the split"
    )
    lines.append("  PatchTST pays neither: it reads the raw window and normalises in the forward pass.")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from data_feeder import FILE_PATH
    from patchtst_model import load_checkpoint

    parser = argparse.ArgumentParser(description="Time all three models on the same windows.")
    parser.add_argument("--weights", default=str(Path(__file__).resolve().parent.parent / "weights" / "patchtst.pt"))
    parser.add_argument("--csv", default=FILE_PATH)
    parser.add_argument("--bars-cache", default=None)
    parser.add_argument("--hours", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    model, payload = load_checkpoint(args.weights, map_location=args.device)
    data_meta = payload.get("data_meta", {})
    hours = args.hours if args.hours is not None else data_meta.get("hours")

    bars = (
        seq.load_bars(args.bars_cache)
        if args.bars_cache and Path(args.bars_cache).exists()
        else seq.load_second_bars(args.csv, hours=hours)
    )
    dataset = seq.build_sequence_dataset(
        bars,
        window=data_meta.get("window", seq.WINDOW_SIZE),
        horizon=data_meta.get("horizon", seq.HORIZON),
        fee_threshold=data_meta.get("fee_threshold", seq.FEE_THRESHOLD),
        train_frac=data_meta.get("train_frac", 0.7),
        val_frac=data_meta.get("val_frac", 0.1),
        train_stride=data_meta.get("train_stride", 1),
        val_stride=data_meta.get("val_stride", 1),
    )
    print(dataset.summary())
    print("\nFitting the tabular baselines on the same windows ...\n")
    results = run_benchmark(
        dataset, model, bars["price"], device=args.device, batch_size=args.batch_size
    )
    print(format_benchmark(results))
    print(f"\nTest ROC-AUC on {results['test_windows']:,} identical windows:")
    for name, auc in results["accuracy"].items():
        print(f"  {name:<22} {auc:.4f}")
