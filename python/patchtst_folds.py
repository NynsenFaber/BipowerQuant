"""
Train one PatchTST side model per walk-forward fold.

The notebook is a thin driver over this: it downloads the tape and mounts Drive,
then calls `train_folds`. Keeping the loop here rather than in a cell means the
code that produced a set of probabilities is version controlled next to the
model, and that a local run and a Colab run are the same computation.

    # locally, on whatever device is available
    python patchtst_folds.py --bars ../data/bars_6m.npz \\
        --out ../data/patchtst_probs.npz --train-stride 20

The output is one probability array per fold, keyed by the fold name that
`walkforward.build_folds` generates, which is what `walkforward.py
--patchtst-probs` expects.

**Every fold scores all of its test windows**, including the ones where no
barrier is touched. Those are not the side model's problem — deciding whether a
window is worth trading is the gate's job — but the backtest needs a side for
every window the gate might let through, so the probability vector has to cover
the whole test month.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

import sequence_matrix as seq
import walkforward as wf
from patchtst_model import (
    PatchTSTClassifier,
    PatchTSTConfig,
    WindowBatcher,
    binary_metrics,
    predict_proba,
    save_checkpoint,
)
from patchtst_train import TrainConfig, estimate_throughput, fit, pick_device

# The validation pass each epoch, as a multiplier on training cost. Measured at
# roughly 1.2-1.3x across runs; used only to project wall time up front.
VAL_OVERHEAD = 1.25


def fold_rows(
    fold: wf.Fold,
    starts: np.ndarray,
    touched: np.ndarray,
    purge: int,
    val_frac: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(train, val, test)` positions into `starts` for one fold.

    Training and validation keep only barrier-touching windows, because a window
    with no side has nothing to teach a side model. Validation is the **last**
    slice of the training months rather than a random sample: early stopping on
    a random split would select against a model that generalises forward in time,
    which is the only direction that matters. A further `purge` rows separate it
    from the training rows so no training label is drawn from a price move inside
    a validation lookback.
    """
    in_train = (starts >= fold.train_lo) & (starts < fold.train_hi - purge)
    in_test = (starts >= fold.test_lo) & (starts < fold.test_hi)

    train_idx = np.flatnonzero(in_train & touched)
    cut = int(train_idx.size * (1.0 - val_frac))
    val_idx = train_idx[cut:]
    train_idx = train_idx[: max(cut - purge, 1)]
    return train_idx, val_idx, np.flatnonzero(in_test)


def stride_for_budget(
    total_rows: int,
    epochs: int,
    rate: float,
    budget_hours: float,
    minimum: int = 2,
    maximum: int = 64,
) -> tuple[int, float]:
    """Smallest training stride whose projected wall time fits the budget.

    Windows overlap by `window - 1` bars, so striding discards far less
    information than it discards rows — which is what makes this a reasonable
    knob to turn automatically rather than a compromise to agonise over.
    Returns `(stride, projected_hours)`; the projection ignores early stopping,
    so it is an upper bound.
    """
    stride = minimum
    while stride < maximum:
        hours = total_rows / stride * epochs * VAL_OVERHEAD / rate / 3600.0
        if hours <= budget_hours:
            return stride, hours
        stride += 1
    return stride, total_rows / stride * epochs * VAL_OVERHEAD / rate / 3600.0


def train_folds(
    bars: dict,
    folds: list[wf.Fold],
    window: int = seq.WINDOW_SIZE,
    horizon: int = seq.HORIZON,
    barrier: float = seq.BARRIER,
    channel_set: str = "raw",
    config: PatchTSTConfig | None = None,
    train_config: TrainConfig | None = None,
    train_stride: int | None = None,
    val_stride: int = 8,
    val_frac: float = 0.10,
    time_budget_hours: float | None = None,
    device: torch.device | str | None = None,
    checkpoint_dir: str | Path | None = None,
    on_fold_complete: Callable[[str, np.ndarray], None] | None = None,
    verbose: bool = True,
) -> tuple[dict[str, np.ndarray], dict]:
    """Fit one model per fold; return per-fold test probabilities and metadata.

    `train_stride=None` with `time_budget_hours` set picks the stride from a
    measured throughput probe. Pass `train_stride` explicitly to override.

    `on_fold_complete`, if given, is called with `(fold.name, probs)` right
    after each fold finishes, so a caller can persist progress incrementally
    instead of losing the whole run to a crash near the end.
    """
    device = torch.device(device) if device is not None else pick_device()
    channels = seq.build_channels(bars, channel_set)
    starts = seq.valid_window_starts(bars["price"].size, window, horizon)
    side, _, defined = seq.triple_barrier(bars["price"], horizon, barrier)

    label_bar = starts + window - 1
    y_side = (side[label_bar] > 0).astype(np.int8)
    touched = (side[label_bar] != 0) & defined[label_bar]
    purge = window + horizon - 1

    cfg = config or PatchTSTConfig(n_channels=channels.shape[1], seq_len=window)
    train_cfg = train_config or TrainConfig()

    # One upload, shared by every fold and split. At two channels and six months
    # this is ~125 MB; materialising windows instead would be tens of GB.
    matrix = torch.as_tensor(channels, dtype=torch.float32).t().contiguous().to(device)

    rows = {f.name: fold_rows(f, starts, touched, purge, val_frac) for f in folds}

    def view(idx: np.ndarray, stride: int = 1) -> WindowBatcher:
        sel = idx[::stride]
        return WindowBatcher(None, starts[sel], y_side[sel], window, device, shared_channels=matrix)

    if train_stride is None:
        if time_budget_hours is None:
            train_stride = 1
            projected = float("nan")
        else:
            probe = PatchTSTClassifier(cfg).to(device)
            rate = estimate_throughput(
                probe, view(rows[folds[0].name][0]), train_cfg, device=device
            )
            del probe
            total = sum(rows[f.name][0].size for f in folds)
            train_stride, projected = stride_for_budget(
                total, train_cfg.epochs, rate, time_budget_hours
            )
            if verbose:
                print(
                    f"measured {rate:,.0f} windows/s | train stride {train_stride} "
                    f"-> projected upper bound {projected:.1f} h "
                    f"(budget {time_budget_hours:.1f} h)"
                )
                if projected > time_budget_hours:
                    print(
                        "!! does not fit even at the maximum stride — lower epochs "
                        "or shorten the training window"
                    )
    else:
        projected = float("nan")

    probabilities: dict[str, np.ndarray] = {}
    summary: dict[str, dict] = {}
    started = time.perf_counter()

    for fold in folds:
        train_idx, val_idx, test_idx = rows[fold.name]
        if verbose:
            print(f"\n{'=' * 74}\n{fold.name}\n{'=' * 74}")
            print(
                f"train {train_idx.size // train_stride:,} (stride {train_stride}) | "
                f"val {val_idx.size // val_stride:,} | test {test_idx.size:,}"
            )

        model = PatchTSTClassifier(cfg).to(device)
        train_meta = fit(
            model,
            view(train_idx, train_stride),
            view(val_idx, val_stride),
            train_cfg,
            device=device,
            verbose=verbose,
        )

        probs = predict_proba(model, view(test_idx), batch_size=4096)
        probabilities[fold.name] = probs.astype(np.float32)

        # AUC is only defined where a side exists; the full vector still ships.
        resolved = touched[test_idx]
        metrics = binary_metrics(y_side[test_idx][resolved], probs[resolved])
        summary[fold.name] = {
            "roc_auc": metrics["roc_auc"],
            "n_test": int(test_idx.size),
            "n_resolved": int(resolved.sum()),
            "best_epoch": train_meta["best_epoch"],
            "best_val_metric": train_meta["best_val_metric"],
            "epochs_run": train_meta["epochs_run"],
        }
        if verbose:
            print(
                f"test ROC-AUC {metrics['roc_auc']:.4f} on {resolved.sum():,} resolved "
                f"of {test_idx.size:,} windows"
            )

        if checkpoint_dir:
            slug = fold.name.replace(" ", "").replace("..", "_").replace("->", "TO")
            save_checkpoint(
                Path(checkpoint_dir) / f"patchtst_{slug}.pt",
                model,
                data_meta={
                    "source": bars["meta"].get("source", "?"),
                    "n_bars": int(bars["meta"]["n_bars"]),
                    "first_ts": int(bars["meta"]["first_ts"]),
                    "window": window,
                    "horizon": horizon,
                    "barrier": barrier,
                    "label_mode": "triple_barrier",
                    "channels": list(seq.CHANNEL_SETS[channel_set]),
                    "fold": fold.as_dict(),
                    "train_stride": train_stride,
                },
                train_meta=train_meta,
                metrics=metrics,
            )

        if on_fold_complete:
            on_fold_complete(fold.name, probabilities[fold.name])

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = (time.perf_counter() - started) / 3600.0
    if verbose:
        print(f"\nall folds done in {elapsed:.2f} h")
    return probabilities, {
        "folds": summary,
        "train_stride": train_stride,
        "projected_hours": projected,
        "elapsed_hours": elapsed,
        "config": cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "device": str(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--bars", nargs="+", required=True)
    parser.add_argument("--out", required=True, help=".npz of per-fold probabilities")
    parser.add_argument("--scheme", default="anchored", choices=wf.SCHEMES)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--window", type=int, default=seq.WINDOW_SIZE)
    parser.add_argument("--horizon", type=int, default=seq.HORIZON)
    parser.add_argument("--barrier", type=float, default=seq.BARRIER)
    parser.add_argument("--channels", default="raw", choices=sorted(seq.CHANNEL_SETS))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--train-stride", type=int, default=None)
    parser.add_argument("--val-stride", type=int, default=8)
    parser.add_argument("--time-budget-hours", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    bars = wf.load_bars(args.bars)
    folds = wf.build_folds(bars["ts"], args.scheme, args.train_months)
    print(f"{bars['meta']['n_bars']:,} bars | {len(folds)} {args.scheme} fold(s)")

    channels = seq.CHANNEL_SETS[args.channels]
    probabilities, meta = train_folds(
        bars,
        folds,
        window=args.window,
        horizon=args.horizon,
        barrier=args.barrier,
        channel_set=args.channels,
        config=PatchTSTConfig(n_channels=len(channels), seq_len=args.window),
        train_config=TrainConfig(epochs=args.epochs, batch_size=args.batch_size),
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        time_budget_hours=args.time_budget_hours,
        device=args.device,
        checkpoint_dir=args.checkpoint_dir,
    )

    np.savez_compressed(args.out, **probabilities)
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"\nwrote {args.out}")
    print(f"{'fold':<26} {'test ROC-AUC':>13}")
    for name, row in meta["folds"].items():
        print(f"{name:<26} {row['roc_auc']:>13.4f}")


if __name__ == "__main__":
    main()
