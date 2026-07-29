"""
Run the PatchTST experiment locally against weights trained in Colab.

    cd python
    python eval_patchtst.py --weights ../weights/patchtst.pt

The checkpoint carries the data recipe it was trained on (hours of tape, window,
horizon, fee threshold, split fractions), so this script rebuilds the *identical*
window population and evaluates on the same held-out slice the notebook reported.
It refuses to guess: if the local CSV does not reproduce the bar series the
checkpoint was built from, it says so instead of quietly scoring a different set.

Metrics land on stdout and are appended to `python/training_logs.txt` in the same
format `train_xgboost.py` and `train_baseline.py` use.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import sequence_matrix as seq
from data_feeder import FILE_PATH
from patchtst_model import (
    binary_metrics,
    describe_checkpoint,
    format_metrics,
    load_checkpoint,
    predict_proba,
    threshold_sweep,
    WindowBatcher,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "patchtst.pt"
LOG_PATH = Path(__file__).with_name("training_logs.txt")


def log_results(model_name: str, acc: float, f1: float, roc_auc: float, extra: str = "") -> None:
    """Same log format as the other baselines, so the file stays one timeline."""
    with open(LOG_PATH, "a") as handle:
        handle.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {model_name}\n")
        handle.write(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}\n")
        if extra:
            handle.write(f"{extra}\n")
        handle.write("-" * 50 + "\n")


def _check_consistency(data_meta: dict, dataset: seq.SequenceDataset, strict: bool) -> list[str]:
    """Compare the rebuilt bar series against the one the checkpoint used."""
    problems = []
    for key, label in (("n_bars", "bar count"), ("first_ts", "first timestamp")):
        expected, actual = data_meta.get(key), dataset.meta.get(key)
        if expected is not None and actual is not None and expected != actual:
            problems.append(f"{label}: checkpoint {expected:,} vs local {actual:,}")
    expected_windows = data_meta.get("split_sizes", {})
    for name, expected in expected_windows.items():
        actual = dataset.splits[name].size if name in dataset.splits else None
        if actual is not None and expected != actual:
            problems.append(f"{name} windows: checkpoint {expected:,} vs local {actual:,}")

    if problems:
        message = "Local data does not match the checkpoint:\n  - " + "\n  - ".join(problems)
        if strict:
            raise SystemExit(
                f"{message}\n\nUse the same CSV (and --bars-cache) the notebook trained on, "
                "or pass --allow-mismatch to score anyway."
            )
        print(f"⚠️  {message}\n   Scoring anyway (--allow-mismatch).")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="checkpoint exported by the notebook")
    parser.add_argument("--csv", default=FILE_PATH, help="raw Binance trades CSV")
    parser.add_argument("--bars-cache", default=None, help="reuse/write a .npz second-bar cache")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--threshold", type=float, default=0.5, help="probability cut-off for the hard label")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda")
    parser.add_argument("--hours", type=float, default=None, help="override the checkpoint's slice length")
    parser.add_argument("--no-sweep", action="store_true", help="skip the threshold sweep table")
    parser.add_argument("--no-log", action="store_true", help="do not append to training_logs.txt")
    parser.add_argument("--allow-mismatch", action="store_true", help="score even if the data differs")
    parser.add_argument("--save-predictions", default=None, help="write probabilities to this .npy")
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(
            f"No checkpoint at {weights_path}.\n"
            "Train one with notebooks/train_patchtst_colab.ipynb, download the .pt it "
            f"produces, and drop it in {DEFAULT_WEIGHTS.parent}/."
        )

    print(f"1. Loading checkpoint {weights_path} ...")
    model, payload = load_checkpoint(weights_path, map_location=args.device)
    print(describe_checkpoint(payload))

    data_meta = payload.get("data_meta", {})
    hours = args.hours if args.hours is not None else data_meta.get("hours")

    print(f"\n2. Rebuilding the bar series ({'full file' if hours is None else f'{hours}h'}) ...")
    dataset = seq.build_from_csv(
        args.csv,
        hours=hours,
        skip_hours=data_meta.get("skip_hours", 0.0),
        cache=args.bars_cache,
        **seq.dataset_kwargs_from(data_meta),
    )
    print(dataset.summary())
    _check_consistency(data_meta, dataset, strict=not args.allow_mismatch)

    channel_count = dataset.channels.shape[1]
    if channel_count != model.cfg.n_channels:
        raise SystemExit(
            f"Checkpoint expects {model.cfg.n_channels} channels, the local builder produced "
            f"{channel_count}. sequence_matrix.py has changed since training."
        )

    print(f"\n3. Scoring the {args.split} split on {args.device} ...")
    starts = dataset.splits[args.split]
    batcher = WindowBatcher(
        dataset.channels,
        starts,
        dataset.labels(args.split),
        seq_len=dataset.window,
        device=args.device,
    )
    model.to(args.device)
    probabilities = predict_proba(model, batcher, batch_size=args.batch_size)
    truth = batcher.numpy_labels()

    metrics = binary_metrics(truth, probabilities, threshold=args.threshold)
    ci = seq.block_bootstrap_auc(truth, probabilities)
    print("\n✅ PatchTST Evaluated Successfully")
    print("=========================================")
    print(format_metrics(metrics))
    print(f"           95% CI [{ci['lo']:.4f}, {ci['hi']:.4f}], "
          f"P(<=0.5) = {ci['p_le_half']:.3f}")
    print("=========================================")
    print(
        f"windows: {metrics['n_windows']:,} | positives: {metrics['n_positive']:,} | "
        f"flagged at p>={args.threshold:g}: {metrics['n_flagged']:,}"
    )

    sweep = []
    if not args.no_sweep:
        sweep = threshold_sweep(truth, probabilities)
        print(
            f"\n🔍 Threshold sweep (base rate {metrics['base_rate']:.2%}, so 0.5 is "
            "only the right cut if the classes are balanced)"
        )
        print(f"{'p>=':>6} {'flagged':>9} {'precision':>10} {'recall':>8} {'F1':>7}")
        for row in sweep:
            print(
                f"{row['threshold']:>6.2f} {row['n_flagged']:>9,} {row['precision']:>10.4f} "
                f"{row['recall']:>8.4f} {row['f1']:>7.4f}"
            )
        print(f"{'(base rate)':>17} {metrics['base_rate']:>10.4f}")

    if args.save_predictions:
        np.save(args.save_predictions, probabilities)
        print(f"\nProbabilities written to {args.save_predictions}")

    if not args.no_log:
        train_meta = payload.get("train_meta", {})
        extra = (
            f"Split: {args.split} | Label: {dataset.meta['label_mode']} at "
            f"+/-{dataset.meta['barrier']:.4%} over {dataset.horizon}s | "
            f"Base rate: {metrics['base_rate']:.2%} | "
            f"Always-0 accuracy: {metrics['majority_accuracy']:.4f}\n"
            f"ROC-AUC 95% CI: [{ci['lo']:.4f}, {ci['hi']:.4f}] | "
            f"P(AUC <= 0.5) = {ci['p_le_half']:.3f}\n"
            f"Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | "
            f"Flagged: {metrics['n_flagged']:,} / {metrics['n_windows']:,}\n"
            f"Checkpoint: {weights_path.name} ({payload.get('created_utc', 'unknown')}) | "
            f"params: {model.n_parameters():,} | best epoch: {train_meta.get('best_epoch', '?')}\n"
            f"Config: {json.dumps(payload.get('config', {}), sort_keys=True)}"
        )
        log_results("PatchTST (Sequence Matrix + Fee Threshold)", metrics["accuracy"], metrics["f1"], metrics["roc_auc"], extra)
        print(f"\nAppended to {LOG_PATH}")


if __name__ == "__main__":
    main()
