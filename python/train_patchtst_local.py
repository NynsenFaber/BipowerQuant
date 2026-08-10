"""
Train PatchTST locally — tuned for Apple Silicon (M1/M2/M3), not for a datacentre GPU.

    cd python
    python train_patchtst_local.py                      # sensible M1 Pro defaults
    python train_patchtst_local.py --months 2026-04 2026-05 2026-06
    python train_patchtst_local.py --time-budget-hours 3

This is the same walk-forward loop the Colab notebook runs (`patchtst_folds.train_folds`);
only the defaults and the guardrails differ. Three things matter on a laptop that do
not matter on a T4:

**Unified memory.** On Apple Silicon the "GPU" allocates from the same pool as
everything else, so the resident channel matrix, the batch gathers and your
browser are competing. This script measures free RAM up front, estimates what the
run needs, and refuses to start a job that will not fit rather than dying an hour
in. `--months` is the lever: each month costs ~105 MB of bars plus ~62 MB of
channels.

**Throughput.** An M1 Pro runs this model at roughly 1,300-1,800 windows/s against
a T4's several thousand, so the stride the time budget picks will be larger. That
is fine — windows overlap by 299 of 300 bars, so a stride of 24 discards far less
information than it discards rows — but do not expect a laptop run and a GPU run
to produce the same numbers.

**Crash safety.** Every fold's probabilities are written as soon as that fold
finishes, so an interrupted run keeps what it already earned and `--resume` skips
those folds on the next attempt. The previous version of this loop lost two hours
of training to an OOM at the very end; that is what this exists to prevent.
"""

from __future__ import annotations

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch

import patchtst_folds as pf
import sequence_matrix as seq
import walkforward as wf
from patchtst_model import PatchTSTConfig
from patchtst_train import TrainConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MONTHS = ["2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]

# Measured on an M1 Pro at the default model size (2 channels, 6 layers, 209k
# parameters, batch 512). Only used for the pre-flight estimate; the real number
# comes from the throughput probe.
ASSUMED_WINDOWS_PER_S = 1500.0


def rss_gb() -> float:
    """Peak resident memory of this process, in GB (macOS reports bytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e9 if sys.platform == "darwin" else peak / 1e6


def free_ram_gb() -> float | None:
    """Physical memory not currently in use, best effort."""
    if sys.platform == "darwin":
        try:
            import subprocess

            out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
            page = 16384
            free = speculative = 0
            for line in out.splitlines():
                if line.startswith("Pages free:"):
                    free = int(line.split(":")[1].strip().rstrip("."))
                elif line.startswith("Pages speculative:"):
                    speculative = int(line.split(":")[1].strip().rstrip("."))
            return (free + speculative) * page / 1e9
        except Exception:
            return None
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except Exception:
        pass
    return None


def estimate_ram_gb(n_months: int, n_channels: int) -> float:
    """Rough working-set estimate for the training phase, in GB.

    Per month: bars are ~2.6 M bars x (8-byte price + 8-byte ts + 3 x 4-byte
    float32) = ~90 MB, channels are `n_channels x 4 bytes` a bar, and the triple
    barrier adds an int32 exit offset plus two int8/bool arrays. Torch itself and
    the device-resident copy of the channel matrix are the constant terms.
    """
    bars = n_months * 2.6e6 * 36 / 1e9
    channels = n_months * 2.6e6 * n_channels * 4 / 1e9
    barriers = n_months * 2.6e6 * 6 / 1e9
    return bars + 2 * channels + barriers + 1.2  # 1.2 GB: torch, python, slack


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=DEFAULT_MONTHS,
        help="months to use, as YYYY-MM (needs a bar cache for each)",
    )
    parser.add_argument("--data-dir", default=str(REPO_ROOT / "data"))
    parser.add_argument(
        "--csv-template",
        default="BTCUSDT-trades-{month}.csv",
        help="used to build a missing month's bar cache",
    )
    parser.add_argument("--out", default=str(REPO_ROOT / "data" / "patchtst_probs.npz"))
    parser.add_argument("--checkpoint-dir", default=str(REPO_ROOT / "weights"))
    parser.add_argument("--resume", action="store_true", help="skip folds already present in --out")

    parser.add_argument("--scheme", default="anchored", choices=wf.SCHEMES)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument(
        "--window-scale",
        default=None,
        choices=sorted(seq.WINDOW_SCALES),
        help="named lookback; overrides --window. "
        + ", ".join(f"{k}={v}s" for k, v in seq.WINDOW_SCALES.items()),
    )
    parser.add_argument("--window", type=int, default=seq.WINDOW_SIZE)
    parser.add_argument("--horizon", type=int, default=seq.HORIZON)
    parser.add_argument("--barrier", type=float, default=seq.BARRIER)
    parser.add_argument("--channels", default="raw", choices=sorted(seq.CHANNEL_SETS))

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--train-stride",
        type=int,
        default=None,
        help="fixed stride; omit to let the time budget choose",
    )
    parser.add_argument("--val-stride", type=int, default=8)
    parser.add_argument("--time-budget-hours", type=float, default=2.0)
    parser.add_argument("--device", default=None, help="mps, cpu, or cuda")
    parser.add_argument(
        "--threads", type=int, default=0, help="torch CPU threads; 0 leaves the default"
    )
    parser.add_argument("--yes", action="store_true", help="skip the memory confirmation")
    args = parser.parse_args()

    window = seq.window_for(args.window_scale) if args.window_scale else args.window

    data_dir = Path(args.data_dir)

    # -- pre-flight: will this fit? --------------------------------------------
    n_channels = len(seq.CHANNEL_SETS[args.channels])
    needed = estimate_ram_gb(len(args.months), n_channels)
    free = free_ram_gb()
    print(f"months            : {len(args.months)} ({args.months[0]} .. {args.months[-1]})")
    print(f"estimated working set: {needed:.1f} GB")
    print(f"free RAM          : {'unknown' if free is None else f'{free:.1f} GB'}")
    if free is not None and needed > free * 0.85:
        print(
            f"\n!! This is likely to run the machine out of memory.\n"
            f"   Use fewer months (--months {' '.join(args.months[-3:])} needs "
            f"~{estimate_ram_gb(3, n_channels):.1f} GB), close other applications, "
            f"or pass --yes to try anyway."
        )
        if not args.yes:
            raise SystemExit(1)

    if args.threads:
        torch.set_num_threads(args.threads)

    # -- bars, one month at a time --------------------------------------------
    paths = []
    for month in args.months:
        cache = data_dir / f"bars_{month}.npz"
        if not cache.exists():
            csv = data_dir / args.csv_template.format(month=month)
            if not csv.exists():
                raise SystemExit(
                    f"Neither {cache.name} nor {csv.name} exists in {data_dir}. "
                    "Download the month from the Binance archive first."
                )
            print(f"building {cache.name} from {csv.name} ...", end=" ", flush=True)
            started = time.perf_counter()
            month_bars = seq.load_second_bars(csv, hours=None)
            seq.save_bars(month_bars, cache)
            del month_bars
            gc.collect()
            print(f"{time.perf_counter() - started:.0f}s")
        paths.append(cache)

    bars = seq.load_bar_caches(paths)
    meta = bars["meta"]
    print(
        f"\n{meta['n_bars']:,} bars | "
        f"{(meta['last_ts'] - meta['first_ts']) / 86400:.0f} days | "
        f"peak RSS so far {rss_gb():.1f} GB"
    )

    folds = wf.build_folds(bars["ts"], args.scheme, args.train_months)
    print(f"{len(folds)} {args.scheme} fold(s):")
    for fold in folds:
        print(f"   {fold.name}")

    # -- resume ----------------------------------------------------------------
    out_path = Path(args.out)
    existing: dict[str, np.ndarray] = {}
    if args.resume and out_path.exists():
        with np.load(out_path, allow_pickle=False) as raw:
            existing = {k: raw[k] for k in raw.files}
        done = [f.name for f in folds if f.name in existing]
        folds = [f for f in folds if f.name not in existing]
        if done:
            print(f"\nresuming: {len(done)} fold(s) already done, {len(folds)} to go")
        if not folds:
            print("nothing left to train")
            return

    device = torch.device(args.device) if args.device else None
    cfg = PatchTSTConfig.for_window(window, n_channels=n_channels)
    train_cfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size)

    def on_fold(name: str, probs: np.ndarray) -> None:
        """Write after every fold, so an interruption never costs a trained model."""
        existing[name] = probs
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **existing)
        print(
            f"   saved {name} -> {out_path} ({len(existing)} fold(s), peak RSS {rss_gb():.1f} GB)"
        )

    started = time.perf_counter()
    _, run_meta = pf.train_folds(
        bars,
        folds,
        window=window,
        horizon=args.horizon,
        barrier=args.barrier,
        channel_set=args.channels,
        config=cfg,
        train_config=train_cfg,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        time_budget_hours=None if args.train_stride else args.time_budget_hours,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        on_fold_complete=on_fold,
    )

    Path(str(out_path.with_suffix("")) + ".json").write_text(
        json.dumps(run_meta, indent=2, default=float)
    )
    print(f"\ndone in {(time.perf_counter() - started) / 3600:.2f} h | peak RSS {rss_gb():.1f} GB")
    print(f"{'fold':<26} {'test ROC-AUC':>13}")
    for name, row in run_meta["folds"].items():
        print(f"{name:<26} {row['roc_auc']:>13.4f}")
    print(
        f"\nNow fold these into the backtest:\n"
        f"  python walkforward.py --bars {' '.join(str(p) for p in paths)} \\\n"
        f"                        --patchtst-probs {out_path}"
    )


if __name__ == "__main__":
    main()
