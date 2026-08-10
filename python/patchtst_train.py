"""
Training loop for the PatchTST classifier.

Kept out of the notebook on purpose: the notebook is a thin driver that sets
knobs and calls `fit`, so the logic that produced a checkpoint is version
controlled next to the model rather than pasted into a cell.

Nothing here is Colab-specific — the same `fit` runs on CPU, it is just slow.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from patchtst_model import PatchTSTClassifier, WindowBatcher, binary_metrics


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 512
    eval_batch_size: int = 4096
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    patience: int = 5  # epochs without improvement before stopping
    early_stop_metric: str = "roc_auc"  # "roc_auc" or "loss"
    amp: bool = True
    seed: int = 42
    pos_weight: float | None = None  # None -> n_neg / n_pos of the training split
    max_pos_weight: float | None = None  # optional cap on that ratio

    def to_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _amp_setup(device: torch.device, enabled: bool):
    """bf16 only where the hardware runs it natively, fp16 + loss scaling otherwise.

    `torch.cuda.is_bf16_supported()` is not the right test on its own: recent
    PyTorch counts *emulated* bf16, so it returns True on a Turing T4 (sm_75) where
    bf16 is slower than fp16. Native bf16 starts at Ampere (sm_80).
    """
    if not enabled or device.type != "cuda":
        return None, None
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8 and torch.cuda.is_bf16_supported():
        return torch.bfloat16, None
    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # older torch
        scaler = torch.cuda.amp.GradScaler()
    return torch.float16, scaler


def _sync(device: torch.device) -> None:
    """Block until queued device work is done, so timings mean something."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _training_step(
    model: PatchTSTClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    amp_dtype: torch.dtype | None,
    scaler,
    grad_clip: float,
    device: torch.device,
) -> torch.Tensor:
    """One optimiser step. Shared by `fit` and `estimate_throughput`.

    Both call this so a throughput estimate cannot drift from what training
    actually costs — the first version of the probe left out gradient clipping
    and autocast and was optimistic by roughly 2x.
    """
    optimizer.zero_grad(set_to_none=True)
    if amp_dtype is not None:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            loss = criterion(model(x), y)
    else:
        loss = criterion(model(x), y)

    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return loss


def estimate_throughput(
    model: PatchTSTClassifier,
    batcher: WindowBatcher,
    cfg: TrainConfig | None = None,
    device: torch.device | None = None,
    warmup: int = 5,
    measured: int = 20,
) -> float:
    """Windows/s of the *real* training step, measured on a copy of the model.

    The caller's weights and optimiser state are untouched — this runs against a
    deep copy — so it is safe to call immediately before `fit`.
    """
    cfg = cfg or TrainConfig()
    device = device or batcher.device
    probe = copy.deepcopy(model).to(device)
    probe.train()

    optimizer = torch.optim.AdamW(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _cosine_schedule(optimizer, max(warmup + measured, 1), cfg.warmup_frac)
    try:
        pos_weight = torch.tensor(batcher.pos_weight(), dtype=torch.float32, device=device)
    except ValueError:  # no positives in this split; irrelevant to timing
        pos_weight = None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    amp_dtype, scaler = _amp_setup(device, cfg.amp)

    started, seen = None, 0
    for step, (x, y) in enumerate(batcher.iter_batches(cfg.batch_size, shuffle=True)):
        if step == warmup:
            _sync(device)
            started, seen = time.perf_counter(), 0
        loss = _training_step(
            probe, x, y, criterion, optimizer, scheduler, amp_dtype, scaler, cfg.grad_clip, device
        )
        # fit() pays this host sync on every step; the probe must pay it too.
        float(loss.item())
        if started is not None:
            seen += y.numel()
            if step >= warmup + measured - 1:
                break
    _sync(device)

    if started is None or seen == 0:
        raise ValueError(
            f"Split has too few batches to probe ({batcher.n_batches(cfg.batch_size)}); "
            "lower `warmup`/`measured` or the batch size."
        )
    elapsed = time.perf_counter() - started
    del probe, optimizer
    return seen / elapsed


def _cosine_schedule(optimizer, total_steps: int, warmup_frac: float):
    warmup = max(1, int(total_steps * warmup_frac))

    def lr_at(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)


@torch.no_grad()
def evaluate(
    model: PatchTSTClassifier,
    batcher: WindowBatcher,
    criterion: nn.Module,
    batch_size: int,
    amp_dtype: torch.dtype | None,
) -> tuple[float, np.ndarray]:
    """Mean loss and per-window probabilities over a split."""
    model.eval()
    device_type = batcher.device.type
    total_loss, seen, probs = 0.0, 0, []
    for x, y in batcher.iter_batches(batch_size):
        if amp_dtype is not None:
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                logits = model(x)
        else:
            logits = model(x)
        logits = logits.float()
        total_loss += criterion(logits, y).item() * y.numel()
        seen += y.numel()
        probs.append(torch.sigmoid(logits).cpu())
    return total_loss / max(seen, 1), torch.cat(probs).numpy()


def fit(
    model: PatchTSTClassifier,
    train_batcher: WindowBatcher,
    val_batcher: WindowBatcher | None = None,
    cfg: TrainConfig | None = None,
    device: torch.device | None = None,
    verbose: bool = True,
) -> dict:
    """Train in place, restoring the best-validating weights before returning.

    Returns a `train_meta` dict that goes straight into the checkpoint.
    """
    cfg = cfg or TrainConfig()
    device = device or train_batcher.device
    set_seed(cfg.seed)
    model.to(device)

    # Mirrors XGBoost's scale_pos_weight: the profit threshold makes positives rare.
    pos_weight = cfg.pos_weight if cfg.pos_weight is not None else train_batcher.pos_weight()
    if cfg.max_pos_weight is not None:
        pos_weight = min(pos_weight, cfg.max_pos_weight)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
    )
    # Validation loss must stay comparable across epochs, so it is unweighted.
    val_criterion = nn.BCEWithLogitsLoss()

    if model.cfg.use_scale_features and float(model.aux_fitted.item()) == 0.0:
        model.fit_aux_normalization(train_batcher, batch_size=cfg.eval_batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = train_batcher.n_batches(cfg.batch_size)
    scheduler = _cosine_schedule(optimizer, steps_per_epoch * cfg.epochs, cfg.warmup_frac)
    amp_dtype, scaler = _amp_setup(device, cfg.amp)
    generator = torch.Generator().manual_seed(cfg.seed)

    if verbose:
        print(
            f"device={device} | params={model.n_parameters():,} | "
            f"train windows={len(train_batcher):,} ({train_batcher.positive_rate():.2%} positive) | "
            f"pos_weight={pos_weight:.3f} | "
            f"amp={'off' if amp_dtype is None else str(amp_dtype).split('.')[-1]}"
        )

    history: list[dict] = []
    best_score = -math.inf
    best_state = None
    best_epoch = -1
    epochs_run = 0
    metric_name = cfg.early_stop_metric

    for epoch in range(cfg.epochs):
        model.train()
        started = time.perf_counter()
        running, seen = 0.0, 0

        for x, y in train_batcher.iter_batches(cfg.batch_size, shuffle=True, generator=generator):
            optimizer.zero_grad(set_to_none=True)
            if amp_dtype is not None:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    loss = criterion(model(x), y)
            else:
                loss = criterion(model(x), y)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
            scheduler.step()

            running += loss.item() * y.numel()
            seen += y.numel()

        epochs_run = epoch + 1
        train_loss = running / max(seen, 1)
        elapsed = time.perf_counter() - started
        record = {
            "epoch": epochs_run,
            "train_loss": train_loss,
            "seconds": elapsed,
            "windows_per_s": seen / max(elapsed, 1e-9),
            "lr": scheduler.get_last_lr()[0],
        }

        if val_batcher is not None and len(val_batcher):
            val_loss, val_prob = evaluate(
                model, val_batcher, val_criterion, cfg.eval_batch_size, amp_dtype
            )
            val_metrics = binary_metrics(val_batcher.numpy_labels(), val_prob)
            record.update(
                {
                    "val_loss": val_loss,
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_f1": val_metrics["f1"],
                }
            )
            if metric_name == "roc_auc" and not math.isnan(val_metrics["roc_auc"]):
                score = val_metrics["roc_auc"]
            else:
                if metric_name == "roc_auc":
                    # A single-class validation split leaves AUC undefined.
                    metric_name = "loss"
                    if verbose:
                        print("  ! validation split has one class only — early stopping on loss")
                score = -val_loss
        else:
            score = -train_loss

        history.append(record)
        if verbose:
            msg = (
                f"epoch {epochs_run:>3}/{cfg.epochs} | train {train_loss:.4f} | "
                f"{elapsed:5.1f}s ({record['windows_per_s']:,.0f} win/s)"
            )
            if "val_loss" in record:
                msg += f" | val {record['val_loss']:.4f} | val AUC {record['val_roc_auc']:.4f}"
            print(msg)

        if score > best_score:
            best_score, best_epoch = score, epochs_run
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        elif cfg.patience and epochs_run - best_epoch >= cfg.patience:
            if verbose:
                print(f"early stop: no improvement for {cfg.patience} epochs")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        if verbose:
            print(
                f"restored weights from epoch {best_epoch} ({metric_name} = {abs(best_score):.4f})"
            )

    return {
        "train_config": cfg.to_dict(),
        "device": str(device),
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "best_val_metric": float(abs(best_score)),
        "early_stop_metric": metric_name,
        "pos_weight": float(pos_weight),
        "history": history,
    }
