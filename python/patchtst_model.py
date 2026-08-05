"""
PatchTST (Nie et al., ICLR 2023) adapted from long-horizon forecasting to the
binary profit-threshold classification problem the tabular baselines solve.

What is kept from the paper
---------------------------
* **Patching.** The 300-bar lookback is cut into overlapping patches of P=16
  bars (stride S=8), so the encoder sees N = 37 tokens instead of 300. Attention
  cost drops by ~(300/37)^2 ~ 66x, and each token carries a sub-series with
  local semantics rather than one meaningless second.
* **Channel independence.** Every channel shares one set of Transformer weights
  and is pushed through the encoder as an independent univariate series, batched
  as (B*M, N, D) exactly as described in the paper's A.1.5.
* **Instance normalisation.** Each window/channel is standardised before
  patching, which is what makes the model survive the volatility regime shifts
  that crypto data is made of.
* **BatchNorm inside the encoder** rather than LayerNorm, per the paper's
  footnote 1.

What is changed for this problem
--------------------------------
* **Head.** The paper's `Flatten + Linear -> T future values` becomes
  `Flatten + Linear -> 1 logit`, trained with `BCEWithLogitsLoss(pos_weight=...)`
  where pos_weight plays the role XGBoost's `scale_pos_weight` plays.
* **Two raw channels, deeper stack.** The model reads `log_return` and `ofi`
  only — the two primitive observables. Everything else the tabular pipeline
  computes (r^2, (pi/2)|r||r_prev|, log volume) is a pointwise function of those
  two and can be formed in the first layer, so feeding it in explicitly spends
  channel width to buy nothing. The saved budget goes into depth: 6 encoder
  layers instead of 3.
* **Scale features.** Instance normalisation throws away the very thing this
  problem cares about — *how* volatile and *how* imbalanced the window was.
  The per-window mean and log-std of every channel (2M numbers) are standardised
  and concatenated into the head, so the aggregate scale is still available.

Efficiency
----------
Windows overlap by 299/300 bars, so materialising them would cost
`n_windows x M x 300` floats (~100 GB for a month). Nothing here ever does:
`WindowBatcher` keeps the (M, n_bars) channel matrix resident on the device
(~64 MB for a month) and cuts each batch with a single gather. Attention runs
through `F.scaled_dot_product_attention`, so PyTorch picks the fused
Flash/mem-efficient kernel automatically.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

CHECKPOINT_FORMAT = 1


# --- Configuration -----------------------------------------------------------


@dataclass
class PatchTSTConfig:
    """Everything needed to rebuild the network from a checkpoint.

    The defaults describe the current model: **2 raw channels, 6 encoder
    layers**. The earlier configuration fed 6 channels through 3 layers for
    almost exactly the same parameter budget, but four of those channels
    (`realized_var`, `bipower`, `log_volume`, `log_trades`) are pointwise
    functions of the other two, so the width was spent re-encoding information
    the network already had. Trading it for depth is the whole change.
    """

    n_channels: int = 2
    seq_len: int = 300
    patch_len: int = 16
    stride: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 6
    d_ff: int = 128
    dropout: float = 0.2
    head_dropout: float = 0.2
    head_hidden: int = 0  # 0 -> a single linear layer on the flattened encoder output
    use_scale_features: bool = True
    norm: str = "batch"  # "batch" (paper) or "layer"
    eps: float = 1e-8

    @property
    def num_patches(self) -> int:
        """N = floor((L - P) / S) + 2, matching the paper's end-padded patching."""
        if self.seq_len < self.patch_len:
            raise ValueError(f"seq_len {self.seq_len} < patch_len {self.patch_len}")
        return (self.seq_len - self.patch_len) // self.stride + 2

    @property
    def n_aux(self) -> int:
        return 2 * self.n_channels if self.use_scale_features else 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> PatchTSTConfig:
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in known})


# --- Building blocks ---------------------------------------------------------


class _BatchNormTokens(nn.Module):
    """BatchNorm over the model dimension of a (batch, tokens, d_model) tensor."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


def _make_norm(kind: str, d_model: int) -> nn.Module:
    if kind == "batch":
        return _BatchNormTokens(d_model)
    if kind == "layer":
        return nn.LayerNorm(d_model)
    raise ValueError(f"norm must be 'batch' or 'layer', got {kind!r}")


class _MultiHeadAttention(nn.Module):
    """Vanilla MHA routed through PyTorch's fused attention kernels.

    No `dropout_p` is passed to `scaled_dot_product_attention`: the MPS backend
    raises `NotImplementedError` for it, which would make the model trainable on
    CUDA and CPU but not on Apple Silicon. Regularisation of the attention path is
    handled by the residual dropout in `_EncoderLayer`, so behaviour is identical
    on every device rather than silently differing by backend.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class _EncoderLayer(nn.Module):
    """Post-norm Transformer block, as drawn in Figure 1(b) of the paper."""

    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.attn = _MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm1 = _make_norm(cfg.norm, cfg.d_model)
        self.norm2 = _make_norm(cfg.norm, cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.drop(self.attn(x)))
        return self.norm2(x + self.drop(self.ff(x)))


# --- The model ---------------------------------------------------------------


class PatchTSTClassifier(nn.Module):
    """Channel-independent patch Transformer with a single-logit head.

    Input:  (batch, n_channels, seq_len) float32
    Output: (batch,) raw logits — feed to `BCEWithLogitsLoss`, sigmoid for probs.
    """

    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.cfg = cfg
        n_patches = cfg.num_patches

        self.value_embedding = nn.Linear(cfg.patch_len, cfg.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(n_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.embed_dropout = nn.Dropout(cfg.dropout)

        self.layers = nn.ModuleList(_EncoderLayer(cfg) for _ in range(cfg.n_layers))

        flat_dim = cfg.n_channels * n_patches * cfg.d_model + cfg.n_aux
        head: list[nn.Module] = [nn.Dropout(cfg.head_dropout)]
        if cfg.head_hidden:
            head += [nn.Linear(flat_dim, cfg.head_hidden), nn.GELU(), nn.Dropout(cfg.head_dropout)]
            flat_dim = cfg.head_hidden
        head.append(nn.Linear(flat_dim, 1))
        self.head = nn.Sequential(*head)

        # Standardisation for the auxiliary scale features. Filled in from the
        # training split by `fit_aux_normalization`; identity until then.
        self.register_buffer("aux_mean", torch.zeros(max(cfg.n_aux, 1)))
        self.register_buffer("aux_std", torch.ones(max(cfg.n_aux, 1)))
        self.register_buffer("aux_fitted", torch.zeros(1))

    # -- pieces, exposed so the training code can reuse them --

    def instance_norm(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Standardise each (window, channel) and return the discarded scale."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        normed = (x - mean) / (std + self.cfg.eps)
        aux = torch.cat([mean.squeeze(-1), torch.log(std.squeeze(-1) + self.cfg.eps)], dim=1)
        return normed, aux

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """(B, M, L) -> (B, M, N, P), padding S repeats of the last value."""
        padded = F.pad(x, (0, self.cfg.stride), mode="replicate")
        return padded.unfold(dimension=-1, size=self.cfg.patch_len, step=self.cfg.stride)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B, M, L) -> per-channel representations (B, M, N, D) and aux stats."""
        b, m, _ = x.shape
        normed, aux = self.instance_norm(x)
        patches = self.patchify(normed)  # (B, M, N, P)

        tokens = self.value_embedding(patches) + self.pos_embedding
        tokens = self.embed_dropout(tokens)

        # Channel independence: fold M into the batch so one set of weights is
        # shared by every series, then unfold it again.
        tokens = tokens.reshape(b * m, self.cfg.num_patches, self.cfg.d_model)
        for layer in self.layers:
            tokens = layer(tokens)
        return tokens.reshape(b, m, self.cfg.num_patches, self.cfg.d_model), aux

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, aux = self.encode(x)
        flat = z.flatten(start_dim=1)
        if self.cfg.use_scale_features:
            aux = (aux - self.aux_mean) / self.aux_std
            flat = torch.cat([flat, aux.to(flat.dtype)], dim=1)
        return self.head(flat).squeeze(-1)

    # -- calibration --

    @torch.no_grad()
    def fit_aux_normalization(self, batcher: WindowBatcher, batch_size: int = 4096) -> None:
        """Set the auxiliary-feature standardisation from a (training) split."""
        if not self.cfg.use_scale_features:
            self.aux_fitted.fill_(1.0)
            return
        # Accumulated on the CPU in float64: these are sums of squares of values
        # spanning ~20 orders of magnitude, so float32 would lose the variance —
        # and MPS refuses float64 outright. The transfer is (batch, 2M) floats.
        total = torch.zeros(self.cfg.n_aux, dtype=torch.float64)
        total_sq = torch.zeros_like(total)
        count = 0
        for x, _ in batcher.iter_batches(batch_size):
            _, aux = self.instance_norm(x)
            # .cpu() first, *then* .double(): a combined .to(cpu, float64) asks the
            # source device for the cast, and MPS has no float64 at all — it
            # silently returns inf rather than raising.
            aux = aux.detach().cpu().double()
            total += aux.sum(0)
            total_sq += (aux * aux).sum(0)
            count += aux.shape[0]
        mean = total / count
        var = torch.clamp(total_sq / count - mean * mean, min=0.0)
        self.aux_mean.copy_(mean.float())
        self.aux_std.copy_(torch.sqrt(var).clamp_min(1e-6).float())
        self.aux_fitted.fill_(1.0)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# --- Batching ----------------------------------------------------------------


class WindowBatcher:
    """Cuts overlapping windows out of a device-resident channel matrix.

    No `DataLoader`, no per-batch host-to-device copies, no materialised window
    tensor: the whole bar series lives on the device once and every batch is one
    gather (`channels[:, starts[:, None] + arange(L)]`).
    """

    def __init__(
        self,
        channels: np.ndarray | torch.Tensor,
        starts: np.ndarray | torch.Tensor,
        labels: np.ndarray | torch.Tensor,
        seq_len: int,
        device: str | torch.device = "cpu",
        shared_channels: torch.Tensor | None = None,
    ):
        self.device = torch.device(device)
        self.seq_len = seq_len
        # (n_bars, M) -> (M, n_bars): the gather then walks contiguous time.
        if shared_channels is not None:
            self.channels = shared_channels
        else:
            self.channels = (
                torch.as_tensor(np.asarray(channels), dtype=torch.float32)
                .t()
                .contiguous()
                .to(self.device)
            )
        self.starts = torch.as_tensor(np.asarray(starts), dtype=torch.long, device=self.device)
        self.labels = torch.as_tensor(np.asarray(labels), dtype=torch.float32, device=self.device)
        self._offsets = torch.arange(seq_len, device=self.device)

    def __len__(self) -> int:
        return int(self.starts.numel())

    def like(self, starts, labels) -> WindowBatcher:
        """A second view over the same device-resident channel matrix."""
        return WindowBatcher(
            None, starts, labels, self.seq_len, self.device, shared_channels=self.channels
        )

    def gather(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        window = self.starts[positions].unsqueeze(1) + self._offsets.unsqueeze(0)  # (B, L)
        x = self.channels[:, window].permute(1, 0, 2).contiguous()  # (B, M, L)
        return x, self.labels[positions]

    def iter_batches(
        self, batch_size: int, shuffle: bool = False, generator: torch.Generator | None = None
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        n = len(self)
        if shuffle:
            order = torch.randperm(n, generator=generator).to(self.device)
        else:
            order = torch.arange(n, device=self.device)
        for lo in range(0, n, batch_size):
            yield self.gather(order[lo : lo + batch_size])

    def n_batches(self, batch_size: int) -> int:
        return (len(self) + batch_size - 1) // batch_size

    def pos_weight(self) -> float:
        """`n_negative / n_positive`, i.e. XGBoost's `scale_pos_weight`."""
        pos = float(self.labels.sum().item())
        if pos == 0:
            raise ValueError(
                "This split contains no positive windows — widen the time range "
                "or lower FEE_THRESHOLD."
            )
        return (len(self) - pos) / pos

    def positive_rate(self) -> float:
        return float(self.labels.mean().item())

    def numpy_labels(self) -> np.ndarray:
        return self.labels.detach().cpu().numpy()


# --- Inference & metrics -----------------------------------------------------


@torch.no_grad()
def predict_proba(
    model: PatchTSTClassifier,
    batcher: WindowBatcher,
    batch_size: int = 2048,
    amp_dtype: torch.dtype | None = None,
) -> np.ndarray:
    """Sigmoid probabilities for every window in `batcher`, in order."""
    model.eval()
    device_type = batcher.device.type
    out = []
    for x, _ in batcher.iter_batches(batch_size):
        if amp_dtype is not None and device_type == "cuda":
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                logits = model(x)
        else:
            logits = model(x)
        out.append(torch.sigmoid(logits.float()).cpu())
    return torch.cat(out).numpy() if out else np.empty(0, dtype=np.float32)


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """The metric set the other baselines report, plus the honesty columns."""
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    base_rate = float(y_true.mean()) if y_true.size else float("nan")
    # A single-class split makes ROC-AUC undefined rather than 0.5.
    auc = float(roc_auc_score(y_true, y_prob)) if 0 < y_true.sum() < y_true.size else float("nan")
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
        "base_rate": base_rate,
        "majority_accuracy": float(1.0 - base_rate),
        "n_windows": int(y_true.size),
        "n_positive": int(y_true.sum()),
        "n_flagged": int(y_pred.sum()),
    }


def format_metrics(metrics: dict) -> str:
    return (
        f"Accuracy:  {metrics['accuracy']:.4f}   (always-0 baseline: {metrics['majority_accuracy']:.4f})\n"
        f"Precision: {metrics['precision']:.4f}   (base rate:         {metrics['base_rate']:.4f})\n"
        f"Recall:    {metrics['recall']:.4f}\n"
        f"F1-Score:  {metrics['f1']:.4f}\n"
        f"ROC-AUC:   {metrics['roc_auc']:.4f}   (0.5000 is a coin flip)"
    )


def threshold_sweep(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float] | None = None
) -> list[dict]:
    """Precision/recall at several cut-offs — 0.5 is arbitrary for a rare event."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return [binary_metrics(y_true, y_prob, t) for t in thresholds]


# --- Checkpoints -------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model: PatchTSTClassifier,
    data_meta: dict | None = None,
    train_meta: dict | None = None,
    metrics: dict | None = None,
) -> Path:
    """Write weights + the config and data recipe needed to reproduce the split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": CHECKPOINT_FORMAT,
            "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "config": model.cfg.to_dict(),
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "data_meta": data_meta or {},
            "train_meta": train_meta or {},
            "metrics": metrics or {},
        },
        path,
    )
    return path


def load_checkpoint(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[PatchTSTClassifier, dict]:
    """Rebuild the exact network the checkpoint was trained with."""
    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # torch < 2.0 has no weights_only
        payload = torch.load(path, map_location=map_location)

    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT:
        raise ValueError(
            f"Checkpoint format v{version} does not match this code (v{CHECKPOINT_FORMAT}). "
            "Re-export it from the training notebook."
        )

    cfg = PatchTSTConfig.from_dict(payload["config"])
    model = PatchTSTClassifier(cfg)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def describe_checkpoint(payload: dict) -> str:
    lines = [
        f"trained: {payload.get('created_utc', 'unknown')}",
        f"config:  {json.dumps(payload.get('config', {}), sort_keys=True)}",
    ]
    data_meta = payload.get("data_meta", {})
    if data_meta:
        lines.append(
            f"data:    {data_meta.get('source', '?')} | {data_meta.get('n_bars', '?'):,} bars | "
            f"window {data_meta.get('window', '?')} | horizon {data_meta.get('horizon', '?')}"
        )
        # Absent on checkpoints predating the switch, which is exactly when it
        # matters most to say which target the weights were fitted against.
        lines.append(
            f"label:   {data_meta.get('label_mode', 'fee_threshold (pre-triple-barrier)')} "
            f"at +/-{data_meta.get('barrier', data_meta.get('fee_threshold', '?'))} | "
            f"channels {data_meta.get('channels', '?')}"
        )
    train_meta = payload.get("train_meta", {})
    if train_meta:
        lines.append(
            f"training: {train_meta.get('epochs_run', '?')} epochs | "
            f"best val {train_meta.get('best_val_metric', float('nan')):.4f} "
            f"({train_meta.get('early_stop_metric', 'roc_auc')}) | device {train_meta.get('device', '?')}"
        )
    return "\n".join(lines)
