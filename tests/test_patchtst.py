"""The sequence model: shapes, batching, metrics, checkpoints, and a training step.

Deliberately tiny configurations throughout — 2 layers, d_model 16, a 32-bar
lookback. The point is to exercise the wiring, not to reproduce a result: a test
that trained the real 209k-parameter model would take longer than the CI budget
and still tell you nothing a shape assertion does not.

The checkpoint tests carry the most weight. A checkpoint consumer refuses to score a
checkpoint whose recorded data recipe disagrees with the rebuilt one, which is
only a safeguard if the recipe survives a save/load round trip intact.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import sequence_matrix as seq
from patchtst_model import (
    PatchTSTClassifier,
    PatchTSTConfig,
    WindowBatcher,
    binary_metrics,
    describe_checkpoint,
    format_metrics,
    load_checkpoint,
    predict_proba,
    save_checkpoint,
    threshold_sweep,
)
from patchtst_train import TrainConfig, evaluate, fit, pick_device, set_seed


def tiny_config(**overrides) -> PatchTSTConfig:
    base = dict(
        n_channels=2, seq_len=32, patch_len=8, stride=4, d_model=16, n_heads=2, n_layers=2, d_ff=32
    )
    return PatchTSTConfig(**{**base, **overrides})


@pytest.fixture
def batcher(bars) -> WindowBatcher:
    """A batcher over the synthetic bars, at the tiny configuration's window size."""
    channels = seq.build_channels(bars, "raw")
    starts = np.arange(0, 500, dtype=np.int64)
    labels = (np.arange(starts.size) % 3 == 0).astype(np.int8)
    return WindowBatcher(channels, starts, labels, seq_len=32, device="cpu")


# --- configuration ------------------------------------------------------------


def test_patch_count_follows_the_papers_formula():
    """N = floor((L - P) / S) + 2, with the end-padding the paper specifies."""
    assert tiny_config(seq_len=32, patch_len=8, stride=4).num_patches == 8
    # The production configuration: 300 bars, patch 16, stride 8 -> 37 tokens.
    assert PatchTSTConfig(seq_len=300, patch_len=16, stride=8).num_patches == 37


def test_a_lookback_shorter_than_a_patch_is_rejected():
    with pytest.raises(ValueError, match="seq_len"):
        _ = tiny_config(seq_len=4, patch_len=8).num_patches


def test_aux_feature_count_is_two_per_channel():
    """Instance norm discards a mean and a log-std per channel; both are fed back."""
    assert tiny_config(n_channels=2).n_aux == 4
    assert tiny_config(n_channels=6).n_aux == 12
    assert tiny_config(n_channels=2, use_scale_features=False).n_aux == 0


def test_config_survives_a_dict_round_trip():
    cfg = tiny_config(dropout=0.3, norm="layer")
    assert PatchTSTConfig.from_dict(cfg.to_dict()) == cfg


def test_config_ignores_unknown_keys_when_rebuilding():
    """A checkpoint from a future version must still load what this code knows."""
    payload = {**tiny_config().to_dict(), "an_option_added_later": 123}
    assert PatchTSTConfig.from_dict(payload).n_channels == 2


# --- forward pass -------------------------------------------------------------


def test_forward_returns_one_logit_per_window():
    model = PatchTSTClassifier(tiny_config())
    out = model(torch.randn(5, 2, 32))

    assert out.shape == (5,)
    assert torch.isfinite(out).all()


def test_patchify_produces_the_expected_token_grid():
    cfg = tiny_config()
    model = PatchTSTClassifier(cfg)
    patches = model.patchify(torch.randn(3, cfg.n_channels, cfg.seq_len))

    assert patches.shape == (3, cfg.n_channels, cfg.num_patches, cfg.patch_len)


def test_instance_norm_standardises_each_window_and_channel():
    model = PatchTSTClassifier(tiny_config())
    # Two channels on wildly different scales — the case instance norm exists for.
    x = torch.stack([torch.randn(4, 32) * 1000 + 5000, torch.randn(4, 32) * 0.001], dim=1)

    normed, aux = model.instance_norm(x)

    assert normed.mean(dim=-1).abs().max() < 1e-4
    assert (normed.std(dim=-1, unbiased=False) - 1.0).abs().max() < 1e-3
    # The discarded scale is handed back as (mean, log std) per channel.
    assert aux.shape == (4, 4)


def test_a_constant_window_does_not_produce_nan():
    """Zero variance divides by `eps`, not by zero — flat windows are real."""
    model = PatchTSTClassifier(tiny_config())
    normed, aux = model.instance_norm(torch.full((2, 2, 32), 7.0))

    assert torch.isfinite(normed).all()
    assert torch.isfinite(aux).all()


def test_scale_features_change_the_prediction():
    """If they did not, concatenating them back into the head would be dead weight."""
    set_seed(0)
    model = PatchTSTClassifier(tiny_config()).eval()
    base = torch.randn(4, 2, 32)

    with torch.no_grad():
        # Same shape, ten times the amplitude: instance norm erases the difference,
        # so only the auxiliary features can carry it.
        assert not torch.allclose(model(base), model(base * 10.0), atol=1e-4)


def test_a_model_without_scale_features_is_blind_to_amplitude():
    """The converse, which is what makes the previous test meaningful."""
    set_seed(0)
    model = PatchTSTClassifier(tiny_config(use_scale_features=False)).eval()
    base = torch.randn(4, 2, 32)

    with torch.no_grad():
        torch.testing.assert_close(model(base), model(base * 10.0), atol=1e-4, rtol=1e-3)


@pytest.mark.parametrize("norm", ["batch", "layer"])
def test_both_normalisations_build_and_run(norm):
    model = PatchTSTClassifier(tiny_config(norm=norm))
    assert model(torch.randn(4, 2, 32)).shape == (4,)


def test_an_unknown_normalisation_is_rejected():
    with pytest.raises(ValueError):
        PatchTSTClassifier(tiny_config(norm="nonsense"))


def test_head_hidden_adds_a_layer_and_parameters():
    plain = PatchTSTClassifier(tiny_config()).n_parameters()
    deeper = PatchTSTClassifier(tiny_config(head_hidden=8)).n_parameters()

    assert deeper != plain


def test_channel_count_is_respected():
    model = PatchTSTClassifier(tiny_config(n_channels=6))
    assert model(torch.randn(3, 6, 32)).shape == (3,)


def test_the_production_configuration_has_the_documented_size():
    """§5.3 quotes 209,029 parameters for 2 channels over 6 layers."""
    model = PatchTSTClassifier(PatchTSTConfig(n_channels=2, seq_len=300))
    assert model.n_parameters() == 209_029


# --- batching -----------------------------------------------------------------


def test_batcher_gathers_windows_with_the_right_shape(batcher):
    x, y = next(batcher.iter_batches(16))

    assert x.shape == (16, 2, 32)
    assert y.shape == (16,)


def test_gathered_windows_match_the_source_channels(bars):
    """The gather is the only path from bars to model input, so it must be exact."""
    channels = seq.build_channels(bars, "raw")
    starts = np.array([0, 7, 100], dtype=np.int64)
    view = WindowBatcher(channels, starts, np.zeros(3, dtype=np.int8), seq_len=32)

    x, _ = view.gather(torch.arange(3))

    for row, start in enumerate(starts):
        expected = channels[start : start + 32, :].T  # (M, L)
        np.testing.assert_allclose(x[row].numpy(), expected, rtol=1e-6)


def test_every_window_appears_exactly_once_per_epoch(batcher):
    seen = sum(x.shape[0] for x, _ in batcher.iter_batches(64))
    assert seen == len(batcher)


def test_shuffling_reorders_without_losing_windows(batcher):
    generator = torch.Generator().manual_seed(0)
    shuffled = torch.cat(
        [y for _, y in batcher.iter_batches(64, shuffle=True, generator=generator)]
    )
    ordered = torch.cat([y for _, y in batcher.iter_batches(64)])

    assert shuffled.shape == ordered.shape
    assert shuffled.sum() == ordered.sum()


def test_batch_count_covers_a_ragged_final_batch(batcher):
    assert batcher.n_batches(64) == (len(batcher) + 63) // 64
    assert batcher.n_batches(len(batcher)) == 1


def test_like_shares_the_channel_matrix(batcher):
    """Views must not copy: one resident matrix is the whole memory argument."""
    other = batcher.like(np.array([0, 1, 2]), np.zeros(3, dtype=np.int8))

    assert other.channels.data_ptr() == batcher.channels.data_ptr()
    assert len(other) == 3


def test_pos_weight_is_the_negative_to_positive_ratio(bars):
    channels = seq.build_channels(bars, "raw")
    labels = np.array([1, 0, 0, 0], dtype=np.int8)
    view = WindowBatcher(channels, np.arange(4), labels, seq_len=32)

    assert view.pos_weight() == pytest.approx(3.0)
    assert view.positive_rate() == pytest.approx(0.25)


def test_a_split_with_no_positives_is_rejected(bars):
    """Training on it would divide by zero; the message says how to fix it."""
    channels = seq.build_channels(bars, "raw")
    view = WindowBatcher(channels, np.arange(4), np.zeros(4, dtype=np.int8), seq_len=32)

    with pytest.raises(ValueError, match="no positive windows"):
        view.pos_weight()


# --- inference and metrics ----------------------------------------------------


def test_predict_proba_returns_probabilities_in_order(batcher):
    model = PatchTSTClassifier(tiny_config())
    probs = predict_proba(model, batcher, batch_size=64)

    assert probs.shape == (len(batcher),)
    assert np.all((probs >= 0.0) & (probs <= 1.0))
    # Unshuffled, so a second pass must line up window for window.
    np.testing.assert_allclose(probs, predict_proba(model, batcher, batch_size=32), atol=1e-6)


def test_predict_proba_on_an_empty_split_returns_an_empty_array(bars):
    channels = seq.build_channels(bars, "raw")
    empty = WindowBatcher(
        channels, np.array([], dtype=np.int64), np.array([], dtype=np.int8), seq_len=32
    )

    assert predict_proba(PatchTSTClassifier(tiny_config()), empty).size == 0


def test_binary_metrics_on_a_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    metrics = binary_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))

    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["base_rate"] == pytest.approx(0.5)


def test_binary_metrics_auc_is_nan_for_a_single_class():
    """Undefined rather than 0.5, so a degenerate split cannot pass as a coin flip."""
    assert np.isnan(
        binary_metrics(np.ones(4, dtype=int), np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"]
    )


def test_binary_metrics_threshold_moves_precision_and_recall():
    y = np.array([0, 1, 1, 1])
    probs = np.array([0.4, 0.45, 0.6, 0.9])

    lenient = binary_metrics(y, probs, threshold=0.3)
    strict = binary_metrics(y, probs, threshold=0.8)

    assert lenient["recall"] > strict["recall"]
    assert strict["precision"] >= lenient["precision"]


def test_format_metrics_is_a_string(bars):
    assert isinstance(format_metrics(binary_metrics(np.array([0, 1]), np.array([0.2, 0.8]))), str)


def test_threshold_sweep_covers_the_requested_cuts():
    y = np.array([0, 0, 1, 1])
    rows = threshold_sweep(y, np.array([0.1, 0.4, 0.6, 0.9]), thresholds=(0.3, 0.5, 0.7))

    assert len(rows) == 3


# --- training -----------------------------------------------------------------


def test_pick_device_returns_something_usable():
    device = pick_device()
    assert isinstance(device, torch.device)
    # An explicit preference is honoured, which is what `--device` relies on.
    assert pick_device("cpu").type == "cpu"


def test_set_seed_makes_initialisation_reproducible():
    set_seed(123)
    first = PatchTSTClassifier(tiny_config()).head[-1].weight.clone()
    set_seed(123)
    second = PatchTSTClassifier(tiny_config()).head[-1].weight.clone()

    torch.testing.assert_close(first, second)


def test_aux_normalisation_is_fitted_from_a_split(batcher):
    model = PatchTSTClassifier(tiny_config())
    assert model.aux_fitted.item() == 0.0

    model.fit_aux_normalization(batcher, batch_size=128)

    assert model.aux_fitted.item() == 1.0
    assert torch.isfinite(model.aux_mean).all()
    # Standard deviations are clamped away from zero, so the head cannot divide by 0.
    assert (model.aux_std > 0).all()


def _loss_on(model: PatchTSTClassifier, view: WindowBatcher) -> float:
    """Mean BCE over a split, at the signature `fit` calls `evaluate` with."""
    loss, _ = evaluate(model, view, torch.nn.BCEWithLogitsLoss(), 128, None)
    return loss


def test_evaluate_reports_a_loss_and_per_window_probabilities(batcher):
    model = PatchTSTClassifier(tiny_config())
    loss, probs = evaluate(model, batcher, torch.nn.BCEWithLogitsLoss(), 128, None)

    assert np.isfinite(loss) and loss > 0.0
    assert probs.shape == (len(batcher),)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


@pytest.mark.slow
def test_fit_runs_and_reports_its_history(bars):
    """One short run end to end: the loop, early stopping bookkeeping, and history."""
    channels = seq.build_channels(bars, "raw")
    starts = np.arange(0, 600, dtype=np.int64)
    labels = (np.arange(starts.size) % 2 == 0).astype(np.int8)

    train = WindowBatcher(channels, starts[:400], labels[:400], seq_len=32)
    val = train.like(starts[400:], labels[400:])

    set_seed(0)
    model = PatchTSTClassifier(tiny_config())
    meta = fit(
        model,
        train,
        val,
        TrainConfig(epochs=2, batch_size=64, eval_batch_size=128, amp=False),
        device=torch.device("cpu"),
        verbose=False,
    )

    assert meta["epochs_run"] == 2
    assert len(meta["history"]) == 2
    assert 1 <= meta["best_epoch"] <= 2
    assert np.isfinite(meta["best_val_metric"])


@pytest.mark.slow
def test_training_reduces_the_loss_on_a_learnable_signal(bars):
    """A label the model can actually see, so a broken optimiser step shows up.

    The first channel is overwritten with the label itself, which is not a claim
    about the market — it is the smallest signal that separates "the gradient
    flows" from "the loss wandered".
    """
    channels = seq.build_channels(bars, "raw").copy()
    starts = np.arange(0, 600, dtype=np.int64)
    labels = (np.arange(starts.size) % 2 == 0).astype(np.int8)
    for start, label in zip(starts, labels):
        channels[start : start + 32, 0] = float(label)

    train = WindowBatcher(channels, starts, labels, seq_len=32)
    set_seed(0)
    model = PatchTSTClassifier(tiny_config())

    before = _loss_on(model, train)
    fit(
        model,
        train,
        train,
        TrainConfig(epochs=3, batch_size=64, eval_batch_size=128, amp=False, lr=3e-3),
        device=torch.device("cpu"),
        verbose=False,
    )
    after = _loss_on(model, train)

    assert after < before


# --- checkpoints --------------------------------------------------------------


def test_checkpoint_round_trip_preserves_predictions(batcher, tmp_path):
    """The property that matters: reloaded weights score identically."""
    set_seed(0)
    model = PatchTSTClassifier(tiny_config())
    model.fit_aux_normalization(batcher, batch_size=128)
    before = predict_proba(model, batcher, batch_size=128)

    path = save_checkpoint(
        tmp_path / "ckpt.pt",
        model,
        data_meta={"source": "synthetic", "window": 32},
        train_meta={"epochs_run": 1},
        metrics={"roc_auc": 0.5},
    )
    reloaded, payload = load_checkpoint(path)

    np.testing.assert_allclose(before, predict_proba(reloaded, batcher, batch_size=128), atol=1e-6)
    assert payload["data_meta"]["source"] == "synthetic"
    assert payload["config"]["seq_len"] == 32


def test_checkpoint_carries_the_aux_normalisation_buffers(batcher, tmp_path):
    """They are not parameters, so a state dict that dropped them would still load.

    It would also silently change every prediction, since the head divides by them.
    """
    model = PatchTSTClassifier(tiny_config())
    model.fit_aux_normalization(batcher, batch_size=128)

    reloaded, _ = load_checkpoint(save_checkpoint(tmp_path / "ckpt.pt", model))

    torch.testing.assert_close(reloaded.aux_mean, model.aux_mean)
    torch.testing.assert_close(reloaded.aux_std, model.aux_std)
    assert reloaded.aux_fitted.item() == 1.0


def test_checkpoint_loads_without_unpickling_arbitrary_objects(tmp_path):
    """`weights_only=True` is what makes a downloaded checkpoint safe to open."""
    path = save_checkpoint(tmp_path / "ckpt.pt", PatchTSTClassifier(tiny_config()))
    payload = torch.load(path, map_location="cpu", weights_only=True)

    assert payload["format_version"] == 1


def test_a_checkpoint_from_another_format_is_refused(tmp_path):
    """Loading it would rebuild the wrong network and report plausible numbers."""
    path = tmp_path / "old.pt"
    torch.save({"format_version": 0, "config": tiny_config().to_dict(), "state_dict": {}}, path)

    with pytest.raises(ValueError, match="format"):
        load_checkpoint(path)


def test_checkpoint_records_the_data_recipe_that_gates_evaluation(tmp_path):
    """A consumer refuses to score on a recipe mismatch, so this must round-trip."""
    recipe = {
        "source": "BTCUSDT-trades-2026-05.csv",
        "n_bars": 2_678_400,
        "window": 300,
        "horizon": 60,
        "barrier": 0.0005,
        "label_mode": "triple_barrier",
        "channels": ["log_return", "ofi"],
    }
    path = save_checkpoint(
        tmp_path / "ckpt.pt", PatchTSTClassifier(tiny_config()), data_meta=recipe
    )
    _, payload = load_checkpoint(path)

    assert payload["data_meta"] == recipe


def test_describe_checkpoint_names_the_label_mode(tmp_path):
    _, payload = load_checkpoint(
        save_checkpoint(
            tmp_path / "ckpt.pt",
            PatchTSTClassifier(tiny_config()),
            data_meta={
                "source": "s",
                "n_bars": 10,
                "label_mode": "triple_barrier",
                "barrier": 0.0005,
                "channels": ["log_return", "ofi"],
            },
        )
    )

    assert "triple_barrier" in describe_checkpoint(payload)


def test_describe_checkpoint_flags_weights_that_did_not_record_their_target(tmp_path):
    """A checkpoint that does not say what it was fitted against must not be scored."""
    _, payload = load_checkpoint(
        save_checkpoint(
            tmp_path / "old.pt",
            PatchTSTClassifier(tiny_config()),
            data_meta={"source": "s", "n_bars": 10, "channels": ["log_return", "ofi"]},
        )
    )

    assert "unrecorded" in describe_checkpoint(payload)


def test_dataset_kwargs_from_defaults_to_the_current_target():
    """A key the checkpoint omitted resolves to what the code would have used anyway."""
    kwargs = seq.dataset_kwargs_from({"window": 300, "horizon": 60, "barrier": 0.0005})
    assert kwargs["label_mode"] == "triple_barrier"


def test_dataset_kwargs_from_preserves_a_recorded_label_mode():
    kwargs = seq.dataset_kwargs_from(
        {"window": 300, "horizon": 60, "barrier": 0.0005, "label_mode": "triple_barrier"}
    )
    assert kwargs["label_mode"] == "triple_barrier"
