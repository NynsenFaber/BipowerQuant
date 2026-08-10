# Trained weights

Drop PatchTST checkpoints exported by
[`notebooks/train_patchtst_colab.ipynb`](../notebooks/train_patchtst_colab.ipynb) in
this directory, then score them with:

```bash
cd python
python eval_patchtst.py --weights ../weights/patchtst_BTCUSDT_2026-05_full.pt \
                        --bars-cache ../data/bars_full.npz
```

`eval_patchtst.py` defaults to `weights/patchtst.pt`, so renaming (or symlinking) a
checkpoint to that name lets you drop the `--weights` flag.

## One set of weights per lookback

The study trains the same architecture at three lookbacks, so everything here
comes in threes. The notebook writes, per scale:

| File | Contents |
| :--- | :--- |
| `patchtst_probs_<scale>.npz` | one out-of-sample probability array per fold, keyed by fold name |
| `patchtst_folds_<scale>.json` | the run's metadata: chosen stride, per-fold AUC, best epoch |
| `patchtst_<fold-slug>.pt` | one checkpoint per fold |

`<scale>` is `short` (5-minute lookback), `mid` (2 hours) or `long` (24 hours).
`patchtst_probs_short.npz` is the run the README's 5-minute results come from;
`patchtst_probs.npz` is the same file under its pre-lookback-study name, kept so older
commands still resolve. The `.npz` files are what the backtest consumes:

```bash
cd python
python walkforward.py --bars ../data/bars_6m.npz --window-scale mid \
                      --patchtst-probs ../weights/patchtst_probs_mid.npz \
                      --out ../data/wf_mid.json
```

**The scale in the filename is not checked against the probabilities**, because
`walkforward.py` only sees an array of numbers keyed by fold name. Scoring the
2-hour probabilities against a 5-minute run would produce a plausible-looking
table rather than an error, so keep the pairing straight. Each checkpoint's
`data_meta` records its `window` and `window_scale`, which is the authoritative
answer to what a given file was trained at.

## What is inside a checkpoint

`.pt` files here are plain `torch.save` dicts — no pickled classes, so they load
under `weights_only=True`:

| Key | Contents |
| :--- | :--- |
| `config` | every `PatchTSTConfig` field, so the network is rebuilt exactly as trained |
| `state_dict` | weights, plus the auxiliary-feature standardisation buffers |
| `data_meta` | the data recipe — see below |
| `train_meta` | training config, epochs run, best epoch, per-epoch history |
| `metrics` | the test metrics the notebook reported |
| `created_utc` | export timestamp |

`data_meta` is the important one. It records the source file, bar count, first
timestamp, hours of tape, window, horizon, **barrier width**, **label mode**,
**channel layout**, split fractions, strides and split sizes — everything
`sequence_matrix.dataset_kwargs_from` needs to rebuild the *identical* window
population from your local CSV. `eval_patchtst.py` compares what it rebuilt
against what the checkpoint recorded and **refuses to score on a mismatch**, so a
local number that disagrees with the notebook is reported as an error rather than
published as a result.

Inspect one without running an evaluation:

```python
from patchtst_model import load_checkpoint, describe_checkpoint

model, payload = load_checkpoint("../weights/patchtst_BTCUSDT_2026-05_full.pt")
print(describe_checkpoint(payload))
print(payload["data_meta"]["label_mode"], payload["data_meta"]["channels"])
print(payload["metrics"])
```

## Checkpoints from before the triple-barrier switch

Older checkpoints carry no `label_mode` and list six channels. They still load and
still reproduce their published numbers: `dataset_kwargs_from` falls back to
`label_mode="fee_threshold"` when the key is absent, and rebuilds the six-channel
matrix from the recorded `channels` list. That is deliberate — scoring old weights
against a label they never saw would be worse than not scoring them at all.

`describe_checkpoint` prints the label mode, so it is always visible which target a
given file was trained against.

## Size and version control

Parameter count grows with the lookback, because the patch embedding is a
`Linear(patch_len, d_model)` and the patch length scales with the window: **209k**
at the short scale, **233k** at the mid, **503k** at the long. Nothing else in the
network changes size. That puts a checkpoint between **0.9 MB and 2.1 MB** —
small enough to commit if you want a result to stay reproducible. `.gitignore`
excludes `weights/*.pt` by default; force-add the ones worth keeping:

```bash
git add -f weights/patchtst_BTCUSDT_2026-05_full.pt
```
