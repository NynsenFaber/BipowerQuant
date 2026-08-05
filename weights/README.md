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

The current configuration is ~209k parameters, so a checkpoint is roughly **0.9 MB** —
small enough to commit if you want a result to stay reproducible. `.gitignore`
excludes `weights/*.pt` by default; force-add the ones worth keeping:

```bash
git add -f weights/patchtst_BTCUSDT_2026-05_full.pt
```
