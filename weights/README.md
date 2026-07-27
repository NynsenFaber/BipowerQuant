# Trained weights

Drop PatchTST checkpoints exported by
[`notebooks/train_patchtst_colab.ipynb`](../notebooks/train_patchtst_colab.ipynb) in
this directory, then score them with:

```bash
cd python
python eval_patchtst.py --weights ../weights/patchtst_BTCUSDT_2026-05_8h.pt
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
| `data_meta` | the data recipe: source file, bar count, hours of tape, window, horizon, fee threshold, split fractions, split sizes |
| `train_meta` | training config, epochs run, best epoch, per-epoch history |
| `metrics` | the test metrics the notebook reported |
| `created_utc` | export timestamp |

`data_meta` is what lets `eval_patchtst.py` rebuild the *identical* window population
from your local CSV and refuse to score if it cannot — so a local number that
disagrees with the notebook is reported as a mismatch rather than published as a
result.

Inspect one without running an evaluation:

```python
from patchtst_model import load_checkpoint, describe_checkpoint
model, payload = load_checkpoint("../weights/patchtst_BTCUSDT_2026-05_8h.pt")
print(describe_checkpoint(payload))
print(payload["metrics"])
```

## Size and version control

The default configuration is ~118k parameters, so a checkpoint is roughly **0.5 MB** —
small enough to commit if you want a result to stay reproducible. `.gitignore`
excludes `weights/*.pt` by default; force-add the ones worth keeping:

```bash
git add -f weights/patchtst_BTCUSDT_2026-05_8h.pt
```
