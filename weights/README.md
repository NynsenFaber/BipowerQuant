# Trained weights

Nothing here is committed by default (`.gitignore` excludes `weights/*.pt` and
`weights/*.npz`). This directory is where the artefacts of a training run land.

## What a run produces

[`notebooks/train_patchtst_colab.ipynb`](../notebooks/train_patchtst_colab.ipynb)
writes two kinds of file:

| File | Contents | Used by |
| :--- | :--- | :--- |
| `patchtst_probs.npz` | one probability array per walk-forward fold | `walkforward.py --patchtst-probs` |
| `patchtst_folds.json` | per-fold metrics, epochs run, the stride the budget picked | reading what the run did |
| `patchtst_<fold>.pt` | the trained network for that fold | inspection, further training |

The backtest only needs the probabilities. Training happens on a GPU and scoring
happens locally, so the notebook exports the *predictions* rather than asking a
laptop to re-run inference over millions of windows:

```bash
cd python
python walkforward.py --bars ../data/bars_6m.npz \
                      --patchtst-probs ../weights/patchtst_probs.npz \
                      --out ../data/wf_final.json
```

`walkforward.py` matches each array to its fold by name and by length, and skips
— loudly — any fold whose probabilities do not line up with the windows it holds
out. A silent mismatch would score a model against a population it never saw.

## What is inside a checkpoint

`.pt` files are plain `torch.save` dicts with no pickled classes, so they load
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
timestamp, window, horizon, **barrier width**, **label mode**, **channel layout**,
split fractions and strides — everything `sequence_matrix.dataset_kwargs_from`
needs to rebuild the *identical* window population from a local bar cache. A
checkpoint that did not record its target cannot be scored safely, and
`describe_checkpoint` prints `label: unrecorded` when that is the case.

Inspect one without running an evaluation:

```python
from patchtst_model import describe_checkpoint, load_checkpoint

model, payload = load_checkpoint("../weights/patchtst_jan-may_-_june.pt")
print(describe_checkpoint(payload))
print(payload["metrics"])
```

## Committing one

The current configuration is ~209k parameters, so a checkpoint is roughly
**0.9 MB** — small enough to commit if you want a result to stay reproducible:

```bash
git add -f weights/patchtst_probs.npz
```
