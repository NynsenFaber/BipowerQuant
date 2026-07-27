# BipowerQuant: High-Frequency Jump-Diffusion Predictor

This repository contains a high-performance predictive engine designed to forecast ultra-short-term cryptocurrency price direction (1-minute horizons) using tick-level data. It bridges a mathematically rigorous stochastic framework with a low-latency engineering pipeline.

By isolating the continuous volatility of an asset from sudden market shocks (jumps) via Bipower Variation, the model feeds purified stochastic signals—alongside microstructural Order Flow Imbalance (OFI)—into an Extreme Gradient Boosting (XGBoost) classifier.

Three models are benchmarked against the same profit-threshold target: a **Logistic Regression** on OFI alone, **XGBoost** on the seven-feature stochastic matrix, and **PatchTST** — a channel-independent patch Transformer that reads the raw 300-bar lookback instead of its aggregates (see [§4](#4-the-patchtst-sequence-baseline) and [§8](#8-training-and-running-patchtst)).



## 1. Mathematical Framework

The core assumption of this engine is that the high-frequency log-price $p_t$ of an asset does not follow a simple random walk, but rather an Ito Jump-Diffusion process. This allows us to model both the standard market noise and the sudden structural breaks caused by institutional block orders or macroeconomic news.

The process is defined as:

$$dp_t = \mu_t dt + \sigma_t dW_t + J_t dN_t$$

Where:
*   $\mu_t dt$: The continuous drift component.
*   $\sigma_t dW_t$: The continuous Brownian motion (diffusion) component, representing standard market volatility.
*   $J_t dN_t$: The discrete jump component driven by a Poisson process, representing sudden market shocks.

### Discretizing the Variance
To utilize this continuous-time process in a discrete Machine Learning model, the C++ mathematical engine calculates specific variance metrics over a rolling window containing $M$ high-frequency returns ($r_{t,i}$).

**1. Realized Variance (Total Risk):**
Realized Variance captures the total quadratic variation of the asset, absorbing both the smooth diffusion and the discrete jumps.

$$RV_t = \sum_{i=1}^{M} r_{t,i}^2$$

**2. Bipower Variation (Continuous Risk):**
Bipower Variation is a robust estimator that mathematically filters out the jumps. By multiplying adjacent absolute returns, the probability of two massive jumps occurring in consecutive microscopic ticks approaches zero, isolating the continuous variance ($\sigma_t$).

$$BPV_t = \frac{\pi}{2} \sum_{i=2}^{M} |r_{t,i}| |r_{t,i-1}|$$

**3. Jump Component (Discrete Shocks):**
By subtracting the continuous risk from the total risk, we isolate the exact magnitude of the market shocks occurring within the window.

$$J_t = \max(RV_t - BPV_t, 0)$$

## 2. Microstructure Features
Alongside the stochastic variance metrics, the engine leverages pure market microstructure.

**Order Flow Imbalance (OFI):**
Using the tick-level `is_buyer_maker` flag, OFI quantifies the immediate buying versus selling pressure at the top of the order book.
*   If `is_buyer_maker == False`: A taker bought from a maker (Aggressive Buy).
*   If `is_buyer_maker == True`: A taker sold to a maker (Aggressive Sell).

OFI tracks the net difference in these aggressive volumes over the lookback window, serving as the foundational momentum benchmark.

### Contextual Features
The raw stochastic metrics are scale-dependent and direction-blind: $BPV_t$ and $J_t$ are strictly non-negative magnitudes, and a given $OFI$ reading means something very different in a quiet window than in a violent one. Three derived features restore that missing context.

**1. Lookback Return:**
The log return across the same 5-minute window the C++ engine consumes, giving the model the trend the shock occurred in.

$$r_{t-5\text{min}} = \log(p_t) - \log(p_{t-300})$$

**2. Volatility-Adjusted OFI:**
Normalising order flow by the continuous volatility of the window puts it on a per-unit-of-risk scale, so that 10 BTC of net aggression in a calm window is not treated identically to 10 BTC during a liquidation cascade. A small $\epsilon = 10^{-8}$ guards against division by zero on completely flat windows.

$$OFI^{adj}_t = \frac{OFI_t}{\sqrt{BPV_t} + \epsilon}$$

**3. Signed Jumps:**
Since $J_t = \max(RV_t - BPV_t, 0)$ is magnitude-only by construction, signing it with the lookback direction distinguishes a shock into an up-trend from a shock into a down-trend.

$$J^{signed}_t = J_t \times \text{sign}(r_{t-5\text{min}})$$

## 3. Sliding Window Architecture

The pipeline processes out-of-core tick data (e.g., Binance BTC/USDT) using a highly dense, overlapping rolling window approach to generate training data.

*   **Lookback Window ($X$ Features):** A strict 5-minute rolling window where the C++ engine computes $BPV_t$, $J_t$, and $OFI$, plus the three contextual features derived from them.
*   **Prediction Horizon ($y$ Target):** The subsequent 1-minute window following the lookback.
*   **Stride Engine:** The window advances in 1-second intervals, meaning one hour of market data yields 3,600 highly contextualized training vectors.

### Profit-Threshold Targets
A naive `1 = price closed higher` label rewards the model for predicting moves too small to trade. At a round-trip taker cost of roughly 5 basis points, a +0.01% "correct" prediction is a losing trade. The target is therefore defined against a `FEE_THRESHOLD` of `0.0005`:

$$y_t = \begin{cases} 1 & \text{if } \frac{p_{t+60}}{p_t} - 1 > 0.0005 \\ 0 & \text{otherwise} \end{cases}$$

Every down move, flat move, and unprofitable up move collapses into class `0`. This is a deliberately harsh relabelling: on the May 2026 BTC/USDT sample it drops the positive class from roughly 50% to **6.8%**, converting a balanced problem into a rare-event detection problem. `scale_pos_weight` is consequently recomputed from the actual training split at every run rather than assumed to be near 1.0.


## 4. The PatchTST Sequence Baseline

The logistic and XGBoost models both see the same thing: **seven numbers per window**. $RV_t$, $BPV_t$, $OFI_t$ and the contextual features are all *sums* over the 5-minute lookback, and a sum is order-blind. A window where a 40 BTC sell wall hit in the first ten seconds and one where it hit in the last ten produce identical feature vectors, even though only the second is still moving the book when the prediction is made.

PatchTST ([Nie et al., ICLR 2023](https://arxiv.org/abs/2211.14730)) is the natural way to give the model back that ordering without abandoning the jump-diffusion framing. It is added here as a third baseline solving **the identical classification problem** — same 5-minute lookback, same 1-minute horizon, same 5 bp fee threshold, same target definition — differing only in what it reads.

### 4.1 From scalars to channels

Instead of the aggregated 7-vector, PatchTST consumes the raw 300 one-second bars of the lookback as $M = 6$ parallel univariate channels:

| Channel | Definition | Relation to the C++ engine |
| :--- | :--- | :--- |
| `log_return` | $r_t = \log p_t - \log p_{t-1}$ | the signed increments the aggregates discard |
| `realized_var` | $r_t^2$ | $\sum_{\text{window}} = RV_t$ |
| `bipower` | $\frac{\pi}{2}\lvert r_t\rvert\lvert r_{t-1}\rvert$ | $\sum_{\text{window}} = BPV_t$ |
| `ofi` | signed traded volume | $\sum_{\text{window}} = OFI_t$ |
| `log_volume` | $\log(1 + q_t)$ | — |
| `log_trades` | $\log(1 + n_t)$ | — |

Channels 2–4 are exactly the per-bar terms `math_engine.hpp` accumulates — verified numerically to $\sim10^{-15}$ against `bipower_core`. **The model is therefore not given different information; it is given the same information un-summed.** Attention across patches can learn a time-weighted, non-linear alternative to $\sum r_{t,i}^2$, and it can recover the plain sum exactly if that is genuinely optimal.

### 4.2 Patching

A single second carries no more meaning than a single character does in a sentence, and 300 tokens of self-attention is $O(300^2)$ per channel. Patching fixes both. The lookback is cut into overlapping sub-series of length $P = 16$ bars with stride $S = 8$:

$$N = \left\lfloor \frac{L - P}{S} \right\rfloor + 2 = \left\lfloor \frac{300 - 16}{8} \right\rfloor + 2 = 37$$

(the $+2$ accounts for the paper's end-padding, which repeats the final value $S$ times). Each token is now a 16-second sub-series with local semantics, and the attention map shrinks from $300^2$ to $37^2$ — a **66×** reduction in attention cost at unchanged look-back length.

### 4.3 Channel independence

All six channels share one set of Transformer weights and are pushed through the encoder as independent univariate series, folding $M$ into the batch dimension as $(B \cdot M, N, D)$. Per the paper's ablation (Table 7) this beats channel-mixing consistently, for reasons that apply with unusual force here: mixing lets a noisy channel project its noise onto every other channel in the embedding space, and order-flow data is mostly noise. Cross-channel information is not lost — it is recombined in the head.

### 4.4 Instance normalisation, and putting the scale back

Each window/channel is standardised to zero mean and unit variance before patching. This is what makes the model survive the volatility regime shifts that a month of crypto tape is made of — a quiet Tuesday and a liquidation cascade become comparable inputs.

But it also deletes precisely what this problem cares about: *how* volatile and *how* imbalanced the window was. So the discarded statistics are standardised and concatenated back into the classification head:

$$\text{aux}_t = \left[\ \mu_c(t),\ \log(\sigma_c(t) + \epsilon)\ \right]_{c=1..M} \in \mathbb{R}^{12}$$

These 12 numbers are close to the tabular feature set by construction — $\mu$ of `log_return` is the 5-minute return divided by 300, and $\mu$ of `realized_var` is exactly $RV_t / 300$ — so **nothing the trees had access to is withheld from the Transformer.** Set `USE_SCALE_FEATURES = False` in the notebook to ablate it and see how much of any lift is the sequence and how much is the aggregates.

### 4.5 Classification head and loss

The paper's `Flatten + Linear → T future values` becomes `Flatten + Linear → 1 logit`, trained with

$$\mathcal{L} = \text{BCEWithLogits}\left(\hat{y}, y;\ w^+ = \frac{n^-}{n^+}\right)$$

where $w^+$ is computed from the actual training split — the direct analogue of XGBoost's `scale_pos_weight`, and equally necessary given a ~7% positive class.

Default size: 3 encoder layers, $D = 64$, $H = 4$ heads, $F = 128$, dropout 0.2 — **118,093 parameters**, roughly 0.5 MB on disk. Deliberately small: with an ROC-AUC hovering near 0.5 across every run so far, the failure mode to fear is a model with enough capacity to memorise 22k autocorrelated windows.

### 4.6 Computational design

Consecutive windows share 299 of their 300 bars, so materialising the training tensor is the obvious trap: $n_{\text{windows}} \times 6 \times 300$ float32s is **19.3 GB for a month** of data. Nothing in this pipeline ever builds it.

| Concern | Approach |
| :--- | :--- |
| **Window memory** | the $(6, n_{\text{bars}})$ channel matrix lives on the device *once* — **64 MB** for a month, a **300×** reduction — and every batch is one gather, `channels[:, starts[:, None] + arange(300)]` |
| **Data loading** | no `DataLoader`, no workers, no per-batch host→device copies — the whole series is already resident |
| **Attention** | `F.scaled_dot_product_attention`, so PyTorch selects the fused Flash / memory-efficient kernel |
| **Precision** | automatic mixed precision — bf16 where supported, fp16 + loss scaling on T4 |
| **Ingestion** | one streaming Polars pass folds 72M ticks into 2.7M second-bars, cached as a ~30 MB `.npz` so no later run re-reads the 5 GB CSV |
| **Autocorrelation** | `TRAIN_STRIDE` thins training windows (validation and test always keep every window) |

### 4.7 Two deliberate differences from the XGBoost pipeline

Both are documented rather than silently applied, because they affect how the numbers compare:

1. **Complete 1-second grid.** `ml_matrix.py` uses `group_by_dynamic`, which emits bars only for seconds that contain trades — so its "60-bar" horizon is not always 60 seconds. `sequence_matrix.py` reindexes onto a gap-free grid with forward-filled prices and zero volume. On BTC/USDT this matters more than expected: **roughly one second in five contains no trade at all.** A patch model needs uniform time spacing to be meaningful, so this is the Phase 5 grid fix, applied here first.
2. **A purged 70/10/20 split.** The trees used a plain chronological 80/20. A neural network needs a validation split for early stopping, so 10% is carved out of the *training* portion — the test split is still the final 20% of the period. Additionally, $L + H - 1 = 359$ windows are dropped before each boundary: without that purge the last training windows are labelled by price moves that fall *inside* the next split's lookback. (Combined with difference 1, the test windows are not bar-for-bar the ones XGBoost saw, so treat the comparison as close rather than exact until `ml_matrix.py` is moved onto the same grid.)

## 5. Pipeline & Technology Stack

To handle billions of ticks without memory overflows, the project implements a polyglot out-of-core architecture.

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Data Ingestion** | Python (Polars) | Lazy evaluation of multi-gigabyte CSV/ZIP files to stream data chunks in a memory-safe manner. |
| **Math Engine** | C++20 | Zero-overhead arrays and loop optimizations to calculate rolling stochastic metrics. |
| **Interoperability** | `pybind11` (CMake) | Compiles the C++ engine into a native Python module (`bipower_core`) for seamless pipeline integration. |
| **Machine Learning** | XGBoost | Iterative tree boosting (`xgb.train`) trained sequentially on streaming data chunks. |
| **Sequence Model** | PyTorch | PatchTST — a channel-independent patch Transformer reading the raw 300-bar lookback. Trained on a Colab GPU, scored locally from an exported checkpoint. |
| **Baseline Benchmark** | Scikit-Learn | A standard Logistic Regression trained exclusively on OFI to prove the alpha generated by the jump-diffusion metrics. |

## 6. Data Source

To replicate the training environment, you will need high-frequency tick data. The pipeline is built to natively process Binance public trade data. 

You can download the exact dataset used in this project directly from the Binance Data Archive:
* **Archive Link:** [Binance Vision: BTC/USDT Spot Trades](https://data.binance.vision/?prefix=data/spot/monthly/trades/BTCUSDT/)
* **Target File:** Download `BTCUSDT-trades-2026-05.zip`

Once downloaded, place the file (or the extracted CSV) into your project directory and ensure the `FILE_PATH` variable in `python/data_feeder.py` points to it.

*(The PatchTST notebook downloads this archive directly inside Colab — see §8 — so no upload is needed to train.)*

## 7. Repository Structure

The codebase is strictly divided between low-level performance execution and high-level pipeline orchestration:

### `src/` (High-Performance C++ Core)
*   **`math_engine.hpp`**: The core C++20 sliding window implementation. Calculates Realized Variance, Bipower Variation, Jumps, and OFI with pre-allocated memory and zero-overhead loops.
*   **`bindings.cpp`**: The `pybind11` wrapper that grants the C++ engine direct memory access to Python NumPy arrays, bypassing costly data serialization.

### `python/` (Data & Machine Learning Pipeline)
*   **`data_feeder.py`**: Utilizes Polars to lazily evaluate and stream out-of-core tick data in manageable hourly chunks.
*   **`ml_matrix.py`**: The feature engineering bridge. Resamples irregular ticks into uniform 1-second bars, queries the C++ math engine, derives the contextual features (lookback return, volatility-adjusted OFI, signed jumps), applies the `FEE_THRESHOLD` target definition, and constructs the final aligned 7-column training matrix.
*   **`train_baseline.py`**: Trains a Scikit-Learn Logistic Regression model exclusively on the OFI feature to establish a foundational directional benchmark.
*   **`train_xgboost.py`**: Trains the XGBoost tree classifier on the full stochastic jump-diffusion matrix, exporting feature importances to compare against the linear baseline. 

### `python/` (PatchTST Sequence Pipeline)
*   **`sequence_matrix.py`**: The sequence counterpart to `ml_matrix.py`. Streams ticks into a gap-free 1-second bar grid, derives the six per-bar channels, applies the identical fee-threshold target, and produces purged chronological splits. Depends only on Polars and NumPy — **not** on `bipower_core` — which is what lets the exact same code run inside a Colab runtime where the C++ extension was never compiled.
*   **`patchtst_model.py`**: The network (patching, channel-independent encoder, instance normalisation, classification head), the `WindowBatcher` that cuts overlapping windows out of a device-resident channel matrix, checkpoint I/O, and the shared metric helpers.
*   **`patchtst_train.py`**: The training loop — AdamW with cosine schedule and warmup, mixed precision, gradient clipping, ROC-AUC early stopping with best-weight restore. Deliberately kept out of the notebook so the code that produced a checkpoint is version controlled beside the model.
*   **`eval_patchtst.py`**: Loads an exported checkpoint, rebuilds the identical window population from the local CSV, scores the held-out split, and appends to `training_logs.txt`.
*   **`benchmark_inference.py`**: Fits the Logistic Regression and XGBoost baselines on *exactly* the windows PatchTST is scored on, then times all three at inference — single-window latency and batched throughput — with every CPU model pinned to one thread. Also reports each model's ROC-AUC on those identical windows, which is the only strictly apples-to-apples accuracy comparison in this repository.

### `notebooks/` & `weights/`
*   **`notebooks/train_patchtst_colab.ipynb`**: The GPU training driver — see §8.
*   **`weights/`**: Where downloaded checkpoints go. See [`weights/README.md`](weights/README.md) for the checkpoint layout.

*(Note: Training scripts automatically append timestamped classification metrics—Accuracy, F1-Score, and ROC-AUC—to a local `training_logs.txt` file for historical tracking).*

## 8. Training and Running PatchTST

Training happens on a free Colab GPU; evaluation happens locally against the exported checkpoint. The checkpoint carries its own data recipe, so the local run reproduces the notebook's window population exactly — or refuses and tells you why.

### Step 1 — Push the branch

The notebook pulls the model code from the repository rather than carrying a copy, so whatever branch holds `python/patchtst_model.py` must exist on the remote:

```bash
git push -u origin add-patchTST
```

*(No remote? Skip it — the notebook's markdown documents the fallback: upload `sequence_matrix.py`, `patchtst_model.py`, `patchtst_train.py` and `data_feeder.py` through the Colab file browser instead.)*

### Step 2 — Train in Colab

Open [`notebooks/train_patchtst_colab.ipynb`](notebooks/train_patchtst_colab.ipynb) in Google Colab, set **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**. It is configured for the full month out of the box and pauses exactly once — at the start, for Google Drive authorisation. (Set `USE_DRIVE = False` to remove even that, at the cost of losing the cache and weights on a disconnect.)

| Section | What it does |
| :--- | :--- |
| **0** | Confirms a GPU is attached and enables TF32 |
| **1** | Clones this repository, sets `BRANCH`, installs Polars |
| **2** | Downloads `BTCUSDT-trades-2026-05.zip` straight from the Binance archive (~1.5 GB) and mounts Drive |
| **3** | **The configuration cell** — every knob in one place |
| **4** | Streams the CSV into 1-second bars (~28 s, ~5 GB peak), caching them to a ~28 MB `.npz` |
| **5** | Builds batchers and the model; prints parameter count and resident memory |
| **6** | **Throughput probe** — times real training steps and estimates the epoch cost *before* you commit a session to it |
| **7** | Trains, with early stopping on validation ROC-AUC |
| **8** | Plots training loss and validation ROC-AUC |
| **9** | Scores the held-out test split, with a threshold sweep |
| **10** | **Inference benchmark** — latency and accuracy for all three models on identical windows |
| **11** | Saves the checkpoint and triggers a browser download |

**On sample size.** `HOURS = 8.0` leaves a test split of 95 minutes — about **16 independent observations** once you account for each window spanning 360 s. That is far too few to separate an edge from noise, which the block bootstrap below makes concrete. `HOURS = None` gives a 6.2-day test split, roughly **1,488 independent episodes**. Use the 8-hour setting to check the pipeline; use the month to draw conclusions.

**On strides.** At stride 1 the month yields ~1.87M training windows that share 299 of every 300 bars. `TRAIN_STRIDE = 6` keeps essentially the same information at a sixth of the epoch cost, and `VAL_STRIDE = 6` does the same for early stopping. Test is always evaluated at stride 1.

### Step 3 — Bring the weights home

Cell 9 calls `files.download(...)`, which drops a `.pt` (~0.5 MB) into your browser's download folder, and also copies it to Drive if you mounted it. Move it into `weights/`:

```bash
mv ~/Downloads/patchtst_BTCUSDT_2026-05_8h.pt weights/
```

### Step 4 — Run the experiment locally

```bash
uv sync                                                        # first time: pulls torch
cd python
python eval_patchtst.py --weights ../weights/patchtst_BTCUSDT_2026-05_8h.pt
```

The script prints the metrics table plus a threshold sweep, and appends a row to `python/training_logs.txt` in the same format as the other baselines.

| Flag | Purpose |
| :--- | :--- |
| `--device mps` | Apple Silicon GPU (verified to give identical results to CPU) |
| `--bars-cache ../data/bars.npz` | build the bar series once, reuse it on every later run |
| `--split val` | score the validation split instead of the test split |
| `--threshold 0.7` | change the probability cut-off for the hard label |
| `--save-predictions probs.npy` | dump per-window probabilities for further analysis |
| `--allow-mismatch` | score anyway when the local data does not match the checkpoint |

**On the mismatch guard.** `eval_patchtst.py` compares the locally rebuilt bar count, first timestamp and split sizes against what the checkpoint recorded. If they differ — a different month, a truncated CSV, an edited `sequence_matrix.py` — it stops rather than reporting a number against a different window population. That is the failure mode most likely to produce a fake result, so it is an error by default rather than a warning.

### Step 5 — Inference cost

The notebook runs this as its second-to-last section; the same benchmark runs locally:

```bash
cd python
python benchmark_inference.py --weights ../weights/<the-file>.pt \
                              --bars-cache ../data/bars_full.npz
```

It reports two numbers per model, because they rank the models differently:

*   **Single-window latency** — the trading number. What a signal costs when a window closes and you must decide before the next tick.
*   **Batched throughput** — the research number. What a sweep or a backtest over a month of windows costs.

Every CPU model is pinned to **one thread**, so the comparison measures models rather than core counts. The tabular models additionally pay a feature-preparation step (300 bars → 7 scalars) that PatchTST does not — it reads the raw window and normalises inside the forward pass — so that cost is reported separately and must be added to their latencies.

Because the baselines have to be fitted in order to be timed, their test ROC-AUC comes out of the same run — measured on windows *identical* to PatchTST's, which the historical table further down is not.

> **macOS note.** torch and xgboost ship separate OpenMP runtimes, and mixing them in one process either segfaults or deadlocks. `benchmark_inference.py` imports xgboost before torch and pins torch to one thread, which is the combination measured to work. If you import torch first (a notebook, a REPL), it raises with instructions instead of dying mid-run. Linux, including Colab, is unaffected.

## Current Training Results

### Phase 4: Contextual Features + Fee Threshold — July 26, 2026

Trained on the first 8 hours of the May 2026 BTC/USDT sample (22,866 windows, 7 features, chronological 80/20 split).

| Metric | XGBoost (Contextual Matrix) | Reference |
| :--- | :--- | :--- |
| **Accuracy** | 0.7711 | 0.9062 by always predicting `0` |
| **Precision** | 0.0934 | 0.0938 test-set base rate |
| **Recall** | 0.1655 | — |
| **F1-Score** | 0.1194 | — |
| **ROC-AUC** | 0.4980 | 0.5000 is a coin flip |

**Read this table carefully.** The headline accuracy jumped from 0.4921 to 0.7711, but that number is an artifact of the relabelling, not a discovery: only 9.4% of the test windows are positive, so a model that predicts `0` unconditionally scores 0.9062. The metrics that actually matter both say the same thing:

*   **Precision (0.0934) sits marginally *below* the base rate (0.0938).** When the model flags a profitable minute, it is right slightly less often than picking a window at random. There is no lift.
*   **ROC-AUC of 0.4980 is statistically indistinguishable from 0.5.** The model has no ability to rank profitable windows above unprofitable ones at any threshold.

The contextual features did not rescue the signal. Compared to the July 2 run, feature importance spread out — `Order Flow Imbalance` remains the single most-used split (0.2338) and the three new features absorb roughly 40% of the total importance between them — but importance measures how often the trees *split* on a feature, not whether those splits generalise. Wide, evenly-distributed importance across seven features with an AUC of 0.50 is the signature of a model fitting noise.

### XGBoost Feature Importances
*   **Order Flow Imbalance:** 0.2338
*   **Bipower Variation:** 0.1605
*   **Jumps:** 0.1369
*   **Vol-Adjusted OFI:** 0.1355
*   **Signed Jumps:** 0.1334
*   **5m Return:** 0.1274
*   **Realized Variance:** 0.0725

### Phase 5: PatchTST — 8-hour pilot run, July 27, 2026

A pilot run on the same 8 hours the Phase 4 XGBoost run used (T4 GPU, 19,550 training windows, early-stopped at epoch 4 of 30).

| Metric | PatchTST | Reference |
| :--- | :--- | :--- |
| **Accuracy** | 0.6999 | 0.9230 by always predicting `0` |
| **Precision** | 0.0801 | 0.0770 test-set base rate |
| **Recall** | 0.2763 | — |
| **F1-Score** | 0.1242 | — |
| **ROC-AUC** | 0.5459 | 0.5000 is a coin flip |

Nominally the best ROC-AUC the project has produced, and inside the 0.52–0.54 band normally called an edge. **It is not one**, for three reasons that a single headline number hides:

1.  **The sample cannot support the claim.** The test split is 1.58 hours; each observation spans 360 s, so there are ~16 independent episodes in it. A moving-block bootstrap over that split gives a 95% CI of **[0.4495, 0.6522]**, with a **23.4%** probability the true value is at or below 0.5.
2.  **A one-line feature beats it.** On the identical windows, a plain rolling sum of $r_t^2$ scores **0.5876** and $\lvert r_{5m}\rvert$ scores 0.5848 — both above PatchTST's 0.5459. The model's probabilities are essentially uncorrelated with realized variance ($+0.018$), so it did not find a subtler version of that signal; it found a weaker, different one.
3.  **The ranking is inverted where it matters.** Precision in the top 1% of predictions is 0.0000, top 5% is 0.0246, top 10% is 0.0511 — all far *below* the 0.0770 base rate, only reaching it around the top quartile. The model's most confident calls are its worst, which makes the ranking unusable for trading even if the AUC were real.

The validation curve shows the expected shape: training loss falling monotonically (1.39 → 0.67) while validation ROC-AUC peaks at epoch 4 (0.6645) and decays. With 118k parameters against 19,550 windows that share 299 of every 300 bars, it starts memorising almost immediately. The 0.12 gap between validation (0.6645) and test (0.5459) is regime drift: positive rates across the three splits are 5.14% / 3.14% / 7.70%.

**The full-month run is the one that can actually answer the question** — a 6.2-day test split holds ~1,488 independent episodes rather than 16. The notebook is configured for it; this section should be replaced with those numbers.

**When reading them, the benchmark to beat is not 0.5 — it is realized variance.** If PatchTST cannot out-rank a single rolling sum out of sample, the sequence is buying nothing, whatever the AUC says.

### History

| Date | Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Jul 2 | Logistic Regression (OFI only, directional target) | 0.5335 | 0.5562 | 0.5343 |
| Jul 2 | XGBoost (4-feature matrix, directional target) | 0.4921 | 0.3432 | 0.4872 |
| Jul 26 | XGBoost (7-feature matrix, fee threshold) | 0.7711 | 0.1194 | 0.4980 |
| Jul 27 | PatchTST (6-channel sequence, 8h pilot) | 0.6999 | 0.1242 | 0.5459 |
| — | PatchTST (6-channel sequence, full month) | *pending* | *pending* | *pending* |

Note that the July 2 and July 26 rows are **not directly comparable** — they are scored against different target definitions on different class balances. The July 2 logistic baseline remains the only run in this project to have posted an ROC-AUC meaningfully above 0.5, and on a single 4-hour chunk that result is well within the range of sampling noise.

### Honest Assessment

Across four runs, no configuration has produced a defensible statistical edge. The jump-diffusion features have not yet demonstrated alpha over the OFI baseline, and the fee-threshold target — while methodologically correct, since it stops rewarding untradeable moves — has so far only made the absence of signal easier to see.

Plausible explanations, roughly in order of how much they are worth chasing:

1.  **Training volume.** 22,866 windows drawn from a single 8-hour block is a thin sample for a rare-event problem, and overlapping 1-second-stride windows are heavily autocorrelated, so the *effective* sample size is far smaller than the row count suggests. The `xgb.train` incremental loop across the full month is the obvious next step.
2.  **Regime specificity.** All results come from one contiguous block of May 2026. Walk-forward validation across multiple days would separate a genuinely absent signal from one that exists only in certain regimes.
3.  **Bar construction.** `group_by_dynamic` emits bars only for seconds that contain trades, so a quiet stretch silently compresses the timeline and the "60-bar" horizon is not always a literal 60 seconds. Reindexing onto a complete 1-second grid with forward-filled prices would make the horizon exact. *(Done for the PatchTST path in `sequence_matrix.py`, which measured the damage: about one second in five carries no trade. Still outstanding for `ml_matrix.py`, so the tabular runs above remain affected.)*
4.  **Horizon and threshold.** A 5-basis-point move within 60 seconds is a demanding bar to clear. Sweeping the horizon and threshold jointly would show whether any tradeable combination carries signal at all.
5.  **Aggregation.** $RV$, $BPV$ and $OFI$ are sums, and a sum cannot distinguish a shock at the start of the window from one at the end. If the signal lives in the *shape* of the five minutes rather than its totals, no amount of tuning the trees will find it. This is the hypothesis PatchTST exists to test — and, being a genuinely different hypothesis rather than a bigger model on the same features, it is the one worth testing next.

---

## Appendix: Classification Metrics Defined

When evaluating the predictive performance of the models, we rely on three standard machine learning metrics to ensure robustness beyond simple win-rates.

### 1. Accuracy
The ratio of correctly predicted observations (both up-ticks and down-ticks) to the total observations. While intuitive, it can be misleading if the market regime is heavily skewed in one direction.

$$ \text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} $$

*(Where TP = True Positives, TN = True Negatives, FP = False Positives, FN = False Negatives)*

### 2. F1-Score
The harmonic mean of Precision and Recall. It is a strictly stricter metric than accuracy and is highly valuable when dealing with asymmetrical market conditions or imbalanced target distributions. 

$$ \text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} $$

*   **Precision:** Out of all the times the model predicted the price would go up, how often was it right?

$$ \text{Precision} = \frac{TP}{TP + FP} $$

*   **Recall:** Out of all the actual times the price went up, how many did the model manage to catch?

$$ \text{Recall} = \frac{TP}{TP + FN} $$

### 3. ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
This measures the model's ability to distinguish between classes across all possible classification thresholds, rather than just a fixed 50% cutoff. 
*   An **AUC of 0.5** means the model has no class separation capacity whatsoever (a coin flip).
*   An **AUC of 1.0** means the model perfectly distinguishes between upward and downward price movements.
In high-frequency quantitative finance, an ROC-AUC consistently above 0.52 to 0.54 is generally considered a strong statistical edge.