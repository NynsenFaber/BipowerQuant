# BipowerQuant: High-Frequency Jump-Diffusion Predictor

This repository contains a high-performance predictive engine designed to forecast ultra-short-term cryptocurrency price direction (1-minute horizons) using tick-level data. It bridges a mathematically rigorous stochastic framework with a low-latency engineering pipeline.

By isolating the continuous volatility of an asset from sudden market shocks (jumps) via Bipower Variation, the model feeds purified stochastic signals—alongside microstructural Order Flow Imbalance (OFI)—into an Extreme Gradient Boosting (XGBoost) classifier.



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


## 4. Pipeline & Technology Stack

To handle billions of ticks without memory overflows, the project implements a polyglot out-of-core architecture.

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Data Ingestion** | Python (Polars) | Lazy evaluation of multi-gigabyte CSV/ZIP files to stream data chunks in a memory-safe manner. |
| **Math Engine** | C++20 | Zero-overhead arrays and loop optimizations to calculate rolling stochastic metrics. |
| **Interoperability** | `pybind11` (CMake) | Compiles the C++ engine into a native Python module (`bipower_core`) for seamless pipeline integration. |
| **Machine Learning** | XGBoost | Iterative tree boosting (`xgb.train`) trained sequentially on streaming data chunks. |
| **Baseline Benchmark** | Scikit-Learn | A standard Logistic Regression trained exclusively on OFI to prove the alpha generated by the jump-diffusion metrics. |

## 4. Data Source

To replicate the training environment, you will need high-frequency tick data. The pipeline is built to natively process Binance public trade data. 

You can download the exact dataset used in this project directly from the Binance Data Archive:
* **Archive Link:** [Binance Vision: BTC/USDT Spot Trades](https://data.binance.vision/?prefix=data/spot/monthly/trades/BTCUSDT/)
* **Target File:** Download `BTCUSDT-trades-2026-05.zip`

Once downloaded, place the file (or the extracted CSV) into your project directory and ensure the `FILE_PATH` variable in `python/data_feeder.py` points to it.

## 6. Repository Structure

The codebase is strictly divided between low-level performance execution and high-level pipeline orchestration:

### `src/` (High-Performance C++ Core)
*   **`math_engine.hpp`**: The core C++20 sliding window implementation. Calculates Realized Variance, Bipower Variation, Jumps, and OFI with pre-allocated memory and zero-overhead loops.
*   **`bindings.cpp`**: The `pybind11` wrapper that grants the C++ engine direct memory access to Python NumPy arrays, bypassing costly data serialization.

### `python/` (Data & Machine Learning Pipeline)
*   **`data_feeder.py`**: Utilizes Polars to lazily evaluate and stream out-of-core tick data in manageable hourly chunks.
*   **`ml_matrix.py`**: The feature engineering bridge. Resamples irregular ticks into uniform 1-second bars, queries the C++ math engine, derives the contextual features (lookback return, volatility-adjusted OFI, signed jumps), applies the `FEE_THRESHOLD` target definition, and constructs the final aligned 7-column training matrix.
*   **`train_baseline.py`**: Trains a Scikit-Learn Logistic Regression model exclusively on the OFI feature to establish a foundational directional benchmark.
*   **`train_xgboost.py`**: Trains the XGBoost tree classifier on the full stochastic jump-diffusion matrix, exporting feature importances to compare against the linear baseline. 

*(Note: Training scripts automatically append timestamped classification metrics—Accuracy, F1-Score, and ROC-AUC—to a local `training_logs.txt` file for historical tracking).*

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

### History

| Date | Model | Accuracy | F1-Score | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Jul 2 | Logistic Regression (OFI only, directional target) | 0.5335 | 0.5562 | 0.5343 |
| Jul 2 | XGBoost (4-feature matrix, directional target) | 0.4921 | 0.3432 | 0.4872 |
| Jul 26 | XGBoost (7-feature matrix, fee threshold) | 0.7711 | 0.1194 | 0.4980 |

Note that the July 2 and July 26 rows are **not directly comparable** — they are scored against different target definitions on different class balances. The July 2 logistic baseline remains the only run in this project to have posted an ROC-AUC meaningfully above 0.5, and on a single 4-hour chunk that result is well within the range of sampling noise.

### Honest Assessment

Across three runs, no configuration has produced a defensible statistical edge. The jump-diffusion features have not yet demonstrated alpha over the OFI baseline, and the fee-threshold target — while methodologically correct, since it stops rewarding untradeable moves — has so far only made the absence of signal easier to see.

Plausible explanations, roughly in order of how much they are worth chasing:

1.  **Training volume.** 22,866 windows drawn from a single 8-hour block is a thin sample for a rare-event problem, and overlapping 1-second-stride windows are heavily autocorrelated, so the *effective* sample size is far smaller than the row count suggests. The `xgb.train` incremental loop across the full month is the obvious next step.
2.  **Regime specificity.** All results come from one contiguous block of May 2026. Walk-forward validation across multiple days would separate a genuinely absent signal from one that exists only in certain regimes.
3.  **Bar construction.** `group_by_dynamic` emits bars only for seconds that contain trades, so a quiet stretch silently compresses the timeline and the "60-bar" horizon is not always a literal 60 seconds. Reindexing onto a complete 1-second grid with forward-filled prices would make the horizon exact.
4.  **Horizon and threshold.** A 5-basis-point move within 60 seconds is a demanding bar to clear. Sweeping the horizon and threshold jointly would show whether any tradeable combination carries signal at all.

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