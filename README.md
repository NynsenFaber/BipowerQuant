# BipowerQuant

[![CI](https://github.com/NynsenFaber/BipowerQuant/actions/workflows/ci.yml/badge.svg)](https://github.com/NynsenFaber/BipowerQuant/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/NynsenFaber/BipowerQuant/branch/main/graph/badge.svg)](https://codecov.io/gh/NynsenFaber/BipowerQuant)
[![Python 3.14](https://img.shields.io/badge/python-3.14-blue.svg)](https://www.python.org/downloads/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**Can you predict which way Bitcoin moves next?**

This repository is a careful attempt at that question on **six months of tick-level BTC/USDT
data** (742 million individual trades). It builds two very different kinds of model on
exactly the same data and compares: a **feature-based** approach, where
handcrafted statistics from stochastic calculus are fed to standard machine learning, and a
**deep learnin** approach, where a Transformer reads the raw price in sequence.

The short version of what it found:

* Predicting **whether** a large move is coming works well and reliably, scoring **0.71–0.76
  ROC-AUC** (a standard accuracy score where 0.50 is a coin flip and 1.0 is perfect) on three
  months the models never saw. This is the volatility question, and the hand-built features
  answer it almost perfectly.
* Predicting **which way** is far harder. The feature-based models get essentially nowhere.
* The Transformer is the one model that consistently makes money before costs: **+2.9 basis
  points per trade** (a basis point is one hundredth of a percent), and it beats every
  feature-based model in **6 of 7** configurations tested. 
* The model predictive edge does not yet cover trading fees. It gets **56% of the way** there, which is a real result, and a very different situation from having no signal at all.

Everything below explains what those statements mean and how they were measured. A full
plain-language glossary is at the end; every trading term is defined there.

---

## 1. The question, and why it is asked this way

### An example

The naive question, *"will the price be higher later?"*, is close to a coin flip and, worse,
it is not worth answering. Getting into a position and back out costs roughly **5 basis
points** (a basis point, or bp, is one hundredth of a percent); mostly the exchange's trading
fee, an assumption rather than a measurement, broken down in full later in this section. A
correct prediction that captures 1 bp still loses money. **Any target has to be worth more
than the toll charged to pursue it.**

So the question is posed as a *trade* rather than a forecast:

> **Starting now, does the price rise 60 bp before it falls 60 bp, within the next hour?**

If it rises first, a buyer wins. If it falls first, they are stopped out. If neither happens,
the hour expires and the position closes at whatever price is available. Why 60 bp and one
hour specifically is not arbitrary; it is derived from the cost arithmetic itself, a few
paragraphs below.

### Labeling the Dataset. The triple barrier

We use the **López de Prado's Triple-Barrier Method**: from the moment of decision, draw two horizontal lines (a profit target above and a
stop-loss below) and one vertical line, the deadline. Whichever line the price touches first
is the label.

Let $p_t$ be the log price at second $t$, $\theta$ the barrier half-width, and $H$ the
deadline. Define the first times each horizontal barrier is crossed:

$$\tau^{+} = \min \lbrace k \le H : p_{t+k} - p_t > \log(1+\theta) \rbrace \qquad \tau^{-} = \min \lbrace k \le H : p_{t+k} - p_t < -\log(1+\theta) \rbrace$$

with $\min \emptyset = \infty$. The label is whichever came first:

$$y_t = \begin{cases} 1 & \tau^{+} < \tau^{-} \quad \text{(profit target first)} \\ 0 & \tau^{-} < \tau^{+} \quad \text{(stop first)} \\ \text{undefined} & \text{neither, within } H \end{cases}$$

Here $\theta = 60\text{ bp}$ and $H = 3600$ seconds. Predictions are made from the preceding
$L = 300$ seconds of market activity.

**The essential property is that both outcomes require the same 60 bp move.** Magnitude is
held constant across the two classes, so no amount of skill at forecasting *volatility* can
score above chance. Only getting the *direction* right can.

### Where 60 bp and one hour come from

These two numbers were not guessed; they were derived, and the derivation is the single most
important design decision in the project.

A position that wins captures roughly the barrier width. One that loses gives back the same.
But **most positions do neither**: they hit the deadline and close at a near-random price
while still paying the full cost. Writing $\rho$ for the fraction of positions that actually
reach a barrier, $\bar G$ for the size of the move captured when one does, and $c$ for the
round-trip cost, the expected profit per trade is

$$\mathbb{E}[\text{P\&L}] = \rho\,\bar G\,(2h-1) - c$$

where $h$ is the **hit rate** (how often the predicted direction is the one that happens) and $2h -1$ is the **expected direction** as we win with probability $h$ and loose with probability $1-h$.
Setting this to zero gives the number that matters:

$$h^{*} = \frac{1}{2}\left(1 + \frac{c}{\rho\,\bar G}\right)$$

$h^{*}$ is the accuracy a model must reach for the strategy to break even. Note that without a round-trip cost is sufficient to win half of the times.

![What a target has to be worth](assets/economics.png)

The curve above is that identity drawn out: for any given accuracy, how large a payoff a
trade must carry to survive its own costs. A target is worth pursuing only if the horizontal
line (what it actually pays) sits above the curve at an accuracy a model can plausibly
reach.

That makes target selection a search. Sweeping 48 combinations of barrier and deadline over
the **training months only**, and choosing the one that minimises $h^{*}$, gives 60 bp over
one hour. Two properties of that choice are worth stating:

| | |
| :--- | ---: |
| Fraction of windows reaching a barrier | 30.1% |
| Average move captured when one is reached | 61.9 bp |
| Round-trip cost | 5.15 bp |
| **Accuracy needed to break even** | **63.8%** |
| …on the filtered population actually traded (§4.3) | **56.2%** |

A short deadline cannot produce those numbers no matter where the barrier is placed: over
one minute, the amount of price movement available is simply too small relative to the fee.
Lengthening the deadline is what makes the target worth pursuing.

### What it costs to trade

The 5.15 bp round trip breaks down as follows, per side:

| Component | Per side | How it is obtained |
| :--- | ---: | :--- |
| Exchange fee | 2.500 bp | **assumed**, a high-volume Binance account tier |
| Slippage | 0.076 bp | measured from the tape |
| Half-spread | 0.001 bp | measured (Roll estimator, tick-size floor) |

**Slippage:** is the gap between the price you expect and the price you get and it is measured
rather than assumed. Across 45,697 qualifying runs it comes to 0.076 bp on average.

One caveat worth keeping in mind while reading every profit figure below: **the exchange fee
is 97% of the cost and cannot be measured from trade data**. It is a property of an account.
A retail account pays about 10 bp per side rather than 2.5, which quadruples the round trip.

---

## 2. The data

![The tape, its activity, and what the labelling does to it](assets/dataset.png)

The pipeline consumes raw Binance public trade data, one row per executed trade.

| | |
| :--- | ---: |
| Source | [Binance Vision: BTC/USDT spot trades](https://data.binance.vision/?prefix=data/spot/monthly/trades/BTCUSDT/) |
| Period | **181 days**, January–June 2026 |
| Individual trades | **742,244,193** |
| Volume | 3,566,529 BTC |
| Price range | \$58,130 – \$97,924 |
| Raw size | ~52 GB uncompressed |

**On the time scale.** The input is genuinely tick-level: 742 M trades, an average of 47 per
second, with bursts past 400,000 an hour. Machine learning models need inputs of a fixed
size, so the first step folds those irregular ticks into **1-second bars**, one row per
second, carrying the price, the volume, and the balance of buying against selling.

| | |
| :--- | ---: |
| 1-second bars | **15,638,400** |
| …carrying at least one trade | 13,923,707 |
| …carrying **none** | 1,714,693 (**11.0%**) |

That last row explains a detail that matters more than it looks. About 11% of seconds contain
no trade at all. A naive grouping would simply emit no row for those seconds, and then a
"3600-bar deadline" would span an arbitrary 60 to 80 minutes of real time depending on how
busy the market happened to be. Since the deadline *defines the label*, that would make the
label itself depend on market activity. Every second therefore gets a bar; quiet ones carry
the last known price, zero volume, and zero order flow.

### Windows and labels

A prediction window slides forward one second at a time, so consecutive windows overlap
heavily: they share 299 of their 300 seconds. Across the 15,634,501 windows in the sample:

| Outcome within one hour | Windows | Share |
| :--- | ---: | ---: |
| Profit target first (`y = 1`) | 2,249,523 | 14.4% |
| Stop first (`y = 0`) | 2,454,606 | 15.7% |
| **Neither (deadline expired)** | **10,930,372** | **69.9%** |

Most hours do not move 60 bp, so most windows carry no direction and are set aside when
training the directional model. What remains is **4.7 M labelled windows**, close to balanced
between the two classes. The slight tilt toward the stop is the market, not a flaw: bitcoin
fell about 35% across these six months, so downward moves came first more often.

The rate is also not stable, and six months of data is what reveals it:

| Month | Windows reaching a barrier within the hour |
| :--- | ---: |
| April | 21.7% |
| May | 13.6% |
| June | 36.2% |

June offers more than two and a half times as many opportunities as May. Any conclusion drawn
from a single month describes that month rather than the market, which is why every result
below is reported per month rather than averaged into one number.

---

## 3. Two ways to model a market

The same 300 seconds of tape can be handed to a model in two fundamentally different ways.
This project builds both and compares them on identical data, identical labels and an
identical backtest. Understanding the difference between them is most of the story.

### 3.1 The feature approach: compress, then learn

Classical quantitative finance does not feed raw data to a model. It first **extracts
features** (a small number of hand-designed statistics, each with a known meaning) and lets
a general-purpose learner work on those. The design of those statistics is where the domain
knowledge lives.

The premise here is that high-frequency prices do not follow a simple random walk but an
**Itô jump-diffusion**:

$$dp_t = \mu_t\,dt + \sigma_t\,dW_t + J_t\,dN_t$$

In words, three things move the price at once: a slow **drift** ($\mu_t$), continuous
**random jitter** ($\sigma_t dW_t$, ordinary volatility), and occasional **jumps**
($J_t dN_t$) caused by block orders, liquidations or news. The valuable property of this
decomposition is that the smooth part and the jump part can be *measured separately* from
discrete observations. Over a rolling window of $M$ one-second returns $r_{t,i}$:

**1. Realized Variance: total risk.** The sum of squared returns. It absorbs both jitter and
jumps, and is the standard estimate of how volatile a window was.

$$RV_t = \sum_{i=1}^{M} r_{t,i}^2$$

**2. Bipower Variation: smooth risk only.** Multiplying *adjacent* absolute returns makes
this estimator blind to jumps: two consecutive seconds both containing a jump is vanishingly
unlikely, so the products are dominated by ordinary volatility.

$$BPV_t = \frac{\pi}{2} \sum_{i=2}^{M} \lvert r_{t,i}\rvert\,\lvert r_{t,i-1}\rvert$$

**3. Jumps: the shocks alone.** The difference between the two isolates what the jumps
contributed. This is the trick that gives the project its name.

$$J_t = \max(RV_t - BPV_t,\ 0)$$

**4. Order Flow Imbalance: who is being aggressive.** Every trade in the archive records
whether the buyer or the seller was the one crossing the spread to get filled. Summing signed
volume measures whether aggressive buyers or aggressive sellers dominated the window.

$$OFI_t = \sum_{i=1}^{M} q_i \cdot s_i, \qquad s_i = +1 \text{ for an aggressive buy}, \ -1 \text{ for an aggressive sell}$$

Where $q_i$ is the volume size. Three further features add context, because $BPV_t$ and $J_t$ are magnitudes and therefore
direction-blind, and a given amount of order flow means something different in a calm market
than in a violent one:

| Feature | Definition | What it adds |
| :--- | :--- | :--- |
| Lookback return | $r_{t-5m}=p_t - p_{t-5m}$ | the trend the shock arrived in |
| Volatility-adjusted OFI | $OFI_t / (\sqrt{BPV_t} + \epsilon)$ | order flow per unit of risk |
| Signed jumps | $J_t \cdot \operatorname{sign}(r_{t-5\text{m}})$ | a shock into a rising vs falling market |

**Those seven numbers are the entire input** to the feature-based models. Five minutes of
market activity (often tens of thousands of individual trades) is compressed to seven
floating-point values, and a model then learns from those. Two learners are used:

* **Logistic Regression**, the simplest possible baseline. Included as a control: the gap
  between it and a more powerful model measures what non-linearity actually buys.
* **XGBoost**, gradient-boosted decision trees. The standard workhorse for tabular data, and
  capable of capturing interactions a linear model cannot.

This approach has real virtues. It is fast, interpretable, needs little data, and each input
has a defensible meaning. But it has one structural limitation, and it turns out to matter.

### 3.2 The limitation: a sum forgets when things happened

**Every one of those seven features is a sum over the window, and a sum is order-blind.**

Consider two five-minute windows. In the first, a large sell order hits in the opening ten
seconds and the market then recovers calmly. In the second, the market is calm and the same
sell order lands in the *final* ten seconds, still reverberating at the moment the decision
is made.

These are entirely different situations. A trader would treat them differently. But their
realized variance is identical, their bipower variation is identical, their order flow
imbalance is identical. **The feature vectors are the same.** Any model
built on them must give both the same answer.

If information about direction lives in the *timing* of events within the window, and
intuitively it should, since a shock that just landed matters more than one that has been
absorbed, then no amount of tuning can extract it from these features. It was destroyed
during compression, before learning began.

### 3.3 The sequence approach: let the model read the tape

The alternative is to skip compression and give the model the sequence itself.

[**PatchTST**](https://arxiv.org/abs/2211.14730) (Nie et al., ICLR 2023) is a Transformer
designed for time series. A Transformer processes a sequence using **attention**, a mechanism
that lets every position in the sequence look at every other position and decide what is
relevant.

Three design choices adapt it to this problem.

**Patching.** Feeding 300 individual seconds as 300 tokens would be wasteful and slow:
attention cost grows with the square of sequence length, and a single second carries about as
much meaning as a single character does in a sentence. Instead the window is cut into
overlapping **patches** of 16 seconds at a stride of 8, producing 37 tokens instead of 300.
Each token then represents a short stretch of market behaviour with local meaning, and
attention cost drops by roughly 66×.

**Two raw inputs, not seven derived ones.** The model reads only `log_return` and `ofi`, the
price change and the signed volume of each second. Everything in §3.1 is a deterministic
function of these two, so the network can construct any of those features internally if they
are useful. Supplying them would spend model capacity to buy nothing. The saved budget goes
into depth instead: **6 encoder layers, 209,029 parameters.**

**Normalisation that preserves scale information.** Each window is normalised individually,
which is what allows the model to work across calm and violent regimes alike. But that
normalisation erases *how* volatile the window was (genuinely useful information). So the
discarded mean and standard deviation are standardised and fed back in near the output layer,
recovering the best of both.

In contrast to standard PatchTST that is a time series prediction, the output is a single number (we use the model as a binary classificator).

### 3.4 What the sequence approach costs

Reading raw sequences is not free, and the price is paid in three places.

**Compute.** Measured on one CPU thread, per prediction:

| Model | Single window | Batched, per window | Windows/second |
| :--- | ---: | ---: | ---: |
| Logistic Regression | 0.002 ms | 0.01 µs | 143,717,617 |
| XGBoost | 0.045 ms | 0.55 µs | 1,807,257 |
| **PatchTST** | **0.934 ms** | **423 µs** | **2,362** |

XGBoost is roughly **20× faster** per prediction and **770× faster** in bulk. Feature
preparation adds 0.0027 ms to the tabular models and closes none of that gap. For a strategy
deciding once per second this is irrelevant, a millisecond is ample. For one competing on
microseconds it would be disqualifying.

**Training.** The tabular models fit in minutes on a laptop. PatchTST needs a GPU; the
results below come from a one-hour run on a Google Colab T4.

**Data.** Sequence models have far more parameters to constrain and need correspondingly more
examples. Six months of tick data is what makes this comparison possible at all.

The question §5 answers is whether that price buys anything.

---

## 4. Measuring it honestly

Three safeguards separate a real result from a flattering one. Each addresses a specific way
this kind of study normally goes wrong.

### 4.1 Walk-forward validation

Models are tested the way time actually runs. Each fold trains only on months strictly
*before* its test month:

| Fold | Trains on | Tests on |
| :--- | :--- | :--- |
| 1 | Jan–Mar | April |
| 2 | Jan–Apr | May |
| 3 | Jan–May | June |

Every reported number is therefore a genuine forecast, and there are three of them rather
than one, so drift between months is visible instead of averaged away.

Because consecutive windows overlap by 299 seconds and each label depends on the *next* hour,
a **purge** of 3,899 windows is dropped at each boundary. Without it, the final training
labels would be determined by price moves occurring inside the test period: the model would
have seen part of its own exam.

### 4.2 Splitting the problem in two

The directional model can only be trained on windows where a barrier was actually reached;
elsewhere there is no direction to learn. But **whether** a barrier will be reached is not
knowable at decision time. Reporting accuracy on that subset would silently assume knowledge
of the outcome.

The fix is **meta-labelling**: two models with different jobs.

| | Question | Trained on |
| :--- | :--- | :--- |
| **Gate** | Is this window worth trading at all? | every window |
| **Side** | Which direction? | windows that resolved |

At decision time both produce a number from past data only. A trade is taken when the gate
clears its threshold, in the direction the side model indicates. The traded population is
therefore selected by a *prediction* rather than by an *outcome*, and the backtest prices
every window that selection admits, including those where the gate turns out to be wrong.

The gate's threshold is set as a percentile of the **training** months' scores. Using the test
month's own distribution would smuggle in future information through the back door.

### 4.3 A backtest with the details that matter

Two elements dominate, and both cut reported performance sharply:

**Positions never overlap.** Windows advance every second, so a simulator that opened a
position on every signal would count the same price move hundreds of times and report a
performance figure inflated by roughly the square root of that overlap. Signals arriving while
a position is open are discarded. At an hour-long deadline this is aggressive: 15.6 M windows
collapse to 2,680 actual trades.

**Costs are charged in full**, including the spread, measured slippage and, for passive
orders, a **queue model**: a limit order joins a first-in-first-out queue and only fills once
the volume ahead of it trades, which means it fills exactly when the market is coming toward
you and not when it runs away.

The gate does something valuable here beyond saving fees. Filtering to high-volatility windows
raises the fraction of trades that reach a barrier from 32.5% to **66.9%**, which lowers the
break-even accuracy from 63.8% to **56.2%**. Trading less often does not merely cost less; it
makes the remaining trades structurally easier to win.

---

## 5. Results

![Gate and side accuracy, fold by fold](assets/walkforward_auc.png)

Accuracy is reported as **ROC-AUC**: the probability that a randomly chosen positive case is
ranked above a randomly chosen negative one. 0.5 is a coin flip, 1.0 is perfect. In
high-frequency finance, consistently above 0.52 is generally considered a real edge.

Profitability is reported two ways. **Gross** is what the price move gave, before any cost;
**net** is what remains after the round trip. The **Sharpe ratio** is return divided by its
own variability, annualised (a measure of return per unit of risk), where negative means
losing money.

### 5.1 Predicting whether: a clear success

The gate asks *will any barrier be reached within the hour?* (the volatility question).

| Test month | XGBoost, 7 features | Realized variance alone, untrained |
| :--- | ---: | ---: |
| April | 0.7184 | 0.7143 |
| May | 0.7079 | 0.7077 |
| June | 0.7561 | 0.7509 |

**This works, and it works on every month tested**, across regimes that differ by a factor of
two and a half in how much they move. Volatility is strongly autocorrelated (busy markets
stay busy), and the jump-diffusion features capture that cleanly.

The second column is the more interesting one. A plain rolling sum of squared returns, with
no training and no parameters at all, comes within **0.005** of gradient-boosted trees every
single time. That is a genuinely useful finding: the feature set is an excellent volatility
estimator, which is exactly what realized variance and bipower variation were designed to be,
and the machine learning adds essentially nothing on top. **If you want this forecast, take
the rolling sum**, one line of code and free.

### 5.2 Predicting which way: where the approaches separate

The side model asks *which barrier comes first?* (the directional question).

| Test month | Logistic (OFI) | Logistic (7 features) | XGBoost | **PatchTST** |
| :--- | ---: | ---: | ---: | ---: |
| April | 0.5013 | 0.5166 | 0.5060 | **0.5256** |
| May | 0.5201 | 0.5320 | **0.5434** | 0.5244 |
| June | 0.5021 | **0.5244** | 0.5035 | 0.5117 |

Read on its own, this table looks like a draw. Every model sits in the 0.50–0.54 band, and
the seven-feature logistic regression edges PatchTST in two of three months.

**But ROC-AUC scores the entire population, and the strategy does not trade the entire
population.** It trades the top decile that the gate admits. A model can rank mediocrely
overall and rank very well exactly where it counts, and that is what happens here. The
backtest is where it shows.

### 5.3 The backtest

![Profit per trade, by model](assets/backtest.png)

Pooled across the three out-of-sample months, one position at a time. The four highlighted
rows share the **same gate at the same threshold**, so they trade an identical set of 785
windows; the only difference between them is which direction each trade was taken. Two kinds
of gate are compared throughout: **RV** is the untrained realized-variance threshold from
§5.1, and **Trained** is the XGBoost gate it was checked against there, at the top 50%, 20%
or 10% of scores. §5.1 found the two nearly indistinguishable at the volatility question
itself; the tables below ask whether that holds once a gate is actually used to select trades.

| Model | Gate | Trades | Gross/trade | Net/trade | Accuracy when resolved | Sharpe |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **PatchTST** | RV, top 10% | 785 | **+2.879 bp** | −2.275 bp | **54.29%** | −2.67 |
| XGBoost | RV, top 10% | 785 | −0.081 bp | −5.234 bp | 49.52% | −6.92 |
| Logistic (7f) | RV, top 10% | 785 | −0.280 bp | −5.433 bp | 50.48% | −3.86 |
| Logistic (OFI) | RV, top 10% | 785 | −1.643 bp | −6.796 bp | 49.90% | −7.91 |
| PatchTST | none | 2,680 | +1.151 bp | −4.003 bp | 53.22% | −11.12 |
| XGBoost | none | 2,680 | −0.326 bp | −5.479 bp | 50.80% | −13.26 |

**Read the gross column**, because it is the one that does not depend on any cost assumption.
On identical windows, PatchTST turns a coin flip into a 54.3% hit rate and **+2.9 bp per
trade**, while all three feature-based models sit at chance and earn nothing.

This is not a single lucky configuration. Across all seven gate settings tested:

| Gate setting | Logistic (OFI) | Logistic (7f) | XGBoost | **PatchTST** |
| :--- | ---: | ---: | ---: | ---: |
| none | −0.475 | −0.111 | −0.326 | **+1.151** |
| RV, top 50% | −0.550 | +0.323 | −0.127 | **+0.915** |
| RV, top 20% | +0.076 | −1.213 | −0.526 | **+1.814** |
| RV, top 10% | −1.643 | −0.280 | −0.081 | **+2.879** |
| Trained, top 50% | −0.823 | +0.044 | +0.360 | **+0.658** |
| Trained, top 20% | +1.537 | +0.886 | −0.808 | **+2.409** |
| Trained, top 10% | +0.319 | −0.412 | **+1.793** | +1.479 |

**PatchTST has the highest gross profit in six of seven configurations**, and is positive in
all seven. The feature-based models scatter around zero and change sign essentially at
random, the signature of no signal. This is the project's central result: **the ordering
information that summary statistics throw away is real, and it is worth roughly 3 basis
points per trade.**

It also validates the reasoning in §3.2. The prediction was that timing within the window
carries directional information invisible to sums. A model that can see ordering finds it; a
model that cannot, does not.

### 5.4 What it does not yet do

The edge is real but it does not cover the toll. At **+2.9 bp of gross against a 5.15 bp
round trip**, the strategy captures about **56% of what it needs**, and net profit per trade
stays negative. The best Sharpe ratio is −2.67, with a 95% confidence interval of
[−7.29, +1.42] that includes zero.

Closing a 2.2 bp gap therefore needs one of: a stronger signal, a lower fee tier, or a
different execution style. The first looks the most promising, for a reason given below.

---

## 6. What this establishes

**The volatility forecast is solid and immediately usable.** Predicting whether a 60 bp move
arrives within the hour scores 0.71–0.76 across three unseen months, and a parameter-free
rolling sum matches trained gradient boosting to within 0.005. This is a finished, reliable
component, useful for position sizing, risk limits, quoting width, or anything that needs to
know how much the market is about to move rather than where it is going.

**Feature engineering solved the volatility problem and could not solve the direction one.**
The jump-diffusion decomposition is excellent at what it was designed for. But its outputs
are sums, and sums cannot distinguish a shock that just landed from one that landed five
minutes ago. That limitation is structural, not a matter of tuning.

**The sequence model finds directional signal the feature models cannot.** +2.9 bp of gross
profit per trade on identical windows, positive in every configuration tested and best in six
of seven, with hit rates of 52–54% against the feature models' 50%. Reading the tape in order
is what makes the difference.

**The cost of that capability is about 20× more compute per prediction and a GPU for
training**, irrelevant at this decision frequency, decisive at a faster one.

**Profitability is close but not reached.** The signal covers 56% of its trading costs. It is
worth being precise about what kind of gap that is: this is not a strategy with no edge, and
it is not one defeated by arithmetic. It is a real, repeatable, out-of-sample edge that is
currently too small to pay a fee.

### Where the remaining gap most likely is

The PatchTST results above come from a **one-hour training run on a single Colab T4**
(a free-tier GPU), with the training stride automatically raised to fit the time budget, which
means the model saw a fraction of the available windows. It reached a real edge anyway.

That makes training budget the most promising lever available: a longer run on better
hardware, using more of the 4.7 M labelled windows, is the cheapest experiment that could
close the remaining 2.2 bp. Nothing about the architecture or the target needs to change to
try it.

---

## 7. Getting started

### Install

```bash
git clone https://github.com/NynsenFaber/BipowerQuant.git
cd BipowerQuant
uv sync                       # or: pip install polars numpy scikit-learn xgboost torch matplotlib
python build.py               # compiles the C++ engine
```

`build.py` needs CMake 3.18+ and a C++20 compiler, and works on Linux, macOS and Windows.
The C++ extension accelerates the rolling feature computation; the sequence pipeline is pure
Polars, NumPy and PyTorch and runs without it, which is what lets the same code run in Colab.

### Run the tests

```bash
uv sync --group test
uv run pytest                 # 260 tests, ~4 seconds
uv run pytest -m "not slow"   # skip the ones that fit models
```

Everything is synthetic and seeded, so the suite never touches the 52 GB of tape. What it
defends, beyond arithmetic:

| Property | Why it is a test rather than a review |
| :--- | :--- |
| **The barrier is symmetric** | Inverting the price flips every direction and leaves timing alone. An asymmetry would manufacture a directional edge from nothing. |
| **Positions never overlap** | A simulator that opens one per signal counts the same move hundreds of times and inflates performance, in the flattering direction, which is how it survives a casual read. |
| **Training ends before testing begins** | Asserted on every fold. A leak here would not fail anything; it would quietly raise every number in §5. |
| **The two feature paths agree** | The C++ engine against an independent NumPy implementation written from the formulas rather than transcribed from the loop. |
| **The break-even identity holds** | `h* = ½(1 + c/(ρG))`, pinned against hand-computed cases. |

### Get the data and build the caches

Download the monthly archives from the
[Binance archive](https://data.binance.vision/?prefix=data/spot/monthly/trades/BTCUSDT/) into
`data/`, then fold them into 1-second bars once; every later run starts instantly:

```bash
cd python
for m in 01 02 03 04 05 06; do
  python -c "import sequence_matrix as s; s.save_bars(
      s.load_second_bars(f'../data/BTCUSDT-trades-2026-$m.csv'), f'../data/bars_2026-$m.npz')"
done
python -c "import sequence_matrix as s; s.save_bars(
    s.load_bar_caches([f'../data/bars_2026-{m:02d}.npz' for m in range(1,7)]), '../data/bars_6m.npz')"
```

### Reproduce the study

```bash
# choose a target from cost arithmetic alone, on training months only
python sweep_barriers.py --bars ../data/bars_6m.npz --geometry-only \
                         --months 2026-01 2026-02 2026-03 \
                         --barriers 5 10 20 30 40 60 100 \
                         --horizons 60 300 900 1800 3600 7200

# fit the gate and side models per fold, then run the backtest grid
python walkforward.py --bars ../data/bars_6m.npz --train-stride 5 \
                      --patchtst-probs ../weights/patchtst_probs.npz \
                      --out ../data/wf_final.json

python make_walkforward_figures.py --result ../data/wf_final.json
```

Useful flags: `--entry maker` for passive execution, `--taker-fee-bps` to test a different
account tier, `--scheme rolling` for a fixed-width training window.

### Train PatchTST

Training happens on a free Colab GPU; the backtest happens locally against the exported
probabilities.

1. Open [notebooks/train_patchtst_colab.ipynb](notebooks/train_patchtst_colab.ipynb) in Colab,
   select a GPU runtime, and *Run all*. It downloads the archives itself, folding each month
   into bars and deleting the CSV before fetching the next.
2. `TIME_BUDGET_HOURS` is enforced rather than advisory: the notebook measures real training
   throughput, projects the cost across every fold, and raises the training stride until the
   run fits. Raise the budget to train on more windows.
3. The final cell writes `patchtst_probs.npz`, one probability array per fold. Put it in
   `weights/` and re-run `walkforward.py --patchtst-probs`.

---

## 8. Technology

| Component | Technology | Role |
| :--- | :--- | :--- |
| Data ingestion | Polars (lazy/streaming) | folds 742 M ticks into 15.6 M bars, one pass per month, flat memory |
| Math engine | C++20 + `pybind11` | rolling variance, bipower variation, jumps and order flow over raw NumPy buffers |
| Tabular models | XGBoost, scikit-learn | the gate, the seven-feature side model, the linear control |
| Sequence model | PyTorch | PatchTST, trained per fold on a Colab GPU |
| Backtest | NumPy | fees, spread, slippage, queue position, non-overlapping positions |
| Tests | pytest + coverage | 260 tests on synthetic data, ~4 s, 87% of the library |
| CI | GitHub Actions | builds the extension and runs the suite on Linux, macOS and Windows |

Two implementation notes that are load-bearing rather than incidental:

* **Windows are never materialised.** Consecutive windows share 299 of 300 seconds, so a
  materialised training tensor would cost tens of gigabytes. The channel matrix instead lives
  on the GPU once (~125 MB) and every batch is a single gather.
* **The triple barrier is vectorised.** Labelling 15.6 M bars naively means billions of
  comparisons; here it is `H` vectorised passes over the series.

---

## Appendix: glossary

Every term used above, in plain language.

### Money and cost

* **bp (basis point):** One hundredth of a percent. 1 bp = 0.01%, 100 bp = 1%. A 60 bp target
  means the price must move 0.6% (about \$450 on a \$75,000 bitcoin).
* **P&L (profit and loss):** What a trade made or lost, quoted here *per trade* in bp so that
  strategies trading at different frequencies are comparable.
* **gross vs net:** Gross is what the price move gave you, before costs. Net is what remains
  after fees, spread and slippage. Gross is read first: zero gross means no edge at all,
  while positive gross with negative net means an edge too small to pay for itself.
* **round trip:** The cost of opening a position *and* closing it. Quoted this way because a
  position you cannot close is not a trade.
* **fee, taker and maker:** The exchange's commission. Taking liquidity immediately (taker)
  costs more than posting an order and waiting (maker).
* **spread (bid-ask spread):** The gap between the best price to buy at and the best price to
  sell at. Crossing it is a cost paid the instant you trade. On BTC/USDT it is tiny (about
  0.001 bp) because the market is extremely liquid.
* **slippage:** The difference between the price you expected and the price you got, because
  your order consumed the best price and continued into worse ones.

### Orders and execution

* **order book:** The exchange's live list of resting buy and sell orders at each price.
* **market vs limit order:** A market order executes immediately at whatever price is
  available. A limit order specifies a price and waits, which may mean never filling.
* **taker vs maker:** Taking means crossing the spread for immediate execution. Making means
  posting a limit order and waiting for someone else to trade against it.
* **queue position:** Limit orders at the same price fill first-in-first-out. Your order fills
  only once the volume ahead of it has traded, which is why passive orders fill reliably
  when the market moves toward you and unreliably when it runs away.
* **adverse selection:** The systematic problem that your passive orders fill precisely when
  filling is bad for you.

### The trade

* **long and short:** Long profits when the price rises; short profits when it falls.
* **profit target and stop-loss:** The price at which a winning position is closed to bank the
  gain, and the price at which a losing one is closed to cap the loss. The two horizontal
  barriers.
* **deadline (vertical barrier):** The time limit. A position reaching neither target nor stop
  is closed at whatever price is available, earning roughly nothing while still paying the
  full round trip.
* **resolved trade:** One that reached the target or the stop, rather than expiring at the
  deadline. The accuracy that decides profitability is measured over resolved trades only.
* **first passage:** Which barrier the price touches *first*. A move that rises 60 bp and then
  collapses is labelled a win, because a real position would already have closed.
* **overshoot:** Prices are checked once per second, so a barrier is usually crossed by a
  little more than exactly 60 bp. The backtest exits at the real price, not the barrier level.

### Describing the market

* **tick:** One executed trade. **Bar:** all activity in a fixed interval, here one second.
* **volatility:** How much the price moves, regardless of direction.
* **realized variance:** The sum of squared returns over a window, the standard measure of
  how volatile that window was.
* **bipower variation:** A volatility measure deliberately blind to sudden jumps, built by
  multiplying adjacent absolute returns. The difference between the two isolates the jumps.
* **jump-diffusion:** A model in which the price moves through both continuous jitter and
  occasional discontinuous jumps.
* **order flow imbalance:** The balance of aggressive buying against aggressive selling.
* **regime:** A period with its own character (calm or violent, trending or ranging). Regimes
  shift, which is why results are reported per month.

### Judging a model

* **lookback:** How much history the model reads before deciding. Here, 300 seconds.
* **out-of-sample:** Data the model has never seen. Every headline number here is
  out-of-sample.
* **walk-forward:** Testing the way time runs: train on January–March, test on April; then
  train on January–April, test on May. Each result is a genuine forecast.
* **purge:** A gap deliberately left between training and test data, so that training labels
  are not decided by price moves inside the test period.
* **look-ahead bias:** Using information unavailable at the moment of the decision. The most
  common way a backtest shows a profit that does not exist.
* **hit rate:** How often the predicted direction is the one that happened.
* **break-even hit rate (h\*):** The accuracy at which a strategy makes exactly zero after
  costs. Used here to choose the target *before* fitting any model.
* **edge:** A genuine, repeatable advantage over chance.
* **gate and side:** This project's split. The gate predicts *whether* a move worth trading
  will happen; the side predicts *which way*. Separate models, because the first is much
  easier than the second and mixing them lets skill at the easy one masquerade as skill at
  the hard one.
* **meta-labelling:** Using a second model to decide whether to act on the first model's
  signal, which allows the strategy to be scored on a population it could have selected in
  advance.
* **ROC-AUC:** The probability that a randomly chosen positive case is ranked above a randomly
  chosen negative one. 0.5 is a coin flip, 1.0 is perfect. Used as the headline metric because
  it does not depend on where the decision threshold is placed.
* **Sharpe ratio:** Average return divided by its standard deviation, annualised, return per
  unit of risk. Computed here on daily P&L and scaled by √365, since crypto trades every day.
* **drawdown:** The decline from a peak in cumulative profit; in practice what decides whether
  a strategy survives long enough to be right.
* **confidence interval:** The range of values consistent with the data. Computed here with a
  *moving-block bootstrap*, which resamples contiguous stretches rather than individual
  observations, necessary because consecutive windows overlap heavily and treating them as
  independent would produce intervals several times too narrow.
