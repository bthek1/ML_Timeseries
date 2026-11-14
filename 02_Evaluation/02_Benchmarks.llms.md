# Benchmarks

What the public forecasting benchmarks measure, how to read their leaderboards without being misled, and two of their protocols run by hand: ETTh1 long-horizon and M4 Hourly.

Author

Benedict Thekkel

## Why benchmarks exist

One dataset can make any method look good. A benchmark fixes many datasets, a split and a metric in advance, so that results from different papers compare. Each one also bakes in choices that decide what “best” means, and a leaderboard number is only meaningful once you know them.

| Benchmark | What it is | Headline metric |
|----|----|----|
| **M4** (2018) | 100,000 series, yearly to hourly, fixed horizons | sMAPE and MASE, combined as OWA (overall weighted average) against a naive method |
| **Monash** (2021) | a few dozen public datasets in one format, a common pretraining source | MASE per dataset |
| **ETT / long-horizon** (2021) | ETTh1/h2/m1/m2, Electricity, Traffic, Weather; horizons 96 to 720 | MSE and MAE on z-normalised data |
| **GIFT-Eval** (2024) | 23 datasets, 144,000 series, 177 million points, built to test foundation models | MASE and CRPS over seasonal naive, geometric mean |
| **fev-bench** (2025) | tasks from the AutoGluon team, many with covariates | win rate and skill score against a reference |

------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ML_Timeseries.backtest import mase_scale, record, smape
from ML_Timeseries.data import bar_chart, fetch, load_m4_hourly, show

from threadpoolctl import threadpool_limits

# nbdev-test runs pages in parallel; per-page OpenMP pools on 4 cores made the gradient-boosting pages 30x slower.
_ = threadpool_limits(1)
```

## 1. How to read a leaderboard

Five questions decide whether a leaderboard number means what it seems to:

1.  **Normalised to what?** GIFT-Eval divides each model’s error by seasonal naive’s error on the same series, then takes the geometric mean, so 0.7 means 30% less error than seasonal naive on a typical series. A raw average would let one large series dominate.
2.  **Zero-shot or trained?** A foundation model scored zero-shot has never seen the test series. A model trained per dataset has. Leaderboards mark the two separately; compare like with like.
3.  **Was the test data in the pretraining corpus?** Many foundation models pretrain on public archives that overlap the benchmark. GIFT-Eval marks entries with possible test leakage; fev-bench replaces a leaky model’s scores on the overlapping datasets. An unflagged result from an overlapping corpus is the same bug as [leak 4](../02_Evaluation/backtesting.llms.md), at the scale of a whole corpus.
4.  **Point or probabilistic?** CRPS rewards a good spread of outcomes; MASE only the median. A model can top one ranking and sit mid-table on the other.
5.  **When was it read?** Leaderboards change monthly. A ranking quoted without a date is not evidence. In September 2026, one paper counted 129 entries on GIFT-Eval.

------------------------------------------------------------------------

## 2. ETTh1: the long-horizon protocol

ETT (Electricity Transformer Temperature, from the Informer paper, 2021) records a power transformer in China every hour for two years. `OT`, the oil temperature, is the target. The long-horizon literature fixes the protocol:

- **Split**: the first 12 months train, the next 4 validate, the next 4 test.
- **Scale**: z-normalise with the training mean and standard deviation; report errors on that scale.
- **Windows**: every possible test window, one step apart. Each test input may reach back into the validation period.
- **Horizons**: 96, 192, 336 and 720 steps. This page uses 96 with a 336-step lookback.

The data comes from the fev benchmark’s public copy on Hugging Face (0.5 MB).

------------------------------------------------------------------------

``` python
ETT_URL = "https://huggingface.co/datasets/autogluon/fev_datasets/resolve/main/ETT/1H/train-00000-of-00001.parquet"
ett = pd.read_parquet(fetch(ETT_URL, "ett_1h.parquet"))
row = ett.set_index("id").loc["ETTh1"]
ot = pd.Series(np.asarray(row["OT"], dtype=float), index=pd.DatetimeIndex(row["timestamp"]), name="OT")
print(f"ETTh1 OT: {len(ot):,} hours from {ot.index[0]:%Y-%m-%d} to {ot.index[-1]:%Y-%m-%d}")
```

    ETTh1 OT: 17,420 hours from 2016-07-01 to 2018-06-26

``` python
H, L = 96, 336
n_train, n_val, n_test = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
mu, sd = ot.iloc[:n_train].mean(), ot.iloc[:n_train].std()
z = ((ot - mu) / sd).to_numpy()


def windows(start, stop):
    "Input/target pairs whose targets start in [start, stop - H]."
    idx = np.arange(start, stop - H + 1)
    X = np.stack([z[i - L:i] for i in idx])
    Y = np.stack([z[i:i + H] for i in idx])
    return X, Y


X_tr, Y_tr = windows(L, n_train)
X_te, Y_te = windows(n_train + n_val, n_train + n_val + n_test)
print(f"{len(X_tr):,} training windows, {len(X_te):,} test windows")
```

    8,209 training windows, 2,785 test windows

Three forecasters, from nothing to a single linear layer:

- **Seasonal naive (24 h)**: repeat the last day four times.
- **Last value**: hold the last observation flat for 96 hours.
- **Linear**: one ridge regression from the 336 inputs to the 96 outputs. This is DLinear (2023) without its trend split, the model that showed many published transformers did not beat a linear map on this benchmark.

------------------------------------------------------------------------

``` python
forecasts = {
    "Seasonal naive (24 h)": np.tile(X_te[:, -24:], H // 24),
    "Last value": np.repeat(X_te[:, -1:], H, axis=1),
    "Linear (ridge, 336 to 96)": Ridge(alpha=10.0).fit(X_tr, Y_tr).predict(X_te),
}
ett_scores = pd.DataFrame({name: {"MSE": np.mean((Y_te - f) ** 2), "MAE": np.mean(np.abs(Y_te - f))}
                           for name, f in forecasts.items()}).T
ett_scores.round(3)
```

|                           | MSE   | MAE   |
|---------------------------|-------|-------|
| Seasonal naive (24 h)     | 0.071 | 0.211 |
| Last value                | 0.069 | 0.203 |
| Linear (ridge, 336 to 96) | 0.058 | 0.180 |

``` python
show(bar_chart(ett_scores.index, {"MSE": ett_scores.MSE, "MAE": ett_scores.MAE}, height="320px"))
```

Two things to take from the numbers. First, holding the last value flat is as good as copying the last day (slightly better here): oil temperature drifts slowly, and its daily cycle is weak compared with the drift over four days. Second, one linear layer cuts the error of both baselines by about a fifth. Any model claimed to be state of the art on ETTh1 has to beat that layer, and papers from 2021 and 2022 often reported transformers that did not.

------------------------------------------------------------------------

## 3. M4 Hourly: the competition protocol

M4 scores each of the 414 hourly series on its fixed 48-hour test horizon, then averages across series:

- **sMAPE** in percent (the M4 definition multiplies by 100);
- **MASE** with a 24-hour season, scaled by each series’ own training history.

Averaging scale-free errors across series is what makes 414 series of very different sizes comparable.

------------------------------------------------------------------------

``` python
m4_train, m4_test = load_m4_hourly()
rows = []
for uid, g in m4_train.groupby("unique_id", sort=False):
    hist = g.y.to_numpy()
    actual = m4_test.loc[m4_test.unique_id == uid, "y"].to_numpy()
    yhat = np.resize(hist[-24:], 48)
    scale = np.mean(np.abs(hist[24:] - hist[:-24]))
    rows.append({"unique_id": uid, "smape": 100 * smape(actual, yhat), "mase": np.mean(np.abs(actual - yhat)) / scale})
m4_scores = pd.DataFrame(rows)
m4_snaive = {"mase": m4_scores.mase.mean(), "smape": m4_scores.smape.mean(), "skill": 0.0}
record("Seasonal naive (24 h)", "baseline", m4_snaive, notebook="02_Evaluation/02_Benchmarks", dataset="m4_hourly")
pd.Series(m4_snaive).round(3)
```

    mase      1.193
    smape    13.912
    skill     0.000
    dtype: float64

The average hides a wide spread. Most series sit between MASE 0.5 and 1.8, but a few reach 6, where yesterday says almost nothing about tomorrow. A method can win on the mean by doing well on those few, which is why M4 also reported ranks and why GIFT-Eval uses a geometric mean.

``` python
m4_scores.mase.describe(percentiles=[0.1, 0.5, 0.9]).round(3)
```

    count    414.000
    mean       1.193
    std        0.660
    min        0.049
    10%        0.480
    50%        1.127
    90%        1.833
    max        6.147
    Name: mase, dtype: float64

## What this unlocks

The M4 Hourly seasonal naive row is now on the [scoreboard](../02_Evaluation/scoreboard.llms.md) as the reference for the global models. The household protocol from [Backtesting](../02_Evaluation/backtesting.llms.md) is this site’s own benchmark: one series, scored the same way on every page.

------------------------------------------------------------------------

Back to top
