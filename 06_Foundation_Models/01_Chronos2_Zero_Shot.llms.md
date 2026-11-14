# Chronos-2 Zero-Shot

A pretrained forecaster that has never seen this house, used with no training at all: univariate, with calendar covariates, and with the sub-meters as extra series.

Author

Benedict Thekkel

## What a time series foundation model is

Every model on this site so far was trained on this house. A **foundation model** is trained once, on a very large and varied corpus of time series, and then forecasts a new series from its context alone: **zero-shot**. It has learned what daily cycles, weekly cycles, trends, spikes and level shifts generally look like, and recognises them in a context it has never seen.

**Chronos-2** (Amazon, October 2025) is an encoder-only transformer with about 120 million parameters. It reads the context in patches, forecasts quantiles for every horizon step at once, and adds **group attention**: several related series (or a target and its covariates) attend to each other, so it can use past-only covariates, known-future covariates and related series in the same forward pass. In spring 2026 it ranked first on GIFT-Eval; the [comparison page](../06_Foundation_Models/foundation_model_comparison.llms.md) has the current standings.

It runs through Amazon’s `chronos-forecasting` package, because no general-purpose library loads it. That package is installed in the root environment only, and imported only on these GPU pages, which carry `skip_exec` and stored outputs (an exception recorded in the repo’s coding conventions). The weights are Apache-2.0.

------------------------------------------------------------------------

``` python
import gc
import logging
import time
import warnings

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from ML_Timeseries.backtest import backtest, future_index, record, score
from ML_Timeseries.data import line_chart, load_household, show

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "06_Foundation_Models/01_Chronos2_Zero_Shot"
CTX = 2048  # hours of context given to the model: about 12 weeks
pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu")
print(f"Chronos-2: {sum(p.numel() for p in pipe.model.parameters()) / 1e6:.0f} M parameters")


def median(out):
    "Chronos-2 returns (quantiles, mean); quantiles[i] is [variates, horizon, levels]."
    return out[0][0][0, :, 0].cpu().numpy()


def run(name, f):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    bt = backtest(y, f)
    sc = score(bt, y, exclude=FILLED) | {"seconds": time.perf_counter() - t0, "peak VRAM GB": torch.cuda.max_memory_allocated() / 1e9}
    print(f"{name}: skill {sc['skill']:+.1%}, MASE {sc['mase']:.3f}, {sc['seconds']:.1f} s for 30 forecasts")
    return sc, bt
```

    Chronos-2: 119 M parameters

## 1. Univariate, zero-shot

The model gets the last 2,048 hours and nothing else: no training, no tuning, no features.

------------------------------------------------------------------------

``` python
def univariate(history, horizon):
    x = torch.tensor(history.to_numpy(dtype=np.float32)[-CTX:])
    return median(pipe.predict_quantiles([x], prediction_length=horizon, quantile_levels=[0.5]))


uni_score, uni_bt = run("Chronos-2, univariate", univariate)
_ = record("Chronos-2, zero-shot", "foundation model", {k: uni_score[k] for k in ("mase", "skill", "mae", "rmse", "wape", "smape", "seconds")},
           notebook=NB, notes="no training; 2,048 h context")
```

    Chronos-2, univariate: skill +21.7%, MASE 0.645, 0.9 s for 30 forecasts

## 2. With calendar covariates

Hour of day and weekday are known for every future hour, so they can be passed as **known-future covariates**: their history alongside the target’s, and their future values for the forecast hours.

------------------------------------------------------------------------

``` python
def calendar(index):
    return {"hour": index.hour.to_numpy(np.float32), "dow": index.dayofweek.to_numpy(np.float32)}


def with_calendar(history, horizon):
    x = history.iloc[-CTX:]
    item = {"target": x.to_numpy(np.float32), "past_covariates": calendar(x.index),
            "future_covariates": calendar(future_index(history, horizon))}
    return median(pipe.predict_quantiles([item], prediction_length=horizon, quantile_levels=[0.5]))


cal_score, _ = run("Chronos-2, + calendar", with_calendar)
```

    Chronos-2, + calendar: skill +21.6%, MASE 0.640, 0.7 s for 30 forecasts

## 3. With the sub-meters as past covariates

The three sub-meters are known up to the origin but not after it, so they go in as **past-only covariates**. Group attention lets the model look across the four series.

------------------------------------------------------------------------

``` python
SUBS = ["sub1_kwh", "sub2_kwh", "sub3_kwh"]


def with_submeters(history, horizon):
    x = history.iloc[-CTX:]
    past = {c: hh[c].loc[x.index].to_numpy(np.float32) for c in SUBS} | calendar(x.index)
    item = {"target": x.to_numpy(np.float32), "past_covariates": past, "future_covariates": calendar(future_index(history, horizon))}
    return median(pipe.predict_quantiles([item], prediction_length=horizon, quantile_levels=[0.5]))


sub_score, _ = run("Chronos-2, + calendar + sub-meters", with_submeters)
```

    Chronos-2, + calendar + sub-meters: skill +21.7%, MASE 0.639, 1.4 s for 30 forecasts

``` python
w = uni_bt.window.max()
last = uni_bt[uni_bt.window == w]
show(line_chart(last.ds.dt.strftime("%a %H:00"), {"actual": last.y, "Chronos-2 (zero-shot)": last.yhat},
                y_name="kWh", height="320px", zoom=False))
```

With no training at all, Chronos-2 beats SARIMA, every gradient-boosting model and every library deep learning model on the [scoreboard](../02_Evaluation/scoreboard.llms.md). Only the two hand-built networks from the deep learning section, trained on this house, do better. Thirty forecasts take about a second on the GPU.

Neither kind of covariate helps. The calendar adds nothing because the model already reads the daily and weekly rhythm from twelve weeks of context, the same reason [sub-meter lags](../01_Foundations/feature_engineering.llms.md) added nothing to gradient boosting: the information is already in the target’s own history. Covariates pay when they carry something the target cannot, such as a weather forecast for a heating load or a price signal.

The practical meaning: for a new meter with a few months of data and no time to build a model, a zero-shot forecast from a foundation model is a strong default on day one.

------------------------------------------------------------------------

Back to top
