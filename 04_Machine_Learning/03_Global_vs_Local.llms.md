# Global vs Local Models

One LightGBM model trained across all 414 M4 Hourly series against one model per series, with the same features: why sharing data across series wins, and why the target transforms decide whether it works at all.

Author

Benedict Thekkel

## Two ways to forecast many series

| Approach | Trained on | Typical models |
|----|----|----|
| **Local** | one series at a time, one parameter set each | ARIMA, ETS, a small LightGBM per series |
| **Global** | all series pooled, one shared parameter set | one LightGBM or neural network for everything |

Classical statistics is local by habit: each series gets its own model. The M4 (2018) and M5 (2020) competitions changed practice. The M5 winners were global LightGBM models, and M4’s winner was a hybrid with global parameters. A global model sees far more examples of “what a daily cycle looks like” or “what happens after a spike” than any single series can show it, which is the same reason a foundation model trained on many datasets can forecast a series it has never seen.

Pooling only works if the series look alike to the model. M4 Hourly mixes series whose levels differ by orders of magnitude, so each one must first be put on a common scale. That is the job of the **target transforms**.

This page uses Nixtla’s `mlforecast`, which builds the lag features, applies the transforms per series, trains one model, and forecasts recursively. The [Nixtla MLForecast](../Libraries/Nixtla/nixtla_mlforecast.llms.md) walkthrough covers its API.

------------------------------------------------------------------------

``` python
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, SeasonalRollingMean
from mlforecast.target_transforms import Differences, LocalStandardScaler
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import record, smape
from ML_Timeseries.data import bar_chart, load_m4_hourly, show

# nbdev-test runs pages in parallel; per-page OpenMP pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
```

``` python
train, test = load_m4_hourly()
NB = "04_Machine_Learning/03_Global_vs_Local"
LAGS = list(range(1, 25)) + [48, 72, 96, 120, 144, 168]
LAG_TRANSFORMS = {24: [SeasonalRollingMean(season_length=24, window_size=7)], 1: [RollingMean(window_size=24)]}
H = 48


def m4_scores(fc: pd.DataFrame, col: str) -> dict:
    "The M4 protocol: per-series sMAPE (percent) and MASE (24 h season), averaged over the 414 series."
    rows = []
    for uid, g in train.groupby("unique_id", sort=False):
        hist = g.y.to_numpy()
        actual = test.loc[test.unique_id == uid, "y"].to_numpy()
        pred = fc.loc[fc.unique_id == uid, col].to_numpy()
        rows.append((100 * smape(actual, pred), np.mean(np.abs(actual - pred)) / np.mean(np.abs(hist[24:] - hist[:-24]))))
    r = np.array(rows)
    return {"mase": float(r[:, 1].mean()), "smape": float(r[:, 0].mean())}
```

## 1. Global: one model for 414 series

30 lags (the last day hour by hour, then each day back to a week), two rolling means, and two per-series transforms applied before training and undone after forecasting:

- **`Differences([24])`**: subtract the same hour yesterday, so the model learns day-on-day *changes*, which look alike across series far more than levels do.
- **`LocalStandardScaler`**: divide each series by its own standard deviation, so a 10 MW series and a 10 kW series contribute equally.

------------------------------------------------------------------------

``` python
def global_model(transforms, name):
    t = time.perf_counter()
    mf = MLForecast(models={"lgbm": lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=127, min_child_samples=50,
                                                       verbose=-1, n_jobs=1, random_state=0)},
                    freq=1, lags=LAGS, lag_transforms=LAG_TRANSFORMS, target_transforms=transforms)
    mf.fit(train, static_features=[])
    sc = m4_scores(mf.predict(H), "lgbm") | {"seconds": time.perf_counter() - t}
    print(f"{name}: MASE {sc['mase']:.3f}, sMAPE {sc['smape']:.2f}, {sc['seconds']:.0f} s")
    return sc


glob = global_model([Differences([24]), LocalStandardScaler()], "global, differenced and scaled")
```

    global, differenced and scaled: MASE 0.887, sMAPE 11.68, 14 s

## 2. Local: one model per series

The same features, transforms and model family, but 414 separate fits. Each local model sees 700 or 960 hours; the global model saw all of them at once. The local trees are smaller (15 leaves instead of 127), because a large tree fitted to one short series memorises it.

------------------------------------------------------------------------

``` python
t = time.perf_counter()
local_fc = []
for uid, g in train.groupby("unique_id", sort=False):
    mf = MLForecast(models={"lgbm": lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=10,
                                                       verbose=-1, n_jobs=1, random_state=0)},
                    freq=1, lags=LAGS, lag_transforms=LAG_TRANSFORMS, target_transforms=[Differences([24]), LocalStandardScaler()])
    mf.fit(g, static_features=[])
    local_fc.append(mf.predict(H))
loc = m4_scores(pd.concat(local_fc), "lgbm") | {"seconds": time.perf_counter() - t}
print(f"local, one model per series: MASE {loc['mase']:.3f}, sMAPE {loc['smape']:.2f}, {loc['seconds']:.0f} s")
```

    local, one model per series: MASE 0.933, sMAPE 12.09, 60 s

## 3. What the transforms are worth

The same global model, trained on raw levels, and on scaled but undifferenced series:

------------------------------------------------------------------------

``` python
glob_raw = global_model([], "global, raw levels")
glob_scaled = global_model([LocalStandardScaler()], "global, scaled only")
```

    global, raw levels: MASE 3.998, sMAPE 12.79, 13 s
    global, scaled only: MASE 1.303, sMAPE 10.96, 12 s

``` python
res = pd.DataFrame({"global, differenced + scaled": glob, "local, differenced + scaled": loc,
                    "global, scaled only": glob_scaled, "global, raw levels": glob_raw}).T
for name, sc in [("LightGBM, global (diff 24 + scaler)", glob), ("LightGBM, local per series (diff 24 + scaler)", loc)]:
    _ = record(name, "machine learning", {"mase": sc["mase"], "smape": sc["smape"], "seconds": sc["seconds"]},
               notebook=NB, dataset="m4_hourly")
res.style.format({"mase": "{:.3f}", "smape": "{:.2f}", "seconds": "{:.0f}"})
```

|                              | mase  | smape | seconds |
|------------------------------|-------|-------|---------|
| global, differenced + scaled | 0.887 | 11.68 | 14      |
| local, differenced + scaled  | 0.933 | 12.09 | 60      |
| global, scaled only          | 1.303 | 10.96 | 12      |
| global, raw levels           | 3.998 | 12.79 | 13      |

``` python
show(bar_chart(res.index, {"MASE": res.mase}, height="320px"))
```

Three results, in order of importance:

- **The transforms matter more than the model.** On raw levels the global model’s MASE is about four times worse: the loss is dominated by the few largest series, and the small ones are fitted badly. A model is only global in a useful sense once the series are on a common scale.
- **With the same features, global beats local** on both metrics, and trains four times faster (one model instead of 414). Against [seasonal naive](../02_Evaluation/benchmarks.llms.md) (MASE 1.193, sMAPE 13.91) it removes about a quarter of the scaled error.
- **The metric picks the transform.** Scaling without differencing gives the best sMAPE of all four and a poor MASE. MASE divides by each series’ day-on-day change, so it rewards a model trained on day-on-day changes; sMAPE is a percentage of the level, so it rewards getting levels right. Decide the metric before tuning anything, as the [Forecast Metrics](../02_Evaluation/forecast_metrics.llms.md) page argued.

The household scoreboard has one series, so global models enter it only indirectly: the deep learning and foundation model sections test models that were trained on many series.

------------------------------------------------------------------------

Back to top
