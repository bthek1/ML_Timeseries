# Drift and Retraining

A deployed forecaster ages: how often to retrain, on how much history, and how to tell real drift from the calendar turning. Measured on the household backtest with LightGBM.

Author

Benedict Thekkel

## Two decisions after deployment

Every model on this site so far trained once, before the first test origin, and then forecast five months without learning anything new. A deployed model faces two decisions:

1.  **Retraining schedule**: how often to refit, and on how much history (everything so far, or a sliding window of recent data).
2.  **Drift monitoring**: how to notice that the data has changed enough to matter, before the error does.

The model is LightGBM with the [shared features](../01_Foundations/feature_engineering.llms.md), the same at every refit.

------------------------------------------------------------------------

``` python
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import backtest, future_index, origins, score
from ML_Timeseries.data import bar_chart, line_chart, load_household, show
from ML_Timeseries.features import make_features

# nbdev-test runs pages in parallel; per-page OpenMP pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
O = origins(y.index)
X = make_features(y)
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=1, random_state=0)
```

## 1. Retraining schedules

Five schedules. At each refit the model sees only data before the current origin.

------------------------------------------------------------------------

``` python
def schedule(every_days, window_days=None):
    models = {}

    def f(history, horizon):
        o = len(history)
        key = (o - O[0]) // (24 * every_days) if every_days else 0
        if key not in models:
            end = o if every_days else O[0]
            start = max(700, end - window_days * 24) if window_days else 700
            models[key] = lgb.LGBMRegressor(**PARAMS).fit(X.iloc[start:end], y.iloc[start:end])
        ext = pd.concat([history.iloc[-800:], pd.Series(np.nan, index=future_index(history, horizon))])
        return models[key].predict(make_features(ext).iloc[-horizon:])
    return f


SCHEDULES = {"train once": (0, None), "every 30 days, all history": (30, None), "every 5 days, all history": (5, None),
             "every 5 days, last 365 days": (5, 365), "every 5 days, last 90 days": (5, 90)}
rows = {}
for name, (every, window) in SCHEDULES.items():
    t0 = time.perf_counter()
    sc = score(backtest(y, schedule(every, window)), y, exclude=FILLED)
    rows[name] = {"skill": sc["skill"], "MASE": sc["mase"], "seconds": time.perf_counter() - t0}
sched = pd.DataFrame(rows).T
show(bar_chart(sched.index, {"skill vs seasonal naive": sched.skill}, horizontal=True, height="300px"))
```

``` python
sched.style.format({"skill": "{:+.1%}", "MASE": "{:.3f}", "seconds": "{:.0f}"})
```

|                             | skill  | MASE  | seconds |
|-----------------------------|--------|-------|---------|
| train once                  | +16.5% | 0.668 | 1       |
| every 30 days, all history  | +17.0% | 0.667 | 4       |
| every 5 days, all history   | +17.5% | 0.657 | 22      |
| every 5 days, last 365 days | +14.5% | 0.688 | 12      |
| every 5 days, last 90 days  | +11.1% | 0.714 | 9       |

**Refitting helps, a little.** Going from one model to a refit every five days adds about one point of skill, for twenty times the training compute. Monthly refits capture half of that gain at a fifth of the cost.

**Forgetting hurts, a lot.** Training only on the last 365 days costs three points; the last 90 days costs six. Recent data is not more relevant here, it is less complete: the annual cycle needs more than a year of history to be learned at all, and a model fitted on summer has never seen a winter evening. “Retrain on recent data” is common advice for drifting series; for a strongly seasonal one it is the wrong default. Keep all history and refit on a fixed schedule, unless monitoring shows the old data has become misleading.

------------------------------------------------------------------------

## 2. Measuring drift: PSI

The **population stability index** (PSI) compares two distributions binned by the reference’s deciles:

$$\text{PSI} = \sum_{i} (p_i - q_i)\,\ln\frac{p_i}{q_i}$$

where $q_i$ and $p_i$ are the shares of the reference and the new data in bin *i*. A common rule of thumb from credit scoring: below 0.1 is stable, 0.1 to 0.25 is a moderate shift, above 0.25 is a large one.

The question that matters for a seasonal series is **compared with what?** Each test month is compared twice: with the whole previous year, and with the same month a year earlier.

------------------------------------------------------------------------

``` python
def psi(reference, new, bins=10):
    edges = np.quantile(reference, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    q = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-4, None)
    p = np.clip(np.histogram(new, edges)[0] / len(new), 1e-4, None)
    return float(np.sum((p - q) * np.log(p / q)))


previous_year = y["2009-06-01":"2010-05-31"].to_numpy()
months = pd.period_range("2010-06", "2010-11", freq="M")
drift = pd.DataFrame({
    "PSI vs previous 12 months": [psi(previous_year, y[str(m)].to_numpy()) for m in months],
    "PSI vs same month last year": [psi(y[str(m - 12)].to_numpy(), y[str(m)].to_numpy()) for m in months],
}, index=[m.strftime("%b %Y") for m in months])
show(bar_chart(drift.index, {c: drift[c] for c in drift}, y_name="PSI", height="320px"))
```

``` python
drift.style.format("{:.2f}")
```

|          | PSI vs previous 12 months | PSI vs same month last year |
|----------|---------------------------|-----------------------------|
| Jun 2010 | 0.11                      | 0.08                        |
| Jul 2010 | 0.35                      | 0.32                        |
| Aug 2010 | 0.89                      | 0.16                        |
| Sep 2010 | 0.11                      | 0.08                        |
| Oct 2010 | 0.35                      | 0.21                        |
| Nov 2010 | 0.04                      | 0.05                        |

Against the previous twelve months, August 2010 looks like a severe shift (PSI 0.89). Against August 2009 it is moderate (0.16). Most of the “drift” is the calendar: August is a holiday month in France, and this household’s August always looks unlike its year. A drift monitor that compares with the last twelve months would page someone every summer for something entirely expected.

July 2010 is different: it is far from the previous year **and** from July 2009 (0.32). That is a genuine change in behaviour, and it matches the [matrix profile](../08_Other_Tasks/matrix_profile.llms.md), where three of the five most unusual windows of the year fell in July 2010.

Practical rules that follow:

- **Compare like with like**: same season, same weekday mix, or residuals of a seasonal model, never raw values against an all-year baseline.
- **Monitor the error, not only the inputs.** Input drift that does not raise the forecast error needs no action; the [backtest](../02_Evaluation/backtesting.llms.md) machinery run on each new week is the most direct monitor there is.
- **Tie retraining to the schedule, and drift to an investigation.** A refit cannot fix a broken meter or a new tenant; a person looking at the data can.

------------------------------------------------------------------------

Back to top
