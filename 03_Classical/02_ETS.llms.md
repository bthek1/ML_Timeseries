# Exponential Smoothing (ETS)

Level, trend and season updated by exponentially weighted averages: why the default fit loses to seasonal naive on hourly data, and how choosing the smoothing for the real horizon turns it into one of the best classical models here.

Author

Benedict Thekkel

## The idea

Exponential smoothing forecasts with a weighted average of the past in which the weights shrink geometrically: the last hour counts most, the one before a little less, and so on. One parameter, $\alpha$ between 0 and 1, sets how fast the weights shrink. A large $\alpha$ follows the data closely; a small one averages over a long memory.

**Holt-Winters** applies the same idea to three states at once. With an additive season of length $m$:

$$\ell_t = \alpha\,(y_t - s_{t-m}) + (1-\alpha)\,\ell_{t-1}$$

$$s_t = \gamma\,(y_t - \ell_t) + (1-\gamma)\,s_{t-m}$$

$$\hat y_{t+h} = \ell_t + s_{t+h-m}$$

The level $\ell$ is the deseasonalised average; the season $s$ is a profile of $m$ values, one per hour of the cycle, each updated only when its hour comes round. A trend state can be added the same way. The **ETS** framework (Error, Trend, Season; Hyndman et al., 2002) writes all these variants as state space models, which gives them likelihoods, prediction intervals and automatic model selection.

------------------------------------------------------------------------

``` python
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import backtest, origins, record, score, seasonal_naive
from ML_Timeseries.data import line_chart, load_household, show

# nbdev-test runs pages in parallel; per-page BLAS pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
warnings.filterwarnings("ignore", module="statsmodels")
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "03_Classical/02_ETS"
WINDOW = 24 * 7 * 8  # each forecast fits on the last 8 weeks
```

## 1. The default fit

statsmodels estimates $\alpha$ and $\gamma$ by minimising the **one-step-ahead** squared error on the training window. Here that is the wrong objective, because the forecast is used 1 to 24 steps ahead.

------------------------------------------------------------------------

``` python
fitted = []


def holt_winters(m, alpha=None, gamma=None):
    "Additive Holt-Winters on the last 8 weeks; smoothing estimated unless given."
    def f(history, horizon):
        model = ExponentialSmoothing(history.iloc[-WINDOW:].to_numpy(), seasonal="add", seasonal_periods=m,
                                     initialization_method="estimated")
        kw = {} if alpha is None else {"smoothing_level": alpha, "smoothing_seasonal": gamma}
        res = model.fit(**kw)
        fitted.append((res.params["smoothing_level"], res.params["smoothing_seasonal"]))
        return res.forecast(horizon)
    return f


default = {}
for m in (24, 168):
    fitted.clear()
    default[m] = score(backtest(y, holt_winters(m)), y, exclude=FILLED)
    a, g = np.median(fitted, axis=0)
    print(f"season {m:3d} h: skill {default[m]['skill']:+.1%}; median fitted alpha {a:.2f}, gamma {g:.2f}")
_ = record("Holt-Winters (24 h), default fit", "classical", default[24], notebook=NB)
```

    season  24 h: skill -13.4%; median fitted alpha 0.59, gamma 0.00
    season 168 h: skill -9.6%; median fitted alpha 0.62, gamma 0.00

Both lose to seasonal naive. The fitted $\alpha$ is around 0.6: the level jumps most of the way to every new hour. The fitted $\gamma$ is 0: the season profile is frozen at its initial estimate and never learns from the eight weeks it is given. For a one-step forecast that is sensible, because the next hour does look like this one (the lag-1 autocorrelation is 0.71). For a forecast of tomorrow evening, the level is then whatever the last hour before midnight happened to be, plus a season profile that was barely updated. The model is excellent at a question nobody asked.

------------------------------------------------------------------------

## 2. Smoothing chosen for the real horizon

The fix is to choose $\alpha$ and $\gamma$ by the error that matters: 24-step forecasts on **validation windows** that end before the first test origin, exactly the honest procedure from [leak 5](../02_Evaluation/backtesting.llms.md). Then the chosen pair runs once on the test windows.

------------------------------------------------------------------------

``` python
first_test = origins(y.index)[0]
val_y = y.iloc[:first_test]
grid = [(a, g) for a in (0.005, 0.01, 0.02, 0.05, 0.1, 0.3) for g in (0.02, 0.05, 0.1, 0.2, 0.3)]
val = {}
for m in (24, 168):
    for a, g in grid:
        val[(m, a, g)] = score(backtest(val_y, holt_winters(m, a, g)), val_y, exclude=FILLED)["mae"]
val = pd.Series(val, name="validation MAE")
best = {m: val[m].idxmin() for m in (24, 168)}
tuned = {m: score(backtest(y, holt_winters(m, *best[m])), y, exclude=FILLED) for m in (24, 168)}
for m in (24, 168):
    print(f"season {m:3d} h: chosen alpha {best[m][0]}, gamma {best[m][1]} -> test skill {tuned[m]['skill']:+.1%}")
m_best = max(tuned, key=lambda m: tuned[m]["skill"])
_ = record(f"Holt-Winters ({m_best} h), smoothing tuned for 24 steps", "classical", tuned[m_best], notebook=NB,
       notes=f"alpha {best[m_best][0]}, gamma {best[m_best][1]}, chosen on validation windows")
```

    season  24 h: chosen alpha 0.005, gamma 0.05 -> test skill +12.3%
    season 168 h: chosen alpha 0.005, gamma 0.2 -> test skill +7.1%

With the smoothing chosen for the real task, Holt-Winters swings from losing to seasonal naive to beating every baseline by a wide margin. (Choosing the pair on the test windows instead would have reported about 16%; the honest validation choice gives a little less, which is [leak 5](../02_Evaluation/backtesting.llms.md) measured again.) Nothing about the model changed; only the objective used to fit it. This is the most general lesson on this page: **fit a forecaster on the horizon you will use it at.** It applies equally to ARIMA, to tree models trained on one-step targets, and to neural networks.

A small $\alpha$ makes the level a long average, so a single unusual hour barely moves it, and the season adapts slowly enough to average several weeks, much like the seasonal window median on the [Baselines](../03_Classical/baselines.llms.md) page.

------------------------------------------------------------------------

``` python
bt_default = backtest(y, holt_winters(24))
bt_tuned = backtest(y, holt_winters(m_best, *best[m_best]))
w = bt_tuned.window.max()
a_, b_ = bt_default[bt_default.window == w], bt_tuned[bt_tuned.window == w]
show(line_chart(b_.ds.dt.strftime("%a %H:00"), {"actual": b_.y, "default fit": a_.yhat, "tuned for 24 steps": b_.yhat},
                y_name="kWh", height="320px", zoom=False))
```

## 3. Where ETS is at home: daily totals

ETS was designed for series like monthly sales or daily totals: smooth, with one season, where the next value really does depend mostly on the recent level. The same method on this house’s **daily** energy, forecasting 7 days ahead over 30 weekly windows, with automatic estimation and a damped trend:

------------------------------------------------------------------------

``` python
daily = load_household("D")["kwh"]


def ets_daily(history, horizon):
    model = ETSModel(history.iloc[-365:].to_numpy(), error="add", trend="add", damped_trend=True,
                     seasonal="add", seasonal_periods=7)
    return model.fit(disp=False).forecast(horizon)


bt_d = backtest(daily, ets_daily, horizon=7, n_windows=30, step=7)
daily_ets = score(bt_d, daily, m=7)
print(f"daily totals, 7 days ahead: ETS(A,Ad,A) skill {daily_ets['skill']:+.1%} against repeating last week")
```

    daily totals, 7 days ahead: ETS(A,Ad,A) skill +19.0% against repeating last week

On daily totals the default fit works, because one step ahead and the forecast horizon are the same kind of question: tomorrow’s total depends on today’s level. This result is on a different series, so it is not on the hourly scoreboard.

| Data | Default ETS | Why |
|----|----|----|
| Hourly, 24 steps ahead | loses to seasonal naive | the one-step fit chases noise |
| Hourly, smoothing tuned for 24 steps | strong | the objective matches the use |
| Daily totals, 7 days ahead | beats repeating last week | smooth series, level matters |

------------------------------------------------------------------------

Back to top
