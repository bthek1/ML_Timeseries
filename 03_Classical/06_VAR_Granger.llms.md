# VAR and Granger Causality

Several series that may drive each other: vector autoregression on the house’s four circuits, Granger tests between them, and why “every circuit Granger-causes every other” says more about the household than about the wiring.

Author

Benedict Thekkel

## The model

A **vector autoregression** VAR(p) forecasts several series together. Each series is a linear function of the last *p* values of *every* series:

$$\mathbf{y}_t = \mathbf{c} + A_1 \mathbf{y}_{t-1} + \dots + A_p \mathbf{y}_{t-p} + \boldsymbol{\varepsilon}_t$$

With *k* series that is $k^2 p$ coefficients, so VARs suit a handful of related series with plenty of history, such as macroeconomic indicators or, here, the four circuits of one house at daily resolution:

| Series                            | Mean kWh per day |
|-----------------------------------|------------------|
| kitchen (sub-meter 1)             | about 1.6        |
| laundry (sub-meter 2)             | about 1.9        |
| water heater and AC (sub-meter 3) | about 9.3        |
| everything else                   | about 13.4       |

------------------------------------------------------------------------

``` python
import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import grangercausalitytests
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import backtest, origins, score, seasonal_naive
from ML_Timeseries.data import bar_chart, load_household, show

# nbdev-test runs pages in parallel; per-page BLAS pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
warnings.filterwarnings("ignore", module="statsmodels")
```

``` python
d = load_household("D")
X = d[["sub1_kwh", "sub2_kwh", "sub3_kwh", "other_kwh"]].set_axis(["kitchen", "laundry", "heater_ac", "other"], axis=1)
total, filled = d["kwh"], d["filled"]
train = X.loc[:total.index[origins(total.index, horizon=7, n_windows=30, step=7)[0] - 1]]
print(f"{len(X)} days; Granger tests and order selection use the {len(train)} days before the first test window")
X.mean().round(2)
```

    1442 days; Granger tests and order selection use the 1232 days before the first test window

    kitchen       1.61
    laundry       1.85
    heater_ac     9.29
    other        13.39
    dtype: float64

## 1. Granger causality

Series A **Granger-causes** B (Granger, 1969) if A’s past improves the forecast of B beyond what B’s own past gives. The test fits two regressions of B, with and without lags of A, and F-tests the difference. It is a statement about **predictive information**, not about cause: two series driven by a third will Granger-cause each other.

Each cell below is the smallest p-value over lags 1 to 7 for “row Granger-causes column”.

------------------------------------------------------------------------

``` python
pv = {}
for target in X:
    for source in X:
        if source == target:
            continue
        with contextlib.redirect_stdout(io.StringIO()):  # the function prints every test
            res = grangercausalitytests(train[[target, source]], maxlag=7)
        pv[(source, target)] = min(res[k][0]["ssr_ftest"][1] for k in res)
granger = pd.Series(pv).unstack()
granger.style.format("{:.4f}", na_rep="")
```

|           | heater_ac | kitchen | laundry | other  |
|-----------|-----------|---------|---------|--------|
| heater_ac |           | 0.0000  | 0.0002  | 0.0000 |
| kitchen   | 0.0232    |         | 0.0001  | 0.0000 |
| laundry   | 0.0008    | 0.0000  |         | 0.0021 |
| other     | 0.0000    | 0.0000  | 0.0000  |        |

Every circuit Granger-causes every other, at the 5% level and mostly far below it. The wiring did not do that. A common cause did: **whether the household is at home and active**. A busy day raises cooking, washing, hot water and lighting together; a holiday week lowers all four. Each circuit’s past is therefore a noisy measurement of that hidden activity level, and so it helps predict every other circuit.

This is the standard trap with Granger tests on observational data. The result is real and useful for forecasting (a busy kitchen yesterday does carry information), and it says nothing about intervening (switching off the kitchen would not lower the water heater’s use).

------------------------------------------------------------------------

## 2. Does the VAR forecast the total better?

If the circuits carry information about each other, forecasting them jointly and summing might beat forecasting the total directly. Seven days ahead, 30 weekly windows over the last 210 days, fitted on the last year at each origin:

------------------------------------------------------------------------

``` python
sel = VAR(train.iloc[-365:].to_numpy()).select_order(14)
print(f"VAR order chosen by AIC: {sel.aic}, by BIC: {sel.bic}")


def var_total(order):
    def f(history, horizon):
        hist = X.loc[:history.index[-1]].iloc[-365:].to_numpy()
        return VAR(hist).fit(order).forecast(hist[-order:], horizon).sum(axis=1)
    return f


def ar_total(history, horizon):
    return AutoReg(history.iloc[-365:].to_numpy(), lags=7).fit().forecast(horizon)


candidates = {"seasonal naive (7 days)": seasonal_naive(7), "AR(7) on the total": ar_total,
              f"VAR({sel.bic}) on the circuits, summed": var_total(sel.bic), "VAR(7) on the circuits, summed": var_total(7)}
daily_scores = pd.DataFrame({n: score(backtest(total, f, horizon=7, n_windows=30, step=7), total, m=7, exclude=filled)
                             for n, f in candidates.items()}).T[["mase", "skill", "mae"]]
daily_scores.style.format({"mase": "{:.3f}", "skill": "{:+.1%}", "mae": "{:.2f}"})
```

    VAR order chosen by AIC: 9, by BIC: 2

|                                | mase  | skill  | mae  |
|--------------------------------|-------|--------|------|
| seasonal naive (7 days)        | 0.832 | +0.0%  | 5.68 |
| AR(7) on the total             | 0.652 | +21.4% | 4.46 |
| VAR(2) on the circuits, summed | 0.684 | +16.5% | 4.74 |
| VAR(7) on the circuits, summed | 0.669 | +19.4% | 4.58 |

``` python
order = daily_scores.sort_values("skill")
show(bar_chart(order.index, {"skill vs repeating last week": order.skill}, horizontal=True, height="300px"))
```

The VAR does **not** beat a plain autoregression on the total. The circuits’ cross-information is real (section 1), but the total already contains it: when the household is busy, the total rises, and the total’s own lags carry that signal. The VAR spends 16 coefficients per lag to rediscover it from four noisier series, and the extra variance costs more than the information gains. AIC asked for 9 lags (144 coefficients from one year of days) and BIC for 2. BIC’s model forecasts worst of the three: two days of history cannot see the weekly cycle. Neither order reaches the plain AR on the total.

When does a VAR pay? When the extra series carry information the target does not: an upstream sensor that moves before the target does, a price that leads demand, a weather series. Splitting one quantity into its parts rarely helps forecast the sum. These are daily results on a 7-day horizon, so they are not on the hourly scoreboard.

------------------------------------------------------------------------

Back to top
