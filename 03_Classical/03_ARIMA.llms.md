# ARIMA

Autoregression, differencing and moving averages; identifying an order from the ACF and PACF, choosing one by AIC, checking the residuals with Ljung-Box, and adding the weekly season with Fourier terms.

Author

Benedict Thekkel

## The model

**ARIMA(p, d, q)** combines three ideas:

- **AR(p)**, autoregression: the next value is a weighted sum of the last *p* values, plus noise.
- **I(d)**, integration: difference the series *d* times first, to remove a unit root (a random-walk component).
- **MA(q)**, moving average: the next value also depends on the last *q* forecast errors, so the model corrects itself after a surprise.

$$y_t = c + \sum_{i=1}^{p} \phi_i\, y_{t-i} + \sum_{j=1}^{q} \theta_j\, \varepsilon_{t-j} + \varepsilon_t$$

**Seasonal ARIMA**, written $(p,d,q)(P,D,Q)_m$, adds the same three terms at multiples of the season $m$: a seasonal AR term at lag 24 says “this hour depends on the same hour yesterday”. The [stationarity tests](../01_Foundations/decomposition_stationarity.llms.md) found no unit root, so $d = 0$ and $D = 0$ here; the seasonality is modelled by seasonal AR and MA terms rather than removed by differencing.

------------------------------------------------------------------------

``` python
import itertools
import warnings

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import backtest, future_index, origins, record, score
from ML_Timeseries.data import bar_chart, line_chart, load_household, show

# nbdev-test runs pages in parallel; per-page BLAS pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
warnings.filterwarnings("ignore", module="statsmodels")
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "03_Classical/03_ARIMA"
FIT = 24 * 21  # each forecast fits on the last 3 weeks
first_test = origins(y.index)[0]
dev = y.iloc[first_test - FIT:first_test]  # the 3 weeks before the first test origin, for identification
```

## 1. Identification from the ACF and PACF

The classical (Box-Jenkins) recipe reads the order off two plots of the training data:

- an **AR(p)** process has a PACF that cuts off after lag *p* and an ACF that decays;
- an **MA(q)** process has an ACF that cuts off after lag *q* and a PACF that decays;
- seasonal terms show the same patterns at lags 24, 48, …

------------------------------------------------------------------------

``` python
lag = np.arange(0, 73)
show(line_chart(lag, {"ACF": acf(dev, nlags=72), "PACF": pacf(dev, nlags=72)}, y_name="correlation", height="320px", zoom=False))
```

The PACF has one large spike at lag 1 and a smaller one at lag 23, the hour just before the same hour yesterday; everything else is small. The ACF decays and rises again towards 24 and 48. That suggests one or two short AR terms plus a seasonal term at 24. Real series rarely match the textbook patterns cleanly, so the plots narrow the search rather than settle it: something like $(1\text{-}2, 0, 0\text{-}1)(0\text{-}1, 0, 0\text{-}1)_{24}$.

------------------------------------------------------------------------

## 2. Choosing the order by AIC

Automatic ARIMA (Hyndman and Khandakar’s `auto.arima`, and its Python ports) searches orders and keeps the one with the lowest **AIC** (Akaike information criterion): the fit’s log-likelihood, penalised by the number of parameters. Here a small grid over the candidates above, fitted on the same three weeks:

------------------------------------------------------------------------

``` python
aic, orders = {}, {}
for p, q, P, Q in itertools.product((1, 2), (0, 1), (0, 1), (0, 1)):
    if P == Q == 0:
        continue
    name = f"({p},0,{q})({P},0,{Q})24"
    orders[name] = ((p, 0, q), (P, 0, Q, 24))
    aic[name] = SARIMAX(dev.to_numpy(), order=orders[name][0], seasonal_order=orders[name][1]).fit(disp=False).aic
aic = pd.Series(aic).sort_values()
chosen = aic.index[0]
print(f"lowest AIC: {chosen}")
aic.head(5).round(1)
```

    lowest AIC: (2,0,1)(1,0,1)24

    (2,0,1)(1,0,1)24    751.4
    (1,0,0)(1,0,1)24    759.7
    (1,0,1)(1,0,1)24    760.0
    (2,0,0)(1,0,1)24    761.0
    (2,0,1)(1,0,0)24    838.0
    dtype: float64

## 3. Checking the residuals: Ljung-Box

If the model has captured the structure, its residuals should look like white noise: no autocorrelation left. The **Ljung-Box test** asks whether the first *k* residual autocorrelations are jointly zero. A small p-value means structure remains.

------------------------------------------------------------------------

``` python
order, seasonal = orders[chosen]
fit = SARIMAX(dev.to_numpy(), order=order, seasonal_order=seasonal).fit(disp=False)
lb = acorr_ljungbox(fit.resid[24:], lags=[1, 24, 48, 168])
lb.round(3)
```

|     | lb_stat | lb_pvalue |
|-----|---------|-----------|
| 1   | 2.286   | 0.131     |
| 24  | 15.065  | 0.919     |
| 48  | 40.289  | 0.778     |
| 168 | 151.870 | 0.809     |

The residuals pass at every lag tested: no autocorrelation is left that the test can detect. Two cautions before reading that as “the model is good”:

- **White residuals are not small residuals.** They mean the model has used all the *linear* structure in these three weeks. The [decomposition](../01_Foundations/decomposition_stationarity.llms.md) already showed that most of an hour’s variation is unpredictable appliance use; Ljung-Box cannot tell you how large that part is.
- **Three weeks hold only three weekly cycles,** so the test has almost no power at lag 168. A weekly pattern can be there and still pass. A seasonal ARIMA handles one season well; a second, much longer one (168 lags) makes the state space large and the fit slow. Section 5 adds it another way.

------------------------------------------------------------------------

## 4. The backtest

The chosen order is refitted on the last three weeks at every origin, then forecasts 24 hours.

------------------------------------------------------------------------

``` python
def sarima(history, horizon):
    res = SARIMAX(history.iloc[-FIT:].to_numpy(), order=order, seasonal_order=seasonal).fit(disp=False)
    return res.forecast(horizon)


bt = backtest(y, sarima)
arima_score = score(bt, y, exclude=FILLED)
record(f"SARIMA {chosen}", "classical", arima_score, notebook=NB, notes="order by AIC on the 3 weeks before the test; refit on 3 weeks per window")
print(f"SARIMA {chosen}: MASE {arima_score['mase']:.3f}, skill {arima_score['skill']:+.1%}, {arima_score['seconds']:.0f} s for 30 fits")
```

    SARIMA (2,0,1)(1,0,1)24: MASE 0.670, skill +17.2%, 20 s for 30 fits

## 5. A second season with Fourier terms

**Dynamic harmonic regression** keeps the seasonal ARIMA for the daily cycle and adds the weekly cycle as regressors: sine and cosine pairs at the 168-hour period. Four pairs give a smooth weekly shape with 8 parameters instead of 168. The Fourier terms are known for any future hour, so they are legitimate inputs to a forecast.

------------------------------------------------------------------------

``` python
T0 = y.index[0]


def fourier(index, period=168, pairs=4):
    t = (index - T0) / pd.Timedelta("1h")
    return np.column_stack([f(2 * np.pi * k * t / period) for k in range(1, pairs + 1) for f in (np.sin, np.cos)])


def dhr(history, horizon):
    x = history.iloc[-FIT:]
    res = SARIMAX(x.to_numpy(), exog=fourier(x.index), order=order, seasonal_order=seasonal).fit(disp=False)
    return res.forecast(horizon, exog=fourier(future_index(history, horizon)))


dev_fit = SARIMAX(dev.to_numpy(), exog=fourier(dev.index), order=order, seasonal_order=seasonal).fit(disp=False)
lb_dhr = acorr_ljungbox(dev_fit.resid[24:], lags=[24, 168])
print("Ljung-Box p-values with weekly Fourier terms:", lb_dhr.lb_pvalue.round(4).to_dict())
bt_dhr = backtest(y, dhr)
dhr_score = score(bt_dhr, y, exclude=FILLED)
record(f"SARIMA {chosen} + weekly Fourier", "classical", dhr_score, notebook=NB, notes="4 Fourier pairs at 168 h as regressors")
print(f"with weekly Fourier terms: MASE {dhr_score['mase']:.3f}, skill {dhr_score['skill']:+.1%}")
```

    Ljung-Box p-values with weekly Fourier terms: {24: 0.893, 168: 0.7869}
    with weekly Fourier terms: MASE 0.695, skill +14.0%

The weekly terms make the forecast slightly *worse*. Three weeks of history give each of the eight Fourier coefficients only three weekly cycles to learn from, so they mostly fit that period’s accidents, and the extra variance costs more than the weekly signal gains. A longer fit window would give the Fourier terms more to learn from and make every fit slower. On this series the seasonal AR term at lag 24 plus two short-term AR terms already capture most of what a day-ahead forecast can use, which matches the [autocorrelation](../01_Foundations/decomposition_stationarity.llms.md) finding that lag 24 and lag 168 carry similar information.

------------------------------------------------------------------------

``` python
w = bt.window.max()
a_, b_ = bt[bt.window == w], bt_dhr[bt_dhr.window == w]
show(line_chart(a_.ds.dt.strftime("%a %H:00"), {"actual": a_.y, "SARIMA": a_.yhat, "SARIMA + weekly Fourier": b_.yhat},
                y_name="kWh", height="320px", zoom=False))
```

## Summary

| Step | Tool | Lesson here |
|----|----|----|
| Identify | ACF, PACF | short AR terms plus a seasonal term at 24 |
| Select | AIC over a small grid | the plots narrow the search; AIC decides |
| Check | Ljung-Box | residuals pass, which says nothing about how large they are |
| Extend | Fourier regressors | with 3 weeks of data, the weekly terms add variance and cost skill |

SARIMA is among the best models on the [scoreboard](../02_Evaluation/scoreboard.llms.md), from a model with a handful of parameters fitted on three weeks. It is also slow next to the baselines and ETS, about 0.7 seconds per fit, which matters when a forecast has to be produced for thousands of meters.

------------------------------------------------------------------------

Back to top
