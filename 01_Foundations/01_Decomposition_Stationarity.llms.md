# Decomposition and Stationarity

Trend, seasonality and noise in one household’s consumption: MSTL, the periodogram, autocorrelation, and what the ADF and KPSS tests do and do not say.

Author

Benedict Thekkel

## What a series is made of

Most forecasting methods assume a series is a sum of parts:

$$y_t = T_t + S_t + R_t$$

- **Trend** $T_t$: the slow level, here the household’s general consumption over months.
- **Seasonality** $S_t$: patterns that repeat on a fixed period. Here there are three: the day (24 hours), the week (168 hours) and the year.
- **Remainder** $R_t$: everything else, including the unpredictable.

Decomposing first tells you which parts a model must capture and how much of the series is predictable at all. If the remainder holds most of the variance, no model will forecast well, and the remainder sets the floor on error.

------------------------------------------------------------------------

``` python
import warnings

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from ML_Timeseries.data import bar_chart, line_chart, load_household, show
```

``` python
hh = load_household()
y = hh["kwh"]
year = y["2009-11-01":"2010-10-31"]
daily = load_household("D")["kwh"]
print(f"hourly: {len(y):,} points, one year for MSTL: {len(year):,}; daily: {len(daily):,}")
```

    hourly: 34,589 points, one year for MSTL: 8,760; daily: 1,442

## 1. MSTL: several seasons at once

STL (Seasonal-Trend decomposition using Loess, Cleveland et al., 1990) splits a series into trend, one seasonal component and a remainder by repeated local regression. It handles a season whose shape drifts over time, which classical decomposition (a fixed average per hour) cannot. **MSTL** (Bandara et al., 2021) runs STL once per seasonal period, so the daily and weekly cycles get separate components.

Here it runs on one year of hourly data with periods of 24 and 168 hours.

------------------------------------------------------------------------

``` python
mstl = MSTL(year, periods=(24, 168)).fit()
parts = pd.DataFrame({"trend": mstl.trend, "daily": mstl.seasonal["seasonal_24"],
                      "weekly": mstl.seasonal["seasonal_168"], "remainder": mstl.resid})
win = parts["2010-01-04":"2010-01-10"]
labels = win.index.strftime("%a %H:00")
show(line_chart(labels, {"actual": year.loc[win.index], "trend + daily + weekly": win.trend + win.daily + win.weekly},
                y_name="kWh", height="340px"))
```

The gap between the two lines is the remainder. This is an in-sample fit: MSTL lets the seasonal shapes drift week by week, so they follow the data more closely here than any forecast made in advance could. Even so, the misses are the sharp spikes, which are appliances switched on at times the calendar cannot predict. The two seasonal components on their own:

``` python
show(line_chart(labels, {"daily (24 h)": win.daily, "weekly (168 h)": win.weekly}, y_name="kWh", height="300px"))
```

How much each part matters is measured by its **strength** (Wang, Smith and Hyndman, 2006): one minus the remainder’s variance as a share of the component plus remainder. 0 means the component adds nothing over noise; 1 means it explains all the variation the remainder would otherwise hold.

$$F_S = \max\left(0,\ 1 - \frac{\operatorname{Var}(R_t)}{\operatorname{Var}(S_t + R_t)}\right)$$

``` python
def strength(component, remainder):
    return max(0.0, 1 - remainder.var() / (component + remainder).var())


annual = STL(daily, period=365, robust=True).fit()
strengths = pd.Series({
    "daily (24 h)": strength(parts.daily, parts.remainder),
    "weekly (168 h)": strength(parts.weekly, parts.remainder),
    "annual (on daily totals)": strength(annual.seasonal, annual.resid),
}, name="strength")
print(f"remainder std {parts.remainder.std():.3f} kWh of total std {year.std():.3f} kWh")
strengths.round(3)
```

    remainder std 0.462 kWh of total std 0.812 kWh

    daily (24 h)                0.568
    weekly (168 h)              0.277
    annual (on daily totals)    0.582
    Name: strength, dtype: float64

The daily cycle is the strongest part of the hourly series, the weekly cycle is weaker, and the annual cycle shows clearly on daily totals (winter heating and lighting). The remainder’s standard deviation is more than half the series’ own: an hour in this house is noisy, because a kettle, an oven or a water heater switches on when someone decides to use it. Any hourly model will leave a large share of the error unexplained.

------------------------------------------------------------------------

## 2. The periodogram: which periods carry power

A discrete Fourier transform rewrites the series as a sum of sine waves. The **periodogram** is the power of each wave. A seasonal period shows as a peak, and so do its **harmonics** (period / 2, period / 3, …), because a daily cycle with a sharp morning and evening peak is not a sine wave and needs several to draw.

------------------------------------------------------------------------

``` python
x = y.to_numpy() - y.mean()
power = np.abs(np.fft.rfft(x)) ** 2
period = 1 / np.fft.rfftfreq(len(x), d=1.0)[1:]
power = power[1:]
band = (period >= 4) & (period <= 400)
peaks, _ = find_peaks(power[band])
cand = pd.DataFrame({"period_h": period[band][peaks], "power": power[band][peaks]}).sort_values("power", ascending=False)
top = []
for p, pw in cand.itertuples(index=False):
    if all(abs(p - q) / q > 0.03 for q, _ in top):
        top.append((p, pw))
    if len(top) == 8:
        break
top = pd.DataFrame(top, columns=["period_h", "power"])
top["share"] = top.power / top.power.sum()
show(bar_chart([f"{p:.0f} h" for p in top.period_h], {"share of peak power": top.share}, height="320px"))
```

The two largest peaks are 12 and 24 hours: the day and its first harmonic (a morning peak and an evening peak, 12 hours apart). Next come 6 hours (the second harmonic), then the week (168 hours) and its own harmonic at 84 hours. These are the periods MSTL was given, found here from the data alone. The FFT finds the seasons; the decomposition then separates them.

------------------------------------------------------------------------

## 3. Autocorrelation

The **autocorrelation function** (ACF) is the correlation between the series and itself shifted by *k* hours. It is the most direct view of which lags a model should use. The **partial** autocorrelation (PACF) removes what shorter lags already explain, so it shows which lags add new information.

------------------------------------------------------------------------

``` python
lags = np.arange(0, 401)
a = acf(y, nlags=400, fft=True)
show(line_chart(lags, {"ACF": a}, y_name="correlation", height="320px"))
```

``` python
p = pacf(y, nlags=48)
print("ACF at 1, 12, 24, 168, 336 h:", {k: round(float(a[k]), 3) for k in (1, 12, 24, 168, 336)})
print("largest PACF after lag 1:", {int(k): round(float(p[k]), 3) for k in np.argsort(-np.abs(p[2:]))[:4] + 2})
```

    ACF at 1, 12, 24, 168, 336 h: {1: 0.713, 12: 0.239, 24: 0.436, 168: 0.463, 336: 0.431}
    largest PACF after lag 1: {23: 0.162, 22: 0.145, 10: 0.107, 47: 0.101}

Three things to read from it:

- **Lag 1 is by far the highest.** The next hour looks like this hour. That is useless for a day-ahead forecast, which must start from lag 24.
- **Lag 168 beats lag 24, slightly.** Over four years, the same hour last week correlates a little better with this hour than the same hour yesterday, because a weekly lag matches both the hour of the day *and* the day of the week. Correlation over the whole series is not forecast accuracy over a test period, though: on the [Baselines](../03_Classical/baselines.llms.md) page, copying last week loses to copying yesterday, because a week-old copy also carries a week-old level.
- **The peaks decay slowly.** Lag 336 (two weeks) is still about as strong as lag 168. The seasonal pattern is stable, which favours models that average several past seasons over ones that copy one.

------------------------------------------------------------------------

## 4. Stationarity: ADF and KPSS

A series is **stationary** when its mean, variance and autocorrelation do not change over time. ARIMA-type models assume it after differencing, so the classical workflow tests for it first. Two tests ask opposite questions:

| Test | Null hypothesis | Small p-value means |
|----|----|----|
| **ADF** (augmented Dickey-Fuller) | the series has a unit root (a random walk part, never returns to a mean) | no unit root |
| **KPSS** (Kwiatkowski-Phillips-Schmidt-Shin) | the series is stationary around a constant level | not stationary |

Used together they separate three cases: both say stationary; both say a unit root, so difference it; or they disagree, which usually means stationary around something the test does not model, such as seasonality.

------------------------------------------------------------------------

``` python
cases = {"hourly, one year": year, "daily totals": daily,
         "daily, differenced": daily.diff().dropna(), "daily, minus last week": daily.diff(7).dropna()}
rows = []
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # KPSS warns when its p-value is outside its lookup table (0.01 to 0.1)
    for name, s in cases.items():
        rows.append({"series": name, "ADF p": adfuller(s, autolag="AIC")[1], "KPSS p": kpss(s, regression="c", nlags="auto")[1]})
tests = pd.DataFrame(rows).set_index("series")
tests.style.format("{:.3f}")
```

|                        | ADF p | KPSS p |
|------------------------|-------|--------|
| series                 |       |        |
| hourly, one year       | 0.000 | 0.010  |
| daily totals           | 0.005 | 0.100  |
| daily, differenced     | 0.000 | 0.100  |
| daily, minus last week | 0.000 | 0.100  |

Reading the table:

- **Daily totals**: ADF rejects a unit root and KPSS does not reject stationarity. Daily consumption returns to its level; it is not a random walk, so differencing it would only add noise.
- **Hourly**: ADF rejects a unit root, and KPSS rejects stationarity too. The two disagree because the hourly series has a strong, deterministic daily cycle, which KPSS reads as a changing mean. The fix is to model the seasonality (seasonal terms, Fourier terms, seasonal lags), not to difference.
- KPSS reports p-values only between 0.01 and 0.1 (it interpolates a table), so 0.010 means “0.01 or less” and 0.100 means “0.1 or more”.

Stationarity tests matter most for ARIMA. Tree models and neural networks do not assume stationarity, but they still forecast badly outside the range they were trained on, so a trend still has to be handled.

------------------------------------------------------------------------

## What this means for the models

| Finding | Consequence |
|----|----|
| Daily and weekly seasons, with harmonics | Fourier terms at 24 h and 168 h are good features; averaging several aligned past weeks beats copying one |
| Lag 168, 24 and 336 all similar | Use several seasonal lags, not one |
| Large remainder | Expect skill well below 50% against seasonal naive for any model |
| No unit root, deterministic seasons | Model seasonality explicitly; avoid differencing the level |
| Annual cycle on daily totals | A model trained on summer will under-forecast winter; the backtest’s test span (June to November) runs into the start of winter |

------------------------------------------------------------------------

Back to top
