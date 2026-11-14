# State Space Models and the Kalman Filter

One framework under ETS and ARIMA: a hidden state that evolves, noisy observations of it, and the Kalman filter that tracks it. Plus the proof, in numbers, that exponential smoothing is a Kalman filter.

Author

Benedict Thekkel

## The model

A **state space model** separates what you want to know from what you can measure:

The **state equation** says how the hidden state evolves:

$$x_t = F\,x_{t-1} + w_t, \quad w_t \sim N(0, Q)$$

The **observation equation** says what the meter reads:

$$y_t = H\,x_t + v_t, \quad v_t \sim N(0, R)$$

For this house, the hidden state could be the household’s underlying consumption level and its daily and weekly profile; the observation is that plus the appliance noise of each hour. ETS and ARIMA are both special cases with particular choices of $F$, $H$, $Q$ and $R$, which is how statsmodels fits them.

The **Kalman filter** (Kalman, 1960) tracks the state through two steps per observation:

1.  **Predict**: push the state estimate and its uncertainty through $F$. Uncertainty grows by $Q$.
2.  **Update**: compare the prediction with the new observation and move the estimate by the **Kalman gain** $K$ times the error. $K$ is large when the state is uncertain and the meter is precise, small when the meter is noisy.

This is the same filter that fuses wheel odometry and IMU readings in a robot (the EKF in ROS 2’s `robot_localization` is its nonlinear extension). Here the “sensor” is the meter and the “pose” is the household’s consumption level.

------------------------------------------------------------------------

``` python
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.structural import UnobservedComponents
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import backtest, record, score
from ML_Timeseries.data import line_chart, load_household, show

# nbdev-test runs pages in parallel; per-page BLAS pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
warnings.filterwarnings("ignore", module="statsmodels")
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
daily = load_household("D")["kwh"]
NB = "03_Classical/05_State_Space_Kalman"
```

## 1. A Kalman filter in twelve lines

The simplest state space model is the **local level**: the state is one number, a level that wanders as a random walk, observed with noise.

$$\ell_t = \ell_{t-1} + w_t \qquad y_t = \ell_t + v_t$$

So $F = H = 1$, and the filter is scalar arithmetic. Run on the daily totals with a level noise $Q$ and a measurement noise $R$:

------------------------------------------------------------------------

``` python
def kalman_local_level(obs, Q, R, level0=None, P0=1e6):
    "Scalar Kalman filter for a local level model; returns filtered levels and gains."
    level = obs[0] if level0 is None else level0
    P = P0
    levels, gains = [], []
    for z in obs:
        P = P + Q                   # predict: the level may have moved, so uncertainty grows
        K = P / (P + R)             # gain: trust the observation in proportion to our uncertainty
        level = level + K * (z - level)  # update: move towards the observation
        P = (1 - K) * P             # the observation reduced the uncertainty
        levels.append(level)
        gains.append(K)
    return np.array(levels), np.array(gains)


x = daily.to_numpy()
R = np.var(np.diff(x)) / 2      # a rough split of the day-to-day variance into noise...
Q = R / 50                      # ...and a level that moves slowly relative to it
levels, gains = kalman_local_level(x, Q, R)
show(line_chart(daily.index.date, {"daily kWh": x, "Kalman level": levels}, y_name="kWh", height="340px"))
```

## 2. Exponential smoothing is a Kalman filter

Watch the gain. It starts near 1 (the first observations are all the filter has) and settles to a constant within a few weeks. Once it is constant, the update step reads

$$\ell_t = \ell_{t-1} + K\,(y_t - \ell_{t-1})$$

$$\ell_t = K\,y_t + (1-K)\,\ell_{t-1},$$

which is **simple exponential smoothing with** $\alpha = K$. The steady-state gain has a closed form in the noise ratio $q = Q/R$:

$$K_\infty = \frac{-q + \sqrt{q^2 + 4q}}{2}$$

So the smoothing parameter of SES is not an arbitrary knob: it encodes a belief about how much the level moves compared with how noisy the readings are. Fitting $\alpha$ is estimating that ratio.

------------------------------------------------------------------------

``` python
q = Q / R
k_inf = (-q + np.sqrt(q * q + 4 * q)) / 2
ses_level, ses_levels = levels[59], []
for z in x[60:]:                    # simple exponential smoothing, started from the filter's day-59 level
    ses_level = k_inf * z + (1 - k_inf) * ses_level
    ses_levels.append(ses_level)
gap = np.max(np.abs(np.array(ses_levels) - levels[60:]))
print(f"steady-state gain from the formula: {k_inf:.5f}; from the filter after 60 days: {gains[60]:.5f}")
print(f"largest difference between SES (alpha = K) and the Kalman level after day 60: {gap:.2e} kWh")
```

    steady-state gain from the formula: 0.13177; from the filter after 60 days: 0.13177
    largest difference between SES (alpha = K) and the Kalman level after day 60: 5.68e-08 kWh

The two agree to rounding error. Every exponential smoothing method is the steady state of a Kalman filter for some state space model; Holt-Winters is the filter for a level, a trend and a seasonal state. The Kalman form adds three things the plain recursion lacks: a likelihood (so the noise ratio can be estimated), prediction intervals, and a principled way to handle **missing observations**: skip the update step and let the uncertainty grow.

------------------------------------------------------------------------

## 3. Missing data for free

A gap is just several predict steps without an update. The filter’s uncertainty widens across the gap and the estimate carries on; no imputation is needed. On the real 121-hour gap in the raw data, a structural model with a daily season, fitted around the gap:

------------------------------------------------------------------------

``` python
raw = load_household(clean=False)["kwh"]
gaps = raw.isna()
run = (gaps != gaps.shift()).cumsum()
longest = gaps.groupby(run).sum().idxmax()
gap_idx = raw.index[(run == longest) & gaps]
around = raw[gap_idx[0] - pd.Timedelta("7D"):gap_idx[-1] + pd.Timedelta("2D")]
uc_gap = UnobservedComponents(around.to_numpy(), level="local level",
                              freq_seasonal=[{"period": 24, "harmonics": 4}]).fit(disp=False)
smoothed = uc_gap.smoothed_state[0] + uc_gap.smoothed_state[1]
show(line_chart(around.index.strftime("%d %b %H:00"), {"observed": around.to_numpy(), "Kalman smoother": smoothed},
                y_name="kWh", height="320px"))
print(f"gap of {len(gap_idx)} hours from {gap_idx[0]:%Y-%m-%d %H:00}, filled by the smoother without any imputation step")
```

    gap of 121 hours from 2010-08-17 21:00, filled by the smoother without any imputation step

The **smoother** (a backward pass after the forward filter) uses observations on both sides of the gap, so it reconnects the level smoothly and keeps the daily cycle running through it. That is the model-based gap filling the [Data Preparation](../01_Foundations/data_preparation.llms.md) page set aside.

------------------------------------------------------------------------

## 4. A structural model on the backtest

statsmodels’ `UnobservedComponents` builds the state space model from named parts, the “structural” approach of Harvey (1989):

- a **local level**;
- a **daily** season and a **weekly** season, each as a few Fourier harmonics (6 for the day, 3 for the week), which keeps the state small;
- an **AR(1)** term for short-lived departures, such as an evening that runs high.

It is refitted by maximum likelihood on the last three weeks at each origin.

------------------------------------------------------------------------

``` python
def structural(history, horizon):
    model = UnobservedComponents(history.iloc[-24 * 21:].to_numpy(), level="local level", autoregressive=1,
                                 freq_seasonal=[{"period": 24, "harmonics": 6}, {"period": 168, "harmonics": 3}])
    return model.fit(disp=False).forecast(horizon)


uc_score = score(backtest(y, structural), y, exclude=FILLED)
record("Structural model (level + daily + weekly + AR1)", "classical", uc_score, notebook=NB)
print(f"structural model: MASE {uc_score['mase']:.3f}, skill {uc_score['skill']:+.1%}")
```

    structural model: MASE 0.733, skill +9.4%

The structural model beats seasonal naive and the default ETS fit, and stays below [SARIMA](../03_Classical/arima.llms.md). Its seasons are smooth (a few harmonics), which loses the sharp edges of the morning and evening peaks that SARIMA’s lag-24 term copies directly. Its strengths are elsewhere: every component is interpretable, it handles gaps natively, and it gives a likelihood for comparing designs.

| Model family | State space view |
|----|----|
| Simple exponential smoothing | steady-state Kalman filter of a local level |
| Holt-Winters, ETS | level, trend and seasonal states, single error source |
| ARIMA | lagged values and errors as the state vector |
| Structural (this page) | named components, each with its own noise |

------------------------------------------------------------------------

Back to top
