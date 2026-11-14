# GARCH: Forecasting the Variance

When the size of the surprises changes over time: testing for ARCH effects, fitting GARCH(1,1) to daily consumption, and checking that its prediction intervals are better calibrated than constant-variance ones.

Author

Benedict Thekkel

## Why forecast the variance

Every model so far forecasts the expected value. Many decisions also need the **uncertainty**: how much reserve to hold, how wide a band to plan within. If the uncertainty itself changes, calm periods followed by turbulent ones, a constant-width interval is too wide in calm weeks and too narrow in turbulent ones.

**ARCH** (autoregressive conditional heteroskedasticity; Engle, 1982) and its generalisation **GARCH** (Bollerslev, 1986) model the variance of tomorrow’s error as a function of recent errors and recent variance:

$$\sigma_t^2 = \omega + \alpha\, \varepsilon_{t-1}^2 + \beta\, \sigma_{t-1}^2$$

A large surprise yesterday ($\varepsilon_{t-1}^2$) raises today’s variance; $\beta$ makes the effect persist. $\alpha + \beta$ close to 1 means volatility comes in long spells. GARCH is the standard tool in finance, where returns are calm for months and then turbulent; Engle shared the 2003 Nobel prize in economics for ARCH.

------------------------------------------------------------------------

``` python
import warnings

import numpy as np
import pandas as pd
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch
from threadpoolctl import threadpool_limits

from ML_Timeseries.data import bar_chart, line_chart, load_household, show

# nbdev-test runs pages in parallel; per-page BLAS pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
warnings.filterwarnings("ignore")
```

## 1. The series: weekly changes in daily consumption

GARCH models the errors of a mean model, so the series should have no strong predictable level left. The change in daily consumption against the same weekday last week removes the weekly cycle and most of the annual level. What remains is the week-on-week surprise, in kWh.

------------------------------------------------------------------------

``` python
daily = load_household("D")["kwh"]
r = (daily - daily.shift(7)).dropna()
lm_p = het_arch(r.to_numpy(), nlags=7)[1]
print(f"{len(r)} days; Engle's ARCH test (7 lags): p = {lm_p:.1e}")
```

    1435 days; Engle's ARCH test (7 lags): p = 2.4e-65

Engle’s test regresses the squared series on its own lags. A tiny p-value means large changes cluster with large changes: the variance is predictable.

------------------------------------------------------------------------

## 2. GARCH(1,1) with an AR(7) mean

The `arch` library fits the mean (an AR(7), to use the last week of changes) and the variance equation together by maximum likelihood.

------------------------------------------------------------------------

``` python
fit = arch_model(r, mean="AR", lags=7, vol="GARCH", p=1, q=1).fit(disp="off")
a, b = fit.params["alpha[1]"], fit.params["beta[1]"]
print(f"alpha {a:.3f}, beta {b:.3f}, persistence alpha + beta = {a + b:.3f}")
vol = fit.conditional_volatility
by_month = vol.groupby(vol.index.month).mean()
show(bar_chart(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
               {"mean conditional volatility (kWh)": by_month.to_numpy()}, height="300px"))
```

    alpha 0.086, beta 0.862, persistence alpha + beta = 0.949

Persistence near 0.95 means a turbulent spell decays slowly, with a half-life of about two weeks. The monthly averages show where the turbulence lives: **winter**. Heating, lighting and more time indoors make winter days both larger and more variable, so a forecast for a January day should carry a wider band than one for August.

------------------------------------------------------------------------

``` python
show(line_chart(r.index.date, {"weekly change (kWh)": r.to_numpy(), "+2 sigma": 2 * vol.to_numpy(), "-2 sigma": -2 * vol.to_numpy()},
                y_name="kWh", height="340px"))
```

## 3. Are the intervals better?

The test that matters: one-step-ahead 90% intervals over the last 300 days, from a GARCH model and from the same AR(7) mean with **constant** variance. Both are refitted every 30 days on the data before. A good interval covers about 90% of outcomes and is as narrow as possible.

------------------------------------------------------------------------

``` python
Z90 = 1.645
test_idx = range(len(r) - 300, len(r))
hits = {"constant variance": [], "GARCH(1,1)": []}
width = {k: [] for k in hits}
for i in test_idx:
    if (i - test_idx[0]) % 30 == 0:
        params = {"GARCH(1,1)": arch_model(r.iloc[:i], mean="AR", lags=7, vol="GARCH", p=1, q=1).fit(disp="off").params,
                  "constant variance": arch_model(r.iloc[:i], mean="AR", lags=7, vol="Constant").fit(disp="off").params}
    for name, vol_kind in (("GARCH(1,1)", "GARCH"), ("constant variance", "Constant")):
        model = arch_model(r.iloc[:i + 1], mean="AR", lags=7, vol=vol_kind, **({"p": 1, "q": 1} if vol_kind == "GARCH" else {}))
        fc = model.fix(params[name]).forecast(horizon=1, start=i - 1, reindex=False)
        mu, sd = fc.mean.iloc[0, 0], np.sqrt(fc.variance.iloc[0, 0])
        hits[name].append(abs(r.iloc[i] - mu) <= Z90 * sd)
        width[name].append(2 * Z90 * sd)
cover = pd.DataFrame({"coverage of 90% interval": {k: np.mean(v) for k, v in hits.items()},
                      "mean width (kWh)": {k: np.mean(v) for k, v in width.items()}})
cover.style.format({"coverage of 90% interval": "{:.1%}", "mean width (kWh)": "{:.1f}"})
```

|                   | coverage of 90% interval | mean width (kWh) |
|-------------------|--------------------------|------------------|
| constant variance | 94.3%                    | 25.8             |
| GARCH(1,1)        | 91.3%                    | 22.5             |

The constant-variance interval over-covers: it is wide enough for winter all year, so in summer it is far wider than it needs to be. GARCH’s intervals land closer to the nominal 90% and are narrower on average, because they tighten in calm spells and widen when turbulence starts. Same mean forecast, more useful band.

The [Probabilistic Forecasting](../07_Probabilistic/probabilistic_forecasting.llms.md) and [Conformal Prediction](../07_Probabilistic/conformal_prediction.llms.md) pages take this further, with quantile models and conformal intervals that need no distributional assumption.

------------------------------------------------------------------------

Back to top
