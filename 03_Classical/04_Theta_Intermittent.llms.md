# Theta and Intermittent Demand

The Theta method that won the M3 competition, and Croston, SBA, TSB and ADIDA for series that are zero most of the time, tested on the kitchen sub-meter.

Author

Benedict Thekkel

## 1. Theta

The **Theta method** (Assimakopoulos and Nikolopoulos, 2000) won the M3 competition in 2000 on 3,003 series. It splits a deseasonalised series into two “theta lines”: one with the curvature removed (the linear trend) and one with the curvature doubled. It extrapolates the first, forecasts the second with simple exponential smoothing, averages the two and puts the season back. Hyndman and Billah (2003) later showed that this is equivalent to **simple exponential smoothing with a drift** equal to half the trend’s slope, which explains why it works: a damped compromise between “the level stays” and “the trend continues”.

------------------------------------------------------------------------

``` python
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.forecasting.theta import ThetaModel
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import backtest, mae, origins, record, rmse, score, seasonal_naive
from ML_Timeseries.data import bar_chart, line_chart, load_household, show

# nbdev-test runs pages in parallel; per-page BLAS pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
warnings.filterwarnings("ignore", module="statsmodels")
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "03_Classical/04_Theta_Intermittent"
WINDOW = 24 * 7 * 8


def theta(history, horizon):
    res = ThetaModel(history.iloc[-WINDOW:].to_numpy(), period=24).fit()
    return np.asarray(res.forecast(horizon))


theta_score = score(backtest(y, theta), y, exclude=FILLED)
record("Theta (24 h season)", "classical", theta_score, notebook=NB)
print(f"Theta: MASE {theta_score['mase']:.3f}, skill {theta_score['skill']:+.1%}")
```

    Theta: MASE 0.831, skill -4.4%

Theta loses slightly to seasonal naive here, for the same reason the default [ETS](../03_Classical/ets.llms.md) fit does: its smoothing is fitted for one step ahead, and the house’s next hour is a poor guide to tomorrow evening. Theta’s strength is many short, trending, low-frequency series such as yearly and monthly business data, which is what M3 was made of. An hourly household load has no trend worth extrapolating over a day.

------------------------------------------------------------------------

## 2. Intermittent demand

Some series are zero most of the time: spare-parts demand, hospital admissions for a rare condition, or this house’s **kitchen** circuit, which draws nothing in 86% of hours. Ordinary methods handle these badly. Exponential smoothing decays towards zero after every zero and jumps after every non-zero, so its forecast is mostly noise.

**Croston’s method** (1972) splits the series into two smaller ones and smooths each separately:

- the **size** of each non-zero demand, $z$;
- the **interval** between non-zero demands, $p$.

The forecast is the demand rate $z / p$, a flat line: the method does not try to say *when* the next demand comes, only how much per period on average. Three refinements:

| Method | Change | Why |
|----|----|----|
| **SBA** (Syntetos and Boylan, 2005) | multiply Croston by $1 - \alpha/2$ | Croston’s ratio of two smoothed values is biased upwards |
| **TSB** (Teunter, Syntetos and Babai, 2011) | smooth the *probability* of a demand every period, instead of the interval | the forecast decays when demand stops, which matters for items going obsolete |
| **ADIDA** (Nikolopoulos et al., 2011) | aggregate to a coarser period (here: days), forecast that, then split it back | at the daily level the series is no longer intermittent |

All four are a few lines of numpy.

------------------------------------------------------------------------

``` python
def croston(x, alpha=0.1, sba=False):
    nz = np.flatnonzero(x > 0)
    if len(nz) == 0:
        return 0.0
    z, p, last = x[nz[0]], nz[0] + 1.0, nz[0]
    for i in nz[1:]:
        z += alpha * (x[i] - z)
        p += alpha * ((i - last) - p)
        last = i
    rate = z / p
    return rate * (1 - alpha / 2) if sba else rate


def tsb(x, alpha=0.1, beta=0.1):
    nz = np.flatnonzero(x > 0)
    if len(nz) == 0:
        return 0.0
    z, prob = x[nz[0]], 1.0 / (nz[0] + 1)
    for v in x[nz[0]:]:
        prob += beta * ((v > 0) - prob)
        if v > 0:
            z += alpha * (v - z)
    return prob * z


def adida(history, horizon, alpha=0.1):
    "Aggregate to days, smooth the daily total, spread it over the hours by the average hourly profile."
    x = history.iloc[-WINDOW:]
    daily = x.resample("D").sum()
    level = daily.iloc[0]
    for v in daily.iloc[1:]:
        level += alpha * (v - level)
    profile = x.groupby(x.index.hour).sum()
    profile = profile / profile.sum() if profile.sum() > 0 else pd.Series(1 / 24, index=range(24))
    hours = (x.index[-1] + pd.to_timedelta(np.arange(1, horizon + 1), "h")).hour
    return level * profile.reindex(hours).to_numpy()


flat = lambda rate_fn: (lambda h, n: np.repeat(rate_fn(h.iloc[-WINDOW:].to_numpy()), n))
methods = {
    "zero": lambda h, n: np.zeros(n),
    "seasonal naive (24 h)": seasonal_naive(24),
    "Croston": flat(croston),
    "SBA": flat(lambda x: croston(x, sba=True)),
    "TSB": flat(tsb),
    "ADIDA (daily, hourly profile)": adida,
}
```

### Scoring a sparse series

The kitchen series is scored on the same 30 windows with three numbers. MAE and RMSE as usual, plus **bias**: total forecast over total actual, minus one. For a stock-keeping decision (how many spare parts, how much hot water) the total over the period matters more than the hour-by-hour match.

------------------------------------------------------------------------

``` python
kitchen = hh["sub1_kwh"]
rows = {}
for name, f in methods.items():
    bt = backtest(kitchen, f)
    rows[name] = {"MAE": mae(bt.y, bt.yhat), "RMSE": rmse(bt.y, bt.yhat), "bias": bt.yhat.sum() / bt.y.sum() - 1}
inter = pd.DataFrame(rows).T
print(f"kitchen: zero in {(kitchen == 0).mean():.0%} of hours")
test_days = [kitchen.iloc[i:i + 24].sum() for i in origins(kitchen.index)]
avg_days = [kitchen.iloc[i - WINDOW:i].sum() / (WINDOW / 24) for i in origins(kitchen.index)]
print(f"the 30 test days used {1 - sum(test_days) / sum(avg_days):.0%} less than an average day of the 8 weeks before each")
inter.style.format({"MAE": "{:.4f}", "RMSE": "{:.4f}", "bias": "{:+.0%}"})
```

    kitchen: zero in 86% of hours
    the 30 test days used 22% less than an average day of the 8 weeks before each

|                               | MAE    | RMSE   | bias  |
|-------------------------------|--------|--------|-------|
| zero                          | 0.0379 | 0.1389 | -100% |
| seasonal naive (24 h)         | 0.0732 | 0.2297 | +39%  |
| Croston                       | 0.0850 | 0.1379 | +54%  |
| SBA                           | 0.0827 | 0.1372 | +46%  |
| TSB                           | 0.0931 | 0.1557 | +80%  |
| ADIDA (daily, hourly profile) | 0.0716 | 0.1362 | +24%  |

``` python
show(bar_chart(inter.index, {"MAE": inter.MAE, "RMSE": inter.RMSE}, y_name="kWh", height="340px"))
```

The table is the whole lesson of intermittent forecasting:

- **The zero forecast has the lowest MAE.** In most hours the kitchen draws nothing, so “nothing” is the median, and MAE rewards the median. It is also useless: its bias is -100%, it never predicts the energy the kitchen actually uses.
- **Seasonal naive** copies yesterday’s spikes into the right hours of today only by luck, so it pays for every misplaced spike twice: once where it predicted one and once where the real one came.
- **Croston and SBA** predict a small flat rate. Their MAE is worse than zero’s and their RMSE slightly better. SBA’s correction lowers Croston’s rate and its bias, as the theory says it should.
- **TSB** does worst of the rate methods here: it updates the demand probability every hour, so a few busy days before an origin push the whole next day up.
- **ADIDA** is the best non-zero method on every measure. Forecasting the daily total is a much easier problem than forecasting sparse hours, and spreading it by the usual hourly profile puts the energy at meal times.
- **Every method over-forecasts the total.** That is the data, not a bug: kitchen use is lumpy, with rare heavy cooking days, so a typical day is well below the average day. The 30 test days happened to use about a fifth less than the average day before them, so even a perfect estimate of the average would have over-forecast them.

So for a sparse series the metric decides the method. Optimise MAE and you will ship the zero forecast; optimise RMSE or bias and you will ship Croston or ADIDA. The [Forecast Metrics](../02_Evaluation/forecast_metrics.llms.md) page showed the same trap from the metric’s side: sMAPE scored this series as the easiest of the three.

These results are for the kitchen circuit, not the total, so they are not on the scoreboard.

------------------------------------------------------------------------

``` python
bt_k = {n: backtest(kitchen, methods[n]) for n in ("Croston", "ADIDA (daily, hourly profile)")}
w = bt_k["Croston"].window.max()
a_, b_ = bt_k["Croston"], bt_k["ADIDA (daily, hourly profile)"]
a_, b_ = a_[a_.window == w], b_[b_.window == w]
show(line_chart(a_.ds.dt.strftime("%a %H:00"), {"actual kitchen": a_.y, "Croston": a_.yhat, "ADIDA": b_.yhat},
                y_name="kWh", height="300px", zoom=False))
```

Back to top
