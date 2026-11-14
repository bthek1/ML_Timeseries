# Baselines

Naive, mean, drift, seasonal naive and seasonal window averages: the forecasts that cost nothing, and the bar every other page has to clear.

Author

Benedict Thekkel

## Why start with the simplest forecasts

A baseline answers one question: how much of this problem is solved before any modelling? On many real series a well-chosen baseline is hard to beat, and a complex model that loses to it is worse than useless, because it looks sophisticated. The M competitions repeatedly found simple methods near the top.

Every baseline here is a few lines of numpy and runs on the same [backtest](../02_Evaluation/backtesting.llms.md) as every other model.

| Baseline | Forecast for each of the next 24 hours |
|----|----|
| **Naive** | the last observed hour |
| **Mean** | the mean of the last four weeks |
| **Drift** | the last value, plus the average change per hour over the last four weeks |
| **Seasonal naive (24 h)** | the same hour yesterday (already on the scoreboard as the reference) |
| **Seasonal naive (168 h)** | the same hour on the same weekday last week |
| **Seasonal window mean / median** | the mean or median of the same hour over the last four weeks |

------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd

from ML_Timeseries.backtest import backtest, record, score, seasonal_naive
from ML_Timeseries.data import bar_chart, line_chart, load_household, show
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "03_Classical/01_Baselines"
FOUR_WEEKS = 24 * 28


def naive(history, horizon):
    return np.repeat(history.iloc[-1], horizon)


def mean(history, horizon):
    return np.repeat(history.iloc[-FOUR_WEEKS:].mean(), horizon)


def drift(history, horizon):
    x = history.iloc[-FOUR_WEEKS:].to_numpy()
    slope = (x[-1] - x[0]) / (len(x) - 1)
    return x[-1] + slope * np.arange(1, horizon + 1)


def seasonal_window(stat, weeks=4):
    "The `stat` (mean or median) of the same hour of the week over the last `weeks` weeks."
    def f(history, horizon):
        x = history.to_numpy()
        past = np.stack([x[len(x) - 168 * k:len(x) - 168 * k + horizon] for k in range(1, weeks + 1)])
        return stat(past, axis=0)
    return f
```

## Results

------------------------------------------------------------------------

``` python
baselines = {
    "Naive": naive,
    "Mean (4 weeks)": mean,
    "Drift (4 weeks)": drift,
    "Seasonal naive (24 h)": seasonal_naive(24),
    "Seasonal naive (168 h)": seasonal_naive(168),
    "Seasonal window mean (4 weeks)": seasonal_window(np.mean),
    "Seasonal window median (4 weeks)": seasonal_window(np.median),
}
results, bts = {}, {}
for name, f in baselines.items():
    bts[name] = backtest(y, f)
    results[name] = score(bts[name], y, exclude=FILLED)
    if name != "Seasonal naive (24 h)":
        record(name, "baseline", results[name], notebook=NB)
table = pd.DataFrame(results).T[["mase", "skill", "mae", "rmse", "wape"]].sort_values("mase")
table.style.format({"mase": "{:.3f}", "skill": "{:+.1%}", "mae": "{:.3f}", "rmse": "{:.3f}", "wape": "{:.1%}"})
```

|                                  | mase  | skill  | mae   | rmse  | wape  |
|----------------------------------|-------|--------|-------|-------|-------|
| Seasonal window median (4 weeks) | 0.764 | +4.0%  | 0.501 | 0.715 | 55.4% |
| Seasonal window mean (4 weeks)   | 0.771 | +4.3%  | 0.500 | 0.693 | 55.2% |
| Seasonal naive (24 h)            | 0.816 | +0.0%  | 0.523 | 0.809 | 57.7% |
| Seasonal naive (168 h)           | 0.841 | -4.5%  | 0.546 | 0.810 | 60.3% |
| Mean (4 weeks)                   | 0.843 | -5.9%  | 0.553 | 0.702 | 61.1% |
| Naive                            | 0.931 | -15.7% | 0.605 | 0.857 | 66.8% |
| Drift (4 weeks)                  | 0.941 | -16.9% | 0.611 | 0.864 | 67.5% |

``` python
order = table.sort_values("skill")
show(bar_chart(order.index, {"skill vs seasonal naive (24 h)": order.skill}, horizontal=True, height="380px"))
```

What the ranking says:

- **Naive, mean and drift lose to seasonal naive.** They ignore the daily cycle, which is most of the predictable signal. Drift is the worst: a straight line through a series that goes up and down every day.
- **Last week loses to yesterday.** The [autocorrelation](../01_Foundations/decomposition_stationarity.llms.md) is slightly higher at lag 168 than at lag 24 over the whole four years, yet on these test windows copying one week ago is worse. A week-old copy gets the weekday right but carries a level that is seven days stale, and in late summer and autumn the level moves. (Before gap-filled hours were excluded from scoring, last week appeared to win; see the [backtest protocol](../02_Evaluation/backtesting.llms.md).)
- **Averaging four weeks beats copying either.** One week contains that week’s accidents: a late dinner, a guest, a load of washing. The average of four keeps the routine and dilutes the accidents. Mean and median are almost tied. The median has the better MASE, because it ignores a single unusual week entirely; the mean has the clearly better RMSE, because one large miss costs it less than it costs a median that sits on a spike. That is the rule from [Forecast Metrics](../02_Evaluation/forecast_metrics.llms.md) in action: the median minimises absolute error, the mean minimises squared error.

A four-week seasonal window is the strongest free forecast for this house. Every model on the later pages should be compared with about +4% skill, not 0%.

------------------------------------------------------------------------

## One window, four forecasts

The last test window shows the difference. Copying a single day reproduces that day’s peaks wherever they happened to fall; the four-week median draws the usual shape at the usual height.

------------------------------------------------------------------------

``` python
w = bts["Naive"].window.max()
pick = ["Naive", "Seasonal naive (24 h)", "Seasonal naive (168 h)", "Seasonal window median (4 weeks)"]
last = bts[pick[0]][bts[pick[0]].window == w]
show(line_chart(last.ds.dt.strftime("%a %H:00"),
                {"actual": last.y} | {p: bts[p][bts[p].window == w].yhat for p in pick},
                y_name="kWh", height="340px", zoom=False))
```

## Why a median of past weeks is hard to beat

The household’s consumption is a routine plus noise. The routine repeats every week; the noise does not. Any forecast that averages several aligned past weeks estimates the routine and throws away noise, and the remainder from the [decomposition](../01_Foundations/decomposition_stationarity.llms.md) is more than half the series’ spread, so most of what is left cannot be forecast from the past alone. Better models win by adapting the routine faster (a warm week, a holiday) or by using other information (the calendar, the weather), not by finding a hidden pattern in the noise.

------------------------------------------------------------------------

Back to top
