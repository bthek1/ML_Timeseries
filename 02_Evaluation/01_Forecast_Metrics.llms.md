# Forecast Metrics

MAE, RMSE, MAPE, sMAPE, WAPE, MASE and skill scores: what each one rewards, where each one breaks, and why this site reports MASE and skill.

Author

Benedict Thekkel

## Why the metric matters

A metric is a decision about which errors hurt. RMSE punishes one large miss more than many small ones. MAPE punishes a miss at 03:00, when the house uses almost nothing, far more than the same miss at 19:00. Pick the wrong metric and the “best” model is best at the wrong thing.

All the metrics below are implemented in `ML_Timeseries.backtest` and computed here on two seasonal naive backtests, so the differences come from the metrics, not the models:

- **daily**: repeat yesterday (season 24 hours);
- **weekly**: repeat the same day last week (season 168 hours).

Each runs on three series from the same house, which differ in scale and in how often they are zero.

------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd

from ML_Timeseries.backtest import backtest, mae, mape, mase_scale, rmse, seasonal_naive, smape, wape
from ML_Timeseries.data import bar_chart, load_household, show
```

``` python
hh = load_household()
observed = ~hh["filled"]  # score only hours the meter recorded, as the backtest protocol does
series = {"total": hh["kwh"], "kitchen": hh["sub1_kwh"], "water heater + AC": hh["sub3_kwh"]}
for name, s in series.items():
    print(f"{name:18s} mean {s.mean():.3f} kWh, zero in {(s == 0).mean():.0%} of hours")
```

    total              mean 1.089 kWh, zero in 0% of hours
    kitchen            mean 0.067 kWh, zero in 86% of hours
    water heater + AC  mean 0.387 kWh, zero in 23% of hours

## 1. Scale-dependent: MAE and RMSE

$$\text{MAE} = \frac{1}{n}\sum |y_t - \hat y_t|$$

$$\text{RMSE} = \sqrt{\frac{1}{n}\sum (y_t - \hat y_t)^2}$$

Both are in the units of the series (kWh here), which makes them easy to read and impossible to compare across series. MAE is minimised by forecasting the **median**; RMSE by forecasting the **mean**. RMSE is always at least MAE, and the gap grows with the size of the largest misses, so RMSE far above MAE means a few big errors dominate.

------------------------------------------------------------------------

## 2. Percentage errors: MAPE and sMAPE

$$\text{MAPE} = \frac{1}{n}\sum \frac{|y_t - \hat y_t|}{|y_t|}$$

$$\text{sMAPE} = \frac{1}{n}\sum \frac{2\,|y_t - \hat y_t|}{|y_t| + |\hat y_t|}$$

Percentages look scale-free, which is why managers like them. They fail in two ways:

- **Division by zero.** One hour with zero consumption makes MAPE infinite. The kitchen meter reads zero in most hours.
- **Asymmetry.** sMAPE was designed to fix MAPE, and it is not symmetric either: for the same absolute miss, forecasting too low costs more than forecasting too high.

------------------------------------------------------------------------

``` python
print(f"actual 1.0, forecast 1.5 (too high): sMAPE {smape([1.0], [1.5]):.3f}")
print(f"actual 1.0, forecast 0.5 (too low):  sMAPE {smape([1.0], [0.5]):.3f}")
```

    actual 1.0, forecast 1.5 (too high): sMAPE 0.400
    actual 1.0, forecast 0.5 (too low):  sMAPE 0.667

## 3. Scale-free and safe: WAPE and MASE

$$\text{WAPE} = \frac{\sum |y_t - \hat y_t|}{\sum |y_t|}$$

$$\text{MASE} = \frac{\text{MAE}}{\frac{1}{T-m}\sum_{t>m} |y_t - y_{t-m}|}$$

**WAPE** divides total error by total demand, so a zero hour adds nothing to the denominator instead of exploding one term. It reads as “the forecast is off by 25% of what the house used”.

**MASE** (Hyndman and Koehler, 2006) divides by the error the seasonal naive forecast made *on the training history*. It is defined for any series with some variation, works across scales, and has a fixed meaning: 1.0 is as good as seasonal naive was in the past. The M4 competition used it alongside sMAPE.

**Skill** is the most direct comparison: `1 - MAE / MAE(reference)`, both on the same test windows. It answers “how much error did this model remove compared with doing nothing clever”.

------------------------------------------------------------------------

``` python
rows = []
for name, s in series.items():
    for label, m in [("daily", 24), ("weekly", 168)]:
        bt = backtest(s, seasonal_naive(m))
        bt = bt[observed.reindex(bt.ds).to_numpy()]
        scale = mase_scale(s, 24)
        mase = bt.groupby("window").apply(lambda g: mae(g.y, g.yhat) / scale.loc[g.origin.iloc[0]], include_groups=False).mean()
        rows.append({"series": name, "reference": label, "MAE": mae(bt.y, bt.yhat), "RMSE": rmse(bt.y, bt.yhat),
                     "MAPE": mape(bt.y, bt.yhat), "sMAPE": smape(bt.y, bt.yhat), "WAPE": wape(bt.y, bt.yhat), "MASE": mase})
table = pd.DataFrame(rows)
table.style.format(precision=3).hide(axis="index")
```

| series            | reference | MAE   | RMSE  | MAPE  | sMAPE | WAPE  | MASE  |
|-------------------|-----------|-------|-------|-------|-------|-------|-------|
| total             | daily     | 0.523 | 0.809 | 0.721 | 0.501 | 0.577 | 0.816 |
| total             | weekly    | 0.546 | 0.810 | 0.803 | 0.553 | 0.603 | 0.841 |
| kitchen           | daily     | 0.077 | 0.237 | inf   | 0.349 | 1.896 | 0.676 |
| kitchen           | weekly    | 0.073 | 0.191 | inf   | 0.375 | 1.795 | 0.635 |
| water heater + AC | daily     | 0.315 | 0.498 | inf   | 0.765 | 0.925 | 0.954 |
| water heater + AC | weekly    | 0.320 | 0.490 | inf   | 0.828 | 0.940 | 0.981 |

Read the table by column:

- **MAE and RMSE** rank the series by size, not by how predictable they are.
- **MAPE** is infinite for both sub-meters, because each has zero hours in the test windows. A metric that cannot rank two models is useless.
- **sMAPE** gives the kitchen the *best* score of the three series. In 86% of its hours both the actual and the forecast are zero, and those terms count as perfect. Yet WAPE shows that the kitchen forecasts’ total error is almost twice the kitchen’s total consumption. sMAPE rewards a series for being empty.
- **WAPE and MASE** stay finite on every series, and on each one they agree about which reference is better: yesterday for the total and the water heater, last week for the kitchen.
- **MASE below 1 for seasonal naive itself** is not a contradiction. The scale comes from the whole history, and the test period (summer into autumn) is calmer than the four-year average, so the same method makes smaller errors there.

------------------------------------------------------------------------

## 4. Where MAPE puts its weight

Split the daily reference’s errors by hour of day. The left bar is each hour’s share of the total absolute error, which is where the energy actually went wrong. The right bar is each hour’s share of the MAPE sum. MAPE gives the night (00:00 to 05:00) more weight than its share of the error, and the evening peak (18:00 to 21:00) less, because a 0.2 kWh miss on a 0.3 kWh night is 67%, while the same miss on a 2 kWh evening is 10%. A model tuned for MAPE learns to get the night right, where little energy is at stake.

------------------------------------------------------------------------

``` python
bt = backtest(hh["kwh"], seasonal_naive(24))
bt = bt[observed.reindex(bt.ds).to_numpy()].copy()
bt["abs_err"] = (bt.y - bt.yhat).abs()
bt["pct_err"] = bt.abs_err / bt.y
by_hour = bt.groupby(bt.ds.dt.hour)[["abs_err", "pct_err"]].sum()
by_hour = by_hour / by_hour.sum()
show(bar_chart([f"{h:02d}" for h in by_hour.index],
               {"share of absolute error": by_hour.abs_err, "share of MAPE": by_hour.pct_err},
               y_name="share", height="360px"))
```

## 5. The choice for this site

| Use | Metric | Why |
|----|----|----|
| Headline, comparable across pages and series | **MASE** | Scale-free, safe with zeros, 1.0 means seasonal naive |
| “Did the model help on these windows?” | **Skill** against seasonal naive | Same windows, same units, reads as % of error removed |
| Business reading for one series | **WAPE** | “Off by X% of consumption” |
| Sensitivity to big misses | **RMSE** next to MAE | A large gap means a few big errors dominate |
| Avoid | MAPE, and sMAPE on sparse series | Infinite, or flattering, on series with zeros |

Probabilistic forecasts need their own metrics (pinball loss, CRPS, coverage). They come with the probabilistic forecasting pages.

------------------------------------------------------------------------

Back to top
