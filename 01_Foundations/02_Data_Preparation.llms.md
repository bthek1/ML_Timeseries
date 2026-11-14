# Data Preparation

Resampling, clock time and daylight saving, filling gaps, outliers and scaling, each tested on the household data rather than assumed.

Author

Benedict Thekkel

## Why preparation gets its own page

Every choice made before a model sees the data changes what the model can learn, and most of them are made silently: a default `mean()` when resampling, a `fillna` borrowed from another project, a timezone added because it seemed tidy. This page makes five of those choices on purpose and measures each one.

------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd

from ML_Timeseries.data import bar_chart, fill_gaps, line_chart, load_household, load_household_raw, show
```

``` python
raw = load_household_raw()
hh = load_household()
hh_raw = load_household(clean=False)
y = hh["kwh"]
```

## 1. Resampling: the aggregate decides what you can see

The meter reports one-minute average power. The forecasts on this site use energy per hour, which is the mean power over the hour. That is the right aggregate for energy, and the wrong one for **peak demand**: a 6 kW oven running for ten minutes and a 1 kW heater running for an hour use the same energy, and a meter that bills on peak power sees them very differently.

------------------------------------------------------------------------

``` python
g = raw["active_kw"].resample("h")
peak_ratio = (g.max() / g.mean()).dropna()
print(f"hourly peak minute / hourly mean: median {peak_ratio.median():.2f}, 90th percentile {peak_ratio.quantile(0.9):.2f}")
day = raw.loc["2009-12-20", "active_kw"]
hourly_mean = day.resample("h").mean().reindex(day.index, method="ffill")
show(line_chart(day.index.strftime("%H:%M"), {"1-minute power": day, "hourly mean": hourly_mean}, y_name="kW", height="320px"))
```

    hourly peak minute / hourly mean: median 1.76, 90th percentile 3.30

In a typical hour the peak minute is almost twice the hourly mean, and in one hour in ten it is more than three times. A forecast of hourly energy cannot answer a question about peak power. If the goal were sizing a battery inverter or forecasting a demand charge, the resampling would have to keep the maximum (`resample("h").max()`), and that is a different target with a different model.

------------------------------------------------------------------------

## 2. Clock time and daylight saving

France changes its clocks twice a year. In local clock time, the hour from 02:00 to 03:00 on the last Sunday of March does not exist, and the hour from 02:00 to 03:00 on the last Sunday of October happens twice. Yet the raw file is a perfect one-minute grid with no gap and no duplicate. So which clock is it?

The household answers it. If the timestamps ignored daylight saving, the family’s routine (getting up, cooking dinner) would jump by an hour in the data at each change. Compare the average daily profile in the three weeks before and after each change, and find the shift that lines them up:

------------------------------------------------------------------------

``` python
changes = ["2007-03-25", "2007-10-28", "2008-03-30", "2008-10-26", "2009-03-29", "2009-10-25", "2010-03-28", "2010-10-31"]
rows = []
for c in changes:
    t = pd.Timestamp(c)
    before = y[t - pd.Timedelta("21D"):t - pd.Timedelta("1h")]
    after = y[t + pd.Timedelta("1D"):t + pd.Timedelta("22D")]
    pb = before.groupby(before.index.hour).mean().to_numpy()
    pa = after.groupby(after.index.hour).mean().to_numpy()
    corr = {k: np.corrcoef(pb, np.roll(pa, k))[0, 1] for k in range(-2, 3)}
    rows.append({"clock change": c, "best shift (h)": max(corr, key=corr.get), "corr at 0": corr[0], "corr at +1": corr[1], "corr at -1": corr[-1]})
dst = pd.DataFrame(rows)
dst.style.format(precision=3).hide(axis="index")
```

| clock change | best shift (h) | corr at 0 | corr at +1 | corr at -1 |
|--------------|----------------|-----------|------------|------------|
| 2007-03-25   | 0              | 0.891     | 0.568      | 0.878      |
| 2007-10-28   | 0              | 0.822     | 0.807      | 0.600      |
| 2008-03-30   | 0              | 0.941     | 0.664      | 0.903      |
| 2008-10-26   | 0              | 0.739     | 0.608      | 0.583      |
| 2009-03-29   | 0              | 0.928     | 0.679      | 0.762      |
| 2009-10-25   | 0              | 0.869     | 0.751      | 0.719      |
| 2010-03-28   | 0              | 0.941     | 0.783      | 0.796      |
| 2010-10-31   | 0              | 0.922     | 0.824      | 0.655      |

The best alignment is a shift of zero at every change. The routine stays at the same clock time across the changes, so the timestamps are **local wall-clock time**. Because the grid is still regular, the logger must have dealt with each change somehow (inventing an hour in spring, merging one in autumn), and the dataset’s documentation does not say how. The practical rules that follow:

- **Keep these timestamps naive.** Localising them fails at the first spring change, because the file contains a time that did not exist:

------------------------------------------------------------------------

``` python
try:
    y.tz_localize("Europe/Paris")
except ValueError as e:
    print(type(e).__name__, ":", str(e).split(".")[0])
```

    ValueError : 2007-03-25 02:00:00 is a nonexistent time due to daylight savings time

- **Calendar features stay correct.** “Hour of day” in naive local time is the hour the household lived by, which is what predicts its behaviour.
- **For your own data, store UTC and convert at the edges.** A logger that records UTC never has a missing or duplicate hour, and converting to local time for calendar features is one line (`tz_convert`).

------------------------------------------------------------------------

## 3. Filling gaps: test the method, do not assume it

The [Datasets](../01_Foundations/datasets.llms.md) loader fills gaps of up to 3 hours by linear interpolation and longer ones from the same hour a week earlier. That rule should be measured, not trusted. The test: take 300 random stretches of complete data for each gap length, hide them, fill them with each method, and score the fill against the hidden truth.

------------------------------------------------------------------------

``` python
rng = np.random.default_rng(0)
complete = hh_raw["kwh"].notna().to_numpy()
lengths = [1, 3, 6, 24, 72]
fill_rows = []
for L in lengths:
    starts = []
    while len(starts) < 300:
        i = int(rng.integers(400, len(y) - 400))
        if complete[i - 340:i + L + 340].all():
            starts.append(i)
    errs = {"linear": [], "yesterday (t-24)": [], "last week (t-168)": [], "loader default": []}
    for i in starts:
        truth = y.iloc[i:i + L].to_numpy()
        s = y.iloc[i - 400:i + L + 400].copy()
        s.iloc[400:400 + L] = np.nan
        errs["linear"].append(np.abs(s.interpolate().iloc[400:400 + L].to_numpy() - truth).mean())
        daily_fill = s.copy()
        for k in (1, 2, 3):
            daily_fill = daily_fill.fillna(s.shift(24 * k))
        errs["yesterday (t-24)"].append(np.abs(daily_fill.iloc[400:400 + L].to_numpy() - truth).mean())
        errs["last week (t-168)"].append(np.abs(s.shift(168).iloc[400:400 + L].to_numpy() - truth).mean())
        errs["loader default"].append(np.abs(fill_gaps(s).iloc[400:400 + L].to_numpy() - truth).mean())
    fill_rows.append({"gap (h)": L} | {k: np.mean(v) for k, v in errs.items()})
fills = pd.DataFrame(fill_rows).set_index("gap (h)")
show(bar_chart([f"{L} h" for L in fills.index], {c: fills[c] for c in fills.columns}, y_name="fill MAE (kWh)", height="360px"))
```

``` python
fills.style.format("{:.3f}").highlight_min(axis=1, props="font-weight: bold")
```

|         | linear | yesterday (t-24) | last week (t-168) | loader default |
|---------|--------|------------------|-------------------|----------------|
| gap (h) |        |                  |                   |                |
| 1       | 0.370  | 0.581            | 0.623             | 0.370          |
| 3       | 0.480  | 0.619            | 0.599             | 0.480          |
| 6       | 0.603  | 0.620            | 0.629             | 0.629          |
| 24      | 0.776  | 0.643            | 0.636             | 0.636          |
| 72      | 0.817  | 0.672            | 0.626             | 0.626          |

The crossover is between 6 and 24 hours. Up to 6 hours, the neighbours on either side are the best guide; beyond a day, interpolation draws a straight line through a night and an evening, and copying last week wins. Yesterday and last week are close; last week is slightly better on long gaps because it also gets the weekday right.

The loader’s threshold of 3 hours is conservative: at 6 hours, interpolation beats last week by a small margin. It stays, because 9 of the 16 real gaps are 1 to 3 hours long and the difference at 6 hours is under 5%.

------------------------------------------------------------------------

## 4. Outliers: extreme is not the same as wrong

A robust way to flag unusual hours: compare each hour with the median of the same hour of the week, and scale by the median absolute deviation (MAD) of that hour of the week. This **robust z-score** is not pulled by the outliers themselves, unlike a mean and standard deviation.

------------------------------------------------------------------------

``` python
how = y.index.dayofweek * 24 + y.index.hour
med = y.groupby(how).transform("median")
mad = (y - med).abs().groupby(how).transform("median") * 1.4826
z = (y - med) / mad
flag = z.abs() > 5
print(f"{flag.sum():,} hours ({flag.mean():.1%}) have |robust z| > 5; largest z {z.max():.0f} at {z.idxmax():%a %Y-%m-%d %H:00} ({y[z.idxmax()]:.2f} kWh)")
night = flag[y.index.hour.isin(range(0, 6))].mean()
print(f"share of night hours (00-05) flagged: {night:.1%}; share of other hours flagged: {flag[~y.index.hour.isin(range(0, 6))].mean():.1%}")
```

    1,402 hours (4.1%) have |robust z| > 5; largest z 39 at Sat 2007-02-03 02:00 (3.50 kWh)
    share of night hours (00-05) flagged: 12.7%; share of other hours flagged: 1.2%

Four percent of hours look extreme, and most of them are nights. A quiet night has a tiny MAD, so one load of washing at 02:00 scores a z of 40. That is a real event in the house, not a sensor error. Removing or clipping it would teach a model that nobody ever runs the washing machine at night, and the model would then be confidently wrong when someone does.

The rule this site follows: **remove values that are physically impossible** (negative energy, a sub-meter above the total), **keep values that are merely rare**, and handle rare values in the loss or the metric, not in the data. The loader’s assertion that sub-meters never exceed the total is the first kind of check.

------------------------------------------------------------------------

## 5. Scaling and transforms

Two transforms matter for the models that follow.

**Per-series scaling.** The total, kitchen and water-heater series differ in mean by a factor of about 16. A global model trained on all of them at once (the machine learning and deep learning sections) would spend its capacity on the large series unless each is scaled by its own statistics first.

**The log transform.** Consumption is right-skewed: most hours are small and a few are large. `log1p` (log of 1 plus the value, safe at zero) makes the distribution closer to symmetric, which helps models that assume Gaussian errors. It works on the total and does little for the kitchen meter, whose skew comes from being zero most of the time; no smooth transform moves a spike of zeros. The cost: a model trained on the log predicts the median, not the mean, after transforming back, so its forecasts of total energy run low unless corrected.

------------------------------------------------------------------------

``` python
stats = pd.DataFrame({name: {"mean kWh": s.mean(), "std kWh": s.std(), "skew": s.skew(), "skew after log1p": np.log1p(s).skew()}
                      for name, s in {"total": y, "kitchen": hh["sub1_kwh"], "water heater + AC": hh["sub3_kwh"]}.items()}).T
stats.style.format("{:.2f}")
```

|                   | mean kWh | std kWh | skew | skew after log1p |
|-------------------|----------|---------|------|------------------|
| total             | 1.09     | 0.90    | 1.30 | 0.56             |
| kitchen           | 0.07     | 0.21    | 4.26 | 3.27             |
| water heater + AC | 0.39     | 0.44    | 0.71 | 0.59             |

``` python
back = np.expm1(np.log1p(y).groupby(y.index.hour).mean())
print(f"mean of the hourly profile {y.groupby(y.index.hour).mean().mean():.3f} kWh; back-transformed log profile {back.mean():.3f} kWh "
      f"({1 - back.mean() / y.groupby(y.index.hour).mean().mean():.0%} low)")
```

    mean of the hourly profile 1.089 kWh; back-transformed log profile 0.967 kWh (11% low)

## Summary

| Step | Choice on this site | Measured reason |
|----|----|----|
| Resample | hourly mean power (energy) | the forecast target is energy; peak power needs `max` and is a different problem |
| Time | naive local time | the routine does not shift at clock changes; localising fails on the spring hour |
| Gaps | interpolate up to 3 h, else last week | interpolation wins to about 6 h, last week wins from a day |
| Outliers | keep | the extreme hours are real appliance use, mostly at night |
| Scaling | per series, inside global models | series means differ by about 16 times |
| Log | optional, and correct the bias | log1p halves the total’s skew (1.30 to 0.56), but its back-transform runs 11% low |

------------------------------------------------------------------------

Back to top
