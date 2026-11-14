# Change-Point Detection

Splitting a series into regimes with PELT and binary segmentation, and a negative result worth knowing: in household consumption, a three-week +40% shift is too ordinary to stand out.

Author

Benedict Thekkel

## Regimes, not points

An anomaly is a moment; a **change point** is the start of a new regime: a household goes on holiday, a heat pump is installed, a tenant moves in. Change-point detection splits a series into segments, each with its own mean (or variance, or distribution), placing the boundaries where the series changes behaviour.

The standard formulation: choose boundaries to minimise the total cost of fitting each segment (for a mean shift, the sum of squared deviations from each segment’s mean) plus a **penalty** per boundary. Without the penalty, every point would be its own segment. Two search methods from the `ruptures` library:

- **PELT** (pruned exact linear time; Killick et al., 2012): the exact optimum for a given penalty, fast because it prunes candidates that can never be optimal.
- **Binary segmentation**: find the single best split, then recurse into each half. Approximate, but it can be asked for a fixed number of boundaries.

The series here is daily household energy, on a log scale so that a 40% rise is the same size whether the base is summer or winter.

------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd
import ruptures as rpt

from ML_Timeseries.data import bar_chart, line_chart, load_household, show
```

``` python
daily = load_household("D")["kwh"]
signal = np.log(daily.clip(lower=0.5))
z = ((signal - signal.mean()) / signal.std()).to_numpy()
```

## 1. Regimes in the real series

PELT with a moderate penalty on the whole four years:

------------------------------------------------------------------------

``` python
breaks = rpt.Pelt(model="l2", min_size=7).fit(z).predict(pen=20)
cps = [daily.index[i] for i in breaks[:-1]]
segment = np.zeros(len(daily))
for a, b in zip([0] + breaks[:-1], breaks):
    segment[a:b] = daily.iloc[a:b].mean()
print(f"{len(cps)} change points: " + ", ".join(c.strftime("%Y-%m-%d") for c in cps))
show(line_chart(daily.index.date, {"daily kWh": daily, "segment mean": segment}, y_name="kWh", height="340px"))
```

    12 change points: 2007-02-24, 2007-03-06, 2007-04-05, 2007-11-06, 2008-02-24, 2008-08-02, 2008-09-01, 2009-06-18, 2009-09-11, 2010-02-28, 2010-07-18, 2010-08-27

The boundaries fall at the household’s large regime changes: the onset and end of winter heating, and the long August 2008 absence, when consumption drops to a few kWh a day for almost three weeks. These are real and useful: a monthly report that says “consumption has been in a new, higher regime since 14 October” is more actionable than a daily anomaly flag.

------------------------------------------------------------------------

## 2. Can it find a change we put there?

To measure detection, three blocks of +40% lasting three weeks are injected into the daily series (March and October 2009, April 2010), giving six true boundaries. The series is first deseasonalised by removing a weekday effect and a smooth annual cycle (three Fourier pairs), so that winter itself is not mistaken for a change. A boundary counts as found if a detected change point lies within three days of it.

------------------------------------------------------------------------

``` python
x = daily.copy()
blocks = []
for start in ("2009-03-02", "2009-10-05", "2010-04-12"):
    s = pd.Timestamp(start)
    x[s:s + pd.Timedelta(days=20)] *= 1.4
    blocks += [s, s + pd.Timedelta(days=21)]
xs = x["2008-01-01":]
t = (xs.index - xs.index[0]).days.to_numpy()
design = np.column_stack([np.ones(len(t))] + [f(2 * np.pi * k * t / 365.25) for k in (1, 2, 3) for f in (np.sin, np.cos)]
                         + [(xs.index.dayofweek == d).astype(float) for d in range(1, 7)])
logx = np.log(xs.clip(lower=1)).to_numpy()
coef, *_ = np.linalg.lstsq(design, logx, rcond=None)
resid = logx - design @ coef
r = (resid - resid.mean()) / resid.std()


def recall(found):
    return np.mean([any(abs((c - b).days) <= 3 for c in found) for b in blocks])


rows = {}
for pen in (5, 10, 20):
    found = [xs.index[i] for i in rpt.Pelt(model="l2", min_size=5).fit(r).predict(pen=pen)[:-1]]
    rows[f"PELT, penalty {pen}"] = {"change points": len(found), "recall of the 6 boundaries": recall(found)}
for n in (6, 15):
    found = [xs.index[i] for i in rpt.Binseg(model="l2", min_size=5).fit(r).predict(n_bkps=n)[:-1]]
    rows[f"binary segmentation, {n} boundaries"] = {"change points": len(found), "recall of the 6 boundaries": recall(found)}
cp_table = pd.DataFrame(rows).T
cp_table.style.format({"change points": "{:.0f}", "recall of the 6 boundaries": "{:.2f}"})
```

|   | change points | recall of the 6 boundaries |
|----|----|----|
| PELT, penalty 5 | 24 | 0.50 |
| PELT, penalty 10 | 18 | 0.50 |
| PELT, penalty 20 | 4 | 0.00 |
| binary segmentation, 6 boundaries | 6 | 0.00 |
| binary segmentation, 15 boundaries | 15 | 0.33 |

Recall stays low however the penalty is set, and even binary segmentation told there are exactly six boundaries places most of them elsewhere. The detectors are not broken. The question is: how unusual is a three-week +40% shift *in this house*?

------------------------------------------------------------------------

``` python
m21 = daily.rolling(21).mean()
ratio = (m21 / m21.shift(21)).dropna()
share_30 = ((ratio >= 1.3) | (ratio <= 1 / 1.3)).mean()
share_40 = (ratio >= 1.4).mean()
print(f"21-day windows whose mean differs from the previous 21 days by 30% or more: {share_30:.0%}")
print(f"21-day windows at least 40% above the previous 21 days: {share_40:.1%}")
```

    21-day windows whose mean differs from the previous 21 days by 30% or more: 20%
    21-day windows at least 40% above the previous 21 days: 5.6%

One three-week window in five already differs from the previous three weeks by 30% or more, and one in eighteen is 40% higher. Cold spells, guests, a run of home-working days: this household changes regime constantly. An injected +40% block is therefore not a special event to the detector; it is one more change among dozens of the same size, and with a limited budget of boundaries the detector spends them on the larger real ones.

So this page records a **negative result**: the goal of finding at least 80% of injected changes is not met, and it is not a tuning problem. Change-point detection answers “where does the behaviour change?”, and in a household the honest answer is “often”. It works when the change of interest is large relative to the series’ natural regime swings: the August 2008 absence, when consumption fell by more than half for three weeks, is found in section 1.

Two ways to make it useful on meter data:

- **Detect against a model, not the raw series.** Run change-point detection on the residuals of a good forecast (the [scoreboard](../02_Evaluation/scoreboard.llms.md) models), where weather and routine are already explained, so what is left is the change that is not.
- **Detect across many meters.** A new heat pump changes one house’s load shape the same way it changes others’; comparing a meter with its peers removes the shared swings that hide the change here.

------------------------------------------------------------------------

Back to top
