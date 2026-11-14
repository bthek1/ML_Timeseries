# Anomaly Detection

Finding meter faults in household energy data: statistical scores, forecast residuals, an isolation forest and two plain rules, scored on 24 faults injected into the test months.

Author

Benedict Thekkel

## What counts as an anomaly

For a meter, the anomalies worth finding are usually **faults**, not unusual behaviour. The [Data Preparation](../01_Foundations/data_preparation.llms.md) page showed that this house’s extreme hours are real appliance use; flagging them would only teach an operator to ignore the alarms. Three fault types matter in practice:

| Fault | What the data shows | Injected |
|----|----|----|
| **Spike** | one hour 3 kWh too high (a bad reading, a double count) | 8 |
| **Dropout** | six hours of exactly zero (meter or link down, filled with zeros) | 8 |
| **Stuck meter** | twelve hours repeating the last good value | 8 |

To measure detection, 24 such faults are injected at random positions into the test months (June to November 2010), with a fixed random seed. An event counts as found if any of its hours is flagged. The cost side is the number of flags on clean hours per week.

------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from threadpoolctl import threadpool_limits

from ML_Timeseries.data import bar_chart, line_chart, load_household, show

# nbdev-test runs pages in parallel; per-page OpenMP pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
```

``` python
hh = load_household()
y = hh["kwh"]
hist = y[:"2010-05-31"]                 # detectors learn from here
test = y["2010-06-01":"2010-11-25"].copy()
rng = np.random.default_rng(42)
starts = sorted(rng.choice(np.arange(200, len(test) - 200), 24, replace=False))
kinds = ["spike"] * 8 + ["dropout"] * 8 + ["stuck"] * 8
rng.shuffle(kinds)
x, events = test.copy(), []
for s, kind in zip(starts, kinds):
    if kind == "spike":
        span = [s]
        x.iloc[s] += 3.0
    elif kind == "dropout":
        span = list(range(s, s + 6))
        x.iloc[span] = 0.0
    else:
        span = list(range(s, s + 12))
        x.iloc[span] = x.iloc[s - 1]
    events.append((kind, x.index[span]))
injected = pd.Series(False, index=x.index)
for _, idx in events:
    injected[idx] = True
print(f"{len(events)} faults injected over {len(x) / 168:.0f} weeks; {injected.sum()} faulty hours")


def evaluate(flags: pd.Series) -> dict:
    flags = flags.reindex(x.index).fillna(False).astype(bool)
    found = {k: [] for k in ("spike", "dropout", "stuck")}
    for kind, idx in events:
        found[kind].append(bool(flags[idx].any()))
    return {"recall": np.mean(sum(found.values(), [])), **{f"{k} recall": np.mean(v) for k, v in found.items()},
            "false flags per week": (flags & ~injected).sum() / (len(x) / 168)}
```

    24 faults injected over 25 weeks; 152 faulty hours

## 1. Four detectors that know nothing about faults

- **Robust z-score**: distance from the median of the same hour of the week, in units of its median absolute deviation, flagged above 5.
- **Forecast residual**: the [seasonal window median](../03_Classical/baselines.llms.md) as the expected value; flag when the actual differs by more than the 99.5% quantile of historical week-on-week changes.
- **Isolation forest** (Liu et al., 2008): an ensemble of random trees that isolates points with few splits; anomalies are easy to isolate. Fitted on rolling six-hour window features of the history, contamination 1%.
- **Spike score**: how far an hour rises above *both* its neighbours, flagged above the 99.9% quantile of the history’s own spike scores.

------------------------------------------------------------------------

``` python
how = lambda idx: idx.dayofweek * 24 + idx.hour
med = hist.groupby(how(hist.index)).median()
mad = (hist - med.reindex(how(hist.index)).to_numpy()).abs().groupby(how(hist.index)).median() * 1.4826
z = (x - med.reindex(how(x.index)).to_numpy()) / mad.reindex(how(x.index)).to_numpy()

full = pd.concat([hist, x])
expected = pd.Series([np.median([full.iloc[i - 168 * k] for k in range(1, 5)]) for i in range(len(hist), len(full))], index=x.index)
resid_thr = np.quantile((hist - hist.shift(168)).abs().dropna(), 0.995)


def window_features(s, reference):
    r = s.rolling(6, center=True, min_periods=1)
    return pd.DataFrame({"value": s, "mean": r.mean(), "std": r.std(), "min": r.min(), "max": r.max(),
                         "zeros": (s == 0).rolling(6, center=True, min_periods=1).sum(),
                         "repeats": (s.diff() == 0).rolling(6, center=True, min_periods=1).sum(),
                         "vs expected": s - reference}).fillna(0)


iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=0)
iso.fit(window_features(hist.iloc[-8000:], hist.shift(168).iloc[-8000:]))
iso_flags = pd.Series(iso.predict(window_features(x, expected)) == -1, index=x.index)

spike = x - np.maximum(x.shift(1), x.shift(-1))
spike_thr = np.quantile((hist - np.maximum(hist.shift(1), hist.shift(-1))).dropna(), 0.999)

detectors = {"robust z-score > 5": z.abs() > 5, "forecast residual": (x - expected).abs() > resid_thr,
             "isolation forest": iso_flags, "spike score": spike > spike_thr}
results = {name: evaluate(f) for name, f in detectors.items()}
pd.DataFrame(results).T.style.format({"recall": "{:.2f}", "spike recall": "{:.2f}", "dropout recall": "{:.2f}",
                                      "stuck recall": "{:.2f}", "false flags per week": "{:.1f}"})
```

|   | recall | spike recall | dropout recall | stuck recall | false flags per week |
|----|----|----|----|----|----|
| robust z-score \> 5 | 0.21 | 0.12 | 0.00 | 0.50 | 7.6 |
| forecast residual | 0.08 | 0.25 | 0.00 | 0.00 | 0.0 |
| isolation forest | 0.21 | 0.62 | 0.00 | 0.00 | 0.6 |
| spike score | 0.29 | 0.88 | 0.00 | 0.00 | 0.1 |

None of them finds dropouts, and only the robust z-score finds stuck meters at all. Six hours of zero at night look exactly like a quiet night; twelve repeated values at an ordinary level look like an ordinary half-day. These faults are **unusual in structure, not in value**, and every detector above looks at values. The spike score is the exception that proves the point: it was designed for one fault’s structure, and it finds that fault.

------------------------------------------------------------------------

## 2. Two rules that know what a fault looks like

A real meter almost never reads exactly the same value four hours running, and a house with a fridge never uses exactly zero for three hours. Two lines of pandas:

------------------------------------------------------------------------

``` python
flat = (x.diff() == 0).rolling(4).sum() >= 4
zero = (x == 0).rolling(3).sum() >= 3
results["rules: flat run or zero run"] = evaluate(flat | zero)
results["rules + spike score"] = evaluate(flat | zero | (spike > spike_thr))
table = pd.DataFrame(results).T
table.style.format({"recall": "{:.2f}", "spike recall": "{:.2f}", "dropout recall": "{:.2f}", "stuck recall": "{:.2f}",
                    "false flags per week": "{:.1f}"})
```

|   | recall | spike recall | dropout recall | stuck recall | false flags per week |
|----|----|----|----|----|----|
| robust z-score \> 5 | 0.21 | 0.12 | 0.00 | 0.50 | 7.6 |
| forecast residual | 0.08 | 0.25 | 0.00 | 0.00 | 0.0 |
| isolation forest | 0.21 | 0.62 | 0.00 | 0.00 | 0.6 |
| spike score | 0.29 | 0.88 | 0.00 | 0.00 | 0.1 |
| rules: flat run or zero run | 0.67 | 0.00 | 1.00 | 1.00 | 0.0 |
| rules + spike score | 0.96 | 0.88 | 1.00 | 1.00 | 0.1 |

``` python
show(bar_chart(table.index, {"share of injected faults found": table.recall}, horizontal=True, height="320px"))
```

The rules catch every dropout and every stuck meter with no false alarms at all, and together with the spike score they find 23 of the 24 faults at about one false flag every ten weeks. The one miss is a spike at 19:00 on 2 October, just before an hour that was itself 3.9 kWh, so it barely rose above its neighbours.

This is the general shape of operational anomaly detection. Generic detectors (scores, residuals, isolation forests) find what is **rare**; faults are often not rare in value, only impossible in structure. Write down what each fault looks like, encode it as a rule, and keep the generic detectors for what is left. They earn their place on faults nobody anticipated, which is also why their false flags need a human to look at them.

The same structural thinking applies to the [Phisaver](../core.llms.md) brief’s anomaly goal: a dead sub-meter, a reversed CT clamp (negative readings) or a sudden constant offset are each one rule, and each would beat a model.

------------------------------------------------------------------------

``` python
kind, idx = next(e for e in events if e[0] == "dropout")
win = x[idx[0] - pd.Timedelta("36h"):idx[-1] + pd.Timedelta("36h")]
show(line_chart(win.index.strftime("%a %H:00"), {"with fault": win, "original": test.loc[win.index]}, y_name="kWh", height="300px", zoom=False))
```

Back to top
