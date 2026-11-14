# The Matrix Profile: Motifs and Discords

For every day-long window, the distance to its nearest match elsewhere in the year: the most repeated day shapes (motifs), the most unusual ones (discords), and the difference between unusual in shape and unusual in size.

Author

Benedict Thekkel

## One array that answers two questions

The **matrix profile** (Yeh, Keogh and colleagues, 2016) takes every subsequence of length *m* in a series and records the distance to its **nearest neighbour**, the most similar subsequence anywhere else (excluding overlaps with itself). From that one array:

- the smallest values are **motifs**: shapes that recur almost exactly, such as the household’s standard weekday;
- the largest values are **discords**: windows with no close match anywhere, the most unusual stretches of the year.

It has no parameters beyond *m*, needs no training, and the STUMPY library computes it exactly for a year of hourly data in seconds. The distance is usually computed on **z-normalised** windows (each window rescaled to mean 0, standard deviation 1), so it compares **shapes** regardless of level; the raw distance compares **values**.

Here *m* = 24: every day-long window of one year of hourly data, about 8,700 of them.

------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd
import stumpy

from ML_Timeseries.data import line_chart, load_household, show
```

``` python
hh = load_household()["2009-11-01":"2010-10-31"]
year, filled = hh["kwh"], hh["filled"].to_numpy()
M = 24


def top_windows(profile, k, largest=True):
    "Indices of the k most extreme windows, at least one window length apart."
    order = np.argsort(-profile if largest else profile)
    picks = []
    for i in order:
        if all(abs(i - j) >= M for j in picks):
            picks.append(i)
        if len(picks) == k:
            break
    return picks


mp = stumpy.stump(year.to_numpy(), m=M)
profile = mp[:, 0].astype(float)
print(f"{len(profile):,} windows; nearest-neighbour distance from {profile.min():.2f} to {profile.max():.2f}")
```

    8,737 windows; nearest-neighbour distance from 0.00 to 4.14

## 1. The motif: the shape this house repeats

The window with the closest match anywhere in the year, plotted with that match:

------------------------------------------------------------------------

``` python
i = top_windows(profile, 1, largest=False)[0]
j = int(mp[i, 1])
print(f"motif: {year.index[i]:%a %d %b %H:00} and its match {year.index[j]:%a %d %b %H:00}, distance {profile[i]:.2f}")
hours = [f"+{h} h" for h in range(M)]
show(line_chart(hours, {f"{year.index[i]:%a %d %b %H:00}": year.iloc[i:i + M], f"{year.index[j]:%a %d %b %H:00}": year.iloc[j:j + M]},
                y_name="kWh", height="300px", zoom=False))
```

    motif: Thu 19 Aug 11:00 and its match Thu 12 Aug 11:00, distance 0.00

A distance of zero means an **exact** copy, and households do not repeat themselves to the watt-hour. This one is not the household at all: the [Datasets](../01_Foundations/datasets.llms.md) loader filled the 121-hour gap from 17 to 22 August 2010 by copying the same hours a week earlier, and the matrix profile found the copy. That is a useful property in its own right: an exact or near-exact motif in meter data is a strong hint of duplicated, imputed or replayed readings.

Excluding every window that touches a filled hour gives the household’s real most-repeated shape:

------------------------------------------------------------------------

``` python
touches_filled = np.array([filled[k:k + M].any() for k in range(len(profile))])
clean = profile.copy()
clean[touches_filled | touches_filled[mp[:, 1].astype(int)]] = np.inf
i2 = top_windows(clean, 1, largest=False)[0]
j2 = int(mp[i2, 1])
print(f"best motif among recorded hours: {year.index[i2]:%a %d %b %H:00} and {year.index[j2]:%a %d %b %H:00}, distance {profile[i2]:.2f}")
show(line_chart(hours, {f"{year.index[i2]:%a %d %b %H:00}": year.iloc[i2:i2 + M], f"{year.index[j2]:%a %d %b %H:00}": year.iloc[j2:j2 + M]},
                y_name="kWh", height="300px", zoom=False))
```

    best motif among recorded hours: Wed 04 Aug 04:00 and Fri 30 Jul 17:00, distance 0.52

## 2. Discords: the most unusual days

The five windows with the most distant nearest neighbour:

------------------------------------------------------------------------

``` python
discords = top_windows(profile, 5)
pd.DataFrame({"window start": [year.index[d].strftime("%a %d %b %Y %H:00") for d in discords],
              "distance to nearest match": profile[discords].round(2),
              "energy in window (kWh)": [year.iloc[d:d + M].sum().round(1) for d in discords]})
```

|     | window start          | distance to nearest match | energy in window (kWh) |
|-----|-----------------------|---------------------------|------------------------|
| 0   | Thu 22 Jul 2010 20:00 | 4.14                      | 17.7                   |
| 1   | Wed 07 Jul 2010 01:00 | 4.13                      | 13.9                   |
| 2   | Sat 06 Mar 2010 06:00 | 4.12                      | 15.0                   |
| 3   | Thu 01 Jul 2010 20:00 | 4.09                      | 18.7                   |
| 4   | Fri 04 Jun 2010 01:00 | 4.06                      | 25.4                   |

``` python
d0 = discords[0]
show(line_chart(hours, {f"discord {year.index[d0]:%d %b %H:00}": year.iloc[d0:d0 + M],
                        f"its nearest match {year.index[int(mp[d0, 1])]:%d %b %H:00}": year.iloc[int(mp[d0, 1]):int(mp[d0, 1]) + M]},
                y_name="kWh", height="300px", zoom=False))
```

## 3. Shape versus size

Would the matrix profile catch a heater left on? Add 2.5 kWh to every hour from 10:00 to 18:00 on three ordinary days (20 kWh extra each, nearly doubling those days), recompute, and see where those days rank among all 8,700 windows, with z-normalised and with raw distances:

------------------------------------------------------------------------

``` python
heater = year.copy()
days = [pd.Timestamp(d) for d in ("2010-01-13", "2010-04-21", "2010-08-04")]
for d in days:
    heater[d + pd.Timedelta(hours=10):d + pd.Timedelta(hours=17)] += 2.5
ranks = {}
for label, norm in (("z-normalised (shape)", True), ("raw (value)", False)):
    p = stumpy.stump(heater.to_numpy(), m=M, normalize=norm)[:, 0].astype(float)
    ranks[label] = [int((p > p[heater.index.get_loc(d + pd.Timedelta(hours=10)) - 4:heater.index.get_loc(d + pd.Timedelta(hours=10)) + 4].max()).sum()) + 1
                    for d in days]
rank_table = pd.DataFrame(ranks, index=[d.strftime("%d %b %Y") for d in days])
rank_table.index.name = "heater day"
rank_table
```

|             | z-normalised (shape) | raw (value) |
|-------------|----------------------|-------------|
| heater day  |                      |             |
| 13 Jan 2010 | 2424                 | 114         |
| 21 Apr 2010 | 6214                 | 128         |
| 04 Aug 2010 | 6467                 | 259         |

**Shape and size are different questions.** A heater left on all day nearly doubles a day’s energy, yet with z-normalised distances those days are unremarkable: adding a constant block to the middle of a day leaves its shape close to many real days. With raw distances they rise to the top 1 to 3% of all windows, but still not into the top five, because this house has dozens of genuinely unusual days (four of the five discords above are summer evenings and nights unlike any other).

So the matrix profile is the right tool for **“is this shape new?”**: a new appliance with its own cycle, a changed routine, a meter replaying old data. For **“is this too much energy?”**, a forecast with an interval (the [probabilistic](../07_Probabilistic/probabilistic_forecasting.llms.md) pages) or a plain rule answers more directly. And the discords are a good list to show a person: each one is an unusual day worth a look, which is exactly what an exploratory tool should produce.

------------------------------------------------------------------------

Back to top
