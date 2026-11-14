# Datasets

The two public datasets every notebook on this site uses: one household’s electricity meter, and the 414 hourly series of the M4 competition.

Author

Benedict Thekkel

## Why one running dataset

Most forecasting tutorials change dataset on every page, so their results cannot be compared. This site uses one dataset for every single-series method, and one for every many-series method. Each notebook then adds a row to the same [scoreboard](../02_Evaluation/scoreboard.llms.md), and the gain of each method is a number, not an impression.

| Dataset | Shape | Used for |
|----|----|----|
| **Household power** (UCI) | 1 house, 1-minute readings, Dec 2006 to Nov 2010, total plus 3 sub-meters | Every single-series forecast, anomaly and change-point notebook |
| **M4 Hourly** | 414 series, 700 or 960 hourly training points each, 48-step test | Global models, hierarchies, benchmarks |

The household data matches the [Phisaver Project](../core.llms.md) brief: total consumption of one home, with sub-meters that show which part of the house drew the power. It is public (UCI, CC BY 4.0), so every number on this site can be reproduced. Phisaver’s own meter data is private and never appears here.

------------------------------------------------------------------------

## Download cache

Files download once into `~/.cache/ml_timeseries` (override with `ML_TIMESERIES_DATA`). The cache sits outside the repo, so no data file is ever committed or published.

------------------------------------------------------------------------

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L39)

### fetch

``` python
def fetch(
    url:str, name:str
)->pathlib.Path:
```

*Download `url` to the cache as `name` unless it is already there.*

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L27)

### cache_dir

``` python
def cache_dir()->pathlib.Path:
```

*Download cache: `$ML_TIMESERIES_DATA`, else `~/.cache/ml_timeseries`.*

## Household power: the raw minutes

The UCI file is a 133 MB semicolon-separated text file inside a 20 MB zip. Missing readings are written as `?`. The first parse takes a few seconds, so the result is cached as Parquet.

| Column        | Unit | Meaning                                            |
|---------------|------|----------------------------------------------------|
| `active_kw`   | kW   | Whole-house active power, averaged over the minute |
| `reactive_kw` | kW   | Reactive power                                     |
| `voltage`     | V    | Mains voltage                                      |
| `current_a`   | A    | Current                                            |
| `sub1_wh`     | Wh   | Kitchen: dishwasher, oven, microwave               |
| `sub2_wh`     | Wh   | Laundry: washer, dryer, fridge, light              |
| `sub3_wh`     | Wh   | Water heater and air conditioner                   |

The sub-meters do not cover the whole house. The rest (lighting, sockets, everything else) is the difference between the total and the three sub-meters.

------------------------------------------------------------------------

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L60)

### load_household_raw

``` python
def load_household_raw()->pandas.core.frame.DataFrame:
```

*One-minute household readings indexed by local timestamp (float32, NaN where missing).*

``` python
raw = load_household_raw()
print(f"{len(raw):,} minutes from {raw.index[0]} to {raw.index[-1]}")
print(f"missing readings: {raw['active_kw'].isna().mean():.2%}")
raw.head(3).astype(float).round(3).set_axis(raw.index[:3].strftime("%H:%M")).T
```

    2,075,259 minutes from 2006-12-16 17:24:00 to 2010-11-26 21:02:00
    missing readings: 1.25%

| ts          | 17:24   | 17:25   | 17:26   |
|-------------|---------|---------|---------|
| active_kw   | 4.216   | 5.360   | 5.374   |
| reactive_kw | 0.418   | 0.436   | 0.498   |
| voltage     | 234.840 | 233.630 | 233.290 |
| current_a   | 18.400  | 23.000  | 23.000  |
| sub1_wh     | 0.000   | 0.000   | 0.000   |
| sub2_wh     | 1.000   | 1.000   | 2.000   |
| sub3_wh     | 17.000  | 16.000  | 17.000  |

## Household power: energy per period

Forecasting works on energy per period, not instantaneous power. [`load_household`](https://bthek1.github.io/ML_Timeseries/01_Foundations/datasets.html#load_household) resamples the minutes into energy in kWh for each period:

- **Total**: mean kW over the period times the period length in hours.
- **Sub-meters**: the sum of their Wh readings, divided by 1000.
- **Other**: total minus the three sub-meters.

A period with fewer than 75% of its minutes present is set to missing rather than averaged from a fragment. An hour with 5 readings would otherwise look like a full hour.

------------------------------------------------------------------------

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L80)

### resample_energy

``` python
def resample_energy(
    raw:pandas.core.frame.DataFrame, freq:str='h', min_coverage:float=0.75
)->pandas.core.frame.DataFrame:
```

*Energy in kWh per `freq` period; periods with too few minutes present become NaN.*

## Filling gaps

Most models cannot take missing values, so the loader can fill them, and it marks every filled period in a boolean `filled` column. The [backtest](../02_Evaluation/backtesting.llms.md) uses that flag to score forecasts only against hours the meter actually recorded. The fill rule depends on the gap length:

- **Short gaps (up to 3 periods)**: linear interpolation. Consumption changes little in three hours.
- **Long gaps**: copy the value from one season earlier (168 hours, the same hour last week). Interpolating across two missing days would draw a straight line through two nights and two evenings, which no house ever does.

The [Data Preparation](../01_Foundations/data_preparation.llms.md) notebook compares these choices. Here they are only the default.

------------------------------------------------------------------------

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L94)

### fill_gaps

``` python
def fill_gaps(
    s:pandas.core.series.Series, season:int=168, max_interp:int=3
)->pandas.core.series.Series:
```

*Interpolate gaps of at most `max_interp` steps; fill longer gaps from `season` steps earlier (or later).*

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L109)

### load_household

``` python
def load_household(
    freq:str='h', clean:bool=True
)->pandas.core.frame.DataFrame:
```

*Household energy in kWh per `freq` period. `clean=True` fills every gap (see [`fill_gaps`](https://bthek1.github.io/ML_Timeseries/01_Foundations/datasets.html#fill_gaps)) and adds a `filled` flag.*

``` python
hourly_raw = load_household(clean=False)
hourly = load_household()
gaps = hourly_raw["kwh"].isna()
run = (gaps != gaps.shift()).cumsum()
lengths = gaps.groupby(run).sum()
lengths = lengths[lengths > 0]
print(f"{len(hourly):,} hours, {gaps.sum()} missing ({gaps.mean():.2%}) in {len(lengths)} gaps")
print(f"longest gap: {lengths.max()} hours; gaps of 1-3 hours: {(lengths <= 3).sum()}")
hourly.drop(columns="filled").describe().T[["mean", "std", "min", "max"]].round(3)
```

    34,589 hours, 438 missing (1.27%) in 16 gaps
    longest gap: 121 hours; gaps of 1-3 hours: 9

|           | mean    | std   | min     | max     |
|-----------|---------|-------|---------|---------|
| kwh       | 1.089   | 0.897 | 0.124   | 6.561   |
| sub1_kwh  | 0.067   | 0.212 | 0.000   | 2.902   |
| sub2_kwh  | 0.078   | 0.251 | 0.000   | 2.786   |
| sub3_kwh  | 0.387   | 0.441 | 0.000   | 1.293   |
| other_kwh | 0.557   | 0.495 | 0.083   | 3.890   |
| voltage   | 240.846 | 2.995 | 225.835 | 251.902 |

## Charts

[`line_chart`](https://bthek1.github.io/ML_Timeseries/01_Foundations/datasets.html#line_chart), [`bar_chart`](https://bthek1.github.io/ML_Timeseries/01_Foundations/datasets.html#bar_chart) and [`show`](https://bthek1.github.io/ML_Timeseries/01_Foundations/datasets.html#show) are the chart helpers the site uses. [`line_chart`](https://bthek1.github.io/ML_Timeseries/01_Foundations/datasets.html#line_chart) draws pyecharts lines with the pinned asset host from the repo conventions and a zoom slider, sized to the page width. [`show`](https://bthek1.github.io/ML_Timeseries/01_Foundations/datasets.html#show) renders it with the chart data compacted, which keeps a notebook with several charts under a few hundred KB.

------------------------------------------------------------------------

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L170)

### show

``` python
def show(
    chart
):
```

*Render a pyecharts chart inline with its JSON compacted (pyecharts pretty-prints one value per line).*

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L150)

### bar_chart

``` python
def bar_chart(
    labels, series:dict, y_name:str='', height:str='400px', horizontal:bool=False
):
```

*pyecharts bar chart of `series` (name -\> values) per label, sized to the page width.*

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L132)

### line_chart

``` python
def line_chart(
    x, series:dict, y_name:str='', height:str='400px', zoom:bool=True
):
```

*pyecharts line chart of `series` (name -\> values) against `x`, sized to the page width.*

### Four years of daily energy

The winter peaks and the August holidays (consumption drops to almost nothing in August 2008) are visible at once. Both matter for every model that follows: the first is annual seasonality, the second is the kind of event no model can predict from the history alone.

``` python
daily = load_household("D")
show(line_chart(daily.index.date, {"total": daily["kwh"], "water heater + AC": daily["sub3_kwh"]},
           y_name="kWh"))
```

### Two weeks of hourly energy

At hourly resolution the daily cycle dominates: a low night, a morning peak and a larger evening peak. Weekends look different from weekdays. These are the two seasonal periods (24 and 168 hours) that the classical models must capture.

``` python
week = hourly.loc["2009-01-05":"2009-01-18"]
show(line_chart(week.index.strftime("%a %d %H:00"), {"total": week["kwh"], "kitchen": week["sub1_kwh"],
                        "laundry": week["sub2_kwh"], "water heater + AC": week["sub3_kwh"]},
           y_name="kWh"))
```

## M4 Hourly

The M4 competition (2018) is the standard many-series benchmark. Its hourly subset has 414 series of different scales and patterns, with a fixed 48-hour test horizon. The files use the long format that the Nixtla libraries expect: one row per `(unique_id, ds, y)`, where `ds` is an integer step.

------------------------------------------------------------------------

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L186)

### to_long

``` python
def to_long(
    s:pandas.core.series.Series, uid:str='house'
)->pandas.core.frame.DataFrame:
```

*One series as a long-format `(unique_id, ds, y)` frame.*

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/data.py#L179)

### load_m4_hourly

``` python
def load_m4_hourly()->tuple[pandas.core.frame.DataFrame, pandas.core.frame.DataFrame]:
```

*M4 Hourly as long-format `(unique_id, ds, y)` frames: `(train, test)`.*

``` python
m4_train, m4_test = load_m4_hourly()
lens = m4_train.groupby("unique_id").size()
print(f"{m4_train.unique_id.nunique()} series; train lengths {sorted(int(v) for v in lens.unique())}; "
      f"test {int(m4_test.groupby('unique_id').size().max())} steps each")
m4_train.head(3)
```

    414 series; train lengths [700, 960]; test 48 steps each

|     | unique_id | ds  | y     |
|-----|-----------|-----|-------|
| 0   | H1        | 1   | 605.0 |
| 1   | H1        | 2   | 586.0 |
| 2   | H1        | 3   | 586.0 |

Back to top
