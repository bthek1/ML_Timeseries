# Feature Engineering

Lags, calendar and holiday features, Fourier terms, rolling statistics and exogenous sub-meters, added one group at a time and scored on the backtest, plus catch22 features that describe a whole day’s shape.

Author

Benedict Thekkel

## Features turn a series into a table

A tree model or a linear model cannot read a time series. It reads rows. Feature engineering builds, for each hour to forecast, a row of numbers that are **known at the forecast origin** and that carry information about that hour.

Every feature here obeys the rule from [Backtesting and Leakage](../02_Evaluation/backtesting.llms.md): it is computed from history plus an empty future, so a feature that would need a future value comes out empty instead of leaking. The model is the same throughout (scikit-learn’s histogram gradient boosting, trained once on everything before the first origin), so every change in score comes from the features.

------------------------------------------------------------------------

``` python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.transformations.panel.catch22 import Catch22

from ML_Timeseries.backtest import backtest, future_index, origins, record, score
from ML_Timeseries.data import bar_chart, load_household, show

from threadpoolctl import threadpool_limits

# nbdev-test runs pages in parallel; per-page OpenMP pools on 4 cores made the gradient-boosting pages 30x slower.
_ = threadpool_limits(1)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
cut = origins(y.index)[0]
```

## 1. The feature groups

| Group | Features | Idea |
|----|----|----|
| **Seasonal lags** | the value 24, 48, 168 and 336 hours earlier | the [autocorrelation](../01_Foundations/decomposition_stationarity.llms.md) peaks; every lag is at least the 24-hour horizon |
| **Calendar** | hour, day of week, month, French public holiday | behaviour follows the clock and the calendar |
| **Fourier terms** | sine and cosine pairs at the daily, weekly and annual periods | smooth seasonal shapes a tree can split on, including the annual cycle that four lags cannot see |
| **Rolling statistics** | mean and standard deviation of the last day, mean of the last week, mean of the same hour over the last four weeks (all ending at the origin) | the recent level and volatility |
| **Sub-meter lags** | kitchen, laundry, water heater and voltage, 24 hours earlier | exogenous series: does knowing *which* appliances ran yesterday help? |

French public holidays: the eight fixed dates plus Easter Monday, Ascension Thursday and Whit Monday, computed from the date of Easter.

------------------------------------------------------------------------

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/features.py#L32)

### make_features

``` python
def make_features(
    s:pandas.core.series.Series, groups:tuple=('calendar', 'fourier', 'rolling'),
    exog:pandas.core.frame.DataFrame | None=None
)->pandas.core.frame.DataFrame:
```

*Feature table for `s`: seasonal lags (24 to 336 h) plus the named groups; every feature is known 24 h ahead.*

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/features.py#L19)

### french_holidays

``` python
def french_holidays(
    years
)->pandas.core.indexes.datetimes.DatetimeIndex:
```

*French public holidays: eight fixed dates plus Easter Monday, Ascension and Whit Monday.*

## 2. One group at a time

Each step adds one group to the previous set, trains the model and runs the full 30-window backtest. Sub-meter values are cut at the origin too: the forecaster sees `hh.loc[:origin]`, nothing later.

------------------------------------------------------------------------

``` python
def gbm_forecaster(groups):
    train = y.iloc[:cut]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
    model.fit(make_features(train, groups, hh).iloc[700:], train.iloc[700:])

    def f(history, horizon):
        ext = pd.concat([history, pd.Series(np.nan, index=future_index(history, horizon))])
        return model.predict(make_features(ext, groups, hh.loc[:history.index[-1]]).iloc[-horizon:])
    return f


steps = [("seasonal lags", ()), ("+ calendar", ("calendar",)), ("+ Fourier", ("calendar", "fourier")),
         ("+ rolling", ("calendar", "fourier", "rolling")), ("+ sub-meters", ("calendar", "fourier", "rolling", "submeters"))]
ablation = {}
for name, groups in steps:
    ablation[name] = score(backtest(y, gbm_forecaster(groups)), y, exclude=FILLED)
abl = pd.DataFrame(ablation).T[["mase", "skill", "mae", "seconds"]]
abl.style.format({"mase": "{:.3f}", "skill": "{:+.1%}", "mae": "{:.3f}", "seconds": "{:.1f}"})
```

|               | mase  | skill  | mae   | seconds |
|---------------|-------|--------|-------|---------|
| seasonal lags | 0.726 | +9.2%  | 0.474 | 0.0     |
| \+ calendar   | 0.677 | +15.5% | 0.442 | 0.1     |
| \+ Fourier    | 0.670 | +16.7% | 0.435 | 0.3     |
| \+ rolling    | 0.665 | +17.0% | 0.434 | 0.4     |
| \+ sub-meters | 0.669 | +16.4% | 0.437 | 0.5     |

``` python
show(bar_chart(abl.index, {"skill vs seasonal naive": abl.skill}, height="320px"))
```

Reading the steps:

- **Seasonal lags alone** already beat seasonal naive, because the model can weigh four past seasons instead of copying one.
- **The calendar is the largest single gain.** Hour and day of week let the model learn the household’s routine directly instead of inferring it from lags, and the holiday flag catches days that look like Sundays.
- **Fourier terms** add a little, mostly through the annual pair: four lags cannot know it is November.
- **Rolling statistics** add almost nothing on top. The same-hour average they include is close to what the lags and calendar already give.
- **Sub-meters do not help.** Knowing that the kitchen was busy yesterday says little about the total tomorrow beyond what the total itself says. An exogenous series earns its place only when it carries information the target does not, such as a weather forecast for heating load.

The best set goes on the scoreboard as the first machine learning row.

------------------------------------------------------------------------

``` python
best_name = abl.mase.idxmin()
best_groups = dict(steps)[best_name]
record("Gradient boosting, engineered features", "machine learning", ablation[best_name],
       notebook="01_Foundations/03_Feature_Engineering", notes=f"lags + {', '.join(best_groups)}")
print(f"recorded: {best_name}, MASE {ablation[best_name]['mase']:.3f}, skill {ablation[best_name]['skill']:+.1%}")
```

    recorded: + rolling, MASE 0.665, skill +17.0%

## 3. Features of a whole series: catch22

The features above describe one hour. Some tasks need features of a whole series or a whole window: classifying a day, clustering households, spotting an unusual week. **catch22** (Lubba et al., 2019) is a set of 22 such features, chosen from thousands of candidate features in the hctsa library because together they classify time series nearly as well as the full set. They measure distribution shape, autocorrelation, entropy and how the series moves between levels. **tsfresh** is the large alternative: hundreds of features with built-in significance tests, slower and more prone to overfitting on small data.

Here catch22 runs on each day’s 24-hour profile (via sktime). The question: do the 22 numbers alone tell a weekday from a weekend?

------------------------------------------------------------------------

``` python
start = y.index[y.index.hour == 0][0]
days = y[start:]
n = len(days) // 24
profiles = days.to_numpy()[:n * 24].reshape(n, 1, 24)
day_index = days.index[::24][:n]
weekend = (day_index.dayofweek >= 5).astype(int)
F = Catch22(col_names="short_str_feat").fit_transform(profiles)
print(f"{F.shape[0]} days x {F.shape[1]} catch22 features")
```

    1440 days x 22 catch22 features

``` python
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
acc = cross_val_score(clf, F.fillna(0), weekend, cv=5).mean()
base = 1 - weekend.mean()
print(f"weekday/weekend accuracy from catch22 alone: {acc:.1%} (always guessing weekday: {base:.1%})")
d = (F[weekend == 1].mean() - F[weekend == 0].mean()) / F.std()
d.abs().sort_values(ascending=False).head(5).rename("standardised difference").round(2)
```

    weekday/weekend accuracy from catch22 alone: 75.0% (always guessing weekday: 71.5%)

    transition_matrix    0.70
    centroid_freq        0.66
    ami_timescale        0.56
    acf_timescale        0.54
    whiten_timescale     0.46
    Name: standardised difference, dtype: float64

The 22 shape features beat the base rate, but not by much. The features that differ most between weekend and weekday measure timescales and frequency content (how fast the profile changes), not level. Most catch22 features ignore *when* in the day something happens, on purpose. That is the trade-off of shape features: they are robust to shifts in time, which is good for clustering households and bad for telling Saturday from Tuesday. For this site’s forecasting task, the hour-level features in sections 1 and 2 do the work.

------------------------------------------------------------------------

Back to top
