# Multi-step Strategies

Recursive, direct and multi-output forecasting with the same LightGBM model and features: how errors compound when a model feeds on its own predictions, measured hour by hour.

Author

Benedict Thekkel

## The problem

A day-ahead forecast is 24 numbers. A regression model naturally predicts one. There are three standard ways to get 24:

| Strategy | How hour *h* is forecast | Risk |
|----|----|----|
| **Recursive** (1 model) | predict hour 1, append it to the history as if observed, predict hour 2, … | errors **compound**: hour 24 rests on 23 guesses |
| **Direct** (24 models) | model *h* maps what is known at the origin straight to hour *h* | 24 times the training; hours forecast independently |
| **MIMO** (1 model, 24 outputs) | all 24 hours at once from the same inputs | needs a model that outputs vectors (linear models, neural networks) |

A fourth trick, used on the [Feature Engineering](../01_Foundations/feature_engineering.llms.md) page, sidesteps the choice: give one model only lags of at least 24 hours, so every input is known for every hour of the day ahead.

All four here use LightGBM with the same raw lags (1, 2, 3, 24, 48, 168 and 336 hours, where available) and the calendar of the target hour.

------------------------------------------------------------------------

``` python
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import backtest, future_index, origins, record, score
from ML_Timeseries.data import line_chart, load_household, show

# nbdev-test runs pages in parallel; per-page OpenMP pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "04_Machine_Learning/02_Multistep_Strategies"
train = y.iloc[:origins(y.index)[0]]
LAGS = [1, 2, 3, 24, 48, 168, 336]
PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=1, random_state=0)


def calendar(index):
    return pd.DataFrame({"hour": index.hour, "dow": index.dayofweek, "month": index.month})
```

## 1. Recursive

One model learns the next hour from the hours just before it. At forecast time its own prediction for hour 1 becomes `lag1` for hour 2, and so on.

------------------------------------------------------------------------

``` python
def one_step_features(s: pd.Series) -> pd.DataFrame:
    X = pd.DataFrame({f"lag{l}": s.shift(l) for l in LAGS}, index=s.index)
    return pd.concat([X, calendar(s.index).set_index(s.index)], axis=1)


one_step = lgb.LGBMRegressor(**PARAMS).fit(one_step_features(train).iloc[336:], train.iloc[336:])


def recursive(history, horizon):
    s = history.iloc[-340:].copy()
    preds = []
    for ts in future_index(history, horizon):
        s.loc[ts] = np.nan
        pred = one_step.predict(one_step_features(s).iloc[[-1]])[0]
        s.loc[ts] = pred  # the guess becomes an input for the next hour
        preds.append(pred)
    return np.array(preds)
```

## 2. Direct

Model *h* (for *h* = 1 to 24) is trained on rows where the inputs are the values known at an origin and the target is the value *h* hours later. Each model uses the freshest values available to it, and none of them ever sees a prediction.

------------------------------------------------------------------------

``` python
def origin_features(s: pd.Series) -> pd.DataFrame:
    "Values known at the end of each hour: the latest value (lag 1 for the next hour) and older lags."
    return pd.DataFrame({f"lag{l}": s.shift(l - 1) for l in LAGS}, index=s.index)


base = origin_features(train)
direct_models = {}
for h in range(1, 25):
    target = train.shift(-h)
    X = pd.concat([base, calendar(train.index + pd.Timedelta(hours=h)).set_index(train.index)], axis=1)
    ok = X.notna().all(axis=1) & target.notna()
    direct_models[h] = lgb.LGBMRegressor(**PARAMS).fit(X[ok], target[ok])


def direct(history, horizon):
    known = origin_features(history.iloc[-340:]).iloc[[-1]].reset_index(drop=True)
    out = []
    for h in range(1, horizon + 1):
        ts = history.index[-1] + pd.Timedelta(hours=h)
        out.append(direct_models[h].predict(pd.concat([known, calendar(pd.DatetimeIndex([ts]))], axis=1))[0])
    return np.array(out)
```

## 3. MIMO

LightGBM predicts one number, so the multi-output strategy uses ridge regression from the [previous page](../04_Machine_Learning/forecasting_as_regression.llms.md): the last 336 hours plus the weekday, to all 24 hours at once.

------------------------------------------------------------------------

``` python
x = train.to_numpy()
mids = [i for i in range(336, len(train) - 24 + 1) if train.index[i].hour == 0]
Xm = np.column_stack([np.stack([x[i - 336:i] for i in mids]), np.eye(7)[[train.index[i].dayofweek for i in mids]]])
mimo_model = Ridge(alpha=100.0).fit(Xm, np.stack([x[i:i + 24] for i in mids]))


def mimo(history, horizon):
    nxt = history.index[-1] + pd.Timedelta(hours=1)
    return mimo_model.predict(np.concatenate([history.to_numpy()[-336:], np.eye(7)[nxt.dayofweek]])[None])[0][:horizon]
```

## 4. One model, lags of at least 24 hours

The sidestep: the same LightGBM, but only lags 24, 48, 168 and 336 hours, so the features of every hour of the day ahead are known at the origin. One model, no recursion.

------------------------------------------------------------------------

``` python
SAFE = [24, 48, 168, 336]


def safe_features(s):
    X = pd.DataFrame({f"lag{l}": s.shift(l) for l in SAFE}, index=s.index)
    return pd.concat([X, calendar(s.index).set_index(s.index)], axis=1)


safe_model = lgb.LGBMRegressor(**PARAMS).fit(safe_features(train).iloc[336:], train.iloc[336:])


def safe_lags(history, horizon):
    ext = pd.concat([history.iloc[-340:], pd.Series(np.nan, index=future_index(history, horizon))])
    return safe_model.predict(safe_features(ext).iloc[-horizon:])
```

## Results

------------------------------------------------------------------------

``` python
strategies = {"Recursive (LightGBM)": recursive, "Direct (24 LightGBM models)": direct,
              "MIMO (ridge)": mimo, "Lags of 24 h or more (LightGBM)": safe_lags}
bts = {name: backtest(y, f) for name, f in strategies.items()}
results = {name: score(bt, y, exclude=FILLED) for name, bt in bts.items()}
for name in ("Recursive (LightGBM)", "Direct (24 LightGBM models)", "Lags of 24 h or more (LightGBM)"):
    _ = record(name, "machine learning", results[name], notebook=NB)
table = pd.DataFrame(results).T[["mase", "skill", "mae", "seconds"]]
table.style.format({"mase": "{:.3f}", "skill": "{:+.1%}", "mae": "{:.3f}", "seconds": "{:.1f}"})
```

|                                 | mase  | skill  | mae   | seconds |
|---------------------------------|-------|--------|-------|---------|
| Recursive (LightGBM)            | 0.726 | +9.2%  | 0.475 | 0.8     |
| Direct (24 LightGBM models)     | 0.661 | +17.2% | 0.433 | 0.5     |
| MIMO (ridge)                    | 0.701 | +12.4% | 0.458 | 0.0     |
| Lags of 24 h or more (LightGBM) | 0.669 | +15.9% | 0.439 | 0.0     |

## Error by hour ahead

The table hides the most important difference. Split the error by forecast step:

------------------------------------------------------------------------

``` python
observed = lambda b: b[~FILLED.reindex(b.ds).to_numpy()]
by_step = pd.DataFrame({name: observed(bt).assign(ae=lambda d: (d.y - d.yhat).abs()).groupby("step").ae.mean()
                        for name, bt in bts.items()})
show(line_chart(by_step.index, {c: by_step[c] for c in by_step}, y_name="MAE (kWh)", height="360px", zoom=False))
```

Read the chart with one caveat: every origin is at midnight, so step *h* is also hour *h* of the day. All four curves rise towards the evening because evening hours are larger and more variable, not only because they are further ahead. Compare the curves with each other, not with themselves.

- **Recursive and direct start identical.** At hour 1 both see the same last value; their errors match.
- **Then the recursive error grows faster.** Over hours 13 to 24 it is about 10% above the direct strategy’s. Each step feeds a guess back in as `lag1`, and the model was trained on real values, not on its own guesses, so it trusts them too much. That is error compounding, measured.
- **“Lags of 24 h or more” is the mirror image.** Its first hours are the worst of the four, because it cannot use the last observed value at all. From midday on it matches the direct strategy, because by then the direct models’ freshest inputs are hours old anyway.
- **MIMO ridge** sits between them: it sees the last value, but as a linear model it uses it less well.

So the direct strategy’s whole advantage lies in the first few hours, which is where its 24 models earn their keep. If only the evening matters, as for a peak-demand forecast, one model with lags of 24 hours or more is almost as good and 24 times cheaper.

------------------------------------------------------------------------

Back to top
