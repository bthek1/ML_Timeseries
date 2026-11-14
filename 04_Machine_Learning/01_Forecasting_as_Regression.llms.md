# Forecasting as Regression

Turning a series into a supervised table: windows, targets and the one rule that keeps them honest, then a linear model and a tree model on the same table.

Author

Benedict Thekkel

## The reduction

Any regression model can forecast once the series is rewritten as rows. Each row is one moment in the past, playing the role of a forecast origin:

- **inputs**: what was known at that moment (recent values, the calendar of the hours ahead);
- **target**: what happened next.

Training on thousands of such rows teaches the model the mapping from “what I knew” to “what came next”. This **reduction** of forecasting to regression is what lets scikit-learn, LightGBM or a neural network forecast at all. Three choices define it:

| Choice | Options | This page |
|----|----|----|
| Window | how far back the inputs reach | 336 hours (two weeks) |
| Target | one step, one specific step, or all 24 at once | all 24 at once (one row per day) |
| Split | which rows train and which test | rows before the first origin train; the [backtest](../02_Evaluation/backtesting.llms.md) tests |

The split is where most mistakes happen. Rows overlap: the row for Tuesday’s forecast and the row for Wednesday’s share 312 of their 336 inputs. A random split puts near-copies of each test row in the training set, which is [leak 1](../02_Evaluation/backtesting.llms.md). Every row in training must end before every row in testing begins.

------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import backtest, origins, record, score
from ML_Timeseries.data import line_chart, load_household, show

# nbdev-test runs pages in parallel; per-page OpenMP and BLAS pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "04_Machine_Learning/01_Forecasting_as_Regression"
WINDOW, H = 336, 24
first_test = origins(y.index)[0]
train = y.iloc[:first_test]
```

## 1. Building the table

One row per midnight in the training data: the 336 hours before it, a one-hot encoding of the weekday being forecast, and the 24 hours that followed as a 24-column target.

------------------------------------------------------------------------

``` python
def daily_rows(s: pd.Series):
    "Inputs (last WINDOW values + weekday one-hot) and 24-hour targets for every midnight with a full window."
    x = s.to_numpy()
    mids = [i for i in range(WINDOW, len(s) - H + 1) if s.index[i].hour == 0]
    X = np.stack([x[i - WINDOW:i] for i in mids])
    dow = np.eye(7)[[s.index[i].dayofweek for i in mids]]
    Y = np.stack([x[i:i + H] for i in mids])
    return np.column_stack([X, dow]), Y


X_train, Y_train = daily_rows(train)
print(f"{X_train.shape[0]} training rows, {X_train.shape[1]} inputs each, {Y_train.shape[1]} targets each")


def as_forecaster(model):
    def f(history, horizon):
        nxt = history.index[-1] + pd.Timedelta(hours=1)
        row = np.concatenate([history.to_numpy()[-WINDOW:], np.eye(7)[nxt.dayofweek]])
        return model.predict(row[None])[0][:horizon]
    return f
```

    1280 training rows, 343 inputs each, 24 targets each

## 2. A linear model: ridge regression

With all 24 hours as targets, one ridge regression is 24 linear autoregressions sharing their inputs: each output hour is a weighted sum of the last two weeks plus a weekday offset. The ridge penalty keeps 343 weights per output from fitting noise. This multi-output form is called **MIMO** (multiple input, multiple output).

------------------------------------------------------------------------

``` python
ridge = Ridge(alpha=100.0).fit(X_train, Y_train)
ridge_score = score(backtest(y, as_forecaster(ridge)), y, exclude=FILLED)
_ = record("Ridge, 336 h window to 24 h (MIMO)", "machine learning", ridge_score, notebook=NB)
print(f"ridge: MASE {ridge_score['mase']:.3f}, skill {ridge_score['skill']:+.1%}")
```

    ridge: MASE 0.701, skill +12.4%

It is tempting to read the weights as “what the model learned”. They resist that reading. For the 19:00 output, the largest weights sit at scattered lags, with alternating signs, not on the seasonal lags the [autocorrelation](../01_Foundations/decomposition_stationarity.llms.md) highlights:

------------------------------------------------------------------------

``` python
lag_to_target = np.arange(WINDOW, 0, -1) + 19  # input k hours before midnight is k + 19 hours before 19:00
w19 = pd.Series(ridge.coef_[19][:WINDOW], index=lag_to_target)
top = w19.abs().sort_values(ascending=False).head(6)
print("largest weights for the 19:00 output, by lag in hours:", {int(lag): round(float(w19[lag]), 3) for lag in top.index})
```

    largest weights for the 19:00 output, by lag in hours: {348: -0.121, 212: -0.117, 38: -0.106, 158: 0.106, 326: 0.104, 349: 0.092}

``` python
show(line_chart(lag_to_target, {"weight on the input at this lag": w19.to_numpy()}, y_name="weight", height="300px"))
```

The inputs are strongly correlated with their neighbours (hour *t* looks like hour *t-1*), so many different weight patterns give nearly the same forecast, and ridge picks one that spreads weight in positive and negative pairs. The forecast is reliable; the individual coefficients are not. Interpreting a model of correlated lags needs tools built for it, such as permutation importance on grouped features, which the production section returns to.

## 3. A tree model on the same table

Gradient boosting cannot output 24 numbers at once, so `MultiOutputRegressor` trains one model per output hour. Same rows, same split, nonlinear model:

------------------------------------------------------------------------

``` python
gbm = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=150, learning_rate=0.05, random_state=0)).fit(X_train, Y_train)
gbm_score = score(backtest(y, as_forecaster(gbm)), y, exclude=FILLED)
_ = record("Gradient boosting, 336 h window to 24 h (one model per hour)", "machine learning", gbm_score, notebook=NB)
print(f"gradient boosting on the raw window: MASE {gbm_score['mase']:.3f}, skill {gbm_score['skill']:+.1%}")
```

    gradient boosting on the raw window: MASE 0.655, skill +18.1%

Gradient boosting on the same raw table beats ridge by a wide margin, and edges past both the engineered-feature model from [Feature Engineering](../01_Foundations/feature_engineering.llms.md) and [SARIMA](../03_Classical/arima.llms.md): it is the best model on the scoreboard so far. Two differences explain it. It trains one model per output hour, so each hour gets its own mapping (the **direct** strategy, compared properly on the [next page](../04_Machine_Learning/multistep_strategies.llms.md)). And trees can act on the recent level nonlinearly (“if yesterday evening ran high, raise tonight”), where ridge can only add up weighted past values.

The cost is 24 models instead of one, and about 40 seconds of training on one core instead of under one second.

------------------------------------------------------------------------

Back to top
