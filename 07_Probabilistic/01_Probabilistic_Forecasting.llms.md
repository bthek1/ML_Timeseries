# Probabilistic Forecasting

Forecasting a range instead of a number: quantiles, the pinball loss, CRPS, and the two checks every interval must pass (does it cover what it claims, and is it no wider than needed).

Author

Benedict Thekkel

## A number is not enough

A point forecast says “1.2 kWh at 19:00”. A decision usually needs more: how high could it plausibly go (to size a battery, or to warn before a peak), how low (to schedule a flexible load). A **probabilistic forecast** answers with a distribution, most often summarised by **quantiles**: the 90% quantile is the value the outcome stays below 90% of the time. Two quantiles make an interval: the 5% and 95% quantiles bound a **90% prediction interval**.

Three questions judge such a forecast:

1.  **Calibration**: does the 90% interval contain the outcome about 90% of the time? Too often means too wide; too rarely means overconfident.
2.  **Sharpness**: among calibrated forecasts, narrower is better. An interval from 0 to 10 kWh always covers and says nothing.
3.  **A single score** that rewards both: the pinball loss and CRPS.

------------------------------------------------------------------------

## 1. The pinball loss and CRPS

The **pinball loss** of a quantile forecast $q_\tau$ at level $\tau$ is

$$L_\tau(y, q) = \begin{cases} \tau\,(y - q) & y \ge q \\ (1 - \tau)\,(q - y) & y < q \end{cases}$$

For the 90% quantile, an outcome above it costs 0.9 per unit of miss; one below costs only 0.1. Minimising it in expectation gives exactly the true 90% quantile, which is why quantile regression trains on it. At $\tau = 0.5$ it is half the absolute error.

The **CRPS** (continuous ranked probability score) is the pinball loss integrated over every level from 0 to 1, times two. It reduces to the absolute error for a point forecast, so it compares probabilistic and point forecasts on the same scale. With a finite set of quantiles, the average pinball loss over those levels approximates it; GIFT-Eval’s CRPS is computed this way.

------------------------------------------------------------------------

    /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/fastcore/docments.py:499: UserWarning: docments unavailable for summarise: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
      warnings.warn(f'docments unavailable for {self.nm}: {type(e).__name__}: {e}')

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/probabilistic.py#L52)

### summarise

``` python
summarise(qb: pandas.core.frame.DataFrame, levels=array([0.05, 0.1 , 0.25, 0.5 , 0.75, 0.9 , 0.95])) -> dict
```

*CRPS, median MAE, and coverage and width of the 80% and 90% intervals.*

    /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/fastcore/docments.py:499: UserWarning: docments unavailable for quantile_backtest: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
      warnings.warn(f'docments unavailable for {self.nm}: {type(e).__name__}: {e}')

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/probabilistic.py#L36)

### quantile_backtest

``` python
quantile_backtest(y: pandas.core.series.Series, forecaster, exclude: pandas.core.series.Series | None = None, levels=array([0.05, 0.1 , 0.25, 0.5 , 0.75, 0.9 , 0.95])) -> pandas.core.frame.DataFrame
```

*Backtest a forecaster returning a (horizon, len(levels)) array; one row per scored hour, quantiles as columns.*

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/probabilistic.py#L30)

### coverage

``` python
def coverage(
    y, lo, hi
)->float:
```

*Share of outcomes inside \[lo, hi\].*

    /opt/hostedtoolcache/Python/3.12.16/x64/lib/python3.12/site-packages/fastcore/docments.py:499: UserWarning: docments unavailable for crps: ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
      warnings.warn(f'docments unavailable for {self.nm}: {type(e).__name__}: {e}')

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/probabilistic.py#L25)

### crps

``` python
crps(y, Q, levels=array([0.05, 0.1 , 0.25, 0.5 , 0.75, 0.9 , 0.95])) -> float
```

*CRPS approximated by twice the mean pinball loss over the quantile `levels` (columns of `Q`).*

------------------------------------------------------------------------

[source](https://github.com/bthek1/ML_Timeseries/blob/main/ML_Timeseries/probabilistic.py#L19)

### pinball

``` python
def pinball(
    y, q, level:float
)->float:
```

*Mean pinball (quantile) loss of quantile forecasts `q` at `level`.*

## 2. Two ways to get quantiles

- **Empirical seasonal quantiles**: take the same hour of the week over the last *k* weeks and read off their quantiles. The probabilistic version of the [seasonal window](../03_Classical/baselines.llms.md) baseline.
- **Quantile regression**: train one LightGBM model per level on the pinball loss, with the [shared features](../01_Foundations/feature_engineering.llms.md). Each model learns its quantile as a function of the hour, the weekday and the recent level.

------------------------------------------------------------------------

``` python
import lightgbm as lgb
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import future_index
from ML_Timeseries.data import line_chart, load_household, show
from ML_Timeseries.features import make_features

# nbdev-test runs pages in parallel; per-page OpenMP pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
train = y.iloc[:origins(y.index)[0]]


def empirical(weeks):
    def f(history, horizon):
        x = history.to_numpy()
        past = np.stack([x[len(x) - 168 * k:len(x) - 168 * k + horizon] for k in range(1, weeks + 1)])
        return np.quantile(past, LEVELS, axis=0).T
    return f


X_train = make_features(train).iloc[700:]
quantile_models = {l: lgb.LGBMRegressor(objective="quantile", alpha=l, n_estimators=300, learning_rate=0.05,
                                        num_leaves=31, verbose=-1, n_jobs=1, random_state=0).fit(X_train, train.iloc[700:])
                   for l in LEVELS}


def quantile_gbm(history, horizon):
    ext = pd.concat([history.iloc[-800:], pd.Series(np.nan, index=future_index(history, horizon))])
    X = make_features(ext).iloc[-horizon:]
    return np.column_stack([quantile_models[l].predict(X) for l in LEVELS])


methods = {"empirical, 4 weeks": empirical(4), "empirical, 8 weeks": empirical(8), "quantile LightGBM": quantile_gbm}
qbs = {name: quantile_backtest(y, f, exclude=FILLED) for name, f in methods.items()}
prob = pd.DataFrame({name: summarise(qb) for name, qb in qbs.items()}).T
prob.style.format({"crps": "{:.3f}", "median MAE": "{:.3f}", "cover 90%": "{:.1%}", "width 90%": "{:.2f}",
                   "cover 80%": "{:.1%}", "width 80%": "{:.2f}"})
```

|                    | crps  | median MAE | cover 90% | width 90% | cover 80% | width 80% |
|--------------------|-------|------------|-----------|-----------|-----------|-----------|
| empirical, 4 weeks | 0.343 | 0.501      | 49.2%     | 0.87      | 44.0%     | 0.77      |
| empirical, 8 weeks | 0.310 | 0.517      | 63.9%     | 1.20      | 56.6%     | 1.01      |
| quantile LightGBM  | 0.243 | 0.419      | 86.3%     | 1.56      | 76.3%     | 1.17      |

**Four weeks of history cannot make a 90% interval.** The 5% and 95% quantiles of four values sit between the smallest and largest of them, so the “90%” band covers only about half the outcomes. Eight weeks helps but is still far off. An empirical quantile needs many samples per tail; this is the same small-sample problem that makes the 4-week median a fine *point* forecast and a poor *interval*.

**Quantile regression comes close**: its 90% interval covers 86% and its 80% interval 76%, a little narrow in both cases, with the best CRPS by a wide margin. It is slightly overconfident because each quantile model learned the spread of the training years, and the test months turn out a little more variable. The [Conformal Prediction](../07_Probabilistic/conformal_prediction.llms.md) page corrects exactly that.

The CRPS ranks the three methods the same way as the median MAE, and it also rewards the quantile model’s calibration: two forecasts with similar medians can differ a lot in CRPS when one of them has honest tails.

------------------------------------------------------------------------

## 3. What a calibrated band looks like

The last test window from quantile LightGBM: the median, the 90% band, and what happened. A good band is narrow at night, when the house is predictable, and wide in the evening, when it is not.

------------------------------------------------------------------------

``` python
qb = qbs["quantile LightGBM"]
last = qb[qb.window == qb.window.max()]
show(line_chart(last.index.strftime("%a %H:00"), {"actual": last.y, "5%": last["q0.05"], "median": last["q0.5"], "95%": last["q0.95"]},
                y_name="kWh", height="340px", zoom=False))
```

``` python
by_hour = qb.assign(width=qb["q0.95"] - qb["q0.05"]).groupby(qb.index.hour).width.mean()
print(f"mean 90% width: {by_hour.loc[[2, 3, 4]].mean():.2f} kWh at 02:00-04:00, {by_hour.loc[[18, 19, 20]].mean():.2f} kWh at 18:00-20:00")
```

    mean 90% width: 0.67 kWh at 02:00-04:00, 2.19 kWh at 18:00-20:00

Back to top
