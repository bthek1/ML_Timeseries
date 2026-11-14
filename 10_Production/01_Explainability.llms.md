# Explainability

Why did the model forecast 2.1 kWh at 19:00? SHAP values from LightGBM’s own tree algorithm, permutation importance by feature group, and why correlated time series features make both harder to read than they look.

Author

Benedict Thekkel

## Two questions

Explainability for a forecaster splits into two questions:

- **Global**: which inputs does the model rely on overall? Answered here by **permutation importance**: shuffle one input across the test rows, re-score, and see how much worse the forecast gets.
- **Local**: why this forecast? Answered by **SHAP values** (Lundberg and Lee, 2017): each feature’s contribution to one prediction, measured against the average prediction, with the contributions adding up exactly to the forecast. For tree ensembles they are exact and fast (Tree SHAP), and LightGBM computes them itself with `predict(X, pred_contrib=True)`, so no extra library is needed.

The model is LightGBM with the [shared features](../01_Foundations/feature_engineering.llms.md) (seasonal lags, calendar, Fourier terms, rolling statistics), trained before the first test origin.

------------------------------------------------------------------------

``` python
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import mae, origins
from ML_Timeseries.data import bar_chart, load_household, show
from ML_Timeseries.features import make_features

# nbdev-test runs pages in parallel; per-page OpenMP pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
O = origins(y.index)
X = make_features(y)
model = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1, n_jobs=1, random_state=0)
model.fit(X.iloc[700:O[0]], y.iloc[700:O[0]])
test_idx = np.concatenate([np.arange(o, o + 24) for o in O])
test_idx = test_idx[~FILLED.to_numpy()[test_idx]]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
base_mae = mae(y_test, model.predict(X_test))
print(f"{X.shape[1]} features; test MAE {base_mae:.3f} kWh over {len(test_idx)} scored hours")
```

    26 features; test MAE 0.436 kWh over 671 scored hours

## 1. Global: permutation importance by feature group

Shuffling one feature at a time understates correlated features: shuffle `sin24_1` and the model still has `cos24_1`, `hour` and the lags carrying the same information. Shuffling **whole groups** together measures what each kind of information is worth.

------------------------------------------------------------------------

``` python
GROUPS = {
    "seasonal lags": [c for c in X if c.startswith("lag")],
    "calendar": ["hour", "dow", "month", "holiday"],
    "Fourier terms": [c for c in X if c.startswith(("sin", "cos"))],
    "rolling statistics": ["mean_24h", "std_24h", "mean_168h", "same_hour_4w"],
}
rng = np.random.default_rng(0)
importance = {}
for name, cols in GROUPS.items():
    worse = []
    for _ in range(5):
        Xp = X_test.copy()
        perm = rng.permutation(len(Xp))
        Xp[cols] = Xp[cols].to_numpy()[perm]  # shuffle the group's rows together, keeping its columns aligned
        worse.append(mae(y_test, model.predict(Xp)) / base_mae - 1)
    importance[name] = np.mean(worse)
imp = pd.Series(importance).sort_values()
show(bar_chart(imp.index, {"MAE increase when shuffled": imp}, horizontal=True, height="260px"))
imp.map("{:+.1%}".format)
```

    calendar               +3.7%
    seasonal lags          +7.0%
    rolling statistics    +10.0%
    Fourier terms         +19.0%
    dtype: str

## 2. Local: SHAP values for one forecast

SHAP values for the 19:00 forecast in the last test window. The bars are each feature’s push away from the model’s average prediction; together with that average they add up to the forecast exactly.

------------------------------------------------------------------------

``` python
row = O[-1] + 19
contrib = model.predict(X.iloc[[row]], pred_contrib=True)[0]
shap = pd.Series(contrib[:-1], index=X.columns)
expected = contrib[-1]
pred = model.predict(X.iloc[[row]])[0]
print(f"{y.index[row]:%a %d %b %H:00}: average prediction {expected:.3f} + sum of SHAP values {shap.sum():+.3f} "
      f"= forecast {pred:.3f} kWh (actual {y.iloc[row]:.3f})")
top = shap.reindex(shap.abs().sort_values(ascending=False).index[:8])[::-1]
show(bar_chart(top.index, {"SHAP value (kWh)": top}, horizontal=True, height="320px"))
```

    Thu 25 Nov 19:00: average prediction 1.095 + sum of SHAP values +1.289 = forecast 2.384 kWh (actual 2.330)

## 3. Global SHAP: mean absolute contribution

Averaging the absolute SHAP values over all test hours gives a second global ranking, per feature:

------------------------------------------------------------------------

``` python
all_shap = pd.DataFrame(model.predict(X_test, pred_contrib=True)[:, :-1], columns=X.columns)
mean_abs = all_shap.abs().mean().sort_values(ascending=False)
group_of = {c: g for g, cols in GROUPS.items() for c in cols}
by_group = mean_abs.groupby(group_of).sum().sort_values(ascending=False)
pd.DataFrame({"top features": mean_abs.head(6).round(3).to_dict()}).T
```

|              | same_hour_4w | sin24_2 | hour | cos8766_1 | lag24 | lag48 |
|--------------|--------------|---------|------|-----------|-------|-------|
| top features | 0.22         | 0.094   | 0.08 | 0.075     | 0.054 | 0.045 |

**The local explanation reads naturally.** For 19:00 on a Thursday in late November, the biggest push above the average comes from `same_hour_4w` (the last four weeks at this hour were high), then a daily Fourier term (evening shape) and the slow annual term (it is nearly winter). The contributions add up to the forecast exactly, which makes them easy to show to someone who asks “why 2.4?”.

**The global rankings disagree with each other, and with the [Feature Engineering](../01_Foundations/feature_engineering.llms.md) ablation.** There, adding the calendar gave the largest gain; here, shuffling the calendar costs the least. Both are correct, because they answer different questions. The daily Fourier terms also encode the hour of day, so once they are in the model, the calendar is largely redundant: shuffle it and the Fourier terms still say it is 19:00. The ablation added the calendar *before* the Fourier terms, so the calendar got the credit for information both groups carry.

That is the general caution for time series features, which are correlated by construction (lags of the same series, several encodings of the same clock):

- importance measures what a feature adds **given the others**, so redundant groups split or lose the credit;
- shuffle correlated features **together**, as groups, or the numbers mostly measure redundancy;
- an explanation describes the **model**, not the household. The model uses `same_hour_4w` heavily because it is a good summary of the routine, not because the household consults it.

------------------------------------------------------------------------

Back to top
