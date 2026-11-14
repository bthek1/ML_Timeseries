# Conformal Prediction

Calibrated intervals around any forecaster, with no distributional assumption: split conformal, conformalised quantile regression, and adaptive conformal inference for a series that drifts. Plus how uncertain a coverage number from 30 days really is.

Author

Benedict Thekkel

## The idea

**Conformal prediction** (Vovk and colleagues, from 2005) turns any point forecaster into an interval forecaster with a coverage guarantee. The recipe:

1.  Forecast a **calibration** set the model did not train on, and record a **nonconformity score** for each outcome, usually the absolute error $|y - \hat y|$.
2.  Take the $\lceil (n+1)(1-\alpha) \rceil / n$ empirical quantile of those scores, $\hat q$.
3.  The interval for a new forecast is $\hat y \pm \hat q$.

If the calibration errors and the new error are **exchangeable** (any order equally likely), the interval covers the outcome with probability at least $1 - \alpha$, whatever the model and whatever the error distribution. Time series break exchangeability: errors cluster, and the distribution drifts. So the guarantee becomes an approximation, and the test is empirical.

Every method here wraps the quantile LightGBM from [Probabilistic Forecasting](../07_Probabilistic/probabilistic_forecasting.llms.md) (its median as the point forecast, its 5% and 95% quantiles where needed). Calibration uses the 30 days before each test origin, so it always reflects the recent past. The target is a 90% interval, with 88% to 92% fixed as the success band before running anything.

------------------------------------------------------------------------

``` python
import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ML_Timeseries.backtest import future_index, origins
from ML_Timeseries.data import bar_chart, line_chart, load_household, show
from ML_Timeseries.features import make_features
from ML_Timeseries.probabilistic import LEVELS, coverage

# nbdev-test runs pages in parallel; per-page OpenMP pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
O = origins(y.index)
train = y.iloc[:O[0]]
X_train = make_features(train).iloc[700:]
models = {l: lgb.LGBMRegressor(objective="quantile", alpha=l, n_estimators=300, learning_rate=0.05, num_leaves=31,
                               verbose=-1, n_jobs=1, random_state=0).fit(X_train, train.iloc[700:]) for l in (0.05, 0.5, 0.95)}
_cache = {}


def quantiles(origin):
    "5%, 50% and 95% forecasts for the 24 hours after `origin` (cached: calibration windows overlap)."
    if origin not in _cache:
        hist = y.iloc[:origin]
        ext = pd.concat([hist.iloc[-800:], pd.Series(np.nan, index=future_index(hist, 24))])
        X = make_features(ext).iloc[-24:]
        _cache[origin] = np.sort(np.column_stack([models[l].predict(X) for l in (0.05, 0.5, 0.95)]), axis=1)
    return _cache[origin]


def actual(origin):
    return y.iloc[origin:origin + 24].to_numpy()


def observed(origin):
    return ~FILLED.iloc[origin:origin + 24].to_numpy()


CAL_DAYS = 30
ALPHA = 0.1
```

## 1. Four ways to build the interval

- **Uncalibrated quantiles**: the LightGBM 5% and 95% quantiles as they are.
- **Split conformal**: median $\pm \hat q$, with $\hat q$ from the absolute errors of the previous 30 days. Same width every hour.
- **CQR** (conformalised quantile regression; Romano et al., 2019): keep the quantile model’s shape and widen or narrow it by a conformal margin. The score is how far the outcome fell outside the quantile band, so the interval stays narrow at night and wide in the evening.
- **ACI** (adaptive conformal inference; Gibbs and Candès, 2021): CQR with a level that **adapts**. After each day, if the interval missed more than 10% of hours, the next day’s level rises, and if it missed less, it falls: $\alpha_{t+1} = \alpha_t + \gamma\,(\alpha - \text{miss rate}_t)$, with $\gamma = 0.05$ fixed in advance. Built for distributions that drift.

------------------------------------------------------------------------

``` python
def conformal_quantile(scores, alpha):
    n = len(scores)
    return np.quantile(scores, min(1.0, max(0.0, np.ceil((n + 1) * (1 - alpha)) / n)))


def run(method, gamma=0.05):
    alpha, hits, widths = ALPHA, [], []
    for o in O:
        cal = [o - 24 * k for k in range(1, CAL_DAYS + 1)]
        Q = quantiles(o)
        if method == "uncalibrated quantiles":
            lo, hi = Q[:, 0], Q[:, 2]
        elif method == "split conformal":
            q = conformal_quantile(np.concatenate([np.abs(actual(c) - quantiles(c)[:, 1]) for c in cal]), ALPHA)
            lo, hi = Q[:, 1] - q, Q[:, 1] + q
        else:  # CQR, and ACI on top of it
            s = np.concatenate([np.maximum(quantiles(c)[:, 0] - actual(c), actual(c) - quantiles(c)[:, 2]) for c in cal])
            q = conformal_quantile(s, alpha if method == "ACI" else ALPHA)
            lo, hi = Q[:, 0] - q, Q[:, 2] + q
        keep = observed(o)
        hit = ((actual(o) >= lo) & (actual(o) <= hi))[keep]
        hits.append(hit)
        widths.append(np.mean((hi - lo)[keep]) if keep.any() else np.nan)
        if method == "ACI" and keep.any():  # a fully gap-filled day carries no information about coverage
            alpha += gamma * (ALPHA - (1 - hit.mean()))
    daily = np.array([h.mean() for h in hits if len(h)])
    return {"coverage": float(np.mean(np.concatenate(hits))), "mean width": float(np.nanmean(widths)),
            "daily coverage std": float(daily.std(ddof=1)), "days": len(daily)}, daily


results, daily = {}, {}
for m in ("uncalibrated quantiles", "split conformal", "CQR", "ACI"):
    results[m], daily[m] = run(m)
table = pd.DataFrame(results).T
table["coverage 95% CI"] = [f"{c - 1.96 * s / np.sqrt(n):.0%} to {c + 1.96 * s / np.sqrt(n):.0%}"
                            for c, s, n in zip(table.coverage, table["daily coverage std"], table.days)]
table[["coverage", "coverage 95% CI", "mean width"]].style.format({"coverage": "{:.1%}", "mean width": "{:.2f}"})
```

|                        | coverage | coverage 95% CI | mean width |
|------------------------|----------|-----------------|------------|
| uncalibrated quantiles | 86.3%    | 82% to 90%      | 1.55       |
| split conformal        | 86.6%    | 81% to 92%      | 1.79       |
| CQR                    | 86.4%    | 82% to 91%      | 1.56       |
| ACI                    | 88.2%    | 84% to 92%      | 1.61       |

``` python
show(bar_chart(table.index, {"coverage of the 90% interval": table.coverage}, height="300px"))
```

Three findings:

- **ACI reaches the target band** (88% to 92%), the only method that does. Its level rises after days that missed too often, so it tracks the test months’ higher variability instead of trusting last month’s errors.
- **CQR is the efficient choice.** It matches split conformal’s coverage with intervals about 13% narrower, because it keeps the quantile model’s hour-by-hour shape: tight at night, wide in the evening. Split conformal pays for one width everywhere.
- **Plain split conformal and CQR did not fix the under-coverage.** Their calibration is the previous 30 days, and the test windows were a little more variable than the days before them. Exchangeability, the assumption behind the guarantee, does not hold for a drifting series, which is exactly the gap ACI was designed to close.

**And the honest caveat:** 30 test days measure coverage to about plus or minus four points. Hours within a day are strongly correlated, so the effective sample is the 30 days, not the 700 hours. Every method’s interval in the table contains 90%. “86% versus 88%” is a real difference in the methods’ behaviour on this data, but it is not one these 30 days can confirm statistically. A coverage claim needs many independent periods, or it is an anecdote with a percentage sign.

------------------------------------------------------------------------

## 2. Coverage day by day

The total hides how coverage moves. ACI’s daily hit rate against the target, with the uncalibrated quantiles for comparison:

------------------------------------------------------------------------

``` python
days = [y.index[o].strftime("%d %b") for o in O if observed(o).any()]
show(line_chart(days, {"ACI": daily["ACI"], "uncalibrated quantiles": daily["uncalibrated quantiles"], "target": np.full(len(days), 0.9)},
                y_name="share of hours covered", height="320px", zoom=False))
```

Back to top
