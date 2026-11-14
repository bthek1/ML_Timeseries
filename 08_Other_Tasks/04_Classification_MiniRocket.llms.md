# Time Series Classification with MiniRocket

Labelling whole series instead of forecasting them: MiniRocket’s random convolutions against catch22’s hand-picked features, on ItalyPowerDemand and on this house’s weekdays and weekends.

Author

Benedict Thekkel

## The task

**Classification** assigns a label to a whole series: this day’s load profile is a winter day or a summer day; this vibration trace is a healthy or a failing bearing; this meter belongs to a house or a shop. The inputs are many short series of equal length; the output is one label each.

**ROCKET** (Dempster, Petitjean and Webb, 2020) made this fast and accurate with a deliberately simple idea: apply thousands of **random convolution kernels** to each series, record two numbers per kernel (the maximum response and the proportion of positive values), and train a linear classifier on the result. No kernel is learned. **MiniRocket** (2021) makes the kernels almost deterministic and is about 75 times faster, with the same accuracy. On the 128-dataset UCR archive, these methods matched far more complex deep models.

The comparison here is **catch22**, the 22 hand-picked shape features from the [Feature Engineering](../01_Foundations/feature_engineering.llms.md) page, with a logistic regression on top. Both come from `sktime`.

------------------------------------------------------------------------

``` python
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sktime.datasets import load_italy_power_demand
from sktime.transformations.panel.catch22 import Catch22
from sktime.transformations.panel.rocket import MiniRocket
from threadpoolctl import threadpool_limits

from ML_Timeseries.data import bar_chart, line_chart, load_household, show

# nbdev-test runs pages in parallel; per-page thread pools would oversubscribe the 4 cores.
_ = threadpool_limits(1)
```

## 1. ItalyPowerDemand

A standard UCR benchmark that ships with `sktime`: 24-hour electricity demand profiles from Italy in 1997, labelled by season (October to March, or April to September). It has 67 training days and 1,029 test days, so the classifier must generalise from very little.

------------------------------------------------------------------------

``` python
X_train, y_train = load_italy_power_demand(split="train", return_type="numpy3D")
X_test, y_test = load_italy_power_demand(split="test", return_type="numpy3D")
print(f"train {X_train.shape}, test {X_test.shape}, classes {sorted(str(c) for c in set(y_train))}")
show(line_chart([f"{h:02d}" for h in range(24)],
                {f"class {c}, mean profile": X_train[y_train == c, 0].mean(axis=0) for c in sorted(set(y_train))},
                y_name="normalised demand", height="280px", zoom=False))
```

    train (67, 1, 24), test (1029, 1, 24), classes ['1', '2']

``` python
def minirocket_accuracy(Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    mr = MiniRocket(random_state=0).fit(Xtr)
    clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)).fit(mr.transform(Xtr), ytr)
    return clf.score(mr.transform(Xte), yte), time.perf_counter() - t0


def catch22_accuracy(Xtr, ytr, Xte, yte):
    t0 = time.perf_counter()
    c22 = Catch22().fit(Xtr)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(c22.transform(Xtr).fillna(0), ytr)
    return clf.score(c22.transform(Xte).fillna(0), yte), time.perf_counter() - t0


italy = {}
for name, fn in (("MiniRocket", minirocket_accuracy), ("catch22 + logistic", catch22_accuracy)):
    acc, secs = fn(X_train, y_train, X_test, y_test)
    italy[name] = {"test accuracy": acc, "seconds": secs}
italy = pd.DataFrame(italy).T
italy.style.format({"test accuracy": "{:.1%}", "seconds": "{:.1f}"})
```

|                    | test accuracy | seconds |
|--------------------|---------------|---------|
| MiniRocket         | 96.5%         | 0.4     |
| catch22 + logistic | 84.3%         | 1.2     |

## 2. Weekday or weekend in this house

The same two methods on this site’s data: each of the 1,440 days’ 24-hour profiles, labelled weekday or weekend, scored by 5-fold cross-validation. On the Feature Engineering page catch22 managed 75% against a 71.5% base rate (always guessing “weekday”).

------------------------------------------------------------------------

``` python
y_h = load_household()["kwh"]
start = y_h.index[y_h.index.hour == 0][0]
days = y_h[start:]
n = len(days) // 24
X_days = days.to_numpy()[:n * 24].reshape(n, 1, 24)
weekend = np.where(days.index[::24][:n].dayofweek >= 5, "weekend", "weekday")
base = (weekend == "weekday").mean()

mr_feats = MiniRocket(random_state=0).fit(X_days).transform(X_days)
acc_mr = cross_val_score(RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)), mr_feats, weekend, cv=5).mean()
c22_feats = Catch22().fit_transform(X_days).fillna(0)
acc_c22 = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)), c22_feats, weekend, cv=5).mean()
house = pd.Series({"always 'weekday'": base, "catch22 + logistic": acc_c22, "MiniRocket": acc_mr}, name="accuracy")
show(bar_chart(house.index, {"accuracy": house}, height="280px"))
house.map("{:.1%}".format)
```

    always 'weekday'      71.5%
    catch22 + logistic    75.0%
    MiniRocket            78.9%
    Name: accuracy, dtype: str

On ItalyPowerDemand, MiniRocket is right on about 96% of test days after seeing 67 examples, in well under a second, and catch22 trails it by 12 points. Twenty-two summary features of the whole series lose information that thousands of random convolutions keep. Each kernel responds to a local pattern at one of several dilations (time scales), and the proportion of the day where the response is positive records how much of the day has that pattern, so ten thousand of them describe the profile far more finely than 22 global statistics.

On this house the gap is smaller and both stay close to the base rate. Weekend and weekday days in one household overlap far more than winter and summer days across a whole country: a Saturday spent at home and a Wednesday spent at home look alike. MiniRocket still extracts more than catch22, and both confirm what the [forecasting pages](../02_Evaluation/scoreboard.llms.md) already showed: this household’s day-to-day variation is mostly noise around a weak weekly pattern.

As a default for a new classification problem on short series, MiniRocket is hard to beat for the effort: no tuning, CPU only, seconds to train.

------------------------------------------------------------------------

Back to top
