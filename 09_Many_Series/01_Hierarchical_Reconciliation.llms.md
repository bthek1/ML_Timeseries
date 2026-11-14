# Hierarchical Reconciliation

When forecasts must add up: the house total and its four circuits forecast separately disagree with their own sum. Bottom-up, top-down, OLS, WLS and MinT reconciliation, written in numpy, and which one helps which level.

Author

Benedict Thekkel

## The problem

Many series come in hierarchies: a house and its circuits, a region and its substations, a company and its stores. Forecast every node separately and the forecasts will not add up: the forecast for the house will differ from the sum of the forecasts for its circuits. A planner then has to choose which number to believe.

**Reconciliation** makes the forecasts **coherent**: every parent equals the sum of its children. The modern view (Hyndman et al., 2011; Wickramasuriya, Athanasopoulos and Hyndman, 2019) writes every method the same way. Stack the base forecasts of all nodes in a vector $\hat y$, and let the **summing matrix** $S$ map the bottom-level series to all nodes. A reconciliation method is a matrix $P$ that maps the base forecasts to bottom-level forecasts; the coherent forecasts are then

$$\tilde y = S\,P\,\hat y$$

- **Bottom-up**: $P$ keeps the bottom-level forecasts. Uses only the leaves.
- **Top-down**: $P$ splits the total by historical proportions. Uses only the total.
- **OLS**: $P = (S^\top S)^{-1} S^\top$. Uses all levels with equal weight.
- **WLS**: $P = (S^\top W^{-1} S)^{-1} S^\top W^{-1}$, with $W$ the diagonal of the base forecasts’ error variances. Weights each level by its accuracy.
- **MinT**: the same formula with $W$ the full error covariance (shrunk), so it also uses how the levels’ errors move together.

MinT (“minimum trace”) minimises the total variance of the coherent forecast errors, and it is optimal when $W$ is known.

Here the hierarchy is this house: the total and its four parts (kitchen, laundry, water heater and AC, everything else). Nixtla’s `hierarchicalforecast` library implements all of these; at five nodes, the matrix algebra is shorter written out.

------------------------------------------------------------------------

``` python
import numpy as np
import pandas as pd

from ML_Timeseries.backtest import origins
from ML_Timeseries.data import bar_chart, load_household, show
```

``` python
hh = load_household()
FILLED = hh["filled"].to_numpy()
O = origins(hh.index)
NODES = ["kwh", "sub1_kwh", "sub2_kwh", "sub3_kwh", "other_kwh"]
LABELS = ["total", "kitchen", "laundry", "heater + AC", "other"]
S = np.vstack([np.ones(4), np.eye(4)])  # total = sum of the four leaves; each leaf is itself
values = {c: hh[c].to_numpy() for c in NODES}
print("summing matrix S (rows: total, kitchen, laundry, heater + AC, other):")
print(S.astype(int))


def base_forecasts(o):
    "Seasonal window median (same hour, last 4 weeks) for every node: a 5 x 24 array."
    return np.stack([np.median(np.stack([values[c][o - 168 * k:o - 168 * k + 24] for k in range(1, 5)]), axis=0) for c in NODES])


def actuals(o):
    return np.stack([values[c][o:o + 24] for c in NODES])
```

    summing matrix S (rows: total, kitchen, laundry, heater + AC, other):
    [[1 1 1 1]
     [1 0 0 0]
     [0 1 0 0]
     [0 0 1 0]
     [0 0 0 1]]

## 1. The base forecasts do not add up

Every node gets the same simple base forecast, the [seasonal window median](../03_Classical/baselines.llms.md). The median of a sum is not the sum of the medians, so the forecasts disagree:

------------------------------------------------------------------------

``` python
gaps = [np.abs(base_forecasts(o)[0] - base_forecasts(o)[1:].sum(axis=0)).max() for o in O]
print(f"largest gap between the total forecast and the sum of its parts: {max(gaps):.2f} kWh in one hour")
```

    largest gap between the total forecast and the sum of its parts: 0.91 kWh in one hour

## 2. Reconciling

The error covariance $W$ comes from the base forecasts’ errors on the 60 days before the first test origin. A full 5 x 5 covariance estimated from data is noisy, so MinT uses the standard **shrinkage** towards its diagonal (half and half here).

------------------------------------------------------------------------

``` python
history = [O[0] - 24 * k for k in range(1, 61)]
residuals = np.concatenate([actuals(o) - base_forecasts(o) for o in history], axis=1)
W = np.cov(residuals)
W_shrunk = 0.5 * np.diag(np.diag(W)) + 0.5 * W
proportions = np.array([hh[c].iloc[:O[0]].sum() for c in NODES[1:]])
proportions = proportions / proportions.sum()


def P_matrix(method):
    if method == "bottom-up":
        return np.hstack([np.zeros((4, 1)), np.eye(4)])
    if method == "top-down":
        return np.hstack([proportions[:, None], np.zeros((4, 4))])
    Winv = np.linalg.inv({"OLS": np.eye(5), "WLS": np.diag(np.diag(W)), "MinT (shrink)": W_shrunk}[method])
    return np.linalg.inv(S.T @ Winv @ S) @ S.T @ Winv


METHODS = ["base (incoherent)", "bottom-up", "top-down", "OLS", "WLS", "MinT (shrink)"]
rows, incoherence = {}, {}
for m in METHODS:
    errs, worst = [], 0.0
    for o in O:
        Yhat = base_forecasts(o)
        R = Yhat if m.startswith("base") else S @ P_matrix(m) @ Yhat
        keep = ~FILLED[o:o + 24]
        errs.append(np.abs(R - actuals(o))[:, keep])
        worst = max(worst, np.abs(R[0] - R[1:].sum(axis=0)).max())
    E = np.concatenate(errs, axis=1)
    rows[m] = dict(zip(LABELS, E.mean(axis=1)))
    incoherence[m] = worst
mae = pd.DataFrame(rows).T
mae["max incoherence (kWh)"] = pd.Series(incoherence)
mae.style.format("{:.4f}")
```

|   | total | kitchen | laundry | heater + AC | other | max incoherence (kWh) |
|----|----|----|----|----|----|----|
| base (incoherent) | 0.5015 | 0.0583 | 0.0669 | 0.2646 | 0.2068 | 0.9093 |
| bottom-up | 0.4793 | 0.0583 | 0.0669 | 0.2646 | 0.2068 | 0.0000 |
| top-down | 0.5015 | 0.0811 | 0.0783 | 0.2991 | 0.2519 | 0.0000 |
| OLS | 0.4957 | 0.0701 | 0.0745 | 0.2702 | 0.2097 | 0.0000 |
| WLS | 0.4875 | 0.0627 | 0.0681 | 0.2717 | 0.2076 | 0.0000 |
| MinT (shrink) | 0.4875 | 0.0626 | 0.0681 | 0.2717 | 0.2076 | 0.0000 |

``` python
rel = mae[LABELS].div(mae.loc["base (incoherent)", LABELS]) - 1
show(bar_chart(LABELS, {m: rel.loc[m] for m in METHODS[1:]}, y_name="MAE change vs base", height="360px"))
```

Every reconciled forecast is coherent to rounding error, so the planner’s dilemma is gone. Which method to use depends on which level matters:

- **Bottom-up is best for the total**, about 4% better than forecasting the total directly, and by construction it leaves the circuits unchanged. The sum of the circuits’ medians is simply a better point forecast of the total than the total’s own median; for skewed, spiky parts the two differ, and on these windows the sum of the parts is closer.
- **WLS and MinT** improve the total by about 3% but make the sparse circuits slightly worse: they move every node towards a compromise, and the kitchen’s near-zero forecasts are pulled up by the total. MinT adds nothing over WLS here; with five nodes and 60 days, the off-diagonal covariances are mostly noise and the shrinkage discards them.
- **Top-down is worst for the circuits**: one fixed share per circuit cannot know that the kitchen peaks at dinner and the water heater at night.
- **OLS** ignores that the circuits’ errors differ in size, and is worse than WLS on four of the five nodes.

The textbook result (MinT is optimal) assumes mean forecasts and well-estimated covariances. These are median forecasts of spiky series with a short calibration history, so the simplest coherent method wins on the level that usually matters. With hundreds of nodes, such as a utility’s substations and feeders, the covariance carries real information and MinT’s advantage returns.

------------------------------------------------------------------------

Back to top
