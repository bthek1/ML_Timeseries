# The Deep Learning Data Pipeline

Windows, lookback length and normalisation, built in plain PyTorch with one small network: the choices that decide whether a deep forecaster learns anything, measured before any architecture is compared.

Author

Benedict Thekkel

## Why the pipeline comes first

Deep forecasting papers compare architectures. In practice, three choices made before the architecture often matter more:

1.  **Windows**: the network sees fixed-length slices of history (the **lookback**) and predicts a fixed-length slice of future (the **horizon**). How long should the lookback be?
2.  **Normalisation**: neural networks train badly on raw values that drift. Normalise globally, or per window?
3.  **Sampling**: which windows go into each training batch.

This page fixes one small network, a multi-layer perceptron (MLP) from the 336 inputs (or other lookbacks) to the 24 outputs, and changes only these choices. Everything runs on the GPU in seconds.

This page trains on the GPU, so it does not run in CI: it carries `skip_exec` and stores its outputs from a run on the site’s machine (RTX 3060, 12 GB).

------------------------------------------------------------------------

``` python
import gc
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from ML_Timeseries.backtest import backtest, origins, record, score
from ML_Timeseries.data import bar_chart, load_household, show

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}" + (f" ({torch.cuda.get_device_name()})" if DEVICE == "cuda" else ""))
```

    device: cuda (NVIDIA GeForce RTX 3060)

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "05_Deep_Learning/01_DL_Data_Pipeline"
H = 24
first_test = origins(y.index)[0]
train_values = y.iloc[:first_test].to_numpy(dtype=np.float32)
```

## 1. Windows

Every hour of the training data can end a lookback, so a series of 29,000 hours gives about 29,000 overlapping training windows. Unlike the [regression page](../04_Machine_Learning/forecasting_as_regression.llms.md), which used one row per midnight, the network trains on windows starting at every hour: more examples of the same patterns, at every phase of the day. The calendar is given as the hour of day and weekday of the first forecast hour.

`torch.Tensor.unfold` builds all windows as a view, without copying the series 29,000 times.

------------------------------------------------------------------------

``` python
def make_windows(values: np.ndarray, index: pd.DatetimeIndex, lookback: int):
    "All (input, calendar, target) windows of a series, as tensors."
    v = torch.tensor(values)
    x = v[:-H].unfold(0, lookback, 1)                 # [n, lookback]
    target = v[lookback:].unfold(0, H, 1)             # [n, H]
    n = min(len(x), len(target))
    first = index[lookback:lookback + n]              # timestamp of the first forecast hour
    cal = torch.tensor(np.column_stack([np.eye(24)[first.hour], np.eye(7)[first.dayofweek]]), dtype=torch.float32)
    return x[:n], cal, target[:n]


x, cal, t = make_windows(train_values, y.index[:first_test], 336)
print(f"{len(x):,} windows: inputs {tuple(x.shape)}, calendar {tuple(cal.shape)}, targets {tuple(t.shape)}")
```

    30,704 windows: inputs (30704, 336), calendar (30704, 31), targets (30704, 24)

## 2. Normalisation

Each setting below trains three networks with different random seeds. The table shows their mean skill, the spread between the best and worst seed, and the skill of their averaged forecast (the **seed ensemble**). A difference smaller than the spread is not a difference.

Three options, all inside the model so the backtest sees raw kWh:

| Option | Inputs and targets divided by | Effect |
|----|----|----|
| **None** | nothing | the network must learn the level itself |
| **Global** | the training mean and standard deviation | fixed scale; drift in the level still shows |
| **Instance (RevIN)** | each window’s own mean and standard deviation, undone on the output | every window looks “level zero”; the network learns shape, and the level is added back |

**RevIN** (reversible instance normalisation; Kim et al., 2022) is the third option with a learnable affine step. It is the single trick most of the modern architectures on the next pages share, because it removes the distribution shift between training years and test months.

------------------------------------------------------------------------

``` python
class MLP(nn.Module):
    def __init__(self, lookback: int, norm: str, mean: float = 0.0, std: float = 1.0, hidden: int = 256):
        super().__init__()
        self.norm, self.mean, self.std = norm, mean, std
        self.net = nn.Sequential(nn.Linear(lookback + 31, hidden), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, H))

    def forward(self, x, cal):
        if self.norm == "instance":
            mu, sd = x.mean(1, keepdim=True), x.std(1, keepdim=True) + 1e-3
        elif self.norm == "global":
            mu, sd = self.mean, self.std
        else:
            mu, sd = 0.0, 1.0
        out = self.net(torch.cat([(x - mu) / sd, cal], 1))
        return out * sd + mu


def train_mlp(lookback: int, norm: str, epochs: int = 15, seed: int = 0):
    torch.manual_seed(seed)
    x, cal, t = make_windows(train_values, y.index[:first_test], lookback)
    model = MLP(lookback, norm, float(train_values.mean()), float(train_values.std())).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x, cal, t = x.to(DEVICE), cal.to(DEVICE), t.to(DEVICE)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(x), device=DEVICE).split(512):
            loss = (model(x[idx], cal[idx]) - t[idx]).abs().mean()  # L1 loss: the backtest scores absolute error
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def mlp_forecaster(model, lookback):
    def f(history, horizon):
        first = history.index[-1] + pd.Timedelta(hours=1)
        xi = torch.tensor(history.to_numpy(dtype=np.float32)[-lookback:])[None].to(DEVICE)
        ci = torch.tensor(np.concatenate([np.eye(24)[first.hour], np.eye(7)[first.dayofweek]]), dtype=torch.float32)[None].to(DEVICE)
        with torch.no_grad():
            return model(xi, ci)[0, :horizon].cpu().numpy()
    return f


SEEDS = (0, 1, 2)


def run(lookback, norm):
    "Train one network per seed; report the mean and spread of their skill, and score the seed ensemble."
    t0 = time.perf_counter()
    models = [train_mlp(lookback, norm, seed=s) for s in SEEDS]
    singles = [score(backtest(y, mlp_forecaster(m, lookback)), y, exclude=FILLED)["skill"] for m in models]
    fs = [mlp_forecaster(m, lookback) for m in models]
    ensemble = score(backtest(y, lambda h, n: np.mean([f(h, n) for f in fs], axis=0)), y, exclude=FILLED)
    ensemble |= {"skill mean": float(np.mean(singles)), "skill spread": float(np.max(singles) - np.min(singles)),
                 "seconds": time.perf_counter() - t0}
    del models, fs
    gc.collect()
    torch.cuda.empty_cache()
    return ensemble
```

``` python
norm_results = {norm: run(336, norm) for norm in ("none", "global", "instance")}
COLS = ["skill mean", "skill spread", "skill", "mase", "seconds"]
FMT = {"skill mean": "{:+.1%}", "skill spread": "{:.1%}", "skill": "{:+.1%}", "mase": "{:.3f}", "seconds": "{:.0f}"}
norm_table = pd.DataFrame(norm_results).T[COLS].rename(columns={"skill": "ensemble skill", "mase": "ensemble MASE"})
norm_table.style.format({"skill mean": "{:+.1%}", "skill spread": "{:.1%}", "ensemble skill": "{:+.1%}", "ensemble MASE": "{:.3f}", "seconds": "{:.0f}"})
```

|          | skill mean | skill spread | ensemble skill | ensemble MASE | seconds |
|----------|------------|--------------|----------------|---------------|---------|
| none     | +17.5%     | 1.0%         | +18.0%         | 0.658         | 3       |
| global   | +17.7%     | 2.7%         | +19.2%         | 0.645         | 2       |
| instance | +17.0%     | 1.5%         | +18.4%         | 0.652         | 2       |

On this series, normalisation makes no measurable difference: the three means sit within a point of each other, inside the seed spread. That is a finding about the data, not about RevIN. The household series has no trend, its values sit near 1 kWh (already a scale a network trains well on), and the test months stay within the range of the training years. RevIN earns its place on series that drift, or when one network trains on many series at different scales, as the [Global vs Local](../04_Machine_Learning/global_vs_local.llms.md) page showed for the scaler. It costs nothing, so the pages that follow keep it on.

The seed ensemble is the more useful result: averaging three networks that differ only in their random start adds one to two points of skill for free, because their errors are partly independent.

------------------------------------------------------------------------

## 3. Lookback length

With instance normalisation fixed, the lookback is the next choice. Longer windows show more weeks of the routine, and more noise to fit.

------------------------------------------------------------------------

``` python
look_results = {lb: (norm_results["instance"] if lb == 336 else run(lb, "instance")) for lb in (48, 168, 336, 672)}
look_table = pd.DataFrame(look_results).T[COLS]
look_table.index.name = "lookback (h)"
show(bar_chart([f"{lb} h" for lb in look_table.index], {"mean skill of 3 seeds": look_table["skill mean"],
                                                         "seed ensemble": look_table.skill}, height="300px"))
```

``` python
best_lb = int(look_table.mase.idxmin())
_ = record(f"MLP (PyTorch), {best_lb} h lookback, 3-seed ensemble", "deep learning",
           {k: look_results[best_lb][k] for k in ("mase", "skill", "mae", "rmse", "wape", "smape", "seconds")}, notebook=NB,
           notes="instance normalisation, 15 epochs, L1 loss")
look_table.style.format(FMT)
```

|              | skill mean | skill spread | skill  | mase  | seconds |
|--------------|------------|--------------|--------|-------|---------|
| lookback (h) |            |              |        |       |         |
| 48           | +21.1%     | 0.3%         | +21.5% | 0.631 | 2       |
| 168          | +21.9%     | 0.4%         | +22.8% | 0.620 | 2       |
| 336          | +17.0%     | 1.5%         | +18.4% | 0.652 | 2       |
| 672          | +12.9%     | 3.9%         | +15.3% | 0.678 | 2       |

**Shorter is better, up to a point.** One week of history gives the best and most stable network; two days is nearly as good; two and four weeks are worse, and the four-week networks disagree with each other the most (the largest seed spread). A longer window adds inputs faster than it adds information: the calendar features already say which weekday is coming, and the [autocorrelation](../01_Foundations/decomposition_stationarity.llms.md) showed little beyond lag 168. With 30,000 training windows, the extra 500 inputs mostly give the network more noise to fit.

This small network, a two-layer MLP trained in seconds, is the best model on the scoreboard so far. Two things set it apart from the gradient-boosting models: it trains on a window ending at **every** hour (30,000 examples, against 1,280 daily rows on the [regression page](../04_Machine_Learning/forecasting_as_regression.llms.md)), and it predicts all 24 hours jointly from the full recent window, so the shape of the last day carries over into the next. The lesson for the rest of this section: before reaching for a bigger architecture, get the windows, the lookback and the ensemble right.

------------------------------------------------------------------------

Back to top
