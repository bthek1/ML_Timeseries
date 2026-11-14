# The MLP Family: N-BEATS, N-HiTS, TiDE and DLinear

Forecasters built from fully connected layers, and the 2023 result that a single linear layer matched many published transformers on long-horizon benchmarks.

Author

Benedict Thekkel

## Four models, one building block

All four map a lookback window straight to the forecast with fully connected layers, with no recurrence and no attention:

| Model | Year | Idea |
|----|----|----|
| **N-BEATS** (Oreshkin et al.) | 2020 | stacks of blocks; each block forecasts part of the series and removes its part of the input (a **residual** “backcast”), so later blocks model what earlier ones missed. It beat the M4 winner on M4 with no time-series-specific components |
| **N-HiTS** (Challu et al.) | 2023 | N-BEATS with each stack reading a pooled (coarser) input and forecasting at a coarser rate, then interpolating: some stacks handle slow patterns, others fast ones, at a fraction of the compute |
| **TiDE** (Das et al., Google) | 2023 | a dense encoder-decoder with residual connections that also takes future covariates, built for long horizons |
| **DLinear** (Zeng et al.) | 2023 | split the window into a moving-average trend and a remainder; one **linear layer** each. That is all |

DLinear’s paper, *Are Transformers Effective for Time Series Forecasting?*, showed that this linear model matched or beat most transformer forecasters of 2021 and 2022 on the standard long-horizon benchmarks. It reset the field: every architecture since is expected to beat a linear baseline first. The [Benchmarks](../02_Evaluation/benchmarks.llms.md) page reproduced the effect on ETTh1 with a ridge layer.

Same protocol as the [previous page](../05_Deep_Learning/rnn_lstm_deepar.llms.md): univariate, one-week lookback, trained once before the first origin. GPU page, so `skip_exec` with stored outputs.

------------------------------------------------------------------------

``` python
import gc
import logging
import time
import warnings

import numpy as np
import pandas as pd
import torch
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE
from neuralforecast.models import NBEATS, NHITS, DLinear, TiDE

from ML_Timeseries.backtest import backtest, origins, record, score
from ML_Timeseries.data import bar_chart, load_household, show, to_long

warnings.filterwarnings("ignore")
for name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(name).setLevel(logging.ERROR)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "05_Deep_Learning/03_MLP_Family"
INPUT, H = 168, 24
train = to_long(y.iloc[:origins(y.index)[0]])
COMMON = dict(h=H, input_size=INPUT, loss=MAE(), max_steps=1000, scaler_type="robust", random_seed=0,
              enable_progress_bar=False, enable_model_summary=False, logger=False,
              accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)


def fit_and_score(model):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    nf = NeuralForecast(models=[model], freq="h")
    nf.fit(train)
    fit_s = time.perf_counter() - t0
    col = type(model).__name__

    def f(history, horizon):
        return nf.predict(df=to_long(history.iloc[-(INPUT + 48):]))[col].to_numpy()[:horizon]

    sc = score(backtest(y, f), y, exclude=FILLED) | {"seconds": time.perf_counter() - t0, "fit seconds": fit_s,
                                                       "peak VRAM GB": torch.cuda.max_memory_allocated() / 1e9,
                                                       "parameters": sum(p.numel() for p in nf.models[0].parameters())}
    del nf
    gc.collect()
    torch.cuda.empty_cache()
    return sc
```

``` python
models = {
    "N-BEATS": NBEATS(**COMMON),
    "N-HiTS": NHITS(**COMMON),
    "TiDE": TiDE(**COMMON),
    "DLinear": DLinear(**COMMON),
}
results = {}
for name, model in models.items():
    results[name] = fit_and_score(model)
    _ = record(f"{name} (neuralforecast)", "deep learning",
               {k: results[name][k] for k in ("mase", "skill", "mae", "rmse", "wape", "smape", "seconds")}, notebook=NB)
table = pd.DataFrame(results).T[["mase", "skill", "fit seconds", "peak VRAM GB", "parameters"]]
table.style.format({"mase": "{:.3f}", "skill": "{:+.1%}", "fit seconds": "{:.0f}", "peak VRAM GB": "{:.2f}", "parameters": "{:,.0f}"})
```

|         | mase  | skill  | fit seconds | peak VRAM GB | parameters |
|---------|-------|--------|-------------|--------------|------------|
| N-BEATS | 0.661 | +17.5% | 7           | 0.10         | 2,781,666  |
| N-HiTS  | 0.676 | +15.3% | 8           | 0.10         | 2,817,570  |
| TiDE    | 0.649 | +18.9% | 6           | 0.10         | 1,497,340  |
| DLinear | 0.683 | +14.6% | 2           | 0.03         | 8,112      |

``` python
order = table.sort_values("skill")
show(bar_chart(order.index, {"skill vs seasonal naive": order.skill}, horizontal=True, height="280px"))
```

TiDE is the strongest of the four and N-BEATS close behind; both beat every classical model. N-HiTS’s coarse-to-fine stacks are aimed at long horizons, and at 24 steps their pooling mostly throws detail away. DLinear finishes last of the four, yet still beats every baseline and the default ETS fit, with 8,112 parameters (one weight per input-output pair, twice) against N-BEATS’s 2.8 million, and in two seconds of training.

That is the DLinear lesson as it applies here: a linear map gets most of the way, and every additional point of skill costs orders of magnitude more parameters. Whether the extra points are worth it depends on the use: for a fleet of meters retrained nightly, DLinear’s cost is hard to argue with.

None of these models sees the calendar on this page. The [data pipeline](../05_Deep_Learning/dl_data_pipeline.llms.md) MLP, which does, is still ahead of all four.

------------------------------------------------------------------------

Back to top
