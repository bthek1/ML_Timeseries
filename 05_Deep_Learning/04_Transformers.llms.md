# Transformers for Time Series: PatchTST, iTransformer and TFT

Attention applied to forecasting three different ways (patches of time, whole variables, and gated covariates), each scored against the one-layer DLinear baseline.

Author

Benedict Thekkel

## What attention adds, and what goes wrong

A transformer lets every input token attend to every other, so a long-range dependency (this hour and the same hour last week) is one step away instead of 168. The first time-series transformers (Informer, Autoformer, FEDformer, 2021 and 2022) treated **each time step as a token**. A single hourly reading carries almost no meaning on its own, so attention between individual readings learned little, and the [DLinear](../05_Deep_Learning/mlp_family.llms.md) paper showed a linear layer matching them.

The models since then fix the tokenisation:

| Model | Token | Idea |
|----|----|----|
| **PatchTST** (Nie et al., 2023) | a **patch** of consecutive steps (here 16 hours) | like words instead of letters: a patch has a shape worth attending to. Each series is handled independently (“channel independence”) |
| **iTransformer** (Liu et al., 2024) | a **whole variable’s** history | attention runs *across variables*, not across time; useful when many related series move together |
| **TFT** (Lim et al., Google, 2021) | time steps, with gating and variable selection | built for many covariates: static, past-only and known-future inputs, with interpretable attention weights |

Same protocol as the previous pages (univariate, one-week lookback, trained once before the first origin), so iTransformer sees a single variable, which removes its main advantage; it is included to show that. GPU page, so `skip_exec` with stored outputs.

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
from neuralforecast.models import TFT, DLinear, PatchTST, iTransformer

from ML_Timeseries.backtest import backtest, load_results, origins, record, score
from ML_Timeseries.data import bar_chart, load_household, show, to_long

warnings.filterwarnings("ignore")
for name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(name).setLevel(logging.ERROR)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "05_Deep_Learning/04_Transformers"
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
    "PatchTST": PatchTST(patch_len=16, stride=8, **COMMON),
    "iTransformer": iTransformer(n_series=1, **COMMON),
    "TFT": TFT(hidden_size=64, **COMMON),
}
results = {}
for name, model in models.items():
    results[name] = fit_and_score(model)
    _ = record(f"{name} (neuralforecast)", "deep learning",
               {k: results[name][k] for k in ("mase", "skill", "mae", "rmse", "wape", "smape", "seconds")}, notebook=NB)
dlinear = load_results().query("model == 'DLinear (neuralforecast)'").iloc[0]
table = pd.DataFrame(results).T[["mase", "skill", "fit seconds", "peak VRAM GB", "parameters"]]
table.loc["DLinear (from the previous page)", ["mase", "skill"]] = [dlinear.mase, dlinear.skill]
table.style.format({"mase": "{:.3f}", "skill": "{:+.1%}", "fit seconds": "{:.0f}", "peak VRAM GB": "{:.2f}", "parameters": "{:,.0f}"}, na_rep="")
```

|   | mase | skill | fit seconds | peak VRAM GB | parameters |
|----|----|----|----|----|----|
| PatchTST | 0.648 | +19.4% | 68 | 0.73 | 466,843 |
| iTransformer | 0.665 | +17.0% | 6 | 0.15 | 6,404,632 |
| TFT | 0.638 | +20.4% | 152 | 4.58 | 220,046 |
| DLinear (from the previous page) | 0.683 | +14.6% |  |  |  |

``` python
order = table.sort_values("skill")
show(bar_chart(order.index, {"skill vs seasonal naive": order.skill}, horizontal=True, height="260px"))
```

TFT scores best of all the neuralforecast models on this site, and PatchTST is close behind. Both pay for it: TFT trains for over two minutes and peaks above 4 GB of GPU memory, against seconds and megabytes for DLinear. iTransformer, given a single series, has one token to attend to, so its attention does nothing and it behaves like an MLP; on many related series it would be a different model.

**The DLinear test.** On this task the linear model’s MASE is about 7% above the best transformer’s, so the gap is wider than the 2023 paper’s headline would suggest. Three reasons, all about the setting rather than the models: the horizon is short (24 steps, where DLinear’s paper used 96 to 720), so there is less long-range structure for a linear map to exploit; the series is noisy and spiky, which rewards nonlinear models that can react to the shape of the last evening; and no model here sees the calendar, so the nonlinear models must infer the weekday from the window itself. The paper’s broader point holds: a single linear layer is within a few points of every architecture on this page, at a thousandth of the cost.

Across the whole deep learning section, the most accurate model is still the small calendar-aware MLP ensemble from the [data pipeline](../05_Deep_Learning/dl_data_pipeline.llms.md) page. Inputs and training setup moved the score more than the choice of architecture did.

------------------------------------------------------------------------

Back to top
