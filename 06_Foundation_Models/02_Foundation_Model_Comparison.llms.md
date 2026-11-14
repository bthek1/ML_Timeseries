# Foundation Model Comparison

Chronos-2, Chronos-Bolt, TiRex and TimesFM 2.5 zero-shot on the same backtest: accuracy, speed, memory and licence, plus a test that catches a model that has already seen the answers.

Author

Benedict Thekkel

## The field on 14 November 2026

Public leaderboards move every few weeks, so a ranking needs a date. The table below is computed from the GIFT-Eval repository’s result files (last updated 13 November 2026) the way the benchmark aggregates them: each model’s MASE and CRPS (a probabilistic error) divided by seasonal naive’s on each of the 97 dataset configurations, then the geometric mean. Lower is better; 0.70 means 30% less error than seasonal naive on a typical configuration. None of these entries is flagged for test-data leakage.

| Model (organisation)         | MASE ratio | CRPS ratio |
|------------------------------|------------|------------|
| TimesFM-3 (Google)           | 0.667      | 0.456      |
| Toto-2.0-2.5B (Datadog)      | 0.696      | 0.476      |
| TiRex-2, zero-shot (NXAI)    | 0.697      | 0.478      |
| **Chronos-2** (Amazon)       | 0.698      | 0.485      |
| **TiRex** (NXAI)             | 0.716      | 0.488      |
| **TimesFM-2.5** (Google)     | 0.705      | 0.490      |
| Moirai-2 (Salesforce)        | 0.728      | 0.516      |
| Sundial base 128M (Tsinghua) | 0.750      | 0.559      |

The bold models run on this page. TimesFM-3, Toto-2.0 and TiRex-2 were not tested: they need packages or checkpoints that this site’s environment does not have yet. Moirai cannot be installed without downgrading the environment.

| Model (weights) | Licence | Package |
|----|----|----|
| Chronos-2 (0.48 GB), Chronos-Bolt small (0.19 GB) | Apache-2.0 | `chronos-forecasting` |
| TiRex (0.28 GB) | NXAI Community License: free for research and small businesses; large companies using it commercially must pay | `tirex-ts` |
| TimesFM 2.5 (0.93 GB) | Apache-2.0 | `transformers` |

Same protocol as every page: 2,048 hours of context, 24 hours ahead, 30 windows, no training. GPU page, so `skip_exec` with stored outputs.

------------------------------------------------------------------------

``` python
import gc
import logging
import time
import warnings

import numpy as np
import pandas as pd
import torch
import tirex
from chronos import BaseChronosPipeline, Chronos2Pipeline
from transformers import TimesFm2_5ModelForPrediction

from ML_Timeseries.backtest import backtest, record, score
from ML_Timeseries.data import bar_chart, load_household, show

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "06_Foundation_Models/02_Foundation_Model_Comparison"
CTX = 2048
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def context(history):
    return torch.tensor(history.to_numpy(dtype=np.float32)[-CTX:])


def run(name, load, forecast):
    "Load a model, backtest it, measure time and memory, then free it before the next one."
    torch.cuda.reset_peak_memory_stats()
    model = load()
    t0 = time.perf_counter()
    bt = backtest(y, lambda h, n: forecast(model, h, n))
    sc = score(bt, y, exclude=FILLED) | {"ms per forecast": 1000 * (time.perf_counter() - t0) / 30,
                                          "peak VRAM GB": torch.cuda.max_memory_allocated() / 1e9}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return sc, bt
```

``` python
MODELS = {
    "Chronos-2": (lambda: Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE),
                  lambda m, h, n: m.predict_quantiles([context(h)], prediction_length=n, quantile_levels=[0.5])[0][0][0, :, 0].cpu().numpy()),
    "Chronos-Bolt (small)": (lambda: BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map=DEVICE),
                             lambda m, h, n: m.predict_quantiles(context(h)[None], prediction_length=n, quantile_levels=[0.5])[0][0, :, 0].cpu().numpy()),
    "TiRex": (lambda: tirex.load_model("NX-AI/TiRex", device=DEVICE),
              lambda m, h, n: m.forecast(context=context(h)[None], prediction_length=n)[0][0, :, 4].cpu().numpy()),  # quantile 0.5
    "TimesFM 2.5": (lambda: TimesFm2_5ModelForPrediction.from_pretrained("google/timesfm-2.5-200m-transformers").to(DEVICE).eval(),
                    lambda m, h, n: m(past_values=[context(h).to(DEVICE)]).mean_predictions[0, :n].float().detach().cpu().numpy()),
}
results, bts = {}, {}
for name, (load, forecast) in MODELS.items():
    with torch.no_grad():
        results[name], bts[name] = run(name, load, forecast)
table = pd.DataFrame(results).T[["mase", "skill", "ms per forecast", "peak VRAM GB"]]
table.style.format({"mase": "{:.3f}", "skill": "{:+.1%}", "ms per forecast": "{:.0f}", "peak VRAM GB": "{:.2f}"})
```

|                      | mase  | skill  | ms per forecast | peak VRAM GB |
|----------------------|-------|--------|-----------------|--------------|
| Chronos-2            | 0.645 | +21.7% | 30              | 0.49         |
| Chronos-Bolt (small) | 0.642 | +20.3% | 9               | 0.20         |
| TiRex                | 0.623 | +23.6% | 162             | 0.30         |
| TimesFM 2.5          | 0.345 | +57.5% | 345             | 1.03         |

## Too good to be true

TimesFM 2.5’s score is far above every other model on this site, including models trained on this very house. A forecaster that good on a noisy household series would be remarkable. The first question to ask of such a result is the one from the [Benchmarks](../02_Evaluation/benchmarks.llms.md) page: **was the test data in the training data?**

This household dataset has been public on UCI since 2012 and is a common example in tutorials and benchmarks, so it may well sit in a pretraining corpus. Google does not publish the list of datasets TimesFM 2.5 was trained on, so this cannot be checked directly. It can be tested from the outside.

**The spike test.** Many of this house’s large hours are appliances switched on at times nothing in the past predicts. A model that *forecasts* should miss most of them. A model that *remembers* the series will hit them. Count, for every hour above 2.5 kWh in the test windows, how often each model predicted more than 2 kWh:

------------------------------------------------------------------------

``` python
spike = bts["Chronos-2"].y > 2.5
check = pd.DataFrame({name: {"spike hours predicted above 2 kWh": (bt.yhat[spike] > 2).mean(),
                             "correlation with actual": np.corrcoef(bt.y, bt.yhat)[0, 1]} for name, bt in bts.items()}).T
print(f"{int(spike.sum())} spike hours in the test windows")
check.style.format({"spike hours predicted above 2 kWh": "{:.0%}", "correlation with actual": "{:.2f}"})
```

    24 spike hours in the test windows

|   | spike hours predicted above 2 kWh | correlation with actual |
|----|----|----|
| Chronos-2 | 25% | 0.50 |
| Chronos-Bolt (small) | 33% | 0.50 |
| TiRex | 25% | 0.53 |
| TimesFM 2.5 | 79% | 0.83 |

TimesFM 2.5 predicts more than three quarters of the unpredictable spikes; the three other models catch a quarter to a third, which is about what a good honest forecaster manages. Its forecasts also track the actual hours far more closely. Adding noise to its context (tested while preparing this page) barely changes the result, which fits recognising a familiar series rather than reading fragile details.

This is consistent with TimesFM 2.5 having seen this household’s data, or a copy of it, during pretraining. It is not proof: Google does not list the 2.5 training corpus, and the test cannot tell which copy it saw. The result is therefore **left off the scoreboard**, and the GIFT-Eval numbers above remain the fair measure of TimesFM 2.5 (where it is a strong but ordinary member of the top group).

The general lesson outlasts this case. Any public dataset may be inside a foundation model’s training data. Before trusting a zero-shot result on a public series, run a recall test like this one, or evaluate on data that has never been published, such as a company’s own meters.

------------------------------------------------------------------------

``` python
clean = [n for n in results if n != "TimesFM 2.5"]
for name in clean[1:]:
    _ = record(f"{name}, zero-shot", "foundation model",
               {k: results[name][k] for k in ("mase", "skill", "mae", "rmse", "wape", "smape")} | {"seconds": results[name]["ms per forecast"] * 30 / 1000},
               notebook=NB, notes="no training; 2,048 h context")
order = table.loc[clean].sort_values("skill")
show(bar_chart(order.index, {"skill vs seasonal naive": order.skill}, horizontal=True, height="260px"))
```

Among the three clean models, TiRex is the most accurate, close to the best model trained on this house, and also the slowest at about a sixth of a second per forecast: it is built on xLSTM, a recurrent architecture that reads the context step by step. Chronos-2 is a little less accurate and five times faster. Chronos-Bolt small, at 9 milliseconds and 0.2 GB, gives up little accuracy for its size; the [small models](../06_Foundation_Models/small_models_on_cpu.llms.md) page runs it without a GPU.

The order here differs from GIFT-Eval, where Chronos-2 sits slightly ahead of TiRex. One series is one draw; the leaderboard averages 97 configurations. Both views agree on the main point: three different pretrained models, used with no training, all forecast this house better than any classical or gradient-boosting model built for it.

------------------------------------------------------------------------

Back to top
