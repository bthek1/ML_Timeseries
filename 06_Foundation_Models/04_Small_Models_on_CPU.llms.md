# Small Models on CPU

Foundation models without a GPU: Chronos-Bolt tiny and small, Chronos-2 and TiRex on the CPU, with the latency and accuracy a meter gateway or a cheap server would see.

Author

Benedict Thekkel

## Why CPU matters

A forecast that runs once an hour per meter does not need a GPU, and most places a forecast is needed (an edge gateway, a small cloud function, a home server) do not have one. The question is how much accuracy survives on a CPU, and how long each forecast takes.

**Chronos-Bolt** (Amazon, 2024) is built for this: it forecasts all quantiles in one forward pass instead of sampling, and comes in sizes from 9 million parameters (tiny) to 205 million (base). Two other small CPU candidates named in this site’s plan could not be tested: IBM’s TTM (Tiny Time Mixers) would downgrade the shared environment’s PyTorch, and TabPFN-TS does not resolve against it at all.

Everything here runs on the CPU only, with the GPU hidden. The page still carries `skip_exec`, because it imports the vendor packages that only the root environment has.

------------------------------------------------------------------------

``` python
import gc
import logging
import os
import time
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # force CPU for every model on this page

import numpy as np
import pandas as pd
import tirex
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline

from ML_Timeseries.backtest import backtest, record, score
from ML_Timeseries.data import bar_chart, load_household, show

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
print(f"CUDA visible: {torch.cuda.is_available()}; CPU threads available: {os.cpu_count()}")
```

    CUDA visible: False; CPU threads available: 4

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "06_Foundation_Models/04_Small_Models_on_CPU"
CTX = 2048


def context(history):
    return torch.tensor(history.to_numpy(dtype=np.float32)[-CTX:])


MODELS = {
    "Chronos-Bolt tiny": (lambda: BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-tiny", device_map="cpu"),
                          lambda m, h, n: m.predict_quantiles(context(h)[None], prediction_length=n, quantile_levels=[0.5])[0][0, :, 0].numpy()),
    "Chronos-Bolt small": (lambda: BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cpu"),
                           lambda m, h, n: m.predict_quantiles(context(h)[None], prediction_length=n, quantile_levels=[0.5])[0][0, :, 0].numpy()),
    "Chronos-2": (lambda: Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cpu"),
                  lambda m, h, n: m.predict_quantiles([context(h)], prediction_length=n, quantile_levels=[0.5])[0][0][0, :, 0].numpy()),
    "TiRex": (lambda: tirex.load_model("NX-AI/TiRex", device="cpu"),
              lambda m, h, n: m.forecast(context=context(h)[None], prediction_length=n)[0][0, :, 4].numpy()),
}
```

## Accuracy and latency

Each model forecasts the 30 backtest windows twice: with four threads (the whole machine) and with one (a fair share on a busy server). Latency is the mean time per 24-hour forecast.

------------------------------------------------------------------------

``` python
rows = {}
for name, (load, forecast) in MODELS.items():
    model = load()
    params = sum(p.numel() for p in (model.model if hasattr(model, "model") else model).parameters())
    row = {"parameters (M)": params / 1e6}
    for threads in (4, 1):
        torch.set_num_threads(threads)
        t0 = time.perf_counter()
        with torch.no_grad():
            bt = backtest(y, lambda h, n: forecast(model, h, n))
        row[f"ms per forecast, {threads} thread{'s' if threads > 1 else ''}"] = 1000 * (time.perf_counter() - t0) / 30
    row |= {k: v for k, v in score(bt, y, exclude=FILLED).items() if k in ("mase", "skill")}
    rows[name] = row
    del model
    gc.collect()
torch.set_num_threads(4)
cpu = pd.DataFrame(rows).T
_ = record("Chronos-Bolt tiny, zero-shot (CPU)", "foundation model",
           {"mase": rows["Chronos-Bolt tiny"]["mase"], "skill": rows["Chronos-Bolt tiny"]["skill"],
            "seconds": rows["Chronos-Bolt tiny"]["ms per forecast, 1 thread"] * 30 / 1000}, notebook=NB, notes="9 M parameters, one CPU thread")
cpu.style.format({"parameters (M)": "{:.0f}", "ms per forecast, 4 threads": "{:.0f}", "ms per forecast, 1 thread": "{:.0f}",
                  "mase": "{:.3f}", "skill": "{:+.1%}"})
```

|   | parameters (M) | ms per forecast, 4 threads | ms per forecast, 1 thread | mase | skill |
|----|----|----|----|----|----|
| Chronos-Bolt tiny | 9 | 10 | 15 | 0.644 | +19.8% |
| Chronos-Bolt small | 48 | 33 | 64 | 0.642 | +20.3% |
| Chronos-2 | 119 | 117 | 272 | 0.645 | +21.7% |
| TiRex | 35 | 118 | 115 | 0.624 | +23.6% |

``` python
show(bar_chart(cpu.index, {"ms per forecast, 1 thread": cpu["ms per forecast, 1 thread"]}, y_name="ms", height="300px"))
```

**Chronos-Bolt tiny is the headline.** Nine million parameters, 15 milliseconds per forecast on a single CPU thread, and a skill within two points of Chronos-2, with no training. That is better than every classical and gradient-boosting model on the scoreboard, from a 30 MB file. At 15 milliseconds, one core produces about 240,000 day-ahead forecasts an hour, enough for a large fleet of meters.

The other rows show how the architectures differ off the GPU:

- **Chronos-2** is the slowest on one thread (about a quarter of a second) and gains the most from four, because its transformer layers are large matrix products that parallelise well.
- **TiRex** is the most accurate and does not speed up with more threads: its xLSTM reads the context one step at a time, and a sequential loop cannot be split across cores. On a GPU it was the slowest of the three for the same reason.
- **Chronos-Bolt small** buys half a point over tiny for four times the latency.

Pick by the deployment, not the leaderboard: tiny where cost and latency dominate, TiRex where the last points of accuracy matter, and check its licence if the use is commercial.

------------------------------------------------------------------------

Back to top
