# Fine-Tuning Chronos-2

Continuing a foundation model’s training on one house’s history: how much a few minutes of fine-tuning adds to a zero-shot forecast, and what it costs in GPU memory.

Author

Benedict Thekkel

## Zero-shot is the floor, not the ceiling

A zero-shot forecast uses only what the model learned from its pretraining corpus. **Fine-tuning** continues training on the target data, so the model can adapt to this house’s particular routine. Two forms:

- **full fine-tuning** updates every weight (here 119 million);
- **LoRA** (low-rank adaptation) freezes the weights and trains small low-rank corrections added to some layers; far fewer parameters and less memory, usually a little less gain.

The risk is the usual one: a model fitted to a few years of one house can forget the general patterns that made it good. A small learning rate and few steps limit that. The setting is fixed in advance (300 steps at a learning rate of 1e-5, batch 32) and not tuned on the test windows; the fine-tuning data is everything before the first test origin, exactly what every other model on the scoreboard trained on.

GPU page, so `skip_exec` with stored outputs. Checkpoints go to a temporary folder and are deleted afterwards.

------------------------------------------------------------------------

``` python
import gc
import logging
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import torch
from chronos import Chronos2Pipeline

from ML_Timeseries.backtest import backtest, origins, record, score
from ML_Timeseries.data import bar_chart, line_chart, load_household, show

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "06_Foundation_Models/03_Fine_Tuning"
CTX, H = 2048, 24
train = y.iloc[:origins(y.index)[0]].to_numpy(np.float32)
base = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda")


def forecaster(pipe):
    def f(history, horizon):
        x = torch.tensor(history.to_numpy(dtype=np.float32)[-CTX:])
        return pipe.predict_quantiles([x], prediction_length=horizon, quantile_levels=[0.5])[0][0][0, :, 0].cpu().numpy()
    return f


def fine_tune(steps, lr=1e-5, mode="full"):
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as d:
        tuned = base.fit([train], prediction_length=H, num_steps=steps, learning_rate=lr, batch_size=32,
                         context_length=CTX, finetune_mode=mode, output_dir=d)
    return tuned, time.perf_counter() - t0, torch.cuda.max_memory_allocated() / 1e9
```

## 1. Zero-shot against fine-tuned

------------------------------------------------------------------------

``` python
zero = score(backtest(y, forecaster(base)), y, exclude=FILLED)
tuned, tune_s, tune_gb = fine_tune(300)
bt_tuned = backtest(y, forecaster(tuned))
ft = score(bt_tuned, y, exclude=FILLED) | {"seconds": tune_s}
_ = record("Chronos-2, fine-tuned (300 steps)", "foundation model", ft, notebook=NB,
           notes="full fine-tuning on the pre-test history, lr 1e-5, batch 32; setting fixed in advance")
print(f"zero-shot: skill {zero['skill']:+.1%}; fine-tuned: skill {ft['skill']:+.1%} "
      f"({tune_s:.0f} s of fine-tuning, peak {tune_gb:.1f} GB GPU memory)")
del tuned
gc.collect()
torch.cuda.empty_cache()
```

\[300/300 01:31, Epoch 1/9223372036854775807\]

| Step | Training Loss |
|------|---------------|
| 100  | 4.349665      |
| 200  | 4.208158      |
| 300  | 4.162796      |

    zero-shot: skill +21.7%; fine-tuned: skill +22.6% (94 s of fine-tuning, peak 5.8 GB GPU memory)

## 2. Sensitivity: longer, and LoRA

Two other settings, shown to see how sensitive the gain is, not to pick a winner (picking on the test windows would be [leak 5](../02_Evaluation/backtesting.llms.md)):

------------------------------------------------------------------------

``` python
runs = {"zero-shot": {"skill": zero["skill"], "seconds": 0.0, "peak GB": 0.0},
        "full, 300 steps (recorded)": {"skill": ft["skill"], "seconds": tune_s, "peak GB": tune_gb}}
for label, steps, mode in (("full, 1000 steps", 1000, "full"), ("LoRA, 300 steps", 300, "lora")):
    model, secs, gb = fine_tune(steps, lr=1e-5 if mode == "full" else 1e-4, mode=mode)
    runs[label] = {"skill": score(backtest(y, forecaster(model)), y, exclude=FILLED)["skill"], "seconds": secs, "peak GB": gb}
    del model
    gc.collect()
    torch.cuda.empty_cache()
sens = pd.DataFrame(runs).T
show(bar_chart(sens.index, {"skill vs seasonal naive": sens.skill}, height="300px"))
```

\[1000/1000 05:06, Epoch 1/9223372036854775807\]

| Step | Training Loss |
|------|---------------|
| 100  | 4.348908      |
| 200  | 4.196870      |
| 300  | 4.109299      |
| 400  | 4.031978      |
| 500  | 3.851957      |
| 600  | 3.836250      |
| 700  | 3.768209      |
| 800  | 3.729778      |
| 900  | 3.719774      |
| 1000 | 3.667917      |

\[300/300 01:36, Epoch 1/9223372036854775807\]

| Step | Training Loss |
|------|---------------|
| 100  | 4.410985      |
| 200  | 4.326057      |
| 300  | 4.321923      |

``` python
sens.style.format({"skill": "{:+.1%}", "seconds": "{:.0f}", "peak GB": "{:.1f}"})
```

|                            | skill  | seconds | peak GB |
|----------------------------|--------|---------|---------|
| zero-shot                  | +21.7% | 0       | 0.0     |
| full, 300 steps (recorded) | +22.6% | 94      | 5.8     |
| full, 1000 steps           | +23.1% | 308     | 5.8     |
| LoRA, 300 steps            | +22.6% | 98      | 3.9     |

Fine-tuning helps, a little. A minute and a half of training adds about one point of skill; three times as many steps add about half a point more. LoRA gets the same gain as the short full fine-tune with two thirds of the GPU memory, because only its small adapter matrices carry gradients and optimiser state. The training-loss logs above keep falling for 1,000 steps, so the model is still learning this house, but the forecasts improve more slowly than the loss does: much of what it learns is noise specific to the training years.

Set against the effort, the pattern is the same as for every model family on this site. The zero-shot model already captures most of what the history can say; adaptation adds the last few percent. For one house that is rarely worth a GPU job. For a fleet of similar meters, one fine-tune shared by all of them is a much better trade, and that is the same argument the [Global vs Local](../04_Machine_Learning/global_vs_local.llms.md) page made for pooling series.

------------------------------------------------------------------------

Back to top
