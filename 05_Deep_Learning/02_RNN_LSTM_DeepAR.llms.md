# RNN, LSTM and DeepAR

Recurrent networks read a series one step at a time and carry a memory forward. LSTM and DeepAR on the household backtest, and why recurrence has given way to models that look at the whole window at once.

Author

Benedict Thekkel

## Recurrence

A **recurrent neural network** (RNN) keeps a hidden state $h_t$ and updates it with each new value: $h_t = f(h_{t-1}, y_t)$. After reading the lookback window, the state summarises it, and a decoder turns the state into the forecast. Plain RNNs forget quickly, because the gradient through many steps shrinks or explodes. The **LSTM** (long short-term memory; Hochreiter and Schmidhuber, 1997) adds gates that decide what to keep, forget and output, which lets it carry information across hundreds of steps.

**DeepAR** (Salinas et al., Amazon, 2020) made recurrent forecasting practical at scale. Three ideas:

- it is **global**: one LSTM trained on many related series;
- it is **probabilistic**: the network outputs the parameters of a distribution for each step (here a Student’s t), not one number, and is trained by maximising the likelihood;
- it is **autoregressive**: at forecast time it samples a value, feeds it back as the next input and continues, so a forecast is a set of sampled paths.

Both models here come from Nixtla’s `neuralforecast` (the [NeuralForecast](../Libraries/Nixtla/nixtla_neuralforecast.llms.md) walkthrough covers the library). They see only the series, a one-week lookback, and train once on everything before the first test origin.

This page trains on the GPU, so it does not run in CI: it carries `skip_exec` and stores its outputs.

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
from neuralforecast.losses.pytorch import DistributionLoss, MAE
from neuralforecast.models import LSTM, DeepAR

from ML_Timeseries.backtest import backtest, origins, record, score
from ML_Timeseries.data import line_chart, load_household, show, to_long

warnings.filterwarnings("ignore")
for name in ("pytorch_lightning", "lightning", "lightning.pytorch", "lightning_fabric"):
    logging.getLogger(name).setLevel(logging.ERROR)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
NB = "05_Deep_Learning/02_RNN_LSTM_DeepAR"
INPUT, H = 168, 24
train = to_long(y.iloc[:origins(y.index)[0]])
TRAINER = dict(random_seed=0, enable_progress_bar=False, enable_model_summary=False, logger=False,
               accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)


def fit_and_score(model, col):
    "Train once before the first origin, then forecast every window from its own history."
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    nf = NeuralForecast(models=[model], freq="h")
    nf.fit(train)
    fit_s = time.perf_counter() - t0

    def f(history, horizon):
        out = nf.predict(df=to_long(history.iloc[-(INPUT + 48):]))
        return out[col if col in out else col.split("-")[0]].to_numpy()[:horizon]

    bt = backtest(y, f)
    sc = score(bt, y, exclude=FILLED) | {"seconds": time.perf_counter() - t0, "fit seconds": fit_s,
                                          "peak VRAM GB": torch.cuda.max_memory_allocated() / 1e9}
    del nf
    gc.collect()
    torch.cuda.empty_cache()
    return sc, bt
```

## 1. LSTM

A two-layer LSTM encoder reads the week; an MLP decoder maps the final state to all 24 hours at once. Trained on mean absolute error, like the scores.

------------------------------------------------------------------------

``` python
lstm_score, lstm_bt = fit_and_score(
    LSTM(h=H, input_size=INPUT, encoder_n_layers=2, encoder_hidden_size=128, decoder_hidden_size=128,
         loss=MAE(), max_steps=1000, scaler_type="robust", **TRAINER), "LSTM")
_ = record("LSTM (neuralforecast)", "deep learning", {k: lstm_score[k] for k in ("mase", "skill", "mae", "rmse", "wape", "smape", "seconds")}, notebook=NB)
print(f"LSTM: skill {lstm_score['skill']:+.1%}, fit {lstm_score['fit seconds']:.0f} s, peak VRAM {lstm_score['peak VRAM GB']:.2f} GB")
```

    LSTM: skill +19.9%, fit 12 s, peak VRAM 0.41 GB

## 2. DeepAR

DeepAR outputs a Student’s t distribution per hour. The point forecast scored here is the **median** of its sampled paths, the value that minimises absolute error.

------------------------------------------------------------------------

``` python
deepar_score, deepar_bt = fit_and_score(
    DeepAR(h=H, input_size=INPUT, lstm_n_layers=2, lstm_hidden_size=128, trajectory_samples=200,
           loss=DistributionLoss(distribution="StudentT", level=[80, 90]), valid_loss=MAE(),
           max_steps=1000, scaler_type="robust", **TRAINER), "DeepAR-median")
_ = record("DeepAR (neuralforecast, median)", "deep learning", {k: deepar_score[k] for k in ("mase", "skill", "mae", "rmse", "wape", "smape", "seconds")}, notebook=NB)
print(f"DeepAR: skill {deepar_score['skill']:+.1%}, fit {deepar_score['fit seconds']:.0f} s, peak VRAM {deepar_score['peak VRAM GB']:.2f} GB")
```

    DeepAR: skill +20.3%, fit 74 s, peak VRAM 3.23 GB

``` python
w = lstm_bt.window.max()
a_, b_ = lstm_bt[lstm_bt.window == w], deepar_bt[deepar_bt.window == w]
show(line_chart(a_.ds.dt.strftime("%a %H:00"), {"actual": a_.y, "LSTM": a_.yhat, "DeepAR (median)": b_.yhat},
                y_name="kWh", height="320px", zoom=False))
```

Both recurrent models do well here: each beats SARIMA and every gradient-boosting model on the scoreboard. DeepAR edges the LSTM slightly, at six times the training time and eight times the GPU memory, because sampling 200 paths per window is expensive. Its real product is the distribution; the median alone is only a by-product. The [Probabilistic Forecasting](../07_Probabilistic/probabilistic_forecasting.llms.md) page shows how to score a distribution (pinball loss, CRPS, coverage).

So why did the field move on from recurrence? Not because it forecasts badly on a task like this. An RNN reads its window one step at a time, so training cannot run in parallel along the time axis, and very long lookbacks are slow and unstable to train. On a 168-step window and a 24-step horizon that cost is small. On the long-horizon benchmarks with lookbacks of 512 steps and horizons of 720, models that look at the whole window at once (the MLPs and transformers on the next two pages) train faster and forecast better.

------------------------------------------------------------------------

Back to top
