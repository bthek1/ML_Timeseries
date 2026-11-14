# ONNX Export

Taking a trained PyTorch forecaster out of Python: export to ONNX, run it with ONNX Runtime, and prove the two give the same forecasts before trusting the faster one.

Author

Benedict Thekkel

## Why export

A model trained in a notebook usually runs somewhere else: a web service, a gateway beside the meter, a job written in another language. **ONNX** (Open Neural Network Exchange) is a file format for the computation graph and its weights; **ONNX Runtime** executes it on CPU or GPU from Python, C++, C#, Java or JavaScript, without PyTorch installed. The deployment then depends on one small runtime instead of the training stack.

Exporting is a translation, and translations can be wrong: an operator with a different default, a shape that was fixed when it should be dynamic. So the rule is to **verify before deploying**: run the same inputs through both and check the outputs agree to floating-point precision.

The model is the calendar-aware MLP from [The Deep Learning Data Pipeline](../05_Deep_Learning/dl_data_pipeline.llms.md) (one-week lookback), retrained here on the CPU in a few seconds. PyTorch’s exporter (`torch.onnx.export` with `dynamo=True`) needs the `onnx` and `onnxscript` packages.

This page carries `skip_exec` and stored outputs for a packaging reason, not a GPU one: the shared environment uses `onnxruntime-gpu`, and listing `onnxruntime` in this site’s CI dependencies would install a second, conflicting copy of the same module. The exported file goes to a temporary folder and is not committed.

------------------------------------------------------------------------

``` python
import contextlib
import io
import os
import tempfile
import time
import warnings

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from torch import nn

from ML_Timeseries.backtest import backtest, origins, score
from ML_Timeseries.data import bar_chart, load_household, show

warnings.filterwarnings("ignore")  # exporter deprecation warnings print local install paths
torch.manual_seed(0)
torch.set_num_threads(4)
```

``` python
hh = load_household()
y, FILLED = hh["kwh"], hh["filled"]
L, H = 168, 24
O = origins(y.index)
train = torch.tensor(y.iloc[:O[0]].to_numpy(np.float32))
idx = y.index[:O[0]]
x = train[:-H].unfold(0, L, 1)
t = train[L:].unfold(0, H, 1)
n = min(len(x), len(t))
first = idx[L:L + n]
cal = torch.tensor(np.column_stack([np.eye(24)[first.hour], np.eye(7)[first.dayofweek]]), dtype=torch.float32)
x, t = x[:n], t[:n]


class MLP(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(L + 31, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, H))

    def forward(self, window, calendar):
        mu = window.mean(1, keepdim=True)
        sd = window.std(1, keepdim=True) + 1e-3
        return self.net(torch.cat([(window - mu) / sd, calendar], 1)) * sd + mu


model = MLP()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
t0 = time.perf_counter()
for _ in range(10):
    for b in torch.randperm(n).split(512):
        loss = (model(x[b], cal[b]) - t[b]).abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
model.eval()
print(f"trained on CPU in {time.perf_counter() - t0:.0f} s; {sum(p.numel() for p in model.parameters()):,} parameters")
```

    trained on CPU in 3 s; 123,160 parameters

## 1. Export

The batch dimension is declared **dynamic**, so the same file serves one forecast or a thousand.

------------------------------------------------------------------------

``` python
workdir = tempfile.mkdtemp()
path = os.path.join(workdir, "mlp_forecaster.onnx")
example = (x[:2], cal[:2])
batch = torch.export.Dim("batch")
with contextlib.redirect_stdout(io.StringIO()):  # the exporter narrates its passes
    torch.onnx.export(model, example, path, dynamo=True, input_names=["window", "calendar"], output_names=["forecast"],
                      dynamic_shapes=({0: batch}, {0: batch}))
files = {f: os.path.getsize(os.path.join(workdir, f)) for f in sorted(os.listdir(workdir))}
print("exported:", ", ".join(f"{f} ({b / 1e6:.2f} MB)" for f, b in files.items()))
```

    exported: mlp_forecaster.onnx (0.02 MB), mlp_forecaster.onnx.data (0.49 MB)

``` python
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4  # explicit, so ONNX Runtime does not try to pin threads (refused inside this LXC container)
session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
print([(i.name, i.shape) for i in session.get_inputs()], "->", [(o.name, o.shape) for o in session.get_outputs()])
```

    [('window', ['batch', 168]), ('calendar', ['batch', 31])] -> [('forecast', ['batch', 24])]

## 2. Verify

All 30 test windows through both, as one batch: the largest absolute difference between the PyTorch and ONNX Runtime forecasts.

------------------------------------------------------------------------

``` python
def inputs(origin):
    hist = y.iloc[:origin]
    nxt = hist.index[-1] + pd.Timedelta(hours=1)
    w = hist.to_numpy(np.float32)[-L:]
    c = np.concatenate([np.eye(24)[nxt.hour], np.eye(7)[nxt.dayofweek]]).astype(np.float32)
    return w, c


W = np.stack([inputs(o)[0] for o in O])
C = np.stack([inputs(o)[1] for o in O])
with torch.no_grad():
    torch_out = model(torch.tensor(W), torch.tensor(C)).numpy()
onnx_out = session.run(None, {"window": W, "calendar": C})[0]
max_diff = float(np.abs(torch_out - onnx_out).max())
print(f"largest difference over {torch_out.size} forecast values: {max_diff:.2e} kWh")
```

    largest difference over 720 forecast values: 9.54e-07 kWh

The exported model is the same model. It also forecasts the same: both score identically on the backtest, because they produce the same numbers.

------------------------------------------------------------------------

``` python
def onnx_forecaster(history, horizon):
    nxt = history.index[-1] + pd.Timedelta(hours=1)
    w = history.to_numpy(np.float32)[-L:][None]
    c = np.concatenate([np.eye(24)[nxt.hour], np.eye(7)[nxt.dayofweek]]).astype(np.float32)[None]
    return session.run(None, {"window": w, "calendar": c})[0][0, :horizon]


onnx_score = score(backtest(y, onnx_forecaster), y, exclude=FILLED)
print(f"ONNX Runtime forecaster on the backtest: skill {onnx_score['skill']:+.1%}, MASE {onnx_score['mase']:.3f}")
```

    ONNX Runtime forecaster on the backtest: skill +20.1%, MASE 0.643

## 3. Speed

One forecast at a time, the way a service answers requests, timed over 2,000 calls after a warm-up:

------------------------------------------------------------------------

``` python
w1, c1 = W[:1], C[:1]
tw, tc = torch.tensor(w1), torch.tensor(C[:1])
timings = {}
for name, call in (("PyTorch (eager)", lambda: model(tw, tc)), ("ONNX Runtime", lambda: session.run(None, {"window": w1, "calendar": c1}))):
    with torch.no_grad():
        for _ in range(100):
            call()
        t0 = time.perf_counter()
        for _ in range(2000):
            call()
    timings[name] = 1e6 * (time.perf_counter() - t0) / 2000
speed = pd.Series(timings, name="microseconds per forecast")
show(bar_chart(speed.index, {"microseconds per forecast": speed}, height="260px"))
speed.round(1)
```

    PyTorch (eager)    37.2
    ONNX Runtime       15.2
    Name: microseconds per forecast, dtype: float64

The export checks out on every count:

- **Identical forecasts**: the largest disagreement over 720 forecast values is about a millionth of a kWh, the rounding of 32-bit floats taken in a different order. The backtest score is the same model’s score.
- **A small artefact**: half a megabyte of weights plus a tiny graph file (PyTorch stores the weights beside the graph). Nothing else is needed to run it.
- **Faster**: ONNX Runtime answers a single forecast more than twice as fast as eager PyTorch on the same CPU threads, because it runs a fixed, optimised graph with no Python in the loop.

For a forecaster serving many meters, the deployed process can then be a small service holding one ONNX file, with PyTorch and the training environment left behind. The verification step is the part not to skip: an export that silently fixed the batch size, or changed a normalisation, would still produce plausible numbers.

------------------------------------------------------------------------

Back to top
