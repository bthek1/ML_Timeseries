# Scoreboard

Every forecasting model on this site, scored on the same backtest: one table per dataset, best first.

Author

Benedict Thekkel

## How to read it

Each forecasting page calls [`record`](https://bthek1.github.io/ML_Timeseries/02_Evaluation/backtesting.html#record) from `ML_Timeseries.backtest`, which writes one JSON file per model into `nbs/results/`. This page reads them all. A page re-run updates its own row; nothing else changes.

- **MASE**: mean absolute scaled error, the headline. Lower is better; 1.0 is as good as seasonal naive was on the history.
- **Skill**: the share of seasonal naive’s error removed on the same test windows. Higher is better; 0 is the reference itself.
- **Seconds**: total forecasting time across the 30 windows, including any refits, on a 4-core CPU or a 12 GB GPU.

The [Backtesting](../02_Evaluation/backtesting.llms.md) page defines the household protocol and [Benchmarks](../02_Evaluation/benchmarks.llms.md) the M4 Hourly one. [Forecast Metrics](../02_Evaluation/forecast_metrics.llms.md) explains why these metrics.

------------------------------------------------------------------------

``` python
from ML_Timeseries.backtest import load_results
from ML_Timeseries.data import bar_chart, show
```

``` python
res = load_results()
cols = ["model", "family", "mase", "skill", "mae", "wape", "seconds", "notebook"]
print(f"{len(res)} rows from {res.notebook.nunique()} notebooks")
```

    40 rows from 22 notebooks

## Household energy, day-ahead

One house, hourly kWh, 30 windows of 24 hours over the last 150 days of the data.

------------------------------------------------------------------------

``` python
hh = res[res.dataset == "household"].reset_index(drop=True)
hh[[c for c in cols if c in hh]].style.format(
    {"mase": "{:.3f}", "skill": "{:+.1%}", "mae": "{:.3f}", "wape": "{:.1%}", "seconds": "{:.1f}"}, na_rep="").hide(axis="index")
```

| model | family | mase | skill | mae | wape | seconds | notebook |
|----|----|----|----|----|----|----|----|
| N-BEATS from scratch (PyTorch), 3-seed ensemble | deep learning | 0.607 | +24.3% | 0.395 | 43.7% | 8.4 | 05_Deep_Learning/05_NBEATS_From_Scratch |
| MLP (PyTorch), 168 h lookback, 3-seed ensemble | deep learning | 0.620 | +22.8% | 0.403 | 44.5% | 1.8 | 05_Deep_Learning/01_DL_Data_Pipeline |
| TiRex, zero-shot | foundation model | 0.623 | +23.6% | 0.399 | 44.1% | 4.9 | 06_Foundation_Models/02_Foundation_Model_Comparison |
| TFT (neuralforecast) | deep learning | 0.638 | +20.4% | 0.416 | 45.9% | 152.5 | 05_Deep_Learning/04_Transformers |
| Chronos-2, fine-tuned (300 steps) | foundation model | 0.639 | +22.6% | 0.404 | 44.6% | 93.8 | 06_Foundation_Models/03_Fine_Tuning |
| DeepAR (neuralforecast, median) | deep learning | 0.641 | +20.3% | 0.416 | 46.0% | 75.1 | 05_Deep_Learning/02_RNN_LSTM_DeepAR |
| Chronos-Bolt (small), zero-shot | foundation model | 0.642 | +20.3% | 0.416 | 46.0% | 0.3 | 06_Foundation_Models/02_Foundation_Model_Comparison |
| Chronos-Bolt tiny, zero-shot (CPU) | foundation model | 0.643 | +19.8% |  |  | 0.4 | 06_Foundation_Models/04_Small_Models_on_CPU |
| Chronos-2, zero-shot | foundation model | 0.644 | +21.7% | 0.409 | 45.2% | 0.9 | 06_Foundation_Models/01_Chronos2_Zero_Shot |
| LSTM (neuralforecast) | deep learning | 0.646 | +19.9% | 0.419 | 46.2% | 12.1 | 05_Deep_Learning/02_RNN_LSTM_DeepAR |
| PatchTST (neuralforecast) | deep learning | 0.648 | +19.4% | 0.421 | 46.5% | 68.7 | 05_Deep_Learning/04_Transformers |
| TiDE (neuralforecast) | deep learning | 0.649 | +18.9% | 0.424 | 46.8% | 6.2 | 05_Deep_Learning/03_MLP_Family |
| Gradient boosting, 336 h window to 24 h (one model per hour) | machine learning | 0.655 | +18.1% | 0.428 | 47.3% | 0.8 | 04_Machine_Learning/01_Forecasting_as_Regression |
| N-BEATS (neuralforecast) | deep learning | 0.660 | +17.5% | 0.431 | 47.6% | 7.2 | 05_Deep_Learning/03_MLP_Family |
| Direct (24 LightGBM models) | machine learning | 0.661 | +17.2% | 0.432 | 47.8% | 1.1 | 04_Machine_Learning/02_Multistep_Strategies |
| XGBoost, engineered features | machine learning | 0.663 | +17.5% | 0.431 | 47.6% | 0.4 | 04_Machine_Learning/05_XGBoost_Energy |
| Gradient boosting, engineered features | machine learning | 0.665 | +17.0% | 0.433 | 47.9% | 2.1 | 01_Foundations/03_Feature_Engineering |
| iTransformer (neuralforecast) | deep learning | 0.665 | +17.0% | 0.434 | 47.9% | 6.2 | 05_Deep_Learning/04_Transformers |
| Lags of 24 h or more (LightGBM) | machine learning | 0.669 | +15.9% | 0.439 | 48.5% | 0.2 | 04_Machine_Learning/02_Multistep_Strategies |
| SARIMA (2,0,1)(1,0,1)24 | classical | 0.670 | +17.2% | 0.433 | 47.8% | 50.1 | 03_Classical/03_ARIMA |
| N-HiTS (neuralforecast) | deep learning | 0.676 | +15.3% | 0.442 | 48.9% | 8.4 | 05_Deep_Learning/03_MLP_Family |
| DLinear (neuralforecast) | deep learning | 0.683 | +14.6% | 0.446 | 49.2% | 2.5 | 05_Deep_Learning/03_MLP_Family |
| SARIMA (2,0,1)(1,0,1)24 + weekly Fourier | classical | 0.695 | +14.0% | 0.450 | 49.6% | 69.2 | 03_Classical/03_ARIMA |
| Holt-Winters (24 h), smoothing tuned for 24 steps | classical | 0.700 | +12.3% | 0.458 | 50.6% | 0.3 | 03_Classical/02_ETS |
| Ridge, 336 h window to 24 h (MIMO) | machine learning | 0.701 | +12.4% | 0.458 | 50.5% | 0.0 | 04_Machine_Learning/01_Forecasting_as_Regression |
| Recursive (LightGBM) | machine learning | 0.726 | +9.2% | 0.475 | 52.4% | 2.8 | 04_Machine_Learning/02_Multistep_Strategies |
| Structural model (level + daily + weekly + AR1) | classical | 0.733 | +9.4% | 0.473 | 52.3% | 37.7 | 03_Classical/05_State_Space_Kalman |
| Seasonal window median (4 weeks) | baseline | 0.764 | +4.0% | 0.501 | 55.4% | 0.0 | 03_Classical/01_Baselines |
| Prophet (daily + weekly + holidays) | classical | 0.766 | +3.9% | 0.502 | 55.5% | 8.7 | 03_Classical/08_Prophet |
| Seasonal window mean (4 weeks) | baseline | 0.771 | +4.3% | 0.500 | 55.2% | 0.0 | 03_Classical/01_Baselines |
| Seasonal naive (24 h) | baseline | 0.816 | +0.0% | 0.522 | 57.7% | 0.0 | 02_Evaluation/00_Backtesting |
| Theta (24 h season) | classical | 0.831 | -4.4% | 0.546 | 60.2% | 1.5 | 03_Classical/04_Theta_Intermittent |
| Seasonal naive (168 h) | baseline | 0.841 | -4.5% | 0.546 | 60.3% | 0.0 | 03_Classical/01_Baselines |
| Mean (4 weeks) | baseline | 0.843 | -5.9% | 0.553 | 61.1% | 0.0 | 03_Classical/01_Baselines |
| Holt-Winters (24 h), default fit | classical | 0.900 | -13.4% | 0.593 | 65.4% | 2.3 | 03_Classical/02_ETS |
| Naive | baseline | 0.931 | -15.7% | 0.605 | 66.8% | 0.0 | 03_Classical/01_Baselines |
| Drift (4 weeks) | baseline | 0.941 | -16.9% | 0.611 | 67.5% | 0.0 | 03_Classical/01_Baselines |

``` python
order = hh.sort_values("skill")
show(bar_chart(order.model, {"skill vs seasonal naive": order.skill}, horizontal=True,
               height=f"{max(240, 40 * len(order) + 80)}px"))
```

## M4 Hourly

414 series, each scored on its fixed 48-hour test horizon, averaged across series.

------------------------------------------------------------------------

``` python
m4 = res[res.dataset == "m4_hourly"].reset_index(drop=True)
m4[[c for c in ["model", "family", "mase", "smape", "skill", "notebook"] if c in m4]].style.format(
    {"mase": "{:.3f}", "smape": "{:.2f}", "skill": "{:+.1%}"}, na_rep="").hide(axis="index")
```

| model | family | mase | smape | skill | notebook |
|----|----|----|----|----|----|
| LightGBM, global (diff 24 + scaler) | machine learning | 0.887 | 11.68 |  | 04_Machine_Learning/03_Global_vs_Local |
| LightGBM, local per series (diff 24 + scaler) | machine learning | 0.933 | 12.09 |  | 04_Machine_Learning/03_Global_vs_Local |
| Seasonal naive (24 h) | baseline | 1.193 | 13.91 | +0.0% | 02_Evaluation/02_Benchmarks |

Back to top
