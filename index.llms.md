# TimeSeries ML

Time series machine learning, method by method: every model runs on the same household energy data and is scored on the same backtest.

Author

Benedict Thekkel

This site is organised around one concrete problem. The problem is household energy: given hourly meter readings, forecast consumption, detect anomalies, and work out which appliances are running from the total draw alone. [Phisaver Project](core.llms.md) states that brief.

The site is a curriculum that follows the field’s topic map (plan: [Knowledge \#28](https://github.com/bthek1/Knowledge/issues/28)). Each section covers one family of methods, from free baselines to pretrained foundation models. The original library walkthroughs (Darts, Nixtla) stay as API references under Libraries.

------------------------------------------------------------------------

## How the site works

Three choices make the pages comparable:

1.  **One dataset.** Every single-series notebook uses the same public household meter data (UCI, one house, 2006 to 2010, with three sub-meters). Every many-series notebook uses M4 Hourly (414 series). [Datasets](01_Foundations/datasets.llms.md) loads both.
2.  **One backtest.** Every forecasting notebook is scored by the same rolling-origin backtest against a seasonal naive forecast ([Backtesting](02_Evaluation/backtesting.llms.md)).
3.  **One scoreboard.** Every forecasting notebook adds a row to a shared results table, so the gain of each method is a number on the same scale ([Scoreboard](02_Evaluation/scoreboard.llms.md)).

The notebooks run on a small machine (4 CPU cores, 8 GB RAM, 12 GB GPU). Pages that run on CPU in a few minutes execute in CI on every push; GPU pages store their outputs.

------------------------------------------------------------------------

## What the scoreboard says

Skill is the share of seasonal naive’s error a model removes on the same 30 day-ahead windows ([Scoreboard](02_Evaluation/scoreboard.llms.md), 14 November 2026).

| Model                                 | Family           | Skill |
|---------------------------------------|------------------|-------|
| N-BEATS written from scratch, 3 seeds | deep learning    | +24%  |
| TiRex, zero-shot (no training)        | foundation model | +24%  |
| Small MLP, one-week lookback, 3 seeds | deep learning    | +23%  |
| Chronos-2, zero-shot                  | foundation model | +22%  |
| Chronos-Bolt tiny on one CPU thread   | foundation model | +20%  |
| Gradient boosting, one model per hour | machine learning | +18%  |
| SARIMA                                | classical        | +17%  |
| Four-week seasonal window             | free baseline    | +4%   |

Three lessons recur across the sections. How a model is fitted and fed matters more than its architecture (Holt-Winters goes from -13% to +12% by fitting for the right horizon). Pretrained models now beat everything built for one series, unless they have seen the test data, which [one of them](06_Foundation_Models/foundation_model_comparison.llms.md) appears to have. And most of an hour in a household is unpredictable: the best model removes about a quarter of the simplest model’s error.

------------------------------------------------------------------------

## Start Here

| Page | Covers |
|----|----|
| [Phisaver Project](core.llms.md) | The brief. What is being predicted (solar, total consumption, aircon) and from which inputs, the three resolutions the meter data arrives at (hourly energy, five-minute power, one-minute sensor readings), and a taxonomy of the model families worth trying: auto-forecast, exponential smoothing, benchmarks, intermittent-demand, multiple-seasonality, and Theta |

------------------------------------------------------------------------

## 01 Foundations

What a series is made of, and how to get it into shape before any model sees it.

| Page | Covers |
|----|----|
| [Datasets](01_Foundations/datasets.llms.md) | The household and M4 Hourly datasets, the download cache, resampling minutes to energy per period, gap filling, and the chart helpers every page uses |
| [Decomposition and Stationarity](01_Foundations/decomposition_stationarity.llms.md) | MSTL with daily and weekly seasons and their strength, the periodogram and its harmonics, ACF and PACF (lag 168 beats lag 24), and ADF against KPSS |
| [Data Preparation](01_Foundations/data_preparation.llms.md) | Peak versus mean power, why the timestamps are local clock time and must stay naive, a measured comparison of gap-filling methods, robust outlier flags, and scaling and log transforms |
| [Feature Engineering](01_Foundations/feature_engineering.llms.md) | Lags, calendar and holidays, Fourier terms, rolling statistics and sub-meters added one group at a time on the backtest (the calendar gives the largest gain), and catch22 features of whole days |
| [FFT Analysis](01_Foundations/fft_analysis.llms.md) | Building a synthetic multi-frequency signal, then using wavelets and FFT features to feed a random forest regressor |

------------------------------------------------------------------------

## 02 Evaluation

How every model on the site is scored, and the ways a score can lie. Read Backtesting before any model page.

| Page | Covers |
|----|----|
| [Backtesting and Leakage](02_Evaluation/backtesting.llms.md) | The rolling-origin protocol (30 day-ahead windows over the last 150 days), the [`backtest`](https://bthek1.github.io/ML_Timeseries/02_Evaluation/backtesting.html#backtest), [`score`](https://bthek1.github.io/ML_Timeseries/02_Evaluation/backtesting.html#score) and [`record`](https://bthek1.github.io/ML_Timeseries/02_Evaluation/backtesting.html#record) functions, and five leakage bugs measured side by side: shuffled CV, short lags, unshifted rolling means, whole-series statistics and tuning on the test set |
| [Forecast Metrics](02_Evaluation/forecast_metrics.llms.md) | MAE, RMSE, MAPE, sMAPE, WAPE, MASE and skill on three series from the same house; why MAPE is infinite and sMAPE flatters on sparse series; why the site reports MASE and skill |
| [Benchmarks](02_Evaluation/benchmarks.llms.md) | M4, Monash, ETT, GIFT-Eval and fev-bench; five questions for reading a leaderboard; the ETTh1 long-horizon protocol (a ridge layer beats both baselines) and the M4 Hourly protocol run by hand |
| [Scoreboard](02_Evaluation/scoreboard.llms.md) | Every model on the site in one table per dataset, best MASE first |

------------------------------------------------------------------------

## 03 Classical Statistical Forecasting

From free baselines to state space models, each scored on the household backtest. The best classical model, SARIMA, is level with gradient boosting.

| Page | Covers |
|----|----|
| [Baselines](03_Classical/baselines.llms.md) | Naive, mean, drift, seasonal naive and four-week seasonal windows; the window mean or median (+4%) is the free forecast to beat |
| [Exponential Smoothing (ETS)](03_Classical/ets.llms.md) | Holt-Winters equations; the default one-step fit loses to seasonal naive, smoothing chosen for 24 steps gives +12%; ETS on daily totals |
| [ARIMA](03_Classical/arima.llms.md) | Identification from ACF and PACF, order by AIC, Ljung-Box residual checks, SARIMA (+17%), and weekly Fourier terms that cost skill on three weeks of data |
| [Theta and Intermittent Demand](03_Classical/theta_intermittent.llms.md) | Theta as SES with drift; Croston, SBA, TSB and ADIDA on the sparse kitchen circuit, and why MAE rewards the zero forecast |
| [State Space and the Kalman Filter](03_Classical/state_space_kalman.llms.md) | A Kalman filter in twelve lines, the proof that SES is its steady state, gaps handled by the smoother, and a structural model on the backtest |
| [VAR and Granger Causality](03_Classical/var_granger.llms.md) | Every circuit Granger-causes every other through a common cause; a VAR on the circuits does not beat an AR on the total |
| [GARCH](03_Classical/garch.llms.md) | Volatility clustering in daily consumption, highest in winter; GARCH intervals closer to 90% coverage and 13% narrower than constant-variance ones |
| [Prophet](03_Classical/prophet.llms.md) | The original hourly-energy tutorial, plus Prophet on this site’s backtest, where it ties the four-week seasonal window |

------------------------------------------------------------------------

## 04 Machine Learning Forecasting

Forecasting rewritten as regression. Gradient boosting with one model per output hour is the best model on the household scoreboard so far (+18%).

| Page | Covers |
|----|----|
| [Forecasting as Regression](04_Machine_Learning/forecasting_as_regression.llms.md) | Building the supervised table without leaks; ridge from a two-week window to 24 hours (+12%), and gradient boosting on the same table (+18%); why ridge weights on correlated lags cannot be read |
| [Multi-step Strategies](04_Machine_Learning/multistep_strategies.llms.md) | Recursive, direct, MIMO and lags-of-24-hours compared hour by hour: recursive errors compound, and direct only wins in the first hours |
| [Global vs Local Models](04_Machine_Learning/global_vs_local.llms.md) | One LightGBM for 414 M4 series against one per series: global wins once the series are differenced and scaled, and the metric decides the transform |
| [XGBoost](04_Machine_Learning/xgboost.llms.md) | The algorithm itself, worked through from first principles |
| [XGBoost: Forecast Energy Consumption](04_Machine_Learning/xgboost_energy.llms.md) | The original PJM energy tutorial, plus XGBoost on this site’s backtest with the shared features (+17.5%) |

------------------------------------------------------------------------

## 05 Deep Learning Forecasting

Neural forecasters trained on the GPU (these pages store their outputs and do not run in CI). The best model on the site lives here: N-BEATS written from scratch, at +24% skill.

| Page | Covers |
|----|----|
| [The Deep Learning Data Pipeline](05_Deep_Learning/dl_data_pipeline.llms.md) | Windows at every hour, normalisation (no measurable effect on this series) and lookback length (one week is best) with a small PyTorch MLP; seed spread and seed ensembles |
| [RNN, LSTM and DeepAR](05_Deep_Learning/rnn_lstm_deepar.llms.md) | Recurrence and gating, DeepAR’s probabilistic autoregression; both near +20%, and why the field moved on anyway |
| [The MLP Family](05_Deep_Learning/mlp_family.llms.md) | N-BEATS, N-HiTS, TiDE and DLinear; DLinear reaches +15% with 8,112 parameters |
| [Transformers](05_Deep_Learning/transformers.llms.md) | PatchTST, iTransformer and TFT; TFT best at +20%, and the DLinear test (a linear layer is 7% behind on this task, and why) |
| [N-BEATS from Scratch](05_Deep_Learning/nbeats_from_scratch.llms.md) | Doubly residual stacking in sixty lines of PyTorch, each block’s contribution plotted, and a result that beats the library version |

------------------------------------------------------------------------

## 06 Foundation Models

Pretrained forecasters used zero-shot, with no training on this house (GPU pages with stored outputs). Three of them beat every classical and gradient-boosting model built for it.

| Page | Covers |
|----|----|
| [Chronos-2 Zero-Shot](06_Foundation_Models/chronos2_zero_shot.llms.md) | What a foundation model is; Chronos-2 univariate (+22%), with calendar covariates and with the sub-meters (no gain) |
| [Foundation Model Comparison](06_Foundation_Models/foundation_model_comparison.llms.md) | The GIFT-Eval standings on 14 Nov 2026 and licences; Chronos-2, Chronos-Bolt, TiRex (+24%) and TimesFM 2.5 on the backtest, and a spike test suggesting TimesFM 2.5 has seen this dataset |
| [Fine-Tuning Chronos-2](06_Foundation_Models/fine_tuning.llms.md) | Full and LoRA fine-tuning on the pre-test history: about one point of gain for 1.5 to 5 minutes of GPU time |
| [Small Models on CPU](06_Foundation_Models/small_models_on_cpu.llms.md) | Chronos-Bolt tiny (9 M parameters, 15 ms per forecast on one thread, +20%), Chronos-2 and TiRex without a GPU |

------------------------------------------------------------------------

## 07 Probabilistic Forecasting

Forecasting ranges instead of numbers, and checking that a “90%” interval really covers 90%.

| Page | Covers |
|----|----|
| [Probabilistic Forecasting](07_Probabilistic/probabilistic_forecasting.llms.md) | Quantiles, pinball loss, CRPS, coverage and width; empirical quantiles from four weeks cover only half of a “90%” band, quantile LightGBM covers 86% |
| [Conformal Prediction](07_Probabilistic/conformal_prediction.llms.md) | Split conformal, CQR and adaptive conformal inference on a drifting series; ACI reaches 88%, CQR is 13% narrower than split conformal, and why 30 days cannot settle a coverage claim |

------------------------------------------------------------------------

## 08 Other Tasks

Time series problems that are not forecasting, each tested on labelled events.

| Page | Covers |
|----|----|
| [Anomaly Detection](08_Other_Tasks/anomaly_detection.llms.md) | 24 injected meter faults (spikes, dropouts, stuck readings); generic detectors miss most of them, two structural rules plus a spike score find 23 of 24 at 0.1 false flags a week |
| [Change-Point Detection](08_Other_Tasks/change_point_detection.llms.md) | PELT and binary segmentation find the real regimes (heating, the August 2008 absence); a negative result on injected +40% shifts, which are ordinary in this house |
| [The Matrix Profile](08_Other_Tasks/matrix_profile.llms.md) | Motifs and discords with STUMPY; the exact motif is the loader’s own gap filling; shape versus size |
| [Classification with MiniRocket](08_Other_Tasks/classification_minirocket.llms.md) | Random convolution kernels against catch22: 96.5% against 84.3% on ItalyPowerDemand, and weekday or weekend in this house |

------------------------------------------------------------------------

## 09 Many Series

| Page | Covers |
|----|----|
| [Hierarchical Reconciliation](09_Many_Series/hierarchical_reconciliation.llms.md) | The house total and its four circuits made to add up: bottom-up, top-down, OLS, WLS and MinT in numpy; bottom-up is best for the total |

------------------------------------------------------------------------

## 10 Production

What happens after the backtest: explaining a forecast, keeping a model current, and shipping it.

| Page | Covers |
|----|----|
| [Explainability](10_Production/explainability.llms.md) | SHAP values from LightGBM itself and grouped permutation importance; why correlated time series features make importance rankings disagree |
| [Drift and Retraining](10_Production/drift_and_retraining.llms.md) | Five retraining schedules (refitting adds a point, forgetting old data costs six) and PSI drift that is mostly the calendar |
| [ONNX Export](10_Production/onnx_export.llms.md) | A PyTorch forecaster exported to ONNX, verified to a millionth of a kWh, and 2.4 times faster in ONNX Runtime |

------------------------------------------------------------------------

## Libraries

API walkthroughs, mostly following each project’s own tutorials and data. The topic pages above link here instead of repeating them.

| Page | Covers |
|----|----|
| [Darts](Libraries/darts.llms.md) | The long tour: training and predicting, inspecting seasonality, hyper-parameter search with Theta, backtesting against history, global models trained across many series, N-BEATS for the deep learning case, covariates and encoders for external data, and probabilistic forecasts with their distributions |
| [Darts Hyper-parameter Optimization](Libraries/darts_hyperparameter_optimization.llms.md) | Tuning a deep model for electricity load with confidence intervals: a linear baseline, then TCN, then Optuna over its parameters, then picking the winner |
| [Nixtla StatsForecast](Libraries/Nixtla/nixtla_statsforecasting.llms.md) | Training many statistical models over many series at once, evaluating them, and selecting the best per series |
| [Nixtla MLForecast](Libraries/Nixtla/nixtla_mlforecast.llms.md) | The machine learning arm, including distributed training via Spark and LightGBM |
| [Nixtla NeuralForecast](Libraries/Nixtla/nixtla_neuralforecast.llms.md) | Fitting LSTM and NHITS on AirPassengers, then plotting the predictions |
| [Nixtla Exogenous Variables](Libraries/Nixtla/nixtla_exogenous_variables.llms.md) | Training and forecasting with external inputs, both historical and known-future |
| [Nixtla Save and Load Models](Libraries/Nixtla/nixtla_neural_save_and_load.llms.md) | Persisting a trained neural forecaster and reading it back |
| [Nixtla Neural Advanced](Libraries/Nixtla/nixtla_neural_advanced.llms.md) | The multi-series version: train several neural models, evaluate, and select per series |
| [Nixtla All Methods](Libraries/Nixtla/nixtla_all_methods.llms.md) | Statistical, machine learning, and neural forecasts over the M5 dataset, cross-validated and evaluated per series and per window |

The FFT forecasting, electricity load and demand peak pages moved to [Electrical Docs](https://bthek1.github.io/Electrical_Docs/).

------------------------------------------------------------------------

## Not Covered Yet

Stated plainly, so the gaps are not mistaken for oversights:

- **Appliance disaggregation**: recovering per-appliance draw from the total signal. The household dataset’s sub-meters make it possible, but no page attempts it.
- **Phisaver’s own data**: it is private, so no page here uses it. It would also be the clean test for foundation models, which may have seen this public dataset.
- **Newer foundation models**: TimesFM-3, Toto-2.0 and TiRex-2 lead GIFT-Eval but are not tested here; Moirai and IBM’s TTM cannot be installed in this environment without downgrades.
- **Weather and other covariates**: no page uses temperature, which drives heating load; every covariate tried (calendar, sub-meters) added nothing over the target’s own history.
- **Probabilistic scoring of the deep and foundation models**: DeepAR, Chronos-2 and TiRex produce quantiles, but only quantile LightGBM and the conformal methods are scored on coverage and CRPS.
- **Change points on forecast residuals**, and **TFT attention explanations**, both proposed on their pages and not run.
- **A winter test period**: the backtest covers late June to late November 2010, so every score here is a summer-into-autumn score.

------------------------------------------------------------------------

Back to top
