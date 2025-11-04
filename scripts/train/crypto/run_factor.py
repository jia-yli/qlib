import os
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import qlib
from qlib.constant import REG_CRYPTO
from qlib.data import D
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.workflow import R
from qlib.contrib.evaluate import risk_analysis, fit_capm
from qlib.contrib.report import analysis_position

freq = "720min" # day, 720min, 240min, 60min, 15min, 30min
_freq = "720min" # 1day, 720min, 240min, 60min, 15min, 30min
deal_price = "open"
provider_uri = f"/capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto/bin/{_freq}"
market = "my_universe"
benchmark = "BTCUSDT"
train_split = ("2021-01-01", "2022-12-31")
valid_split = ("2023-01-01", "2023-12-31")
test_split  = ("2024-01-01", "2025-09-29")

def get_score(provider_uri):
  qlib.init(provider_uri=provider_uri, region=REG_CRYPTO)
  score = D.features(D.instruments(market=market), ["$close/Ref($close, 1) - 1"], start_time=test_split[0], end_time=test_split[1], freq=freq)
  return score

def run_eval(score, provider_uri):
  qlib.init(provider_uri=provider_uri, region=REG_CRYPTO)

  signal = score.iloc[:, 0]
  port_analysis_config = {
    "executor": {
      "class": "SimulatorExecutor",
      "module_path": "qlib.backtest.executor",
      "kwargs": {
        "time_per_step": freq,
        "generate_portfolio_metrics": True,
      },
    },
    "strategy": {
      "class": "TopkDropoutStrategy",
      "module_path": "qlib.contrib.strategy.signal_strategy",
      "kwargs": {
        "signal": signal,
        "topk": 20,
        "n_drop": 2,
      },
    },
    "backtest": {
      "start_time": test_split[0],
      "end_time": test_split[1],
      "account": 1_000_000,
      "benchmark": benchmark,
      "exchange_kwargs": {
        "freq": freq,
        "trade_unit": None,
        "limit_threshold": None,
        "deal_price": deal_price,
        "open_cost": 0.001,
        "close_cost": 0.001,
        "min_cost": 0,
      },
    },
  }

  portfolio_metric_dict, indicator_dict = backtest(executor=port_analysis_config["executor"], strategy=port_analysis_config["strategy"], **port_analysis_config["backtest"])
  report = portfolio_metric_dict[_freq][0]
  position = portfolio_metric_dict[_freq][1]
  indicator = indicator_dict[_freq]
  print(risk_analysis(report["bench"], N=365, mode="product"))
  print(risk_analysis(report["return"]-report["cost"], N=365, mode="product"))
  print(fit_capm(report["return"]-report["cost"], report["bench"], N=365, r_f_annual=2e-2))
  analysis_position.report_graph(report, show_notebook=False, save_path=f"./scripts/train/crypto/{_freq}")
  # import pdb;pdb.set_trace()
  # dt = score.index.get_level_values(1)[-1]
  # score.xs(dt, level=1).sort_values("$close/Ref($close, 1) - 1")
  # sorted(score.xs(dt-pd.to_timedelta('2d'), level=1).sort_values("$close/Ref($close, 1) - 1")[-20:].index)

score = get_score(provider_uri)
run_eval(score, provider_uri)
