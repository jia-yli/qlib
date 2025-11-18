import os
import itertools
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.workflow import R
from qlib.contrib.evaluate import risk_analysis, fit_capm
from qlib.contrib.report import analysis_position

def compare_df(di, dj, eps=1e-12):
  name_i, name_j = f"left", f"right"
  if not di.index.equals(dj.index):
    only_i_idx = di.index.difference(dj.index)
    only_j_idx = dj.index.difference(di.index)
    if len(only_i_idx): print(f"rows only in {name_i}: {list(only_i_idx)}")
    if len(only_j_idx): print(f"rows only in {name_j}: {list(only_j_idx)}")
  else:
    print("Index match")

  if not di.columns.equals(dj.columns):
    only_i_cols = di.columns.difference(dj.columns).tolist()
    only_j_cols = dj.columns.difference(di.columns).tolist()
    if only_i_cols: print(f"cols only in {name_i}: {only_i_cols}")
    if only_j_cols: print(f"cols only in {name_j}: {only_j_cols}")
  else:
    print("Columns match")

  inter_idx = di.index.intersection(dj.index)
  inter_cols = di.columns.intersection(dj.columns)
  ai = di.loc[inter_idx, inter_cols]
  bj = dj.loc[inter_idx, inter_cols]
  same = np.isclose(ai.values, bj.values, equal_nan=True, atol=eps, rtol=0)
  row_has_diff = ~same.all(axis=1)           # bool array per row
  if not row_has_diff.any():
    print("Values match (no differing rows)")

  diff_row_indices = ai.index[row_has_diff].tolist()
  import pdb;pdb.set_trace()
  # sorted(set(e[0] for e in  diff_row_indices))
  print(f"Differing rows: {len(diff_row_indices)}")
  for idx in diff_row_indices:
    print(f"{idx}")

if __name__ == "__main__":
  # config
  provider_uri = "/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock/bin"
  market = "hs300" # sz50, hs300, zz500
  benchmark = "SH000300" # SH000016, SH000300, SH000905
  deal_price = "close"
  freq = "1d"
  assert int(pd.to_timedelta("1day") / pd.to_timedelta(freq)) == 1, "Only support daily frequency now."
  steps_per_year = 246 

  train_split = ("2020-01-01", "2022-12-31")
  valid_split = ("2023-01-01", "2023-12-31")
  test_split  = ("2024-01-01", "2025-09-29")

  factor = "$close/Ref($close, 1) - 1"

  qlib.init(provider_uri=provider_uri, region=REG_CN)
  score = D.features(D.instruments(market=market), [factor], start_time=test_split[0], end_time=test_split[1])

  signal = score.iloc[:, 0]
  port_analysis_config = {
    "executor": {
      "class": "SimulatorExecutor",
      "module_path": "qlib.backtest.executor",
      "kwargs": {
        "time_per_step": "day" if freq == "1d" else freq,
        "generate_portfolio_metrics": True,
      },
    },
    "strategy": {
      "class": "TopkDropoutStrategy",
      "module_path": "qlib.contrib.strategy",
      "kwargs": {
        "signal": signal,
        "topk": 50,
        "n_drop": 5,
      },
    },
    "backtest": {
      "start_time": test_split[0],
      "end_time": test_split[1],
      "account": 1_000_000,
      "benchmark": benchmark,
      "exchange_kwargs": {
        "freq": "day" if freq == "1d" else freq,
        "trade_unit": 100,
        "limit_threshold": 0.095,
        "deal_price": deal_price,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5,
      },
    },
  }

  portfolio_metric_dict, indicator_dict = backtest(executor=port_analysis_config["executor"], strategy=port_analysis_config["strategy"], **port_analysis_config["backtest"])
  report = portfolio_metric_dict["1day" if freq == "1d" else freq][0]
  position = portfolio_metric_dict["1day" if freq == "1d" else freq][1]
  indicator = indicator_dict["1day" if freq == "1d" else freq]
  print(risk_analysis(report["bench"], N=steps_per_year, mode="product"))
  print(risk_analysis(report["return"]-report["cost"], N=steps_per_year, mode="product"))
  print(fit_capm(report["return"]-report["cost"], report["bench"], N=steps_per_year, r_f_annual=2e-2))
  analysis_position.report_graph(report, show_notebook=False, save_path=f"./scripts/train/cn/{market}/{freq}/factor")


