import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt

import qlib
from qlib.constant import REG_CN
from qlib.data import D

market_list = ["sz50", "hs300", "zz500"]
market_benchmarks = ['SH000016', 'SH000300', 'SH000905'] # sz50, hs300, zz500
for market, benchmark in zip(market_list, market_benchmarks):
  provider_uri = f"/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock/bin"
  train_split = ("2021-01-01", "2022-12-31")
  valid_split = ("2023-01-01", "2023-12-31")
  test_split  = ("2024-01-01", "2025-09-29")

  qlib.init(provider_uri=provider_uri, region=REG_CN)
  '''
  Instrument Counts over time
  '''
  # universe = D.list_instruments(D.instruments(market=market), start_time=train_split[0], end_time=test_split[1], freq="day" if freq=="1d" else freq)
  # get_instrument_counts(universe, pd.to_datetime("2024-01-01"))
  df = D.features(D.instruments(market=market), ["$close"], start_time=train_split[0], end_time=test_split[1], freq="day")
  instruments_count = df.groupby("datetime").size()

  fig, ax = plt.subplots(figsize=(8, 6))
  ax.plot(instruments_count.index, instruments_count.values, linestyle='-')
  ax.set_title(f"Instrument Counts in {market}")
  ax.set_xlabel("Time")
  ax.set_ylabel("Count")
  ax.grid(True)
  save_path = os.path.join("/users/ljiayong/projects/qlib/scripts/train/cn/dataset_inspection", f"instruments_count_{market}.png")
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.savefig(save_path)

  '''
  Index Price over time
  '''
  df = D.features([benchmark], ["$close"], start_time=train_split[0], end_time=test_split[1], freq="day")
  fig, ax = plt.subplots(figsize=(8, 6))
  for benchmark in [benchmark]:
    benchmark_price = df.xs(benchmark, level="instrument")["$close"]
    benchmark_price = benchmark_price / benchmark_price.iloc[0]
    ax.plot(benchmark_price.index, benchmark_price, linestyle='-', label=benchmark)
  ax.set_title(f"Index Price in {market}")
  ax.set_xlabel("Time")
  ax.set_ylabel("Price")
  ax.grid(True)
  ax.legend()
  save_path = os.path.join("/users/ljiayong/projects/qlib/scripts/train/cn/dataset_inspection", f"index_price_{market}.png")
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.savefig(save_path)
