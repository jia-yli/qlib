import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import matplotlib.pyplot as plt

import qlib
from qlib.constant import REG_CRYPTO
from qlib.data import D

# sum(1 for time_range_list in universe.values() if any(start_time <= time <= end_time for start_time, end_time in time_range_list))
def get_instrument_counts(universe, time):
  counts = 0
  for symbol, time_range_list in universe.items():
    for start_time, end_time in time_range_list:
      if start_time <= time <= end_time:
        counts += 1
        break
  return counts

# freq_list = ["15min", "30min", "60min", "240min", "720min", "1d"]
freq_list = ["60min", "240min", "1d"]

market_list = ["my_universe", "my_universe_top50", "my_universe_top200", "my_universe_mid150"]
for freq in freq_list:
  for market in market_list:
    provider_uri = f"/capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto/bin/{freq}"
    train_split = ("2021-01-01", "2022-12-31")
    valid_split = ("2023-01-01", "2023-12-31")
    test_split  = ("2024-01-01", "2025-09-29")

    qlib.init(provider_uri=provider_uri, region=REG_CRYPTO)
    '''
    Instrument Counts over time
    '''
    # universe = D.list_instruments(D.instruments(market=market), start_time=train_split[0], end_time=test_split[1], freq="day" if freq=="1d" else freq)
    # get_instrument_counts(universe, pd.to_datetime("2024-01-01"))
    df = D.features(D.instruments(market=market), ["$close"], start_time=train_split[0], end_time=test_split[1], freq="day" if freq=="1d" else freq)
    instruments_count = df.groupby("datetime").size()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(instruments_count.index, instruments_count.values, linestyle='-')
    ax.set_title(f"Instrument Counts in {market} at freq {freq}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.grid(True)
    save_path = os.path.join("/users/ljiayong/projects/qlib/scripts/train/crypto/dataset_inspection", f"instruments_count_{market}_{freq}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    '''
    Index Price over time
    '''
    benchmark1 = market.replace("my_universe", "MYINDEXVOL").replace("_", "").upper()
    benchmark2 = market.replace("my_universe", "MYINDEXEQ").replace("_", "").upper()
    benchmark3 = "BTCUSDT"
    benchmark4 = "ETHUSDT"
    df = D.features([benchmark1, benchmark2, benchmark3, benchmark4], ["$close"], start_time=train_split[0], end_time=test_split[1], freq="day" if freq=="1d" else freq)
    fig, ax = plt.subplots(figsize=(8, 6))
    for benchmark in [benchmark1, benchmark2, benchmark3, benchmark4]:
      benchmark_price = df.xs(benchmark, level="instrument")["$close"]
      benchmark_price = benchmark_price / benchmark_price.iloc[0]
      ax.plot(benchmark_price.index, benchmark_price, linestyle='-', label=benchmark)
    ax.set_title(f"Index Price in {market} at freq {freq}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.grid(True)
    ax.legend()
    save_path = os.path.join("/users/ljiayong/projects/qlib/scripts/train/crypto/dataset_inspection", f"index_price_{market}_{freq}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
