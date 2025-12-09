import os
import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
import matplotlib.pyplot as plt

from tqdm import tqdm
from loguru import logger

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.model.riskmodel import StructuredCovEstimator

def load_quarter_data(symbols):
  processed_quarter_data_path = "/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock/processed/quarter_data"

  dfs = []
  for symbol in symbols:
    df = pd.read_csv(os.path.join(processed_quarter_data_path, f"{symbol}.csv"))
    dfs.append(df)
  
  df = pd.concat(dfs, ignore_index=True)

  df["pubDate"] = pd.to_datetime(df["pubDate"])
  df["statDate"] = pd.to_datetime(df["statDate"])

  return df

def prepare_indexweight(base_config, start_time, method="total_share", price_option="adj"):
  workspace_path = base_config["workspace_path"]
  market = base_config["market"]
  benchmark = base_config["benchmark"]

  qlib.init(**{
    "provider_uri": "/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock/bin",
    "region": REG_CN,
  })

  logger.info("Loading price data...")
  # D.features: df with (instrument, datetime) index
  universe = D.features(
    D.instruments(market), ["$close"], start_time=start_time,
  ).swaplevel().sort_index() # df with (datetime, instrument) index

  if price_option == "raw":
    price_all = D.features(D.instruments("all"), ["$close", "$factor"], start_time=start_time)
    price_all = (price_all["$close"]/price_all["$factor"]).squeeze().unstack(level="instrument") # df with datetime index, instruments col
  elif price_option == "adj":
    price_all = D.features(D.instruments("all"), ["$close"], start_time=start_time).squeeze().unstack(level="instrument") # df with datetime index, instruments col
  else:
    raise ValueError(f"Unknown price_option: {price_option}")

  price_index = D.features(
    [benchmark], ["$close"], start_time=start_time,
  ).squeeze().unstack(level="instrument").squeeze() # series with datetime index

  logger.info("Loading quarter data...")
  all_symbols = universe.index.get_level_values("instrument").unique().tolist()

  quarter_data = load_quarter_data(all_symbols)

  all_weights = {}
  for i in tqdm(range(len(price_all))):
    date = price_all.index[i]
    codes = universe.loc[date].index
    price_data = price_all.loc[date, codes]

    available_quarter_data = quarter_data[(quarter_data["pubDate"] + to_offset("1MS")) <= date]
    latest_quarter_data = available_quarter_data.sort_values('pubDate').groupby('symbol').last().reset_index()

    weights = {}
    total_market_cap = 0
    for code in codes:
      code_data = latest_quarter_data[latest_quarter_data['symbol'] == code].iloc[-1]
      if method == "total_share":
        total_share = code_data['totalShare']
        market_cap = price_data[code] * total_share
      elif method == "liqa_share":
        total_share = code_data['totalShare']
        liqa_share = code_data['liqaShare']
        if pd.isna(liqa_share):
          logger.warning(f"Missing liqaShare for {code} at date {date}. Use totalShare instead.")
          liqa_share = total_share
        market_cap = price_data[code] * liqa_share
      else:
        raise ValueError(f"Unknown method: {method}")
      weights[code] = market_cap
      total_market_cap += market_cap

    weights = {k: v / total_market_cap for k, v in weights.items()}
    all_weights[date] = pd.Series(weights).sort_index()

  all_weights = pd.DataFrame.from_dict(all_weights, orient='index').sort_index()
  # Save weights
  root = os.path.join(workspace_path, f"indexweight")
  os.makedirs(root, exist_ok=True)
  all_weights.to_feather(os.path.join(root, f"{benchmark}_{method}_{price_option}.feather"))

def check_indexweight(base_config, start_time, method="total_share", price_option="adj"):
  workspace_path = base_config["workspace_path"]
  market = base_config["market"]
  benchmark = base_config["benchmark"]

  qlib.init(**{
    "provider_uri": "/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock/bin",
    "region": REG_CN,
  })

  # D.features: df with (instrument, datetime) index
  universe = D.features(
    D.instruments(market), ["$close"], start_time=start_time,
  ).swaplevel().sort_index() # df with (datetime, instrument) index

  if price_option == "raw":
    price_all = D.features(D.instruments("all"), ["$close", "$factor"], start_time=start_time)
    price_all = (price_all["$close"]/price_all["$factor"]).squeeze().unstack(level="instrument") # df with datetime index, instruments col
  elif price_option == "adj":
    price_all = D.features(D.instruments("all"), ["$close"], start_time=start_time).squeeze().unstack(level="instrument") # df with datetime index, instruments col
  else:
    raise ValueError(f"Unknown price_option: {price_option}")

  price_index = D.features(
    [benchmark], ["$close"], start_time=start_time,
  ).squeeze().unstack(level="instrument").squeeze() # series with datetime index

  all_weights = pd.read_feather(os.path.join(workspace_path, "indexweight", f"{benchmark}_{method}_{price_option}.feather"))

  # compare with https://en.wikipedia.org/wiki/CSI_300_Index
  # all_weights.loc[pd.Timestamp("2024-03-08")].sort_values(ascending=False)
  
  # Compare synthesized index with actual benchmark
  synthesized_index = pd.Series(index=price_all.index, dtype=float)
  synthesized_index.iloc[0] = price_index.iloc[0]
  for i in range(1, len(price_all)):
    date = price_all.index[i]
    prev_date = price_all.index[i - 1]

    weights = all_weights.loc[prev_date].dropna()
    ret = (price_all.loc[date, weights.index] / price_all.loc[prev_date, weights.index] - 1).fillna(0)
    synth_ret = (weights * ret).sum()
    synthesized_index.iloc[i] = synthesized_index.iloc[i - 1] * (1 + synth_ret)

  # plot comparison
  fig, axes = plt.subplots(2, 1, figsize=(15, 10))
  # Normalize both to start at 100
  synth_normalized = synthesized_index / synthesized_index.iloc[0]
  bench_normalized = price_index / price_index.iloc[0]
  
  # Plot 1: Both indices
  axes[0].plot(synth_normalized.index, synth_normalized.values, label=f'Synthesized {benchmark}', linewidth=1.5)
  axes[0].plot(bench_normalized.index, bench_normalized.values, label=f'Actual {benchmark}', linewidth=1.5, alpha=0.7)
  axes[0].set_title(f'Index Comparison: {benchmark} ({method}, {price_option})')
  axes[0].set_xlabel('Date')
  axes[0].set_ylabel('Value')
  axes[0].legend()
  axes[0].grid(True, alpha=0.3)
  
  # Plot 2: Difference
  diff = synth_normalized - bench_normalized
  
  axes[1].plot(diff.index, diff.values, label='Difference', color='red', linewidth=1)
  axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
  axes[1].set_title(f'Difference: Synthesized - Actual ({benchmark}, {method}, {price_option})')
  axes[1].set_xlabel('Date')
  axes[1].set_ylabel('Difference')
  axes[1].legend()
  axes[1].grid(True, alpha=0.3)
  
  save_path = os.path.join(os.path.dirname(__file__), f"results/indexweight/{benchmark}_{method}_{price_option}_comparison.png")
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.savefig(save_path, bbox_inches="tight", dpi=500)
  plt.close()

if __name__ == "__main__":
  market = "hs300" # sz50, hs300, zz500
  benchmark = "SH000300" # SH000016, SH000300, SH000905

  workspace_path = "/iopsstor/scratch/cscs/ljiayong/workspace/qlib/risk/cn"

  base_config = {
    "workspace_path": workspace_path,
    "market":     market,
    "benchmark":  benchmark,
  }

  for method in ["total_share", "liqa_share"]:
    for price_option in ["adj", "raw"]:
      prepare_indexweight(base_config, start_time="2022-01-01", method=method, price_option=price_option)

  for method in ["total_share", "liqa_share"]:
    for price_option in ["adj", "raw"]:
      check_indexweight(base_config, start_time="2022-01-01", method=method, price_option=price_option)
  
  '''
  Results:
  1. liq + raw best match
  2. liq/total + adj next
  3. total + raw worst
  '''