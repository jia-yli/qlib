import os
import numpy as np
import pandas as pd

from tqdm import tqdm

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.model.riskmodel import StructuredCovEstimator


def prepare_riskdata(base_config, T, start_time):
  workspace_path = base_config["workspace_path"]
  market = base_config["market"]
  benchmark = base_config["benchmark"]
  deal_price = base_config["deal_price"]
  freq = base_config["freq"]

  qlib.init(**{
    "provider_uri": "/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock/bin",
    "region": REG_CN,
    "exp_manager": {
      "class": "MLflowExpManager",
      "module_path": "qlib.workflow.expm",
      "kwargs": {
        "uri": "file:///" + os.path.join(workspace_path, f"mlrun/prepare_riskdata"),
        "default_exp_name": "default_experiment",
      },
    }
  })

  # D.features: df with (instrument, datetime) index
  universe = D.features(
    D.instruments(market), ["$close"], start_time=start_time,
  ).swaplevel().sort_index() # df with (datetime, instrument) index

  price_all = D.features(
    D.instruments("all"), ["$close"], start_time=start_time,
  ).squeeze().unstack(level="instrument") # df with datetime index, instruments col

  # StructuredCovEstimator is a statistical risk model
  riskmodel = StructuredCovEstimator()

  for i in tqdm(range(T - 1, len(price_all))): # T-window, first result is at T-1
    date = price_all.index[i]
    ref_date = price_all.index[i - T + 1]

    codes = universe.loc[date].index
    price = price_all.loc[ref_date:date, codes] # [T, codes]

    # calculate return and remove extreme return
    ret = price.pct_change()
    ret.clip(ret.quantile(0.025), ret.quantile(0.975), axis=1, inplace=True)

    # run risk model
    F, cov_b, var_u = riskmodel.predict(ret, is_price=False, return_decomposed_components=True)

    # save risk data
    root = os.path.join(workspace_path, "riskdata", date.strftime("%Y%m%d"))
    os.makedirs(root, exist_ok=True)

    pd.DataFrame(F, index=codes).to_pickle(os.path.join(root, "factor_exp.pkl"))
    pd.DataFrame(cov_b).to_pickle(os.path.join(root, "factor_cov.pkl"))
    # for specific_risk we follow the convention to save volatility
    pd.Series(np.sqrt(var_u), index=codes).to_pickle(os.path.join(root, "specific_risk.pkl"))

if __name__ == "__main__":
  data_handler = "Alpha158"
  market = "hs300" # sz50, hs300, zz500
  benchmark = "SH000300" # SH000016, SH000300, SH000905
  deal_price = "close"
  freq = "1d"

  workspace_path = "/iopsstor/scratch/cscs/ljiayong/workspace/qlib/risk/cn"

  base_config = {
    "workspace_path": workspace_path,
    "market":     market,
    "benchmark":  benchmark,
    "deal_price": deal_price,
    "freq":       freq,
  }

  prepare_riskdata(base_config, T=246, start_time="2021-09-01")
