import os
import fire
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import qlib
from qlib.constant import REG_CN
from qlib.backtest import backtest
from qlib.workflow import R
from qlib.contrib.evaluate import risk_analysis, fit_capm
from qlib.contrib.report import analysis_position

def main(workflow="single"):
  data_handler = "Alpha158"
  market = "hs300" # sz50, hs300, zz500
  benchmark = "SH000300" # SH000016, SH000300, SH000905
  deal_price = "close"
  freq = "1d"

  if workflow == "single":
    n_tasks = 1 # number of different hyperparameter optimization tasks = num test splits
    n_folds = 0 # number of cross-validation folds, 0 means no CV
  elif workflow == "rolling_cv":
    n_tasks = 7 # number of different hyperparameter optimization tasks = num test splits
    n_folds = 4 # number of cross-validation folds, 0 means no CV
  k = 5  # top k models to select and plot

  identifier = f"tune_{workflow}_{data_handler}_{market}_{deal_price}_{freq}"

  workspace_path = "/iopsstor/scratch/cscs/ljiayong/workspace/qlib/tune/cn/lightgbm"
  storage = JournalStorage(JournalFileBackend(os.path.join(workspace_path, f"optuna/{identifier}_journal.log")))

  dfs = []
  for task_idx in range(n_tasks):
    study_name = f"task_{task_idx}"
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials_data = []
    for trial in study.trials:
      trial_dict = {
        'trial_number': trial.number,
        'state': trial.state.name,
        'value': trial.value,
      }
      trial_dict.update(trial.user_attrs)
      trials_data.append(trial_dict)
    df = pd.DataFrame(trials_data)
    df = df[df['state'] == 'COMPLETE'].dropna().reset_index(drop=True)

    dfs.append(df)

  qlib.init(**{
    "provider_uri": "/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock/bin",
    "region": REG_CN,
    "exp_manager": {
      "class": "MLflowExpManager",
      "module_path": "qlib.workflow.expm",
      "kwargs": {
        "uri": "file:///" + os.path.join(workspace_path, f"mlrun/{identifier}_runs"),
        "default_exp_name": "default_experiment",
      },
    }
  })

  for task_idx in range(n_tasks):
    df = dfs[task_idx]
    for idx, row in df.iterrows():
      recorder = R.get_recorder(experiment_name=f"task_{task_idx}", recorder_id=row["rid"])
      config = recorder.load_object("config.pkl")
      dataset = recorder.load_object("dataset.pkl")
      model = recorder.load_object("model.pkl")
      segments = config["task"]["dataset"]["kwargs"]["segments"]
      for split in ["valid", "test"]:
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
            "class": "EnhancedIndexingStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {
              "signal": recorder.load_object(f"pred_{split}.pkl"),
              "riskmodel_root": "/iopsstor/scratch/cscs/ljiayong/workspace/qlib/risk/cn/riskdata",
            }
          },
          "backtest": {
            "start_time": segments[split][0],
            "end_time": segments[split][1],
            "account": 1_000_000,
            "benchmark": benchmark,
            "exchange_kwargs": {
              "freq": "day",
              "trade_unit": 100,
              "limit_threshold": 0.095,
              "deal_price": deal_price,
              "open_cost": 0.002,
              "close_cost": 0.002,
              "min_cost": 5
            }
          }
        }
        steps_per_year = 246
        portfolio_metric_dict, indicator_dict = backtest(executor=port_analysis_config["executor"], strategy=port_analysis_config["strategy"], **port_analysis_config["backtest"])
        report = portfolio_metric_dict["1day" if freq == "1d" else freq][0]
        position = portfolio_metric_dict["1day" if freq == "1d" else freq][1]
        indicator = indicator_dict["1day" if freq == "1d" else freq]
        print(risk_analysis(report["bench"], N=steps_per_year, mode="product"))
        print(risk_analysis(report["return"]-report["cost"], N=steps_per_year, mode="product"))
        print(fit_capm(report["return"]-report["cost"], report["bench"], N=steps_per_year, r_f_annual=2e-2))
        analysis_position.report_graph(report, show_notebook=False, save_path=os.path.join(os.path.dirname(__file__), f"plots_{split}"))

if __name__ == "__main__":
  fire.Fire(main)

