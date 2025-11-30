import os
import copy
import fire
import functools

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import numpy as np
import pandas as pd

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict, fill_placeholder, class_casting
from qlib.workflow import R
from qlib.backtest import backtest
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.eva.alpha import calc_ic
from qlib.contrib.evaluate import risk_analysis, fit_capm

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../utils"))
from tune_utils import generate_config
from suggest_config import suggest_lightgbm_config

def run_eval(model, dataset, segment, recorder, config):
  base_config = config["base"]
  freq = base_config["freq"]
  steps_per_year = config["task"]["steps_per_year"]
  segment_start, segment_end = base_config[f"{segment}_split"]

  metrics = dict()
  # prediction
  pred = model.predict(dataset, segment=segment)
  recorder.save_objects(**{f"pred_{segment}.pkl": pred})

  assert isinstance(dataset, DatasetH)
  with class_casting(dataset, DatasetH):
    params = dict(segments=segment, col_set="label", data_key=DataHandlerLP.DK_R)
    try:
      # Assume the backend handler is DataHandlerLP
      raw_label = dataset.prepare(**params)
    except TypeError:
      # The argument number is not right
      params.pop("data_key")
      # The backend handler should be DataHandler
      raw_label = dataset.prepare(**params)
  recorder.save_objects(**{f"label_{segment}.pkl": raw_label})

  metrics.update({
    f"l2_{segment}": ((pred - raw_label.iloc[:, 0]) ** 2).mean(),
  })

  ic, ric = calc_ic(pred, raw_label.iloc[:, 0])
  recorder.save_objects(**{f"ic_{segment}.pkl": ic, f"ric_{segment}.pkl": ric})

  metrics.update({
    f"ic_{segment}": ic.mean(),
    f"icir_{segment}": ic.mean() / ic.std(),
    f"rank_ic_{segment}": ric.mean(),
    f"rank_icir_{segment}": ric.mean() / ric.std(),
  })

  # backtest
  port_analysis_config = fill_placeholder(
    copy.deepcopy(config["task"]["port_analysis_config"]), 
    {
      "<PRED>": pred,
      "<BACKTEST_START>": segment_start,
      "<BACKTEST_END>": segment_end,
    }
  )
  portfolio_metric_dict, indicator_dict = backtest(
    executor=port_analysis_config["executor"], strategy=port_analysis_config["strategy"], **port_analysis_config["backtest"]
  )
  recorder.save_objects(**{f"portfolio_metric_{segment}.pkl": portfolio_metric_dict, f"indicator_{segment}.pkl": indicator_dict})

  report = portfolio_metric_dict["1day" if freq == "1d" else freq][0]
  analysis = dict()
  analysis['bench'] = risk_analysis(report["bench"], N=steps_per_year, mode="product")
  analysis['return_with_cost'] = risk_analysis(report["return"]-report["cost"], N=steps_per_year, mode="product")
  analysis['fit_capm'] = fit_capm(report["return"]-report["cost"], report["bench"], N=steps_per_year, r_f_annual=2e-2)

  metrics.update({
    f'bench_annualized_return_{segment}': analysis['bench'].loc['annualized_return', 'risk'],
    f'bench_information_ratio_{segment}': analysis['bench'].loc['information_ratio', 'risk'],
    f'bench_max_drawdown_{segment}': analysis['bench'].loc['max_drawdown', 'risk'],
    f'annualized_return_{segment}': analysis['return_with_cost'].loc['annualized_return', 'risk'],
    f'excess_annualized_return_{segment}': analysis['return_with_cost'].loc['annualized_return', 'risk'] - analysis['bench'].loc['annualized_return', 'risk'],
    f'information_ratio_{segment}': analysis['return_with_cost'].loc['information_ratio', 'risk'],
    f'max_drawdown_{segment}': analysis['return_with_cost'].loc['max_drawdown', 'risk'],
    f'capm_alpha_{segment}': analysis['fit_capm'].loc['alpha', 'CAPM'],
    f'capm_beta_{segment}': analysis['fit_capm'].loc['beta', 'CAPM'],
    f'capm_alpha_annual_{segment}': analysis['fit_capm'].loc['alpha_annual', 'CAPM'],
  })
  recorder.log_metrics(**metrics)
  return metrics      

def objective(trial, base_config, experiment_name):
  trial_config = suggest_lightgbm_config(trial)
  config = generate_config(base_config=base_config, config=trial_config)

  qlib.init(**config["qlib_init"])

  with R.start(experiment_name=experiment_name):
    recorder = R.get_recorder()
    rid = recorder.id
    trial.set_user_attr("rid", rid)

    recorder.log_params(**flatten_dict(config))
    recorder.save_objects(**{"config.pkl": config})
    # model training
    model = init_instance_by_config(config["task"]["model"])
    dataset = init_instance_by_config(config["task"]["dataset"])

    model.fit(dataset)
    recorder.save_objects(**{"model.pkl": model})

    dataset.config(dump_all=False, recursive=True) # TODO: is this necessary?
    recorder.save_objects(**{"dataset.pkl": dataset}) # TODO: how to reuse dataset later?

    # evaluation: valid split
    metrics_valid = run_eval(model, dataset, segment="valid", recorder=recorder, config=config)
    # evaluation: test split
    metrics_test  = run_eval(model, dataset, segment="test" , recorder=recorder, config=config)

  metrics = metrics_valid | metrics_test

  return metrics["excess_annualized_return_valid"]


def main(n_trials=10):
  data_handler = "Alpha158"
  market = "hs300" # sz50, hs300, zz500
  benchmark = "SH000300" # SH000016, SH000300, SH000905
  deal_price = "close"
  freq = "1d"

  identifier = f"tune_single_{data_handler}_{market}_{deal_price}_{freq}"

  # workspace_path = "/capstor/scratch/cscs/ljiayong/workspace/qlib/tune/cn/lightgbm"
  workspace_path = "/iopsstor/scratch/cscs/ljiayong/workspace/qlib/tune/cn/lightgbm"

  base_config = {
    "exp_manager_uri": "file:///" + os.path.join(workspace_path, f"mlrun/{identifier}_runs"),
    "market":     market,
    "benchmark":  benchmark,
    "deal_price": deal_price,
    "freq":       freq,
    "train_split": ("2020-01-01", "2022-12-31"),
    "valid_split": ("2023-01-01", "2023-12-31"),
    "test_split":  ("2024-01-01", "2025-09-29"),
  }

  storage_path = os.path.join(workspace_path, f"optuna/{identifier}_journal.log")
  os.makedirs(os.path.dirname(storage_path), exist_ok=True)
  storage = JournalStorage(JournalFileBackend(storage_path))

  task_idx = 0 # only one task for single tuning
  study_name = f"task_{task_idx}"
  study = optuna.create_study(
    study_name=study_name,
    storage=storage,
    direction="maximize",
    load_if_exists=True,
  )
  study.optimize(functools.partial(objective, base_config=base_config, experiment_name=study_name), n_trials=n_trials, n_jobs=1)

if __name__ == "__main__":
  fire.Fire(main)