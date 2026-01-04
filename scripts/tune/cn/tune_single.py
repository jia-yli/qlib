import os
import copy
import fire
import functools

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import numpy as np
import pandas as pd

from loguru import logger

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
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))
from tune_utils import generate_config
import suggest_config

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
    f"ic_ir_{segment}": ic.mean() / ic.std(),
    f"ric_{segment}": ric.mean(),
    f"ric_ir_{segment}": ric.mean() / ric.std(),
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

  for k, v in metrics.items():
    if isinstance(v, np.generic):
      metrics[k] = v.item()

  recorder.log_metrics(**metrics)
  logger.info(f"Segment {segment} IC: {metrics[f'ic_{segment}']}, IC IR: {metrics[f'ic_ir_{segment}']}")
  logger.info(f"Segment {segment} RIC: {metrics[f'ric_{segment}']}, RIC IR: {metrics[f'ric_ir_{segment}']}")
  logger.info(f"Segment {segment} Bench Annualized Return: {metrics[f'bench_annualized_return_{segment}']}")
  logger.info(f"Segment {segment} Annualized Return: {metrics[f'annualized_return_{segment}']}")
  logger.info(f"Segment {segment} Excess Annualized Return: {metrics[f'excess_annualized_return_{segment}']}")
  return metrics

def objective(trial, base_config, experiment_name):
  model_name = base_config["model_name"]
  trial_config = eval(f"suggest_config.suggest_{model_name.lower()}_config(trial)")
  config = generate_config(base_config=base_config, config=trial_config)

  qlib.init(**config["qlib_init"])

  with R.start(experiment_name=experiment_name):
    recorder = R.get_recorder()
    rid = recorder.id
    trial.set_user_attr("rid", rid)

    recorder.log_params(**flatten_dict(config))
    recorder.save_objects(**{
      "base_config.pkl": base_config,
      "config.pkl": config,
    })
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

  # user attr for other metrics
  for key, value in metrics.items():
    trial.set_user_attr(key, value)

  return metrics["ic_valid"]


def main(model_name="LightGBM", n_trials=8):
  market = "zz500" # sz50, hs300, zz500
  benchmark = "SH000905" # SH000016, SH000300, SH000905
  deal_price = "close"
  freq = "1d"

  identifier = f"tune_single_{model_name.lower()}_{market}_{deal_price}_{freq}"

  logger.info(f"Start tuning {model_name} with identifier {identifier}")
  logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

  # workspace_path = f"/capstor/scratch/cscs/ljiayong/workspace/qlib/tune/cn/workspace/{identifier}_workspace"
  workspace_path = f"/iopsstor/scratch/cscs/ljiayong/workspace/qlib/tune/cn/workspace/{identifier}_workspace"

  base_config = {
    "exp_manager_uri": "file:///" + os.path.join(workspace_path, f"mlrun/{identifier}_runs"),
    "model_name": model_name,
    "market":     market,
    "benchmark":  benchmark,
    "deal_price": deal_price,
    "freq":       freq,
    "train_split": ("2016-01-01", "2021-12-31"),
    "valid_split": ("2022-01-01", "2023-12-31"),
    "test_split":  ("2024-01-01", "2025-12-10"),
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