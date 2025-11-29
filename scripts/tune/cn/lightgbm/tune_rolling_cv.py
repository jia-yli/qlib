import os
import copy
import optuna
import functools

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from loguru import logger
from filelock import FileLock

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict, fill_placeholder, class_casting
from qlib.workflow import R
from qlib.backtest import backtest
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.eva.alpha import calc_ic
from qlib.contrib.evaluate import risk_analysis, fit_capm

def gen_rolling_splits(
  start_time, # inc
  end_time, # inc
  train_size,
  valid_size,
  test_size,
  inclusive = True,
  freq = "1d",
):
  start_time = pd.Timestamp(start_time)
  end_time = pd.Timestamp(end_time)

  train_offset = to_offset(train_size)
  valid_offset = to_offset(valid_size)
  test_offset  = to_offset(test_size)

  all_segments = []

  train_start = start_time
  while True:
    train_end = train_start + train_offset
    valid_start = train_end
    valid_end = valid_start + valid_offset
    test_start = valid_end
    test_end = test_start + test_offset

    if valid_start <= end_time:
      _valid_end = min(valid_end, end_time + pd.to_timedelta(freq))
    else:
      _valid_end = valid_end
    
    if test_start <= end_time:
      _test_end  = min(test_end , end_time + pd.to_timedelta(freq))
    else:
      _test_end = test_end

    if inclusive:
      segments = {
        "train_split": (train_start.strftime("%Y-%m-%d %H:%M:%S"), (train_end - pd.to_timedelta(freq)).strftime("%Y-%m-%d %H:%M:%S")),
        "valid_split": (valid_start.strftime("%Y-%m-%d %H:%M:%S"), (_valid_end - pd.to_timedelta(freq)).strftime("%Y-%m-%d %H:%M:%S")),
        "test_split" : (test_start .strftime("%Y-%m-%d %H:%M:%S"), (_test_end  - pd.to_timedelta(freq)).strftime("%Y-%m-%d %H:%M:%S")),
      }
    else:
      segments = {
        "train_split": (train_start.strftime("%Y-%m-%d %H:%M:%S"), train_end.strftime("%Y-%m-%d %H:%M:%S")),
        "valid_split": (valid_start.strftime("%Y-%m-%d %H:%M:%S"), _valid_end.strftime("%Y-%m-%d %H:%M:%S")),
        "test_split" : (test_start .strftime("%Y-%m-%d %H:%M:%S"), _test_end .strftime("%Y-%m-%d %H:%M:%S")),
      }

    all_segments.append(segments)

    if test_start > end_time:
      # stop here, last test set is completely out of range
      # end_time is always inclusive
      break
    test_start_next = test_end
    train_start = test_start_next - train_offset - valid_offset

  return all_segments


def generate_config(base_config, config):
  # parse base config
  exp_manager_uri = base_config['exp_manager_uri']

  market = base_config["market"]
  benchmark = base_config["benchmark"]
  deal_price = base_config["deal_price"]

  freq = base_config["freq"]
  assert int(pd.to_timedelta("1day") / pd.to_timedelta(freq)) == 1, "Only support daily frequency now."
  steps_per_year = 246 

  train_start, train_end = base_config["train_split"]
  valid_start, valid_end = base_config["valid_split"]
  test_start , test_end  = base_config["test_split"]

  # load config
  if isinstance(config, dict):
    config_dict = config
  else:
    config_path = Path(config)
    yaml = YAML(typ="safe")
    with config_path.open("r") as f:
      config_dict = yaml.load(f)

  # full config
  '''
  init
  '''
  init_config = {
    "provider_uri": "/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock/bin",
    "region": REG_CN,
    "exp_manager": {
      "class": "MLflowExpManager",
      "module_path": "qlib.workflow.expm",
      "kwargs": {
        "uri": exp_manager_uri,
        "default_exp_name": "default_experiment",
      },
    }
  }

  '''
  model
  '''
  model_config = config_dict["model"]

  '''
  dataset
  '''
  data_handler_config = {
    "start_time": train_start,
    "end_time": test_end,
    "fit_start_time": train_start,
    "fit_end_time": train_end,
    "instruments": market,
    "label": [[f"Ref(${deal_price}, -2)/Ref(${deal_price}, -1) - 1"], ["LABEL0"]],
  }
  if config_dict.get("data_handler_config", None):
    data_handler_config.update(config_dict["data_handler_config"])

  assert set(config_dict["data_handler"].keys()) == {"class", "module_path"}, \
    f"{set(config_dict['data_handler'].keys())} != {{'class', 'module_path'}}"
  assert not ({"handler", "segments"} & set(config_dict["dataset"].get("kwargs", {}).keys())), \
    f"{set(config_dict['dataset'].get('kwargs', {}).keys())} has intersection with {{'handler', 'segments'}}"

  dataset_config = {
    "class": config_dict["dataset"]["class"],
    "module_path": config_dict["dataset"]["module_path"],
    "kwargs": {
      "handler": {
        "class": config_dict["data_handler"]["class"],
        "module_path": config_dict["data_handler"]["module_path"],
        "kwargs": data_handler_config,
      },
      "segments": {
        "train": [train_start, train_end],
        "valid": [valid_start, valid_end],
        "test" : [test_start , test_end ],
      },
    }
  }
  if config_dict["dataset"].get("kwargs", None):
    dataset_config["kwargs"].update(config_dict["dataset"]["kwargs"])

  '''
  record
  '''
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
        "signal": "<PRED>",
        "topk": 50,
        "n_drop": 5
      }
    },
    "backtest": {
      "start_time": "<BACKTEST_START>",
      "end_time": "<BACKTEST_END>",
      "account": 1_000_000,
      "benchmark": benchmark,
      "exchange_kwargs": {
        "freq": "day",
        "trade_unit": 100,
        "limit_threshold": 0.095,
        "deal_price": deal_price,
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "min_cost": 5
      }
    }
  }

  config = {
    "base": base_config,
    "qlib_init": init_config,
    "task": {
      "model": model_config,
      "dataset": dataset_config,
      "port_analysis_config": port_analysis_config,
      "steps_per_year": steps_per_year,
    }
  }
  return config


def suggest_config(trial):
  model_config = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
      "loss": "mse",

      "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
      "num_leaves": trial.suggest_int("num_leaves", 31, 511),
      "max_depth": trial.suggest_int("max_depth", -1, 16),
      "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),

      "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
      "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
      "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),

      "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
      "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),

      # "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
      # no early stopping in dart
    },
  }

  trial_config = {
    "model": model_config,
    "dataset": {
      "class": "DatasetH",
      "module_path": "qlib.data.dataset",
    },
    "data_handler": {
      "class": "Alpha158",
      "module_path": "qlib.contrib.data.handler",
    },
  }
  return trial_config


def run_eval(model, dataset, segment, recorder, config):
  base_config = config["base"]
  freq = base_config["freq"]
  steps_per_year = config["task"]["steps_per_year"]
  segment_start, segment_end = base_config[f"{segment}_split"]
  metrics = dict()
  # prediction
  pred = model.predict(dataset, segment=segment)
  assert isinstance(dataset, DatasetH)
  with class_casting(dataset, DatasetH):
    params = dict(segments=segment, col_set="label", data_key=DataHandlerLP.DK_R)
    try:
      # Assume the backend handler is DataHandlerLP
      raw_label = dataset.prepare(**params)
    except TypeError:
      # The argument number is not right
      del params["data_key"]
      # The backend handler should be DataHandler
      raw_label = dataset.prepare(**params)

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
  portfolio_metric_dict, indicator_dict = backtest(executor=port_analysis_config["executor"], strategy=port_analysis_config["strategy"], **port_analysis_config["backtest"])
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
  recorder.save_objects(**{f"metrics_{segment}.pkl": metrics})
  return metrics    


def run_config(config, experiment_name):
  base_config = config["base"]
  freq = base_config["freq"]

  run_results = {}
  qlib.init(**config["qlib_init"])

  with R.start(experiment_name=experiment_name):
    recorder = R.get_recorder()
    rid = recorder.id
    run_results["rid"] = rid

    recorder.save_objects(**{"config.pkl": config})
    # model training
    model = init_instance_by_config(config["task"]["model"])
    dataset = init_instance_by_config(config["task"]["dataset"])
    model.fit(dataset)
    recorder.save_objects(**{"model.pkl": model})

    # evaluation: valid split
    metrics_valid = run_eval(model, dataset, segment="valid", recorder=recorder, config=config)
    # evaluation: test split
    metrics_test  = run_eval(model, dataset, segment="test" , recorder=recorder, config=config)

  run_results.update(metrics_valid | metrics_test)
  return run_results


def objective(trial, cv_task, base_config, experiment_name):
  trial_config = suggest_config(trial)
  n_folds = len(cv_task["fit"])
  target = np.zeros(n_folds)
  for fit_task_idx, fit_task in enumerate(cv_task["fit"]):
    config = generate_config(
      base_config={**base_config, **fit_task},
      config=trial_config,
    )
    result = run_config(config, experiment_name)
    for key, value in result.items():
      trial.set_user_attr(f"fit_{fit_task_idx}_{key}", value)
    target[fit_task_idx] = result["annualized_return_test"]

  target = (1 + target).prod() ** (1 / n_folds) - 1  # geometric mean
  
  deploy_task = cv_task["deploy"]
  config = generate_config(
    base_config={**base_config, **deploy_task},
    config=trial_config,
  )
  deploy_result = run_config(config, experiment_name)
  for key, value in deploy_result.items():
    trial.set_user_attr(f"deploy_{key}", value)

  return target


def main():
  data_handler = "Alpha158"
  market = "hs300" # sz50, hs300, zz500
  benchmark = "SH000300" # SH000016, SH000300, SH000905
  deal_price = "close"
  freq = "1d"

  dataset_start_time = "2020-01-01"
  deploy_start_time = "2024-01-01"
  deploy_end_time = "2025-09-29"
  train_size = "24MS"
  valid_size = "6MS"
  test_size = "3MS"
  n_folds = 4

  '''
  Generate rolling splits
  '''
  rolling_splits = gen_rolling_splits(
    start_time=dataset_start_time,
    end_time=deploy_end_time,
    train_size=train_size,
    valid_size=valid_size,
    test_size=test_size,
    inclusive=False,
    freq=freq,
  )

  cross_validation_tasks = []
  for split_idx in range(len(rolling_splits)):
    rolling_split = rolling_splits[split_idx]
    if pd.Timestamp(rolling_split["test_split"][0]) >= pd.Timestamp(deploy_start_time):
      # get current task and all n previous tasks
      assert (split_idx - n_folds) >= 0
      cross_validation_task = {
        "fit" : [],
        "deploy" : None
      }
      for fold_idx in range(n_folds):
        cross_validation_task["fit"].append(copy.deepcopy(rolling_splits[split_idx - n_folds + fold_idx]))

      cross_validation_task["deploy"] = copy.deepcopy(rolling_split)
      cross_validation_tasks.append(cross_validation_task)

  '''
  Run Cross Validation
  '''
  base_config = {
    "exp_manager_uri": "file:///" + os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/runs"),
    "market":     market,
    "benchmark":  benchmark,
    "deal_price": deal_price,
    "freq":       freq,
  }

  for cv_task_idx, cv_task in enumerate(cross_validation_tasks[:4]):
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/optuna.db")
    storage_uri = f"sqlite:///{db_path}"
    study_name = f"LightGBM_{data_handler}_{market}_{freq}_task{cv_task_idx}"

    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with FileLock(db_path + ".lock"):
      # logger.warning(f"Delete existing study {study_name} at {storage_uri} if any.")
      # optuna.delete_study(study_name=study_name,storage=storage_uri)
      study = optuna.create_study(
        study_name=study_name, 
        storage=storage_uri, 
        direction="maximize",
        load_if_exists=True,
      )

    study.optimize(functools.partial(objective, cv_task=cv_task, base_config=base_config, experiment_name=study_name), n_trials=1, n_jobs=1)

if __name__ == "__main__":
  main()

