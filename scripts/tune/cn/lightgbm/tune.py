import os
import fire
import qlib
import optuna
import functools

import pandas as pd

from loguru import logger
from filelock import FileLock

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.contrib.report import analysis_position

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
      "start_time": test_start,
      "end_time": test_end,
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

def suggest_config(trial, base_config):
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
  config = generate_config(
    base_config = base_config,
    config = {
      "model": model_config,
      "dataset": {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
      },
      "data_handler": {
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
      },
    },
  )
  return config

def run_config(config, experiment_name):
  freq  = config["base"]["freq"]
  qlib.init(**config["qlib_init"])
  model = init_instance_by_config(config["task"]["model"])
  dataset = init_instance_by_config(config["task"]["dataset"])

  # start exp
  with R.start(experiment_name=experiment_name):
    recorder = R.get_recorder()
    rid = recorder.id
    recorder.save_objects(config=config)

    # model training
    evals_result = dict()
    model.fit(dataset, evals_result=evals_result)
    R.save_objects(**{"model.pkl": model, "evals_result.pkl": evals_result})

    # prediction
    sr = SignalRecord(model, dataset, recorder)
    sr.generate()

    # Signal Analysis
    sar = SigAnaRecord(recorder)
    sar.generate()

    # backtest & analysis
    par = PortAnaRecord(
      recorder, 
      config=config["task"]["port_analysis_config"], 
      N=config["task"]["steps_per_year"], 
      risk_analysis_freq="day" if freq == "1d" else freq
    )
    par.generate()
  
  return rid, experiment_name

def objective(trial, base_config):
  config = suggest_config(trial, base_config)
  rid, experiment_name = run_config(config, experiment_name="tune")

  freq = config["base"]["freq"]
  steps_per_year = config["task"]["steps_per_year"]

  recorder = R.get_recorder(recorder_id=rid, experiment_name=experiment_name)

  evals_result = recorder.load_object("evals_result.pkl")
  report_normal_df = recorder.load_object(f"portfolio_analysis/report_normal_{'1day' if freq == '1d' else freq}.pkl")
  positions = recorder.load_object(f"portfolio_analysis/positions_normal_{'1day' if freq == '1d' else freq}.pkl")
  analysis = recorder.load_object(f"portfolio_analysis/port_analysis_{'1day' if freq == '1d' else freq}.pkl")

  # import pdb;pdb.set_trace()
  # model.model.best_iteration
  # model.model.curret_iteration()
  valid_l2 = min(evals_result["valid"]["l2"])
  trial.set_user_attr(f"valid_l2", min(evals_result["valid"]["l2"]))
  trial.set_user_attr(f'test_annualized_return', analysis['return_with_cost'].loc['annualized_return', 'risk'])
  trial.set_user_attr(
    f'test_excess_annualized_return', 
    analysis['return_with_cost'].loc['annualized_return', 'risk'] - analysis['bench'].loc['annualized_return', 'risk'],
  )
  trial.set_user_attr(f'test_information_ratio',analysis['return_with_cost'].loc['information_ratio', 'risk'])
  trial.set_user_attr(f'test_max_drawdown',analysis['return_with_cost'].loc['max_drawdown', 'risk'])
  return valid_l2


if __name__ == "__main__":
  data_handler = "Alpha158"
  market = "hs300" # sz50, hs300, zz500
  benchmark = "SH000300" # SH000016, SH000300, SH000905
  deal_price = "close"
  freq = "1d"

  base_config = {
    "exp_manager_uri": "file:///" + os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{market}_{data_handler}_{freq}_runs"),
    "market":     market,
    "benchmark":  benchmark,
    "deal_price": deal_price,
    "freq":       freq,
    "train_split": ("2020-01-01", "2022-12-31"),
    "valid_split": ("2023-01-01", "2023-12-31"),
    "test_split":  ("2024-01-01", "2025-09-29"),
  }
  db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/optuna.db")
  storage_uri = f"sqlite:///{db_path}"
  study_name = f"LightGBM_{data_handler}_{freq}"

  os.makedirs(os.path.dirname(db_path), exist_ok=True)

  with FileLock(db_path + ".lock"):
    # logger.warning(f"Delete existing study {study_name} at {storage_uri} if any.")
    # optuna.delete_study(study_name=study_name,storage=storage_uri)
    study = optuna.create_study(
      study_name=study_name, 
      storage=storage_uri, 
      direction="minimize",
      load_if_exists=True,
    )

  study.optimize(functools.partial(objective, base_config=base_config), n_trials=10, n_jobs=1)
