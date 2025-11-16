import os
import fire
import pandas as pd
from pathlib import Path
from ruamel.yaml import YAML

import qlib
from qlib.constant import REG_CN
from qlib.model.trainer import task_train

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
  record_config = [
    {
      "class": "SignalRecord",
      "module_path": "qlib.workflow.record_temp",
      "kwargs": {
        "model": "<MODEL>",
        "dataset": "<DATASET>",
      }
    },
    {
      "class": "SigAnaRecord",
      "module_path": "qlib.workflow.record_temp"
    },
    {
      "class": "PortAnaRecord",
      "module_path": "qlib.workflow.record_temp",
      "kwargs": {
        "config": port_analysis_config,
        "N": steps_per_year,
      },
    },
  ]

  config = {
    "qlib_init": init_config,
    "task": {
      "model": model_config,
      "dataset": dataset_config,
      "record": record_config,
    }
  }
  return config

def run_config(config, experiment_name):
  qlib.init(**config["qlib_init"])
  recorder = task_train(config.get("task"), experiment_name=experiment_name)
  recorder.save_objects(config=config)

def run(experiment_name="workflow"):
  model_name = "GATs"
  data_handler = "Alpha158"
  market = "hs300" # sz50, hs300, zz500
  benchmark = "SH000300" # SH000016, SH000300, SH000905
  deal_price = "close"
  freq = "1d"

  base_config = {
    "exp_manager_uri": "file://" + str(Path("/users/ljiayong/projects/qlib/scripts/train/cn") / f"{market}_{data_handler}_test_run" / f"{freq}"),
    "market":     market,
    "benchmark":  benchmark,
    "deal_price": deal_price,
    "freq":       freq,
    "train_split": ("2020-01-01", "2022-12-31"),
    "valid_split": ("2023-01-01", "2023-12-31"),
    "test_split":  ("2024-01-01", "2025-09-29"),
  }

  config_path = Path(__file__).resolve().parent / f"benchmarks/{model_name}_{market}_{data_handler}.yaml"
  config = generate_config(base_config, config_path)

  run_config(config, experiment_name=experiment_name)

if __name__ == "__main__":
  fire.Fire(run)
