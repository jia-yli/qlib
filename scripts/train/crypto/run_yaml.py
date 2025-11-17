import os
import fire
import pandas as pd
from pathlib import Path
from ruamel.yaml import YAML

import qlib
from qlib.constant import REG_CRYPTO
from qlib.model.trainer import task_train

def generate_config(base_config, config):
  # parse base config
  exp_manager_uri = base_config['exp_manager_uri']

  market = base_config["market"]
  benchmark = base_config["benchmark"]
  deal_price = base_config["deal_price"]

  freq = base_config["freq"]
  trading_days_per_year = 365
  steps_per_year = trading_days_per_year * int(pd.to_timedelta("1day") / pd.to_timedelta(freq)) 

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
    "provider_uri": f"/capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto/bin/{freq}",
    "region": REG_CRYPTO,
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
    "freq": "day" if freq == "1d" else freq,
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
        "topk": 20,
        "n_drop": 2,
      }
    },
    "backtest": {
      "start_time": test_start,
      "end_time": test_end,
      "account": 10_000,
      "benchmark": benchmark,
      "exchange_kwargs": {
        "freq": "day" if freq == "1d" else freq,
        "trade_unit": None,
        "limit_threshold": None,
        "deal_price": deal_price,
        "open_cost": 0.001,
        "close_cost": 0.001,
        "min_cost": 0,
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
  recorder = task_train(config["task"], experiment_name=experiment_name)
  recorder.save_objects(config=config)

def run(experiment_name="workflow"):
  model_name = "LightGBM"
  data_handler = "Alpha158"
  market = "my_universe_top50"
  benchmark = "BTCUSDT"
  freq = "240min" # ["1d", "720min", "240min", "60min", "30min", "15min"]
  deal_price = 'open'

  base_config = {
    "exp_manager_uri": "file://" + str(Path("/users/ljiayong/projects/qlib/scripts/train/crypto") / f"{market}_{data_handler}_test_run" / f"{freq}"),
    "market":     market,
    "benchmark":  benchmark,
    "deal_price": deal_price,
    "freq":       freq,
    "train_split": ("2021-04-01", "2022-12-31"),
    "valid_split": ("2023-01-01", "2023-12-31"),
    "test_split":  ("2024-01-01", "2025-09-29"),
  }

  config_path = Path(__file__).resolve().parent / f"benchmarks/{model_name}_{market}_{data_handler}.yaml"
  config = generate_config(base_config, config_path)

  run_config(config, experiment_name=experiment_name)

if __name__ == "__main__":
  fire.Fire(run)

