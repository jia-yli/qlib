import os
import fire
from pathlib import Path
from ruamel.yaml import YAML
from jinja2 import Template

import qlib
from qlib.constant import REG_CN
from qlib.model.trainer import task_train

def run(experiment_name="workflow", uri_folder="mlruns"):
  model_name = 'GATs'
  dataset = 'Alpha158'
  market = 'hs300'
  benchmark = "SH000300"
  deal_price = 'close'
  config_path = Path(__file__).resolve().parent / f"benchmarks/{model_name}/workflow_config_{model_name.lower()}_{dataset}_{market}.yaml"

  # Define date ranges
  train_start = "2020-01-01"
  train_end = "2022-12-31"
  valid_start = "2023-01-01"
  valid_end = "2023-12-31"
  test_start = "2024-01-01"
  test_end = "2025-09-29"
  
  base_config = {
    "market":      market,
    "benchmark":   benchmark,
    "deal_price":  deal_price,
    "steps_per_year": 246,
    "train_start": train_start,
    "train_end":   train_end,
    "valid_start": valid_start,
    "valid_end":   valid_end,
    "test_start":  test_start,
    "test_end":    test_end,
    "port_analysis_config": {
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
  }

  label = [[f"Ref(${deal_price}, -2)/Ref(${deal_price}, -1) - 1"], ["LABEL0"]]

  # load and run
  with open(config_path, "r") as f:
    template = Template(f.read())
  yaml = YAML(typ="safe", pure=True)
  config = yaml.load(template.render(base_config))

  qlib.init(
    provider_uri = "/capstor/scratch/cscs/ljiayong/datasets/qlib/cn_my_baostock/bin",
    region = REG_CN,
    exp_manager={
      "class": "MLflowExpManager",
      "module_path": "qlib.workflow.expm",
      "kwargs": {
        "uri": "file://" + str(Path("/users/ljiayong/projects/qlib/scripts/train/cn") / uri_folder),
        "default_exp_name": "Experiment",
      },
    }
  )

  task_config = config.get("task")
  task_config['dataset']['kwargs']['handler']['kwargs']['label'] = label

  recorder = task_train(task_config, experiment_name=experiment_name)
  recorder.save_objects(config=config)
  

if __name__ == "__main__":
  fire.Fire(run)
