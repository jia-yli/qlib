import pandas as pd
from pandas.tseries.frequencies import to_offset
from pathlib import Path
from ruamel.yaml import YAML
from qlib.constant import REG_CN

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
        "topk": 20,
        "n_drop": 2
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
        "open_cost": 0.002,
        "close_cost": 0.002,
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
