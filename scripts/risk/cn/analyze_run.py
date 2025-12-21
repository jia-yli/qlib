import os
import fire
from pathlib import Path

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import kendalltau

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.workflow import R

def get_stock_ret(market, deal_price, num_steps, start_time, end_time):
  if not isinstance(num_steps, list):
    num_steps = [num_steps]
  stock_ret = D.features(
    D.instruments(market), 
    [f"Ref(${deal_price}, -{1+n})/Ref(${deal_price}, -1) - 1" for n in num_steps], 
    start_time=start_time, 
    end_time=end_time,
  ).swaplevel().sort_index() # df with (datetime, instrument) index
  stock_ret.columns = [f"ret{n}" for n in num_steps]
  return stock_ret

def get_bucket_hhi(pred_label, bucket_cols=["bucket"], group_cols=["sector"]):
  weight = pred_label.groupby(["datetime"] + bucket_cols + group_cols).size()
  weight = weight / weight.groupby(["datetime"] + bucket_cols).sum()
  hhi = weight.pow(2).groupby(["datetime"] + bucket_cols).sum()
  eff_n = 1/hhi
  hhi = pd.DataFrame({"hhi" : hhi, "eff_n": eff_n})
  return hhi

def analyze_signal(pred, base_config):
  market = base_config["market"]
  benchmark = base_config["benchmark"]
  deal_price = base_config["deal_price"]
  freq = base_config["freq"]

  stock_industry = pd.read_csv("/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock/processed/static_data/stock_industry.csv")
  stock_industry = stock_industry.set_index("symbol").rename_axis("instrument")

  start_time = pred.index.get_level_values("datetime").min()
  end_time = pred.index.get_level_values("datetime").max()

  # Stock return
  all_return_steps = [1, 5, 10]
  stock_ret = D.features(
    D.instruments(market), 
    [f"Ref(${deal_price}, -{1+n})/Ref(${deal_price}, -1) - 1" for n in all_return_steps], 
    start_time=start_time, 
    end_time=end_time,
  ).swaplevel().sort_index() # df with (datetime, instrument) index
  stock_ret.columns = [f"ret{n}" for n in all_return_steps]
  stock_ret = stock_ret.join(stock_industry[["sector", "industry"]], on="instrument")

  '''
  1. IC
  '''
  ic_metrics = {}
  ic_metrics_check_passed = {}
  for num_steps in all_return_steps:
    for method in ["pearson", "spearman"]:
      pred_label = pd.concat([pred.rename("pred"), stock_ret.rename(columns={f"ret{num_steps}": "label"})], axis=1)
      ic = pred_label.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method=method))
      ic_name = f"{num_steps}step_{'r' if method=='spearman' else ''}ic"
      # Compute IC metrics
      ic_metrics[f"{ic_name}_mean"] = ic.mean()
      ic_metrics[f"{ic_name}_ir"] = ic.mean() / ic.std() # 1 > 0.2, 10 > 0.2
      ic_metrics[f"{ic_name}_hitrate"] = (ic > 0).sum() / len(ic) # 10 > 0.55
      ic_metrics[f"{ic_name}_q40"] = ic.quantile(0.4) # > 0.01
      ic_metrics[f"{ic_name}_q50"] = ic.quantile(0.5) # > 0.02
      # Check IC metrics
      ic_metrics_check_passed[f"{ic_name}_mean_passed"] = ic_metrics[f"{ic_name}_mean"] > 0.02
      ic_metrics_check_passed[f"{ic_name}_ir_passed"] = ic_metrics[f"{ic_name}_ir"] > 0.2
      ic_metrics_check_passed[f"{ic_name}_hitrate_passed"] = ic_metrics[f"{ic_name}_hitrate"] > 0.55
      ic_metrics_check_passed[f"{ic_name}_q40_passed"] = ic_metrics[f"{ic_name}_q40"] > 0.01
      ic_metrics_check_passed[f"{ic_name}_q50_passed"] = ic_metrics[f"{ic_name}_q50"] > 0.02

      # Compute rolling window IC metrics
      window_metrics = ic.rolling(10).agg({
        "mean": lambda ic: ic.mean(),
        "ir": lambda ic: ic.mean() / ic.std(), # > 0.2, 0.3, 0.5
        "hitrate": lambda ic: (ic > 0).sum() / len(ic), # > 0.55, 0.6
      })
      window_metrics = window_metrics.quantile(0.4).to_dict()
      ic_metrics[f"{ic_name}_window_mean"] = window_metrics["mean"]
      ic_metrics[f"{ic_name}_window_ir"] = window_metrics["ir"]
      ic_metrics[f"{ic_name}_window_hitrate"] = window_metrics["hitrate"]
      # Check rolling window IC metrics
      ic_metrics_check_passed[f"{ic_name}_window_mean_passed"] = ic_metrics[f"{ic_name}_window_mean"] > 0.02
      ic_metrics_check_passed[f"{ic_name}_window_ir_passed"] = ic_metrics[f"{ic_name}_window_ir"] > 0.2
      ic_metrics_check_passed[f"{ic_name}_window_hitrate_passed"] = ic_metrics[f"{ic_name}_window_hitrate"] > 0.55

  '''
  2. Bucket
  '''
  bucket_metrics = {}
  bucket_metrics_check_passed = {}
  for num_steps in all_return_steps:
    for num_buckets in [5]:
      pred_label = pd.concat([pred.rename("pred"), stock_ret.rename(columns={f"ret{num_steps}": "label"})], axis=1)
      pred_bucket = (pred_label.groupby(level="datetime", group_keys=False)
        .apply(
          lambda x: pd.Series(range(len(s:=x.sort_values("pred", ascending=False))), index=s.index)
          // ((len(s) + num_buckets-1)//num_buckets)
        )
      ).rename("pred_bucket")
      label_bucket = (pred_label.groupby(level="datetime", group_keys=False)
        .apply(
          lambda x: pd.Series(range(len(s:=x.sort_values("label", ascending=False))), index=s.index)
          // ((len(s) + num_buckets-1)//num_buckets)
        )
      ).rename("label_bucket")
      pred_label = pd.concat([pred_label, pred_bucket, label_bucket], axis=1)
      bucket_mean = pred_label.groupby(level="datetime").apply(lambda x: x.groupby("pred_bucket")["label"].mean())
      bucket_mean_adjacent = bucket_mean.diff(periods=-1, axis=1).iloc[:, :-1]/2
      bucket_mean_longshort = (bucket_mean.iloc[:, 0] - bucket_mean.iloc[:, -1])/2
      bucket_mean_longshort_fee = bucket_mean_longshort - 0.2 * 0.003 # 20% turnover with 0.3% cost

      # Bucket cumulative return
      bucket_name = f"{num_steps}step_{num_buckets}bucket"
      bucket_cum = ((1+bucket_mean).cumprod() - 1).iloc[-1].diff(periods=-1).iloc[:-1].rename(lambda x: f"{bucket_name}_cum_{x}")
      bucket_cum_adjacent = ((1+bucket_mean_adjacent).cumprod() - 1).iloc[-1].rename(lambda x: f"{bucket_name}_cum_adjacent_{x}")
      bucket_cum_longshort = ((1+bucket_mean_longshort).cumprod() - 1).iloc[-1]
      bucket_cum_longshort_fee = ((1+bucket_mean_longshort_fee).cumprod() - 1).iloc[-1]

      bucket_metrics.update(bucket_cum.to_dict())
      bucket_metrics.update(bucket_cum_adjacent.to_dict())
      bucket_metrics[f"{bucket_name}_cum_longshort"] = bucket_cum_longshort
      bucket_metrics[f"{bucket_name}_cum_longshort_fee"] = bucket_cum_longshort_fee

      # Check bucket cumulative return
      bucket_metrics_check_passed.update((bucket_cum > 0).rename(lambda x: f"{x}_passed").to_dict())
      bucket_metrics_check_passed.update((bucket_cum_adjacent > 0).rename(lambda x: f"{x}_passed").to_dict())
      bucket_metrics_check_passed[f"{bucket_name}_cum_longshort_passed"] = bucket_cum_longshort > 0
      bucket_metrics_check_passed[f"{bucket_name}_cum_longshort_fee_passed"] = bucket_cum_longshort_fee > 0

      # check window, 0.48 positive
      w=10
      q=0.48
      bucket_window_cum = bucket_mean.rolling(w).apply(lambda x: ((1+x).cumprod() - 1).iloc[-1]).diff(periods=-1, axis=1).iloc[:, :-1].quantile(q).rename(lambda x: f"{bucket_name}_window_cum_{x}")
      bucket_window_cum_adjacent = bucket_mean_adjacent.rolling(w).apply(lambda x: ((1+x).cumprod() - 1).iloc[-1]).quantile(q).rename(lambda x: f"{bucket_name}_window_cum_adjacent_{x}")
      bucket_window_cum_longshort = bucket_mean_longshort.rolling(w).apply(lambda x: ((1+x).cumprod() - 1).iloc[-1]).quantile(q)
      bucket_window_cum_longshort_fee = bucket_mean_longshort_fee.rolling(w).apply(lambda x: ((1+x).cumprod() - 1).iloc[-1]).quantile(q)
      
      bucket_metrics.update(bucket_window_cum.to_dict())
      bucket_metrics.update(bucket_window_cum_adjacent.to_dict())
      bucket_metrics[f"{bucket_name}_window_cum_longshort"] = bucket_window_cum_longshort
      bucket_metrics[f"{bucket_name}_window_cum_longshort_fee"] = bucket_window_cum_longshort_fee

      # Check bucket cumulative return
      bucket_metrics_check_passed.update((bucket_window_cum > 0).rename(lambda x: f"{x}_passed").to_dict())
      bucket_metrics_check_passed.update((bucket_window_cum_adjacent > 0).rename(lambda x: f"{x}_passed").to_dict())
      bucket_metrics_check_passed[f"{bucket_name}_window_cum_longshort_passed"] = bucket_window_cum_longshort > 0
      bucket_metrics_check_passed[f"{bucket_name}_window_cum_longshort_fee_passed"] = bucket_window_cum_longshort_fee > 0
      # TODO: sharpe, max drawdown, etc.

      # HHI
      # hhi_universe = get_bucket_hhi(pred_label, bucket_cols=[], group_cols=["sector"])
      hhi_bucket = get_bucket_hhi(pred_label, bucket_cols=["pred_bucket"], group_cols=["sector"])
      bucket_metrics[f"{bucket_name}_bucket_effn_q10"] = hhi_bucket["eff_n"].quantile(0.1)
      bucket_metrics[f"{bucket_name}_bucket_effn_q30"] = hhi_bucket["eff_n"].quantile(0.3)

      # Check HHI
      bucket_metrics_check_passed[f"{bucket_name}_bucket_effn_q10_passed"] = bucket_metrics[f"{bucket_name}_bucket_effn_q10"] > 2
      bucket_metrics_check_passed[f"{bucket_name}_bucket_effn_q30_passed"] = bucket_metrics[f"{bucket_name}_bucket_effn_q30"] > 3

  '''
  3. Autocorrelation
  '''
  ac_metrics = {}
  ac_metrics_check_passed = {}
  for num_steps in [1]:
    for method in ["pearson", "spearman"]:
      pred_label = pd.concat([pred.rename("pred"), stock_ret.rename(columns={f"ret{num_steps}": "label"})], axis=1)
      pred_label_lag = pred_label.groupby("instrument").shift(num_steps)
      pred_label = pred_label.join(pred_label_lag, how="outer", rsuffix="_lag")
      ac_name = f"{num_steps}step_{'r' if method=='spearman' else ''}ac"
      ac = pred_label.groupby(level="datetime").apply(lambda x: x["pred"].corr(x["pred_lag"], method=method))
      ac_metrics[f"{ac_name}_mean"] = ac.mean() # > 0.3
      ac_metrics[f"{ac_name}_ir"] = ac.mean() / ac.std() # > 1.5
      ac_metrics[f"{ac_name}_q40"] = ac.quantile(0.4) # > 0.3

      # Check AC metrics
      ac_metrics_check_passed[f"{ac_name}_mean_passed"] = ac_metrics[f"{ac_name}_mean"] > 0.3
      ac_metrics_check_passed[f"{ac_name}_ir_passed"] = ac_metrics[f"{ac_name}_ir"] > 1.5
      ac_metrics_check_passed[f"{ac_name}_q40_passed"] = ac_metrics[f"{ac_name}_q40"] > 0.3
  
  '''
  Aggregate Results
  '''
  signal_metrics = {
    **ic_metrics,
    **bucket_metrics,
    **ac_metrics,
  }
  signal_metrics_check_passed = {
    **ic_metrics_check_passed,
    **bucket_metrics_check_passed,
    **ac_metrics_check_passed,
    "signal_all_passed": all(
      list(ic_metrics_check_passed.values()) + 
      list(bucket_metrics_check_passed.values()) + 
      list(ac_metrics_check_passed.values())
    ),
  }

  return signal_metrics, signal_metrics_check_passed

def analyze_run(recorder, benchmark):
  base_config = dict(
    market = "hs300", # sz50, hs300, zz500
    benchmark = "SH000300", # SH000016, SH000300, SH000905
    deal_price = "close",
    freq = "1d",
  )
  freq = base_config["freq"]

  run_metrics = recorder.list_metrics()
  # Load prediction and label
  for split in ["valid", "test"]:
    '''
    load run data
    '''
    # load signal
    pred = recorder.load_object(f"pred_{split}.pkl")

    # load portfolio metrics
    portfolio_metric_dict = recorder.load_object(f"portfolio_metric_{split}.pkl")
    freqs = list(portfolio_metric_dict.keys())
    assert len(freqs) == 1, "Only support one frequency now."
    report = portfolio_metric_dict["1day" if freq == "1d" else freq][0]
    position = portfolio_metric_dict["1day" if freq == "1d" else freq][1]

    '''
    analyze signal
    '''
    signal_metrics, signal_metrics_check_passed = analyze_signal(pred, base_config)
    run_metrics.update({f"{k}_{split}": v for k, v in signal_metrics.items()})
    run_metrics.update({f"{k}_{split}": v for k, v in signal_metrics_check_passed.items()})

  return run_metrics


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

  assert workflow == "single"
  for task_idx in range(n_tasks):
    df = dfs[task_idx]
    for idx, row in df.iterrows():
      recorder = R.get_recorder(experiment_name=f"task_{task_idx}", recorder_id=row["rid"])
      run_metrics = analyze_run(recorder, benchmark=benchmark)
      df.loc[idx, run_metrics.keys()] = run_metrics
    df = df.sort_values("signal_all_passed_valid", ascending=False).reset_index(drop=True)
    save_path = os.path.join(os.path.dirname(__file__), f"results/analyze_run/{identifier}_task{task_idx}.csv")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
  fire.Fire(main)

