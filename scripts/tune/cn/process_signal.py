import os
import fire

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import numpy as np
import pandas as pd

from loguru import logger

import matplotlib.pyplot as plt

import qlib
from qlib.data.dataset.handler import DataHandlerLP
from qlib.constant import REG_CN
from qlib.data import D
from qlib.workflow import R
from qlib.backtest import backtest
from qlib.contrib.evaluate import risk_analysis, fit_capm
from qlib.contrib.report import analysis_position

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

def process_task(task_results, task_idx, save_path):
  # [top_k_idx][fold_idx]
  k = len(task_results)
  n_folds = len(task_results[0]) - 1  # last one is deploy
  steps_per_year = 246

  fold_boundaries = (
    [r["pred"].index.get_level_values("datetime").min() for r in task_results[0]] 
    + [task_results[0][-1]["pred"].index.get_level_values("datetime").max()]
  )

  base_config = task_results[0][0]["base_config"]
  market = base_config["market"]
  deal_price = base_config["deal_price"]
  freq = base_config["freq"]

  all_return_steps = [1, 5, 10]
  stock_ret = D.features(
    D.instruments(market), 
    [f"Ref(${deal_price}, -{1+n})/Ref(${deal_price}, -1) - 1" for n in all_return_steps], 
    start_time=fold_boundaries[0], 
    end_time=fold_boundaries[-1],
  ).swaplevel().sort_index() # df with (datetime, instrument) index
  stock_ret.columns = [f"ret{n}" for n in all_return_steps]

  # top-k signals
  top_k_signals = [] # [top_k_idx][fold_idx]
  for result in task_results:
    signal_lst = []
    for r in result:
      model = r["model"]
      dataset = r["dataset"]
      pred = r["pred"]
      start_time = pred.index.get_level_values("datetime").min().strftime("%Y-%m-%d")
      end_time = pred.index.get_level_values("datetime").max().strftime("%Y-%m-%d")

      dataset.config(handler_kwargs={"end_time": end_time})
      dataset.setup_data(handler_kwargs={"init_type": DataHandlerLP.IT_LS})
      signal = model.predict(dataset, segment=slice(start_time, end_time))
      # assert signal.equals(pred)
      signal_lst.append(signal)
    top_k_signals.append(pd.concat(signal_lst))

  avg_signal = pd.concat(top_k_signals, axis=1)
  avg_signal = avg_signal.groupby("datetime").transform(
    lambda x: (x - x.mean()) / x.std()
  )
  avg_signal = avg_signal.mean(axis=1, skipna=True)
  # pred_label = pd.concat([pred.rename("pred"), stock_ret.rename(columns={f"ret{num_steps}": "label"})], axis=1)
  # ic = pred_label.groupby("datetime").apply(lambda df: df["pred"].corr(df["label"], method=method))
  # import pdb;pdb.set_trace()
  
  all_signals = top_k_signals + [avg_signal]
  all_names = [f"Top-{i+1}" for i in range(k)] + ["Avg"]
  all_reports = []
  for signal in all_signals:
    report = run_sim(signal, base_config)
    all_reports.append(report)

  return_cum = []
  cost_cum = []
  bench_cum = []
  metrics = []

  for idx, df in enumerate(all_reports):
    r = df['return'] - df['cost']
    cum_r = (1 + r).cumprod() - 1
    cum_c = ((1 + cum_r) * df['cost']).cumsum()
    cum_b = (1 + df['bench']).cumprod() - 1

    return_cum.append(cum_r)
    cost_cum.append(cum_c)
    bench_cum.append(cum_b)

    metrics.append((
      risk_analysis(r, N=steps_per_year, mode="product"),
      fit_capm(r, df['bench'], N=steps_per_year, r_f_annual=2e-2)
    ))

  fig, axes = plt.subplots(3, 1, figsize=(6, 10))
  fig.suptitle(f"Task {task_idx} Top-{k} Results")

  # return
  ax = axes[0]
  for i, s in enumerate(return_cum):
    ax.plot(s.index, s.values, label=all_names[i])
  ax.plot(bench_cum[0].index, bench_cum[0].values, label="Bench", linestyle='--', linewidth=1)
  ax.set_title(f"Total Return")
  ax.set_ylabel("Total Return")
  ax.tick_params(labelbottom=False)

  # cost
  ax = axes[1]
  for i, s in enumerate(cost_cum):
    ax.plot(s.index, s.values, label=all_names[i])
  ax.set_title(f"Cost Ratio")
  ax.set_ylabel("Cost Ratio")
  ax.tick_params(labelbottom=False)

  # excess return
  ax = axes[2]
  for i, (r, b) in enumerate(zip(return_cum, bench_cum)):
    excess = r - b
    ax.plot(excess.index, excess.values, label=all_names[i])
  ax.set_title(f"Excess Return")
  ax.set_ylabel("Excess Return")
  ax.tick_params(axis='x', rotation=90)

  for ax in axes.flat:
    ax.grid()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    for b in fold_boundaries:
      ax.axvline(b, linestyle='--', linewidth=1, color='red', alpha=0.5)

  os.makedirs(save_path, exist_ok=True)
  plt.savefig(os.path.join(save_path, f"task_{task_idx}_top_{k}_sim_results.png"), bbox_inches="tight", dpi=500)
  plt.close()

def plot_task(task_results, task_idx, save_path):
  # [top_k_idx][fold_idx]
  k = len(task_results)
  n_folds = len(task_results[0]) - 1  # last one is deploy
  steps_per_year = 246

  top_k_reports = [] # [top_k_idx]
  top_k_preds = []
  for result in task_results:
    report = pd.concat([r["report"] for r in result], axis=0)
    pred = pd.concat([r["pred"] for r in result], axis=0)
    # df = df[~df.index.duplicated(keep='first')]
    top_k_reports.append(report[['return', 'cost', 'bench']])
    top_k_preds.append(pred)

  fold_boundaries = [r["report"].index[0] for r in task_results[0]]

  return_cum = []
  cost_cum = []
  bench_cum = []
  metrics = []

  for idx, df in enumerate(top_k_reports):
    r = df['return'] - df['cost']
    cum_r = (1 + r).cumprod() - 1
    cum_c = ((1 + cum_r) * df['cost']).cumsum()
    cum_b = (1 + df['bench']).cumprod() - 1

    return_cum.append(cum_r)
    cost_cum.append(cum_c)
    bench_cum.append(cum_b)

    metrics.append((
      risk_analysis(r, N=steps_per_year, mode="product"),
      fit_capm(r, df['bench'], N=steps_per_year, r_f_annual=2e-2)
    ))

  fig, axes = plt.subplots(3, 1, figsize=(6, 10))
  fig.suptitle(f"Task {task_idx} Top-{k} Results")

  # return
  ax = axes[0]
  for i, s in enumerate(return_cum):
    ax.plot(s.index, s.values, label=f"Top-{i+1}")
  ax.plot(bench_cum[0].index, bench_cum[0].values, label="Bench", linestyle='--', linewidth=1)
  ax.set_title(f"Total Return")
  ax.set_ylabel("Total Return")
  ax.tick_params(labelbottom=False)

  # cost
  ax = axes[1]
  for i, s in enumerate(cost_cum):
    ax.plot(s.index, s.values, label=f"Top-{i+1}")
  ax.set_title(f"Cost Ratio")
  ax.set_ylabel("Cost Ratio")
  ax.tick_params(labelbottom=False)

  # excess return
  ax = axes[2]
  for i, (r, b) in enumerate(zip(return_cum, bench_cum)):
    excess = r - b
    ax.plot(excess.index, excess.values, label=f"Top-{i+1}")
  ax.set_title(f"Excess Return")
  ax.set_ylabel("Excess Return")
  ax.tick_params(axis='x', rotation=90)

  for ax in axes.flat:
    ax.grid()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    for b in fold_boundaries:
      ax.axvline(b, linestyle='--', linewidth=1, color='red', alpha=0.5)

  os.makedirs(save_path, exist_ok=True)
  plt.savefig(os.path.join(save_path, f"task_{task_idx}_top_{k}_results.png"), bbox_inches="tight", dpi=500)
  plt.close()

def plot_all_tasks(results, save_path):
  # [cv_task_idx][top_k_idx][fold_idx]
  n_tasks = len(results)
  k = len(results[0])
  n_folds = len(results[0][0]) - 1  # last one is deploy
  steps_per_year = 246

  top_k_reports = [] # [top_k_idx]
  top_k_preds = []
  for idx in range(k):
    report = pd.concat([r[idx][n_folds]["report"] for r in results], axis=0)
    pred = pd.concat([r[idx][n_folds]["pred"] for r in results], axis=0)
    # df = df[~df.index.duplicated(keep='first')]
    top_k_reports.append(report[['return', 'cost', 'bench']])
    top_k_preds.append(pred)

  deploy_boundaries = [r[0][n_folds]["report"].index[0] for r in results]

  return_cum = []
  cost_cum = []
  bench_cum = []
  metrics = []

  for idx, df in enumerate(top_k_reports):
    r = df['return'] - df['cost']
    cum_r = (1 + r).cumprod() - 1
    cum_c = ((1 + cum_r) * df['cost']).cumsum()
    cum_b = (1 + df['bench']).cumprod() - 1

    return_cum.append(cum_r)
    cost_cum.append(cum_c)
    bench_cum.append(cum_b)

    metrics.append((
      risk_analysis(r, N=steps_per_year, mode="product"),
      fit_capm(r, df['bench'], N=steps_per_year, r_f_annual=2e-2)
    ))

  fig, axes = plt.subplots(3, 1, figsize=(6, 10))
  fig.suptitle(f"Deploy Top-{k} Results")

  # return
  ax = axes[0]
  for i, s in enumerate(return_cum):
    ax.plot(s.index, s.values, label=f"Top-{i+1}")
  ax.plot(bench_cum[0].index, bench_cum[0].values, label="Bench", linestyle='--', linewidth=1)
  ax.set_title(f"Total Return")
  ax.set_ylabel("Total Return")
  ax.tick_params(labelbottom=False)

  # cost
  ax = axes[1]
  for i, s in enumerate(cost_cum):
    ax.plot(s.index, s.values, label=f"Top-{i+1}")
  ax.set_title(f"Cost Ratio")
  ax.set_ylabel("Cost Ratio")
  ax.tick_params(labelbottom=False)

  # excess return
  ax = axes[2]
  for i, (r, b) in enumerate(zip(return_cum, bench_cum)):
    excess = r - b
    ax.plot(excess.index, excess.values, label=f"Top-{i+1}")
  ax.set_title(f"Excess Return")
  ax.set_ylabel("Excess Return")
  ax.tick_params(axis='x', rotation=90)

  for ax in axes.flat:
    ax.grid()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    for b in deploy_boundaries:
      ax.axvline(b, linestyle='--', linewidth=1, color='red', alpha=0.5)

  os.makedirs(save_path, exist_ok=True)
  plt.savefig(os.path.join(save_path, f"deploy_top_{k}_results.png"), bbox_inches="tight", dpi=500)
  plt.close()

def get_prediction(result):
  base_config = result["base_config"]
  dataset = result["dataset"]
  model = result["model"]

  valid_split = base_config["valid_split"]
  test_split = base_config["test_split"]    

  dataset.config(handler_kwargs={"end_time": test_split[1]})
  dataset.setup_data(handler_kwargs={"init_type": DataHandlerLP.IT_LS})

  start_time_lst = [valid_split[0], test_split[0]]
  end_time = test_split[1]

  signal = [
    model.predict(dataset, segment=slice(start_time, end_time))
    for start_time in start_time_lst
  ]
  # assert signal[1].equals(result["pred_test"])
  return signal

def simulate_signal(signals, base_config):
  benchmark = base_config["benchmark"]
  deal_price = base_config["deal_price"]
  freq = base_config["freq"]

  results = []
  for signal in signals:
    strategy_config = {
      "class": "TopkDropoutStrategy",
      "module_path": "qlib.contrib.strategy",
      "kwargs": {
        "signal": signal,
        "topk": 20,
        "n_drop": 2
      }
    }
    # {
    #   "class": "EnhancedIndexingStrategy",
    #   "module_path": "qlib.contrib.strategy",
    #   "kwargs": {
    #     "signal": recorder.load_object(f"pred_{split}.pkl"),
    #     "riskmodel_root": "/iopsstor/scratch/cscs/ljiayong/workspace/qlib/risk/cn/riskdata",
    #   }
    # }
    port_analysis_config = {
      "executor": {
        "class": "SimulatorExecutor",
        "module_path": "qlib.backtest.executor",
        "kwargs": {
          "time_per_step": "day" if freq == "1d" else freq,
          "generate_portfolio_metrics": True,
        },
      },
      "strategy": strategy_config,
      "backtest": {
        "start_time": signal.index.get_level_values("datetime").min().strftime("%Y-%m-%d"),
        "end_time": signal.index.get_level_values("datetime").max().strftime("%Y-%m-%d"),
        "account": 1_000_000,
        "benchmark": benchmark,
        "exchange_kwargs": {
          "freq": "day",
          "trade_unit": 100,
          "limit_threshold": 0.095,
          "deal_price": deal_price,
          "open_cost": 0.002,
          "close_cost": 0.0025,
          "min_cost": 5
        }
      }
    }
    steps_per_year = 246
    portfolio_metric_dict, indicator_dict = backtest(executor=port_analysis_config["executor"], strategy=port_analysis_config["strategy"], **port_analysis_config["backtest"])
    report = portfolio_metric_dict["1day" if freq == "1d" else freq][0]
    # position = portfolio_metric_dict["1day" if freq == "1d" else freq][1]
    # indicator = indicator_dict["1day" if freq == "1d" else freq]
    # print(risk_analysis(report["bench"], N=steps_per_year, mode="product"))
    # print(risk_analysis(report["return"]-report["cost"], N=steps_per_year, mode="product"))
    # print(fit_capm(report["return"]-report["cost"], report["bench"], N=steps_per_year, r_f_annual=2e-2))
    # analysis_position.report_graph(report, show_notebook=False, save_path=os.path.join(os.path.dirname(__file__), f"plots_{split}"))
    results.append({
      "report": report,
    })
  
  return results

def _plot_simulate_signal_results(all_reports, all_names, title, save_path):
  steps_per_year = 246
  return_cum = []
  cost_cum = []
  bench_cum = []
  metrics = []

  for idx, df in enumerate(all_reports):
    r = df['return'] - df['cost']
    cum_r = (1 + r).cumprod() - 1
    cum_c = ((1 + cum_r) * df['cost']).cumsum()
    cum_b = (1 + df['bench']).cumprod() - 1

    return_cum.append(cum_r)
    cost_cum.append(cum_c)
    bench_cum.append(cum_b)

    metrics.append((
      risk_analysis(r, N=steps_per_year, mode="product"),
      fit_capm(r, df['bench'], N=steps_per_year, r_f_annual=2e-2)
    ))

  fig, axes = plt.subplots(3, 1, figsize=(6, 10))
  fig.suptitle(title)

  # return
  ax = axes[0]
  for i, s in enumerate(return_cum):
    ax.plot(s.index, s.values, label=all_names[i])
  ax.plot(bench_cum[0].index, bench_cum[0].values, label="Bench", linestyle='--', linewidth=1)
  ax.set_title(f"Total Return")
  ax.set_ylabel("Total Return")
  ax.tick_params(labelbottom=False)

  # cost
  ax = axes[1]
  for i, s in enumerate(cost_cum):
    ax.plot(s.index, s.values, label=all_names[i])
  ax.set_title(f"Cost Ratio")
  ax.set_ylabel("Cost Ratio")
  ax.tick_params(labelbottom=False)

  # excess return
  ax = axes[2]
  for i, (r, b) in enumerate(zip(return_cum, bench_cum)):
    excess = r - b
    ax.plot(excess.index, excess.values, label=all_names[i])
  ax.set_title(f"Excess Return")
  ax.set_ylabel("Excess Return")
  ax.tick_params(axis='x', rotation=90)

  for ax in axes.flat:
    ax.grid()
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  plt.savefig(save_path, bbox_inches="tight", dpi=500)
  plt.close()

def plot_simulate_signal_results(signal_results, avg_signal_results, save_path):
  n = len(avg_signal_results)
  k = len(signal_results) // n
  num_splits = len(signal_results[0])
  for split_idx in range(num_splits):
    for group_idx in range(n):
      all_reports = signal_results[group_idx*k:(group_idx+1)*k] + [avg_signal_results[group_idx]]
      all_reports = [r[split_idx]["report"] for r in all_reports]
      all_names = [f"Top-{i+1}" for i in range(k)] + ["Avg"]

      _plot_simulate_signal_results(
        all_reports,
        all_names,
        title=f"Top-{k} Group {group_idx} Split {split_idx} Simulation Results",
        save_path=os.path.join(save_path, f"top_{k}_group_{group_idx}_split_{split_idx}_sim_results.png"),
      )
    
    _plot_simulate_signal_results(
      [r[split_idx]["report"] for r in avg_signal_results],
      [f"Group-{i} Avg" for i in range(n)],
      title=f"Avg Signal Top-{k} Group {group_idx} Split {split_idx} Simulation Results",
      save_path=os.path.join(save_path, f"avg_top_{k}_group_{group_idx}_split_{split_idx}_sim_results.png"),
    )


def main(model_name="GATs"):
  market = "hs300" # sz50, hs300, zz500
  benchmark = "SH000300" # SH000016, SH000300, SH000905
  deal_price = "close"
  freq = "1d"

  k = 5 # top k models to select and plot
  n = 3 # num top-k groups, n*k trials in total

  identifier = f"tune_single_{model_name.lower()}_{market}_{deal_price}_{freq}"

  workspace_path = f"/iopsstor/scratch/cscs/ljiayong/workspace/qlib/tune/cn/workspace/{identifier}_workspace"
  storage = JournalStorage(JournalFileBackend(os.path.join(workspace_path, f"optuna/{identifier}_journal.log")))

  '''
  Results to Sqlite DB
  '''
  dst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{identifier}_sqlite.db")
  os.makedirs(os.path.dirname(dst_path), exist_ok=True)
  os.path.exists(dst_path) and os.remove(dst_path)

  study_names = optuna.study.get_all_study_names(storage=storage)
  for study_name in study_names:
    logger.info(f"Copying study: {study_name}, {len(study_names)} studies in total ...")
    optuna.copy_study(
      from_study_name=study_name,
      from_storage=storage,
      to_storage=f"sqlite:///{dst_path}",
    )

  '''
  Load Trials
  '''
  study_name = f"task_0"
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

  '''
  Init Qlib
  '''
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

  '''
  Load Trial Artifacts
  '''
  results = [] # [n*k]
  df = df.sort_values("value", ascending=False).reset_index(drop=True).head(n*k)
  for idx in range(len(df)):
    recorder = R.get_recorder(experiment_name="task_0", recorder_id=df.at[idx, f"rid"])
    artifacts = {
      "base_config": recorder.load_object("base_config.pkl"),
      "dataset": recorder.load_object("dataset.pkl"),
      "model": recorder.load_object("model.pkl"),
      "pred_valid": recorder.load_object(f"pred_valid.pkl"),
      "pred_test": recorder.load_object(f"pred_test.pkl"),
    }
    results.append(artifacts)

  base_config = results[0]["base_config"]

  '''
  Generate Predictions
  '''
  signals = [get_prediction(result) for result in results] # [n*k][split]
  
  avg_signals = [] # [n][split]
  for group_idx in range(n):
    signal_group = signals[group_idx*k:(group_idx+1)*k]
    avg_signal = []
    for i in range(len(signals[0])):
      avg_s = pd.concat([s[i] for s in signal_group], axis=1)
      avg_s = avg_s.groupby("datetime").transform(lambda x: (x - x.mean()) / x.std())
      avg_s = avg_s.mean(axis=1)
      avg_signal.append(avg_s)
    avg_signals.append(avg_signal)

  '''
  Signal Analysis
  '''
  # analyze_signal_results = [analyze_signal(signal, base_config) for signal in signals]
  # analyze_avg_signal_results = [analyze_signal(avg_signal, base_config) for avg_signal in avg_signals]
  
  # plot_analyze_signal_results(
  #   analyze_signal_results,
  #   analyze_avg_signal_results,
  # )
  
  '''
  Simulation
  '''
  simulate_signal_results = [simulate_signal(signal, base_config) for signal in signals]
  simulate_avg_signal_results = [simulate_signal(avg_signal, base_config) for avg_signal in avg_signals]
  
  plot_simulate_signal_results(
    simulate_signal_results,
    simulate_avg_signal_results,
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/simulation/{identifier}")
  )


if __name__ == "__main__":
  fire.Fire(main)

