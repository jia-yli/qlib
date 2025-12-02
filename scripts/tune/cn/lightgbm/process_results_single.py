import os

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.contrib.evaluate import risk_analysis, fit_capm

def plot_task(task_results, task_idx, save_path):
  # [top_k_idx][fold_idx]
  k = len(task_results)
  n_folds = len(task_results[0]) - 1  # last one is deploy
  steps_per_year = 246

  top_k_reports = [] # [top_k_idx]
  top_k_preds = []
  for result in task_results:
    report = pd.concat([r["report_test"] for r in result], axis=0)
    pred = pd.concat([r["pred_test"] for r in result], axis=0)
    # df = df[~df.index.duplicated(keep='first')]
    top_k_reports.append(report[['return', 'cost', 'bench']])
    top_k_preds.append(pred)

  fold_boundaries = [r["report_test"].index[0] for r in task_results[0]]

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

def plot_all_tasks(results, task_idx, save_path):
  # [cv_task_idx][top_k_idx][fold_idx]
  num_cv_tasks = len(top_k_results)
  k = len(top_k_results[0])
  n_folds = len(top_k_results[0][0]) - 1  # last one is deploy
  steps_per_year = 246

  top_k_dfs = [] # [top_k_idx]
  for idx in range(k):
    result = []
    for cv_task_idx in range(num_cv_tasks):
      result.append(top_k_results[cv_task_idx][idx][n_folds])  # deploy results
    df = pd.concat(result, axis=0)
    # df = df[~df.index.duplicated(keep='first')]
    top_k_dfs.append(df[['return', 'cost', 'bench']])
  
  deploy_start_index = [top_k_results[cv_task_idx][0][n_folds].index[0] for cv_task_idx in range(num_cv_tasks)]

  with_cost_cum = []
  without_cost_cum = []
  bench_cum = []
  with_cost_metrics = []
  without_cost_metrics = []

  for idx, df in enumerate(top_k_dfs):
    # Step returns
    r_wc = df['return'] - df['cost']
    r_wo = df['return']
    r_b  = df['bench']

    # Cumulative series
    cum_wc = (1 + r_wc).cumprod() - 1
    cum_wo = (1 + r_wo).cumprod() - 1
    cum_b  = (1 + r_b).cumprod() - 1

    with_cost_cum.append(cum_wc)
    without_cost_cum.append(cum_wo)
    bench_cum.append(cum_b)

    with_cost_metrics.append((
      risk_analysis(r_wc, N=steps_per_year, mode="product"),
      fit_capm(r_wc, r_b, N=steps_per_year, r_f_annual=2e-2)
    ))
    without_cost_metrics.append((
      risk_analysis(r_wo, N=steps_per_year, mode="product"),
      fit_capm(r_wo, r_b, N=steps_per_year, r_f_annual=2e-2)
    ))

  fig, axes = plt.subplots(2, 2, figsize=(12, 8))
  fig.suptitle(f"Deploy Top-{k} Results by {selection_criterion}")

  for cfg_idx, (r_cum, cfg) in enumerate(zip([with_cost_cum, without_cost_cum], ["with Cost", "without Cost"])):
    ax = axes[0, cfg_idx]
    for i, s in enumerate(r_cum):
      ax.plot(s.index, s.values, label=f"Top-{i+1}")
    ax.plot(bench_cum[0].index, bench_cum[0].values, label="Bench", linestyle='--', linewidth=1)
    ax.set_title(f"Total Return {cfg}")
    ax.set_ylabel("Total Return")
    ax.tick_params(axis='x', labelbottom=False)
    ax.grid()
    ax.legend()

    ax = axes[1, cfg_idx]
    for i, (r, b) in enumerate(zip(r_cum, bench_cum)):
      excess = r - b
      ax.plot(excess.index, excess.values, label=f"Top-{i+1}")
    ax.set_title(f"Excess Return {cfg}")
    ax.set_ylabel("Excess Return")
    ax.tick_params(axis='x', rotation=90)
    ax.grid()
    ax.legend()
  
  for ax in axes.flat:
    for d in deploy_start_index:
      ax.axvline(d, linestyle='--', linewidth=1, color='red', alpha=0.5)

  os.makedirs(save_path, exist_ok=True)
  plt.savefig(os.path.join(save_path, f"deploy_top_{k}_results_by_{selection_criterion}.png"), bbox_inches="tight", dpi=500)
  plt.close()

def main():
  data_handler = "Alpha158"
  market = "hs300" # sz50, hs300, zz500
  benchmark = "SH000300" # SH000016, SH000300, SH000905
  deal_price = "close"
  freq = "1d"

  n_tasks = 1 # number of different hyperparameter optimization tasks = num test splits
  n_folds = 0 # number of cross-validation folds, 0 means no CV
  k = 5  # top k models to select and plot

  identifier = f"tune_single_{data_handler}_{market}_{deal_price}_{freq}"

  workspace_path = "/iopsstor/scratch/cscs/ljiayong/workspace/qlib/tune/cn/lightgbm"
  storage = JournalStorage(JournalFileBackend(os.path.join(workspace_path, f"optuna/{identifier}_journal.log")))

  '''
  To Sqlite DB
  '''
  # dst_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/{identifier}_sqlite.db")
  # os.makedirs(os.path.dirname(dst_path), exist_ok=True)
  # # os.path.exists(dst_path) and os.remove(dst_path)

  # study_names = optuna.study.get_all_study_names(storage=storage)
  # for study_name in study_names:
  #   print(f"Copying study: {study_name}, {len(study_names)} studies in total ...")
  #   optuna.copy_study(
  #     from_study_name=study_name,
  #     from_storage=storage,
  #     to_storage=f"sqlite:///{dst_path}",
  #   )
  
  '''
  Load Top K Results and Plot
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

  save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/plots")
  results = [] # [task_idx][k_idx][fold_idx]
  for task_idx, df in enumerate(dfs):
    df = df.sort_values("value", ascending=False).reset_index(drop=True).head(k)
    task_results = [] # [top_k_idx][fold_idx]
    for top_idx in range(k):
      trial_results = [] # [fold_idx]
      for fold_idx in range(n_folds):
        recorder = R.get_recorder(experiment_name=f"task_{task_idx}", recorder_id=df.loc[top_idx, f"fold_{fold_idx}_rid"])
        trial_results.append({
          "pred_test": recorder.load_object("pred_test.pkl"),
          "report_test": recorder.load_object("portfolio_metric_test.pkl")["1day" if freq == "1d" else freq][0],
        })
      recorder = R.get_recorder(experiment_name=f"task_{task_idx}", recorder_id= df.loc[top_idx, f"rid"])
      trial_results.append({
        "pred_test": recorder.load_object("pred_test.pkl"),
        "report_test": recorder.load_object("portfolio_metric_test.pkl")["1day" if freq == "1d" else freq][0],
      })
      task_results.append(trial_results)
    # plot current cv task
    plot_task(task_results, task_idx, save_path)
    results.append(task_results)
  # plot_all_tasks(results, task_idx, save_path)


if __name__ == "__main__":
  main()

