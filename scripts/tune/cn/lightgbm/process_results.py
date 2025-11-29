import os
import optuna

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.contrib.evaluate import risk_analysis, fit_capm

def plot_cv_task(cv_task_top_k_results, selection_criterion, cv_task_idx, save_path):
  # [top_k_idx][fold_idx]
  k = len(cv_task_top_k_results)
  n_folds = len(cv_task_top_k_results[0]) - 1  # last one is deploy
  steps_per_year = 246

  top_k_dfs = [] # [top_k_idx]
  for result in cv_task_top_k_results:
    df = pd.concat(result, axis=0)
    # df = df[~df.index.duplicated(keep='first')]
    top_k_dfs.append(df[['return', 'cost', 'bench']])
  
  fold_start_index = [df.index[0] for df in cv_task_top_k_results[0]]

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
  fig.suptitle(f"CV Task {cv_task_idx} Top-{k} Results by {selection_criterion}")

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
    for d in fold_start_index:
      ax.axvline(d, linestyle='--', linewidth=1, color='red', alpha=0.5)

  os.makedirs(save_path, exist_ok=True)
  plt.savefig(os.path.join(save_path, f"cv_task_{cv_task_idx}_top_{k}_results_by_{selection_criterion}.png"), bbox_inches="tight", dpi=500)
  plt.close()

def plot_all_cv_task(top_k_results, selection_criterion, save_path):
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

  n_folds = 4

  qlib.init(**{
    "provider_uri": "/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock/bin",
    "region": REG_CN,
    "exp_manager": {
      "class": "MLflowExpManager",
      "module_path": "qlib.workflow.expm",
      "kwargs": {
        "uri": "file:///" + os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results/runs"),
        "default_exp_name": "default_experiment",
      },
    }
  })

  dfs = []
  for cv_task_idx in range(n_folds):
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/optuna.db")
    storage_uri = f"sqlite:///{db_path}"
    study_name = f"LightGBM_{data_handler}_{market}_{freq}_task{cv_task_idx}"
    study = optuna.load_study(study_name=study_name, storage=storage_uri)
    trials_data = []
    for trial in study.trials:
      trial_dict = {
        'trial_number': trial.number,
        'state': trial.state.name,
        'value': trial.value,
      }
      trial_dict.update(trial.params)
      trial_dict.update(trial.user_attrs)
      trials_data.append(trial_dict)
    df = pd.DataFrame(trials_data)
    df = df[df['state'] == 'COMPLETE'].dropna().reset_index(drop=True)

    # model selection based on fit task results
    df["cv_annualized_return"] = (1 + df[[f"fit_{idx}_annualized_return_test" for idx in range(n_folds)]]).prod(axis=1) ** (1 / n_folds) - 1
    df["cv_information_ratio"] = df[[f"fit_{idx}_information_ratio_test" for idx in range(n_folds)]].mean(axis=1)
    dfs.append(df)
  
  k = 5
  for selection_criterion in ["cv_annualized_return", "cv_information_ratio"]:
    top_k_results = [] # [cv_task_idx][top_k_idx][fold_idx]
    for cv_task_idx, df in enumerate(dfs):
      df = df.sort_values(selection_criterion, ascending=False).reset_index(drop=True).head(k)
      cv_task_top_k_results = []
      for top_idx in range(k):
        result = []
        for fit_task_idx in range(n_folds):
          recorder = R.get_recorder(experiment_name=study_name, recorder_id= df.loc[top_idx, f"fit_{fit_task_idx}_rid"])
          report_test = recorder.load_object(f"portfolio_metric_test.pkl")["1day" if freq == "1d" else freq][0]
          result.append(report_test)
        recorder = R.get_recorder(experiment_name=study_name, recorder_id= df.loc[top_idx, f"deploy_rid"])
        report_test = recorder.load_object(f"portfolio_metric_test.pkl")["1day" if freq == "1d" else freq][0]
        result.append(report_test)
        cv_task_top_k_results.append(result)
      # plot current cv task
      save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/plots")
      plot_cv_task(cv_task_top_k_results, selection_criterion, cv_task_idx, save_path)
      top_k_results.append(cv_task_top_k_results)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results/plots")
    plot_all_cv_task(top_k_results, selection_criterion, save_path)


if __name__ == "__main__":
  main()

