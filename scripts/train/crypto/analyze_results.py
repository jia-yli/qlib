import os
import fire
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import qlib
from qlib.constant import REG_CRYPTO
from qlib.workflow import R
from qlib.contrib.report import analysis_position

def main(experiment_name="workflow"):
  data_handler = "Alpha158"
  market = "my_universe_top50"
  freq = "240min"

  output_path = Path("/users/ljiayong/projects/qlib/scripts/train/crypto") / f"{market}_{data_handler}_test_run_results" / f"{freq}"
  output_path.mkdir(parents=True, exist_ok=True)
 
  # Initialize qlib
  qlib.init(
    provider_uri = f"/capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto/bin/{freq}",
    region = REG_CRYPTO,
    exp_manager={
      "class": "MLflowExpManager",
      "module_path": "qlib.workflow.expm",
      "kwargs": {
        "uri": "file:///" + str(Path("/users/ljiayong/projects/qlib/scripts/train/crypto") / f"{market}_{data_handler}_test_run" / f"{freq}"),
        "default_exp_name": "default_experiment",
      },
    }
  )

  # Get experiment by name
  exp = R.get_exp(experiment_name=experiment_name, create=False)
  recorders = exp.list_recorders()
  
  # Collect all metrics
  all_results = []
  plots_dir = output_path / "plots"
  plots_dir.mkdir(exist_ok=True)
  
  print(f"Processing experiment: {experiment_name}")
  
  for recorder_id, recorder_info in recorders.items():
    print(f"Processing recorder: {recorder_id}")
    
    recorder = R.get_recorder(recorder_id=recorder_id, experiment_name=experiment_name)
    
    if recorder.status != "FINISHED":
      print(f"Skipping - Status: {recorder.status}")
      continue
    
    # Extract metrics
    metrics = recorder.list_metrics()
    artifacts = recorder.list_artifacts()

    # Get model info
    config = recorder.load_object("config")
    model_class = config["task"]["model"]["class"]

    # Build results row
    result = {
      "model": model_class,
      "experiment": experiment_name,
      "recorder_id": recorder_id,
      "status": recorder.status
    }

    metrics_to_collect = [
      "ic", "icir", "rank_ic", "rank_icir", 
      "annualized_return", "excess_annualized_return", "information_ratio", "max_drawdown"
    ]
    print(f"Metrics available: {metrics.keys()}")
    print(f"Using metrics for results: {metrics_to_collect}")

    result.update({
      metric: metrics[metric] for metric in metrics_to_collect
    })

    all_results.append(result)

    # Generate plots if portfolio analysis exists
    report_key = f"portfolio_analysis/report_normal_{'1day' if freq == '1d' else freq}.pkl"
    report_df = recorder.load_object(report_key)
    
    # Generate plots
    analysis_position.report_graph(
      report_df,
      show_notebook=False,
      save_path=str(plots_dir / f"{experiment_name}_{model_class}_{recorder_id}")
    )

  df = pd.DataFrame(all_results)
  print(f"Collected results from {len(df)} runs")

  summary = df.groupby('model')[metrics_to_collect].agg(['mean', 'std', 'count'])
  summary.columns = ['_'.join(col).strip() for col in summary.columns]
  summary = summary.reset_index()
  
  df.to_csv(output_path / f"results.csv", index=False)
  summary.to_csv(output_path / f"summary.csv", index=False)

  print(f"Results saved to {output_path}")
  print("Results:")
  print(df)
  print("\nSummary:")
  print(summary)

if __name__ == "__main__":
  fire.Fire(main)