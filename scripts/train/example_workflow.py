import os
import pandas as pd
from datetime import datetime, timezone

import qlib
from qlib.constant import REG_CN
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.contrib.report import analysis_model, analysis_position

def save_figs(figs, path, stem):
    """Save a list/tuple of Plotly figures to HTML (and optionally PNG)."""
    # if not isinstance(figs, (list, tuple)):
    #     figs = [figs]
    for i, fig in enumerate(figs):
        html_path = os.path.join(path, f"{stem}_{i}.html")
        fig.write_html(html_path, include_plotlyjs="cdn")

        # png_path = os.path.join(path, f"{stem}_{i}.png")
        # fig.write_image(png_path, scale=2)

if __name__ == "__main__":
    # config
    provider_uri = "/capstor/scratch/cscs/ljiayong/datasets/qlib/baostock_cn_data_1day_test/bin"  # target_dir
    market = "all" # "all", "csi300"
    benchmark = "SH600000" # "SH600000", "SH000300"
    train_split = ("2024-01-01", "2024-11-30")
    valid_split = ("2024-12-01", "2025-02-28")
    test_split  = ("2025-03-01", "2025-08-31")


    # provider_uri = "/capstor/scratch/cscs/ljiayong/datasets/qlib/cn_data"  # target_dir
    # market = "all"
    # benchmark = "SH600000"
    # train_split = ("2015-01-01", "2015-12-31")
    # valid_split = ("2016-01-01", "2016-12-31")
    # test_split  = ("2017-01-01", "2017-06-30")


    qlib.init(provider_uri=provider_uri, region=REG_CN)
    utc_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    mlrun_path = f"/capstor/scratch/cscs/ljiayong/workspace/qlib/mlruns/{utc_timestamp}"
    mlrun_uri = f"file:///{mlrun_path}"
    R.set_uri(mlrun_uri)

    data_handler_config = {
        "start_time": train_split[0],
        "end_time": test_split[1],
        "fit_start_time": train_split[0],
        "fit_end_time": train_split[1],
        "instruments": market,
    }

    task = {
        "model": {
            "class": "LGBModel",
            "module_path": "qlib.contrib.model.gbdt",
            "kwargs": {
                "loss": "mse",
                "colsample_bytree": 0.8879,
                "learning_rate": 0.0421,
                "subsample": 0.8789,
                "lambda_l1": 205.6999,
                "lambda_l2": 580.9768,
                "max_depth": 8,
                "num_leaves": 210,
                "num_threads": 20,
            },
        },
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": {
                    "class": "Alpha158",
                    "module_path": "qlib.contrib.data.handler",
                    "kwargs": data_handler_config,
                },
                "segments": {
                    "train": train_split,
                    "valid": valid_split,
                    "test" : test_split ,
                },
            },
        },
    }

    model = init_instance_by_config(task["model"])
    dataset = init_instance_by_config(task["dataset"])

    port_analysis_config = {
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 50,
                "n_drop": 5,
            },
        },
        "backtest": {
            "start_time": test_split[0],
            "end_time": test_split[1],
            "account": 1_000_000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "codes": market,
                "trade_unit": 100,
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    # start exp
    with R.start(experiment_name="workflow"):
        rid = R.get_recorder().id
        # model training
        R.log_params(**flatten_dict(task))
        model.fit(dataset)
        R.save_objects(**{"model.pkl": model})

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # Signal Analysis
        sar = SigAnaRecord(recorder)
        sar.generate()

        # backtest & analysis
        par = PortAnaRecord(recorder, port_analysis_config, "day")
        par.generate()

    recorder = R.get_recorder(recorder_id=rid, experiment_name="workflow")

    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    positions = recorder.load_object("portfolio_analysis/positions_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
    save_figs(figs, mlrun_path, "report_normal")

    figs = analysis_position.risk_analysis_graph(analysis_df, report_normal_df, show_notebook=False)
    save_figs(figs, mlrun_path, "risk_analysis")