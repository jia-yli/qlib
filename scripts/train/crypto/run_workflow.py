import os
import pandas as pd
from datetime import datetime, timezone

import qlib
from qlib.constant import REG_CRYPTO
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.contrib.report import analysis_position

def main(freq, _freq):
    # config
    deal_price = "open"
    provider_uri = f"/capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto/bin/{_freq}"
    market = "my_universe"
    benchmark = "BTCUSDT"
    train_split = ("2021-01-01", "2022-12-31")
    valid_split = ("2023-01-01", "2023-12-31")
    test_split  = ("2024-01-01", "2025-09-29")


    qlib.init(provider_uri=provider_uri, region=REG_CRYPTO)
    utc_timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    mlrun_path = f"/capstor/scratch/cscs/ljiayong/workspace/qlib/mlruns/{utc_timestamp}"
    mlrun_uri = f"file://{mlrun_path}"
    R.set_uri(mlrun_uri)

    data_handler_config = {
        "start_time": train_split[0],
        "end_time": test_split[1],
        "fit_start_time": train_split[0],
        "fit_end_time": train_split[1],
        "instruments": market,
        "freq": freq,
        "label": [[f"Ref(${deal_price}, -2)/Ref(${deal_price}, -1) - 1"], ["LABEL0"]]
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
                "time_per_step": freq,
                "generate_portfolio_metrics": True,
            },
        },
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy.signal_strategy",
            "kwargs": {
                "signal": (model, dataset),
                "topk": 20,
                "n_drop": 2,
            },
        },
        "backtest": {
            "start_time": test_split[0],
            "end_time": test_split[1],
            "account": 10_000,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": freq,
                "trade_unit": None,
                "limit_threshold": None,
                "deal_price": deal_price,
                "open_cost": 0.001,
                "close_cost": 0.001,
                "min_cost": 0,
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
        N = 365 * int(pd.to_timedelta("1day") / pd.to_timedelta(_freq))
        par = PortAnaRecord(recorder, config=port_analysis_config, N=N, risk_analysis_freq=freq)
        par.generate()

    recorder = R.get_recorder(recorder_id=rid, experiment_name="workflow")

    pred_df = recorder.load_object("pred.pkl")
    report_normal_df = recorder.load_object(f"portfolio_analysis/report_normal_{_freq}.pkl")
    positions = recorder.load_object(f"portfolio_analysis/positions_normal_{_freq}.pkl")
    analysis_df = recorder.load_object(f"portfolio_analysis/port_analysis_{_freq}.pkl")

    analysis_position.report_graph(report_normal_df, show_notebook=False, save_path=f"./scripts/train/crypto/{_freq}")
    # import pdb;pdb.set_trace()
    # sorted(model.predict(dataset).xs(pd.Timestamp('2024-07-23 00:00:00'), level=0).sort_values(ascending=False).iloc[:20].index)

if __name__ == "__main__":
    freq_lst = ["day", "720min", "240min", "60min", "30min", "15min"]
    _freq_lst = ["1day", "720min", "240min", "60min", "30min", "15min"]
    for freq, _freq in zip(freq_lst, _freq_lst):
        main(freq, _freq)