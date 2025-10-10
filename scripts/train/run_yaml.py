import os
import fire
from pathlib import Path
from ruamel.yaml import YAML
from jinja2 import Template

import qlib
from qlib.config import C
from qlib.model.trainer import task_train

def run(experiment_name="workflow", uri_folder="mlruns"):
    '''
    Alpha158:
    1. DoubleEnsemble: 0.025
    2. LightGBM: 0.085, -0.004(5), -0.015(10)
    3. MLP: not run
    4. TFT (selected 20): need TF
    5. XGBoost
    6. CatBoost
    7. TRA (selected 20): 0.008
    8. TRA: not run
    8. Linear
    9. GATs (selected 20): 0.10
    Alpha360
    1. HIST: not run
    2. IGMTF: not run
    3. TRA
    4. TCTS: failed and start retraining
    5. GATs: 0.025
    6. ADARNN: 0.014
    7. GRU
    8. ADD
    9. LSTM
    10. ALSTM
    11. TCN
    12. LightGBM
    13. Double Ensemble
    '''
    model_name = 'LightGBM'
    dataset = 'Alpha158'
    config_path = f"/users/ljiayong/projects/qlib/scripts/train/benchmarks/{model_name}/workflow_config_{model_name.lower()}_{dataset}_hs300.yaml"

    base_config = {
        "market":      "hs300",
        "benchmark":   "SH000300",
        "train_start": "2020-01-01",
        "train_end":   "2022-12-31",
        "valid_start": "2023-01-01",
        "valid_end":   "2023-12-31",
        "test_start":  "2024-01-01",
        "test_end":    "2025-09-29",
    }

    label = ["Ref($close, -2) / Ref($close, -1) - 1"]

    # load and run
    with open(config_path, "r") as f:
        template = Template(f.read())
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(template.render(base_config))

    qlib.init(
        provider_uri = "/capstor/scratch/cscs/ljiayong/datasets/qlib/cn_my_baostock/bin",
        region = 'cn',
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "file://" + str(Path(os.getcwd()).resolve() / uri_folder),
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
