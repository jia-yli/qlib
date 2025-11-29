def suggest_lightgbm_config(trial):
  model_config = {
    "class": "LGBModel",
    "module_path": "qlib.contrib.model.gbdt",
    "kwargs": {
      "loss": "mse",

      "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
      "num_leaves": trial.suggest_int("num_leaves", 31, 511),
      "max_depth": trial.suggest_int("max_depth", -1, 16),
      "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),

      "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
      "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
      "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),

      "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
      "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),

      # "boosting_type": trial.suggest_categorical("boosting_type", ["gbdt", "dart"]),
      # no early stopping in dart
    },
  }

  trial_config = {
    "model": model_config,
    "dataset": {
      "class": "DatasetH",
      "module_path": "qlib.data.dataset",
    },
    "data_handler": {
      "class": "Alpha158",
      "module_path": "qlib.contrib.data.handler",
    },
  }
  return trial_config