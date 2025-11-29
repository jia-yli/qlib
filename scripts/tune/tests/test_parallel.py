import os
import random
import functools

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

def dummy_objective(
  trial,
  num_float_params=20,
  num_int_params=20,
  num_user_attrs=40,
):
  
  for i in range(num_float_params):
    trial.suggest_float(f"float_param_{i:03d}", 0, 1)
  
  for i in range(num_int_params):
    trial.suggest_int(f"int_param_{i:03d}", 0, 100)

  target = random.random()

  for i in range(num_user_attrs):
    trial.set_user_attr(f"user_attr_{i:03d}", random.random())
  
  return target

def main():
  n_runs = 4
  n_studies = 4
  n_trials = 4

  workspace_path = "/iopsstor/scratch/cscs/ljiayong/workspace/qlib/tune/tests"
  storage_path = os.path.join(workspace_path, "test_parallel_journal.log")
  os.makedirs(os.path.dirname(storage_path), exist_ok=True)
  storage = JournalStorage(JournalFileBackend(storage_path))

  for _ in range(n_runs):
    for study_idx in range(n_studies):
      study_name = f"study_{study_idx}"
      study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
      )
      study.optimize(dummy_objective, n_trials=n_trials, n_jobs=1)

if __name__ == "__main__":
  main()