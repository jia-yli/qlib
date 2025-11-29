import os

import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def main():
  src_journal = "/iopsstor/scratch/cscs/ljiayong/workspace/qlib/tune/tests/test_parallel_journal.log"
  dst_sqlite = os.path.join(os.path.dirname(__file__), "optuna/optuna_journal.db")

  os.makedirs(os.path.dirname(dst_sqlite), exist_ok=True)
  os.path.exists(dst_sqlite) and os.remove(dst_sqlite)

  src_storage = JournalStorage(JournalFileBackend(src_journal))
  study_names = optuna.study.get_all_study_names(storage=src_storage)

  for study_name in study_names:
    print(f"Copying study: {study_name}, {len(study_names)} studies in total ...")
    optuna.copy_study(
      from_study_name=study_name,
      from_storage=src_storage,
      to_storage=f"sqlite:///{dst_sqlite}",
    )
    print(f"Converting study: {study_name} to CSV file ...")
    src_study = optuna.load_study(study_name=study_name, storage=src_storage)
    df = src_study.trials_dataframe(
      attrs=("number", "state", "value", "params", "user_attrs"),
      multi_index=True,
    )
    df.to_csv(os.path.join(os.path.dirname(dst_sqlite), f"study_{study_name}.csv"), index=False)


if __name__ == "__main__":
  main()
