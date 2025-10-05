cd qlib/scripts/data_collector/my_baostock

pip install -r requirements.txt

python collector.py download_data --source_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/cn_my_baostock/src --start 2020-01-01 --end 2025-10-01 --interval 1d
python dump_bin.py dump_all --data_path /capstor/scratch/cscs/ljiayong/datasets/qlib/cn_my_baostock/src --qlib_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/cn_my_baostock/bin --freq day --exclude_fields date,symbol
