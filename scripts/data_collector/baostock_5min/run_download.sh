cd qlib/scripts/data_collector/baostock_5min

pip install -r requirements.txt

python collector.py download_data --source_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/baostock_cn_data_5min_test/raw --start 2025-01-01 --end 2025-01-30 --interval 5min --region HS300
python dump_bin.py dump_all --data_path /capstor/scratch/cscs/ljiayong/datasets/qlib/baostock_cn_data_5min_test/raw --qlib_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/baostock_cn_data_5min_test/bin --freq 5min --exclude_fields date,symbol

python collector.py download_data --source_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/baostock_cn_data_1day_test/raw --start 2024-01-01 --end 2025-09-01 --interval 1d --region HS300
python dump_bin.py dump_all --data_path /capstor/scratch/cscs/ljiayong/datasets/qlib/baostock_cn_data_1day_test/raw --qlib_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/baostock_cn_data_1day_test/bin --freq day --exclude_fields date,symbol
