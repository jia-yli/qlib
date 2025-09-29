cd scripts/data_collector/yahoo

pip install -r requirements.txt

# US market, daily data
python collector.py download_data --source_dir  /capstor/scratch/cscs/ljiayong/datasets/qlib/yahoo_us_data_test/source/raw --start 2023-01-01 --end 2025-07-01 --delay 1 --interval 1d --region US --limit_nums 50

python collector.py normalize_data --source_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/yahoo_us_data_test/source/raw --normalize_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/yahoo_us_data_test/source/normalized --region US --interval 1d

cd ../..
python dump_bin.py dump_all --data_path /capstor/scratch/cscs/ljiayong/datasets/qlib/yahoo_us_data_test/source/normalized --qlib_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/yahoo_us_data_test --freq day --exclude_fields date,symbol --file_suffix .csv

# CN market, daily data
python collector.py download_data --source_dir  /capstor/scratch/cscs/ljiayong/datasets/qlib/yahoo_cn_data_test/source/raw --start 2023-01-01 --end 2025-07-01 --delay 1 --interval 1d --region CN --limit_nums 200

python collector.py normalize_data --source_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/yahoo_cn_data_test/source/raw --normalize_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/yahoo_cn_data_test/source/normalized --region CN --interval 1d

cd ../..
python dump_bin.py dump_all --data_path /capstor/scratch/cscs/ljiayong/datasets/qlib/yahoo_cn_data_test/source/normalized --qlib_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/yahoo_cn_data_test --freq day --exclude_fields date,symbol --file_suffix .csv