# pip install -r requirements.txt


# download from https://api.coingecko.com/api/v3/
python collector.py download_data --source_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/crypto_data/source/1d --start 2023-01-01 --end 2025-07-01 --delay 1 --interval 1d

# normalize
# python collector.py normalize_data --source_dir ~/.qlib/crypto_data/source/1d --normalize_dir ~/.qlib/crypto_data/source/1d_nor --interval 1d --date_field_name date

# dump data
# cd qlib/scripts
# python dump_bin.py dump_all --data_path ~/.qlib/crypto_data/source/1d_nor --qlib_dir ~/.qlib/qlib_data/crypto_data --freq day --date_field_name date --include_fields prices,total_volumes,market_caps
