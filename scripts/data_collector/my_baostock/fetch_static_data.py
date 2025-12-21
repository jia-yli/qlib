import sys
import time
import pandas as pd
import baostock as bs
from pathlib import Path
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.utils import deco_retry

@deco_retry(retry_sleep=5, retry=2)
def get_stock_industry():
  rs = bs.query_stock_industry()
  if rs.error_code == "0":
    data_list = rs.data
    columns = rs.fields
    df = pd.DataFrame(data_list, columns=columns)
  else:
    raise ValueError(f"fetch stock industry error, error_code: {rs.error_code}, error_msg: {rs.error_msg}")
  return df

def normalize_symbol(symbol):
  return str(symbol).replace(".", "").upper()

if __name__ == "__main__":
  save_path = '/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock'
  save_path = Path(save_path)

  instrument_list_save_path = save_path / "raw" / "instrument_list"
  instrument_list_save_path.mkdir(parents=True, exist_ok=True)

  static_data_path = save_path / "raw" / "static_data"
  static_data_path.mkdir(parents=True, exist_ok=True)

  processed_static_data_path = save_path / "processed" / "static_data"
  processed_static_data_path.mkdir(parents=True, exist_ok=True)
  '''
  1. Get Static Data
  '''
  # bs.login()
  # df = get_stock_industry()
  # df.to_csv(static_data_path / "stock_industry.csv", index=False)
  # bs.logout()

  '''
  2. Process Static Data
  '''
  df = pd.read_csv(static_data_path / "stock_industry.csv")
  df.rename(columns={"code": "symbol"}, inplace=True)
  df['symbol'] = df['symbol'].apply(normalize_symbol)

  # 国民经济行业分类 GB/T 4754—2017
  # https://www.beijing.gov.cn/zhengce/zhengcefagui/202304/W020230410621028325997.pdf
  industry = df["industry"].copy() # e.g. J66货币金融服务
  df["sector"] = industry.str.extract(r"^([A-Z])", expand=False)
  df["industry"] = industry.str.extract(r"^([A-Z]\d\d)", expand=False)
  df["industry_name"] = industry.str.replace(r"^[A-Z]\d\d", "", regex=True)
  df.to_csv(processed_static_data_path / "stock_industry.csv", index=False)
