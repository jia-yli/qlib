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
def get_quarter_data(symbol, start_year, end_year, data_type="profit"):
  dfs = []
  for year in range(start_year, end_year + 1):
    for quarter in range(1, 5):
      time.sleep(1)  # avoid query too fast
      logger.info(f"Fetching {symbol} {year} Q{quarter} {data_type} data ......")
      rs = eval(f"bs.query_{data_type}_data")(code=symbol, year=year, quarter=quarter)
      if rs.error_code == "0":
        data_list = rs.data
        columns = rs.fields
        df = pd.DataFrame(data_list, columns=columns)
      else:
        raise ValueError(f"fetch {data_type} data error for {symbol} {year} Q{quarter}, error_code: {rs.error_code}, error_msg: {rs.error_msg}")
      df["year"] = year
      df["quarter"] = quarter
      dfs.append(df)
  
  return pd.concat(dfs, ignore_index=True)

def normalize_symbol(symbol):
  return str(symbol).replace(".", "").upper()

if __name__ == "__main__":
  save_path = '/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock'
  save_path = Path(save_path)

  # index_lst = ['sz50', 'hs300', 'zz500']
  # index_symbols = ['sh.000016', 'sh.000300', 'sh.000905']  # sz50, hs300, zz500

  index_lst = ['hs300']

  instrument_list_save_path = save_path / "raw" / "instrument_list"
  instrument_list_save_path.mkdir(parents=True, exist_ok=True)

  quarter_data_path = save_path / "quarter_data"
  quarter_data_path.mkdir(parents=True, exist_ok=True)

  '''
  1. Get Index Components
  '''
  symbols = set()
  for index in index_lst:
    df = pd.read_feather(instrument_list_save_path / f"{index}_raw.feather")
    symbols.update(df['code'].unique())

  all_symbols = sorted(symbols)

  '''
  3. Fetch Quarter Data for Symbols
  '''
  logger.info(f"Total symbols to fetch quarter data: {len(all_symbols)}")
  bs.login()
  for symbol in all_symbols:
    df = get_quarter_data(symbol, 2019, 2025)
    df.to_csv(quarter_data_path / f"{symbol}.csv", index=False)
  bs.logout()  