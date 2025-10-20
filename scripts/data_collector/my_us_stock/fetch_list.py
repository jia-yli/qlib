import io
import re
import sys
import requests
import pandas as pd
from pathlib import Path
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.utils import deco_retry

@deco_retry(retry_sleep=5, retry=2)
def fetch_wiki(index_name):
  target_url = {
    'S&P500': 'List_of_S%26P_500_companies',
    'S&P400': 'List_of_S%26P_400_companies',
    'S&P600': 'List_of_S%26P_600_companies',
  }
  headers = {
    "User-Agent": (
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36"
    )
  }
  base_url = "https://en.wikipedia.org/wiki"
  url = f"{base_url}/{target_url[index_name]}"

  resp = requests.get(url, headers=headers)
  if resp.status_code != 200:
    raise RuntimeError(f"Failed to fetch {url}, status code {resp.status_code}")

  return {
    'index_name': index_name,
    'content': resp.text,
  }

def clean_ticker_str(ticker):
  if pd.isna(ticker):
    return []
  
  tickers = [t for t in re.split(f'[,\s/]+', ticker)]
  return tickers
  

def get_enlisted_state(tables, current_time, start_time, end_time):
  assert len(tables) == 2
  current_components, historical_changes = tables
  # flatten multi-index and get datetime of change
  historical_changes.columns = [f"{col[0]}{col[1] if col[0] != col[1] else ''}" for col in historical_changes.columns]
  import pdb;pdb.set_trace()
  historical_changes['date'] = pd.to_datetime(historical_changes.iloc[:, 0])
  historical_changes = historical_changes.sort_values(by='date', ascending=False)

  change_log_size = len(historical_changes)
  first_change_date = historical_changes['date'].iloc[change_log_size-1]
  last_change_date = historical_changes['date'].iloc[0]
  logger.info(f"Change log size: {change_log_size}, from {first_change_date} to {last_change_date}")
  if first_change_date > start_time:
    logger.warning(f"First change {first_change_date} is later than start time {start_time}")

  # symbols appeared
  assert(current_components['Symbol'].is_unique)
  current_symbols = sorted(current_components['Symbol'])
  df = pd.DataFrame({symbol: [True] for symbol in current_symbols}, index=[current_time], columns=current_symbols)

  current_resolving_time = current_time
  for change_idx in range(change_log_size):
    change_info = historical_changes.iloc[change_idx]
    change_time = change_info['date']
    # changes are resolved backward from the latest
    if change_time <= current_resolving_time:
      current_resolving_time = change_time
    else:
      raise ValueError
    
    # until start_time
    # component change happens before the open of that day
    if change_time <= start_time:
      break

    # infer prev day state
    # nothing happen if this is already last row
    change_prev_day = change_time - pd.Timedelta(1, 'day')
    earliest_record_time = df.index[-1]
    df.loc[change_prev_day] = df.iloc[len(df) - 1]

    added_tickers = clean_ticker_str(change_info['AddedTicker'])
    removed_tickers = clean_ticker_str(change_info['RemovedTicker'])

    for added_ticker in added_tickers:
      # manually resolve ticker renaming
      if change_time == pd.to_datetime('2021-09-20') and added_ticker == 'CDAY':
        added_ticker = 'DAY'

      ticker_state_curr = df.loc[earliest_record_time, added_ticker]
      assert ticker_state_curr
      df.loc[change_prev_day, added_ticker] = False
    
    for removed_ticker in removed_tickers:
      # manually resolve ticker renaming
      if change_time == pd.to_datetime('2019-03-19') and removed_ticker == 'FOXA':
        removed_ticker = 'TFCFA'
      if change_time == pd.to_datetime('2019-03-19') and removed_ticker == 'FOX':
        removed_ticker = 'TFCF'

      if removed_ticker not in df.columns:
        df[removed_ticker] = False
      ticker_state_curr = df.loc[earliest_record_time, removed_ticker]
      assert not ticker_state_curr
      # if ticker_state_curr:
      #   import pdb;pdb.set_trace()
      df.loc[change_prev_day, removed_ticker] = True

    df = df.copy()

  enlisted_state = df
  return enlisted_state


if __name__ == "__main__":
  start = "2019-01-01" # inc
  end   = "2025-10-01" # inc

  start_time = pd.Timestamp(start)
  end_time = pd.Timestamp(end)
  current_time = pd.Timestamp.now().normalize()

  save_path = '/capstor/scratch/cscs/ljiayong/datasets/qlib/my_us_stock'
  save_path = Path(save_path)

  index_lst = ['S&P500', 'S&P400', 'S&P600']
  index_symbols = ['^GSPC', '^SP400', '^SP600']

  instrument_list_save_path = save_path / "raw" / "instrument_list"
  instrument_list_save_path.mkdir(parents=True, exist_ok=True)

  raw_kline_path = save_path / "raw"
  raw_kline_path.mkdir(parents=True, exist_ok=True)

  '''
  1. Fetch Raw Instrument List Info
  '''
  # results = []
  # for index_name in index_lst:
  #   result = fetch_wiki(index_name)
  #   results.append(result)
  # df = pd.DataFrame(results)
  # df.to_feather(instrument_list_save_path / "raw.feather")

  '''
  2. Process Instrument List
  '''
  df = pd.read_feather(instrument_list_save_path / "raw.feather")
  for index_name in index_lst:
    raw_page_content = df.loc[df['index_name']==index_name, 'content'].iloc[0]
    tables = pd.read_html(io.StringIO(raw_page_content))
    enlisted_state = get_enlisted_state(tables[0:2], current_time, start_time, end_time)
    import pdb;pdb.set_trace()
    


  '''
  3. Fetch Klines for Symbols
  '''

