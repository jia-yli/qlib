import re
import os
import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path
from loguru import logger
from bs4 import BeautifulSoup
from joblib import Parallel, delayed

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.utils import deco_retry

@deco_retry(retry_sleep=5, retry=2)
def fetch_page(url, session=None):
  if session is None:
    resp = requests.get(url)
  else:
    resp = session.get(url)
  # resp.raise_for_status()
  if resp.status_code != 200:
    raise RuntimeError(f"Failed to fetch {url}, status code {resp.status_code}")

  actual_url = resp.url
  m = re.search(r"/web/(\d{14})/", actual_url)
  assert m
  actual_timestamp = pd.Timestamp(m.group(1), tz="UTC")
  content = resp.text

  return {
    'timestamp': actual_timestamp,
    'url': actual_url,
    'content': content,
  }

def parse_page(html_str):
  soup = BeautifulSoup(html_str, "html.parser")
  table = soup.find("table")
  symbols = [e.get_text(strip=True) for e in table.select("p.coin-item-symbol, span.crypto-symbol")]
  # num_symbols = len(symbols)
  # num_rows = len(table.find_all("tr"))
  return symbols

def fetch_all_pages(start_time, end_time, step, save_path):
  save_path.mkdir(parents=True, exist_ok=True)
  if (save_path / "raw.feather").exists():
    df = pd.read_feather(save_path / "raw.feather")
  else:
    df = pd.DataFrame()
  
  seen_comb = set(zip(df.get('fetch_timestamp', []), df.get('page_id', [])))

  results = []
  session = requests.Session()
  base_url = "https://web.archive.org/web/"
  target_url = "https://coinmarketcap.com/"
  current_time = start_time
  while current_time <= end_time:
    time_stamp = current_time.strftime("%Y%m%d%H%M%S")
    url_lst = [f"{base_url}{time_stamp}/{target_url}/"] + [f"{base_url}{time_stamp}/{target_url}/?page={n}" for n in range(2, 7)] # first 6 pages

    for page_id, url in enumerate(url_lst):
      if (current_time, page_id) in seen_comb:
        logger.info(f"Page {page_id} for {current_time} already seen")
        continue
      logger.info(f"Getting page {page_id} for {current_time} ......")
      try:
        result = fetch_page(url, session)
        result['fetch_timestamp'] = current_time
        result['page_id'] = page_id
        result['symbols'] = parse_page(result['content'])
        results.append(result)
        seen_comb.add((current_time, page_id))
      except Exception as e:
        logger.warning(f"Fetch failed for page {page_id} at {current_time}: {e}")
    current_time += step
  
  if results:
    df_new = pd.DataFrame(results)
    df = pd.concat([df, df_new], ignore_index=True)
    df = df.sort_values(["fetch_timestamp", "page_id"])
    df = df.drop_duplicates(subset=["fetch_timestamp", "page_id"])
    df.to_feather(save_path / "raw.feather")
  return df

def get_symbols_from_raw(df):
  df = df[['timestamp', 'page_id', 'symbols']].copy()
  df["month_start"] = (df["timestamp"] + pd.offsets.MonthBegin(1)).dt.normalize()
  df = df[(df["month_start"] >= start_time) & (df["month_start"] <= end_time)]
  df = (
    df[["month_start", "symbols"]]
    .groupby("month_start", as_index=False)
    .agg(lambda s: sorted(set(s.explode().dropna())))
    .set_index("month_start")
  )
  # df = pd.pivot_table(
  #   df, 
  #   values='symbols', 
  #   index='month_start', 
  #   columns='page_id', 
  #   aggfunc=lambda s: sorted(set(s.explode().dropna()))
  # )
  df = df.reindex(pd.date_range(start_time, end_time, freq='MS')).ffill()
  df['symbols'] = df['symbols'].apply(lambda x: x if isinstance(x, list) else [])

  symbol_enlisted_time = {}
  for month_start, symbols in df['symbols'].items():
    for symbol in symbols:
      if symbol not in symbol_enlisted_time.keys():
        symbol_enlisted_time[symbol] = month_start
  symbol_enlisted_time = pd.DataFrame(
    list(symbol_enlisted_time.items()), 
    columns=['symbol', 'enlisted_time'],
  )
  return symbol_enlisted_time

@deco_retry(retry_sleep=5, retry=2)
def _get_kline_data(symbol, interval, start_time, end_time, limit):
  url = 'https://api.binance.com/api/v3/klines'
  start = int(start_time.timestamp()) * 1000
  end = int(end_time.timestamp()) * 1000 - 1 # extra 1 bar will be fetched if no -1, because end is incl
  assert limit <= 1000

  params = {
    'symbol': symbol,
    'interval': interval,
    'startTime': start,
    'endTime': end,
    'limit': limit
  }

  response = requests.get(url, params=params)
  if response.status_code == 200:
    # OK, if no kline, '[]' returns
    return response.json()
  else:
    # Not OK
    logger.warning(f"{symbol} {start_time} to {end_time} {interval} data fetch failed: {response.status_code}")
    if response.status_code == 400:
      if response.json()['code'] == -1121: # {"code":-1121,"msg":"Invalid symbol."}
        # invalid symbol -> not exists
        logger.warning(f"{symbol} not exists")
        return [] # no kline
    if response.status_code in [418, 429]:
      delay = int(response.headers.get("Retry-After", 1))
      logger.warning(f"retry after {delay}s")
      time.sleep(delay)
    raise RuntimeError(f"Network error when fetching data: {response.status_code}")

def get_kline_data(symbol, interval, start_time, end_time):
  # fetch data from start to (end - 1ms)
  fetch_interval_start_time = start_time
  fetch_interval_end_time = end_time
  step = pd.to_timedelta(interval)
  limit = 1000

  all_data = []
  while fetch_interval_start_time < end_time:
    if fetch_interval_end_time > (fetch_interval_start_time + limit * step):
      fetch_interval_end_time = fetch_interval_start_time + limit * step
    # Call your data retrieval function
    logger.info(f"Fetching {symbol} {fetch_interval_start_time} to {fetch_interval_end_time} {interval} data")
    data_segment = _get_kline_data(symbol, interval, fetch_interval_start_time, fetch_interval_end_time, limit)
    all_data.extend(data_segment)
    # Move to the next interval
    fetch_interval_start_time = fetch_interval_end_time
    fetch_interval_end_time = end_time
  
  # Convert to DataFrame for easier handling and storage
  # example response
  # [
  #     [
  #         1499040000000,      // Kline open time
  #         "0.01634790",       // Open price
  #         "0.80000000",       // High price
  #         "0.01575800",       // Low price
  #         "0.01577100",       // Close price
  #         "148976.11427815",  // Volume
  #         1499644799999,      // Kline Close time
  #         "2434.19055334",    // Quote asset volume
  #         308,                // Number of trades
  #         "1756.87402397",    // Taker buy base asset volume
  #         "28.46694368",      // Taker buy quote asset volume
  #         "0"                 // Unused field, ignore.
  #     ]
  # ]
  df = pd.DataFrame(
    all_data, 
    columns=[
      'open_time', 
      'open_price', 
      'high_price', 
      'low_price', 
      'close_price', 
      'base_volume', 
      'close_time', 
      'quote_volume', 
      'num_trades', 
      'taker_base_volume', 
      'taker_quote_volume', 
      'ignore'
    ]
  )
  return df


# def fetch_data(symbol_enlisted_time, interval, start_time, end_time, save_path):
  # df, start_time, end_time = BinanceData.get_kline_data_1m(symbol, start_time, end_time)
  # ts_name = f"{symbol}_{start_time.strftime('%Y%m%d_%H%M%S')}_{start_time.tzinfo}_{end_time.strftime('%Y%m%d_%H%M%S')}_{end_time.tzinfo}"
  # save_path = os.path.join(dataset_base_path, 'raw')
  # os.makedirs(save_path, exist_ok=True)
  # output_file = os.path.join(save_path, f'{ts_name}.csv')
  # df.to_csv(output_file, index=False)
  # print(f"[SUCCESS] Symbol {symbol} data from {start_time} to {end_time} is fetched, with shape {df.shape}, saved to {output_file}")
  # for symbol
  # [get_kline_data() for in]
  # res = Parallel(n_jobs=4)(
  #   delayed(get_kline_data)(
  #     symbol,
  #     interval, 
  #     start_time,
  #     end_time)
  #     for _inst in tqdm(instrument_list)
  # )


if __name__ == "__main__":
  start = "2021-01-01"
  end   = "2025-10-01"

  start_time = pd.Timestamp(start, tz="UTC").normalize()
  end_time = pd.Timestamp(end, tz="UTC").normalize()
  step = pd.Timedelta(days=1)

  save_path = '/capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto'
  save_path = Path(save_path)

  instrument_list_save_path = save_path / "raw" / "instrument_list"
  instrument_list_save_path.mkdir(parents=True, exist_ok=True)

  '''
  1. Fetch Raw Instrument List Info
  '''

  # df = fetch_all_pages(start_time, end_time, step, instrument_list_save_path)

  '''
  2. Process Instrument List
  '''
  df = pd.read_feather(instrument_list_save_path / "raw.feather")
  symbol_enlisted_time = get_symbols_from_raw(df)

  '''
  3. Fetch Daily KLine
  '''
  for symbol, enlisted_time in zip(symbol_enlisted_time['symbol'], symbol_enlisted_time['enlisted_time']):
    df = get_kline_data(symbol+"USDT", '1d', enlisted_time, pd.Timestamp("2021-02-05", tz="UTC").normalize())
    # TODO
    # 1. pair not exist
    # 2. pair no data in period
    # 3. filter stable coin
  

  # pd.pivot_table(df, values
  # all_month_start = pd.date_range(start_time, end_time, freq='MS')
  # all_page_id = pd.Index(df["page_id"].unique(), name="page_id")
  # all_index = pd.Multi
  

  # df['symbols'] = df['content'].apply(parse_page)

  # open("1.html", "w").write(df.loc[1, "content"])
  # # snaps = list_snapshots_between(url, start, end)
  # url = "https://web.archive.org/web/20240216024121/https://coinmarketcap.com/"
  # url = "https://web.archive.org/web/20250810032118/https://coinmarketcap.com/?page=6"
  # r = requests.get(url)
  # r.url

  # import pdb; pdb.set_trace()
  # print(json.dumps(snaps, indent=2))
