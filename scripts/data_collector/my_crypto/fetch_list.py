import re
import os
import sys
import json
import time
import requests
import numpy as np
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
  while current_time < end_time:
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

def get_symbols_from_raw(df, start_time, end_time):
  df = df[['timestamp', 'page_id', 'symbols']].copy()
  df["month_start"] = (df["timestamp"] + pd.offsets.MonthBegin(1)).dt.normalize()
  df = df[(df["month_start"] >= start_time) & (df["month_start"] < end_time)]
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
  df = df.reindex(
    pd.date_range(start_time, end_time, freq='MS', inclusive="left")
  ).ffill()
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
      if response.json()['code'] == -1121:
        # {"code":-1121,"msg":"Invalid symbol."}
        logger.warning(f"{symbol} not exists")
        return [] # no kline
      elif response.json()['code'] == -1100: 
        # {"code":-1100,"msg":"Illegal characters found in parameter 'symbol'; legal range is '^[A-Z0-9-_.]{1,20}$'."}
        logger.warning(f"{symbol} not exists")
        return []
      else:
        logger.warning(f"{symbol}: {response.text}")
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

def process_raw_kline(df, freq):
  # Reformat the raw data: proper index and throw unused cols
  df['date'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
  start_time = df['date'].min()
  end_time   = df['date'].max()
  df = df.set_index('date').reindex(
    pd.date_range(start=start_time, end=end_time, freq=freq)
  )
  '''
  Remaining columns: 
  open_price
  high_price
  low_price
  close_price
  base_volume
  quote_volume
  num_trades
  taker_base_volume
  taker_quote_volume
  '''
  df = df.drop(['open_time', 'close_time', 'ignore'], axis=1)
  # fill missing
  df['close_price'] = df['close_price'].ffill()
  # fill ohl with last close
  df['open_price'] = df['open_price'].combine_first(df['close_price'])
  df['high_price'] = df['high_price'].combine_first(df['close_price'])
  df['low_price']  = df['low_price'] .combine_first(df['close_price'])
  # fill volume with 0
  df['base_volume'] = df['base_volume'].fillna(0)
  df['quote_volume'] = df['quote_volume'].fillna(0)
  df['num_trades'] = df['num_trades'].fillna(0)
  df['taker_base_volume'] = df['taker_base_volume'].fillna(0)
  df['taker_quote_volume'] = df['taker_quote_volume'].fillna(0)

  return df

def resample_data(df, freq):
  agg_dict = {
    'open_price': 'first',
    'high_price': 'max',
    'low_price': 'min',
    'close_price': 'last',
    'base_volume': 'sum',
    'quote_volume': 'sum',
    'num_trades': 'sum',
    'taker_base_volume': 'sum',
    'taker_quote_volume': 'sum',
  }
  
  return df.resample(freq).agg(agg_dict)

def get_enlisted_state(enter_condition, exit_condition):
  # a saturate counter between [0, 1] for traking state
  # 0 is outside and 1 is inside
  enter_condition = enter_condition.astype(float).fillna(0).astype(int) # don't incl if not valid
  exit_condition  = exit_condition .astype(float).fillna(1).astype(int) # kick out if not valid

  assert enter_condition.index.equals(exit_condition.index) and enter_condition.columns.equals(exit_condition.columns)

  delta = enter_condition.values - exit_condition.values

  T, N = delta.shape
  enlisted_state = np.zeros((T, N), dtype=bool)
  s = np.full(N, 0, dtype=bool)
  for t_idx in range(T):
    d = delta[t_idx] # [N] delta for all assets
    enter_ = d == 1
    exit_ = d == -1
    s[enter_] = True
    s[exit_] = False
    enlisted_state[t_idx] = s
  
  return pd.DataFrame(
    enlisted_state, 
    index = enter_condition.index, 
    columns = enter_condition.columns,
  )

def extract_enlisted_regions(enlisted_state):
  '''
  extract boundaries of consecutive regions
  '''
  edges = enlisted_state.ne(enlisted_state.shift(1, fill_value=False))
  region_id = edges.cumsum() # new region after edge: id + 1
  # now only need some fancy groupby to get first and last in the region
  # drop outside id
  enlisted_region_info = (
    region_id.where(enlisted_state)
    .reset_index(names="date")
    .melt(
      id_vars = "date",
      var_name = "symbol",
      value_name = "region_id",
    )
  ).dropna(subset=["region_id"])
  # agg
  enlisted_regions = (
    enlisted_region_info.groupby(["symbol", "region_id"])["date"]
    .agg(
      enlisted_region_start = "min",
      enlisted_region_end_incl = "max",
    )
    .reset_index()
    .drop(columns="region_id")
    .sort_values(["enlisted_region_start", "symbol"])
    .reset_index(drop=True)
  )
  return enlisted_regions

def extract_enlisted_state(enlisted_regions, date_range, symbols):
  enlisted_state = pd.DataFrame(False, index=date_range, columns=symbols)
  for _, row in enlisted_regions.iterrows():
    symbol = row['symbol']
    start = row['enlisted_region_start']
    end_incl = row['enlisted_region_end_incl']
    enlisted_state.loc[start:end_incl, symbol] = True
  return enlisted_state

def synthesize_custom_index(klines, weight, date_range):
  assert weight.index[0] == date_range[0] and weight.index[-1] <= date_range[-1]
  # normalize
  weight = weight.fillna(0)
  weight = weight.loc[:, (weight > 0).any()]
  weight = weight.div(weight.sum(axis=1), axis=0).fillna(0)

  # results
  open_price_mult_lst    = []
  high_price_mult_lst    = []
  low_price_mult_lst     = []
  close_price_mult_lst   = []
  vwap_mult_lst          = []
  quote_volume_lst       = []
  num_trades_lst         = []
  taker_vwap_mult_lst    = []
  taker_quote_volume_lst = []
  # kline
  for trade_symbol in weight.columns:
    kline = klines[trade_symbol]
    kline = kline.reindex(date_range)

    # ffill close
    kline['close_price'] = kline['close_price'].ffill()
    # fill ohl with last close
    kline['open_price'] = kline['open_price'].combine_first(kline['close_price'])
    kline['high_price'] = kline['high_price'].combine_first(kline['close_price'])
    kline['low_price']  = kline['low_price'] .combine_first(kline['close_price'])

    # bfill open
    kline['open_price'] = kline['open_price'].bfill()
    # fill hlc with first open
    kline['high_price']  = kline['high_price'] .combine_first(kline['open_price'])
    kline['low_price']   = kline['low_price']  .combine_first(kline['open_price'])
    kline['close_price'] = kline['close_price'].combine_first(kline['open_price'])

    # fill volume with 0
    kline['base_volume']        = kline['base_volume'].fillna(0)
    kline['quote_volume']       = kline['quote_volume'].fillna(0)
    kline['num_trades']         = kline['num_trades'].fillna(0)
    kline['taker_base_volume']  = kline['taker_base_volume'].fillna(0)
    kline['taker_quote_volume'] = kline['taker_quote_volume'].fillna(0)

    kline['vwap'] = (kline['quote_volume'] / kline['base_volume']).combine_first(kline['open_price'])
    kline['taker_vwap'] = (kline['taker_quote_volume'] / kline['taker_base_volume']).combine_first(kline['open_price'])
    # previous close to compute change
    kline['previous_close'] = kline['close_price'].shift(1)
    # no previous close if prev is padded
    kline.loc[kline['quote_volume'].shift(1, fill_value=0) <= 0, 'previous_close'] = np.nan

    kline['open_price_mult' ] = (kline['open_price']  / kline['previous_close'] - 1).fillna(0)
    kline['high_price_mult' ] = (kline['high_price']  / kline['previous_close'] - 1).fillna(0)
    kline['low_price_mult'  ] = (kline['low_price']   / kline['previous_close'] - 1).fillna(0)
    kline['close_price_mult'] = (kline['close_price'] / kline['previous_close'] - 1).fillna(0)
    kline['vwap_mult'       ] = (kline['vwap']        / kline['previous_close'] - 1).fillna(0)
    kline['taker_vwap_mult' ] = (kline['taker_vwap']  / kline['previous_close'] - 1).fillna(0)

    open_price_mult_lst   .append(kline['open_price_mult'   ].rename(trade_symbol))
    high_price_mult_lst   .append(kline['high_price_mult'   ].rename(trade_symbol))
    low_price_mult_lst    .append(kline['low_price_mult'    ].rename(trade_symbol))
    close_price_mult_lst  .append(kline['close_price_mult'  ].rename(trade_symbol))
    vwap_mult_lst         .append(kline['vwap_mult'         ].rename(trade_symbol))
    quote_volume_lst      .append(kline['quote_volume'      ].rename(trade_symbol))
    num_trades_lst        .append(kline['num_trades'        ].rename(trade_symbol))
    taker_vwap_mult_lst   .append(kline['taker_vwap_mult'   ].rename(trade_symbol))
    taker_quote_volume_lst.append(kline['taker_quote_volume'].rename(trade_symbol))

  open_price_mult    = pd.concat(open_price_mult_lst   , axis=1)
  high_price_mult    = pd.concat(high_price_mult_lst   , axis=1)
  low_price_mult     = pd.concat(low_price_mult_lst    , axis=1)
  close_price_mult   = pd.concat(close_price_mult_lst  , axis=1)
  vwap_mult          = pd.concat(vwap_mult_lst         , axis=1)
  quote_volume       = pd.concat(quote_volume_lst      , axis=1)
  num_trades         = pd.concat(num_trades_lst        , axis=1)
  taker_vwap_mult    = pd.concat(taker_vwap_mult_lst   , axis=1)
  taker_quote_volume = pd.concat(taker_quote_volume_lst, axis=1)

  close_price = []
  adj_weight = []
  index_init_value = 100
  index_current_value = index_init_value
  for i in range(len(weight)):
    start_date = weight.index[i]
    end_date = weight.index[i + 1] if i + 1 < len(weight) else None
    weight_slice = weight.iloc[i:i+1]

    close_price_mult_slice = close_price_mult.loc[(close_price_mult.index >= start_date) & ((close_price_mult.index < end_date) if end_date is not None else True)]
    close_price_change_slice = (1 + close_price_mult_slice).cumprod() - 1 # ratio to the ref point
    close_price_slice = index_current_value * (1 + (weight_slice.reindex(close_price_change_slice.index, method='ffill') * close_price_change_slice).sum(axis=1))
    adj_weight_slice = weight_slice.reindex(close_price_change_slice.index, method='ffill') * (1 + close_price_change_slice)
    adj_weight_slice = adj_weight_slice.div(adj_weight_slice.sum(axis=1), axis=0).fillna(0)
    close_price.append(close_price_slice)
    adj_weight.append(adj_weight_slice)
    index_current_value = close_price_slice.iloc[-1]

  close_price = pd.concat(close_price)
  adj_weight = pd.concat(adj_weight)

  previous_close = close_price.shift(1, fill_value=index_init_value)

  # aggregate by weight
  open_price_mult    = (open_price_mult    * adj_weight).sum(axis=1)
  high_price_mult    = (high_price_mult    * adj_weight).sum(axis=1)
  low_price_mult     = (low_price_mult     * adj_weight).sum(axis=1)
  vwap_mult          = (vwap_mult          * adj_weight).sum(axis=1)
  quote_volume       = (quote_volume       * adj_weight).sum(axis=1)
  num_trades         = (num_trades         * adj_weight).sum(axis=1)
  taker_vwap_mult    = (taker_vwap_mult    * adj_weight).sum(axis=1)
  taker_quote_volume = (taker_quote_volume * adj_weight).sum(axis=1)

  open_price  = previous_close * (1 + open_price_mult)
  high_price  = previous_close * (1 + high_price_mult)
  low_price   = previous_close * (1 + low_price_mult )
  vwap        = previous_close * (1 + vwap_mult      )
  taker_vwap  = previous_close * (1 + taker_vwap_mult)

  base_volume = quote_volume / vwap
  taker_base_volume = taker_quote_volume / taker_vwap

  custom_index_kline = pd.DataFrame(
    {
      'open_price'        : open_price        ,
      'high_price'        : high_price        ,
      'low_price'         : low_price         ,
      'close_price'       : close_price       ,
      'base_volume'       : base_volume       ,
      'quote_volume'      : quote_volume      ,
      'num_trades'        : num_trades        ,
      'taker_base_volume' : taker_base_volume ,
      'taker_quote_volume': taker_quote_volume,
    },
    index = date_range,
  )

  return custom_index_kline

if __name__ == "__main__":
  start = "2021-01-01" # inc
  end   = "2025-10-01" # not inc

  resample_freq_lst = ['15min', '30min', '60min', '240min', '720min', '1d']

  start_time = pd.Timestamp(start, tz="UTC").normalize()
  end_time = pd.Timestamp(end, tz="UTC").normalize()
  step = pd.Timedelta(days=1)

  save_path = '/capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto'
  save_path = Path(save_path)

  instrument_list_save_path = save_path / "raw" / "instrument_list"
  instrument_list_save_path.mkdir(parents=True, exist_ok=True)

  workspace_path = instrument_list_save_path / "workspace"
  workspace_path.mkdir(parents=True, exist_ok=True)

  raw_kline_path = save_path / "raw"
  raw_kline_path.mkdir(parents=True, exist_ok=True)

  resampled_kline_path = save_path / "resampled"
  resampled_kline_path.mkdir(parents=True, exist_ok=True)

  '''
  1. Fetch Raw Instrument List Info
  '''

  # df = fetch_all_pages(start_time, end_time, step, instrument_list_save_path)

  '''
  2. Process Instrument List
  '''
  # df = pd.read_feather(instrument_list_save_path / "raw.feather")
  # symbol_enlisted_time = get_symbols_from_raw(df, start_time, end_time)

  '''
  3. Fetch Daily KLine
  '''
  # trade_symbol_info = {}
  # for symbol, enlisted_time in zip(symbol_enlisted_time['symbol'], symbol_enlisted_time['enlisted_time']):
  #   trade_symbol = symbol + "USDT"
  #   df = get_kline_data(trade_symbol, '1d', start_time, end_time)
  #   trade_symbol_info[trade_symbol] = {
  #     "trade_symbol": trade_symbol,
  #     "enlisted_time": enlisted_time,
  #     "kline_exists": not df.empty,
  #   }
  #   if not df.empty:
  #     df.to_csv(workspace_path / f"{trade_symbol}.csv", index=False)
  # pd.DataFrame(list(trade_symbol_info.values())).to_csv(workspace_path / f"trade_symbol_info.csv", index=False)

  '''
  4. Build Index: Components & Weights
  '''
  # trade_symbol_info = pd.read_csv(workspace_path / f"trade_symbol_info.csv")
  # trade_symbol_klines = {}
  # for trade_symbol, kline_exists in zip(trade_symbol_info['trade_symbol'], trade_symbol_info['kline_exists']):
  #   if kline_exists:
  #     kline_df = pd.read_csv(workspace_path / f"{trade_symbol}.csv")
  #     kline_df = process_raw_kline(kline_df, '1d')
  #     trade_symbol_klines[trade_symbol] = kline_df
  
  # # liquidity screen: 30 day avg > 1M USDT/day
  # volume_threshold = 1_000_000
  # enlist_threshold = 0.15
  # reconstitution_dates = pd.date_range(start_time, end_time, freq='MS', inclusive="left")

  # volume_df = pd.concat({sym: df['quote_volume'] for sym, df in trade_symbol_klines.items()}, axis=1).sort_index()
  # enter_condition = volume_df.shift(1).rolling(30).mean().fillna(0) > volume_threshold * (1 + enlist_threshold)
  # exit_condition  = volume_df.shift(1).rolling(30).mean().fillna(0) < volume_threshold * (1 - enlist_threshold)

  # # filter stable coin
  # change_df = pd.concat({sym: df['close_price'] / df['close_price'].shift(1) - 1 for sym, df in trade_symbol_klines.items()}, axis=1).sort_index()
  # change_eligible = change_df.shift(1).rolling(30).quantile(0.9) > 0.01

  # enter_condition = enter_condition & change_eligible

  # # enter after enlisted time
  # symbol_enlisted_time = pd.to_datetime(
  #   trade_symbol_info[trade_symbol_info['kline_exists']]
  #   .set_index('trade_symbol')['enlisted_time']
  # )

  # causality_mask = pd.DataFrame(
  #   enter_condition.index.values[:, np.newaxis] >= symbol_enlisted_time.reindex(enter_condition.columns).values[np.newaxis, :],
  #   index = enter_condition.index,
  #   columns = enter_condition.columns,
  # )
  # enter_condition = enter_condition & causality_mask

  # enter_condition = enter_condition.reindex(reconstitution_dates)
  # exit_condition = exit_condition.reindex(reconstitution_dates)

  # enlisted_state = get_enlisted_state(enter_condition, exit_condition)
  # enlisted_state.to_feather(instrument_list_save_path / f"enlisted_state.feather")

  # # weight
  # # volume weighted
  # volume_df = pd.concat({sym: df['quote_volume'] for sym, df in trade_symbol_klines.items()}, axis=1).sort_index()
  # weight_volume = volume_df.shift(1).rolling(30).mean().reindex(enlisted_state.index).fillna(0)
  # weight_volume = weight_volume * enlisted_state
  # weight_volume.to_feather(instrument_list_save_path / f"weight_volume.feather")

  # # equal weighted
  # weight_equal = enlisted_state.astype(float)
  # weight_equal.to_feather(instrument_list_save_path / f"weight_equal.feather")

  # # top-N
  # volume_df = pd.concat({sym: df['quote_volume'] for sym, df in trade_symbol_klines.items()}, axis=1).sort_index()
  # volume_score = volume_df.shift(1).rolling(90).mean().reindex(enlisted_state.index).fillna(0) * enlisted_state

  # mask_top50  = volume_score.apply(lambda row: (row >= row.nlargest(50).min()) & (row > 0), axis=1)
  # mask_top200 = volume_score.apply(lambda row: (row >= row.nlargest(200).min()) & (row > 0), axis=1)
  # mask_mid150 = mask_top200 & (~mask_top50)

  # enlisted_state_top50  = mask_top50  & enlisted_state
  # enlisted_state_top200 = mask_top200 & enlisted_state
  # enlisted_state_mid150 = mask_mid150 & enlisted_state

  # for suffix in ['top50', 'top200', 'mid150']:
  #   eval(f"enlisted_state_{suffix}").to_feather(instrument_list_save_path / f"enlisted_state_{suffix}.feather")

  #   weight_volume = volume_score * eval(f"enlisted_state_{suffix}")
  #   weight_volume.to_feather(instrument_list_save_path / f"weight_volume_{suffix}.feather")

  #   # equal weighted
  #   weight_equal = eval(f"enlisted_state_{suffix}").astype(float)
  #   weight_equal.to_feather(instrument_list_save_path / f"weight_equal_{suffix}.feather")

  '''
  5. Fetch 15min Kline
  '''
  # enlisted_state = pd.read_feather(instrument_list_save_path / f"enlisted_state.feather")
  # trade_symbols = enlisted_state.loc[:, enlisted_state.any()].columns
  # for trade_symbol in trade_symbols:
  #   df = get_kline_data(trade_symbol, '15m', start_time, end_time)
  #   assert not df.empty
  #   df.to_csv(raw_kline_path / f"{trade_symbol}.csv", index=False)
  
  '''
  6. Resample & Make Dump-Ready
  '''
  enlisted_state = pd.read_feather(instrument_list_save_path / f"enlisted_state.feather")
  weight_volume = pd.read_feather(instrument_list_save_path / f"weight_volume.feather")
  weight_equal = pd.read_feather(instrument_list_save_path / f"weight_equal.feather")

  enlisted_regions = extract_enlisted_regions(enlisted_state)
  enlisted_regions["enlisted_region_end"] = enlisted_regions["enlisted_region_end_incl"] + pd.offsets.MonthBegin(1)

  for resample_freq in resample_freq_lst:
    resample_freq_instruments_save_path = resampled_kline_path / resample_freq / "instruments"
    resample_freq_instruments_save_path.mkdir(parents=True, exist_ok=True)
    # universe
    pd.concat(
      [
        enlisted_regions['symbol'],
        enlisted_regions['enlisted_region_start'].dt.strftime("%Y-%m-%d %H:%M:%S"),
        (enlisted_regions['enlisted_region_end'] - pd.to_timedelta(resample_freq)).dt.strftime("%Y-%m-%d %H:%M:%S"),
      ],
      axis=1,
    ).to_csv(
      resample_freq_instruments_save_path / "my_universe.txt", sep='\t', index=False, header=False
    )
  
  # (extract_enlisted_state(enlisted_regions, enlisted_state.index, enlisted_state.columns) == enlisted_state).all().all()
  
  suffix_index_info = {}
  for suffix in ['top50', 'top200', 'mid150']:
    enlisted_state_suffix = pd.read_feather(instrument_list_save_path / f"enlisted_state_{suffix}.feather")
    weight_volume_suffix = pd.read_feather(instrument_list_save_path / f"weight_volume_{suffix}.feather")
    weight_equal_suffix = pd.read_feather(instrument_list_save_path / f"weight_equal_{suffix}.feather")
    enlisted_regions_suffix = extract_enlisted_regions(enlisted_state_suffix)
    enlisted_regions_suffix["enlisted_region_end"] = enlisted_regions_suffix["enlisted_region_end_incl"] + pd.offsets.MonthBegin(1)
    # (extract_enlisted_state(enlisted_regions_suffix, enlisted_state_suffix.index, enlisted_state_suffix.columns) == enlisted_state_suffix).all().all()

    suffix_index_info[suffix] = {
      'enlisted_state': enlisted_state_suffix,
      'weight_volume': weight_volume_suffix,
      'weight_equal' : weight_equal_suffix ,
    }

    for resample_freq in resample_freq_lst:
      resample_freq_instruments_save_path = resampled_kline_path / resample_freq / "instruments"
      resample_freq_instruments_save_path.mkdir(parents=True, exist_ok=True)
      # universe
      pd.concat(
        [
          enlisted_regions_suffix['symbol'],
          enlisted_regions_suffix['enlisted_region_start'].dt.strftime("%Y-%m-%d %H:%M:%S"),
          (enlisted_regions_suffix['enlisted_region_end'] - pd.to_timedelta(resample_freq)).dt.strftime("%Y-%m-%d %H:%M:%S"),
        ],
        axis=1,
      ).to_csv(
        resample_freq_instruments_save_path / f"my_universe_{suffix}.txt", sep='\t', index=False, header=False
      )

  logger.info("Loading all raw klines ......")
  trade_symbols = enlisted_state.columns[enlisted_state.any()]
  trade_symbol_klines = {}
  for trade_symbol in trade_symbols:
    kline_df = pd.read_csv(raw_kline_path / f"{trade_symbol}.csv")
    kline_df = process_raw_kline(kline_df, '15min')
    trade_symbol_klines[trade_symbol] = kline_df

  for resample_freq in resample_freq_lst:
    logger.info(f"Resampling to {resample_freq} ......")
    resample_freq_save_path = resampled_kline_path / resample_freq
    resample_freq_save_path.mkdir(parents=True, exist_ok=True)

    # kline
    resampled_klines = {}
    for trade_symbol in trade_symbols:
      kline = trade_symbol_klines[trade_symbol]
      resampled_kline = resample_data(kline, resample_freq)
      resampled_klines[trade_symbol] = resampled_kline

    # index
    logger.info(f"Synthesizing custom index for {resample_freq} ......")
    resampled_klines["MYINDEXVOL"] = synthesize_custom_index(
      resampled_klines, 
      weight_volume, 
      pd.date_range(start_time, end_time, freq=resample_freq, inclusive="left")
    )
    resampled_klines["MYINDEXEQ" ] = synthesize_custom_index(
      resampled_klines, 
      weight_equal, 
      pd.date_range(start_time, end_time, freq=resample_freq, inclusive="left")
    )

    for suffix in suffix_index_info.keys():
      resampled_klines[f"MYINDEXVOL{suffix.upper()}"] = synthesize_custom_index(
        resampled_klines, 
        suffix_index_info[suffix]['weight_volume'], 
        pd.date_range(start_time, end_time, freq=resample_freq, inclusive="left")
      )
      resampled_klines[f"MYINDEXEQ{suffix.upper()}"] = synthesize_custom_index(
        resampled_klines, 
        suffix_index_info[suffix]['weight_equal'], 
        pd.date_range(start_time, end_time, freq=resample_freq, inclusive="left")
      )

    logger.info(f"Saving resampled klines for {resample_freq} ......")
    for trade_symbol in resampled_klines.keys():
      kline = resampled_klines[trade_symbol]
      kline = kline.rename(
        columns = {
          'open_price'        : 'open',
          'high_price'        : 'high',
          'low_price'         : 'low',
          'close_price'       : 'close',
          'base_volume'       : 'volume',
          'quote_volume'      : 'amount',
          'num_trades'        : 'numtrades',
          'taker_base_volume' : 'takervolume',
          'taker_quote_volume': 'takeramount',
        }
      )
      # set nan
      # kline.loc[(kline["volume"] <= 0) | np.isnan(kline["volume"]), :] = np.nan
      kline.insert(0, 'symbol', trade_symbol)
      kline['change'] = kline['close'] / kline["close"].shift(1) - 1
      kline['factor'] = 1
      kline.insert(0, 'date', kline.index.strftime("%Y-%m-%d %H:%M:%S"))

      kline.to_csv(resample_freq_save_path / f"{trade_symbol}.csv", index=False)

