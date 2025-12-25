import os
import sys
import json
import filelock
import pandas as pd
import baostock as bs
from loguru import logger

sys.path.append("/users/ljiayong/projects/qlib/scripts/data_collector/my_utils")
from data_collector_utils import deco_retry

def incremental_fetch(
  func,
  start_time,
  end_time,
  func_kwargs,
  save_path,
  file_name,
  date_key,
  unique_keys,
  margin_end = "5d",
):
  os.makedirs(save_path, exist_ok=True)
  csv_path = os.path.join(save_path, file_name + ".csv")
  parquet_path = os.path.join(save_path, file_name + ".parquet")
  lock_path = os.path.join(save_path, file_name + ".lock")

  # get current data
  if not os.path.exists(parquet_path):
    logger.info(f"Running {func.__name__} from {start_time} to {end_time} ...")
    df = func(start_time, end_time, **func_kwargs)
  else:
    df = pd.read_parquet(parquet_path)
    # fetch missing dates
    current_dates = pd.to_datetime(df[date_key]).dt.tz_localize(start_time.tz)
    current_start = current_dates.min()
    current_end = current_dates.max()
    # add margin
    margin_end = pd.to_timedelta(margin_end)
    current_end = current_end - margin_end

    if start_time < current_start:
      logger.info(f"Running {func.__name__} from {start_time} to {current_start} ...")
      df_start = func(start_time, current_start, **func_kwargs)
      df = pd.concat([df_start, df], ignore_index=True)

    if end_time > current_end:
      logger.info(f"Running {func.__name__} from {current_end} to {end_time} ...")
      df_end = func(current_end, end_time, **func_kwargs)
      df = pd.concat([df, df_end], ignore_index=True)
    
    df = df.drop_duplicates(subset=unique_keys)
    df = df.sort_values(by=unique_keys).reset_index(drop=True)

  df.to_parquet(parquet_path+".tmp")
  os.replace(parquet_path+".tmp", parquet_path)
  df.to_csv(csv_path+".tmp", index=False)
  os.replace(csv_path+".tmp", csv_path)

  return df

@deco_retry(retry_sleep=5, retry=2)
def fetch_trade_dates(start_time, end_time):
  start = start_time.strftime("%Y-%m-%d")
  end   = end_time.strftime("%Y-%m-%d")
  rs = bs.query_trade_dates(start_date=start, end_date=end)
  if rs.error_code == "0":
    data_list = rs.data
    columns = rs.fields # calendar_date, is_trading_day
    df = pd.DataFrame(data_list, columns=columns)
  else:
    raise ValueError(f"Fetch trade dates from {start} to {end} error, code: {rs.error_code}, msg: {rs.error_msg}")
  return df

@deco_retry(retry_sleep=5, retry=2)
def _fetch_index_stocks(date, index):
  date = date.strftime("%Y-%m-%d")
  if index in ['sz50', 'hs300', 'zz500']:
    rs = eval(f"bs.query_{index}_stocks")(date=date) # updateDate, code, code_name
  elif index == 'all':
    rs = eval(f"bs.query_{index}_stock")(day=date) # code, tradeStatus, code_name
  else:
    raise ValueError(f"index {index} not supported")

  if rs.error_code == "0":
    data_list = rs.data
    columns = rs.fields
    df = pd.DataFrame(data_list, columns=columns)
    df["query_date"] = date
  else:
    raise ValueError(f"Fetch {index} stock at {date} error, code: {rs.error_code}, msg: {rs.error_msg}")
  return df

def fetch_index_stocks(start_time, end_time, index, trade_dates):
  # all dates
  # date_range = pd.date_range(start=start_time, end=end_time, freq='1d')
  # trade dates only
  date_range = trade_dates[trade_dates['is_trading_day'] == '1']['calendar_date']
  date_range = pd.to_datetime(date_range).dt.tz_localize(start_time.tz)
  date_range = date_range[(date_range >= start_time) & (date_range <= end_time)]
  results = []
  for date in date_range:
    df = _fetch_index_stocks(date, index)
    results.append(df)
  df = pd.concat(results, ignore_index=True)
  return df

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

@deco_retry(retry_sleep=5, retry=2)
def fetch_kline_data(start_time, end_time, symbol):
  start = start_time.strftime("%Y-%m-%d")
  end   = end_time.strftime("%Y-%m-%d")
  rs = bs.query_history_k_data_plus(
    symbol,
    "date,code,open,high,low,close,volume,amount,turn,tradestatus,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
    start_date=start,
    end_date=end,
    frequency="d",
    adjustflag="3",
  )
  if rs.error_code == "0":
    data_list = rs.data
    columns = rs.fields
    df = pd.DataFrame(data_list, columns=columns)
  else:
    raise ValueError(f"Fetch {symbol} kline data from {start} to {end} error, code: {rs.error_code}, msg: {rs.error_msg}")
  rs_adj = bs.query_history_k_data_plus(
    symbol,
    "date,close",
    start_date=start,
    end_date=end,
    frequency="d",
    adjustflag="1",
  )
  if rs_adj.error_code == "0":
    data_list = rs_adj.data
    columns = ["date", "adjclose"]
    df_adj = pd.DataFrame(data_list, columns=columns)
    df = pd.merge(df, df_adj, on="date", how="left")
  else:
    raise ValueError(f"Fetch {symbol} adj price from {start} to {end} error, code: {rs_adj.error_code}, msg: {rs_adj.error_msg}")
  return df

def normalize_symbol(symbol):
  return str(symbol).replace(".", "").upper()

if __name__ == "__main__":
  # args
  start = "2024-12-01" # inc
  end   = "2025-01-01" # inc
  tz    = 'Asia/Shanghai'

  if end is None:
    now = pd.Timestamp.now(tz='UTC').tz_convert(tz)
    end = now.strftime("%Y-%m-%d")
  
  start_time = pd.Timestamp(start, tz=tz)
  end_time = pd.Timestamp(end, tz=tz)
  assert start_time <= end_time, f"start_time {start_time} should be <= end_time {end_time}"

  save_path = '/capstor/scratch/cscs/ljiayong/datasets/qlib/baostock_incremental'

  index_to_symbol = {
    'sz50' : 'sh.000016',
    'hs300': 'sh.000300',
    'zz500': 'sh.000905',
  }
  # index_lst = ['sz50', 'hs300', 'zz500']
  index_lst = ['sz50']
  index_symbols = [index_to_symbol[x] for x in index_lst]

  '''
  1. Fetch Trade Calendar
  '''
  bs.login()
  trade_dates = incremental_fetch(
    func = fetch_trade_dates,
    start_time = start_time,
    end_time = end_time,
    func_kwargs = {},
    save_path = os.path.join(save_path, "raw", "trade_dates"),
    file_name = "trade_dates",
    date_key = "calendar_date",
    unique_keys = ["calendar_date"],
    margin_end = "5d",
  )
  bs.logout()

  '''
  2. Fetch Instrument List
  '''
  bs.login()
  for index in index_lst:
    df = incremental_fetch(
      func = fetch_index_stocks,
      start_time = start_time,
      end_time = end_time,
      func_kwargs = {"index": index, "trade_dates": trade_dates},
      save_path = os.path.join(save_path, "raw", "instrument_list"),
      file_name = f"{index}_stocks",
      date_key = "query_date",
      unique_keys = ["query_date", "code"],
      margin_end = "5d",
    )
  bs.logout()

  '''
  3. Get Index Components
  '''
  for index in index_lst:
    df = pd.read_parquet(os.path.join(save_path, "raw", "instrument_list", f"{index}_stocks.parquet"))
    df['date'] = pd.to_datetime(df['query_date']).dt.tz_localize(tz) # no tz info needed
    df['is_enlisted'] = 1
    enlisted_state = (
      pd.pivot_table(
        df,
        values = 'is_enlisted',
        index = 'date',
        columns = 'code',
        aggfunc = 'sum',
        fill_value = 0,
      )
      .astype(bool)
      .sort_index()
    )
    enlisted_regions = extract_enlisted_regions(enlisted_state)

    os.makedirs(os.path.join(save_path, "processed", "instrument_list"), exist_ok=True)
    enlisted_state.to_parquet(os.path.join(save_path, "processed", "instrument_list", f"{index}_enlisted_state.parquet"))
    enlisted_regions.to_parquet(os.path.join(save_path, "processed", "instrument_list", f"{index}_enlisted_regions.parquet"))

  '''
  4. Fetch Klines for All Symbols
  '''
  stock_symbols = set()
  for index in index_lst:
    enlisted_regions = pd.read_parquet(os.path.join(save_path, "processed", "instrument_list", f"{index}_enlisted_regions.parquet"))
    stock_symbols.update(enlisted_regions['symbol'].unique())
  all_symbols = sorted(stock_symbols | set(index_symbols))

  bs.login()
  for symbol in all_symbols:
    incremental_fetch(
      func = fetch_kline_data,
      start_time = start_time,
      end_time = end_time,
      func_kwargs = {"symbol": symbol},
      save_path = os.path.join(save_path, "raw", "kline_data"),
      file_name = f"{symbol}",
      date_key = "date",
      unique_keys = ["date"],
      margin_end = "5d",
    )
  bs.logout()  

  '''
  4. Make Data Dump-Ready
  '''
  # symbol_klines = {}
  # trade_calendar = None
  # for i, index in enumerate(index_lst):
  #   enlisted_state = pd.read_feather(instrument_list_save_path / f"{index}_enlisted_state.feather")
  #   trade_calendar = enlisted_state.index
  #   enlisted_regions = extract_enlisted_regions(enlisted_state)

  #   symbols = [index_symbols[i]] + list(enlisted_regions["symbol"].unique())
  #   for symbol in symbols:
  #     if symbol not in symbol_klines.keys():
  #       symbol_klines[symbol] = pd.read_csv(raw_kline_path / f"{symbol}.csv")

  #   enlisted_regions["symbol"] = enlisted_regions["symbol"].apply(normalize_symbol)
  #   enlisted_regions["enlisted_region_start"] = enlisted_regions["enlisted_region_start"].dt.strftime("%Y-%m-%d")
  #   enlisted_regions["enlisted_region_end_incl"] = enlisted_regions["enlisted_region_end_incl"].dt.strftime("%Y-%m-%d")
  #   enlisted_regions[["symbol", "enlisted_region_start", "enlisted_region_end_incl"]].to_csv(
  #     instrument_list_save_path / f"{index}.txt", sep='\t', index=False, header=False
  #   )

  # for symbol, df in symbol_klines.items():
  #   normalized_symbol = normalize_symbol(symbol)
  #   df["date"] = pd.to_datetime(df["date"])
  #   df = df.set_index("date")
  #   df = df.reindex(trade_calendar[(trade_calendar >= df.index.min()) & (trade_calendar <= df.index.max())])
  #   '''
  #   date,code,open,high,low,close,volume,amount,turn,tradestatus,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,adjclose
  #   '''
  #   df.insert(0, 'date', df.index.strftime("%Y-%m-%d"))
  #   df.rename(columns={"code": "symbol"}, inplace=True)
  #   df["symbol"] = normalized_symbol
  #   # fill missing
  #   df["close"] = df["close"].ffill()
  #   # fill ohl with last close
  #   df["open"] = df["open"].combine_first(df["close"])
  #   df["high"] = df["high"].combine_first(df["close"])
  #   df["low"]  = df["low"] .combine_first(df["close"])
  #   # fill volume with 0
  #   df["volume"] = df["volume"].fillna(0)
  #   df["amount"] = df["amount"].fillna(0)
  #   df["turn"]   = df["turn"]  .fillna(0)
  #   df["tradestatus"] = df["tradestatus"].fillna(0)
  #   df.drop(columns="tradestatus", inplace=True)

  #   df["peTTM"] = df["peTTM"].ffill()
  #   df["pbMRQ"] = df["pbMRQ"].ffill()
  #   df["psTTM"] = df["psTTM"].ffill()
  #   df["pcfNcfTTM"] = df["pcfNcfTTM"].ffill()
  #   df["isST"] = df["isST"].ffill()
  #   df.drop(columns="isST", inplace=True)

  #   df["factor"] = df["adjclose"] / df["close"]
  #   df["factor"] = df["factor"].ffill()
  #   df.drop(columns="adjclose", inplace=True)
  #   for col in df.columns:
  #     if col in ["date", "symbol", "amount", "turn", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "factor"]:
  #       pass
  #     elif col in ["open", "high", "low", "close"]:
  #       df[col] = df[col] * df["factor"]
  #     elif col in ["volume"]:
  #       df[col] = df[col] / df["factor"]
  #     else:
  #       raise ValueError(f"Unknown column: {col}")

  #   df["change"] = df["close"] / df["close"].shift(1) - 1

  #   df.to_csv(processed_kline_path / f"{normalized_symbol}.csv", index=False)


