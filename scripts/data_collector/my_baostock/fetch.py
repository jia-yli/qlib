import os
import sys
import json
import shutil
import numpy as np
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
    
    df = df.drop_duplicates(subset=unique_keys, keep='last')
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

@deco_retry(retry_sleep=5, retry=2)
def fetch_stock_industry():
  rs = bs.query_stock_industry()
  if rs.error_code == "0":
    data_list = rs.data
    columns = rs.fields
    df = pd.DataFrame(data_list, columns=columns)
  else:
    raise ValueError(f"Fetch stock industry error, code: {rs.error_code}, msg: {rs.error_msg}")
  return df

def normalize_symbol(symbol):
  return str(symbol).replace(".", "").upper()

if __name__ == "__main__":
  # args
  start = "2010-01-01" # inc
  end   = "2025-12-15" # inc
  tz    = 'Asia/Shanghai'

  index_lst = ['sz50', 'hs300', 'zz500']
  # index_lst = ['hs300']

  save_path = '/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock'

  run_fetch = False
  margin_end = "5d"

  now_utc = pd.Timestamp.now(tz='UTC')
  if end is None:
    end = now_utc.tz_convert(tz).strftime("%Y-%m-%d")
  
  start_time = pd.Timestamp(start, tz=tz)
  end_time = pd.Timestamp(end, tz=tz)
  assert start_time <= end_time, f"start_time {start_time} should be <= end_time {end_time}"

  index_to_symbol = {
    'sz50' : 'sh.000016',
    'hs300': 'sh.000300',
    'zz500': 'sh.000905',
  }
  index_symbols = [index_to_symbol[x] for x in index_lst]

  # backup save path
  now_utc_str = now_utc.strftime("%Y%m%d_%H%M%S") + "_utc"
  backup_path = save_path + "_backup_" + now_utc_str
  if run_fetch:
    if os.path.exists(save_path):
      logger.info(f"Backing up existing data folder {save_path} to {backup_path} ...")
      if os.path.exists(backup_path):
        shutil.rmtree(backup_path)
      shutil.copytree(save_path, backup_path)

  '''
  1. Fetch Trade Calendar
  '''
  if run_fetch:
    bs.login()
    incremental_fetch(
      func = fetch_trade_dates,
      start_time = start_time,
      end_time = end_time,
      func_kwargs = {},
      save_path = os.path.join(save_path, "raw", "trade_dates"),
      file_name = "trade_dates",
      date_key = "calendar_date",
      unique_keys = ["calendar_date"],
      margin_end = margin_end,
    )
    bs.logout()

  '''
  2. Process Trade Calendar
  '''
  trade_dates = pd.read_parquet(os.path.join(save_path, "raw", "trade_dates", "trade_dates.parquet"))
  # trim start and end such that both are trading dates
  date_range = trade_dates[trade_dates['is_trading_day'] == '1']['calendar_date']
  date_range = pd.to_datetime(date_range).dt.tz_localize(tz)
  date_range = date_range[(date_range >= start_time) & (date_range <= end_time)]

  start_time = max(start_time, date_range.min())
  end_time = min(date_range.max(), end_time)

  trade_calendar = date_range.dt.strftime("%Y-%m-%d")
  os.makedirs(os.path.join(save_path, "processed", "trade_dates"), exist_ok=True)
  trade_calendar.to_csv(os.path.join(save_path, "processed", "trade_dates", "trade_calendar.csv"), index=False, header=False)

  '''
  3. Fetch Instrument List
  '''
  if run_fetch:
    bs.login()
    for index in index_lst:
      logger.info(f"Fetching stocks for {index} ...")
      incremental_fetch(
        func = fetch_index_stocks,
        start_time = start_time,
        end_time = end_time,
        func_kwargs = {"index": index, "trade_dates": trade_dates},
        save_path = os.path.join(save_path, "raw", "instrument_list"),
        file_name = f"{index}_stocks",
        date_key = "query_date",
        unique_keys = ["query_date", "code"],
        margin_end = margin_end,
      )
    bs.logout()

  '''
  4. Process Instrument List
  '''
  stock_symbols = set()
  for index in index_lst:
    df = pd.read_parquet(os.path.join(save_path, "raw", "instrument_list", f"{index}_stocks.parquet"))
    df['date'] = pd.to_datetime(df['query_date']).dt.tz_localize(tz)
    df = df[(df['date'] >= start_time) & (df['date'] <= end_time)]
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
    stock_symbols.update(enlisted_regions['symbol'].unique())

    # save enlisted state
    os.makedirs(os.path.join(save_path, "processed", "instrument_list"), exist_ok=True)
    enlisted_state.to_parquet(os.path.join(save_path, "processed", "instrument_list", f"{index}_enlisted_state.parquet"))

    # process & dump enlisted regions to qlib format
    enlisted_regions["symbol"] = enlisted_regions["symbol"].apply(normalize_symbol)
    enlisted_regions["enlisted_region_start"] = enlisted_regions["enlisted_region_start"].dt.strftime("%Y-%m-%d")
    enlisted_regions["enlisted_region_end_incl"] = enlisted_regions["enlisted_region_end_incl"].dt.strftime("%Y-%m-%d")
    enlisted_regions[["symbol", "enlisted_region_start", "enlisted_region_end_incl"]].to_csv(
      os.path.join(save_path, "processed", "instrument_list", f"{index}.txt"), sep='\t', index=False, header=False
    )
  all_symbols = sorted(stock_symbols | set(index_symbols))

  '''
  5. Fetch Klines
  '''
  if run_fetch:
    bs.login()
    for symbol in all_symbols:
      logger.info(f"Fetching kline data for {symbol} ...")
      incremental_fetch(
        func = fetch_kline_data,
        start_time = start_time,
        end_time = end_time,
        func_kwargs = {"symbol": symbol},
        save_path = os.path.join(save_path, "raw", "kline_data"),
        file_name = f"{symbol}",
        date_key = "date",
        unique_keys = ["date"],
        margin_end = margin_end,
      )
    bs.logout()

  '''
  6. Process Klines
  '''
  trade_calendar = pd.read_csv(os.path.join(save_path, "processed", "trade_dates", "trade_calendar.csv"), header=None).iloc[:,0]
  for symbol in all_symbols:
    normalized_symbol = normalize_symbol(symbol)
    df = pd.read_csv(os.path.join(save_path, "raw", "kline_data", f"{symbol}.csv"))
    df = df.set_index("date")
    df = df.reindex(trade_calendar[(trade_calendar >= df.index.min()) & (trade_calendar <= df.index.max())].rename("date"))
    df = df.reset_index()
    '''
    date,code,open,high,low,close,volume,amount,turn,tradestatus,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,adjclose
    '''
    df.rename(columns={"code": "symbol"}, inplace=True)
    df["symbol"] = normalized_symbol
    # fill missing
    df["close"] = df["close"].ffill()
    # fill ohl with last close
    df["open"] = df["open"].combine_first(df["close"])
    df["high"] = df["high"].combine_first(df["close"])
    df["low"]  = df["low"] .combine_first(df["close"])
    # get vwap
    df["vwap"] = df["amount"] / df["volume"]
    df["vwap"] = df["vwap"].combine_first(df["close"])
    # fill volume with 0
    df["volume"] = df["volume"].fillna(0)
    df["amount"] = df["amount"].fillna(0)
    df["turn"]   = df["turn"]  .fillna(0)
    df["tradestatus"] = df["tradestatus"].fillna(0)
    df.drop(columns="tradestatus", inplace=True)

    df["peTTM"] = df["peTTM"].ffill()
    df["pbMRQ"] = df["pbMRQ"].ffill()
    df["psTTM"] = df["psTTM"].ffill()
    df["pcfNcfTTM"] = df["pcfNcfTTM"].ffill()
    df["isST"] = df["isST"].ffill()
    df.drop(columns="isST", inplace=True)

    df["factor"] = df["adjclose"] / df["close"]
    df["factor"] = df["factor"].ffill()
    df.drop(columns="adjclose", inplace=True)
    for col in df.columns:
      if col in ["date", "symbol", "amount", "turn", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "factor"]:
        pass
      elif col in ["open", "high", "low", "close", "vwap"]:
        df[col] = df[col] * df["factor"]
      elif col in ["volume"]:
        df[col] = df[col] / df["factor"]
      else:
        raise ValueError(f"Unknown column: {col}")

    df["change"] = df["close"] / df["close"].shift(1) - 1

    os.makedirs(os.path.join(save_path, "processed", "kline_data"), exist_ok=True)
    df.to_csv(os.path.join(save_path, "processed", "kline_data", f"{normalized_symbol}.csv"), index=False)

  '''
  7. Fetch Stock Industry
  '''
  if run_fetch:
    bs.login()
    logger.info(f"Fetching stock industry ...")
    df = fetch_stock_industry()
    os.makedirs(os.path.join(save_path, "raw", "stock_industry"), exist_ok=True)
    df.to_csv(os.path.join(save_path, "raw", "stock_industry", "stock_industry.csv"), index=False)
    bs.logout()

  '''
  8. Process Stock Industry
  '''
  df = pd.read_csv(os.path.join(save_path, "raw", "stock_industry", "stock_industry.csv"))
  '''
  updateDate,code,code_name,industry,industryClassification
  '''
  df.drop(columns="updateDate", inplace=True)
  df.rename(columns={"code": "symbol"}, inplace=True)
  df['symbol'] = df['symbol'].apply(normalize_symbol)
  df.drop(columns="code_name", inplace=True)

  # 国民经济行业分类 GB/T 4754—2017
  # https://www.beijing.gov.cn/zhengce/zhengcefagui/202304/W020230410621028325997.pdf
  industry = df["industry"].copy() # e.g. J66货币金融服务
  df.drop(columns="industry", inplace=True)
  df["sector"] = industry.str.extract(r"^([A-Z])", expand=False)
  df["industry"] = industry.str.extract(r"^([A-Z]\d\d)", expand=False)
  # df["industry_name"] = industry.str.replace(r"^[A-Z]\d\d", "", regex=True)
  df.drop(columns="industryClassification", inplace=True)

  os.makedirs(os.path.join(save_path, "processed", "stock_industry"), exist_ok=True)
  df.to_csv(os.path.join(save_path, "processed", "stock_industry", "stock_industry.csv"), index=False)

  '''
  9. Dump Klines
  '''
  # 1. calendars
  trade_calendar = pd.read_csv(os.path.join(save_path, "processed", "trade_dates", "trade_calendar.csv"), header=None).iloc[:,0]
  os.makedirs(os.path.join(save_path, "bin", "calendars"), exist_ok=True)
  trade_calendar.to_csv(os.path.join(save_path, "bin", "calendars", "day.txt"), index=False, header=False)
  # 2. instruments
  # all
  all_instruments = []
  for symbol in all_symbols:
    normalized_symbol = normalize_symbol(symbol)
    df = pd.read_csv(os.path.join(save_path, "processed", "kline_data", f"{normalized_symbol}.csv"))
    if df.empty:
      continue
    start_date = df.at[0, "date"]
    end_date   = df.at[len(df)-1, "date"]
    all_instruments.append({
      "symbol": normalized_symbol,
      "enlisted_region_start": start_date,
      "enlisted_region_end_incl": end_date,
    })
  all_instruments = pd.DataFrame(all_instruments)
  os.makedirs(os.path.join(save_path, "bin", "instruments"), exist_ok=True)
  all_instruments[["symbol", "enlisted_region_start", "enlisted_region_end_incl"]].to_csv(
    os.path.join(save_path, "bin", "instruments", f"all.txt"), sep='\t', index=False, header=False
  )
  # index
  for index in index_lst:
    enlisted_regions = pd.read_csv(os.path.join(save_path, "processed", "instrument_list", f"{index}.txt"), sep='\t', header=None)
    enlisted_regions.columns = ["symbol", "enlisted_region_start", "enlisted_region_end_incl"]
    enlisted_regions[["symbol", "enlisted_region_start", "enlisted_region_end_incl"]].to_csv(
      os.path.join(save_path, "bin", "instruments", f"{index}.txt"), sep='\t', index=False, header=False
    )
  # 3. features
  for symbol in all_symbols:
    normalized_symbol = normalize_symbol(symbol)
    df = pd.read_csv(os.path.join(save_path, "processed", "kline_data", f"{normalized_symbol}.csv"))
    if df.empty:
      continue
    symbol_features_path = os.path.join(save_path, "bin", "features", normalized_symbol.lower())
    os.makedirs(symbol_features_path, exist_ok=True)
    date_index = trade_calendar.tolist().index(df.at[0, "date"])
    for field in df.columns:
      if field in ["date", "symbol"]:
        continue
      bin_path = os.path.join(symbol_features_path, f"{field.lower()}.day.bin")
      np.concatenate([[date_index], df[field]]).astype("<f").tofile(bin_path)
