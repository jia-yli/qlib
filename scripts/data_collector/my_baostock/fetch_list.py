import sys
import pandas as pd
import baostock as bs
from pathlib import Path
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.utils import deco_retry

@deco_retry(retry_sleep=5, retry=2)
def get_trade_calendar(start, end):
  rs = bs.query_trade_dates(start_date=start, end_date=end)
  if rs.error_code == "0":
    data_list = rs.data
    columns = rs.fields
    calendar_df = pd.DataFrame(data_list, columns=columns)
  else:
    raise ValueError(f"Error when fetching calendar from {start} to {end}, code: {rs.error_code}")
  trade_calendar_df = calendar_df[~calendar_df["is_trading_day"].isin(["0"])]
  return trade_calendar_df["calendar_date"].values

@deco_retry(retry_sleep=5, retry=2)
def fetch_stocks(date, index):
  if index in ['sz50', 'hs300', 'zz500']:
    rs = eval(f"bs.query_{index}_stocks")(date=date)
  elif index == 'all':
    rs = eval(f"bs.query_{index}_stock")(day=date)
  else:
    raise ValueError(f"index {index} not supported")

  if rs.error_code == "0":
    data_list = rs.data
    columns = rs.fields
    stocks_df = pd.DataFrame(data_list, columns=columns)
  else:
    raise ValueError(f"Error when fetching stock symbols for {index} index at {date}, code: {rs.error_code}")
  return stocks_df

def fetch_records(calendar, index, save_path):
  save_path.mkdir(parents=True, exist_ok=True)
  if (save_path / f"{index}_raw.feather").exists():
    df = pd.read_feather(save_path / f"{index}_raw.feather")
  else:
    df = pd.DataFrame()

  seen_dates = set(df.get('fetch_date', []))

  logger.info(f"Fetching {index} records......")
  results = []
  for date in calendar:
    if date in seen_dates:
      logger.info(f"{date} {index} record already seen")
      continue
    logger.info(f"Fetching {date} {index} record ......")
    try:
      stocks_df = fetch_stocks(date, index)
      stocks_df["fetch_date"] = date
      results.append(stocks_df)
      seen_dates.add(date)
    except Exception as e:
      logger.warning(f"Fetch failed for {date} {index} record: {e}")

  if results:
    df = pd.concat([df] + results, ignore_index=True)
    df = df.sort_values(["fetch_date", "code"])
    df = df.drop_duplicates(subset=["fetch_date", "code"])
    df.to_feather(save_path / f"{index}_raw.feather")
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
def get_kline_data(symbol, start_time, end_time):
  logger.info(f"Fetching {symbol} {start_time} to {end_time} daily data")
  rs = bs.query_history_k_data_plus(
    symbol,
    "date,code,open,high,low,close,volume,amount,turn,tradestatus,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST",
    start_date=str(start_time.strftime("%Y-%m-%d")),
    end_date=str(end_time.strftime("%Y-%m-%d")),
    frequency="d",
    adjustflag="3",
  )
  if rs.error_code == "0":
    data_list = rs.data
    columns = rs.fields
    df = pd.DataFrame(data_list, columns=columns)
  else:
    raise ValueError(f"fetch {symbol} data error, error_code: {rs.error_code}, error_msg: {rs.error_msg}")
  rs_adj = bs.query_history_k_data_plus(
    symbol,
    "date,close",
    start_date=str(start_time.strftime("%Y-%m-%d")),
    end_date=str(end_time.strftime("%Y-%m-%d")),
    frequency="d",
    adjustflag="1",
  )
  if rs_adj.error_code == "0":
    data_list = rs_adj.data
    columns = ["date", "adjclose"]
    df_adj = pd.DataFrame(data_list, columns=columns)
    df = pd.merge(df, df_adj, on="date", how="left")
  else:
    raise ValueError(f"fetch {symbol} adjclose error, error_code: {rs_adj.error_code}, error_msg: {rs_adj.error_msg}")
  return df

def normalize_symbol(symbol):
  return str(symbol).replace(".", "").upper()

if __name__ == "__main__":
  start = "2019-01-01" # inc
  end   = "2025-10-01" # inc

  start_time = pd.Timestamp(start)
  end_time = pd.Timestamp(end)

  save_path = '/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock'
  save_path = Path(save_path)

  index_lst = ['sz50', 'hs300', 'zz500']
  index_symbols = ['sh.000016', 'sh.000300', 'sh.000905']  # sz50, hs300, zz500

  instrument_list_save_path = save_path / "raw" / "instrument_list"
  instrument_list_save_path.mkdir(parents=True, exist_ok=True)

  raw_kline_path = save_path / "raw"
  raw_kline_path.mkdir(parents=True, exist_ok=True)

  processed_kline_path = save_path / "processed"
  processed_kline_path.mkdir(parents=True, exist_ok=True)

  '''
  1. Fetch Raw Instrument List Info
  '''
  # bs.login()
  # trade_calendar = get_trade_calendar(start, end)
  # for index in index_lst:
  #   df = fetch_records(trade_calendar, index, instrument_list_save_path)
  # bs.logout()

  '''
  2. Get Index Components
  '''
  # symbols = set()
  # for index in index_lst:
  #   df = pd.read_feather(instrument_list_save_path / f"{index}_raw.feather")
  #   df['date'] = pd.to_datetime(df['fetch_date'])
  #   df['is_enlisted'] = 1
  #   enlisted_state = (
  #     pd.pivot_table(
  #       df,
  #       values = 'is_enlisted',
  #       index = 'date',
  #       columns = 'code',
  #       aggfunc = 'sum',
  #       fill_value = 0,
  #     )
  #     .astype(bool)
  #     .sort_index()
  #   )
  #   enlisted_state.to_feather(instrument_list_save_path / f"{index}_enlisted_state.feather")
  #   enlisted_regions = extract_enlisted_regions(enlisted_state)
  #   symbols.update(enlisted_regions['symbol'].unique())

  # symbols.update(index_symbols)
  # all_symbols = sorted(symbols)

  '''
  3. Fetch Klines for Symbols
  '''
  # bs.login()
  # for symbol in all_symbols:
  #   df = get_kline_data(symbol, start_time, end_time)
  #   df.to_csv(raw_kline_path / f"{symbol}.csv", index=False)
  # bs.logout()  

  '''
  4. Make Data Dump-Ready
  '''
  symbol_klines = {}
  trade_calendar = None
  for i, index in enumerate(index_lst):
    enlisted_state = pd.read_feather(instrument_list_save_path / f"{index}_enlisted_state.feather")
    trade_calendar = enlisted_state.index
    enlisted_regions = extract_enlisted_regions(enlisted_state)

    symbols = [index_symbols[i]] + list(enlisted_regions["symbol"].unique())
    for symbol in symbols:
      if symbol not in symbol_klines.keys():
        symbol_klines[symbol] = pd.read_csv(raw_kline_path / f"{symbol}.csv")

    enlisted_regions["symbol"] = enlisted_regions["symbol"].apply(normalize_symbol)
    enlisted_regions["enlisted_region_start"] = enlisted_regions["enlisted_region_start"].dt.strftime("%Y-%m-%d")
    enlisted_regions["enlisted_region_end_incl"] = enlisted_regions["enlisted_region_end_incl"].dt.strftime("%Y-%m-%d")
    enlisted_regions[["symbol", "enlisted_region_start", "enlisted_region_end_incl"]].to_csv(
      instrument_list_save_path / f"{index}.txt", sep='\t', index=False, header=False
    )

  for symbol, df in symbol_klines.items():
    normalized_symbol = normalize_symbol(symbol)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    df = df.reindex(trade_calendar[(trade_calendar >= df.index.min()) & (trade_calendar <= df.index.max())])
    '''
    date,code,open,high,low,close,volume,amount,turn,tradestatus,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST,adjclose
    '''
    df.insert(0, 'date', df.index.strftime("%Y-%m-%d"))
    df.rename(columns={"code": "symbol"}, inplace=True)
    df["symbol"] = normalized_symbol
    # fill missing
    df["close"] = df["close"].ffill()
    # fill ohl with last close
    df["open"] = df["open"].combine_first(df["close"])
    df["high"] = df["high"].combine_first(df["close"])
    df["low"]  = df["low"] .combine_first(df["close"])
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
      elif col in ["open", "high", "low", "close"]:
        df[col] = df[col] * df["factor"]
      elif col in ["volume"]:
        df[col] = df[col] / df["factor"]
      else:
        raise ValueError(f"Unknown column: {col}")

    df["change"] = df["close"] / df["close"].shift(1) - 1

    df.to_csv(processed_kline_path / f"{normalized_symbol}.csv", index=False)


