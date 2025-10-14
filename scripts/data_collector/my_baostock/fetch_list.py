import sys
import pandas as pd
import baostock as bs
from pathlib import Path
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.utils import deco_retry

@deco_retry(retry_sleep=5, retry=4)
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

def get_index_stocks(self, index):
    index_stocks = set()
    appearance_spans_df = pd.DataFrame(columns=[
        "code",
        "first_appear_date_idx", "first_appear_date",
        "last_appear_date_idx", "last_appear_date",
    ])
    last_seg_row_idx = {}
    trade_calendar = self.get_trade_calendar()
    with tqdm(total=len(trade_calendar)) as p_bar:
        for date_idx, date in enumerate(trade_calendar):
            @deco_retry(retry_sleep=5, retry=5)
            def fetch_step_stocks(index, date):
                rs = eval(f"bs.query_{index}_stocks")(date=date)
                step_stocks_lst = []
                if rs.error_code != "0":
                    raise ValueError(f"fetch {index} {date} stocks error, error_code: {rs.error_code}, error_msg: {rs.error_msg}")
                while rs.next():
                    row = rs.get_row_data()
                    step_stocks_lst.append(
                        {
                            "date": date,
                            **{k: v for k, v in zip(rs.fields, row)},
                        }
                    )
                step_stocks_df = pd.DataFrame(step_stocks_lst)
                return step_stocks_df
            step_stocks_df = fetch_step_stocks(index, date)

            todays_codes = step_stocks_df["code"].unique().tolist()
            # update index stocks
            index_stocks.update(todays_codes)
            # update appearance spans
            for code in todays_codes:
                if code not in last_seg_row_idx.keys():
                    # first time we ever see this code -> start a new segment
                    new_row = {
                        "code": code,
                        "first_appear_date_idx": date_idx,
                        "first_appear_date": date,
                        "last_appear_date_idx": date_idx,
                        "last_appear_date": date,
                    }
                    appearance_spans_df.loc[len(appearance_spans_df)] = new_row
                    last_seg_row_idx[code] = len(appearance_spans_df) - 1
                else:
                    r = last_seg_row_idx[code]
                    last_idx = int(appearance_spans_df.at[r, "last_appear_date_idx"])

                    if last_idx == date_idx - 1:
                        # "+1 for places where end idx is current - 1" -> extend the same segment
                        appearance_spans_df.at[r, "last_appear_date_idx"] = date_idx
                        appearance_spans_df.at[r, "last_appear_date"] = date
                    elif last_idx < date_idx - 1:
                        # Not contiguous (there is a gap) -> start a *new* segment for this code
                        new_row = {
                            "code": code,
                            "first_appear_date_idx": date_idx,
                            "first_appear_date": date,
                            "last_appear_date_idx": date_idx,
                            "last_appear_date": date,
                        }
                        appearance_spans_df.loc[len(appearance_spans_df)] = new_row
                        last_seg_row_idx[code] = len(appearance_spans_df) - 1
                    # If last_idx >= date_idx, dates are out of order; we ignore since enumerate is increasing.
            p_bar.update()

    save_dir = self.save_dir.joinpath("metadata")
    save_dir.mkdir(parents=True, exist_ok=True)
    appearance_spans_df.to_csv(save_dir.joinpath(f"{index}_symbol_record.csv"), index=False)
    appearance_spans_df["code"] = appearance_spans_df["code"].apply(self.normalize_symbol)
    appearance_spans_df[["code", "first_appear_date", "last_appear_date"]].to_csv(
        save_dir.joinpath(f"{index}.txt"), sep='\t', index=False, header=False
    )
    return sorted(index_stocks)

def get_instrument_list(self):
    index_lst = ['sz50', 'hs300', 'zz500', 'all']
    index_symbols = ['sh.000016', 'sh.000300', 'sh.000905']  # sz50, hs300, zz500

    stock_symbols = []
    for index in index_lst:
        logger.info(f"getting {index} stock symbols......")
        index_stocks = self.get_index_stocks(index)
        logger.info(f"get {len(index_stocks)} symbols for {index}.")
        stock_symbols.extend(index_stocks)
    
    symbol_lst = sorted(list(set(stock_symbols + index_symbols)))
    logger.info(f"total {len(symbol_lst)} symbols.")

    return symbol_lst

@deco_retry(retry_sleep=5, retry=4)
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



# def fetch_all_records(start, end, index_lst):
#     trade_calendar = get_trade_calendar(start, end)
#     for index in index_lst:
#         fetch_records(trade_calendar, index)
    # hs300_stocks = []
    #     trade_calendar = self.get_trade_calendar()
    #     with tqdm(total=len(trade_calendar)) as p_bar:
    #         for date in trade_calendar:
    #             rs = bs.query_hs300_stocks(date=date)
    #             while rs.error_code == "0" and rs.next():
    #                 hs300_stocks.append(rs.get_row_data())
    #             p_bar.update()
    #     return sorted({e[1] for e in hs300_stocks})


if __name__ == "__main__":
    start = "2019-01-01"
    end   = "2025-10-01"

    save_path = '/capstor/scratch/cscs/ljiayong/datasets/qlib/my_baostock'
    save_path = Path(save_path)

    index_lst = ['sz50', 'hs300', 'zz500']
    index_symbols = ['sh.000016', 'sh.000300', 'sh.000905']  # sz50, hs300, zz500

    # Step 1: instrument list
    instrument_list_save_path = save_path / "raw" / "instrument_list"
    instrument_list_save_path.mkdir(parents=True, exist_ok=True)

    bs.login()
    trade_calendar = get_trade_calendar(start, end)
    for index in index_lst:
        df = fetch_records(trade_calendar, index, instrument_list_save_path)
        # import pdb;pdb.set_trace()
    bs.logout()
