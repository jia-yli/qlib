# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import sys
import copy
import fire
import numpy as np
import pandas as pd
import baostock as bs
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Iterable, List


CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.base import BaseCollector, BaseNormalize, BaseRun
from data_collector.utils import deco_retry

class MyBaostockCollector1d(BaseCollector):
    def __init__(
        self,
        save_dir: [str, Path],
        start=None,
        end=None,
        interval="5min",
        max_workers=4,
        max_collector_count=2,
        delay=0,
        check_data_length: int = None,
        limit_nums: int = None,
    ):
        """

        Parameters
        ----------
        save_dir: str
            stock save dir
        max_workers: int
            workers, default 4
        max_collector_count: int
            default 2
        delay: float
            time.sleep(delay), default 0
        interval: str
            freq, value from [5min], default 5min
        start: str
            start datetime, default None
        end: str
            end datetime, default None
        check_data_length: int
            check data length, by default None
        limit_nums: int
            using for debug, by default None
        """
        bs.login()
        super(MyBaostockCollector1d, self).__init__(
            save_dir=save_dir,
            start=start,
            end=end,
            interval=interval,
            max_workers=max_workers,
            max_collector_count=max_collector_count,
            delay=delay,
            check_data_length=check_data_length,
            limit_nums=limit_nums,
        )
        self.raw_dir = save_dir
        self.raw_dir = Path(self.raw_dir).expanduser().resolve().joinpath("raw")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def get_trade_calendar(self):
        _format = "%Y-%m-%d"
        start = self.start_datetime.strftime(_format)
        end = self.end_datetime.strftime(_format)
        rs = bs.query_trade_dates(start_date=start, end_date=end)
        calendar_list = []
        while (rs.error_code == "0") & rs.next():
            calendar_list.append(rs.get_row_data())
        calendar_df = pd.DataFrame(calendar_list, columns=rs.fields)
        trade_calendar_df = calendar_df[~calendar_df["is_trading_day"].isin(["0"])]
        return trade_calendar_df["calendar_date"].values

    @deco_retry(retry_sleep=5, retry=5)
    def get_data(
        self, symbol: str, interval: str, start_datetime: pd.Timestamp, end_datetime: pd.Timestamp
    ) -> pd.DataFrame:
        df = pd.DataFrame()
        rs = bs.query_history_k_data_plus(
            symbol,
            "date,code,open,high,low,close,volume,amount,turn",
            start_date=str(start_datetime.strftime("%Y-%m-%d")),
            end_date=str(end_datetime.strftime("%Y-%m-%d")),
            frequency="d",
            adjustflag="3",
        )
        if rs.error_code == "0" and len(rs.data) > 0:
            data_list = rs.data
            columns = rs.fields
            df = pd.DataFrame(data_list, columns=columns)
        else:
            raise ValueError(f"fetch {symbol} data error, error_code: {rs.error_code}, error_msg: {rs.error_msg}")
        rs_adj = bs.query_history_k_data_plus(
            symbol,
            "date,close",
            start_date=str(start_datetime.strftime("%Y-%m-%d")),
            end_date=str(end_datetime.strftime("%Y-%m-%d")),
            frequency="d",
            adjustflag="1",
        )
        if rs_adj.error_code == "0" and len(rs_adj.data) > 0:
            data_list = rs_adj.data
            columns = ["date", "adjclose"]
            df_adj = pd.DataFrame(data_list, columns=columns)
            df = pd.merge(df, df_adj, on="date", how="left")
        else:
            raise ValueError(f"fetch {symbol} adjclose error, error_code: {rs_adj.error_code}, error_msg: {rs_adj.error_msg}")

        df = df.rename(columns={"code": "symbol"})
        df["symbol"] = df["symbol"].apply(self.normalize_symbol)
        df.to_csv(self.raw_dir.joinpath(f"{self.normalize_symbol(symbol)}.csv"), index=False)

        df = pd.read_csv(self.raw_dir.joinpath(f"{self.normalize_symbol(symbol)}.csv"))
        # process df
        df["factor"] = df["adjclose"].astype(float) / df["close"].astype(float)
        df["factor"] = df["factor"].ffill()
        df.drop(columns=["adjclose"], inplace=True)
        for col in df.columns:
            if col in ["date", "symbol", "amount", "turn", "factor"]:
                pass
            elif col in ["open", "high", "low", "close"]:
                df[col] = df[col].astype(float) * df["factor"]
            elif col in ["volume"]:
                df[col] = df[col].astype(float) / df["factor"]
            else:
                raise ValueError(f"Unknown column: {col}")

        df["change"] = df["close"] / df["close"].shift(1) - 1
        return df

    def get_index_stocks(self, index: str) -> List[str]:
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
        # index_lst = ['sz50', 'hs300', 'zz500']
        # index_symbols = ['sh.000016', 'sh.000300', 'sh.000905']  # sz50, hs300, zz500
        index_lst = ['hs300']
        index_symbols = ['sh.000300']  # sz50, hs300

        stock_symbols = []
        for index in index_lst:
            logger.info(f"getting {index} stock symbols......")
            index_stocks = self.get_index_stocks(index)
            logger.info(f"get {len(index_stocks)} symbols for {index}.")
            stock_symbols.extend(index_stocks)
        
        symbol_lst = sorted(list(set(stock_symbols + index_symbols)))
        logger.info(f"total {len(symbol_lst)} symbols.")

        return symbol_lst

    def normalize_symbol(self, symbol: str):
        return str(symbol).replace(".", "").upper()


class Run(BaseRun):
    def __init__(self, source_dir=None, normalize_dir=None, max_workers=1, interval="1d"):
        """
        Changed the default value of: scripts.data_collector.base.BaseRun.
        """
        super().__init__(source_dir, normalize_dir, max_workers, interval)

    @property
    def collector_class_name(self):
        return f"MyBaostockCollector{self.interval}"

    @property
    def normalize_class_name(self):
        return f"BaostockNormalize{self.region.upper()}{self.interval}"

    @property
    def default_base_dir(self) -> [Path, str]:
        return CUR_DIR

    def download_data(
        self,
        max_collector_count=2,
        delay=0.5,
        start=None,
        end=None,
        check_data_length=None,
        limit_nums=None,
    ):
        """download data from Baostock

        Notes
        -----
            check_data_length, example:
                hs300 5min, a week: 4 * 60 * 5

        Examples
        ---------
            # get hs300 5min data
            $ python collector.py download_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --start 2022-01-01 --end 2022-01-30 --interval 5min --region HS300
        """
        super(Run, self).download_data(max_collector_count, delay, start, end, check_data_length, limit_nums)

    def normalize_data(
        self,
        date_field_name: str = "date",
        symbol_field_name: str = "symbol",
        end_date: str = None,
        qlib_data_1d_dir: str = None,
    ):
        """normalize data

        Attention
        ---------
        qlib_data_1d_dir cannot be None, normalize 5min needs to use 1d data;

            qlib_data_1d can be obtained like this:
                $ python scripts/get_data.py qlib_data --target_dir ~/.qlib/qlib_data/cn_data --interval 1d --region cn --version v3
            or:
                download 1d data, reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#1d-from-yahoo

        Examples
        ---------
            $ python collector.py normalize_data --qlib_data_1d_dir ~/.qlib/qlib_data/cn_data --source_dir ~/.qlib/stock_data/source/hs300_5min_original --normalize_dir ~/.qlib/stock_data/source/hs300_5min_nor --region HS300 --interval 5min
        """
        if qlib_data_1d_dir is None or not Path(qlib_data_1d_dir).expanduser().exists():
            raise ValueError(
                "If normalize 5min, the qlib_data_1d_dir parameter must be set: --qlib_data_1d_dir <user qlib 1d data >, Reference: https://github.com/microsoft/qlib/tree/main/scripts/data_collector/yahoo#automatic-update-of-daily-frequency-datafrom-yahoo-finance"
            )
        super(Run, self).normalize_data(
            date_field_name, symbol_field_name, end_date=end_date, qlib_data_1d_dir=qlib_data_1d_dir
        )


if __name__ == "__main__":
    fire.Fire(Run)
