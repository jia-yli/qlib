import baostock as bs
import pandas as pd

lg = bs.login()
if lg.error_code != '0':
  print('Login failed. Error code:', lg.error_code)
  print('Error message:', lg.error_msg)
  exit()

# def fetch_hs300_stocks(date=None):
#   if date:
#     rs = bs.query_hs300_stocks(date)
#   else:
#     rs = bs.query_hs300_stocks()
  
#   if rs.error_code != '0':
#     print('Error querying HS300 stocks:', rs.error_msg)
#     return pd.DataFrame()
  
#   stocks = []
#   while (rs.error_code == '0') and rs.next():
#     stocks.append(rs.get_row_data())
  
#   return pd.DataFrame(stocks, columns=rs.fields)

# hs300_stocks = fetch_hs300_stocks()

def fetch_day_kline(code, start_date, end_date):
  rs = bs.query_history_k_data_plus(
    code,
    "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
    start_date=start_date, 
    end_date=end_date,
    frequency="d",
    adjustflag="3")
  
  if rs.error_code != '0':
    print(f'Error querying daily k data for {code}:', rs.error_msg)
    return pd.DataFrame()
  
  data_list = []
  while (rs.error_code == '0') and rs.next():
    data_list.append(rs.get_row_data())
  
  return pd.DataFrame(data_list, columns=rs.fields)

data = fetch_day_kline("sh.600000", '2025-09-25', '2025-09-25')

import pdb;pdb.set_trace()

bs.logout()