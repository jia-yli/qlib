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

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.utils import deco_retry


@deco_retry(retry_sleep=3, retry=3)
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
        results.append(result)
        seen_comb.add((current_time, page_id))
      except Exception as e:
        logger.warning(f"Fetch failed for page {page_id} at {current_time}: {e}")
    current_time += step
  
  if results:
    df_new = pd.DataFrame(results)
    df = pd.concat([df, df_new], ignore_index=True).drop_duplicates(
      subset=["fetch_timestamp", "page_id"]
    )
    df.to_feather(save_path / "raw.feather")
    df.to_csv(save_path / "raw.csv", index=False)
  return df

def parse_page(html_str):
  soup = BeautifulSoup(html_str, "html.parser")
  table = soup.find("table")
  symbols = [e.get_text(strip=True) for e in table.select("p.coin-item-symbol, span.crypto-symbol")]
  num_symbols = len(symbols)
  num_rows = len(table.find_all("tr"))
  return num_rows


if __name__ == "__main__":
  # Example inputs (edit as needed)
  start = "2021-01-01"
  end   = "2025-10-01"

  save_path = '/capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto'
  save_path = Path(save_path)

  # Step 1: instrument list
  instrument_list_save_path = save_path / "raw" / "instrument_list"
  instrument_list_save_path.mkdir(parents=True, exist_ok=True)

  start_time = pd.Timestamp(start, tz="UTC")
  end_time = pd.Timestamp(end, tz="UTC")
  step = pd.Timedelta(days=365)

  # df = fetch_all_pages(start_time, end_time, step, instrument_list_save_path)
  df = pd.read_feather(instrument_list_save_path / "raw.feather")
  
  import pdb; pdb.set_trace()

  df['count'] = df['content'].apply(parse_page)

  # open("1.html", "w").write(df.loc[1, "content"])
  # # snaps = list_snapshots_between(url, start, end)
  # url = "https://web.archive.org/web/20240216024121/https://coinmarketcap.com/"
  # url = "https://web.archive.org/web/20250810032118/https://coinmarketcap.com/?page=6"
  # r = requests.get(url)
  # r.url

  import pdb; pdb.set_trace()
  # print(json.dumps(snaps, indent=2))
