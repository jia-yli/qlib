import re
import sys
import json
import requests
import pandas as pd
from pathlib import Path
from loguru import logger

CUR_DIR = Path(__file__).resolve().parent
sys.path.append(str(CUR_DIR.parent.parent))

from data_collector.utils import deco_retry


@deco_retry(retry_sleep=5, retry=5)
def fetch_page(url):
  resp = requests.get(url)
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

# --------- tiny demo (edit these or import the functions) ----------
if __name__ == "__main__":
  # Example inputs (edit as needed)
  start = "2021-01-01"
  end   = "2025-10-01"

  start_time = pd.Timestamp(start, tz="UTC")
  end_time = pd.Timestamp(end, tz="UTC")
  step = pd.Timedelta(days=4*365)

  current_time = start_time
  results = []
  while current_time < end_time:
    base_url = "https://web.archive.org/web/"
    target_url = "https://coinmarketcap.com/"
    time_stamp = current_time.strftime("%Y%m%d%H%M%S")
    url_lst = [f"{base_url}{time_stamp}/{target_url}/"] + [f"{base_url}{time_stamp}/{target_url}/?page={n}" for n in range(2, 7)] # first 6 pages

    for page_id, url in enumerate(url_lst):
      logger.info(f"Getting page {page_id} for {current_time} ......")
      result = fetch_page(url)
      result['page_id'] = page_id
      results.append(result)

    current_time += step
  df = pd.DataFrame(results)
  import pdb; pdb.set_trace()

  # parse content

  # # snaps = list_snapshots_between(url, start, end)
  # url = "https://web.archive.org/web/20240216024121/https://coinmarketcap.com/"
  # url = "https://web.archive.org/web/20250810032118/https://coinmarketcap.com/?page=6"
  # r = requests.get(url)
  # r.url

  import pdb; pdb.set_trace()
  # print(json.dumps(snaps, indent=2))
