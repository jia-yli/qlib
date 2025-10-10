# TODO
1. LGBM @ CN
    - executor-exchange-strategy interaction loop
        - Add, not Mult, it is a bug and lead to wrong results (but keep it for now)
        - Need to resolve all warning and errors in plotting
2. All Models @ CN
    - Model selection? (rolling cv, ...)
3. LGBM @ Crypto
    - diff freq
4. All Models @ Crypto
5. All Models @ US
6. Deploy
7. Extend Workflow RD Agent, Custom Op, Custom Model
8. Extend Dataset (News, ...)

## Get New Data: Universe + Top-N Universe Selection
Raw Data
- CN: BaoStock
- [TODO]US: Way Back Engine + slickcharts -> YahooFinance/Alphavintage/Alpaca
- [TODO]Crypto: Way Back Engine + CoinMarketCap -> binance

News
- CN: http://mrxwlb.com/, cn.govopendata.com/, paper.people.com.cn
- US: Alpaca?
- Crypto: Alpaca?

Test Datast and Loader by `D.features()`, `D.calendar()`

`SH601991` in out csi 300 multiple times. can handle dyn universe

## Op Ideas
1. Cross Section Ops
2. CAPM (Alpha, Beta, Resi) Compare to Benchmark, C-/I- CAPM, APT
3. Jensen alpha: Substract Benchmark Baseline

## Data Source
1. financialdata.net: free stock list?

## Bugs
1. If asset left universe, position stucks when no price info. 
    - should ffill in some part: position is last because of NaN score and exit by last close
2. all metrics in backtest are using add, not mult. All of them are wrong.
    - `/users/ljiayong/projects/qlib/qlib/backtest/account.py` L269
        - `/users/ljiayong/projects/qlib/qlib/backtest/position.py`
    - `/users/ljiayong/projects/qlib/qlib/backtest/report.py` L153
    - `/users/ljiayong/projects/qlib/qlib/contrib/report/analysis_position/report.py` L35
    - `/users/ljiayong/projects/qlib/qlib/contrib/evaluate.py` L27
3. day per year: CN246, US252, CRYPTO365