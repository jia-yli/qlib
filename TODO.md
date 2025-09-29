# TODO
1. Reproduce Qlib benchmark results on CN dataset
    1. Data Handler
        - How data is loaded: data format, feature generation, NaN, time scale, ...
            - Interaction to storage by `qlib/data/storage/file_storage.py`
            - Expression handlers used in `qlib/data/data.py`
        - How data is fed into model
    2. Result Analysis
        - What is reported and how it is computed
        - What is the strategy and how sim & transaction fee works
2. Reprocuce Qlib benchmark results on CN dataset with custom scripts
3. Extend workflow to crypto dataset

## Step 1: Get New Data: Universe + Top-N Selection
Raw Data
- [TODO]CN: BaoStock
- US: Polygon API? -> ?
- [TODO]Crypto: Way Back Engine + CoinMarketCap -> binance

Test Datast and Loader by `D.features()`, `D.calendar()`

`SH601991` in out csi 300 multiple times. can handle dyn universe

Top-N Selection

