# White Maize Price Prediction (South Africa) / SAFEX Futures Forecasting

## The Challenge

White maize is South Africa's staple grain, and its price on the **SAFEX** commodity futures market (now part of the JSE) moves with global grain markets, the rand, the weather and the cost of getting the crop to market. For a farmer, a decision on *when* to sell the crop, hold it or hedge it can be worth a great deal of money.

This project forecasts the **SAFEX white maize price (R/ton)** from its own history plus South African **external drivers**:

| Driver | Why it matters |
|---|---|
| **ZAR/USD exchange rate** | Maize is priced against world markets (import / export parity), so a weaker rand tends to lift local prices |
| **Rainfall (mm)** | Rain in the production regions during the season drives crop size, and crop size drives price |
| **Fuel price** | Fuel feeds into farming costs and into transport and logistics costs |

The models range from classical time-series methods to tree ensembles and recurrent neural networks, and they are all evaluated the same way so they can be compared fairly.

> **Origin.** This project is adapted from the MinneMUDAC 2019 soybean-futures forecasting solution ([original repository](https://github.com/guptapiyush340/Soybean-Price-Prediction---MinneMUDAC-winning-solution)). The model line-up and project structure carry over; the data, features, evaluation and the farmer SELL / HOLD analysis were reworked for the South African white maize market. Results quoted in the original project belong to soybeans and do not apply here.

## Data

Two CSV files, both with a `Date` column, are expected in the project folder:

| File | Columns | Notes |
|---|---|---|
| `safex_prices.csv` | `Date`, `Close` (R/ton) | Optional extras: `Open`, `High`, `Low`, `Volume`. If `High` and `Low` exist, the XGBoost notebook adds the daily trading range as a feature. |
| `sa_factors.csv` | `Date`, `ZAR_USD`, `Rainfall_mm`, `Fuel_Price` | May be daily, weekly or monthly, and may live in one wide file with blanks between observations |

Practical notes:

* **Contract continuity.** SAFEX white maize trades as futures contracts with fixed delivery months. If your price series is stitched from several contracts, back-adjust it or work with one contract at a time; otherwise the jumps at the roll dates look like price moves to the models.
* **Rainfall** is most informative when averaged over the main production areas (Free State, North West and Mpumalanga) rather than taken nationally.
* **Dates** are read as month-first by default. If yours are `dd/mm/yyyy`, add `dayfirst=True` to the `read_csv` calls at the top of each notebook.
* **Column names** are set once at the top of each notebook (`TARGET_COL`, `EXOG_COLS`) if yours differ.

### Alignment (identical in every notebook)

```python
prices = pd.read_csv('safex_prices.csv', parse_dates=['Date'], index_col='Date')
exog_data = pd.read_csv('sa_factors.csv', parse_dates=['Date'], index_col='Date')

exog_data = exog_data.resample('D').last().ffill()      # safeguard, explained below
data = prices.join(exog_data, how='inner')              # keep dates present in both
data = data.resample('B').ffill()                       # business-day calendar, forward-fill gaps
```

The safeguard line matters when a driver is *less frequent* than prices (monthly fuel prices, for instance). Joining first would leave only about twelve matching dates per year and discard the daily prices. Carrying the last known driver value forward to a daily calendar *before* the join avoids that. Business days that are public holidays carry the previous price forward, so you will see a few repeated prices.

## Process Overview

### Machine Learning Journey

**We experiment with multiple prediction algorithms**

| # | Notebook | Model | How it uses the external drivers |
|---|---|---|---|
| 1 | [SARIMAX.ipynb](SARIMAX.ipynb) | SARIMAX (replaces ARIMA) | Exogenous regressors; order tuned by AIC; compared with a driver-free fit |
| 2 | [Prophet.ipynb](Prophet.ipynb) | Prophet | Extra regressors, South African public holidays, yearly (harvest) seasonality |
| 3 | [XGBoost_Final_Model.ipynb](XGBoost_Final_Model.ipynb) | XGBoost with hyper-parameter tuning | Lagged drivers as features; feature importance; SELL / HOLD back-test |
| 4 | [LSTM.ipynb](LSTM.ipynb) | Stacked LSTM | Price and drivers as a multivariate 60-day window |
| 5 | [Seq2Seq.ipynb](Seq2Seq.ipynb) | Encoder-decoder LSTM | Drivers feed the encoder; forecasts the next five trading days |

The original project finished with XGBoost because neural networks need a lot of data and because XGBoost's feature importance shows *which* inputs drive the forecast. The same trade-off applies here, but the comparison table below, filled in from your own runs, should make the final call.

### Common evaluation protocol

* **Chronological 80 / 20 split.** The oldest 80% of days train the model and the newest 20% test it. Nothing is shuffled.
* **Metrics** are MAE and RMSE in rand per ton, plus MAPE (%).
* **Benchmarks.** The naive forecast (the price stays where it is) is reported next to every model. Where the notebook supports it (SARIMAX, Prophet, LSTM), the same model *without* the external drivers is shown too. A model earns its place by beating them.
* **No look-ahead.** The tree and neural models build their inputs only from information available when the forecast would have been made. The scalers are fitted on the training block only.

### Results

Fill in after running the notebooks (each notebook prints its own metrics table).

| Model | Forecast style | MAE (R/ton) | RMSE (R/ton) | MAPE % | Beats naive? |
|---|---|---|---|---|---|
| SARIMAX + drivers | multi-step over the test block | | | | |
| SARIMAX + drivers | one-step-ahead | | | | |
| Prophet + drivers | multi-step over the test block | | | | |
| XGBoost | 1-day ahead | | | | |
| LSTM + drivers | 1-day ahead | | | | |
| Seq2Seq | 5 days ahead (average over days 1 to 5) | | | | |
| Naive benchmark | same horizon as the row it is compared with | | | | |

Rows with different forecast styles are not directly comparable: predicting one day ahead is much easier than predicting a whole test block ahead.

### Profitability for the farmer

The XGBoost notebook ends with a SELL / HOLD back-test. Say a farmer has 100 tons of white maize to sell. For each day of the test block the model forecasts the price five business days ahead:

* forecast **above** today's price plus the cost of holding the maize: **HOLD** and sell five days later
* otherwise: **SELL** today

The back-test compares the cumulative rand gain with "always sell today", "always hold" and a perfect-foresight upper bound. Storage and financing costs differ between farms, so set `HOLD_COST_R_PER_TON` to your own figure; at its default of zero, HOLD is flattered.

Results from your run: _to be added_.

### Reading the scores honestly

* **SARIMAX and Prophet are given the actual driver values for the test block.** That shows how well prices follow the drivers *if the drivers are known*; it is not a pure forecast. For a real forward forecast, future driver values must be supplied as **scenarios** (for example, a rand 5% weaker). Both notebooks include a forward-forecast section built this way.
* **XGBoost, LSTM and Seq2Seq use only lagged information**, so their scores are genuine forecasts at their stated horizon.
* **Commodity prices are close to a random walk.** A model that cannot beat the naive benchmark is not adding value, however good its R-squared looks.

## Known limitations

* Trees (XGBoost) cannot predict outside the range of values they were trained on. The notebook therefore predicts the price *change* by default (`PREDICT = 'diff'`).
* The LSTM notebook scales prices with a min-max scaler fitted on the training block, so it extrapolates poorly if prices in the test block climb well above anything in training; the Seq2Seq notebook avoids this by working with relative prices.
* Neural networks are data-hungry, and a few years of daily prices is not much data for them.
* Weather and fuel data usually arrive at lower frequency than prices; forward-filling them is a simplification.

## Repository layout

```
README.md
requirements.txt
safex_prices.csv              (your data)
sa_factors.csv                (your data)
SARIMAX.ipynb
Prophet.ipynb
XGBoost_Final_Model.ipynb
LSTM.ipynb
Seq2Seq.ipynb
```

## Acknowledgements

Adapted from the MinneMUDAC 2019 soybean price prediction project (first place, Graduate Student division), by Harsh Seksaria, Piyush Gupta, Hamed Khoojinian, Yassine Manane and Pushkar Vengulekar.
