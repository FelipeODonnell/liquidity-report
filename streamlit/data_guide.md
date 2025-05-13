# Izun Liquidity Report - Data Guide

This comprehensive guide describes all data available in the Izun Liquidity Report project. The guide is specifically designed for LLMs to better understand, visualize, and analyze the cryptocurrency market data collected via the CoinGlass API.

## Table of Contents

1. [Data Structure Overview](#data-structure-overview)
2. [Data Format](#data-format)
3. [Timestamp Format](#timestamp-format)
4. [ETF Data](#etf-data)
5. [Futures Data](#futures-data)
   - [Funding Rate](#funding-rate)
   - [Liquidation](#liquidation)
   - [Long-Short Ratio](#long-short-ratio)
   - [Market Data](#market-data)
   - [Open Interest](#open-interest)
   - [Order Book](#order-book)
   - [Taker Buy-Sell](#taker-buy-sell)
6. [Market Indicators](#market-indicators)
7. [On-Chain Data](#on-chain-data)
8. [Options Data](#options-data)
9. [Spot Market Data](#spot-market-data)
   - [Order Book](#spot-order-book)
   - [Spot Market](#spot-market)
   - [Taker Buy-Sell](#spot-taker-buy-sell)
10. [Visualizing the Data](#visualizing-the-data)
11. [Data Analysis Methods](#data-analysis-methods)
12. [Correlation Analysis](#correlation-analysis)
13. [Market Trend Analysis](#market-trend-analysis)

## Data Structure Overview

The data is organized in a date-based folder structure:

```
data/
└── YYYYMMDD/  # Date format: YYYYMMDD (e.g., 20250512)
    ├── etf/
    ├── futures/
    │   ├── funding_rate/
    │   ├── liquidation/
    │   ├── long_short_ratio/
    │   ├── market/
    │   ├── open_interest/
    │   ├── order_book/
    │   └── taker_buy_sell/
    ├── indic/
    ├── on_chain/
    ├── options/
    └── spot/
        ├── order_book_spot/
        ├── spot_market/
        └── taker_buy_sell_spot/
```

Each folder contains Parquet files with data from different CoinGlass API endpoints. The data is focused on four main cryptocurrencies:
- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)
- Ripple (XRP)

## Data Format

All data is stored in Parquet format, which offers several advantages:
- Efficient columnar storage
- Reduced file size through compression
- Fast query performance
- Schema preservation
- Support for complex nested data structures

To read the data in Python:
```python
import pandas as pd

# Example path
file_path = 'data/20250512/etf/api_etf_bitcoin_list.parquet'
df = pd.read_parquet(file_path)
```

## Timestamp Format

Most timestamp fields in the datasets are stored as milliseconds since the UNIX epoch (January 1, 1970). To convert to a readable datetime in Python:

```python
import pandas as pd
from datetime import datetime

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['time'], unit='ms')
```

## ETF Data

The `etf` folder contains data related to Bitcoin and Ethereum Exchange-Traded Funds (ETFs).

### Bitcoin ETF Files

| File | Description | Key Fields |
|------|-------------|------------|
| `api_etf_bitcoin_aum.parquet` | Assets Under Management (AUM) for Bitcoin ETFs | ticker, fund_name, aum_usd, timestamp |
| `api_etf_bitcoin_flow_history.parquet` | Historical fund flows for Bitcoin ETFs | time, fund_flow_usd, price_change_percent |
| `api_etf_bitcoin_history.parquet` | Historical data for Bitcoin ETFs | time, aum_total, fund_flow_total, price_change_percent |
| `api_etf_bitcoin_list.parquet` | List of all Bitcoin ETFs with metadata | ticker, fund_name, region, fund_type, aum_usd, price_usd, price_change_percent |
| `api_etf_bitcoin_list_modified.parquet` | Modified Bitcoin ETF list with additional metrics | ticker, fund_name, region, aum_usd, price_change_percent |
| `api_etf_bitcoin_net_assets_history.parquet` | Historical net asset values for Bitcoin ETFs | time, net_assets_total |
| `api_etf_bitcoin_premium_discount_history.parquet` | Premium/discount history for Bitcoin ETFs | time, premium_discount_percent |
| `api_etf_bitcoin_price_history.parquet` | Price history for Bitcoin ETFs | time, price_usd, price_change_percent |

### Ethereum ETF Files

| File | Description | Key Fields |
|------|-------------|------------|
| `api_etf_ethereum_flow_history.parquet` | Historical fund flows for Ethereum ETFs | time, fund_flow_usd, price_change_percent |
| `api_etf_ethereum_list.parquet` | List of all Ethereum ETFs with metadata | ticker, fund_name, region, fund_type, aum_usd, price_usd |
| `api_etf_ethereum_net_assets_history.parquet` | Historical net asset values for Ethereum ETFs | time, net_assets_total |

### Grayscale Files

| File | Description | Key Fields |
|------|-------------|------------|
| `api_grayscale_holdings_list.parquet` | Grayscale trust holdings details | ticker, fund_name, aum_usd, holdings_per_share, price_usd |
| `api_grayscale_premium_history.parquet` | Historical premium/discount for Grayscale trusts | time, premium_discount_percent |

### Hong Kong ETF Files

| File | Description | Key Fields |
|------|-------------|------------|
| `api_hk_etf_bitcoin_flow_history.parquet` | Fund flow history for Hong Kong Bitcoin ETFs | time, fund_flow_usd, price_change_percent |

## Futures Data

The `futures` folder contains data related to cryptocurrency futures markets, divided into several subcategories.

### Funding Rate

The `funding_rate` folder contains data about funding rates in futures markets.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_futures_fundingRate_accumulated_exchange_list.parquet` | Accumulated funding rates by exchange | exchange_name, symbol, funding_rate, funding_rate_7d |
| `api_futures_fundingRate_exchange_list.parquet` | Current funding rates by exchange | exchange_name, symbol, funding_rate, next_funding_time |
| `api_futures_fundingRate_ohlc_history.parquet` | Historical funding rate OHLC data | time, open, high, low, close |
| `api_futures_fundingRate_oi_weight_ohlc_history_[COIN]_[COIN].parquet` | Open Interest weighted funding rate history for specific coin | time, open, high, low, close |
| `api_futures_fundingRate_vol_weight_ohlc_history_[COIN]_[COIN].parquet` | Volume weighted funding rate history for specific coin | time, open, high, low, close |

### Liquidation

The `liquidation` folder contains data about liquidations in futures markets.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_futures_liquidation_aggregated_coin_history_[COIN]_[COIN].parquet` | Aggregated liquidation history for specific coin | time, short_liquidation_usd, long_liquidation_usd, total_liquidation_usd |
| `api_futures_liquidation_exchange_list_[COIN]_[COIN].parquet` | Liquidations by exchange for specific coin | exchange_name, short_liquidation_usd, long_liquidation_usd, total_liquidation_usd |

### Long-Short Ratio

The `long_short_ratio` folder contains data about long-short positioning in futures markets.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_futures_global_long_short_account_ratio_history.parquet` | Global long-short account ratios | time, long_short_ratio, price |
| `api_futures_taker_buy_sell_volume_exchange_list_[COIN]_[COIN].parquet` | Taker buy/sell volumes by exchange | exchange_name, buy_volume, sell_volume, buy_sell_ratio |
| `api_futures_top_long_short_account_ratio_history.parquet` | Top traders long-short account ratios | time, long_short_ratio, price |
| `api_futures_top_long_short_position_ratio_history.parquet` | Top traders long-short position ratios | time, long_short_ratio, price |

### Market Data

The `market` folder contains general futures market data.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_futures_pairs_markets_[COIN]_[COIN].parquet` | Trading pairs data for specific coin | exchange_name, symbol, price_usd, volume_24h_usd, open_interest_usd |
| `api_price_ohlc_history.parquet` | Price OHLC history | time, open, high, low, close, volume |

### Open Interest

The `open_interest` folder contains data about open interest in futures markets.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_futures_openInterest_exchange_list_[COIN]_[COIN].parquet` | Open interest by exchange for specific coin | exchange_name, open_interest_usd, market_share_percent |
| `api_futures_openInterest_ohlc_aggregated_coin_margin_history_[COIN]_[COIN].parquet` | Aggregated coin-margined open interest history | time, open, high, low, close |
| `api_futures_openInterest_ohlc_aggregated_stablecoin_[COIN]_[COIN].parquet` | Aggregated stablecoin-margined open interest history | time, open, high, low, close |

### Order Book

The `order_book` folder contains data about futures market order books.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_futures_orderbook_aggregated_ask_bids_history_[COIN]_[COIN].parquet` | Aggregated order book history for specific coin | time, asks_amount, bids_amount, asks_bids_ratio |
| `api_futures_orderbook_ask_bids_history.parquet` | General order book history | time, exchange_name, symbol, asks_amount, bids_amount |

### Taker Buy-Sell

The `taker_buy_sell` folder contains data about taker buy/sell volumes in futures markets.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_futures_aggregated_taker_buy_sell_volume_history.parquet` | Aggregated taker buy/sell volume history | time, buy_volume, sell_volume, buy_sell_ratio |
| `api_futures_taker_buy_sell_volume_history_[COIN]_[COIN].parquet` | Taker buy/sell volume history for specific coin | time, buy_volume, sell_volume, buy_sell_ratio, price |

## Market Indicators

The `indic` folder contains various market indicators and metrics.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_bitfinex_margin_long_short.parquet` | Bitfinex margin long/short positions | time, long_short_ratio, price |
| `api_bull_market_peak_indicator.parquet` | Bull market peak indicator | time_list, price_list, indicator_value |
| `api_coinbase_premium_index.parquet` | Coinbase premium index vs other exchanges | time, premium_index, price |
| `api_futures_basis_history.parquet` | Futures basis history | time, basis_percent, price |
| `api_index_200_week_moving_average_heatmap.parquet` | 200-week moving average heatmap | time, ma_value, price_to_ma_ratio |
| `api_index_2_year_ma_multiplier.parquet` | 2-year MA multiplier indicator | time, ma_value, multiplier |
| `api_index_ahr999.parquet` | AHR999 indicator | time, ahr999_value, price |
| `api_index_bitcoin_bubble_index.parquet` | Bitcoin bubble index | time, bubble_index, price |
| `api_index_bitcoin_profitable_days.parquet` | Bitcoin profitable days ratio | time, profitable_days_percent, price |
| `api_index_bitcoin_rainbow_chart.parquet` | Bitcoin rainbow chart bands | time, price, band_values |
| `api_index_fear_greed_history.parquet` | Fear & Greed index history | time_list, data_list (index value), price_list |
| `api_index_golden_ratio_multiplier.parquet` | Golden ratio multiplier bands | time, price, multiplier_bands |
| `api_index_pi_cycle_indicator.parquet` | Pi Cycle top indicator | time, indicator_value, price |
| `api_index_puell_multiple.parquet` | Puell Multiple indicator | time, puell_multiple, price |
| `api_index_stableCoin_marketCap_history.parquet` | Stablecoin market cap history | time, market_cap_usd |

## On-Chain Data

The `on_chain` folder contains blockchain data.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_exchange_balance_list.parquet` | Exchange balance list | exchange_name, total_balance, balance_change_1d, balance_change_percent_7d, balance_change_percent_30d |
| `api_exchange_chain_tx_list.parquet` | Exchange chain transaction list | exchange_name, inflow_amount, outflow_amount, net_flow_amount |

## Options Data

The `options` folder contains cryptocurrency options market data.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_option_info_[COIN]_[COIN].parquet` | Options market information for specific coin | exchange_name, expiry_date, call_open_interest, put_open_interest, put_call_ratio |
| `api_option_max_pain_[COIN]_[COIN].parquet` | Max pain points for options of specific coin | expiry_date, max_pain_price, call_open_interest, put_open_interest |

## Spot Market Data

The `spot` folder contains data related to cryptocurrency spot markets, divided into several subcategories.

### Spot Order Book

The `order_book_spot` folder contains data about spot market order books.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_spot_orderbook_aggregated_ask_bids_history.parquet` | Aggregated spot market order book history | time, asks_amount, bids_amount, asks_bids_ratio |
| `api_spot_orderbook_ask_bids_history_[COIN]_[COIN].parquet` | Spot market order book history for specific coin | time, asks_amount, bids_amount, asks_bids_ratio |

### Spot Market

The `spot_market` folder contains general spot market data.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_spot_pairs_markets_[COIN]_[COIN].parquet` | Spot market trading pairs for specific coin | exchange_name, symbol, price_usd, volume_24h_usd |
| `api_spot_supported_coins.parquet` | List of supported coins on spot markets | coin_symbol, coin_name, market_count |

### Spot Taker Buy-Sell

The `taker_buy_sell_spot` folder contains data about taker buy/sell volumes in spot markets.

| File | Description | Key Fields |
|------|-------------|------------|
| `api_spot_aggregated_taker_buy_sell_volume_history.parquet` | Aggregated spot market taker buy/sell volume history | time, buy_volume, sell_volume, buy_sell_ratio |

## Visualizing the Data

Here are recommended approaches for visualizing different types of data:

### Time Series Data

Most of the datasets contain time series data that can be visualized using:

```python
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Example: Visualizing Bitcoin ETF flows
df = pd.read_parquet('data/20250512/etf/api_etf_bitcoin_flow_history.parquet')

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['time'], unit='ms')

# Using Matplotlib
plt.figure(figsize=(12, 6))
plt.plot(df['datetime'], df['fund_flow_usd'])
plt.title('Bitcoin ETF Fund Flows')
plt.xlabel('Date')
plt.ylabel('Fund Flow (USD)')
plt.grid(True)
plt.show()

# Using Plotly (interactive)
fig = px.line(df, x='datetime', y='fund_flow_usd', 
              title='Bitcoin ETF Fund Flows')
fig.update_layout(xaxis_title='Date', yaxis_title='Fund Flow (USD)')
fig.show()
```

### OHLC and Candlestick Charts

For OHLC (Open-High-Low-Close) data:

```python
import pandas as pd
import plotly.graph_objects as go

# Example: Visualizing funding rate OHLC data
df = pd.read_parquet('data/20250512/futures/funding_rate/api_futures_fundingRate_ohlc_history.parquet')

# Convert timestamp to datetime
df['datetime'] = pd.to_datetime(df['time'], unit='ms')

# Make sure OHLC columns are numeric
for col in ['open', 'high', 'low', 'close']:
    df[col] = pd.to_numeric(df[col])

# Create candlestick chart
fig = go.Figure(data=[go.Candlestick(
    x=df['datetime'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close']
)])

fig.update_layout(
    title='Funding Rate History',
    xaxis_title='Date',
    yaxis_title='Funding Rate (%)',
    xaxis_rangeslider_visible=False
)
fig.show()
```

### Comparison Charts

For comparing multiple metrics:

```python
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Example: Comparing ETF flows with price changes
df = pd.read_parquet('data/20250512/etf/api_etf_bitcoin_flow_history.parquet')
df['datetime'] = pd.to_datetime(df['time'], unit='ms')

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Bar(x=df['datetime'], y=df['fund_flow_usd'], name="Fund Flow (USD)"),
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=df['datetime'], y=df['price_change_percent'], name="Price Change (%)"),
    secondary_y=True,
)

# Set titles
fig.update_layout(
    title_text="Bitcoin ETF Flows vs Price Change"
)
fig.update_xaxes(title_text="Date")
fig.update_yaxes(title_text="Fund Flow (USD)", secondary_y=False)
fig.update_yaxes(title_text="Price Change (%)", secondary_y=True)

fig.show()
```

### Heatmaps and Correlation Matrices

For visualizing relationships between different metrics:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Correlation between different market indicators
# You'll need to merge multiple datasets first
# This is just a conceptual example

# Load different datasets
df1 = pd.read_parquet('data/20250512/indic/api_index_fear_greed_history.parquet')
df1['datetime'] = pd.to_datetime(df1['time_list'], unit='ms')
df1 = df1.rename(columns={'data_list': 'fear_greed', 'price_list': 'price_fg'})

df2 = pd.read_parquet('data/20250512/futures/funding_rate/api_futures_fundingRate_ohlc_history.parquet')
df2['datetime'] = pd.to_datetime(df2['time'], unit='ms')
df2['funding_rate'] = pd.to_numeric(df2['close'])

# Join on nearest datetime
df1.set_index('datetime', inplace=True)
df2.set_index('datetime', inplace=True)
merged = pd.merge_asof(
    df1.reset_index(), 
    df2.reset_index(), 
    on='datetime', 
    direction='nearest'
)

# Calculate correlation matrix
corr_matrix = merged[['fear_greed', 'funding_rate', 'price_fg']].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix: Fear & Greed vs Funding Rate vs Price')
plt.show()
```

## Data Analysis Methods

Here are some recommended analysis methods:

### 1. Market Sentiment Analysis

Use the Fear & Greed Index, long-short ratios, and other sentiment indicators to gauge market sentiment:

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load Fear & Greed Index data
df = pd.read_parquet('data/20250512/indic/api_index_fear_greed_history.parquet')
df['datetime'] = pd.to_datetime(df['time_list'], unit='ms')

# Create sentiment categories
bins = [0, 20, 40, 60, 80, 100]
labels = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed']
df['sentiment'] = pd.cut(df['data_list'], bins=bins, labels=labels)

# Group by sentiment and analyze price behavior
sentiment_stats = df.groupby('sentiment')['price_list'].agg(['mean', 'std', 'min', 'max', 'count'])
print(sentiment_stats)

# Visualize sentiment distribution
plt.figure(figsize=(12, 6))
df['sentiment'].value_counts().plot(kind='bar')
plt.title('Distribution of Market Sentiment')
plt.ylabel('Count')
plt.xlabel('Sentiment Category')
plt.show()
```

### 2. Liquidity Analysis

Analyze order book data to assess market liquidity:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load order book data
df = pd.read_parquet('data/20250512/futures/order_book/api_futures_orderbook_aggregated_ask_bids_history_BTC_BTC.parquet')
df['datetime'] = pd.to_datetime(df['time'], unit='ms')

# Calculate liquidity metrics
df['total_liquidity'] = df['asks_amount'] + df['bids_amount']
df['liquidity_imbalance'] = (df['bids_amount'] - df['asks_amount']) / df['total_liquidity']

# Plot liquidity metrics
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(df['datetime'], df['total_liquidity'])
plt.title('Total Order Book Liquidity (BTC)')
plt.ylabel('Amount')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(df['datetime'], df['liquidity_imbalance'])
plt.title('Liquidity Imbalance (+ = Bid Heavy, - = Ask Heavy)')
plt.ylabel('Imbalance Ratio')
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.grid(True)

plt.tight_layout()
plt.show()
```

### 3. ETF Flow Analysis

Analyze the impact of ETF flows on price:

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

# Load ETF flow data
df = pd.read_parquet('data/20250512/etf/api_etf_bitcoin_flow_history.parquet')
df['datetime'] = pd.to_datetime(df['time'], unit='ms')

# Calculate cumulative flows
df['cumulative_flow'] = df['fund_flow_usd'].cumsum()

# Calculate rolling correlation between flows and price changes
window_size = 30  # 30-day rolling window
df['flow_price_corr'] = df['fund_flow_usd'].rolling(window=window_size).corr(df['price_change_percent'])

# Plot results
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot 1: ETF Flows
ax1.bar(df['datetime'], df['fund_flow_usd'], color='blue', alpha=0.7)
ax1.set_title('Daily ETF Flows')
ax1.set_ylabel('Flow (USD)')
ax1.grid(True)

# Plot 2: Cumulative Flows
ax2.plot(df['datetime'], df['cumulative_flow'], color='green')
ax2.set_title('Cumulative ETF Flows')
ax2.set_ylabel('Cumulative Flow (USD)')
ax2.grid(True)

# Plot 3: Rolling Correlation
ax3.plot(df['datetime'], df['flow_price_corr'], color='purple')
ax3.set_title(f'{window_size}-Day Rolling Correlation: Flows vs Price Changes')
ax3.set_ylabel('Correlation Coefficient')
ax3.set_ylim(-1, 1)
ax3.axhline(y=0, color='r', linestyle='-', alpha=0.3)
ax3.grid(True)

plt.tight_layout()
plt.show()
```

## Correlation Analysis

Analyze correlations between different market metrics:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This is an example that would require merging multiple datasets
# Assume we've already created a merged_data dataframe with key metrics

# Example columns in merged_data:
# - price_change_percent
# - funding_rate
# - open_interest_change
# - long_short_ratio
# - fear_greed_index
# - liquidation_volume

# Calculate correlation matrix
correlation_matrix = merged_data.corr()

# Create heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Crypto Market Metrics')
plt.tight_layout()
plt.show()

# Create pairplot for detailed relationships
sns.pairplot(merged_data, diag_kind='kde')
plt.suptitle('Pairwise Relationships Between Market Metrics', y=1.02)
plt.show()
```

## Market Trend Analysis

Identify market trends using various indicators:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load price data
df = pd.read_parquet('data/20250512/futures/market/api_price_ohlc_history.parquet')
df['datetime'] = pd.to_datetime(df['time'], unit='ms')

# Calculate technical indicators
# Simple Moving Averages
df['sma_50'] = df['close'].rolling(window=50).mean()
df['sma_200'] = df['close'].rolling(window=200).mean()

# Create golden/death cross signal
df['golden_cross'] = np.where(df['sma_50'] > df['sma_200'], 1, 0)
df['golden_cross_change'] = df['golden_cross'].diff()
df['bull_market'] = np.where(df['golden_cross'] == 1, 'Bull', 'Bear')

# Identify trend change points
trend_changes = df[df['golden_cross_change'] != 0].copy()

# Plot results
plt.figure(figsize=(14, 8))
plt.plot(df['datetime'], df['close'], label='Price', alpha=0.5)
plt.plot(df['datetime'], df['sma_50'], label='50-day SMA', linewidth=1)
plt.plot(df['datetime'], df['sma_200'], label='200-day SMA', linewidth=1)

# Highlight bull markets
bull_markets = df[df['golden_cross'] == 1]
for i in range(len(trend_changes) - 1):
    if trend_changes.iloc[i]['golden_cross_change'] > 0:  # Start of bull market
        start_date = trend_changes.iloc[i]['datetime']
        if i+1 < len(trend_changes):
            end_date = trend_changes.iloc[i+1]['datetime']
            plt.axvspan(start_date, end_date, alpha=0.2, color='green')

# Mark golden/death crosses
golden_crosses = trend_changes[trend_changes['golden_cross_change'] > 0]
death_crosses = trend_changes[trend_changes['golden_cross_change'] < 0]

plt.scatter(golden_crosses['datetime'], golden_crosses['close'], marker='^', 
            color='g', s=100, label='Golden Cross')
plt.scatter(death_crosses['datetime'], death_crosses['close'], marker='v', 
            color='r', s=100, label='Death Cross')

plt.title('Bitcoin Price with Golden/Death Crosses')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.yscale('log')
plt.legend()
plt.grid(True)
plt.show()
```

---

This guide provides a comprehensive overview of the data available in the Izun Liquidity Report project, along with methods for visualization and analysis. By understanding the structure and content of this data, LLMs can effectively assist with market analysis, trend identification, and visualization of cryptocurrency market dynamics.