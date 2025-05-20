# Izun Crypto Liquidity Report

A comprehensive dashboard for analyzing cryptocurrency market liquidity data, providing detailed insights for spot and derivatives markets.

## Overview

This project provides real-time and historical analysis of cryptocurrency market liquidity metrics across major crypto assets (BTC, ETH, SOL, XRP). The dashboard presents data through interactive visualizations and analytics, focusing on key liquidity indicators for both spot and futures markets.

## Features

- **Core Liquidity Metrics**: Trading volume, open interest, bid-ask spreads, order book depth, and funding rates
- **Spot Market Analysis**: Dedicated spot price history charts and exchange volume distribution
- **Futures Market Data**: Comprehensive futures market metrics including funding rates, liquidations, and open interest
- **ETF Analysis**: Bitcoin and Ethereum ETF data including flows, premium/discount, and net assets
- **Multi-Asset Analysis**: Compare liquidity metrics across different cryptocurrencies
- **Exchange Comparison**: Analyze market share and liquidity across different exchanges
- **Historical Trends**: View time-series data for all major liquidity metrics
- **Interactive Visualizations**: Drill down into specific data points and time ranges
- **On-Chain Data**: Exchange balance tracking and transaction monitoring

## Data Sources

Data is collected from various cryptocurrency exchanges and aggregators via the CoinGlass API and stored in structured parquet files. Key data categories include:

- **Spot Markets**: Price history, order books, and taker buy/sell data for major spot markets
- **Futures Markets**: Funding rates, liquidations, open interest, and order book data for futures
- **ETF Data**: Bitcoin and Ethereum ETF market data including flows and AUM
- **Indicators**: Market sentiment and technical indicators like Fear & Greed Index
- **On-Chain Data**: Exchange balances and transaction data for major cryptocurrencies
- **Options Data**: Options market data including volume, open interest, and max pain

The system includes sophisticated rate limiting and exponential backoff mechanisms to handle API request limits effectively.

## Project Structure

- `streamlit/`: Main application code
  - `app.py`: Entry point for the Streamlit application
  - `components/`: UI components (charts, tables, metrics, sidebar)
  - `pages/`: Individual dashboard pages for different aspects of liquidity
    - `01_report.py`: Main dashboard with key metrics
    - `02_etf.py`: ETF analysis dashboard
    - `03_futures.py`: Futures market analysis
    - `04_spot.py`: Spot market analysis including price history
    - `05_indicators.py`: Market indicators and sentiment analysis
    - `06_on_chain.py`: On-chain metrics and exchange balances
    - `07_options.py`: Options market analysis
  - `utils/`: Helper functions for data loading, formatting, and configuration
- `coinglass-api/`: API clients for data collection organized by category
- `data/`: Storage for parquet data files, organized by date and category
- `report.py`: Data collection script with rate limiting for CoinGlass API

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```
   streamlit run streamlit/app.py
   ```

## Data Collection

To collect fresh data from the CoinGlass API:

```
python report.py
```

This script fetches data for all configured API endpoints and stores it in parquet format in the data directory. The script includes:

- Smart rate limiting with exponential backoff
- Automatic retry for failed requests
- Special handling for rate-limited endpoints
- Data validation and error handling

## Data Verification

To check which data files have been successfully saved:

```
python data-checker.py
```

This will generate a `data-saved.md` file showing which API endpoints have successfully saved data.

## Technologies

- Python 3.9+
- Streamlit: Web application framework
- Pandas: Data manipulation and analysis
- Plotly: Interactive data visualizations
- Parquet: Efficient columnar storage format
- NumPy: Numerical computing
- Requests: HTTP library for API calls

## Recent Updates

- Added dedicated spot price history visualization using spot market data
- Improved data loading with better error handling and fallback mechanisms
- Enhanced rate limiting for robust API data collection
- Added comprehensive data structure documentation
- Optimized chart rendering and data processing