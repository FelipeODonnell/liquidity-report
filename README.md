# Izun Crypto Liquidity Report

A comprehensive cryptocurrency liquidity dashboard and data collection tool using the CoinGlass API. This application provides detailed market analysis for cryptocurrency markets, including ETFs, futures, spot markets, options, and various market indicators.

## Project Components

The project consists of two main components:
1. **Data Collection**: Python scripts to collect and process data from the CoinGlass API
2. **Data Visualization**: Streamlit application for visualizing collected market data

## Project Setup

### Prerequisites

- Python 3.11 or higher
- pip or [uv](https://github.com/astral-sh/uv) for package management
- Required packages: streamlit, pandas, plotly, pyarrow, numpy
- CoinGlass API key (get one at [CoinGlass](https://coinglass.com/))

### Setup Instructions

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd izun-liquidity-report-v5
   ```

2. **Create a Virtual Environment**
   ```bash
   # Create virtual environment
   python3 -m venv .venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Install all dependencies
   pip install -e .
   
   # Or using uv (faster)
   uv pip install -e .
   ```

4. **Configure API Keys**
   
   Create a `.streamlit/secrets.toml` file with your API keys:
   ```toml
   [coinglass_api]
   api_key = "your_coinglass_api_key_here"
   ```

## Running the Application

### Data Collection

```bash
# Run the data collection script
python report.py

# Specify custom rate limit (defaults to 29 requests/minute)
python report.py --max-rate 15
```

### Dashboard Visualization

```bash
# Recommended method (direct Streamlit launch)
streamlit run streamlit/app.py

# Alternative method (may cause relaunch issues)
python main.py
```

The dashboard will open automatically in your default web browser at http://localhost:8501.

> **Important**: If you experience app relaunch issues, please use the direct Streamlit launch method and refer to the `relaunchfix.md` file for detailed solutions.

## Data Collection Features

- **Date-based Data Organization**: Data is saved in date folders (format: YYYYMMDD)
- **Rate Limiting**: Prevents API rate limit errors (default: 29 requests/minute)
- **Selective Endpoint Execution**: Comment out specific API endpoints in `report.py` to skip them
- **Time Parameter Refresh**: Use `refresh_time.py` to update time ranges for historical data

## Dashboard Features

The Streamlit dashboard provides interactive visualizations across multiple pages:

- **Overview Dashboard**: Key market metrics and trends
- **ETF Analysis**: Bitcoin and Ethereum ETF flows, AUM, and performance
- **Futures Markets**: Funding rates, liquidations, open interest, and market data
- **Spot Markets**: Exchange comparisons, order books, and market data
- **Indicators**: Technical indicators and market sentiment metrics
- **On-Chain Data**: Exchange balances and blockchain transaction data
- **Options Markets**: Open interest, max pain, and options data
- **Historical Data**: Access to historical market data

## API Data Categories

This application collects and visualizes data across several categories:

- **ETF Data**: Bitcoin and Ethereum ETF metrics
- **Futures Market Data**: Funding rates, liquidations, open interest, order books
- **Spot Market Data**: Exchange prices, volumes, order books
- **Market Indicators**: Fear & Greed index, technical indicators
- **On-chain Data**: Exchange balances, transaction volumes
- **Options Data**: Options open interest, max pain points

## Project Structure

```
izun-liquidity-report-v5/
├── coinglass-api/           # API data collection scripts
│   ├── etf/                 # ETF-related API scripts
│   ├── futures/             # Futures market API scripts
│   ├── indic/               # Market indicators API scripts
│   ├── on_chain/            # On-chain data API scripts
│   ├── options/             # Options market API scripts
│   └── spot/                # Spot market API scripts
├── data/                    # Collected data storage
│   └── YYYYMMDD/            # Date-based folders
├── streamlit/               # Streamlit dashboard
│   ├── app.py               # Main dashboard app
│   ├── components/          # Reusable UI components
│   ├── pages/               # Dashboard pages
│   └── utils/               # Utility functions
├── main.py                  # Alternative entry point
├── report.py                # Data collection script
├── refresh_time.py          # Time parameter refresh script
├── pyproject.toml           # Project configuration
├── README.md                # Project documentation
├── userguide.md             # Detailed user guide
├── upload.md                # GitHub & Streamlit deployment guide
└── relaunchfix.md           # Solution for app relaunch issues
```

## Deployment

### GitHub Deployment

Before pushing to GitHub:

1. Ensure sensitive information like API keys are not included in the repository
2. Check that `.streamlit/secrets.toml` is in your `.gitignore` file
3. See `upload.md` for detailed deployment preparation steps

### Streamlit Cloud Deployment

To deploy on Streamlit Cloud:

1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Configure your API keys in Streamlit Cloud's secrets management
4. Set your Python version to 3.11+
5. See `upload.md` for complete instructions

## Troubleshooting

If you encounter issues running the application:

1. **Missing Data**: Ensure data collection has been run successfully 
2. **App Relaunches**: Use direct Streamlit launch: `streamlit run streamlit/app.py`
3. **Import Errors**: Verify all dependencies are installed correctly
4. **API Rate Limiting**: Reduce the rate limit in `report.py` with `--max-rate`
5. **Missing Charts**: Some visualizations require specific data files to be present
6. **API Key Issues**: Verify your CoinGlass API key is correctly configured in `.streamlit/secrets.toml`

For more detailed information, consult the `userguide.md` file.

## Additional Resources

- **User Guide**: For comprehensive usage instructions, see `userguide.md`
- **Relaunch Fix**: For solutions to app relaunch issues, see `relaunchfix.md`
- **Deployment Guide**: For GitHub and Streamlit deployment, see `upload.md`

## Data Storage

All collected data is stored in Parquet format, which provides:
- Efficient compression
- Fast query performance
- Column-oriented storage
- Strong typing

## Target Cryptocurrencies

The application focuses on the following cryptocurrencies:
- Bitcoin (BTC)
- Ethereum (ETH)
- Solana (SOL)
- Ripple (XRP)