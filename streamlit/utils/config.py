"""
Configuration settings for the Izun Crypto Liquidity Report application.
"""

import os

# Application title and branding
APP_TITLE = "Izun Crypto Liquidity Report"
APP_ICON = "📊"

# Asset specific colors
ASSET_COLORS = {
    "BTC": "#F7931A",  # Bitcoin orange
    "ETH": "#627EEA",  # Ethereum blue
    "SOL": "#14F195",  # Solana green
    "XRP": "#23292F",  # XRP dark
}

# Chart colors
CHART_COLORS = {
    "primary": "#3366CC",
    "secondary": "#FF9900",
    "positive": "#4CAF50",  # Green for positive changes
    "negative": "#F44336",  # Red for negative changes
    "neutral": "#9E9E9E",   # Gray for neutral values
}

# Exchange colors for consistent visualization
EXCHANGE_COLORS = {
    "Binance": "#F0B90B", 
    "OKX": "#121212",
    "Bybit": "#FFBC00",
    "dYdX": "#6966FF",
    "Coinbase": "#0052FF",
    "Kraken": "#5741D9",
    "Bitfinex": "#16B157",
    "FTX": "#11A9BC",
    "Huobi": "#347FEB",
    "KuCoin": "#24AE8F",
    "Other": "#9E9E9E"
}

# Metric formatting settings
DEFAULT_CURRENCY_PRECISION = 2
DEFAULT_PERCENTAGE_PRECISION = 2

# Data settings
DATA_BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data'))

# Supported assets
SUPPORTED_ASSETS = ["BTC", "ETH", "SOL", "XRP"]

# Default settings
DEFAULT_TIMEFRAME = "7d"  # 1d, 7d, 30d, 90d, YTD, MAX
DEFAULT_ASSET = "BTC"

