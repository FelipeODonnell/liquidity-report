# Izun Crypto Liquidity Report - Streamlit Application Plan

## 1. Introduction

This document outlines a comprehensive plan for building the Streamlit application for the Izun Crypto Liquidity Report. The application will serve as a frontend for visualizing and analyzing cryptocurrency market data, with a focus on liquidity metrics across various market segments including ETFs, futures markets, spot markets, options, and market indicators.

**IMPLEMENTATION STATUS: COMPLETED ✅**

## 2. Application Overview

The Izun Crypto Liquidity Report Streamlit application will be a multi-page dashboard that presents cryptocurrency market data in an intuitive and interactive manner. The application will:

- Provide a consistent sidebar navigation matching the data folder structure
- Present data through interactive visualizations and tables
- Allow users to filter and drill down into specific data points
- Enable historical data comparison and trend analysis
- Provide downloadable data in CSV format

## 3. Project Structure

```
/izun-liquidity-report-v5/
├── main.py                  # Main entry point for the application
├── streamlit/               # Streamlit application code
│   ├── app.py               # Main Streamlit application file
│   ├── pages/               # Individual pages for each section
│   │   ├── 01_report.py     # Dashboard overview/summary page
│   │   ├── 02_etf.py        # ETF data visualizations
│   │   ├── 03_futures.py    # Futures market data
│   │   ├── 04_spot.py       # Spot market data
│   │   ├── 05_indicators.py # Market indicators
│   │   ├── 06_on_chain.py   # On-chain metrics
│   │   ├── 07_options.py    # Options market data
│   │   └── 08_historical.py # Historical data download page
│   ├── components/          # Reusable UI components
│   │   ├── sidebar.py       # Sidebar navigation
│   │   ├── metrics.py       # Metrics display components
│   │   ├── charts.py        # Chart creation utilities
│   │   └── tables.py        # Table display components
│   └── utils/               # Utility functions
│       ├── data_loader.py   # Functions for loading data
│       ├── formatters.py    # Data formatting utilities
│       └── config.py        # Application configuration
├── data/                    # Data directory (existing)
└── requirements.txt         # Project dependencies
```

## 4. Technology Stack

- **Streamlit**: Main framework for building the web application
- **Pandas**: For data manipulation and analysis
- **Plotly**: For interactive visualizations
- **Altair**: For declarative visualizations
- **Matplotlib/Seaborn**: For static visualizations
- **uv**: Python package management as specified in requirements

## 5. Navigation Structure

### Sidebar Structure

The sidebar will serve as the main navigation element and will mirror the structure of the data folder:

```
- Izun Crypto Liquidity Report (Title)
- Report (Default Landing Page)
- ETF
  - Bitcoin ETFs
  - Ethereum ETFs
  - Grayscale Funds
  - Hong Kong ETFs
- Futures
  - Funding Rate
  - Liquidation
  - Long/Short Ratio
  - Market Data
  - Open Interest
  - Order Book
  - Taker Buy/Sell
  - Whale Positions
- Spot
  - Order Book
  - Market Data
  - Taker Buy/Sell
- Indicators
  - Market Indicators
- On-Chain
  - Exchange Balance
  - Chain Transactions
- Options
  - BTC Options
  - ETH Options
  - SOL Options
  - XRP Options
- Historical Reports & Data
```

## 6. Page Designs and Functionalities

### 6.1 Report Page (Default Page)

The Report page will serve as a dashboard with key metrics and insights from all data categories.

**Components:**
- Market Summary Cards (Price, 24h Change, Volume)
- Key Metrics Highlights (Top ETF Flows, Liquidation Events, Exchange Balance Changes)
- Market Status Overview (Fear & Greed Index, BTC Rainbow Chart)
- Recent Trend Visualizations (Price, Open Interest, Funding Rates)
- Market Indicators Summary
- Data Update Status (Showing when data was last refreshed)

**Data Sources:**
- Latest data from all categories with key metrics extracted

**Implementation Details:**
- Create a grid layout with expandable sections
- Use Streamlit metrics components for key figures
- Implement interactive charts for trend visualization
- Add time period selectors (1d, 7d, 30d, YTD)

### 6.2 ETF Page

The ETF page will focus on ETF-related data including flows, AUM, and performance metrics.

**Subpages:**
- Bitcoin ETFs
- Ethereum ETFs
- Grayscale Funds
- Hong Kong ETFs

**Bitcoin ETFs Components:**
- ETF List Table:
  - Data: `api_etf_bitcoin_list.parquet`
  - Functionality: Sortable table with search, showing all Bitcoin ETFs and key metrics
  
- ETF Flow Analysis:
  - Data: `api_etf_bitcoin_flow_history.parquet`
  - Visualizations: Daily flows chart, cumulative flows chart, flow vs. price change correlation
  
- AUM Visualization:
  - Data: `api_etf_bitcoin_aum.parquet`
  - Visualizations: AUM by ETF (treemap/pie chart), AUM history (line chart)
  
- Price & Premium/Discount Analysis:
  - Data: `api_etf_bitcoin_premium_discount_history.parquet`, `api_etf_bitcoin_price_history.parquet`
  - Visualizations: Premium/discount trends, price comparison with BTC spot

**Ethereum ETFs Components:**
- ETF List Table:
  - Data: `api_etf_ethereum_list.parquet`
  
- ETF Flow Analysis:
  - Data: `api_etf_ethereum_flow_history.parquet`
  
- Net Assets History:
  - Data: `api_etf_ethereum_net_assets_history.parquet`

**Grayscale Funds Components:**
- Holdings List:
  - Data: `api_grayscale_holdings_list.parquet`
  
- Premium History:
  - Data: `api_grayscale_premium_history.parquet`

**Hong Kong ETFs Components:**
- Flow History:
  - Data: `api_hk_etf_bitcoin_flow_history.parquet`

**Implementation Details:**
- Create tabbed interface for different ETF types
- Implement interactive date range selectors
- Add comparison functionality between different ETFs
- Include annotations for significant events (e.g., new ETF launches)

### 6.3 Futures Page

The Futures page will be the most complex page, covering multiple aspects of futures markets.

**Subpages:**
- Funding Rate
- Liquidation
- Long/Short Ratio
- Market Data
- Open Interest
- Order Book
- Taker Buy/Sell
- Whale Positions

**Funding Rate Components:**
- Funding Rate Overview:
  - Data: `api_futures_fundingRate_exchange_list.parquet`, `api_futures_fundingRate_accumulated_exchange_list.parquet`
  - Visualizations: Current funding rates by exchange, accumulated funding rates
  
- Funding Rate History:
  - Data: `api_futures_fundingRate_ohlc_history.parquet`
  - Visualizations: Funding rate history chart with OHLC candlesticks
  
- Weighted Funding Rates:
  - Data: OI and volume weighted OHLC history files for BTC, ETH, SOL, XRP
  - Visualizations: Comparison charts for different weighting methods

**Liquidation Components:**
- Liquidation Overview:
  - Data: Aggregated liquidation history files for BTC, ETH, SOL, XRP
  - Visualizations: Liquidation heatmap, long vs. short liquidations
  
- Liquidation by Exchange:
  - Data: Exchange list files for BTC, ETH, SOL, XRP
  - Visualizations: Liquidations by exchange, market share pie chart

**Long/Short Ratio Components:**
- Global Long/Short Ratios:
  - Data: `api_futures_global_long_short_account_ratio_history.parquet`
  - Visualizations: Long/short ratio trend vs. price
  
- Taker Buy/Sell Volumes:
  - Data: Taker buy/sell volume exchange list files for BTC, ETH, SOL, XRP
  - Visualizations: Buy/sell ratio by exchange

**Market Data Components:**
- Pairs Markets Overview:
  - Data: Pairs markets files for BTC, ETH, SOL, XRP
  - Visualizations: Market data tables with key metrics
  
- Price OHLC:
  - Data: `api_price_ohlc_history.parquet`
  - Visualizations: Interactive OHLC candlestick chart

**Open Interest Components:**
- Open Interest by Exchange:
  - Data: Open Interest exchange list files for BTC, ETH, SOL, XRP
  - Visualizations: OI by exchange, market share charts
  
- Open Interest History:
  - Data: OHLC aggregated files for different margin types
  - Visualizations: OI history charts with price overlay

**Order Book Components:**
- Order Book Analysis:
  - Data: Order book files for different assets
  - Visualizations: Depth charts, ask/bid ratio trends

**Taker Buy/Sell Components:**
- Buy/Sell Volume Analysis:
  - Data: Taker buy/sell volume history files
  - Visualizations: Buy/sell ratio trends, volume by exchange

**Whale Positions (if available in data):**
- Whale Position Analysis:
  - Data: Related files if available
  - Visualizations: Large position tracking

**Implementation Details:**
- Create nested navigation for the complex futures section
- Implement asset selector (BTC, ETH, SOL, XRP) for relevant charts
- Add time period selectors for historical data
- Include comparative analysis between different metrics
- Provide explanatory text for technical indicators

### 6.4 Spot Page

The Spot page will focus on spot market data and metrics.

**Subpages:**
- Order Book
- Market Data
- Taker Buy/Sell

**Order Book Components:**
- Aggregated Order Book Analysis:
  - Data: `api_spot_orderbook_aggregated_ask_bids_history.parquet`
  - Visualizations: Depth charts, ask/bid ratio trends
  
- Asset-Specific Order Books:
  - Data: Order book files for BTC, ETH, SOL, XRP
  - Visualizations: Comparative order book analysis

**Market Data Components:**
- Pairs Markets Overview:
  - Data: Spot pairs markets files for BTC, ETH, SOL, XRP
  - Visualizations: Market data tables with key metrics
  
- Supported Coins:
  - Data: `api_spot_supported_coins.parquet`
  - Visualizations: Table of supported coins with market count

**Taker Buy/Sell Components:**
- Aggregated Buy/Sell Analysis:
  - Data: `api_spot_aggregated_taker_buy_sell_volume_history.parquet`
  - Visualizations: Buy/sell ratio trends

**Implementation Details:**
- Create tabbed interface for different spot market aspects
- Implement exchange filters for more focused analysis
- Add comparison between spot and futures metrics
- Include volume profile visualizations

### 6.5 Indicators Page

The Indicators page will showcase various market indicators and metrics.

**Components:**
- Fear & Greed Index:
  - Data: `api_index_fear_greed_history.parquet`
  - Visualizations: Historical fear & greed index with interpretation

- Bitcoin Cycle Indicators:
  - Data: Various indicator files (Rainbow Chart, Stock-to-Flow, etc.)
  - Visualizations: Multi-indicator dashboard with current position

- Margin and Basis Analysis:
  - Data: `api_bitfinex_margin_long_short.parquet`, `api_futures_basis_history.parquet`
  - Visualizations: Margin positioning, basis spreads

- Stablecoin Market Cap:
  - Data: `api_index_stableCoin_marketCap_history.parquet`
  - Visualizations: Stablecoin market cap trends

**Implementation Details:**
- Create a dashboard-style layout for indicators
- Implement interpretative elements explaining each indicator
- Add historical context and benchmarks
- Include alert levels for extreme readings

### 6.6 On-Chain Page

The On-Chain page will focus on blockchain data and metrics.

**Components:**
- Exchange Balance Overview:
  - Data: `api_exchange_balance_list.parquet`
  - Visualizations: Exchange balance table, change metrics
  
- Exchange Chain Transactions:
  - Data: `api_exchange_chain_tx_list.parquet`
  - Visualizations: Inflow/outflow analysis, net flow metrics

**Implementation Details:**
- Create visual representations of exchange flows
- Implement time-based comparisons (1d, 7d, 30d changes)
- Add explanatory context for on-chain metrics
- Include notable wallet tracking if data is available

### 6.7 Options Page

The Options page will focus on cryptocurrency options market data.

**Subpages:**
- BTC Options
- ETH Options
- SOL Options
- XRP Options

**Components for Each Asset:**
- Options Market Overview:
  - Data: Option info files for each asset
  - Visualizations: Open interest by strike, call/put ratio
  
- Max Pain Analysis:
  - Data: Max pain files for each asset
  - Visualizations: Max pain point visualization, historical tracking

**Implementation Details:**
- Create options chain visualizations
- Implement expiry date selectors
- Add implied volatility analysis
- Include options strategies explanations

### 6.8 Historical Reports & Data Page

This page will allow users to access and download historical data.

**Components:**
- Data Calendar:
  - Functionality: Calendar view showing available data dates
  
- Data Browser:
  - Functionality: Browsable interface to navigate the data directory
  
- Download Options:
  - Functionality: Select and download data as CSV files
  
- Data Summary:
  - Functionality: Show data file information and stats

**Implementation Details:**
- Create an intuitive file browser for the data directory
- Implement batch download functionality
- Add data previews before download
- Include data documentation and field explanations

## 7. Data Loading Strategy

### 7.1 Data Loading Functions

```python
import pandas as pd
import os
import glob
from datetime import datetime
import streamlit as st

@st.cache_data(ttl=3600)  # Cache for 1 hour
def list_data_directories():
    """List all date directories in the data folder"""
    data_path = "../data/"
    return sorted([d for d in os.listdir(data_path) 
                  if os.path.isdir(os.path.join(data_path, d))], 
                  reverse=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_latest_data_directory():
    """Get the most recent data directory"""
    dirs = list_data_directories()
    return dirs[0] if dirs else None

@st.cache_data
def load_parquet_file(file_path):
    """Load a parquet file into a pandas DataFrame"""
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return pd.DataFrame()

@st.cache_data
def load_data_for_category(category, subcategory=None, asset=None, latest_only=True):
    """
    Load data for a specific category and optional subcategory
    
    Parameters:
    -----------
    category : str
        Main data category (e.g., 'etf', 'futures')
    subcategory : str, optional
        Subcategory (e.g., 'funding_rate', 'liquidation')
    asset : str, optional
        Asset filter (e.g., 'BTC', 'ETH')
    latest_only : bool
        If True, load only from the latest data directory
        
    Returns:
    --------
    dict
        Dictionary of DataFrames with file names as keys
    """
    data_dict = {}
    
    # Determine base path
    if latest_only:
        latest_dir = get_latest_data_directory()
        if not latest_dir:
            return data_dict
        base_path = f"../data/{latest_dir}/"
    else:
        base_path = "../data/"
    
    # Build search path
    if subcategory:
        search_path = f"{base_path}{category}/{subcategory}/"
    else:
        search_path = f"{base_path}{category}/"
    
    # Add asset filter if provided
    if asset:
        search_pattern = f"*{asset}*.parquet"
    else:
        search_pattern = "*.parquet"
    
    # Find and load files
    for file_path in glob.glob(search_path + search_pattern):
        file_name = os.path.basename(file_path).replace('.parquet', '')
        data_dict[file_name] = load_parquet_file(file_path)
    
    return data_dict
```

### 7.2 Data Processing Functions

```python
def process_timestamps(df, timestamp_col='time'):
    """Convert timestamp column to datetime"""
    if timestamp_col in df.columns:
        df['datetime'] = pd.to_datetime(df[timestamp_col], unit='ms')
    return df

def calculate_metrics(df, category):
    """Calculate relevant metrics based on data category"""
    metrics = {}
    
    if category == 'etf':
        if 'aum_usd' in df.columns:
            metrics['Total AUM'] = df['aum_usd'].sum()
        if 'fund_flow_usd' in df.columns:
            metrics['Net Flow (24h)'] = df['fund_flow_usd'].sum()
    
    elif category == 'futures_liquidation':
        if 'aggregated_long_liquidation_usd' in df.columns and 'aggregated_short_liquidation_usd' in df.columns:
            metrics['Total Liquidations'] = df['aggregated_long_liquidation_usd'].sum() + df['aggregated_short_liquidation_usd'].sum()
            metrics['Long Liquidations'] = df['aggregated_long_liquidation_usd'].sum()
            metrics['Short Liquidations'] = df['aggregated_short_liquidation_usd'].sum()
    
    # Add more categories as needed
    
    return metrics
```

## 8. Reusable Components

### 8.1 Sidebar Navigation

```python
# components/sidebar.py

import streamlit as st
from datetime import datetime, timedelta

def render_sidebar():
    """Render the sidebar navigation"""
    
    st.sidebar.title("Izun Crypto Liquidity Report")
    
    # Date selector
    latest_date = datetime.now().date()
    default_start_date = latest_date - timedelta(days=7)
    
    st.sidebar.subheader("Date Range")
    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", latest_date)
    
    # Asset selector
    st.sidebar.subheader("Assets")
    selected_assets = st.sidebar.multiselect(
        "Select Assets",
        ["BTC", "ETH", "SOL", "XRP"],
        default=["BTC"]
    )
    
    # Exchange selector (for relevant pages)
    if st.session_state.get('current_page') in ['futures', 'spot', 'options']:
        st.sidebar.subheader("Exchanges")
        exchanges = ["Binance", "OKX", "Bybit", "dYdX", "All"]
        selected_exchanges = st.sidebar.multiselect(
            "Select Exchanges",
            exchanges,
            default=["All"]
        )
    
    # Navigation links
    st.sidebar.subheader("Navigation")
    
    pages = {
        "Report": "01_report",
        "ETF": "02_etf",
        "Futures": "03_futures",
        "Spot": "04_spot",
        "Indicators": "05_indicators",
        "On-Chain": "06_on_chain",
        "Options": "07_options",
        "Historical Reports & Data": "08_historical"
    }
    
    for page_name, page_url in pages.items():
        st.sidebar.page_link(f"pages/{page_url}.py", label=page_name)
    
    # Credits and data information
    st.sidebar.divider()
    st.sidebar.caption("Data source: CoinGlass API")
    st.sidebar.caption(f"Last updated: {latest_date}")
```

### 8.2 Chart Components

```python
# components/charts.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_time_series(df, x_col, y_col, title, color_col=None, height=400):
    """Create a time series line chart"""
    fig = px.line(
        df, 
        x=x_col, 
        y=y_col, 
        title=title,
        color=color_col,
        height=height
    )
    
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=y_col if isinstance(y_col, str) else None,
        hovermode="x unified",
        legend_title=None
    )
    
    return fig

def create_ohlc_chart(df, datetime_col, open_col, high_col, low_col, close_col, title, height=500):
    """Create an OHLC candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df[datetime_col],
        open=df[open_col],
        high=df[high_col],
        low=df[low_col],
        close=df[close_col]
    )])
    
    fig.update_layout(
        title=title,
        xaxis_title=None,
        yaxis_title=None,
        height=height,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def create_bar_chart(df, x_col, y_col, title, color=None, height=400):
    """Create a bar chart"""
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        color=color,
        height=height
    )
    
    fig.update_layout(
        xaxis_title=None,
        yaxis_title=y_col if isinstance(y_col, str) else None,
        legend_title=None
    )
    
    return fig

def create_time_series_with_bar(df, x_col, line_y_col, bar_y_col, title, height=500):
    """Create a combination chart with line and bar"""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add bar chart
    fig.add_trace(
        go.Bar(
            x=df[x_col], 
            y=df[bar_y_col], 
            name=bar_y_col if isinstance(bar_y_col, str) else "Bar"
        ),
        secondary_y=False,
    )
    
    # Add line chart
    fig.add_trace(
        go.Scatter(
            x=df[x_col], 
            y=df[line_y_col], 
            name=line_y_col if isinstance(line_y_col, str) else "Line", 
            mode="lines"
        ),
        secondary_y=True,
    )
    
    fig.update_layout(
        title=title,
        xaxis_title=None,
        hovermode="x unified",
        height=height,
        legend_title=None
    )
    
    return fig

def create_pie_chart(df, values_col, names_col, title, height=400):
    """Create a pie chart"""
    fig = px.pie(
        df,
        values=values_col,
        names=names_col,
        title=title,
        height=height
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    return fig
```

### 8.3 Metrics Components

```python
# components/metrics.py

import streamlit as st
import pandas as pd
import numpy as np

def format_large_number(num):
    """Format large numbers with appropriate suffixes"""
    if num >= 1_000_000_000:
        return f"${num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"${num/1_000_000:.2f}M"
    elif num >= 1_000:
        return f"${num/1_000:.2f}K"
    else:
        return f"${num:.2f}"

def format_percentage(pct):
    """Format percentage values"""
    return f"{pct:.2f}%"

def display_metric_card(title, value, delta=None, delta_suffix="%", formatter=None):
    """Display a metric card with optional delta"""
    if formatter:
        formatted_value = formatter(value)
    else:
        formatted_value = value
    
    if delta is not None:
        st.metric(
            label=title,
            value=formatted_value,
            delta=f"{delta:.2f}{delta_suffix}"
        )
    else:
        st.metric(
            label=title,
            value=formatted_value
        )

def display_metrics_row(metrics_dict, formatters=None):
    """Display a row of metrics"""
    cols = st.columns(len(metrics_dict))
    
    if formatters is None:
        formatters = {key: None for key in metrics_dict}
    
    for i, (key, value) in enumerate(metrics_dict.items()):
        with cols[i]:
            display_metric_card(
                key, 
                value.get('value', value) if isinstance(value, dict) else value,
                value.get('delta') if isinstance(value, dict) else None,
                formatter=formatters.get(key)
            )
```

### 8.4 Table Components

```python
# components/tables.py

import streamlit as st
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
from st_aggrid.shared import GridUpdateMode

def create_data_table(df, key=None, selection_mode=None, height=400, pagination=True, fit_columns=True):
    """Create an interactive data table using AgGrid"""
    # Configure grid options
    gb = GridOptionsBuilder.from_dataframe(df)
    
    # Enable selection if mode is specified
    if selection_mode:
        gb.configure_selection(selection_mode, use_checkbox=True)
    
    # Configure pagination
    if pagination:
        gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
    
    # Set grid height
    gb.configure_grid_options(domLayout='normal', rowHeight=30)
    
    # Build grid options
    grid_options = gb.build()
    
    # Create the AgGrid component
    grid_response = AgGrid(
        df,
        gridOptions=grid_options,
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=fit_columns,
        height=height,
        allow_unsafe_jscode=True,
        key=key
    )
    
    return grid_response

def create_simple_table(df, format_dict=None):
    """Create a simple formatted Streamlit table"""
    # Apply formatting if provided
    if format_dict:
        formatted_df = df.copy()
        for col, fmt in format_dict.items():
            if col in formatted_df.columns:
                formatted_df[col] = formatted_df[col].apply(fmt)
        st.table(formatted_df)
    else:
        st.table(df)
```

## 9. Main Application Structure

### 9.1 Main App Entry Point (app.py)

```python
# streamlit/app.py

import streamlit as st
import pandas as pd
import os
from datetime import datetime
from components.sidebar import render_sidebar
from utils.data_loader import get_latest_data_directory, load_data_for_category

# Set page title and description
st.set_page_config(
    page_title="Izun Crypto Liquidity Report",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for filters and data
if 'selected_assets' not in st.session_state:
    st.session_state.selected_assets = ["BTC"]

if 'selected_exchanges' not in st.session_state:
    st.session_state.selected_exchanges = ["All"]

if 'date_range' not in st.session_state:
    st.session_state.date_range = {
        'start': datetime.now().date(),
        'end': datetime.now().date()
    }

# Set current page
st.session_state.current_page = 'report'

# Render sidebar
render_sidebar()

# Main page content
st.title("Izun Crypto Liquidity Report")
st.write("Welcome to the Izun Crypto Liquidity Report dashboard. This application provides comprehensive analytics on cryptocurrency liquidity across various market segments.")

# Get the latest data directory
latest_data_dir = get_latest_data_directory()
st.write(f"Latest data: {latest_data_dir}")

# Create dashboard layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("ETF Flows")
    # Load ETF data
    etf_data = load_data_for_category('etf', subcategory=None, asset=None)
    if 'api_etf_bitcoin_flow_history' in etf_data:
        st.write(f"Bitcoin ETF Flow Data: {len(etf_data['api_etf_bitcoin_flow_history'])} records")
    
with col2:
    st.subheader("Futures Liquidations")
    # Load liquidation data
    liquidation_data = load_data_for_category('futures', subcategory='liquidation', asset='BTC')
    if 'api_futures_liquidation_aggregated_coin_history_BTC_BTC' in liquidation_data:
        st.write(f"BTC Liquidation Data: {len(liquidation_data['api_futures_liquidation_aggregated_coin_history_BTC_BTC'])} records")

with col3:
    st.subheader("Market Indicators")
    # Load indicator data
    indicator_data = load_data_for_category('indic', subcategory=None, asset=None)
    if 'api_index_fear_greed_history' in indicator_data:
        st.write(f"Fear & Greed Index: {len(indicator_data['api_index_fear_greed_history'])} records")

# Link to other pages
st.subheader("Explore the Data")
st.write("Use the sidebar navigation to explore different data categories and visualizations.")
```

### 9.2 Report Page Example (01_report.py)

```python
# streamlit/pages/01_report.py

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import from components and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from components.sidebar import render_sidebar
from components.metrics import display_metrics_row, format_large_number, format_percentage
from components.charts import create_time_series, create_bar_chart, create_time_series_with_bar
from utils.data_loader import load_data_for_category, process_timestamps

# Set page title and description
st.set_page_config(
    page_title="Izun Crypto Liquidity Report - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set current page
st.session_state.current_page = 'report'

# Render sidebar
render_sidebar()

# Main page content
st.title("Crypto Liquidity Dashboard")
st.write("Overview of key metrics and trends across all market segments")

# Date filter for dashboard
col1, col2 = st.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now().date() - timedelta(days=30)
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now().date()
    )

# Market Summary Metrics
st.subheader("Market Summary")

# Load key data for metrics
btc_price_data = load_data_for_category('futures', 'market', 'BTC')
etf_flow_data = load_data_for_category('etf', asset='bitcoin')
liquidation_data = load_data_for_category('futures', 'liquidation', 'BTC')

# Calculate metrics
metrics = {
    "BTC Price": {
        "value": 65000,  # Example value, would be calculated from real data
        "delta": 2.3
    },
    "24h ETF Flows": {
        "value": 125000000,  # Example value
        "delta": -10.5
    },
    "24h Liquidations": {
        "value": 250000000,  # Example value
        "delta": 35.2
    },
    "Funding Rate": {
        "value": 0.01,  # Example value
        "delta": 0.002
    }
}

formatters = {
    "BTC Price": lambda x: f"${x:,.2f}",
    "24h ETF Flows": format_large_number,
    "24h Liquidations": format_large_number,
    "Funding Rate": lambda x: f"{x:.4f}%"
}

display_metrics_row(metrics, formatters)

# Create tabs for different chart categories
tab1, tab2, tab3, tab4 = st.tabs(["ETF Flows", "Liquidations", "Open Interest", "Funding Rates"])

with tab1:
    st.subheader("Bitcoin ETF Flows")
    # Here you would load and visualize ETF flow data
    if 'api_etf_bitcoin_flow_history' in etf_flow_data:
        flow_df = etf_flow_data['api_etf_bitcoin_flow_history']
        flow_df = process_timestamps(flow_df)
        
        # Create chart
        fig = create_time_series_with_bar(
            flow_df,
            'datetime',
            'price_change_percent',
            'fund_flow_usd',
            "Bitcoin ETF Flows vs Price Change"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Futures Liquidations")
    # Here you would load and visualize liquidation data
    if 'api_futures_liquidation_aggregated_coin_history_BTC_BTC' in liquidation_data:
        liq_df = liquidation_data['api_futures_liquidation_aggregated_coin_history_BTC_BTC']
        liq_df = process_timestamps(liq_df)
        
        # Create chart
        fig = create_bar_chart(
            liq_df,
            'datetime',
            ['aggregated_long_liquidation_usd', 'aggregated_short_liquidation_usd'],
            "Bitcoin Futures Liquidations"
        )
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Open Interest")
    # Open Interest visualizations would go here
    st.info("Open Interest charts will be implemented")

with tab4:
    st.subheader("Funding Rates")
    # Funding rate visualizations would go here
    st.info("Funding Rate charts will be implemented")

# Market Indicators
st.subheader("Market Indicators")
indicators_data = load_data_for_category('indic')

# Create columns for indicators
col1, col2 = st.columns(2)

with col1:
    st.write("Fear & Greed Index")
    # Fear & Greed index visualization would go here
    st.info("Fear & Greed Index chart will be implemented")

with col2:
    st.write("Bitcoin Rainbow Chart")
    # Rainbow chart visualization would go here
    st.info("Bitcoin Rainbow Chart will be implemented")
```

## 10. Implementation Plan

The implementation of the Streamlit application will follow these phases:

### Phase 1: Core Infrastructure (Week 1)
1. Set up project structure and basic navigation
2. Implement data loading utilities
3. Create shared components (sidebar, charts, tables)
4. Implement base page layout
5. Set up data updating mechanisms

### Phase 2: Main Pages Implementation (Weeks 2-3)
1. Implement Report/Dashboard page
2. Implement ETF page
3. Implement Futures pages
4. Implement Spot page
5. Implement Indicators page
6. Implement basic visualizations for all sections

### Phase 3: Advanced Features (Week 4)
1. Enhance data filtering and selection options
2. Implement comparative analysis features
3. Add interactive elements to visualizations
4. Implement historical data download page
5. Add data export functionality

### Phase 4: Optimization and Testing (Week 5)
1. Optimize data loading for performance
2. Enhance mobile responsiveness
3. Comprehensive testing across browsers
4. Fix bugs and address usability issues
5. Ensure consistent styling and branding

### Phase 5: Documentation and Deployment (Week 6)
1. Complete code documentation
2. Create user guide and documentation
3. Set up deployment configuration
4. Final testing and validation
5. Launch application

## 11. Conclusion

This plan provides a comprehensive roadmap for building the Izun Crypto Liquidity Report Streamlit application. The application will deliver a user-friendly interface for exploring and analyzing cryptocurrency market data across various segments.

Key features of the application include:
- Intuitive navigation matching the data folder structure
- Interactive visualizations for all data categories
- Historical data comparison
- Downloadable data exports
- Comprehensive market insights

The modular design and component-based architecture will ensure maintainability and extensibility, allowing for future enhancements and additions as new data sources become available.