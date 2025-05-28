"""
Report page for the Izun Crypto Liquidity Report.

This page serves as the main dashboard/overview page focused on liquidity metrics and intraday changes.
It provides a comprehensive view of market conditions across different assets, with emphasis on
liquidity indicators such as open interest, trading volumes, and market depth.

This report should include:
1. Table of 4 main crypto (ETH, BTC, XRP, SOL) and their changes in the spot and futures market (daily, weekly, monthly change)
   - Data files: 
     - Spot price data: data/[LATEST_DATE]/spot/spot_market/api_spot_price_history_[ASSET].parquet
     - Futures price data: data/[LATEST_DATE]/futures/market/api_futures_pairs_markets_[ASSET].parquet,
       data/[LATEST_DATE]/futures/market/api_price_ohlc_history.parquet
   - Structure: The spot price files contain columns like 'timestamp', 'price', 'change_24h', etc.
     The futures price files contain exchange-specific data with columns like 'price', 'volume_24h', etc.
   - Implementation: Create a table showing each asset with columns for current price, 24h change,
     7d change, 30d change for both spot and futures markets. Default timeframe is intraday (24h).
     Use the tables.py create_crypto_table() or create_formatted_table() component.

2. Chart of the spot crypto market (default is daily change, with the option to show weekly and month change)
   - Data files: data/[LATEST_DATE]/spot/spot_market/api_spot_price_history_[ASSET].parquet
   - Structure: Contains timestamp/datetime and price data for each asset over time
   - Implementation: Create a line chart showing price movements for all 4 main assets (BTC, ETH, XRP, SOL)
     using the charts.py create_time_series() component with a default timeframe of 24h (intraday),
     with options to switch to weekly or monthly views. Will use 'datetime' as x_col and 'price' as y_col.

3. Chart of the funding rate values for different exchanges (default is daily change, with the option to show weekly and month change)
   - Data files: 
     - Per exchange: data/[LATEST_DATE]/futures/funding_rate/api_futures_fundingRate_ohlc_history_[ASSET]_[EXCHANGE].parquet
     - Aggregated: data/[LATEST_DATE]/futures/funding_rate/api_futures_fundingRate_oi_weight_ohlc_history_[ASSET].parquet, 
       data/[LATEST_DATE]/futures/funding_rate/api_futures_fundingRate_vol_weight_ohlc_history_[ASSET].parquet
   - Structure: Contains timestamp, funding rate data for each exchange or aggregated by weighting method
   - Implementation: Create a multi-line chart showing funding rates across major exchanges for a selected asset
     (default BTC) with a default timeframe of 24h (intraday), using charts.py create_time_series().
     Will have options to switch between assets and timeframes.

4. Chart of the open interest changes for different exchanges (default is daily change, with the option to show weekly and month change)
   - Data files:
     - Exchange-specific: data/[LATEST_DATE]/futures/open_interest/api_futures_openInterest_exchange_list_[ASSET].parquet,
       data/[LATEST_DATE]/futures/open_interest/api_futures_openInterest_exchange_history_chart_[ASSET].parquet
     - Aggregated data: data/[LATEST_DATE]/futures/open_interest/api_futures_openInterest_ohlc_aggregated_history_[ASSET].parquet
   - Structure: Contains timestamp, open interest values in USD and coin units across various exchanges
   - Implementation: Create a stacked area chart or multi-line chart showing open interest across major exchanges
     for a selected asset (default BTC) with a default timeframe of 24h (intraday). Will use
     charts.py create_area_chart() or create_time_series() with options to switch between assets and timeframes.

All charts will have a default intraday view (24h) with the ability to toggle to see weekly (7d) and monthly (30d)
timeframes. The implementation will use a single tab containing all 4 elements arranged in a logical flow from
summary table at the top to detailed charts below.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to path to import from components and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components and utilities
from components.sidebar import render_sidebar
from components.metrics import display_metrics_row, display_metric_card
from components.charts import (
    create_time_series, 
    create_ohlc_chart, 
    create_bar_chart, 
    create_time_series_with_bar,
    create_area_chart,
    apply_chart_theme,
    display_chart
)
from components.tables import create_formatted_table, create_crypto_table
from utils.data_loader import (
    get_latest_data_directory, 
    load_data_for_category, 
    process_timestamps,
    get_data_last_updated,
    calculate_metrics,
    load_specific_data_file
)
from utils.formatters import (
    format_currency, 
    format_percentage, 
    format_volume,
    format_timestamp,
    humanize_time_diff
)
from utils.config import APP_TITLE, APP_ICON, SUPPORTED_ASSETS

# Set page config with title and icon
st.set_page_config(
    page_title=f"{APP_TITLE} - Report",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the current page for sidebar navigation
st.session_state.current_page = 'report'

def load_report_data():
    """
    Load data for the report page.
    
    Returns:
    --------
    dict
        Dictionary containing data for the report
    """
    data = {}
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return data
    
    # Load spot market data for BTC, ETH, SOL, XRP
    spot_data = {}
    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
        spot_asset_data = load_data_for_category('spot', 'spot_market', asset, latest_dir)
        if spot_asset_data:
            spot_data[asset] = spot_asset_data
    
    data['spot'] = spot_data
    
    # Load futures market data
    futures_data = {}
    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
        futures_asset_data = load_data_for_category('futures', 'market', asset, latest_dir)
        if futures_asset_data:
            futures_data[asset] = futures_asset_data
    
    data['futures'] = futures_data
    
    # Load funding rate data
    funding_rate_data = {}
    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
        funding_asset_data = load_data_for_category('futures', 'funding_rate', asset, latest_dir)
        if funding_asset_data:
            funding_rate_data[asset] = funding_asset_data
    
    data['funding_rate'] = funding_rate_data
    
    # Load open interest data
    open_interest_data = {}
    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
        oi_asset_data = load_data_for_category('futures', 'open_interest', asset, latest_dir)
        if oi_asset_data:
            open_interest_data[asset] = oi_asset_data
    
    data['open_interest'] = open_interest_data
    
    # Load funding rate exchange list data for tables
    try:
        # Current funding rates - load directly as these are special format files
        # Ensure we have the full path including 'data' directory
        if not latest_dir.startswith('data'):
            data_path = os.path.join('data', latest_dir)
        else:
            data_path = latest_dir
            
        current_funding_file = os.path.join(data_path, 'futures', 'funding_rate', 'api_futures_fundingRate_exchange_list.parquet')
        if os.path.exists(current_funding_file):
            data['current_funding_rates'] = pd.read_parquet(current_funding_file)
            logger.info(f"Loaded current funding rates: {len(data['current_funding_rates'])} rows")
            logger.info(f"Current funding rates columns: {data['current_funding_rates'].columns.tolist()}")
        else:
            logger.warning(f"Current funding rates file not found: {current_funding_file}")
            
        # Accumulated funding rates - load directly as these are special format files  
        accumulated_funding_file = os.path.join(data_path, 'futures', 'funding_rate', 'api_futures_fundingRate_accumulated_exchange_list.parquet')
        if os.path.exists(accumulated_funding_file):
            data['accumulated_funding_rates'] = pd.read_parquet(accumulated_funding_file)
            logger.info(f"Loaded accumulated funding rates: {len(data['accumulated_funding_rates'])} rows")
            logger.info(f"Accumulated funding rates columns: {data['accumulated_funding_rates'].columns.tolist()}")
        else:
            logger.warning(f"Accumulated funding rates file not found: {accumulated_funding_file}")
    except Exception as e:
        logger.error(f"Error loading funding rate exchange list data: {e}")
    
    return data

def prepare_price_comparison_dataframe(data):
    """
    Prepare a comparison dataframe for the price table showing spot and futures data
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with price comparison data for BTC, ETH, SOL, XRP
    """
    comparison_data = []
    
    # Filter to the 4 supported assets
    assets = SUPPORTED_ASSETS[:4]  # BTC, ETH, SOL, XRP
    
    for asset in assets:
        asset_data = {
            'Asset': asset
        }
        
        # Get spot price data
        if 'spot' in data and asset in data['spot']:
            try:
                spot_price_history_key = f'api_spot_price_history_{asset}'
                if spot_price_history_key in data['spot'][asset]:
                    spot_df = data['spot'][asset][spot_price_history_key]
                    spot_df = process_timestamps(spot_df)
                    
                    if not spot_df.empty and 'close' in spot_df.columns:
                        spot_df = spot_df.sort_values('datetime')
                        asset_data['Spot Price'] = spot_df['close'].iloc[-1]
                        
                        # Calculate changes over different periods
                        now = spot_df['datetime'].max()
                        
                        # 24h change
                        day_ago = now - timedelta(days=1)
                        day_df = spot_df[spot_df['datetime'] >= day_ago]
                        if not day_df.empty:
                            first_price = day_df['close'].iloc[0]
                            last_price = day_df['close'].iloc[-1]
                            asset_data['Spot 24h Change'] = (last_price - first_price) / first_price
                        
                        # 7d change
                        week_ago = now - timedelta(days=7)
                        week_df = spot_df[spot_df['datetime'] >= week_ago]
                        if not week_df.empty:
                            first_price = week_df['close'].iloc[0]
                            last_price = week_df['close'].iloc[-1]
                            asset_data['Spot 7d Change'] = (last_price - first_price) / first_price
                        
                        # 30d change
                        month_ago = now - timedelta(days=30)
                        month_df = spot_df[spot_df['datetime'] >= month_ago]
                        if not month_df.empty:
                            first_price = month_df['close'].iloc[0]
                            last_price = month_df['close'].iloc[-1]
                            asset_data['Spot 30d Change'] = (last_price - first_price) / first_price
            except Exception as e:
                logger.error(f"Error processing spot data for {asset}: {e}")
        
        # Get futures price data
        if 'futures' in data and asset in data['futures']:
            try:
                futures_markets_key = f'api_futures_pairs_markets_{asset}'
                if futures_markets_key in data['futures'][asset]:
                    futures_df = data['futures'][asset][futures_markets_key]
                    
                    if not futures_df.empty and 'current_price' in futures_df.columns:
                        # Use the average price across all exchanges
                        asset_data['Futures Price'] = futures_df['current_price'].mean()
                        
                        # Use the average price change across all exchanges
                        if 'price_change_percent_24h' in futures_df.columns:
                            asset_data['Futures 24h Change'] = futures_df['price_change_percent_24h'].mean() / 100
                
                # For 7d and 30d changes, we need OHLC data from price_ohlc_history
                price_ohlc_key = 'api_price_ohlc_history'
                if price_ohlc_key in data['futures'][asset]:
                    ohlc_df = data['futures'][asset][price_ohlc_key]
                    ohlc_df = process_timestamps(ohlc_df)
                    
                    if not ohlc_df.empty and 'close' in ohlc_df.columns:
                        ohlc_df = ohlc_df.sort_values('datetime')
                        
                        now = ohlc_df['datetime'].max()
                        
                        # 7d change
                        week_ago = now - timedelta(days=7)
                        week_df = ohlc_df[ohlc_df['datetime'] >= week_ago]
                        if not week_df.empty:
                            first_price = week_df['close'].iloc[0]
                            last_price = week_df['close'].iloc[-1]
                            asset_data['Futures 7d Change'] = (last_price - first_price) / first_price
                        
                        # 30d change
                        month_ago = now - timedelta(days=30)
                        month_df = ohlc_df[ohlc_df['datetime'] >= month_ago]
                        if not month_df.empty:
                            first_price = month_df['close'].iloc[0]
                            last_price = month_df['close'].iloc[-1]
                            asset_data['Futures 30d Change'] = (last_price - first_price) / first_price
            except Exception as e:
                logger.error(f"Error processing futures data for {asset}: {e}")
        
        comparison_data.append(asset_data)
    
    return pd.DataFrame(comparison_data)

def prepare_spot_price_chart_data(data, timeframe='24h'):
    """
    Prepare data for the spot price chart
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    timeframe : str
        Timeframe to display (24h, 7d, 30d)
    
    Returns:
    --------
    dict
        Dictionary with chart data for each asset
    """
    chart_data = {}
    
    # Filter to the 4 supported assets
    assets = SUPPORTED_ASSETS[:4]  # BTC, ETH, SOL, XRP
    
    for asset in assets:
        if 'spot' in data and asset in data['spot']:
            try:
                spot_price_history_key = f'api_spot_price_history_{asset}'
                if spot_price_history_key in data['spot'][asset]:
                    df = data['spot'][asset][spot_price_history_key]
                    df = process_timestamps(df)
                    
                    if not df.empty and 'close' in df.columns and 'datetime' in df.columns:
                        df = df.sort_values('datetime')
                        
                        # Filter based on timeframe
                        now = df['datetime'].max()
                        
                        if timeframe == '24h':
                            cutoff = now - timedelta(days=1)
                        elif timeframe == '7d':
                            cutoff = now - timedelta(days=7)
                        elif timeframe == '30d':
                            cutoff = now - timedelta(days=30)
                        else:
                            cutoff = now - timedelta(days=1)  # Default to 24h
                        
                        df = df[df['datetime'] >= cutoff]
                        
                        # Normalize price to compare different assets on same scale
                        first_price = df['close'].iloc[0]
                        df['normalized_price'] = df['close'] / first_price * 100
                        
                        chart_data[asset] = df
            except Exception as e:
                logger.error(f"Error preparing spot price chart data for {asset}: {e}")
    
    return chart_data

def prepare_funding_rate_chart_data(data, asset='BTC', timeframe='24h'):
    """
    Prepare data for the funding rate chart
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    asset : str
        Asset to display (BTC, ETH, SOL, XRP)
    timeframe : str
        Timeframe to display (24h, 7d, 30d)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with funding rate data for the selected asset and timeframe
    """
    combined_df = pd.DataFrame()
    
    if 'funding_rate' in data and asset in data['funding_rate']:
        # Get list of exchanges from the available data files
        exchange_dfs = []
        
        for key in data['funding_rate'][asset].keys():
            if f"api_futures_fundingRate_ohlc_history_{asset}_" in key:
                exchange = key.replace(f"api_futures_fundingRate_ohlc_history_{asset}_", "")
                try:
                    df = data['funding_rate'][asset][key]
                    df = process_timestamps(df)
                    
                    if not df.empty and 'close' in df.columns and 'datetime' in df.columns:
                        # Convert string funding rates to float if needed
                        if df['close'].dtype == 'object':
                            df['funding_rate'] = df['close'].astype(float)
                        else:
                            df['funding_rate'] = df['close']
                        
                        # Add exchange column
                        df['exchange'] = exchange
                        
                        # Keep only necessary columns
                        df = df[['datetime', 'exchange', 'funding_rate']]
                        
                        exchange_dfs.append(df)
                except Exception as e:
                    logger.error(f"Error processing funding rate data for {asset} on {exchange}: {e}")
        
        # Combine all exchanges into one dataframe
        if exchange_dfs:
            combined_df = pd.concat(exchange_dfs, ignore_index=True)
            
            # Filter based on timeframe
            if not combined_df.empty:
                now = combined_df['datetime'].max()
                
                if timeframe == '24h':
                    cutoff = now - timedelta(days=1)
                elif timeframe == '7d':
                    cutoff = now - timedelta(days=7)
                elif timeframe == '30d':
                    cutoff = now - timedelta(days=30)
                else:
                    cutoff = now - timedelta(days=1)  # Default to 24h
                
                combined_df = combined_df[combined_df['datetime'] >= cutoff]
    
    return combined_df

def prepare_open_interest_chart_data(data, asset='BTC', timeframe='24h'):
    """
    Prepare data for the open interest chart
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    asset : str
        Asset to display (BTC, ETH, SOL, XRP)
    timeframe : str
        Timeframe to display (24h, 7d, 30d)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with open interest data for the selected asset and timeframe
    """
    result_df = None
    
    if 'open_interest' in data and asset in data['open_interest']:
        # Try to use aggregated history first
        agg_key = f'api_futures_openInterest_ohlc_aggregated_history_{asset}'
        if agg_key in data['open_interest'][asset]:
            try:
                df = data['open_interest'][asset][agg_key]
                df = process_timestamps(df)
                
                if not df.empty and 'close' in df.columns and 'datetime' in df.columns:
                    df = df.sort_values('datetime')
                    
                    # Filter based on timeframe
                    now = df['datetime'].max()
                    
                    if timeframe == '24h':
                        cutoff = now - timedelta(days=1)
                    elif timeframe == '7d':
                        cutoff = now - timedelta(days=7)
                    elif timeframe == '30d':
                        cutoff = now - timedelta(days=30)
                    else:
                        cutoff = now - timedelta(days=1)  # Default to 24h
                    
                    df = df[df['datetime'] >= cutoff]
                    
                    # Prepare for display
                    df['open_interest'] = df['close']
                    result_df = df
            except Exception as e:
                logger.error(f"Error processing aggregated open interest data for {asset}: {e}")
        
        # If we don't have aggregated data or it failed, try exchange-specific data
        if result_df is None or result_df.empty:
            exchange_chart_key = f'api_futures_openInterest_exchange_history_chart_{asset}'
            if exchange_chart_key in data['open_interest'][asset]:
                try:
                    df = data['open_interest'][asset][exchange_chart_key]
                    df = process_timestamps(df)
                    
                    if not df.empty and 'datetime' in df.columns:
                        # This data is likely already in the right format with exchange-specific columns
                        
                        # Filter based on timeframe
                        now = df['datetime'].max()
                        
                        if timeframe == '24h':
                            cutoff = now - timedelta(days=1)
                        elif timeframe == '7d':
                            cutoff = now - timedelta(days=7)
                        elif timeframe == '30d':
                            cutoff = now - timedelta(days=30)
                        else:
                            cutoff = now - timedelta(days=1)  # Default to 24h
                        
                        df = df[df['datetime'] >= cutoff]
                        
                        result_df = df
                except Exception as e:
                    logger.error(f"Error processing exchange open interest history for {asset}: {e}")
                
    return result_df

def create_spot_chart(chart_data, timeframe='24h'):
    """
    Create a line chart of spot prices
    
    Parameters:
    -----------
    chart_data : dict
        Dictionary with chart data for each asset
    timeframe : str
        Timeframe to display (24h, 7d, 30d)
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The chart figure
    """
    if not chart_data:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # Add a trace for each asset
    for asset, df in chart_data.items():
        if not df.empty:
            fig.add_trace(
                go.Scatter(
                    x=df['datetime'],
                    y=df['normalized_price'],
                    mode='lines',
                    name=asset
                )
            )
    
    # Set chart title based on timeframe
    timeframe_text = "24 Hours" if timeframe == '24h' else ("7 Days" if timeframe == '7d' else "30 Days")
    title = f"Spot Price Performance - Past {timeframe_text} (Normalized %)"
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Price Change (%)",
        hovermode="x unified"
    )
    
    return apply_chart_theme(fig)

def create_funding_rate_chart(df, asset='BTC', timeframe='24h'):
    """
    Create a line chart of funding rates
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with funding rate data
    asset : str
        Asset being displayed
    timeframe : str
        Timeframe being displayed
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The chart figure
    """
    if df is None or df.empty:
        return None
    
    # Create time series chart
    timeframe_text = "24 Hours" if timeframe == '24h' else ("7 Days" if timeframe == '7d' else "30 Days")
    title = f"{asset} Funding Rates by Exchange - Past {timeframe_text}"
    
    fig = create_time_series(
        df=df,
        x_col='datetime',
        y_col='funding_rate',
        title=title,
        color_col='exchange'
    )
    
    # Add a zero line
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="red",
        annotation_text="Neutral"
    )
    
    return fig

def calculate_annualized_rate(current_rate, interval_hours):
    """
    Calculate annualized funding rate.
    
    Parameters:
    -----------
    current_rate : float
        The current funding rate
    interval_hours : float
        The funding interval in hours
    
    Returns:
    --------
    float
        Annualized funding rate
    """
    try:
        if pd.isna(current_rate):
            return None
        if pd.isna(interval_hours) or interval_hours <= 0:
            # Default to 8-hour interval if not specified
            interval_hours = 8.0
        # Formula: current_rate × (24 / interval_hours) × 365
        return float(current_rate) * (24 / float(interval_hours)) * 365
    except Exception as e:
        logger.error(f"Error calculating annualized rate: {e}")
        return None

def prepare_current_funding_rate_table(data, selected_assets=None, selected_exchanges=None, 
                                      rate_display='annualized'):
    """
    Prepare current funding rate data for table display.
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    selected_assets : list, optional
        List of assets to display
    selected_exchanges : list, optional
        List of exchanges to display
    rate_display : str
        'annualized' (default) or 'current'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame ready for table display
    """
    if 'current_funding_rates' not in data or data['current_funding_rates'] is None:
        return pd.DataFrame()
    
    try:
        df = data['current_funding_rates'].copy()
        
        # Use stablecoin margin column
        margin_col = 'stablecoin_margin_list'
        if margin_col not in df.columns:
            logger.error(f"Column {margin_col} not found in current funding rates data")
            return pd.DataFrame()
        
        # Initialize list to store processed data
        processed_data = []
        
        # Process each asset
        for idx, row in df.iterrows():
            symbol = row['symbol']
            
            # Skip if asset filter is applied and asset not selected
            if selected_assets and symbol not in selected_assets:
                continue
            
            # Get funding data for the margin type
            funding_list = row[margin_col]
            if not isinstance(funding_list, (list, np.ndarray)) or len(funding_list) == 0:
                continue
            
            # Create a dictionary to store exchange rates for this asset
            asset_data = {'Asset': symbol}
            
            # Process each exchange
            for exchange_data in funding_list:
                if not isinstance(exchange_data, dict):
                    continue
                    
                exchange = exchange_data.get('exchange', '')
                
                # Skip if exchange filter is applied and exchange not selected
                if selected_exchanges and exchange not in selected_exchanges:
                    continue
                
                funding_rate = exchange_data.get('funding_rate', None)
                interval = exchange_data.get('funding_rate_interval', 8.0)
                
                # Calculate rate based on display mode
                if rate_display == 'annualized' and funding_rate is not None:
                    display_rate = calculate_annualized_rate(funding_rate, interval)
                else:
                    display_rate = funding_rate
                
                # Store the rate
                asset_data[exchange] = display_rate
            
            # Only add if we have at least one exchange rate
            if len(asset_data) > 1:  # More than just the 'Asset' key
                processed_data.append(asset_data)
        
        # Create DataFrame from processed data
        result_df = pd.DataFrame(processed_data)
        
        # Sort by asset symbol
        if not result_df.empty:
            result_df = result_df.sort_values('Asset')
            
            # Limit to top 20 assets if no specific assets selected
            if not selected_assets and len(result_df) > 20:
                result_df = result_df.head(20)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error preparing current funding rate table: {e}")
        return pd.DataFrame()

def prepare_accumulated_funding_rate_table(data, selected_assets=None, selected_exchanges=None):
    """
    Prepare accumulated funding rate data for table display.
    Note: Data represents 365 days of accumulated funding.
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    selected_assets : list, optional
        List of assets to display
    selected_exchanges : list, optional
        List of exchanges to display
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame ready for table display with summary row
    """
    if 'accumulated_funding_rates' not in data or data['accumulated_funding_rates'] is None:
        return pd.DataFrame()
    
    try:
        df = data['accumulated_funding_rates'].copy()
        
        # Use stablecoin margin column
        margin_col = 'stablecoin_margin_list'
        if margin_col not in df.columns:
            logger.error(f"Column {margin_col} not found in accumulated funding rates data")
            return pd.DataFrame()
        
        # Initialize list to store processed data
        processed_data = []
        
        # Process each asset
        for idx, row in df.iterrows():
            symbol = row['symbol']
            
            # Skip if asset filter is applied and asset not selected
            if selected_assets and symbol not in selected_assets:
                continue
            
            # Get funding data for the margin type
            funding_list = row[margin_col]
            if not isinstance(funding_list, (list, np.ndarray)) or len(funding_list) == 0:
                continue
            
            # Create a dictionary to store exchange rates for this asset
            asset_data = {'Asset': symbol}
            
            # Process each exchange
            for exchange_data in funding_list:
                if not isinstance(exchange_data, dict):
                    continue
                    
                exchange = exchange_data.get('exchange', '')
                
                # Skip if exchange filter is applied and exchange not selected
                if selected_exchanges and exchange not in selected_exchanges:
                    continue
                
                funding_rate = exchange_data.get('funding_rate', None)
                
                # Store the accumulated rate (already represents 365 days)
                asset_data[exchange] = funding_rate
            
            # Only add if we have at least one exchange rate
            if len(asset_data) > 1:  # More than just the 'Asset' key
                processed_data.append(asset_data)
        
        # Create DataFrame from processed data
        result_df = pd.DataFrame(processed_data)
        
        if not result_df.empty:
            # Sort by asset symbol
            result_df = result_df.sort_values('Asset')
            
            # Limit to top 20 assets if no specific assets selected
            if not selected_assets and len(result_df) > 20:
                result_df = result_df.head(20)
            
            # Calculate average for each exchange (summary row)
            exchange_cols = [col for col in result_df.columns if col != 'Asset']
            if exchange_cols:
                avg_data = {'Asset': 'Average'}
                for exchange in exchange_cols:
                    # Calculate average, ignoring NaN values
                    avg_data[exchange] = result_df[exchange].mean()
                
                # Add summary row
                result_df = pd.concat([result_df, pd.DataFrame([avg_data])], ignore_index=True)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error preparing accumulated funding rate table: {e}")
        return pd.DataFrame()

def create_open_interest_chart(df, asset='BTC', timeframe='24h'):
    """
    Create a chart of open interest
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with open interest data
    asset : str
        Asset being displayed
    timeframe : str
        Timeframe being displayed
    
    Returns:
    --------
    plotly.graph_objects.Figure
        The chart figure
    """
    if df is None or df.empty:
        return None
    
    timeframe_text = "24 Hours" if timeframe == '24h' else ("7 Days" if timeframe == '7d' else "30 Days")
    title = f"{asset} Open Interest - Past {timeframe_text}"
    
    # Check if we have a column for open interest or if this is exchange-specific data
    if 'open_interest' in df.columns:
        # Simple time series
        fig = create_time_series(
            df=df,
            x_col='datetime',
            y_col='open_interest',
            title=title
        )
    else:
        # This is likely exchange-specific data with multiple columns
        # Let's create an area chart with exchanges as separate series
        # First identify exchange columns
        exchange_cols = [col for col in df.columns if col not in ['datetime', 'time']]
        
        if exchange_cols:
            # Create area chart
            fig = create_area_chart(
                df=df,
                x_col='datetime',
                y_col=exchange_cols,
                title=title,
                stacked=True
            )
        else:
            # Fallback
            fig = go.Figure()
            fig.update_layout(title=title)
    
    return fig

def main():
    """Main function to render the report page."""
    
    # Render sidebar
    render_sidebar()
    
    # Page title and description
    st.title("Crypto Liquidity Report")
    st.write("Overview of market conditions, focusing on intraday liquidity metrics")
    
    # Display latest data date
    latest_dir = get_latest_data_directory()
    if latest_dir:
        # Extract date from directory name (format: YYYYMMDD)
        date_str = os.path.basename(latest_dir)
        try:
            # Parse the date string and format it nicely
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            formatted_date = date_obj.strftime('%Y-%m-%d')
            st.caption(f"Latest data from: {formatted_date}")
        except ValueError:
            # If date parsing fails, just show the directory name
            st.caption(f"Latest data from: {date_str}")
    
    # Display loading message
    with st.spinner("Loading data..."):
        data = load_report_data()
    
    # Check if data is available
    if not data:
        st.error("No data available. Please check your data sources.")
        return
    
    # Create a single main section
    main_container = st.container()
    
    with main_container:
        # Create time range selector above charts
        time_ranges = {
            '24h': 'Intraday (24h)',
            '7d': 'Weekly (7d)',
            '30d': 'Monthly (30d)'
        }
        selected_timeframe = st.radio("Select Time Range", options=list(time_ranges.keys()), format_func=lambda x: time_ranges[x], horizontal=True, index=0)
        
        # 1. Summary price table
        st.subheader("Price Comparison: Spot vs Futures Markets")
        
        price_df = prepare_price_comparison_dataframe(data)
        
        if not price_df.empty:
            # Define formatting for the table
            format_dict = {
                'Spot Price': lambda x: format_currency(x, precision=2),
                'Spot 24h Change': lambda x: format_percentage(x, precision=2, show_plus=True),
                'Spot 7d Change': lambda x: format_percentage(x, precision=2, show_plus=True),
                'Spot 30d Change': lambda x: format_percentage(x, precision=2, show_plus=True),
                'Futures Price': lambda x: format_currency(x, precision=2),
                'Futures 24h Change': lambda x: format_percentage(x, precision=2, show_plus=True),
                'Futures 7d Change': lambda x: format_percentage(x, precision=2, show_plus=True),
                'Futures 30d Change': lambda x: format_percentage(x, precision=2, show_plus=True)
            }
            
            create_formatted_table(price_df, format_dict=format_dict)
            
            # Add note about 24h change calculation
            st.caption("Note: The 24h change represents the price difference between today's midnight (00:00 UTC) and yesterday's midnight (00:00 UTC) due to daily data granularity.")
        else:
            st.warning("No price comparison data available.")
        
        # 2. Spot price chart
        st.subheader("Spot Price Performance")
        
        spot_chart_data = prepare_spot_price_chart_data(data, timeframe=selected_timeframe)
        spot_chart = create_spot_chart(spot_chart_data, timeframe=selected_timeframe)
        
        if spot_chart:
            display_chart(spot_chart)
        else:
            st.warning("No spot price chart data available.")
        
        # 3. Funding rate chart
        st.subheader("Funding Rates")
        
        # Asset selector for funding rate chart
        funding_asset = st.selectbox("Select Asset", options=SUPPORTED_ASSETS[:4], index=0)
        
        funding_df = prepare_funding_rate_chart_data(data, asset=funding_asset, timeframe=selected_timeframe)
        funding_chart = create_funding_rate_chart(funding_df, asset=funding_asset, timeframe=selected_timeframe)
        
        if funding_chart:
            display_chart(funding_chart)
        else:
            st.warning(f"No funding rate data available for {funding_asset}.")
        
        # 4. Open interest chart
        st.subheader("Open Interest")
        
        # Asset selector for open interest chart
        oi_asset = st.selectbox("Select Asset for Open Interest", options=SUPPORTED_ASSETS[:4], index=0, key="oi_asset_selector")
        
        oi_df = prepare_open_interest_chart_data(data, asset=oi_asset, timeframe=selected_timeframe)
        oi_chart = create_open_interest_chart(oi_df, asset=oi_asset, timeframe=selected_timeframe)
        
        if oi_chart:
            display_chart(oi_chart)
        else:
            st.warning(f"No open interest data available for {oi_asset}.")
        
        # 5. Funding Rates Analysis Tables
        st.header("Funding Rates Analysis")
        
        # Create filter controls for funding rate tables
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # Get all available assets from the data
            all_assets = []
            if 'current_funding_rates' in data and data['current_funding_rates'] is not None:
                # Check if 'symbol' column exists
                if 'symbol' in data['current_funding_rates'].columns:
                    all_assets = sorted(data['current_funding_rates']['symbol'].unique().tolist())
                else:
                    # Log available columns for debugging
                    logger.warning(f"Expected 'symbol' column not found. Available columns: {data['current_funding_rates'].columns.tolist()}")
                    # Try to find the first column that might contain symbols
                    first_col = data['current_funding_rates'].columns[0]
                    all_assets = sorted(data['current_funding_rates'][first_col].unique().tolist())
            
            # Multi-select for assets (default to top 20)
            default_assets = all_assets[:20] if len(all_assets) > 20 else all_assets
            selected_assets = st.multiselect(
                "Select Assets",
                options=all_assets,
                default=default_assets,
                key="funding_assets"
            )
        
        with filter_col2:
            # Get all available exchanges from stablecoin margin data
            all_exchanges = set()
            if 'current_funding_rates' in data and data['current_funding_rates'] is not None:
                for idx, row in data['current_funding_rates'].iterrows():
                    if 'stablecoin_margin_list' in row and isinstance(row['stablecoin_margin_list'], (list, np.ndarray)):
                        for exchange_data in row['stablecoin_margin_list']:
                            if isinstance(exchange_data, dict) and 'exchange' in exchange_data:
                                all_exchanges.add(exchange_data['exchange'])
            
            all_exchanges = sorted(list(all_exchanges))
            selected_exchanges = st.multiselect(
                "Select Exchanges",
                options=all_exchanges,
                default=all_exchanges[:10] if len(all_exchanges) > 10 else all_exchanges,
                key="funding_exchanges"
            )
        
        # No longer need filter_col3, just set margin_type to stablecoin
        margin_type = 'stablecoin'  # Fixed to stablecoin margin only
        
        # Rate display type selector (only for current rates)
        rate_display_col1, rate_display_col2 = st.columns([1, 3])
        with rate_display_col1:
            rate_display = st.radio(
                "Rate Display",
                options=['annualized', 'current'],
                format_func=lambda x: 'Annualized (Default)' if x == 'annualized' else 'Current Rate',
                horizontal=True,
                key="funding_rate_display"
            )
        
        # Current Funding Rates Table
        st.subheader("Current Funding Rates (Live)")
        
        if rate_display == 'annualized':
            st.caption("Showing annualized rates calculated as: Current Rate × (24 / Funding Interval) × 365")
        else:
            st.caption("Showing raw current funding rates as provided by exchanges")
        
        current_df = prepare_current_funding_rate_table(
            data,
            selected_assets=selected_assets if selected_assets else None,
            selected_exchanges=selected_exchanges if selected_exchanges else None,
            rate_display=rate_display
        )
        
        if not current_df.empty:
            # Create format dictionary for the table
            format_dict = {}
            for col in current_df.columns:
                if col != 'Asset':
                    if rate_display == 'annualized':
                        # Format as percentage with 2 decimal places for annualized
                        format_dict[col] = lambda x: format_percentage(x, precision=2, show_plus=True) if pd.notnull(x) else 'N/A'
                    else:
                        # Format as percentage with 4 decimal places for current rates
                        format_dict[col] = lambda x: format_percentage(x, precision=4, show_plus=True) if pd.notnull(x) else 'N/A'
            
            create_formatted_table(
                current_df,
                format_dict=format_dict,
                emphasize_negatives=True,
                compact_display=True
            )
        else:
            st.info("No current funding rate data available for the selected filters.")
        
        # Accumulated Funding Rates Table
        st.subheader("Accumulated Funding Rates (365 Days)")
        st.caption("Total funding accumulated over the past year (365 days)")
        
        accumulated_df = prepare_accumulated_funding_rate_table(
            data,
            selected_assets=selected_assets if selected_assets else None,
            selected_exchanges=selected_exchanges if selected_exchanges else None
        )
        
        if not accumulated_df.empty:
            # Create format dictionary for the table
            format_dict = {}
            for col in accumulated_df.columns:
                if col != 'Asset':
                    # Format as percentage with 2 decimal places
                    format_dict[col] = lambda x: format_percentage(x, precision=2, show_plus=True) if pd.notnull(x) else 'N/A'
            
            # Apply special styling to the Average row
            def style_average_row(styler):
                # Find the index of the Average row
                avg_idx = accumulated_df[accumulated_df['Asset'] == 'Average'].index
                if not avg_idx.empty:
                    # Apply bold font to the Average row
                    styler = styler.set_properties(
                        subset=pd.IndexSlice[avg_idx, :],
                        **{'font-weight': 'bold', 'background-color': '#f0f0f0'}
                    )
                return styler
            
            # Create the table
            styled_table = create_formatted_table(
                accumulated_df,
                format_dict=format_dict,
                emphasize_negatives=True,
                compact_display=True
            )
            
            # Note about the data
            st.caption("Note: The 'Average' row shows the mean accumulated funding rate across all displayed assets for each exchange.")
        else:
            st.info("No accumulated funding rate data available for the selected filters.")

if __name__ == "__main__":
    main()