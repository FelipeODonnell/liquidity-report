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

logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

st.set_page_config(
    page_title=f"{APP_TITLE} - Report",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return data
    
    spot_data = {}
    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
        spot_asset_data = load_data_for_category('spot', 'spot_market', asset, latest_dir)
        if spot_asset_data:
            spot_data[asset] = spot_asset_data
    
    data['spot'] = spot_data
    
    futures_data = {}
    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
        futures_asset_data = load_data_for_category('futures', 'market', asset, latest_dir)
        if futures_asset_data:
            futures_data[asset] = futures_asset_data
    
    data['futures'] = futures_data
    
    funding_rate_data = {}
    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
        funding_asset_data = load_data_for_category('futures', 'funding_rate', asset, latest_dir)
        if funding_asset_data:
            funding_rate_data[asset] = funding_asset_data
    
    data['funding_rate'] = funding_rate_data
    
    open_interest_data = {}
    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
        oi_asset_data = load_data_for_category('futures', 'open_interest', asset, latest_dir)
        if oi_asset_data:
            open_interest_data[asset] = oi_asset_data
    
    data['open_interest'] = open_interest_data
    
    # Load futures volume data for each asset
    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
        volume_key = f'futures_volume_{asset}'
        volume_file = f'api_futures_taker_buy_sell_volume_history_{asset}'
        volume_data = load_specific_data_file('futures', volume_file, latest_dir, 'taker_buy_sell')
        if volume_data is not None and not volume_data.empty:
            data[volume_key] = volume_data
    
    # Load comprehensive volume data for the new section
    volume_data = {}
    
    # Ensure we have the full path including 'data' directory
    if not latest_dir.startswith('data'):
        data_path = os.path.join('data', latest_dir)
    else:
        data_path = latest_dir
    
    # Load aggregated taker buy/sell volume history
    aggregated_volume_file = os.path.join(data_path, 'futures', 'taker_buy_sell', 'api_futures_aggregated_taker_buy_sell_volume_history.parquet')
    if os.path.exists(aggregated_volume_file):
        data['aggregated_volume_history'] = pd.read_parquet(aggregated_volume_file)
        logger.info(f"Loaded aggregated volume history: {len(data['aggregated_volume_history'])} rows")
    
    # Load asset-specific volume history for all 4 assets
    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
        asset_volume_file = os.path.join(data_path, 'futures', 'taker_buy_sell', f'api_futures_taker_buy_sell_volume_history_{asset}.parquet')
        if os.path.exists(asset_volume_file):
            volume_data[asset] = pd.read_parquet(asset_volume_file)
            logger.info(f"Loaded {asset} volume history: {len(volume_data[asset])} rows")
        
        # Load exchange-specific volume data
        exchange_volume_file = os.path.join(data_path, 'futures', 'long_short_ratio', f'api_futures_taker_buy_sell_volume_exchange_list_{asset}.parquet')
        if os.path.exists(exchange_volume_file):
            volume_data[f'{asset}_exchange_volume'] = pd.read_parquet(exchange_volume_file)
            logger.info(f"Loaded {asset} exchange volume data: {len(volume_data[f'{asset}_exchange_volume'])} rows")
    
    data['volume'] = volume_data
    
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

def prepare_price_volume_combined_data(data, asset='BTC', timeframe='24h'):
    """
    Prepare combined spot price and futures volume data for chart display.
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    asset : str
        Specific asset like 'BTC', 'ETH', 'SOL', 'XRP'
    timeframe : str
        '24h', '7d', or '30d'
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns: datetime, price, total_volume, buy_volume, sell_volume
    """
    result_df = None
    
    try:
        # Get spot price data
        spot_df = None
        if 'spot' in data and asset in data['spot']:
            spot_price_key = f'api_spot_price_history_{asset}'
            if spot_price_key in data['spot'][asset]:
                spot_df = data['spot'][asset][spot_price_key].copy()
                spot_df = process_timestamps(spot_df)
                
                # Check if datetime column was created
                if 'datetime' not in spot_df.columns:
                    logger.warning(f"No datetime column after process_timestamps for {asset} spot data. Columns: {list(spot_df.columns)}")
                    # Try to find a timestamp column and convert it
                    if 'timestamp' in spot_df.columns:
                        spot_df['datetime'] = pd.to_datetime(spot_df['timestamp'], unit='ms')
                    elif 'time' in spot_df.columns:
                        spot_df['datetime'] = pd.to_datetime(spot_df['time'], unit='ms')
                    else:
                        logger.error(f"No timestamp column found for {asset} spot data")
                        return None
                
                if not spot_df.empty and 'close' in spot_df.columns and 'datetime' in spot_df.columns:
                    spot_df = spot_df.sort_values('datetime')
                    spot_df['price'] = spot_df['close']
        
        # Get futures volume data
        volume_df = None
        volume_key = f'futures_volume_{asset}'
        if volume_key in data and data[volume_key] is not None:
            volume_df = data[volume_key].copy()
            
            # Process timestamps to add datetime column
            volume_df = process_timestamps(volume_df, timestamp_col='time')
            
            # Check if datetime column was created
            if 'datetime' not in volume_df.columns:
                logger.warning(f"No datetime column after process_timestamps for {asset} volume data. Columns: {list(volume_df.columns)}")
                # Try to convert time column manually
                if 'time' in volume_df.columns:
                    volume_df['datetime'] = pd.to_datetime(volume_df['time'], unit='ms')
                else:
                    logger.error(f"No time column found for {asset} volume data")
                    return None
            
            # Calculate total volume
            if 'aggregated_buy_volume_usd' in volume_df.columns and 'aggregated_sell_volume_usd' in volume_df.columns:
                volume_df['total_volume'] = volume_df['aggregated_buy_volume_usd'] + volume_df['aggregated_sell_volume_usd']
                volume_df['buy_volume'] = volume_df['aggregated_buy_volume_usd']
                volume_df['sell_volume'] = volume_df['aggregated_sell_volume_usd']
        
        # Merge the dataframes if both exist
        if spot_df is not None and volume_df is not None:
            # Ensure both have datetime column
            if 'datetime' not in spot_df.columns or 'datetime' not in volume_df.columns:
                logger.error(f"Missing datetime column. Spot has: {list(spot_df.columns)}, Volume has: {list(volume_df.columns)}")
                return None
                
            # Sort both by datetime
            spot_df = spot_df.sort_values('datetime')
            volume_df = volume_df.sort_values('datetime')
            
            # Merge on nearest datetime (since they might have different granularities)
            result_df = pd.merge_asof(
                spot_df[['datetime', 'price']].dropna(),
                volume_df[['datetime', 'total_volume', 'buy_volume', 'sell_volume']].dropna(),
                on='datetime',
                direction='nearest'
            )
            
            # Filter based on timeframe
            if not result_df.empty:
                now = result_df['datetime'].max()
                
                if timeframe == '24h' or timeframe == '1d':
                    cutoff = now - timedelta(days=1)
                elif timeframe == '7d' or timeframe == '1w':
                    cutoff = now - timedelta(days=7)
                elif timeframe == '30d' or timeframe == '1m':
                    cutoff = now - timedelta(days=30)
                elif timeframe == '6m':
                    cutoff = now - timedelta(days=180)
                else:
                    cutoff = now - timedelta(days=1)  # Default to 1 day
                
                result_df = result_df[result_df['datetime'] >= cutoff]
                
    except Exception as e:
        logger.error(f"Error preparing price volume combined data for {asset}: {e}")
        st.error(f"Error preparing data for {asset}: {str(e)}")
    
    return result_df

def create_price_volume_combined_chart(df, asset='BTC', timeframe='24h'):
    """
    Create a dual-axis chart with price and volume.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price and volume data
    asset : str
        Asset name for labeling
    timeframe : str
        Display timeframe
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object with dual y-axis
    """
    if df is None or df.empty:
        return None
    
    from plotly.subplots import make_subplots
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Asset colors
    asset_colors = {
        'BTC': '#FF9800',
        'ETH': '#3F51B5',
        'SOL': '#9C27B0',
        'XRP': '#00BCD4'
    }
    
    color = asset_colors.get(asset, '#2196F3')
    
    # Add price line trace (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['price'],
            name=f'{asset} Price',
            line=dict(color=color, width=2),
            hovertemplate='Price: $%{y:,.2f}<br>Time: %{x}<extra></extra>'
        ),
        secondary_y=False,
    )
    
    # Add volume line trace with fill (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df['datetime'],
            y=df['total_volume'],
            name='Futures Volume',
            line=dict(color='#4CAF50', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(76, 175, 80, 0.1)',
            hovertemplate='Volume: $%{y:,.0f}<br>Time: %{x}<extra></extra>',
            yaxis='y2'
        ),
        secondary_y=True,
    )
    
    # Update layout
    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Price (USD)", secondary_y=False, tickformat='$,.0f')
    fig.update_yaxes(title_text="Volume (USD)", secondary_y=True, tickprefix="$", tickformat=",")
    
    # Update overall layout
    timeframe_labels = {
        '24h': 'Last 24 Hours',
        '1d': 'Last 24 Hours',
        '7d': 'Last 7 Days',
        '1w': 'Last Week',
        '30d': 'Last 30 Days',
        '1m': 'Last Month',
        '6m': 'Last 6 Months'
    }
    
    fig.update_layout(
        title=f"{asset} Spot Price & Perpetual Futures Volume - {timeframe_labels.get(timeframe, timeframe)}",
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Apply theme
    fig = apply_chart_theme(fig)
    
    return fig

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

def load_all_accumulated_funding_data(data_path):
    """
    Load accumulated funding rate data for all time periods.
    
    Parameters:
    -----------
    data_path : str
        Path to the data directory
    
    Returns:
    --------
    dict
        Dictionary with {period: DataFrame} for each time period
    """
    import json
    
    periods = ['1d', '7d', '30d', '365d']
    funding_data = {}
    
    for period in periods:
        file_path = os.path.join(
            data_path, 
            'futures', 
            'funding_rate', 
            f'api_futures_fundingRate_accumulated_exchange_list_{period}.parquet'
        )
        
        if os.path.exists(file_path):
            funding_data[period] = pd.read_parquet(file_path)
            logger.info(f"Loaded {period} accumulated funding rates: {len(funding_data[period])} rows")
        else:
            logger.warning(f"File not found: {file_path}")
            funding_data[period] = pd.DataFrame()
    
    return funding_data

def annualize_funding_rate(rate, period):
    """
    Convert accumulated funding rate to annualized rate.
    
    Parameters:
    -----------
    rate : float
        Accumulated funding rate (already as percentage, e.g., 6.89 for 6.89%)
    period : str
        Time period ('1d', '7d', '30d', '365d')
    
    Returns:
    --------
    float
        Annualized rate as percentage
    """
    multipliers = {
        '1d': 365,      # Daily to annual
        '7d': 52.14,    # Weekly to annual (365/7)
        '30d': 12.17,   # Monthly to annual (365/30)
        '365d': 1       # Already annual
    }
    
    return rate * multipliers.get(period, 1)

def prepare_comprehensive_funding_rate_table(funding_data_dict):
    """
    Create comprehensive table showing annualized funding rates across all time periods.
    Only includes top 20 crypto assets by market cap.
    
    Parameters:
    -----------
    funding_data_dict : dict
        Dictionary with DataFrames for each period
        
    Returns:
    --------
    pandas.DataFrame
        Formatted table with annualized rates
    """
    import json
    
    if not funding_data_dict:
        return pd.DataFrame()
    
    # Define top 20 crypto assets to filter for
    TOP_20_ASSETS = [
        'BTC', 'ETH', 'USDT', 'XRP', 'BNB', 'SOL', 'USDC', 'DOGE', 'ADA', 'TRX',
        'WBTC', 'SUI', 'LINK', 'AVAX', 'SHIB', 'LEO', 'BCH', 'XLM', 'HBAR', 'TON'
    ]
    
    all_rows = []
    
    # Process each asset across all time periods
    for period, df in funding_data_dict.items():
        if df.empty:
            continue
            
        for _, row in df.iterrows():
            symbol = row['symbol']
            
            # Only process top 20 crypto assets
            if symbol not in TOP_20_ASSETS:
                continue
            
            # Process stablecoin margin exchanges
            stablecoin_data = row.get('stablecoin_margin_list')
            if stablecoin_data is not None and len(stablecoin_data) > 0:
                try:
                    # Handle both string (JSON) and list formats
                    if isinstance(stablecoin_data, str):
                        exchange_list = json.loads(stablecoin_data)
                    else:
                        exchange_list = stablecoin_data
                    
                    # Process each exchange in the list
                    for exchange_info in exchange_list:
                        if isinstance(exchange_info, dict) and 'exchange' in exchange_info and 'funding_rate' in exchange_info:
                            try:
                                exchange = exchange_info['exchange']
                                funding_rate_value = exchange_info['funding_rate']
                                
                                # Skip if funding_rate is None or empty
                                if funding_rate_value is None or funding_rate_value == '':
                                    continue
                                    
                                rate = float(funding_rate_value)  # Use raw value as-is
                                
                                # Annualize the rate
                                annualized_rate = annualize_funding_rate(rate, period)
                                
                                all_rows.append({
                                    'Asset': symbol,
                                    'Exchange': exchange,
                                    'Period': period,
                                    'Accumulated Rate (%)': f"{rate:.2f}%",
                                    'Annualized Rate (%)': f"{annualized_rate:.2f}%",
                                    'Raw Rate': rate,
                                    'Raw Annualized': annualized_rate
                                })
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Error processing exchange data for {symbol}-{period}: {e}")
                                
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing stablecoin margin data for {symbol}-{period}: {e}")
            
            # Also process token margin exchanges if available
            token_data = row.get('token_margin_list')
            if token_data is not None and len(token_data) > 0:
                try:
                    # Handle both string (JSON) and list formats
                    if isinstance(token_data, str):
                        exchange_list = json.loads(token_data)
                    else:
                        exchange_list = token_data
                    
                    # Process each exchange in the list
                    for exchange_info in exchange_list:
                        if isinstance(exchange_info, dict) and 'exchange' in exchange_info and 'funding_rate' in exchange_info:
                            try:
                                exchange = f"{exchange_info['exchange']}_TOKEN"  # Distinguish from stablecoin
                                funding_rate_value = exchange_info['funding_rate']
                                
                                # Skip if funding_rate is None or empty
                                if funding_rate_value is None or funding_rate_value == '':
                                    continue
                                    
                                rate = float(funding_rate_value)  # Use raw value as-is
                                
                                # Annualize the rate
                                annualized_rate = annualize_funding_rate(rate, period)
                                
                                all_rows.append({
                                    'Asset': symbol,
                                    'Exchange': exchange,
                                    'Period': period,
                                    'Accumulated Rate (%)': f"{rate:.2f}%",
                                    'Annualized Rate (%)': f"{annualized_rate:.2f}%",
                                    'Raw Rate': rate,
                                    'Raw Annualized': annualized_rate
                                })
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Error processing token exchange data for {symbol}-{period}: {e}")
                                
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing token margin data for {symbol}-{period}: {e}")
    
    if not all_rows:
        return pd.DataFrame()
    
    # Create DataFrame
    result_df = pd.DataFrame(all_rows)
    
    # Pivot to show periods as columns
    pivot_df = result_df.pivot_table(
        index=['Asset', 'Exchange'],
        columns='Period',
        values='Raw Annualized',
        aggfunc='first'
    ).reset_index()
    
    # Format columns
    period_columns = ['1d', '7d', '30d', '365d']
    for col in period_columns:
        if col in pivot_df.columns:
            pivot_df[f'{col} Annualized (%)'] = pivot_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Select and order final columns
    final_columns = ['Asset', 'Exchange']
    for period in period_columns:
        if f'{period} Annualized (%)' in pivot_df.columns:
            final_columns.append(f'{period} Annualized (%)')
    
    return pivot_df[final_columns]

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
        
        # 5. Spot Price & Perpetual Futures Volume
        st.subheader("Spot Price & Perpetual Futures Volume")
        
        # Asset and timeframe selectors
        col1, col2, col3 = st.columns([2, 3, 3])
        with col1:
            selected_asset = st.selectbox(
                "Select Asset", 
                options=SUPPORTED_ASSETS[:4],  # BTC, ETH, SOL, XRP
                index=0,
                key="price_volume_asset"
            )
        
        with col2:
            # Custom timeframe selector for this chart
            pv_time_ranges = {
                '1d': '1 Day',
                '1w': '1 Week',
                '1m': '1 Month',
                '6m': '6 Months'
            }
            pv_timeframe = st.selectbox(
                "Time Range",
                options=list(pv_time_ranges.keys()),
                format_func=lambda x: pv_time_ranges[x],
                index=1,  # Default to 1 week
                key="price_volume_timeframe"
            )
        
        # Prepare combined data
        combined_df = prepare_price_volume_combined_data(
            data, 
            asset=selected_asset, 
            timeframe=pv_timeframe
        )
        
        # Create and display chart
        if combined_df is not None and not combined_df.empty:
            price_volume_chart = create_price_volume_combined_chart(
                combined_df, 
                asset=selected_asset, 
                timeframe=pv_timeframe
            )
            if price_volume_chart:
                display_chart(price_volume_chart)
            else:
                st.warning(f"Unable to create chart for {selected_asset}")
        else:
            st.warning(f"No price or volume data available for {selected_asset}")
        
        # 6. Funding Rate Return Analysis
        st.subheader("📊 Annualized Funding Rate Return Analysis")
        st.write("Annualized funding rates across different time periods for top 20 crypto assets by market cap.")
        
        # Get data path for comprehensive analysis
        if not latest_dir.startswith('data'):
            comprehensive_data_path = os.path.join('data', latest_dir)
        else:
            comprehensive_data_path = latest_dir
        
        # Load comprehensive funding data
        comprehensive_funding_data = load_all_accumulated_funding_data(comprehensive_data_path)
        
        if any(not df.empty for df in comprehensive_funding_data.values()):
            # Display which periods have data
            available_periods = [period for period, df in comprehensive_funding_data.items() if not df.empty]
            st.info(f"📈 Available data periods: {', '.join(available_periods)}")
            
            # Create comprehensive table
            with st.spinner("Processing comprehensive funding data..."):
                comprehensive_table = prepare_comprehensive_funding_rate_table(comprehensive_funding_data)
            
            # Period and asset selectors
            comp_col1, comp_col2 = st.columns([2, 1])
            
            with comp_col1:
                display_mode = st.radio(
                    "Display Mode:",
                    ["Summary View", "Detailed View"],
                    horizontal=True,
                    help="Summary shows only BTC and ETH. Detailed shows all top 20 crypto assets.",
                    key="report_comp_display_mode"
                )
            
            with comp_col2:
                # Get unique assets from the comprehensive table for the filter
                available_assets = sorted(comprehensive_table['Asset'].unique().tolist()) if not comprehensive_table.empty else []
                filter_asset = st.selectbox(
                    "Filter by Asset:",
                    ["All"] + available_assets,
                    help="Choose a specific asset to filter the data",
                    key="report_comp_filter_asset"
                )
            
            if not comprehensive_table.empty:
                # Apply filtering based on display mode and asset filter
                if display_mode == "Summary View":
                    # Filter to only show BTC and ETH in summary view
                    summary_assets = ['BTC', 'ETH']
                    display_table = comprehensive_table[comprehensive_table['Asset'].isin(summary_assets)]
                else:
                    display_table = comprehensive_table
                
                # Apply asset filter if specific asset is selected
                if filter_asset != "All":
                    display_table = display_table[display_table['Asset'] == filter_asset]
                
                # Exclude exchanges ending with '_TOKEN'
                display_table = display_table[~display_table['Exchange'].str.endswith('_TOKEN')]
                
                # Format the table for display
                format_dict = {}
                for col in display_table.columns:
                    if 'Annualized (%)' in col:
                        format_dict[col] = lambda x: x  # Already formatted as percentage strings
                
                st.subheader(f"Funding Rate Analysis - {display_mode}")
                create_formatted_table(
                    display_table,
                    format_dict=format_dict
                )
                
                # Explanation section
                st.markdown("""
                **📋 Annualized Funding Rates:**
                - **Asset Filter**: Shows top 20 crypto assets by market cap (BTC, ETH, USDT, XRP, etc.)
                - All rates are displayed as annualized percentages 


                """)
            else:
                st.warning("No comprehensive funding rate data could be processed.")
        else:
            st.warning("No accumulated funding rate data found for any time period.")
        
        # 7. Futures Volume Analysis Section
        st.header("📊 Futures Volume Analysis")
        st.write("Perpetual futures trading volume across assets, exchanges, and time periods.")
        
        # Check if volume data is available
        if 'volume' in data and data['volume']:
            # Create tabs for different views
            volume_tab1, volume_tab2, volume_tab3, volume_tab4 = st.tabs([
                "📈 Asset Volume Overview", 
                "🏢 Exchange Volume Distribution", 
                "📊 Volume Trends & Momentum",
                "🔄 Buy/Sell Flow Analysis"
            ])
            
            with volume_tab1:
                st.subheader("Asset Volume Overview")
                
                # Prepare asset volume summary
                asset_volume_summary = []
                for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
                    if asset in data['volume']:
                        volume_df = data['volume'][asset].copy()
                        if not volume_df.empty and 'aggregated_buy_volume_usd' in volume_df.columns:
                            # Calculate metrics
                            volume_df['total_volume'] = volume_df['aggregated_buy_volume_usd'] + volume_df['aggregated_sell_volume_usd']
                            
                            # Get latest and average volumes
                            latest_volume = volume_df['total_volume'].iloc[-1] if len(volume_df) > 0 else 0
                            avg_volume_7d = volume_df['total_volume'].tail(7).mean() if len(volume_df) >= 7 else latest_volume
                            avg_volume_30d = volume_df['total_volume'].tail(30).mean() if len(volume_df) >= 30 else latest_volume
                            
                            # Calculate growth rates
                            volume_growth_7d = ((latest_volume / avg_volume_7d) - 1) * 100 if avg_volume_7d > 0 else 0
                            volume_growth_30d = ((latest_volume / avg_volume_30d) - 1) * 100 if avg_volume_30d > 0 else 0
                            
                            # Get buy/sell ratio
                            buy_ratio = volume_df['aggregated_buy_volume_usd'].iloc[-1] / volume_df['total_volume'].iloc[-1] if volume_df['total_volume'].iloc[-1] > 0 else 0.5
                            
                            # Check if we have exchange-specific data
                            exchange_count = 0
                            if f'{asset}_exchange_volume' in data['volume']:
                                exchange_df = data['volume'][f'{asset}_exchange_volume']
                                if not exchange_df.empty and 'exchange_list' in exchange_df.columns:
                                    # Count unique exchanges
                                    exchanges = set()
                                    for idx, row in exchange_df.iterrows():
                                        # Handle both dict and list formats for exchange_list
                                        exchange_list = row['exchange_list']
                                        exchanges_to_process = []
                                        
                                        if isinstance(exchange_list, dict):
                                            # Single exchange per row format
                                            exchanges_to_process = [exchange_list]
                                        elif isinstance(exchange_list, list):
                                            # Multiple exchanges per row format
                                            exchanges_to_process = exchange_list
                                        
                                        for ex in exchanges_to_process:
                                            if isinstance(ex, dict) and 'exchange' in ex:
                                                exchanges.add(ex['exchange'])
                                    exchange_count = len(exchanges)
                            
                            asset_volume_summary.append({
                                'Asset': asset,
                                'Latest 24h Volume': latest_volume,
                                'Avg Volume (7d)': avg_volume_7d,
                                'Avg Volume (30d)': avg_volume_30d,
                                'Volume Change (7d)': volume_growth_7d,
                                'Volume Change (30d)': volume_growth_30d,
                                'Buy Ratio': buy_ratio,
                                'Exchange Count': exchange_count
                            })
                
                if asset_volume_summary:
                    volume_summary_df = pd.DataFrame(asset_volume_summary)
                    
                    # Display summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        total_volume_24h = volume_summary_df['Latest 24h Volume'].sum()
                        display_metric_card(
                            "Total 24h Volume",
                            format_currency(total_volume_24h, abbreviate=True),
                            "All Top 4 Assets"
                        )
                    
                    with col2:
                        avg_buy_ratio = volume_summary_df['Buy Ratio'].mean()
                        display_metric_card(
                            "Average Buy Ratio",
                            f"{avg_buy_ratio:.1%}",
                            "Bullish" if avg_buy_ratio > 0.52 else "Bearish" if avg_buy_ratio < 0.48 else "Neutral"
                        )
                    
                    with col3:
                        top_volume_asset = volume_summary_df.loc[volume_summary_df['Latest 24h Volume'].idxmax(), 'Asset']
                        top_volume = volume_summary_df['Latest 24h Volume'].max()
                        display_metric_card(
                            "Highest Volume Asset",
                            top_volume_asset,
                            format_currency(top_volume, abbreviate=True)
                        )
                    
                    with col4:
                        avg_growth_7d = volume_summary_df['Volume Change (7d)'].mean()
                        display_metric_card(
                            "Avg Volume Growth (7d)",
                            f"{avg_growth_7d:+.1f}%",
                            "Increasing" if avg_growth_7d > 5 else "Stable" if avg_growth_7d > -5 else "Decreasing"
                        )
                    
                    # Display detailed table
                    st.subheader("Detailed Volume Metrics by Asset")
                    
                    volume_format_dict = {
                        'Latest 24h Volume': lambda x: format_currency(x, abbreviate=True),
                        'Avg Volume (7d)': lambda x: format_currency(x, abbreviate=True),
                        'Avg Volume (30d)': lambda x: format_currency(x, abbreviate=True),
                        'Volume Change (7d)': lambda x: f"{x:+.1f}%",
                        'Volume Change (30d)': lambda x: f"{x:+.1f}%",
                        'Buy Ratio': lambda x: f"{x:.1%}",
                        'Exchange Count': lambda x: f"{int(x)}" if x > 0 else "N/A"
                    }
                    
                    create_formatted_table(
                        volume_summary_df,
                        format_dict=volume_format_dict,
                        emphasize_negatives=True,
                        compact_display=True
                    )
                    
                    # Volume comparison chart
                    st.subheader("24h Volume Comparison")
                    
                    fig_volume_comp = go.Figure()
                    
                    # Sort by volume for better visualization
                    volume_summary_df_sorted = volume_summary_df.sort_values('Latest 24h Volume', ascending=True)
                    
                    fig_volume_comp.add_trace(go.Bar(
                        x=volume_summary_df_sorted['Latest 24h Volume'],
                        y=volume_summary_df_sorted['Asset'],
                        orientation='h',
                        text=[format_currency(v, abbreviate=True) for v in volume_summary_df_sorted['Latest 24h Volume']],
                        textposition='outside',
                        marker_color='rgba(99, 110, 250, 0.8)'
                    ))
                    
                    fig_volume_comp.update_layout(
                        title="24-Hour Futures Volume by Asset",
                        xaxis_title="Volume (USD)",
                        yaxis_title=None,
                        height=400,
                        showlegend=False
                    )
                    
                    display_chart(apply_chart_theme(fig_volume_comp))
                    
                    st.markdown("""
                    **Volume Analysis Insights:**
                    - Higher volumes indicate more liquid markets with tighter spreads
                    - Buy ratio >52% suggests bullish sentiment, <48% suggests bearish sentiment
                    - Volume growth shows increasing market interest and trading activity
                    - Exchange count indicates market breadth and accessibility
                    """)
            
            with volume_tab2:
                st.subheader("Exchange Volume Distribution")
                
                # Aggregate exchange volumes across all assets
                exchange_volumes = {}
                
                for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
                    if f'{asset}_exchange_volume' in data['volume']:
                        exchange_df = data['volume'][f'{asset}_exchange_volume']
                        if not exchange_df.empty and 'exchange_list' in exchange_df.columns:
                            for idx, row in exchange_df.iterrows():
                                if isinstance(row['exchange_list'], list):
                                    for ex in row['exchange_list']:
                                        if isinstance(ex, dict):
                                            exchange_name = ex.get('exchange', '')
                                            buy_vol = ex.get('buy_vol_usd', 0)
                                            sell_vol = ex.get('sell_vol_usd', 0)
                                            total_vol = buy_vol + sell_vol
                                            
                                            if exchange_name and total_vol > 0:
                                                if exchange_name not in exchange_volumes:
                                                    exchange_volumes[exchange_name] = {
                                                        'total_volume': 0,
                                                        'buy_volume': 0,
                                                        'sell_volume': 0,
                                                        'assets': set()
                                                    }
                                                
                                                exchange_volumes[exchange_name]['total_volume'] += total_vol
                                                exchange_volumes[exchange_name]['buy_volume'] += buy_vol
                                                exchange_volumes[exchange_name]['sell_volume'] += sell_vol
                                                exchange_volumes[exchange_name]['assets'].add(asset)
                
                if exchange_volumes:
                    # Convert to DataFrame
                    exchange_data = []
                    for exchange, metrics in exchange_volumes.items():
                        exchange_data.append({
                            'Exchange': exchange,
                            'Total Volume': metrics['total_volume'],
                            'Buy Volume': metrics['buy_volume'],
                            'Sell Volume': metrics['sell_volume'],
                            'Buy Ratio': metrics['buy_volume'] / metrics['total_volume'] if metrics['total_volume'] > 0 else 0.5,
                            'Asset Count': len(metrics['assets']),
                            'Market Share': 0  # Will calculate after
                        })
                    
                    exchange_df = pd.DataFrame(exchange_data)
                    total_market_volume = exchange_df['Total Volume'].sum()
                    exchange_df['Market Share'] = exchange_df['Total Volume'] / total_market_volume if total_market_volume > 0 else 0
                    
                    # Sort by volume
                    exchange_df = exchange_df.sort_values('Total Volume', ascending=False)
                    
                    # Display top exchanges metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        top_exchange = exchange_df.iloc[0]['Exchange'] if not exchange_df.empty else "N/A"
                        top_share = exchange_df.iloc[0]['Market Share'] if not exchange_df.empty else 0
                        display_metric_card(
                            "Top Exchange",
                            top_exchange,
                            f"{top_share:.1%} market share"
                        )
                    
                    with col2:
                        top_5_share = exchange_df.head(5)['Market Share'].sum() if len(exchange_df) >= 5 else exchange_df['Market Share'].sum()
                        display_metric_card(
                            "Top 5 Exchange Share",
                            f"{top_5_share:.1%}",
                            "Market concentration"
                        )
                    
                    with col3:
                        active_exchanges = len(exchange_df)
                        display_metric_card(
                            "Active Exchanges",
                            str(active_exchanges),
                            "Trading top 4 assets"
                        )
                    
                    # Market share pie chart
                    st.subheader("Exchange Market Share Distribution")
                    
                    # Take top 10 exchanges for pie chart
                    top_exchanges = exchange_df.head(10).copy()
                    if len(exchange_df) > 10:
                        others_share = exchange_df.iloc[10:]['Market Share'].sum()
                        others_volume = exchange_df.iloc[10:]['Total Volume'].sum()
                        top_exchanges = pd.concat([
                            top_exchanges,
                            pd.DataFrame([{
                                'Exchange': 'Others',
                                'Total Volume': others_volume,
                                'Market Share': others_share
                            }])
                        ])
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=top_exchanges['Exchange'],
                        values=top_exchanges['Market Share'],
                        hole=0.4,
                        textinfo='label+percent',
                        textposition='auto'
                    )])
                    
                    fig_pie.update_layout(
                        title="Exchange Market Share (24h Volume)",
                        height=500
                    )
                    
                    display_chart(apply_chart_theme(fig_pie))
                    
                    # Detailed exchange table
                    st.subheader("Exchange Volume Details")
                    
                    display_exchange_df = pd.DataFrame({
                        'Exchange': exchange_df['Exchange'],
                        'Total Volume (24h)': exchange_df['Total Volume'],
                        'Market Share': exchange_df['Market Share'],
                        'Buy Ratio': exchange_df['Buy Ratio'],
                        'Asset Coverage': exchange_df['Asset Count']
                    })
                    
                    exchange_format_dict = {
                        'Total Volume (24h)': lambda x: format_currency(x, abbreviate=True),
                        'Market Share': lambda x: f"{x:.2%}",
                        'Buy Ratio': lambda x: f"{x:.1%}",
                        'Asset Coverage': lambda x: f"{int(x)}/4"
                    }
                    
                    create_formatted_table(
                        display_exchange_df.head(15),  # Show top 15
                        format_dict=exchange_format_dict,
                        emphasize_negatives=False,
                        compact_display=True
                    )
                    
                    st.markdown("""
                    **Exchange Distribution Insights:**
                    - Market concentration shows dominance of top exchanges
                    - Higher asset coverage indicates better liquidity across different markets
                    - Buy ratio variations across exchanges can indicate arbitrage opportunities
                    - Consider exchange reliability and regulatory status when selecting venues
                    """)
            
            with volume_tab3:
                st.subheader("Volume Trends & Momentum Analysis")
                
                # Time period selector
                trend_period = st.selectbox(
                    "Select Analysis Period",
                    options=['7d', '30d', '90d', 'All'],
                    index=1,
                    key='report_volume_trend_period'
                )
                
                # Asset selector for detailed analysis
                selected_trend_assets = st.multiselect(
                    "Select Assets for Trend Analysis",
                    options=SUPPORTED_ASSETS[:4],  # BTC, ETH, SOL, XRP
                    default=['BTC', 'ETH'],
                    key='report_volume_trend_assets'
                )
                
                if selected_trend_assets:
                    # Create volume trend chart
                    fig_trend = go.Figure()
                    
                    for asset in selected_trend_assets:
                        if asset in data['volume']:
                            volume_df = data['volume'][asset].copy()
                            if 'time' in volume_df.columns:
                                volume_df['datetime'] = pd.to_datetime(volume_df['time'], unit='ms')
                                volume_df = volume_df.sort_values('datetime')
                                volume_df['total_volume'] = volume_df['aggregated_buy_volume_usd'] + volume_df['aggregated_sell_volume_usd']
                                
                                # Filter by period
                                if trend_period == '7d':
                                    volume_df = volume_df.tail(7)
                                elif trend_period == '30d':
                                    volume_df = volume_df.tail(30)
                                elif trend_period == '90d':
                                    volume_df = volume_df.tail(90)
                                
                                # Add trace
                                fig_trend.add_trace(go.Scatter(
                                    x=volume_df['datetime'],
                                    y=volume_df['total_volume'],
                                    name=asset,
                                    mode='lines+markers',
                                    line=dict(width=2)
                                ))
                    
                    fig_trend.update_layout(
                        title=f"Volume Trends - {trend_period} Period",
                        xaxis_title="Date",
                        yaxis_title="Daily Volume (USD)",
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    display_chart(apply_chart_theme(fig_trend))
                    
                    # Volume momentum indicators
                    st.subheader("Volume Momentum Indicators")
                    
                    momentum_data = []
                    for asset in selected_trend_assets:
                        if asset in data['volume']:
                            volume_df = data['volume'][asset].copy()
                            if not volume_df.empty and 'aggregated_buy_volume_usd' in volume_df.columns:
                                volume_df['total_volume'] = volume_df['aggregated_buy_volume_usd'] + volume_df['aggregated_sell_volume_usd']
                                
                                # Calculate various momentum indicators
                                if len(volume_df) >= 30:
                                    # Simple moving averages
                                    sma_7 = volume_df['total_volume'].tail(7).mean()
                                    sma_30 = volume_df['total_volume'].tail(30).mean()
                                    current_vol = volume_df['total_volume'].iloc[-1]
                                    
                                    # Momentum calculations
                                    momentum_7d = (current_vol / sma_7 - 1) * 100 if sma_7 > 0 else 0
                                    momentum_30d = (current_vol / sma_30 - 1) * 100 if sma_30 > 0 else 0
                                    
                                    # Volatility
                                    vol_std = volume_df['total_volume'].tail(30).std()
                                    vol_mean = volume_df['total_volume'].tail(30).mean()
                                    vol_cv = (vol_std / vol_mean * 100) if vol_mean > 0 else 0
                                    
                                    # Trend strength (linear regression slope) - using numpy instead of scipy
                                    x = np.arange(len(volume_df.tail(30)))
                                    y = volume_df['total_volume'].tail(30).values
                                    
                                    # Manual linear regression calculation
                                    x_mean = x.mean()
                                    y_mean = y.mean()
                                    numerator = ((x - x_mean) * (y - y_mean)).sum()
                                    denominator = ((x - x_mean) ** 2).sum()
                                    
                                    if denominator != 0:
                                        slope = numerator / denominator
                                        # Calculate R-squared
                                        y_pred = slope * (x - x_mean) + y_mean
                                        ss_res = ((y - y_pred) ** 2).sum()
                                        ss_tot = ((y - y_mean) ** 2).sum()
                                        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                                    else:
                                        slope = 0
                                        r_squared = 0
                                    
                                    momentum_data.append({
                                        'Asset': asset,
                                        'Current Volume': current_vol,
                                        '7D MA': sma_7,
                                        '30D MA': sma_30,
                                        '7D Momentum': momentum_7d,
                                        '30D Momentum': momentum_30d,
                                        'Volatility (CV)': vol_cv,
                                        'Trend Strength': r_squared
                                    })
                    
                    if momentum_data:
                        momentum_df = pd.DataFrame(momentum_data)
                        
                        momentum_format_dict = {
                            'Current Volume': lambda x: format_currency(x, abbreviate=True),
                            '7D MA': lambda x: format_currency(x, abbreviate=True),
                            '30D MA': lambda x: format_currency(x, abbreviate=True),
                            '7D Momentum': lambda x: f"{x:+.1f}%",
                            '30D Momentum': lambda x: f"{x:+.1f}%",
                            'Volatility (CV)': lambda x: f"{x:.1f}%",
                            'Trend Strength': lambda x: f"{x:.2f}"
                        }
                        
                        create_formatted_table(
                            momentum_df,
                            format_dict=momentum_format_dict,
                            emphasize_negatives=True,
                            compact_display=True
                        )
                        
                        st.markdown("""
                        **Momentum Indicators Explained:**
                        - **7D/30D Momentum**: Current volume vs moving average (>0% = above average)
                        - **Volatility (CV)**: Coefficient of variation - higher values indicate more volatile volume
                        - **Trend Strength**: R² value from linear regression (0-1, higher = stronger trend)
                        - Positive momentum with low volatility suggests sustainable volume growth
                        """)
            
            with volume_tab4:
                st.subheader("Buy/Sell Flow Analysis")
                
                # Create buy/sell imbalance chart for all assets
                if 'aggregated_volume_history' in data and data['aggregated_volume_history'] is not None:
                    agg_volume_df = data['aggregated_volume_history'].copy()
                    agg_volume_df['datetime'] = pd.to_datetime(agg_volume_df['time'], unit='ms')
                    agg_volume_df = agg_volume_df.sort_values('datetime')
                    
                    # Calculate metrics
                    agg_volume_df['total_volume'] = agg_volume_df['aggregated_buy_volume_usd'] + agg_volume_df['aggregated_sell_volume_usd']
                    agg_volume_df['buy_ratio'] = agg_volume_df['aggregated_buy_volume_usd'] / agg_volume_df['total_volume']
                    agg_volume_df['net_flow'] = agg_volume_df['aggregated_buy_volume_usd'] - agg_volume_df['aggregated_sell_volume_usd']
                    agg_volume_df['flow_imbalance'] = agg_volume_df['net_flow'] / agg_volume_df['total_volume']
                    
                    # Time period selector for flow analysis
                    flow_period = st.selectbox(
                        "Select Flow Analysis Period",
                        options=['24h', '7d', '30d'],
                        index=1,
                        key='report_flow_period'
                    )
                    
                    # Filter data based on period
                    if flow_period == '24h':
                        flow_df = agg_volume_df.tail(6)  # 6 * 4h = 24h
                    elif flow_period == '7d':
                        flow_df = agg_volume_df.tail(42)  # 42 * 4h = 7 days
                    else:  # 30d
                        flow_df = agg_volume_df.tail(180)  # 180 * 4h = 30 days
                    
                    # Flow metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        cumulative_net_flow = flow_df['net_flow'].sum()
                        display_metric_card(
                            f"Net Flow ({flow_period})",
                            format_currency(cumulative_net_flow, abbreviate=True),
                            "Buy pressure" if cumulative_net_flow > 0 else "Sell pressure"
                        )
                    
                    with col2:
                        avg_buy_ratio = flow_df['buy_ratio'].mean()
                        display_metric_card(
                            f"Avg Buy Ratio ({flow_period})",
                            f"{avg_buy_ratio:.1%}",
                            "Bullish" if avg_buy_ratio > 0.52 else "Bearish" if avg_buy_ratio < 0.48 else "Neutral"
                        )
                    
                    with col3:
                        flow_volatility = flow_df['flow_imbalance'].std()
                        display_metric_card(
                            "Flow Volatility",
                            f"{flow_volatility:.3f}",
                            "High" if flow_volatility > 0.1 else "Moderate" if flow_volatility > 0.05 else "Low"
                        )
                    
                    with col4:
                        # Trend direction based on linear regression - using numpy
                        x = np.arange(len(flow_df))
                        y = flow_df['buy_ratio'].values
                        
                        # Simple slope calculation using numpy
                        if len(x) > 1:
                            # Calculate slope using least squares
                            x_mean = x.mean()
                            y_mean = y.mean()
                            numerator = ((x - x_mean) * (y - y_mean)).sum()
                            denominator = ((x - x_mean) ** 2).sum()
                            slope = numerator / denominator if denominator != 0 else 0
                        else:
                            slope = 0
                        
                        trend_direction = "Increasing" if slope > 0.0001 else "Decreasing" if slope < -0.0001 else "Stable"
                        display_metric_card(
                            "Buy Pressure Trend",
                            trend_direction,
                            f"Slope: {slope:.5f}"
                        )
                    
                    # Create flow imbalance chart
                    fig_flow = go.Figure()
                    
                    # Add buy volume (positive)
                    fig_flow.add_trace(go.Bar(
                        x=flow_df['datetime'],
                        y=flow_df['aggregated_buy_volume_usd'],
                        name='Buy Volume',
                        marker_color='rgba(0, 255, 0, 0.7)',
                        hovertemplate='Buy: $%{y:,.0f}<extra></extra>'
                    ))
                    
                    # Add sell volume (negative)
                    fig_flow.add_trace(go.Bar(
                        x=flow_df['datetime'],
                        y=-flow_df['aggregated_sell_volume_usd'],
                        name='Sell Volume',
                        marker_color='rgba(255, 0, 0, 0.7)',
                        hovertemplate='Sell: $%{y:,.0f}<extra></extra>'
                    ))
                    
                    # Add net flow line
                    fig_flow.add_trace(go.Scatter(
                        x=flow_df['datetime'],
                        y=flow_df['net_flow'],
                        name='Net Flow',
                        line=dict(color='white', width=2),
                        yaxis='y2',
                        hovertemplate='Net: $%{y:,.0f}<extra></extra>'
                    ))
                    
                    fig_flow.update_layout(
                        title=f"Buy/Sell Volume Flow - {flow_period}",
                        xaxis_title="Time",
                        yaxis_title="Volume (USD)",
                        yaxis2=dict(
                            title="Net Flow (USD)",
                            overlaying='y',
                            side='right'
                        ),
                        barmode='relative',
                        hovermode='x unified',
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    display_chart(apply_chart_theme(fig_flow))
                    
                    # Buy/Sell ratio heatmap by asset
                    st.subheader("Buy/Sell Ratio by Asset")
                    
                    # Prepare heatmap data
                    heatmap_data = []
                    for asset in SUPPORTED_ASSETS[:4]:  # BTC, ETH, SOL, XRP
                        if asset in data['volume']:
                            asset_volume_df = data['volume'][asset].copy()
                            if not asset_volume_df.empty and 'aggregated_buy_volume_usd' in asset_volume_df.columns:
                                asset_volume_df['buy_ratio'] = asset_volume_df['aggregated_buy_volume_usd'] / (
                                    asset_volume_df['aggregated_buy_volume_usd'] + asset_volume_df['aggregated_sell_volume_usd']
                                )
                                
                                # Get ratios for different periods
                                current_ratio = asset_volume_df['buy_ratio'].iloc[-1] if len(asset_volume_df) > 0 else 0.5
                                avg_7d = asset_volume_df['buy_ratio'].tail(7).mean() if len(asset_volume_df) >= 7 else current_ratio
                                avg_30d = asset_volume_df['buy_ratio'].tail(30).mean() if len(asset_volume_df) >= 30 else current_ratio
                                
                                heatmap_data.append({
                                    'Asset': asset,
                                    'Current': current_ratio,
                                    '7D Average': avg_7d,
                                    '30D Average': avg_30d
                                })
                    
                    if heatmap_data:
                        heatmap_df = pd.DataFrame(heatmap_data)
                        heatmap_df = heatmap_df.set_index('Asset')
                        
                        # Create heatmap
                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=heatmap_df.values,
                            x=heatmap_df.columns,
                            y=heatmap_df.index,
                            colorscale='RdYlGn',
                            zmid=0.5,
                            text=[[f"{val:.1%}" for val in row] for row in heatmap_df.values],
                            texttemplate="%{text}",
                            textfont={"size": 12},
                            colorbar=dict(title="Buy Ratio")
                        ))
                        
                        fig_heatmap.update_layout(
                            title="Buy/Sell Ratio Heatmap",
                            xaxis_title="Time Period",
                            yaxis_title="Asset",
                            height=400
                        )
                        
                        display_chart(apply_chart_theme(fig_heatmap))
                        
                        st.markdown("""
                        **Buy/Sell Flow Analysis Insights:**
                        - **Net Flow**: Positive = more buying pressure, Negative = more selling pressure
                        - **Buy Ratio >52%**: Strong bullish sentiment, likely to see positive funding rates
                        - **Flow Volatility**: High volatility suggests uncertain market direction
                        - **Trend Analysis**: Consistent buy/sell pressure indicates market conviction
                        - **Cross-Asset Comparison**: Identify which assets have strongest directional bias
                        """)
        else:
            st.warning("No futures volume data available. Please check data sources.")

if __name__ == "__main__":
    main()