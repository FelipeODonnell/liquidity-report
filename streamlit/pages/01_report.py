"""
Report page for the Izun Crypto Liquidity Report.

This page serves as the main dashboard/overview page focused on liquidity metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import json
import traceback
import logging

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
    create_stacked_bar_chart,
    create_pie_chart,
    apply_chart_theme,
    display_chart
)
from components.tables import create_formatted_table, create_etf_table
from utils.data_loader import (
    get_latest_data_directory, 
    load_data_for_category, 
    process_timestamps,
    get_data_last_updated,
    calculate_metrics
)
from utils.formatters import (
    format_currency, 
    format_percentage, 
    format_volume,
    format_timestamp,
    humanize_time_diff
)
from utils.config import APP_TITLE, APP_ICON, ASSET_COLORS, DEFAULT_ASSET, EXCHANGE_COLORS

# Set page config with title and icon
st.set_page_config(
    page_title=f"{APP_TITLE} - Liquidity Report",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set the current page for sidebar navigation
st.session_state.current_page = 'report'

def load_report_data():
    """
    Load data for the report dashboard.
    
    Returns:
    --------
    dict
        Dictionary containing all data needed for the dashboard
    """
    data = {}
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return data
    
    # Get selected asset from session state or use default
    asset = st.session_state.get('selected_asset', DEFAULT_ASSET)
    
    # Load trading volume data
    data['taker_volume'] = load_data_for_category('futures', 'taker_buy_sell', asset)
    
    # Load open interest data
    data['open_interest'] = load_data_for_category('futures', 'open_interest', asset)
    
    # Load order book data for bid-ask spread and depth
    data['order_book'] = load_data_for_category('futures', 'order_book', asset)
    
    # Load funding rate data
    data['funding_rate'] = load_data_for_category('futures', 'funding_rate', asset)
    
    # Load market data for exchange comparisons
    data['market'] = load_data_for_category('futures', 'market', asset)
    
    # Load price data
    data['price'] = load_data_for_category('futures', 'market')
    
    return data

def normalize_funding_rate_data(funding_df, asset):
    """
    Process nested funding rate data structure into a flat DataFrame.
    
    Parameters:
    -----------
    funding_df : pandas.DataFrame
        Raw funding rate data
    asset : str
        Asset to filter for
        
    Returns:
    --------
    pandas.DataFrame
        Normalized funding rate data
    """
    if funding_df.empty:
        return pd.DataFrame()
    
    # Check if it's an OHLC funding rate structure
    if 'time' in funding_df.columns and any(col in funding_df.columns for col in ['open', 'close']):
        try:
            # Process timestamps and convert values to numeric
            processed_df = process_timestamps(funding_df)
            
            # Convert OHLC columns to numeric
            for col in ['open', 'high', 'low', 'close']:
                if col in processed_df.columns:
                    processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
            
            # For consistent column naming, create a funding_rate column
            if 'close' in processed_df.columns and 'funding_rate' not in processed_df.columns:
                processed_df['funding_rate'] = processed_df['close']
                
            logger.info(f"Successfully processed OHLC funding rate data with shape {processed_df.shape}")
            return processed_df
        except Exception as e:
            logger.error(f"Error processing OHLC funding rate data: {e}")
            return pd.DataFrame()
    
    # Check if it's already a flat structure
    if 'funding_rate' in funding_df.columns and 'symbol' in funding_df.columns and 'exchange_name' in funding_df.columns:
        try:
            # Convert funding_rate to numeric
            funding_df['funding_rate'] = pd.to_numeric(funding_df['funding_rate'], errors='coerce')
            
            # Filter for the specific asset
            filtered_df = funding_df[funding_df['symbol'].str.contains(asset, case=False, na=False)]
            
            if not filtered_df.empty:
                logger.info(f"Successfully filtered flat funding rate data for {asset} with shape {filtered_df.shape}")
                return filtered_df
            else:
                logger.warning(f"No funding rate data found for {asset} after filtering")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing flat funding rate data: {e}")
            return pd.DataFrame()
    
    # Check if it has nested lists structure
    if 'symbol' not in funding_df.columns or ('stablecoin_margin_list' not in funding_df.columns and 'token_margin_list' not in funding_df.columns):
        logger.warning("Funding rate data does not have expected nested structure")
        return pd.DataFrame()
    
    # Process nested structure
    normalized_data = []
    
    try:
        for _, row in funding_df.iterrows():
            symbol = row['symbol']
            if asset.lower() not in symbol.lower():
                continue
                
            # Process stablecoin margin list - handle different formats
            if 'stablecoin_margin_list' in row:
                margin_list = row['stablecoin_margin_list']
                
                # Convert to Python list if it's a numpy array or other array-like
                if hasattr(margin_list, '__array__'):
                    try:
                        margin_list = margin_list.tolist()
                    except:
                        logger.warning(f"Could not convert stablecoin_margin_list to list for {symbol}")
                        margin_list = []
                        
                # Process list items
                if isinstance(margin_list, list):
                    for item in margin_list:
                        if isinstance(item, dict) and 'exchange' in item and 'funding_rate' in item:
                            normalized_data.append({
                                'symbol': symbol,
                                'exchange_name': item['exchange'],
                                'funding_rate': pd.to_numeric(item['funding_rate'], errors='coerce'),
                                'margin_type': 'stablecoin'
                            })
                elif isinstance(margin_list, dict) and 'exchange' in margin_list and 'funding_rate' in margin_list:
                    # Handle case where it's a single dict instead of a list
                    normalized_data.append({
                        'symbol': symbol,
                        'exchange_name': margin_list['exchange'],
                        'funding_rate': pd.to_numeric(margin_list['funding_rate'], errors='coerce'),
                        'margin_type': 'stablecoin'
                    })
            
            # Process token margin list - handle different formats
            if 'token_margin_list' in row:
                margin_list = row['token_margin_list']
                
                # Convert to Python list if it's a numpy array or other array-like
                if hasattr(margin_list, '__array__'):
                    try:
                        margin_list = margin_list.tolist()
                    except:
                        logger.warning(f"Could not convert token_margin_list to list for {symbol}")
                        margin_list = []
                        
                # Process list items
                if isinstance(margin_list, list):
                    for item in margin_list:
                        if isinstance(item, dict) and 'exchange' in item and 'funding_rate' in item:
                            normalized_data.append({
                                'symbol': symbol,
                                'exchange_name': item['exchange'],
                                'funding_rate': pd.to_numeric(item['funding_rate'], errors='coerce'),
                                'margin_type': 'token'
                            })
                elif isinstance(margin_list, dict) and 'exchange' in margin_list and 'funding_rate' in margin_list:
                    # Handle case where it's a single dict instead of a list
                    normalized_data.append({
                        'symbol': symbol,
                        'exchange_name': margin_list['exchange'],
                        'funding_rate': pd.to_numeric(margin_list['funding_rate'], errors='coerce'),
                        'margin_type': 'token'
                    })
        
        if normalized_data:
            result_df = pd.DataFrame(normalized_data)
            logger.info(f"Successfully normalized nested funding rate data with shape {result_df.shape}")
            return result_df
        else:
            logger.warning(f"No funding rate data found for {asset} after processing nested structure")
            return pd.DataFrame()
            
    except Exception as e:
        st.warning(f"Error processing funding rate data: {e}")
        logger.error(f"Error processing funding rate data: {e}")
        return pd.DataFrame()

def calculate_spreads_and_depth(orderbook_df, asset, price_levels=[0.01, 0.02, 0.05]):
    """
    Calculate bid-ask spreads and depth at various price levels.
    
    Parameters:
    -----------
    orderbook_df : pandas.DataFrame
        Orderbook data
    asset : str
        Asset to filter for
    price_levels : list
        List of price level percentages for depth calculation
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with calculated spreads and depth
    """
    if orderbook_df.empty:
        return pd.DataFrame()
    
    # Check if we have the required columns
    required_cols = ['bids_usd', 'bids_quantity', 'asks_usd', 'asks_quantity']
    for col in required_cols:
        if col not in orderbook_df.columns:
            logger.warning(f"Missing required column {col} in orderbook data")
            return pd.DataFrame()
    
    try:
        # Process timestamps if needed
        processed_df = orderbook_df.copy()
        if 'datetime' not in processed_df.columns:
            processed_df = process_timestamps(processed_df)
        
        # Calculate basic spread metrics
        processed_df['mid_price'] = (processed_df['asks_usd'] + processed_df['bids_usd']) / 2
        processed_df['spread_usd'] = processed_df['asks_usd'] - processed_df['bids_usd']
        processed_df['spread_pct'] = (processed_df['spread_usd'] / processed_df['mid_price']) * 100
        
        # Calculate basic depth metrics
        processed_df['bid_depth'] = processed_df['bids_quantity'] * processed_df['bids_usd']
        processed_df['ask_depth'] = processed_df['asks_quantity'] * processed_df['asks_usd']
        processed_df['total_depth'] = processed_df['bid_depth'] + processed_df['ask_depth']
        
        # Calculate depth and imbalance at price levels
        for level in price_levels:
            level_pct = int(level * 100)  # Convert to percentage for column naming
            
            # Calculate depth at this price level
            processed_df[f'bid_depth_{level_pct}pct'] = processed_df['bid_depth'] * level
            processed_df[f'ask_depth_{level_pct}pct'] = processed_df['ask_depth'] * level
            processed_df[f'total_depth_{level_pct}pct'] = processed_df[f'bid_depth_{level_pct}pct'] + processed_df[f'ask_depth_{level_pct}pct']
            
            # Calculate imbalance at this price level
            with np.errstate(divide='ignore', invalid='ignore'):
                processed_df[f'depth_imbalance_{level_pct}pct'] = np.where(
                    processed_df[f'ask_depth_{level_pct}pct'] != 0,
                    processed_df[f'bid_depth_{level_pct}pct'] / processed_df[f'ask_depth_{level_pct}pct'],
                    np.nan
                )
        
        return processed_df
    except Exception as e:
        st.warning(f"Error calculating spreads and depth: {e}")
        logger.error(f"Error calculating spreads and depth: {e}\n{traceback.format_exc()}")
        return pd.DataFrame()

def create_fallback_chart(title, message="No data available", height=400, suggestions=None):
    """
    Create an empty chart with a message when data is unavailable.
    
    Parameters:
    -----------
    title : str
        Chart title
    message : str
        Main message to display
    height : int
        Chart height in pixels
    suggestions : list
        Optional list of suggestion strings to display
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Fallback chart with message
    """
    fig = go.Figure()
    
    # Add main message
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color="#555555")
    )
    
    # Add suggestions if provided
    if suggestions and isinstance(suggestions, list):
        suggestions_text = "<br>".join([f"• {s}" for s in suggestions])
        fig.add_annotation(
            text=f"<b>Suggestions:</b><br>{suggestions_text}",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.3,  # Position below main message
            showarrow=False,
            font=dict(size=12, color="#777777"),
            align="left"
        )
    
    # Add icon to make the chart more visually informative
    fig.add_annotation(
        text="📊",  # Chart emoji
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.7,  # Position above main message
        showarrow=False,
        font=dict(size=36, color="#AAAAAA")
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=height,
        plot_bgcolor="#f9f9f9",
        paper_bgcolor="#f9f9f9",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return apply_chart_theme(fig)

def extract_asset_volume_data(data, asset_list):
    """
    Extract volume data for multiple assets.
    
    Parameters:
    -----------
    data : dict
        Data dictionary
    asset_list : list
        List of assets to extract data for
        
    Returns:
    --------
    dict
        Dictionary with volume data by asset
    """
    assets_volume = {}
    
    for asset_name in asset_list:
        for key in data.get('taker_volume', {}):
            if 'history' in key.lower() and asset_name.lower() in key.lower():
                asset_vol_df = data['taker_volume'][key]
                if not asset_vol_df.empty:
                    try:
                        asset_vol_df = process_timestamps(asset_vol_df)
                        if not asset_vol_df.empty and 'taker_buy_volume_usd' in asset_vol_df.columns and 'taker_sell_volume_usd' in asset_vol_df.columns:
                            # Convert string columns to numeric values first
                            asset_vol_df['taker_buy_volume_usd'] = pd.to_numeric(asset_vol_df['taker_buy_volume_usd'], errors='coerce')
                            asset_vol_df['taker_sell_volume_usd'] = pd.to_numeric(asset_vol_df['taker_sell_volume_usd'], errors='coerce')
                            
                            # Calculate total volume for each row
                            asset_vol_df['total_volume'] = asset_vol_df['taker_buy_volume_usd'] + asset_vol_df['taker_sell_volume_usd']
                            
                            # Get latest volume (use last row that has valid data)
                            latest_vol = asset_vol_df.dropna(subset=['total_volume']).iloc[-1] if not asset_vol_df.dropna(subset=['total_volume']).empty else None
                            
                            if latest_vol is not None:
                                assets_volume[asset_name] = latest_vol['total_volume']
                                logger.info(f"Successfully processed volume data for {asset_name}: {assets_volume[asset_name]}")
                            else:
                                logger.warning(f"No valid volume data found for {asset_name} after numeric conversion")
                    except Exception as e:
                        logger.error(f"Error processing volume data for {asset_name}: {e}")
                break
    
    return assets_volume

def extract_asset_oi_data(data, asset_list):
    """
    Extract open interest data for multiple assets.
    
    Parameters:
    -----------
    data : dict
        Data dictionary
    asset_list : list
        List of assets to extract data for
        
    Returns:
    --------
    dict
        Dictionary with OI data by asset
    """
    assets_oi = {}
    
    for asset_name in asset_list:
        for key in data.get('open_interest', {}):
            if 'exchange_list' in key.lower() and asset_name.lower() in key.lower():
                asset_oi_df = data['open_interest'][key]
                if not asset_oi_df.empty and any(col in asset_oi_df.columns for col in ['open_interest_usd', 'open_interest']):
                    try:
                        # Determine which column to use
                        oi_col = 'open_interest_usd' if 'open_interest_usd' in asset_oi_df.columns else 'open_interest' 
                        assets_oi[asset_name] = asset_oi_df[oi_col].sum()
                    except Exception as e:
                        logger.error(f"Error processing OI data for {asset_name}: {e}")
                break
    
    return assets_oi

def main():
    """Main function to render the report dashboard."""
    
    # Render sidebar
    render_sidebar()
    
    # Page title and description
    st.title(f"{APP_ICON} Crypto Liquidity Report")
    st.write("Comprehensive analysis of market liquidity metrics across major crypto assets")
    
    # Display loading message
    with st.spinner("Loading dashboard data..."):
        data = load_report_data()
    
    # Removed data last updated reference
    
    # Check if data is available
    if not data:
        st.error("No data available for the dashboard.")
        return
    
    # Get selected asset from session state or use default
    asset = st.session_state.get('selected_asset', DEFAULT_ASSET)
    
    # ==============================
    # SECTION 1: Core Liquidity Metrics
    # ==============================
    st.header("Core Liquidity Metrics & Analysis")
    
    # Calculate key metrics for the header
    metrics = {}
    
    # Total trading volume (24h)
    volume_24h = None
    volume_history_df = None
    
    # Find volume data
    for key in data.get('taker_volume', {}):
        if 'history' in key.lower() and asset.lower() in key.lower():
            try:
                temp_df = data['taker_volume'][key]
                if not temp_df.empty and all(col in temp_df.columns for col in ['taker_buy_volume_usd', 'taker_sell_volume_usd']):
                    volume_history_df = process_timestamps(temp_df)
                    if not volume_history_df.empty:
                        # Get last day of data
                        last_day = volume_history_df.iloc[-1]
                        volume_24h = last_day['taker_buy_volume_usd'] + last_day['taker_sell_volume_usd']
                        break
            except Exception as e:
                logger.error(f"Error processing volume data: {e}")
    
    # Total open interest
    open_interest_total = None
    oi_exchange_df = None
    
    # Find OI data
    for key in data.get('open_interest', {}):
        if 'exchange_list' in key.lower() and asset.lower() in key.lower():
            try:
                temp_df = data['open_interest'][key]
                if not temp_df.empty:
                    # Normalize column names
                    oi_exchange_df = temp_df.copy()
                    if 'exchange' in oi_exchange_df.columns and 'exchange_name' not in oi_exchange_df.columns:
                        oi_exchange_df = oi_exchange_df.rename(columns={'exchange': 'exchange_name'})
                    
                    # Find the OI column
                    oi_col = None
                    for possible_col in ['open_interest_usd', 'open_interest']:
                        if possible_col in oi_exchange_df.columns:
                            oi_col = possible_col
                            break
                    
                    if oi_col:
                        open_interest_total = oi_exchange_df[oi_col].sum()
                        break
            except Exception as e:
                logger.error(f"Error processing open interest data: {e}")
    
    # Get average spread and depth
    spread_pct = None
    total_depth = None
    spreads_df = None
    
    # Find orderbook data
    for key in data.get('order_book', {}):
        if 'ask_bids_history' in key.lower() and asset.lower() in key.lower():
            try:
                ob_df = data['order_book'][key]
                if not ob_df.empty and all(col in ob_df.columns for col in ['bids_usd', 'asks_usd']):
                    spreads_df = calculate_spreads_and_depth(ob_df, asset)
                    if not spreads_df.empty and 'spread_pct' in spreads_df.columns:
                        spread_pct = spreads_df['spread_pct'].iloc[-1] if len(spreads_df) > 0 else None
                        total_depth = spreads_df['total_depth'].iloc[-1] if len(spreads_df) > 0 and 'total_depth' in spreads_df.columns else None
                        break
            except Exception as e:
                logger.error(f"Error processing orderbook data: {e}")
    
    # If we didn't find asset-specific orderbook data, try generic one
    if spreads_df is None or spreads_df.empty:
        for key in data.get('order_book', {}):
            if 'ask_bids_history' in key.lower() and 'aggregated' not in key.lower():
                try:
                    ob_df = data['order_book'][key]
                    if not ob_df.empty and all(col in ob_df.columns for col in ['bids_usd', 'asks_usd']):
                        spreads_df = calculate_spreads_and_depth(ob_df, asset)
                        if not spreads_df.empty and 'spread_pct' in spreads_df.columns:
                            spread_pct = spreads_df['spread_pct'].iloc[-1] if len(spreads_df) > 0 else None
                            total_depth = spreads_df['total_depth'].iloc[-1] if len(spreads_df) > 0 and 'total_depth' in spreads_df.columns else None
                            break
                except Exception as e:
                    logger.error(f"Error processing generic orderbook data: {e}")
    
    # Get current funding rate
    funding_rate = None
    funding_exchange_df = None
    
    # Find funding rate data
    for key in data.get('funding_rate', {}):
        if 'exchange_list' in key.lower():
            try:
                fr_df = data['funding_rate'][key]
                if not fr_df.empty:
                    # Process and normalize funding rate data
                    funding_exchange_df = normalize_funding_rate_data(fr_df, asset)
                    if not funding_exchange_df.empty and 'funding_rate' in funding_exchange_df.columns:
                        funding_rate = funding_exchange_df['funding_rate'].mean()
                        break
            except Exception as e:
                logger.error(f"Error processing funding rate data: {e}")
    
    # Build metrics for display
    if volume_24h is not None:
        metrics["24h Trading Volume"] = {
            "value": volume_24h,
            "delta": None
        }
    
    if open_interest_total is not None:
        metrics["Total Open Interest"] = {
            "value": open_interest_total,
            "delta": None
        }
    
    if spread_pct is not None:
        metrics["Avg Bid-Ask Spread"] = {
            "value": spread_pct,
            "delta": None,
            "delta_suffix": "%"
        }
    
    if total_depth is not None:
        metrics["Order Book Depth"] = {
            "value": total_depth,
            "delta": None
        }
    
    if funding_rate is not None:
        metrics["Current Funding Rate"] = {
            "value": funding_rate * 100,  # Convert to percentage
            "delta": None,
            "delta_suffix": "%"
        }
    
    # Create formatters for the metrics
    formatters = {
        "24h Trading Volume": lambda x: format_currency(x, abbreviate=True),
        "Total Open Interest": lambda x: format_currency(x, abbreviate=True),
        "Avg Bid-Ask Spread": lambda x: format_percentage(x, precision=3),
        "Order Book Depth": lambda x: format_currency(x, abbreviate=True),
        "Current Funding Rate": lambda x: format_percentage(x, precision=3)
    }
    
    # Display the metrics
    display_metrics_row(metrics, formatters)
    
    # ==============================
    # SECTION 2: Trading Volume Analysis
    # ==============================
    st.subheader("Trading Volume Analysis")
    
    if volume_history_df is not None and not volume_history_df.empty:
        # Add total volume column
        if 'taker_buy_volume_usd' in volume_history_df.columns and 'taker_sell_volume_usd' in volume_history_df.columns:
            volume_history_df['total_volume'] = volume_history_df['taker_buy_volume_usd'] + volume_history_df['taker_sell_volume_usd']
            
            # Create volume chart
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Stacked volume bar chart
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=volume_history_df['datetime'],
                    y=volume_history_df['taker_buy_volume_usd'],
                    name='Buy Volume',
                    marker_color='green'
                ))
                
                fig.add_trace(go.Bar(
                    x=volume_history_df['datetime'],
                    y=volume_history_df['taker_sell_volume_usd'],
                    name='Sell Volume',
                    marker_color='red'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"{asset} Futures Trading Volume",
                    barmode='stack',
                    xaxis_title=None,
                    yaxis_title="Volume (USD)",
                    hovermode="x unified"
                )
                
                display_chart(apply_chart_theme(fig))
            
            with col2:
                # Get volume by exchange
                volume_by_exchange = None
                for key in data.get('market', {}):
                    if 'pairs_markets' in key.lower() and asset.lower() in key.lower():
                        market_df = data['market'][key]
                        if not market_df.empty and 'volume_usd' in market_df.columns and 'exchange_name' in market_df.columns:
                            volume_by_exchange = market_df[['exchange_name', 'volume_usd']].sort_values('volume_usd', ascending=False)
                            break
                
                if volume_by_exchange is not None and not volume_by_exchange.empty:
                    # Create pie chart for volume distribution using improved function
                    # Filter out any "All" values that shouldn't be in the pie chart
                    filtered_volume_data = volume_by_exchange[volume_by_exchange['exchange_name'] != "All"]
                    
                    # Import enhanced pie chart function
                    from utils.chart_utils import create_enhanced_pie_chart
                    
                    fig = create_enhanced_pie_chart(
                        df=filtered_volume_data,
                        values_col='volume_usd',
                        names_col='exchange_name',
                        title=f"{asset} Trading Volume by Exchange",
                        color_map=EXCHANGE_COLORS,
                        exclude_names=["All"],
                        show_top_n=8,  # Show top 8 exchanges
                        min_percent=2.0,  # Group exchanges with less than 2% share
                        height=400
                    )
                    
                    display_chart(fig)
                else:
                    display_chart(create_fallback_chart(f"{asset} Trading Volume by Exchange", "No exchange volume data available"))
        else:
            display_chart(create_fallback_chart(f"{asset} Futures Trading Volume", "Incomplete volume data available"))
    else:
        display_chart(create_fallback_chart(f"{asset} Futures Trading Volume", "No trading volume data available"))
    
    # Volume comparison across assets
    st.subheader("Trading Volume Comparison Across Assets")
    
    # Add time range selection
    col1, col2 = st.columns([2, 1])
    with col2:
        # Time range selector
        time_range = st.selectbox(
            "Select Time Range", 
            ["24h", "7d", "30d", "90d"],
            index=0,
            help="Select time range for volume comparison"
        )
    
    with col1:
        # Add normalization option
        normalize_method = st.radio(
            "Visualization Type",
            ["Absolute Volume", "Market Share (%)", "Volume Relative to BTC"],
            horizontal=True,
            index=0,
            help="Choose how to visualize the volume comparison"
        )
    
    # Get volume data for all major assets
    try:
        assets_volume = extract_asset_volume_data(data, ['BTC', 'ETH', 'SOL', 'XRP'])
        
        # Log success or issues with data
        if assets_volume:
            assets_found = list(assets_volume.keys())
            logger.info(f"Successfully extracted volume data for assets: {assets_found}")
            
            # Check if we're missing any assets and log warning
            missing_assets = [a for a in ['BTC', 'ETH', 'SOL', 'XRP'] if a not in assets_found]
            if missing_assets:
                logger.warning(f"Missing volume data for assets: {missing_assets}")
        else:
            logger.warning("No volume data extracted for any assets")
            
    except Exception as e:
        logger.error(f"Error extracting volume data: {e}")
        assets_volume = {}
        st.error(f"Error loading volume comparison data: {e}")
        
    # Fallback suggestions if data is missing
    volume_fallback_suggestions = [
        "Check that volume data files exist in the data directory",
        "Ensure asset-specific volume files follow the expected naming pattern",
        "Verify that the volume data contains 'taker_buy_volume_usd' and 'taker_sell_volume_usd' columns"
    ]
    
    if assets_volume:
        # Create comparison bar chart
        volume_comp_df = pd.DataFrame({
            'Asset': list(assets_volume.keys()),
            'Volume (USD)': list(assets_volume.values())
        }).sort_values('Volume (USD)', ascending=False)
        
        # Apply normalization if selected
        chart_title = f"{time_range} Trading Volume by Asset"
        y_axis_title = "Volume (USD)"
        
        if normalize_method == "Market Share (%)":
            # Calculate percentage of total
            total_volume = volume_comp_df['Volume (USD)'].sum()
            if total_volume > 0:
                volume_comp_df['Volume (%)'] = (volume_comp_df['Volume (USD)'] / total_volume) * 100
                y_column = 'Volume (%)'
                chart_title = f"{time_range} Trading Volume Share by Asset"
                y_axis_title = "Volume Share (%)"
            else:
                y_column = 'Volume (USD)'
        elif normalize_method == "Volume Relative to BTC":
            # Normalize relative to BTC
            btc_volume = volume_comp_df.loc[volume_comp_df['Asset'] == 'BTC', 'Volume (USD)'].iloc[0] if 'BTC' in volume_comp_df['Asset'].values else 1
            if btc_volume > 0:
                volume_comp_df['Volume (Relative to BTC)'] = volume_comp_df['Volume (USD)'] / btc_volume
                y_column = 'Volume (Relative to BTC)'
                chart_title = f"{time_range} Trading Volume Relative to BTC"
                y_axis_title = "Volume (BTC = 1)"
            else:
                y_column = 'Volume (USD)'
        else:
            y_column = 'Volume (USD)'
        
        # Create bar chart with the selected normalization
        fig = px.bar(
            volume_comp_df,
            x='Asset',
            y=y_column,
            title=chart_title,
            color='Asset',
            color_discrete_map={
                'BTC': ASSET_COLORS.get('BTC', '#F7931A'),
                'ETH': ASSET_COLORS.get('ETH', '#627EEA'),
                'SOL': ASSET_COLORS.get('SOL', '#00FFA3'),
                'XRP': ASSET_COLORS.get('XRP', '#23292F')
            },
            text=y_column
        )
        
        # Customize text display format based on normalization type
        if normalize_method == "Market Share (%)":
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        elif normalize_method == "Volume Relative to BTC":
            fig.update_traces(texttemplate='%{text:.2f}x', textposition='outside')
        else:
            fig.update_traces(texttemplate='%{text:$.2s}', textposition='outside')
        
        fig.update_layout(
            xaxis_title=None,
            yaxis_title=y_axis_title,
            uniformtext_minsize=10,
            uniformtext_mode='hide',
            bargap=0.3
        )
        
        # Format y-axis based on normalization type
        if normalize_method == "Market Share (%)":
            fig.update_yaxes(ticksuffix="%")
        elif normalize_method == "Absolute Volume":
            fig.update_yaxes(tickformat="$.2s")
        
        display_chart(apply_chart_theme(fig))
        
        # Add a table with exact values below the chart
        st.markdown("### Detailed Volume Comparison")
        
        # Prepare detailed table
        if normalize_method == "Market Share (%)" and 'Volume (%)' in volume_comp_df.columns:
            table_df = volume_comp_df[['Asset', 'Volume (USD)', 'Volume (%)']].copy()
            format_dict = {
                'Volume (USD)': lambda x: format_currency(x, abbreviate=True),
                'Volume (%)': lambda x: f"{x:.2f}%"
            }
        elif normalize_method == "Volume Relative to BTC" and 'Volume (Relative to BTC)' in volume_comp_df.columns:
            table_df = volume_comp_df[['Asset', 'Volume (USD)', 'Volume (Relative to BTC)']].copy()
            format_dict = {
                'Volume (USD)': lambda x: format_currency(x, abbreviate=True),
                'Volume (Relative to BTC)': lambda x: f"{x:.2f}x"
            }
        else:
            table_df = volume_comp_df[['Asset', 'Volume (USD)']].copy()
            format_dict = {
                'Volume (USD)': lambda x: format_currency(x, abbreviate=True)
            }
        
        # Display the table
        create_formatted_table(table_df, format_dict=format_dict)
    else:
        display_chart(create_fallback_chart(
            f"{time_range} Trading Volume by Asset", 
            "No volume comparison data available",
            suggestions=volume_fallback_suggestions
        ))
    
    # ==============================
    # SECTION 3: Open Interest Analysis
    # ==============================
    st.subheader("Open Interest Analysis")
    
    # Open Interest history
    oi_history_df = None
    for key in data.get('open_interest', {}):
        if 'aggregated_history' in key.lower() and asset.lower() in key.lower():
            temp_df = data['open_interest'][key]
            if not temp_df.empty:
                oi_history_df = process_timestamps(temp_df)
                break
    
    # If asset-specific OI history not found, try to find any OI history data
    if oi_history_df is None or oi_history_df.empty:
        for key in data.get('open_interest', {}):
            if 'history' in key.lower() and 'ohlc' in key.lower():
                temp_df = data['open_interest'][key]
                if not temp_df.empty:
                    # If we have a symbol column, filter for the asset
                    if 'symbol' in temp_df.columns:
                        temp_df = temp_df[temp_df['symbol'].str.contains(asset, case=False, na=False)]
                    
                    if not temp_df.empty:
                        oi_history_df = process_timestamps(temp_df)
                        break
    
    # Price data for overlay
    price_history_df = None
    for key in data.get('price', {}):
        if 'ohlc_history' in key.lower():
            temp_df = data['price'][key]
            if not temp_df.empty:
                # If we have a symbol column, filter for the asset
                if 'symbol' in temp_df.columns:
                    temp_df = temp_df[temp_df['symbol'].str.contains(asset, case=False, na=False)]
                
                if not temp_df.empty:
                    price_history_df = process_timestamps(temp_df)
                    break
    
    # Create two columns for OI charts
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if oi_history_df is not None and not oi_history_df.empty:
            # Check if we have the required columns
            if all(col in oi_history_df.columns for col in ['datetime', 'close']):
                # Create figure with price overlay if available
                fig = go.Figure()
                
                # Add OI line
                fig.add_trace(go.Scatter(
                    x=oi_history_df['datetime'],
                    y=oi_history_df['close'],
                    name="Open Interest",
                    line=dict(color=ASSET_COLORS.get(asset, '#3366CC'), width=2)
                ))
                
                # Add price overlay if available
                if price_history_df is not None and not price_history_df.empty and 'close' in price_history_df.columns:
                    # Create second y-axis for price
                    fig.add_trace(go.Scatter(
                        x=price_history_df['datetime'],
                        y=price_history_df['close'],
                        name=f"{asset} Price",
                        line=dict(color='#FF9900', width=1.5, dash='dot'),
                        yaxis="y2"
                    ))
                    
                    # Update layout for second y-axis
                    fig.update_layout(
                        yaxis2=dict(
                            title=f"{asset} Price (USD)",
                            overlaying="y",
                            side="right"
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"{asset} Open Interest History",
                    xaxis_title=None,
                    yaxis_title="Open Interest (USD)",
                    hovermode="x unified"
                )
                
                display_chart(apply_chart_theme(fig))
                
                # Set defaults in session state for backward compatibility
                st.session_state.oi_history_time_range = 'All'
                st.session_state.selected_time_range = 'All'
            else:
                display_chart(create_fallback_chart(f"{asset} Open Interest History", "Incomplete open interest data"))
        else:
            display_chart(create_fallback_chart(f"{asset} Open Interest History", "No open interest history data available"))
    
    with col2:
        if oi_exchange_df is not None and not oi_exchange_df.empty:
            # Create pie chart for OI distribution
            # Find the OI column
            oi_col = None
            for possible_col in ['open_interest_usd', 'open_interest']:
                if possible_col in oi_exchange_df.columns:
                    oi_col = possible_col
                    break
                    
            if oi_col and 'exchange_name' in oi_exchange_df.columns:
                oi_by_exchange = oi_exchange_df[['exchange_name', oi_col]].sort_values(oi_col, ascending=False)
                
                # Filter out any "All" values
                filtered_oi_data = oi_by_exchange[oi_by_exchange['exchange_name'] != "All"]
                
                fig = create_pie_chart(
                    df=filtered_oi_data,
                    values_col=oi_col,
                    names_col='exchange_name',
                    title=f"{asset} Open Interest by Exchange",
                    color_map=EXCHANGE_COLORS,
                    exclude_names=["All"],
                    show_top_n=8,  # Show top 8 exchanges
                    min_percent=2.0,  # Group exchanges with less than 2% share
                    height=400
                )
                
                display_chart(fig)
            else:
                display_chart(create_fallback_chart(f"{asset} Open Interest by Exchange", "Incomplete exchange data"))
        else:
            display_chart(create_fallback_chart(f"{asset} Open Interest by Exchange", "No exchange open interest data available"))
    
    # OI Comparison across assets
    st.subheader("Open Interest Comparison Across Assets")
    
    # Get OI data for all major assets
    assets_oi = extract_asset_oi_data(data, ['BTC', 'ETH', 'SOL', 'XRP'])
    
    if assets_oi:
        # Create comparison bar chart
        oi_comp_df = pd.DataFrame({
            'Asset': list(assets_oi.keys()),
            'Open Interest (USD)': list(assets_oi.values())
        }).sort_values('Open Interest (USD)', ascending=False)
        
        fig = px.bar(
            oi_comp_df,
            x='Asset',
            y='Open Interest (USD)',
            title="Open Interest by Asset",
            color='Asset',
            color_discrete_map={
                'BTC': ASSET_COLORS.get('BTC', '#F7931A'),
                'ETH': ASSET_COLORS.get('ETH', '#627EEA'),
                'SOL': ASSET_COLORS.get('SOL', '#00FFA3'),
                'XRP': ASSET_COLORS.get('XRP', '#23292F')
            }
        )
        
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Open Interest (USD)"
        )
        
        # Format y-axis
        fig.update_yaxes(tickformat="$.2s")
        
        display_chart(apply_chart_theme(fig))
    else:
        display_chart(create_fallback_chart("Open Interest by Asset", "No open interest comparison data available"))
    
    # ==============================
    # SECTION 4: Bid-Ask Spread Analysis
    # ==============================
    st.subheader("Bid-Ask Spread Analysis")
    
    if spreads_df is not None and not spreads_df.empty and 'spread_pct' in spreads_df.columns:
        # Create two columns for spread charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Create time series of spread percentage
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=spreads_df['datetime'],
                y=spreads_df['spread_pct'],
                name="Spread Percentage",
                line=dict(color='#3366CC', width=2)
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{asset} Bid-Ask Spread Percentage",
                xaxis_title=None,
                yaxis_title="Spread (%)",
                hovermode="x unified"
            )
            
            display_chart(apply_chart_theme(fig))
            
            # Add basic spread statistics
            spread_stats = {
                'Current': spreads_df['spread_pct'].iloc[-1],
                'Average': spreads_df['spread_pct'].mean(),
                'Min': spreads_df['spread_pct'].min(),
                'Max': spreads_df['spread_pct'].max()
            }
            
            st.markdown("### Spread Statistics")
            stats_cols = st.columns(4)
            
            for i, (label, value) in enumerate(spread_stats.items()):
                with stats_cols[i]:
                    st.metric(label, f"{value:.3f}%")
        
        with col2:
            # Create time series of market depth
            if 'bid_depth' in spreads_df.columns and 'ask_depth' in spreads_df.columns:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=spreads_df['datetime'],
                    y=spreads_df['bid_depth'],
                    name="Bid Depth",
                    line=dict(color='green', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=spreads_df['datetime'],
                    y=spreads_df['ask_depth'],
                    name="Ask Depth",
                    line=dict(color='red', width=2)
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"{asset} Order Book Depth",
                    xaxis_title=None,
                    yaxis_title="Depth (USD)",
                    hovermode="x unified"
                )
                
                # Format y-axis
                fig.update_yaxes(tickformat="$.2s")
                
                display_chart(apply_chart_theme(fig))
                
                # Add depth statistics
                depth_stats = {
                    'Total Depth': spreads_df['total_depth'].iloc[-1],
                    'Bid Depth': spreads_df['bid_depth'].iloc[-1],
                    'Ask Depth': spreads_df['ask_depth'].iloc[-1],
                    'Bid/Ask Ratio': spreads_df['bid_depth'].iloc[-1] / spreads_df['ask_depth'].iloc[-1] if spreads_df['ask_depth'].iloc[-1] != 0 else 0
                }
                
                st.markdown("### Depth Statistics")
                depth_stats_cols = st.columns(4)
                
                with depth_stats_cols[0]:
                    st.metric("Total Depth", format_currency(depth_stats['Total Depth'], abbreviate=True))
                
                with depth_stats_cols[1]:
                    st.metric("Bid Depth", format_currency(depth_stats['Bid Depth'], abbreviate=True))
                
                with depth_stats_cols[2]:
                    st.metric("Ask Depth", format_currency(depth_stats['Ask Depth'], abbreviate=True))
                
                with depth_stats_cols[3]:
                    st.metric("Bid/Ask Ratio", f"{depth_stats['Bid/Ask Ratio']:.2f}")
            else:
                display_chart(create_fallback_chart(f"{asset} Order Book Depth", "No order book depth data available"))
    else:
        col1, col2 = st.columns(2)
        with col1:
            display_chart(create_fallback_chart(f"{asset} Bid-Ask Spread Percentage", "No bid-ask spread data available"))
        with col2:
            display_chart(create_fallback_chart(f"{asset} Order Book Depth", "No order book depth data available"))
    
    # ==============================
    # SECTION 5: Funding Rate Analysis
    # ==============================
    st.subheader("Funding Rate Dynamics")
    
    # Get funding rate history data
    funding_history_df = None
    funding_type = None
    funding_type_display = ""
    funding_data_files_checked = []
    
    try:
        # First try to load the OHLC history data
        for key in data.get('funding_rate', {}):
            if 'ohlc_history' in key.lower():
                funding_data_files_checked.append(key)
                temp_df = data['funding_rate'][key]
                if not temp_df.empty:
                    # Process the data to convert strings to numeric and handle timestamps
                    funding_history_df = process_timestamps(temp_df)
                    
                    # Convert OHLC columns to numeric
                    for col in ['open', 'high', 'low', 'close']:
                        if col in funding_history_df.columns:
                            funding_history_df[col] = pd.to_numeric(funding_history_df[col], errors='coerce')
                    
                    funding_type = "general"
                    funding_type_display = "General"
                    logger.info(f"Found general funding rate OHLC history data with shape {funding_history_df.shape}")
                    break
                else:
                    logger.warning(f"Found empty funding rate OHLC history data file: {key}")
        
        # If we didn't find general funding rate history, try asset-specific ones
        if funding_history_df is None or funding_history_df.empty:
            # Try OI-weighted funding rate data
            for key in data.get('funding_rate', {}):
                if 'oi_weight' in key.lower() and asset.lower() in key.lower():
                    funding_data_files_checked.append(key)
                    temp_df = data['funding_rate'][key]
                    if not temp_df.empty:
                        funding_history_df = process_timestamps(temp_df)
                        
                        # Convert OHLC columns to numeric
                        for col in ['open', 'high', 'low', 'close']:
                            if col in funding_history_df.columns:
                                funding_history_df[col] = pd.to_numeric(funding_history_df[col], errors='coerce')
                        
                        funding_type = "oi_weighted"
                        funding_type_display = "OI-Weighted"
                        logger.info(f"Found OI-weighted funding rate history for {asset} with shape {funding_history_df.shape}")
                        break
                    else:
                        logger.warning(f"Found empty OI-weighted funding rate data file: {key}")
            
            # If still not found, try volume-weighted funding rate data
            if funding_history_df is None or funding_history_df.empty:
                for key in data.get('funding_rate', {}):
                    if 'vol_weight' in key.lower() and asset.lower() in key.lower():
                        funding_data_files_checked.append(key)
                        temp_df = data['funding_rate'][key]
                        if not temp_df.empty:
                            funding_history_df = process_timestamps(temp_df)
                            
                            # Convert OHLC columns to numeric
                            for col in ['open', 'high', 'low', 'close']:
                                if col in funding_history_df.columns:
                                    funding_history_df[col] = pd.to_numeric(funding_history_df[col], errors='coerce')
                            
                            funding_type = "volume_weighted"
                            funding_type_display = "Volume-Weighted"
                            logger.info(f"Found volume-weighted funding rate history for {asset} with shape {funding_history_df.shape}")
                            break
                        else:
                            logger.warning(f"Found empty volume-weighted funding rate data file: {key}")
        
        # Log summary of data loading process
        if funding_history_df is not None and not funding_history_df.empty:
            logger.info(f"Successfully loaded {funding_type_display} funding rate data with {len(funding_history_df)} rows")
        else:
            logger.warning(f"Could not find valid funding rate data. Checked files: {funding_data_files_checked}")
    
    except Exception as e:
        logger.error(f"Error loading funding rate data: {e}")
        st.error(f"Error loading funding rate data: {e}")
    
    # Fallback suggestions if data is missing
    funding_rate_fallback_suggestions = [
        "Check that funding rate data files exist in the funding_rate directory",
        "Ensure asset-specific funding rate files follow the expected naming pattern",
        "Verify that the funding rate data contains OHLC columns with numeric values",
        f"Files checked: {', '.join(funding_data_files_checked)}"
    ]
    
    # If we have funding rate data, calculate moving averages
    if funding_history_df is not None and not funding_history_df.empty:
        # Determine which column to use
        rate_col = 'close' if 'close' in funding_history_df.columns else 'rate' if 'rate' in funding_history_df.columns else 'funding_rate'
        
        if rate_col in funding_history_df.columns:
            # Calculate moving averages
            funding_history_df['7d_ma'] = funding_history_df[rate_col].rolling(window=7).mean()
            funding_history_df['30d_ma'] = funding_history_df[rate_col].rolling(window=30).mean()
            logger.info(f"Calculated moving averages for funding rate data")
    
    # Add time period selector
    time_period = st.radio(
        "Time Period",
        ["All Time", "Last 30 Days", "Last 90 Days", "Last Year"],
        horizontal=True,
        index=1
    )
    
    # Filter data based on selected time period
    if funding_history_df is not None and not funding_history_df.empty and 'datetime' in funding_history_df.columns:
        today = datetime.now()
        if time_period == "Last 30 Days":
            cutoff_date = today - timedelta(days=30)
            filtered_funding_df = funding_history_df[funding_history_df['datetime'] >= cutoff_date]
        elif time_period == "Last 90 Days":
            cutoff_date = today - timedelta(days=90)
            filtered_funding_df = funding_history_df[funding_history_df['datetime'] >= cutoff_date]
        elif time_period == "Last Year":
            cutoff_date = today - timedelta(days=365)
            filtered_funding_df = funding_history_df[funding_history_df['datetime'] >= cutoff_date]
        else:
            filtered_funding_df = funding_history_df
            
        # If filtering returned an empty dataframe, use the original one
        if filtered_funding_df.empty:
            filtered_funding_df = funding_history_df
            st.warning(f"No data available for the selected time period. Showing all available data.")
    else:
        filtered_funding_df = None
    
    # Display funding rate analysis
    col1, col2 = st.columns(2)
    
    with col1:
        if filtered_funding_df is not None and not filtered_funding_df.empty:
            # Check if we have the required columns
            if 'datetime' in filtered_funding_df.columns and any(col in filtered_funding_df.columns for col in ['close', 'rate', 'funding_rate']):
                fig = go.Figure()
                
                # Determine which column to use
                rate_col = 'close' if 'close' in filtered_funding_df.columns else 'rate' if 'rate' in filtered_funding_df.columns else 'funding_rate'
                
                # Add main funding rate line
                fig.add_trace(go.Scatter(
                    x=filtered_funding_df['datetime'],
                    y=filtered_funding_df[rate_col] * 100,  # Convert to percentage
                    name="Funding Rate",
                    line=dict(color=ASSET_COLORS.get(asset, '#3366CC'), width=2)
                ))
                
                # Add moving averages if available
                if '7d_ma' in filtered_funding_df.columns:
                    fig.add_trace(go.Scatter(
                        x=filtered_funding_df['datetime'],
                        y=filtered_funding_df['7d_ma'] * 100,  # Convert to percentage
                        name="7-Day MA",
                        line=dict(color='orange', width=1.5, dash='dash')
                    ))
                
                if '30d_ma' in filtered_funding_df.columns:
                    fig.add_trace(go.Scatter(
                        x=filtered_funding_df['datetime'],
                        y=filtered_funding_df['30d_ma'] * 100,  # Convert to percentage
                        name="30-Day MA",
                        line=dict(color='red', width=1.5, dash='dot')
                    ))
                
                # Add horizontal line at zero
                fig.add_shape(
                    type="line",
                    x0=filtered_funding_df['datetime'].min(),
                    y0=0,
                    x1=filtered_funding_df['datetime'].max(),
                    y1=0,
                    line=dict(
                        color="gray",
                        width=1,
                        dash="dash",
                    )
                )
                
                # Create title based on funding type
                title_suffix = ""
                if funding_type == "oi_weighted":
                    title_suffix = " (OI-Weighted)"
                elif funding_type == "volume_weighted":
                    title_suffix = " (Volume-Weighted)"
                
                # Update layout
                fig.update_layout(
                    title=f"{asset} Funding Rate History{title_suffix}",
                    xaxis_title=None,
                    yaxis_title="Funding Rate (%)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                display_chart(apply_chart_theme(fig))
            else:
                display_chart(create_fallback_chart(
                f"{asset} Funding Rate History", 
                "Incomplete funding rate data - Missing required columns",
                suggestions=funding_rate_fallback_suggestions
            ))
        else:
            display_chart(create_fallback_chart(
                f"{asset} Funding Rate History", 
                "No funding rate history data available",
                suggestions=funding_rate_fallback_suggestions
            ))
    
    with col2:
        # Funding Rate Interpretation
        st.markdown("### Funding Rate Interpretation")
        
        # Calculate current funding rate stats
        current_rate = None
        avg_rate = None
        volatility = None
        
        if filtered_funding_df is not None and not filtered_funding_df.empty:
            rate_col = 'close' if 'close' in filtered_funding_df.columns else 'rate' if 'rate' in filtered_funding_df.columns else 'funding_rate'
            if rate_col in filtered_funding_df.columns:
                current_rate = filtered_funding_df[rate_col].iloc[-1] * 100 if len(filtered_funding_df) > 0 else None
                avg_rate = filtered_funding_df[rate_col].mean() * 100 if len(filtered_funding_df) > 0 else None
                # Calculate volatility (standard deviation)
                volatility = filtered_funding_df[rate_col].std() * 100 if len(filtered_funding_df) > 0 else None
        elif funding_rate is not None:
            current_rate = funding_rate * 100
        
        # Determine market sentiment
        if current_rate is not None:
            if current_rate > 0.05:
                sentiment = "Strongly Bullish"
                sentiment_color = "green"
            elif current_rate > 0.01:
                sentiment = "Moderately Bullish"
                sentiment_color = "lightgreen"
            elif current_rate >= -0.01:
                sentiment = "Neutral"
                sentiment_color = "gray"
            elif current_rate >= -0.05:
                sentiment = "Moderately Bearish"
                sentiment_color = "lightcoral"
            else:
                sentiment = "Strongly Bearish"
                sentiment_color = "red"
            
            st.markdown(f"**Current Market Sentiment:** <span style='color:{sentiment_color}'>{sentiment}</span>", unsafe_allow_html=True)
            
            # Add explanation of what funding rate means
            st.markdown("""
            **Funding Rate Significance:**
            - **Positive Rate:** Longs pay shorts; indicates bullish sentiment
            - **Negative Rate:** Shorts pay longs; indicates bearish sentiment
            - **High Absolute Values:** Potential market extremes or imbalances
            
            **Trading Implications:**
            - High positive rates may signal overheated market conditions
            - High negative rates often occur during market capitulation
            - Extreme rates can precede mean reversion or short-term reversals
            - Moving averages show trend direction and momentum
            """)
            
            # Display key stats
            stats_cols = st.columns(2)
            
            with stats_cols[0]:
                st.metric("Current Rate", f"{current_rate:.3f}%")
                
                if avg_rate is not None:
                    st.metric("Historical Average", f"{avg_rate:.3f}%")
                else:
                    st.metric("Historical Average", "N/A")
            
            with stats_cols[1]:
                if volatility is not None:
                    st.metric("Rate Volatility", f"{volatility:.3f}%")
                else:
                    st.metric("Rate Volatility", "N/A")
                    
                if current_rate is not None:
                    st.metric("Annualized Cost", f"{(current_rate * 3 * 365):.2f}%",
                             help="Estimated annual cost of holding perpetual position at current funding rate")
                else:
                    st.metric("Annualized Cost", "N/A")
            
            # If we have price data, compute correlation between funding rate and price
            if filtered_funding_df is not None and price_history_df is not None:
                st.markdown("### Funding Rate and Price Correlation")
                
                try:
                    # Try to merge dataframes on datetime
                    rate_col = 'close' if 'close' in filtered_funding_df.columns else 'rate' if 'rate' in filtered_funding_df.columns else 'funding_rate'
                    
                    # Ensure both dataframes have datetime as index
                    # First explicitly convert any numeric columns to ensure they're not objects
                    numeric_df = filtered_funding_df.copy()
                    # Ensure the column is numeric before resampling
                    numeric_df[rate_col] = pd.to_numeric(numeric_df[rate_col], errors='coerce')
                    funding_df_resampled = numeric_df.set_index('datetime').resample('1D').mean()
                    
                    # Do the same for price data
                    price_numeric_df = price_history_df.copy()
                    price_numeric_df['close'] = pd.to_numeric(price_numeric_df['close'], errors='coerce')
                    price_df_resampled = price_numeric_df.set_index('datetime').resample('1D').mean()
                    
                    # Merge on index
                    merged_df = pd.merge(
                        funding_df_resampled[rate_col].to_frame('funding_rate'),
                        price_df_resampled['close'].to_frame('price'),
                        left_index=True, right_index=True,
                        how='inner'
                    )
                    
                    if len(merged_df) >= 5:  # At least 5 data points for correlation
                        correlation = merged_df['funding_rate'].corr(merged_df['price'])
                        
                        corr_cols = st.columns(2)
                        with corr_cols[0]:
                            st.metric("Funding-Price Correlation", f"{correlation:.2f}",
                                     help="Correlation between funding rate and price. Range -1 to 1.")
                        
                        with corr_cols[1]:
                            # Interpret correlation
                            if correlation > 0.7:
                                interpretation = "Strong positive correlation"
                            elif correlation > 0.3:
                                interpretation = "Moderate positive correlation"
                            elif correlation > -0.3:
                                interpretation = "Weak correlation"
                            elif correlation > -0.7:
                                interpretation = "Moderate negative correlation"
                            else:
                                interpretation = "Strong negative correlation"
                            
                            st.markdown(f"**Correlation Strength:** {interpretation}")
                            
                        # Add scatter plot of funding rate vs price
                        fig = px.scatter(
                            merged_df.reset_index(), 
                            x='funding_rate',
                            y='price',
                            trendline='ols',
                            title=f"{asset} Funding Rate vs Price Correlation",
                            labels={'funding_rate': 'Funding Rate', 'price': f'{asset} Price (USD)'},
                            color_discrete_sequence=[ASSET_COLORS.get(asset, '#3366CC')]
                        )
                        display_chart(apply_chart_theme(fig))
                    else:
                        st.info("Insufficient data points to calculate correlation.")
                except Exception as e:
                    logger.error(f"Error calculating funding rate-price correlation: {e}")
                    st.info("Could not calculate correlation between funding rate and price.")
        else:
            st.info(f"No funding rate data available for {asset}.")
    
    # Display funding rates by exchange
    st.subheader("Exchange Funding Rate Comparison")
    
    # Add filter for margin type
    margin_filter = None
    if funding_exchange_df is not None and not funding_exchange_df.empty and 'margin_type' in funding_exchange_df.columns:
        margin_types = ['All'] + sorted(funding_exchange_df['margin_type'].unique().tolist())
        margin_filter = st.selectbox(
            "Filter by Margin Type", 
            margin_types,
            index=0,
            help="Filter exchanges by margin type (stablecoin or token margin)"
        )
    
    if funding_exchange_df is not None and not funding_exchange_df.empty and 'funding_rate' in funding_exchange_df.columns:
        # Apply margin type filter if selected
        filtered_df = funding_exchange_df
        if margin_filter and margin_filter != 'All' and 'margin_type' in funding_exchange_df.columns:
            filtered_df = funding_exchange_df[funding_exchange_df['margin_type'] == margin_filter.lower()]
        
        # Make sure we have data after filtering
        if not filtered_df.empty:
            # Sort by funding rate
            filtered_df = filtered_df.sort_values('funding_rate', ascending=False)
            
            # Convert funding rate to percentage
            filtered_df['funding_rate_pct'] = filtered_df['funding_rate'] * 100
            
            # Create bar chart with improved formatting
            fig = px.bar(
                filtered_df.head(15),  # Top 15 exchanges
                x='exchange_name',
                y='funding_rate_pct',
                title=f"{asset} Current Funding Rates by Exchange" + (f" ({margin_filter} Margin)" if margin_filter and margin_filter != 'All' else ""),
                color='funding_rate_pct',
                color_continuous_scale='RdBu_r',
                color_continuous_midpoint=0,
                text='funding_rate_pct'
            )
            
            # Format text labels
            fig.update_traces(
                texttemplate='%{text:.3f}%',
                textposition='outside'
            )
            
            fig.update_layout(
                xaxis_title=None,
                yaxis_title="Funding Rate (%)",
                uniformtext_minsize=8,
                uniformtext_mode='hide'
            )
            
            # Format axis
            fig.update_xaxes(tickangle=-45)
            
            display_chart(apply_chart_theme(fig))
            
            # Add exchange count and stats
            st.markdown(f"**Showing funding rates for {len(filtered_df.head(15))} out of {len(filtered_df)} exchanges**")
            
            # Create two columns for different visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Create funding rate histogram
                hist_fig = px.histogram(
                    filtered_df,
                    x='funding_rate_pct',
                    nbins=20,
                    title=f"{asset} Funding Rate Distribution",
                    labels={'funding_rate_pct': 'Funding Rate (%)'},
                    color_discrete_sequence=[ASSET_COLORS.get(asset, '#3366CC')]
                )
                
                # Add vertical line at mean
                mean_rate = filtered_df['funding_rate_pct'].mean()
                hist_fig.add_vline(
                    x=mean_rate,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Mean: {mean_rate:.3f}%",
                    annotation_position="top right"
                )
                
                display_chart(apply_chart_theme(hist_fig))
            
            with col2:
                # Create funding rate statistics table
                if 'margin_type' in filtered_df.columns:
                    # Group by margin type
                    margin_groups = filtered_df.groupby('margin_type')
                    
                    # Calculate statistics
                    stats_data = []
                    for margin_type, group in margin_groups:
                        stats_data.append({
                            'Margin Type': margin_type.capitalize(),
                            'Average Rate (%)': group['funding_rate_pct'].mean(),
                            'Min Rate (%)': group['funding_rate_pct'].min(),
                            'Max Rate (%)': group['funding_rate_pct'].max(),
                            'Std Dev (%)': group['funding_rate_pct'].std(),
                            'Exchanges': len(group)
                        })
                    
                    # Add overall row if we have multiple margin types
                    if len(stats_data) > 1:
                        stats_data.append({
                            'Margin Type': 'Overall',
                            'Average Rate (%)': filtered_df['funding_rate_pct'].mean(),
                            'Min Rate (%)': filtered_df['funding_rate_pct'].min(),
                            'Max Rate (%)': filtered_df['funding_rate_pct'].max(),
                            'Std Dev (%)': filtered_df['funding_rate_pct'].std(),
                            'Exchanges': len(filtered_df)
                        })
                    
                    # Create DataFrame
                    stats_df = pd.DataFrame(stats_data)
                    
                    # Display statistics
                    st.markdown("### Funding Rate Statistics")
                    
                    # Format values
                    format_dict = {
                        'Average Rate (%)': lambda x: f"{x:.3f}%",
                        'Min Rate (%)': lambda x: f"{x:.3f}%",
                        'Max Rate (%)': lambda x: f"{x:.3f}%",
                        'Std Dev (%)': lambda x: f"{x:.3f}%"
                    }
                    
                    create_formatted_table(stats_df, format_dict=format_dict)
                else:
                    # Simple statistics without margin type grouping
                    st.markdown("### Funding Rate Statistics")
                    
                    stats_data = [{
                        'Statistic': 'Average Rate',
                        'Value': f"{filtered_df['funding_rate_pct'].mean():.3f}%"
                    }, {
                        'Statistic': 'Minimum Rate',
                        'Value': f"{filtered_df['funding_rate_pct'].min():.3f}%"
                    }, {
                        'Statistic': 'Maximum Rate',
                        'Value': f"{filtered_df['funding_rate_pct'].max():.3f}%"
                    }, {
                        'Statistic': 'Standard Deviation',
                        'Value': f"{filtered_df['funding_rate_pct'].std():.3f}%"
                    }, {
                        'Statistic': 'Number of Exchanges',
                        'Value': str(len(filtered_df))
                    }]
                    
                    stats_df = pd.DataFrame(stats_data)
                    create_formatted_table(stats_df)
            
            # Add cross-asset funding rate comparison if we have data for multiple assets
            st.subheader("Cross-Asset Funding Rate Comparison")
            try:
                # Try to load funding rate data for other assets
                asset_funding_data = {}
                for compare_asset in ['BTC', 'ETH', 'SOL', 'XRP']:
                    if compare_asset == asset:
                        # Already have this asset's data
                        avg_rate = filtered_df['funding_rate_pct'].mean()
                        asset_funding_data[compare_asset] = avg_rate
                    else:
                        # Try to find funding rate data for this asset
                        for key in data.get('funding_rate', {}):
                            if 'exchange_list' in key.lower():
                                temp_df = data['funding_rate'][key]
                                if not temp_df.empty:
                                    compare_df = normalize_funding_rate_data(temp_df, compare_asset)
                                    if not compare_df.empty and 'funding_rate' in compare_df.columns:
                                        avg_rate = pd.to_numeric(compare_df['funding_rate'], errors='coerce').mean() * 100
                                        asset_funding_data[compare_asset] = avg_rate
                                        break
                
                # Create comparison chart if we have data for multiple assets
                if len(asset_funding_data) > 1:
                    comp_df = pd.DataFrame({
                        'Asset': list(asset_funding_data.keys()),
                        'Average Funding Rate (%)': list(asset_funding_data.values())
                    })
                    
                    comp_fig = px.bar(
                        comp_df,
                        x='Asset',
                        y='Average Funding Rate (%)',
                        title="Average Funding Rate Comparison Across Assets",
                        color='Asset',
                        color_discrete_map={
                            'BTC': ASSET_COLORS.get('BTC', '#F7931A'),
                            'ETH': ASSET_COLORS.get('ETH', '#627EEA'),
                            'SOL': ASSET_COLORS.get('SOL', '#00FFA3'),
                            'XRP': ASSET_COLORS.get('XRP', '#23292F')
                        },
                        text='Average Funding Rate (%)'
                    )
                    
                    # Format text labels
                    comp_fig.update_traces(
                        texttemplate='%{text:.3f}%',
                        textposition='outside'
                    )
                    
                    display_chart(apply_chart_theme(comp_fig))
                    
                    # Add brief interpretation
                    max_asset = comp_df.loc[comp_df['Average Funding Rate (%)'].idxmax()]['Asset']
                    min_asset = comp_df.loc[comp_df['Average Funding Rate (%)'].idxmin()]['Asset']
                    st.markdown(f"**Funding Rate Analysis:** {max_asset} currently has the highest average funding rate, indicating stronger bullish sentiment, while {min_asset} has the lowest, suggesting relatively less bullish or more bearish sentiment.")
                else:
                    st.info("Insufficient data to compare funding rates across assets.")
            except Exception as e:
                logger.error(f"Error creating cross-asset funding rate comparison: {e}")
                st.info("Could not create cross-asset funding rate comparison.")
        else:
            st.info(f"No {margin_filter} margin funding rate data available for {asset}.")
    else:
        exchange_funding_fallback_suggestions = [
            "Check that exchange funding rate data files exist in the funding_rate directory",
            "Ensure exchange list data contains 'funding_rate' and 'exchange_name' columns",
            "Verify that the data has been properly normalized from nested structures"
        ]
        display_chart(create_fallback_chart(
            f"{asset} Current Funding Rates by Exchange", 
            "No exchange funding rate data available",
            suggestions=exchange_funding_fallback_suggestions
        ))
    
    # ==============================
    # SECTION 6: Exchange Liquidity Comparison
    # ==============================
    st.header("Exchange Liquidity Snapshot")
    
    # Get exchange data from market and open interest
    exchange_data = []
    
    # First collect market data
    market_exchanges = set()
    for key in data.get('market', {}):
        if 'pairs_markets' in key.lower() and asset.lower() in key.lower():
            market_df = data['market'][key]
            if not market_df.empty and 'exchange_name' in market_df.columns:
                for idx, row in market_df.iterrows():
                    exchange_name = row['exchange_name']
                    market_exchanges.add(exchange_name)
                    
                    exchange_dict = {
                        'exchange_name': exchange_name,
                        'volume_usd': row['volume_usd'] if 'volume_usd' in row else None,
                    }
                    
                    # Add additional metrics if available
                    if 'open_interest_usd' in row:
                        exchange_dict['open_interest_usd'] = row['open_interest_usd']
                    
                    if 'funding_rate' in row:
                        exchange_dict['funding_rate'] = row['funding_rate']
                    
                    exchange_data.append(exchange_dict)
    
    # Then add open interest data
    if oi_exchange_df is not None and not oi_exchange_df.empty:
        oi_col = None
        for possible_col in ['open_interest_usd', 'open_interest']:
            if possible_col in oi_exchange_df.columns:
                oi_col = possible_col
                break
                
        if oi_col and 'exchange_name' in oi_exchange_df.columns:
            for idx, row in oi_exchange_df.iterrows():
                exchange_name = row['exchange_name']
                
                # Check if exchange already exists
                existing = [item for item in exchange_data if item['exchange_name'] == exchange_name]
                
                if existing:
                    # Update existing entry
                    existing[0]['open_interest_usd'] = row[oi_col]
                else:
                    # Create new entry
                    exchange_dict = {
                        'exchange_name': exchange_name,
                        'open_interest_usd': row[oi_col],
                        'volume_usd': None
                    }
                    exchange_data.append(exchange_dict)
    
    # Finally add funding rate data
    if funding_exchange_df is not None and not funding_exchange_df.empty:
        for idx, row in funding_exchange_df.iterrows():
            exchange_name = row['exchange_name']
            
            # Check if exchange already exists
            existing = [item for item in exchange_data if item['exchange_name'] == exchange_name]
            
            if existing:
                # Update existing entry
                existing[0]['funding_rate'] = row['funding_rate']
            else:
                # Create new entry
                exchange_dict = {
                    'exchange_name': exchange_name,
                    'funding_rate': row['funding_rate'],
                    'volume_usd': None,
                    'open_interest_usd': None
                }
                exchange_data.append(exchange_dict)
    
    # Create DataFrame and fill missing values
    if exchange_data:
        try:
            exchange_df = pd.DataFrame(exchange_data)
            
            # Merge rows with the same exchange_name by summing numeric columns
            if not exchange_df.empty:
                exchange_df = exchange_df.groupby('exchange_name').agg({
                    'volume_usd': 'sum',
                    'open_interest_usd': 'sum',
                    'funding_rate': 'mean'
                }).reset_index()
                
                # Add market share columns
                if 'volume_usd' in exchange_df.columns:
                    total_volume = exchange_df['volume_usd'].sum(skipna=True)
                    if total_volume > 0:
                        exchange_df['volume_share'] = exchange_df['volume_usd'] / total_volume * 100
                
                if 'open_interest_usd' in exchange_df.columns:
                    total_oi = exchange_df['open_interest_usd'].sum(skipna=True)
                    if total_oi > 0:
                        exchange_df['oi_share'] = exchange_df['open_interest_usd'] / total_oi * 100
                
                # Sort by volume
                if 'volume_usd' in exchange_df.columns:
                    exchange_df = exchange_df.sort_values('volume_usd', ascending=False, na_position='last')
                elif 'open_interest_usd' in exchange_df.columns:
                    exchange_df = exchange_df.sort_values('open_interest_usd', ascending=False, na_position='last')
                
                # Display exchange comparison table
                st.subheader(f"{asset} Exchange Liquidity Comparison")
                
                # Choose which columns to display and format
                display_columns = ['exchange_name']
                format_dict = {}
                
                if 'volume_usd' in exchange_df.columns:
                    display_columns.extend(['volume_usd', 'volume_share'])
                    format_dict['volume_usd'] = lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else "N/A"
                    format_dict['volume_share'] = lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
                
                if 'open_interest_usd' in exchange_df.columns:
                    display_columns.extend(['open_interest_usd', 'oi_share'])
                    format_dict['open_interest_usd'] = lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else "N/A"
                    format_dict['oi_share'] = lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
                
                if 'funding_rate' in exchange_df.columns:
                    display_columns.append('funding_rate')
                    format_dict['funding_rate'] = lambda x: f"{x*100:.3f}%" if pd.notna(x) else "N/A"
                
                # Rename columns for display
                rename_dict = {
                    'exchange_name': 'Exchange',
                    'volume_usd': 'Volume (24h)',
                    'volume_share': 'Volume Share',
                    'open_interest_usd': 'Open Interest',
                    'oi_share': 'OI Share',
                    'funding_rate': 'Funding Rate'
                }
                
                display_df = exchange_df[display_columns].copy()
                display_df = display_df.rename(columns=rename_dict)
                
                # Update format dict keys
                format_dict = {rename_dict.get(k, k): v for k, v in format_dict.items()}
                
                # Create and display the table
                create_formatted_table(display_df.head(15), format_dict=format_dict)
                
                # Create market share visualization
                st.subheader("Market Share Analysis")
                
                # Create two columns for market share charts
                col1, col2 = st.columns(2)
                
                with col1:
                    # Trading volume market share
                    if 'volume_share' in exchange_df.columns:
                        top_volume_exchanges = exchange_df.sort_values('volume_usd', ascending=False).head(8)
                        
                        if not top_volume_exchanges.empty:
                            fig = px.pie(
                                top_volume_exchanges,
                                values='volume_usd',
                                names='exchange_name',
                                title=f"{asset} Trading Volume Market Share",
                                color_discrete_map=EXCHANGE_COLORS
                            )
                            
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            display_chart(apply_chart_theme(fig))
                        else:
                            display_chart(create_fallback_chart(f"{asset} Trading Volume Market Share", "No volume market share data available"))
                    else:
                        display_chart(create_fallback_chart(f"{asset} Trading Volume Market Share", "No volume market share data available"))
                
                with col2:
                    # Open interest market share
                    if 'oi_share' in exchange_df.columns:
                        top_oi_exchanges = exchange_df.sort_values('open_interest_usd', ascending=False).head(8)
                        
                        if not top_oi_exchanges.empty:
                            fig = px.pie(
                                top_oi_exchanges,
                                values='open_interest_usd',
                                names='exchange_name',
                                title=f"{asset} Open Interest Market Share",
                                color_discrete_map=EXCHANGE_COLORS
                            )
                            
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                            display_chart(apply_chart_theme(fig))
                        else:
                            display_chart(create_fallback_chart(f"{asset} Open Interest Market Share", "No open interest market share data available"))
                    else:
                        display_chart(create_fallback_chart(f"{asset} Open Interest Market Share", "No open interest market share data available"))
            else:
                st.info(f"No exchange comparison data available for {asset}.")
        except Exception as e:
            st.warning(f"Error processing exchange data: {e}")
            logger.error(f"Error processing exchange data: {e}\n{traceback.format_exc()}")
            st.info(f"No exchange comparison data available for {asset}.")
    else:
        st.info(f"No exchange comparison data available for {asset}.")
    
    # ==============================
    # Key Market Insights
    # ==============================
    st.header("Key Market Insights")
    
    # Generate insights based on available data
    insights = []
    
    # Volume insights
    if volume_24h is not None:
        insights.append(f"**Trading Volume:** {asset} 24-hour trading volume is {format_currency(volume_24h, abbreviate=True)}.")
    
    # Open interest insights
    if open_interest_total is not None:
        insights.append(f"**Open Interest:** Total {asset} open interest is {format_currency(open_interest_total, abbreviate=True)}.")
    
    # Price data
    latest_price = None
    if price_history_df is not None and not price_history_df.empty and 'close' in price_history_df.columns:
        latest_price = price_history_df['close'].iloc[-1] if len(price_history_df) > 0 else None
        if latest_price is not None:
            insights.append(f"**Current Price:** {asset} is trading at {format_currency(latest_price)}.")
    
    # OI to Price ratio
    if latest_price is not None and open_interest_total is not None:
        oi_price_ratio = open_interest_total / (latest_price * 1000)  # Scale for better readability
        insights.append(f"**OI/Price Ratio:** The ratio of open interest to price is {oi_price_ratio:.2f}.")
    
    # Funding rate insights
    if current_rate is not None:
        sentiment = "bullish" if current_rate > 0 else "bearish"
        intensity = "strongly" if abs(current_rate) > 0.05 else "moderately" if abs(current_rate) > 0.01 else "slightly"
        insights.append(f"**Funding Rate:** Current rate of {current_rate:.3f}% indicates {intensity} {sentiment} sentiment.")
        
        # Annualized cost
        annual_cost = current_rate * 3 * 365  # Assuming funding occurs 3 times per day
        insights.append(f"**Annualized Cost:** Holding a leveraged position costs approximately {annual_cost:.2f}% per year at current rates.")
    
    # Spread insights
    if spread_pct is not None:
        liquidity_desc = "highly liquid" if spread_pct < 0.02 else "moderately liquid" if spread_pct < 0.05 else "less liquid"
        insights.append(f"**Market Liquidity:** Bid-ask spread of {spread_pct:.3f}% indicates a {liquidity_desc} market.")
    
    # Display insights
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("Insufficient data available to generate market insights.")
    

def _test_asset_data_loading():
    """
    Internal function to test data loading for different assets.
    This is a diagnostic function that can be run to verify all assets work correctly.
    """
    assets_to_test = ['BTC', 'ETH', 'SOL', 'XRP']
    results = {}
    
    for test_asset in assets_to_test:
        # Store the current session state
        st.session_state.selected_asset = test_asset
        
        # Get the data for this asset
        data = load_report_data()
        
        # Test all data categories
        volume_data = extract_asset_volume_data(data, [test_asset])
        has_volume = bool(volume_data)
        
        # Test funding rate data
        funding_history_df = None
        has_funding_history = False
        
        # Try to find funding rate data
        for key in data.get('funding_rate', {}):
            if 'ohlc_history' in key.lower() or ('history' in key.lower() and test_asset.lower() in key.lower()):
                temp_df = data['funding_rate'][key]
                if not temp_df.empty:
                    funding_history_df = process_timestamps(temp_df)
                    has_funding_history = not funding_history_df.empty
                    break
        
        # Test exchange funding data
        funding_exchange_df = None
        has_exchange_funding = False
        
        # Try to find exchange funding rate data
        for key in data.get('funding_rate', {}):
            if 'exchange_list' in key.lower():
                fr_df = data['funding_rate'][key]
                if not fr_df.empty:
                    funding_exchange_df = normalize_funding_rate_data(fr_df, test_asset)
                    has_exchange_funding = not funding_exchange_df.empty
                    break
        
        # Store results
        results[test_asset] = {
            'volume': has_volume,
            'funding_history': has_funding_history,
            'exchange_funding': has_exchange_funding
        }
    
    return results

def _display_test_results(test_results):
    """
    Display test results in a nicely formatted table.
    """
    st.header("Asset Data Loading Test Results")
    
    # Create DataFrame from results
    results_df = pd.DataFrame(test_results).T.reset_index()
    results_df.columns = ['Asset', 'Volume Data', 'Funding Rate History', 'Exchange Funding Rates']
    
    # Display results
    st.dataframe(results_df)
    
    # Show overall summary
    all_assets_volume = all(test_results[asset]['volume'] for asset in test_results)
    all_assets_funding = all(test_results[asset]['funding_history'] for asset in test_results)
    all_assets_exchange = all(test_results[asset]['exchange_funding'] for asset in test_results)
    
    if all_assets_volume and all_assets_funding and all_assets_exchange:
        st.success("All data categories available for all assets!")
    else:
        st.warning("Some data categories are missing for certain assets.")
        
        # Provide detailed guidance for missing data
        missing_data = []
        for asset in test_results:
            for category, has_data in test_results[asset].items():
                if not has_data:
                    missing_data.append(f"{asset} is missing {category} data")
        
        if missing_data:
            st.subheader("Missing Data Details")
            for item in missing_data:
                st.markdown(f"- {item}")
            
            st.subheader("Troubleshooting Steps")
            st.markdown("""
            1. **Volume Data**: Check `futures/taker_buy_sell/api_futures_taker_buy_sell_volume_history_[ASSET].parquet`
            2. **Funding Rate History**: Check `futures/funding_rate/api_futures_fundingRate_oi_weight_ohlc_history_[ASSET].parquet`
            3. **Exchange Funding Rates**: Check `futures/funding_rate/api_futures_fundingRate_exchange_list.parquet`
            
            Ensure files exist and have the expected data structure. If a file is missing or empty, it will need to be generated from the API.
            """)

if __name__ == "__main__":
    # Check if we're in test mode
    if 'test_mode' in st.query_params and st.query_params['test_mode'] == 'true':
        test_results = _test_asset_data_loading()
        _display_test_results(test_results)
    else:
        main()