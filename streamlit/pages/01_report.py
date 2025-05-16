"""
Report page for the Izun Crypto Liquidity Report.

This page serves as the main dashboard/overview page focused on liquidity metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from utils.config import APP_TITLE, APP_ICON, ASSET_COLORS, DEFAULT_ASSET, EXCHANGE_COLORS, SUPPORTED_ASSETS

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

def load_report_data(assets=None):
    """
    Load data for the report dashboard for one or more assets.
    
    Parameters:
    -----------
    assets : list, optional
        List of assets to load data for. If None, loads data for the currently
        selected asset from session state.
    
    Returns:
    --------
    dict
        Dictionary containing all data needed for the dashboard, with nested
        dictionaries for each asset
    """
    # Initialize master data dictionary
    all_data = {}
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return all_data
    
    # Determine which assets to load
    if assets is None:
        # Get selected asset from session state or use default
        single_asset = st.session_state.get('selected_asset', DEFAULT_ASSET)
        assets_to_load = [single_asset]
    else:
        assets_to_load = assets
    
    # Load shared data (not asset-specific)
    all_data['shared'] = {
        'price': load_data_for_category('futures', 'market'),
        'long_short': load_data_for_category('futures', 'long_short_ratio')
    }
    
    # Load asset-specific data for each asset
    for asset in assets_to_load:
        asset_data = {}
        
        # Load trading volume data
        asset_data['taker_volume'] = load_data_for_category('futures', 'taker_buy_sell', asset)
        
        # Load open interest data
        asset_data['open_interest'] = load_data_for_category('futures', 'open_interest', asset)
        
        # Load order book data for bid-ask spread and depth
        asset_data['order_book'] = load_data_for_category('futures', 'order_book', asset)
        
        # Load funding rate data
        asset_data['funding_rate'] = load_data_for_category('futures', 'funding_rate', asset)
        
        # Load market data for exchange comparisons
        asset_data['market'] = load_data_for_category('futures', 'market', asset)
        
        # Load liquidation data
        asset_data['liquidation'] = load_data_for_category('futures', 'liquidation', asset)
        
        # Store asset data in the main dictionary
        all_data[asset] = asset_data
    
    return all_data

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
        xaxis=dict(visible=False, type="linear"),
        yaxis=dict(visible=False, type="linear")
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
        for key in data.get(asset_name, {}).get('taker_volume', {}):
            if 'history' in key.lower() and asset_name.lower() in key.lower():
                asset_vol_df = data[asset_name]['taker_volume'][key]
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
        for key in data.get(asset_name, {}).get('open_interest', {}):
            if 'exchange_list' in key.lower() and asset_name.lower() in key.lower():
                asset_oi_df = data[asset_name]['open_interest'][key]
                if not asset_oi_df.empty and any(col in asset_oi_df.columns for col in ['open_interest_usd', 'open_interest']):
                    try:
                        # Determine which column to use
                        oi_col = 'open_interest_usd' if 'open_interest_usd' in asset_oi_df.columns else 'open_interest' 
                        assets_oi[asset_name] = asset_oi_df[oi_col].sum()
                    except Exception as e:
                        logger.error(f"Error processing OI data for {asset_name}: {e}")
                break
    
    return assets_oi

def create_cross_asset_overview(all_data):
    """
    Create a cross-asset overview dashboard displaying comparative metrics.
    
    Parameters:
    -----------
    all_data : dict
        Dictionary containing data for all assets
    """
    # Extract price data
    price_data = {}
    latest_prices = {}
    
    price_history_df = None
    for key in all_data.get('shared', {}).get('price', {}):
        if 'ohlc_history' in key.lower():
            price_history_df = all_data['shared']['price'][key]
            # Process timestamps once
            if not price_history_df.empty:
                price_history_df = process_timestamps(price_history_df)
                break
    
    if price_history_df is not None and not price_history_df.empty:
        # Get latest prices for each asset if available
        for asset in SUPPORTED_ASSETS:
            try:
                # Get latest market price data from pairs
                for key in all_data.get(asset, {}).get('market', {}):
                    if 'pairs_markets' in key.lower() and asset.lower() in key.lower():
                        pairs_df = all_data[asset]['market'][key]
                        if not pairs_df.empty and 'price_usd' in pairs_df.columns:
                            # Calculate weighted average price
                            price_col = 'price_usd'
                            volume_col = None
                            for vol_col in ['volume_usd', 'volume_24h_usd', 'volume']:
                                if vol_col in pairs_df.columns:
                                    volume_col = vol_col
                                    break
                                    
                            if volume_col:
                                # Use volume-weighted average
                                valid_pairs = pairs_df[(pd.notna(pairs_df[price_col])) & (pd.notna(pairs_df[volume_col]))]
                                if not valid_pairs.empty:
                                    valid_pairs = valid_pairs[valid_pairs[volume_col] > 0]
                                    if not valid_pairs.empty:
                                        weighted_price = (valid_pairs[price_col] * valid_pairs[volume_col]).sum() / valid_pairs[volume_col].sum()
                                        latest_prices[asset] = weighted_price
                                        break
                            
                            # Fallback to simple average if no volume data
                            if asset not in latest_prices:
                                avg_price = pairs_df[price_col].mean()
                                latest_prices[asset] = avg_price
                                break
            except Exception as e:
                logger.error(f"Error getting latest price for {asset}: {e}")
    
    # Get trading volume data for all assets
    volume_data = {}
    for asset in SUPPORTED_ASSETS:
        try:
            if asset in all_data:
                asset_volumes = extract_asset_volume_data(all_data, [asset])
                if asset in asset_volumes:
                    volume_data[asset] = asset_volumes[asset]
        except Exception as e:
            logger.error(f"Error extracting volume data for {asset}: {e}")
    
    # Get open interest data for all assets
    oi_data = {}
    for asset in SUPPORTED_ASSETS:
        try:
            if asset in all_data:
                asset_oi = extract_asset_oi_data(all_data, [asset])
                if asset in asset_oi:
                    oi_data[asset] = asset_oi[asset]
        except Exception as e:
            logger.error(f"Error extracting OI data for {asset}: {e}")
    
    # Get funding rate data for all assets
    funding_data = {}
    for asset in SUPPORTED_ASSETS:
        try:
            if asset in all_data:
                # Find funding rate data in asset data
                for key in all_data.get(asset, {}).get('funding_rate', {}):
                    if 'exchange_list' in key.lower():
                        funding_df = all_data[asset]['funding_rate'][key]
                        if not funding_df.empty:
                            # Normalize the funding rate data
                            normalized_df = normalize_funding_rate_data(funding_df, asset)
                            if not normalized_df.empty and 'funding_rate' in normalized_df.columns:
                                # Calculate average funding rate
                                avg_funding_rate = normalized_df['funding_rate'].mean()
                                funding_data[asset] = avg_funding_rate * 100  # Convert to percentage
                                break
        except Exception as e:
            logger.error(f"Error extracting funding rate data for {asset}: {e}")
    
    # Create a comparative price chart
    st.subheader("Price Comparison")
    
    if price_history_df is not None and not price_history_df.empty:
        try:
            # Create price comparison chart
            fig = go.Figure()
            
            for asset in SUPPORTED_ASSETS:
                if asset in latest_prices:
                    fig.add_trace(go.Scatter(
                        x=price_history_df['datetime'],
                        y=price_history_df['close'],
                        name=f"{asset}",
                        line=dict(color=ASSET_COLORS.get(asset, '#3366CC'), width=2)
                    ))
            
            # Update layout
            fig.update_layout(
                title="Asset Price Comparison",
                xaxis_title=None,
                yaxis_title="Price (USD)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            # Format y-axis and ensure linear scale
            fig.update_yaxes(tickformat="$.2f", type="linear")
            fig.update_xaxes(type="linear")
            
            display_chart(apply_chart_theme(fig))
            
            # Show latest prices in a metrics row
            if latest_prices:
                metrics = {}
                formatters = {}
                
                for asset in SUPPORTED_ASSETS:
                    if asset in latest_prices:
                        metrics[f"{asset} Price"] = {
                            "value": latest_prices[asset],
                            "delta": None
                        }
                        formatters[f"{asset} Price"] = lambda x: format_currency(x, precision=2)
                
                if metrics:
                    display_metrics_row(metrics, formatters)
        except Exception as e:
            logger.error(f"Error creating price comparison chart: {e}")
            st.warning("Could not create price comparison chart.")
    
    # Create trading volume comparison
    st.subheader("Trading Volume Comparison")
    
    # Extract volume data for bar chart
    volume_comparison = []
    for asset in SUPPORTED_ASSETS:
        if asset in volume_data:
            volume_comparison.append({
                'Asset': asset,
                'Volume (USD)': volume_data[asset]
            })
    
    if volume_comparison:
        volume_df = pd.DataFrame(volume_comparison)
        
        # Create bar chart
        fig = px.bar(
            volume_df,
            x='Asset',
            y='Volume (USD)',
            title="24h Trading Volume by Asset",
            color='Asset',
            color_discrete_map=ASSET_COLORS,
            text='Volume (USD)'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Volume (USD)"
        )
        
        # Format text labels and y-axis
        fig.update_traces(
            texttemplate='%{y:$.2s}',
            textposition='outside'
        )
        fig.update_yaxes(tickformat="$.2s", type="linear")
        fig.update_xaxes(type="linear")
        
        display_chart(apply_chart_theme(fig))
        
        # Create a table with exact values
        volume_df = volume_df.sort_values('Volume (USD)', ascending=False)
        create_formatted_table(volume_df, format_dict={
            'Volume (USD)': lambda x: format_currency(x, abbreviate=True)
        })
    else:
        st.info("No trading volume data available for comparison.")
    
    # Create open interest comparison
    st.subheader("Open Interest Comparison")
    
    # Extract OI data for bar chart
    oi_comparison = []
    for asset in SUPPORTED_ASSETS:
        if asset in oi_data:
            oi_comparison.append({
                'Asset': asset,
                'Open Interest (USD)': oi_data[asset]
            })
    
    if oi_comparison:
        oi_df = pd.DataFrame(oi_comparison)
        
        # Create bar chart
        fig = px.bar(
            oi_df,
            x='Asset',
            y='Open Interest (USD)',
            title="Open Interest by Asset",
            color='Asset',
            color_discrete_map=ASSET_COLORS,
            text='Open Interest (USD)'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Open Interest (USD)"
        )
        
        # Format text labels and y-axis
        fig.update_traces(
            texttemplate='%{y:$.2s}',
            textposition='outside'
        )
        fig.update_yaxes(tickformat="$.2s", type="linear")
        fig.update_xaxes(type="linear")
        
        display_chart(apply_chart_theme(fig))
        
        # Create a table with exact values
        oi_df = oi_df.sort_values('Open Interest (USD)', ascending=False)
        create_formatted_table(oi_df, format_dict={
            'Open Interest (USD)': lambda x: format_currency(x, abbreviate=True)
        })
    else:
        st.info("No open interest data available for comparison.")
    
    # Create funding rate comparison
    st.subheader("Funding Rate Comparison")
    
    # Extract funding rate data for bar chart
    funding_comparison = []
    for asset in SUPPORTED_ASSETS:
        if asset in funding_data:
            funding_comparison.append({
                'Asset': asset,
                'Funding Rate (%)': funding_data[asset]
            })
    
    if funding_comparison:
        funding_df = pd.DataFrame(funding_comparison)
        
        # Create bar chart with color based on value
        fig = px.bar(
            funding_df,
            x='Asset',
            y='Funding Rate (%)',
            title="Current Funding Rates by Asset",
            color='Funding Rate (%)',
            color_continuous_scale='RdBu_r',
            color_continuous_midpoint=0,
            text='Funding Rate (%)'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Funding Rate (%)"
        )
        
        # Format text labels
        fig.update_traces(
            texttemplate='%{y:.3f}%',
            textposition='outside'
        )
        
        # Ensure linear scale on both axes
        fig.update_yaxes(type="linear")
        fig.update_xaxes(type="linear")
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        display_chart(apply_chart_theme(fig))
        
        # Create a table with exact values and sentiment indicators
        funding_df = funding_df.sort_values('Funding Rate (%)', ascending=False)
        
        # Add sentiment column
        funding_df['Sentiment'] = funding_df['Funding Rate (%)'].apply(
            lambda x: "Strongly Bullish" if x > 0.05 else 
                      "Moderately Bullish" if x > 0.01 else
                      "Neutral" if x >= -0.01 else
                      "Moderately Bearish" if x >= -0.05 else
                      "Strongly Bearish"
        )
        
        create_formatted_table(funding_df, format_dict={
            'Funding Rate (%)': lambda x: f"{x:.3f}%"
        })
    else:
        st.info("No funding rate data available for comparison.")
    
    # Add a combined market health indicator
    st.subheader("Market Health Overview")
    
    # Create metrics for market health
    metrics = {}
    
    # Total ecosystem trading volume
    total_volume = sum(volume_data.get(asset, 0) for asset in SUPPORTED_ASSETS)
    if total_volume > 0:
        metrics["Total Ecosystem Volume"] = {
            "value": total_volume,
            "delta": None
        }
    
    # Total ecosystem open interest
    total_oi = sum(oi_data.get(asset, 0) for asset in SUPPORTED_ASSETS)
    if total_oi > 0:
        metrics["Total Ecosystem OI"] = {
            "value": total_oi,
            "delta": None
        }
    
    # Average funding rate
    valid_funding_rates = [funding_data[asset] for asset in SUPPORTED_ASSETS if asset in funding_data]
    if valid_funding_rates:
        avg_funding = sum(valid_funding_rates) / len(valid_funding_rates)
        sentiment = "bullish" if avg_funding > 0 else "bearish" if avg_funding < 0 else "neutral"
        metrics["Avg Funding Rate"] = {
            "value": avg_funding,
            "delta": sentiment,
            "delta_suffix": ""
        }
    
    # Display metrics
    formatters = {
        "Total Ecosystem Volume": lambda x: format_currency(x, abbreviate=True),
        "Total Ecosystem OI": lambda x: format_currency(x, abbreviate=True),
        "Avg Funding Rate": lambda x: f"{x:.3f}%"
    }
    
    if metrics:
        display_metrics_row(metrics, formatters)
        
        # Add insights based on data
        st.markdown("### Market Insights")
        
        # Generate insights based on the comparative data
        insights = []
        
        # Volume distribution insight
        if volume_comparison:
            volume_df = pd.DataFrame(volume_comparison)
            if not volume_df.empty:
                highest_vol_asset = volume_df.loc[volume_df['Volume (USD)'].idxmax()]['Asset']
                highest_vol = volume_df.loc[volume_df['Volume (USD)'].idxmax()]['Volume (USD)']
                vol_dominance = (highest_vol / total_volume * 100) if total_volume > 0 else 0
                insights.append(f"**Volume Distribution:** {highest_vol_asset} dominates with {vol_dominance:.1f}% of trading volume.")
        
        # OI distribution insight
        if oi_comparison:
            oi_df = pd.DataFrame(oi_comparison)
            if not oi_df.empty:
                highest_oi_asset = oi_df.loc[oi_df['Open Interest (USD)'].idxmax()]['Asset']
                highest_oi = oi_df.loc[oi_df['Open Interest (USD)'].idxmax()]['Open Interest (USD)']
                oi_dominance = (highest_oi / total_oi * 100) if total_oi > 0 else 0
                insights.append(f"**Open Interest:** {highest_oi_asset} leads with {oi_dominance:.1f}% of total open interest.")
        
        # Funding rate insight
        if valid_funding_rates:
            max_funding_asset = None
            max_funding = -9999
            min_funding_asset = None
            min_funding = 9999
            
            for asset in SUPPORTED_ASSETS:
                if asset in funding_data:
                    if funding_data[asset] > max_funding:
                        max_funding = funding_data[asset]
                        max_funding_asset = asset
                    if funding_data[asset] < min_funding:
                        min_funding = funding_data[asset]
                        min_funding_asset = asset
            
            if max_funding_asset and min_funding_asset:
                funding_spread = max_funding - min_funding
                insights.append(f"**Funding Rate Spread:** {max_funding_asset} has the most bullish funding at {max_funding:.3f}%, while {min_funding_asset} has the most bearish at {min_funding:.3f}%, representing a spread of {funding_spread:.3f}%.")
        
        # Display insights
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.info("Insufficient data to generate cross-asset market insights.")
    else:
        st.info("Insufficient data to calculate market health metrics.")


def display_asset_section(data, asset):
    """
    Display a comprehensive analysis section for a specific asset.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing data for the asset
    asset : str
        Asset symbol to analyze
    """
    # ==============================
    # SECTION 1: Core Liquidity Metrics
    # ==============================
    st.subheader("Core Liquidity Metrics")
    
    # Calculate key metrics for the header
    metrics = {}
    
    # Total trading volume (24h)
    volume_24h = None
    volume_history_df = None
    
    # Find volume data
    for key in data.get(asset, {}).get('taker_volume', {}):
        if 'history' in key.lower() and asset.lower() in key.lower():
            try:
                temp_df = data[asset]['taker_volume'][key]
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
    for key in data.get(asset, {}).get('open_interest', {}):
        if 'exchange_list' in key.lower() and asset.lower() in key.lower():
            try:
                temp_df = data[asset]['open_interest'][key]
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
    for key in data.get(asset, {}).get('order_book', {}):
        if 'ask_bids_history' in key.lower() and asset.lower() in key.lower():
            try:
                ob_df = data[asset]['order_book'][key]
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
        for key in data.get(asset, {}).get('order_book', {}):
            if 'ask_bids_history' in key.lower() and 'aggregated' not in key.lower():
                try:
                    ob_df = data[asset]['order_book'][key]
                    if not ob_df.empty and all(col in ob_df.columns for col in ['bids_usd', 'asks_usd']):
                        spreads_df = calculate_spreads_and_depth(ob_df, asset)
                        if not spreads_df.empty and 'spread_pct' in spreads_df.columns:
                            spread_pct = spreads_df['spread_pct'].iloc[-1] if len(spreads_df) > 0 else None
                            total_depth = spreads_df['total_depth'].iloc[-1] if len(spreads_df) > 0 and 'total_depth' in spreads_df.columns else None
                            break
                except Exception as e:
                    logger.error(f"Error processing generic orderbook data: {e}")
    
    # Get current funding rate
    current_rate = None
    funding_exchange_df = None
    
    # Find funding rate data
    for key in data.get(asset, {}).get('funding_rate', {}):
        if 'exchange_list' in key.lower():
            try:
                fr_df = data[asset]['funding_rate'][key]
                if not fr_df.empty:
                    # Process and normalize funding rate data
                    funding_exchange_df = normalize_funding_rate_data(fr_df, asset)
                    if not funding_exchange_df.empty and 'funding_rate' in funding_exchange_df.columns:
                        current_rate = funding_exchange_df['funding_rate'].mean() * 100  # Convert to percentage
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
    
    if current_rate is not None:
        metrics["Current Funding Rate"] = {
            "value": current_rate,
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
                
                # Ensure linear scale on both axes
                fig.update_yaxes(type="linear")
                fig.update_xaxes(type="linear")
                
                display_chart(apply_chart_theme(fig))
            
            with col2:
                # Get volume by exchange
                volume_by_exchange = None
                for key in data.get(asset, {}).get('market', {}):
                    if 'pairs_markets' in key.lower() and asset.lower() in key.lower():
                        market_df = data[asset]['market'][key]
                        if not market_df.empty and 'volume_usd' in market_df.columns and 'exchange_name' in market_df.columns:
                            volume_by_exchange = market_df[['exchange_name', 'volume_usd']].sort_values('volume_usd', ascending=False)
                            break
                
                if volume_by_exchange is not None and not volume_by_exchange.empty:
                    # Create pie chart for volume distribution
                    fig = px.pie(
                        volume_by_exchange.head(8),  # Top 8 exchanges
                        values='volume_usd',
                        names='exchange_name',
                        title=f"{asset} Trading Volume by Exchange",
                        color_discrete_map=EXCHANGE_COLORS
                    )
                    
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    display_chart(apply_chart_theme(fig))
                else:
                    display_chart(create_fallback_chart(f"{asset} Trading Volume by Exchange", "No exchange volume data available"))
                    
            # Buy/Sell ratio analysis
            if 'taker_buy_volume_usd' in volume_history_df.columns and 'taker_sell_volume_usd' in volume_history_df.columns:
                # Calculate buy/sell ratio
                volume_history_df['buy_sell_ratio'] = volume_history_df['taker_buy_volume_usd'] / volume_history_df['taker_sell_volume_usd']
                
                # Create Buy/Sell ratio chart
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=volume_history_df['datetime'],
                    y=volume_history_df['buy_sell_ratio'],
                    name='Buy/Sell Ratio',
                    line=dict(color=ASSET_COLORS.get(asset, '#3366CC'), width=2)
                ))
                
                # Add horizontal line at 1 (equal buy/sell)
                fig.add_shape(
                    type="line",
                    x0=volume_history_df['datetime'].min(),
                    y0=1,
                    x1=volume_history_df['datetime'].max(),
                    y1=1,
                    line=dict(color="gray", width=1, dash="dash"),
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{asset} Buy/Sell Volume Ratio",
                    xaxis_title=None,
                    yaxis_title="Ratio",
                    hovermode="x unified"
                )
                
                # Ensure linear scale on both axes
                fig.update_yaxes(type="linear")
                fig.update_xaxes(type="linear")
                
                display_chart(apply_chart_theme(fig))
                
                # Add interpretation
                latest_ratio = volume_history_df['buy_sell_ratio'].iloc[-1] if not volume_history_df.empty else None
                if latest_ratio is not None:
                    sentiment = "bullish" if latest_ratio > 1 else "bearish"
                    intensity = "strongly" if abs(latest_ratio - 1) > 0.3 else "moderately" if abs(latest_ratio - 1) > 0.1 else "slightly"
                    
                    st.markdown(f"""
                    **Buy/Sell Ratio Interpretation:**
                    - Current Ratio: {latest_ratio:.2f}
                    - Market Sentiment: {intensity.capitalize()} {sentiment}
                    - {'More buying pressure than selling pressure' if latest_ratio > 1 else 'More selling pressure than buying pressure'}
                    """)
                
    else:
        display_chart(create_fallback_chart(f"{asset} Futures Trading Volume", "No trading volume data available"))
    
    # ==============================
    # SECTION 3: Open Interest Analysis
    # ==============================
    st.subheader("Open Interest Analysis")
    
    # Open Interest history
    oi_history_df = None
    for key in data.get(asset, {}).get('open_interest', {}):
        if 'aggregated_history' in key.lower() and asset.lower() in key.lower():
            temp_df = data[asset]['open_interest'][key]
            if not temp_df.empty:
                oi_history_df = process_timestamps(temp_df)
                break
    
    # If asset-specific OI history not found, try to find any OI history data
    if oi_history_df is None or oi_history_df.empty:
        for key in data.get(asset, {}).get('open_interest', {}):
            if 'history' in key.lower() and 'ohlc' in key.lower():
                temp_df = data[asset]['open_interest'][key]
                if not temp_df.empty:
                    # If we have a symbol column, filter for the asset
                    if 'symbol' in temp_df.columns:
                        temp_df = temp_df[temp_df['symbol'].str.contains(asset, case=False, na=False)]
                    
                    if not temp_df.empty:
                        oi_history_df = process_timestamps(temp_df)
                        break
    
    # Price data for overlay
    price_history_df = None
    for key in data.get('shared', {}).get('price', {}):
        if 'ohlc_history' in key.lower():
            temp_df = data['shared']['price'][key]
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
                            side="right",
                            type="linear"
                        )
                    )
                
                # Update layout
                fig.update_layout(
                    title=f"{asset} Open Interest History",
                    xaxis_title=None,
                    yaxis_title="Open Interest (USD)",
                    hovermode="x unified"
                )
                
                # Ensure linear scale on both axes
                fig.update_yaxes(type="linear")
                fig.update_xaxes(type="linear")
                
                display_chart(apply_chart_theme(fig))
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
                
                fig = px.pie(
                    oi_by_exchange.head(8),  # Top 8 exchanges
                    values=oi_col,
                    names='exchange_name',
                    title=f"{asset} Open Interest by Exchange",
                    color_discrete_map=EXCHANGE_COLORS
                )
                
                fig.update_traces(textposition='inside', textinfo='percent+label')
                display_chart(apply_chart_theme(fig))
            else:
                display_chart(create_fallback_chart(f"{asset} Open Interest by Exchange", "Incomplete exchange data"))
        else:
            display_chart(create_fallback_chart(f"{asset} Open Interest by Exchange", "No exchange open interest data available"))
    
    # ==============================
    # SECTION 4: Liquidation Analysis
    # ==============================
    st.subheader("Liquidation Analysis")
    
    # Get liquidation data
    liquidation_df = None
    for key in data.get(asset, {}).get('liquidation', {}):
        if 'history' in key.lower() and asset.lower() in key.lower():
            temp_df = data[asset]['liquidation'][key]
            if not temp_df.empty:
                liquidation_df = process_timestamps(temp_df)
                break
    
    # Check and process liquidation data
    valid_liquidation_data = False
    if liquidation_df is not None and not liquidation_df.empty:
        # Check if we have the required columns
        required_cols = ['aggregated_long_liquidation_usd', 'aggregated_short_liquidation_usd']
        if all(col in liquidation_df.columns for col in required_cols):
            valid_liquidation_data = True
            
            # Create stacked bar chart of liquidations
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=liquidation_df['datetime'],
                y=liquidation_df['aggregated_long_liquidation_usd'],
                name='Long Liquidations',
                marker_color='red'
            ))
            
            fig.add_trace(go.Bar(
                x=liquidation_df['datetime'],
                y=liquidation_df['aggregated_short_liquidation_usd'],
                name='Short Liquidations',
                marker_color='green'
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{asset} Liquidation History",
                barmode='stack',
                xaxis_title=None,
                yaxis_title="Liquidation Volume (USD)",
                hovermode="x unified"
            )
            
            # Format y-axis with linear scale
            fig.update_yaxes(tickformat="$.2s", type="linear")
            fig.update_xaxes(type="linear")
            
            display_chart(apply_chart_theme(fig))
            
            # Calculate liquidation statistics
            total_long_liq = liquidation_df['aggregated_long_liquidation_usd'].sum()
            total_short_liq = liquidation_df['aggregated_short_liquidation_usd'].sum()
            total_liq = total_long_liq + total_short_liq
            
            # Display liquidation metrics
            liq_cols = st.columns(3)
            
            with liq_cols[0]:
                st.metric("Total Liquidations", format_currency(total_liq, abbreviate=True))
                
            with liq_cols[1]:
                st.metric("Long Liquidations", format_currency(total_long_liq, abbreviate=True))
                
            with liq_cols[2]:
                st.metric("Short Liquidations", format_currency(total_short_liq, abbreviate=True))
            
            # Create liquidation ratio analysis
            liquidation_df['long_short_ratio'] = liquidation_df['aggregated_long_liquidation_usd'] / liquidation_df['aggregated_short_liquidation_usd'].replace(0, np.nan)
            
            # Add liquidation to price chart if price data available
            if price_history_df is not None and not price_history_df.empty:
                # Create a subplot with shared x-axis
                fig = make_subplots(
                    rows=2, 
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=(f"{asset} Price", "Liquidation Volume")
                )
                
                # Add price line to first subplot
                fig.add_trace(
                    go.Scatter(
                        x=price_history_df['datetime'],
                        y=price_history_df['close'],
                        name=f"{asset} Price",
                        line=dict(color=ASSET_COLORS.get(asset, '#3366CC'), width=2)
                    ),
                    row=1, col=1
                )
                
                # Add liquidation bars to second subplot
                fig.add_trace(
                    go.Bar(
                        x=liquidation_df['datetime'],
                        y=liquidation_df['aggregated_long_liquidation_usd'],
                        name='Long Liquidations',
                        marker_color='red'
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=liquidation_df['datetime'],
                        y=liquidation_df['aggregated_short_liquidation_usd'],
                        name='Short Liquidations',
                        marker_color='green'
                    ),
                    row=2, col=1
                )
                
                # Update layout
                fig.update_layout(
                    title=f"{asset} Price and Liquidations",
                    height=600,
                    hovermode="x unified",
                    barmode='stack',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                # Format axes
                fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
                fig.update_yaxes(title_text="Liquidation Volume (USD)", row=2, col=1)
                fig.update_xaxes(title_text=None, row=2, col=1)
                
                # Format y-axis with linear scale
                fig.update_yaxes(tickformat="$.2f", type="linear", row=1, col=1)
                fig.update_yaxes(tickformat="$.2s", type="linear", row=2, col=1)
                fig.update_xaxes(type="linear", row=1, col=1)
                fig.update_xaxes(type="linear", row=2, col=1)
                
                display_chart(apply_chart_theme(fig))
        
    if not valid_liquidation_data:
        display_chart(create_fallback_chart(f"{asset} Liquidation History", "No liquidation data available"))
    
    # ==============================
    # SECTION 5: Market Sentiment Analysis
    # ==============================
    st.subheader("Market Sentiment Analysis")
    
    # Get funding rate history
    funding_history_df = None
    for key in data.get(asset, {}).get('funding_rate', {}):
        if 'ohlc_history' in key.lower() or ('history' in key.lower() and asset.lower() in key.lower()):
            temp_df = data[asset]['funding_rate'][key]
            if not temp_df.empty:
                funding_history_df = process_timestamps(temp_df)
                break
    
    # Get long-short ratio data (if available)
    # First try to get from global data
    long_short_df = None
    for key in data.get('shared', {}).get('long_short', {}):
        if 'global_account' in key.lower():
            temp_df = data['shared']['long_short'][key]
            if not temp_df.empty:
                long_short_df = process_timestamps(temp_df)
                break
    
    # Create sentiment indicators
    sentiment_indicators = {}
    sentiment_weights = {}
    
    # Funding Rate (if available)
    if current_rate is not None:
        # Scale from -0.2% to +0.2% to a -1 to +1 scale
        scaled_funding = min(max(current_rate / 20, -1), 1)
        sentiment_indicators['Funding Rate'] = scaled_funding
        sentiment_weights['Funding Rate'] = 2.0  # Higher weight as it's directly reflective of market sentiment
    
    # Long-Short Ratio (if available)
    if long_short_df is not None and not long_short_df.empty and 'global_account_long_short_ratio' in long_short_df.columns:
        latest_ls_ratio = long_short_df['global_account_long_short_ratio'].iloc[-1]
        # Scale from 0.5 to 2.0 to a -1 to +1 scale
        scaled_ls_ratio = min(max((latest_ls_ratio - 1) / 1, -1), 1)
        sentiment_indicators['Long-Short Ratio'] = scaled_ls_ratio
        sentiment_weights['Long-Short Ratio'] = 1.5  # Medium-high weight
    
    # Liquidation Ratio (if available)
    if valid_liquidation_data:
        total_long_liq = liquidation_df['aggregated_long_liquidation_usd'].sum()
        total_short_liq = liquidation_df['aggregated_short_liquidation_usd'].sum()
        
        if total_long_liq > 0 or total_short_liq > 0:
            # More short liquidations is bullish, more long liquidations is bearish
            liq_ratio = (total_short_liq - total_long_liq) / max(total_short_liq + total_long_liq, 1)
            sentiment_indicators['Liquidation Ratio'] = liq_ratio
            sentiment_weights['Liquidation Ratio'] = 1.0  # Medium weight
    
    # Display sentiment gauge chart if we have enough indicators
    if len(sentiment_indicators) >= 2:
        # Calculate weighted average sentiment
        total_weight = sum(sentiment_weights.values())
        weighted_sentiment = sum([sentiment_indicators[k] * sentiment_weights[k] for k in sentiment_indicators]) / total_weight
        
        # Scale to 0-100 for gauge chart (from -1 to +1)
        gauge_value = (weighted_sentiment + 1) * 50
        
        # Determine sentiment category
        if gauge_value >= 70:
            sentiment_category = "Strongly Bullish"
        elif gauge_value >= 55:
            sentiment_category = "Moderately Bullish"
        elif gauge_value > 45:
            sentiment_category = "Neutral"
        elif gauge_value > 30:
            sentiment_category = "Moderately Bearish"
        else:
            sentiment_category = "Strongly Bearish"
        
        # Create gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            title={"text": f"{asset} Market Sentiment"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "royalblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": "red"},
                    {"range": [30, 45], "color": "orange"},
                    {"range": [45, 55], "color": "gray"},
                    {"range": [55, 70], "color": "lightgreen"},
                    {"range": [70, 100], "color": "green"}
                ],
                "threshold": {
                    "line": {"color": "blue", "width": 4},
                    "thickness": 0.75,
                    "value": gauge_value
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # Ensure linear scale on gauge chart
        fig.update_yaxes(type="linear")
        fig.update_xaxes(type="linear")
        
        # Display the gauge in a narrower column
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.plotly_chart(fig, use_container_width=True)
        
        # Display sentiment breakdown
        st.subheader("Sentiment Breakdown")
        
        # Create columns for sentiment metrics
        sentiment_cols = st.columns(len(sentiment_indicators))
        
        for i, (indicator, value) in enumerate(sentiment_indicators.items()):
            with sentiment_cols[i]:
                # Convert the -1 to +1 scale to a descriptive string
                if value > 0.7:
                    sentiment_str = "Strongly Bullish"
                    sentiment_color = "green"
                elif value > 0.3:
                    sentiment_str = "Moderately Bullish"
                    sentiment_color = "lightgreen"
                elif value > -0.3:
                    sentiment_str = "Neutral"
                    sentiment_color = "gray"
                elif value > -0.7:
                    sentiment_str = "Moderately Bearish"
                    sentiment_color = "orange"
                else:
                    sentiment_str = "Strongly Bearish"
                    sentiment_color = "red"
                
                # Display metric
                st.metric(
                    indicator, 
                    sentiment_str,
                    f"Weight: {sentiment_weights[indicator]}"
                )
        
        # Display overall sentiment
        st.markdown(f"""
        **Overall Market Sentiment: <span style='color: {"green" if gauge_value > 55 else "red" if gauge_value < 45 else "gray"}'>{sentiment_category}</span>**
        
        This combined indicator synthesizes multiple market signals including funding rates, long-short ratios, and liquidation patterns to provide a holistic view of market sentiment.
        
        *Note: This is a relative measure and should be used alongside other technical and fundamental analysis.*
        """, unsafe_allow_html=True)
    else:
        st.warning("Insufficient data to calculate combined market sentiment indicator. At least two sentiment indicators are required.")
    
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
    

def main():
    """
    Main function to render the report dashboard with multi-asset support.
    """
    try:
        # Render sidebar
        render_sidebar()
        
        # Page title only
        st.title(f"{APP_ICON} Crypto Liquidity Report")
        
        # Display loading message
        with st.spinner("Loading dashboard data..."):
            # Load data for all supported assets
            all_data = load_report_data(assets=SUPPORTED_ASSETS)
        
        # Check if data is available
        if not all_data:
            st.error("No data available for the dashboard.")
            return
        
        # Create tabs for cross-asset overview and individual assets
        tab_names = ["Cross-Asset Overview"] + SUPPORTED_ASSETS
        tabs = st.tabs(tab_names)
        
        with tabs[0]:  # Cross-Asset Overview tab
            # Cross-asset comparison dashboard
            create_cross_asset_overview(all_data)
        
        # Create asset-specific tabs
        for i, asset in enumerate(SUPPORTED_ASSETS):
            with tabs[i+1]:  # +1 because the first tab is the overview
                # Asset-specific analysis
                display_asset_section(all_data, asset)
                
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Error in main: {e}", exc_info=True)
        return


if __name__ == "__main__":
    # Check if we're in test mode
    if 'test_mode' in st.query_params and st.query_params['test_mode'] == 'true':
        test_results = _test_asset_data_loading()
        _display_test_results(test_results)
    else:
        main()