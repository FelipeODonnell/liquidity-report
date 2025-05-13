"""
Spot page for the Izun Crypto Liquidity Report.

This page displays data and visualizations related to cryptocurrency spot markets.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Add parent directory to path to import from components and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components and utilities
from components.sidebar import render_sidebar
from components.metrics import display_metrics_row, display_metric_card
from components.charts import (
    create_time_series, 
    create_bar_chart, 
    create_time_series_with_bar,
    create_pie_chart,
    apply_chart_theme,
    display_chart
)
from components.tables import create_formatted_table, create_exchange_table
from utils.data_loader import (
    get_latest_data_directory, 
    load_data_for_category, 
    process_timestamps,
    get_data_last_updated,
    calculate_metrics,
    get_available_assets_for_category
)
from utils.formatters import (
    format_currency, 
    format_percentage, 
    format_volume,
    format_timestamp,
    humanize_time_diff
)
from utils.config import APP_TITLE, APP_ICON, ASSET_COLORS

# Set page config with title and icon
st.set_page_config(
    page_title=f"{APP_TITLE} - Spot Markets",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the current page for sidebar navigation
st.session_state.current_page = 'spot'

# Initialize subcategory session state if not exists
if 'spot_subcategory' not in st.session_state:
    st.session_state.spot_subcategory = 'market_data'

def load_spot_data(subcategory, asset):
    """
    Load spot data for the specified subcategory and asset.
    
    Parameters:
    -----------
    subcategory : str
        Subcategory of spot data to load
    asset : str
        Asset to load data for
        
    Returns:
    --------
    dict
        Dictionary containing spot data
    """
    data = {}
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return data
    
    # Load specified subcategory data
    if subcategory == 'market_data':
        data = load_data_for_category('spot', 'spot_market', asset)
    elif subcategory == 'order_book':
        data = load_data_for_category('spot', 'order_book_spot', asset)
    elif subcategory == 'taker_buy_sell':
        data = load_data_for_category('spot', 'taker_buy_sell_spot', asset)
    else:
        st.error(f"Unknown subcategory: {subcategory}")
    
    return data

def render_market_data_page(asset):
    """Render the market data page for the specified asset."""
    st.header(f"{asset} Spot Market Data")
    logger.info(f"Rendering market data page for {asset}")

    # Load market data
    data = load_spot_data('market_data', asset)

    if not data:
        st.info(f"No spot market data available for {asset}.")
        st.write("Spot market data shows information about cryptocurrency trading pairs across various exchanges.")

        # Show empty placeholder layout
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{asset} Price", "$0", "0%")
        with col2:
            st.metric("24h Volume", "$0")

        st.subheader("Price History")
        st.write("No data available. Price history charts will be displayed here when data is loaded.")

        st.subheader("Trading Pairs")
        st.write("No data available. Trading pair information will be displayed here when data is loaded.")
        return

    # Log data keys for debugging
    logger.info(f"Available data keys: {list(data.keys())}")

    # Trading pairs data - try different possible key formats
    possible_pairs_keys = [
        f"api_spot_pairs_markets_{asset}_{asset}",
        f"api_spot_pairs_markets_{asset}",
        "api_spot_pairs_markets",  # Generic fallback
        f"api_spot_pairs_markets"  # Additional format without asset
    ]

    pairs_df = None
    used_key = None
    for key in possible_pairs_keys:
        if key in data and not data[key].empty:
            pairs_df = data[key].copy()  # Create a copy to avoid modifying the original
            used_key = key
            logger.info(f"Found pairs data using key: {key}")
            # If generic key, try to filter for asset
            if key == 'api_spot_pairs_markets' and 'symbol' in pairs_df.columns:
                pairs_df = pairs_df[pairs_df['symbol'].str.contains(asset, case=False, na=False)]
            break

    # Price history - Since the price history file might be missing, we'll try to extract price data from pairs data
    current_price = None
    price_change_24h_pct = None
    total_volume_24h = None

    if pairs_df is not None and not pairs_df.empty:
        try:
            logger.info(f"Processing pairs data with columns: {list(pairs_df.columns)}")

            # Map column names if needed
            column_mapping = {}

            # Check for price column variations
            price_columns = ['price_usd', 'current_price', 'current_price_usd', 'price']
            found_price_col = None
            for col in price_columns:
                if col in pairs_df.columns:
                    found_price_col = col
                    if col != 'price_usd':
                        column_mapping[col] = 'price_usd'
                    break

            # Check for volume column variations
            volume_columns = ['volume_24h_usd', 'volume_24h', 'total_volume_24h_usd', 'volume_usd']
            found_volume_col = None
            for col in volume_columns:
                if col in pairs_df.columns:
                    found_volume_col = col
                    if col != 'volume_24h_usd':
                        column_mapping[col] = 'volume_24h_usd'
                    break

            # Check for price change column variations
            change_columns = ['price_change_percentage_24h', 'change_24h', 'price_change_24h_pct', 'change_24h_pct']
            found_change_col = None
            for col in change_columns:
                if col in pairs_df.columns:
                    found_change_col = col
                    if col != 'price_change_percentage_24h':
                        column_mapping[col] = 'price_change_percentage_24h'
                    break

            # Apply column mapping if needed
            if column_mapping:
                pairs_df = pairs_df.rename(columns=column_mapping)
                logger.info(f"Renamed columns using mapping: {column_mapping}")

            # Convert string columns to numeric
            numeric_cols = ['price_usd', 'volume_24h_usd', 'price_change_percentage_24h']
            for col in numeric_cols:
                if col in pairs_df.columns and pairs_df[col].dtype == 'object':
                    pairs_df[col] = pd.to_numeric(pairs_df[col], errors='coerce')

            # Calculate metrics from the pairs data
            if 'price_usd' in pairs_df.columns:
                # Use volume-weighted average price
                if 'volume_24h_usd' in pairs_df.columns:
                    # Filter out rows with zero volume to avoid division by zero
                    valid_pairs = pairs_df[(pairs_df['volume_24h_usd'] > 0) & (~pd.isna(pairs_df['price_usd']))]
                    if not valid_pairs.empty:
                        total_volume = valid_pairs['volume_24h_usd'].sum()
                        current_price = (valid_pairs['price_usd'] * valid_pairs['volume_24h_usd']).sum() / total_volume
                    else:
                        # Fallback to simple average
                        current_price = pairs_df['price_usd'].mean()
                else:
                    # Simple average if no volume data
                    current_price = pairs_df['price_usd'].mean()

            # Calculate 24h price change
            if 'price_change_percentage_24h' in pairs_df.columns:
                # Use volume-weighted average
                if 'volume_24h_usd' in pairs_df.columns:
                    valid_pairs = pairs_df[(pairs_df['volume_24h_usd'] > 0) & (~pd.isna(pairs_df['price_change_percentage_24h']))]
                    if not valid_pairs.empty:
                        total_volume = valid_pairs['volume_24h_usd'].sum()
                        price_change_24h_pct = (valid_pairs['price_change_percentage_24h'] * valid_pairs['volume_24h_usd']).sum() / total_volume
                    else:
                        price_change_24h_pct = pairs_df['price_change_percentage_24h'].mean()
                else:
                    price_change_24h_pct = pairs_df['price_change_percentage_24h'].mean()

            # Calculate total volume
            if 'volume_24h_usd' in pairs_df.columns:
                total_volume_24h = pairs_df['volume_24h_usd'].sum()

            # Display metrics
            metrics = {}
            formatters = {}

            # Add price metric if available
            if current_price is not None:
                metrics[f"{asset} Spot Price"] = {
                    "value": current_price,
                    "delta": price_change_24h_pct,
                    "delta_suffix": "%"
                }
                formatters[f"{asset} Spot Price"] = lambda x: format_currency(x, precision=2)

            # Add volume metric if available
            if total_volume_24h is not None:
                metrics["24h Volume"] = {
                    "value": total_volume_24h,
                    "delta": None
                }
                formatters["24h Volume"] = lambda x: format_currency(x, abbreviate=True, show_decimals=False)

            if metrics:
                display_metrics_row(metrics, formatters)

            # Now process and display the pairs data
            st.subheader(f"{asset} Trading Pairs")

            # Check if we have the necessary columns
            if 'exchange_name' in pairs_df.columns and 'price_usd' in pairs_df.columns:
                # Sort by volume
                if 'volume_24h_usd' in pairs_df.columns:
                    pairs_df = pairs_df.sort_values(by='volume_24h_usd', ascending=False)

                # Select relevant columns for display
                display_cols = []

                # Always include exchange and symbol
                if 'exchange_name' in pairs_df.columns:
                    display_cols.append('exchange_name')
                if 'symbol' in pairs_df.columns:
                    display_cols.append('symbol')

                # Add price, volume and change if available
                if 'price_usd' in pairs_df.columns:
                    display_cols.append('price_usd')
                if 'volume_24h_usd' in pairs_df.columns:
                    display_cols.append('volume_24h_usd')
                if 'price_change_percentage_24h' in pairs_df.columns:
                    display_cols.append('price_change_percentage_24h')

                # Create a display dataframe with selected columns
                if display_cols:
                    display_df = pairs_df[display_cols].copy()

                    # Format column names for better display
                    column_display_names = {
                        'exchange_name': 'Exchange',
                        'symbol': 'Symbol',
                        'price_usd': 'Price (USD)',
                        'volume_24h_usd': 'Volume (24h)',
                        'price_change_percentage_24h': 'Change (24h)'
                    }

                    display_df = display_df.rename(columns=column_display_names)

                    # Create formatting dictionary
                    format_dict = {}
                    if 'Price (USD)' in display_df.columns:
                        format_dict['Price (USD)'] = lambda x: format_currency(x, precision=2)
                    if 'Volume (24h)' in display_df.columns:
                        format_dict['Volume (24h)'] = lambda x: format_currency(x, abbreviate=True, show_decimals=False)
                    if 'Change (24h)' in display_df.columns:
                        format_dict['Change (24h)'] = lambda x: format_percentage(x, precision=2)

                    # Create and display the table
                    create_formatted_table(display_df, format_dict=format_dict)

                # Create bar chart for volume by exchange if we have the necessary data
                if 'exchange_name' in pairs_df.columns and 'volume_24h_usd' in pairs_df.columns:
                    exchange_volume = pairs_df.groupby('exchange_name')['volume_24h_usd'].sum().reset_index()
                    exchange_volume = exchange_volume.sort_values(by='volume_24h_usd', ascending=False).head(10)

                    if not exchange_volume.empty:
                        st.subheader(f"Top Exchanges by {asset} Volume")

                        fig = px.bar(
                            exchange_volume,
                            x='exchange_name',
                            y='volume_24h_usd',
                            title=f"Top 10 Exchanges by {asset} Spot Trading Volume",
                            color='volume_24h_usd',
                            color_continuous_scale='Viridis'
                        )

                        fig.update_layout(
                            xaxis_title=None,
                            yaxis_title="24h Volume (USD)",
                            coloraxis_showscale=False
                        )

                        # Format y-axis tick labels to use commas with no decimals
                        fig.update_yaxes(
                            tickprefix="$",
                            tickformat=",",
                        )

                        display_chart(apply_chart_theme(fig))

                        # Create pie chart for volume distribution
                        st.subheader(f"{asset} Volume Distribution")

                        fig = create_pie_chart(
                            exchange_volume,
                            'volume_24h_usd',
                            'exchange_name',
                            f"{asset} Spot Volume Distribution by Exchange"
                        )

                        display_chart(fig)
            else:
                st.warning(f"Trading pairs data is missing required columns. Current columns: {list(pairs_df.columns)}")
                st.dataframe(pairs_df.head())  # Show raw data for debugging
        except Exception as e:
            st.error(f"Error processing trading pairs data: {e}")
            logger.error(f"Error in market data processing: {e}", exc_info=True)
            st.info("Unable to display market data due to data format issues.")
    else:
        st.info(f"No trading pairs data available for {asset}.")

    # Supported coins data
    if 'api_spot_supported_coins' in data:
        coins_df = data['api_spot_supported_coins']

        if not coins_df.empty:
            try:
                st.subheader("Supported Coins")

                # Check for required columns
                if 'coin_symbol' not in coins_df.columns or 'market_count' not in coins_df.columns:
                    st.warning("Supported coins data is missing required columns.")
                    st.dataframe(coins_df)  # Show raw data as fallback
                else:
                    # Sort by market count
                    coins_df = coins_df.sort_values(by='market_count', ascending=False)

                    # Create table
                    create_formatted_table(
                        coins_df,
                        format_dict={
                            'market_count': lambda x: format_currency(x, include_symbol=False, show_decimals=False)
                        }
                    )

                    # Create bar chart for top coins by market count
                    fig = px.bar(
                        coins_df.head(20),  # Top 20 coins
                        x='coin_symbol',
                        y='market_count',
                        title="Top 20 Coins by Market Count",
                        color='market_count',
                        color_continuous_scale='Viridis'
                    )

                    fig.update_layout(
                        xaxis_title=None,
                        yaxis_title="Market Count",
                        coloraxis_showscale=False
                    )

                    # Format y-axis to use integers with commas
                    fig.update_yaxes(tickformat=",")

                    display_chart(apply_chart_theme(fig))
            except Exception as e:
                st.error(f"Error processing supported coins data: {e}")
                st.info("Unable to display supported coins data due to data format issues.")
        else:
            st.info("No supported coins data available.")

def render_order_book_page(asset):
    """Render the order book page for the specified asset."""
    st.header(f"{asset} Spot Order Book Analysis")

    # Load order book data
    data = load_spot_data('order_book', asset)

    if not data:
        st.info(f"No spot order book data available for {asset}.")
        st.write("Order book data shows the balance between buy and sell orders in the spot market.")

        # Show empty placeholder layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Asks Amount", "$0", "0%")
        with col2:
            st.metric("Bids Amount", "$0", "0%")
        with col3:
            st.metric("Asks/Bids Ratio", "0.00")

        st.subheader("Order Book History")
        st.write("No data available. Order book charts will be displayed here when data is loaded.")
        return

    # Try different possible key formats for asset-specific data
    possible_asset_keys = [
        f"api_spot_orderbook_ask_bids_history_{asset}_{asset}",  # Double asset format
        f"api_spot_orderbook_ask_bids_history_{asset}",          # Single asset format
    ]

    # Try different possible key formats for aggregated data
    possible_agg_keys = [
        "api_spot_orderbook_aggregated_ask_bids_history",
        "api_spot_orderbook_ask_bids_history",
    ]

    # First try asset-specific data keys
    ob_df = None
    ob_title = ""

    # Try asset-specific keys first
    for key in possible_asset_keys:
        if key in data and not data[key].empty:
            ob_df = data[key]
            ob_title = f"{asset} Spot Order Book"
            logger.info(f"Found order book data using key: {key}")
            break

    # If asset-specific keys didn't work, try aggregated keys
    if ob_df is None or ob_df.empty:
        for key in possible_agg_keys:
            if key in data and not data[key].empty:
                ob_df = data[key]
                ob_title = "Spot Order Book (Aggregated)"
                logger.info(f"Using aggregated order book data with key: {key}")

                # If it's aggregated and has symbol column, filter for the asset
                if 'symbol' in ob_df.columns:
                    ob_df = ob_df[ob_df['symbol'].str.contains(asset, case=False, na=False)]
                break

    if ob_df is not None and not ob_df.empty:
        try:
            # Process dataframe for timestamps
            ob_df = process_timestamps(ob_df, timestamp_col='time')

            # Map column names to expected values if needed
            column_mapping = {}

            # Map asks/bids columns if they have different names
            if 'asks_usd' in ob_df.columns and 'asks_amount' not in ob_df.columns:
                column_mapping['asks_usd'] = 'asks_amount'

            if 'bids_usd' in ob_df.columns and 'bids_amount' not in ob_df.columns:
                column_mapping['bids_usd'] = 'bids_amount'

            # Apply column renaming if needed
            if column_mapping:
                ob_df = ob_df.rename(columns=column_mapping)

            # Calculate asks/bids ratio if it doesn't exist
            if 'asks_bids_ratio' not in ob_df.columns:
                ob_df['asks_bids_ratio'] = ob_df['asks_amount'] / ob_df['bids_amount'].replace(0, float('nan'))

            # Calculate metrics from most recent data point
            recent_ob = ob_df.iloc[-1] if len(ob_df) > 0 else None

            if recent_ob is not None:
                asks_amount = recent_ob['asks_amount']
                bids_amount = recent_ob['bids_amount']
                ratio = recent_ob['asks_bids_ratio']

                # Create metrics
                metrics = {
                    "Asks Amount": {
                        "value": asks_amount,
                        "delta": None
                    },
                    "Bids Amount": {
                        "value": bids_amount,
                        "delta": None
                    },
                    "Asks/Bids Ratio": {
                        "value": ratio,
                        "delta": None
                    }
                }

                formatters = {
                    "Asks Amount": lambda x: format_currency(x, abbreviate=True),
                    "Bids Amount": lambda x: format_currency(x, abbreviate=True),
                    "Asks/Bids Ratio": lambda x: f"{x:.4f}"
                }

                display_metrics_row(metrics, formatters)

            # Create time series chart for asks and bids
            st.subheader(ob_title)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=ob_df['datetime'],
                y=ob_df['asks_amount'],
                name='Asks Amount',
                line=dict(color='red')
            ))

            fig.add_trace(go.Scatter(
                x=ob_df['datetime'],
                y=ob_df['bids_amount'],
                name='Bids Amount',
                line=dict(color='green')
            ))

            # Update layout
            fig.update_layout(
                title=f"{ob_title} - Asks/Bids Amount",
                xaxis_title=None,
                yaxis_title="Amount (USD)",
                hovermode="x unified"
            )

            display_chart(apply_chart_theme(fig))

            # Create ratio chart
            st.subheader(f"{ob_title} - Ask/Bid Ratio")

            fig = px.line(
                ob_df,
                x='datetime',
                y='asks_bids_ratio',
                title=f"{ob_title} - Ask/Bid Ratio"
            )

            # Add reference line at 1 (equal asks and bids)
            fig.add_hline(
                y=1,
                line_dash="dash",
                line_color="gray",
                annotation_text="Equal"
            )

            display_chart(apply_chart_theme(fig))
        except Exception as e:
            st.error(f"Error processing order book data: {e}")
            logger.error(f"Error in order book processing: {e}", exc_info=True)
            st.info("Unable to display order book data due to data format issues.")
    else:
        st.info(f"No order book data available for {asset}.")
    
    # Large limit orders
    if 'api_spot_orderbook_large_limit_order' in data:
        large_orders_df = data['api_spot_orderbook_large_limit_order']
        
        if not large_orders_df.empty:
            # Filter for the selected asset
            asset_orders = large_orders_df[large_orders_df['symbol'].str.contains(asset, case=False, na=False)]
            
            if not asset_orders.empty:
                st.subheader(f"{asset} Large Limit Orders")
                
                # Sort by size
                asset_orders = asset_orders.sort_values(by='amount_usd', ascending=False)
                
                # Create table
                create_formatted_table(
                    asset_orders,
                    format_dict={
                        'price': lambda x: format_currency(x, precision=2),
                        'amount': lambda x: f"{x:.6f}",
                        'amount_usd': lambda x: format_currency(x, abbreviate=True)
                    }
                )
            else:
                st.info(f"No large limit orders available for {asset}.")
        else:
            st.info("No large limit order data available.")
    
    # Order book heatmap
    if 'api_spot_orderbook_heatmap_history' in data:
        heatmap_df = data['api_spot_orderbook_heatmap_history']
        
        if not heatmap_df.empty:
            # Filter for the selected asset
            asset_heatmap = heatmap_df[heatmap_df['symbol'].str.contains(asset, case=False, na=False)]
            
            if not asset_heatmap.empty:
                st.subheader(f"{asset} Order Book Heatmap")
                st.info("Order book heatmap visualization not implemented yet.")
            else:
                st.info(f"No order book heatmap data available for {asset}.")
        else:
            st.info("No order book heatmap data available.")

def render_taker_buy_sell_page(asset):
    """Render the taker buy/sell page for the specified asset."""
    st.header("Spot Market Taker Buy/Sell Analysis")

    # Load taker buy/sell data
    data = load_spot_data('taker_buy_sell', asset)

    if not data:
        st.info("No spot taker buy/sell data available.")
        st.write("Taker buy/sell data shows the balance between buying and selling activities in the spot market.")

        # Show empty placeholder layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Buy Volume", "$0", "0%")
        with col2:
            st.metric("Sell Volume", "$0", "0%")
        with col3:
            st.metric("Buy/Sell Ratio", "0.00")

        st.subheader("Buy/Sell Volume History")
        st.write("No data available. Buy/sell volume charts will be displayed here when data is loaded.")
        return

    # Check if aggregated taker buy/sell history is available
    if 'api_spot_aggregated_taker_buy_sell_volume_history' in data:
        taker_df = data['api_spot_aggregated_taker_buy_sell_volume_history']

        if not taker_df.empty:
            try:
                # Process dataframe with timestamp, explicitly specify 'time' column
                taker_df = process_timestamps(taker_df, timestamp_col='time')

                logger.info(f"Taker buy/sell columns before mapping: {list(taker_df.columns)}")

                # Map column names to expected values
                column_mapping = {}

                # Map the column names based on the actual data structure
                if 'aggregated_buy_volume_usd' in taker_df.columns and 'buy_volume' not in taker_df.columns:
                    column_mapping['aggregated_buy_volume_usd'] = 'buy_volume'

                if 'aggregated_sell_volume_usd' in taker_df.columns and 'sell_volume' not in taker_df.columns:
                    column_mapping['aggregated_sell_volume_usd'] = 'sell_volume'

                # Apply column renaming if needed
                if column_mapping:
                    taker_df = taker_df.rename(columns=column_mapping)
                    logger.info(f"Renamed columns using mapping: {column_mapping}")

                logger.info(f"Taker buy/sell columns after mapping: {list(taker_df.columns)}")

                # Calculate buy/sell ratio if it doesn't exist
                if 'buy_sell_ratio' not in taker_df.columns:
                    taker_df['buy_sell_ratio'] = taker_df['buy_volume'] / taker_df['sell_volume'].replace(0, float('nan'))

                # Check if we have the required columns now
                if 'buy_volume' in taker_df.columns and 'sell_volume' in taker_df.columns:
                    # Calculate metrics from most recent data point
                    recent_taker = taker_df.iloc[-1] if len(taker_df) > 0 else None

                    if recent_taker is not None:
                        buy_volume = recent_taker['buy_volume']
                        sell_volume = recent_taker['sell_volume']
                        ratio = recent_taker['buy_sell_ratio']

                        # Create metrics
                        metrics = {
                            "Buy Volume": {
                                "value": buy_volume,
                                "delta": None
                            },
                            "Sell Volume": {
                                "value": sell_volume,
                                "delta": None
                            },
                            "Buy/Sell Ratio": {
                                "value": ratio,
                                "delta": None
                            }
                        }

                        formatters = {
                            "Buy Volume": lambda x: format_currency(x, abbreviate=True),
                            "Sell Volume": lambda x: format_currency(x, abbreviate=True),
                            "Buy/Sell Ratio": lambda x: f"{x:.4f}" if x is not None else "N/A"
                        }

                        display_metrics_row(metrics, formatters)

                    # Create stacked bar chart for buy/sell volume
                    st.subheader("Taker Buy/Sell Volume History")

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=taker_df['datetime'],
                        y=taker_df['buy_volume'],
                        name='Buy Volume',
                        marker_color='green'
                    ))

                    fig.add_trace(go.Bar(
                        x=taker_df['datetime'],
                        y=taker_df['sell_volume'],
                        name='Sell Volume',
                        marker_color='red'
                    ))

                    # Update layout
                    fig.update_layout(
                        title="Taker Buy/Sell Volume History (All Coins)",
                        barmode='group',
                        xaxis_title=None,
                        yaxis_title="Volume (USD)",
                        hovermode="x unified"
                    )

                    display_chart(apply_chart_theme(fig))

                    # Create buy/sell ratio chart
                    st.subheader("Buy/Sell Ratio History")

                    fig = px.line(
                        taker_df,
                        x='datetime',
                        y='buy_sell_ratio',
                        title="Taker Buy/Sell Ratio History (All Coins)"
                    )

                    # Add reference line at 1 (equal buy and sell)
                    fig.add_hline(
                        y=1,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Equal"
                    )

                    display_chart(apply_chart_theme(fig))

                    # Add analysis of buy/sell ratio
                    st.subheader("Buy/Sell Ratio Analysis")

                    # Calculate average ratio and trend
                    avg_ratio = taker_df['buy_sell_ratio'].mean()

                    # Calculate the trend in the ratio
                    if len(taker_df) > 5:  # Need a reasonable amount of data for trend
                        recent_5 = taker_df.iloc[-5:]
                        avg_5 = recent_5['buy_sell_ratio'].mean()

                        trend_description = ""
                        if avg_5 > 1.05:
                            trend_description = "Currently trending toward **buying pressure** (ratio > 1.05)"
                        elif avg_5 < 0.95:
                            trend_description = "Currently trending toward **selling pressure** (ratio < 0.95)"
                        else:
                            trend_description = "Currently showing **balanced** buying and selling (0.95 ≤ ratio ≤ 1.05)"

                        st.write(f"Average Buy/Sell Ratio: **{avg_ratio:.4f}**")
                        st.write(trend_description)

                        # Add some market interpretation
                        if avg_5 > avg_ratio:
                            st.write("Recent buy/sell ratio is **above** the historical average, suggesting increasing buy pressure.")
                        elif avg_5 < avg_ratio:
                            st.write("Recent buy/sell ratio is **below** the historical average, suggesting increasing sell pressure.")
                        else:
                            st.write("Recent buy/sell ratio is consistent with the historical average.")
                else:
                    st.error(f"Missing required columns after mapping. Current columns: {list(taker_df.columns)}")
                    st.dataframe(taker_df.head())  # Show raw data for debugging
            except Exception as e:
                st.error(f"Error processing taker buy/sell data: {e}")
                logger.error(f"Error in taker buy/sell processing: {e}", exc_info=True)

                # Try to show the raw data for debugging
                st.info("Original data format:")
                try:
                    # Show a few rows of the original data for debugging
                    raw_df = data['api_spot_aggregated_taker_buy_sell_volume_history']
                    st.dataframe(raw_df.head())
                except:
                    st.write("Could not display raw data.")
        else:
            st.info("Taker buy/sell data file is empty.")
    else:
        # Try looking for other possible files
        other_keys = [k for k in data.keys() if 'taker' in k.lower() or 'buy_sell' in k.lower()]

        if other_keys:
            st.info(f"Aggregated taker buy/sell data not found. Found alternative data files: {', '.join(other_keys)}")
            st.write("Please contact the developer to update the implementation for these data files.")
        else:
            st.info("No taker buy/sell data files found in the data directory.")

def main():
    """Main function to render the spot page."""

    # Render sidebar
    render_sidebar()

    # Page title
    st.title("Cryptocurrency Spot Markets")

    # Get asset from session state or use default
    available_assets = get_available_assets_for_category('spot')

    # Set default assets if none are available
    if not available_assets:
        st.warning("No spot data available for any asset. Showing layout with placeholder data.")
        available_assets = ["BTC", "ETH", "SOL", "XRP"]

    asset = st.session_state.get('selected_asset', available_assets[0])

    # Define categories
    spot_categories = [
        "Market Data",
        "Order Book",
        "Taker Buy/Sell"
    ]

    # Create tabs for each category
    tabs = st.tabs(spot_categories)

    # Find the index of the currently active category
    current_subcategory = st.session_state.get('spot_subcategory', 'market_data').replace('_', ' ').title()
    active_tab = 0
    for i, cat in enumerate(spot_categories):
        if cat == current_subcategory:
            active_tab = i
            break

    # We can't programmatically set the active tab in Streamlit,
    # but we can pre-load data for the expected active tab

    # First attempt at rendering, may fail with empty tabs if errors occur
    try:
        with tabs[0]:  # Market Data
            if active_tab == 0 or True:  # Always load since Streamlit may show any tab
                subcategory = 'market_data'
                st.session_state.spot_subcategory = subcategory
                render_market_data_page(asset)

        with tabs[1]:  # Order Book
            if active_tab == 1 or True:  # Always load since Streamlit may show any tab
                subcategory = 'order_book'
                st.session_state.spot_subcategory = subcategory
                render_order_book_page(asset)

        with tabs[2]:  # Taker Buy/Sell
            if active_tab == 2 or True:  # Always load since Streamlit may show any tab
                subcategory = 'taker_buy_sell'
                st.session_state.spot_subcategory = subcategory
                render_taker_buy_sell_page(asset)

    except Exception as e:
        # If there's an error, log it but don't show to user - they'll just see empty tabs
        logger.error(f"Error rendering tabs: {e}")

        # Attempt to render each tab with more robust error handling
        for i, tab in enumerate(tabs):
            with tab:
                subcategory = spot_categories[i].lower().replace(' ', '_')
                try:
                    # Use function dispatch to render the appropriate page
                    if subcategory == 'market_data':
                        render_market_data_page(asset)
                    elif subcategory == 'order_book':
                        render_order_book_page(asset)
                    elif subcategory == 'taker_buy_sell':
                        render_taker_buy_sell_page(asset)
                except Exception as tab_error:
                    st.error(f"Error rendering {subcategory} data: {tab_error}")
                    st.info("There was an error processing the data. This could be due to an unexpected data format or missing data.")
    
    # Add footer
    st.markdown("---")
    st.caption("Izun Crypto Liquidity Report © 2025")
    st.caption("Data provided by CoinGlass API")

if __name__ == "__main__":
    main()