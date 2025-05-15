"""
Spot page for the Izun Crypto Liquidity Report.

This page displays data and visualizations related to cryptocurrency spot markets.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
from utils.config import DATA_BASE_PATH
from components.metrics import display_metrics_row, display_metric_card
from components.charts import (
    create_time_series, 
    create_bar_chart, 
    create_time_series_with_bar,
    create_pie_chart,
    create_ohlc_chart,
    apply_chart_theme,
    display_chart,
    display_filterable_chart
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

# Configure more detailed logging
logger.setLevel(logging.INFO)
fh = logging.FileHandler('spot_page.log')
fh.setLevel(logging.INFO)
logger.addHandler(fh)

# Initialize subcategory session state if not exists
if 'spot_subcategory' not in st.session_state:
    st.session_state.spot_subcategory = 'market_data'  # Backend name stays as 'market_data' for compatibility

def calculate_weighted_average(df, value_col, weight_col):
    """
    Calculate weighted average of a column based on a weight column.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing data
    value_col : str
        Column containing values to average
    weight_col : str
        Column containing weights
        
    Returns:
    --------
    float
        Weighted average
    """
    # Filter to only valid rows (non-NaN and positive weights)
    valid_df = df[df[weight_col] > 0].dropna(subset=[value_col, weight_col])
    
    if valid_df.empty:
        return df[value_col].mean()  # Use simple average if no valid weights
    
    # Calculate weighted average
    return (valid_df[value_col] * valid_df[weight_col]).sum() / valid_df[weight_col].sum()

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
    logger.info(f"Loading spot data for subcategory={subcategory}, asset={asset}")
    data = {}
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    logger.info(f"Latest data directory: {latest_dir}")
    
    if not latest_dir:
        logger.error("No data directories found.")
        logger.error("No data directories found. Please check your data path.")
        return data
    
    # Load specified subcategory data
    try:
        if subcategory == 'market_data':
            logger.info(f"Loading spot market data for {asset}")
            data = load_data_for_category('spot', 'spot_market', asset)
        elif subcategory == 'order_book':
            logger.info(f"Loading order book spot data for {asset}")
            data = load_data_for_category('spot', 'order_book_spot', asset)
        elif subcategory == 'taker_buy_sell':
            logger.info(f"Loading taker buy/sell spot data for {asset}")
            data = load_data_for_category('spot', 'taker_buy_sell_spot', asset)
        else:
            logger.error(f"Unknown subcategory: {subcategory}")
            logger.error(f"Unknown subcategory: {subcategory}")
        
        logger.info(f"Data loaded for {subcategory}, found {len(data)} data objects")
    except Exception as e:
        logger.error(f"Error loading data for {subcategory}/{asset}: {e}", exc_info=True)
        logger.error(f"Error loading data: {str(e)}")
    
    return data

def create_multi_asset_candlestick(assets, market_prices=None):
    """
    Create a candlestick chart showing price data for multiple assets.
    
    Parameters:
    -----------
    assets: list
        List of assets to include in the chart
    market_prices: dict, optional
        Dictionary with current market prices for calculation of price differences
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Candlestick chart with multiple assets
    """
    logger.info(f"Creating multi-asset candlestick chart for: {assets}")
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    price_history_path = os.path.join(DATA_BASE_PATH, latest_dir, 'futures', 'market', 'api_price_ohlc_history.parquet')
    
    if not os.path.exists(price_history_path):
        logger.error(f"Price history file not found at: {price_history_path}")
        return None
    
    try:
        # Load price data
        price_df = pd.read_parquet(price_history_path)
        logger.info(f"Loaded price data with shape: {price_df.shape}, columns: {price_df.columns.tolist()}")
        
        # Convert timestamp to datetime
        if 'time' in price_df.columns:
            # Determine unit based on magnitude
            max_time = price_df['time'].max()
            if max_time > 1e12:
                price_df['datetime'] = pd.to_datetime(price_df['time'], unit='ms')
            else:
                price_df['datetime'] = pd.to_datetime(price_df['time'], unit='s')
        
        # Convert string columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume_usd']:
            if col in price_df.columns and price_df[col].dtype == 'object':
                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')
        
        # Create figure with subplots - one row per asset
        num_assets = len(assets)
        fig = make_subplots(rows=num_assets, cols=1, 
                            shared_xaxes=True, 
                            vertical_spacing=0.05,
                            subplot_titles=[f"{a} Price" for a in assets])
        
        # Add a candlestick trace for each asset
        for i, asset_name in enumerate(assets):
            # Same price data used for all assets since we don't have asset-specific data in this file
            # In a real application, you'd filter by asset
            asset_price = price_df.copy()
            
            # Sort by datetime and take only recent data (last 14 days)
            asset_price = asset_price.sort_values('datetime')
            two_weeks_ago = pd.Timestamp.now() - pd.Timedelta(days=14)
            asset_price = asset_price[asset_price['datetime'] > two_weeks_ago]
            
            # Drop rows with NaN values
            asset_price = asset_price.dropna(subset=['open', 'high', 'low', 'close', 'datetime'])
            
            if not asset_price.empty:
                # Calculate price difference if market prices are provided
                price_annotation = ""
                if market_prices and asset_name in market_prices and market_prices[asset_name]:
                    current_market_price = market_prices[asset_name]
                    latest_close = asset_price['close'].iloc[-1]
                    price_diff = current_market_price - latest_close
                    price_diff_pct = (price_diff / latest_close) * 100 if latest_close != 0 else 0
                    
                    # Update subplot title to include price difference
                    diff_text = f" ({'+' if price_diff >= 0 else ''}{price_diff:.2f}, {'+' if price_diff_pct >= 0 else ''}{price_diff_pct:.2f}%)"
                    price_annotation = diff_text
                    fig.layout.annotations[i].text = f"{asset_name} Price{price_annotation}"
                
                # Add candlestick trace
                fig.add_trace(
                    go.Candlestick(
                        x=asset_price['datetime'],
                        open=asset_price['open'], 
                        high=asset_price['high'],
                        low=asset_price['low'], 
                        close=asset_price['close'],
                        name=asset_name,
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
                
                # Add volume as a bar chart at the bottom of each price chart
                fig.add_trace(
                    go.Bar(
                        x=asset_price['datetime'],
                        y=asset_price['volume_usd'],
                        name='Volume',
                        marker_color='rgba(128,128,128,0.5)',
                        showlegend=False,
                        opacity=0.7,
                        yaxis="y2"
                    ),
                    row=i+1, col=1
                )
                
                # Customize y-axes for this subplot
                fig.update_yaxes(title_text="Price (USD)", row=i+1, col=1)
                
                # If market price is available, add a horizontal line for current market price
                if market_prices and asset_name in market_prices and market_prices[asset_name]:
                    fig.add_hline(
                        y=market_prices[asset_name],
                        line_dash="dash",
                        line_width=2,
                        line_color="rgba(255, 255, 0, 0.7)",
                        annotation_text="Current",
                        annotation_position="bottom right",
                        row=i+1, col=1
                    )
        
        # Update layout
        fig.update_layout(
            title="Asset Price Comparison (OHLC)",
            height=250 * num_assets,  # Height scales with number of assets
            margin=dict(t=50, b=20),
            xaxis_rangeslider_visible=False
        )
        
        # Update each xaxis's rangeslider visibility
        for i in range(num_assets):
            fig.update_xaxes(rangeslider_visible=False, row=i+1, col=1)
        
        # Apply the theme
        fig = apply_chart_theme(fig)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating multi-asset candlestick chart: {e}", exc_info=True)
        # Don't show technical error to user
        return None


def render_market_data_page(asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render the market data page for the specified asset and other selected assets.
    
    Parameters:
    -----------
    asset: str
        Primary asset to display (for backward compatibility)
    all_selected_assets: list
        List of all selected assets to display
    selected_exchanges: list
        List of exchanges to display data for
    selected_time_range: str
        Selected time range for filtering data
    """
    # Write debug info to logs
    logger.info(f"Starting market data page for {asset}")
    logger.info(f"Selected exchanges: {selected_exchanges}")
    
    if all_selected_assets is None or len(all_selected_assets) <= 1:
        st.header(f"{asset} Live Price Data")
        logger.info(f"Rendering live price page for {asset}")
    else:
        asset_str = ", ".join(all_selected_assets)
        st.header(f"Live Price Data: {asset_str}")
        logger.info(f"Rendering live price page for multiple assets: {asset_str}")
        
    # Exchange selector
    # Define available exchanges for spot market data
    available_exchanges = ["Binance", "Coinbase", "Kraken", "OKX", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Add exchange selector
    exchange_col1, exchange_col2 = st.columns([3, 1])
    with exchange_col1:
        selected_exchanges = st.multiselect(
            "Select Exchanges to Display",
            available_exchanges,
            default=default_exchanges,
            key=f"spot_market_exchange_selector_{asset}"
        )
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
        logger.warning("At least one exchange must be selected. Defaulting to 'All'.")
    
    # Store in session state for this section
    st.session_state.selected_spot_market_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges

    # Load market data
    data = load_spot_data('market_data', asset)
    
    # Enhanced logging for debugging
    logger.info(f"Data loaded for {asset}: {bool(data)}")
    if data:
        logger.info(f"Data keys: {list(data.keys())}")
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                logger.info(f"DataFrame '{key}' shape: {value.shape}, columns: {list(value.columns)}")
            else:
                logger.info(f"Non-DataFrame data in key '{key}': {type(value)}")

    if not data:
        logger.info(f"No spot market data available for {asset}.")

        # Show empty placeholder layout
        col1, col2 = st.columns(2)
        with col1:
            st.metric(f"{asset} Price", "$0", "0%")
        with col2:
            st.metric("24h Volume", "$0")

        st.subheader("Price History")
        # Empty placeholder for price history charts

        st.subheader("Trading Pairs")
        # Empty placeholder for trading pair information
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
            
            # Filter by selected exchanges if needed (if not "All")
            if 'exchange_name' in pairs_df.columns and selected_exchanges and "All" not in selected_exchanges:
                pairs_df = pairs_df[pairs_df['exchange_name'].isin(selected_exchanges)]

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
                    
                    # Add Asset Price Comparison Chart section
                    st.subheader(f"{asset} Price Comparison Chart")
                    # Section for asset price comparison
                    
                    # Use SUPPORTED_ASSETS if available or a default list
                    try:
                        from utils.config import SUPPORTED_ASSETS
                        available_comparison_assets = SUPPORTED_ASSETS
                    except:
                        available_comparison_assets = ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA"]
                    
                    # Let user select assets to compare
                    selected_comparison_assets = st.multiselect(
                        "Select assets to compare",
                        available_comparison_assets,
                        default=all_selected_assets if all_selected_assets and len(all_selected_assets) <= 3 else [asset],
                        key="price_comparison_assets"
                    )
                    
                    # Extract current prices from the pairs data
                    market_prices = {}
                    if not pairs_df.empty and 'symbol' in pairs_df.columns and 'price_usd' in pairs_df.columns:
                        for asset_symbol in selected_comparison_assets:
                            asset_rows = pairs_df[pairs_df['symbol'].str.contains(asset_symbol, case=False, na=False)]
                            if not asset_rows.empty and 'price_usd' in asset_rows.columns:
                                # Calculate volume-weighted average price
                                if 'volume_24h_usd' in asset_rows.columns:
                                    try:
                                        market_prices[asset_symbol] = calculate_weighted_average(
                                            asset_rows, 'price_usd', 'volume_24h_usd'
                                        )
                                    except Exception as wp_err:
                                        logger.error(f"Error calculating weighted price: {wp_err}")
                                        # Fallback to simple average
                                        market_prices[asset_symbol] = asset_rows['price_usd'].mean()
                                else:
                                    market_prices[asset_symbol] = asset_rows['price_usd'].mean()
                    
                    if selected_comparison_assets:
                        # Create and display the chart
                        with st.spinner("Creating price comparison chart..."):
                            comparison_chart = create_multi_asset_candlestick(selected_comparison_assets, market_prices)
                            if comparison_chart:
                                st.plotly_chart(comparison_chart, use_container_width=True)
                            else:
                                logger.info("Price comparison chart is not available at this time.")
                    else:
                        logger.info("No assets selected for price comparison chart.")

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
                        
                        
                        # Set defaults in session state for backward compatibility
                        st.session_state.market_data_time_range = 'All'
                        st.session_state.selected_time_range = 'All'

                        # Create pie chart for volume distribution
                        st.subheader(f"{asset} Volume Distribution")

                        fig = create_pie_chart(
                            exchange_volume,
                            'volume_24h_usd',
                            'exchange_name',
                            f"{asset} Spot Volume Distribution by Exchange"
                        )

                        display_chart(fig)
                        
                        # Add Price History Chart
                        st.subheader(f"{asset} Price History")
                        
                        try:
                            # First try to load the price OHLC history data
                            price_history_df = None
                            
                            # Try loading price OHLC data directly from the file
                            try:
                                logger.info("Attempting to load price OHLC data directly from file")
                                price_history_path = os.path.join(DATA_BASE_PATH, get_latest_data_directory(), 'futures', 'market', 'api_price_ohlc_history.parquet')
                                
                                if os.path.exists(price_history_path):
                                    logger.info(f"Loading price history from: {price_history_path}")
                                    price_history_df = pd.read_parquet(price_history_path)
                                    logger.info(f"Successfully loaded price history data: {len(price_history_df)} rows")
                                    
                                    # Log the first few rows to see what the data looks like
                                    logger.info(f"Price history first rows: {price_history_df.head(2).to_dict()}")
                                    
                                    # Log the data types
                                    logger.info(f"Price history data types: {price_history_df.dtypes}")
                                else:
                                    logger.warning(f"Price history file not found at: {price_history_path}")
                                    
                                    # Try the load_data_for_category approach as fallback
                                    logger.info(f"Falling back to loading price history data for {asset} from futures/market")
                                    futures_market_data = load_data_for_category('futures', 'market', asset)
                                    
                                    # Log available keys in futures market data
                                    logger.info(f"Futures market data keys: {list(futures_market_data.keys() if futures_market_data else [])}")
                                    
                                    # Check if price data exists
                                    if futures_market_data and 'api_price_ohlc_history' in futures_market_data and not futures_market_data['api_price_ohlc_history'].empty:
                                        price_history_df = futures_market_data['api_price_ohlc_history'].copy()
                                        logger.info(f"Found price history data via category loader: {len(price_history_df)} rows")
                                    else:
                                        logger.warning(f"No price history data found for {asset} through either method")
                                        price_history_df = None
                            except Exception as load_error:
                                logger.error(f"Error loading price history data: {load_error}", exc_info=True)
                                price_history_df = None
                                
                            if price_history_df is not None and not price_history_df.empty:
                                try:
                                    # Convert timestamp to datetime if it's in milliseconds
                                    logger.info("Processing price data...")
                                    logger.info("Converting timestamps and numeric columns...")
                                    
                                    if 'time' in price_history_df.columns:
                                        logger.info(f"Converting time column: dtype={price_history_df['time'].dtype}, first value={price_history_df['time'].iloc[0]}")
                                        
                                        if price_history_df['time'].dtype == 'int64':
                                            try:
                                                # Check the magnitude of the timestamps to determine unit
                                                max_time = price_history_df['time'].max()
                                                logger.info(f"Maximum timestamp value: {max_time}")
                                                
                                                # Typically if > 1e12, it's milliseconds, if > 1e9 but < 1e12, it's seconds
                                                if max_time > 1e12:
                                                    logger.info("Timestamp appears to be in milliseconds")
                                                    price_history_df['datetime'] = pd.to_datetime(price_history_df['time'], unit='ms')
                                                elif max_time > 1e9:
                                                    logger.info("Timestamp appears to be in seconds")
                                                    price_history_df['datetime'] = pd.to_datetime(price_history_df['time'], unit='s')
                                                else:
                                                    # Default to milliseconds as most common
                                                    logger.info("Using default milliseconds for timestamp")
                                                    price_history_df['datetime'] = pd.to_datetime(price_history_df['time'], unit='ms')
                                                    
                                                logger.info(f"Converted time to datetime: {price_history_df['datetime'].head(2)}")
                                            except Exception as time_err:
                                                logger.error(f"Error converting timestamp: {time_err}")
                                                # Try multiple approaches sequentially
                                                for unit in ['ms', 's', 'us', 'ns']:
                                                    try:
                                                        price_history_df['datetime'] = pd.to_datetime(price_history_df['time'], unit=unit)
                                                        logger.info(f"Successfully converted time using {unit} unit")
                                                        break
                                                    except Exception as unit_err:
                                                        logger.warning(f"Failed {unit} conversion: {unit_err}")
                                        else:
                                            try:
                                                price_history_df['datetime'] = pd.to_datetime(price_history_df['time'])
                                                logger.info(f"Converted string time to datetime: {price_history_df['datetime'].head(2)}")
                                            except Exception as dt_err:
                                                logger.error(f"Could not convert 'time' to datetime: {dt_err}")
                                                logger.warning("Could not process time data correctly")
                                    
                                    # Convert string columns to numeric
                                    logger.info("Converting numeric columns")
                                    numeric_cols = ['open', 'high', 'low', 'close', 'volume_usd']
                                    for col in numeric_cols:
                                        if col in price_history_df.columns:
                                            try:
                                                if price_history_df[col].dtype == 'object':
                                                    logger.info(f"Converting {col} from object to numeric: sample value={price_history_df[col].iloc[0]}")
                                                    # First remove any commas or other non-numeric characters
                                                    if isinstance(price_history_df[col].iloc[0], str):
                                                        price_history_df[col] = price_history_df[col].str.replace(',', '').str.strip()
                                                    price_history_df[col] = pd.to_numeric(price_history_df[col], errors='coerce')
                                                    nan_count = price_history_df[col].isna().sum()
                                                    if nan_count > 0:
                                                        logger.warning(f"NaN count after converting {col}: {nan_count} out of {len(price_history_df)}")
                                            except Exception as num_err:
                                                logger.error(f"Error converting {col} to numeric: {num_err}")
                                                logger.warning(f"Error processing numeric data in {col} column")
                                    
                                    # Create OHLC chart
                                    required_cols = ['open', 'high', 'low', 'close', 'datetime']
                                    missing_cols = [col for col in required_cols if col not in price_history_df.columns]
                                    
                                    if missing_cols:
                                        logger.error(f"Missing required columns for OHLC chart: {missing_cols}. Available columns: {price_history_df.columns.tolist()}")
                                        logger.warning(f"Cannot create OHLC chart - missing required columns: {missing_cols}")
                                    elif price_history_df.empty:
                                        logger.error("Price history DataFrame is empty")
                                        logger.warning("No price history data available to create chart")
                                    else:
                                        # Log information about the data
                                        logger.info(f"Creating OHLC chart with {len(price_history_df)} data points")
                                        
                                        # Check for NaN values
                                        for col in required_cols:
                                            nan_count = price_history_df[col].isna().sum()
                                            if nan_count > 0:
                                                logger.warning(f"Column {col} has {nan_count} NaN values out of {len(price_history_df)} rows")
                                        
                                        try:
                                            # Create OHLC chart with additional safety checks
                                            try:
                                                # Sort by datetime to ensure proper display
                                                price_history_df = price_history_df.sort_values('datetime')
                                                
                                                # Filter out NaN values
                                                price_history_df = price_history_df.dropna(subset=['open', 'high', 'low', 'close', 'datetime'])
                                                
                                                # Ensure data is numeric
                                                for col in ['open', 'high', 'low', 'close']:
                                                    if price_history_df[col].dtype == 'object':
                                                        price_history_df[col] = pd.to_numeric(price_history_df[col], errors='coerce')
                                                
                                                # Drop any rows with NaN after conversion
                                                price_history_df = price_history_df.dropna(subset=['open', 'high', 'low', 'close'])
                                                
                                                # Log the first few rows after cleanup
                                                logger.info(f"Data for charting after cleanup: {len(price_history_df)} rows remaining")
                                                
                                                if not price_history_df.empty:
                                                    # Verify data is valid for charting
                                                    logger.info(f"Sample data for charting: Open={price_history_df['open'].iloc[0]}, High={price_history_df['high'].iloc[0]}, Low={price_history_df['low'].iloc[0]}, Close={price_history_df['close'].iloc[0]}")
                                                    
                                                    fig = create_ohlc_chart(
                                                        price_history_df,
                                                        'datetime',
                                                        'open', 'high', 'low', 'close',
                                                        f"{asset} Price (OHLC)",
                                                        height=500
                                                    )
                                                else:
                                                    logger.error("No valid data points after filtering NaNs")
                                                    fig = go.Figure()
                                                    fig.add_annotation(
                                                        text="Price history data will appear here when available",
                                                        xref="paper", yref="paper",
                                                        x=0.5, y=0.5, showarrow=False,
                                                        font=dict(size=16, color="gray")
                                                    )
                                            except Exception as chart_create_err:
                                                logger.error(f"Error during chart creation: {chart_create_err}")
                                                fig = go.Figure()
                                                fig.add_annotation(
                                                    text="Unable to display price chart at this time",
                                                    xref="paper", yref="paper",
                                                    x=0.5, y=0.5, showarrow=False,
                                                    font=dict(size=16, color="gray")
                                                )
                                            logger.info("Successfully created OHLC chart, now displaying")
                                            display_chart(apply_chart_theme(fig))
                                            logger.info("OHLC chart displayed successfully")
                                        except Exception as chart_error:
                                            logger.error(f"Error creating/displaying OHLC chart: {chart_error}", exc_info=True)
                                        
                                        # Create volume chart
                                        if 'volume_usd' in price_history_df.columns:
                                            st.subheader(f"{asset} Trading Volume History")
                                            logger.info("Attempting to create volume chart")
                                            
                                            # Check required columns
                                            vol_required_cols = ['datetime', 'close', 'volume_usd']
                                            vol_missing_cols = [col for col in vol_required_cols if col not in price_history_df.columns]
                                            
                                            if vol_missing_cols:
                                                logger.error(f"Missing required columns for volume chart: {vol_missing_cols}")
                                                logger.warning(f"Cannot create volume chart - missing columns: {vol_missing_cols}")
                                            else:
                                                try:
                                                    # Check NaN values
                                                    for col in vol_required_cols:
                                                        nan_count = price_history_df[col].isna().sum()
                                                        if nan_count > 0:
                                                            logger.warning(f"Volume chart: Column {col} has {nan_count} NaN values")
                                                    
                                                    # Create the chart with safety checks
                                                    try:
                                                        # Sort by datetime to ensure proper display
                                                        price_history_df = price_history_df.sort_values('datetime')
                                                        
                                                        # Filter out NaN values
                                                        price_history_df = price_history_df.dropna(subset=['close', 'volume_usd', 'datetime'])
                                                        
                                                        # Ensure data is numeric
                                                        for col in ['close', 'volume_usd']:
                                                            if price_history_df[col].dtype == 'object':
                                                                price_history_df[col] = pd.to_numeric(price_history_df[col], errors='coerce')
                                                        
                                                        # Drop any rows with NaN after conversion
                                                        price_history_df = price_history_df.dropna(subset=['close', 'volume_usd'])
                                                        
                                                        # Log status after cleanup
                                                        logger.info(f"Volume chart data after cleanup: {len(price_history_df)} rows remaining")
                                                        
                                                        if not price_history_df.empty:
                                                            # Verify data is valid for charting
                                                            logger.info(f"Sample data for volume chart: Close={price_history_df['close'].iloc[0]}, Volume={price_history_df['volume_usd'].iloc[0]}")
                                                            
                                                            fig = create_time_series_with_bar(
                                                                price_history_df, 
                                                                'datetime', 
                                                                'close', 
                                                                'volume_usd',
                                                                f"{asset} Price vs Volume",
                                                                height=500
                                                            )
                                                        else:
                                                            logger.error("No valid data points for volume chart after filtering NaNs")
                                                            logger.info("Volume history data is not available for this time period.")
                                                            fig = go.Figure()
                                                            fig.add_annotation(
                                                                text="Volume history data will appear here when available",
                                                                xref="paper", yref="paper",
                                                                x=0.5, y=0.5, showarrow=False,
                                                                font=dict(size=16, color="gray")
                                                            )
                                                    except Exception as vol_chart_err:
                                                        logger.error(f"Error during volume chart creation: {vol_chart_err}")
                                                        fig = go.Figure()
                                                        fig.add_annotation(
                                                            text="Unable to display volume chart at this time",
                                                            xref="paper", yref="paper",
                                                            x=0.5, y=0.5, showarrow=False,
                                                            font=dict(size=16, color="gray")
                                                        )
                                                    logger.info("Successfully created volume chart")
                                                    display_chart(apply_chart_theme(fig))
                                                    logger.info("Volume chart displayed successfully")
                                                except Exception as vol_error:
                                                    logger.error(f"Error creating/displaying volume chart: {vol_error}", exc_info=True)
                                except Exception as e:
                                    logger.error(f"Error creating price history charts: {e}", exc_info=True)
                        except Exception as e:
                            logger.error(f"Error loading price history data: {e}", exc_info=True)
                        
                        # Buy vs Sell Volume Analysis
                        try:
                            st.subheader(f"{asset} Buy vs Sell Volume")
                            
                            # Check if we have buy/sell volume data in pairs_df
                            logger.info("Checking for buy/sell volume data")
                            if pairs_df is not None and not pairs_df.empty:
                                # Log available columns
                                logger.info(f"Pairs DF columns: {pairs_df.columns.tolist()}")
                                
                                required_cols = ['buy_volume_usd_24h', 'sell_volume_usd_24h', 'exchange_name']
                                
                                if all(col in pairs_df.columns for col in required_cols):
                                    # Aggregate buy/sell volumes by exchange
                                    buy_sell_df = pairs_df.groupby('exchange_name').agg({
                                        'buy_volume_usd_24h': 'sum',
                                        'sell_volume_usd_24h': 'sum'
                                    }).reset_index()
                                    
                                    # Calculate total and net flow
                                    buy_sell_df['total_volume'] = buy_sell_df['buy_volume_usd_24h'] + buy_sell_df['sell_volume_usd_24h']
                                    buy_sell_df['net_flow'] = buy_sell_df['buy_volume_usd_24h'] - buy_sell_df['sell_volume_usd_24h']
                                    
                                    # Sort by total volume
                                    buy_sell_df = buy_sell_df.sort_values('total_volume', ascending=False).head(10)
                                    
                                    # Create stacked bar chart for buy/sell volumes
                                    fig = go.Figure()
                                    
                                    fig.add_trace(go.Bar(
                                        x=buy_sell_df['exchange_name'],
                                        y=buy_sell_df['buy_volume_usd_24h'],
                                        name='Buy Volume',
                                        marker_color='green'
                                    ))
                                    
                                    fig.add_trace(go.Bar(
                                        x=buy_sell_df['exchange_name'],
                                        y=buy_sell_df['sell_volume_usd_24h'],
                                        name='Sell Volume',
                                        marker_color='red'
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"{asset} 24h Buy vs Sell Volume by Exchange",
                                        xaxis_title=None,
                                        yaxis_title="Volume (USD)",
                                        barmode='stack',
                                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                    )
                                    
                                    # Format y-axis to use currency format
                                    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
                                    
                                    display_chart(apply_chart_theme(fig))
                                    
                                    # Create net flow chart
                                    fig = go.Figure()
                                    
                                    colors = ['green' if x >= 0 else 'red' for x in buy_sell_df['net_flow']]
                                    
                                    fig.add_trace(go.Bar(
                                        x=buy_sell_df['exchange_name'],
                                        y=buy_sell_df['net_flow'],
                                        marker_color=colors,
                                        name='Net Flow'
                                    ))
                                    
                                    fig.update_layout(
                                        title=f"{asset} 24h Net Flow by Exchange (Buy - Sell)",
                                        xaxis_title=None,
                                        yaxis_title="Net Flow (USD)"
                                    )
                                    
                                    # Format y-axis to use currency format
                                    fig.update_yaxes(tickprefix="$", tickformat=",.0f")
                                    
                                    # Add zero line
                                    fig.add_hline(y=0, line_dash="dash", line_color="gray")
                                    
                                    display_chart(apply_chart_theme(fig))
                                    
                                    # Add time period comparison if we have the data
                                    logger.info("Checking for time period comparison data")
                                    time_periods = {
                                        '1h': {'price': 'price_change_percent_1h', 'volume': 'volume_usd_1h'},
                                        '4h': {'price': 'price_change_percent_4h', 'volume': 'volume_usd_4h'},
                                        '12h': {'price': 'price_change_percent_12h', 'volume': 'volume_usd_12h'},
                                        '24h': {'price': 'price_change_percent_24h', 'volume': 'volume_usd_24h'},
                                        '1w': {'price': 'price_change_percent_1w', 'volume': 'volume_usd_1w'}
                                    }
                                    
                                    valid_periods = []
                                    for period, cols in time_periods.items():
                                        if all(col in pairs_df.columns for col in cols.values()):
                                            valid_periods.append(period)
                                    
                                    if valid_periods:
                                        st.subheader(f"{asset} Price Change & Volume by Time Period")
                                        
                                        # Create comparison dataframe
                                        comparison_data = []
                                        
                                        for period in valid_periods:
                                            price_col = time_periods[period]['price']
                                            volume_col = time_periods[period]['volume']
                                            
                                            # Calculate volume-weighted average price change
                                            avg_price_change = calculate_weighted_average(pairs_df, price_col, volume_col)
                                            total_volume = pairs_df[volume_col].sum()
                                            
                                            comparison_data.append({
                                                'Period': period,
                                                'Price Change %': avg_price_change,
                                                'Volume (USD)': total_volume
                                            })
                                        
                                        comparison_df = pd.DataFrame(comparison_data)
                                        
                                        # Create dual-axis chart
                                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                                        
                                        # Add price change line
                                        fig.add_trace(
                                            go.Scatter(
                                                x=comparison_df['Period'],
                                                y=comparison_df['Price Change %'],
                                                name="Price Change %",
                                                line=dict(color=ASSET_COLORS.get(asset, '#3366CC'), width=3)
                                            ),
                                            secondary_y=False,
                                        )
                                        
                                        # Add volume bars
                                        fig.add_trace(
                                            go.Bar(
                                                x=comparison_df['Period'],
                                                y=comparison_df['Volume (USD)'],
                                                name="Volume",
                                                marker_color="rgba(180, 180, 180, 0.7)"
                                            ),
                                            secondary_y=True,
                                        )
                                        
                                        # Set axes titles
                                        fig.update_layout(
                                            title=f"{asset} Price Change & Volume Comparison",
                                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                                        )
                                        
                                        fig.update_yaxes(title_text="Price Change %", secondary_y=False, ticksuffix="%")
                                        fig.update_yaxes(title_text="Volume (USD)", secondary_y=True, tickprefix="$", tickformat=",.0f")
                                        
                                        # Add reference line at zero for price change
                                        fig.add_hline(y=0, line_dash="dash", line_color="gray", secondary_y=False)
                                        
                                        display_chart(apply_chart_theme(fig))
                                else:
                                    logger.warning("Required columns for buy/sell analysis not found in pairs_df")
                                    logger.info("Buy/sell volume data not available. The data is missing required columns.")
                        except Exception as e:
                            logger.error(f"Error in buy/sell volume analysis: {e}", exc_info=True)
            else:
                logger.info("Trading pairs data is not in the expected format. Some features may be unavailable.")
        except Exception as e:
            logger.error(f"Error in market data processing: {e}", exc_info=True)
    else:
        logger.info(f"No trading pairs data available for {asset}.")

    # Supported coins data
    if 'api_spot_supported_coins' in data:
        coins_df = data['api_spot_supported_coins']

        if not coins_df.empty:
            try:
                st.subheader("Supported Coins")

                # Check for required columns
                if 'coin_symbol' not in coins_df.columns or 'market_count' not in coins_df.columns:
                    logger.warning("Supported coins data is missing required columns.")
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
                logger.error(f"Error processing supported coins data: {e}", exc_info=True)
        else:
            logger.info("No supported coins data available.")

def render_order_book_page(asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render the order book page for the specified asset and other selected assets.
    
    Parameters:
    -----------
    asset: str
        Primary asset to display (for backward compatibility)
    all_selected_assets: list
        List of all selected assets to display
    selected_exchanges: list
        List of exchanges to display data for
    selected_time_range: str
        Selected time range for filtering data
    """
    if all_selected_assets is None or len(all_selected_assets) <= 1:
        st.header(f"{asset} Spot Order Book Analysis")
    else:
        asset_str = ", ".join(all_selected_assets)
        st.header(f"Spot Order Book Analysis: {asset_str}")
        
    # Exchange selector
    # Define available exchanges for spot order book
    available_exchanges = ["Binance", "Coinbase", "Kraken", "OKX", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Add exchange selector
    exchange_col1, exchange_col2 = st.columns([3, 1])
    with exchange_col1:
        selected_exchanges = st.multiselect(
            "Select Exchanges to Display",
            available_exchanges,
            default=default_exchanges,
            key=f"spot_orderbook_exchange_selector_{asset}"
        )
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
        logger.warning("At least one exchange must be selected. Defaulting to 'All'.")
    
    # Store in session state for this section
    st.session_state.selected_spot_orderbook_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges
    
    # Load order book data
    data = load_spot_data('order_book', asset)
    
    if not data:
        logger.info(f"No spot order book data available for {asset}.")
        
        # Show empty placeholder layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Asks Amount", "$0")
        with col2:
            st.metric("Bids Amount", "$0")
        with col3:
            st.metric("Asks/Bids Ratio", "0.00")
            
        st.subheader(f"{asset} Order Book History")
        # Empty placeholder for order book history charts
        return
    
    # Get order book history from Coinglass API data
    possible_keys = [
        f"api_spot_orderbook_aggregated_ask_bids_history_{asset}_{asset}",
        f"api_spot_orderbook_aggregated_ask_bids_history_{asset}",
        "api_spot_orderbook_aggregated_ask_bids_history"
    ]
    
    history_df = None
    for key in possible_keys:
        if key in data and not data[key].empty:
            history_df = data[key].copy()
            logger.info(f"Found order book history data using key: {key}")
            break
    
    # Check for individual exchange data
    exchange_key = f"api_spot_orderbook_ask_bids_history_{asset}"
    exchange_df = None
    if exchange_key in data and not data[exchange_key].empty:
        exchange_df = data[exchange_key].copy()
        logger.info(f"Found exchange-specific order book data using key: {exchange_key}")
    
    # Process aggregated order book data if available
    if history_df is not None and not history_df.empty:
        try:
            # Process timestamps if needed
            history_df = process_timestamps(history_df)
            
            # Check for required columns
            required_cols = ['asks_amount', 'bids_amount']
            missing_cols = [col for col in required_cols if col not in history_df.columns]
            
            if missing_cols:
                # Try to map alternative column names
                column_mapping = {}
                
                # Check for asks_usd and bids_usd
                if 'asks_usd' in history_df.columns and 'asks_amount' not in history_df.columns:
                    column_mapping['asks_usd'] = 'asks_amount'
                
                if 'bids_usd' in history_df.columns and 'bids_amount' not in history_df.columns:
                    column_mapping['bids_usd'] = 'bids_amount'
                
                # Apply column mapping if needed
                if column_mapping:
                    history_df = history_df.rename(columns=column_mapping)
                    logger.info(f"Renamed columns using mapping: {column_mapping}")
            
            # Map the available columns to required ones
            logger.info(f"Available order book columns: {history_df.columns.tolist()}")
            
            # Create mappings based on available columns
            asks_col = None
            bids_col = None
            
            # Try to find appropriate columns for asks and bids
            if 'aggregated_asks_usd' in history_df.columns:
                asks_col = 'aggregated_asks_usd'
            elif 'asks_amount' in history_df.columns:
                asks_col = 'asks_amount'
            elif 'asks_usd' in history_df.columns:
                asks_col = 'asks_usd'
                
            if 'aggregated_bids_usd' in history_df.columns:
                bids_col = 'aggregated_bids_usd'
            elif 'bids_amount' in history_df.columns:
                bids_col = 'bids_amount'
            elif 'bids_usd' in history_df.columns:
                bids_col = 'bids_usd'
            
            # Calculate metrics if we have the required columns
            if asks_col and bids_col:
                logger.info(f"Using {asks_col} for asks and {bids_col} for bids")
                
                # Calculate asks/bids ratio
                history_df['asks_bids_ratio'] = history_df[asks_col] / history_df[bids_col].replace(0, float('nan'))
                
                # Get most recent values for metrics
                if not history_df.empty:
                    recent_data = history_df.iloc[-1]
                    asks_amount = recent_data[asks_col]
                    bids_amount = recent_data[bids_col]
                    ratio = asks_amount / bids_amount if bids_amount != 0 else float('nan')
                    
                    # Display metrics
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
                        "Asks/Bids Ratio": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                    }
                    
                    display_metrics_row(metrics, formatters)
                
                # Create time series chart for asks and bids
                st.subheader(f"{asset} Order Book History")
                
                fig = go.Figure()
                
                # Make sure data is sorted by datetime
                history_df = history_df.sort_values('datetime')
                
                fig.add_trace(go.Scatter(
                    x=history_df['datetime'],
                    y=history_df[asks_col],
                    name='Asks Amount',
                    line=dict(color='red')
                ))
                
                fig.add_trace(go.Scatter(
                    x=history_df['datetime'],
                    y=history_df[bids_col],
                    name='Bids Amount',
                    line=dict(color='green')
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"{asset} Order Book Asks/Bids Amount",
                    xaxis_title=None,
                    yaxis_title="Amount (USD)",
                    hovermode="x unified"
                )
                
                display_chart(apply_chart_theme(fig))
                
                # Create ratio chart if asks_bids_ratio exists
                st.subheader(f"{asset} Ask/Bid Ratio")
                
                fig = px.line(
                    history_df,
                    x='datetime',
                    y='asks_bids_ratio',
                    title=f"{asset} Ask/Bid Ratio"
                )
                
                # Add reference line at 1 (equal asks and bids)
                fig.add_hline(
                    y=1,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Equal"
                )
                
                display_chart(apply_chart_theme(fig))
            else:
                logger.info("Order book data is not available in the expected format for this asset.")
        except Exception as e:
            logger.error(f"Error in order book processing: {e}", exc_info=True)
            logger.info("Order book data could not be processed.")
    else:
        logger.info(f"No order book history data available for {asset}.")
    
    # Process exchange-specific order book data if available
    if exchange_df is not None and not exchange_df.empty:
        try:
            # Process timestamps if needed
            exchange_df = process_timestamps(exchange_df)
            
            # Filter by exchange if applicable
            if 'exchange_name' in exchange_df.columns and selected_exchanges and "All" not in selected_exchanges:
                exchange_df = exchange_df[exchange_df['exchange_name'].isin(selected_exchanges)]
            
            # Group by exchange
            if 'exchange_name' in exchange_df.columns:
                exchanges = exchange_df['exchange_name'].unique()
                
                if len(exchanges) > 0:
                    st.subheader("Order Book by Exchange")
                    
                    # Add exchange selector
                    selected_exchange = st.selectbox(
                        "Select Exchange",
                        exchanges,
                        key=f"order_book_exchange_selector_{asset}"
                    )
                    
                    # Filter for selected exchange
                    selected_df = exchange_df[exchange_df['exchange_name'] == selected_exchange]
                    
                    # Map the available columns to required ones
                    logger.info(f"Available exchange order book columns: {selected_df.columns.tolist()}")
                    
                    # Create mappings based on available columns
                    asks_col = None
                    bids_col = None
                    
                    # Try to find appropriate columns for asks and bids
                    if 'aggregated_asks_usd' in selected_df.columns:
                        asks_col = 'aggregated_asks_usd'
                    elif 'asks_amount' in selected_df.columns:
                        asks_col = 'asks_amount'
                    elif 'asks_usd' in selected_df.columns:
                        asks_col = 'asks_usd'
                        
                    if 'aggregated_bids_usd' in selected_df.columns:
                        bids_col = 'aggregated_bids_usd'
                    elif 'bids_amount' in selected_df.columns:
                        bids_col = 'bids_amount'
                    elif 'bids_usd' in selected_df.columns:
                        bids_col = 'bids_usd'
                    
                    # Check for required columns
                    if asks_col and bids_col:
                        logger.info(f"Using {asks_col} for asks and {bids_col} for bids")
                        st.subheader(f"{selected_exchange} {asset} Order Book")
                        
                        # Create time series chart for asks and bids
                        fig = go.Figure()
                        
                        # Make sure data is sorted by datetime
                        selected_df = selected_df.sort_values('datetime')
                        
                        fig.add_trace(go.Scatter(
                            x=selected_df['datetime'],
                            y=selected_df[asks_col],
                            name='Asks Amount',
                            line=dict(color='red')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=selected_df['datetime'],
                            y=selected_df[bids_col],
                            name='Bids Amount',
                            line=dict(color='green')
                        ))
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{selected_exchange} {asset} Order Book",
                            xaxis_title=None,
                            yaxis_title="Amount (USD)",
                            hovermode="x unified"
                        )
                        
                        display_chart(apply_chart_theme(fig))
                        
                        # Calculate ratio
                        ratio_col = 'asks_bids_ratio'
                        selected_df[ratio_col] = selected_df[asks_col] / selected_df[bids_col].replace(0, float('nan'))
                        
                        # Create ratio chart
                        fig = px.line(
                            selected_df,
                            x='datetime',
                            y=ratio_col,
                            title=f"{selected_exchange} {asset} Ask/Bid Ratio"
                        )
                        
                        # Add reference line at 1 (equal asks and bids)
                        fig.add_hline(
                            y=1,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Equal"
                        )
                        
                        display_chart(apply_chart_theme(fig))
                    else:
                        logger.info("Exchange order book data doesn't have the required columns.")
            else:
                logger.info("Exchange-specific order book data is not available for this asset.")
        except Exception as e:
            logger.error(f"Error in exchange order book processing: {e}", exc_info=True)
            logger.info("Exchange order book data could not be processed.")

def render_taker_buy_sell_page(asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render the taker buy/sell page for the specified asset and other selected assets.
    
    Parameters:
    -----------
    asset: str
        Primary asset to display (for backward compatibility)
    all_selected_assets: list
        List of all selected assets to display
    selected_exchanges: list
        List of exchanges to display data for
    selected_time_range: str
        Selected time range for filtering data
    """
    if all_selected_assets is None or len(all_selected_assets) <= 1:
        st.header(f"{asset} Spot Taker Buy/Sell Analysis")
    else:
        asset_str = ", ".join(all_selected_assets)
        st.header(f"Spot Taker Buy/Sell Analysis: {asset_str}")
        
    # Exchange selector
    # Define available exchanges for taker buy/sell
    available_exchanges = ["Binance", "Coinbase", "Kraken", "OKX", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Add exchange selector
    exchange_col1, exchange_col2 = st.columns([3, 1])
    with exchange_col1:
        selected_exchanges = st.multiselect(
            "Select Exchanges to Display",
            available_exchanges,
            default=default_exchanges,
            key=f"taker_buy_sell_exchange_selector_{asset}"
        )
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
        logger.warning("At least one exchange must be selected. Defaulting to 'All'.")
    
    # Store in session state for this section
    st.session_state.selected_taker_buy_sell_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges
    
    # Load taker buy/sell data
    data = load_spot_data('taker_buy_sell', asset)
    
    if not data:
        logger.info(f"No spot taker buy/sell data available for {asset}.")
        
        # Show empty placeholder layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Buy Volume", "$0")
        with col2:
            st.metric("Sell Volume", "$0")
        with col3:
            st.metric("Buy/Sell Ratio", "0.00")
            
        st.subheader(f"{asset} Taker Buy/Sell History")
        # Empty placeholder for taker buy/sell history charts
        return
    
    # Get taker buy/sell history from Coinglass API data
    possible_keys = [
        f"api_spot_aggregated_taker_buy_sell_volume_history_{asset}_{asset}",
        f"api_spot_aggregated_taker_buy_sell_volume_history_{asset}",
        "api_spot_aggregated_taker_buy_sell_volume_history"
    ]
    
    history_df = None
    for key in possible_keys:
        if key in data and not data[key].empty:
            history_df = data[key].copy()
            logger.info(f"Found taker buy/sell history data using key: {key}")
            break
    
    # Process taker buy/sell data if available
    if history_df is not None and not history_df.empty:
        try:
            # Process timestamps if needed
            history_df = process_timestamps(history_df)
            
            # Check for required columns
            # Map the available columns to required ones
            logger.info(f"Available taker buy/sell columns: {history_df.columns.tolist()}")
            
            # Create mappings based on available columns
            buy_col = None
            sell_col = None
            
            # Try to find appropriate columns for buy and sell volumes
            if 'aggregated_buy_volume_usd' in history_df.columns:
                buy_col = 'aggregated_buy_volume_usd'
            elif 'buy_volume' in history_df.columns:
                buy_col = 'buy_volume'
            elif 'taker_buy_volume' in history_df.columns:
                buy_col = 'taker_buy_volume'
            elif 'buy_volume_usd' in history_df.columns:
                buy_col = 'buy_volume_usd'
                
            if 'aggregated_sell_volume_usd' in history_df.columns:
                sell_col = 'aggregated_sell_volume_usd'
            elif 'sell_volume' in history_df.columns:
                sell_col = 'sell_volume'
            elif 'taker_sell_volume' in history_df.columns:
                sell_col = 'taker_sell_volume'
            elif 'sell_volume_usd' in history_df.columns:
                sell_col = 'sell_volume_usd'
            
            # For backward compatibility, create standardized column names
            if buy_col and buy_col != 'buy_volume':
                history_df['buy_volume'] = history_df[buy_col]
                logger.info(f"Created 'buy_volume' from {buy_col}")
                
            if sell_col and sell_col != 'sell_volume':
                history_df['sell_volume'] = history_df[sell_col]
                logger.info(f"Created 'sell_volume' from {sell_col}")
                
            # Calculate metrics if we have the required columns
            if ('buy_volume' in history_df.columns or buy_col) and ('sell_volume' in history_df.columns or sell_col):
                # Calculate buy/sell ratio
                history_df['buy_sell_ratio'] = history_df['buy_volume'] / history_df['sell_volume'].replace(0, float('nan'))
                
                # Get most recent values for metrics
                if not history_df.empty:
                    recent_data = history_df.iloc[-1]
                    buy_volume = recent_data['buy_volume']
                    sell_volume = recent_data['sell_volume']
                    ratio = buy_volume / sell_volume if sell_volume != 0 else float('nan')
                    
                    # Display metrics
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
                        "Buy/Sell Ratio": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                    }
                    
                    display_metrics_row(metrics, formatters)
                
                # Create time series chart for buy and sell volumes
                st.subheader(f"{asset} Taker Buy/Sell History")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=history_df['datetime'],
                    y=history_df['buy_volume'],
                    name='Buy Volume',
                    line=dict(color='green')
                ))
                
                fig.add_trace(go.Scatter(
                    x=history_df['datetime'],
                    y=history_df['sell_volume'],
                    name='Sell Volume',
                    line=dict(color='red')
                ))
                
                # Update layout
                fig.update_layout(
                    title=f"{asset} Taker Buy/Sell Volume",
                    xaxis_title=None,
                    yaxis_title="Volume (USD)",
                    hovermode="x unified"
                )
                
                display_chart(apply_chart_theme(fig))
                
                # Create ratio chart if buy_sell_ratio exists
                st.subheader(f"{asset} Buy/Sell Ratio")
                
                fig = px.line(
                    history_df,
                    x='datetime',
                    y='buy_sell_ratio',
                    title=f"{asset} Buy/Sell Ratio"
                )
                
                # Add reference line at 1 (equal buy and sell)
                fig.add_hline(
                    y=1,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Equal"
                )
                
                display_chart(apply_chart_theme(fig))
                
                # Create net flow chart
                st.subheader(f"{asset} Net Flow (Buy - Sell)")
                
                # Calculate net flow
                history_df['net_flow'] = history_df['buy_volume'] - history_df['sell_volume']
                
                fig = px.bar(
                    history_df,
                    x='datetime',
                    y='net_flow',
                    title=f"{asset} Net Flow (Buy - Sell Volume)",
                    color='net_flow',
                    color_continuous_scale=['red', 'green'],
                    color_continuous_midpoint=0
                )
                
                # Update layout
                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="Net Flow (USD)",
                    coloraxis_showscale=False
                )
                
                # Add reference line at 0
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray"
                )
                
                display_chart(apply_chart_theme(fig))
            else:
                logger.warning(f"Taker buy/sell data is missing required columns. Available columns: {list(history_df.columns)}")
        except Exception as e:
            logger.error(f"Error processing taker buy/sell data: {e}")
            logger.error(f"Error in taker buy/sell processing: {e}", exc_info=True)
            logger.info("Unable to display taker buy/sell data due to data format issues.")
    else:
        logger.warning(f"No taker buy/sell history data available for {asset}.")

def main():
    """Main function to render the spot page."""

    # Render sidebar
    render_sidebar()

    # Page title
    st.title("Cryptocurrency Spot Markets")

    # Get available assets for spot category
    available_assets = get_available_assets_for_category('spot')

    # Set default assets if none are available
    if not available_assets:
        logger.warning("No spot data available for any asset. Showing layout with placeholder data.")
        available_assets = ["BTC", "ETH", "SOL", "XRP"]
    
    # Asset selection with dropdown
    st.subheader("Select Asset to Display")
    
    # Initialize with previously selected asset if available, otherwise default to first asset
    default_asset = st.session_state.get('selected_spot_assets', [available_assets[0]])
    default_index = available_assets.index(default_asset[0]) if default_asset and default_asset[0] in available_assets else 0
    
    # Add dropdown for asset selection (improved from multiselect for better UI)
    selected_asset = st.selectbox(
        "Select asset to display",
        available_assets,
        index=default_index,
        key="spot_assets_selector"
    )
    
    # Use a single asset in a list for compatibility with existing code
    selected_assets = [selected_asset]
    
    # Store selected assets in session state for this page
    st.session_state.selected_spot_assets = selected_assets
    
    # For backward compatibility with existing code, use first selected asset as primary
    asset = selected_assets[0]
    
    # Also update the general selected_asset session state for compatibility
    st.session_state.selected_asset = asset

    # Define categories
    spot_categories = [
        "Live Price",
        "Order Book",
        "Taker Buy/Sell"
    ]

    # Create tabs for each category
    tabs = st.tabs(spot_categories)

    # Find the index of the currently active category
    current_subcategory = st.session_state.get('spot_subcategory', 'market_data').replace('_', ' ').title()
    # Handle special case for 'Market Data' -> 'Live Price' 
    if current_subcategory == "Market Data":
        current_subcategory = "Live Price"
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
                # Use page-specific exchange selection if available, otherwise use general selection
                selected_market_exchanges = st.session_state.get('selected_spot_market_exchanges', 
                                                    st.session_state.get('selected_exchanges', ["All"]))
                render_market_data_page(asset, selected_assets, selected_market_exchanges)

        with tabs[1]:  # Order Book
            if active_tab == 1 or True:  # Always load since Streamlit may show any tab
                subcategory = 'order_book'
                st.session_state.spot_subcategory = subcategory
                # Use page-specific exchange selection if available, otherwise use general selection
                selected_orderbook_exchanges = st.session_state.get('selected_spot_orderbook_exchanges', 
                                                     st.session_state.get('selected_exchanges', ["All"]))
                render_order_book_page(asset, selected_assets, selected_orderbook_exchanges)

        with tabs[2]:  # Taker Buy/Sell
            if active_tab == 2 or True:  # Always load since Streamlit may show any tab
                subcategory = 'taker_buy_sell'
                st.session_state.spot_subcategory = subcategory
                # Use page-specific exchange selection if available, otherwise use general selection
                selected_taker_exchanges = st.session_state.get('selected_taker_buy_sell_exchanges', 
                                                  st.session_state.get('selected_exchanges', ["All"]))
                render_taker_buy_sell_page(asset, selected_assets, selected_taker_exchanges)

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
                        selected_market_exchanges = st.session_state.get('selected_spot_market_exchanges', 
                                                       st.session_state.get('selected_exchanges', ["All"]))
                        render_market_data_page(asset, selected_assets, selected_market_exchanges)
                    elif subcategory == 'order_book':
                        selected_orderbook_exchanges = st.session_state.get('selected_spot_orderbook_exchanges', 
                                                        st.session_state.get('selected_exchanges', ["All"]))
                        render_order_book_page(asset, selected_assets, selected_orderbook_exchanges)
                    elif subcategory == 'taker_buy_sell':
                        selected_taker_exchanges = st.session_state.get('selected_taker_buy_sell_exchanges', 
                                                     st.session_state.get('selected_exchanges', ["All"]))
                        render_taker_buy_sell_page(asset, selected_assets, selected_taker_exchanges)
                except Exception as tab_error:
                    logger.error(f"Error rendering {subcategory} data: {tab_error}")
                    logger.info("There was an error processing the data. This could be due to an unexpected data format or missing data.")
    

if __name__ == "__main__":
    main()