"""
Futures page for the Izun Crypto Liquidity Report.

This page displays data and visualizations related to cryptocurrency futures markets.
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
from components.metrics import display_metrics_row, display_metric_card
from components.charts import (
    create_time_series, 
    create_ohlc_chart, 
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
    page_title=f"{APP_TITLE} - Futures",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the current page for sidebar navigation
st.session_state.current_page = 'futures'

# Initialize subcategory session state if not exists
if 'futures_subcategory' not in st.session_state:
    st.session_state.futures_subcategory = 'funding_rate'

def load_futures_data(subcategory, asset):
    """
    Load futures data for the specified subcategory and asset.

    Parameters:
    -----------
    subcategory : str
        Subcategory of futures data to load
    asset : str
        Asset to load data for

    Returns:
    --------
    dict
        Dictionary containing futures data
    """
    data = {}

    # Get the latest data directory
    latest_dir = get_latest_data_directory()

    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return data

    # Log the data loading request
    logger.info(f"Loading futures data for subcategory={subcategory}, asset={asset}")

    # Load specified subcategory data
    data = load_data_for_category('futures', subcategory, asset)

    # Log what was loaded
    if data:
        logger.info(f"Successfully loaded {len(data)} data files for {asset} {subcategory}")
        logger.debug(f"Loaded keys: {list(data.keys())}")
    else:
        logger.warning(f"No data found for {asset} {subcategory}")

        # Try loading without asset filter as a fallback
        logger.info(f"Attempting to load without asset filter as fallback")
        fallback_data = load_data_for_category('futures', subcategory)

        if fallback_data:
            logger.info(f"Found {len(fallback_data)} files in fallback load")

            # Add any potentially relevant files to the original data dict
            for key, df in fallback_data.items():
                # Only include files that might be relevant to this asset
                if asset.lower() in key.lower() or (
                    hasattr(df, 'columns') and 'symbol' in df.columns and
                    any(asset.lower() in str(s).lower() for s in df['symbol'].unique() if pd.notna(s))
                ):
                    data[key] = df
                    logger.info(f"Added {key} from fallback loading")

            # If we found any relevant data in the fallback
            if data:
                logger.info(f"Found {len(data)} relevant files in fallback load")
            else:
                logger.warning(f"No relevant data found in fallback load")

    # Also load market data if we need price information
    if subcategory in ['open_interest', 'funding_rate', 'long_short_ratio']:
        try:
            logger.info(f"Loading market data for price overlay")
            market_data = load_data_for_category('futures', 'market', asset)
            if market_data:
                data['market'] = market_data
                logger.info(f"Added market data with {len(market_data)} files")
            else:
                # Try without asset filter
                market_data = load_data_for_category('futures', 'market')
                if market_data:
                    data['market'] = market_data
                    logger.info(f"Added market data from fallback with {len(market_data)} files")
        except Exception as e:
            logger.error(f"Error loading market data: {e}")

    return data

def render_funding_rate_page(asset):
    """Render the funding rate page for the specified asset."""
    st.header(f"{asset} Funding Rate Analysis")

    # Load funding rate data
    data = load_futures_data('funding_rate', asset)

    if not data:
        st.info(f"No funding rate data available for {asset}.")
        st.write("Funding rates show the periodic payments between long and short positions in perpetual futures contracts.")

        # Show empty placeholder layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Average Funding Rate", "0.00%")
        with col2:
            st.metric("Highest Funding Rate", "0.00%")
        with col3:
            st.metric("Lowest Funding Rate", "0.00%")

        st.subheader("Current Funding Rates by Exchange")
        st.write("No data available. Funding rate data will show the current funding rates across different exchanges.")

        st.subheader("Funding Rate History")
        st.write("No data available. Historical funding rate charts will be displayed here when data is loaded.")
        return

    # Display metrics
    fr_metrics = {}

    # Current funding rates by exchange
    if 'api_futures_fundingRate_exchange_list' in data:
        # Process the data to normalize it if it has the nested list structure
        fr_df = process_timestamps(data['api_futures_fundingRate_exchange_list'])

        if not fr_df.empty:
            try:
                # Check if required columns exist - depending on how the data was normalized
                # For direct column format
                if 'symbol' in fr_df.columns and 'funding_rate' in fr_df.columns:
                    # Filter for the selected asset
                    asset_fr = fr_df[fr_df['symbol'].str.contains(asset, case=False, na=False)]
                # For normalized exchange list format
                elif 'symbol' in fr_df.columns and 'exchange_name' in fr_df.columns:
                    # Filter for the selected asset
                    asset_fr = fr_df[fr_df['symbol'].str.contains(asset, case=False, na=False)]
                else:
                    st.warning("Funding rate data is missing required columns.")
                    st.write("Available columns:", list(fr_df.columns))
                    asset_fr = pd.DataFrame()

                if not asset_fr.empty:
                    # Calculate metrics
                    avg_fr = asset_fr['funding_rate'].mean()
                    max_fr = asset_fr['funding_rate'].max()
                    min_fr = asset_fr['funding_rate'].min()

                    # Create metrics
                    fr_metrics["Average Funding Rate"] = {
                        "value": avg_fr * 100,  # Convert to percentage
                        "delta": None,
                        "delta_suffix": "%"
                    }

                    fr_metrics["Highest Funding Rate"] = {
                        "value": max_fr * 100,  # Convert to percentage
                        "delta": None,
                        "delta_suffix": "%"
                    }

                    fr_metrics["Lowest Funding Rate"] = {
                        "value": min_fr * 100,  # Convert to percentage
                        "delta": None,
                        "delta_suffix": "%"
                    }

                    # Format funding rates from decimal to percentage
                    formatters = {
                        "Average Funding Rate": lambda x: f"{x:.4f}%",
                        "Highest Funding Rate": lambda x: f"{x:.4f}%",
                        "Lowest Funding Rate": lambda x: f"{x:.4f}%"
                    }

                    display_metrics_row(fr_metrics, formatters)

                    # Current funding rates by exchange
                    st.subheader("Current Funding Rates by Exchange")

                    # Format dataframe for display
                    display_df = asset_fr.copy()
                    display_df['funding_rate'] = display_df['funding_rate'] * 100  # Convert to percentage

                    # Sort by funding rate
                    display_df = display_df.sort_values(by='funding_rate', ascending=False)

                    # Determine columns to display (check if next_funding_time exists)
                    display_columns = ['exchange_name', 'symbol', 'funding_rate']
                    if 'next_funding_time' in display_df.columns:
                        display_columns.append('next_funding_time')

                    # Format dictionary based on available columns
                    format_dict = {'funding_rate': lambda x: f"{x:.6f}%"}
                    if 'next_funding_time' in display_df.columns:
                        format_dict['next_funding_time'] = lambda x: format_timestamp(x, "%Y-%m-%d %H:%M")

                    # Create exchange table
                    create_formatted_table(
                        display_df[display_columns],
                        format_dict=format_dict
                    )

                    # Funding rate distribution chart
                    st.subheader("Funding Rate Distribution by Exchange")

                    # Create bar chart
                    fig = px.bar(
                        display_df,
                        x='exchange_name',
                        y='funding_rate',
                        color='funding_rate',
                        color_continuous_scale='RdBu',
                        color_continuous_midpoint=0,
                        title=f"{asset} Funding Rates by Exchange (%)"
                    )

                    fig.update_layout(
                        xaxis_title=None,
                        yaxis_title="Funding Rate (%)"
                    )

                    display_chart(apply_chart_theme(fig))
                else:
                    st.info(f"No funding rate data available for {asset}.")
            except Exception as e:
                st.error(f"Error processing funding rate data: {e}")
                st.info("Unable to display funding rate metrics due to data format issues.")

    # Funding rate history
    history_options = [
        "Standard History",
        "OI Weighted History",
        "Volume Weighted History"
    ]

    history_type = st.radio("Funding Rate History Type", history_options, horizontal=True)

    if history_type == "Standard History":
        if 'api_futures_fundingRate_ohlc_history' in data:
            fr_history_df = data['api_futures_fundingRate_ohlc_history']

            if not fr_history_df.empty:
                try:
                    # Process dataframe
                    fr_history_df = process_timestamps(fr_history_df)

                    # Check if required columns exist
                    required_cols = ['datetime', 'open', 'high', 'low', 'close']
                    if not all(col in fr_history_df.columns for col in required_cols):
                        st.warning("Funding rate history data is missing required columns for OHLC chart.")
                    else:
                        # Convert funding rates from decimal to percentage
                        for col in ['open', 'high', 'low', 'close']:
                            fr_history_df[col] = fr_history_df[col] * 100

                        # Create OHLC chart
                        st.subheader("Funding Rate History")

                        fig = create_ohlc_chart(
                            fr_history_df,
                            'datetime',
                            'open',
                            'high',
                            'low',
                            'close',
                            f"{asset} Funding Rate History (%)"
                        )

                        # Add zero line
                        fig.add_hline(
                            y=0,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Zero"
                        )

                        display_chart(fig)
                except Exception as e:
                    st.error(f"Error processing funding rate history: {e}")
                    st.info("Unable to display funding rate history chart due to data format issues.")
            else:
                st.info("No funding rate history data available.")
        else:
            st.info("No funding rate history data available.")

    elif history_type == "OI Weighted History":
        # Try different possible key formats
        possible_keys = [
            f"api_futures_fundingRate_oi_weight_ohlc_history_{asset}_{asset}",
            f"api_futures_fundingRate_oi_weight_ohlc_history_{asset}",
            "api_futures_fundingRate_oi_weight_ohlc_history_BTC", # Some files use BTC format for all assets
            "api_futures_fundingRate_oi_weight_ohlc_history_ETH",
            "api_futures_fundingRate_oi_weight_ohlc_history_SOL",
            "api_futures_fundingRate_oi_weight_ohlc_history_XRP"
        ]

        oi_fr_df = None
        oi_weighted_key = None

        for key in possible_keys:
            if key in data and not data[key].empty:
                oi_fr_df = data[key]
                oi_weighted_key = key
                break

        if oi_fr_df is not None:
            try:
                # Handle time column if it exists instead of datetime
                if 'time' in oi_fr_df.columns and 'datetime' not in oi_fr_df.columns:
                    # Convert time column to datetime
                    oi_fr_df['datetime'] = pd.to_datetime(oi_fr_df['time'], unit='ms')
                else:
                    # Process dataframe normally
                    oi_fr_df = process_timestamps(oi_fr_df)

                # Check if required columns exist
                required_cols = ['datetime', 'open', 'high', 'low', 'close']
                if not all(col in oi_fr_df.columns for col in required_cols):
                    # Log the available columns for troubleshooting
                    st.warning(f"OI-weighted funding rate data is missing required columns for OHLC chart. Available columns: {oi_fr_df.columns.tolist()}")
                else:
                    # Convert funding rates from decimal to percentage
                    for col in ['open', 'high', 'low', 'close']:
                        oi_fr_df[col] = oi_fr_df[col] * 100

                    # Create OHLC chart
                    st.subheader("OI-Weighted Funding Rate History")

                    fig = create_ohlc_chart(
                        oi_fr_df,
                        'datetime',
                        'open',
                        'high',
                        'low',
                        'close',
                        f"{asset} OI-Weighted Funding Rate History (%)"
                    )

                    # Add zero line
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Zero"
                    )

                    display_chart(fig)
            except Exception as e:
                st.error(f"Error processing OI-weighted funding rate history: {e}")
                logger.error(f"OI-weighted funding rate error: {e}")
                st.info("Unable to display OI-weighted funding rate history chart due to data format issues.")
        else:
            st.info("No OI-weighted funding rate history data available.")

    elif history_type == "Volume Weighted History":
        # Try different possible key formats
        possible_keys = [
            f"api_futures_fundingRate_vol_weight_ohlc_history_{asset}_{asset}",
            f"api_futures_fundingRate_vol_weight_ohlc_history_{asset}",
            "api_futures_fundingRate_vol_weight_ohlc_history_BTC", # Some files use BTC format for all assets
            "api_futures_fundingRate_vol_weight_ohlc_history_ETH",
            "api_futures_fundingRate_vol_weight_ohlc_history_SOL",
            "api_futures_fundingRate_vol_weight_ohlc_history_XRP"
        ]

        vol_fr_df = None
        vol_weighted_key = None

        for key in possible_keys:
            if key in data and not data[key].empty:
                vol_fr_df = data[key]
                vol_weighted_key = key
                break

        if vol_fr_df is not None:
            try:
                # Handle time column if it exists instead of datetime
                if 'time' in vol_fr_df.columns and 'datetime' not in vol_fr_df.columns:
                    # Convert time column to datetime
                    vol_fr_df['datetime'] = pd.to_datetime(vol_fr_df['time'], unit='ms')
                else:
                    # Process dataframe normally
                    vol_fr_df = process_timestamps(vol_fr_df)

                # Check if required columns exist
                required_cols = ['datetime', 'open', 'high', 'low', 'close']
                if not all(col in vol_fr_df.columns for col in required_cols):
                    # Log the available columns for troubleshooting
                    st.warning(f"Volume-weighted funding rate data is missing required columns for OHLC chart. Available columns: {vol_fr_df.columns.tolist()}")
                else:
                    # Convert funding rates from decimal to percentage
                    for col in ['open', 'high', 'low', 'close']:
                        vol_fr_df[col] = vol_fr_df[col] * 100

                    # Create OHLC chart
                    st.subheader("Volume-Weighted Funding Rate History")

                    fig = create_ohlc_chart(
                        vol_fr_df,
                        'datetime',
                        'open',
                        'high',
                        'low',
                        'close',
                        f"{asset} Volume-Weighted Funding Rate History (%)"
                    )

                    # Add zero line
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Zero"
                    )

                    display_chart(fig)
            except Exception as e:
                st.error(f"Error processing volume-weighted funding rate history: {e}")
                logger.error(f"Volume-weighted funding rate error: {e}")
                st.info("Unable to display volume-weighted funding rate history chart due to data format issues.")
        else:
            st.info("No volume-weighted funding rate history data available.")

    # Funding rate arbitrage if available
    if 'api_futures_fundingRate_arbitrage' in data:
        arb_df = data['api_futures_fundingRate_arbitrage']

        if not arb_df.empty:
            try:
                st.subheader("Funding Rate Arbitrage Opportunities")

                # Check if funding_rate_diff column exists
                if 'funding_rate_diff' not in arb_df.columns:
                    st.warning("Funding rate arbitrage data is missing required columns.")
                else:
                    # Format dataframe for display
                    display_df = arb_df.copy()

                    # Convert funding rates from decimal to percentage
                    if 'funding_rate_diff' in display_df.columns:
                        display_df['funding_rate_diff'] = display_df['funding_rate_diff'] * 100

                    if 'base_rate' in display_df.columns:
                        display_df['base_rate'] = display_df['base_rate'] * 100

                    if 'quote_rate' in display_df.columns:
                        display_df['quote_rate'] = display_df['quote_rate'] * 100

                    # Sort by funding rate difference
                    display_df = display_df.sort_values(by='funding_rate_diff', ascending=False)

                    # Format dictionary based on available columns
                    format_dict = {'funding_rate_diff': lambda x: f"{x:.6f}%"}
                    if 'base_rate' in display_df.columns:
                        format_dict['base_rate'] = lambda x: f"{x:.6f}%"
                    if 'quote_rate' in display_df.columns:
                        format_dict['quote_rate'] = lambda x: f"{x:.6f}%"

                    # Create table
                    create_formatted_table(
                        display_df,
                        format_dict=format_dict
                    )
            except Exception as e:
                st.error(f"Error processing funding rate arbitrage data: {e}")
                st.info("Unable to display funding rate arbitrage data due to data format issues.")
        else:
            st.info("No funding rate arbitrage data available.")

def render_liquidation_page(asset):
    """Render the liquidation page for the specified asset."""
    st.header(f"{asset} Liquidation Analysis")

    # Load liquidation data
    data = load_futures_data('liquidation', asset)

    if not data:
        st.info(f"No liquidation data available for {asset}. Liquidation data shows when traders' positions are forcibly closed due to insufficient margin.")

        # Show empty placeholder layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Liquidations", "$0")
        with col2:
            st.metric("Long Liquidations", "$0")
        with col3:
            st.metric("Short Liquidations", "$0")

        st.subheader(f"{asset} Liquidation History")
        st.info("No liquidation history data available. This chart would show the volume of long and short position liquidations over time.")

        st.subheader(f"{asset} Liquidations by Exchange")
        st.info("No exchange liquidation data available. This section would show which exchanges had the most liquidations.")
        return

    # Aggregated liquidation history - try multiple possible key formats
    liq_df = None
    possible_agg_keys = [
        f"api_futures_liquidation_aggregated_coin_history_{asset}_{asset}",  # Double asset format
        f"api_futures_liquidation_aggregated_coin_history_{asset}",          # Single asset format
        "api_futures_liquidation_aggregated_coin_history"                    # Generic format
    ]

    # Try each key format until we find data
    for key in possible_agg_keys:
        if key in data and not data[key].empty:
            liq_df = data[key]
            logger.info(f"Found aggregated liquidation data using key: {key}")
            break

    if liq_df is not None and not liq_df.empty:
        try:
            # Process dataframe
            liq_df = process_timestamps(liq_df)

            # Calculate metrics
            recent_liq = liq_df.iloc[-1] if len(liq_df) > 0 else None

            if recent_liq is not None:
                # Check column names and map if needed
                column_mapping = {}

                # Check for different column naming patterns
                if 'aggregated_long_liquidation_usd' not in liq_df.columns and 'long_liquidation_usd' in liq_df.columns:
                    column_mapping['long_liquidation_usd'] = 'aggregated_long_liquidation_usd'

                if 'aggregated_short_liquidation_usd' not in liq_df.columns and 'short_liquidation_usd' in liq_df.columns:
                    column_mapping['short_liquidation_usd'] = 'aggregated_short_liquidation_usd'

                # Apply column renaming if needed
                if column_mapping:
                    liq_df = liq_df.rename(columns=column_mapping)

                # Now check if we have the required columns
                if 'aggregated_long_liquidation_usd' in liq_df.columns and 'aggregated_short_liquidation_usd' in liq_df.columns:
                    # Get latest values
                    long_liq = recent_liq['aggregated_long_liquidation_usd']
                    short_liq = recent_liq['aggregated_short_liquidation_usd']
                    total_liq = long_liq + short_liq

                    # Previous day metrics for comparison
                    prev_liq = liq_df.iloc[-2] if len(liq_df) > 1 else None

                    if prev_liq is not None:
                        prev_total = prev_liq['aggregated_long_liquidation_usd'] + prev_liq['aggregated_short_liquidation_usd']
                        total_change = ((total_liq - prev_total) / prev_total) * 100 if prev_total != 0 else None
                    else:
                        total_change = None

                    # Create metrics
                    metrics = {
                        "Total Liquidations": {
                            "value": total_liq,
                            "delta": total_change,
                            "delta_suffix": "%"
                        },
                        "Long Liquidations": {
                            "value": long_liq,
                            "delta": None
                        },
                        "Short Liquidations": {
                            "value": short_liq,
                            "delta": None
                        }
                    }

                    formatters = {
                        "Total Liquidations": lambda x: format_currency(x, abbreviate=True),
                        "Long Liquidations": lambda x: format_currency(x, abbreviate=True),
                        "Short Liquidations": lambda x: format_currency(x, abbreviate=True)
                    }

                    display_metrics_row(metrics, formatters)
                else:
                    st.warning(f"Liquidation data is missing required columns. Available columns: {list(liq_df.columns)}")

                # Create stacked bar chart for liquidations
                st.subheader(f"{asset} Liquidation History")

                fig = go.Figure()

                fig.add_trace(go.Bar(
                    x=liq_df['datetime'],
                    y=liq_df['aggregated_long_liquidation_usd'],
                    name='Long Liquidations',
                    marker_color=ASSET_COLORS.get(asset, '#3366CC')
                ))

                fig.add_trace(go.Bar(
                    x=liq_df['datetime'],
                    y=liq_df['aggregated_short_liquidation_usd'],
                    name='Short Liquidations',
                    marker_color='red'
                ))

                # Update layout
                fig.update_layout(
                    title=f"{asset} Futures Liquidations",
                    barmode='stack',
                    xaxis_title=None,
                    yaxis_title="Liquidation Volume (USD)",
                    hovermode="x unified"
                )

                display_chart(apply_chart_theme(fig))

                # Add long/short ratio chart
                st.subheader("Long/Short Liquidation Ratio")

                # Calculate ratio
                liq_df['long_short_ratio'] = liq_df['aggregated_long_liquidation_usd'] / liq_df['aggregated_short_liquidation_usd'].replace(0, float('nan'))

                # Create line chart
                fig = px.line(
                    liq_df,
                    x='datetime',
                    y='long_short_ratio',
                    title=f"{asset} Long/Short Liquidation Ratio"
                )

                # Add reference line at 1 (equal long and short liquidations)
                fig.add_hline(
                    y=1,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Equal"
                )

                display_chart(apply_chart_theme(fig))
            else:
                st.warning("No recent liquidation data found in the dataset.")
        except Exception as e:
            logger.error(f"Error processing aggregated liquidation data: {e}")
            st.error(f"Error processing liquidation data: {e}")
    else:
        st.info(f"No aggregated liquidation history data available for {asset}.")

    # Liquidation by exchange - try multiple possible key formats
    exchange_liq_df = None
    possible_exchange_keys = [
        f"api_futures_liquidation_exchange_list_{asset}_{asset}",  # Double asset format
        f"api_futures_liquidation_exchange_list_{asset}",          # Single asset format
        "api_futures_liquidation_exchange_list"                    # Generic format
    ]

    # Try each key format until we find data
    for key in possible_exchange_keys:
        if key in data and not data[key].empty:
            exchange_liq_df = data[key]

            # If it's a generic format, filter for the specific asset
            if key == "api_futures_liquidation_exchange_list" and 'symbol' in exchange_liq_df.columns:
                exchange_liq_df = exchange_liq_df[exchange_liq_df['symbol'].str.contains(asset, case=False, na=False)]

            logger.info(f"Found exchange liquidation data using key: {key}")
            break

    if exchange_liq_df is not None and not exchange_liq_df.empty:
        try:
            st.subheader(f"{asset} Liquidations by Exchange")

            # Check column names and map if needed
            column_mapping = {}

            # Check if we need to rename exchange to exchange_name
            if 'exchange' in exchange_liq_df.columns and 'exchange_name' not in exchange_liq_df.columns:
                column_mapping['exchange'] = 'exchange_name'

            # Check for different column naming patterns for liquidation amounts
            if 'longLiquidation_usd' in exchange_liq_df.columns and 'long_liquidation_usd' not in exchange_liq_df.columns:
                column_mapping['longLiquidation_usd'] = 'long_liquidation_usd'

            if 'shortLiquidation_usd' in exchange_liq_df.columns and 'short_liquidation_usd' not in exchange_liq_df.columns:
                column_mapping['shortLiquidation_usd'] = 'short_liquidation_usd'

            if 'liquidation_usd' in exchange_liq_df.columns and 'total_liquidation_usd' not in exchange_liq_df.columns:
                column_mapping['liquidation_usd'] = 'total_liquidation_usd'

            # Apply column renaming if needed
            if column_mapping:
                exchange_liq_df = exchange_liq_df.rename(columns=column_mapping)

            # Calculate total if it doesn't exist
            if 'total_liquidation_usd' not in exchange_liq_df.columns and 'long_liquidation_usd' in exchange_liq_df.columns and 'short_liquidation_usd' in exchange_liq_df.columns:
                exchange_liq_df['total_liquidation_usd'] = exchange_liq_df['long_liquidation_usd'] + exchange_liq_df['short_liquidation_usd']

            # Check if we have the required columns now
            required_cols = ['exchange_name', 'total_liquidation_usd', 'long_liquidation_usd', 'short_liquidation_usd']
            if all(col in exchange_liq_df.columns for col in required_cols):
                # Sort by total liquidation
                exchange_liq_df = exchange_liq_df.sort_values(by='total_liquidation_usd', ascending=False)

                # Create table
                create_formatted_table(
                    exchange_liq_df,
                    format_dict={
                        'short_liquidation_usd': lambda x: format_currency(x, abbreviate=True),
                        'long_liquidation_usd': lambda x: format_currency(x, abbreviate=True),
                        'total_liquidation_usd': lambda x: format_currency(x, abbreviate=True)
                    }
                )

                # Create pie chart for liquidation distribution
                fig = create_pie_chart(
                    exchange_liq_df.head(10),  # Top 10 exchanges
                    'total_liquidation_usd',
                    'exchange_name',
                    f"{asset} Liquidations Distribution by Exchange"
                )

                display_chart(fig)

                # Create stacked bar chart showing long vs short by exchange
                st.subheader("Long vs. Short Liquidations by Exchange")

                top_exchanges = exchange_liq_df.head(10).copy()  # Top 10 exchanges

                # Melt dataframe for stacked bar chart
                melted_df = pd.melt(
                    top_exchanges,
                    id_vars=['exchange_name'],
                    value_vars=['long_liquidation_usd', 'short_liquidation_usd'],
                    var_name='liquidation_type',
                    value_name='liquidation_usd'
                )

                # Create stacked bar chart
                fig = px.bar(
                    melted_df,
                    x='exchange_name',
                    y='liquidation_usd',
                    color='liquidation_type',
                    title=f"{asset} Long vs. Short Liquidations by Exchange",
                    color_discrete_map={
                        'long_liquidation_usd': ASSET_COLORS.get(asset, '#3366CC'),
                        'short_liquidation_usd': 'red'
                    },
                    labels={
                        'liquidation_type': 'Liquidation Type',
                        'liquidation_usd': 'Liquidation Volume (USD)',
                        'exchange_name': 'Exchange'
                    }
                )

                # Rename legend items
                fig.for_each_trace(lambda t: t.update(
                    name=t.name.replace('long_liquidation_usd', 'Long').replace('short_liquidation_usd', 'Short')
                ))

                display_chart(apply_chart_theme(fig))
            else:
                st.warning(f"Exchange liquidation data is missing required columns. Available columns: {list(exchange_liq_df.columns)}")
        except Exception as e:
            logger.error(f"Error processing exchange liquidation data: {e}")
            st.error(f"Error processing exchange liquidation data: {e}")
    else:
        st.info(f"No exchange liquidation data available for {asset}.")

    # Liquidation order data if available - try multiple possible key formats
    liquidation_orders_df = None
    possible_order_keys = [
        'api_futures_liquidation_order',
        f'api_futures_liquidation_order_{asset}',
        'api_futures_liquidation_orders'
    ]

    # Try each key format until we find data
    for key in possible_order_keys:
        if key in data and not data[key].empty:
            liquidation_orders_df = data[key]
            logger.info(f"Found liquidation orders data using key: {key}")
            break

    if liquidation_orders_df is not None and not liquidation_orders_df.empty:
        try:
            st.subheader("Recent Liquidation Orders")

            # Try different timestamp column names
            timestamp_cols = ['created_time', 'time', 'timestamp', 'created_at']
            timestamp_col = next((col for col in timestamp_cols if col in liquidation_orders_df.columns), None)

            if timestamp_col:
                # Process dataframe
                liquidation_orders_df = process_timestamps(liquidation_orders_df, timestamp_col)

                # Filter for selected asset
                if 'symbol' in liquidation_orders_df.columns:
                    asset_orders = liquidation_orders_df[liquidation_orders_df['symbol'].str.contains(asset, case=False, na=False)]

                    if not asset_orders.empty:
                        # Sort by time (newest first)
                        if 'datetime' in asset_orders.columns:
                            asset_orders = asset_orders.sort_values(by='datetime', ascending=False)

                        # Determine which columns to display based on what's available
                        display_cols = ['symbol', 'datetime']
                        format_dict = {'datetime': lambda x: format_timestamp(x, "%Y-%m-%d %H:%M:%S")}

                        if 'price' in asset_orders.columns:
                            display_cols.append('price')
                            format_dict['price'] = lambda x: format_currency(x, precision=2)

                        if 'qty' in asset_orders.columns:
                            display_cols.append('qty')
                            format_dict['qty'] = lambda x: f"{x:.4f}"

                        if 'qty_usd' in asset_orders.columns:
                            display_cols.append('qty_usd')
                            format_dict['qty_usd'] = lambda x: format_currency(x, abbreviate=True)

                        if 'side' in asset_orders.columns:
                            display_cols.append('side')

                        if 'exchange_name' in asset_orders.columns:
                            display_cols.append('exchange_name')

                        # Create table with available columns
                        create_formatted_table(
                            asset_orders[display_cols],
                            format_dict=format_dict
                        )
                    else:
                        st.info(f"No liquidation orders available for {asset}.")
                else:
                    st.warning("Liquidation order data is missing symbol column.")
            else:
                st.warning("Liquidation order data is missing timestamp column.")
        except Exception as e:
            logger.error(f"Error processing liquidation orders: {e}")
            st.error(f"Error processing liquidation orders: {e}")
    else:
        st.info("No liquidation order data available.")

def render_open_interest_page(asset):
    """Render the open interest page for the specified asset."""
    st.header(f"{asset} Open Interest Analysis")
    
    # Load open interest data
    data = load_futures_data('open_interest', asset)
    
    if not data:
        st.error(f"No open interest data available for {asset}.")
        return
    
    # Open interest by exchange - try multiple possible key formats
    oi_exchange_df = None
    possible_keys = [
        f"api_futures_openInterest_exchange_list_{asset}_{asset}",  # Double asset format
        f"api_futures_openInterest_exchange_list_{asset}",          # Single asset format
        "api_futures_openInterest_exchange_list"                    # Generic format
    ]

    for key in possible_keys:
        if key in data and not data[key].empty:
            oi_exchange_df = data[key]

            # If it's a generic format, filter for the specific asset
            if key == "api_futures_openInterest_exchange_list" and 'symbol' in oi_exchange_df.columns:
                oi_exchange_df = oi_exchange_df[oi_exchange_df['symbol'].str.contains(asset, case=False, na=False)]

            # If we found data, break the loop
            if not oi_exchange_df.empty:
                break

    if oi_exchange_df is not None and not oi_exchange_df.empty:
        # Process the dataframe to ensure column names are consistent
        # Rename columns if needed
        column_mapping = {}

        # Check if we need to rename exchange to exchange_name
        if 'exchange' in oi_exchange_df.columns and 'exchange_name' not in oi_exchange_df.columns:
            column_mapping['exchange'] = 'exchange_name'

        # Apply renaming if needed
        if column_mapping:
            oi_exchange_df = oi_exchange_df.rename(columns=column_mapping)

        # Calculate metrics
        if 'open_interest_usd' in oi_exchange_df.columns:
            # Convert to numeric to ensure proper summing
            oi_exchange_df['open_interest_usd'] = pd.to_numeric(oi_exchange_df['open_interest_usd'], errors='coerce')
            total_oi = oi_exchange_df['open_interest_usd'].sum()

            # Add market share percent if it doesn't exist
            if 'market_share_percent' not in oi_exchange_df.columns:
                oi_exchange_df['market_share_percent'] = (oi_exchange_df['open_interest_usd'] / total_oi * 100)

            metrics = {
                "Total Open Interest": {
                    "value": total_oi,
                    "delta": None
                }
            }

            formatters = {
                "Total Open Interest": lambda x: format_currency(x, abbreviate=True)
            }

            display_metrics_row(metrics, formatters)

            # Open interest by exchange
            st.subheader(f"{asset} Open Interest by Exchange")

            # Sort by open interest
            oi_exchange_df = oi_exchange_df.sort_values(by='open_interest_usd', ascending=False)

            # Determine which columns to display
            display_columns = ['exchange_name', 'open_interest_usd']
            if 'market_share_percent' in oi_exchange_df.columns:
                display_columns.append('market_share_percent')

            # Add change columns if they exist
            for col in oi_exchange_df.columns:
                if 'change_percent' in col:
                    display_columns.append(col)

            # Create format dictionary based on available columns
            format_dict = {'open_interest_usd': lambda x: format_currency(x, abbreviate=True)}
            if 'market_share_percent' in oi_exchange_df.columns:
                format_dict['market_share_percent'] = lambda x: format_percentage(x)

            # Add formatters for change columns
            for col in display_columns:
                if 'change_percent' in col:
                    format_dict[col] = lambda x: format_percentage(x)

            # Create table with available columns
            create_formatted_table(
                oi_exchange_df[display_columns],
                format_dict=format_dict
            )

            # Create pie chart for open interest distribution
            fig = create_pie_chart(
                oi_exchange_df.head(10),  # Top 10 exchanges
                'open_interest_usd',
                'exchange_name',
                f"{asset} Open Interest Distribution by Exchange"
            )
            
            display_chart(fig)
    else:
        st.info(f"No exchange open interest data available for {asset}.")
    
    # Open interest history
    st.subheader(f"{asset} Open Interest History")

    oi_type = st.radio(
        "Open Interest Type",
        ["Total", "Coin-Margined", "Stablecoin-Margined"],
        horizontal=True
    )

    # Try multiple possible key formats for each type
    oi_history_df = None
    possible_history_keys = []

    if oi_type == "Total":
        possible_history_keys = [
            f"api_futures_openInterest_ohlc_aggregated_history_{asset}_{asset}",
            f"api_futures_openInterest_ohlc_aggregated_history_{asset}",
            "api_futures_openInterest_ohlc_aggregated_history"
        ]
    elif oi_type == "Coin-Margined":
        possible_history_keys = [
            f"api_futures_openInterest_ohlc_aggregated_coin_margin_history_{asset}_{asset}",
            f"api_futures_openInterest_ohlc_aggregated_coin_margin_history_{asset}",
            "api_futures_openInterest_ohlc_aggregated_coin_margin_history"
        ]
    else:  # Stablecoin-Margined
        possible_history_keys = [
            f"api_futures_openInterest_ohlc_aggregated_stablecoin_{asset}_{asset}",
            f"api_futures_openInterest_ohlc_aggregated_stablecoin_{asset}",
            "api_futures_openInterest_ohlc_aggregated_stablecoin"
        ]

    # Try each key format until we find data
    oi_history_df = None
    for key in possible_history_keys:
        if key in data and not data[key].empty:
            oi_history_df = data[key]
            # If we found data, break the loop
            if not oi_history_df.empty:
                break

    # Check if we found historical data
    if oi_history_df is not None and not oi_history_df.empty:
        # Process dataframe
        oi_history_df = process_timestamps(oi_history_df)

        # Check required columns
        required_cols = ['datetime', 'open', 'high', 'low', 'close']
        if all(col in oi_history_df.columns for col in required_cols):
            # Create OHLC chart
            fig = create_ohlc_chart(
                oi_history_df,
                'datetime',
                'open',
                'high',
                'low',
                'close',
                f"{asset} {oi_type} Open Interest History"
            )

            display_chart(fig)

            # Add price overlay if available
            if 'api_price_ohlc_history' in data.get('market', {}):
                price_df = data['market']['api_price_ohlc_history']

                if not price_df.empty:
                    # Process dataframe
                    price_df = process_timestamps(price_df)

                    if 'datetime' in price_df.columns and 'close' in price_df.columns:
                        try:
                            # OI vs Price chart
                            st.subheader(f"{asset} Open Interest vs. Price")

                            # Sort dataframes
                            oi_sorted = oi_history_df.sort_values('datetime')
                            price_sorted = price_df[['datetime', 'close']].sort_values('datetime')

                            # Merge data on datetime
                            merged_df = pd.merge_asof(
                                oi_sorted,
                                price_sorted,
                                on='datetime',
                                direction='nearest',
                                suffixes=('_oi', '_price')
                            )

                            if not merged_df.empty:
                                # Create dual-axis chart
                                fig = make_subplots(specs=[[{"secondary_y": True}]])

                                # Add open interest line
                                fig.add_trace(
                                    go.Scatter(
                                        x=merged_df['datetime'],
                                        y=merged_df['close_oi'],
                                        name='Open Interest',
                                        line=dict(color='#3366CC')
                                    ),
                                    secondary_y=False
                                )

                                # Add price line
                                fig.add_trace(
                                    go.Scatter(
                                        x=merged_df['datetime'],
                                        y=merged_df['close_price'],
                                        name='Price',
                                        line=dict(color=ASSET_COLORS.get(asset, '#FF9900'))
                                    ),
                                    secondary_y=True
                                )

                                # Update layout
                                fig.update_layout(
                                    title=f"{asset} Open Interest vs. Price",
                                    hovermode="x unified"
                                )

                                # Set axis titles
                                fig.update_yaxes(title_text="Open Interest (USD)", secondary_y=False)
                                fig.update_yaxes(title_text="Price (USD)", secondary_y=True)

                                display_chart(apply_chart_theme(fig))
                            else:
                                logger.warning(f"Merged dataframe is empty for {asset} OI vs Price")
                        except Exception as e:
                            logger.error(f"Error creating OI vs Price chart: {e}")
                            st.warning(f"Could not create price overlay chart: {e}")
                    else:
                        logger.warning(f"Price data missing required columns for {asset}")
        else:
            available_cols = list(oi_history_df.columns)
            logger.warning(f"OI history data missing required columns. Available: {available_cols}")
            st.warning(f"Open interest history data is missing required columns. Available: {available_cols}")
    else:
        # If we couldn't find any data
        st.info(f"No {oi_type.lower()} open interest history data available for {asset}.")
    
    # Open interest history by exchange if available
    # Try multiple possible key formats
    oi_exchange_history_df = None
    possible_exchange_keys = [
        f"api_futures_openInterest_exchange_history_chart_{asset}_{asset}",
        f"api_futures_openInterest_exchange_history_chart_{asset}",
        "api_futures_openInterest_exchange_history_chart"
    ]

    for key in possible_exchange_keys:
        if key in data and not data[key].empty:
            oi_exchange_history_df = data[key]
            # If it's a generic format, filter for the specific asset
            if key == "api_futures_openInterest_exchange_history_chart" and 'symbol' in oi_exchange_history_df.columns:
                oi_exchange_history_df = oi_exchange_history_df[oi_exchange_history_df['symbol'].str.contains(asset, case=False, na=False)]

            # If we found data, break the loop
            if not oi_exchange_history_df.empty:
                break

    if oi_exchange_history_df is not None and not oi_exchange_history_df.empty:
        try:
            st.subheader(f"{asset} Open Interest History by Exchange")

            # Process dataframe
            oi_exchange_history_df = process_timestamps(oi_exchange_history_df)

            # Check required columns
            required_cols = ['datetime', 'exchange_name', 'open_interest_usd']
            column_mapping = {}

            # Map column names if they don't match expected format
            if 'exchange' in oi_exchange_history_df.columns and 'exchange_name' not in oi_exchange_history_df.columns:
                column_mapping['exchange'] = 'exchange_name'

            # Apply column renaming if needed
            if column_mapping:
                oi_exchange_history_df = oi_exchange_history_df.rename(columns=column_mapping)

            # Recheck required columns after potential renaming
            if all(col in oi_exchange_history_df.columns for col in required_cols):
                # Get top exchanges
                try:
                    # Create pivot table
                    pivot_df = oi_exchange_history_df.pivot(index='datetime', columns='exchange_name', values='open_interest_usd')

                    # Calculate average OI for ranking
                    avg_oi = pivot_df.mean()
                    top_exchanges = avg_oi.sort_values(ascending=False).head(5).index.tolist()

                    # Check if we have enough exchanges to display
                    if len(top_exchanges) > 0:
                        # Create multi-line chart for top exchanges
                        fig = px.line(
                            pivot_df,
                            x=pivot_df.index,
                            y=top_exchanges,
                            title=f"{asset} Open Interest History by Exchange (Top {len(top_exchanges)})"
                        )

                        display_chart(apply_chart_theme(fig))
                    else:
                        st.warning("No exchange data available after pivoting")
                except Exception as e:
                    logger.error(f"Error creating exchange history pivot: {e}")
                    st.warning(f"Could not create exchange history visualization: {e}")
            else:
                available_cols = list(oi_exchange_history_df.columns)
                st.warning(f"Exchange history data missing required columns. Available: {available_cols}")
        except Exception as e:
            logger.error(f"Error processing exchange history data: {e}")
            st.warning(f"Could not process exchange history data: {e}")
    else:
        st.info(f"No open interest history by exchange data available for {asset}.")

def render_long_short_ratio_page(asset):
    """Render the long/short ratio page for the specified asset."""
    st.header(f"{asset} Long/Short Ratio Analysis")

    # Load long/short ratio data
    data = load_futures_data('long_short_ratio', asset)

    if not data:
        st.info(f"No long/short ratio data available for {asset}. Long/short ratio shows the balance between long and short positions in the market.")

        # Show empty placeholder layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Buy Volume", "$0")
        with col2:
            st.metric("Sell Volume", "$0")
        with col3:
            st.metric("Buy/Sell Ratio", "0.00")

        st.subheader(f"{asset} Taker Buy/Sell by Exchange")
        st.info("No taker buy/sell data available. This section would show the buy/sell volumes and ratios across different exchanges.")
        return

    # Display tabs for different ratio types
    tab1, tab2, tab3 = st.tabs([
        "Taker Buy/Sell Ratio",
        "Account Ratio",
        "Position Ratio"
    ])

    with tab1:
        # Taker buy/sell volume by exchange - try multiple possible key formats
        taker_exchange_df = None
        possible_exchange_keys = [
            f"api_futures_taker_buy_sell_volume_exchange_list_{asset}_{asset}",  # Double asset format
            f"api_futures_taker_buy_sell_volume_exchange_list_{asset}",          # Single asset format
            "api_futures_taker_buy_sell_volume_exchange_list"                    # Generic format
        ]

        # Try each key format until we find data
        for key in possible_exchange_keys:
            if key in data and not data[key].empty:
                taker_exchange_df = data[key]

                # If it's a generic format, filter for the specific asset
                if key == "api_futures_taker_buy_sell_volume_exchange_list" and 'symbol' in taker_exchange_df.columns:
                    taker_exchange_df = taker_exchange_df[taker_exchange_df['symbol'].str.contains(asset, case=False, na=False)]

                logger.info(f"Found taker buy/sell exchange data using key: {key}")
                break

        if taker_exchange_df is not None and not taker_exchange_df.empty:
            try:
                # Check column names and map if needed
                column_mapping = {}

                # Check for different column naming patterns
                if 'exchange' in taker_exchange_df.columns and 'exchange_name' not in taker_exchange_df.columns:
                    column_mapping['exchange'] = 'exchange_name'

                if 'buy_vol_usd' in taker_exchange_df.columns and 'buy_volume' not in taker_exchange_df.columns:
                    column_mapping['buy_vol_usd'] = 'buy_volume'

                if 'sell_vol_usd' in taker_exchange_df.columns and 'sell_volume' not in taker_exchange_df.columns:
                    column_mapping['sell_vol_usd'] = 'sell_volume'

                if 'buy_ratio' in taker_exchange_df.columns and 'buy_sell_ratio' not in taker_exchange_df.columns:
                    # In some formats, we may need to calculate the ratio
                    if 'sell_ratio' in taker_exchange_df.columns:
                        taker_exchange_df['buy_sell_ratio'] = taker_exchange_df['buy_ratio'] / taker_exchange_df['sell_ratio'].replace(0, float('nan'))

                # Handle nested exchange_list data if present
                if 'exchange_list' in taker_exchange_df.columns:
                    try:
                        logger.info("Found nested exchange_list data, attempting to extract")

                        # Extract exchange-specific data from the nested list
                        exchanges_data = []
                        for idx, row in taker_exchange_df.iterrows():
                            exchange_list = row.get('exchange_list')

                            # If exchange_list is NaN, skip
                            if pd.isna(exchange_list):
                                continue

                            # Handle numpy arrays
                            if hasattr(exchange_list, '__array__'):
                                try:
                                    exchange_list = exchange_list.tolist()
                                except Exception as e_convert:
                                    logger.warning(f"Could not convert exchange_list to list: {e_convert}")
                                    continue

                            # Process list of exchange data
                            if isinstance(exchange_list, list):
                                for exchange_data in exchange_list:
                                    if isinstance(exchange_data, dict):
                                        exchange_item = {
                                            'symbol': row.get('symbol', ''),
                                            'exchange_name': exchange_data.get('exchange', '')
                                        }

                                        # Map buy/sell volumes and ratio
                                        if 'buy_vol' in exchange_data:
                                            exchange_item['buy_volume'] = exchange_data['buy_vol']

                                        if 'sell_vol' in exchange_data:
                                            exchange_item['sell_volume'] = exchange_data['sell_vol']

                                        if 'buy_vol' in exchange_data and 'sell_vol' in exchange_data and exchange_data['sell_vol'] != 0:
                                            exchange_item['buy_sell_ratio'] = exchange_data['buy_vol'] / exchange_data['sell_vol']

                                        exchanges_data.append(exchange_item)

                        # If we extracted data successfully, replace the original dataframe
                        if exchanges_data:
                            taker_exchange_df = pd.DataFrame(exchanges_data)
                            logger.info(f"Successfully extracted exchange data from nested list: {len(taker_exchange_df)} rows")
                    except Exception as e:
                        logger.error(f"Error processing nested exchange_list: {e}")

                # Apply column renaming if needed
                if column_mapping:
                    taker_exchange_df = taker_exchange_df.rename(columns=column_mapping)

                # If exchange_name is missing but we have exchange_list, create exchange_name from it
                if 'exchange_name' not in taker_exchange_df.columns and 'exchange_list' in taker_exchange_df.columns:
                    try:
                        # Add an exchange_name column based on the exchange_list data
                        taker_exchange_df = taker_exchange_df.copy()  # Create a copy to avoid SettingWithCopyWarning

                        # For each row, extract exchange name from exchange_list if it's a dictionary
                        exchange_names = []
                        for i, row in taker_exchange_df.iterrows():
                            if isinstance(row['exchange_list'], dict) and 'exchange' in row['exchange_list']:
                                exchange_names.append(row['exchange_list']['exchange'])
                            else:
                                exchange_names.append("All Exchanges")

                        taker_exchange_df['exchange_name'] = exchange_names
                        logger.info("Added exchange_name from exchange_list dictionary data")
                    except Exception as e:
                        logger.error(f"Error adding exchange_name from exchange_list: {e}")
                        taker_exchange_df['exchange_name'] = "All Exchanges"  # Default value

                # Map buy/sell volume columns if needed
                if 'buy_volume' not in taker_exchange_df.columns:
                    if 'buy_vol_usd' in taker_exchange_df.columns:
                        taker_exchange_df['buy_volume'] = taker_exchange_df['buy_vol_usd']
                    elif 'buy_vol' in taker_exchange_df.columns:
                        taker_exchange_df['buy_volume'] = taker_exchange_df['buy_vol']

                if 'sell_volume' not in taker_exchange_df.columns:
                    if 'sell_vol_usd' in taker_exchange_df.columns:
                        taker_exchange_df['sell_volume'] = taker_exchange_df['sell_vol_usd']
                    elif 'sell_vol' in taker_exchange_df.columns:
                        taker_exchange_df['sell_volume'] = taker_exchange_df['sell_vol']

                # Check if we have the required columns now
                required_cols = ['exchange_name', 'buy_volume', 'sell_volume']
                missing_cols = [col for col in required_cols if col not in taker_exchange_df.columns]

                if missing_cols:
                    st.warning(f"Taker buy/sell data is missing required columns: {missing_cols}. Available columns: {list(taker_exchange_df.columns)}")
                else:
                    # Calculate buy_sell_ratio if it doesn't exist
                    if 'buy_sell_ratio' not in taker_exchange_df.columns:
                        taker_exchange_df['buy_sell_ratio'] = taker_exchange_df['buy_volume'] / taker_exchange_df['sell_volume'].replace(0, float('nan'))

                    # Calculate metrics
                    total_buy = taker_exchange_df['buy_volume'].sum()
                    total_sell = taker_exchange_df['sell_volume'].sum()
                    total_ratio = total_buy / total_sell if total_sell != 0 else None

                    metrics = {
                        "Buy Volume": {
                            "value": total_buy,
                            "delta": None
                        },
                        "Sell Volume": {
                            "value": total_sell,
                            "delta": None
                        },
                        "Buy/Sell Ratio": {
                            "value": total_ratio,
                            "delta": None
                        }
                    }

                    formatters = {
                        "Buy Volume": lambda x: format_currency(x, abbreviate=True),
                        "Sell Volume": lambda x: format_currency(x, abbreviate=True),
                        "Buy/Sell Ratio": lambda x: f"{x:.4f}" if x is not None else "N/A"
                    }

                    display_metrics_row(metrics, formatters)

                    # Taker buy/sell by exchange
                    st.subheader(f"{asset} Taker Buy/Sell by Exchange")

                    # Sort by volume
                    taker_exchange_df['total_volume'] = taker_exchange_df['buy_volume'] + taker_exchange_df['sell_volume']
                    taker_exchange_df = taker_exchange_df.sort_values(by='total_volume', ascending=False)

                    # Create table
                    display_columns = ['exchange_name', 'buy_volume', 'sell_volume', 'buy_sell_ratio']
                    display_df = taker_exchange_df[display_columns].copy()

                    create_formatted_table(
                        display_df,
                        format_dict={
                            'buy_volume': lambda x: format_currency(x, abbreviate=True),
                            'sell_volume': lambda x: format_currency(x, abbreviate=True),
                            'buy_sell_ratio': lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                        }
                    )

                    # Create bar chart for buy/sell ratio by exchange
                    fig = px.bar(
                        taker_exchange_df.head(10),  # Top 10 exchanges
                        x='exchange_name',
                        y='buy_sell_ratio',
                        title=f"{asset} Buy/Sell Ratio by Exchange",
                        color='buy_sell_ratio',
                        color_continuous_scale='RdYlGn',
                        color_continuous_midpoint=1
                    )

                    # Add reference line at 1 (equal buy and sell)
                    fig.add_hline(
                        y=1,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text="Equal"
                    )

                    display_chart(apply_chart_theme(fig))
            except Exception as e:
                logger.error(f"Error processing taker buy/sell exchange data: {e}")
                st.error(f"Error processing taker buy/sell data: {e}")
        else:
            st.info(f"No taker buy/sell data by exchange available for {asset}.")

        # Taker buy/sell volume history - try multiple possible key formats
        taker_history_df = None
        possible_history_keys = [
            f"api_futures_taker_buy_sell_volume_history_{asset}_{asset}",  # Double asset format
            f"api_futures_taker_buy_sell_volume_history_{asset}",          # Single asset format
            "api_futures_taker_buy_sell_volume_history"                    # Generic format
        ]

        # Try each key format until we find data
        for key in possible_history_keys:
            if key in data and not data[key].empty:
                taker_history_df = data[key]

                # If it's a generic format, filter for the specific asset
                if key == "api_futures_taker_buy_sell_volume_history" and 'symbol' in taker_history_df.columns:
                    taker_history_df = taker_history_df[taker_history_df['symbol'].str.contains(asset, case=False, na=False)]

                logger.info(f"Found taker buy/sell history data using key: {key}")
                break

        if taker_history_df is not None and not taker_history_df.empty:
            try:
                # Process dataframe
                taker_history_df = process_timestamps(taker_history_df)

                # Check column names and map if needed
                column_mapping = {}

                # Check for different column naming patterns
                if 'taker_buy_volume_usd' in taker_history_df.columns and 'buy_volume' not in taker_history_df.columns:
                    column_mapping['taker_buy_volume_usd'] = 'buy_volume'

                if 'taker_sell_volume_usd' in taker_history_df.columns and 'sell_volume' not in taker_history_df.columns:
                    column_mapping['taker_sell_volume_usd'] = 'sell_volume'

                # Apply column renaming if needed
                if column_mapping:
                    taker_history_df = taker_history_df.rename(columns=column_mapping)

                # Map additional columns if needed (check for all potential variants)
                if 'buy_volume' not in taker_history_df.columns:
                    if 'buy_vol_usd' in taker_history_df.columns:
                        taker_history_df['buy_volume'] = taker_history_df['buy_vol_usd']

                if 'sell_volume' not in taker_history_df.columns:
                    if 'sell_vol_usd' in taker_history_df.columns:
                        taker_history_df['sell_volume'] = taker_history_df['sell_vol_usd']

                # Calculate buy_sell_ratio if it doesn't exist
                if 'buy_sell_ratio' not in taker_history_df.columns and 'buy_volume' in taker_history_df.columns and 'sell_volume' in taker_history_df.columns:
                    taker_history_df['buy_sell_ratio'] = taker_history_df['buy_volume'] / taker_history_df['sell_volume'].replace(0, float('nan'))

                # Check if we have the required columns now
                required_cols = ['datetime', 'buy_volume', 'sell_volume']
                missing_cols = [col for col in required_cols if col not in taker_history_df.columns]

                if missing_cols:
                    st.warning(f"Taker buy/sell history data is missing required columns: {missing_cols}. Available columns: {list(taker_history_df.columns)}")
                else:
                    # Create stacked bar chart for buy/sell volume
                    st.subheader(f"{asset} Taker Buy/Sell Volume History")

                    fig = go.Figure()

                    fig.add_trace(go.Bar(
                        x=taker_history_df['datetime'],
                        y=taker_history_df['buy_volume'],
                        name='Buy Volume',
                        marker_color='green'
                    ))

                    fig.add_trace(go.Bar(
                        x=taker_history_df['datetime'],
                        y=taker_history_df['sell_volume'],
                        name='Sell Volume',
                        marker_color='red'
                    ))

                    # Update layout
                    fig.update_layout(
                        title=f"{asset} Taker Buy/Sell Volume History",
                        barmode='group',
                        xaxis_title=None,
                        yaxis_title="Volume (USD)",
                        hovermode="x unified"
                    )

                    display_chart(apply_chart_theme(fig))

                    if 'buy_sell_ratio' in taker_history_df.columns:
                        # Create buy/sell ratio chart
                        st.subheader(f"{asset} Buy/Sell Ratio History")

                        fig = px.line(
                            taker_history_df,
                            x='datetime',
                            y='buy_sell_ratio',
                            title=f"{asset} Taker Buy/Sell Ratio History"
                        )

                        # Add reference line at 1 (equal buy and sell)
                        fig.add_hline(
                            y=1,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Equal"
                        )

                        # Add price overlay if available
                        if 'price' in taker_history_df.columns:
                            fig.add_trace(
                                go.Scatter(
                                    x=taker_history_df['datetime'],
                                    y=taker_history_df['price'],
                                    name='Price',
                                    yaxis="y2",
                                    line=dict(color=ASSET_COLORS.get(asset, '#FF9900'))
                                )
                            )

                            fig.update_layout(
                                yaxis2=dict(
                                    title="Price (USD)",
                                    overlaying="y",
                                    side="right"
                                )
                            )

                        display_chart(apply_chart_theme(fig))
            except Exception as e:
                logger.error(f"Error processing taker buy/sell history data: {e}")
                st.error(f"Error processing taker buy/sell history: {e}")
        else:
            st.info(f"No taker buy/sell volume history data available for {asset}.")

    with tab2:
        # Top traders long/short account ratio - try multiple possible key formats
        account_ratio_df = None
        possible_account_keys = [
            'api_futures_top_long_short_account_ratio_history',
            f'api_futures_top_long_short_account_ratio_history_{asset}',
            'api_futures_top_account_ratio_history'
        ]

        # Try each key format until we find data
        for key in possible_account_keys:
            if key in data and not data[key].empty:
                account_ratio_df = data[key]
                logger.info(f"Found account ratio data using key: {key}")
                break

        if account_ratio_df is not None and not account_ratio_df.empty:
            try:
                # Process timestamps (ensuring epoch timestamp in milliseconds is handled correctly)
                if 'time' in account_ratio_df.columns and 'datetime' not in account_ratio_df.columns:
                    # Direct conversion of epoch milliseconds to datetime
                    account_ratio_df['datetime'] = pd.to_datetime(account_ratio_df['time'], unit='ms')
                    logger.info(f"Converted 'time' column directly to datetime for account ratio data")
                else:
                    # Process timestamps using the standard function as fallback
                    account_ratio_df = process_timestamps(account_ratio_df)
                    logger.info(f"Used process_timestamps for account ratio data")

                # Check column names and map if needed
                column_mapping = {}

                # Check for different column naming patterns
                if 'top_account_long_short_ratio' in account_ratio_df.columns and 'long_short_ratio' not in account_ratio_df.columns:
                    column_mapping['top_account_long_short_ratio'] = 'long_short_ratio'

                # Apply column renaming if needed
                if column_mapping:
                    account_ratio_df = account_ratio_df.rename(columns=column_mapping)

                # Make sure percent columns are properly processed
                if 'top_account_long_percent' in account_ratio_df.columns:
                    # Convert to float if not already
                    account_ratio_df['top_account_long_percent'] = account_ratio_df['top_account_long_percent'].astype(float)
                    
                if 'top_account_short_percent' in account_ratio_df.columns:
                    # Convert to float if not already
                    account_ratio_df['top_account_short_percent'] = account_ratio_df['top_account_short_percent'].astype(float)

                # Calculate ratio if it doesn't exist but we have the components
                if 'long_short_ratio' not in account_ratio_df.columns and 'top_account_long_percent' in account_ratio_df.columns and 'top_account_short_percent' in account_ratio_df.columns:
                    account_ratio_df['long_short_ratio'] = account_ratio_df['top_account_long_percent'] / account_ratio_df['top_account_short_percent'].replace(0, float('nan'))
                
                # If we have a ratio column, make sure it's properly processed
                if 'long_short_ratio' in account_ratio_df.columns or 'top_account_long_short_ratio' in account_ratio_df.columns:
                    ratio_col = 'long_short_ratio' if 'long_short_ratio' in account_ratio_df.columns else 'top_account_long_short_ratio'
                    # Convert to float if not already
                    account_ratio_df[ratio_col] = account_ratio_df[ratio_col].astype(float)

                # Create metrics for account ratio
                if 'top_account_long_percent' in account_ratio_df.columns and 'top_account_short_percent' in account_ratio_df.columns:
                    # Get the most recent values
                    latest_data = account_ratio_df.iloc[-1]
                    
                    long_percent = latest_data.get('top_account_long_percent', 0)
                    short_percent = latest_data.get('top_account_short_percent', 0)
                    ratio = latest_data.get('long_short_ratio', 0)
                    
                    metrics = {
                        "Long %": {
                            "value": long_percent,
                            "delta": None,
                            "delta_suffix": "%"
                        },
                        "Short %": {
                            "value": short_percent,
                            "delta": None,
                            "delta_suffix": "%"
                        },
                        "Long/Short Ratio": {
                            "value": ratio,
                            "delta": None
                        }
                    }
                    
                    formatters = {
                        "Long %": lambda x: f"{x:.2f}%",
                        "Short %": lambda x: f"{x:.2f}%",
                        "Long/Short Ratio": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                    }
                    
                    display_metrics_row(metrics, formatters)

                # Check if we have the required columns now
                if 'datetime' in account_ratio_df.columns:
                    ratio_available = False
                    
                    # Try different ratio column names
                    if 'long_short_ratio' in account_ratio_df.columns:
                        ratio_col = 'long_short_ratio'
                        ratio_available = True
                    elif 'top_account_long_short_ratio' in account_ratio_df.columns:
                        ratio_col = 'top_account_long_short_ratio'
                        ratio_available = True
                    
                    # If neither column exists but we have long and short percentages, create ratio
                    elif 'top_account_long_percent' in account_ratio_df.columns and 'top_account_short_percent' in account_ratio_df.columns:
                        account_ratio_df['long_short_ratio'] = account_ratio_df['top_account_long_percent'] / account_ratio_df['top_account_short_percent'].replace(0, float('nan'))
                        ratio_col = 'long_short_ratio'
                        ratio_available = True
                    
                    if ratio_available:
                        # Create chart
                        st.subheader("Top Traders Long/Short Account Ratio")
                        
                        # Drop NaN values to prevent chart errors
                        chart_df = account_ratio_df.dropna(subset=['datetime', ratio_col])
                        
                        if not chart_df.empty:
                            fig = px.line(
                                chart_df,
                                x='datetime',
                                y=ratio_col,
                                title="Top Traders Long/Short Account Ratio"
                            )
                        else:
                            st.warning("Not enough valid data points to display the account ratio chart.")
                            # Skip the rest of this section
                            ratio_available = False

                    # Only add reference line if the chart exists and ratio is available
                    if ratio_available and 'fig' in locals():
                        fig.add_hline(
                            y=1,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Equal"
                        )

                        # Add price overlay if available and chart exists
                        if 'price' in account_ratio_df.columns:
                            # Make sure we have valid price data
                            price_df = account_ratio_df.dropna(subset=['datetime', 'price'])
                            if not price_df.empty:
                                fig.add_trace(
                                    go.Scatter(
                                        x=price_df['datetime'],
                                        y=price_df['price'],
                                        name='Price',
                                        yaxis="y2",
                                        line=dict(color=ASSET_COLORS.get(asset, '#FF9900'))
                                    )
                                )

                                fig.update_layout(
                                    yaxis2=dict(
                                        title="Price (USD)",
                                        overlaying="y",
                                        side="right"
                                    )
                                )

                        # Display the chart if it exists and ratio is available
                        display_chart(apply_chart_theme(fig))
                    
                    # Display Long vs Short percentage chart as well
                    if 'top_account_long_percent' in account_ratio_df.columns and 'top_account_short_percent' in account_ratio_df.columns:
                        st.subheader("Top Traders Long/Short Account Percentages")
                        
                        try:
                            # Create a dataframe for plotting long and short percentages
                            plot_df = account_ratio_df[['datetime', 'top_account_long_percent', 'top_account_short_percent']].copy()
                            
                            # Ensure data is clean - drop NaN values
                            plot_df = plot_df.dropna(subset=['datetime', 'top_account_long_percent', 'top_account_short_percent'])
                            
                            if not plot_df.empty and len(plot_df) > 1:  # Need at least 2 points for a line
                                try:
                                    # Create the figure with two lines
                                    fig_pct = go.Figure()
                                    
                                    fig_pct.add_trace(go.Scatter(
                                        x=plot_df['datetime'],
                                        y=plot_df['top_account_long_percent'],
                                        name='Long %',
                                        line=dict(color='green', width=2)
                                    ))
                                    
                                    fig_pct.add_trace(go.Scatter(
                                        x=plot_df['datetime'],
                                        y=plot_df['top_account_short_percent'],
                                        name='Short %',
                                        line=dict(color='red', width=2)
                                    ))
                                    
                                    fig_pct.update_layout(
                                        title="Top Traders Account Long/Short Percentages",
                                        xaxis_title=None,
                                        yaxis_title="Percentage (%)",
                                        hovermode="x unified"
                                    )
                                    
                                    display_chart(apply_chart_theme(fig_pct))
                                except Exception as e:
                                    logger.error(f"Error creating percentage chart: {e}")
                                    st.warning("Error displaying percentage chart.")
                            else:
                                st.warning("Not enough valid data points to display the percentage chart.")
                        except Exception as e:
                            logger.error(f"Error creating account percentages chart: {e}")
                            st.error(f"Could not create percentage chart: {e}")
                else:
                    st.warning(f"Account ratio data is missing required columns. Available columns: {list(account_ratio_df.columns)}")
            except Exception as e:
                logger.error(f"Error processing account ratio data: {e}")
                st.error(f"Error processing account ratio data: {e}")
        else:
            st.info("No top traders account ratio data available.")

        # Global long/short account ratio - try multiple possible key formats
        global_account_ratio_df = None
        possible_global_keys = [
            'api_futures_global_long_short_account_ratio_history',
            f'api_futures_global_long_short_account_ratio_history_{asset}',
            'api_futures_global_account_ratio_history'
        ]

        # Try each key format until we find data
        for key in possible_global_keys:
            if key in data and not data[key].empty:
                global_account_ratio_df = data[key]
                logger.info(f"Found global account ratio data using key: {key}")
                break

        if global_account_ratio_df is not None and not global_account_ratio_df.empty:
            try:
                # Process timestamps (ensuring epoch timestamp in milliseconds is handled correctly)
                if 'time' in global_account_ratio_df.columns and 'datetime' not in global_account_ratio_df.columns:
                    # Direct conversion of epoch milliseconds to datetime
                    global_account_ratio_df['datetime'] = pd.to_datetime(global_account_ratio_df['time'], unit='ms')
                    logger.info(f"Converted 'time' column directly to datetime for global account ratio data")
                else:
                    # Process timestamps using the standard function as fallback
                    global_account_ratio_df = process_timestamps(global_account_ratio_df)
                    logger.info(f"Used process_timestamps for global account ratio data")

                # Check column names and map if needed
                column_mapping = {}

                # Check for different column naming patterns
                if 'global_account_long_short_ratio' in global_account_ratio_df.columns and 'long_short_ratio' not in global_account_ratio_df.columns:
                    column_mapping['global_account_long_short_ratio'] = 'long_short_ratio'

                # Apply column renaming if needed
                if column_mapping:
                    global_account_ratio_df = global_account_ratio_df.rename(columns=column_mapping)

                # Make sure percent columns are properly processed
                if 'global_account_long_percent' in global_account_ratio_df.columns:
                    # Convert to float if not already
                    global_account_ratio_df['global_account_long_percent'] = global_account_ratio_df['global_account_long_percent'].astype(float)
                    
                if 'global_account_short_percent' in global_account_ratio_df.columns:
                    # Convert to float if not already
                    global_account_ratio_df['global_account_short_percent'] = global_account_ratio_df['global_account_short_percent'].astype(float)

                # Calculate ratio if it doesn't exist but we have the components
                if 'long_short_ratio' not in global_account_ratio_df.columns and 'global_account_long_percent' in global_account_ratio_df.columns and 'global_account_short_percent' in global_account_ratio_df.columns:
                    global_account_ratio_df['long_short_ratio'] = global_account_ratio_df['global_account_long_percent'] / global_account_ratio_df['global_account_short_percent'].replace(0, float('nan'))
                
                # If we have a ratio column, make sure it's properly processed
                if 'long_short_ratio' in global_account_ratio_df.columns or 'global_account_long_short_ratio' in global_account_ratio_df.columns:
                    ratio_col = 'long_short_ratio' if 'long_short_ratio' in global_account_ratio_df.columns else 'global_account_long_short_ratio'
                    # Convert to float if not already
                    global_account_ratio_df[ratio_col] = global_account_ratio_df[ratio_col].astype(float)

                # Create metrics for global account ratio
                if 'global_account_long_percent' in global_account_ratio_df.columns and 'global_account_short_percent' in global_account_ratio_df.columns:
                    # Get the most recent values
                    latest_data = global_account_ratio_df.iloc[-1]
                    
                    long_percent = latest_data.get('global_account_long_percent', 0)
                    short_percent = latest_data.get('global_account_short_percent', 0)
                    ratio = latest_data.get('long_short_ratio', 0)
                    
                    metrics = {
                        "Long %": {
                            "value": long_percent,
                            "delta": None,
                            "delta_suffix": "%"
                        },
                        "Short %": {
                            "value": short_percent,
                            "delta": None,
                            "delta_suffix": "%"
                        },
                        "Long/Short Ratio": {
                            "value": ratio,
                            "delta": None
                        }
                    }
                    
                    formatters = {
                        "Long %": lambda x: f"{x:.2f}%",
                        "Short %": lambda x: f"{x:.2f}%",
                        "Long/Short Ratio": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                    }
                    
                    display_metrics_row(metrics, formatters)

                # Check if we have the required columns now
                if 'datetime' in global_account_ratio_df.columns:
                    ratio_available = False
                    
                    # Try different ratio column names
                    if 'long_short_ratio' in global_account_ratio_df.columns:
                        ratio_col = 'long_short_ratio'
                        ratio_available = True
                    elif 'global_account_long_short_ratio' in global_account_ratio_df.columns:
                        ratio_col = 'global_account_long_short_ratio'
                        ratio_available = True
                    
                    # If neither column exists but we have long and short percentages, create ratio
                    elif 'global_account_long_percent' in global_account_ratio_df.columns and 'global_account_short_percent' in global_account_ratio_df.columns:
                        global_account_ratio_df['long_short_ratio'] = global_account_ratio_df['global_account_long_percent'] / global_account_ratio_df['global_account_short_percent'].replace(0, float('nan'))
                        ratio_col = 'long_short_ratio'
                        ratio_available = True
                    
                    if ratio_available:
                        # Create chart
                        st.subheader("Global Long/Short Account Ratio")
                        
                        # Drop NaN values to prevent chart errors
                        chart_df = global_account_ratio_df.dropna(subset=['datetime', ratio_col])
                        
                        if not chart_df.empty:
                            fig = px.line(
                                chart_df,
                                x='datetime',
                                y=ratio_col,
                                title="Global Long/Short Account Ratio"
                            )
                        else:
                            st.warning("Not enough valid data points to display the global account ratio chart.")
                            # Skip the rest of this section
                            ratio_available = False

                    # Only add reference line if the chart exists and ratio is available
                    if ratio_available and 'fig' in locals():
                        fig.add_hline(
                            y=1,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Equal"
                        )

                        # Add price overlay if available
                        if 'price' in global_account_ratio_df.columns:
                            # Make sure we have valid price data
                            price_df = global_account_ratio_df.dropna(subset=['datetime', 'price'])
                            if not price_df.empty:
                                fig.add_trace(
                                    go.Scatter(
                                        x=price_df['datetime'],
                                        y=price_df['price'],
                                        name='Price',
                                        yaxis="y2",
                                        line=dict(color=ASSET_COLORS.get(asset, '#FF9900'))
                                    )
                                )

                                fig.update_layout(
                                    yaxis2=dict(
                                        title="Price (USD)",
                                        overlaying="y",
                                        side="right"
                                    )
                                )

                        # Display the chart if it exists
                        display_chart(apply_chart_theme(fig))
                    
                    # Display Long vs Short percentage chart as well
                    if 'global_account_long_percent' in global_account_ratio_df.columns and 'global_account_short_percent' in global_account_ratio_df.columns:
                        st.subheader("Global Long/Short Account Percentages")
                        
                        try:
                            # Create a dataframe for plotting long and short percentages
                            plot_df = global_account_ratio_df[['datetime', 'global_account_long_percent', 'global_account_short_percent']].copy()
                            
                            # Ensure data is clean - drop NaN values
                            plot_df = plot_df.dropna(subset=['datetime', 'global_account_long_percent', 'global_account_short_percent'])
                            
                            if not plot_df.empty and len(plot_df) > 1:  # Need at least 2 points for a line
                                try:
                                    # Create the figure with two lines
                                    fig_pct = go.Figure()
                                    
                                    fig_pct.add_trace(go.Scatter(
                                        x=plot_df['datetime'],
                                        y=plot_df['global_account_long_percent'],
                                        name='Long %',
                                        line=dict(color='green', width=2)
                                    ))
                                    
                                    fig_pct.add_trace(go.Scatter(
                                        x=plot_df['datetime'],
                                        y=plot_df['global_account_short_percent'],
                                        name='Short %',
                                        line=dict(color='red', width=2)
                                    ))
                                    
                                    fig_pct.update_layout(
                                        title="Global Account Long/Short Percentages",
                                        xaxis_title=None,
                                        yaxis_title="Percentage (%)",
                                        hovermode="x unified"
                                    )
                                    
                                    display_chart(apply_chart_theme(fig_pct))
                                except Exception as e:
                                    logger.error(f"Error creating global percentage chart: {e}")
                                    st.warning("Error displaying global percentage chart.")
                            else:
                                st.warning("Not enough valid data points to display the global percentage chart.")
                        except Exception as e:
                            logger.error(f"Error processing global account percentages chart: {e}")
                            st.error(f"Could not create global percentage chart: {e}")
                else:
                    st.warning(f"Global account ratio data is missing required columns. Available columns: {list(global_account_ratio_df.columns)}")
            except Exception as e:
                logger.error(f"Error processing global account ratio data: {e}")
                st.error(f"Error processing global account ratio data: {e}")
        else:
            st.info("No global long/short account ratio data available.")

    with tab3:
        # Top traders long/short position ratio - try multiple possible key formats
        position_ratio_df = None
        possible_position_keys = [
            'api_futures_top_long_short_position_ratio_history',
            f'api_futures_top_long_short_position_ratio_history_{asset}',
            'api_futures_top_position_ratio_history'
        ]

        # Try each key format until we find data
        for key in possible_position_keys:
            if key in data and not data[key].empty:
                position_ratio_df = data[key]
                logger.info(f"Found position ratio data using key: {key}")
                break

        if position_ratio_df is not None and not position_ratio_df.empty:
            try:
                # Process timestamps (ensuring epoch timestamp in milliseconds is handled correctly)
                if 'time' in position_ratio_df.columns and 'datetime' not in position_ratio_df.columns:
                    # Direct conversion of epoch milliseconds to datetime
                    position_ratio_df['datetime'] = pd.to_datetime(position_ratio_df['time'], unit='ms')
                    logger.info(f"Converted 'time' column directly to datetime for position ratio data")
                else:
                    # Process timestamps using the standard function as fallback
                    position_ratio_df = process_timestamps(position_ratio_df)
                    logger.info(f"Used process_timestamps for position ratio data")

                # Check column names and map if needed
                column_mapping = {}

                # Check for different column naming patterns
                if 'top_position_long_short_ratio' in position_ratio_df.columns and 'long_short_ratio' not in position_ratio_df.columns:
                    column_mapping['top_position_long_short_ratio'] = 'long_short_ratio'

                # Apply column renaming if needed
                if column_mapping:
                    position_ratio_df = position_ratio_df.rename(columns=column_mapping)

                # Make sure percent columns are properly processed
                if 'top_position_long_percent' in position_ratio_df.columns:
                    # Convert to float if not already
                    position_ratio_df['top_position_long_percent'] = position_ratio_df['top_position_long_percent'].astype(float)
                    
                if 'top_position_short_percent' in position_ratio_df.columns:
                    # Convert to float if not already
                    position_ratio_df['top_position_short_percent'] = position_ratio_df['top_position_short_percent'].astype(float)

                # Calculate ratio if it doesn't exist but we have the components
                if 'long_short_ratio' not in position_ratio_df.columns and 'top_position_long_percent' in position_ratio_df.columns and 'top_position_short_percent' in position_ratio_df.columns:
                    position_ratio_df['long_short_ratio'] = position_ratio_df['top_position_long_percent'] / position_ratio_df['top_position_short_percent'].replace(0, float('nan'))
                
                # If we have a ratio column, make sure it's properly processed
                if 'long_short_ratio' in position_ratio_df.columns or 'top_position_long_short_ratio' in position_ratio_df.columns:
                    ratio_col = 'long_short_ratio' if 'long_short_ratio' in position_ratio_df.columns else 'top_position_long_short_ratio'
                    # Convert to float if not already
                    position_ratio_df[ratio_col] = position_ratio_df[ratio_col].astype(float)

                # Create metrics for position ratio
                if 'top_position_long_percent' in position_ratio_df.columns and 'top_position_short_percent' in position_ratio_df.columns:
                    # Get the most recent values
                    latest_data = position_ratio_df.iloc[-1]
                    
                    long_percent = latest_data.get('top_position_long_percent', 0)
                    short_percent = latest_data.get('top_position_short_percent', 0)
                    ratio = latest_data.get('long_short_ratio', 0)
                    
                    metrics = {
                        "Long %": {
                            "value": long_percent,
                            "delta": None,
                            "delta_suffix": "%"
                        },
                        "Short %": {
                            "value": short_percent,
                            "delta": None,
                            "delta_suffix": "%"
                        },
                        "Long/Short Ratio": {
                            "value": ratio,
                            "delta": None
                        }
                    }
                    
                    formatters = {
                        "Long %": lambda x: f"{x:.2f}%",
                        "Short %": lambda x: f"{x:.2f}%",
                        "Long/Short Ratio": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                    }
                    
                    display_metrics_row(metrics, formatters)

                # Check if we have the required columns now
                if 'datetime' in position_ratio_df.columns:
                    ratio_available = False
                    
                    # Try different ratio column names
                    if 'long_short_ratio' in position_ratio_df.columns:
                        ratio_col = 'long_short_ratio'
                        ratio_available = True
                    elif 'top_position_long_short_ratio' in position_ratio_df.columns:
                        ratio_col = 'top_position_long_short_ratio'
                        ratio_available = True
                    
                    # If neither column exists but we have long and short percentages, create ratio
                    elif 'top_position_long_percent' in position_ratio_df.columns and 'top_position_short_percent' in position_ratio_df.columns:
                        position_ratio_df['long_short_ratio'] = position_ratio_df['top_position_long_percent'] / position_ratio_df['top_position_short_percent'].replace(0, float('nan'))
                        ratio_col = 'long_short_ratio'
                        ratio_available = True
                    
                    if ratio_available:
                        # Create chart
                        st.subheader("Top Traders Long/Short Position Ratio")
                        
                        # Drop NaN values to prevent chart errors
                        chart_df = position_ratio_df.dropna(subset=['datetime', ratio_col])
                        
                        if not chart_df.empty:
                            fig = px.line(
                                chart_df,
                                x='datetime',
                                y=ratio_col,
                                title="Top Traders Long/Short Position Ratio"
                            )
                        else:
                            st.warning("Not enough valid data points to display the position ratio chart.")
                            # Skip the rest of this section
                            ratio_available = False

                    # Only add reference line if the chart exists and ratio is available
                    if ratio_available and 'fig' in locals():
                        fig.add_hline(
                            y=1,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Equal"
                        )

                        # Add price overlay if available
                        if 'price' in position_ratio_df.columns:
                            # Make sure we have valid price data
                            price_df = position_ratio_df.dropna(subset=['datetime', 'price'])
                            if not price_df.empty:
                                fig.add_trace(
                                    go.Scatter(
                                        x=price_df['datetime'],
                                        y=price_df['price'],
                                        name='Price',
                                        yaxis="y2",
                                        line=dict(color=ASSET_COLORS.get(asset, '#FF9900'))
                                    )
                                )

                                fig.update_layout(
                                    yaxis2=dict(
                                        title="Price (USD)",
                                        overlaying="y",
                                        side="right"
                                    )
                                )

                        # Display the chart if it exists
                        display_chart(apply_chart_theme(fig))
                    
                    # Display Long vs Short percentage chart as well
                    if 'top_position_long_percent' in position_ratio_df.columns and 'top_position_short_percent' in position_ratio_df.columns:
                        st.subheader("Top Traders Long/Short Position Percentages")
                        
                        try:
                            # Create a dataframe for plotting long and short percentages
                            plot_df = position_ratio_df[['datetime', 'top_position_long_percent', 'top_position_short_percent']].copy()
                            
                            # Ensure data is clean - drop NaN values
                            plot_df = plot_df.dropna(subset=['datetime', 'top_position_long_percent', 'top_position_short_percent'])
                            
                            if not plot_df.empty and len(plot_df) > 1:  # Need at least 2 points for a line
                                try:
                                    # Create the figure with two lines
                                    fig_pct = go.Figure()
                                    
                                    fig_pct.add_trace(go.Scatter(
                                        x=plot_df['datetime'],
                                        y=plot_df['top_position_long_percent'],
                                        name='Long %',
                                        line=dict(color='green', width=2)
                                    ))
                                    
                                    fig_pct.add_trace(go.Scatter(
                                        x=plot_df['datetime'],
                                        y=plot_df['top_position_short_percent'],
                                        name='Short %',
                                        line=dict(color='red', width=2)
                                    ))
                                    
                                    fig_pct.update_layout(
                                        title="Top Traders Position Long/Short Percentages",
                                        xaxis_title=None,
                                        yaxis_title="Percentage (%)",
                                        hovermode="x unified"
                                    )
                                    
                                    display_chart(apply_chart_theme(fig_pct))
                                except Exception as e:
                                    logger.error(f"Error creating position percentage chart: {e}")
                                    st.warning("Error displaying position percentage chart.")
                            else:
                                st.warning("Not enough valid data points to display the position percentage chart.")
                        except Exception as e:
                            logger.error(f"Error processing position percentages chart: {e}")
                            st.error(f"Could not create position percentage chart: {e}")
                else:
                    st.warning(f"Position ratio data is missing required columns. Available columns: {list(position_ratio_df.columns)}")
            except Exception as e:
                logger.error(f"Error processing position ratio data: {e}")
                st.error(f"Error processing position ratio data: {e}")
        else:
            st.info("No top traders position ratio data available.")

def render_order_book_page(asset):
    """Render the order book page for the specified asset."""
    st.header(f"{asset} Order Book Analysis")

    # Load order book data
    data = load_futures_data('order_book', asset)

    if not data:
        st.info(f"No order book data available for {asset}. Order book data shows the current buy and sell orders across exchanges.")

        # Show empty placeholder layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Asks Amount", "$0")
        with col2:
            st.metric("Bids Amount", "$0")
        with col3:
            st.metric("Asks/Bids Ratio", "0.00")

        st.subheader(f"{asset} Order Book History")
        st.info("No order book history data available. This chart would show the volume of asks and bids over time.")
        return

    # Try to find aggregated order book history with multiple key formats
    ob_df = None
    agg_keys = [
        f"api_futures_orderbook_aggregated_ask_bids_history_{asset}_{asset}",  # Double asset format
        f"api_futures_orderbook_aggregated_ask_bids_history_{asset}",          # Single asset format
        "api_futures_orderbook_aggregated_ask_bids_history"                    # Generic format
    ]

    # Try each key format until we find data
    for key in agg_keys:
        if key in data and not data[key].empty:
            ob_df = data[key]
            logger.info(f"Found aggregated order book data using key: {key}")
            break

    # If no aggregated data, try to use detailed data
    if ob_df is None or ob_df.empty:
        # If detailed order book data is available
        if 'api_futures_orderbook_ask_bids_history' in data and not data['api_futures_orderbook_ask_bids_history'].empty:
            detailed_df = data['api_futures_orderbook_ask_bids_history']

            # Filter for the selected asset if possible
            if 'symbol' in detailed_df.columns:
                detailed_df = detailed_df[detailed_df['symbol'].str.contains(asset, case=False, na=False)]

            # If we found matching data, use it instead of aggregated data
            if not detailed_df.empty:
                ob_df = detailed_df
                logger.info("Using detailed order book data as fallback for aggregated data")

    # Process aggregated or detailed order book history
    if ob_df is not None and not ob_df.empty:
        try:
            # Process dataframe
            ob_df = process_timestamps(ob_df)

            # Map column names if needed
            column_mapping = {}

            # Map asks/bids USD to amount if needed
            if 'asks_usd' in ob_df.columns and 'asks_amount' not in ob_df.columns:
                column_mapping['asks_usd'] = 'asks_amount'

            if 'bids_usd' in ob_df.columns and 'bids_amount' not in ob_df.columns:
                column_mapping['bids_usd'] = 'bids_amount'

            # Map aggregated variations too
            if 'aggregated_asks_usd' in ob_df.columns and 'asks_amount' not in ob_df.columns:
                column_mapping['aggregated_asks_usd'] = 'asks_amount'

            if 'aggregated_bids_usd' in ob_df.columns and 'bids_amount' not in ob_df.columns:
                column_mapping['aggregated_bids_usd'] = 'bids_amount'

            # Apply column renaming if needed
            if column_mapping:
                ob_df = ob_df.rename(columns=column_mapping)

            # Calculate asks_bids_ratio if it doesn't exist
            if 'asks_bids_ratio' not in ob_df.columns and 'asks_amount' in ob_df.columns and 'bids_amount' in ob_df.columns:
                ob_df['asks_bids_ratio'] = ob_df['asks_amount'] / ob_df['bids_amount'].replace(0, float('nan'))

            # Check if we have the required columns now
            required_cols = ['datetime', 'asks_amount', 'bids_amount']
            missing_cols = [col for col in required_cols if col not in ob_df.columns]

            if missing_cols:
                st.warning(f"Order book data is missing required columns: {missing_cols}. Available columns: {list(ob_df.columns)}")
            else:
                # Calculate metrics
                recent_ob = ob_df.iloc[-1] if len(ob_df) > 0 else None

                if recent_ob is not None:
                    asks_amount = recent_ob['asks_amount']
                    bids_amount = recent_ob['bids_amount']

                    # Calculate ratio if it doesn't exist in the dataframe
                    if 'asks_bids_ratio' in recent_ob:
                        ratio = recent_ob['asks_bids_ratio']
                    else:
                        ratio = asks_amount / bids_amount if bids_amount != 0 else float('nan')

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
                        "Asks/Bids Ratio": lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                    }

                    display_metrics_row(metrics, formatters)

                # Create time series chart for asks and bids
                st.subheader(f"{asset} Order Book History")

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
                    title=f"{asset} Order Book Asks/Bids Amount",
                    xaxis_title=None,
                    yaxis_title="Amount (USD)",
                    hovermode="x unified"
                )

                display_chart(apply_chart_theme(fig))

                # Create ratio chart if asks_bids_ratio exists or can be calculated
                if 'asks_bids_ratio' in ob_df.columns:
                    st.subheader(f"{asset} Ask/Bid Ratio")

                    fig = px.line(
                        ob_df,
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
        except Exception as e:
            logger.error(f"Error processing order book history: {e}")
            st.error(f"Error processing order book data: {e}")
    else:
        st.info(f"No aggregated order book history data available for {asset}.")

    # Detailed order book data - look for exchange specific data
    if 'api_futures_orderbook_ask_bids_history' in data:
        detailed_ob_df = data['api_futures_orderbook_ask_bids_history']

        if not detailed_ob_df.empty:
            try:
                # Map column names if needed
                column_mapping = {}

                # Map asks/bids USD to amount if needed
                if 'asks_usd' in detailed_ob_df.columns and 'asks_amount' not in detailed_ob_df.columns:
                    column_mapping['asks_usd'] = 'asks_amount'

                if 'bids_usd' in detailed_ob_df.columns and 'bids_amount' not in detailed_ob_df.columns:
                    column_mapping['bids_usd'] = 'bids_amount'

                # Map quantity columns too if needed
                if 'asks_quantity' in detailed_ob_df.columns and 'asks_qty' not in detailed_ob_df.columns:
                    column_mapping['asks_quantity'] = 'asks_qty'

                if 'bids_quantity' in detailed_ob_df.columns and 'bids_qty' not in detailed_ob_df.columns:
                    column_mapping['bids_quantity'] = 'bids_qty'

                # Apply column renaming if needed
                if column_mapping:
                    detailed_ob_df = detailed_ob_df.rename(columns=column_mapping)

                # Add exchange_name if it doesn't exist but we have exchange
                if 'exchange_name' not in detailed_ob_df.columns and 'exchange' in detailed_ob_df.columns:
                    detailed_ob_df['exchange_name'] = detailed_ob_df['exchange']

                # Add symbol if it doesn't exist (use 'pair' if available)
                if 'symbol' not in detailed_ob_df.columns:
                    if 'pair' in detailed_ob_df.columns:
                        detailed_ob_df['symbol'] = detailed_ob_df['pair']
                    else:
                        # Default to asset name if no symbol/pair info
                        detailed_ob_df['symbol'] = asset + "USDT"  # Default to most common format

                # Filter for the selected asset
                asset_ob = None
                if 'symbol' in detailed_ob_df.columns:
                    asset_ob = detailed_ob_df[detailed_ob_df['symbol'].str.contains(asset, case=False, na=False)]
                else:
                    # If no symbol column, use the whole dataframe
                    asset_ob = detailed_ob_df.copy()

                if asset_ob is not None and not asset_ob.empty:
                    # Process dataframe
                    asset_ob = process_timestamps(asset_ob)

                    # Add exchange_name if it doesn't exist
                    if 'exchange_name' not in asset_ob.columns:
                        # Try to determine from data or default to a fixed value
                        if len(asset_ob) > 0 and 'exchange' in asset_ob.columns:
                            asset_ob['exchange_name'] = asset_ob['exchange']
                        else:
                            # Default exchange name
                            asset_ob['exchange_name'] = "Binance"  # Default to most common exchange

                    # Group by exchange
                    exchanges = asset_ob['exchange_name'].unique()

                    if len(exchanges) > 0:
                        # Add exchange selector
                        selected_exchange = st.selectbox("Select Exchange", exchanges)

                        # Filter for selected exchange
                        exchange_ob = asset_ob[asset_ob['exchange_name'] == selected_exchange]

                        if not exchange_ob.empty:
                            # Check required columns
                            if 'asks_amount' in exchange_ob.columns and 'bids_amount' in exchange_ob.columns:
                                st.subheader(f"{selected_exchange} {asset} Order Book")

                                # Create time series chart for asks and bids
                                fig = go.Figure()

                                fig.add_trace(go.Scatter(
                                    x=exchange_ob['datetime'],
                                    y=exchange_ob['asks_amount'],
                                    name='Asks Amount',
                                    line=dict(color='red')
                                ))

                                fig.add_trace(go.Scatter(
                                    x=exchange_ob['datetime'],
                                    y=exchange_ob['bids_amount'],
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

                                # Create ratio chart
                                exchange_ob['asks_bids_ratio'] = exchange_ob['asks_amount'] / exchange_ob['bids_amount'].replace(0, float('nan'))

                                fig = px.line(
                                    exchange_ob,
                                    x='datetime',
                                    y='asks_bids_ratio',
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
                                st.warning(f"Exchange order book data is missing required columns. Available: {list(exchange_ob.columns)}")
                        else:
                            st.info(f"No data available for {selected_exchange}.")
                    else:
                        st.info("No exchange data available in the order book data.")
                else:
                    st.info(f"No detailed order book data available for {asset}.")
            except Exception as e:
                logger.error(f"Error processing detailed order book data: {e}")
                st.error(f"Error processing detailed order book data: {e}")
        else:
            st.info("No detailed order book data available.")

    # Large limit orders
    if 'api_futures_orderbook_large_limit_order' in data:
        large_orders_df = data['api_futures_orderbook_large_limit_order']

        if not large_orders_df.empty:
            try:
                # Filter for the selected asset if symbol column exists
                asset_orders = None
                if 'symbol' in large_orders_df.columns:
                    asset_orders = large_orders_df[large_orders_df['symbol'].str.contains(asset, case=False, na=False)]
                else:
                    # If no symbol column, use the whole dataframe
                    asset_orders = large_orders_df.copy()

                if asset_orders is not None and not asset_orders.empty:
                    st.subheader(f"{asset} Large Limit Orders")

                    # Sort by size if amount_usd column exists
                    if 'amount_usd' in asset_orders.columns:
                        asset_orders = asset_orders.sort_values(by='amount_usd', ascending=False)

                    # Determine format dictionary based on available columns
                    format_dict = {}

                    if 'price' in asset_orders.columns:
                        format_dict['price'] = lambda x: format_currency(x, precision=2)

                    if 'amount' in asset_orders.columns:
                        format_dict['amount'] = lambda x: f"{x:.6f}"

                    if 'amount_usd' in asset_orders.columns:
                        format_dict['amount_usd'] = lambda x: format_currency(x, abbreviate=True)

                    # Create table with available columns
                    create_formatted_table(
                        asset_orders,
                        format_dict=format_dict
                    )
                else:
                    st.info(f"No large limit orders available for {asset}.")
            except Exception as e:
                logger.error(f"Error processing large limit orders: {e}")
                st.error(f"Error processing large limit orders: {e}")
        else:
            st.info("No large limit order data available.")

def render_market_data_page(asset):
    """Render the market data page for the specified asset."""
    st.header(f"{asset} Futures Market Data")

    # Load market data
    data = load_futures_data('market', asset)

    if not data:
        st.info(f"No market data available for {asset}. Market data shows price, volume, and trading information across exchanges.")

        # Show empty placeholder layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{asset} Price", "$0")
        with col2:
            st.metric("24h Volume", "$0")
        with col3:
            st.metric("24h Change", "0.00%")

        st.subheader(f"{asset} Price History")
        st.info("No price history data available. This chart would show historical price movements over time.")

        st.subheader(f"{asset} Trading Pairs")
        st.info("No trading pairs data available. This section would show volume and other metrics for different exchanges.")
        return

    # Price OHLC data - try multiple possible key formats
    price_df = None
    price_keys = [
        'api_price_ohlc_history',
        f'api_price_ohlc_history_{asset}',
        f'futures_price_ohlc_{asset}'
    ]

    # Try each key format until we find data
    for key in price_keys:
        if key in data and not data[key].empty:
            price_df = data[key]
            logger.info(f"Found price history data using key: {key}")

            # Try to filter for the asset if possible
            if asset.lower() not in key.lower() and 'symbol' in price_df.columns:
                price_df = price_df[price_df['symbol'].str.contains(asset, case=False, na=False)]

            if not price_df.empty:
                break

    if price_df is not None and not price_df.empty:
        try:
            # Process dataframe to extract timestamps
            price_df = process_timestamps(price_df)

            # Convert string numbers to actual numeric values
            numeric_cols = ['open', 'high', 'low', 'close']
            for col in numeric_cols:
                if col in price_df.columns and price_df[col].dtype == 'object':
                    price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

            # Check for volume column with different naming
            if 'volume' not in price_df.columns and 'volume_usd' in price_df.columns:
                price_df['volume'] = pd.to_numeric(price_df['volume_usd'], errors='coerce')

            # Calculate metrics
            if 'close' in price_df.columns:
                # Sort by datetime to ensure latest data is at the end
                price_df = price_df.sort_values('datetime')

                # Get latest price
                latest_price = price_df['close'].iloc[-1] if len(price_df) > 0 else None

                # 24h metrics (use last value vs previous day value)
                if len(price_df) > 1:
                    # Find a value approximately 24h earlier
                    latest_time = price_df['datetime'].iloc[-1]
                    prev_day_time = latest_time - pd.Timedelta(days=1)
                    prev_day_idx = price_df['datetime'].searchsorted(prev_day_time)
                    prev_day_idx = min(prev_day_idx, len(price_df) - 1)

                    prev_price = price_df['close'].iloc[prev_day_idx]
                    price_change_24h = latest_price - prev_price
                    price_change_pct_24h = (price_change_24h / prev_price) * 100 if prev_price != 0 else None
                else:
                    price_change_24h = None
                    price_change_pct_24h = None

                # Create metrics
                metrics = {
                    f"{asset} Price": {
                        "value": latest_price,
                        "delta": price_change_pct_24h,
                        "delta_suffix": "%"
                    }
                }

                # Add volume if available
                if 'volume' in price_df.columns:
                    # Get latest volume (might be same as 24h volume depending on data)
                    latest_volume = price_df['volume'].iloc[-1] if len(price_df) > 0 else None

                    metrics["24h Volume"] = {
                        "value": latest_volume,
                        "delta": None
                    }

                # Create a separate 24h change metric that's more visible
                if price_change_pct_24h is not None:
                    metrics["24h Change"] = {
                        "value": price_change_pct_24h,
                        "delta": None,
                        "delta_suffix": "%"
                    }

                formatters = {
                    f"{asset} Price": lambda x: format_currency(x, precision=2) if pd.notna(x) else "N/A",
                    "24h Volume": lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else "N/A",
                    "24h Change": lambda x: format_percentage(x, precision=2) if pd.notna(x) else "N/A"
                }

                display_metrics_row(metrics, formatters)

                # Create OHLC chart
                st.subheader(f"{asset} Price History")

                try:
                    # Check if we have all required columns for the OHLC chart
                    required_cols = ['datetime', 'open', 'high', 'low', 'close']
                    if all(col in price_df.columns for col in required_cols):
                        # Filter out any rows with NaN values in required columns
                        chart_df = price_df.dropna(subset=required_cols)

                        # Create OHLC chart if we have enough data
                        if len(chart_df) > 1:
                            fig = create_ohlc_chart(
                                chart_df,
                                'datetime',
                                'open',
                                'high',
                                'low',
                                'close',
                                f"{asset} Price History",
                                volume_col='volume' if 'volume' in chart_df.columns else None
                            )

                            display_chart(fig)
                        else:
                            st.warning("Not enough price data to create a meaningful chart.")
                    else:
                        missing = [col for col in required_cols if col not in price_df.columns]
                        st.warning(f"Price data is missing required columns for OHLC chart: {missing}")
                except Exception as chart_e:
                    logger.error(f"Error creating OHLC chart: {chart_e}")
                    st.error(f"Unable to create price chart due to data format issues.")
            else:
                st.warning("Price data is missing the 'close' column required for metrics.")
        except Exception as e:
            logger.error(f"Error processing price history data: {e}")
            st.error(f"Error processing price data: {e}")
    else:
        st.info(f"No price history data available for {asset}.")

    # Trading pairs data - try multiple possible key formats
    pairs_df = None
    pairs_keys = [
        f"api_futures_pairs_markets_{asset}_{asset}",  # Double asset format
        f"api_futures_pairs_markets_{asset}",          # Single asset format
        "api_futures_pairs_markets"                    # Generic format
    ]

    # Try each key format until we find data
    for key in pairs_keys:
        if key in data and not data[key].empty:
            pairs_df = data[key]
            logger.info(f"Found trading pairs data using key: {key}")

            # Filter for the asset if it's a generic key
            if "api_futures_pairs_markets" == key and 'symbol' in pairs_df.columns:
                pairs_df = pairs_df[pairs_df['symbol'].str.contains(asset, case=False, na=False)]

            if not pairs_df.empty:
                break

    if pairs_df is not None and not pairs_df.empty:
        try:
            # Check and map column names if needed
            column_mapping = {}

            # Map typical column naming variations
            if 'price' in pairs_df.columns and 'current_price' not in pairs_df.columns:
                column_mapping['price'] = 'current_price'

            if 'volume_usd' in pairs_df.columns and 'volume_24h_usd' not in pairs_df.columns:
                column_mapping['volume_usd'] = 'volume_24h_usd'

            # Apply column mapping if needed
            if column_mapping:
                pairs_df = pairs_df.rename(columns=column_mapping)

            # Convert string columns to numeric
            numeric_cols = ['current_price', 'volume_24h_usd', 'open_interest_usd']
            for col in numeric_cols:
                if col in pairs_df.columns and pairs_df[col].dtype == 'object':
                    pairs_df[col] = pd.to_numeric(pairs_df[col], errors='coerce')

            st.subheader(f"{asset} Trading Pairs")

            # Determine which columns to display
            if 'volume_24h_usd' in pairs_df.columns:
                # Sort by volume
                pairs_df = pairs_df.sort_values(by='volume_24h_usd', ascending=False)

                # Determine which columns to display based on what's available
                display_columns = []

                # Always include symbol if available
                if 'symbol' in pairs_df.columns:
                    display_columns.append('symbol')

                # Always include exchange if available
                if 'exchange_name' in pairs_df.columns:
                    display_columns.append('exchange_name')

                # Add core trading metrics if available
                for col in ['current_price', 'price_change_percent_24h', 'volume_24h_usd', 'open_interest_usd']:
                    if col in pairs_df.columns:
                        display_columns.append(col)

                # Add funding rate if available
                if 'funding_rate' in pairs_df.columns:
                    display_columns.append('funding_rate')

                # Create format dictionary based on available columns
                format_dict = {}

                if 'current_price' in display_columns:
                    format_dict['current_price'] = lambda x: format_currency(x, precision=2) if pd.notna(x) else "N/A"

                if 'price_change_percent_24h' in display_columns:
                    format_dict['price_change_percent_24h'] = lambda x: format_percentage(x) if pd.notna(x) else "N/A"

                if 'volume_24h_usd' in display_columns:
                    format_dict['volume_24h_usd'] = lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else "N/A"

                if 'open_interest_usd' in display_columns:
                    format_dict['open_interest_usd'] = lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else "N/A"

                if 'funding_rate' in display_columns:
                    format_dict['funding_rate'] = lambda x: format_percentage(x * 100) if pd.notna(x) else "N/A"

                # Create table with available columns
                if display_columns:
                    # Take only the first 20 rows to avoid overwhelming the UI
                    display_df = pairs_df[display_columns].head(20)
                    create_formatted_table(display_df, format_dict=format_dict)
                else:
                    st.warning("No suitable columns found for display in trading pairs data.")

                # Create bar chart for volume by exchange if we have exchange_name and volume data
                if 'exchange_name' in pairs_df.columns and 'volume_24h_usd' in pairs_df.columns:
                    # Group by exchange and sum volumes
                    exchange_volume = pairs_df.groupby('exchange_name')['volume_24h_usd'].sum().reset_index()

                    # Sort and take top 10
                    exchange_volume = exchange_volume.sort_values(by='volume_24h_usd', ascending=False).head(10)

                    if not exchange_volume.empty and len(exchange_volume) > 1:
                        fig = px.bar(
                            exchange_volume,
                            x='exchange_name',
                            y='volume_24h_usd',
                            title=f"Top 10 Exchanges by {asset} Trading Volume",
                            color='volume_24h_usd',
                            color_continuous_scale='Viridis'
                        )

                        fig.update_layout(
                            xaxis_title=None,
                            yaxis_title="24h Volume (USD)",
                            coloraxis_showscale=False
                        )

                        display_chart(apply_chart_theme(fig))

                        # Add pie chart showing market share
                        st.subheader(f"{asset} Trading Volume Market Share")
                        st.write("Distribution of 24h trading volume across exchanges")
                        
                        # Add spacing
                        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)

                        # Calculate market shares
                        total_volume = exchange_volume['volume_24h_usd'].sum()
                        exchange_volume['market_share'] = exchange_volume['volume_24h_usd'] / total_volume * 100

                        # Create a more compact pie chart with better spacing
                        # Limit to top 10 exchanges for better readability
                        top_exchanges = exchange_volume.sort_values(by='volume_24h_usd', ascending=False).head(10)
                        
                        # Create pie chart without a title to prevent overlap
                        fig = px.pie(
                            top_exchanges,
                            values='volume_24h_usd',
                            names='exchange_name',
                            # Removing title to prevent overlap with the subheader above
                            title=""
                        )

                        # Use cleaner text format
                        fig.update_traces(
                            textposition='inside',
                            textinfo='percent', # Only show percentage to avoid overcrowding
                            insidetextorientation='radial' # Improve text readability
                        )
                        
                        # Increase height and add margin to prevent overlap
                        fig.update_layout(
                            height=500,
                            margin=dict(t=60, b=60, l=20, r=20)
                        )

                        display_chart(apply_chart_theme(fig))

                # Create open interest overview if available
                if 'open_interest_usd' in pairs_df.columns and pairs_df['open_interest_usd'].notna().any():
                    st.subheader(f"{asset} Open Interest by Exchange")

                    # Group by exchange and sum open interest
                    if 'exchange_name' in pairs_df.columns:
                        oi_by_exchange = pairs_df.groupby('exchange_name')['open_interest_usd'].sum().reset_index()

                        # Sort and take top 10
                        oi_by_exchange = oi_by_exchange.sort_values(by='open_interest_usd', ascending=False).head(10)

                        if not oi_by_exchange.empty and len(oi_by_exchange) > 1:
                            fig = px.bar(
                                oi_by_exchange,
                                x='exchange_name',
                                y='open_interest_usd',
                                title=f"Top 10 Exchanges by {asset} Open Interest",
                                color='open_interest_usd',
                                color_continuous_scale='Viridis'
                            )

                            fig.update_layout(
                                xaxis_title=None,
                                yaxis_title="Open Interest (USD)",
                                coloraxis_showscale=False
                            )

                            display_chart(apply_chart_theme(fig))
            else:
                st.warning("Trading pairs data is missing the volume_24h_usd column required for metrics.")
        except Exception as e:
            logger.error(f"Error processing trading pairs data: {e}")
            st.error(f"Error processing trading pairs data: {e}")
    else:
        st.info(f"No trading pairs data available for {asset}.")

    # Show funding rates if available
    if 'funding_rate' in pairs_df.columns if pairs_df is not None else False:
        try:
            funding_df = pairs_df[pairs_df['funding_rate'].notna()].copy()

            if not funding_df.empty:
                st.subheader(f"{asset} Funding Rates by Exchange")

                # Ensure funding_rate is numeric
                funding_df['funding_rate'] = pd.to_numeric(funding_df['funding_rate'], errors='coerce')

                # Filter out rows with NaN funding rates
                funding_df = funding_df.dropna(subset=['funding_rate'])

                if not funding_df.empty:
                    # Sort by funding rate
                    funding_df = funding_df.sort_values(by='funding_rate', ascending=False)

                    # Prepare display dataframe with relevant columns
                    display_cols = ['exchange_name', 'symbol', 'funding_rate']
                    if 'next_funding_time' in funding_df.columns:
                        display_cols.append('next_funding_time')

                    # Format dictionary
                    format_dict = {
                        'funding_rate': lambda x: format_percentage(x * 100) if pd.notna(x) else "N/A"
                    }

                    if 'next_funding_time' in display_cols:
                        format_dict['next_funding_time'] = lambda x: format_timestamp(x, "%Y-%m-%d %H:%M") if pd.notna(x) else "N/A"

                    # Create table
                    create_formatted_table(
                        funding_df[display_cols],
                        format_dict=format_dict
                    )

                    # Create bar chart for funding rates by exchange
                    fig = px.bar(
                        funding_df.head(15),  # Top 15 exchanges by funding rate
                        x='exchange_name',
                        y='funding_rate',
                        title=f"{asset} Funding Rates by Exchange",
                        color='funding_rate',
                        color_continuous_scale='RdBu',
                        color_continuous_midpoint=0
                    )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=None,
                        yaxis_title="Funding Rate",
                        yaxis_tickformat='.2%'
                    )

                    # Add reference line at 0
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="gray"
                    )

                    display_chart(apply_chart_theme(fig))
        except Exception as e:
            logger.error(f"Error processing funding rate data: {e}")
            st.warning(f"Unable to display funding rate data due to formatting issues.")

    # If we have liquidation data in the pairs dataframe, show it
    if pairs_df is not None and 'long_liquidation_usd_24h' in pairs_df.columns and 'short_liquidation_usd_24h' in pairs_df.columns:
        try:
            liquidation_df = pairs_df[
                (pairs_df['long_liquidation_usd_24h'].notna()) |
                (pairs_df['short_liquidation_usd_24h'].notna())
            ].copy()

            if not liquidation_df.empty:
                st.subheader(f"{asset} 24h Liquidations by Exchange")

                # Calculate total liquidations and add as column
                liquidation_df['total_liquidation_usd'] = liquidation_df['long_liquidation_usd_24h'].fillna(0) + liquidation_df['short_liquidation_usd_24h'].fillna(0)

                # Sort by total liquidation
                liquidation_df = liquidation_df.sort_values(by='total_liquidation_usd', ascending=False)

                # Group by exchange
                if 'exchange_name' in liquidation_df.columns:
                    liq_by_exchange = liquidation_df.groupby('exchange_name').agg({
                        'long_liquidation_usd_24h': 'sum',
                        'short_liquidation_usd_24h': 'sum',
                        'total_liquidation_usd': 'sum'
                    }).reset_index()

                    # Sort and take top exchanges
                    liq_by_exchange = liq_by_exchange.sort_values(by='total_liquidation_usd', ascending=False).head(10)

                    if not liq_by_exchange.empty:
                        # Create table
                        create_formatted_table(
                            liq_by_exchange,
                            format_dict={
                                'long_liquidation_usd_24h': lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else "$0",
                                'short_liquidation_usd_24h': lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else "$0",
                                'total_liquidation_usd': lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else "$0"
                            }
                        )

                        # Create stacked bar chart for liquidations by exchange
                        if len(liq_by_exchange) > 1:
                            # Melt the dataframe for stacked bars
                            melted_df = pd.melt(
                                liq_by_exchange,
                                id_vars=['exchange_name'],
                                value_vars=['long_liquidation_usd_24h', 'short_liquidation_usd_24h'],
                                var_name='liquidation_type',
                                value_name='liquidation_usd'
                            )

                            # Create stacked bar chart
                            fig = px.bar(
                                melted_df,
                                x='exchange_name',
                                y='liquidation_usd',
                                color='liquidation_type',
                                title=f"{asset} 24h Liquidations by Exchange",
                                color_discrete_map={
                                    'long_liquidation_usd_24h': ASSET_COLORS.get(asset, '#3366CC'),
                                    'short_liquidation_usd_24h': 'red'
                                },
                                labels={
                                    'liquidation_type': 'Liquidation Type',
                                    'liquidation_usd': 'Liquidation Volume (USD)',
                                    'exchange_name': 'Exchange'
                                }
                            )

                            # Rename legend items
                            fig.for_each_trace(lambda t: t.update(
                                name=t.name.replace('long_liquidation_usd_24h', 'Long').replace('short_liquidation_usd_24h', 'Short')
                            ))

                            display_chart(apply_chart_theme(fig))
        except Exception as e:
            logger.error(f"Error processing liquidation data from market pairs: {e}")
            # Don't show error to user as this is a bonus chart

def main():
    """Main function to render the futures page."""

    # Render sidebar
    render_sidebar()

    # Page title
    st.title("Cryptocurrency Futures Markets")

    try:
        # Get asset from session state or use default
        available_assets = get_available_assets_for_category('futures')

        # Set default assets if none are available
        if not available_assets:
            st.warning("No futures data available for any asset. Showing layout with placeholder data.")
            available_assets = ["BTC", "ETH", "SOL", "XRP"]

        asset = st.session_state.get('selected_asset', available_assets[0])

        # Define the categories
        futures_categories = [
            "Funding Rate",
            "Liquidation",
            "Long/Short Ratio",
            "Open Interest",
            "Order Book",
            "Market Data"
        ]

        # Create tabs for each category
        tabs = st.tabs(futures_categories)

        # Find the index of the currently active category
        current_subcategory = st.session_state.get('futures_subcategory', 'funding_rate').replace('_', ' ').title()
        active_tab = 0
        for i, cat in enumerate(futures_categories):
            if cat == current_subcategory:
                active_tab = i
                break

        # We can't programmatically set the active tab in Streamlit,
        # but we can pre-load data for the expected active tab

        # Render each tab with individual error handling
        with tabs[0]:  # Funding Rate
            try:
                subcategory = 'funding_rate'
                st.session_state.futures_subcategory = subcategory
                render_funding_rate_page(asset)
            except Exception as e:
                logger.error(f"Error rendering funding_rate tab: {e}")
                st.error(f"Error displaying funding rate data")
                st.info("There was an error processing the funding rate data. This could be due to an unexpected data format or missing data.")

        with tabs[1]:  # Liquidation
            try:
                subcategory = 'liquidation'
                st.session_state.futures_subcategory = subcategory
                render_liquidation_page(asset)
            except Exception as e:
                logger.error(f"Error rendering liquidation tab: {e}")
                st.error(f"Error displaying liquidation data")
                st.info("There was an error processing the liquidation data. This could be due to an unexpected data format or missing data.")

        with tabs[2]:  # Long/Short Ratio
            try:
                subcategory = 'long_short_ratio'
                st.session_state.futures_subcategory = subcategory
                render_long_short_ratio_page(asset)
            except Exception as e:
                logger.error(f"Error rendering long_short_ratio tab: {e}")
                st.error(f"Error displaying long/short ratio data")
                st.info("There was an error processing the long/short ratio data. This could be due to an unexpected data format or missing data.")

        with tabs[3]:  # Open Interest
            try:
                subcategory = 'open_interest'
                st.session_state.futures_subcategory = subcategory
                render_open_interest_page(asset)
            except Exception as e:
                logger.error(f"Error rendering open_interest tab: {e}")
                st.error(f"Error displaying open interest data")
                st.info("There was an error processing the open interest data. This could be due to an unexpected data format or missing data.")

        with tabs[4]:  # Order Book
            try:
                subcategory = 'order_book'
                st.session_state.futures_subcategory = subcategory
                render_order_book_page(asset)
            except Exception as e:
                logger.error(f"Error rendering order_book tab: {e}")
                st.error(f"Error displaying order book data")
                st.info("There was an error processing the order book data. This could be due to an unexpected data format or missing data.")

        with tabs[5]:  # Market Data
            try:
                subcategory = 'market_data'
                st.session_state.futures_subcategory = subcategory
                render_market_data_page(asset)
            except Exception as e:
                logger.error(f"Error rendering market_data tab: {e}")
                st.error(f"Error displaying market data")
                st.info("There was an error processing the market data. This could be due to an unexpected data format or missing data.")

    except Exception as e:
        # If there's a critical error, log it and show a generic error message
        logger.error(f"Critical error in futures page: {e}")
        st.error("An error occurred while loading the futures page. Please check the logs for more information.")
    
    # Add footer
    st.markdown("---")
    st.caption("Izun Crypto Liquidity Report © 2025")
    st.caption("Data provided by CoinGlass API")

if __name__ == "__main__":
    main()