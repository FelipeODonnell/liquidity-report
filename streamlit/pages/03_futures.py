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
from utils.config import APP_TITLE, APP_ICON, ASSET_COLORS, DATA_BASE_PATH

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

def render_funding_rate_page(asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render the funding rate page for the specified asset and selected exchanges.
    
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
    st.header(f"{asset} Funding Rate Analysis")
    
    # Define available exchanges for funding rate
    available_exchanges = ["Binance", "OKX", "Bybit", "dYdX", "Bitfinex", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Store the available exchanges for use with each chart
    st.session_state.funding_rate_available_exchanges = available_exchanges
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
    
    # Store in session state for this section
    st.session_state.selected_funding_rate_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges
    
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
                    # Filter for the selected asset (exact match only)
                    asset_fr = fr_df[fr_df['symbol'].str.upper() == asset.upper()]
                    
                    # Filter by selected exchanges if needed (if not "All")
                    if selected_exchanges and "All" not in selected_exchanges:
                        asset_fr = asset_fr[asset_fr['exchange_name'].isin(selected_exchanges)]
                # For normalized exchange list format
                elif 'symbol' in fr_df.columns and 'exchange_name' in fr_df.columns:
                    # Filter for the selected asset (exact match only)
                    asset_fr = fr_df[fr_df['symbol'].str.upper() == asset.upper()]
                    
                    # Filter by selected exchanges if needed (if not "All")
                    if selected_exchanges and "All" not in selected_exchanges:
                        asset_fr = asset_fr[asset_fr['exchange_name'].isin(selected_exchanges)]
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
                    
                    # Add a new section for Funding Rate History by Exchange
                    st.subheader("Funding Rate History by Exchange")
                    
                    # Create a container for the filters
                    filter_container = st.container()
                    filter_col1, filter_col2, filter_col3 = filter_container.columns([2, 1, 1])
                    
                    # Define time ranges (max 6M)
                    time_range_options = {
                        "1D": 1,
                        "1W": 7,
                        "1M": 30,
                        "3M": 90,
                        "6M": 180,
                        "All": 180  # Changed to also be 6M (180 days)
                    }
                    
                    # Create time range selector
                    with filter_col1:
                        selected_time_range = st.selectbox(
                            "Select Time Range",
                            list(time_range_options.keys()),
                            index=list(time_range_options.keys()).index("1M"),
                            key=f"funding_rate_time_range_{asset}"
                        )
                        
                    # Add range selectors for y-axis
                    with filter_col2:
                        y_min = st.number_input(
                            "Y-Axis Min (basis points)",
                            value=-10,
                            step=5,
                            key=f"funding_rate_y_min_{asset}"
                        )
                        
                    with filter_col3:
                        y_max = st.number_input(
                            "Y-Axis Max (basis points)",
                            value=1000,
                            step=50,
                            key=f"funding_rate_y_max_{asset}"
                        )
                    
                    # Store in session state for other parts of the app
                    st.session_state.funding_rate_time_range = selected_time_range
                    st.session_state.selected_time_range = selected_time_range
                    
                    # Process exchange-specific funding rate history data
                    funding_history_dfs = {}
                    available_exchanges = []
                    
                    # Find all funding rate history files for this asset
                    for key in data.keys():
                        if f"api_futures_fundingRate_ohlc_history_{asset}_" in key:
                            # Extract exchange name from key
                            exchange_name = key.split(f"api_futures_fundingRate_ohlc_history_{asset}_")[1].replace(".parquet", "")
                            
                            # Process the data
                            df = data[key]
                            if not df.empty:
                                # Convert timestamp to datetime
                                df = process_timestamps(df, timestamp_col="time")
                                
                                # Add to the collection
                                funding_history_dfs[exchange_name] = df
                                available_exchanges.append(exchange_name)
                    
                    if funding_history_dfs:
                        # Create the figure
                        fig = go.Figure()
                        
                        # Apply time filter
                        days_filter = time_range_options[selected_time_range]
                        cutoff_date = None
                        
                        if days_filter:
                            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_filter)
                        
                        # Add a trace for each exchange
                        for exchange, df in funding_history_dfs.items():
                            # Apply time filter if specified
                            if cutoff_date and 'datetime' in df.columns:
                                filtered_df = df[df['datetime'] >= cutoff_date]
                            else:
                                filtered_df = df
                            
                            # Skip if filtered data is empty
                            if filtered_df.empty:
                                continue
                            
                            # Apply filter based on selected exchanges
                            if "All" not in selected_exchanges and exchange not in selected_exchanges:
                                continue
                            
                            # Add trace - use raw values, already in percentage
                            fig.add_trace(go.Scatter(
                                x=filtered_df['datetime'],
                                y=filtered_df['close'],  # Use raw values, already in percentage
                                mode='lines',  # Only use lines mode, no markers or text
                                name=exchange,
                                line=dict(width=2),
                                hovertemplate='%{y:.4f}<extra></extra>',  # No % symbol in hover
                                showlegend=True  # Show the legend for exchange color matching
                            ))
                        
                        # Add a zero line - using filtered date range if applicable
                        if cutoff_date:
                            x0 = cutoff_date
                            x1 = pd.Timestamp.now()
                        else:
                            # Use full date range
                            x0 = min([df['datetime'].min() for df in funding_history_dfs.values() if not df.empty])
                            x1 = max([df['datetime'].max() for df in funding_history_dfs.values() if not df.empty])
                        
                        fig.add_shape(
                            type="line",
                            x0=x0,
                            y0=0,
                            x1=x1,
                            y1=0,
                            line=dict(color="gray", width=1, dash="dash"),
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title=f"{asset} Funding Rate History by Exchange (in basis points)",
                            xaxis_title=None,
                            yaxis_title="Funding Rate (basis points)",  # Specify basis points
                            hovermode="x unified",
                            showlegend=True,  # Show the legend
                            legend=dict(
                                orientation="h",
                                yanchor="bottom", 
                                y=1.02,
                                xanchor="right",
                                x=1,
                                font=dict(size=10),
                                itemsizing="constant",
                                itemwidth=30
                            )
                        )
                        
                        # Format y-axis using user-provided range values (in basis points)
                        # Generate tick values based on the range
                        range_size = y_max - y_min
                        # Determine appropriate tick spacing based on range size
                        if range_size <= 50:
                            tick_step = 5
                        elif range_size <= 200:
                            tick_step = 20
                        elif range_size <= 1000:
                            tick_step = 100
                        else:
                            tick_step = 200
                            
                        # Generate tick values and labels
                        tick_vals = list(range(y_min, y_max + 1, tick_step))
                        tick_text = [str(val) for val in tick_vals]
                        
                        # Update the y-axis
                        fig.update_yaxes(
                            range=[y_min, y_max],  # Use user-provided range
                            tickvals=tick_vals,   # Dynamic tick values
                            ticktext=tick_text,   # Dynamic tick labels
                            title="Funding Rate (basis points)"  # Note that values are in basis points
                        )
                        
                        # Apply theme and display
                        fig = apply_chart_theme(fig)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Add explanation with exchange color key
                        with st.expander("About Funding Rates"):
                            # Create a color key
                            st.markdown("**Funding Rates Explained:**")
                            st.markdown("""
                            - Funding rates are periodic payments between long and short positions in perpetual futures contracts.
                            - Values are shown in basis points (1 basis point = 0.01%).
                            - A value of 100 means 1%, 1000 means 10%, etc.
                            - Positive rates indicate longs pay shorts (bullish sentiment).
                            - Negative rates indicate shorts pay longs (bearish sentiment).
                            - Rates are typically charged every 8 hours, but this varies by exchange.
                            - Higher absolute values indicate stronger market sentiment in either direction.
                            """)
                            
                            # Create a color key for exchanges
                            st.markdown("**Exchange Color Key:**")
                            
                            # Get the exchanges actually being displayed
                            visible_exchanges = []
                            for exchange in available_exchanges:
                                if "All" in selected_exchanges or exchange in selected_exchanges:
                                    visible_exchanges.append(exchange)
                            
                            # Create columns for the color key
                            if visible_exchanges:
                                # Use 3 columns for the display
                                key_cols = st.columns(3)
                                for i, exchange in enumerate(visible_exchanges):
                                    col_idx = i % 3  # Distribute across 3 columns
                                    with key_cols[col_idx]:
                                        # Get color from Plotly's default color sequence
                                        color_idx = i % 10  # Plotly uses 10 default colors
                                        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                                                 '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']
                                        color = colors[color_idx]
                                        st.markdown(f"<span style='color:{color}'>●</span> {exchange}", unsafe_allow_html=True)
                            else:
                                st.info("No exchanges currently selected")
                    else:
                        st.info(f"No detailed funding rate history data available for {asset} by exchange. This chart would show how funding rates have changed over time across different exchanges.")
                    
                    # Add a new section for Open Interest Weighted Funding Rate
                    st.subheader("Funding Rate History - Open Interest Weighted")
                    
                    # Create a container for the filters
                    oi_filter_container = st.container()
                    oi_filter_col1, oi_filter_col2, oi_filter_col3 = oi_filter_container.columns([2, 1, 1])
                    
                    # Define time ranges (max 6M) - reuse the same options 
                    # as the exchange section for consistency
                    
                    # Create time range selector
                    with oi_filter_col1:
                        oi_selected_time_range = st.selectbox(
                            "Select Time Range",
                            list(time_range_options.keys()),
                            index=list(time_range_options.keys()).index("1M"),
                            key=f"oi_funding_rate_time_range_{asset}"
                        )
                        
                    # Add range selectors for y-axis
                    with oi_filter_col2:
                        oi_y_min = st.number_input(
                            "Y-Axis Min (basis points)",
                            value=-10,
                            step=5,
                            key=f"oi_funding_rate_y_min_{asset}"
                        )
                        
                    with oi_filter_col3:
                        oi_y_max = st.number_input(
                            "Y-Axis Max (basis points)",
                            value=1000,
                            step=50,
                            key=f"oi_funding_rate_y_max_{asset}"
                        )
                    
                    # Process OI-weighted funding rate history data
                    oi_funding_history_df = None
                    oi_key = f"api_futures_fundingRate_oi_weight_ohlc_history_{asset}"
                    
                    # Check if the OI-weighted file exists
                    if oi_key in data and not data[oi_key].empty:
                        # Process the data
                        oi_funding_history_df = process_timestamps(data[oi_key], timestamp_col="time")
                        
                        # Apply time filter
                        days_filter = time_range_options[oi_selected_time_range]
                        cutoff_date = None
                        
                        if days_filter:
                            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_filter)
                            
                        # Filter the data by time if needed
                        if cutoff_date and 'datetime' in oi_funding_history_df.columns:
                            oi_filtered_df = oi_funding_history_df[oi_funding_history_df['datetime'] >= cutoff_date]
                        else:
                            oi_filtered_df = oi_funding_history_df
                            
                        if not oi_filtered_df.empty:
                            # Create the figure
                            fig = go.Figure()
                            
                            # Add trace for the OI-weighted funding rate
                            fig.add_trace(go.Scatter(
                                x=oi_filtered_df['datetime'],
                                y=oi_filtered_df['close'],  # Use raw values, already in percentage
                                mode='lines',  # Only use lines mode, no markers or text
                                name="OI-Weighted",
                                line=dict(width=3, color="#00CC96"),  # Make it stand out with a distinct color
                                hovertemplate='%{y:.4f}<extra></extra>',  # No % symbol in hover
                            ))
                            
                            # Add a zero line - using filtered date range
                            if cutoff_date:
                                x0 = cutoff_date
                                x1 = pd.Timestamp.now()
                            else:
                                # Use full date range
                                x0 = oi_filtered_df['datetime'].min()
                                x1 = oi_filtered_df['datetime'].max()
                            
                            fig.add_shape(
                                type="line",
                                x0=x0,
                                y0=0,
                                x1=x1,
                                y1=0,
                                line=dict(color="gray", width=1, dash="dash"),
                            )
                            
                            # Update layout
                            fig.update_layout(
                                title=f"{asset} Open Interest Weighted Funding Rate History (in basis points)",
                                xaxis_title=None,
                                yaxis_title="Funding Rate (basis points)",  # Specify basis points
                                hovermode="x unified",
                            )
                            
                            # Format y-axis using user-provided range values (in basis points)
                            # Generate tick values based on the range
                            range_size = oi_y_max - oi_y_min
                            # Determine appropriate tick spacing based on range size
                            if range_size <= 50:
                                tick_step = 5
                            elif range_size <= 200:
                                tick_step = 20
                            elif range_size <= 1000:
                                tick_step = 100
                            else:
                                tick_step = 200
                                
                            # Generate tick values and labels
                            tick_vals = list(range(oi_y_min, oi_y_max + 1, tick_step))
                            tick_text = [str(val) for val in tick_vals]
                            
                            # Update the y-axis
                            fig.update_yaxes(
                                range=[oi_y_min, oi_y_max],  # Use user-provided range
                                tickvals=tick_vals,   # Dynamic tick values
                                ticktext=tick_text,   # Dynamic tick labels
                                title="Funding Rate (basis points)"  # Note that values are in basis points
                            )
                            
                            # Apply theme and display
                            fig = apply_chart_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation for OI-weighted funding rates
                            with st.expander("About OI-Weighted Funding Rates"):
                                st.markdown("**Open Interest Weighted Funding Rates Explained:**")
                                st.markdown("""
                                - This chart shows funding rates weighted by the Open Interest of each exchange.
                                - Exchanges with higher open interest contribute more to the weighted average.
                                - Values are shown in basis points (1 basis point = 0.01%).
                                - A value of 100 means 1%, 1000 means 10%, etc.
                                - Positive rates indicate longs pay shorts (bullish sentiment).
                                - Negative rates indicate shorts pay longs (bearish sentiment).
                                - This measure provides a more volume-representative view of market sentiment than simple averages.
                                """)
                        else:
                            st.info(f"No data available for the selected time range.")
                    else:
                        st.info(f"No Open Interest weighted funding rate data available for {asset}.")
                    
                    # Add a new section for Volume Weighted Funding Rate
                    st.subheader("Funding Rate History - Volume Weighted")
                    
                    # Create a container for the filters
                    vol_filter_container = st.container()
                    vol_filter_col1, vol_filter_col2, vol_filter_col3 = vol_filter_container.columns([2, 1, 1])
                    
                    # Create time range selector (reuse the same options)
                    with vol_filter_col1:
                        vol_selected_time_range = st.selectbox(
                            "Select Time Range",
                            list(time_range_options.keys()),
                            index=list(time_range_options.keys()).index("1M"),
                            key=f"vol_funding_rate_time_range_{asset}"
                        )
                        
                    # Add range selectors for y-axis
                    with vol_filter_col2:
                        vol_y_min = st.number_input(
                            "Y-Axis Min (basis points)",
                            value=-10,
                            step=5,
                            key=f"vol_funding_rate_y_min_{asset}"
                        )
                        
                    with vol_filter_col3:
                        vol_y_max = st.number_input(
                            "Y-Axis Max (basis points)",
                            value=1000,
                            step=50,
                            key=f"vol_funding_rate_y_max_{asset}"
                        )
                    
                    # Process volume-weighted funding rate history data
                    vol_funding_history_df = None
                    vol_key = f"api_futures_fundingRate_vol_weight_ohlc_history_{asset}"
                    
                    # Check if the volume-weighted file exists
                    if vol_key in data and not data[vol_key].empty:
                        # Process the data
                        vol_funding_history_df = process_timestamps(data[vol_key], timestamp_col="time")
                        
                        # Apply time filter
                        days_filter = time_range_options[vol_selected_time_range]
                        cutoff_date = None
                        
                        if days_filter:
                            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_filter)
                            
                        # Filter the data by time if needed
                        if cutoff_date and 'datetime' in vol_funding_history_df.columns:
                            vol_filtered_df = vol_funding_history_df[vol_funding_history_df['datetime'] >= cutoff_date]
                        else:
                            vol_filtered_df = vol_funding_history_df
                            
                        if not vol_filtered_df.empty:
                            # Create the figure
                            fig = go.Figure()
                            
                            # Add trace for the volume-weighted funding rate
                            fig.add_trace(go.Scatter(
                                x=vol_filtered_df['datetime'],
                                y=vol_filtered_df['close'],  # Use raw values, already in percentage
                                mode='lines',  # Only use lines mode, no markers or text
                                name="Volume-Weighted",
                                line=dict(width=3, color="#FF6692"),  # Make it stand out with a distinct color
                                hovertemplate='%{y:.4f}<extra></extra>',  # No % symbol in hover
                            ))
                            
                            # Add a zero line - using filtered date range
                            if cutoff_date:
                                x0 = cutoff_date
                                x1 = pd.Timestamp.now()
                            else:
                                # Use full date range
                                x0 = vol_filtered_df['datetime'].min()
                                x1 = vol_filtered_df['datetime'].max()
                            
                            fig.add_shape(
                                type="line",
                                x0=x0,
                                y0=0,
                                x1=x1,
                                y1=0,
                                line=dict(color="gray", width=1, dash="dash"),
                            )
                            
                            # Update layout
                            fig.update_layout(
                                title=f"{asset} Volume Weighted Funding Rate History (in basis points)",
                                xaxis_title=None,
                                yaxis_title="Funding Rate (basis points)",  # Specify basis points
                                hovermode="x unified",
                            )
                            
                            # Format y-axis using user-provided range values (in basis points)
                            # Generate tick values based on the range
                            range_size = vol_y_max - vol_y_min
                            # Determine appropriate tick spacing based on range size
                            if range_size <= 50:
                                tick_step = 5
                            elif range_size <= 200:
                                tick_step = 20
                            elif range_size <= 1000:
                                tick_step = 100
                            else:
                                tick_step = 200
                                
                            # Generate tick values and labels
                            tick_vals = list(range(vol_y_min, vol_y_max + 1, tick_step))
                            tick_text = [str(val) for val in tick_vals]
                            
                            # Update the y-axis
                            fig.update_yaxes(
                                range=[vol_y_min, vol_y_max],  # Use user-provided range
                                tickvals=tick_vals,   # Dynamic tick values
                                ticktext=tick_text,   # Dynamic tick labels
                                title="Funding Rate (basis points)"  # Note that values are in basis points
                            )
                            
                            # Apply theme and display
                            fig = apply_chart_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add explanation for volume-weighted funding rates
                            with st.expander("About Volume-Weighted Funding Rates"):
                                st.markdown("**Volume Weighted Funding Rates Explained:**")
                                st.markdown("""
                                - This chart shows funding rates weighted by the trading volume of each exchange.
                                - Exchanges with higher trading volume contribute more to the weighted average.
                                - Values are shown in basis points (1 basis point = 0.01%).
                                - A value of 100 means 1%, 1000 means 10%, etc.
                                - Positive rates indicate longs pay shorts (bullish sentiment).
                                - Negative rates indicate shorts pay longs (bearish sentiment).
                                - This measure highlights the funding rates in the most actively traded markets.
                                """)
                        else:
                            st.info(f"No data available for the selected time range.")
                    else:
                        st.info(f"No Volume weighted funding rate data available for {asset}.")
                    
                else:
                    st.info(f"No funding rate data available for {asset}.")
            except Exception as e:
                st.error(f"Error processing funding rate data: {e}")
                st.info("Unable to display funding rate metrics due to data format issues.")

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

def render_liquidation_page(asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render the liquidation page for the specified asset and selected exchanges.
    
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
    st.header(f"{asset} Liquidation Analysis")
    
    # Define available exchanges for liquidation
    available_exchanges = ["Binance", "OKX", "Bybit", "dYdX", "Bitfinex", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Store the available exchanges for use with each chart
    st.session_state.liquidation_available_exchanges = available_exchanges
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
    
    # Store in session state for this section
    st.session_state.selected_liquidation_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges
    
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

                    # Stats section removed as requested
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
                
                
                # Set defaults in session state for backward compatibility
                st.session_state.liquidation_time_range = 'All'
                st.session_state.selected_time_range = 'All'

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
                
                
                # Set defaults in session state for backward compatibility
                st.session_state.liquidation_ratio_time_range = 'All'
                st.session_state.selected_time_range = 'All'
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
                
            # Filter by selected exchanges if needed (if not "All")
            if 'exchange_name' in exchange_liq_df.columns and selected_exchanges and "All" not in selected_exchanges:
                exchange_liq_df = exchange_liq_df[exchange_liq_df['exchange_name'].isin(selected_exchanges)]

            # Check if we have the required columns now
            required_cols = ['exchange_name', 'total_liquidation_usd', 'long_liquidation_usd', 'short_liquidation_usd']
            if all(col in exchange_liq_df.columns for col in required_cols):
                # Exclude the 'all' row as requested
                exchange_liq_df = exchange_liq_df[~exchange_liq_df['exchange_name'].str.lower().isin(['all', 'ALL', 'All'])]
                
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

                # Create pie chart for liquidation distribution using enhanced version
                from utils.chart_utils import create_enhanced_pie_chart
                
                fig = create_enhanced_pie_chart(
                    exchange_liq_df.head(10),  # Top 10 exchanges
                    'total_liquidation_usd',
                    'exchange_name',
                    f"{asset} Liquidations Distribution by Exchange",
                    show_top_n=8,  # Show top 8 exchanges
                    min_percent=2.0,  # Group exchanges with less than 2% share
                    exclude_names=["All", "all", "ALL"]  # Ensure 'all' is excluded
                )

                display_chart(fig)

                
                # Create stacked bar chart showing long vs short by exchange
                st.subheader("Long vs. Short Liquidations by Exchange")

                # We already filtered out 'All' row from exchange_liq_df earlier
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
        st.info("Binance exchange data shown.")

def render_open_interest_page(asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render the open interest page for the specified asset and selected exchanges.
    
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
    st.header(f"{asset} Open Interest Analysis")
    
    # Exchange selector
    # Define available exchanges for open interest
    available_exchanges = ["Binance", "OKX", "Bybit", "dYdX", "Bitfinex", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Add exchange selector
    exchange_col1, exchange_col2 = st.columns([3, 1])
    with exchange_col1:
        selected_exchanges = st.multiselect(
            "Select Exchanges to Display",
            available_exchanges,
            default=default_exchanges,
            key=f"open_interest_exchange_selector_{asset}"
        )
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
        st.warning("At least one exchange must be selected. Defaulting to 'All'.")
    
    # Store in session state for this section
    st.session_state.selected_oi_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges
    
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
            
        # Filter by selected exchanges if needed (if not "All")
        if 'exchange_name' in oi_exchange_df.columns and selected_exchanges and "All" not in selected_exchanges:
            oi_exchange_df = oi_exchange_df[oi_exchange_df['exchange_name'].isin(selected_exchanges)]

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

            # Exclude the 'all' row as requested
            oi_exchange_df = oi_exchange_df[~oi_exchange_df['exchange_name'].str.lower().isin(['all', 'ALL', 'All'])]
            
            # Recalculate market share percentages after excluding 'all' row
            if 'market_share_percent' in oi_exchange_df.columns:
                total_oi = oi_exchange_df['open_interest_usd'].sum()
                oi_exchange_df['market_share_percent'] = (oi_exchange_df['open_interest_usd'] / total_oi * 100)
            
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

            # Create pie chart for open interest distribution using enhanced version
            from utils.chart_utils import create_enhanced_pie_chart
            
            fig = create_enhanced_pie_chart(
                oi_exchange_df.head(10),  # Top 10 exchanges
                'open_interest_usd',
                'exchange_name',
                f"{asset} Open Interest Distribution by Exchange",
                show_top_n=8,  # Show top 8 exchanges
                min_percent=2.0,  # Group exchanges with less than 2% share
                exclude_names=["All", "all", "ALL"]  # Ensure 'all' is excluded
            )
            
            display_chart(fig)
            
    else:
        st.info(f"No exchange open interest data available for {asset}.")
    
    # Open interest history
    st.subheader(f"{asset} Open Interest History (Coin-Margined)")
    
    # Set defaults in session state for backward compatibility
    st.session_state.oi_time_range = 'All'
    st.session_state.selected_time_range = 'All'

    # Default to Coin-Margined as requested
    oi_type = "Coin-Margined"

    # Try multiple possible key formats for Coin-Margined type
    oi_history_df = None
    possible_history_keys = [
        f"api_futures_openInterest_ohlc_aggregated_coin_margin_history_{asset}_{asset}",
        f"api_futures_openInterest_ohlc_aggregated_coin_margin_history_{asset}",
        "api_futures_openInterest_ohlc_aggregated_coin_margin_history"
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
            
            
            # Set defaults in session state for backward compatibility
            st.session_state.oi_history_time_range = 'All'
            st.session_state.selected_time_range = 'All'

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
    
    # Open Interest History by Exchange section removed as requested

def render_order_book_page(asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render the order book page for the specified asset and selected exchanges.
    
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
    st.header(f"{asset} Order Book Analysis")
    
    # Exchange selector
    # Define available exchanges for order book
    available_exchanges = ["Binance", "OKX", "Bybit", "dYdX", "Bitfinex", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Add exchange selector
    exchange_col1, exchange_col2 = st.columns([3, 1])
    with exchange_col1:
        selected_exchanges = st.multiselect(
            "Select Exchanges to Display",
            available_exchanges,
            default=default_exchanges,
            key=f"order_book_exchange_selector_{asset}"
        )
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
        st.warning("At least one exchange must be selected. Defaulting to 'All'.")
    
    # Store in session state for this section
    st.session_state.selected_order_book_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges
    
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

    # Removed duplicate exchange-specific order book visualization as requested
    # This section previously showed individual exchange order book data
    # The top aggregated charts are kept intact

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

def render_market_data_page(asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render the market data page for the specified asset and selected exchanges.
    
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
    st.header(f"{asset} Futures Market Data")
    
    # Exchange selector
    # Define available exchanges for market data
    available_exchanges = ["Binance", "OKX", "Bybit", "dYdX", "Bitfinex", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Add exchange selector
    exchange_col1, exchange_col2 = st.columns([3, 1])
    with exchange_col1:
        selected_exchanges = st.multiselect(
            "Select Exchanges to Display",
            available_exchanges,
            default=default_exchanges,
            key=f"market_data_exchange_selector_{asset}"
        )
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
        st.warning("At least one exchange must be selected. Defaulting to 'All'.")
    
    # Store in session state for this section
    st.session_state.selected_market_data_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges
    
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
        st.info(f"Limited API data for {asset}.")

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
                    
            # Filter by selected exchanges if needed (if not "All")
            if 'exchange_name' in pairs_df.columns and selected_exchanges and "All" not in selected_exchanges:
                pairs_df = pairs_df[pairs_df['exchange_name'].isin(selected_exchanges)]

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

    # Funding Rates section removed as requested

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

def render_long_short_ratio_page(asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render the long/short ratio page using Binance taker buy/sell volume history data.
    
    Parameters:
    -----------
    asset: str
        Primary asset to display (BTC, ETH, SOL, or XRP)
    all_selected_assets: list
        List of all selected assets to display
    selected_exchanges: list
        List of exchanges to display data for (not used in this implementation)
    selected_time_range: str
        Selected time range for filtering data (not used in this implementation)
    """
    
    st.header(f"{asset} Long/Short Ratio Analysis")
    
    # Add explanation about the data source
    st.info("This data shows the buy/sell volume ratio from Binance futures exchange. A ratio > 1 indicates more buying pressure than selling pressure.")
    
    # Create path to the taker buy/sell data file
    data_dir = get_latest_data_directory()
    if not data_dir:
        st.error("No data directory found.")
        return
        
    file_path = os.path.join(DATA_BASE_PATH, data_dir, 'futures', 'taker_buy_sell', f'api_futures_taker_buy_sell_volume_history_{asset}.parquet')
    
    # Check if file exists
    if not os.path.exists(file_path):
        st.warning(f"No taker buy/sell volume history data available for {asset}.")
        
        # Show empty placeholder layout
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Buy Volume", "$0")
        with col2:
            st.metric("Sell Volume", "$0")
        with col3:
            st.metric("Buy/Sell Ratio", "0.00")
            
        return
    
    # Load the data
    try:
        df = pd.read_parquet(file_path)
        
        if df.empty:
            st.warning(f"The taker buy/sell volume history file for {asset} is empty.")
            return
            
        # Process timestamps
        if 'time' in df.columns:
            # Typically these timestamps are in milliseconds
            df['datetime'] = pd.to_datetime(df['time'], unit='ms')
            
        # Check for required columns
        required_cols = ['aggregated_buy_volume_usd', 'aggregated_sell_volume_usd']
        if not all(col in df.columns for col in required_cols):
            available_cols = ", ".join(df.columns)
            st.error(f"Data is missing required columns. Available columns: {available_cols}")
            return
            
        # Sort by datetime
        df = df.sort_values('datetime')
        
        # Calculate buy/sell ratio
        df['buy_sell_ratio'] = df['aggregated_buy_volume_usd'] / df['aggregated_sell_volume_usd']
        
        # Get most recent values for metrics
        if not df.empty:
            recent_data = df.iloc[-1]
            buy_volume = recent_data['aggregated_buy_volume_usd']
            sell_volume = recent_data['aggregated_sell_volume_usd']
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
        st.subheader(f"{asset} Buy vs Sell Volume History")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['aggregated_buy_volume_usd'],
            name='Buy Volume',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=df['datetime'],
            y=df['aggregated_sell_volume_usd'],
            name='Sell Volume',
            line=dict(color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title=f"{asset} Buy/Sell Volume (Binance Futures)",
            xaxis_title=None,
            yaxis_title="Volume (USD)",
            hovermode="x unified",
            height=500
        )
        
        display_chart(apply_chart_theme(fig))
        
        # Create ratio chart
        st.subheader(f"{asset} Buy/Sell Ratio")
        
        fig = px.line(
            df,
            x='datetime',
            y='buy_sell_ratio',
            title=f"{asset} Buy/Sell Ratio (Binance Futures)",
            color_discrete_sequence=[ASSET_COLORS.get(asset, '#3366CC')]
        )
        
        # Add reference line at 1 (equal buy and sell)
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="gray",
            annotation_text="Equal"
        )
        
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Buy/Sell Ratio",
            height=500
        )
        
        display_chart(apply_chart_theme(fig))
        
        # Create net flow chart (buy - sell)
        st.subheader(f"{asset} Net Volume Flow")
        
        # Calculate net flow
        df['net_flow'] = df['aggregated_buy_volume_usd'] - df['aggregated_sell_volume_usd']
        
        fig = px.bar(
            df,
            x='datetime',
            y='net_flow',
            title=f"{asset} Net Volume Flow (Buy - Sell Volume) - Binance Futures",
            color='net_flow',
            color_continuous_scale=['red', 'green'],
            color_continuous_midpoint=0
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Net Flow (USD)",
            coloraxis_showscale=False,
            height=500
        )
        
        # Add reference line at 0
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="gray"
        )
        
        display_chart(apply_chart_theme(fig))
        
    except Exception as e:
        logger.error(f"Error processing taker buy/sell data for {asset}: {e}")
        st.error(f"Error processing data: {e}")
        st.info(f"There was an error loading the {asset} buy/sell volume data.")

def main():
    """Main function to render the futures page."""

    # Render sidebar
    render_sidebar()

    # Page title
    st.title("Futures")

    try:
        # Get available assets for futures category
        available_assets = get_available_assets_for_category('futures')

        # Set default assets if none are available
        if not available_assets:
            st.warning("No futures data available for any asset. Showing layout with placeholder data.")
            available_assets = ["BTC", "ETH", "SOL", "XRP"]
        
        # Asset selection with dropdown
        st.subheader("Select Asset to Display")
        
        # Initialize with previously selected asset if available, otherwise default to first asset
        default_asset = st.session_state.get('selected_futures_assets', [available_assets[0]])
        default_index = available_assets.index(default_asset[0]) if default_asset and default_asset[0] in available_assets else 0
        
        # Add dropdown for asset selection (improved from multiselect for better UI)
        selected_asset = st.selectbox(
            "Select asset to display",
            available_assets,
            index=default_index
        )
        
        # Use a single asset in a list for compatibility with existing code
        selected_assets = [selected_asset]
        
        # Store selected assets in session state for this page
        st.session_state.selected_futures_assets = selected_assets
        
        # For backward compatibility with existing code, use first selected asset as primary
        asset = selected_assets[0]
        
        # Also update the general selected_asset session state for compatibility
        st.session_state.selected_asset = asset

        # Define the categories
        futures_categories = [
            "Funding Rate",
            "Liquidation",
            "Open Interest",
            "Order Book",
            "Market Data",
            "Long/Short Ratio"
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
                # Use page-specific exchange selection if available, otherwise use general selection
                selected_funding_exchanges = st.session_state.get('selected_funding_rate_exchanges', 
                                                        st.session_state.get('selected_exchanges', ["All"]))
                render_funding_rate_page(asset, selected_assets, selected_funding_exchanges)
            except Exception as e:
                logger.error(f"Error rendering funding_rate tab: {e}")
                st.error(f"Error displaying funding rate data")
                st.info("There was an error processing the funding rate data. This could be due to an unexpected data format or missing data.")

        with tabs[1]:  # Liquidation
            try:
                subcategory = 'liquidation'
                st.session_state.futures_subcategory = subcategory
                # Use page-specific exchange selection if available, otherwise use general selection
                selected_liquidation_exchanges = st.session_state.get('selected_liquidation_exchanges', 
                                                          st.session_state.get('selected_exchanges', ["All"]))
                render_liquidation_page(asset, selected_assets, selected_liquidation_exchanges)
            except Exception as e:
                logger.error(f"Error rendering liquidation tab: {e}")
                st.error(f"Error displaying liquidation data")
                st.info("There was an error processing the liquidation data. This could be due to an unexpected data format or missing data.")

        with tabs[2]:  # Open Interest
            try:
                subcategory = 'open_interest'
                st.session_state.futures_subcategory = subcategory
                # Use page-specific exchange selection if available, otherwise use general selection
                selected_oi_exchanges = st.session_state.get('selected_oi_exchanges', 
                                                 st.session_state.get('selected_exchanges', ["All"]))
                render_open_interest_page(asset, selected_assets, selected_oi_exchanges)
            except Exception as e:
                logger.error(f"Error rendering open_interest tab: {e}")
                st.error(f"Error displaying open interest data")
                st.info("There was an error processing the open interest data. This could be due to an unexpected data format or missing data.")

        with tabs[3]:  # Order Book
            try:
                subcategory = 'order_book'
                st.session_state.futures_subcategory = subcategory
                # Use page-specific exchange selection if available, otherwise use general selection
                selected_orderbook_exchanges = st.session_state.get('selected_order_book_exchanges', 
                                                        st.session_state.get('selected_exchanges', ["All"]))
                render_order_book_page(asset, selected_assets, selected_orderbook_exchanges)
            except Exception as e:
                logger.error(f"Error rendering order_book tab: {e}")
                st.error(f"Error displaying order book data")
                st.info("There was an error processing the order book data. This could be due to an unexpected data format or missing data.")

        with tabs[4]:  # Market Data
            try:
                subcategory = 'market_data'
                st.session_state.futures_subcategory = subcategory
                # Use page-specific exchange selection if available, otherwise use general selection
                selected_market_exchanges = st.session_state.get('selected_market_data_exchanges', 
                                                     st.session_state.get('selected_exchanges', ["All"]))
                render_market_data_page(asset, selected_assets, selected_market_exchanges)
            except Exception as e:
                logger.error(f"Error rendering market_data tab: {e}")
                st.error(f"Error displaying market data")
                st.info("There was an error processing the market data. This could be due to an unexpected data format or missing data.")
                
        with tabs[5]:  # Long/Short Ratio
            try:
                subcategory = 'long_short_ratio'
                st.session_state.futures_subcategory = subcategory
                # Use page-specific exchange selection if available, otherwise use general selection
                selected_ls_exchanges = st.session_state.get('selected_long_short_ratio_exchanges', 
                                                 st.session_state.get('selected_exchanges', ["All"]))
                render_long_short_ratio_page(asset, selected_assets, selected_ls_exchanges)
            except Exception as e:
                logger.error(f"Error rendering long_short_ratio tab: {e}")
                st.error(f"Error displaying long/short ratio data")
                st.info("There was an error processing the long/short ratio data. This could be due to an unexpected data format or missing data.")

    except Exception as e:
        # If there's a critical error, log it and show a generic error message
        logger.error(f"Critical error in futures page: {e}")
        st.error("An error occurred while loading the futures page. Please check the logs for more information.")
    

if __name__ == "__main__":
    main()
