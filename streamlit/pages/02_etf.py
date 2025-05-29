"""
ETF page for the Izun Crypto Liquidity Report.

This page displays data and visualizations related to cryptocurrency ETFs.
"""

import streamlit as st
import pandas as pd
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
from utils.config import APP_TITLE, APP_ICON

st.set_page_config(
    page_title=f"{APP_TITLE} - ETFs",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.session_state.current_page = 'etf'

def load_etf_data():
    """
    Load ETF data for the page.
    
    Returns:
    --------
    dict
        Dictionary containing ETF data
    """
    data = {}
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return data
    
    # Load all ETF data
    data = load_data_for_category('etf')
    
    return data

def main():
    """Main function to render the ETF page."""
    
    # Render sidebar
    render_sidebar()
    
    # Page title and description
    st.title("ETF")
    st.write("Analysis and visualization of Bitcoin and Ethereum ETFs")
    
    # Display loading message
    with st.spinner("Loading ETF data..."):
        data = load_etf_data()
    
    # Removed data last updated reference
    
    # Check if data is available
    if not data:
        st.error("No ETF data available.")
        return
    
    # Create tabs for different ETF types
    tab1, tab2 = st.tabs(["Bitcoin ETFs", "Ethereum ETFs"])
    
    with tab1:
        st.header("Bitcoin ETFs")
        
        # ETF Overview Metrics
        if 'api_etf_bitcoin_list' in data and not data['api_etf_bitcoin_list'].empty:
            btc_etf_df = data['api_etf_bitcoin_list']
            
            # Calculate metrics
            total_aum = pd.to_numeric(btc_etf_df['aum_usd'], errors='coerce').sum()
            us_spot_etfs = btc_etf_df[(btc_etf_df['region'] == 'us') & (btc_etf_df['fund_type'] == 'Spot')]
            total_us_spot_aum = pd.to_numeric(us_spot_etfs['aum_usd'], errors='coerce').sum()
            
            # Get flow data
            if 'api_etf_bitcoin_flow_history' in data and not data['api_etf_bitcoin_flow_history'].empty:
                flow_df = data['api_etf_bitcoin_flow_history']

                # Determine which column to use for flow data
                flow_col = None
                if 'fund_flow_usd' in flow_df.columns:
                    flow_col = 'fund_flow_usd'
                else:
                    # Try to find another suitable column
                    flow_cols = [col for col in flow_df.columns if 'flow' in col.lower()]
                    if flow_cols:
                        flow_col = flow_cols[0]
                        logger.info(f"Using alternative flow column in ETF page: {flow_col}")

                if flow_col:
                    latest_flow = flow_df[flow_col].iloc[-1] if len(flow_df) > 0 else 0

                    # Calculate recent flows
                    flow_df = process_timestamps(flow_df)
                    if 'datetime' in flow_df.columns:
                        recent_df = flow_df[flow_df['datetime'] >= (datetime.now() - timedelta(days=7))]
                        last_7d_flow = recent_df[flow_col].sum() if not recent_df.empty else 0
                    else:
                        logger.warning("No datetime column in flow_df")
                        last_7d_flow = 0
                else:
                    logger.warning("No suitable flow column found in ETF data")
                    latest_flow = 0
                    last_7d_flow = 0
            else:
                latest_flow = None
                last_7d_flow = None
            
            # Display metrics
            metrics = {
                "Total AUM": {
                    "value": total_aum,
                    "delta": None
                },
                "US Spot ETF AUM": {
                    "value": total_us_spot_aum,
                    "delta": None
                },
                "7d Fund Flow": {
                    "value": last_7d_flow,
                    "delta": None
                }
            }
            
            formatters = {
                "Total AUM": lambda x: format_currency(x, abbreviate=True),
                "US Spot ETF AUM": lambda x: format_currency(x, abbreviate=True),
                "7d Fund Flow": lambda x: format_currency(x, abbreviate=True)
            }
            
            display_metrics_row(metrics, formatters)
            
            # Bitcoin ETF List
            st.subheader("Bitcoin ETF List")
            
            # Capitalize region column values
            if 'region' in btc_etf_df.columns:
                btc_etf_df['region'] = btc_etf_df['region'].str.upper()
            
            # Convert aum_usd to numeric for proper sorting
            btc_etf_df['aum_usd'] = pd.to_numeric(btc_etf_df['aum_usd'], errors='coerce')
            
            # Sort by AUM (highest first)
            btc_etf_df = btc_etf_df.sort_values(by='aum_usd', ascending=False)
            
            # Create a copy with numeric values for the chart before formatting
            btc_etf_chart_data = btc_etf_df.copy()
            
            # Format AUM USD values for display in table only
            if 'aum_usd' in btc_etf_df.columns:
                btc_etf_df['aum_usd_formatted'] = btc_etf_df['aum_usd'].apply(
                    lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
                )
                # Replace the original column for display
                btc_etf_df['aum_usd'] = btc_etf_df['aum_usd_formatted']
                btc_etf_df = btc_etf_df.drop('aum_usd_formatted', axis=1)
            
            # Display table
            create_etf_table(btc_etf_df)
            
            # AUM Visualization
            st.subheader("Assets Under Management")
            
            # Create pie chart for AUM distribution with improved formatting
            # Import our enhanced chart utility
            from utils.chart_utils import create_enhanced_pie_chart
            
            # Use the chart data with numeric values for the pie chart
            filtered_etfs = btc_etf_chart_data.copy()
            filtered_etfs = filtered_etfs.dropna(subset=['aum_usd'])
            filtered_etfs = filtered_etfs[filtered_etfs['aum_usd'] > 0]
            
            # Create the enhanced pie chart
            fig = create_enhanced_pie_chart(
                filtered_etfs,
                'aum_usd',
                'ticker',
                "Bitcoin ETF AUM Distribution (Top 10)",
                show_top_n=10,  # Show top 10 ETFs
                min_percent=1.0  # Only group ETFs with less than 1% share
            )
            
            display_chart(fig)
            
            # ETF Flow Analysis
            if 'api_etf_bitcoin_flow_history' in data and not data['api_etf_bitcoin_flow_history'].empty:
                st.subheader("Bitcoin ETF Flows")
                
                flow_df = process_timestamps(data['api_etf_bitcoin_flow_history'])
                
                # Determine which column to use for flow data
                flow_col = 'flow_usd'
                if 'fund_flow_usd' in flow_df.columns:
                    flow_col = 'fund_flow_usd'
                elif 'flow_usd' in flow_df.columns:
                    flow_col = 'flow_usd'
                elif 'etf_flows' in flow_df.columns:
                    flow_col = 'etf_flows'

                # Daily flows chart
                fig = px.bar(
                    flow_df,
                    x='datetime',
                    y=flow_col,
                    title="Daily Bitcoin ETF Flows",
                    color=flow_df[flow_col] > 0,
                    color_discrete_map={True: 'green', False: 'red'}
                )
                
                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="Fund Flow (USD)",
                    showlegend=False
                )
                
                display_chart(apply_chart_theme(fig))
                
                # Set defaults in session state for backward compatibility
                st.session_state.btc_etf_flows_time_range = 'All'
                st.session_state.selected_time_range = 'All'
                
                # Cumulative flows chart
                flow_df['cumulative_flow'] = flow_df[flow_col].cumsum()

                fig = px.line(
                    flow_df,
                    x='datetime',
                    y='cumulative_flow',
                    title="Cumulative Bitcoin ETF Flows"
                )
                
                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="Cumulative Flow (USD)"
                )
                
                display_chart(apply_chart_theme(fig))
                
                # Flow vs. Price Change
                if 'price_change_percent' in flow_df.columns:
                    st.subheader("ETF Flows vs. Price Change")
                    
                    fig = create_time_series_with_bar(
                        flow_df,
                        'datetime',
                        'price_change_percent',
                        flow_col,
                        "Bitcoin ETF Flows vs. Price Change (%)"
                    )
                    
                    display_chart(fig)
            
            # Premium/Discount Analysis
            if 'api_etf_bitcoin_premium_discount_history' in data and not data['api_etf_bitcoin_premium_discount_history'].empty:
                st.subheader("Premium/Discount Analysis")

                premium_df = process_timestamps(data['api_etf_bitcoin_premium_discount_history'])

                # Check if the data was successfully normalized
                if 'premium_discount_details' in premium_df.columns:
                    # Rename column for clarity
                    premium_df = premium_df.rename(columns={'premium_discount_details': 'premium_discount_percent'})

                    # Average premium/discount by ticker
                    avg_premium_df = premium_df.groupby(['ticker', 'datetime'])['premium_discount_percent'].mean().reset_index()

                    # Create time series chart for each ETF
                    st.write("Premium/Discount by ETF (%)")

                    fig = px.line(
                        avg_premium_df,
                        x='datetime',
                        y='premium_discount_percent',
                        color='ticker',
                        title="Bitcoin ETF Premium/Discount to NAV (%)"
                    )

                    # Add zero line
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="NAV"
                    )

                    display_chart(fig)

                    # Also show average across all ETFs
                    st.write("Average Premium/Discount Across All ETFs (%)")

                    avg_all_df = premium_df.groupby('datetime')['premium_discount_percent'].mean().reset_index()

                    fig = create_time_series(
                        avg_all_df,
                        'datetime',
                        'premium_discount_percent',
                        "Average Bitcoin ETF Premium/Discount to NAV (%)"
                    )

                    # Add zero line
                    fig.add_hline(
                        y=0,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="NAV"
                    )

                    display_chart(fig)
                else:
                    st.warning("Premium/Discount data is not in the expected format. Unable to create visualization.")
        else:
            st.info("Bitcoin ETF data not available.")
    
    with tab2:
        st.header("Ethereum ETFs")
        
        # Ethereum ETF Overview
        if 'api_etf_ethereum_list' in data and not data['api_etf_ethereum_list'].empty:
            eth_etf_df = data['api_etf_ethereum_list']
            
            # Calculate metrics
            total_aum = pd.to_numeric(eth_etf_df['aum_usd'], errors='coerce').sum()
            us_spot_etfs = eth_etf_df[(eth_etf_df['region'] == 'us') & (eth_etf_df['fund_type'] == 'Spot')]
            total_us_spot_aum = pd.to_numeric(us_spot_etfs['aum_usd'], errors='coerce').sum()
            
            # Get flow data
            if 'api_etf_ethereum_flow_history' in data and not data['api_etf_ethereum_flow_history'].empty:
                flow_df = data['api_etf_ethereum_flow_history']
                # Determine which column to use for flow data
                flow_col = 'flow_usd'
                if 'fund_flow_usd' in flow_df.columns:
                    flow_col = 'fund_flow_usd'
                elif 'flow_usd' in flow_df.columns:
                    flow_col = 'flow_usd'
                elif 'etf_flows' in flow_df.columns:
                    flow_col = 'etf_flows'

                latest_flow = flow_df[flow_col].iloc[-1] if len(flow_df) > 0 else 0

                # Calculate recent flows
                flow_df = process_timestamps(flow_df)
                recent_df = flow_df[flow_df['datetime'] >= (datetime.now() - timedelta(days=7))]
                last_7d_flow = recent_df[flow_col].sum() if not recent_df.empty else 0
            else:
                latest_flow = None
                last_7d_flow = None
            
            # Display metrics
            metrics = {
                "Total AUM": {
                    "value": total_aum,
                    "delta": None
                },
                "US Spot ETF AUM": {
                    "value": total_us_spot_aum,
                    "delta": None
                },
                "7d Fund Flow": {
                    "value": last_7d_flow,
                    "delta": None
                }
            }
            
            formatters = {
                "Total AUM": lambda x: format_currency(x, abbreviate=True),
                "US Spot ETF AUM": lambda x: format_currency(x, abbreviate=True),
                "7d Fund Flow": lambda x: format_currency(x, abbreviate=True) if x is not None else "N/A"
            }
            
            display_metrics_row(metrics, formatters)
            
            # Ethereum ETF List
            st.subheader("Ethereum ETF List")
            
            # Capitalize region column values
            if 'region' in eth_etf_df.columns:
                eth_etf_df['region'] = eth_etf_df['region'].str.upper()
            
            # Convert aum_usd to numeric for proper sorting
            eth_etf_df['aum_usd'] = pd.to_numeric(eth_etf_df['aum_usd'], errors='coerce')
            
            # Sort by AUM (highest first)
            eth_etf_df = eth_etf_df.sort_values(by='aum_usd', ascending=False)
            
            # Create a copy with numeric values for the chart before formatting
            eth_etf_chart_data = eth_etf_df.copy()
            
            # Format AUM USD values for display in table only
            if 'aum_usd' in eth_etf_df.columns:
                eth_etf_df['aum_usd_formatted'] = eth_etf_df['aum_usd'].apply(
                    lambda x: f"${x:,.0f}" if pd.notnull(x) else "N/A"
                )
                # Replace the original column for display
                eth_etf_df['aum_usd'] = eth_etf_df['aum_usd_formatted']
                eth_etf_df = eth_etf_df.drop('aum_usd_formatted', axis=1)
            
            # Display table
            create_etf_table(eth_etf_df)
            
            # AUM Visualization
            st.subheader("Assets Under Management")
            
            # Create pie chart for AUM distribution with improved formatting
            # Use the chart data with numeric values for the pie chart
            filtered_eth_etfs = eth_etf_chart_data.copy()
            filtered_eth_etfs = filtered_eth_etfs.dropna(subset=['aum_usd'])
            filtered_eth_etfs = filtered_eth_etfs[filtered_eth_etfs['aum_usd'] > 0]
            
            # Create the enhanced pie chart
            fig = create_enhanced_pie_chart(
                filtered_eth_etfs,
                'aum_usd',
                'ticker',
                "Ethereum ETF AUM Distribution",
                show_top_n=10,  # Show top 10 ETFs
                min_percent=1.0  # Only group ETFs with less than 1% share
            )
            
            display_chart(fig)
            
            # ETF Flow Analysis
            if 'api_etf_ethereum_flow_history' in data and not data['api_etf_ethereum_flow_history'].empty:
                st.subheader("Ethereum ETF Flows")
                
                flow_df = process_timestamps(data['api_etf_ethereum_flow_history'])
                
                # Determine which column to use for flow data
                flow_col = 'flow_usd'
                if 'fund_flow_usd' in flow_df.columns:
                    flow_col = 'fund_flow_usd'
                elif 'flow_usd' in flow_df.columns:
                    flow_col = 'flow_usd'
                elif 'etf_flows' in flow_df.columns:
                    flow_col = 'etf_flows'

                # Daily flows chart
                fig = px.bar(
                    flow_df,
                    x='datetime',
                    y=flow_col,
                    title="Daily Ethereum ETF Flows",
                    color=flow_df[flow_col] > 0,
                    color_discrete_map={True: 'green', False: 'red'}
                )
                
                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="Fund Flow (USD)",
                    showlegend=False
                )
                
                display_chart(apply_chart_theme(fig))
                
                # Determine which column to use for flow data
                flow_col = 'flow_usd'
                if 'fund_flow_usd' in flow_df.columns:
                    flow_col = 'fund_flow_usd'
                elif 'flow_usd' in flow_df.columns:
                    flow_col = 'flow_usd'
                elif 'etf_flows' in flow_df.columns:
                    flow_col = 'etf_flows'

                # Cumulative flows chart
                flow_df['cumulative_flow'] = flow_df[flow_col].cumsum()

                fig = px.line(
                    flow_df,
                    x='datetime',
                    y='cumulative_flow',
                    title="Cumulative Ethereum ETF Flows"
                )
                
                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="Cumulative Flow (USD)"
                )
                
                display_chart(apply_chart_theme(fig))
            
            # Net Assets History
            if 'api_etf_ethereum_net_assets_history' in data and not data['api_etf_ethereum_net_assets_history'].empty:
                st.subheader("Net Assets History")

                assets_df = process_timestamps(data['api_etf_ethereum_net_assets_history'])

                # Check which column to use
                assets_col = None

                if 'net_assets_total' in assets_df.columns:
                    assets_col = 'net_assets_total'
                elif 'net_assets_usd' in assets_df.columns:
                    assets_col = 'net_assets_usd'
                else:
                    # Look for columns with 'assets' in the name
                    assets_cols = [col for col in assets_df.columns if 'assets' in col.lower()]
                    assets_col = assets_cols[0] if assets_cols else None

                # Check for nested data structure with data_list
                if assets_col is None and 'data_list' in assets_df.columns:
                    try:
                        logger.info("Attempting to extract net assets from data_list column")

                        # Get first non-null value to check type
                        sample_value = None
                        for idx, val in enumerate(assets_df['data_list']):
                            if pd.notna(val) and val is not None:
                                sample_value = val
                                break

                        if sample_value is not None:
                            # First check if it's a numpy array
                            if hasattr(sample_value, '__array__'):
                                try:
                                    # Try direct conversion to list
                                    processed_val = sample_value.tolist()
                                    if isinstance(processed_val, (int, float)):
                                        # If it converts to a number, create a new column
                                        assets_df['net_assets_usd'] = assets_df['data_list'].apply(
                                            lambda x: x.tolist() if hasattr(x, '__array__') and pd.notna(x) else x)
                                        assets_col = 'net_assets_usd'
                                    elif isinstance(processed_val, list):
                                        # If it's a list, take the first element if it's numeric
                                        if len(processed_val) > 0 and isinstance(processed_val[0], (int, float)):
                                            assets_df['net_assets_usd'] = assets_df['data_list'].apply(
                                                lambda x: x.tolist()[0] if hasattr(x, '__array__') and pd.notna(x) and
                                                        len(x.tolist()) > 0 else 0)
                                            assets_col = 'net_assets_usd'
                                except Exception as e_convert:
                                    logger.warning(f"Failed to convert numpy array in net assets data: {e_convert}")

                            # If not a numpy array or conversion failed, try other approaches
                            if assets_col is None:
                                if isinstance(sample_value, (int, float)):
                                    # It's already a number
                                    assets_df['net_assets_usd'] = assets_df['data_list']
                                    assets_col = 'net_assets_usd'
                                elif isinstance(sample_value, list):
                                    # If it's a list, extract first element if numeric
                                    if len(sample_value) > 0 and isinstance(sample_value[0], (int, float)):
                                        try:
                                            assets_df['net_assets_usd'] = assets_df['data_list'].apply(
                                                lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 0)
                                            assets_col = 'net_assets_usd'
                                        except Exception as e_list:
                                            logger.warning(f"Failed to extract net assets from list: {e_list}")
                                elif isinstance(sample_value, dict):
                                    # It's a dict, look for a key that might contain assets data
                                    assets_keys = [k for k in sample_value.keys() if 'assets' in k.lower()]
                                    if assets_keys:
                                        try:
                                            key = assets_keys[0]
                                            assets_df['net_assets_usd'] = assets_df['data_list'].apply(
                                                lambda x: x.get(key, 0) if isinstance(x, dict) else 0)
                                            assets_col = 'net_assets_usd'
                                        except Exception as e_dict:
                                            logger.warning(f"Failed to extract net assets from dict: {e_dict}")
                        else:
                            logger.warning("Could not find a valid sample in data_list for net assets")
                    except Exception as e:
                        logger.error(f"Error processing nested net assets data: {e}")

                # If we found a valid column, create the chart
                if assets_col:
                    # Make sure the data is sorted by datetime
                    if 'datetime' in assets_df.columns:
                        assets_df = assets_df.sort_values('datetime')

                        # Ensure the assets column is numeric
                        assets_df[assets_col] = pd.to_numeric(assets_df[assets_col], errors='coerce')

                        # Drop any rows with NaN values in the assets column
                        valid_data = assets_df.dropna(subset=[assets_col])

                        if not valid_data.empty:
                            fig = create_time_series(
                                valid_data,
                                'datetime',
                                assets_col,
                                "Ethereum ETF Net Assets History"
                            )
                            display_chart(fig)
                        else:
                            st.warning("No valid net assets data points found after processing")
                    else:
                        st.warning("Net assets data is missing datetime column")
                else:
                    st.warning(f"Could not find or extract net assets data. Available columns: {list(assets_df.columns)}")
            else:
                st.info("Ethereum ETF net assets history data not available.")
        else:
            st.info("Ethereum ETF data not available.")
    
    # Grayscale Funds and Hong Kong ETFs tabs removed as requested
    

if __name__ == "__main__":
    main()