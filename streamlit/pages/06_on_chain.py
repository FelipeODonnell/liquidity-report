"""
On-Chain page for the Izun Crypto Liquidity Report.

This page displays data and visualizations related to exchange balances for multiple cryptocurrencies.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Add parent directory to path to import from components and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components and utilities
from components.sidebar import render_sidebar
from components.metrics import display_metrics_row
from components.charts import create_bar_chart, create_pie_chart, apply_chart_theme, display_chart
from components.tables import create_formatted_table
from utils.formatters import format_currency, format_percentage

# Set page config with title and icon
st.set_page_config(
    page_title="Izun Crypto Liquidity Report - Exchange Balances",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the current page for sidebar navigation
st.session_state.current_page = 'on_chain'

# Define available cryptocurrencies
CRYPTOCURRENCIES = ["BTC", "ETH"]

def load_exchange_balance_data(crypto="BTC"):
    """
    Load exchange balance data for a specific cryptocurrency from the specified path.
    
    Parameters:
    -----------
    crypto : str
        Cryptocurrency symbol (BTC, ETH, XRP, SOL)
    
    Returns:
    --------
    pandas.DataFrame or None
        DataFrame containing exchange balance data or None if file not found
    """
    try:
        # Import directly to access DATA_BASE_PATH
        from utils.config import DATA_BASE_PATH
        
        # Get the latest data directory
        from utils.data_loader import get_latest_data_directory
        latest_dir = get_latest_data_directory()
        
        if not latest_dir:
            st.error("No data directories found. Please check your data setup.")
            return None
        
        # Extract just the directory name from latest_dir (which is a full path)
        latest_dir_name = os.path.basename(latest_dir)
            
        # Construct path using DATA_BASE_PATH
        file_path = os.path.join(DATA_BASE_PATH, latest_dir_name, 'on_chain', f'api_exchange_balance_list_{crypto}.parquet')
        
        logger.info(f"Looking for exchange balance data at: {file_path}")
        
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            return df
        else:
            st.error(f"Exchange balance data file not found for {crypto} at: {file_path}")
            # Try a fallback approach - direct hardcoded path for local testing
            fallback_path = os.path.join("data", latest_dir_name, 'on_chain', f'api_exchange_balance_list_{crypto}.parquet')
            if os.path.exists(fallback_path):
                logger.info(f"Found data at fallback path: {fallback_path}")
                df = pd.read_parquet(fallback_path)
                return df
            return None
    except Exception as e:
        st.error(f"Error loading exchange balance data for {crypto}: {e}")
        logger.error(f"Error loading exchange balance data for {crypto}: {e}")
        return None

def display_exchange_balance_analysis(exchange_balance_df, crypto):
    """
    Display exchange balance analysis for a specific cryptocurrency.
    
    Parameters:
    -----------
    exchange_balance_df : pandas.DataFrame
        DataFrame containing exchange balance data
    crypto : str
        Cryptocurrency symbol (BTC, ETH, XRP, SOL)
    """
    if exchange_balance_df is None or exchange_balance_df.empty:
        st.error(f"No exchange balance data available for {crypto}.")
        return
    
    # Calculate metrics for summary
    try:
        # Calculate total balance
        total_balance = exchange_balance_df['total_balance'].sum()
        
        # Calculate balance changes
        change_1d = exchange_balance_df['balance_change_1d'].sum()
        change_7d = exchange_balance_df['balance_change_7d'].sum() if 'balance_change_7d' in exchange_balance_df.columns else None
        change_30d = exchange_balance_df['balance_change_30d'].sum() if 'balance_change_30d' in exchange_balance_df.columns else None
        
        # Calculate percentage changes
        change_pct_1d = (change_1d / (total_balance - change_1d)) * 100 if total_balance != change_1d else 0
        
        # Calculate weighted averages for percentage changes
        weights = exchange_balance_df['total_balance']
        change_pct_7d = (exchange_balance_df['balance_change_percent_7d'] * weights).sum() / weights.sum() if weights.sum() > 0 and 'balance_change_percent_7d' in exchange_balance_df.columns else None
        change_pct_30d = (exchange_balance_df['balance_change_percent_30d'] * weights).sum() / weights.sum() if weights.sum() > 0 and 'balance_change_percent_30d' in exchange_balance_df.columns else None
        
        # Display metrics
        metrics = {
            f"Total {crypto} Exchange Balance": {
                "value": total_balance,
                "delta": None
            },
            "24h Change %": {
                "value": change_pct_1d,
                "delta": change_1d,
                "delta_formatter": lambda x: format_currency(x, abbreviate=True),
                "delta_suffix": ""
            }
        }
        
        if change_pct_7d is not None:
            metrics["7d Change %"] = {
                "value": change_pct_7d,
                "delta": change_7d,
                "delta_formatter": lambda x: format_currency(x, abbreviate=True),
                "delta_suffix": ""
            }
        
        if change_pct_30d is not None:
            metrics["30d Change %"] = {
                "value": change_pct_30d,
                "delta": change_30d,
                "delta_formatter": lambda x: format_currency(x, abbreviate=True),
                "delta_suffix": ""
            }
        
        formatters = {
            f"Total {crypto} Exchange Balance": lambda x: format_currency(x, abbreviate=True),
            "24h Change %": lambda x: format_percentage(x),
            "7d Change %": lambda x: format_percentage(x),
            "30d Change %": lambda x: format_percentage(x)
        }
        
        display_metrics_row(metrics, formatters)
    except Exception as e:
        st.error(f"Error calculating metrics for {crypto}: {e}")
        logger.error(f"Error calculating metrics for {crypto}: {e}")
    
    # Display exchange balance table
    st.subheader(f"{crypto} Exchange Balance Data")
    
    # Sort by total balance
    exchange_balance_df = exchange_balance_df.sort_values(by='total_balance', ascending=False)
    
    # Create table with formatted columns
    create_formatted_table(
        exchange_balance_df,
        format_dict={
            'total_balance': lambda x: format_currency(x, abbreviate=True),
            'balance_change_1d': lambda x: format_currency(x, abbreviate=True),
            'balance_change_percent_1d': lambda x: format_percentage(x),
            'balance_change_7d': lambda x: format_currency(x, abbreviate=True) if 'balance_change_7d' in exchange_balance_df.columns else None,
            'balance_change_percent_7d': lambda x: format_percentage(x) if 'balance_change_percent_7d' in exchange_balance_df.columns else None,
            'balance_change_30d': lambda x: format_currency(x, abbreviate=True) if 'balance_change_30d' in exchange_balance_df.columns else None,
            'balance_change_percent_30d': lambda x: format_percentage(x) if 'balance_change_percent_30d' in exchange_balance_df.columns else None
        }
    )
    
    # Display visualizations in two columns
    col1, col2 = st.columns(2)
    
    # Create pie chart for exchange distribution
    with col1:
        st.subheader(f"{crypto} Exchange Balance Distribution")
        
        try:
            fig = create_pie_chart(
                exchange_balance_df.head(10),  # Top 10 exchanges
                'total_balance',
                'exchange_name',
                f"{crypto} Exchange Balance Distribution (Top 10)",
                height=500,
                show_outside_labels=True
            )
            
            display_chart(fig)
        except Exception as e:
            st.error(f"Error creating {crypto} balance distribution chart: {e}")
            logger.error(f"Error creating {crypto} balance distribution chart: {e}")
    
    # Create bar chart for 24h balance changes
    with col2:
        st.subheader(f"{crypto} 24h Balance Change by Exchange")
        
        try:
            # Filter for exchanges with significant changes
            significant_changes = exchange_balance_df[exchange_balance_df['balance_change_1d'].abs() > 0]
            
            # Sort by 1d balance change
            significant_changes = significant_changes.sort_values(by='balance_change_1d')
            
            # Get top 5 inflows and outflows
            top_inflows = significant_changes.tail(5)
            top_outflows = significant_changes.head(5)
            top_changes = pd.concat([top_outflows, top_inflows])
            
            # Create horizontal bar chart for better readability
            fig = px.bar(
                top_changes,
                y='exchange_name',
                x='balance_change_1d',
                title=f"Top {crypto} Exchange Balance Changes (24h)",
                color='balance_change_1d',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0,
                orientation='h'
            )
            
            fig.update_layout(
                yaxis_title=None,
                xaxis_title="Balance Change (24h)",
                height=400
            )
            
            display_chart(apply_chart_theme(fig))
        except Exception as e:
            st.error(f"Error creating {crypto} 24h balance change chart: {e}")
            logger.error(f"Error creating {crypto} 24h balance change chart: {e}")
    
    # Create bar charts for 7d and 30d balance changes
    col3, col4 = st.columns(2)
    
    # 7-day balance change
    with col3:
        st.subheader(f"{crypto} 7-Day Balance Change by Exchange")
        
        try:
            if 'balance_change_7d' in exchange_balance_df.columns:
                # Filter for exchanges with significant changes
                significant_changes_7d = exchange_balance_df[exchange_balance_df['balance_change_7d'].abs() > 0]
                
                # Sort by 7d balance change
                significant_changes_7d = significant_changes_7d.sort_values(by='balance_change_7d')
                
                # Get top 5 inflows and outflows
                top_inflows_7d = significant_changes_7d.tail(5)
                top_outflows_7d = significant_changes_7d.head(5)
                top_changes_7d = pd.concat([top_outflows_7d, top_inflows_7d])
                
                # Create horizontal bar chart for better readability
                fig = px.bar(
                    top_changes_7d,
                    y='exchange_name',
                    x='balance_change_7d',
                    title=f"Top {crypto} Exchange Balance Changes (7d)",
                    color='balance_change_7d',
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0,
                    orientation='h'
                )
                
                fig.update_layout(
                    yaxis_title=None,
                    xaxis_title="Balance Change (7d)",
                    height=400
                )
                
                display_chart(apply_chart_theme(fig))
            else:
                st.info(f"{crypto} 7-day balance change data not available.")
        except Exception as e:
            st.error(f"Error creating {crypto} 7-day balance change chart: {e}")
            logger.error(f"Error creating {crypto} 7-day balance change chart: {e}")
    
    # 30-day balance change
    with col4:
        st.subheader(f"{crypto} 30-Day Balance Change by Exchange")
        
        try:
            if 'balance_change_30d' in exchange_balance_df.columns:
                # Filter for exchanges with significant changes
                significant_changes_30d = exchange_balance_df[exchange_balance_df['balance_change_30d'].abs() > 0]
                
                # Sort by 30d balance change
                significant_changes_30d = significant_changes_30d.sort_values(by='balance_change_30d')
                
                # Get top 5 inflows and outflows
                top_inflows_30d = significant_changes_30d.tail(5)
                top_outflows_30d = significant_changes_30d.head(5)
                top_changes_30d = pd.concat([top_outflows_30d, top_inflows_30d])
                
                # Create horizontal bar chart for better readability
                fig = px.bar(
                    top_changes_30d,
                    y='exchange_name',
                    x='balance_change_30d',
                    title=f"Top {crypto} Exchange Balance Changes (30d)",
                    color='balance_change_30d',
                    color_continuous_scale='RdBu',
                    color_continuous_midpoint=0,
                    orientation='h'
                )
                
                fig.update_layout(
                    yaxis_title=None,
                    xaxis_title="Balance Change (30d)",
                    height=400
                )
                
                display_chart(apply_chart_theme(fig))
            else:
                st.info(f"{crypto} 30-day balance change data not available.")
        except Exception as e:
            st.error(f"Error creating {crypto} 30-day balance change chart: {e}")
            logger.error(f"Error creating {crypto} 30-day balance change chart: {e}")

def main():
    """Main function to render the exchange balance page."""
    
    # Render sidebar
    render_sidebar()
    
    # Page title only
    st.title("Exchange Balances")
    
    # Create tabs for each cryptocurrency
    tabs = st.tabs(CRYPTOCURRENCIES)
    
    # Load data for each cryptocurrency and display in respective tabs
    for i, crypto in enumerate(CRYPTOCURRENCIES):
        with tabs[i]:
            st.header(f"{crypto} Exchange Balances")
            
            # Display loading message
            with st.spinner(f"Loading {crypto} exchange balance data..."):
                exchange_balance_df = load_exchange_balance_data(crypto)
            
            # Display analysis for this cryptocurrency
            display_exchange_balance_analysis(exchange_balance_df, crypto)
    
    # Add explanation of exchange balance significance
    st.markdown("""
    ### Exchange Balance Data
    
    Exchange balance represents the total amount of cryptocurrency held on exchanges:
    
    - **Decreasing balance**: Generally considered bullish as it suggests coins are being moved to cold storage for long-term holding
    - **Increasing balance**: Can be bearish as it might indicate investors are preparing to sell
    - **Sudden large outflows**: Could indicate institutional purchases or exchange security issues
    - **Sudden large inflows**: Might precede increased selling pressure
    
    The data shown here provides insights into:
    - Total balance held on major exchanges for BTC and ETH
    - 24-hour, 7-day, and 30-day changes in exchange balances
    - Distribution of balances across top exchanges
    - Specific exchanges experiencing the largest inflows and outflows
    """)

if __name__ == "__main__":
    main()