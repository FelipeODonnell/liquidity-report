"""
On-Chain page for the Izun Crypto Liquidity Report.

This page displays data and visualizations related to cryptocurrency on-chain metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

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
from components.tables import create_formatted_table
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
from utils.config import APP_TITLE, APP_ICON, ASSET_COLORS

# Set page config with title and icon
st.set_page_config(
    page_title=f"{APP_TITLE} - On-Chain Metrics",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the current page for sidebar navigation
st.session_state.current_page = 'on_chain'

def load_onchain_data():
    """
    Load on-chain data for the page.
    
    Returns:
    --------
    dict
        Dictionary containing on-chain data
    """
    data = {}
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return data
    
    # Load on-chain data
    data = load_data_for_category('on_chain')
    
    return data

def render_exchange_balance(data):
    """Render exchange balance visualizations."""
    st.header("Exchange Balance")
    
    if 'api_exchange_balance_list' in data and not data['api_exchange_balance_list'].empty:
        balance_df = data['api_exchange_balance_list']
        
        # Calculate metrics
        total_balance = balance_df['total_balance'].sum()
        
        # Calculate balance changes
        change_1d = balance_df['balance_change_1d'].sum()
        
        # Create percentage changes
        if 'balance_change_percent_7d' in balance_df.columns and 'balance_change_percent_30d' in balance_df.columns:
            # Calculate weighted averages for percentage changes
            weights = balance_df['total_balance']
            change_pct_7d = (balance_df['balance_change_percent_7d'] * weights).sum() / weights.sum() if weights.sum() > 0 else 0
            change_pct_30d = (balance_df['balance_change_percent_30d'] * weights).sum() / weights.sum() if weights.sum() > 0 else 0
        else:
            change_pct_7d = None
            change_pct_30d = None
        
        # Display metrics
        metrics = {
            "Total Exchange Balance": {
                "value": total_balance,
                "delta": None
            },
            "24h Balance Change": {
                "value": change_1d,
                "delta": (change_1d / (total_balance - change_1d)) * 100 if total_balance != change_1d else None,
                "delta_suffix": "%"
            }
        }
        
        if change_pct_7d is not None:
            metrics["7d Balance Change"] = {
                "value": change_pct_7d,
                "delta": None,
                "delta_suffix": "%"
            }
        
        if change_pct_30d is not None:
            metrics["30d Balance Change"] = {
                "value": change_pct_30d,
                "delta": None,
                "delta_suffix": "%"
            }
        
        formatters = {
            "Total Exchange Balance": lambda x: format_currency(x, abbreviate=True),
            "24h Balance Change": lambda x: format_currency(x, abbreviate=True),
            "7d Balance Change": lambda x: format_percentage(x),
            "30d Balance Change": lambda x: format_percentage(x)
        }
        
        display_metrics_row(metrics, formatters)
        
        # Exchange balance table
        st.subheader("Exchange Balance by Exchange")
        
        # Sort by total balance
        balance_df = balance_df.sort_values(by='total_balance', ascending=False)
        
        # Create table
        create_formatted_table(
            balance_df,
            format_dict={
                'total_balance': lambda x: format_currency(x, abbreviate=True),
                'balance_change_1d': lambda x: format_currency(x, abbreviate=True),
                'balance_change_percent_7d': lambda x: format_percentage(x) if 'balance_change_percent_7d' in balance_df.columns else None,
                'balance_change_percent_30d': lambda x: format_percentage(x) if 'balance_change_percent_30d' in balance_df.columns else None
            }
        )
        
        # Create pie chart for exchange distribution
        fig = create_pie_chart(
            balance_df.head(10),  # Top 10 exchanges
            'total_balance',
            'exchange_name',
            "Exchange Balance Distribution (Top 10)"
        )
        
        display_chart(fig)
        
        # Create bar chart for 24h balance change by exchange
        st.subheader("24h Balance Change by Exchange")
        
        # Filter for exchanges with significant changes
        significant_changes = balance_df[balance_df['balance_change_1d'].abs() > 0]
        significant_changes = significant_changes.sort_values(by='balance_change_1d')
        
        # Get top 5 inflows and outflows
        top_inflows = significant_changes.tail(5)
        top_outflows = significant_changes.head(5)
        top_changes = pd.concat([top_outflows, top_inflows])
        
        # Create bar chart
        fig = px.bar(
            top_changes,
            x='exchange_name',
            y='balance_change_1d',
            title="Top Exchange Balance Changes (24h)",
            color='balance_change_1d',
            color_continuous_scale='RdBu',
            color_continuous_midpoint=0
        )
        
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Balance Change"
        )
        
        display_chart(apply_chart_theme(fig))
        
        # Explanation
        st.markdown("""
        ### Understanding Exchange Balance
        
        Exchange balance represents the total amount of cryptocurrency held on exchanges:
        
        - **Decreasing balance**: Generally considered bullish as it suggests coins are being moved to cold storage for long-term holding
        - **Increasing balance**: Can be bearish as it might indicate investors are preparing to sell
        - **Sudden large outflows**: Could indicate institutional purchases or could precede exchange security issues
        - **Sudden large inflows**: Might precede increased selling pressure
        """)
    else:
        st.info("Exchange balance data not available.")

def render_exchange_chain_transactions(data):
    """Render exchange chain transaction visualizations."""
    st.header("Exchange Chain Transactions")
    
    if 'api_exchange_chain_tx_list' in data and not data['api_exchange_chain_tx_list'].empty:
        tx_df = data['api_exchange_chain_tx_list']
        
        # Calculate metrics based on available columns
        if 'amount_usd' in tx_df.columns and 'transfer_type' in tx_df.columns:
            # Using transfer_type to identify inflow/outflow
            # Assuming transfer_type 0 is inflow, 1 is outflow
            inflow_df = tx_df[tx_df['transfer_type'] == 0]
            outflow_df = tx_df[tx_df['transfer_type'] == 1]

            total_inflow = inflow_df['amount_usd'].sum() if not inflow_df.empty else 0
            total_outflow = outflow_df['amount_usd'].sum() if not outflow_df.empty else 0
            net_flow = total_inflow - total_outflow
        else:
            # Fallback if columns don't match expected structure
            total_inflow = 0
            total_outflow = 0
            net_flow = 0
        
        # Display metrics
        metrics = {
            "Total Inflow": {
                "value": total_inflow,
                "delta": None
            },
            "Total Outflow": {
                "value": total_outflow,
                "delta": None
            },
            "Net Flow": {
                "value": net_flow,
                "delta": None
            }
        }
        
        formatters = {
            "Total Inflow": lambda x: format_currency(x, abbreviate=True),
            "Total Outflow": lambda x: format_currency(x, abbreviate=True),
            "Net Flow": lambda x: format_currency(x, abbreviate=True)
        }
        
        display_metrics_row(metrics, formatters)
        
        # Exchange transactions table
        st.subheader("Exchange Chain Transactions by Exchange")
        
        # Sort by amount if available
        if 'amount_usd' in tx_df.columns:
            tx_df = tx_df.sort_values(by='amount_usd')
        
        # Create table
        # Determine which columns to format based on available data
        format_dict = {}
        if 'amount_usd' in tx_df.columns:
            format_dict['amount_usd'] = lambda x: format_currency(x, abbreviate=True)

        # Add any other columns that should be formatted
        if 'asset_quantity' in tx_df.columns:
            format_dict['asset_quantity'] = lambda x: format_volume(x, precision=6)

        create_formatted_table(tx_df, format_dict)
        
        # Create bar chart for amount by exchange if available
        if 'exchange_name' in tx_df.columns and 'amount_usd' in tx_df.columns:
            # Group by exchange
            exchange_amounts = tx_df.groupby('exchange_name')['amount_usd'].sum().reset_index()

            fig = px.bar(
                exchange_amounts,
                x='exchange_name',
                y='amount_usd',
                title="Transfer Amount by Exchange",
                color='amount_usd',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
        
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Net Flow"
        )
        
        display_chart(apply_chart_theme(fig))
        
        # Create grouped bar chart for inflow/outflow by exchange
        st.subheader("Transactions by Exchange")

        # Check if we have the necessary columns
        if 'exchange_name' in tx_df.columns and 'transfer_type' in tx_df.columns and 'amount_usd' in tx_df.columns:
            # Create a new dataframe with inflow/outflow by exchange
            inflow_by_exchange = tx_df[tx_df['transfer_type'] == 0].groupby('exchange_name')['amount_usd'].sum().reset_index()
            inflow_by_exchange['flow_type'] = 'Inflow'
            inflow_by_exchange.rename(columns={'amount_usd': 'amount'}, inplace=True)

            outflow_by_exchange = tx_df[tx_df['transfer_type'] == 1].groupby('exchange_name')['amount_usd'].sum().reset_index()
            outflow_by_exchange['flow_type'] = 'Outflow'
            outflow_by_exchange.rename(columns={'amount_usd': 'amount'}, inplace=True)

            # Combine the dataframes
            melted_df = pd.concat([inflow_by_exchange, outflow_by_exchange])

            # Sort and keep top exchanges
            top_exchanges = melted_df.groupby('exchange_name')['amount'].sum().nlargest(10).index.tolist()
            melted_df = melted_df[melted_df['exchange_name'].isin(top_exchanges)]
        
        # Create grouped bar chart
        fig = px.bar(
            melted_df,
            x='exchange_name',
            y='amount',
            color='flow_type',
            title="Inflow vs. Outflow by Exchange (Top 10 by Volume)",
            barmode='group',
            color_discrete_map={
                'inflow_amount': 'green',
                'outflow_amount': 'red'
            }
        )
        
        # Rename legend items
        fig.for_each_trace(lambda t: t.update(
            name=t.name.replace('inflow_amount', 'Inflow').replace('outflow_amount', 'Outflow')
        ))
        
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Amount"
        )
        
        display_chart(apply_chart_theme(fig))
        
        # Explanation
        st.markdown("""
        ### Understanding Exchange Chain Transactions
        
        Exchange chain transactions show the flow of cryptocurrency to and from exchanges:
        
        - **Inflows**: Coins moving from private wallets to exchanges, potentially for selling
        - **Outflows**: Coins moving from exchanges to private wallets, potentially for long-term holding
        - **Net Flow**: The difference between inflows and outflows, with negative values indicating more coins leaving exchanges
        
        These metrics are important for understanding market sentiment and potential price movements:
        
        - **High outflows (negative net flow)**: Often considered bullish as coins move to storage
        - **High inflows (positive net flow)**: Could indicate increased selling pressure
        """)
    else:
        st.info("Exchange chain transaction data not available.")

def render_exchange_assets(data):
    """Render exchange assets visualizations."""
    st.header("Exchange Assets")
    
    if 'api_exchange_assets' in data and not data['api_exchange_assets'].empty:
        assets_df = data['api_exchange_assets']
        
        # Process the data based on structure
        # This would need to be adapted to the actual data format
        
        st.info("Exchange assets data available but requires custom implementation based on the specific data structure.")
    else:
        st.info("Exchange assets data not available.")

def render_exchange_balance_chart(data):
    """Render exchange balance chart visualizations."""
    st.header("Exchange Balance History")
    
    if 'api_exchange_balance_chart' in data and not data['api_exchange_balance_chart'].empty:
        balance_chart_df = data['api_exchange_balance_chart']
        
        # Process the data based on structure
        # This would need to be adapted to the actual data format
        
        st.info("Exchange balance chart data available but requires custom implementation based on the specific data structure.")
    else:
        st.info("Exchange balance history data not available.")

def main():
    """Main function to render the on-chain page."""
    
    # Render sidebar
    render_sidebar()
    
    # Page title and description
    st.title("On-Chain Metrics")
    st.write("Analysis of cryptocurrency on-chain data and exchange flows")
    
    # Display loading message
    with st.spinner("Loading on-chain data..."):
        data = load_onchain_data()
    
    # Get the last updated time
    last_updated = get_data_last_updated()
    last_updated_str = format_timestamp(last_updated) if last_updated else "Unknown"
    
    # Show data date info
    st.caption(f"Data as of: {last_updated_str}")
    
    # Check if data is available
    if not data:
        st.error("No on-chain data available.")
        return
    
    # Create tabs for different on-chain metrics
    tab1, tab2 = st.tabs([
        "Exchange Balance", 
        "Chain Transactions"
    ])
    
    with tab1:
        render_exchange_balance(data)
    
    with tab2:
        render_exchange_chain_transactions(data)
    
    # Add footer
    st.markdown("---")
    st.caption("Izun Crypto Liquidity Report © 2025")
    st.caption("Data provided by CoinGlass API")

if __name__ == "__main__":
    main()