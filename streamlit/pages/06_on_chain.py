"""
On-Chain page for the Izun Crypto Liquidity Report.

This page displays data and visualizations related to exchange balances for multiple cryptocurrencies,
stablecoin yields, and DeFi pool yields.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
import pytz

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

# Import onchain yields functionality
from utils.onchain.api import get_stablecoin_metadata
from utils.onchain.config import (
    DEFAULT_MIN_TVL_USD, 
    MANUAL_STABLECOIN_DATA, 
    TARGET_YIELD_ASSET_SYMBOLS_LOWER
)
from utils.onchain.data_processing import (
    get_analytics_data,
    get_enhanced_analytics_data,
    get_stablecoin_yields_data,
    get_yield_data,
)
from utils.onchain.formatting import categorize_stablecoin_by_strategy, format_tvl
from utils.onchain.visualization import (
    create_apy_distribution_histogram,
    create_avg_yield_by_strategy_plot,
    create_strategy_distribution_pie,
    create_top_projects_by_tvl,
    create_top_yield_plot,
    create_tvl_vs_apy_scatter,
)

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
        
        # Calculate weighted averages for percentage changes
        weights = exchange_balance_df['total_balance']
        change_pct_1d = (exchange_balance_df['balance_change_percent_1d'] * weights).sum() / weights.sum() if weights.sum() > 0 else 0
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
            "24h Change %": lambda x: format_percentage(x / 100),
            "7d Change %": lambda x: format_percentage(x / 100),
            "30d Change %": lambda x: format_percentage(x / 100)
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
            'balance_change_percent_1d': lambda x: format_percentage(x / 100),
            'balance_change_7d': lambda x: format_currency(x, abbreviate=True) if 'balance_change_7d' in exchange_balance_df.columns else None,
            'balance_change_percent_7d': lambda x: format_percentage(x / 100) if 'balance_change_percent_7d' in exchange_balance_df.columns else None,
            'balance_change_30d': lambda x: format_currency(x, abbreviate=True) if 'balance_change_30d' in exchange_balance_df.columns else None,
            'balance_change_percent_30d': lambda x: format_percentage(x / 100) if 'balance_change_percent_30d' in exchange_balance_df.columns else None
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

def show_stablecoin_section():
    """Display stablecoin yields table and analytics in a single view."""
    # === STABLECOIN YIELDS TABLE SECTION ===
    st.header("Stablecoin Yields")
    st.caption("Data manually updated on a weekly basis")

    # Get the stablecoin yield data
    stablecoin_yield_data = get_stablecoin_yields_data(MANUAL_STABLECOIN_DATA)

    # Check if data is valid
    data_valid = not (
        isinstance(stablecoin_yield_data, pd.DataFrame)
        and ("error" in stablecoin_yield_data.columns or "warning" in stablecoin_yield_data.columns)
    )

    filtered_data = pd.DataFrame()
    options_projects = ["All"]
    options_strategy = ["All"]

    if data_valid and not stablecoin_yield_data.empty:
        filtered_data = stablecoin_yield_data.copy()

        # Add strategy type column for filtering
        filtered_data["Strategy Type"] = filtered_data["Description"].apply(
            categorize_stablecoin_by_strategy
        )

        # Get list of projects and strategies for filters
        if "Project" in filtered_data.columns:
            options_projects.extend(sorted(filtered_data["Project"].dropna().unique()))
        if "Strategy Type" in filtered_data.columns:
            options_strategy.extend(sorted(filtered_data["Strategy Type"].dropna().unique()))

        # Create filters UI
        st.subheader("Filters")
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_ticker = st.text_input(
                "Filter by Ticker Contains",
                placeholder="e.g. susde",
                key="stable_yields_ticker_filter_app",
            )
        with col2:
            filter_project = st.selectbox(
                "Filter by Project",
                options=options_projects,
                index=0,
                key="stable_yields_project_filter_app",
            )
        with col3:
            filter_strategy = st.selectbox(
                "Filter by Strategy Type",
                options=options_strategy,
                index=0,
                key="stable_yields_strategy_filter_app",
            )

        # Apply filters
        if filter_ticker:
            if "Ticker" in filtered_data.columns:
                filtered_data = filtered_data[
                    filtered_data["Ticker"].str.lower().str.contains(filter_ticker.lower())
                ]
            else:
                st.error("Filtering failed: 'Ticker' column not found.")

        if filter_project != "All":
            if "Project" in filtered_data.columns:
                filtered_data = filtered_data[filtered_data["Project"] == filter_project]
            else:
                st.warning("Cannot filter by project: 'Project' column missing.")

        if filter_strategy != "All":
            if "Strategy Type" in filtered_data.columns:
                filtered_data = filtered_data[filtered_data["Strategy Type"] == filter_strategy]
            else:
                st.warning("Cannot filter by strategy: 'Strategy Type' column missing.")

    # Handle error states
    elif (
        isinstance(stablecoin_yield_data, pd.DataFrame) and "error" in stablecoin_yield_data.columns
    ):
        st.error(f"Failed to load stablecoin yield data: {stablecoin_yield_data['error'].iloc[0]}")
    elif (
        isinstance(stablecoin_yield_data, pd.DataFrame)
        and "warning" in stablecoin_yield_data.columns
    ):
        st.warning(
            f"Could not load all stablecoin yield data: {stablecoin_yield_data['warning'].iloc[0]}"
        )
    elif stablecoin_yield_data.empty:
        st.warning("No data loaded for stablecoin yields (Manual data might be empty).")

    # Display divider before data table
    st.divider()

    # Display the filtered data
    if not filtered_data.empty and data_valid:
        st.info(f"Displaying {len(filtered_data)} target stablecoins.")

        display_cols = ["Project", "Ticker", "Yield", "TVL", "Description"]
        cols_to_display = [col for col in display_cols if col in filtered_data.columns]

        # Configure column displays
        column_config = {
            "Project": st.column_config.TextColumn(
                "Project", help="Associated project/protocol (manual)"
            ),
            "Ticker": st.column_config.TextColumn(
                "Ticker", help="Yield-bearing asset symbol (manual)"
            ),
            "Yield": st.column_config.TextColumn("Yield", help="Manually entered APY (%)"),
            "TVL": st.column_config.TextColumn("TVL", help="Manually entered TVL (USD)"),
            "Description": st.column_config.TextColumn(
                "Description", help="Yield source description (manual)"
            ),
            "Staked Proportion": st.column_config.TextColumn(
                "Staked %", help="Staked proportion (manual)"
            ),
        }

        column_config_filtered = {k: v for k, v in column_config.items() if k in cols_to_display}

        st.dataframe(
            filtered_data[cols_to_display],
            column_config=column_config_filtered,
            hide_index=True,
            use_container_width=True,
        )

    # Handle empty results after filtering
    elif (
        filtered_data.empty
        and (filter_ticker or filter_project != "All" or filter_strategy != "All")
        and data_valid
    ):
        st.warning("No stablecoin yields match the selected filter criteria.")
    elif filtered_data.empty and not data_valid:
        pass  # Error already displayed above
    elif filtered_data.empty:
        st.warning("No data available for stablecoin yields overview.")

    # === STABLECOIN ANALYTICS SECTION ===
    st.divider()
    st.header("Stablecoin Analytics")
    st.caption("Analytics derived from data in 'Stablecoin Yields' section above.")

    # Get the enhanced analytics data
    enhanced_analytics_data = get_enhanced_analytics_data(MANUAL_STABLECOIN_DATA)

    # Check if data is valid
    analytics_data_valid = not (
        isinstance(enhanced_analytics_data, pd.DataFrame)
        and any(col in enhanced_analytics_data.columns for col in ["error", "warning"])
    )

    if analytics_data_valid and not enhanced_analytics_data.empty:
        # Top Stablecoins by Yield
        st.subheader("Top Stablecoins by Yield")
        try:
            if (
                "APY" in enhanced_analytics_data.columns
                and "Asset Symbol" in enhanced_analytics_data.columns
            ):
                hover_cols = ["Project", "TVL_USD", "Staked Proportion", "Description"]
                fig_top_yield = create_top_yield_plot(
                    enhanced_analytics_data,
                    top_n=20,
                    color_column="Strategy Type",
                    hover_data_columns=hover_cols,
                )
                st.plotly_chart(fig_top_yield, use_container_width=True)
            else:
                st.warning("Required columns (APY, Asset Symbol) not found for Top Yield plot.")
        except Exception as e:
            st.warning(f"Could not generate Top Yield plot: {e}")
            logging.error(f"Error in Top Yield plot: {e}")

        # Distribution by Strategy Type
        st.divider()
        st.subheader("Distribution by Strategy Type")
        try:
            if "Strategy Type" in enhanced_analytics_data.columns:
                fig_strategy_pie = create_strategy_distribution_pie(enhanced_analytics_data)
                st.plotly_chart(fig_strategy_pie, use_container_width=True)
            else:
                st.warning("Strategy Type information not available for distribution plot.")
        except Exception as e:
            st.warning(f"Could not generate Strategy Distribution plot: {e}")
            logging.error(f"Error in Strategy Distribution plot: {e}")

        # Average Yield by Strategy Type
        st.divider()
        st.subheader("Average Yield by Strategy Type")
        try:
            if (
                "Strategy Type" in enhanced_analytics_data.columns
                and "APY" in enhanced_analytics_data.columns
            ):
                fig_strategy_yield = create_avg_yield_by_strategy_plot(enhanced_analytics_data)
                st.plotly_chart(fig_strategy_yield, use_container_width=True)
            else:
                st.warning("Strategy Type or APY information not available for average yield plot.")
        except Exception as e:
            st.warning(f"Could not generate Average Yield plot: {e}")
            logging.error(f"Error in Average Yield plot: {e}")

        # TVL vs. APY
        st.divider()
        st.subheader("TVL vs. APY")
        try:
            if (
                "TVL_USD" in enhanced_analytics_data.columns
                and "APY" in enhanced_analytics_data.columns
            ):
                hover_cols = ["Strategy Type", "Description", "Staked Proportion"]
                fig_scatter_tvl_apy = create_tvl_vs_apy_scatter(
                    enhanced_analytics_data,
                    color_column="Strategy Type",
                    hover_name="Asset Symbol",
                    hover_data_columns=hover_cols,
                )
                st.plotly_chart(fig_scatter_tvl_apy, use_container_width=True)
            else:
                st.warning(
                    "Required columns (TVL_USD, APY, Project, Asset Symbol) not found for scatter plot."
                )
        except Exception as e:
            st.warning(f"Could not generate TVL vs APY plot: {e}")
            logging.error(f"Error in Stablecoin Analytics TVL vs APY plot: {e}")

    # Handle error states
    elif (
        isinstance(enhanced_analytics_data, pd.DataFrame)
        and "error" in enhanced_analytics_data.columns
    ):
        st.error(
            f"Stablecoin Analytics: Failed to process data - {enhanced_analytics_data['error'].iloc[0]}"
        )
    elif (
        isinstance(enhanced_analytics_data, pd.DataFrame)
        and "warning" in enhanced_analytics_data.columns
    ):
        st.warning(f"Stablecoin Analytics: {enhanced_analytics_data['warning'].iloc[0]}")
        st.warning("Analytics may be incomplete.")
    else:
        st.warning(
            "No data available to generate stablecoin analytics plots (Data might be empty or filtered out)."
        )


def show_pool_section():
    """Display pool yields table and analytics in a single view."""
    # === POOL YIELDS TABLE SECTION ===
    st.header("Pool Yields")

    # Get the current time in London timezone for refresh timestamp
    try:
        london_time = datetime.now(pytz.timezone("Europe/London"))
        refresh_time_str = london_time.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        refresh_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    st.caption(f"Data sourced from DeFiLlama | Last Refreshed: {refresh_time_str}")

    # Get stablecoin metadata for joining
    stablecoin_metadata = get_stablecoin_metadata()

    # Create filter UI
    st.subheader("Filters")
    col1, col2, col3, col4 = st.columns([1, 1.5, 1.5, 1.5])

    with col1:
        min_tvl_input = st.number_input(
            label="Minimum Pool TVL",
            min_value=0,
            max_value=10_000_000_000,
            value=DEFAULT_MIN_TVL_USD,
            step=1_000_000,
            help="Filter pools by minimum Total Value Locked in USD.",
            format="%d",
            key="pool_tvl_filter_app",
        )

    # Get initial data with minimum TVL filter
    initial_yield_data = get_yield_data(
        min_tvl_input, TARGET_YIELD_ASSET_SYMBOLS_LOWER, stablecoin_metadata
    )

    # Initialize filter variables
    selected_projects = []
    selected_types = []
    options_projects = ["All"]
    options_types = ["All"]
    filter_symbol_search = ""

    # Check if data is valid
    data_valid = not (
        isinstance(initial_yield_data, pd.DataFrame)
        and any(col in initial_yield_data.columns for col in ["error", "warning", "warning_tvl"])
    )

    # Prepare filter options if data is valid
    if data_valid and not initial_yield_data.empty:
        try:
            if "Project" in initial_yield_data.columns:
                options_projects.extend(sorted(initial_yield_data["Project"].dropna().unique()))
            if "Type (Peg)" in initial_yield_data.columns:
                options_types.extend(sorted(initial_yield_data["Type (Peg)"].dropna().unique()))
        except KeyError as e:
            st.error(f"Pool Yields: Expected column missing for filter options: {e}")
            data_valid = False
        except Exception as e:
            st.error(f"Pool Yields: Error preparing filter options: {e}")
            data_valid = False

    # Create remaining filter inputs
    with col2:
        filter_symbol_search = st.text_input(
            "Filter Symbol Contains",
            placeholder="e.g. usdc, usde",
            disabled=not data_valid,
            key="pool_symbol_filter_app",
        )

    with col3:
        selected_project_filter = st.selectbox(
            "Filter by Project",
            options=options_projects,
            index=0,
            disabled=not data_valid,
            key="pool_project_filter_app",
        )
        selected_projects = [] if selected_project_filter == "All" else [selected_project_filter]

    with col4:
        selected_type_filter = st.selectbox(
            "Filter by Type (Peg)",
            options=options_types,
            index=0,
            disabled=not data_valid,
            key="pool_type_filter_app",
        )
        selected_types = [] if selected_type_filter == "All" else [selected_type_filter]

    # Filter the data based on user selections
    filtered_yield_data = pd.DataFrame()
    if data_valid and not initial_yield_data.empty:
        filtered_yield_data = initial_yield_data.copy()
        try:
            if filter_symbol_search:
                if "Asset Symbol" in filtered_yield_data.columns:
                    filtered_yield_data = filtered_yield_data[
                        filtered_yield_data["Asset Symbol"]
                        .str.lower()
                        .str.contains(filter_symbol_search.lower())
                    ]
                else:
                    st.warning(
                        "Pool Yields: Cannot filter by symbol: 'Asset Symbol' column missing."
                    )

            if selected_projects:
                if "Project" in filtered_yield_data.columns:
                    filtered_yield_data = filtered_yield_data[
                        filtered_yield_data["Project"].isin(selected_projects)
                    ]
                else:
                    st.warning("Pool Yields: Cannot filter by project: 'Project' column missing.")

            if selected_types:
                if "Type (Peg)" in filtered_yield_data.columns:
                    filtered_yield_data = filtered_yield_data[
                        filtered_yield_data["Type (Peg)"].isin(selected_types)
                    ]
                else:
                    st.warning("Pool Yields: Cannot filter by type: 'Type (Peg)' column missing.")

        except KeyError as e:
            st.error(f"Pool Yields: Error applying filters: Column '{e}' not found.")
            filtered_yield_data = pd.DataFrame()
        except Exception as e:
            st.error(f"Pool Yields: Unexpected error applying filters: {e}")
            filtered_yield_data = pd.DataFrame()

    # Display divider before data table
    st.divider()

    # Handle error states
    if isinstance(initial_yield_data, pd.DataFrame) and "error" in initial_yield_data.columns:
        st.error(
            f"Pool Yields Data fetching failed: {initial_yield_data['error'].iloc[0]}. Please try again later."
        )
    elif isinstance(initial_yield_data, pd.DataFrame) and "warning" in initial_yield_data.columns:
        st.warning(
            f"Pool Yields Initial data load issue: {initial_yield_data['warning'].iloc[0]}. No pools matched the base criteria from API."
        )
    elif (
        isinstance(initial_yield_data, pd.DataFrame) and "warning_tvl" in initial_yield_data.columns
    ):
        st.warning(
            f"Pool Yields Initial data load issue: {initial_yield_data['warning_tvl'].iloc[0]}. No pools found matching API criteria with TVL > {format_tvl(min_tvl_input)}."
        )
    elif not data_valid:
        st.error("Pool Yields: Could not load data or prepare filters correctly.")
    elif filtered_yield_data.empty and (
        filter_symbol_search or selected_projects or selected_types
    ):
        st.warning("Pool Yields: No pools match the selected filter criteria.")
    elif filtered_yield_data.empty:
        st.warning("Pool Yields: No pool data available after initial filtering.")
    else:
        # Display the filtered data
        st.info(f"Displaying {len(filtered_yield_data)} pools from DefiLlama.")

        display_columns = [
            "Chain",
            "Project",
            "Asset Symbol",
            "APY (%)",
            "TVL (USD)",
            "Issuer/Name",
            "Type (Peg)",
        ]
        yield_data_display = filtered_yield_data[
            [col for col in display_columns if col in filtered_yield_data.columns]
        ]

        column_config = {
            "TVL (USD)": st.column_config.TextColumn(
                "TVL (USD)", help="Total Value Locked (Formatted)"
            ),
            "APY (%)": st.column_config.TextColumn(
                "APY (%)", help="Annual Percentage Yield (Formatted)"
            ),
            "Issuer/Name": st.column_config.TextColumn(
                "Issuer/Name", help="Issuing project or name (from metadata)"
            ),
            "Type (Peg)": st.column_config.TextColumn(
                "Type (Peg)", help="Peg type (from metadata)"
            ),
        }

        st.dataframe(
            yield_data_display,
            column_config=column_config,
            hide_index=True,
            use_container_width=True,
        )

    # === POOL YIELD ANALYTICS SECTION ===
    st.divider()
    st.header("Pool Yield Analytics")
    st.write(
        "Visualizations based on the broader stablecoin and yield-asset pool data from API (Default TVL > $10M)."
    )

    # Get stablecoin metadata for joining
    stablecoin_metadata_api = get_stablecoin_metadata()

    # Use default TVL filter for analytics
    analytics_min_tvl_api = DEFAULT_MIN_TVL_USD
    st.info(f"Analytics based on API pools with TVL > {format_tvl(analytics_min_tvl_api)}")

    # Get analytics data
    analytics_data_api = get_analytics_data(
        analytics_min_tvl_api, TARGET_YIELD_ASSET_SYMBOLS_LOWER, stablecoin_metadata_api
    )

    # Check if data is valid
    data_valid_analytics_api = not (
        isinstance(analytics_data_api, pd.DataFrame)
        and any(col in analytics_data_api.columns for col in ["error", "warning", "warning_tvl"])
    )

    if data_valid_analytics_api and not analytics_data_api.empty:
        # APY Distribution
        st.subheader("APY Distribution")
        try:
            if "APY" in analytics_data_api.columns:
                fig_hist = create_apy_distribution_histogram(analytics_data_api)
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.warning("Pool Analytics: APY column not found in API data.")
        except Exception as e:
            st.warning(f"Pool Analytics: Could not generate APY Distribution plot: {e}")

        # TVL vs. APY
        st.divider()
        st.subheader("TVL vs. APY")
        try:
            if "TVL_USD" in analytics_data_api.columns and "APY" in analytics_data_api.columns:
                hover_cols = [
                    "Chain",
                    "Project",
                    "Issuer/Name",
                    "Type (Peg)",
                    "Type (Peg Mechanism)",
                ]
                fig_scatter = create_tvl_vs_apy_scatter(
                    analytics_data_api,
                    color_column="Project",
                    hover_name="Asset Symbol",
                    hover_data_columns=hover_cols,
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("Pool Analytics: Required columns missing for TVL vs APY plot.")
        except Exception as e:
            st.warning(f"Pool Analytics: Could not generate TVL vs APY plot: {e}")

        # Top Pools by APY
        st.divider()
        st.subheader("Top Pools by APY")
        try:
            if "APY" in analytics_data_api.columns and "Asset Symbol" in analytics_data_api.columns:
                hover_cols = ["Project", "Chain", "TVL_USD", "Type (Peg)", "Issuer/Name"]
                fig_bar_apy = create_top_yield_plot(
                    analytics_data_api, top_n=20, color_column=None, hover_data_columns=hover_cols
                )
                st.plotly_chart(fig_bar_apy, use_container_width=True)
            else:
                st.warning("Pool Analytics: Required columns missing for Top APY plot.")
        except Exception as e:
            st.warning(f"Pool Analytics: Could not generate Top Pools by APY plot: {e}")

        # Top 10 Projects by TVL
        st.divider()
        st.subheader("Top 10 Projects by TVL")
        try:
            if "Project" in analytics_data_api.columns and "TVL_USD" in analytics_data_api.columns:
                fig_top_tvl = create_top_projects_by_tvl(analytics_data_api, top_n=10)
                st.plotly_chart(fig_top_tvl, use_container_width=True)
            else:
                st.warning(
                    "Pool Analytics: Cannot generate Top 10 Projects plot: Missing columns (API Data)."
                )
        except Exception as e:
            st.warning(f"Pool Analytics: Could not generate Top Projects by TVL plot: {e}")

    # Handle error states
    elif isinstance(analytics_data_api, pd.DataFrame) and "error" in analytics_data_api.columns:
        st.error(
            f"Pool Analytics: Failed to fetch or process API data - {analytics_data_api['error'].iloc[0]}"
        )
    elif isinstance(analytics_data_api, pd.DataFrame) and (
        "warning" in analytics_data_api.columns or "warning_tvl" in analytics_data_api.columns
    ):
        if "warning" in analytics_data_api.columns:
            st.warning(
                f"Pool Analytics: {analytics_data_api['warning'].iloc[0]}. No API pools matched base criteria."
            )
        elif "warning_tvl" in analytics_data_api.columns:
            st.warning(
                f"Pool Analytics: {analytics_data_api['warning_tvl'].iloc[0]}. No API pools matched TVL criteria."
            )
        st.warning("Pool Analytics (API) may be incomplete due to data limitations.")
    else:
        st.warning("No API data available to generate pool yield analytics plots.")



def main():
    """Main function to render the enhanced on-chain page."""
    
    # Render sidebar
    render_sidebar()
    
    # Page title
    st.title("On-Chain")
    
    # Create enhanced tabs including new yield sections
    tabs = st.tabs(["BTC", "ETH", "Stablecoin Yields", "Pool Yields"])
    
    # Existing BTC tab
    with tabs[0]:
        st.header("BTC Exchange Balances")
        with st.spinner("Loading BTC exchange balance data..."):
            exchange_balance_df = load_exchange_balance_data("BTC")
        display_exchange_balance_analysis(exchange_balance_df, "BTC")
    
    # Existing ETH tab  
    with tabs[1]:
        st.header("ETH Exchange Balances")
        with st.spinner("Loading ETH exchange balance data..."):
            exchange_balance_df = load_exchange_balance_data("ETH")
        display_exchange_balance_analysis(exchange_balance_df, "ETH")
    
    # New Stablecoin Yields tab
    with tabs[2]:
        show_stablecoin_section()
    
    # New Pool Yields tab
    with tabs[3]:
        show_pool_section()
    

if __name__ == "__main__":
    main()