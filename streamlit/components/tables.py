"""
Table display components for the Izun Crypto Liquidity Report application.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.formatters import format_currency, format_percentage, format_timestamp

def create_data_table(df, key=None, selection_mode=None, height=400, pagination=True, fit_columns=True):
    """
    Create an interactive data table.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    key : str, optional
        Unique key for the table component
    selection_mode : str, optional
        Selection mode ('single' or 'multiple')
    height : int
        Table height in pixels
    pagination : bool
        Whether to enable pagination
    fit_columns : bool
        Whether to fit columns to their content
        
    Returns:
    --------
    streamlit.dataframe
        The displayed dataframe with data
    """
    if df.empty:
        st.info("No data available for the table.")
        return None
    
    # Create streamlit dataframe with enhanced features
    return st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=None,  # Can be configured later for specific columns
        height=height
    )

def create_formatted_table(df, format_dict=None, hide_index=True, use_container_width=True, height=None):
    """
    Create a formatted table with custom formatting for columns.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    format_dict : dict, optional
        Dictionary mapping column names to formatting functions
    hide_index : bool
        Whether to hide the index column
    use_container_width : bool
        Whether to use the full container width
    height : int, optional
        Table height in pixels

    Returns:
    --------
    streamlit.dataframe
        The displayed dataframe with formatted data
    """
    if df.empty:
        st.info("No data available for the table.")
        return None

    # Make a copy of the dataframe to avoid modifying the original
    formatted_df = df.copy()

    # Default format dictionary if none provided
    if format_dict is None:
        format_dict = {}

    # Auto-detect numeric columns that might be currency and add default formatting
    for col in formatted_df.columns:
        # Skip columns that already have formatting specified
        if col in format_dict:
            continue

        # Check if column name suggests it's currency
        is_currency_col = any(term in col.lower() for term in
                              ['price', 'volume', 'amount', 'usd', 'value', 'cap', 'aum', 'asset'])

        # If it's a numeric column that might be currency, add default formatting
        if is_currency_col and pd.api.types.is_numeric_dtype(formatted_df[col]):
            format_dict[col] = lambda x: format_currency(x, show_decimals=False)

    # Apply formatting if provided
    for col, fmt_function in format_dict.items():
        if col in formatted_df.columns and fmt_function is not None:
            # Only apply formatting to non-null values
            formatted_df[col] = formatted_df[col].apply(
                lambda x: fmt_function(x) if pd.notnull(x) else 'N/A'
            )

    # Create streamlit dataframe
    return st.dataframe(
        formatted_df,
        use_container_width=use_container_width,
        hide_index=hide_index,
        height=height
    )

def create_crypto_table(df, asset_col=None, price_col=None, change_col=None, volume_col=None, market_cap_col=None):
    """
    Create a table specifically for cryptocurrency data with special formatting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    asset_col : str, optional
        Column containing asset names/symbols
    price_col : str, optional
        Column containing price data
    change_col : str, optional
        Column containing price change data
    volume_col : str, optional
        Column containing volume data
    market_cap_col : str, optional
        Column containing market cap data
        
    Returns:
    --------
    streamlit.dataframe
        The displayed dataframe with formatted data
    """
    if df.empty:
        st.info("No cryptocurrency data available.")
        return None
    
    # Make a copy of the dataframe to avoid modifying the original
    crypto_df = df.copy()
    
    # Create column configuration
    column_config = {}
    
    # Configure asset column if provided
    if asset_col and asset_col in crypto_df.columns:
        column_config[asset_col] = st.column_config.TextColumn(
            "Asset",
            help="Cryptocurrency asset",
            width="medium"
        )
    
    # Configure price column if provided
    if price_col and price_col in crypto_df.columns:
        column_config[price_col] = st.column_config.NumberColumn(
            "Price",
            help="Current price in USD",
            format="$%.2f"
        )
    
    # Configure change column if provided
    if change_col and change_col in crypto_df.columns:
        # Convert to numeric if not already
        crypto_df[change_col] = pd.to_numeric(crypto_df[change_col], errors='coerce')
        
        column_config[change_col] = st.column_config.NumberColumn(
            "Change (24h)",
            help="24-hour price change percentage",
            format="%.2f%%"
        )
    
    # Configure volume column if provided
    if volume_col and volume_col in crypto_df.columns:
        column_config[volume_col] = st.column_config.NumberColumn(
            "Volume (24h)",
            help="24-hour trading volume in USD",
            format="$%.2f"
        )
    
    # Configure market cap column if provided
    if market_cap_col and market_cap_col in crypto_df.columns:
        column_config[market_cap_col] = st.column_config.NumberColumn(
            "Market Cap",
            help="Market capitalization in USD",
            format="$%.2f"
        )
    
    # Create streamlit dataframe with column configuration
    return st.dataframe(
        crypto_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )

def create_etf_table(df):
    """
    Create a table specifically for ETF data with special formatting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ETF data
        
    Returns:
    --------
    streamlit.dataframe
        The displayed dataframe with formatted ETF data
    """
    if df.empty:
        st.info("No ETF data available.")
        return None
    
    # Make a copy of the dataframe to avoid modifying the original
    etf_df = df.copy()
    
    # Define columns to include (if available in the data)
    columns_of_interest = [
        'ticker', 'fund_name', 'region', 'fund_type',
        'price_usd', 'price_change_percent', 'aum_usd', 
        'management_fee_percent', 'premium_discount_percent'
    ]
    
    # Filter columns to only include those that exist in the dataframe
    available_columns = [col for col in columns_of_interest if col in etf_df.columns]
    
    # Create a subset with only the available columns
    etf_subset = etf_df[available_columns].copy()
    
    # Create column configuration
    column_config = {
        'ticker': st.column_config.TextColumn(
            "Ticker",
            help="ETF ticker symbol",
            width="small"
        ),
        'fund_name': st.column_config.TextColumn(
            "Fund Name",
            help="Full ETF fund name",
            width="large"
        ),
        'region': st.column_config.TextColumn(
            "Region",
            help="Geographic region",
            width="small"
        ),
        'fund_type': st.column_config.TextColumn(
            "Type",
            help="Fund type (Spot or Futures)",
            width="small"
        )
    }
    
    # Configure numeric columns if they exist
    if 'price_usd' in available_columns:
        column_config['price_usd'] = st.column_config.NumberColumn(
            "Price (USD)",
            help="Current price in USD",
            format="$%.2f"
        )
    
    if 'price_change_percent' in available_columns:
        column_config['price_change_percent'] = st.column_config.NumberColumn(
            "Change (%)",
            help="Price change percentage",
            format="%.2f%%"
        )
    
    if 'aum_usd' in available_columns:
        column_config['aum_usd'] = st.column_config.NumberColumn(
            "AUM (USD)",
            help="Assets Under Management in USD",
            format="$%.2f"
        )
    
    if 'management_fee_percent' in available_columns:
        column_config['management_fee_percent'] = st.column_config.NumberColumn(
            "Fee (%)",
            help="Management fee percentage",
            format="%.2f%%"
        )
    
    if 'premium_discount_percent' in available_columns:
        column_config['premium_discount_percent'] = st.column_config.NumberColumn(
            "Premium/Discount (%)",
            help="Premium or discount percentage to NAV",
            format="%.2f%%"
        )
    
    # Create streamlit dataframe with column configuration
    return st.dataframe(
        etf_subset,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )

def create_exchange_table(df, has_market_share=False):
    """
    Create a table specifically for exchange data with special formatting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing exchange data
    has_market_share : bool
        Whether the data includes market share information
        
    Returns:
    --------
    streamlit.dataframe
        The displayed dataframe with formatted exchange data
    """
    if df.empty:
        st.info("No exchange data available.")
        return None
    
    # Make a copy of the dataframe to avoid modifying the original
    exchange_df = df.copy()
    
    # Identify columns based on common patterns
    exchange_col = next((col for col in exchange_df.columns if 'exchange' in col.lower()), None)
    volume_col = next((col for col in exchange_df.columns if 'volume' in col.lower() or 'vol_' in col.lower()), None)
    market_share_col = next((col for col in exchange_df.columns if 'share' in col.lower() or 'market_share' in col.lower()), None)
    open_interest_col = next((col for col in exchange_df.columns if 'interest' in col.lower() or 'oi' in col.lower()), None)
    
    # Create column configuration
    column_config = {}
    
    if exchange_col:
        column_config[exchange_col] = st.column_config.TextColumn(
            "Exchange",
            help="Exchange name",
            width="medium"
        )
    
    if volume_col:
        column_config[volume_col] = st.column_config.NumberColumn(
            "Volume (USD)",
            help="Trading volume in USD",
            format="$%.2f"
        )
    
    if market_share_col and has_market_share:
        column_config[market_share_col] = st.column_config.ProgressColumn(
            "Market Share",
            help="Market share percentage",
            format="%.2f%%",
            min_value=0,
            max_value=100
        )
    elif market_share_col:
        column_config[market_share_col] = st.column_config.NumberColumn(
            "Market Share",
            help="Market share percentage",
            format="%.2f%%"
        )
    
    if open_interest_col:
        column_config[open_interest_col] = st.column_config.NumberColumn(
            "Open Interest (USD)",
            help="Open interest in USD",
            format="$%.2f"
        )
    
    # Create streamlit dataframe with column configuration
    return st.dataframe(
        exchange_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )

def create_summary_table(data_dict, metrics=None):
    """
    Create a summary table from a dictionary of metrics.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of data points
    metrics : list, optional
        List of metric names to include in the table
        
    Returns:
    --------
    streamlit.dataframe
        The displayed summary table
    """
    if not data_dict:
        st.info("No summary data available.")
        return None
    
    # Filter metrics if specified
    if metrics:
        filtered_dict = {k: v for k, v in data_dict.items() if k in metrics}
    else:
        filtered_dict = data_dict
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'Metric': filtered_dict.keys(),
        'Value': filtered_dict.values()
    })
    
    # Create streamlit dataframe
    return st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Metric': st.column_config.TextColumn(
                "Metric",
                help="Metric name",
                width="medium"
            ),
            'Value': st.column_config.TextColumn(
                "Value",
                help="Metric value",
                width="medium"
            )
        }
    )

def style_dataframe(df, style_dict):
    """
    Apply styling to a pandas DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to style
    style_dict : dict
        Dictionary mapping column names to styling functions
        
    Returns:
    --------
    pandas.io.formats.style.Styler
        Styled DataFrame
    """
    if df.empty:
        return df
    
    # Create a styler object
    styler = df.style
    
    # Apply styling for each column
    for col, style_func in style_dict.items():
        if col in df.columns:
            styler = style_func(styler, col)
    
    return styler

def background_gradient(styler, column, cmap='RdYlGn', vmin=None, vmax=None):
    """
    Apply a background gradient to a column.
    
    Parameters:
    -----------
    styler : pandas.io.formats.style.Styler
        Styler object
    column : str
        Column to apply gradient to
    cmap : str
        Colormap name
    vmin : float, optional
        Minimum value for scaling
    vmax : float, optional
        Maximum value for scaling
        
    Returns:
    --------
    pandas.io.formats.style.Styler
        Styled DataFrame
    """
    return styler.background_gradient(cmap=cmap, subset=[column], vmin=vmin, vmax=vmax)

def color_negative_red(styler, column):
    """
    Color negative values red and positive values green.
    
    Parameters:
    -----------
    styler : pandas.io.formats.style.Styler
        Styler object
    column : str
        Column to apply coloring to
        
    Returns:
    --------
    pandas.io.formats.style.Styler
        Styled DataFrame
    """
    return styler.applymap(
        lambda x: 'color: red' if isinstance(x, (int, float)) and x < 0 else 'color: green',
        subset=[column]
    )

def create_comparison_table(df1, df2, key_column, value_columns, title1="Before", title2="After"):
    """
    Create a comparison table between two DataFrames.
    
    Parameters:
    -----------
    df1 : pandas.DataFrame
        First DataFrame
    df2 : pandas.DataFrame
        Second DataFrame
    key_column : str
        Column to use as the key for comparison
    value_columns : list
        List of columns to compare
    title1 : str
        Title for the first DataFrame
    title2 : str
        Title for the second DataFrame
        
    Returns:
    --------
    streamlit.dataframe
        The displayed comparison table
    """
    if df1.empty or df2.empty:
        st.info("Not enough data for comparison.")
        return None
    
    # Check if key column exists in both DataFrames
    if key_column not in df1.columns or key_column not in df2.columns:
        st.error(f"Key column '{key_column}' not found in one or both DataFrames.")
        return None
    
    # Filter value columns to only include those that exist in both DataFrames
    common_value_columns = [col for col in value_columns if col in df1.columns and col in df2.columns]
    
    if not common_value_columns:
        st.error("No common value columns found for comparison.")
        return None
    
    # Create merged DataFrame for comparison
    comparison_df = pd.merge(
        df1[[key_column] + common_value_columns],
        df2[[key_column] + common_value_columns],
        on=key_column,
        suffixes=(f' ({title1})', f' ({title2})')
    )
    
    # Calculate differences
    for col in common_value_columns:
        col1 = f"{col} ({title1})"
        col2 = f"{col} ({title2})"
        comparison_df[f"{col} (Diff)"] = comparison_df[col2] - comparison_df[col1]
        comparison_df[f"{col} (% Change)"] = (comparison_df[f"{col} (Diff)"] / comparison_df[col1]) * 100
    
    # Create streamlit dataframe
    return st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )