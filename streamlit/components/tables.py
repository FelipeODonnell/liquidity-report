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
from utils.formatters import format_currency, format_percentage, format_timestamp, format_column_name

def create_data_table(df, key=None, selection_mode=None, height=400, pagination=True, fit_columns=True, format_column_names=True):
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
    format_column_names : bool
        Whether to format column names by replacing underscores with spaces and capitalizing words
        
    Returns:
    --------
    streamlit.dataframe
        The displayed dataframe with data
    """
    if df.empty:
        st.info("No data available for the table.")
        return None
    
    # Format column names if requested
    column_config = None
    if format_column_names:
        column_config = {}
        for col in df.columns:
            # Format column name
            formatted_name = format_column_name(str(col))
            
            # Detect if it's a numeric column
            col_data = df[col].dropna()
            is_numeric = len(col_data) > 0 and pd.api.types.is_numeric_dtype(col_data)
            
            if is_numeric:
                column_config[col] = st.column_config.NumberColumn(
                    formatted_name,
                    help=f"{formatted_name}"
                )
            else:
                column_config[col] = st.column_config.TextColumn(
                    formatted_name,
                    help=f"{formatted_name}"
                )
    
    # Create streamlit dataframe with enhanced features
    return st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
        height=height
    )

def create_formatted_table(df, format_dict=None, hide_index=True, use_container_width=True, height=None, 
                      max_width=None, alignment=None, column_config=None, emphasize_negatives=True, 
                      zebra_striping=False, compact_display=False, custom_css=None, format_column_names=True):
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
    max_width : dict, optional
        Dictionary mapping column names to maximum width values
    alignment : dict, optional
        Dictionary mapping column names to alignment ('left', 'center', 'right')
    column_config : dict, optional
        Streamlit column configuration
    emphasize_negatives : bool
        Whether to highlight negative values in red
    zebra_striping : bool
        Whether to add alternating row colors for better readability
    compact_display : bool
        Whether to use a more compact display format for tables
    custom_css : str, optional
        Custom CSS to apply to the table
    format_column_names : bool
        Whether to format column names by replacing underscores with spaces and capitalizing words

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
        
    # Default alignment dictionary
    if alignment is None:
        alignment = {}

    # Initialize column configuration if not provided
    if column_config is None:
        column_config = {}
        
    # Format column names if requested
    formatted_columns = {}
    if format_column_names:
        for col in formatted_df.columns:
            formatted_columns[col] = format_column_name(str(col))
        
    # Create currency terms list for better detection
    currency_terms = ['price', 'volume', 'amount', 'usd', 'value', 'cap', 'aum', 'asset', 'fund', 'flow', 
                      'liquidation', 'interest', 'depth', 'bid', 'ask', 'spread', 'cost', 'profit', 'revenue',
                      'balance', 'market', 'trade', 'size', 'notional', 'position']
                      
    # Create percentage terms list for better detection
    percentage_terms = ['percent', 'change', 'rate', 'ratio', 'growth', '%', 'pct', 'fee', 'premium', 'discount',
                        'yield', 'return', 'volatility', 'margin', 'allocation', 'weight', 'share', 'dominance',
                        'drawdown', 'gain', 'loss', 'performance', 'change_pct', 'diff_pct', 'delta_pct']

    # Auto-detect numeric columns that might be currency and add default formatting
    for col in formatted_df.columns:
        col_name = str(col).lower()  # Ensure we have a lowercase string for matching
        
        # Skip columns that already have formatting specified
        if col in format_dict:
            continue

        # Check if column name suggests it's currency by comparing with currency terms
        is_currency_col = any(term in col_name for term in currency_terms) or col_name.endswith('_usd')

        # Check if column name suggests it's a percentage by comparing with percentage terms
        is_percentage_col = any(term in col_name for term in percentage_terms) or col_name.endswith('_pct') or col_name.endswith('_percent')

        # Get sample of non-null values to better determine data type and magnitude
        non_null_values = formatted_df[col].dropna()
        is_numeric = len(non_null_values) > 0 and pd.api.types.is_numeric_dtype(non_null_values)
        
        if is_numeric:
            # Calculate stats for better formatting decisions
            max_abs_value = non_null_values.abs().max() if len(non_null_values) > 0 else 0
            has_decimal = any(x != int(x) for x in non_null_values if isinstance(x, (int, float)))
            all_small = all(abs(x) < 0.1 for x in non_null_values if isinstance(x, (int, float)))
            all_large = all(abs(x) >= 1000 for x in non_null_values if isinstance(x, (int, float)))
            
            # Apply automatic formatting based on column type, name, and data characteristics
            if is_currency_col:
                # For currency, adapt formatting based on magnitude
                if all_large:
                    # For large currency values, no decimals
                    format_dict[col] = lambda x: format_currency(x, show_decimals=False, compact=True, abbreviate=compact_display)
                    streamlit_format = "$%,.0f"
                elif all_small:
                    # For very small currency values, show more decimals
                    format_dict[col] = lambda x: format_currency(x, precision=4, show_decimals=True, strip_zeros=True)
                    streamlit_format = "$%,.4f"
                else:
                    # Default currency format
                    format_dict[col] = lambda x: format_currency(x, precision=2, show_decimals=has_decimal, compact=True)
                    streamlit_format = "$%,.2f" if has_decimal else "$%,.0f"
                
                # Set right alignment for currency columns
                if col not in alignment:
                    alignment[col] = 'right'
                    
                # Configure column for streamlit
                display_name = format_column_name(col) if format_column_names else col
                column_config[col] = st.column_config.NumberColumn(
                    display_name,
                    help=f"{col} in USD",
                    format=streamlit_format
                )
                    
            elif is_percentage_col:
                # For percentages, adapt formatting based on magnitude and characteristics
                if all_small:
                    # For very small percentages, show more decimals
                    format_dict[col] = lambda x: format_percentage(x, precision=4, strip_zeros=True, auto_precision=True)
                    streamlit_format = "%.4f%%"
                elif all_large:
                    # For large percentages, reduce decimals
                    format_dict[col] = lambda x: format_percentage(x, precision=1, strip_zeros=True, thousands_separator=True)
                    streamlit_format = "%.1f%%"
                else:
                    # Default percentage format with 2 decimal places
                    format_dict[col] = lambda x: format_percentage(x, precision=2, strip_zeros=True)
                    streamlit_format = "%.2f%%"
                
                # Set right alignment for percentage columns
                if col not in alignment:
                    alignment[col] = 'right'
                    
                # Configure column for streamlit
                display_name = format_column_name(col) if format_column_names else col
                column_config[col] = st.column_config.NumberColumn(
                    display_name,
                    help=f"{col} percentage",
                    format=streamlit_format
                )
            
            # For all other numeric columns, ensure they don't have excessive decimals
            elif is_numeric:
                # Adapt formatting based on magnitude and characteristics
                if max_abs_value >= 1000000:
                    # For very large numbers, use abbreviated format
                    format_dict[col] = lambda x: f"{x:,.1f}M" if abs(x) >= 1000000 else (f"{x:,.1f}K" if abs(x) >= 1000 else f"{x:,.1f}")
                elif max_abs_value >= 1000:
                    # For large numbers, no decimals
                    format_dict[col] = lambda x: f"{x:,.0f}" if isinstance(x, (int, float)) else x
                elif all_small:
                    # For very small numbers, show more decimals
                    format_dict[col] = lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x
                else:
                    # Default numeric format with sensible decimal handling
                    format_dict[col] = lambda x: f"{x:,.2f}".rstrip('0').rstrip('.') if isinstance(x, (int, float)) and x != int(x) else (f"{x:,.0f}" if isinstance(x, (int, float)) else x)
                
                # Configure display name for the column if not already in column_config
                if format_column_names and col not in column_config:
                    display_name = format_column_name(col)
                    column_config[col] = st.column_config.NumberColumn(
                        display_name,
                        format="%.2f" if has_decimal else "%.0f"
                    )

    # Format column names for non-numeric columns if requested
    if format_column_names:
        for col in formatted_df.columns:
            if col not in column_config:  # Skip columns that already have a configuration
                display_name = format_column_name(col)
                column_config[col] = st.column_config.Column(display_name)
    
    # Apply formatting if provided
    for col, fmt_function in format_dict.items():
        if col in formatted_df.columns and fmt_function is not None:
            # Create a closure to capture the current formatter
            def create_formatter(formatter):
                return lambda x: formatter(x) if pd.notnull(x) else 'N/A'
            
            # Apply formatting to non-null values using the closure to avoid issues with lambda functions in loops
            formatted_df[col] = formatted_df[col].apply(create_formatter(fmt_function))

    # Apply custom CSS if provided
    if custom_css:
        st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
    
    # Apply zebra striping if requested
    if zebra_striping:
        st.markdown("""
        <style>
        div[data-testid="stTable"] table tbody tr:nth-child(even) {background-color: #f2f2f2;}
        </style>
        """, unsafe_allow_html=True)
    
    # Apply compact display if requested
    if compact_display:
        st.markdown("""
        <style>
        div[data-testid="stTable"] table {font-size: 0.9rem; line-height: 1.2;}
        div[data-testid="stTable"] th {padding: 0.3rem; white-space: nowrap;}
        div[data-testid="stTable"] td {padding: 0.2rem 0.5rem;}
        </style>
        """, unsafe_allow_html=True)

    # If formatting column names is requested, create a simpler approach
    if format_column_names:
        # Create a new empty column config
        new_column_config = {}
        
        # Go through each column and create a fresh config with the formatted name
        for col in formatted_df.columns:
            # Format the column name
            formatted_name = format_column_name(col)
            
            # Determine column type based on data
            col_data = formatted_df[col].dropna()
            is_numeric = len(col_data) > 0 and pd.api.types.is_numeric_dtype(col_data)
            
            # Detect currency, percentage, or regular column based on name
            col_name = str(col).lower()
            
            # Currency column detection
            is_currency = any(term in col_name for term in currency_terms) or col_name.endswith('_usd')
            
            # Percentage column detection  
            is_percentage = any(term in col_name for term in percentage_terms) or col_name.endswith('_pct') or col_name.endswith('_percent')
            
            # Create appropriate column config based on type
            if is_numeric and is_currency:
                # Currency column
                new_column_config[col] = st.column_config.NumberColumn(
                    formatted_name,
                    help=f"{formatted_name}",
                    format="$%,.2f"
                )
            elif is_numeric and is_percentage:
                # Percentage column
                new_column_config[col] = st.column_config.NumberColumn(
                    formatted_name,
                    help=f"{formatted_name}",
                    format="%.2f%%"
                )
            elif is_numeric:
                # Regular numeric column
                new_column_config[col] = st.column_config.NumberColumn(
                    formatted_name,
                    help=f"{formatted_name}",
                    format="%,g"
                )
            else:
                # Text column
                new_column_config[col] = st.column_config.TextColumn(
                    formatted_name,
                    help=f"{formatted_name}"
                )
        
        # Use the new column config
        column_config = new_column_config
    
    # Create streamlit dataframe with enhanced configuration
    return st.dataframe(
        formatted_df,
        use_container_width=use_container_width,
        hide_index=hide_index,
        height=height,
        column_config=column_config
    )

def create_crypto_table(df, asset_col=None, price_col=None, change_col=None, volume_col=None, market_cap_col=None, format_column_names=True):
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
    format_column_names : bool
        Whether to format column names by removing underscores and capitalizing words
        
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
    
    # Convert change column to numeric if provided
    if change_col and change_col in crypto_df.columns:
        crypto_df[change_col] = pd.to_numeric(crypto_df[change_col], errors='coerce')
    
    # Use simplified approach to create column configuration
    column_config = {}
    
    # Process each column
    for col in crypto_df.columns:
        # Skip columns that don't exist in the dataframe
        if col not in crypto_df.columns:
            continue
        
        # Format column name
        formatted_name = format_column_name(col) if format_column_names else col
        
        # Set specific configurations for known columns
        if col == asset_col:
            column_config[col] = st.column_config.TextColumn(
                formatted_name,
                help="Cryptocurrency asset",
                width="medium"
            )
        elif col == price_col:
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="Current price in USD",
                format="$%.2f"
            )
        elif col == change_col:
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="24-hour price change percentage",
                format="%.2f%%"
            )
        elif col == volume_col:
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="24-hour trading volume in USD",
                format="$%.2f"
            )
        elif col == market_cap_col:
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="Market capitalization in USD",
                format="$%.2f"
            )
        else:
            # Auto-detect column type
            col_data = crypto_df[col].dropna()
            is_numeric = len(col_data) > 0 and pd.api.types.is_numeric_dtype(col_data)
            
            if is_numeric:
                # Detect special types of numeric columns
                col_name = str(col).lower()
                
                if 'price' in col_name or 'usd' in col_name or 'value' in col_name:
                    # Currency column
                    column_config[col] = st.column_config.NumberColumn(
                        formatted_name,
                        help=f"{formatted_name}",
                        format="$%.2f"
                    )
                elif 'percent' in col_name or 'change' in col_name or 'rate' in col_name or 'pct' in col_name:
                    # Percentage column
                    column_config[col] = st.column_config.NumberColumn(
                        formatted_name,
                        help=f"{formatted_name}",
                        format="%.2f%%"
                    )
                else:
                    # Regular numeric column
                    column_config[col] = st.column_config.NumberColumn(
                        formatted_name,
                        help=f"{formatted_name}"
                    )
            else:
                # Text column
                column_config[col] = st.column_config.TextColumn(
                    formatted_name,
                    help=f"{formatted_name}"
                )
    
    # Create streamlit dataframe with column configuration
    return st.dataframe(
        crypto_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )

def create_etf_table(df, format_column_names=True):
    """
    Create a table specifically for ETF data with special formatting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing ETF data
    format_column_names : bool
        Whether to format column names by removing underscores and capitalizing words
        
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
    
    # Create simplified column configuration
    column_config = {}
    
    # Process each column with simplified formatting
    for col in etf_subset.columns:
        # Format the column name
        formatted_name = format_column_name(col) if format_column_names else col
        
        # Determine column type and configuration
        if col == 'ticker':
            column_config[col] = st.column_config.TextColumn(
                formatted_name,
                help="ETF ticker symbol",
                width="small"
            )
        elif col == 'fund_name':
            column_config[col] = st.column_config.TextColumn(
                formatted_name,
                help="Full ETF fund name",
                width="large"
            )
        elif col == 'region':
            column_config[col] = st.column_config.TextColumn(
                formatted_name,
                help="Geographic region",
                width="small"
            )
        elif col == 'fund_type':
            column_config[col] = st.column_config.TextColumn(
                formatted_name,
                help="Fund type (Spot or Futures)",
                width="small"
            )
        elif col == 'price_usd':
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="Current price in USD",
                format="$%.2f"
            )
        elif col == 'price_change_percent':
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="Price change percentage",
                format="%.2f%%"
            )
        elif col == 'aum_usd':
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="Assets Under Management in USD",
                format="$%.2f"
            )
        elif col == 'management_fee_percent':
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="Management fee percentage",
                format="%.2f%%"
            )
        elif col == 'premium_discount_percent':
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="Premium or discount percentage to NAV",
                format="%.2f%%"
            )
        else:
            # Auto-detect column type
            col_data = etf_subset[col].dropna()
            is_numeric = len(col_data) > 0 and pd.api.types.is_numeric_dtype(col_data)
            
            if is_numeric:
                # Detect special types of numeric columns
                col_name = str(col).lower()
                
                if 'price' in col_name or 'usd' in col_name or 'value' in col_name:
                    # Currency column
                    column_config[col] = st.column_config.NumberColumn(
                        formatted_name,
                        help=f"{formatted_name}",
                        format="$%.2f"
                    )
                elif 'percent' in col_name or 'change' in col_name or 'rate' in col_name or 'pct' in col_name:
                    # Percentage column
                    column_config[col] = st.column_config.NumberColumn(
                        formatted_name,
                        help=f"{formatted_name}",
                        format="%.2f%%"
                    )
                else:
                    # Regular numeric column
                    column_config[col] = st.column_config.NumberColumn(
                        formatted_name,
                        help=f"{formatted_name}"
                    )
            else:
                # Text column
                column_config[col] = st.column_config.TextColumn(
                    formatted_name,
                    help=f"{formatted_name}"
                )
    
    # Create streamlit dataframe with column configuration
    return st.dataframe(
        etf_subset,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )

def create_exchange_table(df, has_market_share=False, format_column_names=True):
    """
    Create a table specifically for exchange data with special formatting.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing exchange data
    has_market_share : bool
        Whether the data includes market share information
    format_column_names : bool
        Whether to format column names by removing underscores and capitalizing words
        
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
    
    # Create simplified column configuration
    column_config = {}
    
    # Process each column individually
    for col in exchange_df.columns:
        # Format the column name
        formatted_name = format_column_name(col) if format_column_names else col
        
        # Check if this is a special column we need to handle specifically
        if col == exchange_col:
            column_config[col] = st.column_config.TextColumn(
                formatted_name,
                help="Exchange name",
                width="medium"
            )
        elif col == volume_col:
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="Trading volume in USD",
                format="$%.2f"
            )
        elif col == market_share_col and has_market_share:
            column_config[col] = st.column_config.ProgressColumn(
                formatted_name,
                help="Market share percentage",
                format="%.2f%%",
                min_value=0,
                max_value=100
            )
        elif col == market_share_col:
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="Market share percentage",
                format="%.2f%%"
            )
        elif col == open_interest_col:
            column_config[col] = st.column_config.NumberColumn(
                formatted_name,
                help="Open interest in USD",
                format="$%.2f"
            )
        else:
            # Auto-detect column type
            col_data = exchange_df[col].dropna()
            is_numeric = len(col_data) > 0 and pd.api.types.is_numeric_dtype(col_data)
            
            if is_numeric:
                # Detect special types of numeric columns
                col_name = str(col).lower()
                
                if 'price' in col_name or 'usd' in col_name or 'volume' in col_name or 'value' in col_name:
                    # Currency column
                    column_config[col] = st.column_config.NumberColumn(
                        formatted_name,
                        help=f"{formatted_name}",
                        format="$%.2f"
                    )
                elif 'percent' in col_name or 'change' in col_name or 'rate' in col_name or 'pct' in col_name or 'share' in col_name:
                    # Percentage column
                    column_config[col] = st.column_config.NumberColumn(
                        formatted_name,
                        help=f"{formatted_name}",
                        format="%.2f%%"
                    )
                else:
                    # Regular numeric column
                    column_config[col] = st.column_config.NumberColumn(
                        formatted_name,
                        help=f"{formatted_name}"
                    )
            else:
                # Text column
                column_config[col] = st.column_config.TextColumn(
                    formatted_name,
                    help=f"{formatted_name}"
                )
    
    # Create streamlit dataframe with column configuration
    return st.dataframe(
        exchange_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )

def create_summary_table(data_dict, metrics=None, format_metric_names=True):
    """
    Create a summary table from a dictionary of metrics.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary of data points
    metrics : list, optional
        List of metric names to include in the table
    format_metric_names : bool
        Whether to format metric names by replacing underscores with spaces and capitalizing words
        
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
    
    # Format metric names if requested
    if format_metric_names:
        formatted_dict = {format_column_name(k): v for k, v in filtered_dict.items()}
    else:
        formatted_dict = filtered_dict
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'Metric': formatted_dict.keys(),
        'Value': formatted_dict.values()
    })
    
    # Create a simple column configuration
    column_config = {
        'Metric': st.column_config.TextColumn(
            'Metric',
            help="Metric name",
            width="medium"
        ),
        'Value': st.column_config.TextColumn(
            'Value',
            help="Metric value",
            width="medium"
        )
    }
    
    # Create streamlit dataframe
    return st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
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

def create_comparison_table(df1, df2, key_column, value_columns, title1="Before", title2="After", format_column_names=True):
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
    format_column_names : bool
        Whether to format column names by removing underscores and capitalizing words
        
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
    
    # Generate a simple column configuration for each column
    column_config = {}
    
    if format_column_names:
        # Process each column to create an appropriate configuration
        for col in comparison_df.columns:
            # Format the column name
            formatted_name = format_column_name(col)
            
            # Determine the type of column and create appropriate configuration
            if col == key_column:
                column_config[col] = st.column_config.TextColumn(
                    formatted_name,
                    help=f"{formatted_name}"
                )
            else:
                # Check if it's a numeric column
                column_data = comparison_df[col].dropna()
                is_numeric = len(column_data) > 0 and pd.api.types.is_numeric_dtype(column_data)
                
                if is_numeric:
                    # Determine the type of numeric column
                    is_percentage = "%" in col or "Change" in col
                    is_difference = "Diff" in col
                    
                    if is_percentage:
                        column_config[col] = st.column_config.NumberColumn(
                            formatted_name,
                            help=f"{formatted_name}",
                            format="%.2f%%"
                        )
                    elif is_difference:
                        column_config[col] = st.column_config.NumberColumn(
                            formatted_name,
                            help=f"{formatted_name}",
                            format="%.2f"
                        )
                    else:
                        # Regular numeric column
                        column_config[col] = st.column_config.NumberColumn(
                            formatted_name,
                            help=f"{formatted_name}"
                        )
                else:
                    # Text column
                    column_config[col] = st.column_config.TextColumn(
                        formatted_name,
                        help=f"{formatted_name}"
                    )
    
    # Create streamlit dataframe
    return st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )