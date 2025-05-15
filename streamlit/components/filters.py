"""
Beautiful, compact filter components for Streamlit pages in the Liquidity Report.
"""

# List of major exchanges to ensure they're always included if needed
MAJOR_EXCHANGES = [
    "All", 
    "Binance",
    "Bybit", 
    "OKX", 
    "Deribit", 
    "Bitget", 
    "Hyperliquid", 
    "Kraken", 
    "Coinbase", 
    "dYdX", 
    "Bitfinex",
    "Huobi", 
    "Gate.io",
    "Kucoin",
    "MEXC"
]

import streamlit as st
import logging

# Configure logging
logger = logging.getLogger(__name__)

def compact_exchange_filter(
    available_exchanges,
    default_exchanges=None,
    key_prefix="filter",
    asset="generic",
    help_text="Select exchange to include in the chart",
    label="Exchange"
):
    """
    Creates a beautiful compact exchange dropdown filter designed to be placed near charts.
    
    Parameters:
    -----------
    available_exchanges : list
        List of available exchanges to choose from
    default_exchanges : list or str, optional
        Default exchange to select (if None, will select 'All' if available, otherwise first exchange)
    key_prefix : str
        Prefix for the session state key to ensure uniqueness
    asset : str
        Asset name for unique key generation
    help_text : str
        Help text to display on hover
    label : str
        Label for the filter
        
    Returns:
    --------
    str
        Selected exchange
    """
    try:
        # Ensure 'hyperliquid' is included among available exchanges if not present
        if "hyperliquid" not in [ex.lower() for ex in available_exchanges]:
            available_exchanges = list(available_exchanges) + ["Hyperliquid"]
            
        # Make sure 'All' is the first option if available
        if "All" in available_exchanges:
            available_exchanges = ["All"] + [ex for ex in available_exchanges if ex != "All"]
        
        # Sort the remaining exchanges alphabetically for better organization (keeping 'All' first)
        if "All" in available_exchanges:
            sorted_exchanges = ["All"] + sorted([ex for ex in available_exchanges if ex != "All"])
        else:
            sorted_exchanges = sorted(available_exchanges)
        
        # Set default selection
        if default_exchanges is None:
            if "All" in sorted_exchanges:
                default_exchange = "All"
            elif len(sorted_exchanges) > 0:
                default_exchange = sorted_exchanges[0]
            else:
                default_exchange = None
        elif isinstance(default_exchanges, list) and len(default_exchanges) > 0:
            default_exchange = default_exchanges[0]  # Take the first one if it's a list
        else:
            default_exchange = default_exchanges
                
        # Create a unique key for this filter
        key = f"{key_prefix}_exchange_filter_{asset}"
        
        # Add custom CSS for a beautiful dropdown
        st.markdown("""
        <style>
        div[data-testid="stSelectbox"] > div:first-child > div:first-child {
            padding-top: 5px;
            padding-bottom: 5px;
        }
        div[data-testid="stSelectbox"] > div > div > div {
            padding-top: 2px;
            padding-bottom: 2px;
        }
        div[data-testid="stSelectbox"] label {
            font-size: 14px;
            font-weight: 500;
            color: #555;
        }
        div[data-testid="stSelectbox"] > div:first-child {
            border-radius: 6px;
            box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
            transition: box-shadow 0.2s;
        }
        div[data-testid="stSelectbox"]:hover > div:first-child {
            box-shadow: 0px 3px 5px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create the filter with minimal height and beautiful styling
        container = st.container()
        with container:
            # Create the dropdown (selectbox)
            selected = st.selectbox(
                label,
                sorted_exchanges,
                index=sorted_exchanges.index(default_exchange) if default_exchange in sorted_exchanges else 0,
                key=key,
                help=help_text,
                # Format label for more compact appearance
                label_visibility="visible"
            )
                
        return selected
    except Exception as e:
        logger.error(f"Error creating compact exchange filter: {e}")
        # Return a safe default
        if isinstance(default_exchanges, list) and len(default_exchanges) > 0:
            return default_exchanges[0]
        elif isinstance(default_exchanges, str):
            return default_exchanges
        elif "All" in available_exchanges:
            return "All"
        elif len(available_exchanges) > 0:
            return available_exchanges[0]
        else:
            return "All"

def compact_asset_filter(
    available_assets,
    default_assets=None,
    key_prefix="filter",
    chart_type="generic",
    help_text="Select asset to include in the chart",
    label="Asset"
):
    """
    Creates a beautiful compact asset dropdown filter designed to be placed near charts.
    
    Parameters:
    -----------
    available_assets : list
        List of available assets to choose from
    default_assets : list or str, optional
        Default asset to select (if None, will select first asset)
    key_prefix : str
        Prefix for the session state key to ensure uniqueness
    chart_type : str
        Chart type for unique key generation
    help_text : str
        Help text to display on hover
    label : str
        Label for the filter
        
    Returns:
    --------
    str
        Selected asset
    """
    try:
        # Sort assets alphabetically for better organization
        sorted_assets = sorted(available_assets)
        
        # Set default selection
        if default_assets is None:
            if len(sorted_assets) > 0:
                default_asset = sorted_assets[0]
            else:
                default_asset = None
        elif isinstance(default_assets, list) and len(default_assets) > 0:
            default_asset = default_assets[0]  # Take the first one if it's a list
        else:
            default_asset = default_assets
                
        # Create a unique key for this filter
        key = f"{key_prefix}_asset_filter_{chart_type}"
        
        # Add custom CSS for a beautiful dropdown
        st.markdown("""
        <style>
        div[data-testid="stSelectbox"] > div:first-child > div:first-child {
            padding-top: 5px;
            padding-bottom: 5px;
        }
        div[data-testid="stSelectbox"] > div > div > div {
            padding-top: 2px;
            padding-bottom: 2px;
        }
        div[data-testid="stSelectbox"] label {
            font-size: 14px;
            font-weight: 500;
            color: #555;
        }
        div[data-testid="stSelectbox"] > div:first-child {
            border-radius: 6px;
            box-shadow: 0px 1px 3px rgba(0,0,0,0.1);
            transition: box-shadow 0.2s;
        }
        div[data-testid="stSelectbox"]:hover > div:first-child {
            box-shadow: 0px 3px 5px rgba(0,0,0,0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create the filter with minimal height and beautiful styling
        container = st.container()
        with container:
            # Create the dropdown (selectbox)
            selected = st.selectbox(
                label,
                sorted_assets,
                index=sorted_assets.index(default_asset) if default_asset in sorted_assets else 0,
                key=key,
                help=help_text,
                # Format label for more compact appearance
                label_visibility="visible"
            )
                
        return selected
    except Exception as e:
        logger.error(f"Error creating compact asset filter: {e}")
        # Return a safe default
        if isinstance(default_assets, list) and len(default_assets) > 0:
            return default_assets[0]
        elif isinstance(default_assets, str):
            return default_assets
        elif len(available_assets) > 0:
            return available_assets[0]
        else:
            return ""

def chart_filter_group(
    chart_title,
    available_exchanges=None,
    available_assets=None,
    default_exchange=None,
    default_asset=None,
    key_prefix="chart",
    show_exchanges=True,
    show_assets=True,
    additional_filters=None
):
    """
    Creates a beautiful compact filter group for charts with optional exchange and asset dropdowns.
    
    Parameters:
    -----------
    chart_title : str
        Title to display above the filter group
    available_exchanges : list, optional
        List of available exchanges to choose from
    available_assets : list, optional
        List of available assets to choose from
    default_exchange : str, optional
        Default exchange to select
    default_asset : str, optional
        Default asset to select
    key_prefix : str
        Prefix for the session state key to ensure uniqueness
    show_exchanges : bool
        Whether to show exchange filter
    show_assets : bool
        Whether to show asset filter
    additional_filters : function, optional
        Function that returns additional custom filters
        
    Returns:
    --------
    dict
        Dictionary containing selected filters:
        - 'exchange': Selected exchange
        - 'asset': Selected asset
        - 'custom': Any custom filter values returned by additional_filters
    """
    try:
        # Create result dictionary
        result = {}
        
        # Add custom CSS for the filter group
        st.markdown("""
        <style>
        .filter-group {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 15px;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
        }
        .filter-title {
            font-size: 16px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }
        /* Make Streamlit components more compact */
        .stSelectbox {
            margin-bottom: 0px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Create filter group container
        st.markdown(f"<div class='filter-group'><div class='filter-title'>{chart_title}</div>", unsafe_allow_html=True)
        
        # Determine number of columns based on what filters to show
        num_columns = 0
        if show_exchanges and available_exchanges:
            num_columns += 1
        if show_assets and available_assets:
            num_columns += 1
        if additional_filters:
            num_columns += 1
            
        # Ensure at least 1 column
        num_columns = max(1, num_columns)
        
        # Create columns for filters
        cols = st.columns(num_columns)
        
        # Column index tracker
        col_idx = 0
        
        # Add exchange filter if requested
        if show_exchanges and available_exchanges:
            # Ensure we have major exchanges included
            for ex in MAJOR_EXCHANGES:
                if ex not in available_exchanges and ex != "All":
                    available_exchanges = list(available_exchanges) + [ex]
                    
            with cols[col_idx]:
                selected_exchange = compact_exchange_filter(
                    available_exchanges,
                    default_exchange,
                    key_prefix=f"{key_prefix}_{chart_title.lower().replace(' ', '_')}",
                    label="Exchange"
                )
                result['exchange'] = selected_exchange
            col_idx += 1
            
        # Add asset filter if requested
        if show_assets and available_assets:
            with cols[col_idx]:
                selected_asset = compact_asset_filter(
                    available_assets,
                    default_asset,
                    key_prefix=f"{key_prefix}_{chart_title.lower().replace(' ', '_')}",
                    label="Asset"
                )
                result['asset'] = selected_asset
            col_idx += 1
            
        # Add additional filters if provided
        if additional_filters and callable(additional_filters):
            with cols[col_idx]:
                custom_filter_results = additional_filters()
                if custom_filter_results:
                    result['custom'] = custom_filter_results
                    
        # Close the filter group container
        st.markdown("</div>", unsafe_allow_html=True)
        
        return result
    except Exception as e:
        logger.error(f"Error creating chart filter group: {e}")
        # Return empty result on error
        return {
            'exchange': default_exchange if default_exchange else ('All' if 'All' in (available_exchanges or []) else None),
            'asset': default_asset if default_asset else (available_assets[0] if available_assets and len(available_assets) > 0 else None)
        }