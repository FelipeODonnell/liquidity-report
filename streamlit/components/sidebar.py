"""
Sidebar navigation component for the Izun Crypto Liquidity Report application.
"""

import streamlit as st
from datetime import datetime, timedelta
import sys
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Calculate the correct path to parent directory for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Only add path if it's not already in sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
    logger.info(f"Added to path: {parent_dir}")

try:
    from utils.data_loader import get_data_last_updated, get_available_assets_for_category
    from utils.formatters import format_timestamp, humanize_time_diff
    from utils.config import DATA_BASE_PATH
except ImportError as e:
    logger.error(f"Failed to import required modules in sidebar: {e}")
    # The error will be shown to the user when the component is used

def render_sidebar():
    """
    Render the sidebar navigation with filters and navigation links.
    
    This function creates the standard sidebar used across all pages
    of the application, including filters and navigation links.
    """
    try:
        # Add logo at the top of the sidebar using a relative path for Streamlit Cloud compatibility
        import os
        image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "izun_partners_logo.jpeg")
        st.sidebar.image(image_path, width=120)
        st.sidebar.title("Izun Liquidity Report")
        
        # Check data directory existence and show info
        if not os.path.exists(DATA_BASE_PATH):
            st.sidebar.error("⚠️ Data directory not found")
            st.sidebar.info(f"Looking for data at: {DATA_BASE_PATH}")
        else:
            # Get last updated time
            try:
                last_updated = get_data_last_updated()
                if last_updated:
                    st.sidebar.caption(f"Last updated: {format_timestamp(last_updated, '%Y-%m-%d %H:%M')}")
                    st.sidebar.caption(f"({humanize_time_diff(last_updated)})")
                else:
                    st.sidebar.caption("Last updated: Unknown")
            except Exception as e:
                logger.error(f"Error getting last updated time: {e}")
                st.sidebar.caption("Last updated: Error retrieving update time")
        
        # Determine current page for conditional elements
        current_page = st.session_state.get('current_page', 'report')
    except Exception as e:
        st.sidebar.error(f"Error initializing sidebar: {e}")
        logger.error(f"Sidebar initialization error: {e}", exc_info=True)
        # Set a default page to avoid further errors
        current_page = 'report'
    
    # Asset selector for relevant pages
    try:
        if current_page in ['futures', 'spot', 'options', 'report']:
            available_assets = []
            try:
                if current_page == 'futures':
                    available_assets = get_available_assets_for_category('futures')
                elif current_page == 'spot':
                    available_assets = get_available_assets_for_category('spot')
                elif current_page == 'options':
                    available_assets = get_available_assets_for_category('options')
                elif current_page == 'report':
                    available_assets = ['BTC', 'ETH', 'SOL', 'XRP']  # Default for report page
            except Exception as e:
                logger.error(f"Error getting available assets: {e}")
                # Fallback to default assets
                available_assets = ['BTC', 'ETH', 'SOL', 'XRP']
                st.sidebar.warning("Using default assets due to data loading error")
            
            # Default to BTC if available, otherwise first asset
            default_asset = 'BTC' if 'BTC' in available_assets else (available_assets[0] if available_assets else None)
            
            if available_assets:
                st.sidebar.subheader("Assets")
                selected_asset = st.sidebar.selectbox(
                    "Select Asset",
                    available_assets,
                    index=available_assets.index(default_asset) if default_asset in available_assets else 0
                )
                
                # Store selected asset in session state
                st.session_state.selected_asset = selected_asset
            else:
                st.sidebar.warning("No assets available for selection")
                # Set a default asset to avoid errors
                st.session_state.selected_asset = 'BTC'
    except Exception as e:
        logger.error(f"Error setting up asset selector: {e}")
        st.sidebar.error("Error loading asset selector")
    
    # Date range selector
    try:
        today = datetime.now().date()
        
        # Set default to 3 months, with max allowed of 6 months ago
        default_days = 90  # 3 months
        max_days = 180  # 6 months

        # Adjust based on page if needed
        if current_page == 'report':
            default_days = 90
        elif current_page == 'etf':
            default_days = 90

        default_start_date = today - timedelta(days=default_days)
        min_date = today - timedelta(days=max_days)
        
        # Only show date filter for pages with time series data
        if current_page not in ['historical']:
            st.sidebar.subheader("Date Range")
            
            try:
                # Date range selector
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=default_start_date,
                        min_value=min_date,
                        key=f"start_date_{current_page}"
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=today,
                        key=f"end_date_{current_page}"
                    )
                
                # Store date range in session state
                st.session_state.date_range = {
                    'start': start_date,
                    'end': end_date
                }
            except Exception as e:
                logger.error(f"Error setting up date selector: {e}")
                # Set default dates in session state to avoid errors
                st.session_state.date_range = {
                    'start': default_start_date,
                    'end': today
                }
                st.sidebar.warning("Error loading date selector, using defaults")
    except Exception as e:
        logger.error(f"Error in date range section: {e}")
        # Fall back to default date range
        st.session_state.date_range = {
            'start': datetime.now().date() - timedelta(days=7),
            'end': datetime.now().date()
        }
    
    # Exchange selector for relevant pages
    try:
        if current_page in ['futures', 'spot', 'options']:
            st.sidebar.subheader("Exchanges")
            
            # Dynamic exchange list based on the page
            exchanges = []
            if current_page == 'futures':
                exchanges = ["Binance", "OKX", "Bybit", "dYdX", "Bitfinex", "All"]
            elif current_page == 'spot':
                exchanges = ["Binance", "Coinbase", "Kraken", "OKX", "All"]
            elif current_page == 'options':
                exchanges = ["Deribit", "OKX", "Binance", "All"]
            
            if exchanges:
                selected_exchanges = st.sidebar.multiselect(
                    "Select Exchanges",
                    exchanges,
                    default=["All"]
                )
                
                # Store selected exchanges in session state
                st.session_state.selected_exchanges = selected_exchanges
    except Exception as e:
        logger.error(f"Error in exchange selector: {e}")
        # Set default exchanges in session state
        st.session_state.selected_exchanges = ["All"]
    
    # Navigation links
    try:
        st.sidebar.subheader("Navigation")
        
        # Define pages with their labels and file paths
        pages = [
            {"label": "Report", "path": "app.py", "icon": "📊"},
            {"label": "Futures", "path": "pages/03_futures.py", "icon": "🔄"},
            {"label": "Spot", "path": "pages/04_spot.py", "icon": "💱"},
            {"label": "Options", "path": "pages/07_options.py", "icon": "🎯"},  # Moved below Spot
            {"label": "ETF", "path": "pages/02_etf.py", "icon": "📈"},
            {"label": "Indicators", "path": "pages/05_indicators.py", "icon": "📉"},
            {"label": "On-Chain", "path": "pages/06_on_chain.py", "icon": "⛓️"},
            {"label": "Historical", "path": "pages/08_historical.py", "icon": "📅"}
        ]
        
        # Display navigation links based on visibility settings
        for page in pages:
            # Get the page ID from the label (lowercase)
            page_id = page["label"].lower()

            # Check if the page should be visible (default to True if not in session state)
            is_visible = st.session_state.get('visible_pages', {}).get(page_id, True)

            # Only show the link if the page is visible
            if is_visible:
                try:
                    # Use the page_link feature with relative file paths
                    st.sidebar.page_link(
                        page["path"],
                        label=f"{page['icon']} {page['label']}",
                        use_container_width=True
                    )
                except Exception as e:
                    logger.error(f"Error linking to page {page['label']}: {e}")
                    # Fallback to just showing the label without a link
                    st.sidebar.markdown(f"{page['icon']} {page['label']}")
    except Exception as e:
        logger.error(f"Error creating navigation links: {e}")
        st.sidebar.error("Navigation links could not be loaded")
    
    # Credits and data source information
    try:
        st.sidebar.divider()
        st.sidebar.caption("Data source: CoinGlass API")
        st.sidebar.caption("Izun Crypto Liquidity Report © 2025")
        
        # Removed debug info section to clean up the sidebar
    except Exception as e:
        logger.error(f"Error adding footer information: {e}")
        # Non-critical, so don't show error to user

def render_page_filters(category=None):
    """
    Render specific filters for a page category.
    
    Parameters:
    -----------
    category : str
        The category of the page (e.g., 'etf', 'futures')
    """
    if not category:
        return
    
    # Create a container for the filters
    filter_container = st.container()
    
    with filter_container:
        # Specific filters for different categories
        if category == 'etf':
            col1, col2 = st.columns(2)
            with col1:
                st.selectbox("ETF Type", ["Bitcoin ETFs", "Ethereum ETFs", "Grayscale Funds", "All"])
            with col2:
                st.selectbox("Region", ["US", "Hong Kong", "All"])
                
        elif category == 'futures':
            subcategory = st.session_state.get('futures_subcategory', 'overview')
            
            # Different filters for different subcategories
            if subcategory == 'funding_rate':
                st.selectbox("Weight Method", ["Equal", "Open Interest Weighted", "Volume Weighted"])
            elif subcategory == 'liquidation':
                st.selectbox("Liquidation Type", ["Long", "Short", "Both"])
            elif subcategory == 'open_interest':
                st.selectbox("Margin Type", ["All", "Coin-margined", "Stablecoin-margined"])
                
        elif category == 'spot':
            st.selectbox("Market Type", ["USDT", "USD", "BUSD", "All"])
            
        elif category == 'indicators':
            indicator_type = st.selectbox(
                "Indicator Category",
                ["Market Sentiment", "Bitcoin Cycles", "Technical Indicators", "All"]
            )
            
def get_dynamic_filters():
    """
    Get the current filter settings from session state.
    
    Returns:
    --------
    dict
        Dictionary of current filter settings
    """
    filters = {}
    
    # Add selected asset if it exists
    if 'selected_asset' in st.session_state:
        filters['asset'] = st.session_state.selected_asset
    
    # Add date range if it exists
    if 'date_range' in st.session_state:
        filters['date_range'] = st.session_state.date_range
    
    # Add selected exchanges if they exist
    if 'selected_exchanges' in st.session_state:
        filters['exchanges'] = st.session_state.selected_exchanges
    
    return filters