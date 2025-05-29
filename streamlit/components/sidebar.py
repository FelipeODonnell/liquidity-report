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
            # Removed last updated time references
            pass
        
        # Determine current page for conditional elements
        current_page = st.session_state.get('current_page', 'report')
    except Exception as e:
        st.sidebar.error(f"Error initializing sidebar: {e}")
        logger.error(f"Sidebar initialization error: {e}", exc_info=True)
        # Set a default page to avoid further errors
        current_page = 'report'
    
    # Asset selector has been moved to individual pages
    # Default to BTC to ensure backward compatibility with other parts of the code
    if 'selected_asset' not in st.session_state:
        st.session_state.selected_asset = 'BTC'
    
    # Multi-select asset functionality is now handled in each relevant page
    if 'selected_assets' not in st.session_state:
        st.session_state.selected_assets = ['BTC', 'ETH']
    
    # Time range selection has been moved to individual charts
    # Set default time range for backward compatibility
    try:
        today = datetime.now().date()
        
        # Set default to 3 months
        default_days = 90  # 3 months
        default_start_date = today - timedelta(days=default_days)
        
        # Initialize session state with default time ranges for backward compatibility
        if 'selected_time_range' not in st.session_state:
            st.session_state.selected_time_range = '3M'
            
        # Set default date range in session state to avoid errors
        st.session_state.date_range = {
            'start': default_start_date,
            'end': today
        }
    except Exception as e:
        logger.error(f"Error initializing default time range: {e}")
        # Fall back to default date range
        st.session_state.date_range = {
            'start': datetime.now().date() - timedelta(days=90),
            'end': datetime.now().date()
        }
    
    # Exchange selector has been moved to individual pages
    # Set a default for backward compatibility
    try:
        if 'selected_exchanges' not in st.session_state:
            st.session_state.selected_exchanges = ["All"]
    except Exception as e:
        logger.error(f"Error initializing default exchange selection: {e}")
        # Set default exchanges in session state
        st.session_state.selected_exchanges = ["All"]
    
    # Navigation links
    try:
        st.sidebar.subheader("Navigation")
        
        # Define pages with their labels and file paths
        pages = [
            {"label": "Overview", "path": "app.py", "icon": "📊"},
            {"label": "Report", "path": "pages/01_report.py", "icon": "📋"},
            # {"label": "Basis", "path": "pages/09_basis.py", "icon": "💹"},  # Commented out - still in development
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
    
    # Footer
    try:
        st.sidebar.divider()
        # Removed data source and copyright references
    except Exception as e:
        logger.error(f"Error adding footer: {e}")
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