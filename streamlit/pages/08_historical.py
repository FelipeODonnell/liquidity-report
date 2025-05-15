"""
Historical Reports & Data page for the Izun Crypto Liquidity Report.

This page allows users to access and download historical data.
"""

import streamlit as st
import pandas as pd
import os
import glob
import sys
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Add parent directory to path to import from components and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components and utilities
from components.sidebar import render_sidebar
from utils.data_loader import (
    list_data_directories,
    get_latest_data_directory,
    load_parquet_file,
    convert_df_to_csv
)
from utils.formatters import format_timestamp
from utils.config import APP_TITLE, APP_ICON, DATA_BASE_PATH

# Set page config with title and icon
st.set_page_config(
    page_title=f"{APP_TITLE} - Historical Data",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the current page for sidebar navigation
st.session_state.current_page = 'historical'

# Theme settings removed as requested

# Initialize visible pages settings
if 'visible_pages' not in st.session_state:
    # Default: all pages are visible
    st.session_state.visible_pages = {
        'report': True,
        'etf': True,
        'futures': True,
        'spot': True,
        'indicators': True,
        'on_chain': True,
        'options': True,
        'historical': True
    }

def get_data_categories():
    """
    Get a list of all data categories from the data directory.
    
    Returns:
    --------
    list
        List of category names
    """
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        return []
    
    # Get categories from the directory structure
    categories = []
    data_path = os.path.join(DATA_BASE_PATH, latest_dir)
    
    if os.path.exists(data_path):
        categories = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    return sorted(categories)

def get_subcategories(category, data_dir):
    """
    Get a list of subcategories for a given category.
    
    Parameters:
    -----------
    category : str
        The category to get subcategories for
    data_dir : str
        The data directory to look in
        
    Returns:
    --------
    list
        List of subcategory names
    """
    subcategories = []
    category_path = os.path.join(DATA_BASE_PATH, data_dir, category)
    
    if os.path.exists(category_path):
        subcategories = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
    
    return sorted(subcategories)

def get_data_files(category, subcategory=None, data_dir=None):
    """
    Get a list of data files for a given category and optional subcategory.
    
    Parameters:
    -----------
    category : str
        The category to get files for
    subcategory : str, optional
        The subcategory to get files for
    data_dir : str, optional
        The data directory to look in. If None, the latest is used.
        
    Returns:
    --------
    list
        List of file paths
    """
    if data_dir is None:
        data_dir = get_latest_data_directory()
        if not data_dir:
            return []
    
    if subcategory:
        search_path = os.path.join(DATA_BASE_PATH, data_dir, category, subcategory, "*.parquet")
    else:
        search_path = os.path.join(DATA_BASE_PATH, data_dir, category, "*.parquet")
    
    return sorted(glob.glob(search_path))

def render_historical_data_browser():
    """Render the historical data browser interface."""
    st.header("Historical Data Browser")
    
    # Get list of available data directories
    data_dirs = list_data_directories()
    
    if not data_dirs:
        st.error("No data directories found.")
        return
    
    # Create data directory selector
    selected_dir = st.selectbox(
        "Select Data Date",
        data_dirs,
        format_func=lambda x: format_timestamp(datetime.strptime(x, "%Y%m%d"), "%Y-%m-%d") if x.isdigit() and len(x) == 8 else x,
        index=0  # Default to the latest (first) directory
    )
    
    # Get categories for the selected directory
    categories = get_data_categories()
    
    if not categories:
        st.warning(f"No data categories found in {selected_dir}.")
        return
    
    # Create category selector
    selected_category = st.selectbox(
        "Select Category",
        categories
    )
    
    # Check if the category has subcategories
    subcategories = get_subcategories(selected_category, selected_dir)
    
    if subcategories:
        # Create subcategory selector
        selected_subcategory = st.selectbox(
            "Select Subcategory",
            ["All"] + subcategories
        )
        
        if selected_subcategory == "All":
            # Show all files across subcategories
            all_files = []
            for subcategory in subcategories:
                subcategory_files = get_data_files(selected_category, subcategory, selected_dir)
                all_files.extend(subcategory_files)
            
            files = all_files
        else:
            # Show files for the selected subcategory
            files = get_data_files(selected_category, selected_subcategory, selected_dir)
    else:
        # No subcategories, show files for the category
        files = get_data_files(selected_category, None, selected_dir)
    
    if not files:
        st.warning("No data files found for the selected criteria.")
        return
    
    # Display file list with previews
    st.subheader("Available Data Files")
    
    # Extract file names from paths
    file_names = [os.path.basename(f).replace('.parquet', '') for f in files]
    
    # Create file selector
    selected_file_index = st.selectbox(
        "Select Data File",
        range(len(file_names)),
        format_func=lambda i: file_names[i]
    )
    
    selected_file = files[selected_file_index]
    selected_file_name = file_names[selected_file_index]
    
    # Load and display file preview
    df = load_parquet_file(selected_file)
    
    if df.empty:
        st.error(f"Error loading {selected_file_name} or file is empty.")
        return
    
    # Display file information
    st.markdown(f"**File Path:** `{selected_file}`")
    st.markdown(f"**Rows:** {len(df)}, **Columns:** {len(df.columns)}")
    st.markdown("**Columns:** " + ", ".join(df.columns.tolist()))
    
    # Display preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Create download button
    csv = convert_df_to_csv(df)
    
    st.download_button(
        label=f"Download {selected_file_name} as CSV",
        data=csv,
        file_name=f"{selected_file_name}.csv",
        mime="text/csv"
    )

def render_settings():
    """Render the application settings interface."""
    st.header("Application Settings")

    # Theme Settings section removed as requested

    # Page visibility settings
    st.subheader("Page Visibility Settings")
    st.write("Select which pages to show in the sidebar navigation:")

    # Dictionary of page names and their friendly display names
    page_labels = {
        'report': "📊 Report",
        'etf': "📈 ETF",
        'futures': "🔄 Futures",
        'spot': "💱 Spot",
        'indicators': "📉 Indicators",
        'on_chain': "⛓️ On-Chain",
        'options': "🎯 Options",
        'historical': "📅 Historical"
    }

    # Create toggles for each page
    visible_pages = {}

    for page_id, page_label in page_labels.items():
        is_visible = st.checkbox(
            page_label,
            value=st.session_state.visible_pages.get(page_id, True),
            key=f"visibility_{page_id}"
        )
        visible_pages[page_id] = is_visible

    # Always keep the current page visible to avoid navigation issues
    visible_pages['historical'] = True

    # Update session state if changed
    if visible_pages != st.session_state.visible_pages:
        st.session_state.visible_pages = visible_pages
        st.success("Page visibility settings updated. Navigate to the home page to see changes.")

    # Advanced settings section
    st.subheader("Data Settings")

    # Show data path information
    data_path = os.path.abspath(DATA_BASE_PATH)
    st.info(f"Data directory: {data_path}")

    # Show available data dates
    data_dirs = list_data_directories()
    st.write(f"Available data dates: {len(data_dirs)}")

    if data_dirs:
        latest_date = data_dirs[0]
        formatted_date = format_timestamp(datetime.strptime(latest_date, "%Y%m%d"), "%Y-%m-%d") if latest_date.isdigit() and len(latest_date) == 8 else latest_date
        st.success(f"Latest data from: {formatted_date}")

    # Reset all settings button
    if st.button("Reset All Settings to Default"):
        # No need to reset theme as it's been removed
        st.session_state.visible_pages = {page_id: True for page_id in page_labels.keys()}
        st.success("Page visibility settings have been reset to default values.")
        st.info("Page will reload in a moment...")
        st.rerun()

def main():
    """Main function to render the historical data page."""
    
    # Render sidebar
    render_sidebar()
    
    # Page title and description
    st.title("Historical Reports & Data")
    st.write("Access and download historical cryptocurrency market data")
    
    # Create tabs for different data access methods
    tab1, tab2 = st.tabs([
        "Data Browser",
        "Settings"
    ])

    with tab1:
        render_historical_data_browser()

    with tab2:
        render_settings()
    

if __name__ == "__main__":
    main()