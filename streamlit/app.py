"""
Overview page for the Izun Crypto Liquidity Report.

This serves as the entry point for the Streamlit application and provides an overview
of all available report sections with descriptions.
"""

import streamlit as st
import os
import sys
import logging

# Configure logging (console only, no file logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # Only log to console, not to file
)
logger = logging.getLogger(__name__)

# Ensure the app can find modules properly
file_dir = os.path.dirname(os.path.abspath(__file__))
# Avoid duplicate path addition
if file_dir not in sys.path:
    sys.path.insert(0, file_dir)

try:
    # Import components and utilities
    from components.sidebar import render_sidebar
    from utils.config import APP_TITLE, APP_ICON
    
    logger.info(f"Successfully imported all modules")
except Exception as e:
    st.error(f"Error loading application modules: {e}")
    logger.error(f"Import error: {e}")
    raise

# Set page config with title and icon
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for tracking the current page
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'overview'

# Set the current page for sidebar navigation
st.session_state.current_page = 'overview'

def main():
    """Main function to render the overview dashboard."""
    
    try:
        # Render sidebar
        render_sidebar()
        
        # Page title and description
        st.title("📊 Overview")
        st.subheader("Guide to Izun Liquidity Report")
        
        # Create report section cards
        col1, col2 = st.columns(2)
        
        with col1:
            # Report Section
            st.subheader("📋 Report")
            st.markdown("""
            • Price comparison - Spot vs futures  
            • Performance charts - All assets  
            • Funding rates - By exchange  
            • Open interest - Real-time tracking
            """)
            
            # Futures Section
            st.subheader("🔄 Futures")
            st.markdown("""
            • Open Interest - By exchange  
            • Funding rates - Historical data  
            • Liquidations - Long/short volumes  
            • Long/short ratios - Market positioning  
            • Order book - Depth analysis  
            • Taker flows - Buy/sell pressure
            """)
            
            # Spot Section
            st.subheader("💱 Spot")
            st.markdown("""
            • Price data - OHLC history  
            • Trading pairs - All exchanges  
            • Order books - Bid/ask depth  
            • Volume analysis - 24h metrics  
            • Market share - Exchange comparison
            """)
            
            # Options Section
            st.subheader("🎯 Options")
            st.markdown("""
            • Open interest - Strike distribution  
            • Volume - Daily activity  
            • Max pain - Key levels  
            • Put/call ratio - Sentiment  
            • Exchange data - Multi-venue
            """)
        
        with col2:
            # ETF Section
            st.subheader("📈 ETF")
            st.markdown("""
            • Bitcoin ETFs - Flow tracking  
            • Ethereum ETFs - AUM data  
            • Premium/discount - NAV analysis  
            • Grayscale - GBTC, ETHE metrics  
            • Regional data - US, Hong Kong
            """)
            
            # Indicators Section
            st.subheader("📉 Indicators")
            st.markdown("""
            • Fear & Greed - Market sentiment  
            • Rainbow chart - Price bands  
            • Stock-to-Flow - Scarcity model  
            • MVRV - Valuation metric  
            • Pi Cycle - Top indicator  
            • 200W MA - Heatmap analysis
            """)
            
            # On-Chain Section
            st.subheader("⛓️ On-Chain")
            st.markdown("""
            • Exchange balances - BTC, ETH, XRP  
            • Flow tracking - In/out movements  
            • Network activity - Transaction data  
            • Supply metrics - Distribution  
            • Reserve rankings - By exchange
            """)
            
            # Historical Section
            st.subheader("📅 Historical")
            st.markdown("""
            • Long-term data - Multi-year series  
            • Market cycles - Bull/bear analysis  
            • Correlations - Cross-asset  
            • Custom ranges - Flexible queries  
            • Trend analysis - Pattern detection
            """)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Error in main: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main()