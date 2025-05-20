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
        st.title(f"{APP_ICON} Izun Crypto Liquidity Report")
        st.markdown("""
        This platform provides real-time insights into crypto market liquidity across various metrics and exchanges.
        Use the sidebar navigation to explore different sections of the report.
        """)
        
        # Report Sections
        st.header("Report Sections")
        
        # Create report section cards
        col1, col2 = st.columns(2)
        
        with col1:
            # Futures Section
            st.subheader("🔄 Futures")
            st.markdown("""
            **Comprehensive futures market analysis including:**
            - Open Interest metrics by exchange and asset
            - Funding rates and historical trends
            - Liquidation data analysis
            - Long/short ratio indicators
            - Order book depth and bid-ask spreads
            - Trading volume by exchange
            
            Ideal for understanding derivatives market sentiment and liquidity conditions.
            """)
            
            # ETF Section
            st.subheader("📈 ETF")
            st.markdown("""
            **Detailed analysis of crypto ETF markets:**
            - Bitcoin ETF flow data and AUM tracking
            - Ethereum ETF metrics and comparison
            - Premium/discount analysis
            - Historical price and performance data
            - Market share analysis by issuer
            - Grayscale fund insights
            
            Essential for institutional market tracking and ETF performance comparison.
            """)
            
            # Indicators Section
            st.subheader("📉 Indicators")
            st.markdown("""
            **Market indicators and sentiment analysis:**
            - Fear & Greed Index historical trends
            - Bitcoin cycle indicators (Stock-to-Flow, Rainbow Chart)
            - Bull market peak indicators
            - Golden ratio multiplier
            - Bitcoin profitable days percentage
            - MVRV and Puell Multiple
            - 200-week moving average heatmap
            
            Useful for market cycle analysis and sentiment tracking.
            """)
            
            # Historical Section
            st.subheader("📅 Historical")
            st.markdown("""
            **Long-term historical data and trends:**
            - Extended historical price analysis
            - Comparative performance across market cycles
            - Long-term market metrics and correlations
            - Historical liquidity trends
            
            Ideal for researching long-term market patterns and cyclical behaviors.
            """)
        
        with col2:
            # Spot Section
            st.subheader("💱 Spot")
            st.markdown("""
            **Spot market analysis across exchanges:**
            - Trading volume by exchange and pair
            - Spot order book depth analysis
            - Bid-ask spread monitoring
            - Taker buy/sell volume ratio
            - Market concentration metrics
            - Exchange comparison and market share
            
            Critical for understanding exchange liquidity and spot market trends.
            """)
            
            # Options Section
            st.subheader("🎯 Options")
            st.markdown("""
            **Options market metrics and analysis:**
            - Open interest by strike price and expiry
            - Put/call ratio tracking
            - Options volume analysis
            - Implied volatility surface
            - Max pain analysis
            - Exchange comparison for options markets
            
            Valuable for understanding market sentiment and hedging activity.
            """)
            
            # On-Chain Section
            st.subheader("⛓️ On-Chain")
            st.markdown("""
            **Blockchain network and on-chain analysis:**
            - Exchange inflow/outflow tracking
            - Exchange balance monitoring
            - Network activity metrics
            - Wallet distribution data
            - Chain transaction metrics
            - Supply distribution analysis
            
            Essential for understanding network health and institutional movements.
            """)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")
        logger.error(f"Error in main: {e}", exc_info=True)
        return

if __name__ == "__main__":
    main()