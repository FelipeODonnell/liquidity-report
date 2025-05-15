"""
Test script for testing display_filterable_chart function
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

# Add parent directory to path to import from components and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

# Import components and utilities
from components.charts import (
    create_pie_chart,
    apply_chart_theme,
    display_chart,
    display_filterable_chart
)

# Set page config
st.set_page_config(
    page_title="Chart Test",
    page_icon="📊",
    layout="wide"
)

# Create a sample dataframe for testing
data = {
    'Exchange': ['Binance', 'Bybit', 'OKX', 'Deribit', 'Bitget', 'Hyperliquid'],
    'Value': [100, 80, 60, 40, 30, 20]
}
df = pd.DataFrame(data)

# Create a test pie chart
fig = create_pie_chart(
    df, 
    values_col='Value', 
    names_col='Exchange', 
    title='Test Pie Chart'
)

# Display with filterable chart to test functionality
try:
    st.header("Testing display_filterable_chart")
    
    # Display with exchange filter
    filter_result = display_filterable_chart(
        fig,
        filter_options={
            'exchanges': df['Exchange'].tolist(),  
            'selected_exchanges': 'Binance'
        },
        chart_id="test_chart",
        asset="BTC"
    )
    
    st.write("Filter results:", filter_result)
    
except Exception as e:
    st.error(f"Error occurred: {e}")
    import traceback
    st.code(traceback.format_exc())