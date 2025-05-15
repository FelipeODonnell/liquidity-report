"""
Indicators page for the Izun Crypto Liquidity Report.

This page displays data and visualizations related to cryptocurrency market indicators.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Add parent directory to path to import from components and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components and utilities
from components.sidebar import render_sidebar
from components.metrics import display_metrics_row, display_metric_card
from components.charts import (
    create_time_series, 
    create_bar_chart, 
    create_time_series_with_bar,
    create_pie_chart,
    apply_chart_theme,
    display_chart
)
from components.tables import create_formatted_table
from utils.data_loader import (
    get_latest_data_directory, 
    load_data_for_category, 
    process_timestamps,
    get_data_last_updated,
    calculate_metrics
)
from utils.formatters import (
    format_currency, 
    format_percentage, 
    format_volume,
    format_timestamp,
    humanize_time_diff
)
from utils.config import APP_TITLE, APP_ICON, ASSET_COLORS

# Set page config with title and icon
st.set_page_config(
    page_title=f"{APP_TITLE} - Market Indicators",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the current page for sidebar navigation
st.session_state.current_page = 'indicators'

def load_indicators_data():
    """
    Load indicators data for the page.
    
    Returns:
    --------
    dict
        Dictionary containing indicators data
    """
    data = {}
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return data
    
    # Load indicators data
    data = load_data_for_category('indic')
    
    return data

def render_fear_greed_index(data, selected_time_range=None):
    """Render the fear & greed index visualization.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing indicators data
    selected_time_range : str, optional
        Selected time range for filtering data
    """
    st.header("Fear & Greed Index")

    try:
        if 'api_index_fear_greed_history' in data and not data['api_index_fear_greed_history'].empty:
            try:
                fear_greed_df = data['api_index_fear_greed_history'].copy()

                # Process dataframe
                if 'time_list' in fear_greed_df.columns:
                    fear_greed_df['datetime'] = pd.to_datetime(fear_greed_df['time_list'], unit='ms')
                else:
                    # If time_list is not available, try other columns
                    logger.info(f"Columns in fear_greed_df: {list(fear_greed_df.columns)}")

                    if 'time' in fear_greed_df.columns:
                        fear_greed_df['datetime'] = pd.to_datetime(fear_greed_df['time'], unit='ms')
                    elif 'timestamp' in fear_greed_df.columns:
                        fear_greed_df['datetime'] = pd.to_datetime(fear_greed_df['timestamp'], unit='ms')
                    else:
                        fear_greed_df = process_timestamps(fear_greed_df)

                # Get current value - check for various possible column names
                value_col = None
                for col in ['data_list', 'value', 'index_value', 'fear_greed_value']:
                    if col in fear_greed_df.columns:
                        value_col = col
                        break

                if value_col is None:
                    st.warning("Fear & Greed Index data available but missing value column.")
                    st.dataframe(fear_greed_df.head())
                    return

                current_fg = fear_greed_df[value_col].iloc[-1] if len(fear_greed_df) > 0 else None

                if current_fg is not None:
                    # Determine sentiment based on value
                    if current_fg >= 0 and current_fg < 25:
                        sentiment = "Extreme Fear"
                        color = "red"
                    elif current_fg >= 25 and current_fg < 50:
                        sentiment = "Fear"
                        color = "orange"
                    elif current_fg >= 50 and current_fg < 75:
                        sentiment = "Greed"
                        color = "lightgreen"
                    else:
                        sentiment = "Extreme Greed"
                        color = "green"

                    # Display current value
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        # Create a gauge-like display for the current value
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=current_fg,
                            title={'text': f"Current: {sentiment}"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 25], 'color': 'rgba(255, 0, 0, 0.3)'},
                                    {'range': [25, 50], 'color': 'rgba(255, 165, 0, 0.3)'},
                                    {'range': [50, 75], 'color': 'rgba(144, 238, 144, 0.3)'},
                                    {'range': [75, 100], 'color': 'rgba(0, 128, 0, 0.3)'}
                                ]
                            }
                        ))

                        fig.update_layout(
                            height=300,
                            margin=dict(l=10, r=10, t=50, b=10)
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    with col2:
                        # Create time series chart
                        fig = px.line(
                            fear_greed_df,
                            x='datetime',
                            y=value_col,
                            title="Fear & Greed Index History"
                        )

                        # Add colored background zones
                        fig.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=25, y1=50, fillcolor="orange", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=50, y1=75, fillcolor="lightgreen", opacity=0.1, line_width=0)
                        fig.add_hrect(y0=75, y1=100, fillcolor="green", opacity=0.1, line_width=0)

                        # Add annotations for the zones
                        fig.add_annotation(x=fear_greed_df['datetime'].min(), y=12.5, text="Extreme Fear", showarrow=False, font=dict(color="red"))
                        fig.add_annotation(x=fear_greed_df['datetime'].min(), y=37.5, text="Fear", showarrow=False, font=dict(color="orange"))
                        fig.add_annotation(x=fear_greed_df['datetime'].min(), y=62.5, text="Greed", showarrow=False, font=dict(color="green"))
                        fig.add_annotation(x=fear_greed_df['datetime'].min(), y=87.5, text="Extreme Greed", showarrow=False, font=dict(color="darkgreen"))

                        fig.update_layout(
                            xaxis_title=None,
                            yaxis_title="Fear & Greed Index",
                            yaxis=dict(range=[0, 100])
                        )

                        display_chart(apply_chart_theme(fig))
                        
                        # Set defaults in session state for backward compatibility
                        st.session_state.fear_greed_time_range = 'All'
                        st.session_state.selected_time_range = 'All'

                    # Explanation
                    st.markdown("""
                    ### Understanding the Fear & Greed Index

                    The Fear & Greed Index measures market sentiment on a scale from 0 to 100:

                    - **0-25 (Extreme Fear)**: Investors are very worried, which could represent a buying opportunity.
                    - **26-50 (Fear)**: Investors are fearful, which might indicate undervaluation.
                    - **51-75 (Greed)**: Investors are becoming greedy, potentially signaling overvaluation.
                    - **76-100 (Extreme Greed)**: Investors are excessively greedy, which might precede a market correction.

                    The index incorporates various factors including volatility, market momentum, volume, social media, and surveys.
                    """)

                    # Add price overlay
                    price_col = None
                    for col in ['price_list', 'price', 'btc_price']:
                        if col in fear_greed_df.columns:
                            price_col = col
                            break

                    if price_col is not None:
                        try:
                            st.subheader("Fear & Greed Index vs. Price")

                            # Create dual-axis chart
                            fig = make_subplots(specs=[[{"secondary_y": True}]])

                            # Add fear & greed index line
                            fig.add_trace(
                                go.Scatter(
                                    x=fear_greed_df['datetime'],
                                    y=fear_greed_df[value_col],
                                    name='Fear & Greed Index',
                                    line=dict(color='purple')
                                ),
                                secondary_y=False
                            )

                            # Add price line
                            fig.add_trace(
                                go.Scatter(
                                    x=fear_greed_df['datetime'],
                                    y=fear_greed_df[price_col],
                                    name='BTC Price',
                                    line=dict(color=ASSET_COLORS['BTC'])
                                ),
                                secondary_y=True
                            )

                            # Update layout
                            fig.update_layout(
                                title="Fear & Greed Index vs. BTC Price",
                                hovermode="x unified"
                            )

                            # Set axis titles
                            fig.update_yaxes(title_text="Fear & Greed Index", secondary_y=False, range=[0, 100])
                            fig.update_yaxes(title_text="Price (USD)", secondary_y=True)

                            display_chart(apply_chart_theme(fig))
                            
                            # Set defaults in session state for backward compatibility
                            st.session_state.fear_greed_price_time_range = 'All'
                            st.session_state.selected_time_range = 'All'
                        except Exception as e:
                            logger.error(f"Error creating dual-axis chart: {e}")
                            st.error("Could not create Fear & Greed vs Price chart. Using simple chart instead.")

                            # Fallback to simpler chart
                            try:
                                # Create a simple chart instead
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=fear_greed_df['datetime'],
                                    y=fear_greed_df[value_col],
                                    name='Fear & Greed Index'
                                ))

                                fig.update_layout(
                                    title="Fear & Greed Index",
                                    xaxis_title=None,
                                    yaxis_title="Index Value",
                                    yaxis_range=[0, 100]
                                )

                                display_chart(apply_chart_theme(fig))
                            except Exception as e2:
                                logger.error(f"Error creating fallback chart: {e2}")
                                st.error("Unable to display chart due to data format issues.")
                else:
                    st.info("No current Fear & Greed Index data available.")
            except Exception as e:
                logger.error(f"Error processing Fear & Greed Index data: {e}")
                st.error("Error displaying Fear & Greed Index")
        else:
            st.info("Fear & Greed Index data not available.")
    except Exception as e:
        logger.error(f"Error in render_fear_greed_index: {e}")
        st.error("An error occurred while rendering Fear & Greed Index")

def render_bitcoin_cycles(data, selected_time_range=None):
    """Render Bitcoin cycle indicators.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing indicators data
    selected_time_range : str, optional
        Selected time range for filtering data
    """
    st.header("Bitcoin Cycle Indicators")

    # Rainbow chart removed as requested

    # 200-week moving average
    if 'api_index_200_week_moving_average_heatmap' in data and not data['api_index_200_week_moving_average_heatmap'].empty:
        ma_df = data['api_index_200_week_moving_average_heatmap']

        # Process dataframe
        ma_df = process_timestamps(ma_df)

        st.subheader("200-Week Moving Average")

        # Create dual-line chart
        fig = go.Figure()

        # Add price line
        fig.add_trace(go.Scatter(
            x=ma_df['datetime'],
            y=ma_df['price'],
            name='BTC Price',
            line=dict(color=ASSET_COLORS['BTC'])
        ))

        # Add MA line
        if 'ma_value' in ma_df.columns:
            fig.add_trace(go.Scatter(
                x=ma_df['datetime'],
                y=ma_df['ma_value'],
                name='200-Week MA',
                line=dict(color='red', width=2)
            ))

        # Update layout
        fig.update_layout(
            title="Bitcoin Price vs. 200-Week Moving Average",
            xaxis_title=None,
            yaxis_title="Price (USD)",
            hovermode="x unified",
            yaxis_type="log"
        )

        display_chart(apply_chart_theme(fig))

        # Add ratio chart if available
        if 'price_to_ma_ratio' in ma_df.columns:
            st.subheader("Price to 200-Week MA Ratio")

            # Create ratio chart
            fig = px.line(
                ma_df,
                x='datetime',
                y='price_to_ma_ratio',
                title="Price to 200-Week MA Ratio"
            )

            # Add reference line at 1 (price equals MA)
            fig.add_hline(
                y=1,
                line_dash="dash",
                line_color="gray",
                annotation_text="Equal"
            )

            display_chart(apply_chart_theme(fig))

            # Explanation
            st.markdown("""
            ### Understanding the 200-Week Moving Average

            The 200-Week Moving Average is a key technical indicator for Bitcoin:

            - Price below the 200-Week MA has historically been an excellent buying opportunity
            - When the Price/MA ratio is below 1, Bitcoin is considered undervalued
            - The 200-Week MA has served as strong support during bear markets
            - As Bitcoin matures, the upside when price is above the 200-Week MA tends to decrease
            """)

    # Stock-to-Flow model section removed as requested

    # Other Bitcoin cycle indicators
    cycle_indicators = [
        # Pi Cycle Top Indicator removed as requested
        ("api_index_puell_multiple", "Puell Multiple"),
        ("api_index_2_year_ma_multiplier", "2-Year MA Multiplier"),
        ("api_index_golden_ratio_multiplier", "Golden Ratio Multiplier"),
        ("api_bull_market_peak_indicator", "Bull Market Peak Indicator")
    ]

    for key, title in cycle_indicators:
        try:
            if key in data and not data[key].empty:
                st.subheader(title)

                # Handle the Bull Market Peak Indicator differently as it has a different format
                if key == "api_bull_market_peak_indicator":
                    render_bull_market_peak_indicator(data[key], title)
                    continue

                # Process dataframe for regular time series indicators
                indicator_df = process_timestamps(data[key])

                # Check if datetime column exists - skip if not
                if 'datetime' not in indicator_df.columns:
                    st.info(f"{title} data available but datetime column not found. Unable to display time series chart.")
                    continue

                # Display columns for debugging if needed
                logger.info(f"Columns for {key}: {list(indicator_df.columns)}")

                # Identify indicator value column - expand the search patterns
                indicator_col = next((col for col in indicator_df.columns if
                                      'indicator' in col.lower() or
                                      'multiple' in col.lower() or
                                      'multiplier' in col.lower() or
                                      'value' in col.lower() or
                                      'index' in col.lower() or
                                      'ratio' in col.lower()), None)

                # If we couldn't find a column using patterns, use the first numeric column that isn't price
                if not indicator_col:
                    numeric_cols = [col for col in indicator_df.columns
                                    if col != 'price' and col != 'datetime'
                                    and pd.api.types.is_numeric_dtype(indicator_df[col])]
                    if numeric_cols:
                        indicator_col = numeric_cols[0]
                        logger.info(f"Using {indicator_col} as indicator column for {key}")

                if indicator_col:
                    try:
                        # Create dual-axis chart with price
                        if 'price' in indicator_df.columns:
                            # Create dual-axis chart
                            fig = make_subplots(specs=[[{"secondary_y": True}]])

                            # Add indicator line
                            fig.add_trace(
                                go.Scatter(
                                    x=indicator_df['datetime'],
                                    y=indicator_df[indicator_col],
                                    name=title,
                                    line=dict(color='purple')
                                ),
                                secondary_y=False
                            )

                            # Add price line
                            fig.add_trace(
                                go.Scatter(
                                    x=indicator_df['datetime'],
                                    y=indicator_df['price'],
                                    name='BTC Price',
                                    line=dict(color=ASSET_COLORS['BTC'])
                                ),
                                secondary_y=True
                            )

                            # Update layout
                            fig.update_layout(
                                title=f"{title} vs. BTC Price",
                                hovermode="x unified"
                            )

                            # Set axis titles
                            fig.update_yaxes(title_text=title, secondary_y=False)
                            fig.update_yaxes(title_text="Price (USD)", secondary_y=True)

                            display_chart(apply_chart_theme(fig))
                        else:
                            # Create simple indicator chart
                            fig = px.line(
                                indicator_df,
                                x='datetime',
                                y=indicator_col,
                                title=title
                            )

                            display_chart(apply_chart_theme(fig))
                    except Exception as chart_error:
                        logger.error(f"Error creating chart for {title}: {chart_error}")
                        st.error(f"Could not create chart for {title}. Error: {chart_error}")
                else:
                    st.info(f"{title} data available but indicator column not identified. Available columns: {list(indicator_df.columns)}")

        except Exception as e:
            logger.error(f"Error rendering {title} indicator: {e}")
            st.error(f"Error rendering {title} indicator. Please check the data format.")

    # Add a conclusion to the Bitcoin Cycles section
    st.markdown("""
    ### About Bitcoin Cycle Indicators

    These indicators help track Bitcoin's position in market cycles:

    - The indicators above are tools to identify potential cycle tops and bottoms
    - No single indicator is perfect - they should be used together for confirmation
    - Historical patterns may not repeat exactly in future cycles
    - Always combine technical indicators with fundamental analysis
    """)
    

def render_bull_market_peak_indicator(df, title):
    """Special renderer for the Bull Market Peak Indicator which has a different format."""
    try:
        # Check if dataframe has the expected columns for the bull market peak indicator
        expected_columns = ['indicator_name', 'current_value', 'target_value', 'previous_value',
                           'change_value', 'comparison_type', 'hit_status']

        if not all(col in df.columns for col in expected_columns):
            st.info(f"Missing expected columns in {title} data. Expected: {expected_columns}")
            return

        # Create a bar chart for the indicator values
        fig = go.Figure()

        # Sort indicators by current value (convert to numeric where possible)
        df['numeric_value'] = pd.to_numeric(df['current_value'], errors='coerce')
        df = df.sort_values('numeric_value', ascending=False)

        # Create the chart
        fig.add_trace(go.Bar(
            x=df['indicator_name'],
            y=df['numeric_value'],
            text=df['current_value'],
            textposition='auto',
            marker_color=['green' if hit else 'red' for hit in df['hit_status']],
            name='Current Value'
        ))

        # Add markers for target values where possible
        target_values = []
        for val in df['target_value']:
            try:
                target_values.append(float(val))
            except:
                target_values.append(None)

        if any(val is not None for val in target_values):
            # Add target values as a scatter plot
            fig.add_trace(go.Scatter(
                x=df['indicator_name'],
                y=target_values,
                mode='markers',
                marker=dict(color='blue', size=10, symbol='triangle-down'),
                name='Target Value'
            ))

        # Update layout
        fig.update_layout(
            title=f"{title} Indicators",
            xaxis_title=None,
            yaxis_title="Value",
            xaxis=dict(tickangle=45),
            height=600,
            margin=dict(l=50, r=50, t=50, b=100)
        )

        display_chart(apply_chart_theme(fig))


        # Explanation
        st.markdown("""
        ### Understanding the Bull Market Peak Indicator

        This indicator combines multiple metrics to help identify potential market cycle peaks:

        - **Green indicators**: Currently meeting their target criteria
        - **Red indicators**: Not meeting their target criteria
        - The more indicators that turn green, the higher the probability that we are approaching a market cycle peak
        """)

    except Exception as e:
        st.error(f"Error rendering Bull Market Peak Indicator: {e}")
        logger.error(f"Error rendering Bull Market Peak Indicator: {e}")
        st.info("Unable to render the Bull Market Peak Indicator due to data format issues.")

def render_market_metrics(data):
    """Render market metrics and indicators."""
    st.header("Market Metrics")

    try:
        # Coinbase Premium Index
        if 'api_coinbase_premium_index' in data and not data['api_coinbase_premium_index'].empty:
            try:
                premium_df = data['api_coinbase_premium_index']

                # Process dataframe
                premium_df = process_timestamps(premium_df)

                # Check for available columns and use appropriate ones
                st.subheader("Coinbase Premium Index")

                # Check if we need to rename columns based on available data
                if 'premium_rate' in premium_df.columns and 'premium_index' not in premium_df.columns:
                    # Use premium_rate as the premium_index
                    premium_df['premium_index'] = premium_df['premium_rate']

                # If premium_index is still not available, use premium as a fallback
                if 'premium_index' not in premium_df.columns and 'premium' in premium_df.columns:
                    premium_df['premium_index'] = premium_df['premium'] / 100  # Scale to be similar to a rate

                # If premium_index is still not available, show a warning
                if 'premium_index' not in premium_df.columns:
                    st.warning(f"Coinbase Premium Index data is missing expected columns. Available columns: {list(premium_df.columns)}")
                    st.dataframe(premium_df.head())  # Show data for debugging
                    return

                # Create premium chart
                fig = px.line(
                    premium_df,
                    x='datetime',
                    y='premium_index',
                    title="Coinbase Premium Index"
                )

                # Add zero line
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Zero"
                )

                display_chart(apply_chart_theme(fig))

                # Add price overlay if available
                if 'price' in premium_df.columns:
                    # Create dual-axis chart
                    fig = make_subplots(specs=[[{"secondary_y": True}]])

                    # Add premium index line
                    fig.add_trace(
                        go.Scatter(
                            x=premium_df['datetime'],
                            y=premium_df['premium_index'],
                            name='Coinbase Premium',
                            line=dict(color='purple')
                        ),
                        secondary_y=False
                    )

                    # Add price line
                    fig.add_trace(
                        go.Scatter(
                            x=premium_df['datetime'],
                            y=premium_df['price'],
                            name='BTC Price',
                            line=dict(color=ASSET_COLORS['BTC'])
                        ),
                        secondary_y=True
                    )

                    # Update layout
                    fig.update_layout(
                        title="Coinbase Premium Index vs. BTC Price",
                        hovermode="x unified"
                    )

                    # Set axis titles
                    fig.update_yaxes(title_text="Premium Index", secondary_y=False)
                    fig.update_yaxes(title_text="Price (USD)", secondary_y=True)

                    display_chart(apply_chart_theme(fig))

                    # Explanation
                    st.markdown("""
                    ### Understanding the Coinbase Premium Index

                    The Coinbase Premium Index shows the price difference between Coinbase Pro and Binance:

                    - **Positive premium**: Typically indicates strong buying pressure from US investors
                    - **Negative premium**: May indicate stronger buying in non-US markets
                    - The index is often used to gauge institutional buying interest, as Coinbase is popular among US institutions
                    """)
            except Exception as e:
                logger.error(f"Error rendering Coinbase Premium Index: {e}")
                st.error("Error displaying Coinbase Premium Index")

        # RSI
        if 'api_futures_rsi_list' in data and not data['api_futures_rsi_list'].empty:
            try:
                rsi_df = data['api_futures_rsi_list']

                # Process dataframe if needed
                if 'time' in rsi_df.columns:
                    rsi_df = process_timestamps(rsi_df)

                st.subheader("Relative Strength Index (RSI)")

                # Identify RSI columns
                rsi_cols = [col for col in rsi_df.columns if 'rsi' in col.lower()]

                if rsi_cols:
                    # Create RSI chart
                    fig = go.Figure()

                    for col in rsi_cols:
                        timeframe = col.replace('rsi_', '').upper()
                        fig.add_trace(go.Scatter(
                            x=rsi_df['datetime'] if 'datetime' in rsi_df.columns else rsi_df.index,
                            y=rsi_df[col],
                            name=f'RSI {timeframe}'
                        ))

                    # Add reference lines
                    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")

                    # Update layout
                    fig.update_layout(
                        title="Bitcoin RSI",
                        xaxis_title=None,
                        yaxis_title="RSI",
                        hovermode="x unified",
                        yaxis=dict(range=[0, 100])
                    )

                    display_chart(apply_chart_theme(fig))

                    # Explanation
                    st.markdown("""
                    ### Understanding the Relative Strength Index (RSI)

                    RSI is a momentum oscillator that measures the speed and change of price movements:

                    - **Above 70**: Considered overbought, may indicate a potential reversal down
                    - **Below 30**: Considered oversold, may indicate a potential reversal up
                    - **50 level**: Acts as a centerline, with values above 50 indicating upward momentum and below 50 indicating downward momentum
                    """)
                else:
                    st.info("RSI data available but RSI columns not identified.")
            except Exception as e:
                logger.error(f"Error rendering RSI: {e}")
                st.error("Error displaying RSI data")

        # Bitfinex Margin Long/Short
        if 'api_bitfinex_margin_long_short' in data and not data['api_bitfinex_margin_long_short'].empty:
            try:
                margin_df = data['api_bitfinex_margin_long_short'].copy()

                # Process dataframe
                margin_df = process_timestamps(margin_df)

                st.subheader("Bitfinex Margin Long/Short Ratio")

                # Check if the columns needed for calculations exist
                logger.info(f"Bitfinex Margin columns: {list(margin_df.columns)}")

                # Calculate the long/short ratio if it doesn't exist
                if 'long_short_ratio' not in margin_df.columns and 'long_quantity' in margin_df.columns and 'short_quantity' in margin_df.columns:
                    # Avoid division by zero
                    margin_df['long_short_ratio'] = margin_df['long_quantity'] / margin_df['short_quantity'].replace(0, float('nan'))
                    logger.info("Calculated long_short_ratio from long_quantity and short_quantity")

                # If we still don't have the ratio column, show the available data
                if 'long_short_ratio' not in margin_df.columns:
                    st.info("Long/short ratio data not available. Showing raw data instead.")

                    # Just display a table of what we do have
                    if not margin_df.empty:
                        st.dataframe(margin_df.head(10))

                    # For debugging
                    logger.warning(f"Missing long_short_ratio column in Bitfinex data. Available columns: {list(margin_df.columns)}")
                    return

                # Create margin chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=margin_df['datetime'],
                    y=margin_df['long_short_ratio'],
                    name='Long/Short Ratio'
                ))

                # Add reference line at 1 (equal longs and shorts)
                fig.add_hline(
                    y=1,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="Equal"
                )

                # Update layout
                fig.update_layout(
                    title="Bitfinex Margin Long/Short Ratio",
                    xaxis_title=None,
                    yaxis_title="Ratio",
                    hovermode="x unified"
                )

                display_chart(apply_chart_theme(fig))

                # Add price overlay if available
                if 'price' in margin_df.columns:
                    # Create dual-axis chart
                    fig = make_subplots(specs=[[{"secondary_y": True}]])

                    # Add long/short ratio line
                    fig.add_trace(
                        go.Scatter(
                            x=margin_df['datetime'],
                            y=margin_df['long_short_ratio'],
                            name='Long/Short Ratio',
                            line=dict(color='purple')
                        ),
                        secondary_y=False
                    )

                    # Add price line
                    fig.add_trace(
                        go.Scatter(
                            x=margin_df['datetime'],
                            y=margin_df['price'],
                            name='BTC Price',
                            line=dict(color=ASSET_COLORS['BTC'])
                        ),
                        secondary_y=True
                    )

                    # Update layout
                    fig.update_layout(
                        title="Bitfinex Margin Long/Short Ratio vs. BTC Price",
                        hovermode="x unified"
                    )

                    # Set axis titles
                    fig.update_yaxes(title_text="Long/Short Ratio", secondary_y=False)
                    fig.update_yaxes(title_text="Price (USD)", secondary_y=True)

                    display_chart(apply_chart_theme(fig))
            except Exception as e:
                logger.error(f"Error rendering Bitfinex Margin: {e}")
                st.error("Error displaying Bitfinex Margin Long/Short data")

    except Exception as e:
        logger.error(f"Error in render_market_metrics: {e}")
        st.error("An error occurred while rendering market metrics")

def render_onchain_metrics(data):
    """Render on-chain metrics and indicators."""
    st.header("On-Chain Metrics")

    try:
        # Stablecoin Market Cap
        if 'api_index_stableCoin_marketCap_history' in data and not data['api_index_stableCoin_marketCap_history'].empty:
            try:
                stablecoin_df = data['api_index_stableCoin_marketCap_history'].copy()

                # Process dataframe
                stablecoin_df = process_timestamps(stablecoin_df)

                st.subheader("Stablecoin Market Cap")

                # Check for the market cap column
                market_cap_col = None
                for col in ['market_cap_usd', 'marketCap', 'total_market_cap', 'usd_marketcap']:
                    if col in stablecoin_df.columns:
                        market_cap_col = col
                        break

                if market_cap_col is None:
                    st.warning(f"Stablecoin market cap data available but missing required column. Available columns: {list(stablecoin_df.columns)}")
                    st.dataframe(stablecoin_df.head())
                    return

                # Create market cap chart
                fig = px.line(
                    stablecoin_df,
                    x='datetime',
                    y=market_cap_col,
                    title="Stablecoin Market Cap"
                )

                fig.update_layout(
                    xaxis_title=None,
                    yaxis_title="Market Cap (USD)"
                )

                display_chart(apply_chart_theme(fig))

                # Explanation
                st.markdown("""
                ### Understanding Stablecoin Market Cap

                Stablecoin market cap represents the total value of all stablecoins in circulation:

                - Increasing stablecoin market cap often indicates growing capital on the sidelines ready to enter the crypto market
                - Stablecoins serve as an on-ramp to crypto markets and as a safe haven during volatility
                - High stablecoin market cap relative to total crypto market cap may signal potential buying power
                """)
            except Exception as e:
                logger.error(f"Error rendering Stablecoin Market Cap: {e}")
                st.error("Error displaying Stablecoin Market Cap data")
        else:
            st.info("Stablecoin Market Cap data not available")
    except Exception as e:
        logger.error(f"Error in render_onchain_metrics: {e}")
        st.error("An error occurred while rendering on-chain metrics")

def render_bitcoin_metrics(data):
    """Render Bitcoin-specific metrics and indicators."""
    st.header("Bitcoin Metrics")
    
    # Bitcoin Bubble Index
    if 'api_index_bitcoin_bubble_index' in data and not data['api_index_bitcoin_bubble_index'].empty:
        bubble_df = data['api_index_bitcoin_bubble_index']
        
        # Process dataframe
        bubble_df = process_timestamps(bubble_df)
        
        st.subheader("Bitcoin Bubble Index")
        
        # Create bubble index chart
        fig = px.line(
            bubble_df,
            x='datetime',
            y='bubble_index',
            title="Bitcoin Bubble Index"
        )
        
        # Add reference thresholds
        fig.add_hrect(y0=0, y1=0.5, fillcolor="green", opacity=0.1, line_width=0)
        fig.add_hrect(y0=0.5, y1=0.75, fillcolor="yellow", opacity=0.1, line_width=0)
        fig.add_hrect(y0=0.75, y1=1.0, fillcolor="orange", opacity=0.1, line_width=0)
        fig.add_hrect(y0=1.0, y1=2.0, fillcolor="red", opacity=0.1, line_width=0)
        
        # Add annotations for the zones
        fig.add_annotation(x=bubble_df['datetime'].min(), y=0.25, text="Normal", showarrow=False, font=dict(color="green"))
        fig.add_annotation(x=bubble_df['datetime'].min(), y=0.625, text="Caution", showarrow=False, font=dict(color="orange"))
        fig.add_annotation(x=bubble_df['datetime'].min(), y=0.875, text="Warning", showarrow=False, font=dict(color="red"))
        fig.add_annotation(x=bubble_df['datetime'].min(), y=1.5, text="Bubble Territory", showarrow=False, font=dict(color="darkred"))
        
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Bubble Index",
            yaxis=dict(range=[0, 2])
        )
        
        display_chart(apply_chart_theme(fig))
        
        # Add price overlay if available
        if 'price' in bubble_df.columns:
            # Create dual-axis chart
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add bubble index line
            fig.add_trace(
                go.Scatter(
                    x=bubble_df['datetime'],
                    y=bubble_df['bubble_index'],
                    name='Bubble Index',
                    line=dict(color='purple')
                ),
                secondary_y=False
            )
            
            # Add price line
            fig.add_trace(
                go.Scatter(
                    x=bubble_df['datetime'],
                    y=bubble_df['price'],
                    name='BTC Price',
                    line=dict(color=ASSET_COLORS['BTC'])
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title="Bitcoin Bubble Index vs. BTC Price",
                hovermode="x unified"
            )
            
            # Set axis titles
            fig.update_yaxes(title_text="Bubble Index", secondary_y=False, range=[0, 2])
            fig.update_yaxes(title_text="Price (USD)", secondary_y=True)
            
            display_chart(apply_chart_theme(fig))
    
    # Bitcoin Profitable Days
    if 'api_index_bitcoin_profitable_days' in data and not data['api_index_bitcoin_profitable_days'].empty:
        profitable_df = data['api_index_bitcoin_profitable_days']
        
        # Process dataframe
        profitable_df = process_timestamps(profitable_df)
        
        st.subheader("Bitcoin Profitable Days")
        
        # Create profitable days chart
        fig = px.line(
            profitable_df,
            x='datetime',
            y='profitable_days_percent',
            title="Bitcoin Profitable Days (%)"
        )
        
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Profitable Days (%)",
            yaxis=dict(range=[0, 100])
        )
        
        display_chart(apply_chart_theme(fig))
        
        # Explanation
        st.markdown("""
        ### Understanding Bitcoin Profitable Days
        
        The Bitcoin Profitable Days indicator shows the percentage of days where buying Bitcoin would have been profitable until today:
        
        - Higher percentage (>90%) typically indicates that Bitcoin has been a good investment for most of its history
        - Lower percentage (<80%) might indicate periods of overvaluation or major price corrections
        - This indicator offers perspective on Bitcoin's long-term performance regardless of short-term volatility
        """)

def main():
    """Main function to render the indicators page."""
    try:
        # Render sidebar
        render_sidebar()

        # Page title and description
        st.title("Market Indicators")
        st.write("Analysis of cryptocurrency market indicators and metrics")

        # Display loading message
        with st.spinner("Loading indicators data..."):
            data = load_indicators_data()

        # Get the last updated time
        # Removed data last updated reference

        # Check if data is available
        if not data:
            st.error("No indicators data available.")
            return

        # Create tabs for different indicator categories
        tab1, tab2, tab3 = st.tabs([
            "Sentiment Indicators",
            "Bitcoin Cycles",
            "Market Metrics"
        ])

        # We use try/except blocks for each tab to prevent one tab's errors from affecting others
        try:
            with tab1:
                render_fear_greed_index(data)
        except Exception as e:
            logger.error(f"Error rendering Sentiment Indicators tab: {e}")
            with tab1:
                st.error("Error displaying Sentiment Indicators. Please check the logs for details.")

        try:
            with tab2:
                render_bitcoin_cycles(data)
        except Exception as e:
            logger.error(f"Error rendering Bitcoin Cycles tab: {e}")
            with tab2:
                st.error("Error displaying Bitcoin Cycles. Please check the logs for details.")

        try:
            with tab3:
                render_market_metrics(data)
        except Exception as e:
            logger.error(f"Error rendering Market Metrics tab: {e}")
            with tab3:
                st.error("Error displaying Market Metrics. Please check the logs for details.")


    except Exception as e:
        logger.error(f"Error in main function: {e}")
        st.error("An error occurred while rendering the indicators page")

if __name__ == "__main__":
    main()