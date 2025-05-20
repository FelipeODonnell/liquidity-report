"""
Report page for the Izun Crypto Liquidity Report.

This page serves as the main dashboard/overview page focused on liquidity metrics and intraday changes.
It provides a comprehensive view of market conditions across different assets, with emphasis on
liquidity indicators such as open interest, trading volumes, and market depth.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys
import logging
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

# Add parent directory to path to import from components and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components and utilities
from components.sidebar import render_sidebar
from utils.config import DATA_BASE_PATH, SUPPORTED_ASSETS, ASSET_COLORS, CHART_COLORS, EXCHANGE_COLORS
from components.metrics import display_metrics_row, display_metric_card
from components.charts import (
    create_time_series, 
    create_bar_chart, 
    create_time_series_with_bar,
    create_pie_chart,
    create_ohlc_chart,
    apply_chart_theme,
    display_chart,
    display_filterable_chart
)
from components.tables import create_formatted_table, create_exchange_table
from utils.data_loader import (
    get_latest_data_directory, 
    load_data_for_category, 
    process_timestamps,
    get_available_assets_for_category,
    calculate_metrics,
    get_historical_comparison
)
from utils.formatters import (
    format_currency, 
    format_percentage, 
    format_volume,
    format_timestamp,
    humanize_time_diff
)
from utils.chart_utils import create_enhanced_pie_chart, create_treemap_chart

# Set page config with title and icon
st.set_page_config(
    page_title="Izun Crypto Liquidity Report - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Store the current page in session state for sidebar navigation
st.session_state.current_page = 'report'

# Set logging level
logger.setLevel(logging.INFO)

def load_intraday_metrics():
    """
    Load and compute key intraday metrics across supported assets.
    
    Returns:
    --------
    dict
        Dictionary containing intraday metrics for each asset
    """
    metrics = {}
    
    # Get the latest data directory
    data_dir = get_latest_data_directory()
    if not data_dir:
        logger.error("No data directories found.")
        return metrics
    
    # Loop through supported assets
    for asset in SUPPORTED_ASSETS:
        asset_metrics = {}
        
        try:
            # Load futures data for the asset
            futures_oi = load_data_for_category('futures', 'open_interest', asset, data_dir)
            futures_liquidation = load_data_for_category('futures', 'liquidation', asset, data_dir)
            futures_taker = load_data_for_category('futures', 'taker_buy_sell', asset, data_dir)
            futures_funding = load_data_for_category('futures', 'funding_rate', asset, data_dir)
            
            # Load spot data for the asset
            spot_market = load_data_for_category('spot', 'spot_market', asset, data_dir)
            spot_taker = load_data_for_category('spot', 'taker_buy_sell_spot', asset, data_dir)
            
            # Process open interest data
            if futures_oi:
                # Try to get exchange list data
                oi_key = f"api_futures_openInterest_exchange_list_{asset}"
                if oi_key in futures_oi:
                    oi_df = futures_oi[oi_key]
                    
                    # Find the "All" exchange row if it exists
                    if 'exchange_name' in oi_df.columns:
                        all_row = oi_df[oi_df['exchange_name'] == 'All']
                        if not all_row.empty:
                            # Get total OI value
                            if 'open_interest_usd' in all_row.columns:
                                asset_metrics['total_oi'] = all_row['open_interest_usd'].iloc[0]
                            
                            # Get OI change metrics
                            for period in ['1h', '4h', '24h', '7d', '14d', '30d']:
                                change_col = f'change_{period}_percent'
                                if change_col in all_row.columns:
                                    asset_metrics[f'oi_change_{period}'] = all_row[change_col].iloc[0]
                    
                # Try to get time series OI data for 24h high/low
                oi_history_key = f"api_futures_openInterest_ohlc_aggregated_history_{asset}"
                if oi_history_key in futures_oi:
                    oi_history = futures_oi[oi_history_key]
                    oi_history = process_timestamps(oi_history)
                    
                    # Filter to last 24 hours
                    if 'datetime' in oi_history.columns:
                        last_24h = datetime.now() - timedelta(days=1)
                        oi_24h = oi_history[oi_history['datetime'] >= last_24h]
                        
                        if not oi_24h.empty and 'high' in oi_24h.columns and 'low' in oi_24h.columns:
                            asset_metrics['oi_24h_high'] = oi_24h['high'].max()
                            asset_metrics['oi_24h_low'] = oi_24h['low'].min()
            
            # Process liquidation data
            if futures_liquidation:
                # Try to get aggregated liquidation history
                liq_key = f"api_futures_liquidation_aggregated_coin_history_{asset}"
                if liq_key in futures_liquidation:
                    liq_df = futures_liquidation[liq_key]
                    liq_df = process_timestamps(liq_df)
                    
                    # Filter to last 24 hours
                    if 'datetime' in liq_df.columns:
                        last_24h = datetime.now() - timedelta(days=1)
                        liq_24h = liq_df[liq_df['datetime'] >= last_24h]
                        
                        # Get long and short liquidations
                        if not liq_24h.empty:
                            if 'long_liquidation_usd' in liq_24h.columns:
                                asset_metrics['long_liquidation_24h'] = liq_24h['long_liquidation_usd'].sum()
                            if 'short_liquidation_usd' in liq_24h.columns:
                                asset_metrics['short_liquidation_24h'] = liq_24h['short_liquidation_usd'].sum()
            
            # Process taker buy/sell data
            if futures_taker:
                # Try to get aggregated taker data
                taker_key = f"api_futures_taker_buy_sell_volume_history_{asset}"
                if taker_key in futures_taker:
                    taker_df = futures_taker[taker_key]
                    taker_df = process_timestamps(taker_df)
                    
                    # Filter to last 24 hours
                    if 'datetime' in taker_df.columns:
                        last_24h = datetime.now() - timedelta(days=1)
                        taker_24h = taker_df[taker_df['datetime'] >= last_24h]
                        
                        # Get buy and sell volumes
                        if not taker_24h.empty:
                            # Check for various column name patterns
                            buy_cols = ['buy_volume', 'buy_vol', 'taker_buy_volume']
                            sell_cols = ['sell_volume', 'sell_vol', 'taker_sell_volume']
                            
                            # Find the first matching column
                            buy_col = next((col for col in buy_cols if col in taker_24h.columns), None)
                            sell_col = next((col for col in sell_cols if col in taker_24h.columns), None)
                            
                            if buy_col and sell_col:
                                asset_metrics['taker_buy_volume_24h'] = taker_24h[buy_col].sum()
                                asset_metrics['taker_sell_volume_24h'] = taker_24h[sell_col].sum()
                                
                                # Calculate ratio
                                if asset_metrics['taker_sell_volume_24h'] > 0:
                                    asset_metrics['taker_buy_sell_ratio_24h'] = (
                                        asset_metrics['taker_buy_volume_24h'] / 
                                        asset_metrics['taker_sell_volume_24h']
                                    )
            
            # Process funding rate data
            if futures_funding:
                # Try to get aggregated funding rate data
                funding_key = "api_futures_fundingRate_ohlc_history"
                if funding_key in futures_funding:
                    funding_df = futures_funding[funding_key]
                    funding_df = process_timestamps(funding_df)
                    
                    # Filter to last 24 hours
                    if 'datetime' in funding_df.columns:
                        last_24h = datetime.now() - timedelta(days=1)
                        funding_24h = funding_df[funding_df['datetime'] >= last_24h]
                        
                        # Get average funding rate
                        if not funding_24h.empty and 'close' in funding_24h.columns:
                            asset_metrics['avg_funding_rate_24h'] = funding_24h['close'].mean()
            
            # Process spot market data to get price and volume
            if spot_market:
                # Look for price data
                price_keys = [
                    f"api_spot_pairs_markets_{asset}",
                    f"api_spot_price_history_{asset}"
                ]
                
                for key in price_keys:
                    if key in spot_market and not spot_market[key].empty:
                        price_df = spot_market[key]
                        
                        # Check for price column
                        price_cols = ['price_usd', 'price', 'close', 'last']
                        price_col = next((col for col in price_cols if col in price_df.columns), None)
                        
                        if price_col:
                            # Get current price (assuming the first row is the most recent)
                            asset_metrics['current_price'] = price_df[price_col].iloc[0]
                            
                            # Get price change if available
                            change_cols = ['price_change_percentage_24h', 'change_24h', 'change_percentage_24h']
                            change_col = next((col for col in change_cols if col in price_df.columns), None)
                            
                            if change_col:
                                asset_metrics['price_change_24h'] = price_df[change_col].iloc[0]
                        
                        # Get volume data if available
                        volume_cols = ['volume_24h_usd', 'volume_usd', 'volume']
                        volume_col = next((col for col in volume_cols if col in price_df.columns), None)
                        
                        if volume_col:
                            asset_metrics['spot_volume_24h'] = price_df[volume_col].iloc[0]
                        
                        break
            
            metrics[asset] = asset_metrics
            
        except Exception as e:
            logger.error(f"Error loading intraday metrics for {asset}: {e}")
            # Continue with next asset
            continue
    
    return metrics

def create_key_metrics_table(metrics):
    """
    Create a formatted table with key metrics for all assets.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics for each asset
    
    Returns:
    --------
    pandas.DataFrame
        Formatted metrics table
    """
    table_data = []
    
    # Define the metrics to include in the table
    display_metrics = [
        {'key': 'current_price', 'name': 'Price (USD)', 'format': lambda x: format_currency(x, precision=2)},
        {'key': 'price_change_24h', 'name': 'Price Change (24h)', 'format': lambda x: format_percentage(x, show_plus=True)},
        {'key': 'total_oi', 'name': 'Open Interest', 'format': lambda x: format_currency(x, abbreviate=True)},
        {'key': 'oi_change_24h', 'name': 'OI Change (24h)', 'format': lambda x: format_percentage(x, show_plus=True)},
        {'key': 'taker_buy_sell_ratio_24h', 'name': 'Buy/Sell Ratio', 'format': lambda x: f"{x:.2f}"},
        {'key': 'avg_funding_rate_24h', 'name': 'Funding Rate (avg)', 'format': lambda x: format_percentage(x, show_plus=True)},
        {'key': 'long_liquidation_24h', 'name': 'Long Liquidations', 'format': lambda x: format_currency(x, abbreviate=True)},
        {'key': 'short_liquidation_24h', 'name': 'Short Liquidations', 'format': lambda x: format_currency(x, abbreviate=True)}
    ]
    
    # Create rows for each asset
    for asset in SUPPORTED_ASSETS:
        if asset in metrics:
            row = {'Asset': asset}
            
            # Add metrics to row
            for metric in display_metrics:
                key = metric['key']
                if key in metrics[asset]:
                    # Format the value
                    value = metrics[asset][key]
                    row[metric['name']] = value
                else:
                    row[metric['name']] = None
            
            table_data.append(row)
    
    # Create DataFrame
    if table_data:
        df = pd.DataFrame(table_data)
        return df
    else:
        return pd.DataFrame(columns=['Asset'] + [m['name'] for m in display_metrics])

def calculate_market_health_score(metrics, asset):
    """
    Calculate a market health score for the given asset based on various metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary containing metrics for all assets
    asset : str
        Asset symbol to calculate score for
    
    Returns:
    --------
    float
        Market health score (0-100)
    dict
        Component scores and contributing factors
    """
    if asset not in metrics:
        return 50, {'overall': 50}  # Default neutral score
    
    asset_metrics = metrics[asset]
    component_scores = {}
    
    # 1. Price momentum (20% weight)
    price_score = 50  # Neutral default
    if 'price_change_24h' in asset_metrics:
        price_change = asset_metrics['price_change_24h']
        # Map price change to 0-100 score
        # Positive price change → higher score
        if price_change > 10:
            price_score = 100
        elif price_change > 5:
            price_score = 90
        elif price_change > 2:
            price_score = 80
        elif price_change > 0:
            price_score = 65
        elif price_change > -2:
            price_score = 45
        elif price_change > -5:
            price_score = 30
        elif price_change > -10:
            price_score = 20
        else:
            price_score = 10
    component_scores['price_momentum'] = price_score
    
    # 2. Open Interest health (20% weight)
    oi_score = 50  # Neutral default
    if 'oi_change_24h' in asset_metrics:
        oi_change = asset_metrics['oi_change_24h']
        # Map OI change to 0-100 score
        # Moderate OI growth is healthy, extreme growth may indicate a bubble
        if 1 < oi_change <= 5:
            oi_score = 90  # Healthy growth
        elif 0 < oi_change <= 1:
            oi_score = 70  # Slight growth
        elif -1 <= oi_change <= 0:
            oi_score = 50  # Slight decline
        elif -5 <= oi_change < -1:
            oi_score = 30  # Moderate decline
        elif oi_change > 10:
            oi_score = 40  # Too rapid growth - might indicate excessive leverage
        elif oi_change < -5:
            oi_score = 20  # Sharp decline - might indicate market exit
        else:
            oi_score = 60  # Other scenarios
    component_scores['open_interest'] = oi_score
    
    # 3. Buy/Sell balance (20% weight)
    buysell_score = 50  # Neutral default
    if 'taker_buy_sell_ratio_24h' in asset_metrics:
        ratio = asset_metrics['taker_buy_sell_ratio_24h']
        # Map buy/sell ratio to 0-100 score
        # Ratio around 1.0 is balanced, higher ratio indicates more buying pressure
        if 0.95 <= ratio <= 1.05:
            buysell_score = 50  # Balanced market
        elif 1.05 < ratio <= 1.2:
            buysell_score = 70  # Moderate buying pressure
        elif 1.2 < ratio <= 1.5:
            buysell_score = 85  # Strong buying pressure
        elif ratio > 1.5:
            buysell_score = 95  # Very strong buying pressure
        elif 0.8 <= ratio < 0.95:
            buysell_score = 40  # Moderate selling pressure
        elif 0.5 <= ratio < 0.8:
            buysell_score = 25  # Strong selling pressure
        elif ratio < 0.5:
            buysell_score = 10  # Very strong selling pressure
    component_scores['buy_sell_balance'] = buysell_score
    
    # 4. Funding rate health (20% weight)
    funding_score = 50  # Neutral default
    if 'avg_funding_rate_24h' in asset_metrics:
        funding_rate = asset_metrics['avg_funding_rate_24h']
        # Map funding rate to 0-100 score
        # Moderate positive funding is healthy, extreme values indicate imbalance
        if -0.01 <= funding_rate <= 0.01:
            funding_score = 70  # Balanced funding
        elif 0.01 < funding_rate <= 0.03:
            funding_score = 60  # Slightly positive
        elif 0.03 < funding_rate <= 0.1:
            funding_score = 40  # Moderately high funding
        elif funding_rate > 0.1:
            funding_score = 20  # Very high funding - potential reversal signal
        elif -0.03 <= funding_rate < -0.01:
            funding_score = 60  # Slightly negative
        elif -0.1 <= funding_rate < -0.03:
            funding_score = 40  # Moderately negative funding
        elif funding_rate < -0.1:
            funding_score = 20  # Very negative funding - potential reversal signal
    component_scores['funding_rate'] = funding_score
    
    # 5. Liquidation risk (20% weight)
    liquidation_score = 50  # Neutral default
    if 'long_liquidation_24h' in asset_metrics and 'short_liquidation_24h' in asset_metrics:
        long_liq = asset_metrics['long_liquidation_24h']
        short_liq = asset_metrics['short_liquidation_24h']
        total_liq = long_liq + short_liq
        
        # Try to get OI for relative comparison
        oi = asset_metrics.get('total_oi', 0)
        
        # Calculate liquidation as percentage of OI
        liq_percent = (total_liq / oi * 100) if oi > 0 else 0
        
        # Map liquidation percentage to score
        if liq_percent < 1:
            liquidation_score = 90  # Very low liquidations
        elif 1 <= liq_percent < 3:
            liquidation_score = 70  # Low liquidations
        elif 3 <= liq_percent < 5:
            liquidation_score = 50  # Moderate liquidations
        elif 5 <= liq_percent < 10:
            liquidation_score = 30  # High liquidations
        elif liq_percent >= 10:
            liquidation_score = 10  # Very high liquidations
    component_scores['liquidation_risk'] = liquidation_score
    
    # Calculate weighted average for overall score
    weights = {
        'price_momentum': 0.2,
        'open_interest': 0.2,
        'buy_sell_balance': 0.2,
        'funding_rate': 0.2,
        'liquidation_risk': 0.2
    }
    
    overall_score = sum(score * weights[component] for component, score in component_scores.items())
    component_scores['overall'] = overall_score
    
    return overall_score, component_scores

def format_health_score(score):
    """
    Format health score with color and description.
    
    Parameters:
    -----------
    score : float
        Health score (0-100)
    
    Returns:
    --------
    str
        Description of health
    str
        Color code for the score
    """
    if score >= 80:
        return "Very Healthy", "#1c7c54"  # Green
    elif score >= 65:
        return "Healthy", "#4CAF50"  # Light green
    elif score >= 45:
        return "Neutral", "#FFC107"  # Yellow
    elif score >= 30:
        return "Weak", "#FF9800"  # Orange
    else:
        return "Unhealthy", "#F44336"  # Red

def create_market_health_gauge(score, title="Market Health"):
    """
    Create a gauge chart for market health.
    
    Parameters:
    -----------
    score : float
        Health score (0-100)
    title : str
        Chart title
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Gauge chart
    """
    # Get description and color
    description, color = format_health_score(score)
    
    # Create gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{title}<br><span style='font-size:0.8em;'>{description}</span>"},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 30], 'color': 'rgba(244, 67, 54, 0.3)'},  # Red
                {'range': [30, 45], 'color': 'rgba(255, 152, 0, 0.3)'},  # Orange
                {'range': [45, 65], 'color': 'rgba(255, 193, 7, 0.3)'},  # Yellow
                {'range': [65, 80], 'color': 'rgba(76, 175, 80, 0.3)'},  # Light green
                {'range': [80, 100], 'color': 'rgba(28, 124, 84, 0.3)'}  # Green
            ],
            'threshold': {
                'line': {'color': 'white', 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    # Update layout
    fig.update_layout(
        height=200,
        margin=dict(l=30, r=30, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    
    return fig

def render_market_health_section(metrics):
    """
    Render the market health section with indicators and scores.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of intraday metrics for all assets
    """
    st.subheader("Market Health Overview")
    
    # Create columns for each asset
    cols = st.columns(len(SUPPORTED_ASSETS))
    
    # Add health indicators for each asset
    for i, asset in enumerate(SUPPORTED_ASSETS):
        with cols[i]:
            if asset in metrics:
                # Calculate health score
                score, components = calculate_market_health_score(metrics, asset)
                
                # Display asset name and score
                st.markdown(f"### {asset}")
                
                # Create gauge chart
                gauge_fig = create_market_health_gauge(score, f"{asset} Health")
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Add component scores in a compact format
                if components:
                    # Format component labels
                    component_labels = {
                        'price_momentum': 'Price Momentum',
                        'open_interest': 'Open Interest',
                        'buy_sell_balance': 'Buy/Sell Balance',
                        'funding_rate': 'Funding Rate',
                        'liquidation_risk': 'Liquidation Risk'
                    }
                    
                    # Display components in a compact table
                    component_data = []
                    for component, component_score in components.items():
                        if component != 'overall':
                            description, color = format_health_score(component_score)
                            component_data.append({
                                'Component': component_labels.get(component, component),
                                'Score': f"{component_score:.0f}/100",
                                'Status': description
                            })
                    
                    if component_data:
                        component_df = pd.DataFrame(component_data)
                        st.dataframe(component_df, hide_index=True, use_container_width=True)
            else:
                st.markdown(f"### {asset}")
                st.info("No data available for health calculation")

def render_intraday_changes(metrics):
    """
    Render the intraday changes section showing price and OI changes.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of intraday metrics for all assets
    """
    st.subheader("Intraday Changes")
    
    # Define time periods to display
    periods = ['1h', '4h', '24h']
    
    # Create columns for each metric type
    metric_types = ["Price Change %", "Open Interest Change %"]
    
    # Create a formatted table
    table_data = []
    
    # Add data for each asset
    for asset in SUPPORTED_ASSETS:
        if asset in metrics:
            asset_metrics = metrics[asset]
            row = {'Asset': asset}
            
            # Add price change metrics
            for period in periods:
                price_key = f'price_change_{period}'
                if price_key in asset_metrics:
                    row[f'Price {period}'] = asset_metrics[price_key]
                else:
                    row[f'Price {period}'] = None
            
            # Add OI change metrics
            for period in periods:
                oi_key = f'oi_change_{period}'
                if oi_key in asset_metrics:
                    row[f'OI {period}'] = asset_metrics[oi_key]
                else:
                    row[f'OI {period}'] = None
            
            table_data.append(row)
    
    # Create DataFrame
    if table_data:
        df = pd.DataFrame(table_data)
        
        # Formatting function for numeric columns
        def format_change(val):
            if pd.isna(val):
                return ""
            color = "green" if val > 0 else "red" if val < 0 else "white"
            sign = "+" if val > 0 else ""
            return f"<span style='color:{color}'>{sign}{val:.2f}%</span>"
        
        # Create styled table
        if len(df) > 0:
            # Create two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Price Changes")
                # Select only price columns
                price_cols = ['Asset'] + [f'Price {p}' for p in periods]
                price_df = df[price_cols]
                
                # Apply styling
                st.write(price_df.style.format({
                    col: format_change for col in price_cols if col != 'Asset'
                }).to_html(), unsafe_allow_html=True)
            
            with col2:
                st.markdown("#### Open Interest Changes")
                # Select only OI columns
                oi_cols = ['Asset'] + [f'OI {p}' for p in periods]
                oi_df = df[oi_cols]
                
                # Apply styling
                st.write(oi_df.style.format({
                    col: format_change for col in oi_cols if col != 'Asset'
                }).to_html(), unsafe_allow_html=True)
    else:
        st.info("No intraday change data available")

def render_futures_liquidity_section(metrics, selected_asset='BTC'):
    """
    Render the futures liquidity section with key metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of intraday metrics for all assets
    selected_asset : str
        Selected asset to display detailed metrics for
    """
    st.subheader(f"Futures Market Liquidity ({selected_asset})")
    
    if selected_asset in metrics:
        asset_metrics = metrics[selected_asset]
        
        # Create three columns for key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Open Interest
            if 'total_oi' in asset_metrics:
                oi_value = asset_metrics['total_oi']
                oi_change = asset_metrics.get('oi_change_24h', 0)
                
                display_metric_card(
                    "Total Open Interest",
                    oi_value,
                    oi_change,
                    is_currency=True,
                    delta_is_percent=True,
                    abbreviate=True
                )
            else:
                st.metric("Total Open Interest", "N/A")
        
        with col2:
            # Taker Buy/Sell Ratio
            if 'taker_buy_sell_ratio_24h' in asset_metrics:
                ratio = asset_metrics['taker_buy_sell_ratio_24h']
                # Compare to neutral 1.0 for delta
                ratio_delta = (ratio - 1.0) * 100  # Convert to percentage difference from 1.0
                
                display_metric_card(
                    "Taker Buy/Sell Ratio (24h)",
                    ratio,
                    ratio_delta,
                    is_currency=False,
                    delta_is_percent=True,
                    formatter=lambda x: f"{x:.2f}",
                    delta_formatter=lambda x: f"{x:+.2f}%"
                )
            else:
                st.metric("Taker Buy/Sell Ratio (24h)", "N/A")
        
        with col3:
            # Funding Rate
            if 'avg_funding_rate_24h' in asset_metrics:
                funding_rate = asset_metrics['avg_funding_rate_24h']
                
                display_metric_card(
                    "Average Funding Rate (24h)",
                    funding_rate,
                    None,
                    is_currency=False,
                    formatter=lambda x: format_percentage(x, precision=4, show_plus=True)
                )
            else:
                st.metric("Average Funding Rate (24h)", "N/A")
        
        # Add additional metrics in a second row
        col4, col5, col6 = st.columns(3)
        
        with col4:
            # Long Liquidations
            if 'long_liquidation_24h' in asset_metrics:
                long_liq = asset_metrics['long_liquidation_24h']
                
                display_metric_card(
                    "Long Liquidations (24h)",
                    long_liq,
                    None,
                    is_currency=True,
                    abbreviate=True
                )
            else:
                st.metric("Long Liquidations (24h)", "N/A")
        
        with col5:
            # Short Liquidations
            if 'short_liquidation_24h' in asset_metrics:
                short_liq = asset_metrics['short_liquidation_24h']
                
                display_metric_card(
                    "Short Liquidations (24h)",
                    short_liq,
                    None,
                    is_currency=True,
                    abbreviate=True
                )
            else:
                st.metric("Short Liquidations (24h)", "N/A")
        
        with col6:
            # Liquidation Ratio
            if 'long_liquidation_24h' in asset_metrics and 'short_liquidation_24h' in asset_metrics:
                long_liq = asset_metrics['long_liquidation_24h']
                short_liq = asset_metrics['short_liquidation_24h']
                
                # Calculate ratio (handle division by zero)
                if short_liq > 0:
                    liq_ratio = long_liq / short_liq
                else:
                    liq_ratio = float('inf') if long_liq > 0 else 0
                
                # Format ratio for display
                if liq_ratio == float('inf'):
                    liq_ratio_display = "∞"
                else:
                    liq_ratio_display = f"{liq_ratio:.2f}"
                
                st.metric("Long/Short Liquidation Ratio", liq_ratio_display)
            else:
                st.metric("Long/Short Liquidation Ratio", "N/A")
        
        # Create expander for visualizations
        with st.expander("Liquidity Visualizations", expanded=True):
            # Load specific liquidity data
            try:
                # Get the latest data directory
                data_dir = get_latest_data_directory()
                
                # Load OI by exchange data
                oi_data = load_data_for_category('futures', 'open_interest', selected_asset, data_dir)
                oi_exchange_key = f"api_futures_openInterest_exchange_list_{selected_asset}"
                
                if oi_data and oi_exchange_key in oi_data:
                    oi_df = oi_data[oi_exchange_key]
                    
                    # Check if data has required columns
                    if 'exchange_name' in oi_df.columns and 'open_interest_usd' in oi_df.columns:
                        # Remove 'All' row
                        oi_df = oi_df[oi_df['exchange_name'] != 'All']
                        
                        # Sort by open interest value
                        oi_df = oi_df.sort_values('open_interest_usd', ascending=False)
                        
                        # Create bar chart showing OI by exchange
                        oi_fig = create_bar_chart(
                            oi_df,
                            'exchange_name',
                            'open_interest_usd',
                            f"{selected_asset} Open Interest by Exchange",
                            color_discrete_map=EXCHANGE_COLORS
                        )
                        
                        st.plotly_chart(oi_fig, use_container_width=True)
                        
                        # Calculate OI by margin type
                        if 'stablecoin_open_interest_usd' in oi_df.columns and 'coin_open_interest_usd' in oi_df.columns:
                            # Sum OI by margin type
                            stablecoin_oi = oi_df['stablecoin_open_interest_usd'].sum()
                            coin_oi = oi_df['coin_open_interest_usd'].sum()
                            
                            # Create pie chart
                            margin_data = pd.DataFrame({
                                'margin_type': ['Stablecoin-Margined', 'Coin-Margined'],
                                'open_interest_usd': [stablecoin_oi, coin_oi]
                            })
                            
                            margin_fig = create_pie_chart(
                                margin_data,
                                'open_interest_usd',
                                'margin_type',
                                f"{selected_asset} Open Interest by Margin Type"
                            )
                            
                            st.plotly_chart(margin_fig, use_container_width=True)
                
                # Load liquidation data
                liquidation_data = load_data_for_category('futures', 'liquidation', selected_asset, data_dir)
                liq_history_key = f"api_futures_liquidation_aggregated_coin_history_{selected_asset}"
                
                if liquidation_data and liq_history_key in liquidation_data:
                    liq_df = liquidation_data[liq_history_key]
                    liq_df = process_timestamps(liq_df)
                    
                    # Create liquidation chart
                    if 'datetime' in liq_df.columns and 'long_liquidation_usd' in liq_df.columns and 'short_liquidation_usd' in liq_df.columns:
                        # Filter to last 7 days
                        last_7d = datetime.now() - timedelta(days=7)
                        liq_7d = liq_df[liq_df['datetime'] >= last_7d]
                        
                        # Create stacked bar chart
                        liq_fig = go.Figure()
                        
                        # Add long liquidations
                        liq_fig.add_trace(go.Bar(
                            x=liq_7d['datetime'],
                            y=liq_7d['long_liquidation_usd'],
                            name='Long Liquidations',
                            marker_color='red'
                        ))
                        
                        # Add short liquidations
                        liq_fig.add_trace(go.Bar(
                            x=liq_7d['datetime'],
                            y=liq_7d['short_liquidation_usd'],
                            name='Short Liquidations',
                            marker_color='green'
                        ))
                        
                        # Update layout
                        liq_fig.update_layout(
                            title=f"{selected_asset} Liquidations (Last 7 Days)",
                            barmode='group',
                            xaxis_title=None,
                            yaxis_title="Liquidation Amount (USD)",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        # Apply theme
                        liq_fig = apply_chart_theme(liq_fig)
                        
                        st.plotly_chart(liq_fig, use_container_width=True)
            
            except Exception as e:
                logger.error(f"Error rendering liquidity visualizations: {e}")
                st.warning("Error loading liquidity visualizations")
    else:
        st.info(f"No liquidity data available for {selected_asset}")

def render_market_concentration_section(metrics):
    """
    Render the market concentration section showing distribution of liquidity.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of intraday metrics for all assets
    """
    st.subheader("Market Concentration")
    
    # Create OI comparison chart
    oi_data = []
    for asset in SUPPORTED_ASSETS:
        if asset in metrics and 'total_oi' in metrics[asset]:
            oi_data.append({
                'Asset': asset,
                'Open Interest': metrics[asset]['total_oi']
            })
    
    if oi_data:
        oi_df = pd.DataFrame(oi_data)
        
        # Create bar chart for OI
        oi_fig = go.Figure(go.Bar(
            x=oi_df['Asset'],
            y=oi_df['Open Interest'],
            text=oi_df['Open Interest'].apply(lambda x: format_currency(x, abbreviate=True)),
            textposition='auto',
            marker_color=[ASSET_COLORS.get(asset, '#3366CC') for asset in oi_df['Asset']]
        ))
        
        # Update layout
        oi_fig.update_layout(
            title="Open Interest Comparison",
            xaxis_title=None,
            yaxis_title="Open Interest (USD)",
            yaxis=dict(
                tickformat=",",
                tickprefix="$"
            )
        )
        
        # Apply theme
        oi_fig = apply_chart_theme(oi_fig)
        
        # Add the chart
        st.plotly_chart(oi_fig, use_container_width=True)
    
    # Try to load exchange-specific data
    try:
        # Get the latest data directory
        data_dir = get_latest_data_directory()
        
        # Choose the first asset with available data
        for asset in SUPPORTED_ASSETS:
            # Load OI by exchange data
            oi_data = load_data_for_category('futures', 'open_interest', asset, data_dir)
            oi_exchange_key = f"api_futures_openInterest_exchange_list_{asset}"
            
            if oi_data and oi_exchange_key in oi_data:
                oi_df = oi_data[oi_exchange_key]
                
                # Check if data has required columns
                if 'exchange_name' in oi_df.columns and 'open_interest_usd' in oi_df.columns:
                    # Remove 'All' row
                    oi_df = oi_df[oi_df['exchange_name'] != 'All']
                    
                    # Sort by open interest value
                    oi_df = oi_df.sort_values('open_interest_usd', ascending=False)
                    
                    # Create treemap showing market concentration
                    treemap_data = []
                    for i, row in oi_df.iterrows():
                        treemap_data.append({
                            'Exchange': row['exchange_name'],
                            'Asset': asset,
                            'Open Interest': row['open_interest_usd']
                        })
                    
                    treemap_df = pd.DataFrame(treemap_data)
                    
                    # Create treemap
                    if not treemap_df.empty:
                        treemap_fig = create_treemap_chart(
                            treemap_df,
                            'Open Interest',
                            'Exchange',
                            'Asset',
                            f"{asset} Open Interest by Exchange"
                        )
                        
                        st.plotly_chart(treemap_fig, use_container_width=True)
                    
                    break  # Only show for one asset
    
    except Exception as e:
        logger.error(f"Error rendering market concentration: {e}")
        st.warning("Error loading market concentration data")

def main():
    """Main function to render the report page."""
    
    # Render sidebar
    render_sidebar()
    
    # Page title
    st.title("Overview")
    
    # Add last updated info
    last_updated = get_latest_data_directory()
    if last_updated:
        try:
            last_updated_date = datetime.strptime(last_updated, "%Y%m%d")
            st.caption(f"Data last updated: {last_updated_date.strftime('%Y-%m-%d')}")
        except:
            st.caption(f"Data last updated: {last_updated}")
    
    # Load intraday metrics
    metrics = load_intraday_metrics()
    
    if not metrics:
        st.error("No data available. Please check the data directory.")
        return
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs([
        "Market Overview", 
        "Market Health", 
        "Futures Liquidity", 
        "Market Concentration"
    ])
    
    with tab1:
        # Overview tab
        # Display key metrics table
        metrics_df = create_key_metrics_table(metrics)
        
        # Add conditional formatting
        def highlight_positive(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return 'color: green'
                elif val < 0:
                    return 'color: red'
            return ''
        
        # Apply formatting and show table
        if not metrics_df.empty:
            # Create formatted dictionary for numeric columns
            format_dict = {
                'Price (USD)': lambda x: format_currency(x, precision=2) if pd.notna(x) else 'N/A',
                'Price Change (24h)': lambda x: format_percentage(x, show_plus=True) if pd.notna(x) else 'N/A',
                'Open Interest': lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else 'N/A',
                'OI Change (24h)': lambda x: format_percentage(x, show_plus=True) if pd.notna(x) else 'N/A',
                'Buy/Sell Ratio': lambda x: f"{x:.2f}" if pd.notna(x) else 'N/A',
                'Funding Rate (avg)': lambda x: format_percentage(x, show_plus=True, precision=4) if pd.notna(x) else 'N/A',
                'Long Liquidations': lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else 'N/A',
                'Short Liquidations': lambda x: format_currency(x, abbreviate=True) if pd.notna(x) else 'N/A'
            }
            
            # Display table with formatting
            create_formatted_table(metrics_df, format_dict=format_dict)
        else:
            st.info("No metrics data available")
        
        # Add intraday changes section
        render_intraday_changes(metrics)
    
    with tab2:
        # Market Health tab
        render_market_health_section(metrics)
    
    with tab3:
        # Futures Liquidity tab
        # Add asset selector
        selected_asset = st.selectbox(
            "Select Asset",
            SUPPORTED_ASSETS,
            index=0,  # Default to BTC
            key="futures_liquidity_asset_selector"
        )
        
        # Update current asset in session state
        st.session_state.selected_asset = selected_asset
        
        # Render liquidity section for selected asset
        render_futures_liquidity_section(metrics, selected_asset)
    
    with tab4:
        # Market Concentration tab
        render_market_concentration_section(metrics)

if __name__ == "__main__":
    main()