"""
Metrics display components for the Izun Crypto Liquidity Report application.
"""

import streamlit as st
import pandas as pd
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.formatters import format_currency, format_percentage, format_volume, format_delta

def display_metric_card(title, value, delta=None, delta_suffix="%", formatter=None, help_text=None):
    """
    Display a metric card with optional delta.
    
    Parameters:
    -----------
    title : str
        The title of the metric
    value : float or str
        The value to display
    delta : float, optional
        The delta value to display
    delta_suffix : str
        The suffix for the delta value (e.g., "%")
    formatter : callable, optional
        Function to format the value
    help_text : str, optional
        Help text to display on hover
    """
    # Format the value if a formatter is provided
    if formatter and value is not None:
        formatted_value = formatter(value)
    else:
        formatted_value = value
    
    # Format the delta if provided
    if delta is not None:
        # Check if delta is a string already
        if isinstance(delta, str):
            formatted_delta = delta
        else:
            # Format with + sign for positive values
            if delta > 0:
                if delta_suffix == "%":
                    formatted_delta = f"+{format_percentage(delta)}"
                else:
                    formatted_delta = f"+{delta}{delta_suffix}"
            else:
                if delta_suffix == "%":
                    formatted_delta = format_percentage(delta)
                else:
                    formatted_delta = f"{delta}{delta_suffix}"
        
        # Display metric with delta
        if help_text:
            with st.container():
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.metric(
                        label=title,
                        value=formatted_value,
                        delta=formatted_delta
                    )
                with col2:
                    st.markdown(f"<div style='margin-top: 30px'>{st.info('ℹ️')}</div>", unsafe_allow_html=True)
                    st.info(help_text)
        else:
            st.metric(
                label=title,
                value=formatted_value,
                delta=formatted_delta
            )
    else:
        # Display metric without delta
        if help_text:
            with st.container():
                col1, col2 = st.columns([0.9, 0.1])
                with col1:
                    st.metric(
                        label=title,
                        value=formatted_value
                    )
                with col2:
                    st.markdown(f"<div style='margin-top: 30px'>{st.info('ℹ️')}</div>", unsafe_allow_html=True)
                    st.info(help_text)
        else:
            st.metric(
                label=title,
                value=formatted_value
            )

def display_metrics_row(metrics_dict, formatters=None, columns=None, help_texts=None):
    """
    Display a row of metrics.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary of metrics where keys are titles and values are either:
        - Direct values to display
        - Dictionaries with 'value' and 'delta' keys
    formatters : dict, optional
        Dictionary of formatter functions for each metric
    columns : int, optional
        Number of columns to use. If None, use the number of metrics.
    help_texts : dict, optional
        Dictionary of help texts for each metric
    """
    # Determine number of columns
    num_metrics = len(metrics_dict)
    if columns is None:
        columns = num_metrics

    # Create default formatters and help texts if not provided
    if formatters is None:
        formatters = {}

    # Apply default currency formatting for metrics with currency-related names
    for key in metrics_dict.keys():
        if key not in formatters:
            is_currency = any(term in key.lower() for term in
                            ['price', 'volume', 'amount', 'usd', 'value', 'cap', 'aum', 'assets'])
            if is_currency:
                formatters[key] = lambda x: format_currency(x, show_decimals=False)

    if help_texts is None:
        help_texts = {key: None for key in metrics_dict}

    # Create columns for the metrics
    cols = st.columns(columns)

    # Display each metric in its column
    for i, (key, value) in enumerate(metrics_dict.items()):
        col_index = i % columns
        with cols[col_index]:
            if isinstance(value, dict) and 'value' in value:
                display_metric_card(
                    key,
                    value['value'],
                    value.get('delta'),
                    value.get('delta_suffix', '%'),
                    formatter=formatters.get(key),
                    help_text=help_texts.get(key)
                )
            else:
                display_metric_card(
                    key,
                    value,
                    formatter=formatters.get(key),
                    help_text=help_texts.get(key)
                )

def display_kpi_metrics(df, category, title=None, cols=3):
    """
    Display KPI metrics for a specific data category.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    category : str
        Category of the data (e.g., 'etf', 'futures')
    title : str, optional
        Title to display above the metrics
    cols : int
        Number of columns to use for the metrics
    """
    if df.empty:
        if title:
            st.subheader(title)
        st.warning("No data available for KPI metrics.")
        return

    # Optional title
    if title:
        st.subheader(title)

    # Define KPI metrics based on category
    metrics = {}
    formatters = {}

    if category == 'etf':
        if 'aum_usd' in df.columns:
            metrics['Total AUM'] = format_currency(df['aum_usd'].sum(), abbreviate=True, show_decimals=False)
        if 'fund_flow_usd' in df.columns:
            metrics['Net Flow (24h)'] = format_currency(df['fund_flow_usd'].sum(), abbreviate=True, show_decimals=False)
        if 'premium_discount_percent' in df.columns:
            metrics['Avg. Premium/Discount'] = format_percentage(df['premium_discount_percent'].mean())

    elif category == 'futures_liquidation':
        if 'aggregated_long_liquidation_usd' in df.columns and 'aggregated_short_liquidation_usd' in df.columns:
            long_liq = df['aggregated_long_liquidation_usd'].sum()
            short_liq = df['aggregated_short_liquidation_usd'].sum()
            total_liq = long_liq + short_liq

            metrics['Total Liquidations'] = format_currency(total_liq, abbreviate=True, show_decimals=False)
            metrics['Long Liquidations'] = format_currency(long_liq, abbreviate=True, show_decimals=False)
            metrics['Short Liquidations'] = format_currency(short_liq, abbreviate=True, show_decimals=False)

            # Add ratio if both are non-zero
            if long_liq > 0 and short_liq > 0:
                ratio = long_liq / short_liq
                metrics['Long/Short Ratio'] = f"{ratio:.2f}"

    elif category == 'futures_open_interest':
        if 'open_interest_usd' in df.columns:
            metrics['Total Open Interest'] = format_currency(df['open_interest_usd'].sum(), abbreviate=True, show_decimals=False)

        # Add more metrics specific to Open Interest

    elif category == 'futures_funding_rate':
        if 'funding_rate' in df.columns:
            funding_rates = pd.to_numeric(df['funding_rate'], errors='coerce')
            metrics['Average Funding Rate'] = format_percentage(funding_rates.mean())
            metrics['Max Funding Rate'] = format_percentage(funding_rates.max())
            metrics['Min Funding Rate'] = format_percentage(funding_rates.min())

    # Create columns and display metrics
    if metrics:
        columns = st.columns(cols)
        for i, (metric, value) in enumerate(metrics.items()):
            with columns[i % cols]:
                st.metric(label=metric, value=value)
    else:
        st.info("No KPI metrics available for this data category.")

def display_comparison_metrics(current_value, historical_values, title, formatter=None):
    """
    Display a metric with comparisons to historical values.
    
    Parameters:
    -----------
    current_value : float
        The current value
    historical_values : dict
        Dictionary of historical values with periods as keys (e.g., '1d', '7d', '30d')
    title : str
        The title of the metric
    formatter : callable, optional
        Function to format the values
    """
    # Format the current value
    if formatter:
        formatted_current = formatter(current_value)
    else:
        formatted_current = current_value
    
    # Display the current value
    st.metric(label=title, value=formatted_current)
    
    # Display comparisons
    cols = st.columns(len(historical_values))
    
    for i, (period, value) in enumerate(historical_values.items()):
        with cols[i]:
            # Calculate the change
            change = current_value - value
            pct_change = (change / value) * 100 if value != 0 else 0
            
            # Format the values
            if formatter:
                formatted_value = formatter(value)
            else:
                formatted_value = value
            
            # Display the comparison
            st.metric(
                label=f"{period} ago", 
                value=formatted_value,
                delta=f"{pct_change:+.2f}%"
            )

def create_metric_card_html(title, value, delta=None, delta_color=None, is_currency=False, 
                          is_percentage=False, is_large=False, help_text=None):
    """
    Create a custom HTML metric card.
    
    Parameters:
    -----------
    title : str
        The title of the metric
    value : str or float
        The value to display
    delta : str or float, optional
        The delta value to display
    delta_color : str, optional
        The color of the delta ('positive', 'negative', or 'neutral')
    is_currency : bool
        Whether the value is a currency
    is_percentage : bool
        Whether the value is a percentage
    is_large : bool
        Whether to use a large card
    help_text : str, optional
        Help text to display on hover
    
    Returns:
    --------
    str
        HTML string for the metric card
    """
    # Format the value
    if is_currency:
        if isinstance(value, (int, float)):
            formatted_value = format_currency(value, abbreviate=True)
        else:
            formatted_value = value
    elif is_percentage:
        if isinstance(value, (int, float)):
            formatted_value = format_percentage(value)
        else:
            formatted_value = value
    else:
        formatted_value = value
    
    # Format the delta
    if delta is not None:
        # Determine delta color
        if delta_color is None:
            if isinstance(delta, (int, float)):
                delta_color = 'positive' if delta > 0 else ('neutral' if delta == 0 else 'negative')
            else:
                # Try to extract the sign from the string
                delta_color = 'positive' if '+' in str(delta) else ('neutral' if '0' in str(delta) else 'negative')
        
        # Format delta with color
        delta_html = f'<div class="delta {delta_color}">{delta}</div>'
    else:
        delta_html = ''
    
    # Determine card size class
    size_class = 'large' if is_large else 'normal'
    
    # Help text tooltip
    help_html = f'<div class="help-tooltip" title="{help_text}">ℹ️</div>' if help_text else ''
    
    # Generate the HTML
    html = f"""
    <div class="metric-card {size_class}">
        <div class="title">{title} {help_html}</div>
        <div class="value">{formatted_value}</div>
        {delta_html}
    </div>
    """
    
    # Add CSS styles
    html += """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .metric-card.large {
        padding: 20px;
    }
    .metric-card .title {
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 5px;
    }
    .metric-card.large .title {
        font-size: 1.1rem;
    }
    .metric-card .value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #212529;
    }
    .metric-card.large .value {
        font-size: 2rem;
    }
    .metric-card .delta {
        font-size: 0.9rem;
        margin-top: 5px;
    }
    .metric-card .delta.positive {
        color: #28a745;
    }
    .metric-card .delta.negative {
        color: #dc3545;
    }
    .metric-card .delta.neutral {
        color: #6c757d;
    }
    .help-tooltip {
        display: inline-block;
        cursor: help;
        margin-left: 5px;
    }
    </style>
    """
    
    return html