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

def display_metric_card(title, value, delta=None, delta_suffix="%", formatter=None, help_text=None, delta_formatter=None, 
                      compact=False, color_delta=True, debug_info=None, custom_css=None, hide_label=False):
    """
    Display a metric card with optional delta, improved formatting.
    
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
    delta_formatter : callable, optional
        Function to format the delta value (overrides default)
    compact : bool
        Whether to use a more compact display format
    color_delta : bool
        Whether to color the delta (green for positive, red for negative)
    debug_info : str, optional
        Debug information to display in dev mode
    custom_css : str, optional
        Custom CSS to apply to the metric card
    hide_label : bool
        Whether to hide the label/title (useful for compact displays)
    """
    # Format the value based on the type
    if value is None:
        formatted_value = "N/A"
    elif formatter and callable(formatter):
        formatted_value = formatter(value)
    elif isinstance(value, (int, float)):
        # Auto-detect if it looks like a currency by checking the title
        title_lower = title.lower()
        is_currency = any(term in title_lower for term in 
                         ['price', 'volume', 'amount', 'usd', 'value', 'cap', 'aum', 'assets', 'fund', 'flow', 
                          'liquidation', 'interest', 'depth', 'bid', 'ask', 'spread', 'cost', 'profit',
                          'balance', 'market', 'trade']) or title_lower.endswith('usd')
        
        if is_currency:
            # Format as currency without decimals for better headline stats
            if abs(value) < 0.01 and value != 0:
                # Special handling for very small currency values
                formatted_value = format_currency(value, precision=6, show_decimals=True, compact=compact, strip_zeros=True)
            elif abs(value) < 1 and value != 0:
                # Special handling for small currency values
                formatted_value = format_currency(value, precision=4, show_decimals=True, compact=compact, strip_zeros=True)
            elif abs(value) < 10:
                # Special handling for currency values under 10
                formatted_value = format_currency(value, precision=2, show_decimals=True, compact=compact)
            else:
                # Format larger currency values without decimals
                formatted_value = format_currency(value, show_decimals=False, compact=compact, abbreviate=compact)
        else:
            # See if it might be a percentage
            is_percentage = any(term in title_lower for term in 
                              ['percent', 'change', 'rate', 'ratio', 'growth', '%', 'pct', 'fee', 'premium', 'discount',
                               'yield', 'return', 'volatility', 'margin']) or title_lower.endswith('%') or title_lower.endswith('pct')
            
            if is_percentage:
                # Format percentage based on magnitude
                if abs(value) < 0.01 and value != 0:
                    # Very small percentage
                    formatted_value = format_percentage(value, precision=4, strip_zeros=True, auto_precision=True)
                elif abs(value) < 0.1 and value != 0:
                    # Small percentage
                    formatted_value = format_percentage(value, precision=3, strip_zeros=True)
                else:
                    # Regular percentage
                    formatted_value = format_percentage(value, strip_zeros=True)
            else:
                # Just format with commas for readability, adjust precision based on magnitude
                if value == 0:
                    formatted_value = "0"
                elif abs(value) < 0.001:
                    formatted_value = f"{value:.6f}"
                elif abs(value) < 0.01:
                    formatted_value = f"{value:.4f}"
                elif abs(value) < 1:
                    formatted_value = f"{value:.2f}"
                elif abs(value) < 10:
                    formatted_value = f"{value:.1f}"
                elif abs(value) >= 1000000 and compact:
                    # Abbreviate large numbers in compact mode
                    formatted_value = f"{value/1000000:.1f}M" if abs(value) >= 1000000 else f"{value/1000:.1f}K"
                else:
                    # Apply commas as thousands separators
                    formatted_value = f"{value:,.0f}"
    else:
        # For strings or other types, use as is
        formatted_value = str(value)
    
    # Format the delta if provided
    if delta is not None:
        # Use provided delta formatter if available
        if delta_formatter:
            formatted_delta = delta_formatter(delta)
        # Check if delta is a string already
        elif isinstance(delta, str):
            formatted_delta = delta
        else:
            # Special handling for percentage delta
            if delta_suffix == "%":
                # Format percentage delta with plus sign for positive values
                if abs(delta) < 0.01 and delta != 0:
                    formatted_delta = format_percentage(delta, precision=4, show_plus=True, strip_zeros=True, auto_precision=True)
                elif abs(delta) < 0.1 and delta != 0:
                    formatted_delta = format_percentage(delta, precision=3, show_plus=True, strip_zeros=True)
                else:
                    formatted_delta = format_percentage(delta, show_plus=True, strip_zeros=True)
            # Special handling for currency delta
            elif delta_suffix == "$":
                # Format currency delta with appropriate precision
                if abs(delta) < 0.01 and delta != 0:
                    formatted_delta = format_currency(delta, precision=4, show_decimals=True, compact=compact, strip_zeros=True)
                elif abs(delta) < 1 and delta != 0:
                    formatted_delta = format_currency(delta, precision=2, show_decimals=True, compact=compact)
                else:
                    formatted_delta = format_currency(delta, show_decimals=False, compact=compact, abbreviate=compact)
                # Ensure plus sign for positive values
                if delta > 0 and not str(formatted_delta).startswith('+'):
                    formatted_delta = f"+{formatted_delta}"
            else:
                # Add sign and suffix for other types
                if delta > 0:
                    formatted_delta = f"+{delta}{delta_suffix}"
                else:
                    formatted_delta = f"{delta}{delta_suffix}"
        
        # Apply custom CSS if provided
        if custom_css:
            st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
        
        # Apply custom styling for compact mode
        if compact:
            st.markdown("""
            <style>
            div[data-testid="stMetric"] > div:first-child {
                margin-bottom: -0.5rem;
            }
            div[data-testid="stMetric"] > div:nth-child(2) {
                font-size: 1.5rem;
            }
            div[data-testid="stMetric"] > div:nth-child(3) {
                font-size: 0.8rem;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Generate a modified title if hiding the label is requested
        display_title = "" if hide_label else title
        
        # Display metric with delta
        if help_text:
            with st.container():
                # Adjust column widths based on compact mode
                col_ratio = [0.95, 0.05] if compact else [0.9, 0.1]
                col1, col2 = st.columns(col_ratio)
                with col1:
                    st.metric(
                        label=display_title,
                        value=formatted_value,
                        delta=formatted_delta,
                        delta_color="off" if not color_delta else "normal",
                        help=debug_info if debug_info else None
                    )
                with col2:
                    # Adjust the info icon position based on compact mode
                    margin_top = '15px' if compact else '30px'
                    st.markdown(f"<div style='margin-top: {margin_top}'>{st.info('ℹ️')}</div>", unsafe_allow_html=True)
                    st.info(help_text)
        else:
            st.metric(
                label=display_title,
                value=formatted_value,
                delta=formatted_delta,
                delta_color="off" if not color_delta else "normal",
                help=debug_info if debug_info else None
            )
    else:
        # Display metric without delta
        # Apply custom CSS if provided
        if custom_css:
            st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
        
        # Apply custom styling for compact mode
        if compact:
            st.markdown("""
            <style>
            div[data-testid="stMetric"] > div:first-child {
                margin-bottom: -0.5rem;
            }
            div[data-testid="stMetric"] > div:nth-child(2) {
                font-size: 1.5rem;
            }
            </style>
            """, unsafe_allow_html=True)
        
        # Generate a modified title if hiding the label is requested
        display_title = "" if hide_label else title
        
        if help_text:
            with st.container():
                # Adjust column widths based on compact mode
                col_ratio = [0.95, 0.05] if compact else [0.9, 0.1]
                col1, col2 = st.columns(col_ratio)
                with col1:
                    st.metric(
                        label=display_title,
                        value=formatted_value,
                        help=debug_info if debug_info else None
                    )
                with col2:
                    # Adjust the info icon position based on compact mode
                    margin_top = '15px' if compact else '30px'
                    st.markdown(f"<div style='margin-top: {margin_top}'>{st.info('ℹ️')}</div>", unsafe_allow_html=True)
                    st.info(help_text)
        else:
            st.metric(
                label=display_title,
                value=formatted_value,
                help=debug_info if debug_info else None
            )

def display_metrics_row(metrics_dict, formatters=None, columns=None, help_texts=None, spacing=None, 
                   auto_detect_formatters=True, column_widths=None, compact=False, delta_colors=True,
                   debug_mode=False, custom_css=None, hide_labels=False, min_column_width=None,
                   equal_height=True, vertical_spacing='10px'):
    """
    Display a row of metrics with improved formatting and spacing.

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
    spacing : str, optional
        CSS margin to add between metrics (e.g., '10px')
    auto_detect_formatters : bool
        Whether to automatically detect and apply formatters
    column_widths : list, optional
        Custom column widths (list of numbers that should add up to 1)
    compact : bool
        Whether to use a more compact display format
    delta_colors : bool
        Whether to color deltas (green for positive, red for negative)
    debug_mode : bool
        Whether to display debug information in tooltips
    custom_css : str, optional
        Custom CSS to apply to the metrics row
    hide_labels : bool
        Whether to hide metric labels (useful for compact displays)
    min_column_width : int, optional
        Minimum width in pixels for each column
    equal_height : bool
        Whether to enforce equal height for all metrics
    vertical_spacing : str
        CSS margin-bottom to add between metric rows (e.g., '10px')
    """
    # Determine number of columns
    num_metrics = len(metrics_dict)
    if columns is None:
        # In compact mode, we can display more columns
        max_columns = 6 if compact else 4
        columns = min(num_metrics, max_columns)  # Limit default based on mode
    
    # Ensure we don't try to create more columns than metrics
    columns = min(columns, num_metrics)
    
    # Use custom column widths if provided
    if column_widths and len(column_widths) >= columns:
        cols = st.columns(column_widths[:columns])
    else:
        # Create evenly spaced columns for the metrics
        cols = st.columns(columns)

    # Create default formatters and help texts if not provided
    if formatters is None:
        formatters = {}

    # Auto-detect appropriate formatters based on metric names when enabled
    if auto_detect_formatters:
        # Define currency and percentage terms for better detection
        currency_terms = ['price', 'volume', 'amount', 'usd', 'value', 'cap', 'aum', 'assets',
                         'liquidation', 'interest', 'depth', 'bid', 'ask', 'spread', 'fund', 'flow',
                         'cost', 'profit', 'revenue', 'balance', 'market', 'trade', 'size', 'notional']
                         
        percentage_terms = ['percent', 'change', 'rate', 'ratio', 'growth', '%', 'pct', 'fee', 'premium', 'discount',
                           'yield', 'return', 'volatility', 'margin', 'allocation', 'weight', 'share', 'dominance']
                           
        # Create formatters where they don't already exist
        for key in metrics_dict.keys():
            if key not in formatters:
                key_lower = key.lower()
                
                # Check if it's likely a currency value by key name
                is_currency = any(term in key_lower for term in currency_terms) or key_lower.endswith('usd')
                
                # Check if it's likely a percentage value by key name
                is_percentage = any(term in key_lower for term in percentage_terms) or key_lower.endswith('%') or key_lower.endswith('pct')
                
                # Check if it's likely a count/integer value by key name
                is_count = any(term in key_lower for term in ['count', 'num', 'total', 'size', 'amount']) and not is_currency
                
                # Create appropriate formatter based on detected type
                if is_currency:
                    # For currency values, adapt formatting based on whether we're in compact mode
                    if compact:
                        # More compact display with abbreviations for large values
                        formatters[key] = lambda x: format_currency(x, show_decimals=False, compact=True, abbreviate=True)
                    else:
                        # Regular display with no decimals for cleaner headline stats
                        formatters[key] = lambda x: format_currency(x, show_decimals=False, compact=True, abbreviate=False)
                elif is_percentage:
                    # For percentage values, create a closure to capture proper precision
                    # Standard percentage with trailing zeros stripped
                    formatters[key] = lambda x: format_percentage(x, strip_zeros=True, auto_precision=True)
                elif is_count:
                    # For count values, format with commas but no decimals
                    formatters[key] = lambda x: f"{int(x):,}" if isinstance(x, (int, float)) else str(x)
                else:
                    # Default formatter for other number types
                    formatters[key] = lambda x: (f"{x:,.2f}" if isinstance(x, (int, float)) and x != int(x) else 
                                              (f"{int(x):,}" if isinstance(x, (int, float)) else str(x)))

    # Create default help texts if not provided
    if help_texts is None:
        help_texts = {key: None for key in metrics_dict}

    # Add spacing between metrics if requested
    css_parts = []
    if spacing:
        css_parts.append(f"[data-testid=\"stMetric\"] {{ margin: {spacing}; }}")
        
    if equal_height:
        css_parts.append("[data-testid=\"stMetric\"] { height: 100%; }")
    
    if compact:
        css_parts.append("""
        [data-testid="stMetric"] > div:first-child { /* Label */
            font-size: 0.8rem !important;
            margin-bottom: -0.5rem !important;
        }
        [data-testid="stMetric"] > div:nth-child(2) { /* Value */
            font-size: 1.4rem !important;
        }
        [data-testid="stMetric"] > div:nth-child(3) { /* Delta */
            font-size: 0.75rem !important;
        }
        """)
    
    if min_column_width:
        css_parts.append(f"[data-testid=\"column\"] {{ min-width: {min_column_width}px; }}")
    
    # Add vertical spacing between metric rows
    if vertical_spacing:
        css_parts.append(f"[data-testid=\"stHorizontalBlock\"] {{ margin-bottom: {vertical_spacing}; }}")
    
    # Apply custom CSS if provided
    if custom_css:
        css_parts.append(custom_css)
    
    # Combine all CSS parts and apply them
    if css_parts:
        combined_css = "\n".join(css_parts)
        st.markdown(f"<style>{combined_css}</style>", unsafe_allow_html=True)

    # Display each metric in its column with proper formatting
    for i, (key, value) in enumerate(metrics_dict.items()):
        col_index = i % columns
        
        # Prepare debug info if debug mode is enabled
        debug_info = None
        if debug_mode:
            if isinstance(value, dict) and 'value' in value:
                debug_info = f"Value: {value['value']} ({type(value['value']).__name__})\nDelta: {value.get('delta')} ({type(value.get('delta')).__name__ if value.get('delta') is not None else 'None'})"
            else:
                debug_info = f"Value: {value} ({type(value).__name__})"
        
        with cols[col_index]:
            if isinstance(value, dict) and 'value' in value:
                # For value/delta dict format
                delta_value = value.get('delta')
                
                # Determine appropriate delta_suffix
                delta_suffix = value.get('delta_suffix', '%')
                
                # Check if this is a currency metric for proper delta formatting
                if key in formatters:
                    formatter_str = str(formatters[key])
                    # If this is a currency formatter, use $ for delta
                    if 'format_currency' in formatter_str or ('$' in formatter_str and any(term in formatter_str for term in ['compact', 'abbreviate', 'currency'])):
                        delta_suffix = '$'
                
                # Display the metric with appropriate formatting
                display_metric_card(
                    key,
                    value['value'],
                    delta_value,
                    delta_suffix,
                    formatter=formatters.get(key),
                    help_text=help_texts.get(key),
                    delta_formatter=value.get('delta_formatter'),
                    compact=compact,
                    color_delta=delta_colors,
                    debug_info=debug_info,
                    hide_label=hide_labels
                )
            else:
                # For direct value format
                display_metric_card(
                    key,
                    value,
                    formatter=formatters.get(key),
                    help_text=help_texts.get(key),
                    compact=compact,
                    debug_info=debug_info,
                    hide_label=hide_labels
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