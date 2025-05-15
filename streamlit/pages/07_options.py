"""
Options page for the Izun Crypto Liquidity Report.

This page displays data and visualizations related to cryptocurrency options markets.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

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
    calculate_metrics,
    get_available_assets_for_category
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
    page_title=f"{APP_TITLE} - Options Markets",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the current page for sidebar navigation
st.session_state.current_page = 'options'

def load_options_data(asset):
    """
    Load options data for the specified asset.
    
    Parameters:
    -----------
    asset : str
        Asset to load data for
        
    Returns:
    --------
    dict
        Dictionary containing options data
    """
    data = {}
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return data
    
    # Load options data
    data = load_data_for_category('options', None, asset)
    
    return data

def render_options_info(data, asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render options market information.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing options data
    asset : str
        Primary asset to display (for backward compatibility)
    all_selected_assets : list, optional
        List of all selected assets to display
    selected_exchanges : list, optional
        List of exchanges to display data for
    selected_time_range : str, optional
        Selected time range for filtering data
    """
    st.header(f"{asset} Options Market Overview")
    
    # Exchange selector
    # Define available exchanges for options market data
    available_exchanges = ["Deribit", "OKX", "Binance", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Add exchange selector
    exchange_col1, exchange_col2 = st.columns([3, 1])
    with exchange_col1:
        selected_exchanges = st.multiselect(
            "Select Exchanges to Display",
            available_exchanges,
            default=default_exchanges,
            key=f"options_info_exchange_selector_{asset}"
        )
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
        st.warning("At least one exchange must be selected. Defaulting to 'All'.")
    
    # Store in session state for this section
    st.session_state.selected_options_info_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges

    # Options info data - try different possible key formats
    options_info_key = None
    possible_keys = [
        f"api_option_info_{asset}_{asset}",
        f"api_option_info_{asset}",
        "api_option_info"  # Generic fallback
    ]

    for key in possible_keys:
        if key in data and not data[key].empty:
            options_info_key = key
            break

    if options_info_key is not None and not data[options_info_key].empty:
        try:
            # Make a copy to avoid modifying the original dataframe
            options_df = data[options_info_key].copy()

            # Log the columns for debugging
            st.session_state['debug_columns'] = list(options_df.columns)

            # Filter for the asset if needed (for generic key)
            if options_info_key == "api_option_info" and 'symbol' in options_df.columns:
                options_df = options_df[options_df['symbol'].str.contains(asset, case=False, na=False)]
                
            # Filter by selected exchanges if needed (if not "All")
            if 'exchange_name' in options_df.columns and selected_exchanges and "All" not in selected_exchanges:
                options_df = options_df[options_df['exchange_name'].isin(selected_exchanges)]

            # Check if we have the key metrics
            has_oi = any(col in options_df.columns for col in ['open_interest', 'open_interest_usd'])

            # Calculate metrics based on available columns
            if has_oi:
                # Determine which OI column to use - prefer usd value
                oi_col = 'open_interest_usd' if 'open_interest_usd' in options_df.columns else 'open_interest'

                # Sum total OI across exchanges
                total_oi = options_df[oi_col].sum()

                # Calculate volume if available
                total_volume = options_df['volume_usd_24h'].sum() if 'volume_usd_24h' in options_df.columns else 0

                # Show 24h OI change if available
                oi_change_24h = None
                if 'open_interest_change_24h' in options_df.columns:
                    # Weighted average change by OI
                    weighted_change = (options_df['open_interest_change_24h'] * options_df[oi_col]).sum() / options_df[oi_col].sum() if options_df[oi_col].sum() > 0 else 0
                    oi_change_24h = weighted_change
            else:
                # Fallback if no OI data available
                total_oi = 0
                total_volume = 0
                oi_change_24h = None

            # Display metrics
            metrics = {
                "Total Open Interest": {
                    "value": total_oi,
                    "delta": oi_change_24h
                }
            }

            # Add 24h volume if available
            if total_volume > 0:
                metrics["24h Trading Volume"] = {
                    "value": total_volume,
                    "delta": None
                }

            formatters = {
                "Total Open Interest": lambda x: format_currency(x, abbreviate=True, show_decimals=False),
                "24h Trading Volume": lambda x: format_currency(x, abbreviate=True, show_decimals=False)
            }

            display_metrics_row(metrics, formatters)

            # Exchange breakdown
            st.subheader(f"{asset} Options Market by Exchange")

            # Create a formatted table for the exchange data
            display_cols = ['exchange_name']

            # Add important columns if they exist
            if oi_col in options_df.columns:
                display_cols.append(oi_col)
            if 'oi_market_share' in options_df.columns:
                display_cols.append('oi_market_share')
            if 'volume_usd_24h' in options_df.columns:
                display_cols.append('volume_usd_24h')
            if 'open_interest_change_24h' in options_df.columns:
                display_cols.append('open_interest_change_24h')
            if 'volume_change_percent_24h' in options_df.columns:
                display_cols.append('volume_change_percent_24h')

            # Create a display dataframe with selected columns
            if len(display_cols) > 1:  # Need at least exchange_name plus one more column
                display_df = options_df[display_cols].copy()

                # Format column names for better display
                column_display_names = {
                    'exchange_name': 'Exchange',
                    'open_interest': 'Open Interest (Contracts)',
                    'open_interest_usd': 'Open Interest (USD)',
                    'oi_market_share': 'Market Share (%)',
                    'volume_usd_24h': '24h Volume (USD)',
                    'open_interest_change_24h': '24h OI Change (%)',
                    'volume_change_percent_24h': '24h Volume Change (%)'
                }

                # Apply renaming for columns that exist
                rename_dict = {col: column_display_names[col] for col in display_cols if col in column_display_names}
                display_df = display_df.rename(columns=rename_dict)

                # Create format dictionary for the columns we have
                format_dict = {}
                if 'Open Interest (USD)' in display_df.columns:
                    format_dict['Open Interest (USD)'] = lambda x: format_currency(x, abbreviate=True, show_decimals=False)
                if 'Open Interest (Contracts)' in display_df.columns:
                    format_dict['Open Interest (Contracts)'] = lambda x: format_currency(x, include_symbol=False, abbreviate=True, show_decimals=False)
                if 'Market Share (%)' in display_df.columns:
                    format_dict['Market Share (%)'] = lambda x: format_percentage(x, precision=2)
                if '24h Volume (USD)' in display_df.columns:
                    format_dict['24h Volume (USD)'] = lambda x: format_currency(x, abbreviate=True, show_decimals=False)
                if '24h OI Change (%)' in display_df.columns:
                    format_dict['24h OI Change (%)'] = lambda x: format_percentage(x, precision=2)
                if '24h Volume Change (%)' in display_df.columns:
                    format_dict['24h Volume Change (%)'] = lambda x: format_percentage(x, precision=2)

                # Create and display the table
                create_formatted_table(display_df, format_dict=format_dict)
            else:
                st.info("Limited exchange data available.")
                st.dataframe(options_df)

            # Create pie chart for market share if available
            if 'exchange_name' in options_df.columns and has_oi:
                st.subheader(f"{asset} Options Open Interest Market Share")

                # Create copy for visualization
                viz_df = options_df.copy()

                # Calculate market share if not already present
                if 'oi_market_share' not in viz_df.columns:
                    viz_df['oi_market_share'] = viz_df[oi_col] / viz_df[oi_col].sum() * 100

                # Sort by open interest
                viz_df = viz_df.sort_values(by=oi_col, ascending=False)

                # Create pie chart
                fig = create_pie_chart(
                    viz_df,
                    'oi_market_share' if 'oi_market_share' in viz_df.columns else oi_col,
                    'exchange_name',
                    f"{asset} Options Market Share by Exchange"
                )

                display_chart(fig)
                
                # The pie chart represents current market share, not historical data
                
                # Maintain compatibility with session state
                default_time_range = selected_time_range if selected_time_range else st.session_state.get('selected_time_range', '3M')
                st.session_state.options_market_time_range = default_time_range
                st.session_state.selected_time_range = default_time_range

            # Explanation
            st.markdown(f"""
            ### Understanding {asset} Options Market

            Options are financial derivatives that give the holder the right, but not the obligation, to buy (call) or sell (put) an asset at a predetermined price (strike price) within a specific time period.

            **Key Metrics**:

            - **Open Interest (OI)**: The total number of outstanding options contracts that have not been settled
            - **Market Share**: Percentage of total open interest held by each exchange
            - **24h Volume**: Trading volume over the past 24 hours in USD
            - **OI Change (24h)**: Percentage change in open interest over the past 24 hours
            """)
        except Exception as e:
            st.error(f"Error processing options data: {e}")
            st.info(f"Debug info - columns: {st.session_state.get('debug_columns', 'No columns data')}")
            st.dataframe(data[options_info_key].head(5))  # Display raw data for debugging
    else:
        st.info(f"No options market information available for {asset}.")

def render_max_pain(data, asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render options max pain visualization.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing options data
    asset : str
        Primary asset to display (for backward compatibility)
    all_selected_assets : list, optional
        List of all selected assets to display
    selected_exchanges : list, optional
        List of exchanges to display data for
    selected_time_range : str, optional
        Selected time range for filtering data
    """
    st.header(f"{asset} Options Max Pain")
    
    # Exchange selector
    # Define available exchanges for options max pain
    available_exchanges = ["Deribit", "OKX", "Binance", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Add exchange selector
    exchange_col1, exchange_col2 = st.columns([3, 1])
    with exchange_col1:
        selected_exchanges = st.multiselect(
            "Select Exchanges to Display",
            available_exchanges,
            default=default_exchanges,
            key=f"options_max_pain_exchange_selector_{asset}"
        )
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
        st.warning("At least one exchange must be selected. Defaulting to 'All'.")
    
    # Store in session state for this section
    st.session_state.selected_max_pain_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges

    # Max pain data - try different possible key formats
    max_pain_key = None
    possible_keys = [
        f"api_option_max_pain_{asset}_{asset}",
        f"api_option_max_pain_{asset}",
        "api_option_max_pain"  # Generic fallback
    ]

    for key in possible_keys:
        if key in data and not data[key].empty:
            max_pain_key = key
            break

    if max_pain_key is not None and not data[max_pain_key].empty:
        try:
            # Make a copy to avoid modifying the original dataframe
            max_pain_df = data[max_pain_key].copy()

            # Log columns for debugging
            st.session_state['max_pain_columns'] = list(max_pain_df.columns)

            # Filter for the specific asset if using a generic key
            if max_pain_key == "api_option_max_pain" and 'symbol' in max_pain_df.columns:
                max_pain_df = max_pain_df[max_pain_df['symbol'].str.contains(asset, case=False, na=False)]
                
            # Filter by selected exchanges if needed (if not "All")
            if 'exchange_name' in max_pain_df.columns and selected_exchanges and "All" not in selected_exchanges:
                max_pain_df = max_pain_df[max_pain_df['exchange_name'].isin(selected_exchanges)]

            # Check and rename columns if needed
            date_col = None
            for col_name in ['date', 'expiry_date', 'expiration', 'expiry']:
                if col_name in max_pain_df.columns:
                    date_col = col_name
                    break

            # Check for max pain price column
            price_col = None
            for col_name in ['max_pain_price', 'max_pain', 'max_pain_value']:
                if col_name in max_pain_df.columns:
                    price_col = col_name
                    break

            # Call and put OI columns
            call_oi_col = None
            put_oi_col = None
            for call_col in ['call_open_interest', 'call_oi', 'calloi']:
                if call_col in max_pain_df.columns:
                    call_oi_col = call_col
                    break
            for put_col in ['put_open_interest', 'put_oi', 'putoi']:
                if put_col in max_pain_df.columns:
                    put_oi_col = put_col
                    break

            # Check if we have the required data
            if price_col is not None and date_col is not None:
                # Convert date to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(max_pain_df[date_col]):
                    # Check if the date column is a string that looks like YYMMDD
                    if isinstance(max_pain_df[date_col].iloc[0], str) and len(max_pain_df[date_col].iloc[0]) == 6:
                        # Convert YYMMDD to datetime (assuming 20YY for year)
                        max_pain_df['expiry_date'] = pd.to_datetime(
                            max_pain_df[date_col].apply(lambda x: f"20{x[:2]}-{x[2:4]}-{x[4:6]}"),
                            format="%Y-%m-%d"
                        )
                    else:
                        # Try regular conversion
                        max_pain_df['expiry_date'] = pd.to_datetime(max_pain_df[date_col])
                else:
                    # Already datetime, just rename for consistency
                    max_pain_df['expiry_date'] = max_pain_df[date_col]

                # Sort by expiry date
                max_pain_df = max_pain_df.sort_values(by='expiry_date')

                # Calculate call/put OI if we have the data
                if call_oi_col is not None and put_oi_col is not None:
                    max_pain_df['call_open_interest'] = max_pain_df[call_oi_col]
                    max_pain_df['put_open_interest'] = max_pain_df[put_oi_col]
                    max_pain_df['put_call_ratio'] = max_pain_df['put_open_interest'] / max_pain_df['call_open_interest'].replace(0, float('nan'))

                # Display nearest expiry max pain
                future_expiries = max_pain_df[max_pain_df['expiry_date'] >= datetime.now()]
                nearest_expiry = future_expiries.iloc[0] if not future_expiries.empty else None

                if nearest_expiry is not None:
                    st.subheader("Nearest Expiry Max Pain")

                    # Format expiry date
                    expiry_str = nearest_expiry['expiry_date'].strftime('%Y-%m-%d')

                    # Display metrics
                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("Expiry Date", expiry_str)

                    with col2:
                        st.metric("Max Pain Price", format_currency(nearest_expiry[price_col], precision=2))

                # Create max pain chart by expiry
                st.subheader(f"{asset} Max Pain by Expiry Date")

                # Format dates for display
                max_pain_df['expiry_display'] = max_pain_df['expiry_date'].dt.strftime('%Y-%m-%d')

                # Make sure future_expiries has the expiry_display column
                future_expiries = max_pain_df[max_pain_df['expiry_date'] >= datetime.now()].copy()

                if not future_expiries.empty:
                    try:
                        # Create bar chart
                        fig = px.bar(
                            future_expiries,
                            x='expiry_display',
                            y=price_col,
                            title=f"{asset} Max Pain by Expiry Date",
                            color=price_col,
                            color_continuous_scale='Viridis'
                        )

                        fig.update_layout(
                            xaxis_title=None,
                            yaxis_title="Max Pain Price (USD)",
                            coloraxis_showscale=False
                        )

                        # Format y-axis tick labels
                        fig.update_yaxes(
                            tickprefix="$",
                            tickformat=",",
                        )

                        display_chart(apply_chart_theme(fig))
                        
                        
                        # Set defaults in session state for backward compatibility
                        st.session_state.max_pain_time_range = 'All'
                        st.session_state.selected_time_range = 'All'
                    except Exception as chart_error:
                        st.error(f"Error creating max pain chart: {chart_error}")
                        # Show the data in a table instead
                        st.write("Showing data in table format instead:")
                        st.dataframe(future_expiries[[date_col, price_col, 'expiry_date', 'expiry_display']].head(10))

                # Create OI distribution and put/call ratio charts if data is available
                if call_oi_col is not None and put_oi_col is not None:
                    try:
                        # Make sure we're working with a copy to avoid modifying the original
                        oi_data = future_expiries.copy()

                        # Calculate total OI and sort for display
                        oi_data['total_oi'] = oi_data['call_open_interest'] + oi_data['put_open_interest']
                        oi_data = oi_data.sort_values('expiry_date')

                        st.subheader(f"{asset} Call/Put Open Interest by Expiry Date")

                        # Create a dataframe with just the columns we need for the chart
                        chart_data = pd.DataFrame({
                            'expiry_display': oi_data['expiry_display'],
                            'call_open_interest': oi_data['call_open_interest'],
                            'put_open_interest': oi_data['put_open_interest']
                        })

                        # Melt dataframe for grouped bar chart
                        melted_df = pd.melt(
                            chart_data,
                            id_vars=['expiry_display'],
                            value_vars=['call_open_interest', 'put_open_interest'],
                            var_name='option_type',
                            value_name='open_interest'
                        )

                        # Create grouped bar chart
                        fig = px.bar(
                            melted_df,
                            x='expiry_display',
                            y='open_interest',
                            color='option_type',
                            title=f"{asset} Call/Put Open Interest by Expiry Date",
                            barmode='group',
                            color_discrete_map={
                                'call_open_interest': 'green',
                                'put_open_interest': 'red'
                            }
                        )

                        # Rename legend items
                        fig.for_each_trace(lambda t: t.update(
                            name=t.name.replace('call_open_interest', 'Call OI').replace('put_open_interest', 'Put OI')
                        ))

                        fig.update_layout(
                            xaxis_title=None,
                            yaxis_title="Open Interest"
                        )

                        # Format y-axis tick labels
                        fig.update_yaxes(
                            tickformat=",",
                        )

                        display_chart(apply_chart_theme(fig))
                    except Exception as e:
                        st.error(f"Error creating open interest charts: {e}")
                        # Show the data in a table instead
                        st.write("Showing OI data in table format instead:")
                        try:
                            display_cols = ['expiry_date', 'call_open_interest', 'put_open_interest']
                            if 'expiry_display' in future_expiries.columns:
                                display_cols = ['expiry_display'] + display_cols
                            st.dataframe(future_expiries[display_cols].head(10))
                        except Exception:
                            st.dataframe(future_expiries.head(10))

                    # Put/Call Ratio Chart
                    try:
                        st.subheader(f"{asset} Put/Call Ratio by Expiry Date")

                        # Create a dataframe with just the columns we need for the ratio chart
                        ratio_data = pd.DataFrame({
                            'expiry_display': oi_data['expiry_display'],
                            'put_call_ratio': oi_data['put_call_ratio']
                        })

                        fig = px.bar(
                            ratio_data,
                            x='expiry_display',
                            y='put_call_ratio',
                            title=f"{asset} Put/Call Ratio by Expiry Date",
                            color='put_call_ratio',
                            color_continuous_scale='RdYlGn_r',  # Red for high P/C ratio (bearish), green for low (bullish)
                            color_continuous_midpoint=1
                        )

                        # Add reference line at 1 (equal puts and calls)
                        fig.add_hline(
                            y=1,
                            line_dash="dash",
                            line_color="gray",
                            annotation_text="Equal"
                        )

                        fig.update_layout(
                            xaxis_title=None,
                            yaxis_title="Put/Call Ratio"
                        )

                        display_chart(apply_chart_theme(fig))
                    except Exception as e:
                        st.error(f"Error creating put/call ratio chart: {e}")
                        # Show the ratio data in a table
                        st.write("Showing Put/Call Ratio data in table format:")
                        try:
                            if 'put_call_ratio' in future_expiries.columns:
                                display_cols = ['expiry_date', 'put_call_ratio']
                                if 'expiry_display' in future_expiries.columns:
                                    display_cols = ['expiry_display'] + display_cols
                                st.dataframe(future_expiries[display_cols].head(10))
                            else:
                                st.info("Put/Call Ratio data not available")
                        except Exception:
                            st.info("Could not display Put/Call Ratio data")

                # Explanation
                st.markdown("""
                ### Understanding Max Pain

                Max Pain is the price point where options traders would experience the maximum loss upon expiration:

                - It's calculated as the strike price where the total value of all outstanding options (puts and calls) would be minimized
                - Option writers (sellers) benefit when the underlying asset closes at the max pain price at expiration
                - There's a theory that the market may gravitate toward the max pain price as expiration approaches
                - Max pain can be used as a potential support/resistance level or price target

                **Key Metrics**:

                - **Max Pain Price**: The price at which option holders would lose the most money
                - **Put/Call Ratio**: The ratio of put open interest to call open interest, which can indicate market sentiment
                  - Ratio > 1: More puts than calls (potentially bearish)
                  - Ratio < 1: More calls than puts (potentially bullish)
                """)
            else:
                missing_cols = []
                if date_col is None:
                    missing_cols.append("expiry date")
                if price_col is None:
                    missing_cols.append("max pain price")

                st.info(f"Max pain data available for {asset} but missing required columns: {', '.join(missing_cols)}.")
                st.info(f"Available columns: {', '.join(st.session_state.get('max_pain_columns', []))}")
                st.dataframe(max_pain_df.head(5))  # Show raw data for debugging
        except Exception as e:
            st.error(f"Error processing max pain data: {e}")
            st.info(f"Debug info - columns: {st.session_state.get('max_pain_columns', 'No columns data')}")
            st.dataframe(data[max_pain_key].head(5))  # Display raw data for debugging
    else:
        st.info(f"No max pain data available for {asset}.")

def render_options_volume(data, asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render options volume visualization.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing options data
    asset : str
        Primary asset to display (for backward compatibility)
    all_selected_assets : list, optional
        List of all selected assets to display
    selected_exchanges : list, optional
        List of exchanges to display data for
    selected_time_range : str, optional
        Selected time range for filtering data
    """
    st.header(f"{asset} Options Volume")
    
    # Exchange selector
    # Define available exchanges for options volume
    available_exchanges = ["Deribit", "OKX", "Binance", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Add exchange selector
    exchange_col1, exchange_col2 = st.columns([3, 1])
    with exchange_col1:
        selected_exchanges = st.multiselect(
            "Select Exchanges to Display",
            available_exchanges,
            default=default_exchanges,
            key=f"options_volume_exchange_selector_{asset}"
        )
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
        st.warning("At least one exchange must be selected. Defaulting to 'All'.")
    
    # Store in session state for this section
    st.session_state.selected_options_volume_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges
    
    # Options volume data
    if 'api_option_exchange_vol_history' in data and not data['api_option_exchange_vol_history'].empty:
        volume_df = data['api_option_exchange_vol_history']
        
        # Filter for the specific asset if needed
        asset_volume = volume_df
        if 'symbol' in volume_df.columns:
            asset_volume = volume_df[volume_df['symbol'].str.contains(asset, case=False, na=False)]
            
        # Filter by selected exchanges if needed (if not "All")
        if 'exchange_name' in asset_volume.columns and selected_exchanges and "All" not in selected_exchanges:
            asset_volume = asset_volume[asset_volume['exchange_name'].isin(selected_exchanges)]
        
        if not asset_volume.empty:
            # Process dataframe
            asset_volume = process_timestamps(asset_volume)
            
            # Create volume chart
            st.subheader(f"{asset} Options Volume History")
            
            # Create time series chart
            fig = create_time_series(
                asset_volume,
                'datetime',
                'volume',
                f"{asset} Options Volume History",
                color_col='exchange_name' if 'exchange_name' in asset_volume.columns else None,
                height=500
            )
            
            display_chart(fig)
            
            
            # Set defaults in session state for backward compatibility
            st.session_state.options_volume_time_range = 'All'
            st.session_state.selected_time_range = 'All'
            
            # Create by exchange breakdown if available
            if 'exchange_name' in asset_volume.columns:
                st.subheader(f"{asset} Options Volume by Exchange")
                
                # Group by exchange
                exchange_volume = asset_volume.groupby('exchange_name')['volume'].sum().reset_index()
                exchange_volume = exchange_volume.sort_values(by='volume', ascending=False)
                
                # Create pie chart
                fig = create_pie_chart(
                    exchange_volume,
                    'volume',
                    'exchange_name',
                    f"{asset} Options Volume by Exchange"
                )
                
                display_chart(fig)
                
                # The pie chart represents current volume distribution
        else:
            st.info(f"No options volume data available for {asset}.")
    else:
        st.info(f"No options volume data available.")

def render_options_oi(data, asset, all_selected_assets=None, selected_exchanges=None, selected_time_range=None):
    """Render options open interest visualization.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing options data
    asset : str
        Primary asset to display (for backward compatibility)
    all_selected_assets : list, optional
        List of all selected assets to display
    selected_exchanges : list, optional
        List of exchanges to display data for
    selected_time_range : str, optional
        Selected time range for filtering data
    """
    st.header(f"{asset} Options Open Interest")
    
    # Exchange selector
    # Define available exchanges for options OI
    available_exchanges = ["Deribit", "OKX", "Binance", "All"]
    
    # Default to session state if it exists, otherwise use All
    default_exchanges = selected_exchanges if selected_exchanges else ["All"]
    
    # Add exchange selector
    exchange_col1, exchange_col2 = st.columns([3, 1])
    with exchange_col1:
        selected_exchanges = st.multiselect(
            "Select Exchanges to Display",
            available_exchanges,
            default=default_exchanges,
            key=f"options_oi_exchange_selector_{asset}"
        )
    
    # Ensure at least one exchange is selected
    if not selected_exchanges:
        selected_exchanges = ["All"]
        st.warning("At least one exchange must be selected. Defaulting to 'All'.")
    
    # Store in session state for this section
    st.session_state.selected_options_oi_exchanges = selected_exchanges
    
    # For backward compatibility
    st.session_state.selected_exchanges = selected_exchanges
    
    # Options OI data
    if 'api_option_exchange_oi_history' in data and not data['api_option_exchange_oi_history'].empty:
        oi_df = data['api_option_exchange_oi_history']
        
        # Filter for the specific asset if needed
        asset_oi = oi_df
        if 'symbol' in oi_df.columns:
            asset_oi = oi_df[oi_df['symbol'].str.contains(asset, case=False, na=False)]
            
        # Filter by selected exchanges if needed (if not "All")
        if 'exchange_name' in asset_oi.columns and selected_exchanges and "All" not in selected_exchanges:
            asset_oi = asset_oi[asset_oi['exchange_name'].isin(selected_exchanges)]
        
        if not asset_oi.empty:
            # Process dataframe
            asset_oi = process_timestamps(asset_oi)
            
            # Create OI chart
            st.subheader(f"{asset} Options Open Interest History")
            
            # Create time series chart
            fig = create_time_series(
                asset_oi,
                'datetime',
                'open_interest',
                f"{asset} Options Open Interest History",
                color_col='exchange_name' if 'exchange_name' in asset_oi.columns else None,
                height=500
            )
            
            display_chart(fig)
            
            
            # Set defaults in session state for backward compatibility
            st.session_state.options_oi_time_range = 'All'
            st.session_state.selected_time_range = 'All'
            
            # Create by exchange breakdown if available
            if 'exchange_name' in asset_oi.columns:
                st.subheader(f"{asset} Options Open Interest by Exchange")
                
                # Group by exchange
                exchange_oi = asset_oi.groupby('exchange_name')['open_interest'].sum().reset_index()
                exchange_oi = exchange_oi.sort_values(by='open_interest', ascending=False)
                
                # Create pie chart
                fig = create_pie_chart(
                    exchange_oi,
                    'open_interest',
                    'exchange_name',
                    f"{asset} Options Open Interest by Exchange"
                )
                
                display_chart(fig)
                
                # The pie chart represents current open interest distribution
        else:
            st.info(f"No options open interest data available for {asset}.")
    else:
        st.info(f"No options open interest data available.")

def main():
    """Main function to render the options page."""
    try:
        # Render sidebar
        render_sidebar()

        # Page title and description
        st.title("Cryptocurrency Options Markets")
        st.write("Analysis of cryptocurrency options markets and metrics")

        # Get available assets for options category
        available_assets = get_available_assets_for_category('options')

        if not available_assets:
            st.error("No options data available for any asset.")
            return
        
        # Asset selection with dropdown
        st.subheader("Select Asset to Display")
        
        # Initialize with previously selected asset if available, otherwise default to first asset
        default_asset = st.session_state.get('selected_options_assets', [available_assets[0]])
        default_index = available_assets.index(default_asset[0]) if default_asset and default_asset[0] in available_assets else 0
        
        # Add dropdown for asset selection (improved from multiselect for better UI)
        selected_asset = st.selectbox(
            "Select asset to display",
            available_assets,
            index=default_index,
            key="options_assets_selector"
        )
        
        # Use a single asset in a list for compatibility with existing code
        selected_assets = [selected_asset]
        
        # Store selected assets in session state for this page
        st.session_state.selected_options_assets = selected_assets
        
        # For backward compatibility with existing code, use first selected asset as primary
        asset = selected_assets[0]
        
        # Also update the general selected_asset session state for compatibility
        st.session_state.selected_asset = asset

        # Display loading message
        with st.spinner(f"Loading {asset} options data..."):
            data = load_options_data(asset)

        # Get the last updated time
        # Removed data last updated reference

        # Check if data is available
        if not data:
            st.error(f"No options data available for {asset}.")
            return

        # Initialize session state for tracking active tab if not exists
        if 'options_active_tab' not in st.session_state:
            st.session_state.options_active_tab = "Market Overview"

        # Check if max pain data is available for this asset
        has_max_pain = False
        for key in [f"api_option_max_pain_{asset}_{asset}", f"api_option_max_pain_{asset}", "api_option_max_pain"]:
            if key in data and not data[key].empty:
                has_max_pain = True
                break

        # Create tabs list - always include Market Overview
        tab_options = ["Market Overview"]

        # Add Max Pain tab if data is available
        if has_max_pain:
            tab_options.append("Max Pain")

        # Create tabs for different options metrics
        tabs = st.tabs(tab_options)

        # Render the appropriate content in each tab
        for i, tab_name in enumerate(tab_options):
            with tabs[i]:
                if tab_name == "Market Overview":
                    # Get time range from session state
                    selected_time_range = st.session_state.get('selected_time_range', '3M')
                    render_options_info(data, asset, selected_assets, None, selected_time_range)
                elif tab_name == "Max Pain":
                    selected_time_range = st.session_state.get('selected_time_range', '3M')
                    render_max_pain(data, asset, selected_assets, None, selected_time_range)

    except Exception as e:
        st.error(f"Error rendering options page: {e}")
        # Show at least some data if available
        if 'data' in locals() and data:
            st.info("Available data keys:")
            st.write(list(data.keys()))

if __name__ == "__main__":
    main()
