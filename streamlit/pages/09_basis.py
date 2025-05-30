"""
Basis Trading Analysis page for the Izun Crypto Liquidity Report.

This page provides comprehensive analysis of basis trading opportunities across
the top 10 cryptocurrencies, including annualized return calculations, historical
performance, and funding rate analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to path to import from components and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import components and utilities
from components.sidebar import render_sidebar
from components.metrics import display_metrics_row, display_metric_card
from components.charts import (
    create_time_series, 
    create_bar_chart,
    create_area_chart,
    apply_chart_theme,
    display_chart
)
from components.tables import create_formatted_table, create_crypto_table
from utils.data_loader import (
    get_latest_data_directory, 
    load_data_for_category,
    process_timestamps,
    get_data_last_updated,
    load_specific_data_file
)
from utils.formatters import (
    format_currency, 
    format_percentage, 
    format_volume,
    format_timestamp
)
from utils.config import APP_TITLE, APP_ICON, SUPPORTED_ASSETS

# Set page config
st.set_page_config(
    page_title=f"{APP_TITLE} - Basis Trading",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set the current page for sidebar navigation
st.session_state.current_page = 'basis'

# Define top 10 cryptocurrencies by market cap
TOP_10_ASSETS = ['BTC', 'ETH', 'SOL', 'XRP', 'BNB', 'ADA', 'DOGE', 'AVAX', 'DOT', 'MATIC']

def get_top_10_assets():
    """Return list of top 10 crypto assets by market cap."""
    return TOP_10_ASSETS

def load_basis_data():
    """
    Load all data required for basis analysis.
    
    Returns:
    --------
    dict
        Dictionary containing all required data
    """
    data = {}
    
    # Get the latest data directory
    latest_dir = get_latest_data_directory()
    
    if not latest_dir:
        st.error("No data directories found. Please check your data path.")
        return data
    
    try:
        # Load basis history data
        basis_history = load_specific_data_file('indic', '', 'api_futures_basis_history', latest_dir)
        if basis_history is not None:
            data['basis_history'] = basis_history
            logger.info(f"Loaded basis history: {len(basis_history)} rows")
        
        # Load funding rate exchange list data
        # Ensure we have the full path including 'data' directory
        if not latest_dir.startswith('data'):
            data_path = os.path.join('data', latest_dir)
        else:
            data_path = latest_dir
            
        # Current funding rates
        current_funding_file = os.path.join(data_path, 'futures', 'funding_rate', 'api_futures_fundingRate_exchange_list.parquet')
        if os.path.exists(current_funding_file):
            data['current_funding_rates'] = pd.read_parquet(current_funding_file)
            logger.info(f"Loaded current funding rates: {len(data['current_funding_rates'])} rows")
        
        # Accumulated funding rates (365d - maintains backward compatibility)
        accumulated_funding_file = os.path.join(data_path, 'futures', 'funding_rate', 'api_futures_fundingRate_accumulated_exchange_list_365d.parquet')
        if os.path.exists(accumulated_funding_file):
            data['accumulated_funding_rates'] = pd.read_parquet(accumulated_funding_file)
            logger.info(f"Loaded 365d accumulated funding rates: {len(data['accumulated_funding_rates'])} rows")
        
        # Load spot price data for top 10 assets
        spot_data = {}
        for asset in TOP_10_ASSETS:
            spot_asset_data = load_data_for_category('spot', 'spot_market', asset, latest_dir)
            if spot_asset_data:
                spot_data[asset] = spot_asset_data
        data['spot'] = spot_data
        
        # Load futures market data for top 10 assets
        futures_data = {}
        for asset in TOP_10_ASSETS:
            futures_asset_data = load_data_for_category('futures', 'market', asset, latest_dir)
            if futures_asset_data:
                futures_data[asset] = futures_asset_data
        data['futures'] = futures_data
        
        # Load funding rate history for detailed analysis
        funding_rate_data = {}
        for asset in TOP_10_ASSETS[:4]:  # Load detailed data for top 4 only to save memory
            funding_asset_data = load_data_for_category('futures', 'funding_rate', asset, latest_dir)
            if funding_asset_data:
                funding_rate_data[asset] = funding_asset_data
        data['funding_rate'] = funding_rate_data
        
        # Load volume data for analysis
        volume_data = {}
        
        # Load aggregated taker buy/sell volume history
        aggregated_volume_file = os.path.join(data_path, 'futures', 'taker_buy_sell', 'api_futures_aggregated_taker_buy_sell_volume_history.parquet')
        if os.path.exists(aggregated_volume_file):
            data['aggregated_volume_history'] = pd.read_parquet(aggregated_volume_file)
            logger.info(f"Loaded aggregated volume history: {len(data['aggregated_volume_history'])} rows")
        
        # Load asset-specific volume history for top 10 assets
        for asset in TOP_10_ASSETS:
            asset_volume_file = os.path.join(data_path, 'futures', 'taker_buy_sell', f'api_futures_taker_buy_sell_volume_history_{asset}.parquet')
            if os.path.exists(asset_volume_file):
                volume_data[asset] = pd.read_parquet(asset_volume_file)
                logger.info(f"Loaded {asset} volume history: {len(volume_data[asset])} rows")
            
            # Load exchange-specific volume data
            exchange_volume_file = os.path.join(data_path, 'futures', 'long_short_ratio', f'api_futures_taker_buy_sell_volume_exchange_list_{asset}.parquet')
            if os.path.exists(exchange_volume_file):
                volume_data[f'{asset}_exchange_volume'] = pd.read_parquet(exchange_volume_file)
                logger.info(f"Loaded {asset} exchange volume data: {len(volume_data[f'{asset}_exchange_volume'])} rows")
        
        data['volume'] = volume_data
        
        # Load open interest data
        oi_data = {}
        
        # Load OI history charts for top 10 assets
        for asset in TOP_10_ASSETS:
            oi_history_file = os.path.join(data_path, 'futures', 'open_interest', f'api_futures_openInterest_exchange_history_chart_{asset}.parquet')
            if os.path.exists(oi_history_file):
                oi_data[f'{asset}_history'] = pd.read_parquet(oi_history_file)
                logger.info(f"Loaded {asset} OI history: {len(oi_data[f'{asset}_history'])} rows")
            
            # Load current OI by exchange
            oi_exchange_file = os.path.join(data_path, 'futures', 'open_interest', f'api_futures_openInterest_exchange_list_{asset}.parquet')
            if os.path.exists(oi_exchange_file):
                oi_data[f'{asset}_exchange'] = pd.read_parquet(oi_exchange_file)
                logger.info(f"Loaded {asset} OI exchange data: {len(oi_data[f'{asset}_exchange'])} rows")
            
            # Load aggregated OI OHLC history
            oi_ohlc_file = os.path.join(data_path, 'futures', 'open_interest', f'api_futures_openInterest_ohlc_aggregated_history_{asset}.parquet')
            if os.path.exists(oi_ohlc_file):
                oi_data[f'{asset}_ohlc'] = pd.read_parquet(oi_ohlc_file)
                logger.info(f"Loaded {asset} OI OHLC data: {len(oi_data[f'{asset}_ohlc'])} rows")
        
        data['open_interest'] = oi_data
        
    except Exception as e:
        logger.error(f"Error loading basis data: {e}")
        st.error(f"Error loading data: {str(e)}")
    
    return data

def calculate_annualized_rate(current_rate, interval_hours):
    """
    Calculate annualized funding rate.
    
    Parameters:
    -----------
    current_rate : float
        The current funding rate
    interval_hours : float
        The funding interval in hours
    
    Returns:
    --------
    float
        Annualized funding rate as percentage
    """
    try:
        if pd.isna(current_rate):
            return None
        if pd.isna(interval_hours) or interval_hours <= 0:
            # Default to 8-hour interval if not specified
            interval_hours = 8.0
        # Formula: current_rate × (24 / interval_hours) × 365
        # Convert to percentage
        return float(current_rate) * (24 / float(interval_hours)) * 365 * 100
    except Exception as e:
        logger.error(f"Error calculating annualized rate: {e}")
        return None

def load_all_accumulated_funding_data(data_path):
    """
    Load accumulated funding rate data for all time periods.
    
    Parameters:
    -----------
    data_path : str
        Path to the data directory
    
    Returns:
    --------
    dict
        Dictionary with {period: DataFrame} for each time period
    """
    import json
    
    periods = ['1d', '7d', '30d', '365d']
    funding_data = {}
    
    for period in periods:
        file_path = os.path.join(
            data_path, 
            'futures', 
            'funding_rate', 
            f'api_futures_fundingRate_accumulated_exchange_list_{period}.parquet'
        )
        
        if os.path.exists(file_path):
            funding_data[period] = pd.read_parquet(file_path)
            logger.info(f"Loaded {period} accumulated funding rates: {len(funding_data[period])} rows")
        else:
            logger.warning(f"File not found: {file_path}")
            funding_data[period] = pd.DataFrame()
    
    return funding_data

def annualize_funding_rate(rate, period):
    """
    Convert accumulated funding rate to annualized rate.
    
    Parameters:
    -----------
    rate : float
        Accumulated funding rate (as decimal, e.g., 0.0689 for 6.89%)
    period : str
        Time period ('1d', '7d', '30d', '365d')
    
    Returns:
    --------
    float
        Annualized rate as percentage
    """
    multipliers = {
        '1d': 365,      # Daily to annual
        '7d': 52.14,    # Weekly to annual (365/7)
        '30d': 12.17,   # Monthly to annual (365/30)
        '365d': 1       # Already annual
    }
    
    return rate * multipliers.get(period, 1) * 100

def prepare_comprehensive_funding_rate_table(funding_data_dict):
    """
    Create comprehensive table showing annualized funding rates across all time periods.
    Only includes top 20 crypto assets by market cap.
    
    Parameters:
    -----------
    funding_data_dict : dict
        Dictionary with DataFrames for each period
        
    Returns:
    --------
    pandas.DataFrame
        Formatted table with annualized rates
    """
    import json
    
    if not funding_data_dict:
        return pd.DataFrame()
    
    # Define top 20 crypto assets to filter for
    TOP_20_ASSETS = [
        'BTC', 'ETH', 'USDT', 'XRP', 'BNB', 'SOL', 'USDC', 'DOGE', 'ADA', 'TRX',
        'WBTC', 'SUI', 'LINK', 'AVAX', 'SHIB', 'LEO', 'BCH', 'XLM', 'HBAR', 'TON'
    ]
    
    all_rows = []
    
    # Process each asset across all time periods
    for period, df in funding_data_dict.items():
        if df.empty:
            continue
            
        for _, row in df.iterrows():
            symbol = row['symbol']
            
            # Only process top 20 crypto assets
            if symbol not in TOP_20_ASSETS:
                continue
            
            # Process stablecoin margin exchanges
            stablecoin_data = row.get('stablecoin_margin_list')
            if stablecoin_data is not None and len(stablecoin_data) > 0:
                try:
                    # Handle both string (JSON) and list formats
                    if isinstance(stablecoin_data, str):
                        exchange_list = json.loads(stablecoin_data)
                    else:
                        exchange_list = stablecoin_data
                    
                    # Process each exchange in the list
                    for exchange_info in exchange_list:
                        if isinstance(exchange_info, dict) and 'exchange' in exchange_info and 'funding_rate' in exchange_info:
                            try:
                                exchange = exchange_info['exchange']
                                funding_rate_value = exchange_info['funding_rate']
                                
                                # Skip if funding_rate is None or empty
                                if funding_rate_value is None or funding_rate_value == '':
                                    continue
                                    
                                rate = float(funding_rate_value)  # Already a decimal
                                
                                # Annualize the rate
                                annualized_rate = annualize_funding_rate(rate, period)
                                
                                all_rows.append({
                                    'Asset': symbol,
                                    'Exchange': exchange,
                                    'Period': period,
                                    'Accumulated Rate (%)': f"{rate * 100:.2f}%",
                                    'Annualized Rate (%)': f"{annualized_rate:.2f}%",
                                    'Raw Rate': rate,
                                    'Raw Annualized': annualized_rate
                                })
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Error processing exchange data for {symbol}-{period}: {e}")
                                
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing stablecoin margin data for {symbol}-{period}: {e}")
            
            # Also process token margin exchanges if available
            token_data = row.get('token_margin_list')
            if token_data is not None and len(token_data) > 0:
                try:
                    # Handle both string (JSON) and list formats
                    if isinstance(token_data, str):
                        exchange_list = json.loads(token_data)
                    else:
                        exchange_list = token_data
                    
                    # Process each exchange in the list
                    for exchange_info in exchange_list:
                        if isinstance(exchange_info, dict) and 'exchange' in exchange_info and 'funding_rate' in exchange_info:
                            try:
                                exchange = f"{exchange_info['exchange']}_TOKEN"  # Distinguish from stablecoin
                                funding_rate_value = exchange_info['funding_rate']
                                
                                # Skip if funding_rate is None or empty
                                if funding_rate_value is None or funding_rate_value == '':
                                    continue
                                    
                                rate = float(funding_rate_value)  # Already a decimal
                                
                                # Annualize the rate
                                annualized_rate = annualize_funding_rate(rate, period)
                                
                                all_rows.append({
                                    'Asset': symbol,
                                    'Exchange': exchange,
                                    'Period': period,
                                    'Accumulated Rate (%)': f"{rate * 100:.2f}%",
                                    'Annualized Rate (%)': f"{annualized_rate:.2f}%",
                                    'Raw Rate': rate,
                                    'Raw Annualized': annualized_rate
                                })
                            except (ValueError, KeyError) as e:
                                logger.warning(f"Error processing token exchange data for {symbol}-{period}: {e}")
                                
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"Error parsing token margin data for {symbol}-{period}: {e}")
    
    if not all_rows:
        return pd.DataFrame()
    
    # Create DataFrame
    result_df = pd.DataFrame(all_rows)
    
    # Pivot to show periods as columns
    pivot_df = result_df.pivot_table(
        index=['Asset', 'Exchange'],
        columns='Period',
        values='Raw Annualized',
        aggfunc='first'
    ).reset_index()
    
    # Format columns
    period_columns = ['1d', '7d', '30d', '365d']
    for col in period_columns:
        if col in pivot_df.columns:
            pivot_df[f'{col} Annualized (%)'] = pivot_df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A")
    
    # Select and order final columns
    final_columns = ['Asset', 'Exchange']
    for period in period_columns:
        if f'{period} Annualized (%)' in pivot_df.columns:
            final_columns.append(f'{period} Annualized (%)')
    
    return pivot_df[final_columns]

def calculate_basis_trade_metrics(asset, spot_price, futures_price, funding_rate, interval_hours=8):
    """
    Calculate all metrics for a single basis trade.
    
    Parameters:
    -----------
    asset : str
        Asset symbol
    spot_price : float
        Current spot price
    futures_price : float
        Current futures price
    funding_rate : float
        Current funding rate (as decimal)
    interval_hours : float
        Funding interval in hours
    
    Returns:
    --------
    dict
        Dictionary with all calculated metrics
    """
    metrics = {
        'asset': asset,
        'spot_price': spot_price,
        'futures_price': futures_price,
        'basis_pct': 0,
        'funding_rate': funding_rate,
        'annualized_return': 0,
        'daily_return': 0
    }
    
    try:
        # Calculate basis percentage
        if spot_price > 0:
            metrics['basis_pct'] = ((futures_price - spot_price) / spot_price) * 100
        
        # Calculate annualized return from funding
        if funding_rate is not None:
            # Funding rate is already a decimal (e.g., 0.0001 for 0.01%)
            periods_per_day = 24 / interval_hours
            daily_rate = funding_rate * periods_per_day
            metrics['daily_return'] = daily_rate * 100  # Convert to percentage
            metrics['annualized_return'] = daily_rate * 365 * 100  # Convert to percentage
            
    except Exception as e:
        logger.error(f"Error calculating metrics for {asset}: {e}")
    
    return metrics

def prepare_basis_returns_table(data, top_n=10):
    """
    Create comprehensive returns table for top N assets.
    
    Parameters:
    -----------
    data : dict
        Loaded data dictionary
    top_n : int
        Number of top assets to include
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with basis trade returns for top assets
    """
    if 'current_funding_rates' not in data or data['current_funding_rates'] is None:
        return pd.DataFrame()
    
    try:
        returns_data = []
        funding_df = data['current_funding_rates']
        
        # Process each asset in top 10
        for asset in TOP_10_ASSETS[:top_n]:
            # Get funding rate data
            asset_row = funding_df[funding_df['symbol'] == asset]
            if asset_row.empty:
                continue
            
            # Get spot and futures prices
            spot_price = None
            futures_price = None
            
            # Get spot price
            if 'spot' in data and asset in data['spot']:
                spot_key = f'api_spot_price_history_{asset}'
                if spot_key in data['spot'][asset]:
                    spot_df = data['spot'][asset][spot_key]
                    if not spot_df.empty and 'close' in spot_df.columns:
                        spot_price = spot_df['close'].iloc[-1]
            
            # Get futures price
            if 'futures' in data and asset in data['futures']:
                futures_key = f'api_futures_pairs_markets_{asset}'
                if futures_key in data['futures'][asset]:
                    futures_df = data['futures'][asset][futures_key]
                    if not futures_df.empty and 'current_price' in futures_df.columns:
                        futures_price = futures_df['current_price'].mean()
            
            # Get funding rates from stablecoin margin
            funding_list = asset_row.iloc[0]['stablecoin_margin_list']
            if not isinstance(funding_list, (list, np.ndarray)) or len(funding_list) == 0:
                continue
            
            # Calculate average funding rate across exchanges
            funding_rates = []
            best_exchange = None
            best_rate = -float('inf')
            
            for exchange_data in funding_list:
                if isinstance(exchange_data, dict):
                    rate = exchange_data.get('funding_rate', 0)
                    interval = exchange_data.get('funding_rate_interval', 8)
                    exchange = exchange_data.get('exchange', '')
                    
                    if rate is not None and rate != 0:
                        funding_rates.append((rate, interval))
                        annualized = calculate_annualized_rate(rate, interval)
                        if annualized and annualized > best_rate:
                            best_rate = annualized
                            best_exchange = exchange
            
            if not funding_rates or spot_price is None or futures_price is None:
                continue
            
            # Calculate average funding rate
            avg_rate = np.mean([r[0] for r in funding_rates])
            avg_interval = 8  # Default to 8 hours
            
            # Calculate metrics
            metrics = calculate_basis_trade_metrics(
                asset, spot_price, futures_price, avg_rate, avg_interval
            )
            metrics['best_exchange'] = best_exchange
            metrics['best_rate'] = best_rate
            
            # Add volume metrics
            volume_metrics = calculate_volume_weighted_funding(data, asset)
            if volume_metrics:
                metrics['volume_weighted_rate'] = volume_metrics.get('volume_weighted_rate')
                metrics['total_volume'] = volume_metrics.get('total_volume', 0)
                metrics['exchange_count'] = volume_metrics.get('exchange_count', 0)
                metrics['annualized_volume_weighted'] = volume_metrics.get('annualized_weighted_rate')
            
            # Add volume imbalance metrics
            imbalance_metrics = analyze_volume_imbalance(data, asset)
            if imbalance_metrics:
                metrics['buy_ratio'] = imbalance_metrics.get('current_buy_ratio', 0.5)
                metrics['volume_volatility'] = imbalance_metrics.get('volume_volatility', 0)
            
            # Add volume momentum metrics
            momentum_metrics = calculate_volume_momentum(data, asset)
            if momentum_metrics:
                metrics['volume_momentum_7d'] = momentum_metrics.get('momentum_7d', 0)
                metrics['volume_momentum_30d'] = momentum_metrics.get('momentum_30d', 0)
            
            # Add OI metrics
            oi_metrics = calculate_price_weighted_oi(data, asset)
            if oi_metrics:
                metrics['oi_per_price'] = oi_metrics.get('oi_per_price', 0)
                metrics['oi_1d_change'] = oi_metrics.get('oi_1d_change', 0)
                metrics['oi_7d_change'] = oi_metrics.get('oi_7d_change', 0)
                metrics['oi_exchange_count'] = oi_metrics.get('exchange_count', 0)
            
            returns_data.append(metrics)
        
        # Create DataFrame
        if returns_data:
            df = pd.DataFrame(returns_data)
            # Sort by annualized return descending
            df = df.sort_values('annualized_return', ascending=False)
            return df
        
    except Exception as e:
        logger.error(f"Error preparing basis returns table: {e}")
    
    return pd.DataFrame()

def create_returns_history_chart(data, assets, timeframe='7d'):
    """
    Create line chart of historical annualized returns.
    
    Parameters:
    -----------
    data : dict
        Loaded data dictionary
    assets : list
        List of assets to include
    timeframe : str
        Time period to display
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Line chart of returns
    """
    fig = go.Figure()
    
    try:
        # This would require historical funding rate data processing
        # For now, create a placeholder
        st.info("Historical returns chart will be implemented with historical funding rate data")
        
    except Exception as e:
        logger.error(f"Error creating returns history chart: {e}")
    
    return fig

def create_key_metrics(data):
    """
    Create key metrics cards for the dashboard.
    
    Parameters:
    -----------
    data : dict
        Loaded data dictionary
    
    Returns:
    --------
    dict
        Dictionary of metrics to display
    """
    metrics = {
        'avg_return': 0,
        'best_asset': 'N/A',
        'best_return': 0,
        'positive_count': 0
    }
    
    try:
        # Get returns table
        returns_df = prepare_basis_returns_table(data)
        
        if not returns_df.empty:
            metrics['avg_return'] = returns_df['annualized_return'].mean()
            best_row = returns_df.iloc[0]
            metrics['best_asset'] = best_row['asset']
            metrics['best_return'] = best_row['annualized_return']
            metrics['positive_count'] = len(returns_df[returns_df['annualized_return'] > 0])
            
    except Exception as e:
        logger.error(f"Error calculating key metrics: {e}")
    
    return metrics

# Volume Analysis Functions
def calculate_volume_weighted_funding(data, asset):
    """
    Calculate volume-weighted average funding rates for an asset.
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    asset : str
        Asset symbol
        
    Returns:
    --------
    dict
        Volume-weighted funding metrics
    """
    try:
        if 'current_funding_rates' not in data or 'volume' not in data:
            return {}
        
        # Get funding rate data
        funding_df = data['current_funding_rates']
        asset_funding = funding_df[funding_df['symbol'] == asset]
        
        if asset_funding.empty:
            return {}
        
        # Get market data for volume information
        if 'futures' in data and asset in data['futures']:
            market_key = f'api_futures_pairs_markets_{asset}'
            if market_key in data['futures'][asset]:
                market_df = data['futures'][asset][market_key]
                
                # Create exchange volume mapping
                exchange_volumes = {}
                for _, row in market_df.iterrows():
                    exchange = row.get('exchange_name', '')
                    volume = row.get('volume_usd', 0)
                    if exchange and volume > 0:
                        exchange_volumes[exchange.upper()] = volume
                
                # Calculate volume-weighted funding rate
                funding_list = asset_funding.iloc[0]['stablecoin_margin_list']
                if isinstance(funding_list, (list, np.ndarray)):
                    total_weighted_rate = 0
                    total_volume = 0
                    
                    for exchange_data in funding_list:
                        if isinstance(exchange_data, dict):
                            exchange = exchange_data.get('exchange', '').upper()
                            rate = exchange_data.get('funding_rate', 0)
                            volume = exchange_volumes.get(exchange, 0)
                            
                            if volume > 0 and rate is not None:
                                total_weighted_rate += rate * volume
                                total_volume += volume
                    
                    if total_volume > 0:
                        weighted_avg_rate = total_weighted_rate / total_volume
                        return {
                            'volume_weighted_rate': weighted_avg_rate,
                            'total_volume': total_volume,
                            'exchange_count': len(exchange_volumes),
                            'annualized_weighted_rate': calculate_annualized_rate(weighted_avg_rate, 8)
                        }
        
        return {}
        
    except Exception as e:
        logger.error(f"Error calculating volume-weighted funding for {asset}: {e}")
        return {}

def analyze_volume_imbalance(data, asset):
    """
    Calculate buy/sell imbalance trends for an asset.
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    asset : str
        Asset symbol
        
    Returns:
    --------
    dict
        Volume imbalance metrics
    """
    try:
        if 'volume' not in data or asset not in data['volume']:
            return {}
        
        volume_df = data['volume'][asset].copy()
        
        if 'time' in volume_df.columns:
            volume_df['datetime'] = pd.to_datetime(volume_df['time'], unit='ms')
            volume_df = volume_df.sort_values('datetime')
        
        # Calculate imbalance metrics
        if 'aggregated_buy_volume_usd' in volume_df.columns and 'aggregated_sell_volume_usd' in volume_df.columns:
            volume_df['total_volume'] = volume_df['aggregated_buy_volume_usd'] + volume_df['aggregated_sell_volume_usd']
            volume_df['buy_ratio'] = volume_df['aggregated_buy_volume_usd'] / volume_df['total_volume']
            volume_df['imbalance'] = (volume_df['aggregated_buy_volume_usd'] - volume_df['aggregated_sell_volume_usd']) / volume_df['total_volume']
            
            # Calculate recent metrics (last 30 days)
            recent_df = volume_df.tail(30)
            
            return {
                'current_buy_ratio': volume_df['buy_ratio'].iloc[-1] if not volume_df.empty else 0,
                'avg_buy_ratio_30d': recent_df['buy_ratio'].mean(),
                'imbalance_trend': recent_df['imbalance'].mean(),
                'volume_volatility': recent_df['total_volume'].std() / recent_df['total_volume'].mean() if recent_df['total_volume'].mean() > 0 else 0,
                'current_total_volume': volume_df['total_volume'].iloc[-1] if not volume_df.empty else 0
            }
        
        return {}
        
    except Exception as e:
        logger.error(f"Error analyzing volume imbalance for {asset}: {e}")
        return {}

def calculate_volume_momentum(data, asset, periods=[7, 30]):
    """
    Calculate volume momentum indicators for an asset.
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    asset : str
        Asset symbol
    periods : list
        List of periods for moving averages
        
    Returns:
    --------
    dict
        Volume momentum metrics
    """
    try:
        if 'volume' not in data or asset not in data['volume']:
            return {}
        
        volume_df = data['volume'][asset].copy()
        
        if 'time' in volume_df.columns:
            volume_df['datetime'] = pd.to_datetime(volume_df['time'], unit='ms')
            volume_df = volume_df.sort_values('datetime')
        
        if 'aggregated_buy_volume_usd' in volume_df.columns and 'aggregated_sell_volume_usd' in volume_df.columns:
            volume_df['total_volume'] = volume_df['aggregated_buy_volume_usd'] + volume_df['aggregated_sell_volume_usd']
            
            momentum_metrics = {}
            
            for period in periods:
                if len(volume_df) >= period:
                    ma = volume_df['total_volume'].rolling(window=period).mean()
                    current_vol = volume_df['total_volume'].iloc[-1]
                    recent_ma = ma.iloc[-1]
                    
                    momentum_metrics[f'ma_{period}d'] = recent_ma
                    momentum_metrics[f'momentum_{period}d'] = (current_vol / recent_ma - 1) * 100 if recent_ma > 0 else 0
            
            return momentum_metrics
        
        return {}
        
    except Exception as e:
        logger.error(f"Error calculating volume momentum for {asset}: {e}")
        return {}

def calculate_price_weighted_oi(data, asset):
    """
    Calculate price-normalized open interest metrics.
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary  
    asset : str
        Asset symbol
        
    Returns:
    --------
    dict
        Price-weighted OI metrics
    """
    try:
        if 'open_interest' not in data:
            return {}
        
        oi_history_key = f'{asset}_history'
        if oi_history_key in data['open_interest']:
            oi_df = data['open_interest'][oi_history_key].copy()
            
            if 'timestamp' in oi_df.columns and 'price' in oi_df.columns:
                oi_df['datetime'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
                oi_df = oi_df.sort_values('datetime')
                
                # Calculate total OI across all exchanges
                exchange_cols = [col for col in oi_df.columns if col not in ['timestamp', 'price', 'datetime']]
                oi_df['total_oi'] = oi_df[exchange_cols].sum(axis=1, skipna=True)
                
                # Calculate price-normalized OI
                oi_df['oi_per_price'] = oi_df['total_oi'] / oi_df['price']
                
                # Calculate momentum metrics
                if len(oi_df) > 1:
                    oi_1d_change = (oi_df['total_oi'].iloc[-1] / oi_df['total_oi'].iloc[-2] - 1) * 100 if oi_df['total_oi'].iloc[-2] > 0 else 0
                    oi_7d_change = (oi_df['total_oi'].iloc[-1] / oi_df['total_oi'].iloc[-8] - 1) * 100 if len(oi_df) >= 8 and oi_df['total_oi'].iloc[-8] > 0 else 0
                    
                    return {
                        'current_oi': oi_df['total_oi'].iloc[-1],
                        'current_price': oi_df['price'].iloc[-1],
                        'oi_per_price': oi_df['oi_per_price'].iloc[-1],
                        'oi_1d_change': oi_1d_change,
                        'oi_7d_change': oi_7d_change,
                        'exchange_count': len(exchange_cols)
                    }
        
        return {}
        
    except Exception as e:
        logger.error(f"Error calculating price-weighted OI for {asset}: {e}")
        return {}

# Copy funding rate functions from report page
def prepare_current_funding_rate_table(data, selected_assets=None, selected_exchanges=None, 
                                      rate_display='annualized'):
    """
    Prepare current funding rate data for table display.
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    selected_assets : list, optional
        List of assets to display
    selected_exchanges : list, optional
        List of exchanges to display
    rate_display : str
        'annualized' (default) or 'current'
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame ready for table display
    """
    if 'current_funding_rates' not in data or data['current_funding_rates'] is None:
        return pd.DataFrame()
    
    try:
        df = data['current_funding_rates'].copy()
        
        # Use stablecoin margin column
        margin_col = 'stablecoin_margin_list'
        if margin_col not in df.columns:
            logger.error(f"Column {margin_col} not found in current funding rates data")
            return pd.DataFrame()
        
        # Initialize list to store processed data
        processed_data = []
        
        # Process each asset
        for idx, row in df.iterrows():
            symbol = row['symbol']
            
            # Skip if asset filter is applied and asset not selected
            if selected_assets and symbol not in selected_assets:
                continue
            
            # Get funding data for the margin type
            funding_list = row[margin_col]
            if not isinstance(funding_list, (list, np.ndarray)) or len(funding_list) == 0:
                continue
            
            # Create a dictionary to store exchange rates for this asset
            asset_data = {'Asset': symbol}
            
            # Process each exchange
            for exchange_data in funding_list:
                if not isinstance(exchange_data, dict):
                    continue
                    
                exchange = exchange_data.get('exchange', '')
                
                # Skip if exchange filter is applied and exchange not selected
                if selected_exchanges and exchange not in selected_exchanges:
                    continue
                
                funding_rate = exchange_data.get('funding_rate', None)
                interval = exchange_data.get('funding_rate_interval', 8.0)
                
                # Calculate rate based on display mode
                if rate_display == 'annualized' and funding_rate is not None:
                    display_rate = calculate_annualized_rate(funding_rate, interval)
                else:
                    display_rate = funding_rate
                
                # Store the rate
                asset_data[exchange] = display_rate
            
            # Only add if we have at least one exchange rate
            if len(asset_data) > 1:  # More than just the 'Asset' key
                processed_data.append(asset_data)
        
        # Create DataFrame from processed data
        result_df = pd.DataFrame(processed_data)
        
        # Sort by asset symbol
        if not result_df.empty:
            result_df = result_df.sort_values('Asset')
            
            # Limit to top 20 assets if no specific assets selected
            if not selected_assets and len(result_df) > 20:
                result_df = result_df.head(20)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error preparing current funding rate table: {e}")
        return pd.DataFrame()

def prepare_accumulated_funding_rate_table(data, selected_assets=None, selected_exchanges=None):
    """
    Prepare accumulated funding rate data for table display.
    Note: Data represents 365 days of accumulated funding.
    
    Parameters:
    -----------
    data : dict
        The loaded data dictionary
    selected_assets : list, optional
        List of assets to display
    selected_exchanges : list, optional
        List of exchanges to display
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame ready for table display with summary row
    """
    if 'accumulated_funding_rates' not in data or data['accumulated_funding_rates'] is None:
        return pd.DataFrame()
    
    try:
        df = data['accumulated_funding_rates'].copy()
        
        # Use stablecoin margin column
        margin_col = 'stablecoin_margin_list'
        if margin_col not in df.columns:
            logger.error(f"Column {margin_col} not found in accumulated funding rates data")
            return pd.DataFrame()
        
        # Initialize list to store processed data
        processed_data = []
        
        # Process each asset
        for idx, row in df.iterrows():
            symbol = row['symbol']
            
            # Skip if asset filter is applied and asset not selected
            if selected_assets and symbol not in selected_assets:
                continue
            
            # Get funding data for the margin type
            funding_list = row[margin_col]
            if not isinstance(funding_list, (list, np.ndarray)) or len(funding_list) == 0:
                continue
            
            # Create a dictionary to store exchange rates for this asset
            asset_data = {'Asset': symbol}
            
            # Process each exchange
            for exchange_data in funding_list:
                if not isinstance(exchange_data, dict):
                    continue
                    
                exchange = exchange_data.get('exchange', '')
                
                # Skip if exchange filter is applied and exchange not selected
                if selected_exchanges and exchange not in selected_exchanges:
                    continue
                
                funding_rate = exchange_data.get('funding_rate', None)
                
                # Store the accumulated rate (already represents 365 days)
                asset_data[exchange] = funding_rate
            
            # Only add if we have at least one exchange rate
            if len(asset_data) > 1:  # More than just the 'Asset' key
                processed_data.append(asset_data)
        
        # Create DataFrame from processed data
        result_df = pd.DataFrame(processed_data)
        
        if not result_df.empty:
            # Sort by asset symbol
            result_df = result_df.sort_values('Asset')
            
            # Limit to top 20 assets if no specific assets selected
            if not selected_assets and len(result_df) > 20:
                result_df = result_df.head(20)
            
            # Calculate average for each exchange (summary row)
            exchange_cols = [col for col in result_df.columns if col != 'Asset']
            if exchange_cols:
                avg_data = {'Asset': 'Average'}
                for exchange in exchange_cols:
                    # Calculate average, ignoring NaN values
                    avg_data[exchange] = result_df[exchange].mean()
                
                # Add summary row
                result_df = pd.concat([result_df, pd.DataFrame([avg_data])], ignore_index=True)
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error preparing accumulated funding rate table: {e}")
        return pd.DataFrame()

def main():
    """Main function to render the basis trading page."""
    
    # Render sidebar
    render_sidebar()
    
    # Page header
    st.title("Basis Trading Analysis (In Progress)")
    st.write("Comprehensive analysis of basis trading opportunities across the top 10 cryptocurrencies")
    
    # Display latest data date
    latest_dir = get_latest_data_directory()
    if latest_dir:
        date_str = os.path.basename(latest_dir)
        try:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            formatted_date = date_obj.strftime('%Y-%m-%d')
            st.caption(f"Latest data from: {formatted_date}")
        except ValueError:
            st.caption(f"Latest data from: {date_str}")
    
    # Add introduction to basis trading
    with st.expander("What is Basis Trading?", expanded=False):
        st.markdown("""
        **Basis trading** is a market-neutral strategy that profits from the price difference between spot and perpetual futures markets.
        
        **How it works:**
        1. **Long Spot**: Buy the cryptocurrency in the spot market
        2. **Short Perpetual**: Sell an equal amount in the perpetual futures market
        3. **Collect Funding**: Earn funding payments (when positive) paid by long positions to short positions
        
        **Key advantages:**
        - Market neutral: Profit regardless of price direction
        - Predictable income: Funding payments occur every 8 hours
        - Low risk: Main risks are exchange/counterparty and funding rate changes
        
        **Funding rate mechanics:**
        - Positive funding: Longs pay shorts (bullish market sentiment)
        - Negative funding: Shorts pay longs (bearish market sentiment)
        - Rates adjust to keep perpetual prices aligned with spot prices
        """)
    
    # Load data
    with st.spinner("Loading basis trading data..."):
        data = load_basis_data()
    
    if not data:
        st.error("No data available. Please check your data sources.")
        return
    
    # Calculate key metrics
    metrics = create_key_metrics(data)
    
    # Display key metrics dashboard
    st.header("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        display_metric_card(
            "Average Return",
            format_percentage(metrics['avg_return'], precision=2, show_plus=True),
            "Across Top 10 Assets"
        )
    
    with col2:
        display_metric_card(
            "Best Opportunity",
            metrics['best_asset'],
            format_percentage(metrics['best_return'], precision=2, show_plus=True)
        )
    
    with col3:
        display_metric_card(
            "Positive Returns",
            f"{metrics['positive_count']}/10",
            "Assets with positive funding"
        )
    
    with col4:
        display_metric_card(
            "Data Status",
            "Live",
            "Real-time funding rates"
        )
    
    # Add key metrics explanation
    st.caption("""
    **Key Metrics Explained:** Average Return shows the mean annualized return across all top 10 assets | 
    Best Opportunity highlights the asset with the highest current funding rate | 
    Positive Returns counts how many assets have positive funding (profitable for basis trades) | 
    Data Status confirms real-time data freshness
    """)
    
    # Basis Trade Return Analysis
    st.header("Basis Trade Returns - Top 10 Cryptocurrencies")
    
    returns_df = prepare_basis_returns_table(data, top_n=10)
    
    if not returns_df.empty:
        # Prepare display dataframe with enhanced metrics
        display_df = pd.DataFrame({
            'Asset': returns_df['asset'],
            'Spot Price': returns_df['spot_price'],
            'Futures Price': returns_df['futures_price'],
            'Current Basis %': returns_df['basis_pct'],
            'Avg Funding Rate': returns_df['funding_rate'],
            'Annualized Return %': returns_df['annualized_return'],
            'Volume Weighted Rate': returns_df.get('volume_weighted_rate', 0),
            '24h Volume': returns_df.get('total_volume', 0),
            'Buy Ratio': returns_df.get('buy_ratio', 0.5),
            'Vol Momentum 7d': returns_df.get('volume_momentum_7d', 0),
            'OI Change 7d': returns_df.get('oi_7d_change', 0),
            'Exchange Count': returns_df.get('exchange_count', 0),
            'Best Exchange': returns_df['best_exchange']
        })
        
        # Create format dictionary
        format_dict = {
            'Spot Price': lambda x: format_currency(x, precision=2),
            'Futures Price': lambda x: format_currency(x, precision=2),
            'Current Basis %': lambda x: format_percentage(x, precision=3, show_plus=True),
            'Avg Funding Rate': lambda x: f"{x:.4f}%" if pd.notnull(x) else 'N/A',
            'Annualized Return %': lambda x: format_percentage(x, precision=2, show_plus=True),
            'Volume Weighted Rate': lambda x: f"{x:.4f}%" if pd.notnull(x) else 'N/A',
            '24h Volume': lambda x: format_currency(x, abbreviate=True) if pd.notnull(x) else 'N/A',
            'Buy Ratio': lambda x: f"{x:.1%}" if pd.notnull(x) else 'N/A',
            'Vol Momentum 7d': lambda x: f"{x:+.1f}%" if pd.notnull(x) else 'N/A',
            'OI Change 7d': lambda x: f"{x:+.1f}%" if pd.notnull(x) else 'N/A',
            'Exchange Count': lambda x: f"{int(x)}" if pd.notnull(x) else '0'
        }
        
        # Display table
        create_formatted_table(
            display_df,
            format_dict=format_dict,
            emphasize_negatives=True,
            compact_display=True
        )
        
        st.caption("Note: Annualized returns are calculated as: Funding Rate × (24 / Funding Interval) × 365")
        
        # Add descriptive text
        st.markdown("""
        **Understanding this enhanced table:**
        - **Spot Price**: Current price in the spot market where assets are bought/sold for immediate delivery
        - **Futures Price**: Current price in the perpetual futures market
        - **Current Basis %**: The percentage difference between futures and spot prices (premium/discount)
        - **Avg Funding Rate**: Average funding rate across all exchanges (8-hour rate)
        - **Annualized Return %**: Projected yearly return from collecting funding payments
        - **Volume Weighted Rate**: Funding rate weighted by exchange trading volume (more accurate for liquid trades)
        - **24h Volume**: Total trading volume across all exchanges
        - **Buy Ratio**: Percentage of volume that is buy orders (>50% = bullish bias)
        - **Vol Momentum 7d**: 7-day volume momentum (positive = increasing activity)
        - **OI Change 7d**: 7-day open interest change (positive = growing market interest)
        - **Exchange Count**: Number of exchanges offering this asset
        - **Best Exchange**: Exchange offering the highest annualized funding rate
        
        Positive funding rates indicate longs pay shorts, making basis trades (long spot + short perp) profitable.
        Higher volume and stable OI growth indicate more reliable funding rate sustainability.
        """)
    else:
        st.warning("No basis trade data available for the selected assets.")
    
    # Volume Analysis Dashboard
    st.header("Volume Analysis Dashboard")
    
    if 'aggregated_volume_history' in data and data['aggregated_volume_history'] is not None:
        volume_df = data['aggregated_volume_history'].copy()
        volume_df['datetime'] = pd.to_datetime(volume_df['time'], unit='ms')
        volume_df = volume_df.sort_values('datetime')
        
        # Calculate volume metrics
        volume_df['total_volume'] = volume_df['aggregated_buy_volume_usd'] + volume_df['aggregated_sell_volume_usd']
        volume_df['buy_ratio'] = volume_df['aggregated_buy_volume_usd'] / volume_df['total_volume']
        volume_df['net_flow'] = volume_df['aggregated_buy_volume_usd'] - volume_df['aggregated_sell_volume_usd']
        
        # Display volume metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_volume = volume_df['total_volume'].iloc[-1] if not volume_df.empty else 0
            display_metric_card(
                "Current 4h Volume",
                format_currency(current_volume, abbreviate=True),
                "Cross-market total"
            )
        
        with col2:
            current_buy_ratio = volume_df['buy_ratio'].iloc[-1] if not volume_df.empty else 0.5
            display_metric_card(
                "Current Buy Ratio",
                f"{current_buy_ratio:.1%}",
                "Bullish" if current_buy_ratio > 0.52 else "Bearish" if current_buy_ratio < 0.48 else "Neutral"
            )
        
        with col3:
            avg_volume_7d = volume_df['total_volume'].tail(42).mean() if len(volume_df) >= 42 else current_volume  # 7 days * 6 (4h intervals)
            volume_change = (current_volume / avg_volume_7d - 1) * 100 if avg_volume_7d > 0 else 0
            display_metric_card(
                "Volume vs 7d Avg",
                f"{volume_change:+.1f}%",
                "Above average" if volume_change > 0 else "Below average"
            )
        
        with col4:
            net_flow_24h = volume_df['net_flow'].tail(6).sum() if len(volume_df) >= 6 else 0  # Last 24h (6 * 4h)
            display_metric_card(
                "24h Net Flow",
                f"${net_flow_24h/1e9:.1f}B" if abs(net_flow_24h) > 1e9 else f"${net_flow_24h/1e6:.0f}M",
                "Buy pressure" if net_flow_24h > 0 else "Sell pressure" if net_flow_24h < 0 else "Balanced"
            )
        
        # Volume flow chart
        st.subheader("Volume Flow Analysis")
        
        # Show last 30 days of data
        recent_volume = volume_df.tail(180)  # 30 days * 6 intervals
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recent_volume['datetime'],
            y=recent_volume['aggregated_buy_volume_usd'],
            name='Buy Volume',
            fill='tonexty',
            fillcolor='rgba(0, 255, 0, 0.3)',
            line=dict(color='green', width=1.5)
        ))
        
        fig.add_trace(go.Scatter(
            x=recent_volume['datetime'],
            y=-recent_volume['aggregated_sell_volume_usd'],  # Negative for visual separation
            name='Sell Volume',
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=1.5)
        ))
        
        fig.update_layout(
            title="Buy vs Sell Volume Flow (Last 30 Days)",
            xaxis_title=None,
            yaxis_title="Volume (USD)",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        display_chart(apply_chart_theme(fig))
        
        st.markdown("""
        **Volume Analysis Insights:**
        - **Buy Ratio >52%**: Strong bullish sentiment, funding rates likely to increase
        - **High Volume + Positive Net Flow**: Strong demand, good for long-term basis trades
        - **Volume Momentum**: Increasing volume suggests growing market interest
        - **Volume Volatility**: High volatility may indicate unstable funding rates
        """)
    
    # Open Interest Analysis Dashboard  
    st.header("Open Interest Analysis Dashboard")
    
    # Create summary OI metrics across top assets
    oi_summary_data = []
    for asset in TOP_10_ASSETS[:4]:  # Top 4 assets for performance
        oi_metrics = calculate_price_weighted_oi(data, asset)
        if oi_metrics:
            oi_summary_data.append({
                'Asset': asset,
                'Current OI': oi_metrics.get('current_oi', 0),
                'Price': oi_metrics.get('current_price', 0),
                'OI/Price': oi_metrics.get('oi_per_price', 0),
                '1d Change': oi_metrics.get('oi_1d_change', 0),
                '7d Change': oi_metrics.get('oi_7d_change', 0),
                'Exchanges': oi_metrics.get('exchange_count', 0)
            })
    
    if oi_summary_data:
        oi_df = pd.DataFrame(oi_summary_data)
        
        # Display OI summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_oi = oi_df['Current OI'].sum()
            display_metric_card(
                "Total OI (Top 4)",
                format_currency(total_oi, abbreviate=True),
                "BTC, ETH, SOL, XRP"
            )
        
        with col2:
            avg_1d_change = oi_df['1d Change'].mean()
            display_metric_card(
                "Avg OI Change (1d)",
                f"{avg_1d_change:+.1f}%",
                "Growing" if avg_1d_change > 1 else "Stable" if avg_1d_change > -1 else "Declining"
            )
        
        with col3:
            avg_7d_change = oi_df['7d Change'].mean()
            display_metric_card(
                "Avg OI Change (7d)",
                f"{avg_7d_change:+.1f}%",
                "Strong growth" if avg_7d_change > 5 else "Growing" if avg_7d_change > 0 else "Declining"
            )
        
        with col4:
            avg_exchanges = oi_df['Exchanges'].mean()
            display_metric_card(
                "Avg Exchange Count",
                f"{avg_exchanges:.0f}",
                "Good distribution" if avg_exchanges >= 15 else "Limited"
            )
        
        # OI Summary Table
        st.subheader("Open Interest Summary")
        
        display_oi_df = pd.DataFrame({
            'Asset': oi_df['Asset'],
            'Current OI': oi_df['Current OI'],
            'Price': oi_df['Price'], 
            'OI/Price Ratio': oi_df['OI/Price'],
            '1d Change': oi_df['1d Change'],
            '7d Change': oi_df['7d Change'],
            'Exchange Count': oi_df['Exchanges']
        })
        
        oi_format_dict = {
            'Current OI': lambda x: format_currency(x, abbreviate=True),
            'Price': lambda x: format_currency(x, precision=2),
            'OI/Price Ratio': lambda x: f"{x:.0f}" if pd.notnull(x) else 'N/A',
            '1d Change': lambda x: f"{x:+.1f}%" if pd.notnull(x) else 'N/A',
            '7d Change': lambda x: f"{x:+.1f}%" if pd.notnull(x) else 'N/A',
            'Exchange Count': lambda x: f"{int(x)}" if pd.notnull(x) else '0'
        }
        
        create_formatted_table(
            display_oi_df,
            format_dict=oi_format_dict,
            emphasize_negatives=True,
            compact_display=True
        )
        
        st.markdown("""
        **Open Interest Analysis Insights:**
        - **Rising OI + Stable Funding**: Sustainable basis trading environment
        - **OI/Price Ratio**: Higher ratios indicate more leveraged markets
        - **Exchange Distribution**: More exchanges = better liquidity and lower counterparty risk
        - **OI Momentum**: Consistent growth suggests growing institutional interest
        """)
    
    # Historical Basis Chart (if data available)
    if 'basis_history' in data and data['basis_history'] is not None:
        st.header("Historical Basis Spread")
        
        basis_df = data['basis_history']
        basis_df = process_timestamps(basis_df)
        
        if not basis_df.empty:
            # Create time series chart
            fig = create_time_series(
                df=basis_df,
                x_col='datetime',
                y_col='close_basis',
                title="Historical Basis Spread (%)",
                y_title="Basis %"
            )
            
            display_chart(fig)
            
            st.markdown("""
            **Understanding this chart:**
            - Shows the historical difference between futures and spot prices as a percentage
            - Positive values indicate futures trading at a premium (contango)
            - Negative values indicate futures trading at a discount (backwardation)
            - Persistent positive basis suggests bullish market sentiment
            - Basis convergence to zero occurs at futures expiry (for dated futures)
            """)
        else:
            st.info("No historical basis data available")
    
    # Funding Rate Analysis Section
    st.header("Funding Rate Analysis")
    
    # Create filter controls
    filter_col1, filter_col2 = st.columns(2)
    
    with filter_col1:
        # Get all available assets
        all_assets = []
        if 'current_funding_rates' in data and data['current_funding_rates'] is not None:
            if 'symbol' in data['current_funding_rates'].columns:
                all_assets = sorted(data['current_funding_rates']['symbol'].unique().tolist())
        
        # Multi-select for assets (default to top 20)
        default_assets = all_assets[:20] if len(all_assets) > 20 else all_assets
        selected_assets = st.multiselect(
            "Select Assets",
            options=all_assets,
            default=default_assets,
            key="funding_assets_basis"
        )
    
    with filter_col2:
        # Get all available exchanges
        all_exchanges = set()
        if 'current_funding_rates' in data and data['current_funding_rates'] is not None:
            for idx, row in data['current_funding_rates'].iterrows():
                if 'stablecoin_margin_list' in row and isinstance(row['stablecoin_margin_list'], (list, np.ndarray)):
                    for exchange_data in row['stablecoin_margin_list']:
                        if isinstance(exchange_data, dict) and 'exchange' in exchange_data:
                            all_exchanges.add(exchange_data['exchange'])
        
        all_exchanges = sorted(list(all_exchanges))
        selected_exchanges = st.multiselect(
            "Select Exchanges",
            options=all_exchanges,
            default=all_exchanges[:10] if len(all_exchanges) > 10 else all_exchanges,
            key="funding_exchanges_basis"
        )
    
    # Rate display type selector
    rate_display = st.radio(
        "Rate Display",
        options=['annualized', 'current'],
        format_func=lambda x: 'Annualized (Default)' if x == 'annualized' else 'Current Rate',
        horizontal=True,
        key="funding_rate_display_basis"
    )
    
    # Current Funding Rates Table
    st.subheader("Current Funding Rates by Exchange")
    
    if rate_display == 'annualized':
        st.caption("Showing annualized rates calculated as: Current Rate × (24 / Funding Interval) × 365")
    else:
        st.caption("Showing raw current funding rates as provided by exchanges")
    
    current_df = prepare_current_funding_rate_table(
        data,
        selected_assets=selected_assets if selected_assets else None,
        selected_exchanges=selected_exchanges if selected_exchanges else None,
        rate_display=rate_display
    )
    
    if not current_df.empty:
        # Create format dictionary
        format_dict = {}
        for col in current_df.columns:
            if col != 'Asset':
                if rate_display == 'annualized':
                    format_dict[col] = lambda x: f"{x:.2f}%" if pd.notnull(x) else 'N/A'
                else:
                    format_dict[col] = lambda x: f"{x:.4f}%" if pd.notnull(x) else 'N/A'
        
        create_formatted_table(
            current_df,
            format_dict=format_dict,
            emphasize_negatives=True,
            compact_display=True
        )
        
        st.markdown("""
        **Understanding current funding rates:**
        - Funding rates are periodic payments between long and short positions in perpetual futures
        - Positive rates: Long positions pay short positions (bullish market)
        - Negative rates: Short positions pay long positions (bearish market)
        - Most exchanges use 8-hour funding intervals (3 payments per day)
        - Annualized rates help compare returns with traditional investments
        - Higher funding rates indicate stronger directional bias in the market
        """)
    else:
        st.info("No current funding rate data available for the selected filters.")
    
    # Accumulated Funding Rates Table
    st.subheader("365-Day Accumulated Funding")
    st.caption("Total funding accumulated over the past year (365 days)")
    
    accumulated_df = prepare_accumulated_funding_rate_table(
        data,
        selected_assets=selected_assets if selected_assets else None,
        selected_exchanges=selected_exchanges if selected_exchanges else None
    )
    
    if not accumulated_df.empty:
        # Create format dictionary
        format_dict = {}
        for col in accumulated_df.columns:
            if col != 'Asset':
                format_dict[col] = lambda x: f"{x:.2f}%" if pd.notnull(x) else 'N/A'
        
        create_formatted_table(
            accumulated_df,
            format_dict=format_dict,
            emphasize_negatives=True,
            compact_display=True
        )
        
        st.caption("Note: The 'Average' row shows the mean accumulated funding rate across all displayed assets for each exchange.")
        
        st.markdown("""
        **Understanding accumulated funding:**
        - Shows total funding payments accumulated over 365 days
        - Represents the actual return a basis trader would have earned
        - Accounts for both positive and negative funding periods
        - Higher accumulated values indicate consistently positive funding
        - The 'Average' row helps identify exchanges with the best overall returns
        - Past performance does not guarantee future results
        """)
    else:
        st.info("No accumulated funding rate data available for the selected filters.")
    
    # Comprehensive Funding Rate Analysis
    st.header("📊 Comprehensive Funding Rate Analysis")
    st.write("Annualized funding rates across different time periods for top 20 crypto assets by market cap.")
    
    # Get data path for comprehensive analysis
    if not latest_dir.startswith('data'):
        comprehensive_data_path = os.path.join('data', latest_dir)
    else:
        comprehensive_data_path = latest_dir
    
    # Load comprehensive funding data
    comprehensive_funding_data = load_all_accumulated_funding_data(comprehensive_data_path)
    
    if any(not df.empty for df in comprehensive_funding_data.values()):
        # Display which periods have data
        available_periods = [period for period, df in comprehensive_funding_data.items() if not df.empty]
        st.info(f"📈 Available data periods: {', '.join(available_periods)}")
        
        # Period selector
        comp_col1, comp_col2 = st.columns([2, 1])
        
        with comp_col1:
            display_mode = st.radio(
                "Display Mode:",
                ["Summary View", "Detailed View"],
                horizontal=True,
                help="Summary shows top 20 entries by annualized rates. Detailed shows all top 20 crypto assets.",
                key="comp_display_mode"
            )
        
        with comp_col2:
            sort_period = st.selectbox(
                "Sort by Period:",
                ["365d", "30d", "7d", "1d"],
                help="Choose which time period to use for sorting",
                key="comp_sort_period"
            )
        
        # Create comprehensive table
        with st.spinner("Processing comprehensive funding data..."):
            comprehensive_table = prepare_comprehensive_funding_rate_table(comprehensive_funding_data)
        
        if not comprehensive_table.empty:
            # Apply filtering based on display mode
            if display_mode == "Summary View":
                # Show top 20 by selected period
                sort_col = f'{sort_period} Annualized (%)'
                if sort_col in comprehensive_table.columns:
                    # Convert percentage strings to numeric for sorting
                    comprehensive_table['sort_value'] = comprehensive_table[sort_col].str.replace('%', '').str.replace('N/A', '0').astype(float)
                    display_table = comprehensive_table.nlargest(20, 'sort_value').drop('sort_value', axis=1)
                else:
                    display_table = comprehensive_table.head(20)
                    st.warning(f"Sort column '{sort_col}' not found. Showing first 20 rows.")
            else:
                display_table = comprehensive_table
            
            # Format the table for display
            format_dict = {}
            for col in display_table.columns:
                if 'Annualized (%)' in col:
                    format_dict[col] = lambda x: x  # Already formatted as percentage strings
            
            st.subheader(f"Funding Rate Analysis - {display_mode}")
            create_formatted_table(
                display_table,
                format_dict=format_dict
            )
            
            # Add insights
            st.markdown("**💡 Key Insights:**")
            
            # Calculate some statistics
            insights = []
            for period in ['1d', '7d', '30d', '365d']:
                col_name = f'{period} Annualized (%)'
                if col_name in comprehensive_table.columns:
                    # Extract numeric values for analysis
                    numeric_values = comprehensive_table[col_name].str.replace('%', '').str.replace('N/A', '').replace('', np.nan)
                    numeric_values = pd.to_numeric(numeric_values, errors='coerce').dropna()
                    
                    if not numeric_values.empty:
                        avg_rate = numeric_values.mean()
                        max_rate = numeric_values.max()
                        max_asset_exchange = comprehensive_table.loc[
                            comprehensive_table[col_name] == f"{max_rate:.2f}%", 
                            ['Asset', 'Exchange']
                        ]
                        
                        if not max_asset_exchange.empty:
                            asset = max_asset_exchange.iloc[0]['Asset']
                            exchange = max_asset_exchange.iloc[0]['Exchange']
                            insights.append(f"**{period} Period**: Average annualized rate: {avg_rate:.2f}%, Highest: {max_rate:.2f}% ({asset} on {exchange})")
            
            for insight in insights:
                st.write(f"- {insight}")
            
            # Add download functionality
            if not comprehensive_table.empty:
                csv_data = comprehensive_table.to_csv(index=False)
                st.download_button(
                    label="📥 Download Comprehensive Funding Rate Data (CSV)",
                    data=csv_data,
                    file_name=f"comprehensive_funding_rates_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="download_comprehensive_funding"
                )
            
            # Explanation section
            st.markdown("""
            **📋 Understanding Annualized Funding Rates:**
            - **Asset Filter**: Only shows top 20 crypto assets by market cap (BTC, ETH, USDT, XRP, etc.)
            - **1d Period**: Daily accumulated rate × 365 = Annual equivalent
            - **7d Period**: Weekly accumulated rate × 52.14 = Annual equivalent  
            - **30d Period**: Monthly accumulated rate × 12.17 = Annual equivalent
            - **365d Period**: Already represents the full annual accumulated rate
            - All rates are displayed as annualized percentages for easy comparison
            - Higher values indicate more profitable funding collection opportunities
            - Consider both rate magnitude and consistency across time periods
            """)
        else:
            st.warning("No comprehensive funding rate data could be processed.")
    else:
        st.warning("No accumulated funding rate data found for any time period.")
    
    # Carry Trade Calculator
    st.header("Carry Trade Calculator")
    
    calc_col1, calc_col2, calc_col3, calc_col4 = st.columns(4)
    
    with calc_col1:
        calc_asset = st.selectbox(
            "Select Asset",
            options=TOP_10_ASSETS,
            key="calc_asset"
        )
    
    with calc_col2:
        position_size = st.number_input(
            "Position Size (USD)",
            min_value=1000,
            max_value=10000000,
            value=100000,
            step=10000,
            key="calc_size"
        )
    
    with calc_col3:
        holding_days = st.number_input(
            "Holding Period (days)",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            key="calc_days"
        )
    
    with calc_col4:
        st.write("")  # Spacer
        calc_button = st.button("Calculate Returns", key="calc_button")
    
    if calc_button:
        # Get the metrics for selected asset
        returns_df = prepare_basis_returns_table(data, top_n=10)
        asset_data = returns_df[returns_df['asset'] == calc_asset]
        
        if not asset_data.empty:
            row = asset_data.iloc[0]
            
            # Calculate returns
            annual_return_pct = row['annualized_return']
            daily_return_pct = row['daily_return']
            
            # Calculate dollar amounts
            daily_funding = position_size * (daily_return_pct / 100)
            total_funding = daily_funding * holding_days
            annual_funding = daily_funding * 365
            
            # Display results
            result_col1, result_col2, result_col3 = st.columns(3)
            
            with result_col1:
                st.metric(
                    "Annualized Return",
                    format_percentage(annual_return_pct, precision=2, show_plus=True)
                )
                st.metric(
                    "Daily Funding",
                    f"+{format_currency(daily_funding, precision=2)}" if daily_funding > 0 else format_currency(daily_funding, precision=2)
                )
            
            with result_col2:
                st.metric(
                    f"Total Return ({holding_days} days)",
                    f"+{format_currency(total_funding, precision=2)}" if total_funding > 0 else format_currency(total_funding, precision=2)
                )
                st.metric(
                    "Annual Funding Income",
                    f"+{format_currency(annual_funding, precision=2)}" if annual_funding > 0 else format_currency(annual_funding, precision=2)
                )
            
            with result_col3:
                st.metric(
                    "Required Capital",
                    format_currency(position_size * 2, precision=0),
                    help="Spot position + Margin for futures"
                )
                st.metric(
                    "Best Exchange",
                    row['best_exchange'] if pd.notnull(row['best_exchange']) else "N/A"
                )
        else:
            st.warning(f"No data available for {calc_asset}")
    
    # Add calculator explanation
    st.markdown("""
    **Understanding the Carry Trade Calculator:**
    - **Position Size**: The USD amount you want to allocate to the spot position
    - **Required Capital**: Total capital needed (2x position size for spot + futures margin)
    - **Daily Funding**: Expected daily income from funding payments
    - **Holding Period Return**: Total expected return over your selected timeframe
    - **Annual Funding Income**: Projected yearly income if rates remain constant
    
    **Important considerations:**
    - Funding rates are variable and can change rapidly
    - Negative funding periods will reduce returns
    - Transaction costs and slippage are not included
    - Exchange risk and counterparty risk should be considered
    - This is a market-neutral strategy but still carries risks
    """)
    
    # Comprehensive Futures Volume Analysis Section
    st.header("📊 Comprehensive Futures Volume Analysis")
    st.write("Deep dive into perpetual futures trading volume across assets, exchanges, and time periods.")
    
    # Check if volume data is available
    if 'volume' in data and data['volume']:
        # Create tabs for different views
        volume_tab1, volume_tab2, volume_tab3, volume_tab4 = st.tabs([
            "📈 Asset Volume Overview", 
            "🏢 Exchange Volume Distribution", 
            "📊 Volume Trends & Momentum",
            "🔄 Buy/Sell Flow Analysis"
        ])
        
        with volume_tab1:
            st.subheader("Asset Volume Overview")
            
            # Prepare asset volume summary
            asset_volume_summary = []
            for asset in TOP_10_ASSETS:
                if asset in data['volume']:
                    volume_df = data['volume'][asset].copy()
                    if not volume_df.empty and 'aggregated_buy_volume_usd' in volume_df.columns:
                        # Calculate metrics
                        volume_df['total_volume'] = volume_df['aggregated_buy_volume_usd'] + volume_df['aggregated_sell_volume_usd']
                        
                        # Get latest and average volumes
                        latest_volume = volume_df['total_volume'].iloc[-1] if len(volume_df) > 0 else 0
                        avg_volume_7d = volume_df['total_volume'].tail(7).mean() if len(volume_df) >= 7 else latest_volume
                        avg_volume_30d = volume_df['total_volume'].tail(30).mean() if len(volume_df) >= 30 else latest_volume
                        
                        # Calculate growth rates
                        volume_growth_7d = ((latest_volume / avg_volume_7d) - 1) * 100 if avg_volume_7d > 0 else 0
                        volume_growth_30d = ((latest_volume / avg_volume_30d) - 1) * 100 if avg_volume_30d > 0 else 0
                        
                        # Get buy/sell ratio
                        buy_ratio = volume_df['aggregated_buy_volume_usd'].iloc[-1] / volume_df['total_volume'].iloc[-1] if volume_df['total_volume'].iloc[-1] > 0 else 0.5
                        
                        # Check if we have exchange-specific data
                        exchange_count = 0
                        if f'{asset}_exchange_volume' in data['volume']:
                            exchange_df = data['volume'][f'{asset}_exchange_volume']
                            if not exchange_df.empty and 'exchange_list' in exchange_df.columns:
                                # Count unique exchanges
                                exchanges = set()
                                for idx, row in exchange_df.iterrows():
                                    if isinstance(row['exchange_list'], list):
                                        for ex in row['exchange_list']:
                                            if isinstance(ex, dict) and 'exchange' in ex:
                                                exchanges.add(ex['exchange'])
                                exchange_count = len(exchanges)
                        
                        asset_volume_summary.append({
                            'Asset': asset,
                            'Latest 24h Volume': latest_volume,
                            'Avg Volume (7d)': avg_volume_7d,
                            'Avg Volume (30d)': avg_volume_30d,
                            'Volume Change (7d)': volume_growth_7d,
                            'Volume Change (30d)': volume_growth_30d,
                            'Buy Ratio': buy_ratio,
                            'Exchange Count': exchange_count
                        })
            
            if asset_volume_summary:
                volume_summary_df = pd.DataFrame(asset_volume_summary)
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    total_volume_24h = volume_summary_df['Latest 24h Volume'].sum()
                    display_metric_card(
                        "Total 24h Volume",
                        format_currency(total_volume_24h, abbreviate=True),
                        "All Top 10 Assets"
                    )
                
                with col2:
                    avg_buy_ratio = volume_summary_df['Buy Ratio'].mean()
                    display_metric_card(
                        "Average Buy Ratio",
                        f"{avg_buy_ratio:.1%}",
                        "Bullish" if avg_buy_ratio > 0.52 else "Bearish" if avg_buy_ratio < 0.48 else "Neutral"
                    )
                
                with col3:
                    top_volume_asset = volume_summary_df.loc[volume_summary_df['Latest 24h Volume'].idxmax(), 'Asset']
                    top_volume = volume_summary_df['Latest 24h Volume'].max()
                    display_metric_card(
                        "Highest Volume Asset",
                        top_volume_asset,
                        format_currency(top_volume, abbreviate=True)
                    )
                
                with col4:
                    avg_growth_7d = volume_summary_df['Volume Change (7d)'].mean()
                    display_metric_card(
                        "Avg Volume Growth (7d)",
                        f"{avg_growth_7d:+.1f}%",
                        "Increasing" if avg_growth_7d > 5 else "Stable" if avg_growth_7d > -5 else "Decreasing"
                    )
                
                # Display detailed table
                st.subheader("Detailed Volume Metrics by Asset")
                
                volume_format_dict = {
                    'Latest 24h Volume': lambda x: format_currency(x, abbreviate=True),
                    'Avg Volume (7d)': lambda x: format_currency(x, abbreviate=True),
                    'Avg Volume (30d)': lambda x: format_currency(x, abbreviate=True),
                    'Volume Change (7d)': lambda x: f"{x:+.1f}%",
                    'Volume Change (30d)': lambda x: f"{x:+.1f}%",
                    'Buy Ratio': lambda x: f"{x:.1%}",
                    'Exchange Count': lambda x: f"{int(x)}" if x > 0 else "N/A"
                }
                
                create_formatted_table(
                    volume_summary_df,
                    format_dict=volume_format_dict,
                    emphasize_negatives=True,
                    compact_display=True
                )
                
                # Volume comparison chart
                st.subheader("24h Volume Comparison")
                
                fig_volume_comp = go.Figure()
                
                # Sort by volume for better visualization
                volume_summary_df_sorted = volume_summary_df.sort_values('Latest 24h Volume', ascending=True)
                
                fig_volume_comp.add_trace(go.Bar(
                    x=volume_summary_df_sorted['Latest 24h Volume'],
                    y=volume_summary_df_sorted['Asset'],
                    orientation='h',
                    text=[format_currency(v, abbreviate=True) for v in volume_summary_df_sorted['Latest 24h Volume']],
                    textposition='outside',
                    marker_color='rgba(99, 110, 250, 0.8)'
                ))
                
                fig_volume_comp.update_layout(
                    title="24-Hour Futures Volume by Asset",
                    xaxis_title="Volume (USD)",
                    yaxis_title=None,
                    height=400,
                    showlegend=False
                )
                
                display_chart(apply_chart_theme(fig_volume_comp))
                
                st.markdown("""
                **Volume Analysis Insights:**
                - Higher volumes indicate more liquid markets with tighter spreads
                - Buy ratio >52% suggests bullish sentiment, <48% suggests bearish sentiment
                - Volume growth shows increasing market interest and trading activity
                - Exchange count indicates market breadth and accessibility
                """)
        
        with volume_tab2:
            st.subheader("Exchange Volume Distribution")
            
            # Aggregate exchange volumes across all assets
            exchange_volumes = {}
            
            for asset in TOP_10_ASSETS:
                if f'{asset}_exchange_volume' in data['volume']:
                    exchange_df = data['volume'][f'{asset}_exchange_volume']
                    if not exchange_df.empty and 'exchange_list' in exchange_df.columns:
                        for idx, row in exchange_df.iterrows():
                            if isinstance(row['exchange_list'], list):
                                for ex in row['exchange_list']:
                                    if isinstance(ex, dict):
                                        exchange_name = ex.get('exchange', '')
                                        buy_vol = ex.get('buy_vol_usd', 0)
                                        sell_vol = ex.get('sell_vol_usd', 0)
                                        total_vol = buy_vol + sell_vol
                                        
                                        if exchange_name and total_vol > 0:
                                            if exchange_name not in exchange_volumes:
                                                exchange_volumes[exchange_name] = {
                                                    'total_volume': 0,
                                                    'buy_volume': 0,
                                                    'sell_volume': 0,
                                                    'assets': set()
                                                }
                                            
                                            exchange_volumes[exchange_name]['total_volume'] += total_vol
                                            exchange_volumes[exchange_name]['buy_volume'] += buy_vol
                                            exchange_volumes[exchange_name]['sell_volume'] += sell_vol
                                            exchange_volumes[exchange_name]['assets'].add(asset)
            
            if exchange_volumes:
                # Convert to DataFrame
                exchange_data = []
                for exchange, metrics in exchange_volumes.items():
                    exchange_data.append({
                        'Exchange': exchange,
                        'Total Volume': metrics['total_volume'],
                        'Buy Volume': metrics['buy_volume'],
                        'Sell Volume': metrics['sell_volume'],
                        'Buy Ratio': metrics['buy_volume'] / metrics['total_volume'] if metrics['total_volume'] > 0 else 0.5,
                        'Asset Count': len(metrics['assets']),
                        'Market Share': 0  # Will calculate after
                    })
                
                exchange_df = pd.DataFrame(exchange_data)
                total_market_volume = exchange_df['Total Volume'].sum()
                exchange_df['Market Share'] = exchange_df['Total Volume'] / total_market_volume if total_market_volume > 0 else 0
                
                # Sort by volume
                exchange_df = exchange_df.sort_values('Total Volume', ascending=False)
                
                # Display top exchanges metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    top_exchange = exchange_df.iloc[0]['Exchange'] if not exchange_df.empty else "N/A"
                    top_share = exchange_df.iloc[0]['Market Share'] if not exchange_df.empty else 0
                    display_metric_card(
                        "Top Exchange",
                        top_exchange,
                        f"{top_share:.1%} market share"
                    )
                
                with col2:
                    top_5_share = exchange_df.head(5)['Market Share'].sum() if len(exchange_df) >= 5 else exchange_df['Market Share'].sum()
                    display_metric_card(
                        "Top 5 Exchange Share",
                        f"{top_5_share:.1%}",
                        "Market concentration"
                    )
                
                with col3:
                    active_exchanges = len(exchange_df)
                    display_metric_card(
                        "Active Exchanges",
                        str(active_exchanges),
                        "Trading top 10 assets"
                    )
                
                # Market share pie chart
                st.subheader("Exchange Market Share Distribution")
                
                # Take top 10 exchanges for pie chart
                top_exchanges = exchange_df.head(10).copy()
                if len(exchange_df) > 10:
                    others_share = exchange_df.iloc[10:]['Market Share'].sum()
                    others_volume = exchange_df.iloc[10:]['Total Volume'].sum()
                    top_exchanges = pd.concat([
                        top_exchanges,
                        pd.DataFrame([{
                            'Exchange': 'Others',
                            'Total Volume': others_volume,
                            'Market Share': others_share
                        }])
                    ])
                
                fig_pie = go.Figure(data=[go.Pie(
                    labels=top_exchanges['Exchange'],
                    values=top_exchanges['Market Share'],
                    hole=0.4,
                    textinfo='label+percent',
                    textposition='auto'
                )])
                
                fig_pie.update_layout(
                    title="Exchange Market Share (24h Volume)",
                    height=500
                )
                
                display_chart(apply_chart_theme(fig_pie))
                
                # Detailed exchange table
                st.subheader("Exchange Volume Details")
                
                display_exchange_df = pd.DataFrame({
                    'Exchange': exchange_df['Exchange'],
                    'Total Volume (24h)': exchange_df['Total Volume'],
                    'Market Share': exchange_df['Market Share'],
                    'Buy Ratio': exchange_df['Buy Ratio'],
                    'Asset Coverage': exchange_df['Asset Count']
                })
                
                exchange_format_dict = {
                    'Total Volume (24h)': lambda x: format_currency(x, abbreviate=True),
                    'Market Share': lambda x: f"{x:.2%}",
                    'Buy Ratio': lambda x: f"{x:.1%}",
                    'Asset Coverage': lambda x: f"{int(x)}/10"
                }
                
                create_formatted_table(
                    display_exchange_df.head(15),  # Show top 15
                    format_dict=exchange_format_dict,
                    emphasize_negatives=False,
                    compact_display=True
                )
                
                st.markdown("""
                **Exchange Distribution Insights:**
                - Market concentration shows dominance of top exchanges
                - Higher asset coverage indicates better liquidity across different markets
                - Buy ratio variations across exchanges can indicate arbitrage opportunities
                - Consider exchange reliability and regulatory status when selecting venues
                """)
        
        with volume_tab3:
            st.subheader("Volume Trends & Momentum Analysis")
            
            # Time period selector
            trend_period = st.selectbox(
                "Select Analysis Period",
                options=['7d', '30d', '90d', 'All'],
                index=1,
                key='volume_trend_period'
            )
            
            # Asset selector for detailed analysis
            selected_trend_assets = st.multiselect(
                "Select Assets for Trend Analysis",
                options=TOP_10_ASSETS,
                default=['BTC', 'ETH', 'SOL'],
                key='volume_trend_assets'
            )
            
            if selected_trend_assets:
                # Create volume trend chart
                fig_trend = go.Figure()
                
                for asset in selected_trend_assets:
                    if asset in data['volume']:
                        volume_df = data['volume'][asset].copy()
                        if 'time' in volume_df.columns:
                            volume_df['datetime'] = pd.to_datetime(volume_df['time'], unit='ms')
                            volume_df = volume_df.sort_values('datetime')
                            volume_df['total_volume'] = volume_df['aggregated_buy_volume_usd'] + volume_df['aggregated_sell_volume_usd']
                            
                            # Filter by period
                            if trend_period == '7d':
                                volume_df = volume_df.tail(7)
                            elif trend_period == '30d':
                                volume_df = volume_df.tail(30)
                            elif trend_period == '90d':
                                volume_df = volume_df.tail(90)
                            
                            # Add trace
                            fig_trend.add_trace(go.Scatter(
                                x=volume_df['datetime'],
                                y=volume_df['total_volume'],
                                name=asset,
                                mode='lines+markers',
                                line=dict(width=2)
                            ))
                
                fig_trend.update_layout(
                    title=f"Volume Trends - {trend_period} Period",
                    xaxis_title="Date",
                    yaxis_title="Daily Volume (USD)",
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                display_chart(apply_chart_theme(fig_trend))
                
                # Volume momentum indicators
                st.subheader("Volume Momentum Indicators")
                
                momentum_data = []
                for asset in selected_trend_assets:
                    if asset in data['volume']:
                        volume_df = data['volume'][asset].copy()
                        if not volume_df.empty and 'aggregated_buy_volume_usd' in volume_df.columns:
                            volume_df['total_volume'] = volume_df['aggregated_buy_volume_usd'] + volume_df['aggregated_sell_volume_usd']
                            
                            # Calculate various momentum indicators
                            if len(volume_df) >= 30:
                                # Simple moving averages
                                sma_7 = volume_df['total_volume'].tail(7).mean()
                                sma_30 = volume_df['total_volume'].tail(30).mean()
                                current_vol = volume_df['total_volume'].iloc[-1]
                                
                                # Momentum calculations
                                momentum_7d = (current_vol / sma_7 - 1) * 100 if sma_7 > 0 else 0
                                momentum_30d = (current_vol / sma_30 - 1) * 100 if sma_30 > 0 else 0
                                
                                # Volatility
                                vol_std = volume_df['total_volume'].tail(30).std()
                                vol_mean = volume_df['total_volume'].tail(30).mean()
                                vol_cv = (vol_std / vol_mean * 100) if vol_mean > 0 else 0
                                
                                # Trend strength (linear regression slope)
                                # Using numpy instead of scipy for linear regression
                                x = np.arange(len(volume_df.tail(30)))
                                y = volume_df['total_volume'].tail(30).values
                                
                                # Calculate linear regression manually
                                n = len(x)
                                x_mean = np.mean(x)
                                y_mean = np.mean(y)
                                
                                # Calculate slope and r-squared
                                numerator = np.sum((x - x_mean) * (y - y_mean))
                                denominator = np.sum((x - x_mean) ** 2)
                                
                                if denominator != 0:
                                    slope = numerator / denominator
                                    y_pred = slope * (x - x_mean) + y_mean
                                    ss_tot = np.sum((y - y_mean) ** 2)
                                    ss_res = np.sum((y - y_pred) ** 2)
                                    trend_strength = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0  # R-squared
                                else:
                                    slope = 0
                                    trend_strength = 0
                                
                                momentum_data.append({
                                    'Asset': asset,
                                    'Current Volume': current_vol,
                                    '7D MA': sma_7,
                                    '30D MA': sma_30,
                                    '7D Momentum': momentum_7d,
                                    '30D Momentum': momentum_30d,
                                    'Volatility (CV)': vol_cv,
                                    'Trend Strength': trend_strength
                                })
                
                if momentum_data:
                    momentum_df = pd.DataFrame(momentum_data)
                    
                    momentum_format_dict = {
                        'Current Volume': lambda x: format_currency(x, abbreviate=True),
                        '7D MA': lambda x: format_currency(x, abbreviate=True),
                        '30D MA': lambda x: format_currency(x, abbreviate=True),
                        '7D Momentum': lambda x: f"{x:+.1f}%",
                        '30D Momentum': lambda x: f"{x:+.1f}%",
                        'Volatility (CV)': lambda x: f"{x:.1f}%",
                        'Trend Strength': lambda x: f"{x:.2f}"
                    }
                    
                    create_formatted_table(
                        momentum_df,
                        format_dict=momentum_format_dict,
                        emphasize_negatives=True,
                        compact_display=True
                    )
                    
                    st.markdown("""
                    **Momentum Indicators Explained:**
                    - **7D/30D Momentum**: Current volume vs moving average (>0% = above average)
                    - **Volatility (CV)**: Coefficient of variation - higher values indicate more volatile volume
                    - **Trend Strength**: R² value from linear regression (0-1, higher = stronger trend)
                    - Positive momentum with low volatility suggests sustainable volume growth
                    """)
        
        with volume_tab4:
            st.subheader("Buy/Sell Flow Analysis")
            
            # Create buy/sell imbalance chart for all assets
            if 'aggregated_volume_history' in data and data['aggregated_volume_history'] is not None:
                agg_volume_df = data['aggregated_volume_history'].copy()
                agg_volume_df['datetime'] = pd.to_datetime(agg_volume_df['time'], unit='ms')
                agg_volume_df = agg_volume_df.sort_values('datetime')
                
                # Calculate metrics
                agg_volume_df['total_volume'] = agg_volume_df['aggregated_buy_volume_usd'] + agg_volume_df['aggregated_sell_volume_usd']
                agg_volume_df['buy_ratio'] = agg_volume_df['aggregated_buy_volume_usd'] / agg_volume_df['total_volume']
                agg_volume_df['net_flow'] = agg_volume_df['aggregated_buy_volume_usd'] - agg_volume_df['aggregated_sell_volume_usd']
                agg_volume_df['flow_imbalance'] = agg_volume_df['net_flow'] / agg_volume_df['total_volume']
                
                # Time period selector for flow analysis
                flow_period = st.selectbox(
                    "Select Flow Analysis Period",
                    options=['24h', '7d', '30d'],
                    index=1,
                    key='flow_period'
                )
                
                # Filter data based on period
                if flow_period == '24h':
                    flow_df = agg_volume_df.tail(6)  # 6 * 4h = 24h
                elif flow_period == '7d':
                    flow_df = agg_volume_df.tail(42)  # 42 * 4h = 7 days
                else:  # 30d
                    flow_df = agg_volume_df.tail(180)  # 180 * 4h = 30 days
                
                # Flow metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    cumulative_net_flow = flow_df['net_flow'].sum()
                    display_metric_card(
                        f"Net Flow ({flow_period})",
                        format_currency(cumulative_net_flow, abbreviate=True),
                        "Buy pressure" if cumulative_net_flow > 0 else "Sell pressure"
                    )
                
                with col2:
                    avg_buy_ratio = flow_df['buy_ratio'].mean()
                    display_metric_card(
                        f"Avg Buy Ratio ({flow_period})",
                        f"{avg_buy_ratio:.1%}",
                        "Bullish" if avg_buy_ratio > 0.52 else "Bearish" if avg_buy_ratio < 0.48 else "Neutral"
                    )
                
                with col3:
                    flow_volatility = flow_df['flow_imbalance'].std()
                    display_metric_card(
                        "Flow Volatility",
                        f"{flow_volatility:.3f}",
                        "High" if flow_volatility > 0.1 else "Moderate" if flow_volatility > 0.05 else "Low"
                    )
                
                with col4:
                    # Trend direction based on linear regression
                    # Using numpy instead of scipy for linear regression
                    x = np.arange(len(flow_df))
                    y = flow_df['buy_ratio'].values
                    
                    # Calculate slope manually
                    n = len(x)
                    x_mean = np.mean(x)
                    y_mean = np.mean(y)
                    
                    numerator = np.sum((x - x_mean) * (y - y_mean))
                    denominator = np.sum((x - x_mean) ** 2)
                    
                    slope = numerator / denominator if denominator != 0 else 0
                    trend_direction = "Increasing" if slope > 0.0001 else "Decreasing" if slope < -0.0001 else "Stable"
                    display_metric_card(
                        "Buy Pressure Trend",
                        trend_direction,
                        f"Slope: {slope:.5f}"
                    )
                
                # Create flow imbalance chart
                fig_flow = go.Figure()
                
                # Add buy volume (positive)
                fig_flow.add_trace(go.Bar(
                    x=flow_df['datetime'],
                    y=flow_df['aggregated_buy_volume_usd'],
                    name='Buy Volume',
                    marker_color='rgba(0, 255, 0, 0.7)',
                    hovertemplate='Buy: $%{y:,.0f}<extra></extra>'
                ))
                
                # Add sell volume (negative)
                fig_flow.add_trace(go.Bar(
                    x=flow_df['datetime'],
                    y=-flow_df['aggregated_sell_volume_usd'],
                    name='Sell Volume',
                    marker_color='rgba(255, 0, 0, 0.7)',
                    hovertemplate='Sell: $%{y:,.0f}<extra></extra>'
                ))
                
                # Add net flow line
                fig_flow.add_trace(go.Scatter(
                    x=flow_df['datetime'],
                    y=flow_df['net_flow'],
                    name='Net Flow',
                    line=dict(color='white', width=2),
                    yaxis='y2',
                    hovertemplate='Net: $%{y:,.0f}<extra></extra>'
                ))
                
                fig_flow.update_layout(
                    title=f"Buy/Sell Volume Flow - {flow_period}",
                    xaxis_title="Time",
                    yaxis_title="Volume (USD)",
                    yaxis2=dict(
                        title="Net Flow (USD)",
                        overlaying='y',
                        side='right'
                    ),
                    barmode='relative',
                    hovermode='x unified',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                
                display_chart(apply_chart_theme(fig_flow))
                
                # Buy/Sell ratio heatmap by asset
                st.subheader("Buy/Sell Ratio by Asset")
                
                # Prepare heatmap data
                heatmap_data = []
                for asset in TOP_10_ASSETS:
                    if asset in data['volume']:
                        asset_volume_df = data['volume'][asset].copy()
                        if not asset_volume_df.empty and 'aggregated_buy_volume_usd' in asset_volume_df.columns:
                            asset_volume_df['buy_ratio'] = asset_volume_df['aggregated_buy_volume_usd'] / (
                                asset_volume_df['aggregated_buy_volume_usd'] + asset_volume_df['aggregated_sell_volume_usd']
                            )
                            
                            # Get ratios for different periods
                            current_ratio = asset_volume_df['buy_ratio'].iloc[-1] if len(asset_volume_df) > 0 else 0.5
                            avg_7d = asset_volume_df['buy_ratio'].tail(7).mean() if len(asset_volume_df) >= 7 else current_ratio
                            avg_30d = asset_volume_df['buy_ratio'].tail(30).mean() if len(asset_volume_df) >= 30 else current_ratio
                            
                            heatmap_data.append({
                                'Asset': asset,
                                'Current': current_ratio,
                                '7D Average': avg_7d,
                                '30D Average': avg_30d
                            })
                
                if heatmap_data:
                    heatmap_df = pd.DataFrame(heatmap_data)
                    heatmap_df = heatmap_df.set_index('Asset')
                    
                    # Create heatmap
                    fig_heatmap = go.Figure(data=go.Heatmap(
                        z=heatmap_df.values,
                        x=heatmap_df.columns,
                        y=heatmap_df.index,
                        colorscale='RdYlGn',
                        zmid=0.5,
                        text=[[f"{val:.1%}" for val in row] for row in heatmap_df.values],
                        texttemplate="%{text}",
                        textfont={"size": 12},
                        colorbar=dict(title="Buy Ratio")
                    ))
                    
                    fig_heatmap.update_layout(
                        title="Buy/Sell Ratio Heatmap",
                        xaxis_title="Time Period",
                        yaxis_title="Asset",
                        height=400
                    )
                    
                    display_chart(apply_chart_theme(fig_heatmap))
                    
                    st.markdown("""
                    **Buy/Sell Flow Analysis Insights:**
                    - **Net Flow**: Positive = more buying pressure, Negative = more selling pressure
                    - **Buy Ratio >52%**: Strong bullish sentiment, likely to see positive funding rates
                    - **Flow Volatility**: High volatility suggests uncertain market direction
                    - **Trend Analysis**: Consistent buy/sell pressure indicates market conviction
                    - **Cross-Asset Comparison**: Identify which assets have strongest directional bias
                    """)
    else:
        st.warning("No futures volume data available. Please check data sources.")

if __name__ == "__main__":
    main()