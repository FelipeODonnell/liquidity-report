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
        
        # Accumulated funding rates
        accumulated_funding_file = os.path.join(data_path, 'futures', 'funding_rate', 'api_futures_fundingRate_accumulated_exchange_list.parquet')
        if os.path.exists(accumulated_funding_file):
            data['accumulated_funding_rates'] = pd.read_parquet(accumulated_funding_file)
            logger.info(f"Loaded accumulated funding rates: {len(data['accumulated_funding_rates'])} rows")
        
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

if __name__ == "__main__":
    main()