#!/usr/bin/env python3
"""
CoinGlass API Data Collection Tool

This script runs all API files in the coinglass-api folder and saves data to
a new folder with today's date. Comment out any files you don't want to run.
"""

import os
import subprocess
import sys
from datetime import datetime
import re
import time
from collections import deque

# List of API files to run
# Comment out any file you don't want to run by adding a # at the beginning of the line
API_FILES = [
    # ETF files
    "coinglass-api/etf/api_etf_bitcoin_aum.py",
    "coinglass-api/etf/api_etf_bitcoin_detail.py",
    "coinglass-api/etf/api_etf_bitcoin_flow_history.py",
    "coinglass-api/etf/api_etf_bitcoin_history.py",
    "coinglass-api/etf/api_etf_bitcoin_list.py",
    "coinglass-api/etf/api_etf_bitcoin_list_modified.py",
    "coinglass-api/etf/api_etf_bitcoin_net_assets_history.py",
    "coinglass-api/etf/api_etf_bitcoin_premium_discount_history.py",
    "coinglass-api/etf/api_etf_bitcoin_price_history.py",
    "coinglass-api/etf/api_etf_ethereum_flow_history.py",
    "coinglass-api/etf/api_etf_ethereum_list.py",
    "coinglass-api/etf/api_etf_ethereum_net_assets_history.py",
    "coinglass-api/etf/api_grayscale_holdings_list.py",
    "coinglass-api/etf/api_grayscale_premium_history.py",
    "coinglass-api/etf/api_hk_etf_bitcoin_flow_history.py",
    # FUTURES files
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_accumulated_exchange_list.py",
    # "coinglass-api/futures/funding_rate/api_futures_fundingRate_arbitrage.py",
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_exchange_list.py",
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_ohlc_history.py",
    # Adding symbol-specific versions for funding rate files
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_oi_weight_ohlc_history_BTC.py",
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_oi_weight_ohlc_history_ETH.py",
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_oi_weight_ohlc_history_XRP.py",
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_oi_weight_ohlc_history_SOL.py",
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_vol_weight_ohlc_history_BTC.py",
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_vol_weight_ohlc_history_ETH.py",
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_vol_weight_ohlc_history_XRP.py",
    "coinglass-api/futures/funding_rate/api_futures_fundingRate_vol_weight_ohlc_history_SOL.py",
    # Adding symbol-specific versions for liquidation files
    "coinglass-api/futures/liquidation/api_futures_liquidation_aggregated_coin_history_BTC.py",
    "coinglass-api/futures/liquidation/api_futures_liquidation_aggregated_coin_history_ETH.py",
    "coinglass-api/futures/liquidation/api_futures_liquidation_aggregated_coin_history_XRP.py",
    "coinglass-api/futures/liquidation/api_futures_liquidation_aggregated_coin_history_SOL.py",
    # "coinglass-api/futures/liquidation/api_futures_liquidation_aggregated_coin_map.py",
    # "coinglass-api/futures/liquidation/api_futures_liquidation_coin_heatmap_model1.py",
    # "coinglass-api/futures/liquidation/api_futures_liquidation_coin_heatmap_model2.py",
    # "coinglass-api/futures/liquidation/api_futures_liquidation_coin_heatmap_model3.py",
    # "coinglass-api/futures/liquidation/api_futures_liquidation_coin_list.py",
    "coinglass-api/futures/liquidation/api_futures_liquidation_exchange_list_BTC.py",
    "coinglass-api/futures/liquidation/api_futures_liquidation_exchange_list_ETH.py",
    "coinglass-api/futures/liquidation/api_futures_liquidation_exchange_list_XRP.py",
    "coinglass-api/futures/liquidation/api_futures_liquidation_exchange_list_SOL.py",
    # "coinglass-api/futures/liquidation/api_futures_liquidation_order.py",
    # "coinglass-api/futures/liquidation/api_futures_liquidation_pair_heatmap_model1.py",
    # "coinglass-api/futures/liquidation/api_futures_liquidation_pair_heatmap_model2.py",
    # "coinglass-api/futures/liquidation/api_futures_liquidation_pair_heatmap_model3.py",
    "coinglass-api/futures/liquidation/api_futures_liquidation_pair_history.py",
    # "coinglass-api/futures/liquidation/api_futures_liquidation_pair_map.py",

    # Long/short ratio files
    "coinglass-api/futures/long_short_ratio/api_futures_global_long_short_account_ratio_history.py",
    "coinglass-api/futures/long_short_ratio/api_futures_taker_buy_sell_volume_exchange_list_BTC.py",
    "coinglass-api/futures/long_short_ratio/api_futures_taker_buy_sell_volume_exchange_list_ETH.py",
    "coinglass-api/futures/long_short_ratio/api_futures_taker_buy_sell_volume_exchange_list_XRP.py",
    "coinglass-api/futures/long_short_ratio/api_futures_taker_buy_sell_volume_exchange_list_SOL.py",
    "coinglass-api/futures/long_short_ratio/api_futures_top_long_short_account_ratio_history.py",
    "coinglass-api/futures/long_short_ratio/api_futures_top_long_short_position_ratio_history.py",

    # Futures market files
    # "coinglass-api/futures/market/api_futures_coins_markets.py",
    "coinglass-api/futures/market/api_futures_pairs_markets_BTC.py",
    "coinglass-api/futures/market/api_futures_pairs_markets_ETH.py",
    "coinglass-api/futures/market/api_futures_pairs_markets_XRP.py",
    "coinglass-api/futures/market/api_futures_pairs_markets_SOL.py",
    "coinglass-api/futures/market/api_price_ohlc_history.py",

    # Open interest files
    "coinglass-api/futures/open_interest/api_futures_openInterest_exchange_history_chart_BTC.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_exchange_history_chart_ETH.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_exchange_history_chart_XRP.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_exchange_history_chart_SOL.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_exchange_list_BTC.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_exchange_list_ETH.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_exchange_list_XRP.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_exchange_list_SOL.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_coin_margin_history_BTC.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_coin_margin_history_ETH.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_coin_margin_history_XRP.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_coin_margin_history_SOL.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_history_BTC.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_history_ETH.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_history_XRP.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_history_SOL.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_stablecoin_BTC.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_stablecoin_ETH.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_stablecoin_XRP.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_stablecoin_SOL.py",
    "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_history.py",

    # Order book files
    "coinglass-api/futures/order_book/api_futures_orderbook_aggregated_ask_bids_history_BTC.py",
    "coinglass-api/futures/order_book/api_futures_orderbook_aggregated_ask_bids_history_ETH.py",
    "coinglass-api/futures/order_book/api_futures_orderbook_aggregated_ask_bids_history_XRP.py",
    "coinglass-api/futures/order_book/api_futures_orderbook_aggregated_ask_bids_history_SOL.py",
    "coinglass-api/futures/order_book/api_futures_orderbook_ask_bids_history.py",
    # "coinglass-api/futures/order_book/api_futures_orderbook_heatmap_history.py",
    # "coinglass-api/futures/order_book/api_futures_orderbook_large_limit_order.py",
    # "coinglass-api/futures/order_book/api_futures_orderbook_large_limit_order_history.py",

    # Taker buy/sell files
    "coinglass-api/futures/taker_buy_sell/api_futures_aggregated_taker_buy_sell_volume_history.py",
    "coinglass-api/futures/taker_buy_sell/api_futures_taker_buy_sell_volume_history_BTC.py",
    "coinglass-api/futures/taker_buy_sell/api_futures_taker_buy_sell_volume_history_ETH.py",
    "coinglass-api/futures/taker_buy_sell/api_futures_taker_buy_sell_volume_history_XRP.py",
    "coinglass-api/futures/taker_buy_sell/api_futures_taker_buy_sell_volume_history_SOL.py",
    # "coinglass-api/futures/whale_positions/api_hyperliquid_whale_alert.py",
    # "coinglass-api/futures/whale_positions/api_hyperliquid_whale_position.py",
    # INDIC files
    "coinglass-api/indic/api_bitfinex_margin_long_short.py",
    # "coinglass-api/indic/api_borrow_interest_rate_history.py",
    "coinglass-api/indic/api_bull_market_peak_indicator.py",
    "coinglass-api/indic/api_coinbase_premium_index.py",
    "coinglass-api/indic/api_futures_basis_history.py",
    # "coinglass-api/indic/api_futures_rsi_list.py",
    "coinglass-api/indic/api_index_200_week_moving_average_heatmap.py",
    "coinglass-api/indic/api_index_2_year_ma_multiplier.py",
    "coinglass-api/indic/api_index_ahr999.py",
    "coinglass-api/indic/api_index_bitcoin_bubble_index.py",
    "coinglass-api/indic/api_index_bitcoin_profitable_days.py",
    "coinglass-api/indic/api_index_bitcoin_rainbow_chart.py",
    "coinglass-api/indic/api_index_fear_greed_history.py",
    "coinglass-api/indic/api_index_golden_ratio_multiplier.py",
    "coinglass-api/indic/api_index_pi_cycle_indicator.py",
    "coinglass-api/indic/api_index_puell_multiple.py",
    "coinglass-api/indic/api_index_stableCoin_marketCap_history.py",
    "coinglass-api/indic/api_index_stock_flow.py",
    # ON_CHAIN files
    "coinglass-api/on_chain/api_exchange_assets.py",
    "coinglass-api/on_chain/api_exchange_balance_chart.py",
    "coinglass-api/on_chain/api_exchange_balance_list.py",
    "coinglass-api/on_chain/api_exchange_chain_tx_list.py",
    # OPTIONS files
    "coinglass-api/options/api_option_exchange_oi_history.py",
    "coinglass-api/options/api_option_exchange_vol_history.py",
    # Symbol-specific options files
    "coinglass-api/options/api_option_info_BTC.py",
    "coinglass-api/options/api_option_info_ETH.py",
    "coinglass-api/options/api_option_info_XRP.py",
    "coinglass-api/options/api_option_info_SOL.py",
    "coinglass-api/options/api_option_max_pain_BTC.py",
    "coinglass-api/options/api_option_max_pain_ETH.py",
    "coinglass-api/options/api_option_max_pain_XRP.py",
    "coinglass-api/options/api_option_max_pain_SOL.py",

    # SPOT files
    "coinglass-api/spot/order_book_spot/api_spot_orderbook_aggregated_ask_bids_history.py",
    # Symbol-specific spot order book files
    "coinglass-api/spot/order_book_spot/api_spot_orderbook_ask_bids_history_BTC.py",
    "coinglass-api/spot/order_book_spot/api_spot_orderbook_ask_bids_history_ETH.py",
    "coinglass-api/spot/order_book_spot/api_spot_orderbook_ask_bids_history_XRP.py",
    "coinglass-api/spot/order_book_spot/api_spot_orderbook_ask_bids_history_SOL.py",
    # "coinglass-api/spot/order_book_spot/api_spot_orderbook_heatmap_history.py",
    # "coinglass-api/spot/order_book_spot/api_spot_orderbook_large_limit_order.py",
    # "coinglass-api/spot/order_book_spot/api_spot_orderbook_large_limit_order_history.py",
    # "coinglass-api/spot/spot_market/api_spot_coins_markets.py",
    # Symbol-specific spot market files
    "coinglass-api/spot/spot_market/api_spot_pairs_markets_BTC.py",
    "coinglass-api/spot/spot_market/api_spot_pairs_markets_ETH.py",
    "coinglass-api/spot/spot_market/api_spot_pairs_markets_XRP.py",
    "coinglass-api/spot/spot_market/api_spot_pairs_markets_SOL.py",
    "coinglass-api/spot/spot_market/api_spot_price_history_BTC.py",
    "coinglass-api/spot/spot_market/api_spot_price_history_ETH.py",
    "coinglass-api/spot/spot_market/api_spot_price_history_XRP.py",
    "coinglass-api/spot/spot_market/api_spot_price_history_SOL.py",
    "coinglass-api/spot/spot_market/api_spot_supported_coins.py",
    "coinglass-api/spot/spot_market/api_spot_supported_exchange_pairs.py",
    "coinglass-api/spot/taker_buy_sell_spot/api_spot_aggregated_taker_buy_sell_volume_history.py",
    # Symbol-specific spot taker buy/sell files
    "coinglass-api/spot/taker_buy_sell_spot/api_spot_taker_buy_sell_volume_history_BTC.py",
    "coinglass-api/spot/taker_buy_sell_spot/api_spot_taker_buy_sell_volume_history_ETH.py",
    "coinglass-api/spot/taker_buy_sell_spot/api_spot_taker_buy_sell_volume_history_XRP.py",
    "coinglass-api/spot/taker_buy_sell_spot/api_spot_taker_buy_sell_volume_history_SOL.py",
]

class RateLimiter:
    """
    Rate limiter to ensure we don't exceed API rate limits.
    Implements a sliding window to track requests over the past minute.
    """
    def __init__(self, max_requests_per_minute=29):
        self.max_requests = max_requests_per_minute
        self.window_size = 60  # 60 seconds = 1 minute
        self.requests = deque()
    
    def wait_if_needed(self):
        """
        Wait if we've reached the rate limit until we can make another request.
        Returns the waiting time in seconds, or 0 if no wait was needed.
        """
        current_time = time.time()
        
        # Remove requests that are older than the window size
        while self.requests and current_time - self.requests[0] > self.window_size:
            self.requests.popleft()
        
        # If we've reached the limit
        if len(self.requests) >= self.max_requests:
            # Calculate how long to wait
            oldest_request = self.requests[0]
            wait_time = oldest_request + self.window_size - current_time
            
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                # Recalculate current time after waiting
                current_time = time.time()
                return wait_time
        
        # Record this request
        self.requests.append(current_time)
        return 0

def create_date_folder():
    """Create a folder with today's date and mirror the coinglass-api structure."""
    current_date = datetime.now().strftime('%Y%m%d')
    date_folder = os.path.join('data', current_date)
    
    # Create the date folder
    if not os.path.exists(date_folder):
        os.makedirs(date_folder)
        print(f"Created folder: {date_folder}")
    
    # Mirror the coinglass-api folder structure
    for api_file in API_FILES:
        if api_file.startswith('#'):
            continue
            
        # Extract directory part
        dir_path = os.path.dirname(api_file)
        # Replace coinglass-api with the date folder
        output_path = dir_path.replace('coinglass-api', date_folder)
        
        # Create the directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            print(f"Created directory: {output_path}")
    
    return current_date

def run_api_files():
    """Run all non-commented API files with rate limiting."""
    if not API_FILES:
        print("No API files found. Run with --discover first to populate the list.")
        return
        
    current_date = create_date_folder()
    
    print(f"\nRunning API files for date: {current_date}\n")
    
    total_files = 0
    successful = 0
    
    # Initialize rate limiter (29 requests per minute)
    rate_limiter = RateLimiter(max_requests_per_minute=29)
    
    for api_file in API_FILES:
        # Skip commented files
        if api_file.startswith('#'):
            print(f"Skipping (commented out): {api_file}")
            continue
            
        total_files += 1
        
        # Apply rate limiting
        wait_time = rate_limiter.wait_if_needed()
        if wait_time > 0:
            print(f"Continuing after rate limit wait...")
        
        print(f"Running: {api_file}")
        try:
            subprocess.run([sys.executable, api_file], check=True)
            print(f"✓ Completed: {api_file}")
            successful += 1
        except subprocess.CalledProcessError as e:
            print(f"✗ Error running: {api_file}")
            print(f"  Error details: {e}")
    
    print(f"\nCompleted {successful} of {total_files} API requests.")
    print(f"Data saved to: data/{current_date}/")

def discover_api_files():
    """
    Scan the coinglass-api directory for all Python API files.
    Will update the API_FILES list in this script file.
    """
    root_dir = 'coinglass-api'
    api_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py') and filename.startswith('api_'):
                file_path = os.path.join(dirpath, filename)
                api_files.append(file_path)
    
    # Sort the files to group them by directory
    api_files.sort()
    
    print(f"Found {len(api_files)} API files")
    
    # Update this script with the discovered files
    update_script_with_files(api_files)
    
    return api_files

def update_script_with_files(api_files):
    """Update this script with the discovered API files."""
    script_path = os.path.abspath(__file__)
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find where to insert the API files list
    pattern = r"API_FILES = \[\n.*?\]"
    
    # Create the new API_FILES block with categories as comments
    api_files_block = "API_FILES = [\n"
    
    # Track current category for grouping
    current_category = None
    
    for file_path in api_files:
        # Extract category from path
        parts = file_path.split(os.sep)
        if len(parts) > 2:
            category = parts[1]  # First subdirectory after coinglass-api
            if category != current_category:
                current_category = category
                api_files_block += f"    # {category.upper()} files\n"
        
        api_files_block += f"    \"{file_path}\",\n"
    
    api_files_block += "]"
    
    # Replace the existing API_FILES list with the new one
    # Use re.DOTALL to match across multiple lines
    if re.search(pattern, content, re.DOTALL):
        updated_content = re.sub(pattern, api_files_block, content, flags=re.DOTALL)
        
        with open(script_path, 'w') as f:
            f.write(updated_content)
        
        print(f"Updated {script_path} with {len(api_files)} API files")
    else:
        print("Could not find API_FILES pattern in the script to update")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run CoinGlass API files and save data")
    parser.add_argument('--discover', action='store_true', help='Discover and update API files list')
    parser.add_argument('--date', type=str, help='Specify date folder to use (format: YYYYMMDD)')
    parser.add_argument('--max-rate', type=int, default=29, help='Maximum requests per minute (default: 29)')
    
    args = parser.parse_args()
    
    if args.discover:
        discover_api_files()
    else:
        run_api_files()

if __name__ == "__main__":
    main()