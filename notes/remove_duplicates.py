#!/usr/bin/env python3
"""
Script to remove base API files that now have symbol-specific versions.
"""

import os
import shutil
from pathlib import Path

def main():
    """
    Identify and remove base files that have corresponding symbol-specific versions.
    """
    # List of files to remove (base files with symbol-specific versions)
    files_to_remove = [
        # Options files
        "coinglass-api/options/api_option_max_pain.py",
        "coinglass-api/options/api_option_info.py",
        
        # Spot files
        "coinglass-api/spot/spot_market/api_spot_pairs_markets.py",
        "coinglass-api/spot/spot_market/api_spot_price_history.py",
        "coinglass-api/spot/order_book_spot/api_spot_orderbook_ask_bids_history.py",
        "coinglass-api/spot/taker_buy_sell_spot/api_spot_taker_buy_sell_volume_history.py",
        
        # Futures files
        "coinglass-api/futures/taker_buy_sell/api_futures_taker_buy_sell_volume_history.py",
        "coinglass-api/futures/order_book/api_futures_orderbook_aggregated_ask_bids_history.py",
        "coinglass-api/futures/liquidation/api_futures_liquidation_aggregated_coin_history.py",
        "coinglass-api/futures/liquidation/api_futures_liquidation_exchange_list.py",
        "coinglass-api/futures/long_short_ratio/api_futures_taker_buy_sell_volume_exchange_list.py",
        "coinglass-api/futures/funding_rate/api_futures_fundingRate_oi_weight_ohlc_history.py",
        "coinglass-api/futures/funding_rate/api_futures_fundingRate_vol_weight_ohlc_history.py",
        "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_history.py",
        "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_stablecoin.py",
        "coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_coin_margin_history.py",
        "coinglass-api/futures/open_interest/api_futures_openInterest_exchange_list.py",
        "coinglass-api/futures/open_interest/api_futures_openInterest_exchange_history_chart.py",
        "coinglass-api/futures/market/api_futures_pairs_markets.py",
    ]
    
    # Create a backup directory
    backup_dir = "duplicate_files_backup"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Process each file
    files_removed = 0
    files_backed_up = 0
    
    for file_path in files_to_remove:
        file_path = file_path.replace('/', os.sep)  # Ensure OS-appropriate path separators
        full_path = os.path.join(os.getcwd(), file_path)
        
        if os.path.exists(full_path):
            try:
                # Create backup directory structure
                backup_path = os.path.join(backup_dir, file_path)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                
                # Backup the file
                shutil.copy2(full_path, backup_path)
                files_backed_up += 1
                
                # Remove the original file
                os.remove(full_path)
                files_removed += 1
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"Skipped (not found): {file_path}")
    
    print(f"\nSummary:")
    print(f"  - {files_removed} duplicate files removed")
    print(f"  - {files_backed_up} files backed up to '{backup_dir}' directory")
    print(f"  - All symbol-specific versions (_BTC, _ETH, _XRP, _SOL) have been preserved")
    
    # Check if there are any files in the backup directory
    if files_backed_up == 0:
        try:
            os.rmdir(backup_dir)
            print(f"  - Removed empty backup directory")
        except:
            pass

if __name__ == "__main__":
    main()