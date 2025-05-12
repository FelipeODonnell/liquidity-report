#!/usr/bin/env python3
"""
Futures Symbol Files Generator

This script creates separate API files for different cryptocurrency symbols (BTC, ETH, XRP, SOL)
for futures-related API endpoints in the coinglass-api folder.
"""

import os
import re
import shutil
from pathlib import Path

# Define the target cryptocurrencies
CRYPTO_SYMBOLS = ['BTC', 'ETH', 'XRP', 'SOL']

# Define files to modify
FILES_TO_MODIFY = [
    # Futures Taker Buy/Sell files
    'coinglass-api/futures/taker_buy_sell/api_futures_taker_buy_sell_volume_history.py',
    
    # Futures Orderbook files
    'coinglass-api/futures/order_book/api_futures_orderbook_aggregated_ask_bids_history.py',
    
    # Futures Liquidation files
    'coinglass-api/futures/liquidation/api_futures_liquidation_aggregated_coin_history.py',
    'coinglass-api/futures/liquidation/api_futures_liquidation_exchange_list.py',
    
    # Futures Exchange Taker Buy/Sell Ratio
    'coinglass-api/futures/long_short_ratio/api_futures_taker_buy_sell_volume_exchange_list.py',
    
    # Futures Funding Rate files
    'coinglass-api/futures/funding_rate/api_futures_fundingRate_oi_weight_ohlc_history.py',
    'coinglass-api/futures/funding_rate/api_futures_fundingRate_vol_weight_ohlc_history.py',
    
    # Futures Open Interest files
    'coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_history.py',
    'coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_stablecoin.py',
    'coinglass-api/futures/open_interest/api_futures_openInterest_ohlc_aggregated_coin_margin_history.py',
    'coinglass-api/futures/open_interest/api_futures_openInterest_exchange_list.py',
    'coinglass-api/futures/open_interest/api_futures_openInterest_exchange_history_chart.py',
    
    # Futures Market files
    'coinglass-api/futures/market/api_futures_pairs_markets.py',
]

def create_modified_file(src_path, symbol):
    """
    Create a new file with the symbol suffix and modify its content to use the given symbol.
    
    Args:
        src_path (str): Path to the source file
        symbol (str): Cryptocurrency symbol (BTC, ETH, etc.)
    
    Returns:
        str: Path to the created file
    """
    # Create the destination file path with the symbol suffix
    src_path = Path(src_path)
    dest_filename = f"{src_path.stem}_{symbol}{src_path.suffix}"
    dest_path = src_path.parent / dest_filename
    
    # Read the source file
    with open(src_path, 'r') as f:
        content = f.read()
    
    # Modify the content based on the file type and URL patterns
    
    # Handle futures taker buy/sell volume history
    if 'taker_buy_sell_volume_history.py' in str(src_path):
        # For these files, we need to update the trading pair (BTCUSDT to ETHUSDT, etc.)
        trading_pair = f"{symbol}USDT"
        content = re.sub(
            r'(symbol=)([^&"]+)',
            f'\\1{trading_pair}',
            content
        )
    
    # Handle futures orderbook files
    elif 'orderbook_aggregated_ask_bids_history.py' in str(src_path):
        # These also use trading pairs
        trading_pair = f"{symbol}USDT"
        content = re.sub(
            r'(symbol=)([^&"]+)',
            f'\\1{trading_pair}',
            content
        )
    
    # Handle futures liquidation files
    elif 'liquidation_aggregated_coin_history.py' in str(src_path):
        # These use the symbol parameter
        content = re.sub(
            r'(symbol=)([^&"]+)',
            f'\\1{symbol}',
            content
        )
    elif 'liquidation_exchange_list.py' in str(src_path):
        # Check if the URL already has a symbol parameter
        if 'symbol=' not in content:
            # Add symbol parameter if it's not already there
            url_pattern = r'(url = "[^"]+)(")'
            if '?' in re.search(url_pattern, content).group(1):
                # URL already has a query parameter, add with &
                content = re.sub(url_pattern, f'\\1&symbol={symbol}\\2', content)
            else:
                # URL has no query parameters yet, add with ?
                content = re.sub(url_pattern, f'\\1?symbol={symbol}\\2', content)
        else:
            # Update existing symbol parameter
            content = re.sub(
                r'(symbol=)([^&"]+)',
                f'\\1{symbol}',
                content
            )
    
    # Handle futures taker buy/sell volume exchange list
    elif 'taker_buy_sell_volume_exchange_list.py' in str(src_path):
        # Check if the URL already has a symbol parameter
        if 'symbol=' not in content:
            # Add symbol parameter if it's not already there
            url_pattern = r'(url = "[^"]+)(")'
            if '?' in re.search(url_pattern, content).group(1):
                # URL already has a query parameter, add with &
                content = re.sub(url_pattern, f'\\1&symbol={symbol}\\2', content)
            else:
                # URL has no query parameters yet, add with ?
                content = re.sub(url_pattern, f'\\1?symbol={symbol}\\2', content)
        else:
            # Update existing symbol parameter
            content = re.sub(
                r'(symbol=)([^&"]+)',
                f'\\1{symbol}',
                content
            )
    
    # Handle funding rate files
    elif 'fundingRate_oi_weight_ohlc_history.py' in str(src_path) or 'fundingRate_vol_weight_ohlc_history.py' in str(src_path):
        content = re.sub(
            r'(symbol=)([^&"]+)',
            f'\\1{symbol}',
            content
        )
    
    # Handle open interest files
    elif 'openInterest_ohlc_aggregated' in str(src_path) or 'openInterest_exchange' in str(src_path):
        if 'symbol=' in content:
            content = re.sub(
                r'(symbol=)([^&"]+)',
                f'\\1{symbol}',
                content
            )
        else:
            # If symbol parameter doesn't exist, add it
            url_pattern = r'(url = "[^"]+)(")'
            if '?' in re.search(url_pattern, content).group(1):
                # URL already has a query parameter, add with &
                content = re.sub(url_pattern, f'\\1&symbol={symbol}\\2', content)
            else:
                # URL has no query parameters yet, add with ?
                content = re.sub(url_pattern, f'\\1?symbol={symbol}\\2', content)
    
    # Handle futures market pairs markets file
    elif 'futures_pairs_markets.py' in str(src_path):
        if 'symbol=' in content:
            content = re.sub(
                r'(symbol=)([^&"]+)',
                f'\\1{symbol}',
                content
            )
        else:
            # If symbol parameter doesn't exist, add it
            url_pattern = r'(url = "[^"]+)(")'
            if '?' in re.search(url_pattern, content).group(1):
                # URL already has a query parameter, add with &
                content = re.sub(url_pattern, f'\\1&symbol={symbol}\\2', content)
            else:
                # URL has no query parameters yet, add with ?
                content = re.sub(url_pattern, f'\\1?symbol={symbol}\\2', content)
    
    # For all files, update the output filename to include the symbol
    content = re.sub(
        r'(base_filename = os\.path\.splitext\(os\.path\.basename\(__file__\)\)\[0\])',
        f'\\1\n        # Add symbol to base filename\n        base_filename = f"{{base_filename}}_{symbol}"',
        content
    )
    
    # Write the modified content to the destination file
    with open(dest_path, 'w') as f:
        f.write(content)
    
    return str(dest_path)

def main():
    """Generate symbol-specific versions of the API files."""
    created_files = []
    
    print("Creating symbol-specific futures API files...")
    
    # Process each file for each symbol
    for file_path in FILES_TO_MODIFY:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue
            
        print(f"\nProcessing base file: {file_path}")
        
        # Check the file content to decide how to handle it
        with open(file_path, 'r') as f:
            file_content = f.read()
        
        for symbol in CRYPTO_SYMBOLS:
            # Create the modified file
            new_file = create_modified_file(file_path, symbol)
            created_files.append(new_file)
            print(f"  - Created: {new_file}")
    
    print(f"\nCreated {len(created_files)} new symbol-specific files")
    print("\nFutures symbol files creation complete!")

if __name__ == "__main__":
    main()