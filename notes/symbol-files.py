#!/usr/bin/env python3
"""
Symbol Files Generator

This script creates separate API files for different cryptocurrency symbols (BTC, ETH, XRP, SOL).
It identifies API endpoints that need to be symbol-specific and generates separate files for each symbol.
"""

import os
import re
import shutil
from pathlib import Path

# Define the target cryptocurrencies
CRYPTO_SYMBOLS = ['BTC', 'ETH', 'XRP', 'SOL']

# Define files to modify
FILES_TO_MODIFY = [
    # Options files
    'coinglass-api/options/api_option_max_pain.py',
    'coinglass-api/options/api_option_info.py',
    
    # Spot market files
    'coinglass-api/spot/spot_market/api_spot_pairs_markets.py',
    'coinglass-api/spot/spot_market/api_spot_price_history.py',
    
    # Spot orderbook file
    'coinglass-api/spot/order_book_spot/api_spot_orderbook_ask_bids_history.py',
    
    # Spot taker buy/sell file
    'coinglass-api/spot/taker_buy_sell_spot/api_spot_taker_buy_sell_volume_history.py'
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
    
    # Modify the content based on the file type
    if 'option_max_pain.py' in str(src_path):
        # For option_max_pain.py: Change the symbol in the URL
        content = re.sub(
            r'(url = "https://open-api-v4\.coinglass\.com/api/option/max-pain\?symbol=)([^&"]+)(&exchange=Deribit")',
            f'\\1{symbol}\\3',
            content
        )
    elif 'option_info.py' in str(src_path):
        # For option_info.py: Change the symbol in the URL
        content = re.sub(
            r'(url = "https://open-api-v4\.coinglass\.com/api/option/info\?symbol=)([^"]+)(")',
            f'\\1{symbol}\\3',
            content
        )
    elif 'spot_pairs_markets.py' in str(src_path):
        # For spot_pairs_markets.py: Change the symbol in the URL
        content = re.sub(
            r'(url = "https://open-api-v4\.coinglass\.com/api/spot/pairs-markets\?symbol=)([^"]+)(")',
            f'\\1{symbol}\\3',
            content
        )
    elif 'spot_price_history.py' in str(src_path):
        # For spot_price_history.py: Change the trading pair in the URL
        trading_pair = f"{symbol}USDT"
        content = re.sub(
            r'(symbol=)([^&"]+)',
            f'\\1{trading_pair}',
            content
        )
    elif 'orderbook_ask_bids_history.py' in str(src_path):
        # For orderbook files: Change the trading pair to match the symbol
        # Convert BTC to BTCUSDT, ETH to ETHUSDT, etc.
        trading_pair = f"{symbol}USDT"
        content = re.sub(
            r'(symbol=)([^&"]+)',
            f'\\1{trading_pair}',
            content
        )
    elif 'taker_buy_sell_volume_history.py' in str(src_path):
        # For taker buy/sell files: Change the trading pair to match the symbol
        trading_pair = f"{symbol}USDT"
        content = re.sub(
            r'(symbol=)([^&"]+)',
            f'\\1{trading_pair}',
            content
        )
    
    # Also update the output filename in the code to include the symbol
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
    
    print("Creating symbol-specific API files...")
    
    # Process each file for each symbol
    for file_path in FILES_TO_MODIFY:
        if not os.path.exists(file_path):
            print(f"Warning: File not found - {file_path}")
            continue
            
        print(f"\nProcessing base file: {file_path}")
        
        for symbol in CRYPTO_SYMBOLS:
            # Check if the file already contains this symbol
            with open(file_path, 'r') as f:
                file_content = f.read()
                
            # For BTC files, we'll create symbol-specific versions for all symbols
            # including BTC to maintain consistency
            
            # Create the modified file
            new_file = create_modified_file(file_path, symbol)
            created_files.append(new_file)
            print(f"  - Created: {new_file}")
    
    print(f"\nCreated {len(created_files)} new symbol-specific files")
    print("\nSymbol files creation complete!")

if __name__ == "__main__":
    main()