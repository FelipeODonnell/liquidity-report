"""
Test script for the formatters module, particularly the new column_name_formatter
"""

import sys
import os
import pandas as pd

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.formatters import format_column_name

def test_column_name_formatter():
    """Test the format_column_name function with various column name examples"""
    test_cases = [
        # Basic cases
        ('price', 'Price'),
        ('volume_24h', 'Volume 24h'),
        ('market_cap', 'Market Cap'),
        
        # Special cases with abbreviations
        ('price_usd', 'Price USD'),
        ('btc_dominance', 'BTC Dominance'),
        ('roi_30d', 'ROI 30d'),
        ('roi_24h', 'ROI 24h'),
        ('eth_gas_price', 'ETH Gas Price'),
        
        # Percentage cases
        ('change_pct', 'Change %'),
        ('market_share_percent', 'Market Share %'),
        
        # Mixed cases
        ('eth_btc_ratio', 'ETH BTC Ratio'),
        ('usdt_market_cap_usd', 'USDT Market Cap USD'),
        ('l1_l2_bridge_tvl', 'L1 L2 Bridge TVL'),
        
        # Edge cases
        ('', ''),
        ('a', 'A'),
        ('_', ' '),
        ('__', '  '),
        ('a_b_c', 'A B C')
    ]
    
    results = []
    for original, expected in test_cases:
        actual = format_column_name(original)
        results.append({
            'Original': original,
            'Expected': expected,
            'Actual': actual,
            'Passed': actual == expected
        })
    
    # Create a DataFrame for nicer display
    results_df = pd.DataFrame(results)
    
    # Print results
    print("Column Name Formatter Test Results:")
    print(results_df)
    print(f"Tests passed: {results_df['Passed'].sum()} out of {len(results_df)}")

if __name__ == "__main__":
    test_column_name_formatter()