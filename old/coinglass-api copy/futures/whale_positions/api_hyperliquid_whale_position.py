
'''
Upgrade needed
'''

import requests
import json
import pandas as pd
import os
from datetime import datetime


url = "https://open-api-v4.coinglass.com/api/hyperliquid/whale-position"

headers = {
    "accept": "application/json",
    "CG-API-KEY": "a5b89c9d85dc40ffb8144fbecf0fb18f"
}

response = requests.get(url, headers=headers)

print(response.text)

# Save response data as CSV
try:
    # Parse JSON response
    response_data = json.loads(response.text)
    
    # Check if response contains data
    if response_data.get('code') == '0' and 'data' in response_data:
        # Convert to DataFrame
        df = pd.DataFrame(response_data['data'])
        
        # Get current date for folder name
        current_date = datetime.now().strftime('%Y%m%d')
        
        # Create directory structure if it doesn't exist
        # Use current API file path to determine output path
        api_path = os.path.dirname(os.path.abspath(__file__))
        relative_path = os.path.relpath(api_path, 'coinglass-api')
        output_dir = os.path.join('data', current_date, relative_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename without timestamp
        base_filename = os.path.splitext(os.path.basename(__file__))[0]
        file_path = os.path.join(output_dir, f"{base_filename}.csv")
        
        # Save as CSV
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
except Exception as e:
    print(f"Error saving data: {e}")

'''
Response:


JSON

{
  "code": "0",
  "msg": "success",
  "data": [
    {
      "user": "0x20c2d95a3dfdca9e9ad12794d5fa6fad99da44f5", // User address
      "symbol": "ETH",                                   // Token symbol
      "position_size": -44727.1273,                      // Position size (positive: long, negative: short)
      "entry_price": 2249.7,                             // Entry price
      "mark_price": 1645.8,                              // Current mark price
      "liq_price": 2358.2766,                            // Liquidation price
      "leverage": 25,                                    // Leverage
      "margin_balance": 2943581.7019,                    // Margin balance (USD)
      "position_value_usd": 73589542.5467,               // Position value (USD)
      "unrealized_pnl": 27033236.424,                    // Unrealized PnL (USD)
      "funding_fee": -3107520.7373,                      // Funding fee (USD)
      "margin_mode": "cross",                            // Margin mode (e.g., cross / isolated)
      "create_time": 1741680802000,                      // Entry time (timestamp in ms)
      "update_time": 1745219966000                       // Last updated time (timestamp in ms)
    },
    {
      "user": "0xf967239debef10dbc78e9bbbb2d8a16b72a614eb",
      "symbol": "BTC",
      "position_size": -800,
      "entry_price": 84931.3,
      "mark_price": 87427,
      "liq_price": 92263.798,
      "leverage": 15,
      "margin_balance": 4812076.3896,
      "position_value_usd": 69921600,
      "unrealized_pnl": -1976493.6819,
      "funding_fee": 14390.0346,
      "margin_mode": "isolated",
      "create_time": 1743982804000,
      "update_time": 1745219969000
    }
  ]
}
'''