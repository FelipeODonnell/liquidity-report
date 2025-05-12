
'''

upgrade needed
'''

import requests
import json
import pandas as pd
import os
from datetime import datetime


url = "https://open-api-v4.coinglass.com/api/spot/orderbook/large-limit-order?exchange=Binance&symbol=BTCUSDT"

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
Response Data
JSON

{
  "code": "0",
  "msg": "success",
  "data": [
    {
      "id": 2536823422,
      "exchange_name": "Binance",                // Exchange name
      "symbol": "BTCUSDT",                       // Trading pair
      "base_asset": "BTC",                       // Base asset
      "quote_asset": "USDT",                     // Quote asset
      "price": 20000,                            // Order price
      "start_time": 1742615617000,               // Order start time (timestamp)
      "start_quantity": 109.42454,               // Initial quantity
      "start_usd_value": 2188490.8,              // Initial order value in USD
      "current_quantity": 104.00705,             // Current remaining quantity
      "current_usd_value": 2080141,              // Current remaining value in USD
      "current_time": 1745287267958,             // Current update time (timestamp)
      "executed_volume": 0,                      // Executed volume
      "executed_usd_value": 0,                   // Executed value in USD
      "trade_count": 0,                          // Number of executed trades
      "order_side": 2,                           // Order side: 1 = Buy, 2 = Sell
      "order_state": 1                           // Order state: 1 = Active, 2 = Completed
    }
  ]
}
Query Params
exchange
string
required
Defaults to Binance
Exchange name (e.g., Binance). Retrieve supported exchanges via the 'support-exchange-pair' API.

Binance
symbol
string
required
Defaults to BTCUSDT
Trading pair (e.g., BTCUSDT). Retrieve supported pairs via the 'support-exchange-pair' API.

'''