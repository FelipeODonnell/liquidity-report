'''

upgrade needed 

'''

import requests
import json
import pandas as pd
import os
from datetime import datetime


url = "https://open-api-v4.coinglass.com/api/futures/liquidation/order?exchange=Binance&symbol=BTC&min_liquidation_amount=10000"

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
      "exchange_name": "BINANCE", // Exchange name
      "symbol": "BTCUSDT", // Trading pair symbol
      "base_asset": "BTC", // Base asset
      "price": 87535.9, // Liquidation price
      "usd_value": 205534.2932, // Transaction amount (USD)
      "side": 2, // Order direction (1: Buy, 2: Sell)
      "time": 1745216319263 // Timestamp
    },
    {
      "exchange_name": "BINANCE", // Exchange name
      "symbol": "BTCUSDT", // Trading pair symbol
      "base_asset": "BTC", // Base asset
      "price": 87465.2, // Liquidation price
      "usd_value": 15918.6664, // Transaction amount (USD)
      "side": 2, // Order direction (1: Buy, 2: Sell)
      "time": 1745215647165 // Timestamp
    }
  ]
}
Query Params
exchange
string
required
Defaults to Binance
Exchange name (e.g., Binance, OKX). Retrieve supported exchanges via the 'support-exchange-pair' API.

Binance
symbol
string
required
Defaults to BTC
Trading coin (e.g., BTC). Retrieve supported coins via the 'support-coins' API.

BTC
min_liquidation_amount
string
required
Defaults to 10000
Minimum threshold for liquidation events.

10000
start_time
int64
Start timestamp in milliseconds (e.g., 1641522717000).

end_time
int64
End timestamp in milliseconds (e.g., 1641522717000).

'''