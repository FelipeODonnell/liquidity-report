
'''
upgrade needed
'''

import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow
import streamlit as st





url = "https://open-api-v4.coinglass.com/api/futures/orderbook/large-limit-order?exchange=Binance&symbol=BTCUSDT&start_time=1731498631418&end_time=1747133431418"

headers = {
    "accept": "application/json",
    "CG-API-KEY": st.secrets["coinglass_api"]["api_key"]
}

response = requests.get(url, headers=headers)

print(response.text)

# Save response data as parquet
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
        file_path = os.path.join(output_dir, f"{base_filename}.parquet")
        
        # Save as parquet
        df.to_parquet(file_path, index=False)
        print(f"Data saved to {file_path}")
except Exception as e:
    print(f"Error saving data: {e}")

'''
Response Data
JSON

{
  "code": "0",
  "msg": "success",
  "data":[
  {
    "id": 2868159989,
    "exchange_name": "Binance",            // Exchange name
    "symbol": "BTCUSDT",                   // Trading pair
    "base_asset": "BTC",                   // Base asset
    "quote_asset": "USDT",                 // Quote asset

    "price": 56932,                  // Order price
    "start_time": 1722964242000,           // Order start time (ms)
    "start_quantity": 28.39774,            // Initial order quantity
    "start_usd_value": 1616740.1337,       // Initial USD value

    "current_quantity": 18.77405,          // Current remaining quantity
    "current_usd_value": 1068844.21,       // Current USD value
    "current_time": 1722964272000,         // Current time (ms)

    "executed_volume": 0,                  // Executed volume
    "executed_usd_value": 0,               // Executed USD value
    "trade_count": 0,                      // Number of trades

    "order_side": 2,                       // Order side: 1 - Sell, 2 - Buy
    "order_state": 1                       // Order state: 1 - Open, 2 - Filled, 3      - Canceled
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
Trading pair (e.g., BTCUSDT). Retrieve supported pair via the 'support-exchange-pair' API.
'''