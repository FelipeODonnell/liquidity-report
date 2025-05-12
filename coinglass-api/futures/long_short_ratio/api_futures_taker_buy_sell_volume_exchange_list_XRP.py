import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/futures/taker-buy-sell-volume/exchange-list?symbol=XRP&range=h4&start_time=1731436209918&end_time=1747071009918"

headers = {
    "accept": "application/json",
    "CG-API-KEY": "a5b89c9d85dc40ffb8144fbecf0fb18f"
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
        # Add symbol to base filename
        base_filename = f"{base_filename}_XRP"
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
  "data": {
    "symbol": "BTC", // Token symbol
    "buy_ratio": 51.01, // Buy ratio (%)
    "sell_ratio": 48.99, // Sell ratio (%)
    "buy_vol_usd": 1112108532.1688, // Total buy volume (USD)
    "sell_vol_usd": 1068220541.0417, // Total sell volume (USD)
    "exchange_list": [ // Buy/sell data per exchange
      {
        "exchange": "Binance", // Exchange name
        "buy_ratio": 49.22, // Buy ratio (%)
        "sell_ratio": 50.78, // Sell ratio (%)
        "buy_vol_usd": 240077939.5811, // Buy volume (USD)
        "sell_vol_usd": 247674925.1653 // Sell volume (USD)
      },
      {
        "exchange": "OKX", // Exchange name
        "buy_ratio": 50.84, // Buy ratio (%)
        "sell_ratio": 49.16, // Sell ratio (%)
        "buy_vol_usd": 108435724.6214, // Buy volume (USD)
        "sell_vol_usd": 104834502.5904 // Sell volume (USD)
      }
    ]
  }
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin (e.g., BTC). Retrieve supported coins via the 'support-coins' API.

BTC
range
string
required
Defaults to h1
Time range for the data (e.g., 5m, 15m, 30m, 1h, 4h,12h, 24h).

'''