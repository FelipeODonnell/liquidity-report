
import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/futures/funding-rate/accumulated-exchange-list?range=1d"

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
  "data": [
    {
      "symbol": "BTC", // Symbol
      "stablecoin_margin_list": [ // Accumulated funding rate for USDT/USD margin mode
        {
          "exchange": "BINANCE", // Exchange name
          "funding_rate": 0.001873 // Accumulated funding rate
        },
        {
          "exchange": "OKX", // Exchange name
          "funding_rate": 0.00775484 // Accumulated funding rate
        }
      ],

      "token_margin_list": [ // Accumulated funding rate for coin-margined mode
        {
          "exchange": "BINANCE", // Exchange name
          "funding_rate": -0.003149 // Accumulated funding rate
        }
      ]
    }
  ]
}
Query Params
range
string
required
Defaults to 1d
Time range for the data (e.g.,1d, 7d, 30d, 365d).

'''