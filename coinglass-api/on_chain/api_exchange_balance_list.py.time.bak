

import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/exchange/balance/list?symbol=BTC"

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
      "exchange_name": "Coinbase",               // Exchange name
      "total_balance": 716590.351233,            // Total balance
      "balance_change_1d": 638.797302,           // Balance change in 24 hours
      "balance_change_percent_1d": 0.09,         // Balance change percentage in 24 hours (%)
      "balance_change_7d": 799.967408,           // Balance change in 7 days
      "balance_change_percent_7d": 0.11,         // Balance change percentage in 7 days (%)
      "balance_change_30d": -29121.977486,       // Balance change in 30 days
      "balance_change_percent_30d": -3.91        // Balance change percentage in 30 days (%)
    },
    {
      "exchange_name": "Binance",                // Exchange name
      "total_balance": 582344.497738,            // Total balance
      "balance_change_1d": 505.682778,           // Balance change in 24 hours
      "balance_change_percent_1d": 0.09,         // Balance change percentage in 24 hours (%)
      "balance_change_7d": -3784.88544,          // Balance change in 7 days
      "balance_change_percent_7d": -0.65,        // Balance change percentage in 7 days (%)
      "balance_change_30d": 3753.870055,         // Balance change in 30 days
      "balance_change_percent_30d": 0.65         // Balance change percentage in 30 days (%)
    }
  ]
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin eg. BTC , ETH

'''