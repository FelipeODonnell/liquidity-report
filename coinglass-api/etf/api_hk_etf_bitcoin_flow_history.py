
import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/hk-etf/bitcoin/flow-history"

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
      "timestamp": 1714435200000,                     // Date (timestamp in milliseconds)
      "flow_usd": 247866000,                          // Total capital inflow (USD)
      "price_usd": 63842.4,                           // BTC price on that date (USD)
      "etf_flows": [                                  // ETF capital flow details
        {
          "etf_ticker": "CHINAAMC",                   // ETF ticker
          "flow_usd": 123610690                       // Capital inflow for this ETF (USD)
        },
        {
          "etf_ticker": "HARVEST",                    // ETF ticker
          "flow_usd": 63138000                        // Capital inflow for this ETF (USD)
        },
        {
          "etf_ticker": "BOSERA&HASHKEY",             // ETF ticker
          "flow_usd": 61117310                        // Capital inflow for this ETF (USD)
        }
      ]
    }
  ]
}

'''