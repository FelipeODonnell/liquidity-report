
'''
Upgrade needed again

'''

import requests
import json
import pandas as pd
import os
from datetime import datetime


url = "https://open-api-v4.coinglass.com/api/futures/funding-rate/arbitrage?usd=10000"

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
      "symbol": "SUPRA", // Token symbol
      "buy": { //  (lower funding rate)
        "exchange": "MEXC", // Exchange name
        "open_interest_usd": 848218.2833, // Open interest in USD on the exchange
        "funding_rate_interval": 4, // Funding rate interval (hours)
        "funding_rate": -0.994 // Current funding rate (%)
      },
      "sell": { //  (higher funding rate)
        "exchange": "Gate.io", // Exchange name
        "open_interest_usd": 448263.5072, // Open interest in USD on the exchange
        "funding_rate_interval": 4, // Funding rate interval (hours)
        "funding_rate": 0.005 // Current funding rate (%)
      },
      "apr": 2187.81, // Annual Percentage Rate (APR, %)
      "funding": 0.999, // Funding rate difference (between long and short)
      "fee": 0.03, // Total trading fee (both sides)
      "spread": -0.09, // Price spread between platforms (%)
      "next_funding_time": 1745222400000 // Next funding settlement time (timestamp in milliseconds)
    }
  ]
}
Query Params
usd
int64
required
Defaults to 10000
Investment principal for arbitrage (e.g., 10000).


'''