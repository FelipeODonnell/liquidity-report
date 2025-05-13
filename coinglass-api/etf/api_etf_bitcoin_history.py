

import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow
import streamlit as st





url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/history?ticker=GBTC&start_time=1731498631418&end_time=1747133431418"

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
  "data": [
    {
      "assets_date": 1706486400000,           // Net asset date (timestamp in milliseconds)
      "btc_holdings": 496573.8166,            // BTC holdings
      "market_date": 1706486400000,           // Market price date (timestamp in milliseconds)
      "market_price": 38.51,                  // Market price (USD)
      "name": "Grayscale Bitcoin Trust",      // ETF name
      "nav": 38.57,                           // Net Asset Value per share (USD)
      "net_assets": 21431132778.35,           // Total net assets (USD)
      "premium_discount": -0.16,              // Premium/discount percentage
      "shares_outstanding": 555700100,        // Total shares outstanding
      "ticker": "GBTC"                        // ETF ticker
    }
  ]
}
Query Params
ticker
string
required
Defaults to GBTC
ETF ticker symbol (e.g., GBTC, IBIT).

'''