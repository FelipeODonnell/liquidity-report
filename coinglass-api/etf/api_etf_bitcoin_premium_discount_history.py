
import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow
import streamlit as st





url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/premium-discount/history"

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
      "timestamp": 1706227200000,                 // Date (timestamp in milliseconds)
      "list": [
        {
          "ticker": "GBTC",                       // ETF ticker
          "nav_usd": 37.51,                        // Net Asset Value (USD)
          "market_price_usd": 37.51,               // Market price (USD)
          "premium_discount_details": 0            // Premium/Discount percentage
        },
        {
          "ticker": "IBIT",                       // ETF ticker
          "nav_usd": 23.94,                        // Net Asset Value (USD)
          "market_price_usd": 23.99,               // Market price (USD)
          "premium_discount_details": 0.22         // Premium/Discount percentage
        },
        {
          "ticker": "FBTC",                       // ETF ticker
          "nav_usd": 36.720807,                    // Net Asset Value (USD)
          "market_price_usd": 36.75,               // Market price (USD)
          "premium_discount_details": 0.0795       // Premium/Discount percentage
        }
      ]
    }
  ]
}
Query Params
ticker
string
ETF ticker symbol (e.g., GBTC, IBIT).

'''