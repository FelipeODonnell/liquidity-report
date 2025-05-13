
import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow
import streamlit as st





url = "https://open-api-v4.coinglass.com/api/etf/ethereum/list"

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
      "ticker": "ETHA",                                  // ETF ticker
      "name": "iShares Ethereum Trust ETF",              // ETF name
      "region": "us",                                    // Region
      "market_status": "closed",                         // Market status
      "primary_exchange": "XNAS",                        // Primary exchange
      "cik_code": "0002000638",                          // CIK code
      "type": "Spot",                                    // Type
      "market_cap": "544896000.00",                      // Market capitalization
      "list_date": 1721692800000,                        // Listing date
      "shares_outstanding": "28800000",                  // Shares outstanding
      "aum": "",                                         // Assets under management
      "management_fee_percent": "0.25",                  // Management fee percentage
      "last_trade_time": 1722988779939,                  // Last trade time
      "last_quote_time": 1722988799379,                  // Last quote time
      "volume_quantity": 5592645,                        // Volume quantity
      "volume_usd": 106447049.343,                       // Volume in USD
      "price": 18.92,                                    // Market price
      "price_change": 0.67,                              // Price change
      "price_change_percent": 3.67,                      // Price change percentage
      "asset_info": {
        "nav": 18.11,                                  // Net asset value
        "premium_discount": 0.77,                      // Premium/discount
        "holding_quantity": 237882.8821,                 // Holding quantity
        "change_percent_1d": 0,                        // 1-day change percentage
        "change_quantity_1d": 0,                         // 1-day change quantity
        "change_percent_7d": 56.69,                    // 7-day change percentage
        "change_quantity_7d": 86060.9115,                // 7-day change quantity
        "date": "2024-08-05"                           // Data date
      },
      "update_time": 1722995656637                       // Update time
    }
  ]
}

'''