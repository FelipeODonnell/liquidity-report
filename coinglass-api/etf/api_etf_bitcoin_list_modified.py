import requests
import json
import pandas as pd
import os
from datetime import datetime

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/list"

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
        
        # Create directory if it doesn't exist
        os.makedirs('data/coinglass/etf', exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = f'data/coinglass/etf/api_etf_bitcoin_list_{timestamp}.csv'
        
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
      "ticker": "GBTC",                                // ETF identifier
      "fund_name": "Grayscale Bitcoin Trust ETF",      // Fund name
      "region": "us",                                  // Region
      "market_status": "early_trading",                // Market status (open/closed/early_trading, etc.)
      "primary_exchange": "ARCX",                      // Primary exchange
      "cik_code": "0001588489",                        // CIK code (unique identifier)
      "fund_type": "Spot",                             // Fund type (Spot/ETF/Futures, etc.)
      "list_date": 1424822400000,                      // Listing date (timestamp in milliseconds)
      "shares_outstanding": "240750100",               // Shares outstanding
      "aum_usd": "16137543152.34",                     // Assets under management (USD)
      "management_fee_percent": "1.5",                 // Management fee (%)
      "last_trade_time": 1745225312958,                // Last trade time (timestamp in milliseconds)
      "last_quote_time": 1745225389483,                // Last quote time (timestamp in milliseconds)
      "volume_quantity": 1068092,                      // Volume quantity
      "volume_usd": 71485902.2312,                     // Volume in USD
      "price_change_usd": 0.47,                        // Price change (USD)
      "price_change_percent": 0.71,                    // Price change (%)
      "asset_details": {
        "net_asset_value_usd": 67.03,                  // Net asset value (USD)
        "premium_discount_percent": 0.09,              // Premium/discount rate (%)
        "btc_holding": 190124.5441,                    // BTC balance
        "btc_change_percent_24h": 0,                   // 24h BTC change (%)
        "btc_change_24h": -7.8136,                     // 24h BTC change amount
        "btc_change_percent_7d": -0.32,                // 7d BTC change (%)
        "btc_change_7d": -615.563,                     // 7d BTC change amount
        "update_date": "2025-04-17"                    // Update date
      },
      "update_timestamp": 1745224505000                // Data update timestamp (milliseconds)
    }
  ]
}

'''