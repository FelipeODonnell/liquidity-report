
import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/detail?ticker=GBTC"

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
  "data": {
    "ticker_info": {
      "id": 1,                                         // ETF ID
      "ticker": "GBTC",                                // ETF ticker symbol
      "name": "Grayscale Bitcoin Trust ETF",           // ETF name
      "market": "stocks",                              // Market type
      "region": "us",                                  // Region
      "primary_exchange": "ARCX",                      // Primary exchange
      "fund_type": "ETV",                              // Fund type
      "active": "true",                                // Whether the ETF is active
      "currency_name": "usd",                          // Currency name
      "cik_code": "0001588489",                        // CIK code
      "composite_figi": "BBG008748J88",                // Composite FIGI
      "share_class_figi": "BBG008748J97",              // Share class FIGI
      "phone_number": "212-668-1427",                  // Contact phone number
      "tag": "BTC",                                    // Asset tag
      "type2": "Spot",                                 // Additional product type
      "address": {
        "address_1": "{\"address2\":\"4TH FLOOR\",\"city\":\"STAMFORD\",\"address1\":\"290 HARBOR DRIVE\",\"state\":\"CT\",\"postal_code\":\"06902\"}"
      },                                               // Company address (as a JSON string)
      "sic_code": "6221",                              // SIC code
      "sic_description": "COMMODITY CONTRACTS BROKERS & DEALERS", // Industry description
      "ticker_root": "GBTC",                           // Ticker root
      "list_date": 1424822400000,                      // Listing date (timestamp in ms)
      "share_class_shares_outstanding": "240750100",   // Shares outstanding
      "round_lot": "100",                              // Round lot size
      "status": 1,                                     // Status
      "update_time": 1745224505000                     // Last update time (timestamp in ms)
    },
    "market_status": "early_trading",                  // Market status
    "name": "Grayscale Bitcoin Trust ETF",             // ETF name
    "ticker": "GBTC",                                  // ETF ticker
    "type": "stocks",                                  // Market type
    "session": {
      "change": 2.22,                                  // Price change
      "change_percent": 3.309,                         // Change percentage (%)
      "early_trading_change": 2.22,                    // Pre-market change
      "early_trading_change_percent": 3.309,           // Pre-market change percentage (%)
      "close": 67.09,                                  // Previous closing price
      "high": 67.56,                                   // Highest price
      "low": 66.15,                                    // Lowest price
      "open": 66.86,                                   // Opening price
      "volume": 1068092,                               // Trading volume
      "previous_close": 67.09,                         // Previous close
      "price": 69.31                                   // Latest price
    },
    "last_quote": {
      "last_updated": 1745226801708029700,             // Last update time (timestamp in nanoseconds)
      "timeframe": "REAL-TIME",                        // Timeframe type
      "ask": 69.29,                                    // Ask price
      "ask_size": 34,                                  // Ask size
      "ask_exchange": 8,                               // Ask exchange code
      "bid": 69.18,                                    // Bid price
      "bid_size": 3,                                   // Bid size
      "bid_exchange": 11                               // Bid exchange code
    },
    "last_trade": {
      "last_updated": 1745226730467043600,             // Last trade update time (timestamp in nanoseconds)
      "timeframe": "REAL-TIME",                        // Timeframe type
      "id": "62879131651684",                          // Trade ID
      "price": 69.31,                                  // Trade price
      "size": 30,                                      // Trade volume
      "exchange": 12,                                  // Exchange code
      "conditions": [12, 37]                           // Trade condition codes
    },
    "performance": {
      "low_price_52week": 39.56,                       // 52-week low
      "high_price_52week": 86.11,                      // 52-week high
      "high_price_52week_date": 1734238800000,         // 52-week high date (timestamp)
      "low_price_52week_date": 1722744000000,          // 52-week low date (timestamp)
      "ydt_change_percent": -12.23,                    // Year-to-date change (%)
      "year_change_percent": 13.98,                    // 1-year change (%)
      "avg_vol_usd_10d": 518227                        // 10-day average trading value (USD)
    }
  }
}
Query Params
ticker
string
required
Defaults to GBTC
ETF ticker symbol (e.g., GBTC, IBIT).

'''