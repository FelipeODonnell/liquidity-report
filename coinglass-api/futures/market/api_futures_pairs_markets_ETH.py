

import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/futures/pairs-markets?symbol=ETH&start_time=1731436209918&end_time=1747071009918"

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
        base_filename = f"{base_filename}_ETH"
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
      "instrument_id": "BTCUSDT", // Futures trading pair
      "exchange_name": "Binance", // Exchange name
      "symbol": "BTC/USDT", // Trading pair symbol

      "current_price": 84604.3, // Current price
      "index_price": 84646.66222222, // Index price
      "price_change_percent_24h": 0.67, // 24h price change (%)

      "volume_usd": 11317580109.5041, // 24h trading volume (USD)
      "volume_usd_change_percent_24h": -32.13, // 24h volume change (%)

      "long_volume_usd": 5800829746.047, // Long trade volume (USD)
      "short_volume_usd": 5516750363.4571, // Short trade volume (USD)
      "long_volume_quantity": 1130850, // Number of long trades
      "short_volume_quantity": 1162710, // Number of short trades

      "open_interest_quantity": 77881.234, // Open interest quantity (contracts)
      "open_interest_usd": 6589095073.8296, // Open interest value (USD)
      "open_interest_change_percent_24h": 1.9, // 24h open interest change (%)

      "long_liquidation_usd_24h": 3654182.12, // Long liquidations in past 24h (USD)
      "short_liquidation_usd_24h": 4099047.79, // Short liquidations in past 24h (USD)

      "funding_rate": 0.002007, // Current funding rate
      "next_funding_time": 1744963200000, // Next funding time (timestamp)

      "open_interest_volume_radio": 0.5822, // Open interest to volume ratio
      "oi_vol_ratio_change_percent_24h": 50.13 // 24h ratio change (%)
    },
    {
      "instrument_id": "BTC_USDT", // Futures trading pair
      "exchange_name": "Gate.io", // Exchange name
      "symbol": "BTC/USDT", // Trading pair symbol

      "current_price": 84616.3, // Current price
      "index_price": 84643.36, // Index price
      "price_change_percent_24h": 0.69, // 24h price change (%)

      "volume_usd": 1711484049.255, // 24h trading volume (USD)
      "volume_usd_change_percent_24h": -67.03, // 24h volume change (%)

      "long_volume_usd": 870432407.5966, // Long trade volume (USD)
      "short_volume_usd": 841051641.6584, // Short trade volume (USD)
      "long_volume_quantity": 210027, // Number of long trades
      "short_volume_quantity": 218777, // Number of short trades

      "open_interest_quantity": 69477.278, // Open interest quantity (contracts)
      "open_interest_usd": 5878785139.331, // Open interest value (USD)
      "open_interest_change_percent_24h": 3.82, // 24h open interest change (%)

      "long_liquidation_usd_24h": 1502896.68, // Long liquidations in past 24h (USD)
      "short_liquidation_usd_24h": 1037959.7, // Short liquidations in past 24h (USD)

      "funding_rate": 0.0022, // Current funding rate

      "open_interest_volume_radio": 3.4349, // Open interest to volume ratio
      "oi_vol_ratio_change_percent_24h": 214.93 // 24h ratio change (%)
    }
  ]
}

'''