'''

UPGRADE NEEDED

'''

import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/futures/coins-markets"

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
      "symbol": "BTC",  // Cryptocurrency symbol
      "current_price": 84773.6,  // Current price (in USD)
      "avg_funding_rate_by_oi": 0.00196,  // Average funding rate weighted by open interest
      "avg_funding_rate_by_vol": 0.002647,  // Average funding rate weighted by volume
      "market_cap_usd": 1683310500117.051,  // Market capitalization (in USD)
      "open_interest_market_cap_ratio": 0.0327,  // Ratio of open interest to market capitalization
      "open_interest_usd": 55002072334.9376,  // Open interest (in USD)
      "open_interest_quantity": 648525.0328,  // Open interest quantity (number of contracts)
      "open_interest_volume_ratio": 0.7936,  // Ratio of open interest to volume
      "price_change_percent_5m": -0.02,  // Price change percentage in the last 5 minutes
      "price_change_percent_15m": 0.06,  // Price change percentage in the last 15 minutes
      "price_change_percent_30m": 0.03,  // Price change percentage in the last 30 minutes
      "price_change_percent_1h": -0.1,  // Price change percentage in the last 1 hour
      "price_change_percent_4h": -0.15,  // Price change percentage in the last 4 hours
      "price_change_percent_12h": 0.15,  // Price change percentage in the last 12 hours
      "price_change_percent_24h": 1.06,  // Price change percentage in the last 24 hours
      "open_interest_change_percent_5m": 0,  // Open interest change percentage in the last 5 minutes
      "open_interest_change_percent_15m": 0.04,  // Open interest change percentage in the last 15 minutes
      "open_interest_change_percent_30m": 0,  // Open interest change percentage in the last 30 minutes
      "open_interest_change_percent_1h": -0.01,  // Open interest change percentage in the last 1 hour
      "open_interest_change_percent_4h": 0.17,  // Open interest change percentage in the last 4 hours
      "open_interest_change_percent_24h": 4.58,  // Open interest change percentage in the last 24 hours
      "volume_change_percent_5m": -0.03,  // Volume change percentage in the last 5 minutes
      "volume_change_percent_15m": -0.73,  // Volume change percentage in the last 15 minutes
      "volume_change_percent_30m": -1.13,  // Volume change percentage in the last 30 minutes
      "volume_change_percent_1h": -2.33,  // Volume change percentage in the last 1 hour
      "volume_change_percent_4h": -5.83,  // Volume change percentage in the last 4 hours
      "volume_change_percent_24h": -26.38,  // Volume change percentage in the last 24 hours
      "volume_change_usd_1h": 69310404660.3795,  // Volume change in USD in the last 1 hour
      "volume_change_usd_4h": -4290959532.4414644,  // Volume change in USD in the last 4 hours
      "volume_change_usd_24h": -24835757605.82467,  // Volume change in USD in the last 24 hours
      "long_short_ratio_5m": 1.2523,  // Long/short ratio in the last 5 minutes
      "long_short_ratio_15m": 0.9928,  // Long/short ratio in the last 15 minutes
      "long_short_ratio_30m": 1.0695,  // Long/short ratio in the last 30 minutes
      "long_short_ratio_1h": 1.0068,  // Long/short ratio in the last 1 hour
      "long_short_ratio_4h": 1.0504,  // Long/short ratio in the last 4 hours
      "long_short_ratio_12h": 1.0317,  // Long/short ratio in the last 12 hours
      "long_short_ratio_24h": 1.0313,  // Long/short ratio in the last 24 hours
      "liquidation_usd_1h": 33621.85192,  // Total liquidation amount in USD in the last 1 hour
      "long_liquidation_usd_1h": 22178.4681,  // Long liquidation amount in USD in the last 1 hour
      "short_liquidation_usd_1h": 11443.38382,  // Short liquidation amount in USD in the last 1 hour
      "liquidation_usd_4h": 222210.47117,  // Total liquidation amount in USD in the last 4 hours
      "long_liquidation_usd_4h": 179415.77249,  // Long liquidation amount in USD in the last 4 hours
      "short_liquidation_usd_4h": 42794.69868,  // Short liquidation amount in USD in the last 4 hours
      "liquidation_usd_12h": 11895453.392145,  // Total liquidation amount in USD in the last 12 hours
      "long_liquidation_usd_12h": 10223351.23772,  // Long liquidation amount in USD in the last 12 hours
      "short_liquidation_usd_12h": 1672102.154425,  // Short liquidation amount in USD in the last 12 hours
      "liquidation_usd_24h": 27519292.973646,  // Total liquidation amount in USD in the last 24 hours
      "long_liquidation_usd_24h": 17793322.595016,  // Long liquidation amount in USD in the last 24 hours
      "short_liquidation_usd_24h": 9725970.37863  // Short liquidation amount in USD in the last 24 hours
    },
    {
      "symbol": "ETH",  // Cryptocurrency symbol
      "current_price": 1582.55,  // Current price (in USD)
      "avg_funding_rate_by_oi": 0.001631,  // Average funding rate weighted by open interest
      "avg_funding_rate_by_vol": -0.000601,  // Average funding rate weighted by volume
      "market_cap_usd": 190821695398.62064,  // Market capitalization (in USD)
      "open_interest_market_cap_ratio": 0.0925,  // Ratio of open interest to market capitalization
      "open_interest_usd": 17657693967.0459,  // Open interest (in USD)
      "open_interest_quantity": 11160428.5065,  // Open interest quantity (number of contracts)
      "open_interest_volume_ratio": 0.5398,  // Ratio of open interest to volume
      "price_change_percent_5m": 0.07,  // Price change percentage in the last 5 minutes
      "price_change_percent_15m": 0.25,  // Price change percentage in the last 15 minutes
      "price_change_percent_30m": 0.07,  // Price change percentage in the last 30 minutes
      "price_change_percent_1h": -0.11,  // Price change percentage in the last 1 hour
      "price_change_percent_4h": -0.05,  // Price change percentage in the last 4 hours
      "price_change_percent_12h": -0.02,  // Price change percentage in the last 12 hours
      "price_change_percent_24h": 0.16  // Price change percentage in the last 24 hours
      // ... subsequent fields follow in a similar manner
    }
  ]
}

'''