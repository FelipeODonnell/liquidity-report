'''

upgrade needed 

'''

import requests
import json
import pandas as pd
import os
from datetime import datetime


url = "https://open-api-v4.coinglass.com/api/futures/liquidation/coin-list?exchange=Binance"

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
      "symbol": "BTC", // Token symbol
      "liquidation_usd_24h": 82280481.50425325, // Total liquidation amount in the past 24 hours (USD)
      "long_liquidation_usd_24h": 68437447.33734027, // Long position liquidation in the past 24 hours (USD)
      "short_liquidation_usd_24h": 13843034.16691298, // Short position liquidation in the past 24 hours (USD)

      "liquidation_usd_12h": 68331844.36224127, // Total liquidation in the past 12 hours
      "long_liquidation_usd_12h": 66614158.47451427, // Long liquidation (12h)
      "short_liquidation_usd_12h": 1717685.887727, // Short liquidation (12h)

      "liquidation_usd_4h": 11381137.080643, // Total liquidation in the past 4 hours
      "long_liquidation_usd_4h": 10921633.272973, // Long liquidation (4h)
      "short_liquidation_usd_4h": 459503.80767, // Short liquidation (4h)

      "liquidation_usd_1h": 3283635.95309, // Total liquidation in the past 1 hour
      "long_liquidation_usd_1h": 3182915.16289, // Long liquidation (1h)
      "short_liquidation_usd_1h": 100720.7902 // Short liquidation (1h)
    }
  ]
}
Query Params
exchange
string
required
Defaults to Binance
Futures exchange names (e.g., Binance, OKX) .Retrieve supported exchanges via the 'support-exchange-pair' API.

'''