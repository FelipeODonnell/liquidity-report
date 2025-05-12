import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/futures/liquidation/exchange-list?range=4h&symbol=XRP&start_time=1731436209918&end_time=1747071009918"

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
        base_filename = f"{base_filename}_XRP"
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
      "exchange": "All", // Total data from all exchanges
      "liquidation_usd": 14673519.81739075, // Total liquidation amount (USD)
      "long_liquidation_usd": 451394.17404598, // Long position liquidation amount (USD)
      "short_liquidation_usd": 14222125.64334477 // Short position liquidation amount (USD)
    },
    {
      "exchange": "Bybit", // Exchange name
      "liquidation_usd": 4585290.13404, // Total liquidation amount (USD)
      "long_liquidation_usd": 104560.13885, // Long position liquidation (USD)
      "short_liquidation_usd": 4480729.99519 // Short position liquidation (USD)
    }
  ]
}
Query Params
symbol
string
Defaults to BTC
Trading coin (e.g., BTC). Retrieve supported coins via the 'support-coins' API.

range
string
required
Defaults to 1h
Time range for data aggregation. Supported values: 1h, 4h, 12h, 24h.

1h


'''