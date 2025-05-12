

import requests
import json
import pandas as pd
import os
from datetime import datetime


url = "https://open-api-v4.coinglass.com/api/bitfinex-margin-long-short?symbol=BTC&interval=1d"

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
      "time": 1658880000,              // Timestamp, representing the data's corresponding time point
      "long_quantity": 104637.94,       // Long position quantity
      "short_quantity": 2828.53        // Short position quantity
    },
    {
      "time": 1658966400,              // Timestamp, representing the data's corresponding time point
      "long_quantity": 105259.46,       // Long position quantity
      "short_quantity": 2847.84        // Short position quantity
    }
    // More data entries...
  ]
}
Query Params
symbol
string
required
Defaults to BTC
BTC,ETH

BTC
limit
int32
Number of results per request. Default: 1000, Maximum: 4500.

interval
string
required
Defaults to 1d
Data aggregation time interval. Supported values: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w.

1d
start_time
int64
Start timestamp in milliseconds (e.g., 1641522717000).

end_time
int64
End timestamp in milliseconds (e.g., 1641522717000).

'''