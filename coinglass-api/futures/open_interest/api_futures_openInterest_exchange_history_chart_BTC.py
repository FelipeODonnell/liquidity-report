import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/futures/open-interest/exchange-history-chart?symbol=BTC&range=12h&start_time=1731436209918&end_time=1747071009918"

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
        base_filename = f"{base_filename}_BTC"
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
    "time_list": [1721649600000, ...], // List of timestamps (in milliseconds)
    "price_list": [67490.3, ...], // List of prices corresponding to each timestamp

    "data_map": { // Open interest data of futures from each exchange
      "Binance": [8018229234, ...], // Binance open interest (corresponds to time_list)
      "Bitmex": [395160842, ...] // BitMEX open interest (corresponds to time_list)
      // ...
    }
  }
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin (e.g., BTC). Check supported coins through the 'support-coins' API.

BTC
range
string
required
Defaults to 12h
Time range for the data (e.g., all, 1m, 15m, 1h, 4h, 12h).

12h
unit
string
Defaults to usd
Unit for the returned data, choose between 'usd' or 'coin'.

'''