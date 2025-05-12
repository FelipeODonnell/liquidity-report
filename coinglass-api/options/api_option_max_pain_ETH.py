
import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/option/max-pain?symbol=ETH&exchange=Deribit&start_time=1731436209918&end_time=1747071009918"

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
      "date": "250422",                                   // Date (YYMMDD format)
      "call_open_interest_market_value": 1616749.22,      // Call option market value (USD)
      "put_open_interest": 512.5,                         // Put option open interest (contracts)
      "put_open_interest_market_value": 49687.62,         // Put option market value (USD)
      "max_pain_price": "84000",                          // Max pain price
      "call_open_interest": 953.7,                        // Call option open interest (contracts)
      "call_open_interest_notional": 83519113.56,         // Call option notional value (USD)
      "put_open_interest_notional": 44881569.13           // Put option notional value (USD)
    },
    {
      "date": "250423",                                   // Date (YYMMDD format)
      "call_open_interest_market_value": 2274700.52,      // Call option market value (USD)
      "put_open_interest": 1204.3,                        // Put option open interest (contracts)
      "put_open_interest_market_value": 374536.01,        // Put option market value (USD)
      "max_pain_price": "85000",                          // Max pain price
      "call_open_interest": 1302.2,                       // Call option open interest (contracts)
      "call_open_interest_notional": 114040373.53,        // Call option notional value (USD)
      "put_open_interest_notional": 105465691.73          // Put option notional value (USD)
    }
  ]
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin (e.g., BTC,ETH).

BTC
exchange
string
required
Defaults to Deribit
Exchange name (e.g., Deribit, Binance, OKX).

'''