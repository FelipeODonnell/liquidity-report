

'''
upgrade needed

'''

import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow
import streamlit as st





url = "https://open-api-v4.coinglass.com/api/futures/liquidation/aggregated-heatmap/model3?symbol=BTC&range=3d&start_time=1731498631418&end_time=1747133431418"

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
  "data": {
    "y_axis": [47968.54, 48000.00, 48031.46], // Y-axis price levels
    "liquidation_leverage_data": [
      [5, 124, 2288867.26], // Each array: [X-axis index, Y-axis index, liquidation amount in USD]
      [6, 123, 318624.82],
      [7, 122, 1527940.12]
    ],
    "price_candlesticks": [
      [
        1722676500, // Timestamp (seconds)
        "61486",    // Open price
        "61596.4",  // High price
        "61434.4",  // Low price
        "61539.9",  // Close price
        "63753192.1129" // Trading volume (USD)
      ],
      [
        1722676800,
        "61539.9",
        "61610.0",
        "61480.0",
        "61590.5",
        "42311820.8720"
      ]
    ]
  }
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin (e.g., BTC). Retrieve supported coins via the 'support-coins' API.

BTC
range
string
required
Defaults to 3d
Time range for data aggregation. Supported values: 12h, 24h, 3d, 7d, 30d, 90d, 180d, 1y.

'''