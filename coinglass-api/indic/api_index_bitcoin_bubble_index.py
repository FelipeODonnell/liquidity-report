

import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/index/bitcoin/bubble-index"

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
      "price": 0.0495,                          // Current price
      "bubble_index": -29.59827206,             // Bubble index
      "google_trend_percent": 0.0287,           // Google trend percentage
      "mining_difficulty": 181.543,             // Mining difficulty
      "transaction_count": 235,                 // Transaction count
      "address_send_count": 390,                // Address send count
      "tweet_count": 0,                         // Tweet count
      "date_string": "2010-07-17"               // Date string
    },
    {
      "price": 0.0726,                          // Current price
      "bubble_index": -29.30591863,             // Bubble index
      "google_trend_percent": 0.0365,           // Google trend percentage
      "mining_difficulty": 181.543,             // Mining difficulty
      "transaction_count": 248,                 // Transaction count
      "address_send_count": 424,                // Address send count
      "tweet_count": 0,                         // Tweet count
      "date_string": "2010-07-18"               // Date string
    }
  ]
}

'''