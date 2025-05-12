

import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/index/golden-ratio-multiplier"

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
      "low_bull_high_2": 0.14,                       // Bull market low-high ratio coefficient
      "timestamp": 1282003200000,                    // Timestamp (in milliseconds)
      "price": 0.07,                                 // Current price
      "ma_350": 0.07,                                // 350-day moving average
      "accumulation_high_1_6": 0.11200000000000002,  // Accumulation high ratio (1/6 golden ratio)
      "x_3": 0.21000000000000002,                    // Golden ratio multiple x3
      "x_5": 0.35000000000000003,                    // Golden ratio multiple x5
      "x_8": 0.56,                                   // Golden ratio multiple x8
      "x_13": 0.9100000000000001,                    // Golden ratio multiple x13
      "x_21": 1.4700000000000002                     // Golden ratio multiple x21
    },
    ...
  ]
}

'''