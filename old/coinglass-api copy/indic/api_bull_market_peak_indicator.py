
import requests
import json
import pandas as pd
import os
from datetime import datetime


url = "https://open-api-v4.coinglass.com/api/bull-market-peak-indicator"

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
      "indicator_name": "Bitcoin Ahr999 Index",  // Indicator name
      "current_value": "0.78",                  // Current value
      "target_value": "4",                      // Target value
      "previous_value": "0.77",                 // Previous value
      "change_value": "0.0009811160359081",     // Change value
      "comparison_type": ">=",                  // Comparison type
      "hit_status": false                       // Hit status (whether the target condition is met)
    },
    {
      "indicator_name": "Pi Cycle Top Indicator",  // Indicator name
      "current_value": "85073.0",                  // Current value
      "target_value": "154582",                    // Target value
      "previous_value": "85127.0",                 // Previous value
      "change_value": "-54.0",                     // Change value
      "comparison_type": ">=",                     // Comparison type
      "hit_status": false                          // Hit status (whether the target condition is met)
    }
  ]
}

'''