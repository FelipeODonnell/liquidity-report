
import requests
import json
import pandas as pd
import os
from datetime import datetime


url = "https://open-api-v4.coinglass.com/api/spot/orderbook/history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=5"

headers = {"CG-API-KEY": "a5b89c9d85dc40ffb8144fbecf0fb18f"}

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
        [
            1723611600,
            [
                [
                    56420, //Price
                    4.777 //Quantity
                ],
                [
                    40300,
                    2.191
                ]
            ],
            [
                [
                    56420,
                    4.777
                ],
                [
                    40300,
                    2.191
                ]
            ]
        ]
    ],
    "success": true
}
Query Params
exchange
string
required
Defaults to Binance
Spot exchange names (e.g., Binance, OKX) .Retrieve supported exchanges via the 'support-exchange-pair' API.

Binance
symbol
string
required
Defaults to BTCUSDT
Supported trading pairs (e.g., BTCUSDT, ETHUSDT). Tick sizes: BTCUSDT (TickSize=20), ETHUSDT (TickSize=0.5).

BTCUSDT
interval
string
required
Defaults to 1h
Time intervals for data aggregation. Supported values: 1h, 4h, 8h, 12h, 1d.

1h
limit
string
required
Defaults to 5
Number of results per request. Default: 1000, Maximum: 4500.

5
start_time
string
Start timestamp in milliseconds (e.g., 1641522717000).

end_time
string
End timestamp in milliseconds (e.g., 1641522717000).

'''