

import requests
import json
import pandas as pd
import os
from datetime import datetime
import pyarrow



url = "https://open-api-v4.coinglass.com/api/exchange/chain/tx/list"

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
      "transaction_hash": "0xb8d08182d2de176ac42dceba9ff82a7a5fe650c1e56285d6810daac9561ff9dc", // Transaction hash
      "asset_symbol": "USDT",                       // Asset symbol
      "amount_usd": 9998.5,                         // Amount in USD
      "asset_quantity": 9998.5,                     // Quantity
      "exchange_name": "Coinbase",                 // Exchange name
      "transfer_type": 1,                           // Transfer type: 1 = Inflow, 2 = Outflow, 3 = Internal transfer
      "from_address": "0x16c6897438c4f0c7894862d884549e8564c4025f", // From address
      "to_address": "0xa9d1e08c7793af67e9d92fe308d5697fb81d3e43",   // To address
      "transaction_time": 1745224211                // Transaction time (timestamp in seconds)
    },
    {
      "transaction_hash": "0x033c56cb05654f2c360235eff99c84f0eee9c6330fc7012930adfe0a88789c0f", // Transaction hash
      "asset_symbol": "MANA",                       // Asset symbol
      "amount_usd": 6368.95196834,                  // Amount in USD
      "asset_quantity": 20091.33113044,             // Quantity
      "exchange_name": "Binance",                  // Exchange name
      "transfer_type": 1,                           // Transfer type: 1 = Inflow, 2 = Outflow, 3 = Internal transfer
      "from_address": "0x06fd4ba7973a0d39a91734bbc35bc2bcaa99e3b0", // From address
      "to_address": "0x28c6c06298d514db089934071355e5743bf21d60",   // To address
      "transaction_time": 1745224211                // Transaction time (timestamp in seconds)
    }
  ]
}
Query Params
symbol
string
Must be a token symbol supported by the ERC-20 protocol on the Ethereum (ETH) network.

start_time
int64
Defaults to 1706089927315
Start timestamp in milliseconds (e.g., 1706089927315).

min_usd
double
Minimum transfer amount filter, specified in USD.

per_page
int32
Defaults to 10
Number of results per page.

page
int32
Defaults to 1
Page number for pagination, default: 1.

'''