

import requests

url = "https://open-api-v4.coinglass.com/api/etf/ethereum/flow-history"

headers = {
    "accept": "application/json",
    "CG-API-KEY": "a5b89c9d85dc40ffb8144fbecf0fb18f"
}

response = requests.get(url, headers=headers)

print(response.text)

'''
Response Data
JSON

{
  "code": "0",
  "msg": "success",
  "data": [
    {
      "timestamp": 1721692800000,            // Timestamp
      "change_usd": 106600000,              // Flow in/out (USD)
      "price": 3438.09,                     // Current price
      "close_price": 3481.01,               // Close price
      "etf_flows": [                        // ETF flow list
        {
          "ticker": "ETHA",             // ETF ticker
          "change_usd": 266500000       // ETF flow (USD)
        },
        {
          "ticker": "FETH",             // ETF ticker
          "change_usd": 71300000        // ETF flow (USD)
        }
      ]
    }
  ]
}

'''