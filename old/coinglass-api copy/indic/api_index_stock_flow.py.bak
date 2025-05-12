
import requests

url = "https://open-api-v4.coinglass.com/api/index/stock-flow"

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
      "timestamp": 1282003200000,       // Timestamp (in milliseconds)
      "price": 0.07,                     // Price on the given day
      "next_halving": 834               // Days remaining until the next halving
    },
    {
      "timestamp": 1282089600000,       // Timestamp (in milliseconds)
      "price": 0.068,                    // Price on the given day
      "next_halving": 833               // Days remaining until the next halving
    }

  ]
}

'''