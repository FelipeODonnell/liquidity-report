

import requests

url = "https://open-api-v4.coinglass.com/api/index/bitcoin/profitable-days"

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
      "side": 1,                              // Trade direction, 1 represents buy, 0 represents sell
      "timestamp": 1282003200000,             // Timestamp (in milliseconds)
      "price": 0.07                           // Price
    },
    {
      "timestamp": 1282089600000,             // Timestamp (in milliseconds)
      "price": 0.068,                         // Price
      "side": 1                               // Trade direction, 1 represents buy, 0 represents sell
    }
  ]
}

'''