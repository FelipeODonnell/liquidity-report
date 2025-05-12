

import requests

url = "https://open-api-v4.coinglass.com/api/index/stableCoin-marketCap-history"

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
      "data_list": [4611285.1141, ...],         // List of values
      "price_list": [, ...],                    // List of prices
      "time_list": [1636588800, ...]            // List of timestamps
    }
  ]
}

'''