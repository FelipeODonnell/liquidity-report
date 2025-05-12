

import requests

url = "https://open-api-v4.coinglass.com/api/exchange/balance/chart?symbol=BTC"

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
      "time_list": [1691460000000, ...],       // Array of timestamps (in milliseconds)
      "price_list": [29140.9, ...],            // Array of prices corresponding to each timestamp
      "data_map": {                            // Balance data by exchange
        "huobi": [15167.03527, ...],           // Balance data from Huobi exchange
        "gate": [23412.723, ...],              // Balance data from Gate exchange
        ...
      }
    }
  ]
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin eg. BTC, ETH

'''