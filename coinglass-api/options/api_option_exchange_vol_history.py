

import requests

url = "https://open-api-v4.coinglass.com/api/option/exchange-vol-history?symbol=BTC&unit=USD"

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
      "data_map": {                            // Volume data by exchange
        "huobi": [15167.03527, ...],           // Volume data from Huobi exchange
        "gate": [23412.723, ...],              // Volume data from Gate exchange
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
Trading coin (e.g., BTC,ETH).

BTC
unit
string
required
Defaults to USD
Specify the unit for the returned data. Supported values depend on the symbol. If symbol is BTC, choose between USD or BTC. For ETH, choose between USD or ETH.
'''