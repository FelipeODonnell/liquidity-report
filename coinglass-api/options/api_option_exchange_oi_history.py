
import requests

url = "https://open-api-v4.coinglass.com/api/option/exchange-oi-history?symbol=BTC&unit=USD&range=4h"

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
      "data_map": {                            // Open Interest (OI) data by exchange
        "huobi": [15167.03527, ...],           // OI data from Huobi exchange
        "gate": [23412.723, ...],              // OI data from Gate exchange
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

USD
range
string
required
Defaults to 1h
Time range for the data. Supported values: 1h, 4h, 12h, all.

'''