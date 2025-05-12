
import requests

url = "https://open-api-v4.coinglass.com/api/futures/funding-rate/accumulated-exchange-list?range=1d"

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
      "symbol": "BTC", // Symbol
      "stablecoin_margin_list": [ // Accumulated funding rate for USDT/USD margin mode
        {
          "exchange": "BINANCE", // Exchange name
          "funding_rate": 0.001873 // Accumulated funding rate
        },
        {
          "exchange": "OKX", // Exchange name
          "funding_rate": 0.00775484 // Accumulated funding rate
        }
      ],

      "token_margin_list": [ // Accumulated funding rate for coin-margined mode
        {
          "exchange": "BINANCE", // Exchange name
          "funding_rate": -0.003149 // Accumulated funding rate
        }
      ]
    }
  ]
}
Query Params
range
string
required
Defaults to 1d
Time range for the data (e.g.,1d, 7d, 30d, 365d).

'''