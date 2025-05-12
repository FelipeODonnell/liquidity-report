import requests

url = "https://open-api-v4.coinglass.com/api/futures/funding-rate/exchange-list"

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
      "stablecoin_margin_list": [ // USDT/USD margin mode
        {
          "exchange": "Binance", // Exchange
          "funding_rate_interval": 8, // Funding rate interval (hours)
          "funding_rate": 0.007343, // Current funding rate
          "next_funding_time": 1745222400000 // Next funding time (milliseconds)
        },
        {
          "exchange": "OKX", // Exchange
          "funding_rate_interval": 8, // Funding rate interval (hours)
          "funding_rate": 0.00736901950628, // Current funding rate
          "next_funding_time": 1745222400000 // Next funding time (milliseconds)
        }
      ],
      "token_margin_list": [ // Coin-margined mode
        {
          "exchange": "Binance", // Exchange
          "funding_rate_interval": 8, // Funding rate interval (hours)
          "funding_rate": -0.001829, // Current funding rate
          "next_funding_time": 1745222400000 // Next funding time (milliseconds)
        }
      ]
    }
  ]
}

'''