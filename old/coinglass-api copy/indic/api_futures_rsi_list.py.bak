

import requests

url = "https://open-api-v4.coinglass.com/api/futures/rsi/list"

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
      "symbol": "BTC",                                  // Token symbol, e.g., BTC = Bitcoin
      "rsi_15m": 54.71,                                 // RSI (Relative Strength Index) over 15 minutes
      "price_change_percent_15m": 0.04,                 // Price change percentage over 15 minutes
      "rsi_1h": 71.91,                                  // RSI over 1 hour
      "price_change_percent_1h": -0.23,                 // Price change percentage over 1 hour
      "rsi_4h": 72.12,                                  // RSI over 4 hours
      "price_change_percent_4h": -0.09,                 // Price change percentage over 4 hours
      "rsi_12h": 62.33,                                 // RSI over 12 hours
      "price_change_percent_12h": 2.72,                 // Price change percentage over 12 hours
      "rsi_24h": 57.88,                                 // RSI over 24 hours
      "price_change_percent_24h": 3.4,                  // Price change percentage over 24 hours
      "rsi_1w": 52.04,                                  // RSI over 1 week
      "price_change_percent_1w": 2.6,                   // Price change percentage over 1 week
      "current_price": 87348.6                          // Current market price
    },
    {
      "symbol": "ETH",                                  // Token symbol, e.g., ETH = Ethereum
      "rsi_15m": 54.35,                                 // RSI over 15 minutes
      "price_change_percent_15m": -0.13,                // Price change percentage over 15 minutes
      "rsi_1h": 67.93,                                  // RSI over 1 hour
      "price_change_percent_1h": -0.26,                 // Price change percentage over 1 hour
      "rsi_4h": 63.6,                                   // RSI over 4 hours
      "price_change_percent_4h": 0.2,                   // Price change percentage over 4 hours
      "rsi_12h": 52.09,                                 // RSI over 12 hours
      "price_change_percent_12h": 3.41,                 // Price change percentage over 12 hours
      "rsi_24h": 45.03,                                 // RSI over 24 hours
      "price_change_percent_24h": 3.27,                 // Price change percentage over 24 hours
      "rsi_1w": 33.31,                                  // RSI over 1 week
      "price_change_percent_1w": 3.45,                  // Price change percentage over 1 week
      "current_price": 1641.36                          // Current market price
    }
  ]
  
'''