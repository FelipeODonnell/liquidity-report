
import requests

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/flow-history"

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
      "timestamp": 1704931200000,                   // Date (timestamp in milliseconds)
      "flow_usd": 655300000,                         // Total daily capital flow (USD)
      "price_usd": 46663,                            // BTC current price (USD)
      "etf_flows": [                                 // ETF capital flow breakdown
        {
          "etf_ticker": "GBTC",                      // ETF ticker
          "flow_usd": -95100000                      // Capital outflow (USD)
        },
        {
          "etf_ticker": "IBIT",                      // ETF ticker
          "flow_usd": 111700000                      // Capital inflow (USD)
        },
        {
          "etf_ticker": "FBTC",                      // ETF ticker
          "flow_usd": 227000000                      // Capital inflow (USD)
        }
      ]
    }
  ]
}

'''