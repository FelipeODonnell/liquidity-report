
import requests

url = "https://open-api-v4.coinglass.com/api/hk-etf/bitcoin/flow-history"

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
      "timestamp": 1714435200000,                     // Date (timestamp in milliseconds)
      "flow_usd": 247866000,                          // Total capital inflow (USD)
      "price_usd": 63842.4,                           // BTC price on that date (USD)
      "etf_flows": [                                  // ETF capital flow details
        {
          "etf_ticker": "CHINAAMC",                   // ETF ticker
          "flow_usd": 123610690                       // Capital inflow for this ETF (USD)
        },
        {
          "etf_ticker": "HARVEST",                    // ETF ticker
          "flow_usd": 63138000                        // Capital inflow for this ETF (USD)
        },
        {
          "etf_ticker": "BOSERA&HASHKEY",             // ETF ticker
          "flow_usd": 61117310                        // Capital inflow for this ETF (USD)
        }
      ]
    }
  ]
}

'''