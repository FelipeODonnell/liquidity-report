

import requests

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/history?ticker=GBTC"

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
      "assets_date": 1706486400000,           // Net asset date (timestamp in milliseconds)
      "btc_holdings": 496573.8166,            // BTC holdings
      "market_date": 1706486400000,           // Market price date (timestamp in milliseconds)
      "market_price": 38.51,                  // Market price (USD)
      "name": "Grayscale Bitcoin Trust",      // ETF name
      "nav": 38.57,                           // Net Asset Value per share (USD)
      "net_assets": 21431132778.35,           // Total net assets (USD)
      "premium_discount": -0.16,              // Premium/discount percentage
      "shares_outstanding": 555700100,        // Total shares outstanding
      "ticker": "GBTC"                        // ETF ticker
    }
  ]
}
Query Params
ticker
string
required
Defaults to GBTC
ETF ticker symbol (e.g., GBTC, IBIT).

'''