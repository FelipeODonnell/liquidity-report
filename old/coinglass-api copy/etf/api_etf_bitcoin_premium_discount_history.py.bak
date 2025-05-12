
import requests

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/premium-discount/history"

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
      "timestamp": 1706227200000,                 // Date (timestamp in milliseconds)
      "list": [
        {
          "ticker": "GBTC",                       // ETF ticker
          "nav_usd": 37.51,                        // Net Asset Value (USD)
          "market_price_usd": 37.51,               // Market price (USD)
          "premium_discount_details": 0            // Premium/Discount percentage
        },
        {
          "ticker": "IBIT",                       // ETF ticker
          "nav_usd": 23.94,                        // Net Asset Value (USD)
          "market_price_usd": 23.99,               // Market price (USD)
          "premium_discount_details": 0.22         // Premium/Discount percentage
        },
        {
          "ticker": "FBTC",                       // ETF ticker
          "nav_usd": 36.720807,                    // Net Asset Value (USD)
          "market_price_usd": 36.75,               // Market price (USD)
          "premium_discount_details": 0.0795       // Premium/Discount percentage
        }
      ]
    }
  ]
}
Query Params
ticker
string
ETF ticker symbol (e.g., GBTC, IBIT).

'''