

import requests

url = "https://open-api-v4.coinglass.com/api/etf/bitcoin/price/history?ticker=GBTC&range=1d"

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
  "message": "success",
  "data": [
    {
      "time": 1731056460000,   // timestamp in milliseconds
      "open": 60.47,                // Opening price
      "high": 60.47,                // Highest price
      "low": 60.47,                 // Lowest price
      "close": 60.47,               // Closing price
      "volume": 100                // Trading volume
    },
    ...
  ]
}
Query Params
ticker
string
required
Defaults to GBTC
ETF ticker symbol (e.g., GBTC, IBIT).

GBTC
range
string
required
Defaults to 1d
Time range for the data (e.g., 1d,7d,all).

'''