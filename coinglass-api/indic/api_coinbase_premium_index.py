

import requests

url = "https://open-api-v4.coinglass.com/api/coinbase-premium-index?interval=1d"

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
      "time": 1658880000,         // Timestamp (in seconds)
      "premium": 5.55,            // Premium amount (USD)
      "premium_rate": 0.0261      // Premium rate (e.g., 0.0261 = 2.61%)

    },
    {
       "time": 1658880000,         // Timestamp (in seconds)
       "premium": 5.55,            // Premium amount (USD)
       "premium_rate": 0.0261      // Premium rate (e.g., 0.0261 = 2.61%)

    }
  ]
}
Query Params
interval
string
required
Defaults to 1d
Data aggregation time interval. Supported values: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w.

1d
limit
int32
Number of results per request. Default: 1000, Maximum: 4500.

start_time
int64
Start timestamp in milliseconds (e.g., 1641522717000).

end_time
int64
End timestamp in milliseconds (e.g., 1641522717000).

'''