
import requests

url = "https://open-api-v4.coinglass.com/api/futures/funding-rate/vol-weight-history?symbol=BTC&interval=1d"

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
      "time": 1658880000000, // Timestamp (milliseconds)
      "open": "0.004603",     // Opening funding rate
      "high": "0.009388",     // Highest funding rate
      "low": "-0.005063",     // Lowest funding rate
      "close": "0.009229"     // Closing funding rate
    },
    {
      "time": 1658966400000, // Timestamp (milliseconds)
      "open": "0.009229",     // Opening funding rate
      "high": "0.01",         // Highest funding rate
      "low": "0.007794",      // Lowest funding rate
      "close": "0.01"         // Closing funding rate
    }
  ]
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin (e.g., BTC). Retrieve supported coins via the 'support-coins' API.

BTC
interval
string
required
Defaults to 1d
Time interval for data aggregation. Supported values: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w

1d
limit
int32
Defaults to 10
Number of results per request. Default: 1000, Maximum: 4500

start_time
int64
Start timestamp in milliseconds (e.g., 1641522717000).

end_time
int64
End timestamp in milliseconds (e.g., 1641522717000).

'''