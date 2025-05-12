
import requests

url = "https://open-api-v4.coinglass.com/api/futures/basis/history?exchange=Binance&symbol=BTCUSDT&interval=4h"

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
      "time": 1741629600000,            // Timestamp (in milliseconds)
      "open_basis": 0.0504,             // Opening basis (%) - the basis at the start of the interval
      "close_basis": 0.0445,            // Closing basis (%) - the basis at the end of the interval
      "open_change": 39.5,              // Percentage change in basis at opening compared to previous period
      "close_change": 34.56             // Percentage change in basis at closing compared to previous period
    },
    {
      "time": 1741633200000,            // Timestamp (in milliseconds)
      "open_basis": 0.0446,             // Opening basis (%)
      "close_basis": 0.03,              // Closing basis (%)
      "open_change": 34.65,             // Opening basis change (%)
      "close_change": 23.74             // Closing basis change (%)
    }
  ]
}
Query Params
exchange
string
required
Defaults to Binance
Futures exchange names (e.g., Binance, OKX) .Retrieve supported exchanges via the 'support-exchange-pair' API.

Binance
symbol
string
required
Defaults to BTCUSDT
Trading pair (e.g., BTCUSDT). Retrieve supported pairs via the 'support-exchange-pair' API.

BTCUSDT
interval
string
required
Defaults to 1h
Data aggregation time interval. Supported values: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w.

1h
limit
string
Defaults to 10
Number of results per request. Default: 1000, Maximum: 4500.

start_time
string
Start timestamp in milliseconds (e.g., 1641522717000).

end_time
string
End timestamp in milliseconds (e.g., 1641522717000).

'''