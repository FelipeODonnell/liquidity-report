

import requests

url = "https://open-api-v4.coinglass.com/api/spot/orderbook/ask-bids-history?exchange=Binance&symbol=BTCUSDT&interval=1d"

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
      "bids_usd": 81639959.9338,        // Total long position amount (USD)
      "bids_quantity": 1276.645,        // Total long quantity
      "asks_usd": 78533053.6862,        // Total short position amount (USD)
      "asks_quantity": 1217.125,        // Total short quantity
      "time": 1714003200000             // Timestamp (in milliseconds)
    },
    {
      "bids_usd": 62345879.8821,
      "bids_quantity": 980.473,
      "asks_usd": 65918423.4715,
      "asks_quantity": 1021.644,
      "time": 1714089600000
    }
  ]
}
Query Params
exchange
string
required
Defaults to Binance
Exchange name (e.g., Binance). Retrieve supported exchanges via the 'support-exchange-pair' API.

Binance
symbol
string
required
Defaults to BTCUSDT
Trading pair (e.g., BTCUSDT). Check supported pairs through the 'support-exchange-pair' API.

BTCUSDT
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

range
string
Defaults to 1
Depth percentage (e.g., 0.25, 0.5, 0.75, 1, 2, 3, 5, 10).

'''