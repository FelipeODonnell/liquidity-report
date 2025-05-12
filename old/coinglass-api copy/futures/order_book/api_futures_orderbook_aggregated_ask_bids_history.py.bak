
import requests

url = "https://open-api-v4.coinglass.com/api/futures/orderbook/aggregated-ask-bids-history?exchange_list=Binance&symbol=BTC&interval=h1"

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
      "aggregated_bids_usd": 12679537.0806,         // Aggregated long amount (USD)
      "aggregated_bids_quantity": 197.99861,        // Aggregated long quantity
      "aggregated_asks_usd": 10985519.9268,         // Aggregated short amount (USD)
      "aggregated_asks_quantity": 170.382,          // Aggregated short quantity
      "time": 1714003200000                         // Timestamp (milliseconds)
    },
    {
      "aggregated_bids_usd": 18423845.1947,
      "aggregated_bids_quantity": 265.483,
      "aggregated_asks_usd": 17384271.5521,
      "aggregated_asks_quantity": 240.785,
      "time": 1714089600000
    }
  ]
}
Query Params
exchange_list
string
required
Defaults to Binance
List of exchange names to retrieve data from (e.g., 'ALL', or 'Binance, OKX, Bybit')

Binance
symbol
string
required
Defaults to BTC
Trading coin (e.g., BTC). Retrieve supported coins via the 'support-coins' API.

BTC
interval
string
required
Defaults to h1
Data aggregation time interval. Supported values: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w.

h1
limit
int32
Defaults to 500
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