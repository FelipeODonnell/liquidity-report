

import requests

url = "https://open-api-v4.coinglass.com/api/futures/aggregated-taker-buy-sell-volume/history?exchange_list=Binance&symbol=BTC&interval=h1"

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
      "time": 1741622400000, // Timestamp in milliseconds
      "aggregated_buy_volume_usd": 968834542.3787, // Aggregated buy volume (USD)
      "aggregated_sell_volume_usd": 1054582654.8138 // Aggregated sell volume (USD)
    },
    {
      "time": 1741626000000,
      "aggregated_buy_volume_usd": 1430620763.2041,
      "aggregated_sell_volume_usd": 1559166911.2821
    },
    {
      "time": 1741629600000,
      "aggregated_buy_volume_usd": 1897261721.0129,
      "aggregated_sell_volume_usd": 2003812276.7812
    }
  ]
}
Query Params
exchange_list
string
required
Defaults to Binance
exchange_list: List of exchange names to retrieve data from (e.g., 'Binance, OKX, Bybit')

Binance
symbol
string
required
Defaults to BTC
Trading pair (e.g., BTC). Retrieve supported coins via the 'support-coins' API.

BTC
interval
string
required
Defaults to h1
Time interval for data aggregation. Supported values: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w

h1
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

unit
string
Defaults to usd
Unit for the returned data, choose between 'usd' or 'coin'.

'''