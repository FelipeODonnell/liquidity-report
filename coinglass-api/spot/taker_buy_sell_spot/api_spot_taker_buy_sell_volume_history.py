
import requests

url = "https://open-api-v4.coinglass.com/api/spot/taker-buy-sell-volume/history?exchange=Binance&symbol=BTCUSDT&interval=h4"

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
      "taker_buy_volume_usd": "10551.033", // Taker buy volume (USD)
      "taker_sell_volume_usd": "11308" // Taker sell volume (USD)
    },
    {
      "time": 1741626000000,
      "taker_buy_volume_usd": "15484.245",
      "taker_sell_volume_usd": "16316.118"
    },
    {
      "time": 1741629600000,
      "taker_buy_volume_usd": "20340.501",
      "taker_sell_volume_usd": "18977.660"
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
Trading pair (e.g., BTCUSDT). Retrieve supported pairs via the 'support-exchange-pair' API.

BTCUSDT
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
'''