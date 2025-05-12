import requests

url = "https://open-api-v4.coinglass.com/api/futures/liquidation/history?exchange=Binance&symbol=BTCUSDT&interval=1d"

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
      "long_liquidation_usd": "2369935.19562", // Long position liquidation amount (USD)
      "short_liquidation_usd": "6947459.43674" // Short position liquidation amount (USD)
    },
    {
      "time": 1658966400000, // Timestamp (milliseconds)
      "long_liquidation_usd": "5118407.85124", // Long position liquidation amount (USD)
      "short_liquidation_usd": "8517330.44192" // Short position liquidation amount (USD)
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