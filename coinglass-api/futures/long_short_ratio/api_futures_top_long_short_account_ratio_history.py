import requests

url = "https://open-api-v4.coinglass.com/api/futures/top-long-short-account-ratio/history?exchange=Binance&symbol=BTCUSDT&interval=h4"

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
      "time": 1741615200000, // Timestamp (in milliseconds)
      "top_account_long_percent": 73.3, // Long position percentage of top accounts (%)
      "top_account_short_percent": 26.7, // Short position percentage of top accounts (%)
      "top_account_long_short_ratio": 2.75 // Long/Short ratio of top accounts
    },
    {
      "time": 1741618800000, // Timestamp (in milliseconds)
      "top_account_long_percent": 74.18, // Long position percentage of top accounts (%)
      "top_account_short_percent": 25.82, // Short position percentage of top accounts (%)
      "top_account_long_short_ratio": 2.87 // Long/Short ratio of top accounts
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