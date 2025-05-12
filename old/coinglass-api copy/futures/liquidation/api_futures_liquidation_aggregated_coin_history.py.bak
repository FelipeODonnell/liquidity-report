import requests

url = "https://open-api-v4.coinglass.com/api/futures/liquidation/aggregated-history?exchange_list=Binance&symbol=BTC&interval=1d"

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
      "time": 1658966400000, // Timestamp (milliseconds) 
      "aggregated_long_liquidation_usd": 5916885.14234,//aggregated Long position liquidation amount (USD)
      "aggregated_short_liquidation_usd": 12969583.87632 //aggregated Short position liquidation amount (USD)
    },
    {
      "time": 1659052800000,  // Timestamp (milliseconds)
      "aggregated_long_liquidation_usd": 5345708.23191, //aggregated Long position liquidation amount (USD)
      "aggregated_short_liquidation_usd": 6454875.54909 //aggregated Short position liquidation amount (USD)
    },
  ]
}

Query Params
exchange_list
string
required
Defaults to Binance
List of exchange names to retrieve data from (e.g., 'Binance, OKX, Bybit')

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
Defaults to 1d
Time interval for data aggregation. Supported values: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w.

1d
limit
int32
Defaults to 10
Number of results per request. Default: 1000, Maximum: 4500.

start_time
int64
Start timestamp in milliseconds (e.g., 1641522717000).

end_time
int64
End timestamp in milliseconds (e.g., 1641522717000).

'''