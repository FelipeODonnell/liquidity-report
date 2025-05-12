
'''

Upgrade for higher frequency 

'''

import requests

url = "https://open-api-v4.coinglass.com/api/futures/price/history?exchange=Binance&symbol=BTCUSDT&interval=4h&limit=10"

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
  "data": [
    {
      "time": 1745366400000,
      "open": "93404.9",
      "high": "93864.9",
      "low": "92730",
      "close": "92858.2",
      "volume_usd": "1166471854.3026"
    },
    {
      "time": 1745370000000,
      "open": "92858.2",
      "high": "93464.8",
      "low": "92552",
      "close": "92603.8",
      "volume_usd": "871812560.3437"
    },
    ...
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
Trading pair (e.g., BTCUSDT). Check supported pairs through the 'support-exchange-pair' API.

BTCUSDT
interval
string
required
Defaults to 1h
Data aggregation time interval. Supported values: 1m, 3m, 5m, 15m, 30m, 1h, 4h, 6h, 8h, 12h, 1d, 1w.

1h
limit
int32
required
Defaults to 10
Number of results per request. Default: 1000, Maximum: 4500.

10
start_time
int64
Start timestamp in milliseconds (e.g., 1641522717).

end_time
int64
End timestamp in milliseconds (e.g., 1641522717).



'''