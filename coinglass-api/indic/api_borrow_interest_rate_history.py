

import requests

url = "https://open-api-v4.coinglass.com/api/borrow-interest-rate/history?exchange=Binance&symbol=BTC&interval=h1"

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
      "time": 1741636800,                  // Timestamp (in seconds)
      "interest_rate": 0.002989            // daily Interest rate
    },
    {
      "time": 1741640400,                  // Timestamp (in seconds)
      "interest_rate": 0.002989            // daily Interest rate
    }
  ]
}
Query Params
exchange
string
required
Defaults to Binance
Exchange name support:Binance ,OKX,Bybit

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
Defaults to 1706089927315
Start timestamp in milliseconds (e.g., 1641522717000).

end_time
int64
End timestamp in milliseconds (e.g., 1641522717000).

'''