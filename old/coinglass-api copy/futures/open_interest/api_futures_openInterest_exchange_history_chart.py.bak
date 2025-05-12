import requests

url = "https://open-api-v4.coinglass.com/api/futures/open-interest/exchange-history-chart?symbol=BTC&range=12h"

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
  "data": {
    "time_list": [1721649600000, ...], // List of timestamps (in milliseconds)
    "price_list": [67490.3, ...], // List of prices corresponding to each timestamp

    "data_map": { // Open interest data of futures from each exchange
      "Binance": [8018229234, ...], // Binance open interest (corresponds to time_list)
      "Bitmex": [395160842, ...] // BitMEX open interest (corresponds to time_list)
      // ...
    }
  }
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin (e.g., BTC). Check supported coins through the 'support-coins' API.

BTC
range
string
required
Defaults to 12h
Time range for the data (e.g., all, 1m, 15m, 1h, 4h, 12h).

12h
unit
string
Defaults to usd
Unit for the returned data, choose between 'usd' or 'coin'.

'''