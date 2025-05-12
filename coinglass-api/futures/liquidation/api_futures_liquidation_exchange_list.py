import requests

url = "https://open-api-v4.coinglass.com/api/futures/liquidation/exchange-list?range=4h"

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
      "exchange": "All", // Total data from all exchanges
      "liquidation_usd": 14673519.81739075, // Total liquidation amount (USD)
      "long_liquidation_usd": 451394.17404598, // Long position liquidation amount (USD)
      "short_liquidation_usd": 14222125.64334477 // Short position liquidation amount (USD)
    },
    {
      "exchange": "Bybit", // Exchange name
      "liquidation_usd": 4585290.13404, // Total liquidation amount (USD)
      "long_liquidation_usd": 104560.13885, // Long position liquidation (USD)
      "short_liquidation_usd": 4480729.99519 // Short position liquidation (USD)
    }
  ]
}
Query Params
symbol
string
Defaults to BTC
Trading coin (e.g., BTC). Retrieve supported coins via the 'support-coins' API.

range
string
required
Defaults to 1h
Time range for data aggregation. Supported values: 1h, 4h, 12h, 24h.

1h


'''