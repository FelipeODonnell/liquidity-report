
import requests

url = "https://open-api-v4.coinglass.com/api/option/info?symbol=BTC"

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
      "exchange_name": "All",                           // Exchange name
      "open_interest": 361038.78,                       // Open interest (contracts)
      "oi_market_share": 100,                           // Market share (%)
      "open_interest_change_24h": 2.72,                 // 24h open interest change (%)
      "open_interest_usd": 31623069708.138245,          // Open interest value (USD)
      "volume_usd_24h": 2764676957.0569425,             // 24h trading volume (USD)
      "volume_change_percent_24h": 303.1                // 24h volume change (%)
    },
    {
      "exchange_name": "Deribit",                       // Exchange name
      "open_interest": 262641.9,                        // Open interest (contracts)
      "oi_market_share": 72.74,                         // Market share (%)
      "open_interest_change_24h": 2.57,                 // 24h open interest change (%)
      "open_interest_usd": 23005403973.349,             // Open interest value (USD)
      "volume_usd_24h": 2080336672.709                  // 24h trading volume (USD)
    }
  ]
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin (e.g., BTC,ETH).

'''