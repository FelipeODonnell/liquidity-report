

import requests

url = "https://open-api-v4.coinglass.com/api/exchange/balance/list?symbol=BTC"

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
      "exchange_name": "Coinbase",               // Exchange name
      "total_balance": 716590.351233,            // Total balance
      "balance_change_1d": 638.797302,           // Balance change in 24 hours
      "balance_change_percent_1d": 0.09,         // Balance change percentage in 24 hours (%)
      "balance_change_7d": 799.967408,           // Balance change in 7 days
      "balance_change_percent_7d": 0.11,         // Balance change percentage in 7 days (%)
      "balance_change_30d": -29121.977486,       // Balance change in 30 days
      "balance_change_percent_30d": -3.91        // Balance change percentage in 30 days (%)
    },
    {
      "exchange_name": "Binance",                // Exchange name
      "total_balance": 582344.497738,            // Total balance
      "balance_change_1d": 505.682778,           // Balance change in 24 hours
      "balance_change_percent_1d": 0.09,         // Balance change percentage in 24 hours (%)
      "balance_change_7d": -3784.88544,          // Balance change in 7 days
      "balance_change_percent_7d": -0.65,        // Balance change percentage in 7 days (%)
      "balance_change_30d": 3753.870055,         // Balance change in 30 days
      "balance_change_percent_30d": 0.65         // Balance change percentage in 30 days (%)
    }
  ]
}
Query Params
symbol
string
required
Defaults to BTC
Trading coin eg. BTC , ETH

'''