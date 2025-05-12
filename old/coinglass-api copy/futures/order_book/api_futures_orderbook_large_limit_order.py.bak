
'''
upgrade needed
'''

import requests

url = "https://open-api-v4.coinglass.com/api/futures/orderbook/large-limit-order?exchange=Binance&symbol=BTCUSDT"

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
  "data":[
  {
    "id": 2868159989,
    "exchange_name": "Binance",            // Exchange name
    "symbol": "BTCUSDT",                   // Trading pair
    "base_asset": "BTC",                   // Base asset
    "quote_asset": "USDT",                 // Quote asset

    "price": 56932,                  // Order price
    "start_time": 1722964242000,           // Order start time (ms)
    "start_quantity": 28.39774,            // Initial order quantity
    "start_usd_value": 1616740.1337,       // Initial USD value

    "current_quantity": 18.77405,          // Current remaining quantity
    "current_usd_value": 1068844.21,       // Current USD value
    "current_time": 1722964272000,         // Current time (ms)

    "executed_volume": 0,                  // Executed volume
    "executed_usd_value": 0,               // Executed USD value
    "trade_count": 0,                      // Number of trades

    "order_side": 2,                       // Order side: 1 - Sell, 2 - Buy
    "order_state": 1                       // Order state: 1 - Open, 2 - Filled, 3      - Canceled
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
Trading pair (e.g., BTCUSDT). Retrieve supported pair via the 'support-exchange-pair' API.
'''